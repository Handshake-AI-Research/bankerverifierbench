"""Reward Kit runner — read the verifier output a Harbor staging run produced.

Like Gandalf, Reward Kit runs *inside* the Harbor container during staging
(see ``adapters/btb/template/_tests-rewardkit/test.sh``). Its output lives
at ``<staged_trial_dir>/verifier/{reward.json,reward-details.json}``. This
runner reads those files and translates them into a :class:`Verdict`.

Output shape (rewardkit 0.1.x)
-------------------------------
``reward.json``::

    {"rubric": 0.83}     # weighted score per Reward.name; we ship one Reward.

``reward-details.json``::

    {
      "rubric": {
        "score": 0.83,
        "kind": "agent",
        "judge": {...},
        "judge_output": "...",
        "criteria": [
          {
            "name": "powerpoint_file_is_created_...",
            "value": 1.0,                   # rounded; binary criteria → 0.0 or 1.0
            "raw": 1,
            "weight": 3,
            "description": "...",
            "reasoning": "..."
          },
          ...
        ]
      }
    }

Mapping into :class:`Verdict`:

* ``criteria[i].value > 0``   → ``CriterionVerdict.satisfied`` (binary)
* ``criteria[i].value``       → ``CriterionVerdict.confidence`` (continuous in [0,1])
* ``criteria[i].reasoning``   → ``CriterionVerdict.reason``
* ``criteria[i].raw``         → ``CriterionVerdict.raw_verdict`` (stringified)

Token / cost totals are not exposed in ``reward-details.json``; left at
zero (consistent with how AAAJ / archipelago / autorubric report when
their native output omits totals — the per-call jsonl trace in
``judge_trajectories/`` is the source of truth).

Empty-workspace short-circuit
-----------------------------
The shipped ``test.sh`` writes ``reward.json = {"rubric": 0.0}`` and
``reward-details.json = {"reason": "empty_workspace", "rewards": {}}``
when the agent produced no artifacts. We detect that shape and return
"all UNMET" with the reason set to ``"empty_workspace"`` so it shows up
distinctly in scoring.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..inputs import InputBundle, validate_bundle
from ..verdict import CriterionVerdict, TokenUsage, Verdict
from .base import Capabilities


# Files in jobs/<staging>/<trial>/verifier/ to copy into the trace dir.
_REWARDKIT_TRACE_FILES = (
    "reward.json",
    "input-audit.json",
    "input-audit-after.json",
    "judge-config.json",
    "mcp-audit.json",
    "isolation-audit.json",
    "reward-details.json",
    "test-stdout.txt",
    "judge-trajectory-copy-stderr.txt",
    "workspace-copy-stderr.txt",
)
# The Claude Code session JSONL(s) — one per judge session, copied in by
# test.sh after rewardkit returns. Used for failure analysis.
_REWARDKIT_TRACE_DIRS = ("judge_trajectories",)


def capabilities() -> Capabilities:
    # Like gandalf, rewardkit's verdict is produced *during* staging; we
    # just read the result. Bundle workspace/trajectory aren't consumed
    # here. Declaring no requirements keeps rewardkit out of the
    # workspace/trajectory skip logic.
    return Capabilities(
        needs_workspace=False,
        needs_trajectory=False,
        accepts_placeholder_trajectory=True,
    )


def invoke(
    bundle: InputBundle,
    *,
    model: str | None = None,
    trace_dir: Path | None = None,
    verbose: bool = False,  # accepted for orchestrator API parity; no-op
) -> Verdict:
    """Translate the staged Reward Kit verifier output into a Verdict."""
    validate_bundle(bundle)
    if bundle.trial_dir is None:
        raise FileNotFoundError(
            f"No staged trial dir for {bundle.task_id}; run `stage` first."
        )
    verifier_dir = bundle.trial_dir / "verifier"
    details_path = verifier_dir / "reward-details.json"
    reward_path = verifier_dir / "reward.json"
    if not details_path.is_file() or not reward_path.is_file():
        raise FileNotFoundError(
            f"Reward Kit verifier output not found at "
            f"{verifier_dir}/{{reward,reward-details}}.json — was the cohort "
            f"staged with --verifier-kind rewardkit?"
        )

    if trace_dir is not None:
        _copy_rewardkit_trace(verifier_dir, trace_dir)

    details = json.loads(details_path.read_text())
    if bundle.input_identity:
        from adapters.bvb.contract import EXPECTED_MCP_TOOLS, REPLAY_VIEW
        audit = json.loads((verifier_dir / "input-audit.json").read_text())
        if set(audit.get("mcp_tools", [])) != EXPECTED_MCP_TOOLS:
            raise ValueError("Inner judge MCP tool access was not verified")
        if audit.get('judge_view_validated') is not True or audit.get('judge_workdir') != REPLAY_VIEW:
            raise ValueError('Full replay judge view was not verified')
        if audit.get("status") != "validated" or audit.get("identity") != bundle.input_identity:
            raise ValueError("RewardKit input audit differs from canonical selected rollout")
        for row, expected in zip(details.get("rubric", {}).get("criteria", []), bundle.rubric):
            if row.get("description") != expected["criterion"] or float(row.get("weight", 0)) != float(expected["weight"]):
                raise ValueError("RewardKit native rubric differs from canonical human rubric")

    # Effective model: explicit override wins; otherwise read it back from
    # the rewardkit details (judge.model for AgentJudge, judge.judge for
    # LLMJudge), falling back to the dataset task.toml metadata.
    effective_model = model or _model_from_details(details) or _model_from_dataset(bundle.task_dir) or ""

    criteria = _criteria_from_details(details, bundle.rubric,
                                     require_explanations=bool(bundle.input_identity))

    return Verdict(
        verifier="rewardkit",
        model=effective_model,
        task_id=bundle.task_id,
        gt_task_id=bundle.gt_task_id,
        rollout_id=bundle.trial_dir.name,
        cohort=bundle.cohort,
        criteria=criteria,
        tokens=TokenUsage(),  # not exposed in reward-details.json
        cost_usd=0.0,
    )


# ---------------------------------------------------------------------------
# Detail-dict → CriterionVerdict translation
# ---------------------------------------------------------------------------

def _criteria_from_details(
    details: dict,
    rubric: list[dict],
    *, require_explanations: bool = False,
) -> list[CriterionVerdict]:
    """Pull the per-criterion list out of reward-details.json.

    Tolerates two shapes:

    * Normal: ``{<reward-name>: {"criteria": [...]}}`` — one entry per Reward.
    * Empty-workspace short-circuit: ``{"reason": "empty_workspace", "rewards": {}}``.
      Returns ``rubric_n`` UNMET verdicts with reason="empty_workspace".
    """
    rubric_n = len(rubric)
    if details.get("reason") == "empty_workspace":
        return [
            CriterionVerdict(
                index=i,
                satisfied=False,
                confidence=0.0,
                reason="empty_workspace (no agent artifacts produced)",
                raw_verdict="empty_workspace",
            )
            for i in range(rubric_n)
        ]

    reward = details.get("rubric")
    if not isinstance(reward, dict) or not isinstance(reward.get("criteria"), list):
        raise ValueError(
            "reward-details.json has no canonical 'rubric.criteria' list; "
            f"keys={list(details)!r}"
        )
    criteria_list = reward["criteria"]
    if len(criteria_list) != rubric_n:
        raise ValueError(
            "reward-details.json criterion count mismatch: "
            f"expected {rubric_n}, found {len(criteria_list)}"
        )
    for index, criterion in enumerate(criteria_list):
        if not isinstance(criterion, dict):
            raise ValueError(
                f"reward-details.json criterion {index} is not an object"
            )
        expected = str(rubric[index].get("criterion", ""))
        observed = criterion.get("description")
        if observed is not None and str(observed) != expected:
            raise ValueError(
                "reward-details.json criterion order/text mismatch at "
                f"index {index}"
            )
        if require_explanations and not str(criterion.get('reasoning') or '').strip():
            raise ValueError(f"RewardKit criterion {index} has no grading explanation")

    return [
        CriterionVerdict(
            index=i,
            satisfied=float(c.get("value", 0) or 0) > 0,
            # value is rewardkit's normalised score in [0,1]. For binary
            # criteria it's 0 or 1; for likert/numeric it's a continuous
            # mid-scale value.
            confidence=float(c.get("value", 0) or 0),
            reason=str(c.get("reasoning", "") or ""),
            raw_verdict=str(c.get("raw", c.get("value", ""))),
        )
        for i, c in enumerate(criteria_list)
    ]


def _model_from_details(details: dict) -> str | None:
    """Read the effective judge model out of reward-details.json.

    Rewardkit's ``judge`` block carries either ``model`` (AgentJudge: the
    underlying LLM the agent CLI uses) or ``judge`` (LLMJudge: the model
    name itself). Try both.
    """
    for value in details.values():
        if not isinstance(value, dict):
            continue
        judge = value.get("judge")
        if isinstance(judge, dict):
            for key in ("model", "judge"):
                m = judge.get(key)
                if isinstance(m, str) and m:
                    return m
    return None


def _model_from_dataset(task_dir: Path) -> str | None:
    """Fallback: read [metadata].verifier_model from the dataset task.toml.

    The rewardkit BTB adapter writes this line. Keeps the model field on
    the verdict populated even when the in-container details file lacks
    a model name (e.g. truncated output).
    """
    task_toml = task_dir / "task.toml"
    if not task_toml.is_file():
        return None
    in_metadata = False
    for line in task_toml.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_metadata = stripped == "[metadata]"
            continue
        if not in_metadata:
            continue
        if stripped.startswith("verifier_model"):
            _, _, value = stripped.partition("=")
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                return value[1:-1]
    return None


def _copy_rewardkit_trace(verifier_dir: Path, trace_dir: Path) -> None:
    """Copy rewardkit's native verifier output into ``trace_dir``.

    Best-effort: missing files are skipped silently. Any I/O error is
    non-fatal — the verdict has already been parsed.
    """
    trace_dir.mkdir(parents=True, exist_ok=True)
    for fname in _REWARDKIT_TRACE_FILES:
        src = verifier_dir / fname
        if src.is_file():
            try:
                shutil.copy2(src, trace_dir / fname)
            except OSError:
                pass
    for dname in _REWARDKIT_TRACE_DIRS:
        src = verifier_dir / dname
        if src.is_dir():
            dst = trace_dir / dname
            try:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            except OSError:
                pass
