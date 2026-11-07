"""Gandalf runner — read the verifier output a Harbor staging run produced.

Gandalf runs *inside* the Harbor container during the staging phase (when
verification is enabled, which is our default for `stage`). Its output
lives at ``<staged_trial_dir>/verifier/info.json``. This runner reads
that file and translates it into a :class:`Verdict`.

The schema mapping:

* ``info.criterion_results[i].met``       → ``criteria[i].satisfied``
* ``info.criterion_results[i].reasoning`` → ``criteria[i].reason``
* ``info.llm_usage.prompt_tokens``        → ``tokens.prompt_tokens``
* ``info.llm_usage.cache_read_tokens``    → ``tokens.cached_prompt_tokens``
* ``info.llm_usage.completion_tokens``    → ``tokens.completion_tokens``
* ``info.llm_usage.cost_usd``             → ``cost_usd``

Gandalf gets MCP access and the full Harbor sandbox during staging, so the
verdict here reflects the real production verifier — no degraded mode.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from ..inputs import InputBundle, validate_bundle
from ..verdict import CriterionVerdict, TokenUsage, Verdict
from .base import Capabilities


# Files in jobs/<staging>/<trial>/verifier/ to copy into the trace dir.
# These are gandalf's native artifacts — small enough (~few hundred KB
# total across all batch splits) to copy rather than symlink, so the
# trace dir survives a `rm -rf jobs/` later.
_GANDALF_TRACE_FILES = (
    "info.json",
    "input-audit.json",
    "mcp-audit.json",
    "actual-judge-inputs.json",
    "judge-clone-audit.json",
    "runtime-versions.json",
    "reward.json",
    "reward.txt",
    "test-stdout.txt",
)
# Match both the single-batch ``judge_trace_batch.txt``
# and the multi-batch ``judge_trace_batch_split{N}.txt`` (BTB). These
# files are the verifier-agent's own trajectory — its prompts, tool
# calls, and reasoning while grading — so failure analysis depends on
# them being persisted out of jobs/ before that dir is rm -rf'd.
_GANDALF_TRACE_GLOBS = ("judge_trace_batch*.txt",)


def capabilities() -> Capabilities:
    # Gandalf's verdict is produced *during* staging (Harbor + MCP); we just
    # read the result. So at evaluate time we need a staged trial dir with
    # ``verifier/info.json``. The bundle's workspace/trajectory aren't
    # consumed here, but the orchestrator passes the same bundle to all
    # runners — declaring no requirements keeps gandalf out of the
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
    """Translate the staged Gandalf verifier output into a Verdict.

    Args:
        bundle: Must have ``trial_dir`` set (i.e. produced via a staging
            run). The trial dir's ``verifier/info.json`` is read.
        model: Override model name in the verdict. When None we read it
            from the trial's ``grader.toml`` if available, falling back
            to whatever the bundle's grader.toml says.
        trace_dir: If given, copy gandalf's native verifier artifacts
            (info.json, reward.json, judge_trace_batch_split*.txt,
            test-stdout.txt) here so post-hoc inspection survives later
            ``rm -rf jobs/``. Created if it doesn't exist.
    """
    validate_bundle(bundle)
    if bundle.trial_dir is None:
        raise FileNotFoundError(
            f"No staged trial dir for {bundle.task_id}; run `stage` first."
        )
    verifier_dir = bundle.trial_dir / "verifier"
    info_path = verifier_dir / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"Gandalf verifier output not found: {info_path} "
            f"(was Harbor verification disabled during staging?)"
        )

    if trace_dir is not None:
        _copy_gandalf_trace(verifier_dir, trace_dir)

    info = json.loads(info_path.read_text())
    if bundle.input_identity:
        from adapters.bvb.contract import EXPECTED_MCP_TOOLS, REPLAY_VIEW
        audit = json.loads((verifier_dir / "input-audit.json").read_text())
        if set(audit.get("mcp_tools", [])) != EXPECTED_MCP_TOOLS:
            raise ValueError("Inner judge MCP tool access was not verified")
        if audit.get('judge_view_validated') is not True or audit.get('judge_workdir') != REPLAY_VIEW:
            raise ValueError('Full replay judge view was not verified')
        if audit.get("status") != "validated" or audit.get("identity") != bundle.input_identity:
            raise ValueError("Gandalf input audit does not match canonical selected rollout")
    rows = info.get("criterion_results", [])
    if len(rows) != len(bundle.rubric):
        raise ValueError("Gandalf result count differs from canonical rubric")
    for index, (row, expected) in enumerate(zip(rows, bundle.rubric)):
        if row.get("criterion") != expected["criterion"] or float(row.get("weight", 0)) != float(expected.get("weight", 1)):
            raise ValueError(f"Gandalf result criterion {index} differs from canonical rubric")
        if not isinstance(row.get("met"), bool):
            raise ValueError(f"Gandalf criterion {index} has an invalid decision")

    if model is None:
        model = _read_grader_model(bundle.trial_dir) or _grader_model_from_dataset(bundle.task_dir) or ""

    criteria = [
        CriterionVerdict(
            index=i,
            satisfied=bool(cr["met"]),
            reason=str(cr.get("reasoning", "")),
        )
        for i, cr in enumerate(info.get("criterion_results", []))
    ]

    usage = info.get("llm_usage", {}) or {}
    tokens = TokenUsage(
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        cached_prompt_tokens=int(usage.get("cache_read_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
    )
    cost_usd = float(usage.get("cost_usd", 0.0) or 0.0)

    return Verdict(
        verifier="gandalf",
        model=model,
        task_id=bundle.task_id,
        gt_task_id=bundle.gt_task_id,
        rollout_id=bundle.trial_dir.name,
        cohort=bundle.cohort,
        criteria=criteria,
        tokens=tokens,
        cost_usd=cost_usd,
    )


# ---------------------------------------------------------------------------
# grader.toml model extraction (best-effort)
# ---------------------------------------------------------------------------

_MODEL_TOML_RE = re.compile(r'^\s*model\s*=\s*"([^"]+)"', re.MULTILINE)


def _read_grader_model(trial_dir: Path) -> str | None:
    """Look in known places inside the trial dir for grader.toml's model."""
    for c in (trial_dir / "verifier" / "grader.toml", trial_dir / "tests" / "grader.toml"):
        if c.is_file():
            m = _MODEL_TOML_RE.search(c.read_text())
            if m:
                return m.group(1)
    return None


def _grader_model_from_dataset(task_dir: Path) -> str | None:
    """Fallback: read the grader model from the dataset task dir.

    BTB tasks declare it in ``tests/grader.toml`` as ``model = "..."``.
    """
    btb_path = task_dir / "tests" / "grader.toml"
    if btb_path.is_file():
        m = _MODEL_TOML_RE.search(btb_path.read_text())
        if m:
            return m.group(1)
    return None


def _copy_gandalf_trace(verifier_dir: Path, trace_dir: Path) -> None:
    """Copy gandalf's native verifier output into ``trace_dir``.

    Best-effort: missing files are skipped silently (different gandalf
    configs produce different file sets — e.g. single-batch runs have
    no ``judge_trace_batch_split1.txt``). Any I/O error reading the
    source dir is non-fatal — the verdict has already been parsed by
    the caller. Failures here would lose post-hoc inspection but not
    grading accuracy, so we don't want to abort the run on them.
    """
    trace_dir.mkdir(parents=True, exist_ok=True)
    for fname in _GANDALF_TRACE_FILES:
        src = verifier_dir / fname
        if src.is_file():
            try:
                shutil.copy2(src, trace_dir / fname)
            except OSError:
                pass
    for pattern in _GANDALF_TRACE_GLOBS:
        for src in verifier_dir.glob(pattern):
            try:
                shutil.copy2(src, trace_dir / src.name)
            except OSError:
                pass
