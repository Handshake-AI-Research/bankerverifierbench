"""AutoRubric runner — invokes worker.py in the autorubric uv venv.

The worker (at ``verifiers/autorubric/worker.py``) runs in its own venv
that pins ``autorubric==1.0.1``. We pass it JSON in / JSON out via two
tempfiles, then translate the raw verdict structure into a canonical
:class:`Verdict`.

CANNOT_ASSESS handling
----------------------
AutoRubric's verdict enum is ``MET`` / ``UNMET`` / ``CANNOT_ASSESS``.
By default we map ``CANNOT_ASSESS`` to ``UNMET`` (i.e. a 0-contribution
failure), keeping precision/recall comparable across all verifiers.
The original verdict is preserved in ``CriterionVerdict.raw_verdict``
so the choice is reversible at scoring time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..inputs import InputBundle, validate_bundle
from ..verdict import CriterionVerdict, TokenUsage, Verdict
from ._jsonl_stats import jsonl_stats
from .base import Capabilities


# Total to_grade text is truncated to this size after assembly.
MAX_TRAJECTORY_BYTES = 1_500_000


# Default location of the per-verifier uv project (next to this file's repo).
DEFAULT_AUTORUBRIC_PROJECT = (
    Path(__file__).resolve().parent.parent.parent / "verifiers" / "autorubric"
)


def capabilities() -> Capabilities:
    return Capabilities(
        needs_workspace=False,       # ignored by design
        needs_trajectory=True,
        accepts_placeholder_trajectory=False,
    )


def invoke(
    bundle: InputBundle,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    cannot_assess_as_unmet: bool = True,
    max_parallel_requests: int | None = None,
    project_dir: Path | None = None,
    timeout_sec: int = 1800,
    request_timeout_sec: float = 60.0,
    trace_dir: Path | None = None,
    verbose: bool = False,
    litellm_version: str | None = None,
) -> Verdict:
    """Run AutoRubric grading on a bundle's trajectory; return a Verdict.

    When ``trace_dir`` is set, the worker's full ``result.json`` (which
    is itself the full per-criterion trace — verdicts, reasons, ensemble
    agreement, token usage) is copied to ``trace_dir/result.json``.
    """
    validate_bundle(bundle)
    if bundle.trajectory_path is None:
        raise FileNotFoundError(
            f"AutoRubric requires a trajectory; bundle for {bundle.task_id} has none."
        )

    project_dir = project_dir or DEFAULT_AUTORUBRIC_PROJECT
    worker = project_dir / "worker.py"
    if not worker.is_file():
        raise FileNotFoundError(f"autorubric worker not found: {worker}")

    effective_model = model or _grader_model_from_bundle(bundle) or ""

    rubric_items = [
        {"weight": float(item.get("weight", 1)), "requirement": item["criterion"]}
        for item in bundle.rubric
    ]
    to_grade = _build_to_grade(bundle.trajectory_path)

    args = {
        "rubric_items": rubric_items,
        "to_grade": to_grade,
        "query": bundle.instructions,
        "model": effective_model,
        "temperature": temperature,
        "max_parallel_requests": max_parallel_requests,
        "request_timeout_sec": request_timeout_sec,
        "cannot_assess_as_unmet": cannot_assess_as_unmet,
        "litellm_version": litellm_version,
    }

    raw = _run_worker(
        project_dir, worker, args, timeout_sec,
        trace_dir=trace_dir, verbose=verbose,
    )
    verdict = _raw_to_verdict(raw, bundle, effective_model, cannot_assess_as_unmet)
    # Override tokens/cost from JSONL when present — keeps reporting
    # consistent across verifiers and gives us a cross-check on the
    # worker totals. autorubric's own totals are already correct, so
    # this is normally a no-op semantically.
    if trace_dir is not None:
        stats = jsonl_stats(trace_dir / "llm_calls.jsonl")
        if stats is not None:
            update: dict[str, Any] = {"tokens": stats.tokens}
            if stats.cost_usd is not None:
                update["cost_usd"] = stats.cost_usd
            verdict = verdict.model_copy(update=update)
    return verdict


# ---------------------------------------------------------------------------
# Worker invocation
# ---------------------------------------------------------------------------

def _run_worker(
    project_dir: Path, worker: Path, args: dict[str, Any], timeout_sec: int,
    *,
    trace_dir: Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Shell out to the worker, returning its parsed JSON result.

    When ``verbose``, the worker's stdout/stderr stream live to the
    parent (autorubric prints a tqdm progress bar there).
    """
    with tempfile.TemporaryDirectory(prefix="autorubric_") as tmp:
        args_path = Path(tmp) / "args.json"
        result_path = Path(tmp) / "result.json"
        args_path.write_text(json.dumps(args))
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / "worker-inputs.json").write_text(json.dumps(args, indent=2))

        cmd = [
            "uv", "run", "--project", str(project_dir),
            "python", str(worker), str(args_path), str(result_path),
        ]
        env = {**os.environ}
        from ._runtime import configure_worker
        configure_worker(cmd, env, args.get("litellm_version"))
        # Worker reads VERIFIER_TRACE_LLM_CALLS at startup; if set, every
        # litellm chat completion call is appended to the file.
        # See verifiers/_shared/litellm_trace.py.
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            env["VERIFIER_TRACE_LLM_CALLS"] = str(
                (trace_dir / "llm_calls.jsonl").resolve()
            )
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=not verbose, text=True,
            timeout=timeout_sec, check=False,
        )
        if trace_dir is not None:
            (trace_dir / "worker-stdout.txt").write_text(proc.stdout or "")
            (trace_dir / "worker-stderr.txt").write_text(proc.stderr or "")
        if proc.returncode != 0:
            if not verbose:
                sys.stderr.write(proc.stderr or "")
            raise RuntimeError(
                f"autorubric worker exited {proc.returncode}; "
                f"see stderr above"
            )
        if not result_path.is_file():
            raise RuntimeError(
                f"autorubric worker produced no result.json at {result_path}"
            )

        result_text = result_path.read_text()
        # Persist the worker's raw output before tempdir cleanup. The
        # autorubric result.json is itself the full per-criterion trace
        # (verdicts, reasons, ensemble agreement, token usage). Best-
        # effort — verdict has already been parsed.
        if trace_dir is not None:
            try:
                trace_dir.mkdir(parents=True, exist_ok=True)
                (trace_dir / "result.json").write_text(result_text)
            except OSError:
                pass

        return json.loads(result_text)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _build_to_grade(atif_path: Path) -> str:
    """Render ATIF trajectory as text."""
    atif = json.loads(atif_path.read_text())
    parts: list[str] = ["# Agent Trajectory\n"]
    for i, step in enumerate(atif.get("steps", []) or []):
        source = step.get("source", "unknown")
        evidence = {key: step[key] for key in (
            "message", "reasoning_content", "tool_calls", "observation"
        ) if key in step}
        if evidence:
            parts.append(f"\n## Step {i} — {source}\n\n"
                         + json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    blob = "".join(parts)
    if len(blob) > MAX_TRAJECTORY_BYTES:
        blob = blob[:MAX_TRAJECTORY_BYTES] + "\n\n... [to_grade truncated]"
    return blob


def _verdict_to_satisfied(verdict_str: str) -> bool:
    """Map AutoRubric's verdict string to canonical bool.

    With ``cannot_assess_as_unmet=True`` (the default), CANNOT_ASSESS / NULL
    are treated as UNMET (False). The original verdict is preserved in
    ``raw_verdict`` so a re-score with a different policy is possible.
    """
    return verdict_str == "MET"


def _raw_to_verdict(
    raw: dict[str, Any], bundle: InputBundle, model: str, cannot_assess_as_unmet: bool
) -> Verdict:
    if bundle.input_identity:
        rows = raw.get("criteria", [])
        if len(rows) != len(bundle.rubric) or any(
            row.get("requirement") != expected["criterion"]
            or float(row.get("weight", 0)) != float(expected["weight"])
            for row, expected in zip(rows, bundle.rubric)
        ):
            raise ValueError("AutoRubric native criteria differ from canonical human rubric")
        if any(row.get("verdict") not in {"MET", "UNMET", "CANNOT_ASSESS"} for row in rows):
            raise ValueError("AutoRubric returned an invalid or missing criterion verdict")
    criteria: list[CriterionVerdict] = []
    for i, c in enumerate(raw.get("criteria", []) or []):
        verdict_str = c.get("verdict", "NULL")
        criteria.append(CriterionVerdict(
            index=i,
            satisfied=_verdict_to_satisfied(verdict_str),
            reason=str(c.get("reason", "") or ""),
            raw_verdict=verdict_str,
        ))

    usage = raw.get("token_usage") or {}
    tokens = TokenUsage(
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        cached_prompt_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
    )

    return Verdict(
        verifier="autorubric",
        model=model,
        task_id=bundle.task_id,
        gt_task_id=bundle.gt_task_id,
        rollout_id=bundle.trial_dir.name if bundle.trial_dir else None,
        cohort=bundle.cohort,
        criteria=criteria,
        tokens=tokens,
        cost_usd=float(raw.get("completion_cost") or 0.0),
    )


def _grader_model_from_bundle(bundle: InputBundle) -> str | None:
    from ..inputs import _read_grader_toml
    return _read_grader_toml(bundle.task_dir).get("model") or None
