"""Agent-as-a-Judge runner — invokes worker.py in the AAAJ uv venv.

The worker (at ``verifiers/aaaj/worker.py``) runs in its own venv that
has the locally-modified agent-as-a-judge source installed (editable)
plus AAAJ's heavy dep tree (litellm, spacy, sentence-transformers,
torch, etc.). We pass it JSON in / JSON out via two tempfiles, then
translate the raw structure into a canonical :class:`Verdict`.

AAAJ uses OpenAI ``text-embedding-3-large`` for retrieval (or whatever
``AAAJ_EMBEDDING_MODEL`` is set to in the locally-modified
``embeddings.py``). ``OPENAI_API_KEY`` should be set in the host env;
the worker falls back to GEMINI/ANTHROPIC keys when OPENAI is missing.
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


DEFAULT_AAAJ_PROJECT = (
    Path(__file__).resolve().parent.parent.parent / "verifiers" / "aaaj"
)


# Last-resort fallback when the orchestrator passes ``model=None`` (e.g.
# when this runner is invoked directly outside of ``evaluate``). Normal
# evaluate-driven runs always receive an explicit model via
# ``evaluate.DEFAULT_MODEL`` (currently ``gemini/gemini-3-flash-preview``)
# unless the user supplied ``--model``.
DEFAULT_MODEL = "gemini/gemini-3-flash-preview"


# AAAJ uses an embedding model for trajectory retrieval (DevTextRetrieve)
# and code search. The published runs all used OpenAI's
# text-embedding-3-large. With the env var unset, AAAJ falls back to a
# local sentence-transformers model (all-MiniLM-L6-v2) — which is free
# but produces lower-quality retrieval and isn't comparable to the
# reference numbers. We set the env var on the worker subprocess so
# AAAJ's embeddings.py picks the LiteLLM path. Override by exporting
# AAAJ_EMBEDDING_MODEL before invoking evaluate; unset to force the
# local fallback.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


def capabilities() -> Capabilities:
    return Capabilities(
        needs_workspace=True,        # JudgeAgent reads files from workspace
        needs_trajectory=False,      # gray-box mode primarily inspects files
        accepts_placeholder_trajectory=True,
    )


def invoke(
    bundle: InputBundle,
    *,
    model: str | None = None,
    setting: str = "gray_box",
    planning: str = "efficient (no planning)",
    max_workers: int = 1,
    majority_vote: int = 1,
    project_dir: Path | None = None,
    timeout_sec: int = 3600,
    trace_dir: Path | None = None,
    verbose: bool = False,
    litellm_version: str | None = None,
) -> Verdict:
    """Run AAAJ on a bundle's workspace + trajectory; return a Verdict.

    When ``trace_dir`` is set, AAAJ's full per-criterion judgment file
    (containing ``judge_stats`` with retrieval + LLM-call details) is
    written to ``trace_dir/judgment.json`` for post-hoc inspection.
    """
    validate_bundle(bundle)
    if bundle.workspace_dir is None:
        raise FileNotFoundError(
            f"AAAJ requires a staged workspace; bundle for {bundle.task_id} has none."
        )

    project_dir = project_dir or DEFAULT_AAAJ_PROJECT
    worker = project_dir / "worker.py"
    if not worker.is_file():
        raise FileNotFoundError(f"aaaj worker not found: {worker}")

    # Order: explicit ``model`` arg > module-level DEFAULT_MODEL.
    # We deliberately do NOT fall back to bundle.task_dir's grader.toml —
    # AAAJ's worker should run with whatever the orchestrator chose
    # (normally evaluate.DEFAULT_MODEL when no --model was passed).
    effective_model = model or DEFAULT_MODEL

    trace_path: str | None = None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = str((trace_dir / "judgment.json").resolve())

    args = {
        "task_id": bundle.gt_task_id,        # AAAJ uses the GT id as instance name
        "rubric": bundle.rubric,
        "instructions": bundle.instructions,
        "workspace_dir": str(bundle.workspace_dir.resolve()),
        "trajectory_path": (
            str(bundle.trajectory_path.resolve())
            if bundle.trajectory_path is not None else None
        ),
        "model": effective_model,
        "setting": setting,
        "planning": planning,
        "max_workers": max_workers,
        "majority_vote": majority_vote,
        "trace_path": trace_path,
        "litellm_version": litellm_version,
    }
    raw = _run_worker(project_dir, worker, args, timeout_sec, verbose=verbose)
    verdict = _raw_to_verdict(raw, bundle, effective_model)
    # AAAJ's worker totals only cover the judge phase due to a
    # ``dict.update`` accumulation bug upstream — see check_requirement
    # in agent_as_a_judge/agent.py. Recompute from the JSONL when we
    # have it, so locate + retrieve phases are included.
    if trace_dir is not None:
        verdict = _override_with_jsonl(verdict, trace_dir / "llm_calls.jsonl")
    return verdict


def _run_worker(
    project_dir: Path, worker: Path, args: dict[str, Any], timeout_sec: int,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Shell out to the worker; return its parsed result.json.

    When ``verbose`` is True, the worker's stdout/stderr stream live to
    the parent's stdout/stderr (no capture). Useful for watching AAAJ's
    per-criterion progress. The trade-off: on failure we can't include
    stderr in the raised exception, since it's already on the user's
    terminal.
    """
    with tempfile.TemporaryDirectory(prefix="aaaj_") as tmp:
        args_path = Path(tmp) / "args.json"
        result_path = Path(tmp) / "result.json"
        args_path.write_text(json.dumps(args))

        cmd = [
            "uv", "run", "--project", str(project_dir),
            "python", str(worker), str(args_path), str(result_path),
        ]
        env = {**os.environ}
        from ._runtime import configure_worker
        configure_worker(cmd, env, args.get("litellm_version"))
        # AAAJ_EMBEDDING_MODEL drives DevTextRetrieve's embedding choice
        # (see verifiers/aaaj/agent_as_a_judge/module/embeddings.py).
        # Default to OpenAI's text-embedding-3-large to match the
        # reference setup; respect a host-shell override.
        env.setdefault("AAAJ_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        # Worker reads VERIFIER_TRACE_LLM_CALLS at startup; if set, every
        # litellm.completion / acompletion call is appended to the file.
        # See verifiers/_shared/litellm_trace.py.
        trace_path_str = args.get("trace_path")
        if trace_path_str:
            trace_dir_path = Path(trace_path_str).parent
            trace_dir_path.mkdir(parents=True, exist_ok=True)
            env["VERIFIER_TRACE_LLM_CALLS"] = str(
                (trace_dir_path / "llm_calls.jsonl").resolve()
            )
            (trace_dir_path / "worker-inputs.json").write_text(json.dumps(args, indent=2))
        proc = subprocess.run(
            cmd, env=env,
            capture_output=not verbose, text=True,
            timeout=timeout_sec, check=False,
        )
        if trace_path_str:
            (trace_dir_path / "worker-stdout.txt").write_text(proc.stdout or "")
            (trace_dir_path / "worker-stderr.txt").write_text(proc.stderr or "")
            if result_path.is_file():
                (trace_dir_path / "worker-result.json").write_text(result_path.read_text())
        if proc.returncode != 0:
            if not verbose:
                sys.stderr.write(proc.stderr or "")
            raise RuntimeError(
                f"aaaj worker exited {proc.returncode}; "
                f"see stderr {'above' if verbose else 'above'}"
            )
        if not result_path.is_file():
            raise RuntimeError(
                f"aaaj worker produced no result.json at {result_path}"
            )
        return json.loads(result_path.read_text())


def _raw_to_verdict(
    raw: dict[str, Any], bundle: InputBundle, model: str
) -> Verdict:
    if bundle.input_identity:
        rows = raw.get("criteria", [])
        if len(rows) != len(bundle.rubric) or any(row.get("criterion") != expected["criterion"] for row, expected in zip(rows, bundle.rubric)):
            raise ValueError("AAAJ native criteria differ from canonical human rubric")
    criteria: list[CriterionVerdict] = []
    for i, c in enumerate(raw.get("criteria", []) or []):
        sat = c.get("satisfied")
        # AAAJ may emit None when its tool loop couldn't decide. Map to
        # False (UNMET) — the same policy AutoRubric uses for
        # CANNOT_ASSESS, so precision/recall are comparable across
        # all verifiers.
        satisfied_bool = bool(sat) if sat is not None else False
        criteria.append(CriterionVerdict(
            index=i,
            satisfied=satisfied_bool,
            reason=str(c.get("reason", "") or ""),
            raw_verdict=(
                ("MET" if satisfied_bool else "UNMET")
                if sat is not None else "UNDETERMINED"
            ),
        ))

    return Verdict(
        verifier="aaaj",
        model=model,
        task_id=bundle.task_id,
        gt_task_id=bundle.gt_task_id,
        rollout_id=bundle.trial_dir.name if bundle.trial_dir else None,
        cohort=bundle.cohort,
        criteria=criteria,
        tokens=TokenUsage(
            prompt_tokens=int(raw.get("total_input_tokens", 0) or 0),
            cached_prompt_tokens=0,    # AAAJ doesn't report cache hits
            completion_tokens=int(raw.get("total_output_tokens", 0) or 0),
        ),
        cost_usd=float(raw.get("total_cost", 0.0) or 0.0),
    )


def _override_with_jsonl(verdict: Verdict, jsonl_path: Path) -> Verdict:
    """Replace verdict.tokens (and cost_usd if known) with JSONL totals.

    No-op when the JSONL is missing/empty. When the JSONL has cost data
    for every record, override cost too; otherwise keep the worker's
    cost (AAAJ's per-call ``litellm.completion_cost`` is at least
    judge-phase-correct, even if undercount).
    """
    stats = jsonl_stats(jsonl_path)
    if stats is None:
        return verdict
    update: dict[str, Any] = {"tokens": stats.tokens}
    if stats.cost_usd is not None:
        update["cost_usd"] = stats.cost_usd
    return verdict.model_copy(update=update)
