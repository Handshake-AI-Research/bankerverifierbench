"""Archipelago grading runner — invokes upstream `runner.main` in its uv venv.

Unlike AAAJ and AutoRubric, archipelago already exposes a usable
subprocess interface: ``python -m runner.main --verifiers ... --output ...``
takes JSON config files and writes a JSON output. So we don't need a
worker.py here — the runner just builds the 5 config JSONs + 2
workspace zips in a tempdir, invokes ``uv run --project
verifiers/archipelago/grading/`` against the upstream CLI, then
translates the result.

Archipelago is *non-agentic*: native artifact selection and criterion judging
use the rubric criterion, frozen task instructions, final answer, and extracted
initial/final workspace changes. The native output_llm workflow does not read
the full intermediate trajectory. We use the locally-modified output_llm/main.py which
surfaces token/cost in `verifier_result_values`.

Threshold convention
--------------------
Archipelago's ``score`` is in ``[0, 1]``. We map ``score > threshold`` to
``satisfied=True`` (default 0.5). Configurable via ``threshold`` kwarg
so calibration choices are reversible without re-running the verifier.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

from ..inputs import InputBundle, validate_bundle
from ..verdict import CriterionVerdict, TokenUsage, Verdict
from ._jsonl_stats import jsonl_stats
from .base import Capabilities


DEFAULT_ARCHIPELAGO_PROJECT = (
    Path(__file__).resolve().parent.parent.parent / "verifiers" / "archipelago" / "grading"
)


def capabilities() -> Capabilities:
    return Capabilities(
        needs_workspace=True,
        needs_trajectory=True,
        accepts_placeholder_trajectory=False,
    )


def invoke(
    bundle: InputBundle,
    *,
    model: str | None = None,
    project_dir: Path | None = None,
    threshold: float = 0.5,
    timeout_sec: int = 1800,
    trace_dir: Path | None = None,
    verbose: bool = False,
    litellm_version: str | None = None,
) -> Verdict:
    """Run archipelago grading on a bundle; return a Verdict.

    When ``trace_dir`` is set, archipelago's full ``output.json`` (the
    pre-translation, per-verifier-result structure with all rationales
    + evaluated artifact lists) is copied to
    ``trace_dir/output.json`` for post-hoc inspection.
    """
    validate_bundle(bundle)
    if bundle.workspace_dir is None or bundle.trajectory_path is None:
        raise FileNotFoundError(
            f"Archipelago requires workspace + trajectory; "
            f"bundle for {bundle.task_id} missing one or both."
        )

    project_dir = project_dir or DEFAULT_ARCHIPELAGO_PROJECT
    if not project_dir.is_dir():
        raise FileNotFoundError(
            f"archipelago grading project not found: {project_dir}"
        )

    effective_model = model or _grader_model_from_bundle(bundle) or ""

    verifiers = _rubric_to_archipelago_verifiers(bundle)
    eval_configs = [{
        "eval_config_id": "ec_output_llm",
        "eval_config_name": "Output LLM Verifier",
        "eval_defn_id": "output_llm",
        "eval_config_values": {},
    }]
    scoring_config = {
        "scoring_config_id": "sc_default",
        "scoring_config_name": "Simple Scoring",
        "scoring_defn_id": "template",
        "scoring_config_values": {},
    }
    grading_settings = {
        "llm_judge_model": effective_model,
        "llm_judge_extra_args": None,
    }
    trajectory = _atif_to_archipelago_trajectory(
        bundle.trajectory_path, bundle.instructions
    )

    # Path to the shared LiteLLM call recorder. Injected via PYTHONPATH +
    # a `-c` bootstrap so it's installed before ``runner.main`` pulls
    # litellm in. See verifiers/_shared/litellm_trace.py.
    shared_dir = Path(__file__).resolve().parent.parent.parent / "verifiers" / "_shared"

    output: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="arch_") as tmpdir:
        tmp = Path(tmpdir)

        verifiers_path = tmp / "verifiers.json"
        eval_configs_path = tmp / "eval_configs.json"
        scoring_config_path = tmp / "scoring_config.json"
        grading_settings_path = tmp / "grading_settings.json"
        trajectory_path = tmp / "trajectory.json"
        initial_snapshot = tmp / "initial_snapshot.zip"
        final_snapshot = tmp / "final_snapshot.zip"
        output_path = tmp / "output.json"

        verifiers_path.write_text(json.dumps(verifiers, indent=2))
        eval_configs_path.write_text(json.dumps(eval_configs, indent=2))
        scoring_config_path.write_text(json.dumps(scoring_config, indent=2))
        grading_settings_path.write_text(json.dumps(grading_settings, indent=2))
        trajectory_path.write_text(json.dumps(trajectory, indent=2))

        initial_workspace = bundle.task_dir / "environment" / "input"
        if bundle.input_identity and not initial_workspace.is_dir():
            raise FileNotFoundError("Archipelago requires the actual initial task workspace")
        _zip_workspace(initial_workspace, initial_snapshot)
        _zip_workspace(bundle.workspace_dir, final_snapshot)
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            for path in (verifiers_path, eval_configs_path, scoring_config_path,
                         grading_settings_path, trajectory_path):
                (trace_dir / path.name).write_bytes(path.read_bytes())
            (trace_dir / "snapshot-inputs.json").write_text(json.dumps({
                "initial_workspace": str(initial_workspace.resolve()),
                "final_workspace": str(bundle.workspace_dir.resolve()),
                "initial_files": zipfile.ZipFile(initial_snapshot).namelist(),
                "final_files": zipfile.ZipFile(final_snapshot).namelist(),
            }, indent=2))

        # Bootstrap: archipelago_bootstrap.py (in verifiers/_shared/, on
        # PYTHONPATH below) installs the LiteLLM recorder + per-criterion
        # tagger, then runs runner.main as __main__. The tagger wraps
        # output_llm.llm_judge_eval so every litellm call carries the
        # criterion identity — the ATIF normalizer groups by these tags.
        cmd = [
            "uv", "run", "--project", str(project_dir),
            "python", "-m", "archipelago_bootstrap",
            "--grading-run-id", str(uuid.uuid4()),
            "--trajectory-id", str(uuid.uuid4()),
            "--initial-snapshot", str(initial_snapshot),
            "--final-snapshot", str(final_snapshot),
            "--trajectory", str(trajectory_path),
            "--grading-settings", str(grading_settings_path),
            "--verifiers", str(verifiers_path),
            "--eval-configs", str(eval_configs_path),
            "--scoring-config", str(scoring_config_path),
            "--output", str(output_path),
        ]
        env = {**os.environ}
        from ._runtime import configure_worker
        configure_worker(cmd, env, litellm_version)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{shared_dir}{os.pathsep}{existing_pp}" if existing_pp else str(shared_dir)
        )
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            env["VERIFIER_TRACE_LLM_CALLS"] = str((trace_dir / "llm_calls.jsonl").resolve())
        proc = subprocess.run(
            cmd, cwd=project_dir,
            env=env,
            timeout=timeout_sec, check=False,
            capture_output=not verbose, text=True,
        )
        if trace_dir is not None:
            (trace_dir / "worker-stdout.txt").write_text(proc.stdout or "")
            (trace_dir / "worker-stderr.txt").write_text(proc.stderr or "")
        if proc.returncode != 0:
            if not verbose:
                sys.stderr.write(proc.stderr or "")
            raise RuntimeError(
                f"archipelago runner exited {proc.returncode} for "
                f"{bundle.task_id}; see stderr above"
            )
        if not output_path.is_file():
            raise RuntimeError(
                f"archipelago runner produced no output.json for {bundle.task_id}"
            )
        output = json.loads(output_path.read_text())

        # Persist the raw archipelago output before the tempdir is
        # cleaned up. Best-effort — verdict has already been parsed.
        if trace_dir is not None:
            try:
                trace_dir.mkdir(parents=True, exist_ok=True)
                (trace_dir / "output.json").write_text(output_path.read_text())
            except OSError:
                pass

    verdict = _output_to_verdict(output, bundle, effective_model, threshold)
    # Override tokens/cost from JSONL when present — keeps reporting
    # consistent across verifiers. archipelago's own totals are already
    # correct, so this is normally a no-op semantically.
    if trace_dir is not None:
        stats = jsonl_stats(trace_dir / "llm_calls.jsonl")
        if stats is not None:
            update: dict[str, Any] = {"tokens": stats.tokens}
            if stats.cost_usd is not None:
                update["cost_usd"] = stats.cost_usd
            verdict = verdict.model_copy(update=update)
    return verdict


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _rubric_to_archipelago_verifiers(bundle: InputBundle) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, item in enumerate(bundle.rubric):
        out.append({
            "verifier_id": f"ver_{i:04d}",
            "verifier_version": 1,
            "world_id": None,
            "task_id": bundle.gt_task_id,
            "eval_config_id": "ec_output_llm",
            "verifier_values": {
                "criteria": item["criterion"],
                "is_primary_objective": True,
                "btb_weight": item.get("weight", 1),
                "btb_category": item.get("category", ""),
            },
            "verifier_index": i,
            "verifier_dependencies": None,
        })
    return out


def _atif_to_archipelago_trajectory(atif_path: Path, instructions: str) -> dict[str, Any]:
    """Convert ATIF to archipelago's AgentTrajectoryOutput shape.

    Prepends task instructions as the first user message so archipelago's
    ``extract_task_prompt()`` can find them. Last assistant message
    becomes the ``output.text``.
    """
    atif = json.loads(atif_path.read_text())
    messages: list[dict[str, str]] = [{"role": "user", "content": instructions}]
    for step in atif.get("steps", []):
        source = step.get("source", "")
        evidence = {key: step[key] for key in (
            "message", "reasoning_content", "tool_calls", "observation"
        ) if key in step}
        message = json.dumps(evidence, ensure_ascii=False) if evidence else ""
        if not message:
            continue
        if source == "user":
            role = "user"
        elif source == "agent":
            role = "assistant"
        else:
            role = "system"
        messages.append({"role": role, "content": message})

    output_text = ""
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            # The last assistant text is the output; structured tool evidence
            # remains in messages rather than replacing the final response.
            evidence = json.loads(msg["content"])
            output_text = str(evidence.get("message", ""))
            break

    return {
        "messages": messages,
        "output": {"text": output_text} if output_text else None,
        "status": "completed",
        "time_elapsed": 0.0,
    }


def _write_empty_snapshot(zip_path: Path) -> None:
    """Write a valid ZIP containing only a placeholder under ``filesystem/``."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("filesystem/.empty", "")


def _zip_workspace(source_dir: Path, zip_path: Path) -> None:
    """Zip ``source_dir`` under a ``filesystem/`` prefix."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        wrote_any = False
        if source_dir.is_dir():
            for root, _dirs, files in os.walk(source_dir):
                for fname in files:
                    full = Path(root) / fname
                    arc = Path("filesystem") / full.relative_to(source_dir)
                    zf.write(full, str(arc))
                    wrote_any = True
        if not wrote_any:
            zf.writestr("filesystem/.empty", "")


def _output_to_verdict(
    output: dict[str, Any], bundle: InputBundle, model: str, threshold: float
) -> Verdict:
    """Translate archipelago's output.json into a Verdict.

    Sort by ``verifier_index`` (we set it during config generation) before
    zipping with the rubric.
    """
    results = list(output.get("verifier_results", []))

    def _index_of(vr: dict[str, Any]) -> int:
        if "verifier_index" in vr:
            return int(vr["verifier_index"])
        vid = vr.get("verifier_id", "ver_0000")
        try:
            return int(vid.split("_")[-1])
        except ValueError:
            return 0

    results.sort(key=_index_of)
    if bundle.input_identity and [_index_of(r) for r in results] != list(range(len(bundle.rubric))):
        raise ValueError('Archipelago criterion identities are missing, duplicated or out of range')

    criteria: list[CriterionVerdict] = []
    total_in = 0
    total_cached = 0
    total_out = 0
    total_cost = 0.0
    for i, vr in enumerate(results):
        score = vr.get("score")
        score_f = float(score) if score is not None else None
        satisfied = (score_f is not None) and (score_f > threshold)

        vals = vr.get("verifier_result_values", {}) or {}
        # The locally-modified output_llm/main.py surfaces these.
        # If running against unmodified upstream, they'll all be 0.
        total_in += int(vals.get("prompt_tokens", 0) or 0)
        total_cached += int(vals.get("cache_read_tokens", 0) or 0)
        total_out += int(vals.get("completion_tokens", 0) or 0)
        total_cost += float(vals.get("cost_usd", 0) or 0)

        # Archipelago's output_llm eval stores the judge rationale under
        # ``grade_rationale`` (matches their old field name; preserved by
        # local mods). Fall back to other plausible keys for
        # future-proofing if the upstream renames it.
        reason = (
            vals.get("grade_rationale")
            or vals.get("reasoning")
            or vr.get("reason")
            or ""
        )
        criteria.append(CriterionVerdict(
            index=i,
            satisfied=satisfied,
            confidence=score_f,
            reason=str(reason),
            raw_verdict=f"score={score_f:.4f}" if score_f is not None else "score=null",
        ))

    return Verdict(
        verifier="archipelago",
        model=model,
        task_id=bundle.task_id,
        gt_task_id=bundle.gt_task_id,
        rollout_id=bundle.trial_dir.name if bundle.trial_dir else None,
        cohort=bundle.cohort,
        criteria=criteria,
        tokens=TokenUsage(
            prompt_tokens=total_in,
            cached_prompt_tokens=total_cached,
            completion_tokens=total_out,
        ),
        cost_usd=total_cost,
    )


def _grader_model_from_bundle(bundle: InputBundle) -> str | None:
    from ..inputs import _read_grader_toml
    return _read_grader_toml(bundle.task_dir).get("model") or None
