"""Agent-as-a-Judge worker — JSON-in / JSON-out wrapper.

Runs in this directory's uv venv (which has the locally-modified
agent-as-a-judge as an editable install plus all of its heavy
dependencies). Invoked from the eval harness via:

    uv run --project verifier-eval/verifiers/aaaj/ worker.py args.json result.json

The runner passes paths to a *staged* workspace + trajectory + the
synthesized rubric instance file; the worker symlinks them into a
private working directory, drives ``JudgeAgent.judge_anything``, then
emits the resulting requirements + per-criterion ``judge_stats`` as
JSON for the runner to translate.

``args.json`` schema:

    {
        "task_id": str,
        "rubric": [{"criterion": str, "category": str, "weight": float}, ...],
        "instructions": str,
        "workspace_dir": str,            # absolute path
        "trajectory_path": str | null,   # absolute path; may be missing
        "model": str,                    # LiteLLM model id
        "setting": "gray_box" | "black_box",
        "planning": "efficient (no planning)" | "comprehensive (no planning)" | "planning",
        "max_workers": int,
        "majority_vote": int,
        "trace_path": str | null         # if set, copy AAAJ's full
                                         # judgment.json (per-criterion
                                         # judge_stats incl. retrieval +
                                         # tool-call traces) here for
                                         # post-hoc inspection
    }

``result.json`` schema:

    {
        "criteria": [
            {
                "criterion": str,
                "satisfied": bool | null,    # null = AAAJ couldn't decide
                "reason": str,
                "input_tokens": int,
                "output_tokens": int,
                "cost": float
            },
            ...
        ],
        "total_input_tokens": int,
        "total_output_tokens": int,
        "total_cost": float
    }

On failure: nonzero exit, traceback to stderr.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Install the shared LiteLLM call recorder before any AAAJ import pulls
# in litellm. Activated by VERIFIER_TRACE_LLM_CALLS env var; no-op
# otherwise. See verifiers/_shared/litellm_trace.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
try:
    import litellm_trace  # type: ignore[import-not-found]
    litellm_trace.install()
except ImportError:
    pass


def _atif_to_aaaj_trajectory(atif_path: Path | None) -> list[dict]:
    """Convert ATIF to AAAJ's step-array format. Empty list if no trajectory."""
    if atif_path is None or not atif_path.is_file():
        return []
    atif = json.loads(atif_path.read_text())
    out: list[dict] = []
    for i, step in enumerate(atif.get("steps", []) or []):
        source = step.get("source", "")
        message = step.get("message", "")
        if not message and not step.get("tool_calls") and not step.get("observation"):
            continue
        entry: dict = {"step": i}
        if source == "user":
            entry["user_message"] = message
            entry["agent"] = {"agent_name": "user", "thought": "", "action": ""}
        else:
            entry["user_message"] = ""
            entry["agent"] = {
                "agent_name": source or "agent",
                "thought": str(message) + "\n" + str(step.get("reasoning_content", "")),
                "action": json.dumps(step.get("tool_calls", []), ensure_ascii=False),
            }
        entry["environment"] = json.dumps(step.get("observation", {}), ensure_ascii=False)
        out.append(entry)
    return out


def _build_instance(args: dict) -> dict:
    requirements = []
    for i, item in enumerate(args["rubric"]):
        requirements.append({
            "requirement_id": i,
            "criteria": item["criterion"],
            "category": item.get("category", ""),
            "satisfied": None,
        })
    return {
        "name": args["task_id"],
        "query": args["instructions"],
        "requirements": requirements,
    }


def _judgment_to_result(judgment: dict, n_rubric: int) -> dict:
    stats_by_index: dict[int, dict] = {}
    for js in judgment.get("judge_stats", []) or []:
        idx = js.get("requirement_index")
        if idx is not None:
            stats_by_index[idx] = js

    criteria_out: list[dict] = []
    total_in = 0
    total_out = 0
    total_cost = 0.0
    for i in range(n_rubric):
        req = (judgment.get("requirements") or [{}] * n_rubric)[i]
        stat = stats_by_index.get(i, {})
        llm_stats = stat.get("llm_stats", {}) or {}

        satisfied = req.get("satisfied")
        if satisfied is None:
            satisfied = stat.get("satisfied")
        # Preserve null when AAAJ couldn't decide; runner maps null → False.
        sat_out: bool | None = bool(satisfied) if satisfied is not None else None

        reason = llm_stats.get("reason", "") or req.get("reason", "")
        if isinstance(reason, list):
            reason = "\n---\n".join(str(r) for r in reason)

        ip = int(llm_stats.get("input_tokens", 0) or 0)
        op = int(llm_stats.get("output_tokens", 0) or 0)
        c = float(llm_stats.get("cost", 0.0) or 0.0)
        total_in += ip
        total_out += op
        total_cost += c

        criteria_out.append({
            "criterion": req.get("criteria", ""),
            "satisfied": sat_out,
            "reason": str(reason or ""),
            "input_tokens": ip,
            "output_tokens": op,
            "cost": c,
        })

    return {
        "criteria": criteria_out,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cost": total_cost,
    }


def _patch_aaaj_for_tagging() -> None:
    """Wrap AAAJ's per-requirement / per-phase methods in litellm_trace.tag().

    The 3-call pattern per criterion is: locate_file → llm_summary
    (trajectory) → ask.check (judge). We tag each so the resulting JSONL
    is self-describing. ``_evaluate_requirement`` adds the
    ``criterion_index`` + ``criterion`` for the whole pack. Idempotent.
    """
    try:
        import litellm_trace  # type: ignore[import-not-found]
    except ImportError:
        return
    from agent_as_a_judge.agent import JudgeAgent  # type: ignore[import-not-found]
    from agent_as_a_judge.module.ask import DevAsk  # type: ignore[import-not-found]
    from agent_as_a_judge.module.locate import DevLocate  # type: ignore[import-not-found]
    from agent_as_a_judge.module.text_retrieve import (  # type: ignore[import-not-found]
        DevTextRetrieve,
    )

    def _wrap(cls, name: str, **tag_kv: object) -> None:
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, "_tagged", False):
            return

        def wrapper(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            # Resolve dynamic tags (criterion text from the first
            # positional arg for criterion-aware methods).
            dyn = dict(tag_kv)
            if dyn.get("_criterion_arg") and args:
                dyn["criterion"] = str(args[0])[:300]
                dyn.pop("_criterion_arg")
            with litellm_trace.tag(**dyn):
                return orig(self, *args, **kwargs)

        wrapper._tagged = True  # type: ignore[attr-defined]
        setattr(cls, name, wrapper)

    def _wrap_evaluate(cls) -> None:
        orig = getattr(cls, "_evaluate_requirement", None)
        if orig is None or getattr(orig, "_tagged", False):
            return

        def wrapper(self, i, requirement, user_query):  # type: ignore[no-untyped-def]
            criterion = (
                requirement.get("criteria") if isinstance(requirement, dict) else None
            )
            with litellm_trace.tag(
                criterion_index=int(i),
                criterion=str(criterion or "")[:300],
            ):
                return orig(self, i, requirement, user_query)

        wrapper._tagged = True  # type: ignore[attr-defined]
        setattr(cls, "_evaluate_requirement", wrapper)

    _wrap_evaluate(JudgeAgent)
    _wrap(DevLocate, "locate_file", phase="locate", _criterion_arg=True)
    _wrap(DevTextRetrieve, "llm_summary", phase="retrieve", _criterion_arg=True)
    _wrap(DevAsk, "check", phase="judge", _criterion_arg=True)


def _run(args: dict) -> dict:
    from model_runtime import configure
    configure()
    # Lazy imports — the eval-harness venv doesn't have these.
    from agent_as_a_judge.agent import JudgeAgent  # type: ignore[import-not-found]
    from agent_as_a_judge.config import AgentConfig  # type: ignore[import-not-found]

    _patch_aaaj_for_tagging()

    # AAAJ inspects DEFAULT_LLM (the judge model id) and OPENAI_API_KEY
    # for embeddings (text-embedding-3-large). The judge model itself is
    # routed through litellm — when the model is non-OpenAI, AAAJ's
    # patched LLM-construction sites (agent.py, module/{locate,planning,
    # text_retrieve}.py) pass api_key=None, letting litellm pull the
    # right provider key (GEMINI_API_KEY, ANTHROPIC_API_KEY, …) from env.
    # So we leave OPENAI_API_KEY alone here — keeping it valid for the
    # OpenAI embedding model — and rely on the patched judge-LLM path.
    os.environ["DEFAULT_LLM"] = args["model"]
    if not os.environ.get("OPENAI_API_KEY"):
        for fallback in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "LLM_API_KEY"):
            if os.environ.get(fallback):
                os.environ["OPENAI_API_KEY"] = os.environ[fallback]
                break

    instance_data = _build_instance(args)
    aaaj_trajectory = _atif_to_aaaj_trajectory(
        Path(args["trajectory_path"]) if args.get("trajectory_path") else None
    )
    workspace_dir = Path(args["workspace_dir"]).resolve()

    with tempfile.TemporaryDirectory(prefix="aaaj_worker_") as tmpdir:
        tmp = Path(tmpdir)
        instance_path = tmp / f"{args['task_id']}.json"
        instance_path.write_text(json.dumps(instance_data, indent=2))
        trajectory_path = tmp / f"{args['task_id']}_trajectory.json"
        trajectory_path.write_text(json.dumps(aaaj_trajectory, indent=2))

        judge_dir = tmp / "judge_workspace"
        judge_dir.mkdir(parents=True, exist_ok=True)

        ws_link = tmp / "ws"
        os.symlink(workspace_dir, ws_link)

        config = AgentConfig(
            include_dirs=None,
            exclude_dirs=["__pycache__", ".git", "venv", "env"],
            exclude_files=[".DS_Store"],
            setting=args.get("setting", "gray_box"),
            planning=args.get("planning", "efficient (no planning)"),
            judge_dir=judge_dir,
            workspace_dir=ws_link.parent,
            instance_dir=tmp,
            trajectory_file=trajectory_path.parent,
            majority_vote=args.get("majority_vote", 1),
            critical_threshold=0.5,
            max_workers=args.get("max_workers", 1),
        )

        prev_cwd = os.getcwd()
        try:
            os.chdir(ws_link)
            judge = JudgeAgent(
                workspace=ws_link,
                instance=instance_path,
                judge_dir=judge_dir,
                trajectory_file=trajectory_path,
                config=config,
            )
            judge.judge_anything()
            judgment_file = judge_dir / instance_path.name
            if judgment_file.exists():
                judgment = json.loads(judgment_file.read_text())
            else:
                judgment = {**instance_data, "judge_stats": judge.judge_stats}

            # Persist the rich judgment (per-criterion judge_stats +
            # whatever AAAJ wrote to judge_dir) outside the tempdir so
            # it survives this process. Best-effort — the verdict is
            # already extracted, so I/O failures here shouldn't abort.
            trace_path = args.get("trace_path")
            if trace_path:
                try:
                    out = Path(trace_path)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(judgment, indent=2))
                except OSError:
                    pass
        finally:
            os.chdir(prev_cwd)

    return _judgment_to_result(judgment, n_rubric=len(args["rubric"]))


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <args.json> <result.json>", file=sys.stderr)
        sys.exit(2)
    args_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])

    args = json.loads(args_path.read_text())
    result = _run(args)
    result_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
