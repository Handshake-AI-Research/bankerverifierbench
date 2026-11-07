"""AutoRubric worker — JSON-in / JSON-out wrapper.

Runs in this directory's uv venv (which has ``autorubric==1.0.1``).
Invoked from the eval harness via:

    uv run --project verifier-eval/verifiers/autorubric/ worker.py args.json result.json

``args.json`` schema (written by the verifier_runner.runners.autorubric):

    {
        "rubric_items": [{"weight": float, "requirement": str}, ...],
        "to_grade": str,                     # rendered trajectory text
        "query": str,                        # task instructions
        "model": str,                        # LiteLLM model id
        "temperature": float,
        "max_parallel_requests": int | null,
        "cannot_assess_as_unmet": bool
    }

``result.json`` schema (written here, consumed by the runner):

    {
        "score": float,                      # autorubric's weighted score
        "raw_score": float | null,
        "llm_raw_score": float | null,
        "completion_cost": float | null,
        "cannot_assess_count": int | null,
        "token_usage": {
            "prompt_tokens": int,
            "completion_tokens": int,
            "cache_read_input_tokens": int,
        } | null,
        "criteria": [
            {
                "requirement": str,
                "weight": float,
                "verdict": "MET" | "UNMET" | "CANNOT_ASSESS" | "NULL",
                "reason": str,
                "agreement": float | null     # ensemble agreement, if applicable
            },
            ...
        ]
    }

On failure: nonzero exit, traceback to stderr. Caller surfaces the
stderr text in the resulting Verdict skip / error record.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Install the shared LiteLLM call recorder before autorubric pulls in
# litellm. Activated by VERIFIER_TRACE_LLM_CALLS env var; no-op
# otherwise. See verifiers/_shared/litellm_trace.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
try:
    import litellm_trace  # type: ignore[import-not-found]
    litellm_trace.install()
except ImportError:
    pass


def _verdict_str(verdict_obj) -> str:  # type: ignore[no-untyped-def]
    if verdict_obj is None:
        return "NULL"
    name = getattr(verdict_obj, "name", None)
    return str(name) if name is not None else str(verdict_obj)


def _patch_criterion_grader_for_tagging(CriterionGrader) -> None:  # type: ignore[no-untyped-def]
    """Wrap CriterionGrader._judge_single_criterion in litellm_trace.tag().

    Stamps ``criterion_index``, ``criterion``, and ``phase="judge"`` on
    every litellm call made while grading that criterion. Lets the
    downstream ATIF normalizer group records by criterion without
    parsing the prompt text. No-op if litellm_trace isn't importable.
    """
    try:
        import litellm_trace  # type: ignore[import-not-found]
    except ImportError:
        return
    orig = getattr(CriterionGrader, "_judge_single_criterion", None)
    if orig is None or getattr(orig, "_tagged", False):
        return

    async def tagged(self, judge, criterion, idx, *args, **kwargs):  # type: ignore[no-untyped-def]
        requirement = getattr(criterion, "requirement", None) or str(criterion)
        with litellm_trace.tag(
            criterion_index=idx,
            criterion=str(requirement)[:300],
            phase="judge",
        ):
            return await orig(self, judge, criterion, idx, *args, **kwargs)

    tagged._tagged = True  # type: ignore[attr-defined]
    CriterionGrader._judge_single_criterion = tagged


async def _run(args: dict) -> dict:
    from model_runtime import configure
    configure()
    # Lazy imports — fast --help / fast crash on bad venv.
    from autorubric import (  # type: ignore[import-not-found]
        CannotAssessConfig,
        CannotAssessStrategy,
        LLMConfig,
        Rubric,
    )
    from autorubric.graders import CriterionGrader  # type: ignore[import-not-found]

    _patch_criterion_grader_for_tagging(CriterionGrader)

    rubric = Rubric.from_dict(args["rubric_items"])

    llm_kwargs: dict = {
        "model": args["model"],
        "temperature": args.get("temperature", 0.0),
        # Native LLMConfig options: allow quota windows to reset without
        # changing judge prompts or turning exhausted requests into defects.
        "max_retries": 10,
        "retry_min_wait": 15.0,
        "retry_max_wait": 120.0,
        "timeout": args.get("request_timeout_sec", 60.0),
    }
    if args.get("max_parallel_requests") is not None:
        llm_kwargs["max_parallel_requests"] = args["max_parallel_requests"]
    llm_config = LLMConfig(**llm_kwargs)

    grader_kwargs: dict = {"llm_config": llm_config}
    if args.get("cannot_assess_as_unmet", True):
        grader_kwargs["cannot_assess_config"] = CannotAssessConfig(
            strategy=CannotAssessStrategy.FAIL,
        )
    grader = CriterionGrader(**grader_kwargs)

    result = await rubric.grade(
        to_grade=args["to_grade"],
        grader=grader,
        query=args["query"],
    )

    criteria_out: list[dict] = []
    for cr in (result.report or []):
        if hasattr(cr, "final_verdict"):
            verdict_obj = cr.final_verdict
            reason = cr.final_reason
            requirement = cr.criterion.requirement
            weight = cr.criterion.weight
            agreement = getattr(cr, "agreement", None)
        else:
            verdict_obj = cr.verdict
            reason = cr.reason
            requirement = cr.requirement
            weight = cr.weight
            agreement = None
        criteria_out.append({
            "requirement": requirement,
            "weight": float(weight),
            "verdict": _verdict_str(verdict_obj),
            "reason": str(reason or ""),
            "agreement": float(agreement) if agreement is not None else None,
        })

    usage = result.token_usage
    token_usage = (
        {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "cache_read_input_tokens": int(
                getattr(usage, "cache_read_input_tokens", 0) or 0
            ),
        }
        if usage is not None
        else None
    )

    return {
        "transport_retry_config": {k: llm_kwargs[k] for k in
                                   ('max_retries', 'retry_min_wait', 'retry_max_wait', 'timeout')},
        "score": float(result.score) if result.score is not None else 0.0,
        "raw_score": (
            float(result.raw_score) if result.raw_score is not None else None
        ),
        "llm_raw_score": (
            float(result.llm_raw_score) if result.llm_raw_score is not None else None
        ),
        "completion_cost": (
            float(result.completion_cost) if result.completion_cost is not None else None
        ),
        "cannot_assess_count": (
            int(result.cannot_assess_count)
            if result.cannot_assess_count is not None
            else None
        ),
        "token_usage": token_usage,
        "criteria": criteria_out,
    }


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <args.json> <result.json>", file=sys.stderr)
        sys.exit(2)
    args_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])

    args = json.loads(args_path.read_text())
    result = asyncio.run(_run(args))
    result_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
