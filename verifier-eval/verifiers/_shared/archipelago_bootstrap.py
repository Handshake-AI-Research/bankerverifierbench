"""Archipelago bootstrap — install the LiteLLM recorder + per-criterion
tagger, then run ``runner.main`` as ``__main__``.

Invoked from the host runner via:

    python -m archipelago_bootstrap <argv for runner.main>

(``PYTHONPATH`` includes ``verifier-eval/verifiers/_shared/`` so this
module and ``litellm_trace`` are importable.)

The tagger wraps ``runner.evals.output_llm.main.llm_judge_eval`` so
every litellm call inside picks up the (verifier_id, verifier_index,
criterion) of the current criterion. The downstream ATIF normalizer
reads these tags from each JSONL record's ``metadata`` to group
records by criterion without re-parsing prompts.

Two patch points are needed because the registry stores a *bound
reference* to the original function captured at import time. Patching
the module attribute alone wouldn't redirect the registry's call.
"""

from __future__ import annotations


def _patch() -> None:
    from model_runtime import configure
    configure()
    import litellm_trace  # type: ignore[import-not-found]

    litellm_trace.install()

    from runner.evals.output_llm import main as _olm  # type: ignore[import-not-found]
    from runner.evals.registry import EVAL_REGISTRY  # type: ignore[import-not-found]

    orig = _olm.llm_judge_eval

    async def tagged(input):  # type: ignore[no-untyped-def]
        vv = input.verifier.verifier_values or {}
        with litellm_trace.tag(
            verifier_id=getattr(input.verifier, "verifier_id", None),
            criterion_index=getattr(input.verifier, "verifier_index", None),
            criterion=str(vv.get("criteria", ""))[:300],
            phase="judge",
        ):
            return await orig(input)

    _olm.llm_judge_eval = tagged
    eval_defn = EVAL_REGISTRY.get("output_llm")
    if eval_defn is not None:
        eval_defn.eval_impl = tagged


def main() -> None:
    _patch()
    import runpy

    runpy.run_module("runner.main", run_name="__main__")


if __name__ == "__main__":
    main()
