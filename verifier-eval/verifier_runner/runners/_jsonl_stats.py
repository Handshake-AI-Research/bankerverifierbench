"""Read a worker's ``llm_calls.jsonl`` and sum tokens + cost.

The JSONL is written by ``verifiers/_shared/litellm_trace.py`` — one
record per LiteLLM chat completion the verifier-agent made. Summing
across records gives the *true* cumulative resource use, which can
differ from a worker-reported total when the upstream verifier
mis-accumulates internally (notably AAAJ, where ``total_llm_stats``
is built with ``dict.update`` and silently overwrites earlier phases).

Usage from a runner::

    stats = jsonl_stats(trace_dir / "llm_calls.jsonl")
    if stats is not None:
        verdict = verdict.model_copy(update={
            "tokens": stats.tokens,
            "cost_usd": stats.cost_usd if stats.cost_usd is not None else verdict.cost_usd,
        })

When ``cost_usd`` is None for any record (LiteLLM didn't recognize the
model), we return None for the cumulative cost — the runner then keeps
the worker-reported number rather than reporting an undercount.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..verdict import TokenUsage


@dataclass(frozen=True)
class JsonlStats:
    n_calls: int
    tokens: TokenUsage
    cost_usd: float | None    # None ⇒ at least one record had no cost data


def jsonl_stats(jsonl_path: Path) -> JsonlStats | None:
    """Sum tokens + cost across every record in ``jsonl_path``.

    Returns None when the file is missing or empty — caller keeps the
    worker-reported totals.

    Token counts are summed from each record's ``response.usage``;
    OpenAI-style ``prompt_tokens_details.cached_tokens`` flows into
    ``cached_prompt_tokens``. Cost comes from the per-call ``cost_usd``
    field that the recorder populates via ``litellm.completion_cost``.
    """
    if not jsonl_path.is_file():
        return None

    n = 0
    p_in = 0
    c_in = 0
    w_in = 0
    p_out = 0
    r_out = 0
    cost_total = 0.0
    cost_unknown = False

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1

            usage = ((rec.get("response") or {}).get("usage") or {})
            p_in += int(usage.get("prompt_tokens") or 0)
            p_out += int(usage.get("completion_tokens") or 0)
            details = usage.get("prompt_tokens_details") or {}
            c_in += int(details.get("cached_tokens") or 0)
            w_in += int(details.get("cache_write_tokens") or details.get("cache_creation_tokens") or 0)
            r_out += int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)

            cost = rec.get("cost_usd")
            if cost is None:
                cost_unknown = True
            else:
                try:
                    cost_total += float(cost)
                except (TypeError, ValueError):
                    cost_unknown = True

    if n == 0:
        return None

    return JsonlStats(
        n_calls=n,
        tokens=TokenUsage(
            prompt_tokens=p_in,
            cached_prompt_tokens=c_in,
            cache_write_prompt_tokens=w_in,
            completion_tokens=p_out,
            reasoning_completion_tokens=r_out,
        ),
        cost_usd=None if cost_unknown else cost_total,
    )
