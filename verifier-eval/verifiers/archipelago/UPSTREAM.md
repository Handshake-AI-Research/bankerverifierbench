# Archipelago grading — provenance & local modifications

## Source

Snapshot of [`Mercor-Intelligence/archipelago`](https://github.com/Mercor-Intelligence/archipelago) at commit:

```
3f4a8234a27d71a17e0aa19e60f116440bcd1481
```

…re-taken on **2026-05-01**.

Only the `grading/` subtree is snapshotted — the upstream repo also
contains `agents/`, `environment/`, `examples/`, `mcp_servers/`, none of
which are needed to grade staged BTB rollouts.

## Local modifications

The vendored copy has **4 modified source files** (plus a regenerated
`uv.lock`) on top of upstream sha `3f4a8234`, none of them committed
upstream.

| File | Change |
|---|---|
| `grading/runner/main.py` | `VERIFIER_CONCURRENCY_LIMIT` lowered from 15 to 10 |
| `grading/runner/evals/output_llm/main.py` | Adds prompt/cache/completion token + `cost_usd` aggregation across the JSON-validation retry loop, surfaces them in `verifier_result_values`. Without this change all per-archipelago token/cost metrics would be zero. |
| `grading/runner/evals/output_llm/utils/services/artifact_reference.py` | Adds a content-addressed (sha256-keyed) in-memory cache for reference-artifact extractions, with per-key `asyncio.Lock`s so concurrent criteria referencing the same golden file share a single Reducto round-trip instead of each issuing their own. Cache hits return a deep copy so per-caller mutation (truncation, image capping) doesn't bleed across criteria. Pure additive, no behaviour change on miss. |
| `grading/runner/utils/file_extraction/factory.py` | Restored to `os.getenv("REDUCTO_API_KEY")` (set this env var in `.env` before running archipelago grading). |
| `grading/uv.lock` | Regenerated for the local Python; will be regenerated again on first `uv sync` in this snapshot, so the diff isn't load-bearing. |

The three genuine code improvements (concurrency cap + token/cost
aggregation + reference-extraction cache) are preserved in the snapshot.

See `local-modifications.patch` for the verbatim diff against
upstream sha `3f4a8234`.
