# Agent-as-a-Judge — provenance & local modifications

## Source

Snapshot of [`metauto-ai/agent-as-a-judge`](https://github.com/metauto-ai/agent-as-a-judge) at commit:

```
43c2fa7b84ef0c1522e8250576e6798bace29dd0
```

…taken on **2026-05-01**.

## Local modifications

The vendored copy carries **9 modified files + 1 new file** that are
**not committed upstream**. They are preserved both as the snapshotted
source on disk (the working state, applied) and as a patch file for
archival (`local-modifications.patch`).

| File | Change |
|---|---|
| `agent_as_a_judge/agent.py` | 149 lines — judge_anything refactor + parallelism plumbing. Also adds the missing `file=buf` kwarg to the `Console(...)` constructor in the combined-panel capture path so the rendered text actually lands in the buffer. `LLM(...)` constructor in `__init__` passes `api_key=None` instead of `os.getenv("OPENAI_API_KEY")`, letting litellm pick the right provider key per model id. |
| `agent_as_a_judge/config.py` | Added `max_workers: int = 1` config field for criteria-level parallelism |
| `agent_as_a_judge/llm/provider.py` | (a) `num_retries=3 → 10`, `retry_min_wait=1 → 4`, `retry_max_wait=10 → 120` so a single LLM call can ride out a Gemini 429 / RESOURCE_EXHAUSTED quota window; (b) `llm_temperature` and `llm_top_p` defaults flipped from `0.7` / `0.9` to `None`, and they are now only forwarded to litellm when explicitly set, so reasoning models (gpt-5, gemini-3-thinking, …) use provider-side defaults. |
| `agent_as_a_judge/module/ask.py` | Added `ThreadPoolExecutor` for majority-vote parallelism. `_initialize_llm()` passes `api_key=None` so non-OpenAI judge models receive the right provider key. |
| `agent_as_a_judge/module/code_search.py` | Small adjustments (5 lines) |
| `agent_as_a_judge/module/locate.py` | Drops the hard-coded `temperature=1.0` from the `do_completion(...)` call so the opt-in sampling behaviour in `provider.py` takes effect. `_initialize_llm()` passes `api_key=None` (and stops requiring `OPENAI_API_KEY` to be set) so Gemini/Anthropic judge models work. |
| `agent_as_a_judge/module/planning.py` | Drops hard-coded `temperature=1.0` from `do_completion(...)`. `Planning.__init__` passes `api_key=None` to `LLM(...)`. |
| `agent_as_a_judge/module/read.py` | Video-frame and image `LLM(...)` sites use provider routing with `api_key=None`. The image reader's hard-coded OpenAI endpoint was removed after Gemini smoke tests demonstrated seven failed image calls. |
| `agent_as_a_judge/module/text_retrieve.py` | 7 lines — retrieval tweak. Drops hard-coded `temperature=1.0` from `do_completion(...)` so reasoning-model calls don't reject the request. `DevTextRetrieve.__init__` passes `api_key=None` to `LLM(...)`. |
| `agent_as_a_judge/module/embeddings.py` | **New file** — `LiteLLMEmbedder`, drop-in for `SentenceTransformer.encode()` so embeddings can come from any LiteLLM provider (e.g. OpenAI `text-embedding-3-large`) instead of forcing a local sentence-transformers model |

These local modifications are research-grade adaptations needed for the
multi-judge evaluation in this paper; the exact diff is in
`local-modifications.patch`.

## Adjudicated BVB input corrections (2026-10-06)

The efficient workflow explicitly includes the native `user_query` evidence
action, which the comprehensive workflow already used. Otherwise the exact
task prompt supplied by the adapter was omitted from the judge's evidence.
The worker includes all workspace directories in graph construction and
preserves tool calls and observations when converting the reviewed ATIF
trajectory. Native retrieval and evidence truncation limits remain in effect;
these runs are adapted BVB evaluations, not unmodified paper reproductions.

The completion budget is 16,384 tokens per call. A corrected-input Gemini smoke
still produced 43 length-truncated responses out of 207 calls with the native
2,048-token budget: reasoning consumed almost all the budget, leaving partial
file paths and explanations. Those insufficient-budget runs are excluded.

The per-request timeout is 90 seconds, with native retry handling extended to
LiteLLM timeout errors. A 30-second request timeout exhausted retries near the
end of the 207-call smoke test. The outer worker allows 5,400 seconds so native
backoff can finish; transient failures and final responses remain in the trace.
