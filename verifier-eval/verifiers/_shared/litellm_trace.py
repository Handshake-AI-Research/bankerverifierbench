"""LiteLLM call recorder — monkey-patches litellm to log every chat
completion to a JSONL file.

Used by the per-verifier worker scripts (aaaj, autorubric, archipelago)
to capture the *verifier-agent's own trajectory* — i.e. every prompt
the judge LLM saw and every completion it produced, so a failure can
be replayed/debugged after the fact.

Activated by setting ``VERIFIER_TRACE_LLM_CALLS`` to a writable file
path before litellm is first imported. The runner sets this env var
to ``trace_dir/llm_calls.jsonl`` when ``trace_dir`` is given.

Each line of the JSONL is one record:

    {
        "ts": "2026-05-03T11:23:04.123456+00:00",
        "fn": "completion" | "acompletion",
        "model": "openai/gpt-5.4-nano",
        "messages": [{"role": "user", "content": "..."}, ...],
        "kwargs": {"temperature": 0.0, ...},   # all non-message kwargs
        "metadata": {"criterion_index": 3, ...},  # tags from tag() context
        "duration_s": 1.234,
        "response": {                          # litellm ModelResponse dict
            "choices": [{"message": {"role": "assistant", "content": "..."}, ...}],
            "usage": {...},
            "model": "...",
            "id": "..."
        } | None,
        "cost_usd": 0.000123 | null,           # litellm.completion_cost(); null on lookup failure
        "error": "..." | None                  # only on failure
    }

The JSONL is the per-call source of truth for cost/token accounting:
verifier runners read it post-hoc to override the worker-reported
totals with cumulative numbers (some upstream verifiers — notably
AAAJ — only report the last phase's stats due to a ``dict.update``
bug in their accumulator).

Both sync (``litellm.completion``) and async (``litellm.acompletion``)
entry points are wrapped. The wrapper never alters call semantics —
on exception we record ``error`` and re-raise.

The wrapper survives litellm's internal retry/router layers because
it patches the public top-level callables.

Tagging
-------
Workers can stamp arbitrary key/value tags onto every call inside a
context, so the JSONL is self-describing for downstream normalization
(e.g. an ATIF post-processor can group by ``criterion_index`` without
re-parsing prompt text). Usage::

    import litellm_trace
    for i, criterion in enumerate(rubric):
        with litellm_trace.tag(criterion_index=i, phase="judge"):
            grade(criterion)   # all litellm calls inside pick up the tag

Tags merge with anything caller passes as ``metadata=`` kwarg on the
litellm call itself. Tags also nest — inner ``tag()`` overrides outer
on key collision, restored on context exit. Context-local, so parallel threads
and asynchronous criterion tasks are isolated.
"""

from __future__ import annotations

import contextlib
import contextvars
import datetime as _dt
import json
import os
import threading
import time
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()
_INSTALLED = False
_TAGS = contextvars.ContextVar("verifier_call_tags", default=())


def _current_tags() -> dict[str, Any]:
    """Return the current execution context's tag stack as a flat dict."""
    stack = _TAGS.get()
    if not stack:
        return {}
    merged: dict[str, Any] = {}
    for frame in stack:
        merged.update(frame)
    return merged


@contextlib.contextmanager
def tag(**kv: Any):
    """Stamp ``kv`` onto every recorded litellm call inside this block.

    Nests cleanly — inner keys shadow outer keys on collision, restored
    on exit. Context-local: parallel graders don't see each other's tags.
    No-op if the recorder isn't installed (so callers can use ``tag()``
    unconditionally without worrying about whether tracing is enabled).
    """
    token = _TAGS.set((*_TAGS.get(), dict(kv)))
    try:
        yield
    finally:
        _TAGS.reset(token)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _redact_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop messages (logged separately) and obvious secrets. Keep everything else."""
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k == "messages":
            continue
        if k in ("api_key", "aws_secret_access_key", "azure_ad_token"):
            out[k] = "<redacted>"
            continue
        try:
            json.dumps(v, default=str)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = repr(v)
    return out


def _response_to_jsonable(resp: Any) -> Any:
    """Best-effort conversion of a LiteLLM ModelResponse to a plain dict.

    Falls back to ``str(resp)`` so we never lose the line entirely.
    """
    if resp is None:
        return None
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(resp, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # pylint: disable=broad-except
                pass
    try:
        json.dumps(resp, default=str)
        return resp
    except (TypeError, ValueError):
        return {"_repr": repr(resp)}


def _write(path: Path, record: dict[str, Any]) -> None:
    line = json.dumps(record, default=str, ensure_ascii=False)
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")


def install() -> bool:
    """Install the recorder. Idempotent. Returns True iff patched.

    Reads ``VERIFIER_TRACE_LLM_CALLS`` lazily so the env var can be set
    after import (the worker entry point sets it before calling install).
    """
    global _INSTALLED  # noqa: PLW0603

    trace_path_str = os.environ.get("VERIFIER_TRACE_LLM_CALLS")
    if not trace_path_str:
        return False
    if _INSTALLED:
        return True

    try:
        import litellm  # type: ignore[import-not-found]
    except ImportError:
        return False

    trace_path = Path(trace_path_str)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    sync_orig = getattr(litellm, "completion", None)
    async_orig = getattr(litellm, "acompletion", None)

    def _cost_for(resp: Any) -> float | None:
        """Best-effort per-call cost via ``litellm.completion_cost``.

        Returns None when LiteLLM doesn't have pricing for the model
        (e.g. an internal proxy alias) or any other failure — the field
        is then `null` in the JSONL and consumers fall back to the
        worker-reported total.
        """
        if resp is None:
            return None
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:  # pylint: disable=broad-except
            return None
        try:
            return float(cost) if cost is not None else None
        except (TypeError, ValueError):
            return None

    def _build_metadata(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge thread-local tag stack with any ``metadata=`` caller passed.

        Caller-supplied keys override stack tags on collision (most-specific
        wins). Returns a JSON-safe dict; non-serializable values are repr'd.
        """
        merged = _current_tags()
        caller_meta = kwargs.get("metadata")
        if isinstance(caller_meta, dict):
            merged = {**merged, **caller_meta}
        out: dict[str, Any] = {}
        for k, v in merged.items():
            try:
                json.dumps(v, default=str)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = repr(v)
        return out

    def _sync_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        t0 = time.monotonic()
        record: dict[str, Any] = {
            "ts": _now(),
            "fn": "completion",
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages"),
            "kwargs": _redact_kwargs(kwargs),
            "metadata": _build_metadata(kwargs),
            "response": None,
            "cost_usd": None,
            "error": None,
        }
        resp = None
        try:
            resp = sync_orig(*args, **kwargs)
            record["response"] = _response_to_jsonable(resp)
            return resp
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["duration_s"] = round(time.monotonic() - t0, 4)
            record["cost_usd"] = _cost_for(resp)
            try:
                _write(trace_path, record)
            except OSError:
                pass

    async def _async_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        t0 = time.monotonic()
        record: dict[str, Any] = {
            "ts": _now(),
            "fn": "acompletion",
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages"),
            "kwargs": _redact_kwargs(kwargs),
            "metadata": _build_metadata(kwargs),
            "response": None,
            "cost_usd": None,
            "error": None,
        }
        resp = None
        try:
            resp = await async_orig(*args, **kwargs)
            record["response"] = _response_to_jsonable(resp)
            return resp
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["duration_s"] = round(time.monotonic() - t0, 4)
            record["cost_usd"] = _cost_for(resp)
            try:
                _write(trace_path, record)
            except OSError:
                pass

    if sync_orig is not None:
        litellm.completion = _sync_wrapper  # type: ignore[assignment]
    if async_orig is not None:
        litellm.acompletion = _async_wrapper  # type: ignore[assignment]

    embedding_orig = getattr(litellm, "embedding", None)
    if embedding_orig is not None:
        def _embedding_wrapper(*args, **kwargs):
            started = time.monotonic()
            response = None
            record = dict(ts=_now(), fn="embedding", model=kwargs.get("model"),
                          kwargs=_redact_kwargs(kwargs), metadata=_build_metadata(kwargs),
                          response=None, cost_usd=None, error=None)
            try:
                response = embedding_orig(*args, **kwargs)
                record["response"] = _response_to_jsonable(response)
                return response
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record["duration_s"] = round(time.monotonic() - started, 4)
                record["cost_usd"] = _cost_for(response)
                _write(trace_path, record)
        litellm.embedding = _embedding_wrapper

    _INSTALLED = True
    return True


# Allow `python -c "import litellm_trace; litellm_trace.install()"` style
# bootstrap, and also auto-install when imported directly if the env var
# is already set.
if os.environ.get("VERIFIER_TRACE_LLM_CALLS"):
    install()
