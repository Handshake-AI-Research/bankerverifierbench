"""Convert per-verifier raw traces into ATIF-shaped trajectories.

Why
---
Each verifier emits traces in its own format (``llm_calls.jsonl`` for
aaaj/archipelago/autorubric; ``judge_trace_batch*.txt`` for gandalf;
Claude-Code-CLI session JSONL for rewardkit). For failure analysis we
want a uniform per-verifier "agent trajectory" so a downstream tool —
diff visualizer, replay harness, judge debugger — can consume them
with one parser.

ATIF is the schema the *task agent's* trajectory uses. Reusing it for
the verifier agent means existing visualizers work for free; the only
caveat is that ATIF's ``tool_calls`` field is empty for non-agentic
verifiers (aaaj/archipelago/autorubric never call tools).

Output shape (per (verifier, task)):

    results/<run_id>/traces_atif/<verifier>/<task_id>.json

    {
      "schema_version": "ATIF-v1.6",
      "session_id": "<verifier>-<task_id>",
      "agent": {"name": "<verifier>:<model>", "version": "1.0.0"},
      "metadata": {
        "verifier": "...", "model": "...", "task_id": "...",
        "n_calls": int, "n_criteria": int | null
      },
      "steps": [
        {
          "step_id": int,
          "source": "system" | "user" | "agent" | "tool",
          "message": str,
          "tool_calls": [...],     # only populated for gandalf
          "metadata": {            # custom field — preserved through ATIF
            "criterion_index": int | null,
            "criterion": str | null,
            "phase": str | null,   # aaaj only: "locate" | "retrieve" | "judge"
            "duration_s": float | null,
            "model": str | null,
            "ts": str | null,
            "fn": "completion" | "acompletion" | null,
            "error": str | null
          }
        },
        ...
      ]
    }

Each LLM call is unrolled into N+1 steps: the N input messages
(system/user) followed by one agent step holding the response. For
gandalf, the existing ``judge_trace_batch.txt`` text is parsed into
the same step shape — see ``_parse_gandalf_text``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


ATIF_SCHEMA_VERSION = "ATIF-v1.6"


# ---------------------------------------------------------------------------
# JSONL → ATIF (aaaj, archipelago, autorubric)
# ---------------------------------------------------------------------------

_ROLE_TO_SOURCE = {
    "system": "system",
    "user": "user",
    "tool": "tool",
    "assistant": "agent",
    "function": "tool",
}


def _step_meta(rec: dict[str, Any]) -> dict[str, Any]:
    """Build the per-step metadata block from a JSONL record."""
    md = dict(rec.get("metadata") or {})
    md["duration_s"] = rec.get("duration_s")
    md["model"] = rec.get("model")
    md["ts"] = rec.get("ts")
    md["fn"] = rec.get("fn")
    md["error"] = rec.get("error")
    return md


def _response_text_and_tool_calls(rec: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Pull the assistant's content + tool_calls out of a litellm response.

    Returns ``("", [])`` on missing/odd shapes — we still emit the agent
    step so the caller knows a call happened, just without text.
    """
    resp = rec.get("response") or {}
    if not isinstance(resp, dict):
        return "", []
    choices = resp.get("choices") or []
    if not choices:
        return "", []
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content") or ""
    raw_tcs = msg.get("tool_calls") or []
    tool_calls: list[dict[str, Any]] = []
    for tc in raw_tcs:
        fn = (tc or {}).get("function") or {}
        name = fn.get("name") or tc.get("name") or "unknown"
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        tool_calls.append({"name": name, "args": args or {}})
    return str(content), tool_calls


def _jsonl_to_steps(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    step_id = 0
    for rec in records:
        meta = _step_meta(rec)
        for m in rec.get("messages") or []:
            role = (m or {}).get("role", "system")
            source = _ROLE_TO_SOURCE.get(role, "system")
            content = m.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, default=str, ensure_ascii=False)
            step_id += 1
            steps.append({
                "step_id": step_id,
                "source": source,
                "message": content,
                "tool_calls": [],
                "metadata": meta,
            })
        text, tcs = _response_text_and_tool_calls(rec)
        step_id += 1
        steps.append({
            "step_id": step_id,
            "source": "agent",
            "message": text,
            "tool_calls": tcs,
            "metadata": meta,
        })
    return steps


def normalize_jsonl(
    jsonl_path: Path,
    *,
    verifier: str,
    task_id: str,
) -> dict[str, Any]:
    """Read a verifier's ``llm_calls.jsonl`` and emit one ATIF doc."""
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    steps = _jsonl_to_steps(records)
    model = next((r.get("model") for r in records if r.get("model")), "")
    crit_indices = {
        rec.get("metadata", {}).get("criterion_index")
        for rec in records
        if isinstance(rec.get("metadata"), dict)
    }
    crit_indices.discard(None)

    return {
        "schema_version": ATIF_SCHEMA_VERSION,
        "session_id": f"{verifier}-{task_id}",
        "agent": {"name": f"{verifier}:{model}" if model else verifier,
                  "version": "1.0.0"},
        "metadata": {
            "verifier": verifier,
            "model": model,
            "task_id": task_id,
            "n_calls": len(records),
            "n_criteria": len(crit_indices) or None,
        },
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Gandalf text → ATIF
# ---------------------------------------------------------------------------

# Section header in judge_trace_batch.txt looks like:
#   "System Prompt ─────────────"
#   "Message from User ─────────"
#   "Agent Action ──────────────"
#   "Observation ───────────────"
# The dashes are the U+2500 box-drawing char repeated.
_HEADER_RE = re.compile(
    r"^(System Prompt|Message from User|Agent Action|Observation|Agent Thought)\s+─{3,}\s*$",
    re.MULTILINE,
)

_HEADER_TO_SOURCE = {
    "System Prompt": "system",
    "Message from User": "user",
    "Agent Action": "agent",
    "Agent Thought": "agent",
    "Observation": "tool",
}


def _parse_gandalf_text(text: str) -> list[tuple[str, str]]:
    """Split gandalf's pretty-printed trace into (header, body) chunks.

    Body is everything between this header line and the next, stripped
    of trailing whitespace. The first chunk before any header is dropped
    (it's the harbor-test wrapper output, not part of the agent loop).
    """
    headers = list(_HEADER_RE.finditer(text))
    if not headers:
        return []
    chunks: list[tuple[str, str]] = []
    for i, m in enumerate(headers):
        header = m.group(1)
        body_start = m.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[body_start:body_end].strip()
        chunks.append((header, body))
    return chunks


# Pull tool name + arg blob from an "Agent Action" body. Gandalf renders
# tool calls inline like "→ tool_name(arg1=..., arg2=...)" or as a JSON
# block. We do best-effort: find the first identifier followed by "(".
_TOOL_CALL_RE = re.compile(r"\b([a-z][a-z0-9_]*(?:__[a-z0-9_]+)?)\s*\(")


def _extract_gandalf_tool_call(body: str) -> list[dict[str, Any]]:
    """Best-effort: extract one tool name from an Agent Action body.

    Args are NOT round-tripped (gandalf pretty-wraps them with text-
    rendering line breaks; reconstructing the original JSON would
    require a real parser). The name alone is enough for failure
    analysis to follow which tools the judge used.
    """
    m = _TOOL_CALL_RE.search(body)
    if not m:
        return []
    return [{"name": m.group(1), "args": {}}]


def normalize_gandalf(
    trace_dir: Path,
    *,
    task_id: str,
    model: str = "",
) -> dict[str, Any]:
    """Convert a gandalf trace dir to an ATIF doc.

    Reads ``judge_trace_batch*.txt`` (concatenated in name order if
    multiple — multi-batch BTB runs split the trace) and parses
    section headers into steps. Falls back to an empty steps list if
    no .txt is present (e.g. an aborted run).
    """
    txt_files = sorted(trace_dir.glob("judge_trace_batch*.txt"))
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in txt_files)

    steps: list[dict[str, Any]] = []
    for i, (header, body) in enumerate(_parse_gandalf_text(text), start=1):
        source = _HEADER_TO_SOURCE.get(header, "system")
        tool_calls = (
            _extract_gandalf_tool_call(body) if source == "agent" else []
        )
        steps.append({
            "step_id": i,
            "source": source,
            "message": body,
            "tool_calls": tool_calls,
            "metadata": {"section": header},
        })

    return {
        "schema_version": ATIF_SCHEMA_VERSION,
        "session_id": f"gandalf-{task_id}",
        "agent": {"name": f"gandalf:{model}" if model else "gandalf",
                  "version": "1.0.0"},
        "metadata": {
            "verifier": "gandalf",
            "model": model,
            "task_id": task_id,
            "n_steps": len(steps),
        },
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Claude Code session JSONL → ATIF (rewardkit)
# ---------------------------------------------------------------------------
#
# Rewardkit's agent-judge backend is the Claude Code CLI. CC writes its
# session log as a JSONL stream under
# ``~/.claude/projects/-home-agent-workspace/<session-uuid>.jsonl``,
# which the verifier ``test.sh`` mirrors into
# ``<trial>/verifier/judge_trajectories/``. The orchestrator copies that
# subtree into ``traces/rewardkit/<task>/judge_trajectories/`` for each
# (task) trace.
#
# Record types observed:
#   queue-operation   ← prompt enqueued; first record carries the system prompt
#   user              ← message.role=user with stringified or block content
#   attachment        ← deferred-tools delta (lists added MCP/builtin tools)
#   assistant         ← message.role=assistant with block content
#                       (text / thinking / tool_use / structured_output)
#   last-prompt       ← session metadata, ignored
#   ai-title          ← session metadata, ignored
#
# We map this stream into ATIF by:
#   * Each ``user`` record → one ``user`` (or ``tool``, when content is
#     a tool_result block) step.
#   * Each ``assistant`` record → one ``agent`` step. Its ``content`` is
#     a list of typed blocks; we surface text/thinking blocks as the
#     step's message and ``tool_use`` blocks as the step's tool_calls.
#   * ``attachment`` records become ``system`` steps with the deferred-
#     tools list serialised in metadata, so a downstream viewer can see
#     when the MCP surface activated.
#   * Other record types (queue-operation, last-prompt, ai-title) are
#     dropped — they're session bookkeeping, not part of the agent loop.

def _cc_message_text(content: Any) -> str:
    """Stringify a Claude Code message content payload.

    The payload is either a bare string (legacy) or a list of typed
    blocks (current). For the list form we concatenate the text/thinking
    blocks; tool_use / tool_result blocks are surfaced separately.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, default=str, ensure_ascii=False)
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        t = block.get("type")
        if t == "text":
            parts.append(str(block.get("text", "")))
        elif t == "thinking":
            parts.append(f"[thinking] {block.get('thinking', '')}")
        elif t == "tool_result":
            # Tool results live on user-role records; render the textual
            # output for the step body. Other blocks already capture
            # tool calls themselves.
            inner = block.get("content")
            if isinstance(inner, list):
                inner_text = []
                for ib in inner:
                    if isinstance(ib, dict) and ib.get("type") == "text":
                        inner_text.append(str(ib.get("text", "")))
                parts.append("\n".join(inner_text))
            elif isinstance(inner, str):
                parts.append(inner)
    return "\n\n".join(p for p in parts if p)


def _cc_tool_calls(content: Any) -> list[dict[str, Any]]:
    """Extract tool_use blocks from a Claude Code assistant message."""
    if not isinstance(content, list):
        return []
    calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            calls.append({
                "name": block.get("name", "?"),
                "args": block.get("input") or {},
            })
    return calls


def _cc_user_source(content: Any) -> str:
    """Decide whether a CC user-role record is really a user prompt or a tool result.

    Claude Code packs tool outputs as user-role messages with
    ``content=[{"type":"tool_result", ...}]``. ATIF distinguishes those
    via the ``tool`` source type so a viewer can render them differently
    from human-authored user turns.
    """
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return "tool"
    return "user"


def normalize_rewardkit(
    trace_dir: Path,
    *,
    task_id: str,
    model: str = "",
) -> dict[str, Any]:
    """Convert a rewardkit trace dir to an ATIF doc.

    Reads ``judge_trajectories/*.jsonl`` (one or more Claude Code
    session files; usually one per trial). Falls back to an empty steps
    list when the dir is empty, e.g. an aborted run before CC wrote any
    sessions.
    """
    sessions_dir = trace_dir / "judge_trajectories"
    jsonl_files: list[Path] = []
    if sessions_dir.is_dir():
        jsonl_files = sorted(p for p in sessions_dir.glob("*.jsonl") if p.is_file())

    steps: list[dict[str, Any]] = []
    step_id = 0
    inferred_model = model
    n_assistant_turns = 0
    tool_call_counts: dict[str, int] = {}

    for jsonl in jsonl_files:
        with jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = rec.get("type")

                if t == "user":
                    msg = rec.get("message") or {}
                    content = msg.get("content")
                    source = _cc_user_source(content)
                    text = _cc_message_text(content)
                    step_id += 1
                    steps.append({
                        "step_id": step_id,
                        "source": source,
                        "message": text,
                        "tool_calls": [],
                        "metadata": {
                            "ts": rec.get("timestamp"),
                            "cwd": rec.get("cwd"),
                            "uuid": rec.get("uuid"),
                        },
                    })
                elif t == "assistant":
                    n_assistant_turns += 1
                    msg = rec.get("message") or {}
                    if not inferred_model:
                        inferred_model = msg.get("model") or ""
                    content = msg.get("content")
                    text = _cc_message_text(content)
                    tool_calls = _cc_tool_calls(content)
                    for tc in tool_calls:
                        n = tc.get("name", "?")
                        tool_call_counts[n] = tool_call_counts.get(n, 0) + 1
                    step_id += 1
                    usage = msg.get("usage") or {}
                    steps.append({
                        "step_id": step_id,
                        "source": "agent",
                        "message": text,
                        "tool_calls": tool_calls,
                        "metadata": {
                            "ts": rec.get("timestamp"),
                            "model": msg.get("model"),
                            "stop_reason": msg.get("stop_reason"),
                            "usage": usage or None,
                            "uuid": rec.get("uuid"),
                        },
                    })
                elif t == "attachment":
                    att = rec.get("attachment") or {}
                    if att.get("type") == "deferred_tools_delta":
                        added = att.get("addedNames") or []
                        step_id += 1
                        steps.append({
                            "step_id": step_id,
                            "source": "system",
                            "message": (
                                f"[deferred_tools_delta] {len(added)} tools added; "
                                f"first 5: {added[:5]}"
                            ),
                            "tool_calls": [],
                            "metadata": {
                                "ts": rec.get("timestamp"),
                                "added_tools": added,
                            },
                        })
                # queue-operation / last-prompt / ai-title: skip silently.

    return {
        "schema_version": ATIF_SCHEMA_VERSION,
        "session_id": f"rewardkit-{task_id}",
        "agent": {
            "name": f"rewardkit:{inferred_model}" if inferred_model else "rewardkit",
            "version": "1.0.0",
        },
        "metadata": {
            "verifier": "rewardkit",
            "model": inferred_model,
            "task_id": task_id,
            "n_steps": len(steps),
            "n_assistant_turns": n_assistant_turns,
            "tool_call_counts": tool_call_counts,
            "n_session_files": len(jsonl_files),
        },
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Run-level orchestration
# ---------------------------------------------------------------------------

def normalize_run(run_dir: Path) -> dict[str, Any]:
    """Normalize every verifier×task trace under ``run_dir/traces/``.

    Writes ``run_dir/traces_atif/<verifier>/<task_id>.json`` for each
    found trace. Returns a summary dict for the CLI to print.

    Skips silently when the input file is missing — callers that
    haven't run a verifier on a task simply get no ATIF output.
    """
    traces_root = run_dir / "traces"
    if not traces_root.is_dir():
        raise FileNotFoundError(f"no traces/ dir under {run_dir}")
    out_root = run_dir / "traces_atif"

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "out_dir": str(out_root),
        "written": [],
        "skipped": [],
    }

    verdicts_root = run_dir / "verdicts"

    for verifier_dir in sorted(traces_root.iterdir()):
        if not verifier_dir.is_dir():
            continue
        verifier = verifier_dir.name
        for task_dir in sorted(verifier_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_id = task_dir.name
            out_dir = out_root / verifier
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{task_id}.json"

            if verifier == "gandalf":
                # Pull the model from the persisted Verdict JSON; the trace
                # dir alone doesn't capture it (info.json only has usage
                # totals), but the runner stamped it into the Verdict.
                model = _model_from_verdict(verdicts_root, verifier, task_id) or ""
                doc = normalize_gandalf(task_dir, task_id=task_id, model=model)
            elif verifier == "rewardkit":
                # Rewardkit's native trace is the Claude Code session
                # JSONL stream (different shape from the litellm-flavour
                # JSONL the other host-side verifiers emit). Skip cleanly
                # if no session files were captured (aborted trial,
                # empty-workspace short-circuit).
                sessions = task_dir / "judge_trajectories"
                if not sessions.is_dir() or not any(sessions.glob("*.jsonl")):
                    summary["skipped"].append(
                        {"verifier": verifier, "task_id": task_id,
                         "reason": "no judge_trajectories/*.jsonl"}
                    )
                    continue
                model = _model_from_verdict(verdicts_root, verifier, task_id) or ""
                doc = normalize_rewardkit(task_dir, task_id=task_id, model=model)
            else:
                jsonl = task_dir / "llm_calls.jsonl"
                if not jsonl.is_file():
                    summary["skipped"].append(
                        {"verifier": verifier, "task_id": task_id,
                         "reason": "no llm_calls.jsonl"}
                    )
                    continue
                doc = normalize_jsonl(jsonl, verifier=verifier, task_id=task_id)

            out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
            summary["written"].append({
                "verifier": verifier, "task_id": task_id,
                "out": str(out_path),
                "n_steps": len(doc["steps"]),
            })

    return summary


def _model_from_verdict(verdicts_root: Path, verifier: str, task_id: str) -> str | None:
    """Read the model name out of the canonical Verdict JSON, if present."""
    p = verdicts_root / verifier / f"{task_id}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get("model") or None
    except (json.JSONDecodeError, OSError):
        return None
