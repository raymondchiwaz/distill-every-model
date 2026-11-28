"""Import user-visible turns from local coding agent sessions."""
from __future__ import annotations

import json
import sqlite3
import zipfile
from collections.abc import Collection

from ..consent import CONSENT_VERSION, commercial_attested
from ..history_sources import SOURCES, history_files, is_child_history
from ..models import ACTIVITY_PAYLOAD_FIELDS, ACTIVITY_RESULT_FIELDS, CaptureEvent, sha256_text, ulid
from ..privacy import redact, redact_tool_payload
from ..vault import merge_history_activity

from .history_readers import Message, TurnActivity, _tool_event, read_turns

CODEX_TOOL_NAMES = {"local_shell_call": "shell", "web_search_call": "web search",
                    "tool_search_call": "tool search", "image_generation_call": "image generation"}
CODEX_TOOL_CALLS = {"function_call", "custom_tool_call", *CODEX_TOOL_NAMES}
CODEX_TOOL_RESULTS = {"function_call_output", "custom_tool_call_output",
                      "tool_search_output", "local_shell_call_output"}


def _text(content, kinds):
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n".join(part["text"] for part in content if isinstance(part, dict)
                     and part.get("type") in kinds and isinstance(part.get("text"), str)).strip()


def _claude(row, *, child=False):
    msg = row.get("message")
    if not isinstance(msg, dict) or (row.get("isSidechain") and not child):
        return None
    kind = row.get("type")
    content = msg.get("content")
    blocks = [block for block in content if isinstance(block, dict)] if isinstance(content, list) else []
    if kind == "user":
        results = [_tool_event(block, result=True) for block in blocks if block.get("type") == "tool_result"]
        if results:
            return Message("activity", "", activity=[step for step, _ in results],
                           tool_details={i: detail for i, (_, detail) in enumerate(results)})
        text = _text(content, {"text"})
        if row.get("isMeta") or row.get("isCompactSummary") or text.startswith(
            ("<local-command-", "<command-name>", "<system-reminder>")
        ):
            return None
        return Message("prompt", text)
    if kind == "assistant":
        model = msg.get("model")
        model = model if isinstance(model, str) else ""
        call_types = ("tool_use", "server_tool_use", "mcp_tool_use")
        if msg.get("stop_reason") == "tool_use" or any(
            block.get("type") in call_types or str(block.get("type", "")).endswith("_tool_result") for block in blocks
        ):
            activity = []
            tool_details = {}
            for block in blocks:
                block_type = block.get("type")
                if block_type in call_types or str(block_type).endswith("_tool_result"):
                    step, detail = _tool_event(block, result=block_type not in call_types)
                    tool_details[len(activity)] = detail
                    activity.append(step)
                elif block_type == "text" and isinstance(block.get("text"), str):
                    activity.append(("progress", block["text"]))
            return Message("activity", "", model, activity, tool_details)
        return Message("response", _text(content, {"text"}), model)
    return None


def _codex(row, legacy):
    p = row.get("payload")
    if not isinstance(p, dict):
        return None
    kind = p.get("type")
    if row.get("type") == "event_msg":
        if kind in ("turn_aborted", "task_started"):
            return Message("boundary", "")
        if kind == "user_message":
            return Message("prompt", _text(p.get("message"), {"text"}))
        if kind == "agent_message" and p.get("phase") != "commentary":
            return Message("response", _text(p.get("message"), {"text"}))
        if kind == "agent_message" and p.get("phase") == "commentary":
            return Message("activity", "", activity=[("progress", _text(p.get("message"), {"text"}))])
        if kind == "agent_reasoning" and isinstance(p.get("text"), str):
            return Message("activity", "", activity=[("rationale", p["text"])])
        item = p.get("item")
        if kind == "item_completed" and isinstance(item, dict):
            if item.get("type") == "UserMessage":
                return Message("prompt", _text(item.get("content"), {"text", "Text"}))
            if item.get("type") == "AgentMessage" and item.get("phase") != "commentary":
                return Message("response", _text(item.get("content"), {"Text", "text"}))
            if item.get("type") == "AgentMessage" and item.get("phase") == "commentary":
                return Message("activity", "", activity=[("progress", _text(item.get("content"), {"Text", "text"}))])
    if row.get("type") != "response_item":
        return None
    if kind == "reasoning" and isinstance(p.get("summary"), list):
        summary = _text(p["summary"], {"summary_text"})
        return Message("activity", "", activity=[("rationale", summary)])
    if kind == "message":
        role = p.get("role")
        if role == "assistant":
            text = _text(p.get("content"), {"output_text"})
            if p.get("phase") == "commentary":
                return Message("activity", "", activity=[("progress", text)])
            return Message("response", text)
        if legacy and role == "user":
            text = _text(p.get("content"), {"input_text"})
            if text.startswith(("<environment_context>", "# AGENTS.md instructions", "<permissions instructions>")):
                text = ""
            return Message("prompt", text)
    return None


CODEX_EVENT_TOOLS = {
    "exec_command": "exec_command",
    "mcp_tool_call": "",
    "web_search": "web search",
    "dynamic_tool_call": "",
    "patch_apply": "apply_patch",
}
CODEX_ITEM_TOOLS = {
    "CommandExecution": "exec_command", "McpToolCall": "", "WebSearch": "web search",
    "DynamicToolCall": "", "FileChange": "apply_patch", "ImageView": "view_image",
    "ImageGeneration": "image generation", "CollabAgentToolCall": "agent",
}


def _codex_tool_event(row, payload):
    kind = payload.get("type")
    if not isinstance(kind, str):
        return None
    call_id = payload.get("call_id") or payload.get("id")
    if row.get("type") == "response_item":
        if kind in CODEX_TOOL_CALLS:
            return _tool_event({**payload, "name": payload.get("name") or CODEX_TOOL_NAMES.get(kind, "tool")})
        if kind in CODEX_TOOL_RESULTS:
            return _tool_event(payload, result=True)
    if row.get("type") != "event_msg":
        return None
    if kind in ("item_started", "item_completed"):
        item = payload.get("item")
        if not isinstance(item, dict) or item.get("type") not in CODEX_ITEM_TOOLS:
            return None
        data = item
        name = CODEX_ITEM_TOOLS[item["type"]]
        complete = kind == "item_completed"
        call_id = item.get("call_id") or item.get("id")
    else:
        event, _, phase = kind.rpartition("_")
        if event not in CODEX_EVENT_TOOLS or phase not in ("begin", "end"):
            return None
        data = payload
        name = CODEX_EVENT_TOOLS[event]
        complete = phase == "end"
    invocation = data.get("invocation")
    invocation = invocation if isinstance(invocation, dict) else {}
    name = data.get("tool") or invocation.get("tool") or name
    fields = {**data, "name": name, "call_id": call_id}
    if "status" not in fields and not complete:
        fields["status"] = "running"
    for key in ("command", "query", "changes", "path"):
        if data.get(key):
            fields.setdefault("input", {key: data[key]})
            break
    for key in ("aggregated_output", "results"):
        if data.get(key) is not None:
            fields.setdefault("output", data[key])
            break
    step, detail = _tool_event(fields, result=complete)
    if complete and "input" in fields:
        detail.setdefault("input", fields["input"])
    if invocation.get("arguments") is not None:
        detail.setdefault("input", invocation["arguments"])
    return step, detail


def _rows(stream):
    for line_no, line in enumerate(stream, 1):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            row = None
        yield line_no, row if isinstance(row, dict) else None


def _turns(stream, source, filename, stats, *, child=False):
    # Older Codex rollouts contain only wire messages. Prefer user-facing
    # events whenever available so injected context cannot replace the prompt.
    legacy = False
    if source == "codex":
        legacy = True
        for _, row in _rows(stream):
            msg = _codex(row, False) if row else None
            if msg and msg.role == "prompt":
                legacy = False
                break
        stream.seek(0)
    session = filename
    prompt = answer = None
    turn = TurnActivity()
    finished = []
    call_turns = {}
    start = 0
    model = ""
    for line_no, row in _rows(stream):
        msg = None
        if row is None:
            stats["malformed_lines"] += 1
            msg = Message("boundary", "")
        elif source == "claude-code":
            if child:
                session = row.get("agentId") or filename
            else:
                session = row.get("sessionId") or session
            msg = _claude(row, child=child)
        else:
            payload = row.get("payload")
            if isinstance(payload, dict):
                if row.get("type") == "session_meta":
                    if isinstance(payload.get("source"), dict) and "subagent" in payload["source"]:
                        child = True
                    session = payload.get("id") or session
                elif row.get("type") == "turn_context":
                    model = payload["model"] if isinstance(payload.get("model"), str) and payload["model"] else model
                else:
                    event = _codex_tool_event(row, payload)
                    if event:
                        step, detail = event
                        call_id = detail.get("call_id")
                        # Later event mirrors and background results may arrive after another turn starts.
                        owner = call_turns.get(call_id, turn) if step[0] == "tool_result" else turn
                        if call_id:
                            call_turns[call_id] = owner
                        if model and owner is turn:
                            detail.setdefault("model", model)
                        owner.add(step, detail)
            msg = _codex(row, legacy)
        if msg is None:
            continue
        if msg.role in ("prompt", "boundary"):
            if answer or turn.steps:
                finished.append((session, start, prompt, answer, turn, model))
            prompt, answer, start = msg.text, None, line_no
            turn = TurnActivity()
        elif prompt or msg.activity:
            if source == "claude-code" and answer and any(kind == "tool_call" for kind, _ in msg.activity):
                progress = ("progress", answer.text)
                if progress not in turn.steps and progress not in msg.activity:
                    turn.add(progress, {"model": answer.model} if answer.model else None)
                answer = None
            for index, step in enumerate(msg.activity):
                if step[0] in ("tool_call", "tool_result") or step not in turn.steps:
                    detail = dict(msg.tool_details.get(index, {}))
                    if msg.model or model:
                        detail["model"] = msg.model or model
                    turn.add(step, detail)
            if msg.role == "response" and msg.text:
                answer = msg
                if not answer.model:
                    answer.model = model
    finished.append((session, start, prompt, answer, turn, model))
    for name, position, request, response, activity, last_model in finished:
        saved = activity.finish(response, model=last_model, tools_only=child)
        if saved and (request or saved.role == "activity"):
            yield name, position, "" if child else request or "", saved


def _prepare_activity(activity, config, *, keep_code=None, tool_details=None):
    steps = []
    for index, (kind, text) in enumerate(activity):
        if not isinstance(text, str) or not text.strip():
            continue
        prepared = redact(text, config, keep_code=keep_code)
        if prepared.scrubbed or kind == "tool_call":
            step = {"type": kind, "text": prepared.text if prepared.scrubbed else "Tool name held locally"}
            for field, value in (tool_details or {}).get(index, {}).items():
                if field == "model":
                    label = redact(value, config)
                    if label.scrubbed:
                        step["model"] = label.text
                    continue
                if field in ACTIVITY_RESULT_FIELDS:
                    step[field] = value
                    continue
                if field not in ACTIVITY_PAYLOAD_FIELDS:
                    continue
                payload = redact_tool_payload(value, config, keep_code=keep_code)
                step[field] = payload.text
                if payload.quarantine_reason:
                    step[f"{field}_hold"] = payload.quarantine_reason
            steps.append(step)
    return steps


def _file_turns(file, source, reader, stats):
    if reader in ("claude", "codex") and file.suffix.lower() == ".jsonl":
        with file.open("rb") as stream:
            yield from _turns(stream, source, file.stem, stats, child=is_child_history(file))
    else:
        yield from read_turns(file, reader, stats)


def check_history_consent(config, vault):
    if config.get("privacy", "consent_mode") != "default-deny":
        raise ValueError("History import requires default-deny consent mode.")
    if vault.latest_decision("onboard", CONSENT_VERSION) != "approved":
        raise ValueError("Run dem onboard and accept capture before importing chat history.")
    if (not commercial_attested(config)
            or vault.latest_decision("commercial", CONSENT_VERSION) != "attested"):
        raise ValueError("Run dem attest before importing commercial chat history.")
    if not config.enabled("capture", "capture_prompts"):
        raise ValueError("Enable prompt import with dem config capture capture_prompts true.")


def import_history(config, vault, source, path=None, seen=None, *, conversation_ids: Collection[str] | None = None):
    check_history_consent(config, vault)
    files = history_files(source, path, config.data_dir / "history")
    reader = SOURCES[source].reader
    disabled = vault.disabled_history_chats(source)
    keep_code = config.enabled("capture", "capture_code")
    stats = dict(files=len(files), imported=0, already_imported=0, activity_updated=0, activity_only=0,
                 disabled_skipped=0, malformed_lines=0, unreadable_files=0)
    for file in files:
        try:
            stamp = file.stat()
            signature = (stamp.st_dev, stamp.st_ino, stamp.st_size, stamp.st_mtime_ns, disabled, keep_code)
            wal = file.with_name(file.name + "-wal")
            if file.suffix.lower() in (".sqlite", ".db") and wal.is_file():
                wal_info = wal.stat()
                signature += (wal_info.st_size, wal_info.st_mtime_ns)
            if source == "antigravity" and file.name == "transcript.jsonl":
                sibling = file.with_name("transcript_full.jsonl")
                if sibling.is_file() and not sibling.is_symlink():
                    sibling_info = sibling.stat()
                    signature += (sibling_info.st_size, sibling_info.st_mtime_ns)
            if seen is not None and seen.get((source, file)) == signature:
                continue
            for session, start, prompt, answer in _file_turns(file, source, reader, stats):
                conv = vault.keys.tag(f"{source}:{session}")
                if conversation_ids is not None and conv not in conversation_ids:
                    continue
                if conv in disabled:
                    stats["disabled_skipped"] += 1
                    continue
                unfinished_id = vault.keys.tag(f"activity:{conv}:{start}")
                if answer.role == "activity":
                    activity = _prepare_activity(answer.activity, config, tool_details=answer.tool_details)
                    if not activity:
                        continue
                    preview = redact(prompt, config)
                    event = CaptureEvent(
                        id=unfinished_id, source=source, surface=source, role="activity", consent="local",
                        text="", conversation_id=conv, turn_index=start * 2, model_hint=answer.model,
                        raw_hash=sha256_text(prompt) if prompt else "",
                        redacted_text=preview.text if preview.scrubbed else "", activity=activity,
                        quarantine_reason=preview.quarantine_reason,
                        metadata={"history_activity_origin": answer.local_reason} if answer.local_reason else {},
                    )
                    if vault.save_unfinished_history(event):
                        stats["activity_only"] += 1
                    continue
                turn = vault.keys.tag(f"{conv}:{start}:{sha256_text(prompt)}:{sha256_text(answer.text)}")
                if vault.history_imported(turn):
                    stats["already_imported"] += 1
                    if answer.activity:
                        previous = vault.get_history_activity(turn)
                        activity = _prepare_activity(answer.activity, config, tool_details=answer.tool_details)
                        restore_code = (keep_code and previous is not None
                                        and _prepare_activity(answer.activity, config, keep_code=False,
                                                              tool_details=answer.tool_details) == previous)
                        if not restore_code:
                            activity = merge_history_activity(previous, activity)
                        if activity and activity != previous:
                            if vault.add_history_activity(turn, conv, activity, previous=previous):
                                stats["activity_updated"] += 1
                    continue
                events = []
                for index, (role, text) in enumerate((("prompt", prompt), ("response", answer.text))):
                    prepared = redact(text, config)
                    events.append(CaptureEvent(
                        id=ulid(), source=source, surface=source, role=role, text=text,
                        conversation_id=conv, turn_index=start * 2 + index,
                        raw_hash=sha256_text(text), model_hint=answer.model,
                        redacted_text=prepared.text, redactions=prepared.redactions,
                        activity=(_prepare_activity(answer.activity, config, tool_details=answer.tool_details)
                                  if role == "response" else []),
                        quarantine_reason=prepared.quarantine_reason,
                        metadata={"history_turn": turn, "derived_from": "proprietary-api-output"},
                    ))
                stats[vault.add_history_pair(turn, *events, unfinished_id=unfinished_id)] += 1
        except (sqlite3.Error, zipfile.BadZipFile, OSError, ValueError, RuntimeError):
            stats["unreadable_files"] += 1
            continue
        if seen is not None:
            seen[(source, file)] = signature
    return stats
