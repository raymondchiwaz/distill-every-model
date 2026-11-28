"""Import user-visible turns from local coding agent sessions."""
from __future__ import annotations

import json
import sqlite3
import zipfile
from collections.abc import Collection

from ..consent import CONSENT_VERSION, commercial_attested
from ..history_sources import SOURCES, history_files
from ..models import ACTIVITY_PAYLOAD_FIELDS, ACTIVITY_RESULT_FIELDS, CaptureEvent, sha256_text, ulid
from ..privacy import CODE_BLOCK_REMOVED, redact, redact_tool_payload

from .history_readers import Message, read_turns

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


def _claude(row, details, calls):
    msg = row.get("message")
    if not isinstance(msg, dict) or row.get("isSidechain"):
        return None
    kind = row.get("type")
    content = msg.get("content")
    if kind == "user":
        if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_result"
                                             for b in content):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                call_id = block.get("tool_use_id")
                if isinstance(call_id, str) and call_id in calls:
                    _record_tool_result(details[calls[call_id]], block, output_key="content")
            return None
        text = _text(content, {"text"})
        if row.get("isMeta") or row.get("isCompactSummary") or text.startswith(
            ("<local-command-", "<command-name>", "<system-reminder>")
        ):
            text = ""
        return Message("prompt", text)
    if kind == "assistant":
        model = msg.get("model")
        model = model if isinstance(model, str) else ""
        blocks = [block for block in content if isinstance(block, dict)] if isinstance(content, list) else []
        if msg.get("stop_reason") == "tool_use" or any(b.get("type") == "tool_use" for b in blocks):
            activity = []
            tool_details = {}
            for block in blocks:
                if block.get("type") == "tool_use":
                    detail = {"call_id": block.get("id")}
                    if block.get("input") is not None:
                        detail["input"] = block["input"]
                    tool_details[len(activity)] = detail
                    activity.append(("tool_call", block.get("name") or "tool"))
                elif block.get("type") == "text" and isinstance(block.get("text"), str):
                    activity.append(("progress", block["text"]))
            return Message("response", "", model, activity=activity, tool_details=tool_details)
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


def _codex_tool_activity(payload, activity, details, calls, model):
    kind = payload.get("type")
    if not isinstance(kind, str):
        return
    call_id = payload.get("call_id") or payload.get("id")
    if kind in CODEX_TOOL_CALLS:
        index = len(activity)
        name = payload.get("name") or CODEX_TOOL_NAMES.get(kind, "tool")
        activity.append(("tool_call", name))
        detail = {"model": model} if model else {}
        for key in ("arguments", "input", "action"):
            if payload.get(key) is not None:
                detail["input"] = payload[key]
                break
        if payload.get("result") is not None:
            detail["output"] = payload["result"]
        details[index] = detail
        if isinstance(call_id, str):
            calls[call_id] = index
    elif kind in CODEX_TOOL_RESULTS and isinstance(call_id, str) and call_id in calls:
        _record_tool_result(details[calls[call_id]], payload)


def _record_tool_result(detail, result, *, output_key="output"):
    if result.get(output_key) is not None:
        detail["output"] = result[output_key]
    if type(result.get("exit_code")) is int:
        detail["exit_code"] = str(result["exit_code"])
    if isinstance(result.get("is_error"), bool):
        detail["is_error"] = str(result["is_error"]).lower()


def _rows(stream):
    for line_no, line in enumerate(stream, 1):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            row = None
        yield line_no, row if isinstance(row, dict) else None


def _turns(stream, source, filename, stats):
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
    activity = []
    tool_details = {}
    calls = {}
    start = 0
    model = ""
    for line_no, row in _rows(stream):
        if row is None:
            stats["malformed_lines"] += 1
            prompt = answer = None
            activity = []
            tool_details, calls = {}, {}
            continue
        if source == "claude-code":
            session = row.get("sessionId") or session
            msg = _claude(row, tool_details, calls)
        else:
            payload = row.get("payload")
            if isinstance(payload, dict):
                if row.get("type") == "session_meta":
                    if isinstance(payload.get("source"), dict) and "subagent" in payload["source"]:
                        return
                    session = payload.get("id") or session
                elif row.get("type") == "turn_context":
                    if isinstance(payload.get("model"), str) and payload["model"]:
                        model = payload["model"]
                elif row.get("type") == "response_item" and prompt:
                    _codex_tool_activity(payload, activity, tool_details, calls, model)
                elif row.get("type") == "event_msg" and payload.get("type") == "exec_command_end":
                    call_id = payload.get("call_id")
                    if isinstance(call_id, str) and call_id in calls:
                        _record_tool_result(tool_details[calls[call_id]], payload, output_key="aggregated_output")
            msg = _codex(row, legacy)
        if msg is None:
            continue
        if msg.role in ("prompt", "boundary"):
            if prompt and answer:
                answer.activity = activity
                answer.tool_details = tool_details
                yield session, start, prompt, answer
            prompt, answer, start = msg.text, None, line_no
            activity = []
            tool_details, calls = {}, {}
        elif prompt:
            if (source == "claude-code" and answer and not msg.text
                    and any(kind == "tool_call" for kind, _ in msg.activity)):
                progress = ("progress", answer.text)
                if progress not in activity and progress not in msg.activity:
                    if answer.model:
                        tool_details[len(activity)] = {"model": answer.model}
                    activity.append(progress)
            for index, step in enumerate(msg.activity):
                if step[0] == "tool_call" or step not in activity:
                    detail = dict(msg.tool_details.get(index, {}))
                    call_id = detail.pop("call_id", None)
                    if msg.model or model:
                        detail["model"] = msg.model or model
                    if detail:
                        tool_details[len(activity)] = detail
                    if isinstance(call_id, str):
                        calls[call_id] = len(activity)
                    activity.append(step)
            if msg.role == "response":
                answer = msg if msg.text else None
                if answer and not answer.model:
                    answer.model = model
    if prompt and answer:
        answer.activity = activity
        answer.tool_details = tool_details
        yield session, start, prompt, answer


def _prepare_activity(activity, config, *, keep_code=None, tool_details=None):
    steps = []
    for index, (kind, text) in enumerate(activity):
        if not isinstance(text, str) or not text.strip():
            continue
        prepared = redact(text, config, keep_code=keep_code)
        if prepared.scrubbed:
            step = {"type": kind, "text": prepared.text}
            for field, value in (tool_details or {}).get(index, {}).items():
                if field == "model":
                    label = redact(value, config)
                    if label.scrubbed:
                        step["model"] = label.text
                    continue
                if field in ACTIVITY_RESULT_FIELDS:
                    step[field] = value
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
            yield from _turns(stream, source, file.stem, stats)
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


def import_history(config, vault, source, path=None, seen=None, *, activity_for: Collection[str] | None = None):
    check_history_consent(config, vault)
    files = history_files(source, path, config.data_dir / "history")
    reader = SOURCES[source].reader
    disabled = vault.disabled_history_chats(source)
    keep_code = config.enabled("capture", "capture_code")
    stats = dict(files=len(files), imported=0, already_imported=0, activity_updated=0, disabled_skipped=0,
                 malformed_lines=0, unreadable_files=0)
    for file in files:
        try:
            stamp = file.stat()
            signature = (stamp.st_dev, stamp.st_ino, stamp.st_size, stamp.st_mtime_ns, disabled, keep_code)
            wal = file.with_name(file.name + "-wal")
            if file.suffix.lower() in (".sqlite", ".db") and wal.is_file():
                wal_info = wal.stat()
                signature += (wal_info.st_size, wal_info.st_mtime_ns)
            if seen is not None and seen.get((source, file)) == signature:
                continue
            for session, start, prompt, answer in _file_turns(file, source, reader, stats):
                conv = vault.keys.tag(f"{source}:{session}")
                if activity_for is not None and conv not in activity_for:
                    continue
                if conv in disabled:
                    stats["disabled_skipped"] += 1
                    continue
                turn = vault.keys.tag(f"{conv}:{start}:{sha256_text(prompt)}:{sha256_text(answer.text)}")
                if vault.history_imported(turn):
                    stats["already_imported"] += 1
                    if answer.activity:
                        previous = vault.get_history_activity(turn)
                        restore_code = (keep_code and previous is not None
                                        and any(CODE_BLOCK_REMOVED in step["text"] for step in previous)
                                        and _prepare_activity(answer.activity, config, keep_code=False) == previous)
                        if previous is None or restore_code or any(answer.tool_details.values()):
                            activity = _prepare_activity(answer.activity, config, tool_details=answer.tool_details)
                            old_details = sum(field in step for step in (previous or []) for field in ACTIVITY_PAYLOAD_FIELDS)
                            new_details = sum(field in step for step in activity for field in ACTIVITY_PAYLOAD_FIELDS)
                            if ((activity or restore_code) and activity != previous and new_details >= old_details
                                    and vault.add_history_activity(turn, conv, activity, previous=previous)):
                                stats["activity_updated"] += 1
                    continue
                if activity_for is not None:
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
                stats[vault.add_history_pair(turn, *events)] += 1
        except (sqlite3.Error, zipfile.BadZipFile, OSError, ValueError, RuntimeError):
            stats["unreadable_files"] += 1
            continue
        if seen is not None:
            seen[(source, file)] = signature
    return stats
