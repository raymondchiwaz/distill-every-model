"""Read user-facing turns and structured activity from local harness formats."""
from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import zipfile
from dataclasses import dataclass, field
from io import BufferedReader
from itertools import groupby
from pathlib import PurePosixPath

from zstandard import ZstdDecompressor, ZstdError

from ..history_sources import DEEPSEEK_LOG_NAME, is_child_history


@dataclass
class Message:
    role: str
    text: str
    model: str = ""
    activity: list[tuple[str, str]] = field(default_factory=list)
    tool_details: dict[int, dict] = field(default_factory=dict)
    local_reason: str = ""


def _obj(value):
    return value if isinstance(value, dict) else {}


def _items(value):
    return value if isinstance(value, list) else ()


def _string(value):
    return value if isinstance(value, str) else ""


def _text(value):
    if isinstance(value, str):
        return value.strip()
    parts = []
    for part in _items(value):
        block = _obj(part)
        if (block.get("type") in (None, "text", "input_text", "output_text")
                and not block.get("thought")
                and not any(key in block for key in ("functionCall", "functionResponse", "tool_use"))):
            text = _string(block.get("text"))
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


CALL_TYPES = {"tool_use", "tool-call", "tool_call", "toolCall", "toolRequest", "tool_request",
              "toolUse", "tool", "function_call"}
RESULT_TYPES = {"tool_result", "tool-result", "toolResult", "toolResponse", "tool_response", "function-call-output",
                "function_call_output"}
RESULT_ROLES = {"tool", "toolResult", "tool_result", "function"}
CALL_FIELDS = ("tool_calls", "toolCalls", "toolRequests", "toolUses")


def _first(row, *keys):
    return next((row[key] for key in keys if row.get(key) is not None), None)


def _identifier(row):
    value = _first(row, "tool_call_id", "toolCallId", "toolCallID", "tool_use_id", "toolUseId",
                   "call_id", "callID", "id")
    return str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) else ""


def _name(row):
    return _string(_first(row, "name", "toolName", "tool_name", "tool"))


def _tool_details(row, *, result=False):
    detail = {}
    if not result:
        value = _first(row, "input", "args", "arguments", "parameters", "parsedArgs", "action")
        if value is not None:
            detail["input"] = value
    output_fields = ("output", "result", "response", "observation", "error")
    if result or isinstance(row.get("toolResult"), dict):
        output_fields += ("content", "text")
    value = _first(row, *output_fields)
    if value is not None:
        detail["output"] = value
    is_error = _first(row, "is_error", "isError")
    success = row.get("success")
    if isinstance(is_error, bool):
        detail["is_error"] = str(is_error).lower()
    elif isinstance(success, bool):
        detail["is_error"] = str(not success).lower()
    code = _first(row, "exit_code", "exitCode")
    if type(code) is int:
        detail["exit_code"] = str(code)
    status = _string(row.get("status")).lower().replace("_", "-")
    if status in ("pending", "waiting", "running", "in-progress", "requested", "partial"):
        detail["status"] = {"waiting": "pending", "in-progress": "running"}.get(status, status)
    elif status in ("cancelled", "canceled", "aborted", "rejected", "denied", "skipped", "declined"):
        detail["status"] = {"canceled": "cancelled"}.get(status, status)
    elif status in ("error", "failed", "failure"):
        detail["is_error"] = "true"
    elif status in ("completed", "complete", "success", "succeeded", "ok"):
        detail.setdefault("is_error", "false")
    return detail


def _tool_event(row, *, result=False):
    # Goose wraps MCP requests/results in an externally tagged Result.
    wrapped = _obj(_first(row, "functionCall", "functionResponse", "toolUse", "toolResult",
                          "toolCall", "function"))
    inner = _obj(wrapped.get("value")) or wrapped
    state = _obj(row.get("state"))
    fields = {**row, **inner, **state}
    if state.get("content") is not None:
        fields["output"] = state["content"]
    detail = _tool_details(fields, result=result)
    call_id = _identifier(wrapped) or _identifier(row)
    if call_id:
        detail["call_id"] = call_id
    if wrapped.get("status") == "error" and isinstance(wrapped.get("value"), str):
        detail.update(output=wrapped["value"], is_error="true")
    return ("tool_result" if result else "tool_call", _name(inner) or _name(row)), detail


def _chat(row, role=None, text=None):
    role = role or row.get("role")
    if row.get("isMeta") or row.get("isSidechain"):
        return None
    content = row.get("content") if text is None else text
    model = _string(row.get("model"))
    if role in RESULT_ROLES:
        activity, details = _structured_activity(row, content, include_calls=False)
        if activity:
            return Message("activity", "", model, activity, details)
        step, detail = _tool_event({**row, "content": content}, result=True)
        return Message("activity", "", model, [step], {0: detail})
    if role not in ("user", "assistant", "model", "gemini"):
        return None
    activity, details = _structured_activity(row, content, include_calls=role != "user")
    result = _text(content)
    is_tool_call = any(kind == "tool_call" for kind, _ in activity)
    gemini_answer = role == "gemini" and bool(result) and bool(_items(row.get("toolCalls")))
    if (row.get("phase") == "commentary" or (is_tool_call and not gemini_answer)
            or any(kind == "tool_result" for kind, _ in activity)
            or any(row.get(key) in ("tool_use", "toolUse", "tool-calls", "tool_calls", "aborted", "error", "abort")
                   for key in ("stop_reason", "stopReason", "finishReason", "finish"))):
        if role != "user" and result:
            activity.append(("progress", result))
        return Message("activity", "", model, activity, details)
    if result.startswith(("<system-reminder>", "<environment_context>", "<local-command-")):
        return None
    return Message("prompt" if role == "user" else "response", result, model, activity, details)


def _structured_activity(row, content, *, include_calls):
    summary = _string(row.get("reasoning_summary") or row.get("reasoningSummary"))
    activity = [("rationale", summary)] if summary and include_calls else []
    details = {}

    def append(step, detail):
        details[len(activity)] = detail
        activity.append(step)

    calls = next((_items(row.get(key)) for key in CALL_FIELDS if _items(row.get(key))), ())
    for part in map(_obj, _items(content)):
        kind = _string(part.get("type"))
        if kind in RESULT_TYPES or any(key in part for key in ("functionResponse", "toolResult")):
            append(*_tool_event(part, result=True))
        elif include_calls and (kind in CALL_TYPES or any(key in part for key in ("functionCall", "toolUse"))):
            append(*_tool_event(part))
    if include_calls:
        for call in calls:
            if isinstance(call, str):
                append(("tool_call", call), {})
            elif isinstance(call, dict):
                append(*_tool_event(call))
    return activity, details


def _activity_message(row, *, result=False, model=""):
    step, detail = _tool_event(row, result=result)
    return Message("activity", "", model, [step], {0: detail})


def _extension(row):
    message_type = row.get("type")
    kind = row.get(message_type) if message_type in ("ask", "say") else None
    if row.get("partial") or not isinstance(kind, str):
        return None
    text = row.get("text")
    if kind in ("tool", "command", "use_mcp_server", "browser_action_launch", "browser_action"):
        try:
            payload = _obj(json.loads(text)) if isinstance(text, str) else _obj(text)
        except ValueError:
            payload = {}
        name = _name(payload) or ("browser_action" if kind == "browser_action_launch" else kind)
        fields = {"name": name, "input": payload or text}
        if row["type"] == "ask":
            fields["status"] = "requested"
        return _activity_message(fields)
    if kind in ("command_output", "mcp_server_response", "browser_action_result", "user_feedback_diff"):
        parent = {"command_output": "command", "mcp_server_response": "use_mcp_server",
                  "browser_action_result": "browser_action", "user_feedback_diff": ""}[kind]
        message = _activity_message({"name": parent, "output": text}, result=True)
        message.tool_details[0]["match_latest"] = True
        return message
    role = {
        "task": "user", "user_feedback": "user", "text": "assistant",
        "completion_result": "assistant",
    }.get(kind)
    return _chat({}, role, text) if role else None


def _copilot(row):
    data = _obj(row.get("data"))
    kind = row.get("type")
    if kind == "tool.execution_start":
        return _activity_message(data)
    if kind == "tool.execution_complete":
        message = _activity_message(data, result=True)
        message.tool_details[0].setdefault("status", "result saved")
        return message
    role = {"user.message": "user", "assistant.message": "assistant"}.get(kind)
    if not role or data.get("agentId") or data.get("parentToolCallId"):
        return None
    return _chat(data, role, data.get("content"))


def _openhands(row):
    action = _obj(row.get("action"))
    kind = row.get("kind")
    if kind == "ActionEvent":
        event = _activity_message(row)
        if row.get("tool_name") == "finish" and _string(action.get("message")):
            event.role = "response"
            event.text = action["message"]
        return event
    if kind in ("ObservationEvent", "UserRejectObservation", "AgentErrorEvent"):
        if not (row.get("tool_call_id") or row.get("action_id") or row.get("tool_name")):
            return None
        observation = _obj(row.get("observation"))
        details = {**row, **observation, "tool_call_id": row.get("tool_call_id") or row.get("action_id")}
        if kind == "UserRejectObservation":
            details["status"] = "rejected"
        elif kind == "AgentErrorEvent":
            details["is_error"] = True
        return _activity_message(details, result=True)
    if kind == "MessageEvent":
        role = {"user": "user", "agent": "assistant", "assistant": "assistant"}.get(row.get("source"))
        return _chat(_obj(row.get("llm_message")), role)
    if row.get("action") in ("message", "finish"):
        args = _obj(row.get("args"))
        role = {"user": "user", "agent": "assistant"}.get(row.get("source"))
        return _chat({}, role, args.get("content") or args.get("final_thought"))
    if isinstance(row.get("action"), str) and row.get("source") == "agent":
        return _activity_message({"name": row["action"], "id": row.get("id"), "input": row.get("args")})
    if isinstance(row.get("observation"), str):
        extras = _obj(row.get("extras"))
        return _activity_message({**extras, "name": row["observation"], "call_id": row.get("cause"),
                                  "output": row.get("content")}, result=True)
    return None


def _message(reader, row):
    if reader == "opencode":
        return _chat({**row, "model": _opencode_model(row)})
    if reader == "copilot":
        return _copilot(row)
    if reader in ("cursor", "droid", "pi", "continue", "qwen"):
        inner = _obj(row.get("message"))
        if reader == "pi" and row.get("type") != "message":
            return None
        if reader == "cursor":
            role = row.get("role") or {1: "user", 2: "assistant"}.get(row.get("type"))
            return _chat(inner or row, role)
        if reader == "qwen":
            if row.get("type") not in ("user", "assistant", "tool", "tool_result") or row.get("teammateTurn"):
                return None
            content = inner.get("parts") or inner.get("content")
            visible = row.get("displayContent")
            return _chat({**row, **inner, "model": row.get("model") or inner.get("model")}, row["type"],
                         content if isinstance(content, list) else visible or content)
        if reader == "continue" and _items(row.get("toolCallStates")):
            fields = inner or row
            calls = list(_items(fields.get("toolCalls")))
            for value in map(_obj, row["toolCallStates"]):
                call = _obj(value.get("toolCall"))
                if call:
                    calls.append({**call, **{key: item for key, item in value.items()
                                            if key in ("parsedArgs", "status", "output")}})
            return _chat({**fields, "toolCalls": calls}, row.get("role"))
        return _chat(inner or row, row.get("role"))
    if reader == "gemini":
        content = row.get("content") or row.get("parts")
        role = row.get("type") or row.get("role")
        return _chat(row, role, content if isinstance(content, list) else row.get("displayContent") or content)
    if reader == "extension":
        return _extension(row)
    if reader == "kiro":
        message = _obj(row.get("message")) or _obj(row.get("data")) or row
        kind = row.get("kind")
        content = message.get("content") or message.get("parts")
        if isinstance(content, list):
            content = [{**_obj(part), **_obj(_obj(part).get("data")),
                        "type": _obj(part).get("type") or _obj(part).get("kind")} for part in content]
        role = {"Prompt": "user", "UserMessage": "user", "AssistantMessage": "assistant",
                "ToolResult": "tool"}.get(kind)
        return _chat(message, role, content)
    if reader == "openhands":
        return _openhands(row)
    if reader == "antigravity":
        if row.get("type") == "USER_INPUT":
            content = _string(row.get("content"))
            match = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", content, re.S)
            return _chat({}, "user", match[1] if match else content.split("<ADDITIONAL_METADATA>", 1)[0])
        if row.get("type") == "PLANNER_RESPONSE":
            return _chat(row, "assistant")
        if row.get("type") in ("TOOL_CALL", "TOOL_RESULT", "TOOL_RESPONSE"):
            return _activity_message(row, result=row["type"] != "TOOL_CALL")
    return _chat(row)


def _session_id(reader, file):
    if file.name in ("context.jsonl", "events.jsonl", "ui_messages.json"):
        return file.parent.name
    if reader in ("vibe", "kiro") and file.stem == "messages":
        return file.parent.name
    if (reader == "antigravity" and file.name in ("transcript.jsonl", "transcript_full.jsonl")
            and len(file.parents) >= 3):
        return file.parents[2].name
    return file.stem


class TurnActivity:
    def __init__(self):
        self.steps = []
        self.details = {}
        self.calls = {}

    def add(self, step, detail=None):
        kind, name = step
        detail = dict(detail or {})
        call_id = detail.pop("call_id", "")
        match_latest = detail.pop("match_latest", False)
        if kind in ("tool_call", "tool_result"):
            index = self.calls.get(call_id)
            if match_latest and index is None:
                index = next((i for i in reversed(range(len(self.steps)))
                              if self.steps[i][0] == "tool_call" and
                              (not name or self.steps[i][1] == name)), None)
            if index is None and kind == "tool_result" and not call_id and name:
                matches = [i for i, (step_kind, label) in enumerate(self.steps)
                           if step_kind == "tool_call" and label == name
                           and not any(key in self.details[i] for key in ("output", "is_error", "exit_code"))]
                if len(matches) == 1:
                    index = matches[0]
            if index is None:
                index = len(self.steps)
                self.steps.append(("tool_call", name or "Unidentified tool"))
                if call_id:
                    self.calls[call_id] = index
                self.details[index] = {}
            elif kind == "tool_call" and name and self.steps[index][1] in ("Unidentified tool", "tool"):
                self.steps[index] = ("tool_call", name)
            current = self.details[index]
            if "model" in current:
                detail.pop("model", None)
            if kind == "tool_result" or "is_error" in detail or "exit_code" in detail:
                current.pop("status", None)
                if kind == "tool_result" and not any(key in detail for key in ("is_error", "exit_code", "status", "output")):
                    detail["status"] = "result saved"
            elif current.get("output") is not None or "is_error" in current or current.get("status") in {
                "result saved", "cancelled", "aborted", "rejected", "denied", "skipped", "declined",
            }:
                detail.pop("status", None)
            current.update(detail)
            return
        if detail:
            self.details[len(self.steps)] = detail
        self.steps.append(step)

    def finish(self, answer, *, model="", tools_only=False):
        tools = [i for i, (kind, _) in enumerate(self.steps) if kind == "tool_call"]
        if tools_only:
            if not tools:
                return None
            return Message("activity", "", model, [self.steps[i] for i in tools],
                           {index: self.details[i] for index, i in enumerate(tools)}, local_reason="subagent")
        if answer is None:
            if not tools:
                return None
            answer = Message("activity", "", model)
        answer.activity = self.steps
        answer.tool_details = self.details
        return answer


def _pairs(rows, reader, session, *, child=False):
    prompt = answer = None
    turn = TurnActivity()
    finished = []
    call_turns = {}
    start = 0
    for position, row in rows:
        msg = row if isinstance(row, Message) else _message(reader, row) if row is not None else None
        if row is None or (msg and msg.role == "prompt"):
            if answer or turn.steps:
                finished.append((start, prompt, answer, turn))
            prompt, answer, start = (msg.text if msg else None), None, position
            turn = TurnActivity()
        elif msg and (prompt or msg.activity):
            if msg.role == "activity" and any(kind == "tool_call" for kind, _ in msg.activity) and answer:
                turn.add(("progress", answer.text), {"model": answer.model} if answer.model else None)
                answer = None
            for index, step in enumerate(msg.activity):
                detail = dict(msg.tool_details.get(index, {}))
                if msg.model:
                    detail.setdefault("model", msg.model)
                call_id = detail.get("call_id")
                owner = call_turns.get(call_id, turn) if step[0] == "tool_result" else turn
                if call_id:
                    call_turns[call_id] = owner
                owner.add(step, detail)
            if prompt and msg.role == "response" and msg.text:
                answer = msg
    finished.append((start, prompt, answer, turn))
    for position, request, response, activity in finished:
        saved = activity.finish(response, tools_only=child)
        if saved and (request or saved.role == "activity"):
            yield session, position, "" if child else request or "", saved


def _jsonl(stream, stats):
    for i, line in enumerate(stream, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            row = None
        if not isinstance(row, dict):
            stats["malformed_lines"] += 1
            row = None
        yield i, row


def _deepseek_message(row):
    kind = row.get("type")
    data = _obj(row.get("data"))
    if kind == "tool/call":
        return _activity_message({**data, "call_id": data.get("callId")})
    if kind in ("tool/ptc-dispatch-start", "tool/code-dispatch-start"):
        return _activity_message({**data, "call_id": data.get("subCallId"), "status": "running"})
    if kind in ("tool/ptc-dispatch", "tool/code-dispatch"):
        return _activity_message({
            **data, "call_id": data.get("subCallId"),
            "output": _text(data.get("content")) or _obj(data.get("error")).get("reason"),
        })
    if row.get("surfaceOp", "append") != "append":
        return None  # Surface rewrites from compaction are not new chat messages.
    if kind == "user/message" and _obj(data.get("source")).get("kind") == "user":
        return _chat(data, "user")
    if kind == "assistant/message":
        message = _obj(data.get("message")) or data
        source = _obj(message.get("source")) or _obj(message.get("provenance"))
        return _chat({"content": message.get("content"), "model": source.get("model"),
                      "phase": "commentary" if data.get("interrupted") else ""}, "assistant")
    if kind == "tool/result":
        message = _obj(data.get("message")) or data
        wrapper = next((part for part in map(_obj, _items(message.get("content")))
                        if part.get("type") == "tool-result"), message)
        failure = _obj(data.get("error"))
        return _activity_message({
            "call_id": _identifier(wrapper) or wrapper.get("callId") or _obj(message.get("source")).get("callId"),
            "output": _text(wrapper.get("content")) or _string(failure.get("reason")) or None,
            "isError": wrapper.get("isError", bool(failure)),
        }, result=True)
    return None


def _deepseek_turns(stream, stats, *, child):
    rows = _jsonl(stream, stats)
    _, header = next(rows, (0, None))
    if (not header or header.get("type") != "session" or not _string(header.get("id"))
            or type(header.get("version")) is not int or not 0 <= header["version"] <= 4):
        raise ValueError("Unsupported DeepSeek session header")
    child |= header.get("origin") == "subagent"

    def messages():
        inherited = header.get("isSeeded") is True
        seed_length = header.get("seedLength")
        turn_position = 0
        for position, row in rows:
            if row is None:
                turn_position += 1
                yield turn_position, None
                continue
            kind = row.get("type")
            if kind == "session/end-seed":
                inherited = False
                continue
            if inherited or (type(seed_length) is int and position <= seed_length + 1):
                continue
            if kind in ("turn/start", "turn/end"):
                turn_position += 1
                yield turn_position, None
                continue
            message = _deepseek_message(row)
            if message is not None:
                # Logical turn positions survive migrations that insert unrelated events.
                if message.role == "prompt":
                    turn_position += 1
                yield turn_position, message

    yield from _pairs(messages(), "messages", header["id"], child=child)


def _read_deepseek(file, stats):
    child = is_child_history(file)
    if file.suffix.lower() == ".zip":
        with zipfile.ZipFile(file) as archive:
            logs = []
            for entry in archive.infolist():
                path = PurePosixPath(entry.filename)
                if (path.suffix == ".jsonl" and DEEPSEEK_LOG_NAME.fullmatch(path.name)
                        and (len(path.parts) == 1 or (len(path.parts) == 3 and path.parts[0] == "subagents"))):
                    logs.append((entry, child or is_child_history(path)))
            if sum(entry.file_size for entry, _ in logs) > 128 * 1024 * 1024:
                raise ValueError("Transcript archive is too large")
            for entry, nested in logs:
                with archive.open(entry) as stream:
                    yield from _deepseek_turns(stream, stats, child=nested)
        return
    with file.open("rb") as stream:
        if file.suffix.lower() == ".zstd":
            try:
                with BufferedReader(ZstdDecompressor().stream_reader(stream, read_across_frames=True)) as decoded:
                    yield from _deepseek_turns(decoded, stats, child=child)
            except ZstdError as error:
                raise ValueError("Unreadable DeepSeek Zstandard session") from error
        else:
            yield from _deepseek_turns(stream, stats, child=child)


ANTIGRAVITY_RESULT_STATUS = {
    "DONE": "completed", "ERROR": "failed", "INVALID": "failed", "RUNNING": "running",
    "CANCELLED": "cancelled", "CANCELED": "cancelled", "SKIPPED": "skipped",
}


def _antigravity_key(row):
    return (row.get("step_index"), _string(row.get("type")),
            _string(row.get("source")), _string(row.get("status")))


def _antigravity_rows(rows, file):
    # The shorter transcript has the row positions used by prior imports. Use
    # the full sibling for tool payloads that the shorter log marks as truncated.
    full = {}
    sibling = file.with_name("transcript_full.jsonl")
    if file.name == "transcript.jsonl" and not sibling.is_symlink():
        try:
            with sibling.open("rb") as stream:
                for _, row in _jsonl(stream, {"malformed_lines": 0}):
                    if row and type(row.get("step_index")) is int:
                        full[_antigravity_key(row)] = row
        except OSError:
            pass

    pending = {}
    for position, row in rows:
        if row is None:
            pending.clear()
            yield position, row
            continue
        kind, source, step = row.get("type"), row.get("source"), row.get("step_index")
        if kind == "PLANNER_RESPONSE":
            pending.clear()
            calls = _items(row.get("tool_calls"))
            if type(step) is int and calls:
                complete = full.get(_antigravity_key(row), {})
                if "tool_calls" in _items(row.get("truncated_fields")):
                    calls = _items(complete.get("tool_calls")) or calls
                identified = []
                for offset, raw in enumerate(calls, 1):
                    call = _obj(raw) or {"name": _string(raw)}
                    call_id = _identifier(call) or f"antigravity:{position}:{offset}"
                    identified.append({**call, "tool_call_id": call_id})
                    pending[step + offset] = call_id
                row = {**row, "tool_calls": identified}
        elif kind == "USER_INPUT":
            pending.clear()
        elif kind == "GENERIC" and source == "MODEL" and type(step) is int and step in pending:
            status = ANTIGRAVITY_RESULT_STATUS.get(_string(row.get("status")))
            if status:
                result = full.get(_antigravity_key(row), row) if "content" in _items(row.get("truncated_fields")) else row
                output = result.get("content")
                if result.get("error"):
                    output = {"content": output, "error": result["error"]} if output else result["error"]
                fields = {"tool_call_id": pending[step], "output": output, "status": status}
                if status != "running":
                    pending.pop(step)
                yield position, _activity_message(fields, result=True)
                continue
        yield position, row


def _pi_branch(rows):
    nodes = {row["id"]: (index, row) for index, row in rows if row and isinstance(row.get("id"), str)
             and "parentId" in row}
    if not nodes:
        return rows
    chain = []
    leaf = next(reversed(nodes))
    while isinstance(leaf, str) and leaf in nodes:
        index, row = nodes.pop(leaf)
        chain.append((index, row))
        leaf = row.get("parentId")
    return reversed(chain)


def _gemini_history(rows):
    # Gemini JSONL can rewind and amend messages after the initial write.
    messages = {}
    for position, record in rows:
        if record is None:
            continue
        header = _obj(record.get("$set"))
        if isinstance(header.get("messages"), list):
            messages = {(_string(obj.get("id")) or index): (position + index, obj)
                        for index, obj in enumerate(map(_obj, header["messages"]))}
        rewind = _string(record.get("$rewindTo"))
        if rewind in messages:
            while messages and next(reversed(messages)) != rewind:
                messages.popitem()
        patch = _obj(record.get("$patch"))
        for update in [patch, *map(_obj, _items(patch.get("updates")))]:
            key = update.get("id")
            if isinstance(key, str) and key in messages:
                index, msg = messages[key]
                messages[key] = index, {**msg, **update}
        for key in _items(patch.get("removeIds")):
            if isinstance(key, str):
                messages.pop(key, None)
        if record.get("type") in ("user", "gemini", "tool", "tool_result"):
            key = record.get("id")
            messages[key if isinstance(key, str) else position] = position, record
    return messages.values()


def _markdown(stream, reader, session):
    user_heading = re.compile(r"^#{1,6}\s+(?:\*\*)?(?:user|you|human|prompt)(?:\*\*)?\s*:?[ \t]*$", re.I)
    assistant_heading = re.compile(
        r"^#{1,6}\s+(?:\*\*)?(?:assistant|amp|kiro|agent|response)(?:\*\*)?\s*:?[ \t]*$", re.I
    )
    role = ""
    body = []
    start = 0
    fence = None
    tool_output = []

    def emit():
        if role:
            yield session, start, {"role": role, "content": "\n".join(body)}
        if tool_output:
            yield session, start, _activity_message({"name": "Aider tool output",
                                                    "output": "\n".join(tool_output)}, result=True)
            tool_output.clear()

    for i, raw in enumerate(stream, 1):
        line = raw.decode("utf-8", "replace").rstrip()
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            char = marker[1][0]
            if fence is None or fence == char:
                fence = char if fence is None else None
            body.append(line)
            continue
        if fence is None:
            if reader == "aider" and line.startswith("# aider chat started at "):
                yield from emit()
                session = f"{session.split(':', 1)[0]}:{line[2:].strip()}:{i}"
                role, body = "", []
                continue
            if reader == "aider":
                if line.startswith("#### "):
                    text = line[5:]
                    if role == "user":
                        body.append(text)
                        continue
                    yield from emit()
                    role, start, body = "user", i, [text]
                    continue
                if line.startswith("> ") or line == ">":
                    tool_output.append(line.removeprefix(">").removeprefix(" "))
                    continue
                if not line.strip() and role == "user":
                    continue
                if role == "user":
                    yield from emit()
                    role, start, body = "assistant", i, []
            else:
                heading = "user" if user_heading.match(line) else "assistant" if assistant_heading.match(line) else ""
                if reader == "cursor":
                    heading = {"user:": "user", "assistant:": "assistant"}.get(line, heading)
                if heading:
                    if role:
                        yield session, start, {"role": role, "content": "\n".join(body)}
                    role, start, body = heading, i, []
                    continue
        if role:
            body.append(line)
    yield from emit()


def _opencode_model(message):
    model = message.get("model")
    return _string(message.get("modelID") or _obj(model).get("id") or model)


def _opencode_tool_name(part):
    return _string(part.get("name") or part.get("tool"))


def _opencode_messages(conn):
    latest = conn.execute("SELECT 1 FROM sqlite_schema WHERE type='table' AND name='session_v2'").fetchone()
    if latest:
        rows = conn.execute(
            "SELECT m.session_id, m.id, m.data, p.data, m.type, s.parent_id "
            "FROM session_message m JOIN session_v2 s ON s.id=m.session_id "
            "LEFT JOIN session_part p ON p.session_id=m.session_id AND p.message_id=m.id "
            "AND p.type IN ('text', 'tool') "
            "WHERE m.type IN ('user','assistant','model-switched') "
            "ORDER BY m.session_id, m.seq, p.ordinal"
        )
    else:
        rows = conn.execute(
            "SELECT m.session_id, m.id, m.data, p.data, NULL, s.parent_id "
            "FROM message m JOIN session s ON s.id=m.session_id "
            "LEFT JOIN part p ON p.message_id=m.id "
            "ORDER BY m.session_id, m.time_created, m.id, p.id"
        )
    previous_session = None
    child = False
    activity = []
    tool_details = {}
    for (session, _), block_rows in groupby(rows, key=lambda row: (row[0], row[1])):
        if session != previous_session:
            if activity:
                yield previous_session, Message("activity", "", activity=activity, tool_details=tool_details), True, child
            activity = []
            tool_details = {}
            previous_session = session
        blocks = list(block_rows)
        child = bool(blocks[0][5])
        message = _obj(json.loads(blocks[0][2]))
        if message.get("summary") is True:
            continue
        role = blocks[0][4] or message.get("role")
        model = _opencode_model(message)
        if role == "model-switched":
            if model:
                tool_details[len(activity)] = {"model": model}
                activity.append(("model_change", model))
            continue
        parts = [_obj(json.loads(row[3])) for row in blocks if row[3]]
        if role == "user" and activity:
            trailing = [(step, tool_details.get(i, {})) for i, step in enumerate(activity)
                        if step[0] != "model_change"]
            if trailing:
                steps, details = zip(*trailing)
                yield session, Message("activity", "", activity=list(steps),
                                       tool_details=dict(enumerate(details))), True, child
            activity = [step for step in activity if step[0] == "model_change"]
            tool_details = {index: {"model": label} for index, (_, label) in enumerate(activity)}
        if role == "assistant" and any(part.get("type") == "tool" for part in parts):
            # Keep the reader's existing row positions so saved turns can gain
            # activity without being imported a second time.
            for part in parts:
                if part.get("type") == "tool":
                    name = _opencode_tool_name(part)
                    if name:
                        _, detail = _tool_event(part)
                        if model:
                            detail["model"] = model
                        tool_details[len(activity)] = detail
                        activity.append(("tool_call", name))
                elif part.get("type") == "text" and not part.get("synthetic"):
                    progress = _string(part.get("text"))
                    if progress:
                        if model:
                            tool_details[len(activity)] = {"model": model}
                        activity.append(("progress", progress))
            continue
        content = message.get("text") if latest and role == "user" else parts or message.get("content")
        message_row = {**message, "role": role, "model": model, "content": content}
        parsed = _message("opencode", message_row)
        if parsed and parsed.role != "prompt":
            offset = len(activity)
            tool_details.update((offset + index, detail) for index, detail in parsed.tool_details.items())
            parsed.activity[:0] = activity
            parsed.tool_details = tool_details
            activity = []
            tool_details = {}
        yield session, parsed or message_row, False, child
    if activity:
        yield previous_session, Message("activity", "", activity=activity, tool_details=tool_details), True, child


def _db_turns(file, reader):
    with contextlib.closing(sqlite3.connect(file.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("PRAGMA trusted_schema=OFF")
        if reader == "opencode":
            yield from _opencode_messages(conn)
        elif reader == "goose":
            rows = conn.execute(
                "SELECT session_id, role, content_json, metadata_json FROM messages "
                "ORDER BY session_id, created_timestamp, id"
            )
            for session, role, content, metadata in rows:
                if _obj(json.loads(metadata or "{}")).get("userVisible", True):
                    yield session, {"role": role, "content": json.loads(content)}, False, False
                else:
                    row = _chat({"role": role, "content": json.loads(content)})
                    if row and row.activity:
                        row.role, row.text = "activity", ""
                        tools = [(step, row.tool_details.get(i, {})) for i, step in enumerate(row.activity)
                                 if step[0] in ("tool_call", "tool_result")]
                        if tools:
                            steps, details = zip(*tools)
                            yield session, Message("activity", "", activity=list(steps),
                                                   tool_details=dict(enumerate(details))), True, False
        elif reader == "crush":
            columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
            summary_filter = "AND NOT m.is_summary_message" if "is_summary_message" in columns else ""
            rows = conn.execute(
                "SELECT m.session_id, m.role, m.parts, m.model, s.parent_session_id FROM messages m "
                "JOIN sessions s ON s.id=m.session_id WHERE 1=1 "
                f"{summary_filter} ORDER BY m.session_id, m.created_at, m.rowid"
            )
            for session, role, parts, model, parent in rows:
                content = [{**_obj(part.get("data")), "type": part.get("type"),
                            "text": _obj(part.get("data")).get("text") or _obj(part.get("data")).get("Text")}
                           for part in map(_obj, _items(json.loads(parts)))]
                yield session, {"role": role, "content": content, "model": model}, False, bool(parent)


def _json_rows(document, reader):
    if isinstance(document, list):
        return [_obj(row) for row in document]
    doc = _obj(document)
    if reader == "continue":
        return _items(doc.get("history"))
    if reader == "opencode":
        return [{**_obj(_obj(row).get("info")), "content": _obj(row).get("parts")}
                for row in _items(doc.get("messages"))]
    if reader == "amp":
        return _items(doc.get("messages") or _obj(doc.get("thread")).get("messages"))
    return _items(doc.get("messages") or doc.get("events") or doc.get("trajectory"))


def read_turns(file, reader, stats):
    """Use a source-specific decoder, then pair consecutive human/assistant turns."""
    if reader == "deepseek":
        yield from _read_deepseek(file, stats)
        return
    session = _session_id(reader, file)
    child = is_child_history(file)
    suffix = file.suffix.lower()
    if suffix in (".sqlite", ".db"):
        for (chat_id, nested), entries in groupby(_db_turns(file, reader), key=lambda row: (row[0], row[3])):
            position = 0

            def positioned_rows(entries):
                nonlocal position
                for _, row, synthetic, _ in entries:
                    if not synthetic:
                        position += 1
                    yield position, row

            yield from _pairs(positioned_rows(entries), "messages", chat_id, child=child or nested)
        return

    if suffix == ".zip":
        if reader not in ("kiro", "openhands"):
            return
        with zipfile.ZipFile(file) as archive:
            names = [entry for entry in archive.infolist() if not entry.is_dir()]
            if sum(entry.file_size for entry in names) > 128 * 1024 * 1024:
                raise ValueError("Transcript archive is too large")
            if reader == "openhands":
                events = [entry for entry in names
                          if re.fullmatch(r"\d+(-[^/]*)?\.json", PurePosixPath(entry.filename).name)]
                events.sort(key=lambda entry: int(re.match(r"\d+", PurePosixPath(entry.filename).name)[0]))
                rows = ((index, _obj(json.loads(archive.read(entry)))) for index, entry in enumerate(events, 1))
                yield from _pairs(rows, reader, session, child=child)
            else:
                candidates = [entry for entry in names if PurePosixPath(entry.filename).name == "messages.jsonl"]
                for entry in candidates:
                    with archive.open(entry) as stream:
                        path = PurePosixPath(entry.filename)
                        nested = child or is_child_history(path)
                        chat_id = f"{session}:{path.parent}" if nested else path.parent.name or session
                        yield from _pairs(_jsonl(stream, stats), reader, chat_id, child=nested)
        return

    with file.open("rb") as stream:
        if suffix in (".md", ".txt"):
            for chat_id, rows in groupby(_markdown(stream, reader, session), key=lambda row: row[0]):
                yield from _pairs(((index, row) for _, index, row in rows), "messages", chat_id, child=child)
        elif suffix == ".json":
            document = json.load(stream)
            header = _obj(document)
            session = (header.get("sessionId") or header.get("id")
                       or _obj(header.get("info")).get("id") or _obj(header.get("thread")).get("id") or session)
            child |= header.get("kind") == "subagent"
            rows = _json_rows(document, reader)
            yield from _pairs(((i, _obj(row)) for i, row in enumerate(rows, 1)), reader, str(session), child=child)
        elif suffix == ".jsonl":
            rows = _jsonl(stream, stats)
            if reader == "antigravity":
                rows = _antigravity_rows(rows, file)
            if reader in ("pi", "gemini", "droid", "qwen"):
                collected = list(rows)
                header = next((row for _, row in collected if row), {})
                meta = _obj(header.get("$set")) or header
                child |= meta.get("kind") == "subagent" or bool(meta.get("isSidechain"))
                session = (meta.get("sessionId")
                           or (meta.get("id") if reader in ("pi", "droid") else None) or session)
                rows = _pi_branch(collected) if reader == "pi" else collected
                if reader == "gemini":
                    rows = _gemini_history(collected)
            yield from _pairs(rows, reader, str(session), child=child)
