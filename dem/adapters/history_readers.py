"""Read user-facing turns and structured activity from local harness formats."""
from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import zipfile
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import PurePosixPath


@dataclass
class Message:
    role: str
    text: str
    model: str = ""
    activity: list[tuple[str, str]] = field(default_factory=list)
    tool_details: dict[int, dict] = field(default_factory=dict)


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


def _chat(row, role=None, text=None):
    role = role or row.get("role")
    if role not in ("user", "assistant", "model", "gemini"):
        return None
    if row.get("isMeta") or row.get("isSidechain"):
        return None
    content = row.get("content") if text is None else text
    model = _string(row.get("model"))
    activity = _structured_activity(row, content) if role != "user" else []
    if row.get("phase") == "commentary":
        progress = _text(content) if role != "user" else ""
        if progress:
            activity.append(("progress", progress))
        return Message("activity", "", model, activity)
    if (row.get("tool_calls") or row.get("toolCalls") or row.get("toolRequests")
            or row.get("stop_reason") in ("tool_use", "tool_calls")
            or row.get("stopReason") in ("toolUse", "aborted", "error")
            or row.get("finishReason") in ("tool-calls", "error", "abort")
            or row.get("finish") in ("tool-calls", "tool_calls", "error", "abort")):
        return Message("activity", "", model, activity)
    if any(any(key in _obj(part) for key in ("functionCall", "functionResponse"))
           or _obj(part).get("type") in ("tool_use", "tool_result", "tool-call", "tool-result")
           for part in _items(content)):
        return Message("activity", "", model, activity)
    result = _text(content)
    if result.startswith(("<system-reminder>", "<environment_context>", "<local-command-")):
        return None
    return Message("prompt" if role == "user" else "response", result, model, activity)


def _structured_activity(row, content):
    summary = _string(row.get("reasoning_summary") or row.get("reasoningSummary"))
    activity = [("rationale", summary)] if summary else []
    calls = _items(row.get("tool_calls") or row.get("toolCalls") or row.get("toolRequests"))
    if not calls:
        for part in map(_obj, _items(content)):
            if part.get("type") in ("tool_use", "tool-call", "toolCall", "tool") or "functionCall" in part:
                call = _obj(part.get("functionCall")) or part
                name = call.get("name") or call.get("toolName") or call.get("tool")
                if isinstance(name, str):
                    activity.append(("tool_call", name))
        return activity
    for call in map(_obj, calls):
        name = _obj(call.get("function")).get("name") or call.get("name") or call.get("toolName")
        if isinstance(name, str):
            activity.append(("tool_call", name))
    return activity


def _extension(row):
    if row.get("type") == "ask" and row.get("ask") in ("followup", "tool", "command"):
        return None
    if row.get("type") != "say":
        return None
    role = {
        "task": "user", "user_feedback": "user", "text": "assistant",
        "completion_result": "assistant",
    }.get(row.get("say"))
    return _chat({}, role, row.get("text"))


def _message(reader, row):
    if reader == "opencode":
        message = _chat({**row, "model": _opencode_model(row)})
        if message and message.role != "prompt":
            tool_steps = [(index, name) for index, (kind, name) in enumerate(message.activity) if kind == "tool_call"]
            parts = [part for part in map(_obj, _items(row.get("content")))
                     if part.get("type") == "tool" and _opencode_tool_name(part)]
            if [name for _, name in tool_steps] == [_opencode_tool_name(part) for part in parts]:
                message.tool_details = {index: _opencode_tool_details(part) for (index, _), part in zip(tool_steps, parts)}
        return message
    if reader == "copilot":
        data = _obj(row.get("data"))
        role = {"user.message": "user", "assistant.message": "assistant"}.get(row.get("type"))
        if data.get("agentId") or data.get("parentToolCallId"):
            return None
        return _chat(data, role, data.get("content"))
    if reader in ("cursor", "droid", "pi", "continue", "qwen"):
        inner = _obj(row.get("message"))
        if reader == "pi" and row.get("type") != "message":
            return None
        if reader == "cursor":
            role = row.get("role") or {1: "user", 2: "assistant"}.get(row.get("type"))
            return _chat(inner or row, role)
        if reader == "qwen":
            if row.get("type") not in ("user", "assistant") or row.get("teammateTurn"):
                return None
            text = row.get("displayContent") or inner.get("parts") or inner.get("content")
            return _chat({**inner, "model": row.get("model")}, row["type"], text)
        return _chat(inner or row, row.get("role"))
    if reader == "gemini":
        return _chat(row, row.get("type") or row.get("role"),
                     row.get("displayContent") or row.get("content") or row.get("parts"))
    if reader == "extension":
        return _extension(row)
    if reader == "kiro":
        message = _obj(row.get("message")) or _obj(row.get("data")) or row
        role = {"Prompt": "user", "AssistantMessage": "assistant"}.get(row.get("kind"))
        return _chat(message, role)
    if reader == "openhands":
        if row.get("action") in ("message", "finish"):
            args = _obj(row.get("args"))
            role = {"user": "user", "agent": "assistant"}.get(row.get("source"))
            return _chat({}, role, args.get("content") or args.get("final_thought"))
        kind = row.get("kind")
        if kind == "MessageEvent":
            msg = _obj(row.get("llm_message"))
            role = {"user": "user", "agent": "assistant"}.get(row.get("source"))
            return _chat(msg, role)
        if kind == "ActionEvent" and row.get("tool_name") == "finish":
            action = _obj(row.get("action"))
            return _chat({}, "assistant", action.get("message"))
        return None
    if reader == "antigravity":
        if row.get("type") == "USER_INPUT":
            content = _string(row.get("content"))
            match = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", content, re.S)
            return _chat({}, "user", match[1] if match else content.split("<ADDITIONAL_METADATA>", 1)[0])
        if row.get("type") == "PLANNER_RESPONSE":
            return _chat(row, "assistant")
    return _chat(row)


def _session_id(reader, file):
    if file.name in ("context.jsonl", "events.jsonl", "ui_messages.json"):
        return file.parent.name
    if reader in ("vibe", "kiro") and file.stem == "messages":
        return file.parent.name
    if reader == "antigravity" and file.name == "transcript.jsonl" and len(file.parents) >= 3:
        return file.parents[2].name
    return file.stem


def _pairs(rows, reader, session):
    prompt = answer = None
    activity = []
    tool_details = {}
    for position, row in rows:
        if row is None:
            prompt = answer = None
            activity = []
            tool_details = {}
            continue
        msg = row if isinstance(row, Message) else _message(reader, row)
        if msg is None:
            continue
        if msg.role == "prompt":
            if prompt and answer:
                answer.activity = activity
                answer.tool_details = tool_details
                yield session, start, prompt, answer
            prompt, answer, start = msg.text, None, position
            activity = []
            tool_details = {}
        elif prompt:
            if msg.role == "activity" and any(kind == "tool_call" for kind, _ in msg.activity):
                if answer:
                    if answer.model:
                        tool_details[len(activity)] = {"model": answer.model}
                    activity.append(("progress", answer.text))
                answer = None
            for index, _ in enumerate(msg.activity):
                detail = dict(msg.tool_details.get(index, {}))
                if msg.model:
                    detail.setdefault("model", msg.model)
                if detail:
                    tool_details[len(activity) + index] = detail
            activity.extend(msg.activity)
            if msg.role == "response" and msg.text:
                answer = msg
    if prompt and answer:
        answer.activity = activity
        answer.tool_details = tool_details
        yield session, start, prompt, answer


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
        if record.get("type") in ("user", "gemini"):
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
                if role:
                    yield session, start, {"role": role, "content": "\n".join(body)}
                session = f"{session.split(':', 1)[0]}:{line[2:].strip()}:{i}"
                role, body = "", []
                continue
            if reader == "aider":
                if line.startswith("#### "):
                    text = line[5:]
                    if role == "user":
                        body.append(text)
                        continue
                    if role:
                        yield session, start, {"role": role, "content": "\n".join(body)}
                    role, start, body = "user", i, [text]
                    continue
                if line.startswith("> ") or not line.strip() and role == "user":
                    continue
                if role == "user":
                    yield session, start, {"role": role, "content": "\n".join(body)}
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
    if role:
        yield session, start, {"role": role, "content": "\n".join(body)}


def _opencode_model(message):
    model = message.get("model")
    return _string(message.get("modelID") or _obj(model).get("id") or model)


def _opencode_tool_name(part):
    return _string(part.get("name") or part.get("tool"))


def _opencode_tool_details(part):
    state = _obj(part.get("state"))
    details = {}
    if state.get("input") is not None:
        details["input"] = state["input"]
    if state.get("status") in ("completed", "error"):
        details["is_error"] = "true" if state["status"] == "error" else "false"
    for key in ("content", "output", "error"):
        if state.get(key) is not None:
            details["output"] = state[key]
            break
    return details


def _opencode_messages(conn):
    latest = conn.execute("SELECT 1 FROM sqlite_schema WHERE type='table' AND name='session_v2'").fetchone()
    if latest:
        rows = conn.execute(
            "SELECT m.session_id, m.id, m.data, p.data, m.type "
            "FROM session_message m JOIN session_v2 s ON s.id=m.session_id "
            "LEFT JOIN session_part p ON p.session_id=m.session_id AND p.message_id=m.id "
            "AND p.type IN ('text', 'tool') "
            "WHERE s.parent_id IS NULL AND m.type IN ('user','assistant','model-switched') "
            "ORDER BY m.session_id, m.seq, p.ordinal"
        )
    else:
        rows = conn.execute(
            "SELECT m.session_id, m.id, m.data, p.data, NULL "
            "FROM message m JOIN session s ON s.id=m.session_id "
            "LEFT JOIN part p ON p.message_id=m.id "
            "WHERE s.parent_id IS NULL ORDER BY m.session_id, m.time_created, m.id, p.id"
        )
    previous_session = None
    activity = []
    tool_details = {}
    for (session, _), block_rows in groupby(rows, key=lambda row: (row[0], row[1])):
        if session != previous_session:
            activity = []
            tool_details = {}
            previous_session = session
        blocks = list(block_rows)
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
        if role == "user":
            activity = [step for step in activity if step[0] == "model_change"]
            tool_details = {index: {"model": model} for index, (_, model) in enumerate(activity)}
        if role == "assistant" and any(part.get("type") == "tool" for part in parts):
            # Keep the reader's existing row positions so saved turns can gain
            # activity without being imported a second time.
            for part in parts:
                if part.get("type") == "tool":
                    name = _opencode_tool_name(part)
                    if name:
                        detail = _opencode_tool_details(part)
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
        yield session, parsed or message_row


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
                    yield session, {"role": role, "content": json.loads(content)}
        elif reader == "crush":
            columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
            summary_filter = "AND NOT m.is_summary_message" if "is_summary_message" in columns else ""
            rows = conn.execute(
                "SELECT m.session_id, m.role, m.parts, m.model FROM messages m "
                "JOIN sessions s ON s.id=m.session_id WHERE COALESCE(s.parent_session_id,'')='' "
                f"{summary_filter} ORDER BY m.session_id, m.created_at, m.rowid"
            )
            for session, role, parts, model in rows:
                content = [{"type": part.get("type"), "text": _obj(part.get("data")).get("text")
                            or _obj(part.get("data")).get("Text")}
                           for part in map(_obj, _items(json.loads(parts)))]
                yield session, {"role": role, "content": content, "model": model}


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
    session = _session_id(reader, file)
    suffix = file.suffix.lower()
    if suffix in (".sqlite", ".db"):
        for chat_id, entries in groupby(_db_turns(file, reader), key=lambda row: row[0]):
            yield from _pairs(enumerate((row for _, row in entries), 1), "messages", chat_id)
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
                yield from _pairs(rows, reader, session)
            else:
                candidates = [entry for entry in names if PurePosixPath(entry.filename).name == "messages.jsonl"
                              and "sub-executions" not in PurePosixPath(entry.filename).parts]
                for entry in candidates:
                    with archive.open(entry) as stream:
                        chat_id = PurePosixPath(entry.filename).parent.name or session
                        yield from _pairs(_jsonl(stream, stats), reader, chat_id)
        return

    with file.open("rb") as stream:
        if suffix in (".md", ".txt"):
            for chat_id, rows in groupby(_markdown(stream, reader, session), key=lambda row: row[0]):
                yield from _pairs(((index, row) for _, index, row in rows), "messages", chat_id)
        elif suffix == ".json":
            document = json.load(stream)
            header = _obj(document)
            session = (header.get("sessionId") or header.get("id")
                       or _obj(header.get("info")).get("id") or _obj(header.get("thread")).get("id") or session)
            if header.get("kind") == "subagent":
                return
            rows = _json_rows(document, reader)
            yield from _pairs(((i, _obj(row)) for i, row in enumerate(rows, 1)), reader, str(session))
        elif suffix == ".jsonl":
            rows = _jsonl(stream, stats)
            if reader in ("pi", "gemini", "droid", "qwen"):
                collected = list(rows)
                header = next((row for _, row in collected if row), {})
                meta = _obj(header.get("$set")) or header
                if meta.get("kind") == "subagent" or meta.get("isSidechain"):
                    return
                session = (meta.get("sessionId")
                           or (meta.get("id") if reader in ("pi", "droid") else None) or session)
                rows = _pi_branch(collected) if reader == "pi" else collected
                if reader == "gemini":
                    rows = _gemini_history(collected)
            yield from _pairs(rows, reader, str(session))
