"""Import user-visible turns from local coding agent sessions."""
from __future__ import annotations

import json
import sqlite3
import zipfile

from ..consent import CONSENT_VERSION, commercial_attested
from ..history_sources import SOURCES, history_files
from ..models import CaptureEvent, sha256_text, ulid
from ..privacy import redact


from .history_readers import Message, read_turns


def _text(content, kinds):
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n".join(part["text"] for part in content if isinstance(part, dict)
                     and part.get("type") in kinds and isinstance(part.get("text"), str)).strip()


def _claude(row):
    msg = row.get("message")
    if not isinstance(msg, dict) or row.get("isSidechain"):
        return None
    kind = row.get("type")
    content = msg.get("content")
    if kind == "user":
        if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_result"
                                             for b in content):
            return None
        text = _text(content, {"text"})
        if row.get("isMeta") or row.get("isCompactSummary") or text.startswith(
            ("<local-command-", "<command-name>", "<system-reminder>")
        ):
            text = ""
        return Message("prompt", text)
    if kind == "assistant":
        if msg.get("stop_reason") == "tool_use" or (
            isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_use"
                                              for b in content)
        ):
            return Message("response", "")
        return Message("response", _text(content, {"text"}), msg.get("model") or "")
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
        item = p.get("item")
        if kind == "item_completed" and isinstance(item, dict):
            if item.get("type") == "UserMessage":
                return Message("prompt", _text(item.get("content"), {"text", "Text"}))
            if item.get("type") == "AgentMessage" and item.get("phase") != "commentary":
                return Message("response", _text(item.get("content"), {"Text", "text"}))
    if row.get("type") == "response_item" and kind == "message":
        role = p.get("role")
        if role == "assistant" and p.get("phase") != "commentary":
            return Message("response", _text(p.get("content"), {"output_text"}))
        if legacy and role == "user":
            text = _text(p.get("content"), {"input_text"})
            if text.startswith(("<environment_context>", "# AGENTS.md instructions", "<permissions instructions>")):
                text = ""
            return Message("prompt", text)
    return None


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
    start = 0
    model = ""
    for line_no, row in _rows(stream):
        if row is None:
            stats["malformed_lines"] += 1
            prompt = answer = None
            continue
        if source == "claude-code":
            session = row.get("sessionId") or session
            msg = _claude(row)
        else:
            payload = row.get("payload")
            if isinstance(payload, dict):
                if row.get("type") == "session_meta":
                    if isinstance(payload.get("source"), dict) and "subagent" in payload["source"]:
                        return
                    session = payload.get("id") or session
                elif row.get("type") == "turn_context":
                    model = payload.get("model") or ""
            msg = _codex(row, legacy)
        if msg is None:
            continue
        if msg.role in ("prompt", "boundary"):
            if prompt and answer:
                yield session, start, prompt, answer
            prompt, answer, start = msg.text, None, line_no
        elif prompt:
            answer = msg if msg.text else None
            if answer and not answer.model:
                answer.model = model
    if prompt and answer:
        yield session, start, prompt, answer


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


def import_history(config, vault, source, path=None, seen=None):
    check_history_consent(config, vault)
    files = history_files(source, path, config.data_dir / "history")
    reader = SOURCES[source].reader
    disabled = vault.disabled_history_chats(source)
    stats = dict(files=len(files), imported=0, already_imported=0, disabled_skipped=0,
                 malformed_lines=0, unreadable_files=0)
    for file in files:
        try:
            stamp = file.stat()
            signature = (stamp.st_dev, stamp.st_ino, stamp.st_size, stamp.st_mtime_ns, disabled)
            wal = file.with_name(file.name + "-wal")
            if file.suffix.lower() in (".sqlite", ".db") and wal.is_file():
                wal_info = wal.stat()
                signature += (wal_info.st_size, wal_info.st_mtime_ns)
            if seen is not None and seen.get((source, file)) == signature:
                continue
            for session, start, prompt, answer in _file_turns(file, source, reader, stats):
                conv = vault.keys.tag(f"{source}:{session}")
                if conv in disabled:
                    stats["disabled_skipped"] += 1
                    continue
                turn = vault.keys.tag(f"{conv}:{start}:{sha256_text(prompt)}:{sha256_text(answer.text)}")
                if vault.history_imported(turn):
                    stats["already_imported"] += 1
                    continue
                events = []
                for index, (role, text) in enumerate((("prompt", prompt), ("response", answer.text))):
                    prepared = redact(text, config)
                    events.append(CaptureEvent(
                        id=ulid(), source=source, surface=source, role=role, text=text,
                        conversation_id=conv, turn_index=start * 2 + index,
                        raw_hash=sha256_text(text), model_hint=answer.model,
                        redacted_text=prepared.text, redactions=prepared.redactions,
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
