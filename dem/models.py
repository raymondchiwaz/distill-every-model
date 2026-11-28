"""Core data structures. All IDs are ULIDs; hashes are SHA-256."""
from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
SCHEMA_VERSION = "1.0.0"
ACTIVITY_PAYLOAD_FIELDS = ("input", "output")
ACTIVITY_RESULT_FIELDS = ("exit_code", "is_error", "status")
TASK_OUTCOMES = {"solved": "Solved", "partial": "Partly solved", "unsolved": "Unsolved"}


def ulid() -> str:
    ts = int(time.time() * 1000)
    out = ""
    for _ in range(10):
        out = CROCKFORD[ts % 32] + out
        ts //= 32
    rand = secrets.randbits(80)
    for _ in range(16):
        out += CROCKFORD[rand % 32]
        rand //= 32
    return out


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class CaptureEvent:
    id: str
    source: str                 # ollama | api_proxy | cli | browser | manual
    surface: str                # e.g. localhost:11434, chatgpt.com
    role: str                   # prompt | response | system | activity
    text: str
    conversation_id: str
    turn_index: int
    model_hint: str = ""
    timestamp: str = field(default_factory=now_utc)
    raw_hash: str = ""
    consent: str = "pending"    # pending | approved | rejected | redacted | local
    redacted_text: str | None = None
    redactions: list[dict] = field(default_factory=list)
    quarantine_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    activity: list[dict[str, str]] = field(default_factory=list)
    task_outcome: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CaptureEvent":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class DatasetRecord:
    id: str
    prompt: str
    response: str
    messages: list[dict]
    metadata: dict[str, Any]
    quality: dict[str, Any]
    redactions: list[dict]
    license: str
    rights_attestation: str
    source_tier: str            # clean | restricted
    derived_from: str           # open-model-output | proprietary-api-output | human-written
    consent_token: str
    pii_scrubbed: bool = True
    schema_version: str = SCHEMA_VERSION
    raw_hash: str = ""
    activity: list[dict[str, str]] = field(default_factory=list)
    redaction_summary: list[dict[str, str | int]] = field(default_factory=list)
    task_outcome: dict[str, str] | None = None
    evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetRecord":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})
