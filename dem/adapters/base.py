"""Shared capture sink and HTTP relay used by all adapters."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable

from ..config import Config
from ..consent import check_capture
from ..models import CaptureEvent, sha256_text, ulid
from ..privacy import RedactionResult, redact
from ..vault import Vault


class CaptureSink:
    """Single funnel: consent check -> redaction -> encrypted storage."""

    def __init__(self, config: Config, vault: Vault):
        self.config = config
        self.vault = vault

    def prepare(self, text: str) -> RedactionResult:
        return redact(text or "", self.config)

    def ingest(
        self,
        *,
        text: str,
        role: str,
        source: str,
        surface: str,
        conversation_id: str,
        turn_index: int,
        model_hint: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CaptureEvent | None:
        stub = CaptureEvent(
            id=ulid(), source=source, surface=surface, role=role, text=text,
            conversation_id=conversation_id, turn_index=turn_index,
            model_hint=model_hint,
        )
        allowed, reason = check_capture(self.config, stub)
        if not allowed:
            self.vault.log_audit("capture:blocked", f"{source}:{reason}", "consent-v3")
            return None
        prepared = self.prepare(text)
        stub.raw_hash = sha256_text(text)
        stub.redacted_text = prepared.text
        stub.redactions = prepared.redactions
        stub.quarantine_reason = prepared.quarantine_reason
        stub.metadata = metadata or {}
        stub.consent = "pending"
        self.vault.add_capture(stub)
        return stub


def relay_http(
    handler,
    upstream_url: str,
    body: bytes,
    content_type: str = "application/json",
    pass_headers: Callable[[], dict[str, str]] | None = None,
) -> bytes:
    """POST body to upstream, stream bytes back to the client live, return full body.

    Only chat/completion endpoints are captured by callers; auth headers are
    forwarded for the upstream call but are never stored or logged.
    """
    headers = {"Content-Type": content_type}
    if pass_headers:
        headers.update(pass_headers())
    req = urllib.request.Request(upstream_url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        ctype = resp.headers.get("Content-Type", content_type)
        handler.send_response(resp.status)
        handler.send_header("Content-Type", ctype)
        handler.end_headers()
        total = bytearray()
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            total.extend(chunk)
            try:
                handler.wfile.write(chunk)
                handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
    return bytes(total)


def parse_ndjson_chunks(raw: bytes) -> list[dict]:
    out = []
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def parse_sse(raw: bytes) -> list[dict]:
    out = []
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload in ("", "[DONE]"):
                continue
            try:
                out.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
    return out
