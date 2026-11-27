"""Generic base-URL API proxy (OpenAI-compatible / Anthropic-style endpoints).

Apps point their base URL at 127.0.0.1:<api_port>; the proxy forwards to the
configured upstream and tees chat responses into the local vault. Paid-model
outputs are tagged with their origin and grouped in the `restricted` partition.
"""
from __future__ import annotations

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..config import Config
from ..models import sha256_text
from ..vault import Vault
from .base import CaptureSink, parse_ndjson_chunks, parse_sse, relay_http

CAPTURE_PATHS = {"/v1/chat/completions", "/chat/completions", "/v1/messages", "/messages"}


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return str(content or "")


def _prompt_from(body: dict) -> str:
    return "\n".join(
        _text_of(m.get("content"))
        for m in body.get("messages", [])
        if m.get("role") == "user"
    )


def _response_from(raw: bytes, is_sse: bool) -> str:
    objects = parse_sse(raw) if is_sse else parse_ndjson_chunks(raw)
    parts = []
    for obj in objects:
        if "choices" in obj:
            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            part = delta.get("content") or message.get("content") or ""
            parts.append(_text_of(part))
        elif obj.get("type") == "content_block_delta":
            parts.append(_text_of((obj.get("delta") or {}).get("text", "")))
        elif obj.get("type") == "content_block_start":
            parts.append(_text_of((obj.get("content_block") or {}).get("text", "")))
    return "".join(parts)


def make_handler(config: Config, sink: CaptureSink):
    upstream = config.get("capture", "api_upstream", "").rstrip("/")

    class ApiHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _auth_headers(self):
            headers = {}
            for name in ("Authorization", "x-api-key", "anthropic-version"):
                value = self.headers.get(name)
                if value:
                    headers[name] = value
            return headers

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else b""
            ctype = self.headers.get("Content-Type", "application/json")
            url = upstream + self.path
            try:
                raw = relay_http(self, url, body, content_type=ctype,
                                 pass_headers=self._auth_headers)
            except Exception as exc:
                self.send_error(502, f"upstream error: {type(exc).__name__}")
                return
            if parsed.path not in CAPTURE_PATHS or not upstream:
                return
            try:
                request = json.loads(body or b"{}")
            except json.JSONDecodeError:
                return
            prompt = _prompt_from(request)
            response = _response_from(raw, "text/event-stream" in ctype)
            if not prompt.strip() or not response.strip():
                return
            surface = urllib.parse.urlparse(upstream).netloc or "api"
            conv_id = "api-" + sha256_text(prompt)[:16]
            prompt_preview = sink.prepare(prompt)
            sink.ingest(
                text=response, role="response", source="api_proxy", surface=surface,
                conversation_id=conv_id, turn_index=1,
                model_hint=str(request.get("model", "")),
                metadata={
                    "endpoint": parsed.path,
                    "prompt_text": prompt_preview.text,
                    "derived_from": "proprietary-api-output",
                },
            )
            if config.enabled("capture", "capture_prompts"):
                sink.ingest(
                    text=prompt, role="prompt", source="api_proxy", surface=surface,
                    conversation_id=conv_id, turn_index=0,
                    model_hint=str(request.get("model", "")),
                    metadata={"derived_from": "proprietary-api-output"},
                )

    return ApiHandler


def start_api_proxy(config: Config, vault: Vault) -> tuple[ThreadingHTTPServer, threading.Thread] | None:
    if not config.enabled("capture", "api_proxy") or not config.get("capture", "api_upstream"):
        return None
    sink = CaptureSink(config, vault)
    handler = make_handler(config, sink)
    port = int(config.get("capture", "api_port", 11436))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, name="dem-api", daemon=True)
    thread.start()
    return server, thread
