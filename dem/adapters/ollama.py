"""Reverse proxy in front of Ollama/local models — the cleanest, lowest-risk source.

Captures /api/chat and /api/generate request+response, relays bytes live so the
client experience is unchanged, and emits encrypted CaptureEvents on completion.
"""
from __future__ import annotations

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..config import Config
from ..models import sha256_text
from ..vault import Vault
from .base import CaptureSink, parse_ndjson_chunks, relay_http


def _prompt_from(body: dict, endpoint: str) -> str:
    if endpoint == "/api/generate":
        return body.get("prompt", "")
    messages = body.get("messages", [])
    return "\n".join(
        str(m.get("content", "")) for m in messages if m.get("role") == "user"
    )


def _content_from(chunks: list[dict], endpoint: str) -> str:
    parts = []
    for obj in chunks:
        if endpoint == "/api/chat":
            part = (obj.get("message") or {}).get("content", "")
        else:
            part = obj.get("response", "")
        if part:
            parts.append(str(part))
    return "".join(parts)


def make_handler(config: Config, sink: CaptureSink):
    upstream = config.get("capture", "ollama_upstream", "http://127.0.0.1:11434").rstrip("/")

    class OllamaHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass  # no logs of URLs/content (privacy)

        def _auth_headers(self):
            auth = self.headers.get("Authorization")
            return {"Authorization": auth} if auth else {}

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", 0) or 0)
            return self.rfile.read(length) if length else b""

        def do_GET(self):
            url = upstream + self.path
            try:
                relay_http(self, url, b"", pass_headers=self._auth_headers)
            except Exception as exc:
                self.send_error(502, f"upstream error: {type(exc).__name__}")

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            body = self._read_body()
            endpoint = parsed.path
            url = upstream + self.path
            try:
                raw = relay_http(self, url, body, pass_headers=self._auth_headers)
            except Exception as exc:
                self.send_error(502, f"upstream error: {type(exc).__name__}")
                return
            if endpoint not in ("/api/chat", "/api/generate"):
                return
            try:
                request = json.loads(body or b"{}")
            except json.JSONDecodeError:
                return
            prompt = _prompt_from(request, endpoint)
            if not prompt.strip():
                return
            response = _content_from(parse_ndjson_chunks(raw), endpoint)
            if not response.strip():
                return
            conv_id = "ollama-" + sha256_text(prompt)[:16]
            prompt_preview = sink.prepare(prompt)
            sink.ingest(
                text=response, role="response", source="ollama",
                surface=f"ollama:{upstream}", conversation_id=conv_id,
                turn_index=1, model_hint=str(request.get("model", "")),
                metadata={
                    "endpoint": endpoint,
                    "prompt_text": prompt_preview.text,  # already scrubbed
                    "capture_prompts_enabled": config.enabled("capture", "capture_prompts"),
                },
            )
            if config.enabled("capture", "capture_prompts"):
                sink.ingest(
                    text=prompt, role="prompt", source="ollama",
                    surface=f"ollama:{upstream}", conversation_id=conv_id,
                    turn_index=0, model_hint=str(request.get("model", "")),
                )

    return OllamaHandler


def start_ollama_proxy(config: Config, vault: Vault) -> tuple[ThreadingHTTPServer, threading.Thread]:
    sink = CaptureSink(config, vault)
    handler = make_handler(config, sink)
    port = int(config.get("capture", "ollama_port", 11435))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, name="dem-ollama", daemon=True)
    thread.start()
    return server, thread
