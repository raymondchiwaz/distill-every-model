"""Token-gated review UI served exclusively on localhost."""
from __future__ import annotations

import difflib
import html
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

from .adapters.history import check_history_consent, import_history
from .config import Config
from .consent import CONSENT_TEXT, CONSENT_VERSION, approve_capture, record_commercial_attestation
from .history_sources import HISTORY_SOURCES, SOURCES, history_files
from .vault import Vault

STYLE = Path(__file__).with_name("review.css").read_text(encoding="utf-8")
SOURCE_NAMES = {key: source.name for key, source in SOURCES.items()}
escape = html.escape


def _page(body: str, chats_url: str, captures_url: str, section: str) -> str:
    def nav(label, url, icon, name):
        current = ' aria-current="page"' if section == name else ""
        return f'<a{current} href="{escape(url)}"><span aria-hidden="true">{icon}</span>{label}</a>'

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DEM · Review workspace</title><style>{STYLE}</style></head>
<body>
<aside class="sidebar">
  <div class="brand"><span class="mark" aria-hidden="true">D</span><div>DEM<small>Distill Every Model</small></div></div>
  <nav aria-label="Workspace">
    {nav("Chat history", chats_url, "▤", "history")}
    {nav("Other captures", captures_url, "≡", "captures")}
  </nav>
  <div class="local"><span></span> Local workspace</div>
</aside>
<main>{body}</main>
</body></html>"""


def _source(source: str) -> str:
    symbol = "✳" if source == "claude-code" else "⌘"
    return f'<span class="source {source}"><b aria-hidden="true">{symbol}</b>{SOURCE_NAMES[source]}</span>'


def make_handler(config: Config, vault: Vault, token: str):
    # The review server shares a SQLite connection between its request threads.
    lock = threading.Lock()

    def url(path="/", **params):
        return path + "?" + urlencode({"token": token, **params})

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, body: str, status: int = 200):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; "
                             "form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _route(self):
            request = urlsplit(self.path)
            self.query = {key: values[0] for key, values in parse_qs(request.query).items()}
            if not secrets.compare_digest(self.query.get("token", "").encode(), token.encode()):
                self._send("Unauthorized", 401)
                return None
            return request.path

        def do_GET(self):
            route = self._route()
            with lock:
                if route == "/":
                    self._index()
                elif route == "/chat":
                    self._chat()
                elif route == "/captures":
                    self._captures()
                elif route is not None:
                    self._send("Not found", 404)

        def do_POST(self):
            route = self._route()
            if route is None:
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._send("Invalid form length", 400)
            if length < 0:
                return self._send("Invalid form length", 400)
            form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
            with lock:
                if route == "/chat/toggle":
                    chat_id = self.query.get("chat", "")
                    if not chat_id:
                        return self._send("Chat not found", 404)
                    enabled = form.get("enabled", [""])[0]
                    if enabled not in ("0", "1"):
                        return self._send("Invalid chat setting", 400)
                    vault.set_history_chat_enabled(chat_id, enabled == "1")
                    return self._redirect(self._return_url())
                if route == "/import":
                    return self._import()
                if route == "/import/consent":
                    if form.get("consent") != ["yes"]:
                        return self._consent("Confirm that you agree to import the local chats.")
                    current_config = Config(path=config.path)
                    if current_config.get("privacy", "consent_mode") != "default-deny":
                        return self._index("History import requires default-deny consent mode.")
                    current_config.set("capture", "capture_prompts", True)
                    record_commercial_attestation(current_config, vault)
                    if vault.latest_decision("onboard", CONSENT_VERSION) != "approved":
                        approve_capture(vault)
                    return self._import()
                if route in ("/approve", "/reject", "/edit"):
                    event = vault.get_capture(self.query.get("id", ""))
                    if event is None:
                        return self._send("Not found", 404)
                    if route == "/reject":
                        vault.update_consent(event.id, "rejected")
                    elif event.quarantine_reason:
                        return self._send("This message is quarantined.", 400)
                    elif route == "/approve":
                        vault.update_consent(event.id, "approved")
                    else:
                        text = form.get("text", [""])[0].strip()
                        if text:
                            vault.edit_capture(event.id, text)
                    target = url("/chat", chat=event.conversation_id) if event.source in HISTORY_SOURCES else url("/captures")
                    return self._redirect(target)
                if route == "/bulk-approve-clean":
                    for event in vault.list_captures("pending", include_history=False):
                        if event.source in ("ollama", "manual") and not event.quarantine_reason:
                            vault.update_consent(event.id, "approved")
                    return self._redirect(url("/captures"))
                self._send("Not found", 404)

        def _redirect(self, target):
            self.send_response(303)
            self.send_header("Location", target)
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()

        def _render(self, body, section="history"):
            self._send(_page(body, url(), url("/captures"), section))

        def _import(self):
            current_config = Config(path=config.path)
            try:
                check_history_consent(current_config, vault)
            except ValueError as exc:
                if current_config.get("privacy", "consent_mode") != "default-deny":
                    return self._index(str(exc))
                return self._consent()
            results = [import_history(current_config, vault, source) for source in HISTORY_SOURCES]
            return self._redirect(url(imported=sum(r["imported"] for r in results),
                                      unreadable=sum(r["unreadable_files"] for r in results)))

        def _consent(self, notice=""):
            status = f'<p class="notice" role="status">{escape(notice)}</p>' if notice else ""
            self._render(f"""
<a class="back" href="{escape(url())}">← All chats</a>
<header><div class="intro"><h1>Allow local chat imports</h1>
<p>DEM will copy saved coding-agent prompts and responses into its encrypted local review queue.</p></div></header>
<article class="message">{status}{self._consent_form()}</article>""")

        def _consent_form(self):
            return f"""
<details><summary>Read the DEM consent policy</summary><pre>{escape(CONSENT_TEXT)}</pre></details>
<form method="post" action="{escape(url('/import/consent'))}">
<label class="consent"><input type="checkbox" name="consent" value="yes" required>
<span>I agree to DEM’s consent policy and authorize local capture of my prompts and
responses, including outputs from paid models I use.</span></label>
<button class="primary">Agree and import chats</button>
</form>"""

        def _empty_history(self):
            detected, inventory = self._history_inventory()
            current_config = Config(path=config.path)
            action = f'<form method="post" action="{escape(url("/import"))}"><button class="primary">Import chats</button></form>'
            if current_config.get("privacy", "consent_mode") != "default-deny":
                guidance = "Enable default-deny consent mode to import local chats."
                action = ""
            elif not any(detected.values()):
                guidance = "Save a session in a supported tool or add a local export, then refresh this page."
                action = ""
            else:
                try:
                    check_history_consent(current_config, vault)
                except ValueError:
                    guidance = "Your saved chats will appear here after you consent to import them."
                    action = self._consent_form()
                else:
                    guidance = "Import the saved conversations to review them and choose which chats to include."
            return (f'<article class="message empty-history"><h2>No imported chats yet</h2>'
                    f'<p>{guidance}</p>{inventory}{action}</article>')

        def _history_inventory(self):
            export_root = config.data_dir / "history"
            detected = {source: len(history_files(source, export_root=export_root))
                        for source in HISTORY_SOURCES}

            def inventory_item(source, count):
                label = "source files" if count != 1 else "source file"
                return f'<li>{_source(source)} <strong>{count}</strong> {label}</li>'

            found = "".join(inventory_item(source, count) for source, count in detected.items() if count)
            empty = "".join(inventory_item(source, 0) for source, count in detected.items() if not count)
            return detected, (
                f'<ul aria-label="Detected local history files">{found}</ul>' if found else ""
            ) + (
                f'<details><summary>Other supported tools ({len(SOURCES) - sum(bool(n) for n in detected.values())})'
                f'</summary><ul>{empty}</ul><p>Place exports under <code>{escape(str(export_root))}'
                '/&lt;source&gt;</code>, then choose Import chats.</p></details>' if empty else ""
            )

        def _return_url(self):
            if self.query.get("view") == "chat":
                return url("/chat", chat=self.query["chat"])
            filters = {key: self.query[key] for key in ("source", "state") if key in self.query}
            return url(**filters)

        def _switch(self, chat, **context):
            enabled = bool(chat["enabled"])
            state = "On" if enabled else "Off"
            checked = str(enabled).lower()
            action = escape(url("/chat/toggle", chat=chat["conversation_id"], **context))
            return (f'<form method="post" action="{action}" class="toggle">'
                    f'<button type="submit" role="switch" aria-checked="{checked}" '
                    f'aria-label="Include chat: {escape(self._title(chat))}" name="enabled" value="{int(not enabled)}">'
                    f'<span class="track" aria-hidden="true"><span></span></span><span aria-hidden="true">{state}</span>'
                    '</button></form>')

        @staticmethod
        def _title(chat):
            return chat["title"] or f"Conversation {chat['conversation_id'][:8]}"

        def _index(self, notice=""):
            chats = vault.history_chats()
            source = self.query.get("source", "")
            state = self.query.get("state", "")
            visible = [c for c in chats if (not source or c["source"] == source)
                       and (not state or bool(c["enabled"]) == (state == "on"))]
            enabled = sum(c["enabled"] for c in chats)
            pending = sum(c["pending"] for c in chats if c["enabled"])
            if "imported" in self.query:
                pairs = self.query["imported"]
                notice = f"Imported {pairs} new {'chat pair' if pairs == '1' else 'chat pairs'}."
                if pairs == "0" and not chats and self.query.get("unreadable", "0") == "0":
                    notice += " The import found no completed prompt and response pairs."
                if self.query.get("unreadable", "0") != "0":
                    notice += f" Could not read {self.query['unreadable']} transcript files."
            status = f'<p class="notice" role="status">{escape(notice)}</p>' if notice else ""
            if not chats:
                return self._render(f"""
<header><div class="intro"><h1>Chat history</h1>
<p>Choose which conversations DEM uses for training data.</p></div></header>
{status}{self._empty_history()}""")
            source_options = []
            for key, label in (("", "All sources"), *SOURCE_NAMES.items()):
                count = sum(not key or c["source"] == key for c in chats)
                selected = " selected" if source == key else ""
                source_options.append(f'<option value="{key}"{selected}>{label} ({count})</option>')
            options = "".join(
                f'<option value="{key}"{" selected" if state == key else ""}>{label}</option>'
                for key, label in (("", "Any status"), ("on", "On"), ("off", "Off"))
            )
            cards = []
            for chat in visible:
                pending_count = chat["pending"]
                review = f'<span class="badge">{pending_count} to review</span>' if pending_count else ""
                if chat["quarantined"]:
                    review += f'<span class="badge">Quarantined · {chat["quarantined"]}</span>'
                detail = escape(url("/chat", chat=chat["conversation_id"]))
                cards.append(
                    '<li class="chat-row"><div class="chat-main">'
                    f'<div class="chat-meta">{_source(chat["source"])} {review}</div>'
                    f'<a class="chat-title" href="{detail}">{escape(self._title(chat))}</a>'
                    f'<span class="chat-meta">{chat["messages"]} messages</span></div>'
                    f'{self._switch(chat, source=source, state=state)}</li>'
                )
            listing = (
                f'<ul class="chat-list">{"".join(cards)}</ul>' if cards else
                '<div class="empty"><h2>No chats match these filters</h2>'
                f'<a href="{escape(url())}">View all chats</a></div>'
            )
            self._render(f"""
<header><div class="intro"><h1>Chat history</h1>
<p>Choose which conversations DEM uses for training data.</p></div>
<form method="post" action="{escape(url('/import'))}"><button class="primary">Import chats</button></form></header>
{status}
<div class="stats" aria-label="Chat summary">
  <div><span>Total chats</span><strong>{len(chats)}</strong></div>
  <div><span>Enabled chats</span><strong>{enabled}</strong></div>
  <div><span>Messages to review</span><strong>{pending}</strong></div>
</div>
<div class="section-intro"><h2>Conversations</h2>
<p>Turning a chat off pauses imports and excludes it from new releases.</p></div>
<section class="panel" aria-label="Conversations">
  <div class="toolbar">
    <form class="filter" action="/" method="get"><input type="hidden" name="token" value="{escape(token)}">
      <select name="source" aria-label="History source">{"".join(source_options)}</select>
      <select name="state" aria-label="Chat status">{options}</select><button>Filter</button>
    </form>
  </div>
  {listing}
</section>""")

        def _chat(self):
            chat_id = self.query.get("chat", "")
            chat = next((chat for chat in vault.history_chats() if chat["conversation_id"] == chat_id), None)
            if chat is None:
                return self._send("Chat not found", 404)
            messages = "".join(self._message(event) for event in vault.history_messages(chat_id))
            if not messages:
                messages = '<div class="empty"><h2>No saved messages</h2><p>Turn on this chat and import again.</p></div>'
            self._render(f"""
<a class="back" href="{escape(url())}">← All chats</a>
<header><div class="intro">{_source(chat["source"])}<h1>{escape(self._title(chat))}</h1>
<p>Review both messages in each pair before building a release.</p></div>
{self._switch(chat, view="chat")}</header>
<div class="messages">{messages}</div>""")

        def _message(self, event):
            role = "Prompt" if event.role == "prompt" else "Response"
            turn = event.metadata.get("history_turn", "")
            pair = f" · Pair {escape(turn[:8])}" if turn else ""
            label = f"{role}{pair}" if turn else f"{escape(event.source)} · {role}"
            reject = escape(url("/reject", id=event.id))
            if event.quarantine_reason:
                return (f'<article class="message"><h2>{label}</h2>'
                        f'<p class="notice">{escape(event.quarantine_reason)}</p>'
                        f'<form method="post" action="{reject}"><button>Discard</button></form></article>')
            redacted = event.redacted_text or ""
            final = escape(redacted)
            diff = escape("".join(difflib.ndiff(event.text.splitlines(keepends=True),
                                               redacted.splitlines(keepends=True))))
            approve = escape(url("/approve", id=event.id))
            edit = escape(url("/edit", id=event.id))
            return f"""<article class="message"><div class="message-head"><h2>{label}</h2>
<span class="badge">{escape(event.consent.capitalize())}</span></div>
<pre>{final}</pre>
<details><summary>Redaction diff</summary><pre>{diff}</pre></details>
<div class="actions">
<form method="post" action="{approve}"><button class="primary">Approve</button></form>
<form method="post" action="{reject}"><button>Reject</button></form></div>
<details><summary>Edit final text</summary>
<form method="post" action="{edit}">
<textarea name="text" aria-label="Final text" rows="5">{final}</textarea>
<button>Approve edited text</button></form></details></article>"""

        def _captures(self):
            events = vault.list_captures("pending", include_history=False)
            messages = "".join(self._message(event) for event in events)
            self._render(f"""<header><div class="intro"><h1>Other captures</h1>
<p>Review pending captures from local models, proxies, and manual entry.</p></div>
<form method="post" action="{escape(url('/bulk-approve-clean'))}"><button>Approve local captures</button></form>
</header><div class="messages">{messages or '<div class="empty"><h2>No pending captures</h2></div>'}</div>""",
                         "captures")

    return Handler


def _start_review(config: Config, vault: Vault):
    token = config.get("review", "token") or secrets.token_urlsafe(24)
    config.set("review", "token", token)
    config.save()
    port = int(config.get("review", "port", 8787))
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(config, vault, token))
    return server, f"http://127.0.0.1:{server.server_port}/?{urlencode({'token': token})}"


def start_review(config: Config, vault: Vault) -> str:
    server, url = _start_review(config, vault)
    threading.Thread(target=server.serve_forever, name="dem-review", daemon=True).start()
    return url


def serve_blocking(config: Config, vault: Vault) -> None:
    server, url = _start_review(config, vault)
    with server:
        print("DEM review UI (localhost only):", url, flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nReview UI stopped.")
