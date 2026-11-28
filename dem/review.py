"""Token-gated review UI served exclusively on localhost."""
from __future__ import annotations

import difflib
import html
import json
import re
import secrets
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

from huggingface_hub import get_token

from .adapters.history import check_history_consent, import_history
from .community import community_repo, prepare_contribution, submit_contribution
from .config import Config
from .consent import CONSENT_TEXT, CONSENT_VERSION, approve_capture, record_commercial_attestation
from .history_sources import HISTORY_SOURCES, SOURCES, history_files
from .models import ACTIVITY_PAYLOAD_FIELDS, TASK_OUTCOMES
from .pipeline.evidence import collect_task_evidence
from .pipeline.formatter import build_records
from .privacy import summarize_redactions
from .publisher import HF_ENDPOINT, PublishError, hub_token, prepare_private_chats, upload_hub_bundle
from .vault import Vault

STYLE = Path(__file__).with_name("review.css").read_text(encoding="utf-8")
SOURCE_NAMES = {key: source.name for key, source in SOURCES.items()}
escape = html.escape
DISPLAY_CODE_FENCE = re.compile(
    r"^ {0,3}(?P<fence>(?P<marker>`|~)(?P=marker){2,})(?P<info>[^\r\n]*)\r?\n"
    r"(?P<body>[\s\S]*?)^ {0,3}(?P=fence)(?P=marker)*[ \t]*(?=\r?$)", re.MULTILINE,
)


def _message_text(text: str) -> str:
    parts = []
    end = 0
    for match in DISPLAY_CODE_FENCE.finditer(text):
        prose = text[end:match.start()].strip("\r\n")
        if prose:
            parts.append(f'<pre class="message-text">{escape(prose)}</pre>')
        info = match["info"].split(maxsplit=1)
        label = escape(info[0] if info else "Code")
        parts.append(f'<figure class="code-block"><figcaption>{label}</figcaption>'
                     f'<pre tabindex="0"><code>{escape(match["body"])}</code></pre></figure>')
        end = match.end()
    prose = text[end:].strip("\r\n")
    if prose:
        parts.append(f'<pre class="message-text">{escape(prose)}</pre>')
    return "".join(parts)


def _tool_anchor(event_id: str, index: int) -> str:
    return f"tool-{event_id}-{index}"


def _tool_step(step: dict[str, str], event_id: str, index: int, expand: bool) -> str:
    anchor = escape(_tool_anchor(event_id, index))
    label = f'<span>Tool call</span><code>{escape(step["text"])}</code>'
    if not any(field in step for field in ACTIVITY_PAYLOAD_FIELDS):
        return f'<li id="{anchor}" class="tool-step">{label}</li>'
    sections = []
    if step.get("model"):
        sections.append(f'<p class="tool-note">Model <code>{escape(step["model"])}</code></p>')
    for field in ACTIVITY_PAYLOAD_FIELDS:
        if field not in step:
            sections.append(f'<p class="tool-note">{field.capitalize()} unavailable in the saved call.</p>')
            continue
        hold = step.get(f"{field}_hold")
        note = f'<p class="tool-note">Kept local after privacy screening ({escape(hold)}).</p>' if hold else ""
        payload = escape(step[field]) if step[field] else "(empty)"
        sections.append(f'<h3>{field.capitalize()}</h3>{note}<pre tabindex="0">{payload}</pre>')
    return (f'<li id="{anchor}" class="tool-step"><details class="tool-detail"{" open" if expand else ""}>'
            f'<summary>{label}</summary><div class="tool-payload">{"".join(sections)}</div></details></li>')


def _visible_activity(event):
    return [] if event.quarantine_reason and event.consent == "rejected" else event.activity


def _model_anchor(event_id: str, index: int | None = None) -> str:
    return f"model-{event_id}" if index is None else f"model-{event_id}-{index}"


def _evidence_panel(evidence: dict | None, outcome: str | None, *, can_review: bool) -> str:
    if not evidence and not outcome and not can_review:
        return ""
    content = []
    jump = '<a href="#task-outcome">Review task outcome</a>' if can_review else ""
    if outcome in TASK_OUTCOMES:
        content.append(f'<p>Contributor outcome <strong>{TASK_OUTCOMES[outcome]}</strong></p>')
    if evidence:
        if evidence["checks"]:
            checks = "".join(f'<li><code>{escape(check["name"])}</code> <strong>{escape(check["status"])}</strong></li>'
                             for check in evidence["checks"])
            content.append(f'<h3>Checks reported by tools</h3><ul class="evidence-checks">{checks}</ul>')
        if evidence["files_changed"]:
            files = "".join(f'<li><code>{escape(path)}</code></li>' for path in evidence["files_changed"])
            count = len(evidence["files_changed"])
            label = "file" if count == 1 else "files"
            content.append(f'<details><summary>{count} {label} changed</summary><ul>{files}</ul></details>')
        errors = evidence["tool_errors"]
        error_label = "error" if errors == 1 else "errors"
        if errors:
            content.append(f'<p>{errors} reported tool {error_label}</p>')
    else:
        content.append('<p>No file changes, tool errors, or check results could be identified from the saved activity.</p>')
    return (f'<section class="message evidence"><div class="tool-overview-head"><h2>Task evidence</h2>{jump}</div>'
            f'{"".join(content)}</section>')


def _page(body: str, chats_url: str, captures_url: str, hub_url: str, section: str) -> str:
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
    {nav("Hugging Face", hub_url, "↑", "hub")}
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
    session_hub_token = ""

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
                elif route == "/huggingface":
                    self._hub()
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
            if route in ("/huggingface/upload", "/huggingface/community") and length > 16_384:
                return self._send("Upload form is too large", 413)
            form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
            if route == "/huggingface/upload":
                return self._upload_dataset(form)
            if route == "/huggingface/community":
                return self._submit_community(form)
            with lock:
                if route == "/chat/activity":
                    return self._refresh_activity()
                if route == "/chat/approve":
                    chat_id = self.query.get("chat", "")
                    snapshot = form.get("snapshot", [""])[0]
                    outcome = form.get("task_outcome", [None])[0]
                    target = form.get("outcome_target", [""])[0]
                    if not vault.approve_history_chat(chat_id, snapshot, outcome=outcome, outcome_target=target):
                        return self._chat("This chat changed. Review the current text and try again.")
                    return self._redirect(url("/huggingface", chat=chat_id))
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
                    return self._import(form.get("path", [""])[0].strip())
                if route == "/import/consent":
                    path = form.get("path", [""])[0].strip()
                    if form.get("consent") != ["yes"]:
                        return self._consent("Confirm that you agree to import the local chats.", path)
                    current_config = Config(path=config.path)
                    if current_config.get("privacy", "consent_mode") != "default-deny":
                        return self._index("History import requires default-deny consent mode.")
                    current_config.set("capture", "capture_prompts", True)
                    record_commercial_attestation(current_config, vault)
                    if vault.latest_decision("onboard", CONSENT_VERSION) != "approved":
                        approve_capture(vault)
                    return self._import(path)
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
            self._send(_page(body, url(), url("/captures"), url("/huggingface"), section))

        def _upload_snapshot(self, records):
            payload = [(record.consent_token, record.prompt, record.response, record.activity,
                        record.task_outcome, record.metadata.get("model"))
                       for record in records]
            return vault.keys.tag(json.dumps(payload, sort_keys=True))

        def _hub(self, notice="", repo="", show_token=False, community_notice=""):
            current_config = Config(path=config.path)
            chat_id = self.query.get("chat")
            records = build_records(vault, current_config, conversation_id=chat_id)
            status = f'<p class="notice" role="status">{escape(notice)}</p>' if notice else ""
            header = '<header><div class="intro"><h1>Upload to Hugging Face</h1></div></header>'
            if chat_id:
                header = f'<a class="back" href="{escape(url("/chat", chat=chat_id))}">← Back to chat</a>' + header
            if not records:
                return self._render(f"""{header}{status}
<article class="message hub-card"><h2>Approve a chat to get started</h2>
<p>Open a chat, review its prompts and answers, then choose <strong>Approve chat and continue</strong>.</p>
<div class="actions">
<a class="button primary" href="{escape(url())}">Review chats</a>
<a href="{escape(url('/captures'))}">Other captures</a>
</div></article>""", "hub")

            snapshot = self._upload_snapshot(records)
            configured = current_config.get("publish", "repo")
            if configured == "dem/distill-every-model":
                configured = ""
            repo = repo or configured or f"dem-chats-{snapshot[:8]}"
            repo = str(repo)
            try:
                connected = session_hub_token or hub_token(config.data_dir / "keys")
            except OSError:
                connected = False
            token_field = f"""
<label>Hugging Face write token
<input name="hf_token" type="password" autocomplete="off" maxlength="512" placeholder="Paste your token">
</label>
<p><a href="{HF_ENDPOINT}/settings/tokens" target="_blank" rel="noopener noreferrer">Get a write token ↗</a></p>"""
            need_token = show_token or not connected
            count = len(records)
            pairs = "prompt and answer pair" if count == 1 else "prompt and answer pairs"
            selected = "from this chat" if chat_id else "from all enabled captures"
            action = url("/huggingface/upload", chat=chat_id) if chat_id else url("/huggingface/upload")
            community = self._community_form(current_config, records, chat_id, community_notice) if chat_id else ""
            private = f"""
  <h2>{count} approved {pairs} {selected}</h2>
  <p>Your prompts, answers, and any reviewed activity will go to a private dataset. DEM checks them for duplicates and sensitive information first.</p>
  <form class="hub-form" method="post" action="{escape(action)}">
    <input type="hidden" name="snapshot" value="{escape(snapshot)}">
    {token_field if need_token else ""}
    <p>By uploading, you confirm you have permission to upload these chats to Hugging Face.</p>
    <button class="primary">Upload to Hugging Face</button>
    <details>
      <summary>Dataset name and sign-in</summary>
      <label>Dataset name<input name="repo" value="{escape(repo)}" maxlength="193" required></label>
      {"" if need_token else token_field}
    </details>
  </form>"""
            if chat_id:
                private = f'<details{" open" if notice else ""}><summary>Upload to a private dataset</summary>{private}</details>'
            else:
                private = f'<p><a href="{escape(url())}">Choose a chat to contribute to a public dataset</a></p>{private}'
            self._render(f'{header}{community}{status}<article class="message hub-card">{private}</article>', "hub")

        def _community_form(self, current_config, records, chat_id, notice):
            contribution = prepare_contribution(records, current_config)
            configured = str(current_config.get("publish", "community_repo") or "").strip()
            status = f'<p class="notice" role="status">{escape(notice)}</p>' if notice else ""
            header = f'<h2>Contribute to the community</h2>{status}'
            if not contribution.count:
                return (f'<article class="message hub-card">{header}'
                        '<p>No approved pairs in this chat passed the community upload checks.</p></article>')

            destination = (f'<p>Hugging Face dataset <strong>{escape(configured)}</strong></p>' if configured else
                           '<label>Community dataset<input name="repo" placeholder="owner/dataset" '
                           'maxlength="193" required></label>')
            review = escape(contribution.payload.decode("utf-8"))
            skipped = f'<p>Pairs left out by the upload checks {contribution.skipped}</p>' if contribution.skipped else ""
            token_field = "" if get_token() and not notice else f"""
<p>Use your Hugging Face account. Run <code>hf auth login</code> or enter a token that can open pull requests.</p>
<label>Your Hugging Face token
<input name="hf_token" type="password" autocomplete="off" maxlength="512">
</label>
<p><a href="{HF_ENDPOINT}/settings/tokens" target="_blank" rel="noopener noreferrer">Hugging Face tokens ↗</a></p>"""
            return f"""<article class="message hub-card">{header}
<p>{contribution.count} reviewed pairs will be public as soon as the pull request opens.</p>{skipped}
<details><summary>Review the exact upload</summary><pre class="message-text" tabindex="0">{review}</pre></details>
<form class="hub-form" method="post" action="{escape(url('/huggingface/community', chat=chat_id))}">
  <input type="hidden" name="snapshot" value="{contribution.review_token(configured)}">
  {destination}{token_field}
  <p>By sending, you confirm you have permission to share this content publicly under the dataset's terms.</p>
  <button class="primary">Send to community</button>
</form></article>"""

        def _submit_community(self, form):
            chat_id = self.query.get("chat")
            if not chat_id:
                return self._send("Choose a specific chat to contribute.", 400)
            current_config = Config(path=config.path)
            configured = str(current_config.get("publish", "community_repo") or "").strip()
            with lock:
                records = build_records(vault, current_config, conversation_id=chat_id)
                contribution = prepare_contribution(records, current_config)
                snapshot = form.get("snapshot", [""])[0]
                if not secrets.compare_digest(snapshot.encode(), contribution.review_token(configured).encode()):
                    return self._hub(community_notice="The chat or dataset changed. Review the updated upload.")
            try:
                repo = community_repo(current_config, configured or form.get("repo", [""])[0])
                result = submit_contribution(contribution, repo, token=form.get("hf_token", [""])[0])
            except (PublishError, OSError) as exc:
                with lock:
                    return self._hub(community_notice=str(exc))
            with lock:
                vault.add_publish(result["repo"], result["commit"], result["manifest_sha256"], result["count"])
            self._render(f"""
<header><div class="intro"><h1>Sent to the community</h1>
<p>{result['count']} chat pairs submitted for review on Hugging Face.</p></div></header>
<a class="button primary" href="{escape(result['destination'])}" target="_blank" rel="noopener noreferrer">
Open pull request ↗</a>""", "hub")

        def _upload_dataset(self, form):
            nonlocal session_hub_token
            repo = form.get("repo", [""])[0].strip()
            write_token = form.get("hf_token", [""])[0].strip()
            snapshot = form.get("snapshot", [""])[0]
            current_config = Config(path=config.path)
            with lock:
                records = build_records(vault, current_config, conversation_id=self.query.get("chat"))
                if not records or not secrets.compare_digest(snapshot.encode(), self._upload_snapshot(records).encode()):
                    return self._hub("Your approved chats changed. Review the updated selection and upload again.", repo)
                current_token = session_hub_token
            try:
                with tempfile.TemporaryDirectory(prefix="dem-huggingface-") as temp:
                    folder = Path(temp)
                    bundle = prepare_private_chats(records, folder, current_config)
                    result = upload_hub_bundle(folder, bundle, current_config, config.data_dir / "keys",
                                               repo=repo or f"dem-chats-{snapshot[:8]}",
                                               token=write_token or current_token)
            except PublishError as exc:
                with lock:
                    return self._hub(str(exc), repo, show_token=True)
            except (OSError, ValueError, KeyError, TypeError):
                with lock:
                    return self._hub("Could not complete the upload. Please try again.", repo)
            with lock:
                if write_token:
                    session_hub_token = write_token
                vault.add_publish(result["repo"], result["commit"], result["manifest_sha256"], result["count"])
            count = result["count"]
            pairs = "pair" if count == 1 else "pairs"
            self._render(f"""
<header><div class="intro"><h1>Dataset uploaded</h1>
<p>{count} {pairs} uploaded to your private dataset.</p></div></header>
<a class="button primary" href="{escape(result['destination'])}" target="_blank" rel="noopener noreferrer">
Open on Hugging Face ↗</a>""", "hub")

        def _selected_source(self):
            source = self.query.get("source", "")
            return source if source in SOURCES else ""

        def _import(self, path=""):
            source = self._selected_source()
            if path and (not source or len(path) > 4096):
                return self._index("Choose a source and enter a valid local transcript path.")
            current_config = Config(path=config.path)
            try:
                check_history_consent(current_config, vault)
            except ValueError as exc:
                if current_config.get("privacy", "consent_mode") != "default-deny":
                    return self._index(str(exc))
                return self._consent(path=path)
            try:
                sources = (source,) if source else HISTORY_SOURCES
                results = [import_history(current_config, vault, key, path=path or None) for key in sources]
            except (OSError, ValueError) as exc:
                return self._index(str(exc))
            return self._redirect(url(source=source, files=sum(r["files"] for r in results),
                                      imported=sum(r["imported"] for r in results),
                                      activity=sum(r["activity_updated"] for r in results),
                                      unreadable=sum(r["unreadable_files"] for r in results)))

        def _refresh_activity(self):
            chat_id = self.query.get("chat", "")
            chat = next((chat for chat in vault.history_chats() if chat["conversation_id"] == chat_id), None)
            if chat is None:
                return self._send("Chat not found", 404)
            if not chat["enabled"]:
                return self._chat("Turn this chat on to look for saved activity.")
            try:
                result = import_history(Config(path=config.path), vault, chat["source"], activity_for={chat_id})
            except ValueError as exc:
                return self._chat(str(exc))
            if result["activity_updated"]:
                return self._chat("Added saved activity. Review it before uploading.")
            if result["unreadable_files"]:
                return self._chat("DEM could not read some saved transcripts.")
            return self._chat("The saved transcript has no additional activity for this chat.")

        def _consent(self, notice="", path=""):
            status = f'<p class="notice" role="status">{escape(notice)}</p>' if notice else ""
            self._render(f"""
<a class="back" href="{escape(url(source=self._selected_source()))}">← Chat history</a>
<header><div class="intro"><h1>Allow local chat imports</h1>
<p>DEM will copy saved prompts, answers, and available activity into its encrypted local review queue.</p></div></header>
<article class="message">{status}{self._consent_form(path)}</article>""")

        def _consent_form(self, path=""):
            original_path = f'<input type="hidden" name="path" value="{escape(path)}">' if path else ""
            return f"""
<details><summary>Read the DEM consent policy</summary><pre>{escape(CONSENT_TEXT)}</pre></details>
<form method="post" action="{escape(url('/import/consent', source=self._selected_source()))}">
{original_path}
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

        def _source_switcher(self, chats, source):
            links = []
            for key in ("", "codex", "claude-code", "opencode"):
                name = SOURCE_NAMES[key] if key else "All chats"
                count = sum(not key or chat["source"] == key for chat in chats)
                current = ' aria-current="page"' if key == source else ""
                links.append(f'<a href="{escape(url(source=key))}"{current}>{name} <span>{count}</span></a>')
            return f'<nav class="source-switcher" aria-label="Chat sources">{"".join(links)}</nav>'

        def _import_from_path(self, source, files):
            action = escape(url("/import", source=source))
            return f"""
<details class="message source-import"{'' if files else ' open'}>
  <summary>Import from a file or folder</summary>
  <form class="filter" method="post" action="{action}">
    <label for="source-path">Local transcript path</label>
    <input id="source-path" name="path" maxlength="4096" placeholder="/path/to/transcript.jsonl" required>
    <button>Import</button>
  </form>
</details>"""

        def _index(self, notice=""):
            chats = vault.history_chats()
            source = self._selected_source()
            state = self.query.get("state", "")
            selected = [c for c in chats if not source or c["source"] == source]
            visible = [c for c in selected if not state or bool(c["enabled"]) == (state == "on")]
            enabled = sum(c["enabled"] for c in selected)
            pending = sum(c["pending"] for c in selected if c["enabled"])
            switcher = self._source_switcher(chats, source)
            if "imported" in self.query:
                pairs = self.query["imported"]
                notice = f"Imported {pairs} new {'chat pair' if pairs == '1' else 'chat pairs'}."
                if self.query.get("files") == "0":
                    notice = "No local transcripts found."
                elif pairs == "0" and not selected and self.query.get("unreadable", "0") == "0":
                    notice += " The files had no completed prompt and response pairs."
                if self.query.get("unreadable", "0") != "0":
                    notice += f" Could not read {self.query['unreadable']} transcript files."
                if self.query.get("activity", "0") != "0":
                    notice += " Added activity to previously imported chats. Review those chats before uploading."
            status = f'<p class="notice" role="status">{escape(notice)}</p>' if notice else ""
            if not chats and not source:
                return self._render(f"""
<header><div class="intro"><h1>Chat history</h1>
<p>Review saved prompts, answers, and chat activity before uploading.</p></div></header>
{switcher}{status}{self._empty_history()}""")
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
            import_options = ""
            if source:
                source_name = SOURCE_NAMES[source]
                files = history_files(source, export_root=config.data_dir / "history")
                import_options = self._import_from_path(source, files)
            if cards:
                listing = f'<ul class="chat-list">{"".join(cards)}</ul>'
            elif source and not selected:
                explanation = ("Local transcripts are ready to import." if files
                               else "DEM found no saved transcripts on this device. You can import one from a local file or folder.")
                listing = f'<div class="empty"><h2>No {source_name} chats yet</h2><p>{explanation}</p></div>'
            else:
                listing = ('<div class="empty"><h2>No chats match these filters</h2>'
                           f'<a href="{escape(url(source=source))}">Clear filters</a></div>')
            import_label = f"Import {SOURCE_NAMES[source]} chats" if source else "Import chats"
            self._render(f"""
<header><div class="intro"><h1>Chat history</h1>
<p>Review saved prompts, answers, and chat activity before uploading.</p></div>
<form method="post" action="{escape(url('/import', source=source))}"><button class="primary">{import_label}</button></form></header>
{switcher}{status}
<div class="stats" aria-label="Chat summary">
  <div><span>Total chats</span><strong>{len(selected)}</strong></div>
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
</section>{import_options}""")

        def _chat(self, notice=""):
            chat_id = self.query.get("chat", "")
            chat = next((chat for chat in vault.history_chats() if chat["conversation_id"] == chat_id), None)
            if chat is None:
                return self._send("Chat not found", 404)
            events = vault.history_messages(chat_id)
            first_activity = next((event.id for event in events if _visible_activity(event)), None)
            expand_tools = self.query.get("tools") == "open"
            messages = []
            model_history = []
            previous_model = ""
            tools = {}
            have_details = False
            for event in events:
                switches = {}
                for index, step in enumerate(_visible_activity(event)):
                    model = step.get("model")
                    if model and model != previous_model:
                        model_history.append((model, _model_anchor(event.id, index)))
                        switches[index] = "Switched to" if previous_model or step["type"] == "model_change" else "Using"
                        previous_model = model
                    if step["type"] == "tool_call":
                        name = step["text"]
                        have_details |= any(field in step for field in ACTIVITY_PAYLOAD_FIELDS)
                        anchor, count = tools.get(name, (_tool_anchor(event.id, index), 0))
                        tools[name] = anchor, count + 1
                model = event.model_hint if event.role == "response" else ""
                response_switch = False
                if model and model != previous_model:
                    model_history.append((model, _model_anchor(event.id)))
                    response_switch = bool(previous_model)
                    previous_model = model
                messages.append(self._message(event, activity_anchor=event.id == first_activity,
                                              expand_tools=expand_tools, switches=switches,
                                              response_switch=response_switch))
            snapshot, _ = vault.history_review_snapshot(events)
            outcome_target = vault.history_outcome_target(events)
            outcome = outcome_target.task_outcome if outcome_target else None
            evidence = collect_task_evidence(step for event in events
                                             if not event.quarantine_reason and event.consent != "rejected"
                                             for step in event.activity)
            if not messages:
                messages.append('<div class="empty"><h2>No saved messages</h2><p>Turn on this chat and import again.</p></div>')
            model_status = ""
            if model_history:
                names = "".join(f'<li><a href="#{escape(anchor)}"><code>{escape(model)}</code></a></li>'
                                for model, anchor in model_history)
                model_status = f'<section class="message chat-activity-status"><h2>Models used</h2><ol class="model-list">{names}</ol></section>'
            status = f'<p class="notice" role="status">{escape(notice)}</p>' if notice else ""
            activity_link = '<p><a href="#chat-activity">View chat activity</a></p>' if first_activity else ""
            activity_status = ""
            if tools:
                names = "".join(f'<li><a href="#{escape(anchor)}"><code>{escape(name)}</code>'
                                f'<span>{count}</span></a></li>' for name, (anchor, count) in tools.items())
                toggle = ""
                if have_details:
                    href = url("/chat", chat=chat_id, tools="closed" if expand_tools else "open") + "#tool-calls"
                    label = "Collapse all tool calls" if expand_tools else "Expand all tool calls"
                    toggle = f'<a href="{escape(href)}">{label}</a>'
                activity_status = f"""
<section class="message chat-activity-status" id="tool-calls">
<div class="tool-overview-head"><h2>Tool calls</h2>{toggle}</div>
<ul class="tool-list">{names}</ul></section>"""
            elif not first_activity and events and chat["enabled"]:
                activity_status = f"""
<section class="message chat-activity-status"><h2>Chat activity</h2>
<p>No tool calls or rationale summaries saved in DEM for this chat.</p>
<form method="post" action="{escape(url('/chat/activity', chat=chat_id))}"><button>Check saved chat for activity</button></form>
</section>"""
            guidance = ("Review the prompts, answers, and activity below." if snapshot
                        else "Review or edit the text below.")
            action = ""
            if chat["enabled"] and (snapshot or outcome_target):
                outcome_field = ""
                if outcome_target:
                    selected = outcome or "unspecified"
                    options = "".join(f'<option value="{key}"{" selected" if key == selected else ""}>{name}</option>'
                                      for key, name in (("unspecified", "Leave unspecified"), *TASK_OUTCOMES.items()))
                    outcome_field = f"""
  <input type="hidden" name="outcome_target" value="{escape(outcome_target.id)}">
  <div class="outcome-field"><label for="task-outcome">Did this solve the task? <span>Optional</span></label>
  <select id="task-outcome" name="task_outcome">{options}</select></div>"""
                label = "Approve chat and continue" if snapshot else "Continue to upload"
                action = f"""
<form class="review-actions" method="post" action="{escape(url('/chat/approve', chat=chat_id))}">
  <input type="hidden" name="snapshot" value="{escape(snapshot)}">
  {outcome_field}
  <button class="primary">{label}</button>
</form>"""
            evidence_panel = _evidence_panel(evidence, outcome, can_review=bool(outcome_target and chat["enabled"]))
            self._render(f"""
<a class="back" href="{escape(url())}">← All chats</a>
<header><div class="intro">{_source(chat["source"])}<h1>{escape(self._title(chat))}</h1>
<p>{guidance}</p>{activity_link}</div>
{self._switch(chat, view="chat")}</header>
{status}{model_status}{evidence_panel}{activity_status}<div class="messages">{"".join(messages)}</div>
<div class="chat-approve">{action}</div>""")

        def _message(self, event, *, activity_anchor=False, expand_tools=False, switches=(), response_switch=False):
            role = "Prompt" if event.role == "prompt" else "Response"
            turn = event.metadata.get("history_turn", "")
            pair = f" · Pair {escape(turn[:8])}" if turn else ""
            label = f"{role}{pair}" if turn else f"{escape(event.source)} · {role}"
            reject = escape(url("/reject", id=event.id))
            model_badge = ""
            if event.role == "response" and event.model_hint:
                prefix = "Switched to " if response_switch else ""
                model_badge = (f'<span class="model-badge" id="{escape(_model_anchor(event.id))}">'
                               f'{prefix}<code>{escape(event.model_hint)}</code></span>')
            activity = ""
            if _visible_activity(event):
                steps = []
                for index, step in enumerate(_visible_activity(event)):
                    if index in switches:
                        anchor = escape(_model_anchor(event.id, index))
                        model = f'<strong>{switches[index]} <code>{escape(step["model"])}</code></strong>'
                        steps.append(f'<li class="model-switch" id="{anchor}">{model}</li>')
                    if step["type"] == "model_change":
                        continue
                    if step["type"] == "tool_call":
                        steps.append(_tool_step(step, event.id, index, expand_tools))
                    else:
                        title = "Rationale summary" if step["type"] == "rationale" else "Progress note"
                        steps.append(f'<li><strong>{title}</strong>{_message_text(step["text"])}</li>')
                anchor = ' id="chat-activity"' if activity_anchor else ""
                activity = (f'<details class="activity"{anchor} open><summary>Chat activity</summary>'
                            f'<ol>{"".join(steps)}</ol></details>')
            if event.quarantine_reason:
                return (f'<article class="message"><div class="message-head"><h2>{label}</h2>{model_badge}</div>'
                        f'<p class="notice">{role} held locally ({escape(event.quarantine_reason)}).</p>{activity}'
                        f'<form method="post" action="{reject}"><button>Discard</button></form></article>')
            redacted = event.redacted_text or ""
            final = escape(redacted)
            rendered_text = _message_text(redacted)
            counts = summarize_redactions(redacted)
            redaction_summary = " · ".join(f"{escape(category.replace('_', ' ').lower())} ({count})"
                                          for category, count in counts.items())
            redactions = f'<p class="review-redactions">Redacted {redaction_summary}</p>' if counts else ""
            diff = escape("".join(difflib.ndiff(event.text.splitlines(keepends=True),
                                               redacted.splitlines(keepends=True))))
            approve = escape(url("/approve", id=event.id))
            edit = escape(url("/edit", id=event.id))
            return f"""<article class="message"><div class="message-head"><h2>{label}</h2>
{model_badge}<span class="badge">{escape(event.consent.capitalize())}</span></div>
{activity}
{rendered_text}
{redactions}
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
