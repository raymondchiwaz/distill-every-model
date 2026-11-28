"""dem — command line interface. All operations are local unless explicitly published."""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import time

from . import __version__, HARD_NO
from .adapters.base import CaptureSink
from .adapters.cli_wrapper import run_wrapped
from .adapters.history import import_history
from .community import community_repo, prepare_contribution, submit_contribution
from .config import load as load_config
from .consent import CONSENT_TEXT, CONSENT_VERSION, attest_commercial, consent_hash, onboard
from .crypto import Keyring
from .history_sources import HISTORY_SOURCES
from .models import sha256_text, ulid
from .pipeline.formatter import build_records, records_from_captures
from .pipeline.release import build_release
from .publisher import PublishError, _latest_release, default_repo, dry_run, export_hub_bundle, publish
from .review import serve_blocking
from .vault import Vault


def _context():
    config = load_config()
    dirs = config.ensure_layout()
    keys = Keyring(dirs["keys"])
    vault = Vault(dirs["vault"], keys)
    return config, dirs, keys, vault


def cmd_init(args) -> int:
    config, dirs, keys, vault = _context()
    config.save()
    print("DEM initialized.")
    print("  data dir:", config.data_dir)
    print("  key file:", dirs["keys"] / "master.key", "(0600, separate from vault)")
    print("  consent:", CONSENT_VERSION, consent_hash())
    print("  license default:", config.get("publish", "license"))
    print("\nHard no's:", HARD_NO)
    print("\nNext: `dem onboard`, then `dem daemon`.")
    return 0


def cmd_onboard(args) -> int:
    config, dirs, keys, vault = _context()
    receipt = onboard(config, vault)
    print("Receipt:", receipt[:200])
    return 0


def cmd_attest(args) -> int:
    config, dirs, keys, vault = _context()
    attest_commercial(config, vault)
    return 0


def cmd_config(args) -> int:
    config, _, _, vault = _context()
    if args.value is None:
        print(f"{args.section}.{args.key} = {config.get(args.section, args.key)}")
    else:
        value = args.value
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        elif value.isdigit():
            value = int(value)
        config.set(args.section, args.key, value)
        config.save()
        print(f"set {args.section}.{args.key} = {value}")
        if (args.section, args.key, value) == ("capture", "capture_code", True):
            result = vault.restore_code_blocks(config)
            if any(result.values()):
                print(json.dumps(result, sort_keys=True))
    return 0


def cmd_captures(args) -> int:
    _, _, _, vault = _context()
    events = vault.list_captures(args.consent, limit=args.limit)
    for event in events:
        flag = "QUARANTINED" if event.quarantine_reason else ""
        text = (event.redacted_text or event.text).replace("\n", " ")[:100]
        print(f"{event.id}  {event.consent:10s} {event.source:10s} {flag}  {text}")
    return 0


def cmd_decide(args) -> int:
    _, _, _, vault = _context()
    decision = "approved" if args.command == "approve" else "rejected"
    if args.all:
        events = vault.list_captures("pending", limit=100000)
        for event in events:
            if not event.quarantine_reason:
                vault.update_consent(event.id, decision)
        print(f"{decision}: {len(events)} pending events processed "
              "(quarantined items are never published).")
    else:
        event = vault.get_capture(args.id)
        if event is None:
            print("capture not found")
            return 1
        if event.role == "activity":
            print("This activity is stored locally until the turn has a final reply.")
            return 1
        vault.update_consent(args.id, decision)
        print(f"{decision}: {args.id}")
    return 0


def cmd_ingest(args) -> int:
    """Manual test capture — feeds the exact same sink used by the proxies."""
    config, _, _, vault = _context()
    sink = CaptureSink(config, vault)
    prompt_preview = sink.prepare(args.prompt)
    conv_id = "manual-" + sha256_text(args.prompt)[:16]
    event = sink.ingest(
        text=args.response, role="response", source="manual",
        surface=args.surface, conversation_id=conv_id, turn_index=1,
        model_hint=args.model,
        metadata={"prompt_text": prompt_preview.text, "derived_from": args.derived_from},
    )
    if not event:
        print("Capture blocked by consent policy.")
        return 1
    print("Captured:", event.id, "| quarantine:", event.quarantine_reason or "none")
    print("Redacted response:", event.redacted_text[:200])
    return 0


def cmd_import_history(args) -> int:
    config, _, _, vault = _context()
    try:
        sources = ("claude-code", "codex") if args.source == "both" else (args.source,)
        if args.source == "all":
            sources = HISTORY_SOURCES
        if args.path and len(sources) != 1:
            raise ValueError("Select exactly one history source when using --path.")
        seen = {} if args.watch else None
        first_pass = True
        while True:
            failed = False
            for source in sources:
                stats = import_history(config, vault, source, args.path, seen=seen)
                if first_pass or any(stats[key] for key in ("imported", "activity_updated", "activity_only", "unreadable_files")):
                    print(f"{source} {json.dumps(stats)}", flush=True)
                failed |= bool(stats["unreadable_files"])
            if not args.watch:
                return int(failed)
            if first_pass:
                print("Watching saved chats. Press Ctrl-C to stop.", flush=True)
            first_pass = False
            time.sleep(10)
            config = load_config()
            if config.data_dir != vault.path.parent.parent or (config.data_dir / "stop").exists():
                return 0
    except KeyboardInterrupt:
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        vault.close()


def cmd_review(args) -> int:
    config, _, _, vault = _context()
    serve_blocking(config, vault, open_browser=args.open_browser)
    return 0


def cmd_release(args) -> int:
    config, dirs, _, vault = _context()
    records = build_records(vault, config)
    print(f"Built {len(records)} candidate records from approved captures.")
    result = build_release(records, dirs["releases"], config, args.version, vault=vault)
    print(json.dumps(result, indent=2))
    return 0


def cmd_publish(args) -> int:
    config, dirs, _, vault = _context()
    try:
        release_dir = _latest_release(dirs["releases"], args.version)
        if args.dry_run:
            print(json.dumps(dry_run(release_dir), indent=2, ensure_ascii=False))
            print("\nNo network calls were made.")
            return 0
        if args.export:
            result = export_hub_bundle(release_dir, dirs["export"])
            print(f"Hugging Face upload folder: {result['destination']}")
            print("Create a dataset at https://huggingface.co/new-dataset and upload the files to its root.")
            return 0

        version = args.version
        checksum = None
        if version is None and sys.stdin.isatty():
            preview = dry_run(release_dir)
            rows = preview["counts"]["clean"]
            version = preview["version"]
            if not rows:
                raise PublishError("This release has no rows eligible for Hugging Face. Restricted records stay local.")
            visibility = "public" if args.public else "private"
            repo = args.repo or default_repo(config, version)
            print(f"Release v{version} has {rows} uploadable rows. Destination is the {visibility} dataset {repo}.")
            if input("Upload this release? [y/N] ").strip().lower() not in ("y", "yes"):
                return 0
            checksum = preview["manifest_sha256"]
        result = publish(release_dir, config, vault, dirs["keys"], version,
                         repo=args.repo, public=args.public, expected_manifest=checksum)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        vault.close()


def cmd_contribute(args) -> int:
    config, _, _, vault = _context()
    try:
        if args.list:
            if args.send:
                raise PublishError("Select chats with --chat to submit a reviewed contribution.")
            chats = [{"id": chat["conversation_id"], "source": chat["source"], "title": chat["title"]}
                     for chat in vault.history_chats() if chat["enabled"] and chat["messages"]]
            print(json.dumps(chats, ensure_ascii=False))
            return 0

        repo = community_repo(config, args.repo)
        chat_ids = list(dict.fromkeys(args.chat))
        records = []
        held = 0
        for chat_id in chat_ids:
            messages = vault.history_messages(chat_id, enabled_only=True)
            if not messages:
                raise PublishError(f"Chat {chat_id} is unavailable or disabled.")
            reviewable = [event for event in messages if event.consent in ("pending", "approved", "redacted")
                          and not event.quarantine_reason]
            pairs = records_from_captures(reviewable, vault, config)
            held += sum(event.role == "response" for event in messages) - len(pairs)
            records.extend(pairs)

        contribution = prepare_contribution(records, config)
        review_token = contribution.review_token(repo)
        if args.send is None:
            result = {"repo": repo, "chats": chat_ids, "review_token": review_token,
                      "count": contribution.count, "skipped": contribution.skipped + held,
                      "rows": [json.loads(line) for line in contribution.payload.splitlines()]}
        else:
            if not secrets.compare_digest(review_token.encode(), args.send.encode()):
                raise PublishError("The contribution or dataset changed. Review the new preview before sending.")
            result = submit_contribution(contribution, repo, via_hf_cli=args.via_hf_cli)
            result["skipped"] += held
            vault.add_publish(result["repo"], result["commit"], result["manifest_sha256"], result["count"])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, PublishError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        vault.close()


def cmd_status(args) -> int:
    config, dirs, _, vault = _context()
    print("DEM", __version__)
    print("data dir:", config.data_dir)
    print("consent policy:", CONSENT_VERSION, consent_hash())
    print("commercial sources:", config.get("capture", "commercial_surfaces_default"))
    print("default license:", config.get("publish", "license"))
    print(json.dumps(vault.stats(), indent=2))
    for release in sorted(dirs["releases"].glob("v*")):
        if release.is_dir():
            print("release:", release.name)
    return 0


def cmd_kill(args) -> int:
    config, *_ = _context()
    flag = config.data_dir / "stop"
    flag.write_text("stop\n")
    print("Kill switch set: daemon will stop capture on its next tick.")
    print("Note: already-published data may be cached/mirrored; retraction cannot guarantee erasure.")
    return 0


def cmd_purge(args) -> int:
    config, dirs, _, vault = _context()
    answer = input('Type "PURGE" to permanently delete all local captures, records and logs: ').strip()
    if answer != "PURGE":
        print("Cancelled.")
        return 1
    vault.purge()
    for name in ("master.key", "hmac.key"):
        path = dirs["keys"] / name
        if path.exists():
            path.unlink()
    print("Local vault wiped and keys destroyed. Published copies cannot be erased this way.")
    return 0


def cmd_run(args) -> int:
    config, *_ = _context()
    return run_wrapped(config, args.command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dem", description="Distill Every Model — local-first, consent-based LLM output commons")
    parser.add_argument("--version", action="version", version=f"dem {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="initialize data dir, keys, vault, config").set_defaults(func=cmd_init)
    sub.add_parser("onboard", help="first-run consent wizard").set_defaults(func=cmd_onboard)
    sub.add_parser("attest", help="opt in to ChatGPT/Claude/Gemini capture").set_defaults(func=cmd_attest)

    cfg = sub.add_parser("config", help="get/set config")
    cfg.add_argument("section")
    cfg.add_argument("key")
    cfg.add_argument("value", nargs="?")
    cfg.set_defaults(func=cmd_config)

    cap = sub.add_parser("captures", help="list captured events")
    cap.add_argument("--consent", choices=["pending", "approved", "rejected", "redacted"])
    cap.add_argument("--limit", type=int, default=100)
    cap.set_defaults(func=cmd_captures)

    ing = sub.add_parser("ingest", help="manual test capture")
    ing.add_argument("--prompt", required=True)
    ing.add_argument("--response", required=True)
    ing.add_argument("--surface", default="manual")
    ing.add_argument("--model", default="")
    ing.add_argument("--derived-from", default="open-model-output")
    ing.set_defaults(func=cmd_ingest)

    history = sub.add_parser("import-history", help="import saved coding chats for review")
    history.add_argument("source", choices=[*HISTORY_SOURCES, "all", "both"])
    history.add_argument("--path", help="a transcript file or directory")
    history.add_argument("--watch", action="store_true", help="also import new saved turns until Ctrl-C")
    history.set_defaults(func=cmd_import_history)

    for action in ("approve", "reject"):
        dec = sub.add_parser(action, help=f"{action} a pending capture")
        dec.add_argument("id", nargs="?")
        dec.add_argument("--all", action="store_true", help=f"{action} all pending non-quarantined captures")
        dec.set_defaults(func=cmd_decide)

    review = sub.add_parser("review", help="start localhost review UI")
    review.add_argument("--open", dest="open_browser", action="store_true", help="open the workspace in your browser")
    review.set_defaults(func=cmd_review)
    sub.add_parser("daemon", help="start capture daemon (or: demd)")

    rel = sub.add_parser("release", help="build a versioned release from approved records")
    rel.add_argument("--version")
    rel.set_defaults(func=cmd_release)

    pub = sub.add_parser("publish", help="preview, export, or upload a Hugging Face dataset")
    mode = pub.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="preview without network calls")
    mode.add_argument("--export", action="store_true", help="write files for manual upload")
    pub.add_argument("--version", help="select a specific release")
    pub.add_argument("--repo", help="Hugging Face dataset name or owner/dataset")
    pub.add_argument("--public", action="store_true", help="upload to a public dataset")
    pub.set_defaults(func=cmd_publish)

    contribute = sub.add_parser("contribute", help="review selected chats or send them as a public Hugging Face PR")
    selection = contribute.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list", action="store_true", help="list enabled imported chats")
    selection.add_argument("--chat", action="append", help="pick a chat ID from --list (repeat for multiple chats)")
    contribute.add_argument("--repo", default="", help="override the configured public Hugging Face owner/dataset")
    contribute.add_argument("--send", metavar="REVIEW_TOKEN", help="send the exact reviewed preview to Hugging Face")
    contribute.add_argument("--via-hf-cli", action="store_true", help="send using hf upload --create-pr")
    contribute.set_defaults(func=cmd_contribute)

    sub.add_parser("status", help="show status and vault stats").set_defaults(func=cmd_status)
    sub.add_parser("kill", help="trigger daemon kill switch").set_defaults(func=cmd_kill)
    sub.add_parser("purge", help="emergency wipe of local data").set_defaults(func=cmd_purge)

    run = sub.add_parser("run", help="wrap a command through capture proxies")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(func=cmd_run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "daemon":
        from .daemon import main as daemon_main
        return daemon_main()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
