"""dem — command line interface. All operations are local unless explicitly published."""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import __version__, HARD_NO
from .adapters.base import CaptureSink
from .adapters.cli_wrapper import run_wrapped
from .adapters.history import import_history
from .config import load as load_config
from .consent import CONSENT_TEXT, CONSENT_VERSION, attest_commercial, consent_hash, onboard
from .crypto import Keyring
from .history_sources import HISTORY_SOURCES
from .models import sha256_text, ulid
from .pipeline.formatter import build_records
from .pipeline.release import build_release
from .publisher import _latest_release, dry_run, publish
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
    config, *_ = _context()
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
        if not vault.get_capture(args.id):
            print("capture not found")
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
                if first_pass or stats["imported"] or stats["unreadable_files"]:
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
    serve_blocking(config, vault)
    return 0


def cmd_release(args) -> int:
    config, dirs, _, vault = _context()
    records = build_records(vault, config)
    print(f"Built {len(records)} candidate records from approved captures.")
    result = build_release(records, dirs["releases"], config, args.version, vault=vault)
    print(json.dumps(result, indent=2))
    return 0


def cmd_publish(args) -> int:
    config, dirs, keys, vault = _context()
    if args.dry_run:
        release_dir = _latest_release(dirs["releases"], args.version)
        print(json.dumps(dry_run(release_dir), indent=2, ensure_ascii=False))
        print("\nNo network calls were made.")
        return 0
    result = publish(_latest_release(dirs["releases"], args.version), config, vault,
                     dirs["keys"], version=args.version)
    print(json.dumps(result, indent=2))
    return 0


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

    sub.add_parser("review", help="start localhost review UI").set_defaults(func=cmd_review)
    sub.add_parser("daemon", help="start capture daemon (or: demd)")

    rel = sub.add_parser("release", help="build a versioned release from approved records")
    rel.add_argument("--version")
    rel.set_defaults(func=cmd_release)

    pub = sub.add_parser("publish", help="dry-run preview or publish the latest release")
    pub.add_argument("--dry-run", action="store_true")
    pub.add_argument("--version")
    pub.set_defaults(func=cmd_publish)

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
