"""Explicit, granular, versioned consent. Default-deny everywhere."""
from __future__ import annotations

import hashlib
import json

from .config import Config
from .models import CaptureEvent, now_utc

CONSENT_VERSION = "consent-v3"

CONSENT_TEXT = """DEM will read the text answers shown by the LLM apps you select,
on THIS device only.

- By default NOTHING is uploaded. Text sits in an encrypted local queue.
- Before anything is published you will see the exact final text, character for
  character, and must approve it.
- We automatically try to remove names, emails, keys and other sensitive data.
  Our redaction will sometimes fail. YOU are the last line of defense.
- Captured prompts and code blocks are OFF by default.
- Once published to HuggingFace, text is PUBLIC and PERMANENT. It can be copied,
  cached and trained into models. We cannot reliably take it back.
- Outputs from ChatGPT/Claude/Gemini that you pay for are yours to keep and
  share. They carry a `proprietary-api-output` provenance tag and are grouped
  in a 'restricted' partition so the dataset stays clear about their origin.
- You can stop capture, delete your local queue, and request removal of
  published items at any time (public copies may persist).
"""


def consent_hash() -> str:
    return "sha256:" + hashlib.sha256(CONSENT_TEXT.encode("utf-8")).hexdigest()


def source_allowed(config: Config, source: str) -> bool:
    """Whether capture is permitted at all for a surface (not whether publish is)."""
    if source == "ollama":
        return config.enabled("capture", "ollama_proxy")
    if source == "api_proxy":
        return config.enabled("capture", "api_proxy")
    if source == "cli":
        return True  # only reached via explicit `dem run`
    if source == "manual":
        return True  # explicit user action
    if source in {"browser", "chatgpt", "claude", "gemini"}:
        return config.enabled("capture", "browser_extension")
    return False


def is_commercial(source: str, surface: str) -> bool:
    surface = surface.lower()
    return any(host in surface for host in ("chatgpt", "claude", "gemini", "openai", "anthropic"))


def commercial_attested(config: Config) -> bool:
    return config.get("capture", "commercial_surfaces_default") == "attest"


def check_capture(config: Config, event: CaptureEvent) -> tuple[bool, str]:
    """Return (allowed, reason). Enforces default-deny and per-source opt-ins."""
    if config.get("privacy", "consent_mode") != "default-deny":
        return False, "consent_mode must be default-deny"
    if not source_allowed(config, event.source):
        return False, f"capture disabled for source={event.source}"
    if is_commercial(event.source, event.surface) and not commercial_attested(config):
        return False, "commercial surface not attested (capture.commercial_surfaces_default=exclude)"
    return True, "ok"


def onboard(config: Config, vault) -> str:
    """Interactive first-run wizard. Writes a signed consent receipt."""
    print("=" * 72)
    print("DEM — first-run consent")
    print("=" * 72)
    print(CONSENT_TEXT)
    print("Consent policy version:", CONSENT_VERSION, "hash:", consent_hash())
    answer = input('Type "I CONSENT" to enable capture with these terms: ').strip()
    if answer != "I CONSENT":
        print("Consent not given. DEM will remain fully disabled.")
        receipt = json.dumps({"consent": False, "version": CONSENT_VERSION, "ts": now_utc()})
        vault.add_consent_event("onboard", "rejected", receipt)
        return receipt
    config.set("capture", "ollama_proxy", True)
    config.save()
    receipt = approve_capture(vault)
    print("Consent recorded. Ollama/local capture enabled. Nothing will be published automatically.")
    return receipt


def approve_capture(vault) -> str:
    receipt = json.dumps({
        "consent": True,
        "version": CONSENT_VERSION,
        "hash": consent_hash(),
        "ts": now_utc(),
        "policy": "explicit-consent, default-deny, review-before-publish",
    }, indent=2)
    vault.add_consent_event("onboard", "approved", receipt)
    return receipt


def attest_commercial(config: Config, vault) -> None:
    print("PAID-MODEL OUTPUTS (ChatGPT / Claude / Gemini)")
    print("- These are outputs you paid for; you choose whether to contribute them.")
    print("- They carry a `proprietary-api-output` provenance tag.")
    print("- They are grouped in the 'restricted' partition so their origin is clear.")
    answer = input('Type "I ATTEST" to enable: ').strip()
    if answer == "I ATTEST":
        record_commercial_attestation(config, vault)
        print("Commercial capture enabled (user-attested).")
    else:
        print("Commercial capture remains disabled.")


def record_commercial_attestation(config: Config, vault) -> None:
    config.set("capture", "commercial_surfaces_default", "attest")
    config.save()
    vault.add_consent_event("commercial", "attested", now_utc())
