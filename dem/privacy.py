"""Local redaction engine: secrets + PII + fail-closed quarantine.

Detectors are regex/heuristic (no heavy ML dependency in v0.3). They are
imperfect by design; human review is mandatory. Detection precision/recall
must be measured on a held-out set before 1.0.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .config import Config

# (category, compiled pattern) — category drives the typed placeholder.
DETECTORS: list[tuple[str, re.Pattern]] = [
    ("SECRET", re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,}|xox[baprs]-[A-Za-z0-9-]{10,})\b")),
    ("SECRET", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),  # JWT
    ("SECRET", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")),
    ("SECRET", re.compile(r"(?:postgres|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s:@]+:[^\s@]+@[^\s]+")),
    ("SECRET", re.compile(r"(?i)\b(?:api[_-]?key|token|password|passwd|secret)['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}['\"]?")),
    ("SECRET", re.compile(r"(?i)\b(?:proxy[-_])?authorization['\"]?\s*[:=]\s*['\"]?"
                          r"(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{3,}")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{3,4}[ .-]?\d{3,4}(?!\d)")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
]

HIGH_RISK_MARKERS = re.compile(
    r"(?i)\b(nDa|confidential|proprietary|attorney[- ]client|medical record|diagnos\w+|"
    r"patient|ssn|social security|passport|date of birth|minor|under 18|child)\b"
)

CODE_FENCE = re.compile(r"```[^\n]*\n[\s\S]*?```")
CODE_BLOCK_REMOVED = "[CODE_BLOCK_REMOVED]"
TOOL_CREDENTIAL_KEY = re.compile(
    r"(?:authorization|proxy[_-]?authorization|cookies?|set[_-]?cookie|(?:api|private)[_-]?key|"
    r"(?:(?:access|refresh|auth)[_-]?)?token|password|passwd|secret|credentials?)", re.IGNORECASE,
)
REDACTION_MARKERS = re.compile(r"\[(CODE_BLOCK_REMOVED|(?:" + "|".join(sorted({kind for kind, _ in DETECTORS}))
                               + r")_\d+)\]")


def summarize_redactions(text: str) -> dict[str, int]:
    return dict(Counter(marker if marker == "CODE_BLOCK_REMOVED" else marker.rsplit("_", 1)[0]
                        for marker in REDACTION_MARKERS.findall(text)))


@dataclass
class RedactionResult:
    text: str
    redactions: list[dict] = field(default_factory=list)
    quarantine_reason: str | None = None

    @property
    def scrubbed(self) -> bool:
        return self.quarantine_reason is None


def redact_tool_payload(value, config: Config, *, keep_code: bool | None = None) -> RedactionResult:
    masked = False

    def scrub_json(item):
        nonlocal masked
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                if TOOL_CREDENTIAL_KEY.fullmatch(key) and child:
                    result[key] = "[SECRET_1]"
                    masked = True
                else:
                    result[key] = scrub_json(child)
            return result
        if isinstance(item, list):
            return [scrub_json(child) for child in item]
        return item

    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            decoded = None
        if isinstance(decoded, (dict, list)):
            value = decoded
    if isinstance(value, list) and value and all(
        isinstance(item, dict) and item.keys() == {"type", "text"}
        and item["type"] in ("input_text", "output_text", "text") and isinstance(item["text"], str)
        for item in value
    ):
        value = "\n".join(item["text"] for item in value)
    if not isinstance(value, str):
        value = json.dumps(scrub_json(value), indent=2, ensure_ascii=False)
    if not value.strip():
        return RedactionResult("")
    result = redact(value, config, keep_code=keep_code)
    if masked and config.enabled("privacy", "fail_closed"):
        result.quarantine_reason = "contained credential/financial identifier"
    return result


def luhn(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    total, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {c: value.count(c) for c in set(value)}
    return -sum((n / len(value)) * math.log2(n / len(value)) for n in counts.values())


def _replace_spans(text: str, spans: list[tuple[int, int, str]], counters: dict[str, int]) -> tuple[str, list[dict]]:
    if not spans:
        return text, []
    # Longest-first at the same start, then left-to-right; no overlaps.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    applied: list[tuple[int, int, str]] = []
    end = -1
    for start, stop, category in spans:
        if start < end:
            continue
        applied.append((start, stop, category))
        end = stop
    out, last, log = [], 0, []
    for start, stop, category in applied:
        out.append(text[last:start])
        counters[category] = counters.get(category, 0) + 1
        placeholder = f"[{category}_{counters[category]}]"
        out.append(placeholder)
        log.append({"category": category, "span": [start, stop], "replacement": placeholder})
        last = stop
    out.append(text[last:])
    return "".join(out), log


def redact(text: str, config: Config, *, keep_code: bool | None = None) -> RedactionResult:
    """Scrub secrets/PII and quarantine high-risk content. Fail closed."""
    if not text or not text.strip():
        return RedactionResult(text="", quarantine_reason="empty")

    working = text
    redactions: list[dict] = []
    counters: dict[str, int] = {}

    if keep_code is None:
        keep_code = config.enabled("capture", "capture_code")
    if not keep_code:
        working = CODE_FENCE.sub(CODE_BLOCK_REMOVED, working)

    if config.enabled("privacy", "secret_scan"):
        spans: list[tuple[int, int, str]] = []
        for category, pattern in DETECTORS:
            for match in pattern.finditer(working):
                value = match.group(0)
                if category == "CREDIT_CARD" and not luhn(value):
                    continue
                if category == "PHONE" and len(re.sub(r"\D", "", value)) < 7:
                    continue
                if category == "SECRET" and _shannon_entropy(value) < 2.5 and len(value) < 20:
                    continue
                spans.append((match.start(), match.end(), category))
        # Generic high-entropy token detection (short strings skipped).
        for match in re.finditer(r"\b[A-Za-z0-9_+/=-]{28,}\b", working):
            if _shannon_entropy(match.group(0)) >= 4.0:
                spans.append((match.start(), match.end(), "SECRET"))
        working, found = _replace_spans(working, spans, counters)
        redactions.extend(found)
        if any(r["category"] in {"SECRET", "CREDIT_CARD", "SSN", "IBAN"} for r in found):
            if config.enabled("privacy", "fail_closed"):
                return RedactionResult(working, redactions, "contained credential/financial identifier")

    # Third-party / special-category data we refuse to publish at all.
    if HIGH_RISK_MARKERS.search(working):
        return RedactionResult(working, redactions, "possible third-party/special-category data")

    # Quarantine pasted conversations (other people's words).
    if working.count("> ") > 3 or len(re.findall(r"(?m)^\s*(?:From|To|Subject):", working)) > 2:
        return RedactionResult(working, redactions, "looks like pasted third-party conversation")

    # Final pass on non-secret PII if redaction enabled.
    if config.enabled("privacy", "pii_redaction"):
        spans = []
        for category, pattern in DETECTORS:
            if category == "SECRET":
                continue
            for match in pattern.finditer(working):
                value = match.group(0)
                if category == "CREDIT_CARD" and not luhn(value):
                    continue
                spans.append((match.start(), match.end(), category))
        working, found = _replace_spans(working, spans, counters)
        redactions.extend(found)

    return RedactionResult(text=working.strip(), redactions=redactions, quarantine_reason=None)
