"""Quality scoring + hard gates before publication."""
from __future__ import annotations

import math
import re

from ..config import Config
from ..privacy import redact

TOXIC = re.compile(r"(?i)\b(?:fuck|shit|idiot|stupid|kill yourself|hate (?:you|them)|racist|slur)\b")
INJECTION = re.compile(r"(?i)(ignore (?:all |the )?(?:previous|prior|above) instructions|"
                      r"reveal (?:your|the) system prompt|print your instructions|"
                      r"\<\|system\|>|assistant: system)")
URL = re.compile(r"https?://\S+|\[\d+\]\(http")
HALLU_PHRASES = re.compile(r"(?i)(as of my (?:training|knowledge)|i (?:cannot|can't) browse|"
                           r"according to my training data)")
GATE = 0.60
PIPELINE_VERSION = "dem-quality-0.3.0"


def _length_fit(tokens: int) -> float:
    if 64 <= tokens <= 1024:
        return 1.0
    if tokens < 16 or tokens > 2048:
        return 0.0
    if tokens < 64:
        return 0.4 + 0.6 * (tokens - 16) / 48
    return 0.4 + 0.6 * (2048 - tokens) / 1024


def _repetition_ratio(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    if len(words) < 20:
        return 0.0
    quads = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return 1.0 - (len(set(quads)) / len(quads))


def score(record, config: Config) -> tuple[float, dict]:
    prompt = record.prompt or ""
    response = record.response or ""
    response_tokens = len(response.split())

    length_fit = _length_fit(response_tokens)
    prompt_is_task = bool(re.search(r"[?！？!]|\b(?:explain|write|list|calculate|compare|summarize|code|why|how)\b",
                                    prompt.lower()))
    instruction = 0.5 + 0.3 * prompt_is_task + 0.2 * (1 if response_tokens > 16 else 0)
    coherence = 1.0 - min(1.0, _repetition_ratio(response) * 2)
    not_toxic = 0.0 if TOXIC.search(response) else 1.0
    hallu_flags = len(URL.findall(response)) + len(HALLU_PHRASES.findall(response))
    not_hallu = max(0.0, 1.0 - 0.15 * hallu_flags)
    format_clean = 0.0 if INJECTION.search(prompt + "\n" + response) else 1.0

    score_value = (
        0.30 * length_fit + 0.20 * instruction + 0.20 * coherence +
        0.10 * not_toxic + 0.10 * not_hallu + 0.10 * format_clean
    )
    # Degenerate/repetitive output can never clear the gate, however long it is.
    repetition = _repetition_ratio(response)
    if repetition > 0.5:
        score_value *= max(0.2, 1.0 - repetition)
    signals = {
        "length_fit": round(length_fit, 3),
        "instruction_following": round(instruction, 3),
        "coherence": round(coherence, 3),
        "not_toxic": not_toxic,
        "not_hallucination_flag": round(not_hallu, 3),
        "format_clean": format_clean,
    }
    return round(score_value, 3), signals


def passes(record, config: Config) -> tuple[bool, str]:
    """Hard gates. Returns (ok, reason)."""
    if not record.prompt.strip() or not record.response.strip():
        return False, "empty prompt/response"
    if record.source_tier == "clean" and record.derived_from == "proprietary-api-output":
        return False, "proprietary output in clean tier"
    if not record.pii_scrubbed:
        return False, "not scrubbed"
    if _repetition_ratio(record.response) > 0.4:
        return False, "repetitive/degenerate output"
    # Defense in depth: re-scan the finished record.
    for text in (record.prompt, record.response):
        result = redact(text, config)
        if result.quarantine_reason:
            return False, f"pII rescan: {result.quarantine_reason}"
    if INJECTION.search(record.prompt + "\n" + record.response):
        return False, "prompt-injection junk"
    return True, "ok"


def score_all(records: list, config: Config):
    """Annotate records in place; return (accepted, rejected_with_reasons)."""
    accepted, rejected = [], []
    for record in records:
        value, signals = score(record, config)
        record.quality = {"score": value, "signals": signals, "pipeline": PIPELINE_VERSION}
        ok, reason = passes(record, config)
        if ok and value >= GATE:
            accepted.append(record)
        else:
            rejected.append((record.id, reason if not ok else f"score {value} < {GATE}"))
    return accepted, rejected
