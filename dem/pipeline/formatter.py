"""Turn approved, redacted capture pairs into canonical DatasetRecords."""
from __future__ import annotations

from ..consent import CONSENT_VERSION, is_commercial
from ..config import Config
from ..models import DatasetRecord, sha256_text, ulid
from ..vault import Vault


def build_records(vault: Vault, config: Config) -> list[DatasetRecord]:
    approved = vault.approved_captures()
    prompts = {c.conversation_id: c for c in approved if c.role == "prompt"}

    history_prompts = {c.metadata["history_turn"]: c for c in approved
                      if c.role == "prompt" and "history_turn" in c.metadata}

    records: list[DatasetRecord] = []
    for event in approved:
        if event.role != "response":
            continue
        prompt_text = (event.metadata or {}).get("prompt_text") or ""
        if "history_turn" in event.metadata:
            prompt = history_prompts.get(event.metadata["history_turn"])
            prompt_text = prompt.redacted_text if prompt else ""
        elif not prompt_text and event.conversation_id in prompts:
            prompt_text = prompts[event.conversation_id].redacted_text or ""
        response_text = event.redacted_text or ""
        if not prompt_text.strip() or not response_text.strip():
            continue  # never publish an unpaired / empty record

        derived = (event.metadata or {}).get("derived_from", "open-model-output")
        commercial = is_commercial(event.source, event.surface) or derived == "proprietary-api-output"
        tier = "restricted" if commercial else "clean"
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text},
        ]
        canonical = prompt_text + "\n" + response_text
        records.append(DatasetRecord(
            id=ulid(),
            prompt=prompt_text,
            response=response_text,
            messages=messages,
            metadata={
                "model": event.model_hint,
                "source": event.source,
                "surface": event.surface,
                "conversation_id": event.conversation_id,
                "timestamp_day": event.timestamp[:10],  # coarsened
                "capture_confidence": 1.0,
                "consent_version": CONSENT_VERSION,
            },
            quality={},
            redactions=event.redactions,
            license=config.get("publish", "license", "Apache-2.0"),
            rights_attestation=(
                "user_owned_paid_output" if commercial else "user_owned_output"
            ),
            source_tier=tier,
            derived_from=derived,
            consent_token=vault.keys.tag(event.id + CONSENT_VERSION),
            pii_scrubbed=True,
            raw_hash=sha256_text(canonical),
        ))
    return records
