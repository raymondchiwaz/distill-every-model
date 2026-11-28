"""Turn approved, redacted capture pairs into canonical DatasetRecords."""
from __future__ import annotations

from ..consent import CONSENT_VERSION, is_commercial
from ..config import Config
from ..models import ACTIVITY_PAYLOAD_FIELDS, ACTIVITY_RESULT_FIELDS, CaptureEvent, DatasetRecord, sha256_text, ulid
from ..privacy import redact, summarize_redactions
from ..vault import Vault
from .evidence import collect_task_evidence


def build_records(vault: Vault, config: Config, *, conversation_id: str | None = None,
                  conversation_ids: tuple[str, ...] | None = None) -> list[DatasetRecord]:
    captures = vault.approved_captures(conversation_id, conversation_ids=conversation_ids)
    return records_from_captures(captures, vault, config)


def records_from_captures(captures: list[CaptureEvent], vault: Vault, config: Config) -> list[DatasetRecord]:
    prompts = {c.conversation_id: c for c in captures if c.role == "prompt"}

    history_prompts = {c.metadata["history_turn"]: c for c in captures
                      if c.role == "prompt" and "history_turn" in c.metadata}

    records: list[DatasetRecord] = []
    for event in captures:
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
        model = redact(event.model_hint, config)
        activity = []
        for step in event.activity:
            reviewed = {"type": step["type"], "text": step["text"]}
            for field in ACTIVITY_PAYLOAD_FIELDS:
                if field in step and f"{field}_hold" not in step:
                    reviewed[field] = step[field]
            reviewed.update((field, step[field]) for field in (*ACTIVITY_RESULT_FIELDS, "model") if field in step)
            activity.append(reviewed)
        redaction_summary = [
            {"field": field, "category": category, "count": count}
            for field, text in (("prompt", prompt_text), ("response", response_text),
                                ("activity", "\n".join(value for step in activity for key, value in step.items()
                                                      if key != "type")))
            for category, count in summarize_redactions(text).items()
        ]
        records.append(DatasetRecord(
            id=ulid(),
            prompt=prompt_text,
            response=response_text,
            messages=messages,
            metadata={
                "model": model.text if model.scrubbed else "",
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
            activity=activity,
            redaction_summary=redaction_summary,
            task_outcome={"status": event.task_outcome, "source": "contributor"} if event.task_outcome else None,
            evidence=collect_task_evidence(activity),
        ))
    return records
