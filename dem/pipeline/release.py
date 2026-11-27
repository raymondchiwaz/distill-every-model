"""Build immutable, versioned dataset releases with manifest + QC report."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .. import __version__
from ..config import Config
from .dedup import Deduplicator
from .quality import GATE, PIPELINE_VERSION, score_all


def _bump(version: str | None, releases_dir: Path) -> str:
    if version:
        return version
    version_file = releases_dir / "VERSION"
    if version_file.exists():
        major, minor, patch = version_file.read_text().strip().split(".")
        return f"{major}.{minor}.{int(patch) + 1}"
    return "0.3.0"


def _split_bucket(record) -> str:
    conv = (record.metadata or {}).get("conversation_id", record.id)
    n = int(hashlib.sha256(conv.encode()).hexdigest()[:8], 16) % 100
    if n < 1:
        return "test"
    if n < 2:
        return "validation"
    return "train"


def _dataset_card(version: str, config: Config, counts: dict) -> str:
    license_name = config.get("publish", "license", "Apache-2.0")
    return f"""---
license: {license_name}
language: en
task_categories:
- question-answering
- text-generation
size_categories: n<1K
tags:
- dem
- distill-every-model
- consent-based
---

# DEM — Distill Every Model (v{version})

> Local-first, explicit-consent capture of LLM outputs into an open training-data commons.

## Dataset summary
Consented, locally redacted prompt/response pairs captured from LLM tools. Records
are deduplicated, quality-gated (score >= {GATE}), and tagged by origin:
`clean` (open-model / user-written) and `restricted` (outputs from paid models
you subscribed to or paid for per call). Both come from your own sessions.

## Consent & collection
Every record carries a versioned consent token, was reviewed by the contributor,
and was scrubbed locally before upload. No silent capture, no auto-upload.
Consent policy: `consent-v3` (default-deny, review-before-publish).
Consent may be withdrawn; removals propagate to the next versioned release, with
the honest caveat that already-published copies cannot be reliably erased.

## Privacy
Local multi-pass secret/PII redaction with typed placeholders; fail-closed
quarantine of credentials, financial identifiers, third-party conversations and
special-category data. Timestamps coarsened to day; node pseudonyms rotated.
Publish a `qc_report.json` with every release.

## Licensing
**{license_name} is a software license applied here to data.** It is permissive
and includes an express patent grant. Third-party content is excluded by the
redaction/review pipeline. Contributors may tag records `ODC-By-1.0` or
`CC-BY-4.0` instead. Downstream users should do their own review of both configs.

## Limitations & known risks
- Automated redaction is imperfect; human review is mandatory, not decorative.
- Sybil resistance is imperfect without identity.
- Dataset may reflect model and user biases.

## Curation
Deduplicator + quality pipeline version: `{PIPELINE_VERSION}` (DEM {__version__}).
Counts: {json.dumps(counts)}.

## Citation
```bibtex
@misc{{dem{version.replace('.', '')},
  title = {{DEM — Distill Every Model, v{version}}},
  year = {{2026}},
  note = {{Consent-based open training-data commons}}
}}
```
"""


def build_release(records: list, releases_dir: Path, config: Config,
                  version: str | None = None, vault=None) -> dict:
    releases_dir.mkdir(parents=True, exist_ok=True)
    version = _bump(version, releases_dir)
    release_dir = releases_dir / f"v{version}"
    release_dir.mkdir(parents=True, exist_ok=True)

    dedup = Deduplicator()
    dedup_result = dedup.run(records)
    accepted, rejected = score_all(dedup_result.unique, config)
    if vault is not None:
        for record in accepted:
            vault.add_record(record)

    counts = {
        "input": len(records),
        "after_dedup": len(dedup_result.unique),
        "duplicates_removed": len(dedup_result.duplicates),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "clean": sum(1 for r in accepted if r.source_tier == "clean"),
        "restricted": sum(1 for r in accepted if r.source_tier == "restricted"),
    }

    data_root = release_dir / "data"
    file_hashes: dict[str, str] = {}
    for tier in ("clean", "restricted"):
        tier_records = [r for r in accepted if r.source_tier == tier]
        for split in ("train", "validation", "test"):
            split_records = [r for r in tier_records if _split_bucket(r) == split]
            out_dir = data_root / tier / split
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "train-00000-of-00001.jsonl"
            with open(path, "w", encoding="utf-8") as fh:
                for record in split_records:
                    fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            file_hashes[str(path.relative_to(release_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
            counts[f"{tier}_{split}"] = len(split_records)

    manifest = {
        "version": version,
        "dem_version": __version__,
        "files": file_hashes,
        "quality_gate": GATE,
        "pipeline": PIPELINE_VERSION,
    }
    (release_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_sha = hashlib.sha256(
        json.dumps(file_hashes, sort_keys=True).encode()
    ).hexdigest()
    (release_dir / "manifest.sha256").write_text(manifest_sha + "\n", encoding="utf-8")
    (release_dir / "qc_report.json").write_text(json.dumps({
        "version": version, "counts": counts, "rejected": rejected,
        "duplicates": dedup_result.duplicates,
        "gates": {"quality_score_min": GATE, "pii_rescan": True, "injection_filter": True},
    }, indent=2), encoding="utf-8")
    (release_dir / "README.md").write_text(_dataset_card(version, config, counts), encoding="utf-8")
    (releases_dir / "VERSION").write_text(version + "\n", encoding="utf-8")
    return {
        "version": version,
        "release_dir": str(release_dir),
        "manifest_sha256": manifest_sha,
        "counts": counts,
    }
