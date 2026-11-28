"""Prepare reviewed releases for Hugging Face and upload them explicitly."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .config import Config
from .consent import is_commercial
from .models import DatasetRecord
from .pipeline.release import RELEASE_VERSION, SHARD_NAME, SPLITS, curate_records
from .vault import Vault

HF_ENDPOINT = "https://huggingface.co"


class PublishError(ValueError):
    """A release or destination needs attention before it can be uploaded."""


def _latest_release(releases_dir: Path, version: str | None = None) -> Path:
    if version:
        if not RELEASE_VERSION.fullmatch(version):
            raise PublishError("Use a release version like 0.3.0.")
        path = releases_dir / f"v{version}"
        if not path.is_dir():
            raise FileNotFoundError(f"Release {version} was not found.")
        return path
    versions = sorted(
        (p for p in releases_dir.glob("v*") if p.is_dir() and RELEASE_VERSION.fullmatch(p.name[1:])),
        key=lambda p: [int(x) for x in p.name[1:].split(".")],
    )
    if not versions:
        raise FileNotFoundError("Prepare a dataset first with `dem release` or in `dem review`.")
    return versions[-1]


def _read_manifest(release_dir: Path) -> tuple[dict, str]:
    manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = (release_dir / "manifest.sha256").read_text(encoding="utf-8").strip()
    actual = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
    if expected != actual or release_dir.name != f"v{manifest['version']}":
        raise PublishError("The release manifest failed verification. Prepare a new dataset.")
    return manifest, actual


def dry_run(release_dir: Path) -> dict:
    """Preview a release without accessing the network."""
    manifest, checksum = _read_manifest(release_dir)
    qc = json.loads((release_dir / "qc_report.json").read_text(encoding="utf-8"))
    sample = []
    for split in SPLITS:
        sample_path = release_dir / "data" / "clean" / split / SHARD_NAME
        if sample_path.is_file():
            with sample_path.open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    sample.append({field: record[field][:160] for field in ("prompt", "response")})
                    if len(sample) == 2:
                        break
        if len(sample) == 2:
            break
    return {
        "version": manifest["version"],
        "manifest_sha256": checksum,
        "files": list(manifest["files"]),
        "counts": qc["counts"],
        "sample": sample,
        "card": (release_dir / "README.md").read_text(encoding="utf-8")[:500],
    }


def default_repo(config: Config, version: str) -> str:
    configured = config.get("publish", "repo", "")
    if configured and configured != "dem/distill-every-model":
        return str(configured)
    return f"distill-every-model-v{version}"


def prepare_hub_bundle(release_dir: Path, folder: Path, expected_manifest: str | None = None) -> dict:
    """Stage verified, eligible JSONL files and a card in an empty local folder."""
    manifest, checksum = _read_manifest(release_dir)
    if expected_manifest is not None and checksum != expected_manifest:
        raise PublishError("The release changed since preview. Review the latest dataset before uploading.")

    counts = {}
    licenses = set()
    for split in SPLITS:
        relative = Path("data", "clean", split, SHARD_NAME)
        digest = hashlib.sha256()
        count = 0
        output = folder / f"{split}.jsonl"
        with (release_dir / relative).open("rb") as source, output.open("wb") as target:
            for line in source:
                digest.update(line)
                record = json.loads(line)
                metadata = record["metadata"]
                if (record["source_tier"] != "clean"
                        or record["derived_from"] == "proprietary-api-output"
                        or is_commercial(metadata["source"], metadata["surface"])):
                    raise PublishError("The upload files contain restricted records. Prepare a new dataset.")
                target.write(line)
                count += 1
                licenses.add(record["license"].lower())
        if digest.hexdigest() != manifest["files"].get(relative.as_posix()):
            raise PublishError("The release data failed verification. Prepare a new dataset.")
        if count:
            counts[split] = count
        else:
            output.unlink()

    if not counts:
        raise PublishError("This release has no rows eligible for Hugging Face. Restricted records stay local.")

    license_ids = {"apache-2.0": "apache-2.0", "cc-by-4.0": "cc-by-4.0", "odc-by-1.0": "odc-by"}
    license_id = license_ids.get(next(iter(licenses))) if len(licenses) == 1 else None
    version = manifest["version"]
    _write_hub_card(folder, counts, f"Distill Every Model v{version}",
                    "Each row includes source and license metadata. See its `license` field for its terms.",
                    license_id)
    return {"version": version, "manifest_sha256": checksum, "count": sum(counts.values())}


def prepare_private_chats(records: list[DatasetRecord], folder: Path, config: Config) -> dict:
    """Package approved chat pairs for an explicitly private upload."""
    accepted, _, _ = curate_records(records, config)
    if not accepted:
        raise PublishError("No pairs passed the dataset checks. Review more chats and try again.")

    digest = hashlib.sha256()
    with (folder / "train.jsonl").open("wb") as target:
        for record in accepted:
            row = chat_row(record)
            line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            target.write(line)
            digest.update(line)
    _write_hub_card(folder, {"train": len(accepted)}, "Approved chats",
                    "Rows include source, redactions, and available tool activity, contributor outcomes and recorded checks. "
                    "The upload grants no additional license to the data.")
    return {"count": len(accepted), "manifest_sha256": digest.hexdigest(), "private_only": True}


def chat_row(record: DatasetRecord) -> dict:
    return {"prompt": record.prompt, "response": record.response, "messages": record.messages,
            "source": record.metadata["source"], "model": record.metadata.get("model") or None,
            "derived_from": record.derived_from,
            "activity": record.activity, "redactions": record.redaction_summary,
            "task_outcome": record.task_outcome, "evidence": record.evidence}


def _write_hub_card(folder: Path, counts: dict, title: str, description: str,
                    license_id: str | None = None) -> None:
    license_meta = f"license: {license_id}\n" if license_id else ""
    files = "".join(f"  - split: {split}\n    path: {split}.jsonl\n" for split in counts)
    rows = "".join(f"| {split} | {count} |\n" for split, count in counts.items())
    (folder / "README.md").write_text(f"""---
{license_meta}task_categories:
- text-generation
tags:
- distill-every-model
configs:
- config_name: default
  data_files:
{files}---

# {title}

Contributor-reviewed prompt and response pairs. DEM redacts, deduplicates, and
quality-checks pairs locally before export. Each JSON Lines record includes
`prompt`, `response`, and `messages`. {description}
Automated redaction may miss sensitive information.

| Split | Rows |
| --- | ---: |
{rows}
""", encoding="utf-8")


def export_hub_bundle(release_dir: Path, export_dir: Path) -> dict:
    export_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    folder = Path(tempfile.mkdtemp(prefix=f"dem-huggingface-{release_dir.name}-", dir=export_dir))
    try:
        result = prepare_hub_bundle(release_dir, folder)
    except Exception:
        shutil.rmtree(folder)
        raise
    return {**result, "destination": str(folder)}


def hub_token(keys_dir: Path) -> str | None:
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    legacy_token = keys_dir / "hf.token"
    return (os.environ.get("HF_TOKEN")
            or (legacy_token.read_text().strip() if legacy_token.is_file() else None) or get_token())


def hub_connection_errors() -> tuple[type[Exception], ...]:
    from huggingface_hub.utils import OfflineModeIsEnabled

    try:
        from httpx import RequestError
    except ImportError:
        return (OSError, OfflineModeIsEnabled)  # The requests-based client raises OSError subclasses.
    return (OSError, OfflineModeIsEnabled, RequestError)


def upload_hub_bundle(folder: Path, bundle: dict, config: Config, keys_dir: Path, *,
                      repo: str | None = None, public: bool = False, token: str = "") -> dict:
    if public and bundle.get("private_only"):
        raise PublishError("Approved chats can only be uploaded to a private dataset from this screen.")
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError, HFValidationError, validate_repo_id
    except ImportError as exc:
        raise PublishError("Install the Hugging Face client with `pip install -U huggingface_hub`.") from exc
    version = bundle.get("version")
    fallback_repo = default_repo(config, version) if version else "dem-chats"
    repo = (repo or fallback_repo).strip()
    try:
        validate_repo_id(repo)
    except HFValidationError as exc:
        raise PublishError("Enter a dataset name or owner/dataset, such as my-training-data.") from exc

    token = token.strip() or hub_token(keys_dir)
    if not token:
        raise PublishError("Paste a Hugging Face write token or sign in with `hf auth login` and retry.")

    api = HfApi(endpoint=HF_ENDPOINT, token=token)
    try:
        destination = api.create_repo(repo, repo_type="dataset", private=not public, exist_ok=True)
        info = api.repo_info(repo_id=destination.repo_id, repo_type="dataset")
        if info.private != (not public):
            visibility = "private" if info.private else "public"
            if bundle.get("private_only"):
                raise PublishError("That dataset is public. Choose a new name for a private dataset.")
            raise PublishError(f"That dataset is already {visibility}. Match its visibility or choose a new name.")
        commit = api.upload_folder(
            folder_path=folder, repo_id=destination.repo_id, repo_type="dataset",
            commit_message=f"dem-release-v{version}" if version else "Upload approved chats", parent_commit=info.sha,
        )
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status in (401, 403, 404):
            message = "Hugging Face denied access. Check the dataset owner and your token's write access."
        elif status in (409, 412):
            message = "The Hugging Face repository changed during upload. Check the dataset and retry."
        else:
            message = "Hugging Face could not complete the upload. Try again or download the files."
        raise PublishError(message) from exc
    except hub_connection_errors() as exc:
        raise PublishError("Could not reach Hugging Face. Try again or download the files.") from exc

    return {**bundle, "repo": destination.repo_id, "destination": str(destination), "commit": commit.oid}


def publish(release_dir: Path, config: Config, vault: Vault, keys_dir: Path,
            version: str | None = None, *, repo: str | None = None, public: bool = False,
            expected_manifest: str | None = None) -> dict:
    """Publish an explicitly selected release and log the resulting Hub commit."""
    if config.enabled("publish", "dry_run_required") and version is None:
        raise PublishError("Preview with `dem publish --dry-run`, then pass `--version` to upload.")
    if version is not None:
        release_dir = _latest_release(release_dir.parent, version)
    with tempfile.TemporaryDirectory(prefix="dem-huggingface-") as temp:
        folder = Path(temp)
        bundle = prepare_hub_bundle(release_dir, folder, expected_manifest)
        result = upload_hub_bundle(folder, bundle, config, keys_dir, repo=repo, public=public)
    vault.add_publish(result["repo"], result["commit"], result["manifest_sha256"], result["count"])
    return result
