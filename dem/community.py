"""Prepare exact previews and submit selected chat pairs as Hugging Face pull requests."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import unquote

from huggingface_hub import HfApi, get_token
from huggingface_hub.utils import HfHubHTTPError, HFValidationError, validate_repo_id

from .config import Config
from .models import DatasetRecord
from .pipeline.release import curate_records
from .privacy import redact, redact_tool_payload
from .publisher import HF_ENDPOINT, PublishError, chat_row, hub_connection_errors


@dataclass(frozen=True)
class Contribution:
    payload: bytes
    count: int
    skipped: int

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    def review_token(self, repo: str) -> str:
        return hashlib.sha256(repo.encode() + b"\n" + self.payload).hexdigest()


def community_repo(config: Config, requested: str = "") -> str:
    repo = str(requested or config.get("publish", "community_repo") or "").strip()
    try:
        validate_repo_id(repo)
        if repo.count("/") != 1:
            raise HFValidationError("Missing dataset owner")
    except HFValidationError as exc:
        raise PublishError("Enter the public Hugging Face dataset as owner/dataset.") from exc
    return repo


def _has_sensitive_text(value, config: Config) -> bool:
    if isinstance(value, dict):
        return any(_has_sensitive_text(child, config) for child in value.values())
    if isinstance(value, list):
        return any(_has_sensitive_text(child, config) for child in value)
    if isinstance(value, str) and value.strip():
        result = redact(value, config, keep_code=True)
        return bool(result.quarantine_reason or result.redactions)
    return False


def prepare_contribution(records: list[DatasetRecord], config: Config) -> Contribution:
    screening = deepcopy(config)
    for setting in ("secret_scan", "pii_redaction", "fail_closed"):
        screening.set("privacy", setting, True)

    eligible = []
    for record in records:
        row = chat_row(record)
        if row["derived_from"] == "proprietary-api-output":
            # This fixed provenance label is metadata, even though the privacy rules flag the word in chat text.
            del row["derived_from"]
        if _has_sensitive_text(row, screening):
            continue
        if any(redact_tool_payload(step[field], screening, keep_code=True).quarantine_reason
               for step in record.activity for field in ("input", "output") if step.get(field)):
            continue
        # Keep deduplication stable across preview and send, even when captures are rebuilt.
        eligible.append(replace(record, id=record.consent_token))
    accepted, _, _ = curate_records(eligible, screening)
    payload = "".join(json.dumps(chat_row(record), ensure_ascii=False, sort_keys=True) + "\n"
                      for record in accepted).encode("utf-8")
    return Contribution(payload, len(accepted), len(records) - len(accepted))


def _upload_with_hf_cli(payload: bytes, repo: str, path: str, token: str) -> str:
    with tempfile.TemporaryDirectory(prefix="dem-contribution-") as folder:
        local_file = Path(folder) / "submission.jsonl"
        local_file.write_bytes(payload)
        try:
            result = subprocess.run(
                ["hf", "upload", repo, str(local_file), path, "--repo-type", "dataset", "--create-pr",
                 "--commit-message", "Contribute reviewed DEM chats", "--quiet"],
                env={**os.environ, "HF_ENDPOINT": HF_ENDPOINT, "HF_TOKEN": token},
                capture_output=True, text=True, check=True,
            )
        except FileNotFoundError as exc:
            raise PublishError("Install the Hugging Face CLI with `pip install -U huggingface_hub`.") from exc
        except subprocess.CalledProcessError as exc:
            raise PublishError("The Hugging Face CLI could not confirm the upload. Check the dataset and sign-in before trying again.") from exc

    dataset = f"{HF_ENDPOINT}/datasets/{repo}"
    urls = unquote(result.stdout)
    match = re.search(re.escape(dataset) + r"/(?:discussions/|(?:blob|tree)/refs/pr/)(\d+)\b", urls)
    return f"{dataset}/discussions/{match[1]}" if match else f"{dataset}/discussions"


def submit_contribution(contribution: Contribution, repo: str, *, token: str = "", via_hf_cli: bool = False) -> dict:
    if not contribution.count:
        raise PublishError("No selected pairs passed the community upload checks.")
    token = token.strip() or get_token()
    if not token:
        raise PublishError("Sign in with `hf auth login`, then send again.")

    api = HfApi(endpoint=HF_ENDPOINT, token=token)
    path = f"data/submissions/{contribution.checksum}.jsonl"
    try:
        info = api.repo_info(repo, repo_type="dataset", token=False)
        if info.private is not False:
            raise PublishError("Community contributions require a public dataset.")
        if api.file_exists(repo, path, repo_type="dataset", token=False):
            raise PublishError("These pairs are already in this community dataset.")
        if via_hf_cli:
            destination = _upload_with_hf_cli(contribution.payload, repo, path, token)
            commit_sha = None
        else:
            commit = api.upload_file(
                path_or_fileobj=contribution.payload, path_in_repo=path, repo_id=repo, repo_type="dataset",
                create_pr=True, parent_commit=info.sha, commit_message="Contribute reviewed DEM chats",
            )
            destination = commit.pr_url or commit.commit_url
            commit_sha = commit.oid
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status in (401, 403, 404):
            message = "Check the public dataset, enable pull requests, and sign in with a token that can open them."
        elif status in (409, 412):
            message = "The dataset changed during submission. Check Hugging Face before trying again."
        else:
            message = "Hugging Face could not confirm the submission. Check the dataset before trying again."
        raise PublishError(message) from exc
    except hub_connection_errors() as exc:
        raise PublishError("Could not confirm the submission. Check Hugging Face before trying again.") from exc

    return {"repo": repo, "destination": destination, "commit": commit_sha,
            "manifest_sha256": contribution.checksum, "count": contribution.count, "skipped": contribution.skipped}
