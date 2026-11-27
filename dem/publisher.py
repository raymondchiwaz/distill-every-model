"""Dry-run preview + explicit publish (personal HF repo, or local export fallback)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from .config import Config
from .models import now_utc
from .vault import Vault


def _latest_release(releases_dir: Path, version: str | None = None) -> Path:
    if version:
        path = releases_dir / f"v{version}"
        if not path.exists():
            raise FileNotFoundError(f"release {version} not found")
        return path
    versions = sorted(
        (p for p in releases_dir.glob("v*") if p.is_dir()),
        key=lambda p: [int(x) for x in p.name[1:].split(".")],
    )
    if not versions:
        raise FileNotFoundError("no releases built yet — run `dem release` first")
    return versions[-1]


def dry_run(release_dir: Path) -> dict:
    """Fully local — makes no network calls."""
    manifest = json.loads((release_dir / "manifest.json").read_text())
    qc = json.loads((release_dir / "qc_report.json").read_text())
    sample_path = release_dir / "data" / "clean" / "train" / "train-00000-of-00001.jsonl"
    sample = []
    if sample_path.exists():
        with open(sample_path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 2:
                    break
                rec = json.loads(line)
                sample.append({"prompt": rec["prompt"][:160], "response": rec["response"][:160]})
    return {
        "version": manifest["version"],
        "files": list(manifest["files"].keys()),
        "counts": qc["counts"],
        "sample": sample,
        "card": (release_dir / "README.md").read_text()[:500],
    }


def _hf_token(keys_dir: Path) -> str | None:
    env = os.environ.get("HF_TOKEN")
    if env:
        return env
    path = keys_dir / "hf.token"
    return path.read_text().strip() if path.exists() else None


def publish(release_dir: Path, config: Config, vault: Vault, keys_dir: Path,
            version: str | None = None) -> dict:
    """Publish a reviewed release. Writes a publish_log entry in the vault."""
    if config.enabled("publish", "dry_run_required") and version is None:
        raise SystemExit("dry_run_required: review `dem publish --dry-run <version>` first, "
                         "then pass --version <version>")
    releases_dir = release_dir.parent
    release_dir = _latest_release(releases_dir, version)
    manifest = json.loads((release_dir / "manifest.json").read_text())
    expected = (release_dir / "manifest.sha256").read_text().strip()
    actual = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
    if expected != actual:
        raise RuntimeError("manifest integrity check failed — refusing to publish")

    repo = config.get("publish", "repo", "dem/distill-every-model")
    token = _hf_token(keys_dir)
    if token:
        restricted = release_dir / "data" / "restricted"
        if any(path.stat().st_size for path in restricted.rglob("*") if path.is_file()):
            raise SystemExit("This release contains restricted records. Keep it local until "
                             "a separate restricted publishing destination is supported.")
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise SystemExit("huggingface_hub not installed (pip install dem[hub])") from exc
        api = HfApi(token=token)
        api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(release_dir), repo_id=repo, repo_type="dataset",
            commit_message=f"dem-release-v{manifest['version']}",
        )
        commit = api.repo_info(repo_id=repo, repo_type="dataset").sha
        destination = f"https://huggingface.co/datasets/{repo}"
    else:
        export_root = config.data_dir / "export" / f"v{manifest['version']}"
        if export_root.exists():
            shutil.rmtree(export_root)
        shutil.copytree(release_dir, export_root)
        commit, destination = "local-export", str(export_root)
        print("No HF token found — exported locally. Add HF_TOKEN or keys/hf.token to publish.")

    vault.add_publish(repo, commit, expected, manifest["version"])
    return {"version": manifest["version"], "destination": destination, "commit": commit}
