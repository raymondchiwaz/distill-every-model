"""Configuration with safe defaults. Loaded from <data_dir>/config.toml."""
from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path

DEFAULT_COMMUNITY_REPO = "Distilleverymodel/community-chats"

DEFAULTS: dict = {
    "general": {"data_dir": "", "node_id": ""},
    "capture": {
        "ollama_proxy": True,
        "ollama_upstream": "http://127.0.0.1:11434",
        "ollama_port": 11435,
        "api_proxy": False,
        "api_upstream": "",
        "api_port": 11436,
        "browser_extension": False,
        "clipboard": False,
        "commercial_surfaces_default": "exclude",  # exclude | attest
        "capture_prompts": False,
        "capture_code": True,
    },
    "privacy": {
        "consent_mode": "default-deny",
        "pii_redaction": True,
        "secret_scan": True,
        "fail_closed": True,
    },
    "retention": {"raw_ttl_days": 30},
    "publish": {
        "target": "huggingface",
        "license": "Apache-2.0",
        "cadence": "manual",
        "dry_run_required": True,
        "repo": "",
        "community_repo": DEFAULT_COMMUNITY_REPO,
    },
    "review": {"port": 8787, "token": ""},
}


def default_data_dir() -> Path:
    env = os.environ.get("DEM_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".local" / "share" / "dem").resolve()


class Config:
    def __init__(self, path: Path | None = None):
        self._data = copy.deepcopy(DEFAULTS)
        self.path = path or (default_data_dir() / "config.toml")
        if self.path.exists():
            with open(self.path, "rb") as fh:
                loaded = tomllib.load(fh)
            for section, values in loaded.items():
                if section in self._data and isinstance(values, dict):
                    self._data[section].update(values)
        if not str(self.get("publish", "community_repo") or "").strip():
            self.set("publish", "community_repo", DEFAULT_COMMUNITY_REPO)

    @property
    def data_dir(self) -> Path:
        configured = self.get("general", "data_dir")
        return Path(configured).expanduser().resolve() if configured else default_data_dir()

    @property
    def sections(self) -> dict:
        return self._data

    def get(self, section: str, key: str, default=None):
        return self._data.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value) -> None:
        self._data.setdefault(section, {})[key] = value

    def enabled(self, section: str, key: str) -> bool:
        return bool(self.get(section, key, False))

    def save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for section, values in self._data.items():
            lines.append(f"[{section}]")
            for key, value in values.items():
                if isinstance(value, bool):
                    rendered = "true" if value else "false"
                elif isinstance(value, (int, float)):
                    rendered = str(value)
                else:
                    rendered = '""' if value == "" else f'"{str(value)}"'
                lines.append(f"{key} = {rendered}")
            lines.append("")
        self.path = self.data_dir / "config.toml"
        self.path.write_text("\n".join(lines), encoding="utf-8")

    def ensure_layout(self) -> dict[str, Path]:
        dirs = {
            "vault": self.data_dir / "vault",
            "keys": self.data_dir / "keys",
            "records": self.data_dir / "records",
            "releases": self.data_dir / "releases",
            "audit": self.data_dir / "audit",
            "export": self.data_dir / "export",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        return dirs


def load() -> Config:
    return Config()
