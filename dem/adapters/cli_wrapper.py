"""CLI wrappers: route coding assistants through the local proxies.

`dem run -- <command>` injects base-URL env vars so tools that honor them are
captured through the API/Ollama proxies. No PTY/keystroke capture is enabled by
default — that remains an explicit, separate opt-in for a later release.
"""
from __future__ import annotations

import os
import subprocess

from ..config import Config


def wrapped_env(config: Config) -> dict[str, str]:
    env = os.environ.copy()
    if config.enabled("capture", "ollama_proxy"):
        port = config.get("capture", "ollama_port", 11435)
        env["OLLAMA_HOST"] = f"http://127.0.0.1:{port}"
    if config.enabled("capture", "api_proxy") and config.get("capture", "api_upstream"):
        port = config.get("capture", "api_port", 11436)
        env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
    return env


def run_wrapped(config: Config, argv: list[str]) -> int:
    if not argv:
        print("usage: dem run -- <command> [args...]")
        return 2
    print("DEM wrapper: routing supported tools through local capture proxies.")
    return subprocess.call(argv, env=wrapped_env(config))
