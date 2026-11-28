"""Names and local transcript locations shared by the CLI and review UI."""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HistorySource:
    name: str
    reader: str


SOURCES = {
    "claude-code": HistorySource("Claude Code", "claude"),
    "codex": HistorySource("Codex", "codex"),
    "aider": HistorySource("Aider", "aider"),
    "amp": HistorySource("Amp", "amp"),
    "antigravity": HistorySource("Google Antigravity", "antigravity"),
    "cline": HistorySource("Cline", "extension"),
    "continue": HistorySource("Continue", "continue"),
    "copilot": HistorySource("GitHub Copilot CLI", "copilot"),
    "crush": HistorySource("Crush", "crush"),
    "cursor": HistorySource("Cursor", "cursor"),
    "deepseek": HistorySource("DeepSeek Harness", "deepseek"),
    "droid": HistorySource("Factory Droid", "droid"),
    "gemini-cli": HistorySource("Gemini CLI", "gemini"),
    "goose": HistorySource("Goose", "goose"),
    "kilo-code": HistorySource("Kilo Code", "extension"),
    "kimi-cli": HistorySource("Kimi Code CLI", "kimi"),
    "kiro": HistorySource("Kiro", "kiro"),
    "mistral-vibe": HistorySource("Mistral Vibe", "vibe"),
    "opencode": HistorySource("OpenCode", "opencode"),
    "openhands": HistorySource("OpenHands", "openhands"),
    "pi": HistorySource("Pi", "pi"),
    "qwen-code": HistorySource("Qwen Code", "qwen"),
    "roo-code": HistorySource("Roo Code", "extension"),
}
HISTORY_SOURCES = tuple(SOURCES)
SOURCE_PLACEHOLDERS = ", ".join("?" for _ in HISTORY_SOURCES)
EXPORT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".db", ".sqlite", ".zip"}
SUBAGENT_FOLDERS = {"subagents", "sub-executions"}
DEEPSEEK_LOG_NAME = re.compile(r"session(?:\.v([1-9][0-9]*))?\.jsonl(?:\.zstd)?")


def is_child_history(path):
    return bool(SUBAGENT_FOLDERS.intersection(path.parts) or path.name.startswith("agent-"))


def _home(env, default):
    return Path(os.getenv(env) or default).expanduser()


def _config_root():
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support"
    if os.name == "nt":
        return _home("APPDATA", "~/AppData/Roaming")
    return _home("XDG_CONFIG_HOME", "~/.config")


def _data_root():
    if sys.platform == "darwin" or os.name == "nt":
        return _config_root()
    return _home("XDG_DATA_HOME", "~/.local/share")


def _native_locations(source):
    data = _data_root()
    if source in ("cline", "roo-code", "kilo-code"):
        extension = {"cline": "saoudrizwan.claude-dev", "roo-code": "rooveterinaryinc.roo-cline",
                     "kilo-code": "kilocode.kilo-code"}[source]
        return [
            (_config_root() / editor / "User/globalStorage" / extension, "tasks/*/ui_messages.json")
            for editor in ("Code", "Code - Insiders", "VSCodium", "Cursor", "Windsurf")
        ] + [
            (Path.home() / engine / "data/User/globalStorage" / extension, "tasks/*/ui_messages.json")
            for engine in (".vscode-server", ".vscode-server-insiders")
        ]
    locations = {
        "claude-code": [(_home("CLAUDE_CONFIG_DIR", "~/.claude"), "projects/*/*.jsonl"),
                        (_home("CLAUDE_CONFIG_DIR", "~/.claude"), "projects/*/*/subagents/agent-*.jsonl")],
        "codex": [(_home("CODEX_HOME", "~/.codex"), "sessions/**/rollout-*.jsonl"),
                  (_home("CODEX_HOME", "~/.codex"), "archived_sessions/**/rollout-*.jsonl")],
        "aider": [(Path.cwd(), ".aider.chat.history.md")],
        "amp": [(_home("AMP_THREADS_DIR", data / "amp/threads"), "T-*.json")],
        "antigravity": [
            (Path.home() / ".gemini" / directory, f"brain/*/.system_generated/logs/{name}")
            for directory in ("antigravity", "antigravity-cli", "antigravity-ide")
            for name in ("transcript.jsonl", "transcript_full.jsonl")
        ],
        "continue": [(_home("CONTINUE_GLOBAL_DIR", "~/.continue"), "sessions/*.json")],
        "copilot": [(_home("COPILOT_HOME", "~/.copilot"), "session-state/*/events.jsonl")],
        "crush": [(Path.cwd(), ".crush/crush.db")],
        "cursor": [(Path.home() / ".cursor", "projects/*/agent-transcripts/**/*.jsonl"),
                   (Path.home() / ".cursor", "projects/*/agent-transcripts/*.txt")],
        "deepseek": [
            (_home("DSH_HOME", "~/.dsh"), f"sessions/*/*/{name}")
            for name in ("session.jsonl*", "session.v*.jsonl*")
        ],
        "droid": [(_home("FACTORY_DIR", "~/.factory"), "sessions/*/*.jsonl")],
        "gemini-cli": [(_home("GEMINI_CLI_HOME", "~") / ".gemini", "tmp/*/chats/session-*.json*")],
        "goose": [(_home("GOOSE_PATH_ROOT", data / "goose"), "sessions/sessions.db")],
        "kimi-cli": [(_home("KIMI_SHARE_DIR", "~/.kimi"), "sessions/*/*/context.jsonl")],
        "kiro": [(_home("KIRO_HOME", "~/.kiro"), "sessions/cli/**/messages.jsonl"),
                 (_home("KIRO_HOME", "~/.kiro"), "sessions/cli/**/events.jsonl")],
        "mistral-vibe": [(_home("VIBE_HOME", "~/.vibe"), "logs/session/*/messages.jsonl"),
                         (_home("VIBE_HOME", "~/.vibe"), "logs/session/*/messages.json")],
        "opencode": [(_home("XDG_DATA_HOME", "~/.local/share") / "opencode", "opencode.db")],
        "pi": [(_home("PI_CODING_AGENT_DIR", "~/.pi/agent"), "sessions/*/*.jsonl")],
        "qwen-code": [(_home("QWEN_CODE_HOME", "~") / ".qwen", "projects/*/chats/*.jsonl")],
    }
    return locations.get(source, ())


def _history_file_type(source, file):
    if source == "deepseek":
        return file.suffix.lower() == ".zip" or file.name.lower().endswith((".jsonl", ".jsonl.zstd"))
    return file.suffix.lower() in EXPORT_SUFFIXES


def history_files(source, path=None, export_root=None):
    """Only inspect filenames during discovery. Open files after import consent."""
    if source not in SOURCES:
        raise ValueError(f"Unknown history source: {source}")
    if path is not None:
        selected = Path(path).expanduser()
        if selected.is_file() and _history_file_type(source, selected):
            return [selected]
        if not selected.is_dir():
            raise ValueError("--path must be an existing transcript file or directory")
        candidates = selected.rglob("*")
    else:
        paths = {file for root, pattern in _native_locations(source) for file in root.glob(pattern)}
        if export_root:
            paths.update((Path(export_root) / source).rglob("*"))
        candidates = paths
    files = {file for file in candidates
             if _history_file_type(source, file)
             and not file.name.startswith("sessions-index")
             and file.name not in {"sessions.json", "session.json", "metadata.json"}
             and file.is_file() and not file.is_symlink()}
    if source == "antigravity":
        files = {file for file in files if file.name != "transcript_full.jsonl"
                 or file.with_name("transcript.jsonl") not in files}
    if source == "deepseek":
        # DeepSeek keeps older generations after migrating a session in place.
        latest = {}
        for file in sorted(files):
            match = DEEPSEEK_LOG_NAME.fullmatch(file.name)
            if match is None:
                continue
            generation = (int(match[1] or 0), file.suffix == ".zstd")
            if file.parent not in latest or generation > latest[file.parent][0]:
                latest[file.parent] = generation, file
        files = ({file for _, file in latest.values()}
                 | {file for file in files if not DEEPSEEK_LOG_NAME.fullmatch(file.name)})
    return sorted(files)
