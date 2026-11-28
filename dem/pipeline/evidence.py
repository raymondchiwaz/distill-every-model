"""Summarize outcomes reported by recorded tools without re-running them."""
from __future__ import annotations

import json
import re
import shlex
from collections.abc import Iterable
from pathlib import PurePosixPath

SHELL_TOOLS = {"bash", "exec_command", "local_shell_call", "shell", "terminal"}
FILE_TOOLS = {"apply_patch", "edit", "multiedit", "write", "edit_file", "write_file", "create_file"}
EXIT_CODE = re.compile(r"(?m)^Process exited with code (-?\d+)\s*$")
PATCH_FILE = re.compile(r"^\*\*\* (?:Add|Update|Delete|Move to) File: (.+)$|^\*\*\* Move to: (.+)$", re.MULTILINE)
JS_TOOL_CALL = re.compile(r"\btools\.[A-Za-z_]\w*\s*\(")
JS_COMMAND = re.compile(
    r"""\btools\.exec_command\s*\(\s*\{\s*cmd\s*:\s*("(?:\\.|[^"\\])*"|'[^'\\]*'|`[^`\\]*`)(?=\s*[,}])"""
)
EXEC_RESULT = re.compile(r'\{\s*"chunk_id"\s*:')


def _tool_name(step: dict[str, str]) -> str:
    return step["text"].rsplit(".", 1)[-1].lower()


def _output(step: dict[str, str]) -> str:
    return "" if step.get("output_hold") else step.get("output", "")


def _tool_result(step: dict[str, str], name: str) -> str | None:
    result = step.get("exit_code")
    if result is not None and re.fullmatch(r"-?\d+", result):
        return "passed" if int(result) == 0 else "failed"
    if step.get("is_error") in ("true", "false"):
        return "failed" if step["is_error"] == "true" else "passed"
    output = _output(step)
    if name in SHELL_TOOLS and output.startswith("Chunk ID:"):
        # Match only Codex's execution header, before the command's stdout.
        header = re.split(r"\n(?:Final )?[Oo]utput:", output, maxsplit=1)[0]
        match = EXIT_CODE.search(header)
        if match:
            return "passed" if int(match[1]) == 0 else "failed"
    if name == "apply_patch" and output.startswith("Success. Updated the following files:"):
        return "passed"
    return None


def _check_name(command: str) -> str | None:
    # A shell wrapper reports only the final exit status of a compound command.
    if re.search(r"[;&|<>\n\x60$]", command):
        return None
    try:
        args = shlex.split(command)
    except ValueError:
        return None
    if not args:
        return None
    executable = PurePosixPath(args[0]).name
    if re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", executable) and args[1:2] == ["-m"]:
        executable, args = (args[2], args[2:]) if len(args) > 2 else ("", [])
    if executable in ("pytest", "unittest"):
        return executable
    if executable in ("cargo", "go", "dotnet", "make") and args[1:2] in (["test"], ["build"], ["check"]):
        return f"{executable} {args[1]}"
    if executable in ("npm", "pnpm", "yarn", "bun"):
        action = args[2:3] if args[1:2] == ["run"] else args[1:2]
        if action in (["test"], ["build"], ["lint"], ["typecheck"]):
            return f"{executable} {action[0]}"
    if executable == "git" and args[1:2] == ["diff"] and "--check" in args[2:]:
        return "git diff --check"
    if executable == "ruff" and args[1:2] == ["check"]:
        return "ruff check"
    if executable in ("mypy", "pyright"):
        return executable
    return None


def _js_shell_result(code: str, output: str) -> tuple[str, str] | None:
    # The code-mode wrapper is understandable when it has one literal shell
    # invocation and emits one complete runner result. Treat other logs as text.
    command = JS_COMMAND.search(code)
    if command is None or len(JS_TOOL_CALL.findall(code)) != 1:
        return None
    result_starts = list(EXEC_RESULT.finditer(output))
    if len(result_starts) != 1:
        return None
    literal = command[1]
    try:
        recorded, _ = json.JSONDecoder().raw_decode(output, result_starts[0].start())
        command_text = json.loads(literal) if literal.startswith('"') else literal[1:-1]
    except ValueError:
        return None
    if (type(recorded.get("exit_code")) is not int or not isinstance(recorded.get("chunk_id"), str)
            or not isinstance(recorded.get("wall_time_seconds"), (int, float)) or "${" in command_text):
        return None
    return command_text, "passed" if recorded["exit_code"] == 0 else "failed"


def _display_path(value: str, workdir: str = "") -> str:
    if not value or any(marker in value for marker in ("[SECRET_", "[EMAIL_", "\n", "\r")):
        return ""
    path = PurePosixPath(value.replace("\\", "/"))
    if workdir:
        base = PurePosixPath(workdir.replace("\\", "/"))
        if path.is_relative_to(base):
            path = path.relative_to(base)
    if path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:/", path.as_posix()):
        return path.name
    return path.as_posix() if path.name else ""


def collect_task_evidence(activity: Iterable[dict[str, str]]) -> dict | None:
    files: dict[str, None] = {}
    checks = []
    errors = 0
    for step in activity:
        if step["type"] != "tool_call":
            continue
        name = _tool_name(step)
        result = _tool_result(step, name)
        raw_input = "" if step.get("input_hold") else step.get("input", "")
        if name == "js" and raw_input:
            shell = _js_shell_result(raw_input, _output(step))
            if shell:
                command, result = shell
                check = _check_name(command)
                if check:
                    checks.append({"name": check, "status": result, "source": "recorded_tool"})
        errors += result == "failed"
        if not raw_input:
            continue
        try:
            parsed = json.loads(raw_input)
        except ValueError:
            parsed = None
        arguments = parsed if isinstance(parsed, dict) else {}
        if name in SHELL_TOOLS and result is not None:
            command = arguments.get("cmd") or arguments.get("command") or raw_input
            check = _check_name(command) if isinstance(command, str) else None
            if check:
                checks.append({"name": check, "status": result, "source": "recorded_tool"})
        if name in FILE_TOOLS and result == "passed":
            paths = [arguments[key] for key in ("file_path", "filePath", "path") if isinstance(arguments.get(key), str)]
            if name == "apply_patch":
                patch = arguments.get("patch")
                paths.extend(first or second for first, second in PATCH_FILE.findall(patch if isinstance(patch, str) else raw_input))
            workdir = arguments.get("workdir") or arguments.get("cwd")
            for value in paths:
                display = _display_path(value, workdir if isinstance(workdir, str) else "")
                if display:
                    files[display] = None
    if not files and not checks and not errors:
        return None
    return {"files_changed": list(files), "tool_errors": errors, "checks": checks}
