"""Run the Python CLI shipped inside the npm package."""

import hashlib
import os
from pathlib import Path
import runpy
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory
import tomllib
import venv


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")


def cache_root() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")


def prepare_runtime() -> Path:
    project = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    python_build = (sys.version, sys.base_prefix, sysconfig.get_platform())
    fingerprint = hashlib.sha256((repr(python_build) + project).encode()).hexdigest()[:20]
    cache = (cache_root() / "distill-every-model").resolve()
    runtime = cache / fingerprint
    if (runtime / PYTHON).is_file():
        return runtime / PYTHON

    dependencies = tomllib.loads(project)["project"]["dependencies"]
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    print("Setting up DEM's Python dependencies for first use…", file=sys.stderr)
    with TemporaryDirectory(prefix="setup-", dir=cache) as staging:
        environment = Path(staging) / "runtime"
        venv.EnvBuilder(with_pip=True).create(environment)
        subprocess.run(
            [environment / PYTHON, "-I", "-m", "pip", "install", "--no-input",
             "--disable-pip-version-check", "--quiet", *dependencies],
            check=True,
        )
        # Run the interpreter directly. Console scripts in a venv retain its original path.
        try:
            environment.rename(runtime)
        except OSError:
            if not (runtime / PYTHON).is_file():
                raise
    return runtime / PYTHON


def main() -> int:
    if sys.argv[1] == "--run":
        module = sys.argv[2]
        del sys.argv[1:3]
        sys.path.insert(0, str(PACKAGE_ROOT))
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return 0

    try:
        python = prepare_runtime()
        command = [str(python), "-I", __file__, "--run", *sys.argv[1:]]
        if os.name != "nt":
            os.execv(python, command)
        return subprocess.call(command)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Could not set up or start DEM: {error}", file=sys.stderr)
        print("Check that Python has venv and pip support and can reach the Python package index.",
              file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
