#!/bin/sh
# Install the released DEM CLI for the current user on macOS or Linux.
set -eu

install_dem() {
    version=0.3.2
    runtime="${XDG_DATA_HOME:-$HOME/.local/share}/distill-every-model/cli-$version"
    bin_dir="${DEM_BIN_DIR:-$HOME/.local/bin}"
    python="${DEM_PYTHON:-python3}"
    package="https://github.com/raymondchiwaz/distill-every-model/releases/download/v$version/distilleverymodel-distill-every-model-$version.tgz"
    digest="47ebaf2b22e2fb75246e884119e3c18ab83ac45a93da1bc86f8fe6cb3c29abe48a1e72f1d5528b9c9203842786bf99227602a716e11e9353931ca9a2de117edb"

    "$python" -c 'import venv, ensurepip, sys; sys.exit(sys.version_info < (3, 11))' || {
        echo "Install Python 3.11 or newer with venv and pip support, then rerun the installer." >&2
        return 1
    }
    mkdir -p "$bin_dir" "$(dirname "$runtime")"
    for name in dem demd; do
        if [ -e "$bin_dir/$name" ] || [ -L "$bin_dir/$name" ]; then
            case "$(readlink "$bin_dir/$name" || true)" in
            "$(dirname "$runtime")"/cli-*/bin/"$name") ;;
            *)
                echo "An existing $bin_dir/$name belongs to another installation. Set DEM_BIN_DIR to a different directory." >&2
                return 1
                ;;
            esac
        fi
    done
    if [ ! -f "$runtime/.installed" ]; then
        mkdir "$runtime.lock" || {
            echo "A setup may be running. Check $runtime.lock before retrying." >&2
            return 1
        }
        trap 'rmdir "$runtime.lock"' 0
        trap 'exit 1' HUP INT TERM
        "$python" -m venv "$runtime"
        "$runtime/bin/python" -m pip install --disable-pip-version-check "dem-distill-every-model @ $package#sha512=$digest"
        "$runtime/bin/dem" --version
        touch "$runtime/.installed"
    fi
    for name in dem demd; do
        if [ "$(readlink "$bin_dir/$name" || true)" != "$runtime/bin/$name" ]; then
            ln -sfn "$runtime/bin/$name" "$bin_dir/$name"
        fi
    done
    case ":$PATH:" in
        *":$bin_dir:"*) echo 'Installed. Run dem review --open to begin.' ;;
        *) echo "Installed in $bin_dir. Add this directory to PATH or run \"$bin_dir/dem\" review --open." ;;
    esac
}

install_dem
