#!/usr/bin/env bash
# WirelessJEV Field Platform: set up (first run) and start.
#
#   ./field_platform.sh            the real site: asks for the SmartCar SSH password, then opens the platform
#   ./field_platform.sh --demo     rehearse with a synthetic site and car (no car, no password)
#   ./field_platform.sh --setup-only   set up and stop (for example the evening before)
#
# The full kit carries everything: the code, WirelessJEV's d3-lite model (hf-cache/) and, for
# Apple-Silicon Macs with Python 3.12, every Python package (wheels/). There the first run needs
# no internet; on other laptops it downloads the same pinned packages once. After setup the
# platform runs fully offline. macOS and Linux. Extra options go to the platform (e.g. --port 8771).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
VENV="$HERE/.field-venv"
WHEELS="$HERE/wheels/macos-arm64-py312"
ARGS=(); for a in "$@"; do [ "$a" = "--with-wirelessjev" ] || ARGS+=("$a"); done   # old flag: d3-lite is always included now
set -- "${ARGS[@]+"${ARGS[@]}"}"
PORT=8770
prev=""; for a in "$@"; do case "$a" in --port=*) PORT="${a#--port=}";; esac; [ "$prev" = "--port" ] && PORT="$a"; prev="$a"; done

find_python() {
  for p in python3.12 python3.13 python3; do
    if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
      command -v "$p"; return 0
    fi
  done
  return 1
}

if [ ! -f "$VENV/.setup-complete" ]; then
  PY="$(find_python)" || {
    if command -v uv >/dev/null 2>&1; then
      echo "Python 3.12 not found; getting it with uv..."
      uv python install 3.12 >/dev/null && PY="$(uv python find 3.12)"
    fi
  }
  if [ -z "${PY:-}" ]; then
    echo "This computer's Python is too old: the platform needs Python 3.12 or newer." >&2
    echo "Install it from https://www.python.org/downloads/ (the macOS installer), or with Homebrew: brew install python@3.12" >&2
    echo "Then run ./field_platform.sh again." >&2
    exit 1
  fi
  if [ -f "$HERE/MODEL_SHA256.txt" ]; then
    echo "Checking the WirelessJEV model files..."
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 -c --quiet "$HERE/MODEL_SHA256.txt"; else sha256sum -c --quiet "$HERE/MODEL_SHA256.txt"; fi || {
      echo "The model files are damaged or incomplete: unzip the kit again." >&2; exit 1; }
  fi
  echo "First run: setting up $VENV with $PY ..."
  rm -rf "$VENV"
  "$PY" -m venv "$VENV"
  REQ=(-r "$HERE/requirements-field.txt" -r "$HERE/requirements-wirelessjev.txt")
  PLAT="$("$VENV/bin/python" -c 'import platform, sys; print(f"{platform.system()}-{platform.machine()}-{sys.version_info.major}.{sys.version_info.minor}")')"
  if [ -d "$WHEELS" ] && [ "$PLAT" = "Darwin-arm64-3.12" ] && "$VENV/bin/python" -m pip install --quiet --no-cache-dir --disable-pip-version-check --no-index --find-links "$WHEELS" "${REQ[@]}"; then
    echo "Installed from the kit (no internet used)."
  else
    echo "Downloading the pinned packages for this computer ($PLAT; needs internet, about 0.7 GB)..."
    "$VENV/bin/python" -m pip install --quiet --upgrade pip
    "$VENV/bin/python" -m pip install --quiet "${REQ[@]}"
  fi
  # Make this folder's code importable in the environment (no build, nothing else installed).
  SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  echo "$HERE/src" > "$SITE/wirelessjev-field.pth"
  touch "$VENV/.setup-complete"
  echo "Set up."
fi

# Everything runs from this folder, offline: the bundled model, never the internet.
export HF_HOME="$HERE/hf-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/wj-field-mpl"

"$VENV/bin/python" -c 'import wirelessjev.platform' || {
  echo "The environment in $VENV looks broken. Delete that folder and run this script again." >&2
  exit 1
}
for a in "$@"; do [ "$a" = "--setup-only" ] && { echo "Setup complete; start it with ./field_platform.sh"; exit 0; }; done

DEMO=0; for a in "$@"; do [ "$a" = "--demo" ] && DEMO=1; done
if [ "$DEMO" = 0 ] && [ -z "${WJ_SSH_PASSWORD:-}" ]; then
  read -r -s -p "SmartCar SSH password (not shown; press Enter to skip and use read-only analysis only): " WJ_SSH_PASSWORD
  echo
  export WJ_SSH_PASSWORD
fi

URL="http://127.0.0.1:$PORT"
"$VENV/bin/python" -m wirelessjev.platform "$@" &
PID=$!
trap 'kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null' INT TERM
for _ in $(seq 1 100); do
  if "$VENV/bin/python" -c "import urllib.request; urllib.request.urlopen('$URL/api/state', timeout=5)" 2>/dev/null; then break; fi
  sleep 0.3
done
echo "Open $URL if it did not open by itself. Press Ctrl+C here to stop the platform."
if [ -z "${WJ_NO_BROWSER:-}" ]; then
  if command -v open >/dev/null 2>&1; then open "$URL"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1; fi
fi
wait "$PID"
