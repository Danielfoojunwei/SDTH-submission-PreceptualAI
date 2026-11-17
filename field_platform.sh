#!/usr/bin/env bash
# WirelessJEV Field Platform: set up (first run) and start.
#
#   ./field_platform.sh            the real site: asks for the SmartCar SSH password, then opens the platform
#   ./field_platform.sh --demo     rehearse with a synthetic site and car (no car, no password)
#
# First run needs internet (it installs four Python packages into .field-venv next to this
# script). Later runs work offline. macOS and Linux. Any extra options go to the platform,
# for example --port 8771.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
VENV="$HERE/.field-venv"
PORT=8770
for a in "$@"; do case "$a" in --port=*) PORT="${a#--port=}";; esac; done
prev=""; for a in "$@"; do [ "$prev" = "--port" ] && PORT="$a"; prev="$a"; done

find_python() {
  for p in python3.13 python3.12 python3; do
    if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
      command -v "$p"; return 0
    fi
  done
  return 1
}

if [ ! -x "$VENV/bin/python" ]; then
  PY="$(find_python)" || {
    if command -v uv >/dev/null 2>&1; then
      echo "Python 3.12 not found; getting it with uv..."
      uv python install 3.12 >/dev/null && PY="$(uv python find 3.12)"
    fi
  }
  if [ -z "${PY:-}" ]; then
    echo "This Mac's Python is too old: the platform needs Python 3.12 or newer." >&2
    echo "Install it from https://www.python.org/downloads/ (the macOS installer), or with Homebrew: brew install python@3.12" >&2
    echo "Then run ./field_platform.sh again." >&2
    exit 1
  fi
  echo "First run: setting up $VENV with $PY (needs internet, about a minute)..."
  "$PY" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet -r "$HERE/requirements-field.txt"
  # Make this folder's code importable in the environment (no build, nothing else installed).
  SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  echo "$HERE/src" > "$SITE/wirelessjev-field.pth"
  echo "Set up."
fi

"$VENV/bin/python" -c 'import wirelessjev.platform' || {
  echo "The environment in $VENV looks broken. Delete that folder and run this script again (with internet)." >&2
  exit 1
}

export MPLCONFIGDIR="${TMPDIR:-/tmp}/wj-field-mpl"
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
for _ in $(seq 1 50); do
  if "$VENV/bin/python" -c "import urllib.request,sys; urllib.request.urlopen('$URL/api/state', timeout=1)" 2>/dev/null; then break; fi
  sleep 0.2
done
echo "Open $URL if it did not open by itself. Press Ctrl+C here to stop the platform."
if [ -z "${WJ_NO_BROWSER:-}" ]; then
  if command -v open >/dev/null 2>&1; then open "$URL"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1; fi
fi
wait "$PID"
