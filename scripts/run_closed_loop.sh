#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run --frozen python -m wirelessjev.closed_loop.study --workers 4
uv run --frozen python scripts/replay_closed_loop.py
uv run --frozen python scripts/report_closed_loop.py
uv run --frozen python scripts/report_closed_loop.py --audit
uv run --frozen python -m pytest -q
