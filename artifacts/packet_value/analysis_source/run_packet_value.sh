#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for phase in data train calibrate development; do
  uv run --frozen python -m wirelessjev.packet_value.study --phase "$phase" --workers 4
done
uv run --frozen python scripts/review_packet_development.py
uv run --frozen python scripts/diagnose_packet_value.py
uv run --frozen python scripts/report_packet_value.py
uv run --frozen python scripts/report_packet_value.py --audit
uv run --frozen python -m pytest -q
