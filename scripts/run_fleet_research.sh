#!/usr/bin/env bash
# Run sequentially; do not launch concurrent copies against one artifact dir.
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run --frozen python -m wirelessjev.fleetwm.study --phase train --workers 3
uv run --frozen python scripts/analyze_fleet_validation.py
uv run --frozen python -m wirelessjev.fleetwm.study --phase select --workers 3
uv run --frozen python -m wirelessjev.fleet_strong_reference --phase selection --workers 3
uv run --frozen python -m wirelessjev.fleetwm.study --phase test --workers 3
uv run --frozen python -m wirelessjev.fleet_strong_reference --phase test --workers 3
uv run --frozen python scripts/verify_fleet_forecasts.py
uv run --frozen python scripts/report_fleet_world_model.py
uv run --frozen python scripts/audit_fleet_world_model.py
uv run --frozen python -m pytest -q
