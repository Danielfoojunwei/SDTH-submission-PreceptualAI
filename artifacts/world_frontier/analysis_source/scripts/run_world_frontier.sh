#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run --frozen python -m wirelessjev.frontier.study --phase train --workers 3
uv run --frozen python scripts/calibrate_frontier_timing.py
uv run --frozen python -m wirelessjev.frontier.study --phase select --workers 4
uv run --frozen python -m wirelessjev.frontier.study --phase test --workers 4
uv run --frozen python scripts/report_world_frontier.py
uv run --frozen python scripts/audit_world_frontier.py
uv run --frozen python -m wirelessjev.frontier_confirmation --workers 4
uv run --frozen python scripts/report_frontier_confirmation.py
uv run --frozen python scripts/report_frontier_confirmation.py --audit
uv run --frozen python -m wirelessjev.hardware_budget --workers 4
uv run --frozen python scripts/report_hardware_controls.py hardware_budget
uv run --frozen python scripts/report_hardware_controls.py hardware_budget --audit
uv run --frozen python -m wirelessjev.handover_admission --workers 4
uv run --frozen python scripts/report_hardware_controls.py handover_admission
uv run --frozen python scripts/report_hardware_controls.py handover_admission --audit
uv run --frozen python -m pytest -q
