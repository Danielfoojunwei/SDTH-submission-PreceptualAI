# Compact-model independent confirmation

With sensing and compute charged, the fixed distilled model produces **228.12** misses per million versus **251.04** for optimised RadioMap. The paired difference is **-22.92 ppm**, 95% site-bootstrap interval **[-85.94, 39.06]**.
The primary superiority criterion fails.

The candidate was fixed after the first eight primary warehouses and before these eight new sites. All weights, the planner and the measured service ledger were frozen. The original frontier selection and all its negative results remain unchanged.

| Method | Misses / million | Forecast RMSE (dB) | Route-change RMSE (dB) | Warm compute charge (ms) |
|---|---:|---:|---:|---:|
| Optimised RadioMap | 251.04 | 7.81 | 2.29 | 67.20 |
| Previous JEPA | 217.19 | 10.23 | 2.69 | 139.86 |
| Three-member hybrid | 223.44 | 4.96 | 2.15 | 232.74 |
| Single distilled model | 228.12 | 5.11 | 2.21 | 53.05 |

Eight fresh warehouses, 100 robots, 120 seconds each; 1,920,000 commands per arm. Scans occur at the declared 1 Hz cadence with 30 ms outage; 30 ms handovers and a single-worker measured-service FIFO are charged.

The single model has a 4.39× smaller warm CPU budget than the frozen three-member hybrid. This ratio is from the fixed microbenchmark, not a physical robot measurement. Secondary paired comparisons are in `summary.json`; a RadioMap win is not automatically a win over the older neural references.

## Remaining limits

This remains the same synthetic canonical warehouse family, with idealised scanner phases and attached-link measurements, precomputed motion and mean-field supervision. Timing charges are measured CPU service profiles; scan/handover durations are assumptions. Arbitrary CAD, real radio hardware, full native counterfactual packet training and end-to-end fleet timing remain open.

Reproduce after the frontier study with `uv run --frozen python -m wirelessjev.frontier_confirmation --workers 4`, then `uv run --frozen python scripts/report_frontier_confirmation.py`. Audit with the latter script and `--audit`.
