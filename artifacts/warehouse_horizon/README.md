# Route-aware horizon evidence

All observations and actions are simulated. The campaign uses actual pinned
d3 inference, actual archived NIST Wi-Fi curves and six trained compact models.
It does not claim physical SmartCar qualification or SOTA performance.

- `status.json`, `manifest.json`: completed report and checkpoint hashes.
- `executed_sources.tar.gz`, `provenance.json`: exact primary executed code.
- `checkpoints/`: six finite compact checkpoints; pretrained d3 weights are
  downloaded separately from the pinned public model repository.
- `reader.json`, `selection.json`: actual mission-reading receipts and
  validation-only recipes sealed before held-out evaluation.
- `offline.json`, `closedloop.json`, `snapshot.json`: held-out forecast results,
  all 30 control rollouts and summaries with crossed bootstrap intervals.
- `selector/`: separately versioned post-holdout selector, 42 validation
  rollouts, 24 fresh audit rollouts, sealed policy and exact executed sources.
  Decompress `gate_audit.json.gz` to recover the unchanged raw report. Its
  original SHA-256 and compressed-file hash are in the selector manifest.
- `examples/`: real held-out simulated observation and actual CLI output;
  explicitly caller-provided contracts are distinguished from model inference.
- `forecast_and_control.{png,svg}`: exportable scientific figures. Forecast
  shading is the three-seed range; control intervals resample seeds and scenes.

The numerical planner improves over budgeted strongest-signal roaming in this
specific twelve-switch policy scenario. Unconstrained strongest-signal roaming
has higher service in the primary holdout and violates that policy budget.
Learned models fail the closed-loop acceptance criteria; all eighteen guarded
audit outcomes exactly reproduce numerical outcomes, rather than demonstrating
eighteen learned successes.

Protocols, commands and limitations: `docs/WAREHOUSE_HORIZON.md`,
`docs/WAREHOUSE_HORIZON_RESULTS.md`, `docs/WAREHOUSE_HORIZON_SELECTOR.md`.
