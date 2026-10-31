# Route-aware horizon campaign

This follow-up preserves every v2 result. It addresses the one-step versus
30-second mismatch and the four-plan candidate bottleneck. It is a separately
versioned simulation study; it does not qualify physical SmartCars.

## Fixed protocol before held-out evaluation

- Six 5-second outcomes per 30-second action commitment; model inputs contain
  past capped scans/positions and the committed route, never actual future
  robot positions, propagation coefficients or evaluation outcomes.
- Ten-minute episodes, independent training/validation/test environments.
  Counterfactual actions share the same future mobility and blockage. Looking
  ahead must not advance the live episode or consume its random streams.
- A censored-observation, nominal-geometry radio belief is the strong numerical
  baseline. Residual MLP and temporal graph models predict per-robot service,
  critical-rate violations and signal at all six offsets. Known switch
  interruptions are charged once, across the actual affected intervals.
- Build partial-move candidates under hard switch budgets and visible-target
  checks. Compare budgeted strongest-scan, route-aware numerical planning,
  residual MLP and graph planning with the same search budget. Keep an
  unconstrained strongest-scan reference separate from budget-feasible arms.
  The primary budget is twelve as an explicit operator-policy scenario.
  This is not the existing transaction concurrency limit: batches of twelve
  concurrent transactions could permit more switches across thirty seconds.
  The simulator assumes successful simultaneous switches and does not model
  a transaction scheduler or establish a hardware switching-rate limit.
- Use actual pinned d3 inference for the mission reader; record abstentions.
  Evaluate a declared mission schedule, per-robot service, critical-rate
  violations, reconnections, constraint compliance and complete local compute.
- Three independent compact-model seeds; evaluate each on every held-out
  environment, rather than coupling one model seed with one environment.
  Select checkpoints, residual strength and risk calibration on validation
  environments only. Retain negative results and every predeclared arm.

## Important modelling limits

Radio scans and delivery counters are simulated. The nominal CAD layout is
public input; hidden wall/material parameters are not. The supplied committed
route can diverge from realised motion after a task finishes. Critical-rate
violations are the twin's rate proxy, not physical packet deadlines. Scan
interruption is reported as unmodelled in the primary protocol. Four seconds
of switch interruption is an assumption pending hardware calibration.

No broader foundation-model, ANSpect waveform, real-radio or universal robot
claim follows from this campaign.

## Mission semantics and proposal interface

The two predeclared missions maximise either delivered demand fraction or the
mean service of the lowest-served ten percent of robots, minus one times the
critical-rate-violation fraction. The risk term is averaged over the six
five-second intervals; it is not the probability of any failure in thirty
seconds. Both missions impose at most twelve switches at each decision.

`scripts/plan_warehouse_horizon.py` accepts a validated observation JSON and
either a complete controlled-language mission or a caller-supplied typed
contract. Language input executes the actual pinned d3 model; a provided
contract is explicitly labelled as such. The gateway rejects unsupported
extra clauses, inconsistent units, stale hardware timestamps and invalid scan
bounds. It writes a proposal and six forecast arrays, never a car command.
Self-reported hardware origin is not authenticated by a JSON label, and the
models remain validated on simulation only.

The archived six compact checkpoints are usable directly. Numerical planning
is the default. Requesting `--model-key 0-False` applies the sealed closed-loop
guard; in this campaign every learned model was rejected and uses the numerical
recipe instead. Missing guard evidence also falls back to numerical planning;
modified receipts fail validation. `--research-residual` explicitly enables an
unguarded residual for simulation inputs only. The guard is empirical, not
a physical safety proof or an authentication mechanism.

```bash
uv run --frozen python scripts/plan_warehouse_horizon.py \
  --observation artifacts/warehouse_horizon/examples/observation.json \
  --contract artifacts/warehouse_horizon/examples/contract.json \
  --out /tmp/warehouse-proposal.json
```

To execute the actual language reader, replace `--contract ...` with
`--mission 'Reconnect at most 12 robots per decision; maximise total data delivered. Apply a penalty of 1 per fraction of robots with critical-link failures.'`.
Only this complete grammar is supported. Probabilities over typed model answers
describe interpretation, not physical failure probabilities.

## Reproduction

Install the locked dependencies with `uv sync --frozen`. Actual NIST Wi-Fi
link curves are preserved in `artifacts/warehouse_v2/studies/wifi_link_curves.json`.
The pinned d3 weights are downloaded automatically if they are not cached;
these public weights require no Hugging Face token. Compact training runs on CPU.

```bash
uv run --frozen python scripts/run_warehouse_horizon.py --out /workspace/training/warehouse-horizon
uv run --frozen python scripts/archive_warehouse_horizon.py
uv run --frozen python scripts/validate_warehouse_horizon_gate.py \
  --run /workspace/training/warehouse-horizon \
  --out /workspace/training/warehouse-horizon-selector
uv run --frozen python scripts/archive_warehouse_horizon_gate.py
uv run --frozen python scripts/plot_warehouse_horizon.py
```

Completed controllers verify archived receipts and return without retraining.
For a new experiment, use new output directories; the run captures exact executed
sources and refuses to mix changed sources into unfinished work. The selector
is a separate post-holdout study, with its own source capsule and fresh audit.
Changes to the current implementation do not rewrite either historical capsule.

Results: [primary holdout](WAREHOUSE_HORIZON_RESULTS.md) and
[fresh-scene selector audit](WAREHOUSE_HORIZON_SELECTOR.md). The forecast plot
shows the minimum-to-maximum range across three trained seeds; the control
interval separately resamples model seeds and environments. Neither treats nine
crossed observations as nine independent fleets.
