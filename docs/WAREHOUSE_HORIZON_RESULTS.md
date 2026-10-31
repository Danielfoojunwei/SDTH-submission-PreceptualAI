# Route-aware horizon results

This separately versioned campaign forecasts six five-second outcomes across each thirty-second commitment. Inputs are past censored scans, previous delivery telemetry, public AP metadata/CAD and committed routes; actual future positions and hidden propagation remain evaluation-only.

Training uses 18 ten-minute environments, validation six, and held-out evaluation three. Both compact architectures have three independently trained seeds; each is evaluated on all three held-out environments. The original E1–E5/E7 and warehouse v2 reports are unchanged.

The mission changes from fleet data to worst-decile service halfway through each ten-minute rollout. Both include an explicit critical-rate-violation penalty and a hard twelve-switch budget. d3 interprets the two mission families with actual pinned weights; numeric fields are compiled from the declared controlled language. These two statements do not establish broad language understanding.

| Controller | Runs | Mission utility ↑ | Service fraction ↑ | Worst decile ↑ | Critical-violation robot-steps ↓ | Reconnections ↓ | Budget violations | Mean p95 compute (ms) |
|---|---|---|---|---|---|---|---|---|
| stay | 3 | -0.7368 | 0.1490 | 0.0263 | 9780.0 | 0.0 | 0 | 0.1 |
| budgeted_strongest | 3 | -0.6423 | 0.2024 | 0.0713 | 8971.0 | 240.0 | 0 | 0.2 |
| unconstrained_strongest | 3 | -0.1107 | 0.5184 | 0.3768 | 4478.7 | 1549.7 | 60 | 0.2 |
| numerical_planner | 3 | -0.4387 | 0.3070 | 0.1133 | 7652.0 | 239.3 | 0 | 143.8 |
| residual_mlp | 9 | -0.4352 | 0.3057 | 0.0916 | 7655.0 | 239.2 | 0 | 150.6 |
| residual_graph | 9 | -0.4408 | 0.2994 | 0.0981 | 7687.1 | 239.7 | 0 | 156.5 |

The twelve-switch cap is an operator-policy scenario, not a hardware rate limit or the existing maximum concurrent transaction count. Batched execution could permit more switches across thirty seconds. The unconstrained strongest-scan arm violates this declared budget and has higher service in these scenes. Primary comparisons therefore concern budgeted missions only. Every planner uses the same candidate construction and local-search round limit, with independently validated visible targets and move counts.

## Paired learned-minus-numerical differences

- residual_mlp, mean_mission_utility: 0.0035, crossed-bootstrap 95% interval [-0.0065, 0.0121].
- residual_mlp, critical_violation_robot_steps: 3.0000, crossed-bootstrap 95% interval [-70.0000, 74.0000].
- residual_mlp, worst_decile_mean_service: -0.0217, crossed-bootstrap 95% interval [-0.0281, -0.0170].
- residual_graph, mean_mission_utility: -0.0021, crossed-bootstrap 95% interval [-0.0203, 0.0155].
- residual_graph, critical_violation_robot_steps: 35.1111, crossed-bootstrap 95% interval [-155.3333, 184.3333].
- residual_graph, worst_decile_mean_service: -0.0152, crossed-bootstrap 95% interval [-0.0246, -0.0046].

The bootstrap resamples model seeds and environments separately. Three environments remain a small evaluation set. Intervals do not establish physical-fleet reliability or universal benefit.

## Individual simulated robots

Against budgeted strongest-scan, descriptive comparisons count a change only when mean service differs by more than one percentage point. Each row is one environment and one model seed, not another independent physical fleet. Some robots can regress even when the fleet objective improves. Full counts and worst examples are in `snapshot.json`.

## Validation-only selection

- Model 0-False: residual strength 0.5, risk temperature 1.327.
- Model 0-True: residual strength 0.5, risk temperature 1.335.
- Model 1-False: residual strength 1.0, risk temperature 0.972.
- Model 1-True: residual strength 0.5, risk temperature 1.313.
- Model 2-False: residual strength 0.5, risk temperature 1.324.
- Model 2-True: residual strength 0.5, risk temperature 1.309.

A zero residual strength means the numerical baseline was selected; it is not a learned advantage. Raw learned forecasts and validated blends are both retained in `offline.json`. Selection was sealed before accessing test outcomes. Checkpoint epoch selection includes the zero-residual initial state as a declared candidate.

## Limits

All radio observations, delivery counters and actions are simulated. The NIST payload-error curves do not constitute hardware calibration or a packet-level warehouse validation. Four seconds of switch interruption is assumed and charged only across affected intervals; scan interruption and actuation failures remain unmodelled. Critical-rate violations are not application packet deadlines. Local compute timing includes belief construction and planning, but excludes physical sensing, first-use language inference and actuation.

There is no physical-car, ANSpect-waveform, general foundation-model or SOTA claim. Exact source capsules, report hashes and finite checkpoint hashes are archived with the results.

The separately versioned post-holdout selector and fresh-environment audit are documented in [WAREHOUSE_HORIZON_SELECTOR.md](WAREHOUSE_HORIZON_SELECTOR.md). They preserve these original outcomes. Software and actual proposal-interface execution receipts are recorded in `artifacts/warehouse_horizon/verification.json`.
