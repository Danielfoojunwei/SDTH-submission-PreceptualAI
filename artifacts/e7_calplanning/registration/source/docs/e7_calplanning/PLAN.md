# E7 calibrated-map planning diagnostic

This finite diagnostic asks whether the fixed geometry-and-censor calibration changes the original E7 analytical planner's decisions and truth-scored mission utility. It compares `corrected_v2` with `CalibratedMap(geometry=True,censored=True)` on the same retained reports, action sets, starts and search budgets. It preserves all previous reports and defaults.

The result is **ORACLE-CONDITIONED, DESCRIPTIVE**. Environment seeds 15–29 were already examined in the optional calibration study. This is a downstream endpoint on opened environments, not fresh confirmation or evidence of a deployable observation-only system. It does not open the original SERVICE holdouts, run packet or ns-3 simulation, establish an SLA, or actuate hardware.

## Data and separation

Each registered environment regenerates its original single 15-minute survey (180 five-second simulator steps). The three nominal 1/5/15-minute settings reproduce the legacy **nested flattened tail-row fractions**, including the same fractions of RSSI, censored nonreports, MCS-derived link floors and busy reports. These are not chronological prefixes: the old `Survey` has no report timestamps and row rates vary. The nominal setting is not a measured elapsed acquisition-time guarantee. Legacy zero-count slicing behavior is retained.

The fitter receives a small immutable whitelist: AP XY, band, transmit power, assumed-known AP height, nominal warehouse rack geometry, and reported RSSI/censor bounds. It does not receive the hidden environment, true RF values, walls, materials or propagation coefficients. The exact dynamically imported implementation in `docs/experiments/e7_calibration_study.py` is included in the new source capsule alongside every frozen reused source. Both maps use D=200 random features, length 8 m, prior standard deviation 8 dB, observation noise 4 dB, the same per-environment random-feature seed, and kappa=0. The calibrated arm fixes both geometry and censored calibration on; no variant selection follows these planner results.

Planning is explicitly conditioned on identical truth-derived top-eight visibility masks, contention domains/interferer graph, truth-configured adjacent-channel leakage matrix, protected-channel flags, current associations and a shared truth-derived legacy-policy start. Other shared inputs are public planned robot positions, known flow demands and protocol rate tables. Those shared oracle fields are saved separately from the fit whitelist. Their admission prevents claiming a fully observed-only planning experiment.

The two maps independently infer external-interference estimates from the identical measured MCS-floor reports and infer availability from identical channel-busy reports using the existing E7 calculation. They use the same original analytical `BatchEvaluator` and `FastPlanner`. True current/future UL/DL, hidden interference and future forklift blockage are stored in a separate scoring archive. All belief-selected actions and search statistics are written before the scoring reader evaluates them. True RF never selects the calibration variant or map coefficients.

The retained map checkpoint contains the actual fitted mean coefficients, random features, distance-law parameters and rack factors. Covariance matrices are not copied: this fixed endpoint uses kappa=0 and consumes means only. The checkpoint makes no posterior-calibration or uncertainty claim. The passive verifier reconstructs every mean and planner input from those coefficients, public inputs and reports, then replays all reported utility arithmetic without fitting or searching.

## Fixed endpoint and compute

The five original warehouse missions are `teleop_assist`, `peak_picking`, `map_rollout`, `shift_fairness` and `scanner_coexistence`. The utility uses the original analytical airtime/rate/interference calculation, a 30-second decision horizon, snapshots at now/+10/+20 seconds and the original four-second handover charge. Each search starts from current and shared legacy plans, uses NumPy float64, `screen_iters=1`, multi-move search, and at most 40 iterations.

There are 15 environments × 3 nominal tail budgets × 2 maps = **90 fits**; 450 belief searches and 75 shared truth-informed reference searches = **525 searches**, at most 21,000 search iterations. For 100 robots with at most eight visible choices, the frozen search accounts for at most 3+40×806=32,243 evaluated plan rows per search, hence at most 16,927,575 across the diagnostic. Actual fit/search counts and timing statistics are retained. Execution uses at most two processes with each numerical thread limit set to one.

The truth-informed search is a bounded local-search reference, not an exact/global optimum. Its signed gap can be negative if an arm finds a better truth-scored plan; negative values remain unchanged. The primary endpoint is each environment's mean over the five paired mission utilities, `geometry_censored − corrected_v2`, separately for each nominal budget. All 225 mission/budget pairs and every harm remain visible, nested under **15 environment units**. No significance claim treats missions or robots as independent samples. Map RMSE on audible, inaudible and all current links is secondary, computed only after decisions are saved.

## Execution and audit

Before scientific generation, fitting or planning, `write-spec`, software-only tests, `preflight`, and `register` create an immutable source capsule. Root then commits and pushes it; `publish` verifies the remote commit and exact Git archive. Every scientific entry point checks these bindings and all preserved native source/evidence bytes. A durable exclusive attempt is created before each environment is generated. A second exclusive, canonical generation marker consumes that attempt before the first generator call, so calling the input seam directly cannot repeat generation. Any interrupted, nonempty case without its completed receipt is refused; no automatic repeat extends the budget. Completed cases may be read only after their exact canonical membership, source/protocol identity, reachable visible-or-start actions and arithmetic pass verification. The final closure binds every retained evidence file and rejects extra cases, extra outputs or changed bytes; its replay never generates, fits or searches.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run --frozen python -m wirelessjev.e7_calplanning.study preflight
uv run --frozen python -m wirelessjev.e7_calplanning.study register
# Root commits/pushes this exact capsule before publishing its proof.
uv run --frozen python -m wirelessjev.e7_calplanning.study publish --commit COMMIT --branch BRANCH
# Scientific execution occurs only after the separate root GO.
uv run --frozen python -m wirelessjev.e7_calplanning.study run --workers 2
uv run --frozen python -m wirelessjev.e7_calplanning.study audit
```

Software tests use explicit tiny analytic fixtures only. They never generate the registered fifteen environments or run this diagnostic. All old reports, production defaults, original SERVICE holdouts and hardware remain unchanged.
