# JEPA runtime repair: diagnosis and validation scope

The repair changes forecast completion accounting and exact planned-outage
integration. Increasing an allowance alone repairs neither problem. Frozen
`semantic_jepa` and `selected_jepa` source, registrations, results and rejected
claims remain unchanged. This diagnosis opened no new scientific site or RF
world and performed no training or scientific retiming.

## Completed archive evidence

The 128 selected-confirmation rollouts and eight replays contain these actual
warm-request timing summaries:

| Family | Calls | Median | p95 | p99 | Maximum | Sealed compute allowance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RadioMap | 1,026 | 57.77 ms | 76.93 ms | 80.95 ms | **284.94 ms** | 120 ms |
| Context JEPA | 2,850 | 101.92 ms | 137.73 ms | 148.60 ms | 288.60 ms | 350 ms |
| Matched supervised | 2,850 | 102.05 ms | 137.47 ms | 149.91 ms | 234.96 ms | 350 ms |
| Original three-member FleetJEPA | 1,026 | 355.00 ms | 414.77 ms | 443.66 ms | 603.64 ms | 850 ms |

These descriptive request summaries are not independent-site confidence
intervals. The only overrun was RadioMap at site `2114005`, 0.03 s handover,
decision 10: `0.2849378939936287` s against a `0.12` s compute allowance. Its
forecast was scheduled at slot 1062 using the sealed 0.62 s service, including
the shared 0.50 s delay. The original report retains its global runtime veto.
Its packet results describe the registered envelope, rather than an
observed-wall-time deployment.

The cause remains unknown: no simultaneous CPU, scheduler, garbage-collection
or throttling data were recorded. Historical scientific workers explicitly
set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
Unpinned diagnostic timing cannot explain the historical spike.

Cold/refit/warm charges were 1.63/0.93/0.35 s for a new neural member,
1.46/1.51/0.85 s for the original ensemble, and 0.63/0.62/0.12 s for RadioMap,
all plus 0.50 s. The new neural member has a slower cold charge. Its 0.85 s
warm service fits a 1 s issuance interval, while the original ensemble's
1.35 s can accumulate FIFO delay. This package-level distinction does not
isolate the effect of the JEPA objective.

## Completion and controller repairs

The frozen predictor timer excludes `Observations.before()` copying its
growing history prefix, output storage, hashing and queue work. The frozen
FIFO uses `profile + extra`; an actual overrun is recorded afterward without
moving availability.

New `CompletionSafeForecasts` records predictor and history-snapshot-through-
output/metadata/hash elapsed time separately. It schedules:

```text
actual_compute = max(actual_prediction_seconds, actual_issue_seconds)
charged_compute = max(profile[kind], actual_compute)
service = charged_compute + shared_extra
ready = max(issue_time, predecessor_ready) + service
```

Slots round upward with the predecessor's floating-point epsilon. Different
sub-profile measured times retain deterministic slots. Invalid timings,
history axes, outputs or regimes fail closed. Complete records precede ready
publication. Validated consumption records identify slot, forecast decision,
ready slot and age only when a fresh forecast is returned.

In the **new prospectively registered contract**, a profile exceedance is a
diagnostic; actual-cost release must still cover measured work. Undercharged
availability, causal/integrity failure and malformed consumption remain
vetoes. This does not waive any old-study overrun veto.

`RecordedScheduleReplay` is explicitly **offline**. It validates the entire
original decision/cutoff/shape/hash/timing/FIFO ledger before predicting,
recomputes forecasts with `CompletionSafeForecasts`, and exposes original
availability. Fresh times and whether they exceed original charged compute
are separate fields. Original schedule replay tests reproducibility, not
fresh real-time coverage. New live-cost replications report timing/output
differences independently. Original source/archive binding belongs to the
registered runner.

The 0.03 s predecessor `OutageBudgetGate` starts candidate integration at the
1 s sample, omitting 0.97 s. New `ExactOutageAdmission` uses frozen
`OutageUtility`/`integral_from` at both costs. For constant failure `p` over
horizon `H`, `stay=H*p`, `move=cost+(H-cost)*p`, so
`move-stay=cost*(1-p)`. Identical APs receive no artificial switching gain.
The actual PHY-surrogate unit fixture reproduces and removes the old credit.

Wire admission as `roam_gate=admission`, `handover_admission=None`. The
unchanged native kernel then retains RSSI recovery after planned-move vetoes.
A separate handover gate can reject a move while leaving `confident` true,
suppressing recovery. This is a shared **controller-package change** for
every comparator, not solely a numerical integration fix or a JEPA improvement.
The PHY failure table remains a surrogate.

Controller/admission/scoring CPU is explicitly outside forecast issue timing
and is not individually scheduled. Startup is separate. The common 500 ms
extra delay remains an experimental scheduling/other-work assumption. No
physical-device timing, production safety or hard real-time guarantee follows.
Deployment requires real asynchronous completion callbacks, timeouts,
cancellation and measured dispatch/transport behavior.

## Bounded numerical kernel diagnostic

A scratch calculation used 100 cached query rows from already observed
validation site `2101000`, nine horizons and 16 public CAD APs/three bands.
Coordinates came from quantized cached features; the rows are not 100
independent robots. The validation NPZ hash matched its receipt:
`48f2d4e0558e812b994c8393374ba4c53967b9754740ec5d63e133ce83a02231`.
A fixed synthetic positive-definite numeric posterior exercised arithmetic
only. This was not a fitted radio map, RF accuracy result or scientific
benchmark. No private future labels were loaded and no RF world was created.

With all three thread limits set to one, the loaded NumPy BLAS function
`scipy_openblas_get_num_threads64_()` confirmed one thread. After three
warmups, 30 calls per path gave:

| Numeric fixture path | Wall median | Wall p95 | Wall max | CPU median |
| --- | ---: | ---: | ---: | ---: |
| Existing FastMap | 56.84 ms | 69.64 ms | 82.77 ms | 56.82 ms |
| Shared per-AP geometry across bands | 46.96 ms | 54.60 ms | 61.97 ms | 46.88 ms |

Float32 means/std were bitwise equal on that fixture; hashes:
`aace444735ff748d9418df3b504210dc4bd7e254c4a45f8f1299aa9e77b82a42` and
`6feb63564798fb7207e89b4f5a3f14d9e9aacf3ad1aba14175b3428ecf3d4e69`.
The median reduction was about 17%; one fixture establishes no multi-context
tail improvement. No caching/arithmetic optimization was committed. An earlier
unpinned scratch process used roughly four CPU seconds per wall second; its
timing is excluded from matched-environment conclusions and cannot explain
the old spike.

## Further work requires its own validation

- Share distance/log-distance/rack features per AP while retaining each
  band's Fourier features and reduction order. Validate fitted posteriors,
  scan densities and multiple contexts before adopting it.
- Reuse covariance workspace and bound batches. Current quadratic forms cost
  about 225 million FLOPs per 100-row request; each full float64 feature or
  product batch is about 17.6 MB. Diagonal covariance, float32, Cholesky or
  changed reductions need new uncertainty/control validation.
- Cache fixed EM right-hand-side terms and observed/censored features across
  holdout/final fits, retaining row order. Avoid subtracting held-out Gram
  matrices when exact numerical equivalence is required.
- Batch new-row fleet geometry, retain empty-row timestamp/expiry/denominator,
  cache six-row context and use integer last-heard times. Changed floating-
  point recurrences need equivalence checks. KD-tree caching does not help
  these runs: robot-ID holdout keeps `disc_far=0`.
- Before a new scientific run, freeze the new namespace, controller wiring,
  model set, timing rule, fresh sites and gates. Calibrate full request work
  on declared already observed cold/warm/refit, missing/dense-scan, history-
  length and stationary/drift contexts with actual worker/thread settings.
  Retain all wall/CPU samples, startup, affinity/quota and throttling deltas.
- Report every comparator, actual undercharging, profile diagnostics,
  consumption/queue age, unconsumed jobs, fallback use and live replications.
  A sampled maximum is not a worst-case bound; bounded map corrections do not
  guarantee bounded cumulative packet loss, production safety or SOTA.
