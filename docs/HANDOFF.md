# WirelessJEV research handoff

The starting point for this continuation was GitHub `main` at `3c5ce3f` in
<https://github.com/Danielfoojunwei/SDTH-submission-PreceptualAI>.
The cloud development environment has Python 3.12, locked dependencies and a
CPU-only test/simulation workflow. See Git history and the associated review
branch for commit/push status; external publications are tracked separately.

## Route-aware horizon continuation

The subsequent "continue building" request adds the protocol in
`docs/WAREHOUSE_HORIZON.md`, with results in `WAREHOUSE_HORIZON_RESULTS.md`
and `WAREHOUSE_HORIZON_SELECTOR.md`. The six independently trained compact
checkpoints and exact executed sources are portable in
`artifacts/warehouse_horizon/`; original E1–E5/E7 and v2 evidence is untouched.
The CPU campaign completed on 31 October 2026 at 09:48 Singapore time; its
fresh-scene selector audit completed at 10:00. No training worker needs restarting.

`/workspace/training/warehouse-horizon` contains the full training campaign
(18 training, six validation, three held-out environments; three seeds per
architecture; 30 full-cross control rollouts). The selector at
`/workspace/training/warehouse-horizon-selector` contains 42 validation and
24 fresh audit rollouts. It was designed after the original holdout; its
policy was sealed before audit environments 143000–143002 were evaluated.
An earlier validation-only selector attempt in `warehouse-horizon-gate`
was stopped before audit access; `superseded.json` explains the whole-schedule
replacement. That failed attempt is not a completed experiment.

All six residuals fail the closed-loop checks. The proposal CLI therefore
uses numerical planning, including when a rejected residual is requested.
An explicit `--research-residual` option permits simulated-input research
only. Actual d3 inference interprets the declared controlled mission grammar;
caller-supplied contracts are labelled separately. No car command is sent.
The observation schema does not authenticate a self-reported hardware origin.

On the fresh audit, numerical mean service is 0.326 versus 0.237 for budgeted
strongest-signal roaming. The twelve-switch cap is a declared operator-policy
scenario, not the existing concurrent transaction limit or a physical rate
limit. Unconstrained roaming has higher service in the original holdout.
These results do not qualify all robots, actual hardware or SOTA performance.

## Warehouse v2 continuation

The user subsequently authorised all six warehouse research directions and
requested actual dependencies and execution. The v2 source, protocols and
reproduction commands are in `docs/WAREHOUSE_V2.md`; interpreted results are in
`docs/WAREHOUSE_V2_RESULTS.md`. This continuation uses the separate branch
`codex/warehouse-v2-research`, preserving the original review branch below.
The implementation adapts pinned ns-3 examples, executes Sionna, and runs real
d3-lite, Qwen3.5-0.8B and Chronos-Bolt weights. It does not reproduce ANSpect
proprietary code or use physical SmartCar measurements. Hardware requirements
are in `docs/WAREHOUSE_HARDWARE_DATA.md`.

Workspace runs are under `/workspace/training/warehouse-v2-{core,studies,factorized,generative}`.
Use their status/receipts, archived report hashes and exact source snapshots
before resuming anything; completed compact-model training should not restart.
The historical large-model experiments still have one training seed. Three
new compact-model seeds must not be reported as three original E1–E5 runs.

## Original completion scope

The user returned the scope to warehouse connectivity: finish WirelessJEV and
RadioJEV E1–E5 and corrected E7. E6 remains excluded. Router, E8, JEPA/world
models, general robotics, custom PHY and satellite work are deferred.
`docs/WAREHOUSE_RESULTS.md` is the consolidated completion/result record.

**Completed:** all nine E1–E5 controller stages, validation, report archiving,
Studio rebuild and review-branch push finished on 30 October 2026 at 20:16
Singapore time. The automated archive commit is `bc63d21`; no training worker
needs restarting. Corrected E7 is complete. Only the specified one-seed
synthetic run is complete; independent seeds and physical qualification remain
separate work.

## E7 environment learning

Two censoring defects were identified in `src/wirelessjev/envlearn.py`:

1. A full eight-result scan used `min(reports + [-90])` as the upper bound for
   every omitted BSSID. This always produced −90 dBm, even when a BSSID was
   omitted solely because eight stronger APs occupied the scan slots.
2. The fitter treated violated bounds as point observations at bound − 3 dB.
   These invented measurements gained influence as surveys grew.

Scans now rank noisy measurements before truncating to eight APs, distinguish
the scan-capacity bound from the detection floor, and fit a censored Gaussian
likelihood with the same measurement-noise model as the detected samples.
Inverse-sampling weights account for retaining non-reports every third scan;
otherwise detections would be overrepresented in the fit.
Regression tests cover the scan bounds, the likelihood against an independent
scalar solution, recovery of a known censored signal and numerical stability.

The historical `artifacts/envlearn/report.json` is retained unchanged. Its
audible-path RMSE was 4.98 / 6.19 / 7.90 dB for 1 / 5 / 15 minutes. Corrected
results belong in `artifacts/envlearn_corrected/report.json`; see README for
the measured results and reproduction command. Current inaudible-path RMSE
is reported separately so it cannot be confused with audible-path accuracy.
This is not a guarantee that each environment improves monotonically with
more driving, or that never-observed interference paths are well calibrated.
The truth planner is a local-search reference, not a proven global optimum.

The corrected full sweep completed: 15 environments, five missions each.
Audible RMSE is 4.04 / 3.95 / 3.99 dB at 1 / 5 / 15-minute budgets; plan gap
is 0.202 / 0.182 / 0.159. Five environments still worsen from 1 to 15 minutes,
so the result does not support a monotonic-more-data claim. Currently
inaudible-path RMSE is 15.03 dB for the 15-minute map (all paths: 12.23 dB).
Those paths are not identical to paths never measured during the survey.

A subsequent matched-survey, map-only calibration study on new seeds 15–29
combines censored global calibration with nominal rack geometry. Inaudible
RMSE falls from 18.14 / 16.10 / 15.71 dB to 14.86 / 13.78 / 14.23 dB at
1 / 5 / 15 minutes; all 15 environments improve on that subset. Audible error
is nearly unchanged. This is not a downstream planner result, so the default
planner remains v2. The experiment, raw evidence and paired confidence
intervals are in `docs/experiments/e7_calibration_study.py` and
`artifacts/envlearn_calibration/report.json`. Distant-link error remains large.

Validation includes the full corrected E7 sweep, the additional calibration
study and regression tests. The original E1 LoRA and E7 reports remain unchanged.

## RadioJEV E1–E5

- **Complete:** E1 `radio_lora`, in `artifacts/radiojev/e1_radio_lora.json`:
  held-out regret 0.0928 ± 0.0222 (200 cases), new-mission regret 0.0744 ± 0.0196
  (240 cases), about 3.35–3.99 s per decision on the original machine.
- The numeric-only E1 result is also retained in `e1_no_llm.json`. It has no
  new-mission evaluation. Its small known-mission regret is not evidence of
  language generalisation.
- Those files describe the historical Mac run. The separate cloud run has
  completed all E1 and E2 arms, including the previously interrupted frozen
  arm and paired intent baseline. E6 remains excluded.

The new paired E1 intent baseline has held-out / new-mission regret
0.0140 ± 0.0066 / 0.0251 ± 0.0074, versus LoRA's 0.1068 ± 0.0240 /
0.0721 ± 0.0136. E2 radio-Q reaches normalized regret 0.1170 ± 0.0370 /
0.0779 ± 0.0253 versus intent's 0.1987 ± 0.0461 / 0.3482 ± 0.0674.
Do not compare E1 regret with E2 normalized regret directly. These are
single-seed results, with confidence intervals over evaluation cases.
See `artifacts/radiojev_cloud/README.md` for the full tables and provenance.

The resume runner is portable Bash and resolves the checkout from its own path:

```bash
bash artifacts/radiojev/run_all.sh --dry-run
bash artifacts/radiojev/run_all.sh
```

It skips the completed LoRA arm and smoke chain, skips completed stage reports,
preserves checkpoints, waits for competing jobs and stops on the first error.
An optional positional argument selects the competing-job `pgrep` pattern.
Read logs under `artifacts/radiojev/`; logs and data caches remain ignored.

**Full rerun:** the user authorized all training, including
regenerating the missing LoRA checkpoint. The isolated run directory is
`/workspace/training/radiojev/full`; inspect its `status.json`, per-stage logs
and `manifest.json`. All 330 E1/E2/E4-E5 scenes are prepared. Completed stages
are tracked in `status.json` and the consolidated warehouse results. At the
scope-reset checkpoint, E1–E3 were complete and E4/E5's local arm had finished;
the remaining comparisons were running. Those comparisons have now completed.
The four E1–E3 radio checkpoints were loaded
and checked for finite tensors. Completed reports and hashes are archived in
`artifacts/radiojev_cloud/`.
The model downloaded and loaded successfully after network access changed;
no Hugging Face token is required for the public model. The worker has no GPU.
The old Mac estimate of 1–2 days does not estimate cloud CPU runtime.

The original controller stopped when E4/E5 passed an E3 `heads` checkpoint to
the E2 `encoder` warm-start loader. `Bank` now initialises the multitask model
without an E2 warm start when E3 exists, then restores all E3 heads, LoRA and
20 readout rows. Three regression cases cover E3 with/without an older E2
checkpoint and the E2-only fallback.
The failed attempt was preserved under the run directory's `attempts/` and
the remaining stage resumed with the versioned `scripts/train_radiojev.py`.
Its source hashes and predecessor checkpoint are recorded separately in
`artifacts/radiojev_cloud/e45_resume_provenance.json`.

Do not restart a healthy job. Future full runs can use the versioned
`scripts/train_radiojev.py --run-dir <new-or-stopped-run-directory>`.
Its lock prevents duplicate controllers. Completed scenes/stages are retained;
an interrupted arm restarts because optimizer/epoch continuation is not
implemented. Results in the new run directory do not overwrite the published
reports until explicitly validated/promoted.

The independent `scripts/finalize_radiojev.py --run-dir <run> --watch` validates
all nine completed controller stages, full evaluation counts, finite report
metrics, E4 signed logs, checkpoint tensors and recorded training-source hashes.
It preserves archived raw reports, generates `docs/WAREHOUSE_RESULTS.md`, saves
an exportable cloud snapshot and rebuilds Studio. It never starts or changes
training. With `--publish-branch codex/e7-censoring-training-progress`, it commits
and pushes only validated generated evidence on that existing branch, provided
the checkout is clean. It never merges. Inspect `finalization_status.json` and
`finalizer.log` under the run directory. A stopped worker or failed validation
is reported as failure rather than completion.

E3 association normalized regret improved from 0.1170 to 0.1046. DSA greedy
reward is 1.000 in one 500-step evaluation episode; MLP baselines reach
0.263 / 0.384. Traffic accuracy falls below the rule baselines on all three
questions. Keep these mixed results separate; one episode does not establish
cross-layout DSA robustness. The admission label audit found only "lower"
labels in all 106 training and 54 test cases; 100% accuracy matches a constant
answer and does not establish admission-policy generalisation. The audit is
retained in `artifacts/radiojev_cloud/e3_label_audit.json`, reproduced by
`docs/experiments/e3_label_audit.py`. A separately versioned admission dataset
with meaningful class coverage is needed before a learning claim.
Independent training seeds remain necessary for stronger paper claims.
The router design in `docs/ROUTER_EXPERIMENT.md` is deferred.

E4's local / FedAvg / personalised-swarm mean normalized regrets are
0.1820 / 0.1781 / 0.1820. All three shared merges were rejected, so the shared
model accepted no E4 site contributions. E5 cannot establish removal of an
accepted contribution from that shared model. Its retained-data distillation
also increased attack AUC (0.6167 → 0.6489) and retained regret
(0.1801 → 0.2068). Preserve the negative result; do not claim validated
unlearning. Full per-site comparisons and the retraining baseline are in
`docs/WAREHOUSE_RESULTS.md`.

## Studio and paper

The exporter and Evidence page now include E1 status and E7 reports, with
historical/corrected pipeline labels. Rebuild after completing any new run:

```bash
uv run --frozen python -m wirelessjev.studio_data
python3 studio/build_artifact.py
```

The distributable is `studio/dist/wirelessjev-studio.html` (ignored by Git).
Use `studio_data --run-dir /workspace/training/radiojev/full` to include the
active rerun's completed tests and epoch-validation snapshot. It keeps those
results separate from published benchmarks and does not claim live telemetry.
The published Studio at
<https://claude.ai/artifact/NiohoyLYJsy5vEKE5BBswC> needs a separate update by
someone with access to that artifact. This environment has no tool to update
that Claude artifact; rebuilding locally does not republish it. The old update
watch is not active. README and the paper proposal describe the current limits.

## Hardware and remaining research

- Run `docs/ONE_CAR_TEST.md` on actual hardware: preflight, read-only status,
  capability probe, scan, proposal, supervised switch and switch-back. Only
  fake-car checks have run. Change the factory SSH password on the car.
- Switch-stage timestamps (association/authentication/DHCP/first packet) remain
  proposed. No physical car, radio or credential was changed here.
- PHY realism with Sionna, route-aware rate/width/band/link adaptation, and E8
  sensed inputs remain proposals awaiting a separate implementation decision.
- Further Wi-Fi 7, joint motion/radio, cross-layer, safety/task-assignment,
  Bluetooth, spectrum-image and embodied-AI ideas remain unstarted. CSI/SDR/
  openwifi/5G work needs hardware. Multi-car trials need the other nine cars.
- The didjital R6.1 branch is restored at `/workspace/didjital-r6.1`, commit
  `a458a90`. Its root, reference and R6.1 software suites passed (3 + 67 + 117
  tests, no skips), plus a frozen synthetic demo and native gate. See
  `docs/R6_COMPATIBILITY.md`. A real runtime bridge, private SmartCar source
  audit and hardware integration remain outstanding.
- The earlier token-expiry estimate and `refs/backup/sdth/` were Mac-specific
  handoff notes, not verified cloud state. Recheck them on that machine. GitHub
  reads here work through the platform's existing authentication.
