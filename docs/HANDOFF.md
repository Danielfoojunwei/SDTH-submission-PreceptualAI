# WirelessJEV research handoff

The starting point for this continuation was GitHub `main` at `3c5ce3f` in
<https://github.com/Danielfoojunwei/SDTH-submission-PreceptualAI>.
The cloud development environment has Python 3.12, locked dependencies and a
CPU-only test/simulation workflow. See Git history and the associated review
branch for commit/push status; external publications are tracked separately.

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
- **Restart required:** E1 `radio_frozen`, stopped without a completed report.
- **Pending:** E1 `intent`; E2 `radio_q`, `e1_zero`, `intent`; E3 multitask;
  E4/E5 swarm and unlearning. E6 remains excluded.

The separate intent-reader + planner benchmark remains the stronger practical
result: held-out shortlist regret 0.027 and about 134 ms per re-plan. This is
not a paired E1 comparison; E1's intent baseline is still pending. Do not
claim that native radio inputs outperform intent reading plus exact planning.

The resume runner is portable Bash and resolves the checkout from its own path:

```bash
bash artifacts/radiojev/run_all.sh --dry-run
bash artifacts/radiojev/run_all.sh
```

It skips the completed LoRA arm and smoke chain, skips completed stage reports,
preserves checkpoints, waits for competing jobs and stops on the first error.
An optional positional argument selects the competing-job `pgrep` pattern.
Read logs under `artifacts/radiojev/`; logs and data caches remain ignored.

**Full rerun now active:** the user authorized all training, including
regenerating the missing LoRA checkpoint. The isolated run directory is
`/workspace/training/radiojev/full`; inspect its `status.json`, per-stage logs
and `manifest.json`. All 330 E1/E2/E4-E5 scenes are prepared, and the new E1
numeric-only arm completed. LoRA training is active; later stages are queued.
The model downloaded and loaded successfully after network access changed;
no Hugging Face token is required for the public model. The worker has no GPU.
The old Mac estimate of 1–2 days does not estimate cloud CPU runtime.

The existing job uses the original local controller and must not be restarted
just to adopt a new launcher. Future full runs can use the versioned
`scripts/train_radiojev.py --run-dir <new-or-stopped-run-directory>`.
Its lock prevents duplicate controllers. Completed scenes/stages are retained;
an interrupted arm restarts because optimizer/epoch continuation is not
implemented. Results in the new run directory do not overwrite the published
reports until explicitly validated/promoted.

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
