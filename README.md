# WirelessJEV

Source: [Danielfoojunwei/SDTH-submission-PreceptualAI](https://github.com/Danielfoojunwei/SDTH-submission-PreceptualAI).
Current experiment status and restart prerequisites are in [docs/HANDOFF.md](docs/HANDOFF.md).
The restored didjital R6.1 reference and its executed software checks are
documented in [docs/R6_COMPATIBILITY.md](docs/R6_COMPATIBILITY.md).

The active milestone is **warehouse connectivity: WirelessJEV/RadioJEV E1–E5
and corrected E7**. E6 remains excluded. Router, E8, JEPA/world-model,
general-robotics, custom-PHY and satellite extensions are deferred. See the
consolidated [warehouse results](docs/WAREHOUSE_RESULTS.md) for completion status.

Mission-conditioned Wi-Fi coordination for a 10-car Raspbot fleet. A frozen open System One model chooses the fleet plan. Exact numerical oracles teach and score it. A swarm protocol shares what the cars learn.

Milestone 1 runs entirely in a digital twin. Nothing here touches a car, a radio or a network.

## Hardware it targets (from SmartCar.zip)
Every car is the same hardware as the existing SmartCar (`src/wirelessjev/hardware.py`):
- **Platform:** a Yahboom Raspbot on a Raspberry Pi.
- **Management radio:** the onboard brcmfmac SDIO radio (`wlan0`) runs the management hotspot at 192.168.1.11. WirelessJEV never scans, selects or modifies it; SmartCarGuard owns its recovery.
- **Customer radio:** an Intel BE200 on PCIe (`8086:272b`, `wlan1`). It is the only radio acted on, and the only action is choosing a BSSID.
- **Wire format:** exactly the existing `SmartCarCmd.py` JSON: `status`, and `connect` with ssid/bssid/security plus the Wi-Fi password, which SmartCarCmd requires, just as the PC app sends it. Plans, receipts and logs never contain the password; the real transport (`carlink.py`) injects it from the operator's WiFi.json at send time.
- **Measured anchors:**
  - about 4 s per BSSID switch;
  - 5 s policy interval;
  - -80 dBm legacy reconnect threshold;
  - VHT 80 MHz with 2 spatial streams;
  - at most 8 visible APs.
- Customer APs are not ours, so AP channel planning is modelled but disabled.

## Architecture
1. **Numeric layer**: digital twin (`twin/`), exact evaluator and paper reward families (`oracle.py`), teachers (`teachers/`), and the candidate generator with hard filters (`candidates.py`). It produces up to 7 complete 10-car plans plus DEFER, each described by its consequences.
2. **System One layer**: frozen `vllm-sr/d3-lite@03560c2` (Apache-2.0) in `policy.py`. It picks a plan from the mission text. In the default mode, each option is scored separately by a yes/no question.
3. **Distillation**: one 1024-parameter readout direction (backbone frozen), distilled from the exact oracle (`distill.py`). No language-model teacher is used.
4. **Swarm**: validated merge with a signed log, plus reputation/quarantine and two-tier priors (`swarm/`).
5. **Traffic**: d3-lite typed labels plus frozen Chronos-Bolt forecasts (`traffic.py`).

How each Bowen Shen / Feng Li / Kwok-Yan Lam paper is used is in [docs/PAPER_MAP.md](docs/PAPER_MAP.md).

## 100-robot AGV/AMR warehouse
`twin/warehouse.py`:
- 120 × 80 m racked hall, with rack loss per row crossed;
- 16 dual-band ceiling AP sites on a fixed customer channel plan;
- scanner-occupied channels;
- 100 route-planned AMRs on the same SmartCar hardware.

Plans are scored over the robots' planned positions for the next 20 s.

```bash
uv run --frozen python -m wirelessjev.bench_warehouse --scenes 50 --out artifacts/warehouse
```
```bash
uv run --frozen python -m wirelessjev.closedloop --warehouse --total 600 --adapter artifacts/warehouse/adapters/warehouse --out artifacts/closedloop_warehouse.json
```

## Intent-to-objective WirelessJEV (fast loop)
d3-lite reads the mission text into objective weights through typed score questions (`intent.py`). This happens once per mission change and is cached; the score readout is distilled.

An exact batched planner (`fastplan.py`) then optimises those weights every step. It scores every robot × BSSID move in one batch (numpy or torch, CPU or GPU) and accepts only exactly verified moves.

Results on the 100-robot warehouse:
- **Plan gap to the best known plan:** 0.016, versus 0.179 for the previous option-wise choice.
- **Held-out paraphrase shortlist regret:** 0.027.
- **Decision latency:** about 134 ms, versus about 15 s before.
```bash
uv run --frozen python -m wirelessjev.bench_intent --out artifacts/intent
```
```bash
uv run --frozen python -m wirelessjev.closedloop --warehouse --policies wirelessjev_fast --intent-adapter artifacts/intent/adapters/intent --out artifacts/closedloop_warehouse_fast.json
```

## Development setup

Use Python 3.12 and `uv sync --frozen` to install the locked dependencies.
The test suite and digital-twin baseline simulations run on CPU without model
downloads or physical cars:

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --frozen pytest -q tests
node --test tests/studio_results.test.mjs
```

Model benchmarks additionally download the pinned d3-lite runtime/weights and,
where used, Chronos-Bolt. Their original Mac latencies do not describe this
CPU cloud worker. Real-car commands require separate hardware access.

## RadioJEV E1–E5 status

The full cloud run covers all E1/E2 arms, E3 multitask, and E4/E5 site adaptation
and forgetting. Its validated completion status and complete comparisons are
recorded in [docs/WAREHOUSE_RESULTS.md](docs/WAREHOUSE_RESULTS.md). Raw
completed reports, source provenance and checkpoint hashes are retained in
[artifacts/radiojev_cloud](artifacts/radiojev_cloud/README.md).

| E1 arm | Held-out regret ± 95% CI (n=200) | New-mission regret ± 95% CI (n=240) | CPU latency |
|---|---|---|---|
| Native-radio LoRA | 0.107 ± 0.024 | 0.072 ± 0.014 | 4.31 s |
| Native-radio frozen | 0.107 ± 0.023 | 0.084 ± 0.019 | 4.62 s |
| Intent + planner | 0.014 ± 0.007 | 0.025 ± 0.007 | 468 ms |
| Numeric encoder | 0.00042 ± 0.00018 | Not evaluated | 2.15 ms |

Intent + planning has lower observed regret than both native-radio arms in
this paired E1 evaluation. The numeric-only arm has no new-mission evaluation.
E2 radio-Q reaches normalized regret **0.117 ± 0.037** on held-out missions
and **0.078 ± 0.025** on new missions (120 cases each), versus intent's
0.199 ± 0.046 / 0.348 ± 0.067 and E1 zero-shot's 0.291 ± 0.059 /
0.329 ± 0.069. E2 uses a different metric from E1; its CPU decision latency
is 6.42 s for radio-Q versus 591 ms for intent. This is one training seed,
with intervals over evaluation cases rather than independent training runs.

E3 reduces association normalized regret from 0.1170 to 0.1046. Its DSA
greedy reward is 1.000 over one 500-step evaluation episode, versus 0.263 /
0.384 for the 1,500 / 20,000-step MLP baselines. Traffic accuracy is
0.369 / 0.231 / 0.338 (trend / event / volatility), below the corresponding
rule baselines of 0.708 / 0.738 / 0.923. This is mixed multitask evidence,
not a claim of general improvement across tasks. Admission's 100% accuracy
matches a constant prediction: all 106 training and 54 test labels are "lower".

E2 tests the existing weight-based reader, not a planner with explicit mission
constraints. Its mission implementations mix thresholds, soft penalties and
spatial predicates. The separate router proposal is deferred while the
specified warehouse runs are completed.

The original Mac LoRA result (0.093 ± 0.022 held-out, 0.074 ± 0.020 new-mission,
3.35–3.99 s per decision) remains unchanged in `artifacts/radiojev/`.
That directory's runner can resume the original run once its missing checkpoint
is restored. Preview it without starting model work:

```bash
bash artifacts/radiojev/run_all.sh --dry-run
```

The runner skips the completed LoRA arm and smoke tests, preserves checkpoints
and stops on failures. Before the full run, restore the completed
`artifacts/radiojev/ckpt_radio_lora.pt` from the original Mac: E2 requires it,
and Git contains its report but not its weights. See the handoff for the model
download and compute prerequisites.

For a complete fresh run, including regeneration of missing LoRA weights, use
the portable controller with a separate output directory:

```bash
uv run --frozen python scripts/train_radiojev.py --run-dir /workspace/training/radiojev/full --plan
uv run --frozen python scripts/train_radiojev.py --run-dir /workspace/training/radiojev/full
```

It caches each prepared scene, retains completed stages and prevents duplicate
controllers. The cloud run resumed E4/E5 under the versioned controller after
the original launcher stopped; do not launch a second controller against it.
An interrupted arm restarts its training; optimizer/epoch continuation is not
implemented.

To finish report validation, archive all completed stages and rebuild Studio
when the active controller completes, run the independent finalizer:

```bash
uv run --frozen python scripts/finalize_radiojev.py --run-dir /workspace/training/radiojev/full --watch
```

It never changes training or treats partial logs as final results. The optional
`--publish-branch <existing-review-branch>` commits and pushes generated evidence
only after validation, and refuses a changed working tree. It does not merge.
Its live progress is `finalization_status.json` in the run directory.

## E7 environment learning

The original report had an accuracy regression: audible-path RMSE rose from
4.98 dB after one minute to 7.90 dB after fifteen minutes. The corrected scan
uses the weakest reported signal as the censoring bound when eight scan slots
are full, and −90 dBm only for the detection floor. The fitter now uses a
censored Gaussian likelihood instead of invented point measurements, with
inverse-sampling weights for thinned non-reports.

The original evidence is preserved in `artifacts/envlearn/report.json`.
Corrected results are written separately to `artifacts/envlearn_corrected/`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python -m wirelessjev.envlearn --envs 15 --workers 4 --out artifacts/envlearn_corrected
```

The report separates audible, inaudible and all-path RMSE. Map accuracy and
plan gap are distinct metrics; additional driving need not help every
environment. Weakly observed distant paths remain a limitation, and the truth
planner uses local search rather than a proven optimum.

Completed corrected sweep: **15 environments × 5 missions**, using the same
environment seeds as the historical report. Mean per-environment audible RMSE:

| Survey budget | Historical pipeline | Corrected pipeline | Corrected plan gap |
|---|---:|---:|---:|
| 1 minute | 4.98 dB | 4.04 dB | 0.202 |
| 5 minutes | 6.19 dB | 3.95 dB | 0.182 |
| 15 minutes | 7.90 dB | 3.99 dB | 0.159 |

The large historical regression is removed, but the corrected sweep **does
not establish monotonic improvement with more data**: 5 of 15 environments
still have higher audible RMSE at 15 minutes than at 1 minute. The 15-minute
model's currently inaudible-path RMSE is **15.03 dB**, and all-path RMSE is
**12.23 dB**. Currently inaudible paths (truth RSSI < −90 dBm) comprise 63.3%
of evaluated links; this is not a measurement of the fraction never heard
during the survey. The 5-minute pessimistic planner's plan gap is 0.144;
its RMSE includes its uncertainty margin and is not the posterior-mean error.

A separate **map-only calibration study** uses 15 new environment seeds
(15–29), comparing matched surveys at every budget. Censored global calibration
with nominal rack geometry lowers inaudible-path RMSE from **18.14 / 16.10 /
15.71 dB** to **14.86 / 13.78 / 14.23 dB** at 1 / 5 / 15 minutes. At 15 minutes,
audible-path error is essentially unchanged (3.96 → 3.94 dB). The inaudible-path
error improves in all 15 environments, but remains large. This study does not
validate downstream plan quality or real-site performance, so the default
planner is unchanged. Full results and exploratory paired confidence intervals
are in `artifacts/envlearn_calibration/report.json`.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python docs/experiments/e7_calibration_study.py --out artifacts/new_calibration_study
```

## Studio (front end)
A static app in `studio/` that shows only the exported artifacts, including E1 status and E7's corrected and historical evidence. Its sections are overview, fleet map replay, decisions, evidence, swarm & trust, execution, research map, and hardware & data.
```bash
uv run --frozen python -m wirelessjev.studio_data
```
```bash
python3 -m http.server 8766 --bind 127.0.0.1 --directory studio
```
On a local machine, open port 8766 in a browser. Cloud onboarding validates
the server with local HTTP requests. `python3 studio/build_artifact.py` builds
`studio/dist/wirelessjev-studio.html`; updating the external Claude artifact
requires separate access and publication.

To include the active run's completed results and epoch-validation snapshot:

```bash
uv run --frozen python -m wirelessjev.studio_data --run-dir /workspace/training/radiojev/full
```

This keeps published benchmark results separate and marks stopped processes
when exported on the training machine.

## Run
```bash
uv run --frozen pytest -q tests
```
```bash
uv run --frozen python -m wirelessjev.bench --scenes 80 --out artifacts/bench
```
```bash
uv run --frozen python -m wirelessjev.closedloop --adapter artifacts/bench/adapters/fleet
```
```bash
uv run --frozen python -m wirelessjev.report
```

## Boundaries
- No motor, camera or navigation control: SmartCar.zip ships no such interface.
- No OCUDU, 5G NR or custom-PHY execution. Those need Linux plus SDR hardware.
- d3-lite takes about 1–3 s per decision on the M5 GPU. That suits the 5 s policy interval. Anything sub-10 ms stays conventional.
- Real-car tooling: `python -m wirelessjev.onecar` (see [docs/ONE_CAR_TEST.md](docs/ONE_CAR_TEST.md)) does read-only status/scan and one operator-confirmed BSSID switch, with the host key pinned. The fleet `SSHTransport` in `execution.py` still refuses to send.
