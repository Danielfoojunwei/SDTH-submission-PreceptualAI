# WirelessJEV

Source: [Danielfoojunwei/SDTH-submission-PreceptualAI](https://github.com/Danielfoojunwei/SDTH-submission-PreceptualAI).
Current experiment status and restart prerequisites are in [docs/HANDOFF.md](docs/HANDOFF.md).
The restored didjital R6.1 reference and its executed software checks are
documented in [docs/R6_COMPATIBILITY.md](docs/R6_COMPATIBILITY.md).

## Current system and evidence

[RadioJEPA supplies an experimental physical RF-field model; WirelessJEV supplies
typed mission decisions, bounded planning and an independent gate](docs/research_status/CURRENT_SYSTEM.md).
Public geometry, reported RSSI/censor bounds and declared receiver assumptions
produce per-link predictions. Supported proposals retain explicit timing,
compute and evidence limits. **RadioMap + WirelessJEV remains the application
default.**

For a real car, use the [portable SmartCar deployment workflow](docs/SMARTCAR_NUMPY_DEPLOYMENT.md).
It separates a stdlib Pi collector from desktop numerical mapping, pinned
RadioJEPA shadow inference and a separately confirmed, journaled switch.
The [Torch-free package](artifacts/smartcar_numpy/README.md) runs all three
released models with NumPy/SciPy; the registered CPU-only comparison measures
1.92–3.98× faster conditioning/rendering and 4.30–4.65× lower peak process RSS,
while preserving their predictions. These are fixed software workloads on
the cloud CPU, with no real-car or RF-accuracy improvement claim.
It requires actual position-tagged scans and verified management access;
no physical-car or SOTA qualification is claimed.

The [zero-capable spatial-residual study](docs/nested_radio/RESULTS.md) completes
**15 fits, 28,800 committed updates and 84 fresh Sionna RT cases**. On 24 TEST
layouts, validation-selected nested joint has sparse heard-RSSI RMSE
**2.637330 versus 2.710522 dB** for the strongest validation-selected numerical
control (fixed hierarchy): **2.70% lower**. All four registered comparisons have
negative family-adjusted intervals. Shrinkage improves matched hierarchy
regression by **0.188225 dB**; joint-view agreement adds only **0.002226 dB**.
These are static RF predictions, with no new warehouse packet runs.

The selected models prune **all eight learned local feature channels**; the
retained residual uses fixed Fourier features around shared physics. These
results support the tested shrinkage package, without establishing rich learned
local representations or practical service value from the tiny joint gain.
Material-change error essentially ties and slightly favors the numerical
control. The **578-source / 2,770-evidence** seal and
[first independent replay](artifacts/nested_radio_review/independent.json) pass.
The [interactive 3D viewer](artifacts/nested_radio_review/viewer/index.html)
shows every TEST layout, comparison, training checkpoint and effective gate.

The separate [shared/AP-local warehouse study](docs/hierarchical_radio/RESULTS.md)
retains all 186 RT cases, 15 earlier fits, 34 simulations and its negative field
and full-history-map comparisons. Its fixed weights differ from the new RF
model. The new field gain does not reverse those service results or qualify
physical-device transfer. Learned dynamics and production validation remain open.

The [causal warehouse-control study](docs/research_status/SPATIAL_CONTROL.md)
completes **48 fresh 100-robot simulations plus two exact parity runs**. It adds
spatial report memory, a latest-own support guard, delivered central-command
execution and separately preinstalled client-local recovery with paid scans.
With 30 ms handovers, support + recovery reduces the fixed regression model's
mean command loss from **1.635625% to 1.079375%** (34.0% fewer), while RadioMap
remains better at **1.044375%** in all four episode pairs. Mean episode maximum
silence falls from **5.8175 to 0.355 seconds**. At four-second switching cost,
recovery harms **all four episodes** for both forecasters. The 543-source /
224-evidence closure and [independent passive audit](artifacts/spatial_control_review/result.json)
pass. These gains repair control-stack behavior; they do not establish a JEPA
objective advantage, learned dynamics or production qualification.
The [local result explorer](artifacts/spatial_control_review/index.html) shows
all six arms, both handover costs, paired harms and the two control paths.

The current [adaptive RF-condition assay](docs/research_status/ADAPTIVE_RESULTS.md)
completes **12 fits and 174 actual Sionna RT cases**. Across 24 TEST layouts,
validation-selected regression has sparse condition-delta RMSE **1.520106 versus
1.654205 dB** for RadioMap (**8.1% lower**); both registered RF-dependence
contrasts pass their adjusted intervals. Matched joint/JEPA is worse at
**1.534382 dB**, and pooled Bayes nearly ties at **1.524141 dB**. Detector
Brier improvement is unconfirmed. The canonical raw replays pass all 540 DEV and
3,240 TEST records; the completed seal binds **504 source files and 11,792
evidence files**, with a passing [independent raw replay](artifacts/adaptive_radio_review/result.json). These are static RF-state results, without packet, real-device or
SOTA qualification.

The [native 100-robot diagnostic](docs/research_status/NATIVE_RESULTS.md) completes
18 simulations and retains its adverse fast-handover outcomes and 9.74-second
recovery outlier. The [completed E7 calibration-to-planner diagnostic](docs/research_status/E7_CALPLANNING_RESULTS.md)
has better map accuracy but mixed analytical utility changes:
**+0.03148 / −0.01072 / −0.00233** across nominal 1/5/15-minute tail-row settings.
Its 225 pairs are descriptive, oracle-conditioned results on opened environments.
[Measured CSI](docs/research_status/CSI_RESULTS.md) also retains its negative
model result. None of these results qualifies a fleet controller.

The latest [tracked-suite proof](artifacts/smartcar_numpy_review/software_suite.json)
records **4,138 passing tests**, 165 test modules, one expected fork warning
and unchanged hashes for 606 source/config/test files. The preceding
[3,633-test RF snapshot](artifacts/nested_radio_review/software_suite_importlib.json) and the earlier
[2,978-test snapshot](artifacts/spatial_control_review/final_software_suite.json) and
[2,655-test snapshot](artifacts/research_review/release_suite.json) are preserved;
overlapping R6, exporter and spatial preflight counts are not added again.
The [read-only R6 bridge](docs/research_status/R6_BRIDGE.md),
[local evidence Studio](docs/research_status/STUDIO_EXPORT.md),
[execution ledger](docs/research_status/EXECUTION_LEDGER.md) and
[remaining-work matrix](docs/research_status/REMAINING.md) distinguish implemented
software, completed experiments and open qualification work. The external
Studio/deck has not been republished, and original SERVICE layouts remain unopened.

## Additional registered studies

The [static physical-field study](docs/spectral_scene/RESULTS.md) completes 15
fits and 84 Sionna scenes. Its 24-layout spatial RF RMSE is 3.622 versus
RadioMap-operator 4.843 dB, with no confirmed matched-regression/JEPA advantage.
Its [architecture and scope](docs/spectral_scene/ARCHITECTURE.md) and frozen
source-bound reports remain separate from the current adaptive assay.

The [pilot/local-control study](docs/PILOT_FASTLOOP_RESULTS.md) completes
**96 native executions and 22 matched offline forecast sequences**. On eight
reserved 100-robot simulated sites, the preselected combined JEPA package has
**44,108 versus 46,093 missed commands** for equally augmented RadioMap:
**4.594583% versus 4.801354% loss**, or **4.31% fewer misses**. Registered
observed/pending bounds and relative engineering gates pass. On identical
forecast inputs, JEPA's 1–5 s route-change RMSE is **2.241 versus 2.319 dB**,
**3.34% lower**, with a separate four-test-adjusted accuracy pass.

**Both combined-versus-base comparisons fail qualification.** Component means
do not establish that the pilot/fast interventions improve either model;
JEPA's own comparison fails worst-decile and silence margins. Matched AP-ranking
regret also worsens. JEPA uses about **2.81× more summed forecast work**, and
local-helper wall-time outliers prevent a real-time claim. The
[raw audit](artifacts/pilot_fastloop/verification/raw_audit.json) reproduces all
runs, replays, metrics and gates. Neural weights stay fixed; causal map fitting
continues. **1,861 core tests and 50 separate guard-cache fixtures pass.**
RadioMap remains the default; hardware validation is still pending.

The preceding [JEPA enabling-stack study](docs/FRESHNESS_STACK_RESULTS.md) completes
**80 main evaluations, eight offline replays and eight fresh-cost repetitions**,
each with 100 simulated robots. With neural weights fixed, JEPA command loss
falls **4.7452% → 3.8100%**: **8,978 fewer misses, a 19.71% relative reduction**.
The final package passes registered statistical and relative engineering gates.
RadioMap improves **24.47%**, to **3.6980%** loss; final JEPA fails their
superiority and worst-decile comparisons.

The four fixed stages test FIFO, busy-server admission before inference,
STR duplication disabled, and conditional full scans. Admission alone cuts
JEPA held source age **18.97 → 4.34 s** and computed jobs **60 → 29**, but
worsens loss to **4.9125%**. Descriptive subsequent stages improve delivery
after radio-resource and sensing changes. Performed scans still cost 90 ms;
the final stage performs about 53% fewer. JEPA maximum silence **8.35 → 8.37 s**
passes the declared 50 ms allowance while failing zero deterioration.

That study's preflight has **1,489 passing software tests**, with one expected warning. The
[independent verifier](scripts/verify_freshness_stack.py) reproduces the
published 446-file source and raw evidence. RadioMap remains the application
default. These finite simulations establish no physical, production,
JEPA-objective or SOTA qualification.

The [fixed-weight JEPA controller repair](docs/JEPAREPAIR_RESULTS.md) fixes
forecast completion accounting, fractional handover-cost integration and
local recovery from delivered scan evidence. At assumed **30 ms handovers**,
the original JEPA ensemble loses **1.03938%** of commands versus **1.07010%**
with its legacy controller: **295 fewer misses, a 2.87% relative reduction**
on eight fresh 100-robot sites. Both registered statistical and engineering
gates pass for that comparison. Superiority over repaired RadioMap is still
unconfirmed; RadioMap's own mean improvement fails its maximum-silence margin.
The aggressive four-second arm roughly doubles development loss and is
rejected before its slow confirmation cases. Its negative result remains
visible. These are fixed-weight simulated controller-package results, not a
new JEPA-objective advantage. [Runtime diagnosis](docs/JEPAREPAIR_RUNTIME_DIAGNOSIS.md)
and [prediction-to-control diagnosis](docs/JEPAREPAIR_DIAGNOSIS_ARCHITECTURE.md)
explain the repaired execution path and remaining representation mismatch.

The separately registered [cost-aware four-second recovery study](docs/COSTAWARE_RECOVERY_RESULTS.md)
also completes all **eight development, 32 confirmation, four offline replay
and four fresh-cost repetition cases**. Repaired JEPA loses **5.80104%** of
commands versus repaired RadioMap's **6.18688%**, a statistically supported
mean difference under the three-comparison budget and pending-command bound.
Its maximum silence is worse, **10.03 versus 8.34 seconds**, so the combined
acceptance gate fails. Both repairs reduce observed loss and maximum silence
relative to their own legacy controllers, but neither mean improvement is
statistically confirmed. This fresh cohort does not replace the aggressive
slow-arm rejection. That study’s archived suite has **1,245 passing software tests**, with one expected warning;
the [repair](scripts/verify_jepa_repair.py) and
[cost-aware](scripts/verify_costaware_recovery.py) verifiers check the archived
source, actual charged computation, genuine scan evidence, replays and reports.
RadioMap remains the application default. No physical or production activation
has occurred.

The [compact fleet-JEPA comparison](docs/SELECTED_JEPA_RESULTS.md) completes
**12 fits, 240 native evaluations and eight exact replays**, with 100 simulated
robots per evaluation. The new 13,970-parameter inference model preserves AP/band
identity, trains on genuinely delivered simulated observations, and predicts
each horizon directly around RadioMap with a bounded correction. Predicted
residuals are never recursively added; no cumulative missed-command bound follows.

**JEPA superiority is not confirmed.** On eight fresh sites, context JEPA loses
1.0818% / 4.7900% of commands at assumed 30 ms / four-second handovers, versus
RadioMap's 1.0976% / 4.8496%. All six statistical comparisons fail; matched
supervised and the original JEPA ensemble have lower means. A RadioMap timing
overrun also triggers the registered global qualification veto. The separate
[semantic-JEPA study](docs/SEMANTIC_JEPA_RESULTS.md) retains its development
rejection and unused holdouts. An all-six-site validation audit finds finite,
nonconstant representations, which does not establish useful RF abstraction
or eliminate every form of collapse. That study's archived suite has **895
passing software tests**, with one expected warning; the [original](scripts/verify_semantic_jepa.py) and
[selected](scripts/verify_selected_jepa.py) verifiers reproduce the actual
archives. RadioMap remains the application default; no physical, production
or SOTA qualification follows.

The [costed RF-to-packet study](docs/CONTINUAL_PACKET_RESULTS.md) completes
**224 evaluations and six exact replays**, each with 100 simulated robots.
Neither response memory nor Bayesian updating passes its registered service
gate. At assumed four-second handovers, canonical RadioMap loses **13.6070%**
of commands versus **13.8309% / 13.8321%** for staggered memory/Bayesian
corrections. No primary or information contrast establishes superiority;
the worst-decile margins also fail. This procedural managed-interference
study has different conditions from earlier experiments. It adds no JEPA,
hardware or production claim. RadioMap remains the application default.
The opt-in [SQLite execution journal](docs/DURABLE_EXECUTION.md) and
[spawned proposal worker](docs/PROCESS_ISOLATION.md) add local obligation
recovery and supervised process termination without changing existing services.
That milestone’s archived suite passed **697 tests**, with one expected fork-misuse warning;
[the final verifier](scripts/verify_continual_packet.py) reproduces the frozen
report and checks source, archives and prior evidence.

The [continual response study](docs/CONTINUAL_RADIO_STATUS.md) completes six
registered response-learning fits on 24 real Sionna RT contexts. **JEPA does
not beat the numerical controls.** Independent, staggered feedback reduces
Bayesian response error by **77.1%** versus synchronized feedback at equal
transmitter-on airtime; this is a simulation measurement-design result, not a
packet-service gain. Bounded device/context memory learns identifiable action
effects while leaving unsupported queries unknown.

Execution semantics v2 fixes duplicate issue and reservation recovery; runtime
guards bound work and reject stale forecasts. No learned candidate is production
qualified. The [research watch](docs/RAN_RESEARCH_WATCH.md) retains 47 live,
unreviewed leads and configures daily 08:00 Singapore discovery after its
workflow reaches the default branch. Discovery never promotes a model.
That milestone's archived suite passes **582 tests**, including cached pretrained DINOv3
integration. [Verification and evidence hashes](artifacts/ran_research/verification.json)
bind the completed work; this is software verification, not hardware qualification.
The last recorded GitHub CI attempt was blocked before job execution by an
account billing lock; daily discovery requires working Actions and the workflow merged.

The latest [representation and DINOv3 work](docs/VISUAL_RADIO_STATUS.md) audits
six trained critics and completes six new projection-ablation fits. It finds
concentrated, informative latents; a separate future projection improves latent
matching but **does not consistently improve service prediction**. Real frozen
DINOv3 now extracts spatial features from licensed warehouse/robot photographs.
New physical RF factors and a proposal-only hierarchy preserve link identity,
missing observations, capture/compute timing and canonical recovery. Visual/RF
fusion remains untrained, and cross-warehouse improvement is unconfirmed.
The [coupled RT probe](docs/VISUAL_RADIO_RT.md) now also computes real Sionna RT
channels and DINOv3 features from the same warehouse geometry and two blocker
poses. This supplies paired static simulation inputs, with no learned RF score.
Its archived **400 software tests pass**, including actual cached pretrained DINOv3
integration. The [verification receipt](artifacts/visual_radio/verification.json)
links source and evidence hashes; original experiment archives remain unchanged.

The [service-world diagnostic](docs/SERVICE_WORLD_RESULTS.md) adds a
68,021-parameter **untrained, inactive** model with AP/band memory, causal
measurement masks and action-conditioned latent dynamics. Its
[physical contract](docs/SERVICE_WORLD_FIRST_PRINCIPLES.md) checks receiver
freshness, exact outage/deadline floors and bounded interventions that retain
canonical recovery. The [primary-source audit](docs/SERVICE_WORLD_PRIOR_ART.md)
finds direct prior art; novelty and SOTA are **not established**. Official
upstream component forwards are documented in the
[baseline-access audit](docs/SERVICE_WORLD_BASELINES.md), with no matched
trained warehouse comparison. RadioMap remains the active application default.
The corrected diagnostic completes **184 simulator runs and 144 independent
interventions** across four fresh 100-robot sites at two handover costs. Eight
actions reduce focal-robot misses in hindsight; this is sampled headroom, not a
learned policy result. All **318 software tests** and the source/input audit pass.

The [packet-action JEPA study](docs/PACKET_VALUE_RESULTS.md) completes **six
matched fits**, 930 native counterfactual labels from 1,370 packet runs, and
64 development evaluations with 100 simulated robots. It is **rejected on
development**: JEPA loses 24.598% versus canonical RadioMap's 1.045% at 30 ms
handovers, and 12.780% versus 7.515% at four seconds. The uncertainty correction
and ten-second stay commitment suppress necessary recovery. The eight reserved
confirmation sites remain untouched; JEPA superiority is unconfirmed.
The [baseline-preserving shadow interface](docs/PACKET_VALUE_SAFETY.md) keeps
critic advice outside active switching, and a qualification guard rejects this
profile. The application continues using RadioMap. Pilot mistakes and all
negative outcomes are preserved explicitly. Its archived **221 software tests** and the
independent evidence audit pass; that does not qualify the rejected model.

The [delivered-telemetry confirmation](docs/CLOSED_LOOP_RESULTS.md) adds **120
native evaluations on eight fresh test sites**, each with 100 simulated robots.
Actual scan completion, packet delivery, local neighbor caches and queued
forecasts now form a feedback loop. Frozen models are compared without more
training. A development-selected outage-utility controller reduces compact-model
command loss from **5.849% to 5.635%** under assumed four-second handovers;
the paired-site interval excludes zero. At assumed 30 ms handovers it worsens
loss from **1.035% to 1.318%** on all eight sites. RadioMap reaches 5.603% with
the slow utility profile; model control superiority remains unconfirmed.
All **192 tests pass**, and four selected cases replay with identical forecasts,
report deliveries and packet counts. Remaining loss is too high for deployment.
The [design](docs/CLOSED_LOOP_DESIGN.md) records observation and timing assumptions.

The [optimisation frontier](docs/WORLD_FRONTIER_RESULTS.md) adds **nine fits,
432 native runs and 42 fresh test sites** across model, runtime, sensing and
handover-admission experiments. A distilled single model uses a measured
**53 ms** warm CPU budget per 100-robot forecast versus **233 ms** for the
three-member hybrid (4.39× smaller). RadioMap's covariance calculation is
optimised too. The [independent charged-cost test](docs/FRONTIER_CONFIRMATION_RESULTS.md)
has 228.12 versus 251.04 misses/million, but its interval spans zero; robust
control superiority is not established. New simulations charge forecast queues
and scan outages. Separate [sensing/budget selection](docs/HARDWARE_BUDGET_RESULTS.md)
and [all-handover admission](docs/HANDOVER_ADMISSION_RESULTS.md) reduce losses
under assumed four-second handovers. The latter reduces distilled-model loss
from 8.85% to 5.92% on six fresh sites, still too high for deployment.
`OptimizedWorldModel` and `AsyncForecastService` expose an opt-in proposal runtime.
These are simulated results; hardware qualification and broader layouts remain open.

The [fleet world-model study](docs/FLEET_WORLD_MODEL_RESULTS.md) trains nine
models with causal fleet memory, rack geometry, route conditioning and
mean-field/AP-score supervision. Its [independent confirmation](docs/FLEET_CONFIRMATION_RESULTS.md)
fixes the best discovered hybrid before eight additional 100-robot warehouses:
**167.19 → 133.33 missed commands per million**, **20.2% fewer**, with paired
95% interval [-68.75, -5.73] ppm. Forecast RMSE falls from 7.09 to 4.57 dB.
The JEPA hybrid reaches 128.12 ppm as a secondary comparison; the supervised
primary and previous JEPA prevent a claim that JEPA pretraining is necessary.
The full follow-up contains **760 native runs and 40 fresh sites**.
`wirelessjev.fleet_control.FleetWorldModel(..., family="confirmed")` exposes
the frozen primary candidate explicitly. RadioMap remains the application
default. The hybrid retains RadioMap's fitting cost; results use offline
causal forecasts and a 30 ms native handover assumption, not physical cars.

The [JEPA gate repair](docs/JEPA_GATE_REPAIR_RESULTS.md) fixes a mismatch between
noisy individual-link uncertainty and the uncertainty of an AP-switching gain.
On six fresh 100-robot scenes, a development-selected gate penalty reduces
misses from **329.17 to 111.11 per million**; a decision-calibrated gain gate
reaches 133.33. The frozen forecaster is unchanged. An explicit
`wirelessjev.jepa_control.controller()` profile exposes the validated research
configuration; RadioMap remains the default. The 270-run follow-up, six exact
historical replays and all traces are archived. In that study, stress favors fixed split,
and no physical handover or safety guarantee is established.

The original [contracted world-model study](docs/CONTRACTED_RESULTS.md) completed **20
device-JEPA fits and 366 native rollouts on 24 fresh scenes**. It adds censored
scan inputs, future-route conditioning, ensemble uncertainty and a finite
AP-regret training objective. Ungated JEPA already beats the fixed rule, the
gate worsens its primary mean, and regret training has no robust advantage.
The [model](docs/CONTRACTED_MODEL.md), [theory](docs/CONTRACTED_THEORY.md),
[18-paper review](docs/CONTRACTED_PRIOR_ART.md) and portable checksummed archive
are included. RadioMap + WirelessJEV remains the default; no hardware or
foundation-model/SOTA claim follows.

The original completed milestone is **warehouse connectivity: WirelessJEV/RadioJEV
E1–E5 and corrected E7**; its [results](docs/WAREHOUSE_RESULTS.md) are preserved.
The new [warehouse v2 campaign](docs/WAREHOUSE_V2.md) adds structured missions,
sensed-input temporal models, three independent compact-model seeds, active
sensing, site adaptation/removal, real pretrained baselines, and executed
Sionna/ns-3 PHY/MAC studies. See [v2 findings](docs/WAREHOUSE_V2_RESULTS.md).
E6, general robotics, custom on-air waveforms and satellite work remain excluded.
Physical-car qualification remains pending; no foundation-model or SOTA claim
follows from these experiments.

The [30-second horizon extension](docs/WAREHOUSE_HORIZON.md) now trains six
compact forecasters, plans partial switches under explicit mission budgets,
and exposes a validated, proposal-only CLI. On three fresh simulated scenes,
route-aware numerical planning raises mean service from 0.237 to 0.326 against
budgeted strongest-signal roaming. All six learned residuals fail the
closed-loop acceptance checks and fall back to that numerical planner.
The twelve-switch cap is an operator-policy scenario, not a hardware rate
limit; unconstrained roaming performs better on service in the original
horizon holdout. Read the [full results](docs/WAREHOUSE_HORIZON_RESULTS.md)
and [prospective selector audit](docs/WAREHOUSE_HORIZON_SELECTOR.md).

The [compositional mission and radio audit](docs/WAREHOUSE_V3.md) adds hard
per-offset floors, explicit groups, conditional and lexicographic priorities,
masked/JEPA-style pretraining, a full-priority cached-mission scorer, measured
native packet events and a complete local cost frontier. Canonical missions
use a complete compiler by default; actual neural clause verification is
available explicitly. The [v3 results](docs/WAREHOUSE_V3_RESULTS.md) preserve
the failed gates and separate simulations from hardware measurements.
Both implementations currently target Wi-Fi; [protocol-independent adapters
are not implemented](docs/PROTOCOL_BOUNDARY.md).

The [native packet-supervision campaign](docs/WAREHOUSE_NATIVE.md) completed
58 simulations and six model fits across three seeds per architecture.
[Final results](docs/WAREHOUSE_NATIVE_RESULTS.md) show about 62% lower forecast
error, but learned utility of 0.7010 versus numerical 0.6980 has an improvement
interval spanning zero. No learned candidate passed validation; numerical
planning remains the default. The critical-floor risk labels contain only
misses, so zero Brier does not establish useful risk prediction.

WirelessJEV turns warehouse missions into typed constraints and priorities for
planning. RadioJEV learns radio consequences from observations and proposed
actions. The current proposal path uses the complete controlled language and
numerical belief by default; actual System One verification and learned scorers
are explicit research options. The original ten-car prototype and subsequent
100-/200-robot warehouse experiments run in simulation.

Milestone 1 runs entirely in a digital twin. Nothing here touches a car, a radio or a network.

## Hardware it targets (from SmartCar.zip)
Every car is the same hardware as the existing SmartCar (`src/wirelessjev/hardware.py`):
- **Platform:** a Yahboom Raspbot on a Raspberry Pi.
- **Management radio:** the onboard brcmfmac SDIO radio (`wlan0`) runs the management hotspot at 192.168.1.11. WirelessJEV never scans, selects or modifies it; SmartCarGuard owns its recovery.
- **Customer radio:** an Intel BE200 on PCIe (`8086:272b`, `wlan1`). It is the only radio acted on, and the only action is choosing a BSSID.
- **Wire format:** exactly the existing `SmartCarCmd.py` JSON: `status`, and `connect` with ssid/bssid/security plus the Wi-Fi password, which SmartCarCmd requires, just as the PC app sends it. Plans, receipts and logs never contain the password; the real transport (`carlink.py`) injects it from the operator's WiFi.json at send time.
- **Target settings and reported anchors:**
  - an assumed 4 s per BSSID switch, pending physical stage measurements;
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

The full cloud run **completed all nine controller stages**: all E1/E2 arms,
E3 multitask, and E4/E5 site adaptation and forgetting. Corrected E7 is also
complete; E6 remains excluded. The validated completion status and comparisons are
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

E4 mean normalized regret is 0.1820 (local), 0.1781 (FedAvg) and 0.1820
(personalised swarm), versus 0.1729 for the initial E3 model. All three shared
merges were rejected; swarm therefore matches local-only training. E5 does
not demonstrate forgetting: its shared model had accepted no site updates,
and retained-data distillation increased attack AUC from 0.6167 to 0.6489
while worsening retained regret from 0.1801 to 0.2068. These negative results
remain part of the archived evidence.

E2 tests the existing weight-based reader, not a planner with explicit mission
constraints. Its mission implementations mix thresholds, soft penalties and
spatial predicates. The separate router proposal remains deferred.

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
controllers. The completed cloud run resumed E4/E5 under the versioned
controller after the original launcher stopped. Do not restart completed work.
An interrupted arm restarts its training; optimizer/epoch continuation is not
implemented.

For future runs, the independent finalizer waits for training completion,
validates and archives all stages, and rebuilds Studio:

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
