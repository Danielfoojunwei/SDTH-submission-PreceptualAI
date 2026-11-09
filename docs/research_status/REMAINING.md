# RadioJEPA and WirelessJEV: evidence and remaining work

Snapshot: **9 November 2026, 15:55 UTC**. This maps the
[archived user plan](../plans/RADIOJEPA_USER_PLAN.md) to retained evidence.
It is a status document outside scientific source collectors. It neither
amends a frozen experiment nor declares any original phase complete.

**RadioMap remains the application default. No new RadioJEPA head or fleet
controller is production qualified.** The original 24 SERVICE confirmation
specifications remain unopened. A completed software component, an RF-field
gain, a localization result and a native packet-service qualification are
different evidence levels.

## What the completed studies establish

| Evidence | Retained result | Boundary |
|---|---|---|
| Measured factory array-power pilot | Six cells × three training seeds; the best reported mean is 2.650 dB for `joint_vicreg`. [Results](../RADIOJEPA_RESULTS.md). | Referenced channel power, not dBm or a packet-service result. The same-regularizer JEPA/supervised comparison does not establish a JEPA advantage. |
| Static spectral field | 84 Sionna RT scenes, 15 fits and 24 FIELD layouts. Spatial RMSE is 3.622 versus RadioMap-operator 4.843 dB; p95 absolute error 6.920 versus 9.500 dB; same-link 6 GHz 3.734 versus 4.604 dB. [Results](../spectral_scene/RESULTS.md). | Exact matched regression is 3.631 dB and RF-value-erased geometry/availability 3.660 dB. The gain supports the architecture, without isolating the embedding objective or dependence on rich RF input. Every primary query was heard. The closure binds 475 source and 5,161 evidence files. |
| Real phone/location baseline | TUJI1, five held-out phones, official held-out reference positions, N=0/1/4/16 supports. AP-specific calibration changes mean localization error from 4.516 to 4.225 m at N=16. [Results](../device_transfer/RESULTS.md). | Classical calibration/localization, not a RadioJEPA encoder. One site; device/day effects are confounded; two phone-level tails worsen. No isolated hardware-inverse or arbitrary-device claim. Closure: 487 source and 514 evidence files. |
| Measured complex-CSI completion | Six neural fits and numerical controls on real DICHASUS. Joint mean packet NMSE is 1.083 on cf06, 1.080 on ad00 and 1.069 on dc01. Corresponding strong numerical values include LMMSE 1.000 on cf06, Fourier 1.000 on ad00 and linear interpolation 0.0876 on dc01. [Raw report](../../artifacts/channel_scene/report.json), [passing raw audit](../../artifacts/channel_scene/verification/raw.json). | The neural result is negative. This is digitally masked, same-packet/same-antenna pilot completion, not a learned scatterer scene or service result. Acquired prefixes were previously inspected; no globally unseen-data claim. The 491-source closure requires the separately stored prediction capsule for complete replay. |

The static model's near-tie with RF-value-erased input and the complex-CSI
losses must accompany its RadioMap field gains. They do not justify declaring
representation collapse solved, JEPA uniquely useful, or specialist SOTA beaten.

## Current experimental tracks and integration

| Track at this snapshot | What exists | What is still required |
|---|---|---|
| Adaptive RF conditions | Source/protocol published before execution. All 12 fits completed; VAL selected regression. The [DEV receipt](../../artifacts/adaptive_radio/development.json) records passing checks with 152 censored query events. The [plan](../adaptive_radio/PLAN.md) fixes 24 TRAIN/6 VAL/4 DEV/24 conditional TEST layouts and three material conditions per CAD. | Finish source-bound raw verification, qualification and the conditional fresh TEST/report/closure sequence. A DEV predicate is not a completed TEST result. TEST stays closed after a veto. Static condition changes are not temporal dynamics or network delivery. |
| Native warehouse bridge | Published-source [diagnostic](NATIVE_RESULTS.md) completed: four fresh episodes × 0.03/4 s handovers, 100 simulated robots, 12 s, eight paired cases plus an exact Map round-trip wiring pair. All 18 simulations and the passive audit completed; closure binds 512 source/56 evidence files. | Fixed seed-0 regression worsens misses in all four fast-handover cases; slow mean improves with two of four cases harmed. One gap reaches 9.74 s and forecast computation rises. Diagnose field transfer, cost and recovery under a new protocol; do not promote or retune on these frozen cases. Original SERVICE remains unopened. |
| Bounded multistep planning | [Beam/exhaustive search](../../src/wirelessjev/adaptive_radio/planning.py) uses the typed outcome cost, exact forecast slicing, finite path/evaluation/time caps and known association handovers. Analytic exhaustive and native single-step parity tests pass. | This advances association over exogenous RF forecasts, independently per robot. It is not learned action dynamics, joint fleet control, CEM/MPPI, arbitrary STL or proof of service benefit. The first native probe uses the shared exact held-action utility; it does not experimentally validate this multistep planner. |
| RF-to-service boundary | [Service adapter](../../src/wirelessjev/adaptive_radio/service.py) declares reference gauge, TX, noise, linear interference, reciprocity, actual carriers and PHY approximations. RF residual and packet-label calibration have separate contracts. | No packet-risk calibrator has been fitted/qualified for a new learned fleet controller. RSSI coverage alone cannot certify command misses, latency or silence. The native 6.5 GHz probe is extrapolation, not the separate RT 6.0 GHz endpoint. |

Native episodes share a warehouse family, not four measured buildings. The
native radio/queue loop affects associations, service and delivered reports;
physical trajectories remain exogenous. Results must retain both gains and
harms, worst-decile misses, per-robot maximum gaps, video/bulk and actual paid
compute. The 0.03 s and 4 s handover conditions must stay separate.

## Mapping the original plan

| Original phase | Established slice | Remaining prerequisites |
|---|---|---|
| **0: benchmarks and honesty** | Licensed acquisitions, primary-source/theory audits, fixed protocols, strong numerical controls, matched supervision, source publication and passive evidence checks exist. | Reproduce task/split-matched specialist references before SOTA claims. Existing public reference numbers, code reuse and downloaded data do not establish leaderboard parity or novelty. |
| **1: good views and devices** | Actual multiband/reciprocal RT solves, explicit simulated frontends, separate private physics and public reports; measured CSI/RSSI corpora. | Truly synchronized multi-device/robot observations and phase/time reference evidence where coherence is claimed. Real CFO/STO/IQ/AGC/antenna effects, dynamic blockers/interference and delivery clocks are not validated by static simulated offset/quantizer profiles. |
| **2a: RSSI operators and priors** | Frequency/FSPL, interval/Tobit likelihoods, prior ablations and the completed field architecture. Adaptive RF-necessity study is executing. | Demonstrate gains requiring current RF, against amplitude-erased and geometry-only controls, under fresh split/gate rules. Do not retune the finished FIELD test. PPR remains ineligible where phases/materials/physical factors are unidentified. |
| **2b: front-end inversion** | Simulated relative-offset onboarding and a real classical five-phone calibration curve. | Train/evaluate a new RadioJEPA encoder across physical devices; distinguish device, scene/day and missing information. Relative calibration does not identify device 0's absolute bias. Clipping, missing bands and quantization cannot generally be inverted uniquely. |
| **2c: complex scene/scatterers** | Complex readers, gauge/coherence contracts, Fourier operators and one completed negative pilot-completion study. | Diagnose pilot observability, sampling/coherence and calibration with training/validation data; design a new task/protocol if needed. A Fourier coefficient is not an identified object, path, material or absolute delay. Demonstrate physical identifiability before naming scatterer tokens. |
| **2d: dynamics and hierarchy** | OU/Jakes/filter utilities, physical-unit operators, hierarchy schemas and software checks. | Fit and evaluate an observation-driven slow/mid/fast belief on actual event-time sequences. Test revisits, rollout drift, uncertainty, change detection and action support. Static RF conditions and analytic dynamics helpers do not establish stable learned rollouts or nonaccumulating residual error. |
| **2e: DINO/camera** | Actual frozen DINOv3 features for five approximately aligned frames, with source/timing receipts. | Calibrated camera geometry, proven synchronization, overlay-leakage audit, adequate paired data and a trained matched RF–vision experiment. The existing receipt explicitly establishes no RF gain or sensing/safety qualification. |
| **3a–b: interface, mission and planning** | Typed predicted states/actions/outcomes, exact RadioMap utility parity, checked numeric mission schema, bounded single-/multistep software, external head routing and refusal on unavailable bounds. | The learned native comparison is pending. No general text/STL compiler, action-conditioned physical transition model or calibrated joint chance guarantee exists. Gate permissions remain external; conformal/bin calibration does not grant validity under arbitrary drift or per-query selection. |
| **3c–g: expanded WirelessJEV** | Legacy sensing/load/MLO heuristics, accounting, shared report maps and numeric decision provenance remain available. | New RadioJEPA value-of-information sensing, joint band/MLO/MCS/scan planning, coordinated capacity/interference assignment, shared online scene belief, route advisories, faithful physical explanations and labeled sensing/change alerts each need implementation and an independent endpoint experiment. The new API intentionally rejects unsupported MLO/power/width/routes/mechanisms. |
| **4: real transfer** | Actual measured-channel experiments and real classical held-phone/location evaluation. | Neural any-device onboarding, device↔device and sim↔real alignment, session-held-out transfer where clocks exist, own multi-device capture, and protocol ACK/retry/deadline labels. TUJI1 lacks session timestamps and cannot supply simultaneous-view or isolated session/device evidence. |
| **5: scale and qualification** | Multiple seeds, source-bound splits, independent gates, raw prediction checks and completed closures for narrow tasks. | Dataset/model/view/site scaling, complete first-principle ablations, longer independently reserved service episodes, adequate power analysis, calibrated tails, deployment compute/memory/energy and hardware/rollback evidence. Four short native episodes do not replace 24-site SERVICE confirmation. |

## Execution order and stopping rules

1. Finish and audit the published adaptive RF study. A DEV veto is a result;
   it does not authorize new seeds, substituted layouts, retuning or TEST.
2. Preserve the completed native diagnostic, including fast-handover harm,
   its 9.74-second recovery outlier and higher forecast cost. Investigate
   transfer, completion and recovery with separate hypotheses; changes need
   new source/protocol and fresh comparisons. The slow-handover mean gain
   supplies no production or SERVICE permission.
3. Preserve the measured-CSI negative result. Confirm sampling, gauge and
   observation sufficiency on eligible data before choosing another model.
   Any revised task or post-test architecture requires a new protocol.
4. Build the dynamic and protocol-supervised slices with independent
   calibration units and action support. Test actual packet/deadline/silence
   labels, tails and paid sensing/transport/compute under the same gate.
5. Only then pursue a separately permitted SERVICE confirmation and physical
   device/fleet validation. Driver capabilities, measurement clocks, absolute
   reference calibration, access, supervision and rollback remain hardware
   prerequisites. No source receipt or simulator result grants actuation.

No result here completes the full RadioJEPA program, establishes a universal
foundation model, or qualifies connectivity for arbitrary robots, protocols
and environments. Those are research targets with additional observable-data,
model-support and service requirements.
