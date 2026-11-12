# RadioJEPA and WirelessJEV: evidence and remaining work

Snapshot: **12 November 2026, UTC**. This maps the
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
| Zero-capable residual and larger-budget RF study | All 15 fits / 28,800 committed updates, 84 fresh RT cases plus 90 reused TRAIN/VAL cases, 5,928 prediction streams and four primary contrasts complete. Selected `nested_joint` sparse RMSE is **2.637330 dB**, versus validation-selected fixed-feature hierarchy **2.710522** and global-eight regression **2.651669**. All four adjusted 98.75% intervals favor their candidates. The 578-source/2,770-evidence seal and independent replay pass. [Results](../nested_radio/RESULTS.md), [source publication](../../artifacts/nested_radio/registration/git.json). | The matched joint gain is only **0.002226 dB**. All eight learned-local gates are zero at the selected checkpoints of all six nested members; the active local design uses fixed Fourier channels, with trained shared priors and discrepancy. This supports a narrow RF shrinkage result, not learned local abstraction, solved collapse, temporal dynamics or specialist SOTA. Material-delta error is essentially tied and slightly favors the numerical reference (0.906549 versus 0.906270 dB). This RF-only phase runs zero new native simulations or hardware actions; earlier packet deficits remain. |
| Shared/AP-local spatial posterior | 186 actual RT cases, 15 fits and 34 warehouse runs; 567-source/4,900-evidence seal and independent replay pass. [Results](../hierarchical_radio/RESULTS.md). | Selected hierarchy loses sparse RSSI RMSE to global-24 and numerical Map (3.088402 versus 2.995352/2.951064 dB). Matched agreement improves by 0.034739 dB; the small global-eight control has the lowest descriptive mean. Native hierarchy has higher mean misses than full-history Map at both costs. No default promotion, collapse diagnosis, converged-training or temporal/SOTA claim. |
| Measured factory array-power pilot | Six cells × three training seeds; the best reported mean is 2.650 dB for `joint_vicreg`. [Results](../RADIOJEPA_RESULTS.md). | Referenced channel power, not dBm or a packet-service result. The same-regularizer JEPA/supervised comparison does not establish a JEPA advantage. |
| Static spectral field | 84 Sionna RT scenes, 15 fits and 24 FIELD layouts. Spatial RMSE is 3.622 versus RadioMap-operator 4.843 dB; p95 absolute error 6.920 versus 9.500 dB; same-link 6 GHz 3.734 versus 4.604 dB. [Results](../spectral_scene/RESULTS.md). | Exact matched regression is 3.631 dB and RF-value-erased geometry/availability 3.660 dB. The gain supports the architecture, without isolating the embedding objective or dependence on rich RF input. Every primary query was heard. The closure binds 475 source and 5,161 evidence files. |
| Adaptive RF conditions | 12 fits, 174 Sionna RT cases and 24 TEST layouts. Selected regression's sparse condition-delta RMSE is 1.520106 versus RadioMap 1.654205 and RF-erased 1.885921 dB; adjusted contrasts support current-RF dependence. [Results](ADAPTIVE_RESULTS.md). | Joint alignment is worse at 1.534382 dB; pooled Bayes nearly ties at 1.524141 dB; detector benefit is unconfirmed. Both canonical raw replays and the 504-source/11,792-evidence seal pass; independent raw replay also passes. Static simulated states and known calibrated frontends do not establish temporal learning or physical transfer. |
| E7 calibration-to-planner diagnostic | All 15 opened environments, 90 fits, 525 searches and 225 pairs complete; 525-source/382-evidence closure and independent audit pass. [Results](E7_CALPLANNING_RESULTS.md). | Map RMSE improves, while mean utility changes +0.031479/−0.010718/−0.002326 at nominal 1/5/15-minute tail settings; 7/7/9 environments are harmed. Oracle-conditioned descriptive analytical utility, not fresh confirmation, chronological acquisition time, packet benefit or default promotion. |
| Spatial support and local recovery | Fixed regression weights/64-token encoder, causal memory/source-band admission, causal central commands and separately preinstalled paid own-client recovery are built. All 48 fresh + two parity simulations, 543-source/224-evidence seal and independent audit complete. [Architecture/results](SPATIAL_CONTROL.md), [all contrasts](EXECUTION_LEDGER.md#spatial-control-contrasts). | At 0.03 s support/local recovery/both improve regression's mean, with 1/0/0 miss harms; both still loses to Map + recovery on all four episodes. At 4 s support has no miss/gap harms versus regression, while recovery/both harm all four; Map recovery also harms all four. No new representation training, JEPA-objective, production or hardware claim. |
| Real phone/location baseline | TUJI1, five held-out phones, official held-out reference positions, N=0/1/4/16 supports. AP-specific calibration changes mean localization error from 4.516 to 4.225 m at N=16. [Results](../device_transfer/RESULTS.md). | Classical calibration/localization, not a RadioJEPA encoder. One site; device/day effects are confounded; two phone-level tails worsen. No isolated hardware-inverse or arbitrary-device claim. Closure: 487 source and 514 evidence files. |
| Measured complex-CSI completion | Six neural fits and numerical controls on real DICHASUS. Joint mean packet NMSE is 1.083 on cf06, 1.080 on ad00 and 1.069 on dc01. Corresponding strong numerical values include LMMSE 1.000 on cf06, Fourier 1.000 on ad00 and linear interpolation 0.0876 on dc01. [Raw report](../../artifacts/channel_scene/report.json), [passing raw audit](../../artifacts/channel_scene/verification/raw.json). | The neural result is negative. This is digitally masked, same-packet/same-antenna pilot completion, not a learned scatterer scene or service result. Acquired prefixes were previously inspected; no globally unseen-data claim. The 491-source closure requires the separately stored prediction capsule for complete replay. |

The static model's near-tie with RF-value-erased input, the adaptive model's
confirmed RF dependence but worse alignment branch, and the complex-CSI losses
must all remain visible. The completed zero-capable follow-up supplies a fresh,
conditional point-prediction gain without erasing those negative results or
the preceding hierarchy's service losses. Its all-zero learned-local gates
do not establish whether those features were unnecessary or optimization
removed useful capacity. None of these results declares collapse solved,
JEPA uniquely useful, or specialist SOTA beaten.

## Current experimental tracks and integration

| Track at this snapshot | What exists | What is still required |
|---|---|---|
| Nested RF model | The [published protocol](../nested_radio/PLAN.md) and source commit `328c9d1292c216eb973862988f5656b10651bcc6` precede fitting and fresh tracing. All fifteen longer fits and 84 new cases, raw checkpoint selection, four adjusted comparisons and [independent replay](../../artifacts/nested_radio_review/independent.json) are complete. TRAIN/VAL alone selects `nested_joint` and the fixed-feature numerical hierarchy. | No continuation fit or rescoring of this TEST is needed. Fresh packet-service transfer, complete cost measurements, calibrated tails and physical trials have not been performed for these weights. Gate/budget/regularizer changes form a package, and the small matched agreement gain does not isolate a general JEPA advantage. |
| Adaptive RF conditions | Source/protocol preceded execution; 12 fits and all 174 cases complete. VAL selected regression and DEV permitted TEST. All 24 TEST layouts support the delta endpoint; the completed report, canonical raw replays and seal cover 540 DEV/3,240 TEST records. [Results](ADAPTIVE_RESULTS.md). The sealed Studio card and independent review are complete. | No replacement fit or TEST tuning is needed. Static material changes are not temporal dynamics, real-device transfer or network delivery. |
| Native warehouse bridge | Published-source [diagnostic](NATIVE_RESULTS.md) completed: four fresh episodes × 0.03/4 s handovers, 100 simulated robots, 12 s, eight paired cases plus an exact Map round-trip wiring pair. All 18 simulations and the passive audit completed; closure binds 512 source/56 evidence files. | Fixed seed-0 regression worsens misses in all four fast-handover cases; slow mean improves with two of four cases harmed. The 9.74-second outlier and higher compute remain. The separate completed spatial-control comparison examines support/recovery without replacing these outcomes or qualifying field transfer. Original SERVICE remains unopened. |
| Spatial control and executable recovery | The [stack](SPATIAL_CONTROL.md) uses typed spatial reports, latest-own 2.4/5.2 GHz support with queued server-delivery refresh, paid causal central commands and own-client local recovery. All 50 fixed-weight runs and passive reviews complete. | This control experiment retains its unchanged encoder and all four-second recovery harms. The separate [spatial hierarchy](../hierarchical_radio/RESULTS.md) is now trained and evaluated, with a bounded 192-report route-coverage selector and local recovery off. Learned event-time dynamics and qualified service heads remain future work; neither study establishes production transfer. |
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
| **2a: RSSI operators and priors** | Frequency/FSPL, interval/Tobit likelihoods, strong numerical controls and completed field assays. The zero-capable follow-up confirms selected-model gains over its validation-selected numerical reference and global-eight control on 24 fresh TEST geometries, conditional on three trained members. Earlier adaptive RF-dependence evidence remains retained. | Replicate the narrow gain across wider scene/device/budget settings and isolate useful learned abstraction beyond shared physics and fixed spatial features. Keep the nearly tied material-delta result visible. Do not retune completed TEST/FIELD cases. PPR remains ineligible where phases/materials/physical factors are unidentified. |
| **2b: front-end inversion** | Simulated relative-offset onboarding and a real classical five-phone calibration curve. | Train/evaluate a new RadioJEPA encoder across physical devices; distinguish device, scene/day and missing information. Relative calibration does not identify device 0's absolute bias. Clipping, missing bands and quantization cannot generally be inverted uniquely. |
| **2c: complex scene/scatterers** | Complex readers, gauge/coherence contracts, Fourier operators and one completed negative pilot-completion study. | Diagnose pilot observability, sampling/coherence and calibration with training/validation data; design a new task/protocol if needed. A Fourier coefficient is not an identified object, path, material or absolute delay. Demonstrate physical identifiability before naming scatterer tokens. |
| **2d: dynamics and hierarchy** | Causal memory, bounded route-coverage selection and the retained 186-case/15-fit/34-run hierarchy comparison; the separate zero-capable residual and larger-budget successor now completes its fresh RF evaluation with four favorable primary intervals. | Learned local features are gated off, and this phase has no new packet or event-time experiment. Learned query-specific retrieval, transitions, revisit/change detection, rollout stability and service heads remain unvalidated. A static RF gain and avoiding predicted-measurement feedback do not prove nonaccumulating errors. See the [nested architecture](../nested_radio/ARCHITECTURE.md) for the exact zero/one property and its limits. |
| **2e: DINO/camera** | Actual frozen DINOv3 features for five approximately aligned frames, with source/timing receipts. | Calibrated camera geometry, proven synchronization, overlay-leakage audit, adequate paired data and a trained matched RF–vision experiment. The existing receipt explicitly establishes no RF gain or sensing/safety qualification. |
| **3a–b: interface, mission and planning** | Typed states/actions/outcomes, exact RadioMap utility parity, checked numeric missions, bounded planning and refusal on unavailable heads; causal central command creation/delivery/expiry plus separately declared own-client recovery are now implemented and tested in the completed 50-run diagnostic. | No general text/STL compiler, learned action-conditioned transition, calibrated joint chance guarantee or authenticated hardware authority exists. Source-band support is not packet-risk qualification. Gate permissions remain external; conformal/bin calibration does not grant validity under arbitrary drift or query selection. |
| **3c–g: expanded WirelessJEV** | Legacy sensing/load/MLO heuristics, accounting, shared maps, bounded AP/band/MCS software, target-nullspace/information-per-cost mathematics and read-only R6 bridge; new paid autonomous scanning, shared cadence, admission refresh and own-client execution lifecycle are built. | Qualified RF value-of-information sensing, joint fleet capacity/interference optimization, MLO/width/power/route-supported actions, learned shared event-time belief and calibrated alerts still need endpoint evidence. A paid recovery heuristic is not a learned sensing policy or safety certificate. Existing planners and R6 previews do not supply authenticated action authority or qualified service forecasts. |
| **4: real transfer** | Actual measured-channel experiments and real classical held-phone/location evaluation. | Neural any-device onboarding, device↔device and sim↔real alignment, session-held-out transfer where clocks exist, own multi-device capture, and protocol ACK/retry/deadline labels. TUJI1 lacks session timestamps and cannot supply simultaneous-view or isolated session/device evidence. |
| **5: scale and qualification** | Multiple seeds, source-bound splits, independent gates, raw prediction checks and completed closures for narrow tasks. | Dataset/model/view/site scaling, complete first-principle ablations, longer independently reserved service episodes, adequate power analysis, calibrated tails, deployment compute/memory/energy and hardware/rollback evidence. Four short native episodes do not replace 24-site SERVICE confirmation. |

## Execution order and stopping rules

1. Preserve the completed adaptive 504-source/11,792-evidence seal, passing
   independent raw replay and reproducible Studio export. Fits, TEST
   predictions, scoring and canonical raw replays are complete; no replacement
   seeds or TEST retuning are authorized. Preserve the completed mixed E7
   planning result without promoting it.
2. Preserve both completed native diagnostics: the original 9.74-second
   outlier/fast-handover harms and the new 48+2 support/recovery comparison.
   Its source-band guard and local recovery are built, not pending software.
   Retain all handover-specific harms, especially four-second local recovery.
   The new central kernel is shared by fresh arms and differs from the old
   diagnostic; cross-study differences do not isolate one intervention.
3. Preserve the measured-CSI negative result. Confirm sampling, gauge and
   observation sufficiency on eligible data before choosing another model.
   Any revised task or post-test architecture requires a new protocol.
4. Preserve both the negative shared/AP-local comparison and the completed
   zero-capable RF successor, including its small agreement gain, inactive
   learned-local channels and nearly tied material-delta endpoint. Do not
   extend or retune these completed TEST runs. A new, separately registered
   packet-transfer comparison must test the preselected weights against both
   full-history RadioMap and equally informed numerical/global controls,
   retaining tails, paid sensing/transport/compute and both handover costs.
   Event-time dynamics and service heads still need causal sequence labels,
   independent calibration units and supported actions.
5. Only then pursue a separately permitted SERVICE confirmation and physical
   device/fleet validation. Driver capabilities, measurement clocks, absolute
   reference calibration, access, supervision and rollback remain hardware
   prerequisites. No source receipt or simulator result grants actuation.

No result here completes the full RadioJEPA program, establishes a universal
foundation model, or qualifies connectivity for arbitrary robots, protocols
and environments. Those are research targets with additional observable-data,
model-support and service requirements.

The latest [software receipt](../../artifacts/nested_radio_review/software_suite_importlib.json)
records **3,633 passing tests**, 153 modules and 575 unchanged source/config/test
files. The [3,416-test snapshot](../../artifacts/hierarchical_radio_review/software_suite_importlib.json)
and all earlier scientific evidence remain retained; overlapping test totals
are not added. Ten scientific closures now cover the completed narrow studies.
Software checks and an RF-field improvement add no independent robot episode,
calibrated service head or hardware evidence. **RadioMap + WirelessJEV remains
the application default.**
