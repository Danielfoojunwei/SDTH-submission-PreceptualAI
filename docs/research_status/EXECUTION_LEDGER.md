# Execution ledger

Experiment snapshot: **12 November 2026, UTC**. The current [architecture](CURRENT_SYSTEM.md), [nested-RF results](../nested_radio/RESULTS.md), [shared/AP-local results](../hierarchical_radio/RESULTS.md), [adaptive results](ADAPTIVE_RESULTS.md), [local Studio](STUDIO_EXPORT.md), [read-only R6 bridge](R6_BRIDGE.md), [mixed E7 planning diagnostic](E7_CALPLANNING_RESULTS.md) and [spatial support/recovery stack](SPATIAL_CONTROL.md) reconcile implemented work with retained evidence. The completed nested-RF phase has 15 fits, 28,800 committed updates, 84 fresh actual RT cases, 90 reused TRAIN/VAL cases and 5,928 prediction records. Its validation-selected model improves all four registered primary comparisons on 24 fresh TEST geometries, with a passing 578-source/2,770-evidence seal and independent passive replay. This is a bounded static-RF result: learned local features switch off in all six nested members, the incremental agreement gain is tiny, and no new native simulation or hardware action ran. The earlier hierarchy's negative field/native results and all frozen studies remain unchanged. This ledger starts no training or scientific case. Neither a field win nor test success proves that every E helps 100 real SmartCars. **RadioMap + WirelessJEV remains the default.**

## Verified software and source record

The latest [software receipt](../../artifacts/nested_radio_review/software_suite_importlib.json) and [log](../../artifacts/nested_radio_review/software_suite_importlib.log) record **3,633 passed tests**, 153 test modules and identical before/after hashes for 575 source/config/test files. All ten canonical study closures pass; the nested [evidence seal](../../artifacts/nested_radio/verification/manifest.json) adds the completed phase while its [source receipt](../../artifacts/nested_radio/registration/source.json) retains the nine prior closure identities. Software checks and passive replays add no fit, forward, RT case or native simulation.

The preceding hierarchy [software receipt](../../artifacts/hierarchical_radio_review/software_suite_importlib.json) and [log](../../artifacts/hierarchical_radio_review/software_suite_importlib.log) retain **3,416 passed tests**, 149 test modules and identical before/after hashes for 564 source/config/test files. Pytest used isolated import mode to avoid duplicate test-module basenames; the earlier collection failure is retained separately. The [2,978-test proof](../../artifacts/spatial_control_review/final_software_suite.json) remains unchanged. Overlapping suite totals are not added to scientific sample counts.

The preceding [release-suite receipt](../../artifacts/research_review/release_suite.json) and [log](../../artifacts/research_review/release_suite.log) retain **2,655 passed tests**, one expected multiprocessing/fork warning, 130 tracked modules and 421 unchanged source/config/test files. Its 244.34-second pytest and 246.88-second subprocess timings remain historical software evidence.

The earlier [whole-suite receipt](../../artifacts/research_review/whole_suite.json) and [log](../../artifacts/research_review/whole_suite.log), committed at `1026b8c`, retain their separate **2,547-test**, 124-file scope unchanged. Deferred untracked `observation_frontier` work is outside these tracked-suite proofs. Test success is software evidence, not RF accuracy, deployment qualification or a physical fleet trial.

Adaptive RF and CSI source were published at `9bd34ab688d8b74dc2e05df8a6c55fed87257f20`; publication verification is retained in commit `3b1050d` and the respective [adaptive](../../artifacts/adaptive_radio/registration/git.json) and [CSI](../../artifacts/channel_scene/registration/git.json) receipts. The [native bridge receipt](../../artifacts/native_bridge/registration/git.json) binds `fa0a3b3070f37ec9398f0124c1d807f474a68e88`. The separate [operator profile](../../artifacts/native_bridge_ops/registration/git.json) binds `0bc5b4360d8c8023af6891b53374b4a576f26077`. These publication records are distinct from a merge, release or actuation authorization. The spatial-control [543-file source receipt](../../artifacts/spatial_control/registration/source.json), [publication proof](../../artifacts/spatial_control/registration/git.json) and [finite plan](../spatial_control/PLAN.md) bind commit `e185261e515fbd3993a749ab4815d7b738af16c5` before any fresh world or forward. The continuation is reviewed in [PR #25](https://github.com/Danielfoojunwei/SDTH-submission-PreceptualAI/pull/25); its publication record is not deployment authority.

The new [567-file hierarchy source receipt](../../artifacts/hierarchical_radio/registration/source.json)
and [publication proof](../../artifacts/hierarchical_radio/registration/git.json)
bind `8fe31db9974df2e65a016af6b2e16cdb27c3cab2` before collection or fitting.
[PR #26](https://github.com/Danielfoojunwei/SDTH-submission-PreceptualAI/pull/26)
contains its completed experiment, [first independent replay](../../artifacts/hierarchical_radio_review/independent.json)
and [3D viewer](../../artifacts/hierarchical_radio_review/viewer/index.html).
No original large-model seed replication or hardware execution follows from it.

The nested [578-file source receipt](../../artifacts/nested_radio/registration/source.json),
SHA-256 `a9a1fa1670ab5c6801cb828d1e059d1ff25e09b15720252e8a73174bfadbd06b`,
and [publication proof](../../artifacts/nested_radio/registration/git.json) bind
`328c9d1292c216eb973862988f5656b10651bcc6` before its fits, fresh RT or predictions.
[PR #27](https://github.com/Danielfoojunwei/SDTH-submission-PreceptualAI/pull/27)
contains the finite [registered plan](../nested_radio/PLAN.md), completed
[report](../../artifacts/nested_radio/report.json),
[independent replay](../../artifacts/nested_radio_review/independent.json) and
[3D viewer](../../artifacts/nested_radio_review/viewer/index.html). The replay
rejoins all 5,928 prediction records without constructing a model or running a
scientific case. The separate [browser review](../../artifacts/nested_radio_review/viewer/browser_review.json)
checks 576 geometry/state/AP/carrier choices against 217 unchanged input hashes.

## Original E experiments: no stopped worker remains

All nine report files in [RadioJEV cloud provenance](../../artifacts/radiojev_cloud/provenance.json) exist and match their recorded SHA256 values. It declares `run_complete: true`, no pending stages and successful exit codes for all nine stages. The original Mac files are preserved separately. The canonical original-result table is [WAREHOUSE_RESULTS.md](../WAREHOUSE_RESULTS.md).

| Item | Execution status and evidence | Remaining claim boundary |
|---|---|---|
| **E1** | Completed: `e1_no_llm`, `e1_radio_lora`, `e1_radio_frozen`, `e1_intent` under [radiojev_cloud](../../artifacts/radiojev_cloud). The formerly interrupted frozen arm was restarted and finished. | Intent + planner wins the paired cloud language comparison. Numeric-only has no new-mission evaluation. Original large-model training seed is **0 only**; evaluation-case intervals are not training-seed uncertainty. |
| **E2** | Completed: `e2_radio_q`, `e2_e1_zero`, `e2_intent`, same provenance. | Radio-Q beats the original fixed-weight reader on E2; this does not establish superiority over the later structured evaluator. E1/E2 metrics differ. One original training chain. |
| **E3** | Completed: [multitask report](../../artifacts/radiojev_cloud/e3.json) and [label audit](../../artifacts/radiojev_cloud/e3_label_audit.json). | Association improves; traffic tasks lose to rules. DSA uses one 500-step episode. Original admission labels contain only `lower`, so its accuracy is not generalization. Later all-label compact experiments do not repair this historical dataset retroactively. |
| **E4/E5** | Completed: [joint report](../../artifacts/radiojev_cloud/e45.json), [resume-source receipt](../../artifacts/radiojev_cloud/e45_resume_provenance.json), signed-log/checkpoint validation in provenance. | All three E4 shared merges were rejected; no accepted contribution existed to forget in that shared state. E5 distillation worsens reported attack AUC and retained regret. These are completed negative/limited results, not unfinished jobs to rerun until successful. |
| **E6** | **Explicitly excluded.** | Do not add it to the completion denominator or silently start it. |
| **E7** | Corrected full 15-environment [report](../../artifacts/envlearn_corrected/report.json), opened-seed [map calibration study](../../artifacts/envlearn_calibration/report.json) and historical [buggy report](../../artifacts/envlearn/report.json) are preserved. The exact [calibration-to-planner diagnostic](E7_CALPLANNING_RESULTS.md) now completes 15 environments, 90 fits, 525 searches and 225 paired outcomes, with independent audit and 525-source/382-evidence seal. | Map RMSE improves, but analytical utility changes +0.031479/−0.010718/−0.002326 at nominal 1/5/15-minute nested tail-row fractions; 7/7/9 environments are harmed. Opened seeds 15–29 and shared oracle metadata make this descriptive, not fresh confirmation, chronological acquisition time, packet benefit or default promotion. |

The old “E3 evaluating, E4/E5 queued, frozen E1 lost” text is obsolete. No original E1–E5 worker needs restarting. Four cloud checkpoint identities are recorded in provenance; a hash receipt does not mean every large checkpoint is embedded in the Git repository.

## Later warehouse work already built and executed

| Earlier proposed work | Actual retained implementation/evidence | Status to carry forward |
|---|---|---|
| Stronger mission semantics and routing | [Warehouse scope checklist](../WAREHOUSE_SCOPE_CHECKLIST.md), [v2 results](../WAREHOUSE_V2_RESULTS.md), [v3 results](../WAREHOUSE_V3_RESULTS.md); typed compiler, capability/defer routing, cached/latency-aware comparisons and compositional proposal CLI. | Built; language coverage remains limited. The original [mixed E1/E2 router experiment](../ROUTER_EXPERIMENT.md) is still deferred as specified; later routing code is not that experiment. |
| Sensed-input, temporal and compact multitask models | Completed [v2 stage status](../../artifacts/warehouse_v2/studies/status.json), core/factorized receipts, and [v3 manifest](../../artifacts/warehouse_v3/manifest.json); independent compact-model chains, all-label admission, traffic/Chronos controls and shift tests. | These E8-like capabilities are not unstarted, but the proposed FTM/Bluetooth/stock-input fusion is not complete. No separately numbered canonical E8 result or three-seed replication of original E1–E5 follows. Learned control and transfer results remain mixed. |
| PHY realism and rate-margin studies | Executed Sionna coded-QAM study, NIST link curves, 0/3/6 dB rate margins and native ns-3 packet comparisons documented in [v2](../WAREHOUSE_V2_RESULTS.md), [v3](../WAREHOUSE_V3_RESULTS.md) and [native results](../WAREHOUSE_NATIVE_RESULTS.md). | Dependencies and real software execution exist. Generic LDPC/QAM is not a full Wi-Fi PHY, calibrated rack propagation, custom over-the-air modulation or reproduction of ANSpect proprietary algorithms. |
| Admission, shared adaptation and meaningful removal controls | Later all-label data and three-chain controls; an unconditional shared research target actually incorporates the forgotten site in v3. | Completed comparisons: release gates still reject; approximate v3 removal matches retained retraining in 0/3 seeds. Preserve failures; no certified deletion or successful shared production release. |
| E7-style paid sensing and fleet tails | Later matched scan-budget and downstream service studies; per-robot misses, silence and worst-decile metrics. | Completed narrow studies, not a blanket solution for inaudible interference or benefit to every robot. These differ from the original map-only E7 protocol. |
| Switch-stage timing software | Tracked [handover module](../../src/wirelessjev/handover.py), [one-car integration](../../src/wirelessjev/onecar.py) and [workflow](../ONE_CAR_TEST.md) already capture/validate available association, authentication, IP/DHCP and first-packet events. | Software is built; actual on-car stages and distributions remain unmeasured. Missing events are not to be invented. The newer observer in deferred work is not a hardware result. |
| Earlier JEPA/RadioMap control campaigns | Completed result/closure records, including [independent compact confirmation](../FRONTIER_CONFIRMATION_RESULTS.md). | Several results are null or adverse; confirmation did not establish the stated superiority criterion. They are not pending fits merely because JEPA did not win. |

## Current RadioJEPA phase

| Track | Status at this snapshot | Evidence and limitation |
|---|---|---|
| **Zero-capable nested RF field** | All **15 fits × 1,920 updates**, **84 fresh actual RT cases**, **90 reused TRAIN/VAL cases**, **60 VAL checkpoint archives** and **5,928 prediction records** complete. The 578-source/2,770-evidence seal and independent passive replay pass. | [Results](../nested_radio/RESULTS.md), [architecture](../nested_radio/ARCHITECTURE.md), [report](../../artifacts/nested_radio/report.json). VAL selects `nested_joint`: sparse heard-RSSI RMSE **2.637330 dB**, versus validation-selected fixed-feature hierarchy **2.710522** and global-eight regression **2.651669**. All four adjusted primary intervals favor their candidates. All eight learned-local gates are exactly zero in all six nested members: retained shared physics and shrunken fixed Fourier features define the selected fitted function, not a rich learned RF representation. Agreement contributes only **0.002226 dB**; material-delta error essentially ties the numerical reference with a slight numerical advantage. **Zero new native simulations or hardware actions**, no service qualification or default promotion. |
| **Shared/AP-local physical field** | All **15 fits, 186 RT cases, 60 raw VAL archives, 4,416 forecast records and 34 native runs** complete. The 567-source/4,900-evidence seal and first independent replay pass. | [Results](../hierarchical_radio/RESULTS.md), [interpretation](../hierarchical_radio/INTERPRETATION.md). Selected hierarchy sparse RMSE 3.088402 dB exceeds global-24 2.995352 and numerical Map 2.951064; matched agreement improves by 0.034739 dB. Native hierarchy misses 1.278125%/2.778125% exceed full-history Map 1.165%/2.3725% at 0.03/4 s. Static conditioning, a larger inferred state and a modest agreement gain establish no temporal, physical-fleet or SOTA qualification. |
| **Adaptive reported-RF assay** | All **12 fits**, **174 RT cases**, TEST predictions/scoring, canonical raw replays and 504-source/11,792-evidence closure complete. VAL selects **regression**. Sparse condition-delta RMSE is 1.520106 versus RadioMap 1.654205 and RF-erased 1.885921 dB; the two adjusted RF-adaptation contrasts pass on 24 geometries. Independent raw replay also passes. | [Results](ADAPTIVE_RESULTS.md), [report](../../artifacts/adaptive_radio/report.json), [canonical audit](../../artifacts/adaptive_radio/verification/audit.json). Joint alignment is worse; pooled Bayes nearly ties; Brier benefit is unconfirmed. Static material states, known TRAIN-calibrated frontends and correlated direction/device reports do not establish dynamics, real transfer, packet reliability or SERVICE permission. |
| **Measured CSI** | Six fits, three numerical references, three recordings × 18 model/view outputs, independent scoring and sealed closure complete. | [Results](CSI_RESULTS.md), [interpretation](CSI_INTERPRETATION.md), [independent review](../../artifacts/channel_scene_review/review.json). Negative model result; exact pilot aliasing limits identifiability. No learned temporal dynamics or scatterer identification. |
| **Measured phone transfer** | Five held-out phones, 300 descriptive support/method/draw cases and independent audit complete. | [Results](../device_transfer/RESULTS.md), [review](../../artifacts/device_transfer_review/review.json). Classical localization/calibration; three support draws are not three pretrained neural models. TUJI lacks session timestamps; device/day effects are confounded. |
| **Native bridge** | Nine paired cases, **18 simulations**, passive audits and 512-source/56-evidence closure complete. | [NATIVE_RESULTS.md](NATIVE_RESULTS.md). Fixed seed-0 regression harms all four fast-handover episodes; slow-handover mean improves with two episodes harmed. The 9.74-second outlier and [context diagnosis](NATIVE_CONTEXT.md) remain evidence, not patched-away cases. |
| **Spatial support and client-local recovery** | Built causal report memory/source-band guard with queued server-delivery refresh, causal central handovers and separately preinstalled paid own-client recovery. All **48 fresh + two parity simulations**, 543-source/224-evidence seal and independent passive review complete. No new neural training or encoder-context change; ordinary numerical RadioMap history fitting continues. | [Architecture/results](SPATIAL_CONTROL.md), [all contrasts below](#spatial-control-contrasts). Support helps regression means at both handover costs; fast combined regression still loses to Map + recovery in all four episodes. Four-second recovery harms all four episodes. Fixed weights and extra admission history prevent JEPA-objective attribution. |
| **Geometry operator optimization** | Registered 48-public-plan profile and independent audit complete; exact operand bytes preserved. | [NATIVE_PERFORMANCE.md](NATIVE_PERFORMANCE.md): median 278→98 ms, **2.83× operand-call speedup** under shared CPU load. Frozen runner and all 18 native outcomes unchanged; no full-model, queue, service or SLA speedup measured. |
| **Sensing-design mathematics** | Target-nullspace and explicitly prior-conditioned operator tools implemented/tested. | [SENSING_DESIGN.md](SENSING_DESIGN.md). Mathematical observability/information tools, not an executed on-air acquisition policy or a calibrated fleet endpoint gain. |

## Nested-RF primary contrasts

Differences below are candidate minus reference in dB; negative favors the
candidate. All three member metrics and static states are averaged within each
of 24 fresh TEST geometries. The four-family adjusted central 98.75% intervals
use 20,000 paired whole-geometry bootstrap draws and remain conditional on the
three trained members. Both model and numerical-reference selection use VAL
heard-RSSI RMSE; the numerical choice is among all four frozen TRAIN-fit priors.

| Registered comparison | Mean RMSE difference, dB | Central 98.75% interval, dB |
|---|---:|---:|
| Selected nested joint − best numerical | -0.073192 | [-0.126150, -0.023013] |
| Selected nested joint − global-eight regression | -0.014339 | [-0.023057, -0.005348] |
| Nested regression − ungated hierarchy | -0.188225 | [-0.233787, -0.147843] |
| Nested joint − nested regression | -0.002226 | [-0.004345, -0.000883] |

The gate package changes initialization, feature scaling and effective-prior RMS
regularization together; this comparison does not isolate one cause. Its exact
global-only submodel is a same-remaining-weight function property, not a promise
of noninferiority to an independently trained baseline. The observed learned-local
shutdown is neither evidence of a rich learned representation nor whole-model
collapse: shared physical structure and fixed Fourier variation remain. Secondary
material-delta RMSE is **0.906549 dB** for the selected model and **0.906270 dB**
for the selected numerical reference. This near tie, the tiny agreement increment,
limited procedural geometry families and absence of packet/hardware evidence
bound the conclusion; specialist SOTA, learned temporal dynamics and 100-car
service benefit are unestablished. Original SERVICE cases remain closed.

## Spatial-control contrasts

All 48 fresh simulations use six fixed arms across four procedural episodes per handover cost, with 100 robots for 20 seconds after 60 seconds of warm history. Two separate disabled-extension runs check predecessor parity. All fresh arms use the revised causal central-command semantics; comparisons with the older native diagnostic do not isolate one intervention. The encoder, its 64-token selection and seed-0 regression checkpoint stay fixed. Support uses additional delivered-history admission information, while recovery is separately preinstalled own-client authority.

The table retains every registered contrast. Changes are candidate minus baseline; lower miss fraction and shorter gaps are better. Miss changes are **percentage points**, not relative percentages. Gap changes average episode maxima; harm counts are episodes out of four, not individual robots. These are descriptive results with no confirmatory p-values or production claim.

| Handover | Candidate − baseline | Mean miss change, pp | Mean maximum-gap change, ms | Miss harms / 4 | Gap harms / 4 |
|---|---|---:|---:|---:|---:|
| 0.03 s | regression − radiomap | +0.591250 | +5532.5 | 4 | 4 |
| 0.03 s | regression_support − regression | -0.528125 | -5247.5 | 1 | 0 |
| 0.03 s | regression_recovery − regression | -0.329375 | -3232.5 | 0 | 0 |
| 0.03 s | regression_both − regression | -0.556250 | -5462.5 | 0 | 0 |
| 0.03 s | radiomap_recovery − radiomap | +0.000000 | +0.0 | 0 | 0 |
| 0.03 s | regression_both − radiomap_recovery | +0.035000 | +70.0 | 4 | 2 |
| 4 s | regression − radiomap | -0.044375 | -1067.5 | 2 | 0 |
| 4 s | regression_support − regression | -0.015625 | -50.0 | 0 | 0 |
| 4 s | regression_recovery − regression | +0.268125 | +3195.0 | 4 | 4 |
| 4 s | regression_both − regression | +0.266875 | +3195.0 | 4 | 4 |
| 4 s | radiomap_recovery − radiomap | +0.222500 | +2127.5 | 4 | 4 |
| 4 s | regression_both − radiomap_recovery | +0.000000 | +0.0 | 3 | 0 |

At 0.03 s, support and local recovery improve regression’s mean separately and together; support alone still harms one episode on misses. The combined package misses 1.079375% versus 1.044375% for Map + recovery and loses on misses in all four episodes. At 4 s, support alone makes a small mean improvement without miss/gap harms; local recovery and both harm all four episodes versus regression, and Map recovery harms all four versus Map. Equal slow mean loss between regression + both and Map + recovery (1.629375%) conceals three episode miss harms for the former.

All per-episode misses, worst-decile loss, episode maximum silence, end-pending packets, observed-plus-pending bounds, video/bulk, realized scans and actual paid compute remain in the [complete report](../../artifacts/spatial_control/report.json). Shared sensing capabilities/cadence do not force identical realized observation histories. A support threshold is not calibrated risk; local recovery does not prove reachable safety. The original native negative result, measured-CSI loss and mixed E7 result remain unchanged.

The [canonical seal](../../artifacts/spatial_control/verification/manifest.json) binds **543 source files and 224 evidence files**, SHA-256 `f652af5c0c1671a978ad0b2ebefcf35d7e5a7474dc36b373d63b4f60aa41d28f`. The [independent passive review](../../artifacts/spatial_control_review/result.json) passed without scientific reruns or parser changes, SHA-256 `5f79df8dd60f4e45eb3ac300813a00d2c5a8fd8f724c4d380ce16fa2b2461df3`. It verifies all 48+2 runs, 1,920,000 fresh control releases, 376 support queries, 737 central commands/729 executions, 44 local handovers, 19,178 scan attempts and 19,325 previously named frozen files. These audited event counts do not multiply the four distinct procedural episode seeds or justify independent physical-building/reset evidence.

## Genuinely unfinished work and blockers

| Work | Classification and concrete next requirement |
|---|---|
| Original large-model replication | **Unfinished statistical/training work, requiring a separate licensed/budgeted campaign and protocol.** Original E1–E5 remain one seed; at least two further independent upstream training chains are needed for a three-run claim. Bind source, seed and checkpoint dependencies in separate run directories; do not reinterpret compact-model seeds or reuse only evaluation seeds. No additional original chain is running or queued as a restart here. |
| Exact E7 calibration promotion | **The matched diagnostic is complete with mixed results**, not an unfinished job to rerun until successful. [All 225 pairs](../../artifacts/e7_calplanning_review/paired_rows.csv) remain. Deployment/default promotion still lacks supported planning benefit, observation-only qualification and service evidence; later different-feature studies do not remove these limits. |
| didjital R6.1 integration | **Restoration and a minimal read-only software bridge are complete.** The original [qualification](../R6_COMPATIBILITY.md) covers commit `a458a90…`, 3+67+117 checks and synthetic demos. The new [R6 bridge](R6_BRIDGE.md) passes [77 source-bound checks](../../artifacts/r6_bridge/software_tests.json): actual isolated `plan_missions` parity, explicit units/service forecasts/owner registry, unqualified RF→DEFER and conservative offline journal views preserving UNKNOWN. Live journal migration, authenticated ownership/clock mapping, transition/reconciliation authority and physical integration remain unfinished; no backend is activated. Keep the two `wirelessjev` packages isolated. The private SmartCar source is still unavailable. |
| Current adaptive assay finalization | **Registered run and canonical closure complete.** Independent raw replay also passes; sealed Studio export is complete. No fit, TEST prediction or canonical replay remains queued; do not launch replacement seeds, retune on TEST or consume original SERVICE cases. |
| Nested-RF finalization and learned-local interpretation | **Finite phase, canonical closure, independent replay and local viewer complete.** No fit, fresh RT, TEST forecast or checkpoint score remains queued. Its static primary RF improvements do not complete learned temporal dynamics, native packet transfer or physical trials. All learned-local gates are zero in all six nested members; retain that diagnostic and the tiny agreement increment when defining any new representation experiment. A separately registered campaign is required before new fitting or held-out evaluation; no default is promoted. |
| Dynamic world model, calibrated service heads and wider controls | **Research/implementation backlog under the current plan**, detailed in [REMAINING.md](REMAINING.md): learned event-time/AP-region belief, query-aware encoder retrieval, action-supported packet/retry/deadline heads, calibrated tails and supported joint actions. Causal spatial memory, source-band support and own-client recovery software are already built and evaluated; their mixed results do not establish reliable service or learned dynamics. A new finite protocol must precede each additional experiment. |
| SmartCar archive, access and physical trials | **Hardware/access blocked.** The requested `SmartCar.zip` and deployed scripts are unavailable in the accessible cloud inputs; no pinned car identity/configuration or usable private-network route is supplied. The [one-car procedure](../ONE_CAR_TEST.md) defines the required access and supervision steps. Actual source audit, read-only capability/status, supervised switch/back, stage traces, scan cost, factory-password change and one-/small-/100-car qualification remain physical work. No credentials or endpoint are guessed and no radio was actuated. |
| Studio and presentation publication | **Local exports are built and browser-checked.** The existing 11-card [Studio export](STUDIO_EXPORT.md) includes adaptive results, measured CSI/phone, native/R6 and the mixed E7 result while preserving its legacy bundle. A separate [nested-RF viewer](../../artifacts/nested_radio_review/viewer/index.html) now presents all nine controls, four primary contrasts and actual public CAD; its 3D panels are explicitly selected-seed-zero descriptive views. Incorporating the new phase into the broader Studio/deck and external Claude republication remain unfinished. Local rebuilding is not external publication. |
| Protocol-agnostic/space/extra PHY extensions | **Excluded or deferred**, not hidden completion requirements. [PROTOCOL_BOUNDARY.md](../PROTOCOL_BOUNDARY.md) remains Wi-Fi-specific: no validated 5G/LTE/Bluetooth/satellite adapter. E6 stays excluded; universal robots/environments, satellite collision avoidance and custom on-air waveforms are not completed by this warehouse campaign. |
| Old delivered-observation campaign | **Superseded/unlaunched, not an active training queue.** Untracked `observation_frontier` code/protocol drafts are preserved and excluded from the tracked-suite proof. The newer archived RadioJEPA user plan governs current work; do not silently start that older campaign alongside it. |

Mac-specific token-expiry and backup-ref notes are not verified cloud tasks. Repeated user requests to continue do not turn missing hardware, a rejected release, an excluded E or a failed hypothesis into a completed result.
