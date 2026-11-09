# Adaptive RF-condition results

The validation-selected **regression** model uses current RF observations to predict material-conditioned RF changes more accurately than the registered RadioMap-operator and RF-value-erased controls. This branch retains the neural physical-field architecture with cross-view agreement disabled. On 24 TEST layouts its sparse-context condition-delta RMSE is **1.520106 versus 1.654205 dB** for RadioMap, an **8.1% reduction**. The matched joint/JEPA branch is slightly worse than regression. A strong pooled Bayesian control is close at **1.524141 dB**; specialist SOTA and a unique JEPA benefit are not established.

**Audit status:** both canonical passive raw replays pass, including 540 DEV and 3,240 TEST prediction records. The completed seal binds **504 source files and 11,792 evidence files**. Its manifest SHA-256 is `912f543b363972b56c54c8210cb44ae79a80fa9220d2950cd5f0fc3a89a0fe2c`. The [independent raw replay](../../artifacts/adaptive_radio_review/result.json) also passes all 174 RT cases, 12 checkpoints, 3,780 forecast records and all four adjusted intervals. The registered run, canonical closure and independent numerical review are complete.

## Registered primary comparisons

All four comparisons use sparse A-context, 24 whole geometries and three trained members. Each member's metric is computed first, then averaged within geometry; predictions are not ensembled. The three RF conditions and correlated receiver views are nested within geometry. The 20,000-sample paired whole-geometry bootstrap uses seed 21749 and central **98.75% intervals**, adjusted for the four-comparison family. Intervals are conditional on these three trained members, not uncertainty over arbitrary future retraining.

| Candidate − reference | Metric | Candidate | Reference | Paired difference | Adjusted 98.75% interval | Result |
|---|---|---:|---:|---:|---|---|
| Selected regression − RF-value-erased | Condition-delta RMSE, dB | 1.520106 | 1.885921 | −0.365815 | [−0.691336, −0.129131] | RF-dependent gain supported |
| Selected regression − RadioMap-operator | Condition-delta RMSE, dB | 1.520106 | 1.654205 | −0.134099 | [−0.225948, −0.060057] | RadioMap contrast supported |
| Joint/JEPA − matched regression | Condition-delta RMSE, dB | 1.534382 | 1.520106 | +0.014275 | [+0.003849, +0.027271] | Joint branch is worse |
| Selected regression − RadioMap-operator | Detection Brier score | 0.033220 | 0.035162 | −0.001941 | [−0.006490, +0.001943] | Detector benefit unconfirmed |

The primary delta compares reported-RSSI changes for fixed condition pairs 0→1 and 0→2 at identical query indices, where both reports are heard. All 24 geometries support the endpoint. Censoring and pair availability remain separately recorded; an unheard report is not converted into an invented point value. `rf_erased` removes amplitude while retaining heard/censored status and availability; it is not a strict CAD-only control.

## Descriptive spatial and held-band results

These summaries include **current-context members only** (`stale == false`). Stale-context interventions are separate records and are not averaged into these values. Within each geometry, point RMSE and p95 pool heard queries across its three conditions. Learned cells average three trained-member metrics; each numerical Map/Bayes control contributes one result per geometry. The table then averages geometry metrics. It is not a globally pooled RMSE/p95 or a prediction ensemble. These metrics are distinct from the registered condition-delta endpoint.

| Current-context regime | Model | Mean layout point RMSE, dB | Mean layout p95 absolute error, dB | Condition-delta RMSE, dB |
|---|---|---:|---:|---:|
| Full, 24 A-report attempts | Regression | 3.262079 | 6.484892 | 1.452712 |
| Full, 24 A-report attempts | RadioMap-operator | 3.678918 | 7.158500 | 1.568139 |
| Full, 24 A-report attempts | Pooled Bayesian-operator | 3.352924 | 6.664532 | 1.443138 |
| Sparse, 8 A-report attempts | Regression | 3.359850 | 6.559538 | 1.520106 |
| Sparse, 8 A-report attempts | RadioMap-operator | 3.618041 | 7.195816 | 1.654205 |
| Sparse, 8 A-report attempts | Pooled Bayesian-operator | 3.412823 | 6.777803 | 1.524141 |

The separate **6.0 GHz held-band** assay uses sparse 2.4/5.2 GHz context and queries the even physical links. It does not enter training, branch selection or the DEV gate. Regression point RMSE is **3.512667 dB**, RadioMap **3.534810 dB**, and pooled Bayes **3.210488 dB**. Their condition-delta RMSE values are **1.364924 / 1.504331 / 1.368440 dB**, respectively. These descriptive values do not establish a held-band superiority claim. Native-bridge 6.5 GHz extrapolation is a different endpoint.

The supplementary frozen-transfer model also has lower sparse mean point RMSE (**3.322304 dB**) than selected regression. Branch selection and the registered primary claims concern condition-delta accuracy; they do not select a universal winner across controls or endpoints.

## Data, model and scope

The completed campaign has **12 fits** (four cells × seeds 0/1/2) and **174 actual Sionna RT cases**: 72 TRAIN, 18 validation, 12 DEV and 72 TEST. Each layout retains identical public CAD, positions, transmitter power and RT sampling seed while metal/wood/brick boundary responses change the physical solve. Material labels, condition identifiers, source filenames, private channel gain and true front-end offsets are excluded from model features.

The main TEST target ledger contains **9,431 heard + 937 censored + 0 missing = 10,368 reported query events**, including 6,240 both-heard paired-delta targets. Full and sparse regimes reuse those targets; training seeds do not multiply independent data. Three simulated TRAIN-calibrated frontends and both directions provide correlated measurements, not independent buildings or physical-device transfer. The held-band ledger is separate: 4,348 heard + 836 censored + 0 missing = 5,184 reports, with 2,851 paired-delta targets. Device-0 gauge and quantization/censoring assumptions remain explicit.

Full view A schedules 24 reports: six even poses × two APs × two observed bands. Sparse A schedules eight reports at the same two planned poses, 0 and 10. View B supplies the paired opposite-direction/device training view; main inference uses A only. Scheduled report availability is not a measured network-delivery or capture-time experiment. The model predicts an approximate static physical-field posterior with a bounded geometry correction. Changing static materials is not learned temporal dynamics, an action-conditioned transition, identified scatterers, or a packet/deadline/silence guarantee.

The appropriate evidence claim is **current-RF-dependent adaptation on this registered static simulated assay**. Joint alignment does not improve the matched result, the strong numerical control nearly ties the primary metric, and Brier improvement is unconfirmed. RadioMap remains the application default; original SERVICE layouts remain unopened. The [native diagnostic](NATIVE_RESULTS.md), [mixed E7 planner result](E7_CALPLANNING_RESULTS.md), [measured CSI result](CSI_RESULTS.md), [current architecture](CURRENT_SYSTEM.md) and [remaining prerequisites](REMAINING.md) retain their separate evidence boundaries.

## Provenance

- [Registered plan](../adaptive_radio/PLAN.md), [selection](../../artifacts/adaptive_radio/selection.json), [DEV gate](../../artifacts/adaptive_radio/development.json).
- [Official report](../../artifacts/adaptive_radio/report.json), [member/layout metrics](../../artifacts/adaptive_radio/results/test.json), [canonical passive raw audit](../../artifacts/adaptive_radio/verification/audit.json).
- [504-file source receipt](../../artifacts/adaptive_radio/registration/source.json) and [publication proof](../../artifacts/adaptive_radio/registration/git.json). Source receipt SHA-256: `4bb2415118970fad4a95bbd4575d7ef799e10242faa168329d24f358ed75c3fc`; protocol SHA-256: `dfb6c7da4e9665081dd6660b7a1fafaa93ac83a592af90411874b5d46ee28352`.
- Independent raw-review SHA-256: `ca59d75dfb44e507ec8427097cd3b27ee1cd8cce4ee1a10a0299c78ec55f9c5a`. The separate review invokes no model/fit/RT execution.
- The canonical audit reconstructs public inputs, targets, schedules, metrics and baseline priors with **zero model constructors, fits, forwards or RT solves**. The [final manifest](../../artifacts/adaptive_radio/verification/manifest.json) binds the completed closure. The separate E7 diagnostic has a 525-file source capsule; it is not the adaptive source count.

This page is outside all frozen scientific source collectors and changes no registered source, model or result.
