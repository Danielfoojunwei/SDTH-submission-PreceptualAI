# Where JEPA has evidence, and where it still loses

**JEPA has promising native RF diagnostics, but the current final package does not beat RadioMap on command delivery.** This is a post-hoc audit of all 64 published confirmation cases, with no new worlds, model calls, fitting, timing or selection.

The [machine-readable diagnosis](../artifacts/jepa_advantage_diagnosis/diagnosis.json), [input/output receipt](../artifacts/jepa_advantage_diagnosis/receipt.json) and [audit script](../scripts/audit_jepa_advantages.py) preserve every stage, site and condition. All five earlier source closures and the published 376-source/665-evidence entries were checked before and after.

## Final-stack comparison on all eight sites

Differences are JEPA minus RadioMap. Intervals below are **unadjusted descriptive paired whole-site 95% intervals**, not new superiority tests or a way to select a winner. These 20,000 descriptive resamples use seed 20261105 and do not replace the original registered intervals or decisions. Different accepted query times, histories and map pipelines prevent interpreting RF rows as matched-input accuracy tests.

| Quantity | JEPA | RadioMap | Difference | Descriptive 95% interval | Site directions |
|---|---:|---:|---:|---|---|
| Command loss (%) | 3.810000 | 3.698021 | +0.111979 | [+0.038125, +0.193646] | 2 lower / 0 tied / 6 higher |
| Worst-decile loss (%) | 11.690625 | 11.552083 | +0.138542 | [-0.636458, +0.986458] | 3 lower / 0 tied / 5 higher |
| Original video (%) | 93.107925 | 93.236498 | -0.128573 | [-0.457935, +0.231662] | 6 lower / 0 tied / 2 higher |
| Bulk (fleet Mbps) | 474.745000 | 472.835770 | +1.909230 | [-6.132881, +12.718904] | 5 lower / 0 tied / 3 higher |
| Native SNR RMSE (dB) | 7.011824 | 8.261672 | -1.249848 | [-2.204598, -0.145408] | 6 lower / 0 tied / 2 higher |
| Native route-delta RMSE (dB) | 4.721072 | 4.887427 | -0.166355 | [-0.236111, -0.094858] | 8 lower / 0 tied / 0 higher |
| Native AP regret (dB) | 0.702601 | 0.762151 | -0.059550 | [-0.107009, -0.007715] | 6 lower / 0 tied / 2 higher |
| Measured forecast elapsed work (s/trial) | 13.825714 | 4.529198 | +9.296516 | [+8.790708, +9.779594] | 0 lower / 0 tied / 8 higher |

The strongest consistent RF pattern is route-delta error: JEPA is lower on all eight native site summaries. SNR RMSE and AP regret are lower on six. These are leads for a fair matched-query study, not proof that the JEPA objective is better. Its RF error is lower while its final command-loss mean is higher; prediction quality is not the service endpoint.

The bulk point mean is slightly higher for JEPA, but only three sites favor it and one large gain drives the average. The interval retains that uncertainty. Original-video and worst-decile point means favor RadioMap; their descriptive intervals retain uncertainty. The maximum over every robot and all eight sites is **8.37 seconds for both**; the mean of site maxima is a different quantity and does not create a new continuity claim.

## Stationary and drifting conditions

Even site IDs use the stationary case; odd IDs introduce hidden mean-field drift. This grouping follows the frozen source but is a **post-hoc four-site diagnostic**, not an independently qualified subgroup. All subgroup values and intervals remain in the JSON.

| Condition | Sites | JEPA loss (%) | Map loss (%) | JEPA native SNR RMSE | Map native SNR RMSE | JEPA route-delta RMSE | Map route-delta RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| stationary | 4 | 2.195833 | 2.097292 | 6.609084 | 7.502295 | 4.894390 | 5.033266 |
| drift | 4 | 5.424167 | 5.298750 | 7.414565 | 9.021050 | 4.547753 | 4.741587 |

Both groups have lower native RF point-error means for JEPA but higher command-loss means. The drift group therefore does not rescue a claim of superior dynamic control. Query/history differences remain in both groups.

## What the architecture offers, and what it does not establish

The current learned reference conditions on delivered fleet histories, geometry and planned route queries, and shares neural weights across robots/AP-band queries. Those components can represent nonlinear context and route-conditioned effects. Their practical value still needs task-level tests against numerical and matched-supervised controls.

This particular JEPA is a **hybrid** that continues fitting a RadioMap. Its map pipeline differs from the strong-prior numerical reference. It is neither map-free nor demonstrated intrinsically cheaper. The final JEPA package spends about three times the summed measured forecast elapsed work despite making fewer calls. RadioMap already has online assimilation and posterior uncertainty, so those properties are not exclusive JEPA advantages.

Earlier evidence is mixed rather than erased: a separate fast/idealized-observation hybrid study has a positive supervised primary and a favorable secondary JEPA point result; it does not prove JEPA-objective necessity. The compact context study fails its registered superiority comparisons, and the RT-backed response-identification JEPA loses to numerical Bayesian identification. Those cohorts cannot be pooled with this one.

## Concrete leads for new gains

1. **Decouple fast radio control from slow model arrivals.** The current controller updates native planning/control-link choices on forecast consumption. Latest admission makes the ensemble fresher but reduces its callbacks from 43 to 29 per trial; latest-only service worsens. Reusing a cached forecast with current pilots for a separately timed local control loop could retain prediction value without more neural forwards. The causal effect requires a new fixed comparison; existing faster MCS updates must not be presented as newly invented.
2. **Improve genuinely delivered observation density.** Associated-link pilot reports may provide positive measurements without full scans. Partial reports must not mark every unmeasured AP as censored, must preserve capture position/time/AP/band identity, and must pay reporting/availability costs. Numerical assimilation must receive the same evidence. A model must handle or be trained for the resulting mask distribution.
3. **Measure decision-relevant prediction on common inputs.** Route-delta and AP-relative contrasts are more directly used by control than a static absolute offset. Match history, map prior, query times and labels, then compare JEPA with a matched supervised model. Lower native error in this audit is not that test.

These are hypotheses for prospective work, not newly validated improvements. There is no physical-car, production, SOTA, objective-necessity, arbitrary-layout or indefinite-safety claim. No previous result, model or qualification rule changed.
