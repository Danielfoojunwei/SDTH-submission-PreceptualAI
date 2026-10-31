# Full-window native packet results (v4)

The campaign completed on **31 October 2026 at 17:50 Singapore time**: 37 development and 21 fresh-test native simulations, six independently fitted neural checkpoints (three seeds per architecture), a linear specialist, and a verified archive of 214 reports plus three source capsules. No training worker remains active.

These are ns-3 simulations of 100 robots with thirty-second decisions. They do not qualify a physical fleet. The protocol is in [WAREHOUSE_NATIVE.md](WAREHOUSE_NATIVE.md), with raw evidence under [artifacts/warehouse_native](../artifacts/warehouse_native/).

| Method | Utility | Mean service | Timely critical demand | Worst-decile service | Forecast RMSE |
|---|---|---|---|---|---|
| Stay | 0.6631 | 0.7351 | 0.5910 | 0.1093 | — |
| Budgeted map signal | 0.6872 | 0.7630 | 0.6114 | 0.1781 | — |
| Numerical | 0.6980 | 0.7688 | 0.6273 | 0.1650 | 0.4127 |
| Linear residual | 0.7015 | 0.7716 | 0.6314 | 0.1913 | 0.2074 |
| MLP, three seeds | 0.7010 | 0.7709 | 0.6312 | 0.1920 | 0.1573 |
| AP/channel graph, three seeds | 0.7010 | 0.7709 | 0.6312 | 0.1920 | 0.1592 |

Utility equally weights service and critical demand delivered within fifty milliseconds. Nominal-demand denominators include the assumed four-second application mute. These quantities differ from historical E1/E2 regret and the v3 generated-packet miss fraction.

## What improved and what did not

Native supervision reduces neural forecast RMSE by about 62% relative to the numerical prior. Mean learned utility rises from 0.6980 to 0.7010; worst-decile service rises from 0.1650 to 0.1920 in aggregate. These mean differences do not establish a reliable control advantage.

The neural utility difference is **+0.0030**, with a crossed training-seed/scene 95% interval of **[−0.0078, +0.0139]**. All six neural models chose identical plans on the six test scenes, despite different fitted weights and forecast errors. The linear specialist scores 0.7015, slightly above the neural methods. There is no graph-model control advantage over that specialist.

**Zero of six neural candidates passed the frozen validation gate; the linear specialist also failed.** The numerical planner remains the research default. Most validation candidates improved only one scene and tied the other two, leaving the lower improvement bound at zero; graph seed 1 regressed. Test diagnostics also show a critical-service regression in one scene. Test results did not reopen selection or change the thresholds.

## Critical-floor risk is not validated

The independent post-run audit recomputed every native endpoint target and counted labels once per unique scene/plan. train: 2600 miss / 0 met, validation: 1100 miss / 0 met, test: 2100 miss / 0 met. The aggregate critical floor ratio is 1.0 throughout. Every case misses that floor.

Consequently, the reported zero neural Brier/ECE equals the performance of a constant always-miss predictor. It does **not** demonstrate event discrimination, meaningful risk calibration or safe service. The original numerical outputs remain unchanged; [postrun_audit.json](../artifacts/warehouse_native/postrun_audit.json) records this limitation separately. A future risk experiment needs application-grounded targets and predeclared conditions containing both outcomes; this holdout must not be reused to tune a replacement threshold.

## Remaining scope

There are six test environments and one packet realisation per unique action. The route, log-distance map and radio configuration are known declarations. Four shortlisted plans do not cover the global action space. Per-robot improvements are not guaranteed by a fleet average.

Associations are fixed inside each native process; the application mute does not simulate WPA, DHCP or persistent closed-loop roaming. Racks/materials, scan cost and actual handovers still require calibration and hardware trials. Scoring timings use already prepared features and exclude candidate construction, sensors and actuation; they do not establish an end-to-end compute advantage. The implementation remains Wi-Fi-specific and no SOTA/foundation-model claim follows from this campaign.
