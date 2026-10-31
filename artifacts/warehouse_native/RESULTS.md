# Warehouse native packet results (v4)

Actual ns-3 packet simulations with 100 robots, a full thirty-second decision window, eight training scenes, three validation scenes and six new test scenes. Three independent training seeds per neural architecture. No physical car or persistent roaming loop was tested.

| Method | Utility | Mean service | Timely critical demand | Worst-decile service | Candidate regret | Forecast RMSE |
|---|---|---|---|---|---|---|
| numerical | 0.6980 | 0.7688 | 0.6273 | 0.1650 | 0.0069 | 0.4127 |
| ridge | 0.7015 | 0.7716 | 0.6314 | 0.1913 | 0.0034 | 0.2074 |
| mlp-0 | 0.7010 | 0.7709 | 0.6312 | 0.1920 | 0.0039 | 0.1599 |
| mlp-1 | 0.7010 | 0.7709 | 0.6312 | 0.1920 | 0.0039 | 0.1552 |
| mlp-2 | 0.7010 | 0.7709 | 0.6312 | 0.1920 | 0.0039 | 0.1567 |
| graph-0 | 0.7010 | 0.7709 | 0.6312 | 0.1920 | 0.0039 | 0.1585 |
| graph-1 | 0.7010 | 0.7709 | 0.6312 | 0.1920 | 0.0039 | 0.1571 |
| graph-2 | 0.7010 | 0.7709 | 0.6312 | 0.1920 | 0.0039 | 0.1619 |
| stay | 0.6631 | 0.7351 | 0.5910 | 0.1093 | 0.0419 | — |
| budgeted_map_signal | 0.6872 | 0.7630 | 0.6114 | 0.1781 | 0.0177 | — |
| numerical_tail | 0.6936 | 0.7652 | 0.6220 | 0.1873 | 0.0113 | — |
| candidate_oracle | 0.7049 | 0.7746 | 0.6353 | 0.1875 | 0.0000 | — |

Validation-selected research default: `numerical`. Raw research-model scores do not override failed gates. The candidate oracle is a retrospective upper bound within four proposals, not a global optimum.

Utility is half mean service plus half critical demand delivered within fifty milliseconds. Denominators include nominal demand during the assumed four-second application mute. These metrics differ from historical E1/E2 regret and v3 generated-packet miss fractions.

The map, PHY configuration and future route are known simulation declarations. There are no RF scan inputs. Propagation lacks racks/material calibration; association, WPA and DHCP are not simulated as actual handovers. Six scenes and one packet realisation per candidate give exploratory evidence. See [the protocol](../../docs/WAREHOUSE_NATIVE.md) and the archived raw reports, checkpoints and source capsules.

Post-run interpretation and constant-risk-label limitations: [final analysis](../../docs/WAREHOUSE_NATIVE_RESULTS.md). Raw results and frozen selection remain unchanged.
