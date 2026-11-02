# Admission applies to both planned and reactive handovers

100 simulated robots per site, 60 seconds per run, four-second handovers, 90 ms scan outages and an additional 500 ms compute delay. These are assumed hardware timings, not device measurements.

Six fresh sites compare the same fixed physical-outage rule on planned proposals alone versus both planned and reactive proposals. No model or threshold was fitted.

| Forecaster | planned_only loss (%) | all_handovers loss (%) | Selected/all bulk Mbps | Selected/all handovers per run |
|---|---:|---:|---:|---:|
| Optimised RadioMap | 8.948 | 6.021 | 470.25 | 58.2 |
| Previous JEPA | 9.595 | 7.096 | 489.72 | 85.5 |
| Three-member hybrid | 8.987 | 5.998 | 470.72 | 56.2 |
| Distilled model | 8.851 | 5.920 | 462.44 | 50.0 |

Primary paired difference: -29311.11 ppm, nominal 95% site-bootstrap interval [-34759.72, -24737.50]. The interval excludes zero.

## Mechanism and trade-offs

The physical-outage budget estimates failure area over thirty seconds with the archived ns-3/Nakagami surrogate. It admits a planned switch only when the estimated loss from staying exceeds four seconds of certain outage plus candidate loss afterward. This is an approximate admission heuristic, not a risk certificate.

The Boolean confidence gate chooses whether the planner or reactive rule proposes a move. The new admission hook separately filters proposals from both paths. It can also withhold a necessary emergency switch when the forecast is wrong, so live risk monitoring and hardware qualification remain required.

Sensing cadence changes both data availability and scan outage. The forecasters see masked real scan streams at the selected cadence, not dense scans at a sparse-scan cost. The FIFO uses the already frozen cadence-specific processing profile.

## Limits

Synthetic canonical warehouse, known CAD/localisation/noise, idealised scanner phases and attached-link measurements, fixed robot trajectories and mean-field training. Scan-report transport availability is not coupled to each policy's radio outages. The expected-failure surrogate omits full MAC/queue interactions; actual packet outcomes are reported separately. High remaining loss is a failed deployment condition, not a successful physical-car trial.

Compare percentages only within this cohort. The earlier 30 ms and slow-hardware cohorts use different seeds. A lower miss rate can cost throughput; bulk, video, handover counts and stop time are all archived in `summary.json`.
