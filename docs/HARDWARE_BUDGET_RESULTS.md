# Slow-hardware sensing and handover budget

100 simulated robots per site, 60 seconds per run, four-second handovers, 90 ms scan outages and an additional 500 ms compute delay. These are assumed hardware timings, not device measurements.

Four development sites selected cadence and gate independently for every reference. Four additional sites test the frozen configurations.

| Forecaster | default loss (%) | selected loss (%) | Selected/all bulk Mbps | Selected/all handovers per run |
|---|---:|---:|---:|---:|
| Optimised RadioMap | 14.426 | 7.593 | 493.06 | 103.8 |
| Previous JEPA | 15.985 | 8.431 | 491.00 | 116.0 |
| Three-member hybrid | 15.143 | 7.497 | 492.83 | 103.5 |
| Distilled model | 14.849 | 7.419 | 492.99 | 100.8 |

Primary paired difference: -1739.58 ppm, nominal 95% site-bootstrap interval [-4306.25, 216.67]. The interval spans zero; superiority is not established.

## Mechanism and trade-offs

The physical-outage budget estimates failure area over thirty seconds with the archived ns-3/Nakagami surrogate. It admits a planned switch only when the estimated loss from staying exceeds four seconds of certain outage plus candidate loss afterward. This is an approximate admission heuristic, not a risk certificate.

The selected experiment keeps the original reactive fallback. Consequently, a rejected planned switch can still be followed by a reactive handover. The separate all-handover experiment tests that interface distinction.

Sensing cadence changes both data availability and scan outage. The forecasters see masked real scan streams at the selected cadence, not dense scans at a sparse-scan cost. The FIFO uses the already frozen cadence-specific processing profile.

## Limits

Synthetic canonical warehouse, known CAD/localisation/noise, idealised scanner phases and attached-link measurements, fixed robot trajectories and mean-field training. Scan-report transport availability is not coupled to each policy's radio outages. The expected-failure surrogate omits full MAC/queue interactions; actual packet outcomes are reported separately. High remaining loss is a failed deployment condition, not a successful physical-car trial.

Compare percentages only within this cohort. The earlier 30 ms and slow-hardware cohorts use different seeds. A lower miss rate can cost throughput; bulk, video, handover counts and stop time are all archived in `summary.json`.
