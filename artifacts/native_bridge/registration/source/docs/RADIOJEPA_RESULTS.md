# RadioJEPA measured-view pilot

These are measured factory-channel array-power results. They are not SmartCar command-service results or a global SOTA claim.

The RadioMap application default and all production qualification flags remain unchanged.

| Model | Factory RF RMSE, dB (mean ± run SD) | Frozen XY probe error, m |
|---|---:|---:|
| joint_none | 2.900 ± 0.174 | 2.524 |
| joint_sigreg | 2.908 ± 0.329 | 2.443 |
| joint_vicreg | 2.650 ± 0.135 | 2.397 |
| temporal_sigreg | 2.748 ± 0.150 | 2.458 |
| regression_sigreg | 2.732 ± 0.203 | 2.476 |
| unstructured_sigreg | 3.158 ± 0.118 | 2.984 |

Three independently trained members per cell; member metrics are averaged, predictions are not ensembled. The split has 519 common training rows, 130 validation rows and 155 spatially held-out factory rows.

The report also retains both held-out recordings, constant/linear controls, projector and backbone ranks, cross-view alignment, and pooled/token/power interventions. All metrics use referenced channel power, not dBm or SINR.

PPR remains ineligible for this power-only experiment: phases/path factors are not identified. The 24 service-confirmation geometry specifications remain unopened.

The first planner implementation reproduces RadioMap utility; it does not qualify a new learned outage head. Multi-step dynamics, actual device onboarding and matched native packet-service qualification are further requirements.

[Raw report](../artifacts/radiojepa/report.json) · [Execution scope](RADIOJEPA_EXECUTION.md) · [User plan](plans/RADIOJEPA_USER_PLAN.md)
