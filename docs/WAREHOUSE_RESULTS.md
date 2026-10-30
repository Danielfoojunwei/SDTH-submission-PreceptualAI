# Warehouse connectivity results

Snapshot: 2026-10-30T12:16:05.012571+00:00 (UTC).

Scope: WirelessJEV and RadioJEV warehouse connectivity, E1–E5 and E7. E6 is excluded. Router, E8, foundation-model, custom-PHY and space extensions are deferred.

All nine E1–E5 controller stages completed and passed report/checkpoint validation.

These are synthetic, single-training-seed results. Case-level confidence intervals do not measure independent-training variation or scene clustering. No physical-car trial is claimed.

## E1: mission-conditioned plan selection

| Arm | Test regret ± 95% CI (n=200) | New-mission regret ± 95% CI (n=240) | Test latency |
|---|---|---|---|
| radio_lora | 0.1068 ± 0.0240 | 0.0721 ± 0.0136 | 4309.93 ms |
| radio_frozen | 0.1069 ± 0.0228 | 0.0837 ± 0.0194 | 4621.62 ms |
| intent | 0.0140 ± 0.0066 | 0.0251 ± 0.0074 | 467.90 ms |
| no_llm | 0.0004 ± 0.0002 | Not evaluated | 2.15 ms |

## E2: missions beyond the fixed-weight reader interface

| Arm | Test normalized regret ± 95% CI (n=120) | New-mission normalized regret ± 95% CI (n=120) | Test latency |
|---|---|---|---|
| radio_q | 0.1170 ± 0.0370 | 0.0779 ± 0.0253 | 6420.15 ms |
| e1_zero | 0.2905 ± 0.0594 | 0.3288 ± 0.0693 | 7390.47 ms |
| intent | 0.1987 ± 0.0461 | 0.3482 ± 0.0674 | 591.41 ms |

E1 and E2 use different metrics. The E2 baseline is the current weight-based reader, not a structured parser with explicit constraints.

## E3: multitask

Association normalized regret: 0.1170 → 0.1046.

DSA greedy reward: 1.000 over one 500-step evaluation episode. This does not establish robustness across channel layouts.

| Traffic task | Before | After | Rule |
|---|---|---|---|
| trend | 0.400 | 0.369 | 0.708 |
| event | 0.446 | 0.231 | 0.738 |
| volatility | 0.262 | 0.338 | 0.923 |

Admission labels: train {'lower': 106} (n=106), test {'lower': 54} (n=54). Test majority-class accuracy is 1.000; the one-class dataset does not establish admission-policy generalisation.

## E4/E5: site adaptation and forgetting

| Arm | Mean normalized regret | Worst site | Signed log verifies |
|---|---|---|---|
| local | 0.1820 | 0.2023 | True |
| fedavg_all | 0.1781 | 0.2220 | True |
| swarm_pers | 0.1820 | 0.2023 | True |

| Site | Initial E3 | Local | FedAvg | Personalised swarm |
|---|---|---|---|---|
| S1_normal | 0.1834 | 0.2023 | 0.1774 | 0.2023 |
| S2_heavy_scanners | 0.1726 | 0.1875 | 0.1599 | 0.1875 |
| S3_dense_racks | 0.1525 | 0.1597 | 0.1529 | 0.1597 |
| S4_light_racks | 0.1833 | 0.1783 | 0.2220 | 0.1783 |

Forgotten site: S2_heavy_scanners.

| State | Membership-inference AUC | Retained normalized regret |
|---|---|---|
| Before | 0.6167 | 0.1801 |
| Retrain without site | 0.6231 | 0.1962 |
| Retained-data distillation | 0.6489 | 0.2068 |
| Initial model | 0.6231 | — |

Elapsed costs: retraining 8478.7 s; distillation 1687.7 s.

AUC measures one loss-based membership attack, not certified data deletion. The historical `gated_distill` field names are preserved; the implemented student distils retained-site outputs without a separate final release gate. E5's stated scope is the shared component; personalised retained-site states are reused for its retained-service evaluation.

## E7: corrected environment learning

Completed on 15 environments, five missions per environment.

| Survey | Audible RMSE (dB) | Plan gap to truth search |
|---|---|---|
| 1 min | 4.042 | 0.202 |
| 5 min | 3.948 | 0.182 |
| 15 min | 3.989 | 0.159 |

The severe historical censoring regression is corrected, but additional data does not improve every environment. Truth search is a local-search reference. The optional map-only calibration study is not promoted to the planner.

## Evidence and remaining limits

Raw cloud reports and provenance: `artifacts/radiojev_cloud/`. Original Mac reports remain unchanged in `artifacts/radiojev/`; original E7 evidence remains in `artifacts/envlearn/`.

Independent training seeds, a meaningful admission dataset, calibration planning validation, real-car qualification and external Claude Studio publication remain separate work. Completing the specified experiment run does not establish those claims.
