# E7 calibration and planner utility

All **15 previously opened simulated environments**, **90 map fits** and **525 bounded planner searches** completed. The registered saved-arithmetic replay and an independent result reducer passed. This finishes the fixed calibrated-map-to-planner diagnostic; it does not qualify a production controller.

The geometry-and-censor calibrated map changes mean analytical mission utility by **+0.031479** / **-0.010718** / **-0.002326** at the nominal 1/5/15-minute tail-row settings. Higher is better. The table retains both environment-level and individual mission-pair harms.

Map accuracy improves at every setting. Planning gains are mixed: mean utility falls at the 5- and 15-minute tail settings, with 7–9 of 15 environments harmed at each setting. These results do not establish a consistent planning benefit or support production promotion.

| Nominal tail setting | Mean corrected-v2 utility | Mean calibrated utility | Paired mean delta | Environment harms / 15 | Mission-pair harms / 75 |
|---|---:|---:|---:|---:|---:|
| 1 min | 1.721519 | 1.752999 | +0.031479 | 7 | 33 |
| 5 min | 1.793632 | 1.782914 | -0.010718 | 7 | 32 |
| 15 min | 1.804391 | 1.802065 | -0.002326 | 9 | 38 |

Each primary delta averages the five missions within one environment before averaging the 15 environments. Harms use the registered strict negative-utility-delta rule. All **225 paired outcomes**, including every harm and every signed reference gap, are in [paired_rows.csv](../../artifacts/e7_calplanning_review/paired_rows.csv); the original per-environment receipts remain authoritative. There is no robot-level or mission-level independence claim.

| Environment seed | 1-min tail delta | 5-min tail delta | 15-min tail delta |
|---|---:|---:|---:|
| 15 | +0.019691 | +0.061481 | +0.026502 |
| 16 | -0.059638 | +0.135775 | -0.049923 |
| 17 | +0.014677 | -0.013578 | -0.021870 |
| 18 | +0.392158 | -0.395917 | -0.008403 |
| 19 | +0.009686 | +0.000008 | +0.000847 |
| 20 | +0.136599 | +0.058729 | +0.192519 |
| 21 | -0.018903 | -0.020563 | -0.038459 |
| 22 | +0.114089 | +0.008764 | -0.016470 |
| 23 | -0.058110 | +0.032975 | -0.066100 |
| 24 | +0.013579 | -0.009375 | -0.014964 |
| 25 | -0.015402 | -0.081412 | +0.005733 |
| 26 | -0.034294 | -0.008469 | -0.028352 |
| 27 | -0.043005 | +0.040632 | -0.023188 |
| 28 | +0.030932 | +0.035349 | +0.004134 |
| 29 | -0.029868 | -0.005167 | +0.003109 |

## Map prediction and the planning reference

| Nominal tail | Map | Audible RMSE, dB | Inaudible RMSE, dB | All-link RMSE, dB |
|---|---|---:|---:|---:|
| 1 min | corrected_v2 | 3.970103 | 18.143398 | 14.141412 |
| 1 min | geometry_censored | 3.925554 | 14.859901 | 11.789254 |
| 5 min | corrected_v2 | 3.766661 | 16.102044 | 12.679902 |
| 5 min | geometry_censored | 3.753968 | 13.779736 | 11.042702 |
| 15 min | corrected_v2 | 3.960098 | 15.711471 | 12.518741 |
| 15 min | geometry_censored | 3.940399 | 14.226637 | 11.474654 |

The 270 regenerated map-RMSE values match the retained optional calibration report with maximum absolute difference **0 dB**. This is a passive reproducibility check, with no extra fits or searches.

The truth-informed reference is the registered 40-iteration local search, not a guaranteed global optimum. Signed gaps are preserved, including negative values where a selected plan outperforms that bounded reference. It uses the same reference once per environment/mission across all three tail settings.

## Scope and evidence

This is **ORACLE-CONDITIONED and DESCRIPTIVE**, on already opened calibration seeds 15–29. Both arms share truth-derived top-eight visibility masks, contention/interference metadata, adjacent-channel leakage, protected-channel flags and starting plans. Map fitting receives only reported RSSI/censor bounds, AP geometry/band/power, assumed-known height and nominal rack geometry; hidden propagation and true RF are separate scoring inputs. External-interference estimation from MCS floors and availability estimation from busy reports use the identical inherited algorithm in both arms.

The nominal 1/5/15-minute budgets are nested flattened **tail-row fractions** of one maximum-length survey. They are not chronological prefixes or measured elapsed-acquisition guarantees. The endpoint is the original two-band, three-snapshot analytical association-planning utility with a 30-second horizon and four-second handover charge. It is not packet simulation, ns-3, closed-loop service qualification, a calibrated risk head, an SLA or hardware evidence. Original reports, production defaults and original SERVICE holdouts remain unchanged.

Both maps use the pinned D=200 random features and kappa=0. Retained map checkpoints replay fitted means; they do not claim calibrated posterior spread. The finite run used two numerical workers with BLAS/OMP/MKL thread limits one. No interrupted attempt was repeated and no new variant was selected from these results.

Actual recorded searches: **525**; total search iterations: **10,445 / 21,000** maximum; evaluated plan rows: **7,974,560 / 16,927,575** maximum. The 90 fits took 168.1 summed worker-seconds; searches took 2562.9 summed worker-seconds. These are diagnostic execution timings, not an inference-speed comparison.

- [Registered plan](../e7_calplanning/PLAN.md), [protocol](../../artifacts/e7_calplanning/protocol.json), and [source receipt](../../artifacts/e7_calplanning/registration/source.json).
- Published source commit: `8549a00e4f7148949109bba63a2a17befb22ec93`; [publication proof](../../artifacts/e7_calplanning/registration/git.json).
- [Aggregate report](../../artifacts/e7_calplanning/report.json), [passive audit](../../artifacts/e7_calplanning/verification/audit.json), and [sealed manifest](../../artifacts/e7_calplanning/verification/manifest.json).
- [Independent reducer](../../artifacts/e7_calplanning_review/independent_aggregate.py), [independent audit](../../artifacts/e7_calplanning_review/audit.json), and [all 225 pairs](../../artifacts/e7_calplanning_review/paired_rows.csv).
- Closure binds **525 source files** and **382 evidence files**. Source receipt SHA-256: `7559bda6e136e39bb9befb74cc68d1ca9595823f6b7376ec1b843999784d46df`; protocol SHA-256: `adf503f90507043759ecf76c62fd9c720bbe9ba87a60c1ba2afbc9e2a12662da`.
