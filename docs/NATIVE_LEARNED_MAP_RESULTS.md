# RadioMap v0 drives the trust gate: results

- **Protocol:** [NATIVE_LEARNED_MAP.md](NATIVE_LEARNED_MAP.md), frozen in `182563f` before the test run.
- **Runs:** fresh scenes 520–525, 100 robots, 300 s each.
- **Raw data:** [`artifacts/native/learned_campaign.json`](../artifacts/native/learned_campaign.json).
- **Evidence class:** simulation only.

## The learned map
| Survey | RMSE (dB) | Bias (dB) | Predicted std (dB) | 68% coverage | 90% coverage (per scene) |
|---|---|---|---|---|---|
| 60 s | 3.80 | −1.86 | 2.18 | 0.43 | 0.67 (0.58–0.73) |
| 300 s | 3.50 | −1.96 | 1.73 | 0.35 | 0.57 (0.49–0.64) |

- **Accuracy is good.** It is better than corrected E7's audible-link error of about 4.0 dB, on a different twin, so not a like-for-like comparison.
- **The uncertainty is overconfident.** It is worse than on the development scene (0.72–0.83) and gets worse with *more* data, because the posterior shrinks while the structural error does not.
- **Bias.** About −1.9 dB, because the map learns experienced signal, which includes the fading mean.

## Control misses (means over six scenes)
| Condition | Arm | Control misses | Worst-decile robot | p99 gap (ms) | Airtime | Per scene (per 10,000) |
|---|---|---|---|---|---|---|
| 60 s survey | mlo_split | 0.068% | 0.205% | 370 | 7.62 | 6.1–7.3 |
| 60 s survey | native_learned (ungated) | 0.017% | 0.134% | 468 | 6.44 | 0.9–2.4 |
| 60 s survey | **gated_learned** | **0.013%** | **0.064%** | **208** | 6.38 | 1.2–1.5 |
| 300 s survey | mlo_split | 0.066% | 0.198% | 260 | 7.71 | 5.6–8.0 |
| 300 s survey | native_learned (ungated) | 0.025% | 0.210% | 394 | 6.61 | 1.1–5.7 |
| 300 s survey | **gated_learned** | **0.015%** | **0.091%** | 214 | 6.55 | 1.0–1.7 |
| 300 s, noisy 2.4 GHz | mlo_split | 0.582% | 1.430% | 1531 | 8.56 | 43.5–70.6 |
| 300 s, noisy 2.4 GHz | native_learned (ungated) | 0.301% | 1.426% | 3329 | 7.34 | 22.5–35.6 |
| 300 s, noisy 2.4 GHz | **gated_learned** | **0.171%** | **0.737%** | 1090 | 7.28 | 13.1–23.6 |

The ×1.5-inflated gate performed about the same as the calibrated gate (within ±0.003 percentage points in every condition).

## Verdicts
| | Result | Verdict |
|---|---|---|
| L1 | Gated beats `mlo_split` at a 300 s survey in 6/6 scenes (4.4× fewer misses) | **Pass** |
| L2 | Gated beats `mlo_split` at a 60 s survey in 6/6 (5.2×) | **Pass** |
| L3 | With noisy 2.4 GHz, gated beats ungated in 6/6 (0.171% vs 0.301%) | **Pass** |
| L4 | 90% coverage = 0.62 (required at least 0.70) | **Fail** |

## What it means
1. **The full loop works end to end without being told the answer.** Robots survey with capped scans, RadioMap learns a map and its uncertainty, and WirelessJEV plans with the trust gate.
   - It beats the fixed rule **4–5×** in clean conditions and **3.4×** when 2.4 GHz is noisy, with 15% less airtime.
   - Its worst-served robots improve **2–3×**.
2. **One minute of driving is enough.** A 60 s survey did as well as 300 s, and slightly better on control misses.
3. **The gate helps even with overconfident uncertainty.** Its job is to rank *where* the map is less reliable, not to state exact probabilities. The ranking survives miscalibration.
4. **Calibration is the open problem.** Coverage falls as the survey grows (0.67 → 0.57). The discrepancy term is a single global constant. A location-dependent discrepancy, or conformal calibration on held-out robots, is the next fix. It matters because a future planner that uses probabilities directly would be misled.

## Limits
- Simulation only.
- The survey's fading and blockage model is separate from, but similar to, the evaluation's hidden processes.
- AP positions and building geometry are assumed known from the CAD.
- The noise floor is assumed known per band.
- Six scenes, one realisation each.
