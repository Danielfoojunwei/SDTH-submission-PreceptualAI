# Trust-gated planning and safety broadcast: results

- **Protocol:** [NATIVE_GATE.md](NATIVE_GATE.md), frozen in `3c24c6a` before any test run.
- **Runs:** fresh test scenes 510–515, 100 robots, 300 s each.
- **Map error:** varies in space (a log-normal multiplier with RMS 1), unlike the earlier campaign's uniform error.
- **Raw data:** [`artifacts/native/gate_campaign.json`](../artifacts/native/gate_campaign.json).
- **Evidence class:** simulation only.

## Control misses (means over six scenes)
| Condition | Arm | Control misses | Worst-decile robot | p99 longest gap (ms) | Airtime | Per scene (per 10,000) |
|---|---|---|---|---|---|---|
| σ = 8 dB | mlo_split | 0.064% | 0.198% | 314 | 8.33 | 5.4, 7.7, 6.0, 6.8, 6.8, 6.0 |
| σ = 8 dB | native (ungated) | 2.640% | 10.261% | 16668 | 8.72 | 106–397 |
| σ = 8 dB | **native_gated (z = 2)** | **0.056%** | 0.199% | 333 | 7.93 | 4.1, 7.0, 4.8, 5.0, 5.1, 7.7 |
| σ = 8 dB | native_gated_z1 | 0.152% | 0.742% | 943 | 7.53 | 4.0–35.5 |
| σ = 4 dB | mlo_split | 0.064% | 0.198% | 314 | 8.33 | 5.4–7.7 |
| σ = 4 dB | native (ungated) | 0.237% | 1.926% | 2754 | 7.38 | 3.1–69.0 |
| σ = 4 dB | **native_gated (z = 2)** | **0.025%** | **0.106%** | **252** | 7.36 | 2.1–2.8 |
| σ = 4 dB | native_gated_z1 | 0.030% | 0.166% | 362 | 7.06 | 1.7–7.8 |
| σ = 4 dB, noisy 2.4 GHz | mlo_split | 0.554% | 1.370% | 1156 | 9.29 | 41.9–71.0 |
| σ = 4 dB, noisy 2.4 GHz | native (ungated) | 1.692% | 7.509% | 15883 | 8.35 | 43.0–350.9 |
| σ = 4 dB, noisy 2.4 GHz | **native_gated (z = 2)** | **0.240%** | **0.742%** | 1100 | 8.70 | 19.9–29.0 |
| σ = 4 dB, noisy 2.4 GHz | native_gated_z1 | 0.272% | 1.116% | 1099 | 8.01 | 15.3–61.6 |

`native_gated_overconfident` (scale 0.5) is mathematically identical to z = 1, because halving σ doubles z. Its rows match z = 1 exactly, so they are omitted above.

## Verdicts
| | Result | Verdict |
|---|---|---|
| G1 | Gated beats `mlo_split` at σ = 8 dB in 5/6 scenes | **Pass** (by a small margin: 0.056% vs 0.064%) |
| G2 | Gated is within 1.2× of ungated in 6/6 (trivially, since ungated fails), and beats `mlo_split` in 6/6 | **Pass** |
| C (exploratory) | With noisy 2.4 GHz, gated beats `mlo_split` in 6/6 (0.240% vs 0.554%) | Holds |
| B1 | `layered_0.5` is no worse than multicast in 0/6 scenes | **Fail** |

## What changed in our understanding
1. **When map error varies in space, ungated planning fails even at an average of 4 dB.**
   - The earlier campaign, which used uniform map error, overstated the native stack's robustness.
   - Ungated planning roams into confidently-wrong areas: 0.237% misses and 2.75 s worst gaps, worse than the fixed rule.
2. **The trust gate is what makes the planner deployable.** It beats the fixed rule in every condition:
   - σ = 4 dB: **2.6×** better (0.025% vs 0.064%), with 12% less airtime.
   - Noisy 2.4 GHz: **2.3×** better.
   - σ = 8 dB: it matches the fixed rule and does not fall below it.
   - Its worst-served robots improve too.
3. **The development-tuned threshold was wrong.** On development scene 400, z = 1 looked better. On the fresh test scenes the pre-registered z = 2 was better in every condition. Pre-registration prevented a tuning mistake.
4. **The fixed rule only works while 2.4 GHz is clean.** A +10 dB noise rise there raises its misses 9× (0.064% → 0.554%). In real warehouses, where 2.4 GHz is busy, adaptive planning is needed.

## Safety broadcast (no retries)
| Condition | Scheme | Mean miss | Worst-robot miss |
|---|---|---|---|
| σ = 4 dB | multicast_mcs0 | 6.4e-05 | 3.5e-03 |
| σ = 4 dB | multicast_x3 | 5.4e-05 | 3.2e-03 |
| σ = 4 dB | layered_0.2 | 3.7e-02 | 4.8e-02 |
| σ = 4 dB | layered_0.5 | 1.5e-04 | 1.9e-03 |
| σ = 4 dB | layered_1.0 | 1.6e-05 | 1.0e-03 |
| noisy 2.4 GHz | multicast_mcs0 | 8.8e-04 | 1.6e-02 |
| noisy 2.4 GHz | layered_0.5 | 6.0e-04 | 1.1e-02 |
| noisy 2.4 GHz | layered_1.0 | 4.1e-04 | 1.0e-02 |

B1 fails as pre-registered: at half-loaded downlink, the layered message misses more often on average than one MCS 0 multicast frame. The picture is mixed rather than flat, though:
- With a fully loaded downlink, layered beats multicast in both conditions.
- Its worst-robot miss is lower, even at f = 0.5.
- In the noisy 2.4 GHz condition, layered at f = 0.5 is also better on average.

**Multicast airtime cost.** At 1 Hz it is about 0.05% per AP, so layering saves almost nothing. The idea stays an SDR research lead, not a priority.

## Implications for RadioJEV and WirelessJEV
- **RadioJEV must provide calibrated uncertainty, not just predictions.** The gate depends on knowing *where* the map is wrong. Calibration should be a training objective, checked with held-out coverage.
- **WirelessJEV's core skill is selective autonomy.** It plans where confident and defers to robust rules elsewhere. This is the first configuration that beats the fixed rule in every condition tested.

## Limits
- Simulation only.
- The planner is given the *true* local error std. A real RadioJEV would estimate it, and miscalibration would matter. The overconfident arm (equivalent to z = 1) loses in every condition.
- Roam costs are assumed.
- The 2.4 GHz noise rise is a single value.
- Six scenes.
