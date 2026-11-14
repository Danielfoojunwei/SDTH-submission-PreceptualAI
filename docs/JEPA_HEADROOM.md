# Headroom ladder: how much could any learned radio model gain? (pre-registered)

**Why.** Across Codex's JEPA and world-model studies, learned models forecast
RF better than RadioMap but do not control better. Before building another
model, this diagnostic measures the ceiling: the same controller is given
oracle information in rungs. Whatever rung 4 achieves, no forecaster can beat.

## Rungs (same controller, same scenes, same hidden draws)
| Rung | What the controller knows |
|---|---|
| 1 | RadioMap learned from a 60 s survey, trust-gated (today's system) |
| 2 | The perfect mean signal field (no map error) |
| 3 | Rung 2 plus the *current* blockage state of every robot–AP path |
| 4 | Rung 3 plus fading and blockage for the next 1 s |

In rungs 3–4 the oracle picks, every 100 ms, the AP whose worst case over the
known window is best (5 dB hysteresis), and the second path likewise; the
normal planner's roaming is switched off so the two do not fight.

## Runs
- Development scenes 400–403, 100 robots, 120 s, realistic and stress radio.
- Two radio configurations: single radio, and two AP paths (second BE200).

## Kill criterion, fixed before running
For each radio configuration, pooled over scenes in the realistic condition:
if rung 4 cuts missed commands by **less than 25%** versus rung 1, learned
forecasting for *control* is stopped for that configuration. JEPA work then
continues only for RF mapping and deployment.

Where the ceiling is high, the gap between rungs says what to learn: rung 1→2
is static map accuracy; 2→3 is knowing blockage now (estimable from live
telemetry); 3→4 is predicting the next second.

Results are appended below after the run.

## Results (development scenes 400–403, 960,000 control packets per cell)
Raw data: [`artifacts/native/headroom_ladder.json`](../artifacts/native/headroom_ladder.json). Misses per million.

| Radio | Condition | Rung 1: learned map | Rung 2: perfect map | Rung 3: + blockage now | Rung 4: + next 1 s | Rung 4 vs 1 |
|---|---|---|---|---|---|---|
| Single | Realistic | 267.7 | 244.8 | 91.7 | 83.3 | −69% |
| Single | Stress | 2,691.7 | 2,071.9 | 997.9 | 363.5 | −86% |
| Two AP paths | Realistic | 2.1 | 2.1 | 2.1 | 0.0 | 2 misses → 0 |
| Two AP paths | Stress | 96.9 | 53.1 | 37.5 | 29.2 | −70% |

**Kill criterion:** not triggered for either radio configuration. The two-path realistic cell has only 2 misses in 960,000, so its "100%" is a floor effect, not evidence; the two-path case rests on the stress row.

**What the rungs say:**
1. **A perfect static map is worth little:** 9% on a single radio in realistic conditions. This is why better RF error has not produced better control: the map was never the bottleneck.
2. **Knowing which path is blocked *right now* is the prize:** a further 63% on a single radio (244.8 → 91.7). This is state estimation of the present, not forecasting, and the present is observable in live telemetry.
3. **Foresight of the next second adds little in realistic conditions** (91.7 → 83.3) and a lot under stress on a single radio (997.9 → 363.5).
4. **Under stress the map matters more:** with two paths, a perfect map alone removes 45% of misses.

**Consequence for the learned model:** the target is a *blockage and outage detector on live telemetry*, with a simple statistical detector as the baseline it must beat. Static signal prediction is not the target.

**Limits:** development scenes, 120 s each, one realisation; blockage here is independent per robot–AP path, so fleet-shared blockage information has no value in this twin by construction.
