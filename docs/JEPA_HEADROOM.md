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
