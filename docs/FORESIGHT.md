# Foresight from obstacle sensing: warn before a path is blocked (pre-registered)

Radio telemetry cannot see a blockage coming; a predictor given nearby
forklift positions can ([OUTAGE_RISK.md](OUTAGE_RISK.md)). This builds that
predictor with realistic sensing, gives the controller a cheap response to a
warning, and tests whether misses fall. Blockage is the moving-blocker model
([MOVING_BLOCKERS.md](MOVING_BLOCKERS.md)) throughout.

## What is modelled
**Sensing** (`outage_risk/foresight.py`). Each robot's obstacle sensing reports forklifts:
- within 10 m and not hidden behind a rack;
- with 10% of detections missed and 0.2 m position noise;
- with velocity from two detections 0.5 s apart (so about 0.6 m/s of velocity noise).

**Warning sources.** Each produces, every 100 ms, a yes/no per robot and AP: "this path will be blocked within the next second".
- **Truth:** the perfect warning. Headroom only.
- **Hand-written rule, site known:** each sensed forklift keeps its estimated velocity; a path is flagged if a forklift will sit on the vulnerable part of the line to that AP. It is given the site geometry (how much of the line a forklift can block).
- **Hand-written rule, whole line:** the same, without knowing the AP height.
- **Learned:** a small network (two hidden layers of 64). Inputs, in a frame aligned with the line to the AP: line length, the robot's planned velocity, and for the two nearest sensed forklifts their position and velocity along and across the line. No site geometry is given. Trained on scenes 400–401 (340,448 rows).

**Response** (`native/sim.py`, `foresee`). A warning triggers an ordinary roam, done between packets, so a wrong warning costs a roam and not a miss.
- One radio: leave a path that is about to be blocked if another AP scores better; do not plan a roam into a flagged path.
- Two AP paths: the first path is left alone (the per-packet switch already covers it). The second path is re-pointed only when both paths are about to be blocked together.

## How it was developed (development scenes only)
**1. Kill gate: is a perfect warning worth anything?** Scene 400, 60 s: one radio 850 → 333 (realistic) and 3,608 → 2,925 (stress); two AP paths 558 → 125 (stress). Yes.

**2. Prediction on test scenes 402–403** (blockages on the nearest AP foreseen within a second; share of warnings that were right):

| Source | Foreseen | Right when it warns |
|---|---|---|
| Hand-written, site known | 62–65% | 77–81% |
| Hand-written, whole line | 64–67% | 58–65% |
| Learned (3 seeds) | 70–74% | 73–80% |
| Hand-written, perfect sensing | 92% | 97–98% |

**3. In the control loop, first response** (scenes 402–403, 120 s; missed per million):

| Arm | No warning | Truth | Hand-written, site known | Learned |
|---|---|---|---|---|
| One radio, realistic | 285 | 144 | 640 | 519 |
| One radio, stress | 3,125 | 1,769 | 4,048 | 5,127 |
| Two AP paths, realistic | 8.3 | 10.4 | 2.1 | 2.1 |
| Two AP paths, stress | 196 | 119 | 244 | 192 |

Perfect warnings cut misses 40–50%. Realistic warnings made one radio
much worse. The miss causes show why: robots were warned off one AP and
sent to another that was itself blocked, by a forklift beyond their 10 m
sensing range. "No warning for that AP" is not "that AP is clear".
Realistic warnings also caused 3–4 times more moves than perfect ones.

**4. Fix: only move to an AP within 20 m, and warn less often** (learned threshold 0.5 → 0.8). Within 20 m the robot can see the whole vulnerable stretch of the line. Same scenes:

| Arm | No warning | Truth | Hand-written, site known | Learned | Learned, strict |
|---|---|---|---|---|---|
| One radio, realistic | 285 | 154 | 396 | 273 | 325 |
| One radio, stress | 3,125 | 1,758 | 4,983 | 4,319 | 2,842 |
| Two AP paths, realistic | 8.3 | 10.4 | 2.1 | 2.1 | 8.3 |
| Two AP paths, stress | 196 | 110 | 156 | 169 | 148 |

- **Two AP paths, stress:** realistic warnings now help, 14–24%. This is the one case worth testing.
- **One radio:** break-even at best. Not ready.
- **Two AP paths, realistic:** 1 to 5 misses per arm; no reading possible.

Scenes 402–403 were used to choose the 20 m limit and the 0.8 threshold, so
they are tuning scenes now and support no claim.

## Limits
- Sensing is a model, not a lidar. Real pipelines have false objects, latency and object-tracking errors that are not modelled.
- Only the 13 forklifts exist. People and pallets are not sensed or modelled.
- The learned predictor is trained and tested on the same site layout and the same blocking geometry. Whether it transfers to another site is untested.
- The robot is assumed to know its own planned path one second ahead.

## Test
Fresh scenes 580–585, 100 robots, 300 s, learned map from a 60 s survey,
moving blockers. Frozen in this commit: the predictor
(`artifacts/native/foresight_model.pt`), warning threshold 0.8, move limit
20 m, and the sensing settings above.

**Conditions:** `main` and `stress`.

**Arms:** `nr_30ms`; `single`, `single+foresee[truth]`,
`single+foresee[learned]`; `shared`, `shared+foresee[truth]`,
`shared+foresee[learned]`, `shared+foresee[geometric]` (hand-written, site
known). All foresight arms use the 20 m limit.

**Hypotheses** (rates pooled over the six scenes):
- **F1 (`stress`, the claim):** `shared+foresee[learned]` has at least 10% fewer misses than `shared`, and fewer in at least 4 of 6 scenes.
- **F2 (`stress`, headroom):** `shared+foresee[truth]` has at least 30% fewer misses than `shared`.
- **F3 (`main`, no harm):** `shared+foresee[learned]` ≤ `shared` + 5 per million.
- **F4 (`stress`, learned against hand-written):** `shared+foresee[learned]` ≤ `shared+foresee[geometric]`. The development scenes gave 148 against 156, so this is expected to be marginal.
- **F5 (one radio, expected to fail):** `single+foresee[learned]` ≤ `single` in both conditions. Development gave 9% fewer in stress and 14% more in the realistic condition. It is tested so the answer is on record.

**Reported, not tested:** every arm against `nr_30ms`; moves per arm; miss causes.

Run: `python -m wirelessjev.native.run_fleet foresight --workers 4`.
Results are appended below after the run.
