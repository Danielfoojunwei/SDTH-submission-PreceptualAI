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

**Response** (`native_ext/sim_fleet.py`, `foresee`). A warning triggers an ordinary roam, done between packets, so a wrong warning costs a roam and not a miss.
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

Run: `python -m wirelessjev.native_ext.run_fleet foresight --workers 4`.
Results are appended below after the run.

## Results (15 Nov 2026)
Scenes 580–585, 100 robots, 300 s, 600,000 control packets per scene per
arm. Raw rows: `artifacts/native/foresight.json`. Intervals are 95%
bootstrap over the six scenes; ratio intervals are paired by scene. All
figures are simulation, with simulated sensing.

**Verdicts**

| # | Hypothesis | Result |
|---|---|---|
| F1 | Two AP paths, `stress`: learned warnings cut misses ≥ 10%, in ≥ 4 of 6 scenes | **Pass**: 24% fewer (229.7 → 175.0), 5 of 6 scenes; ratio 0.76 [0.66–0.91] |
| F2 | Headroom: perfect warnings cut them ≥ 30% | **Pass**: 41% fewer (229.7 → 136.7) |
| F3 | No harm in `main` (within 5 per million) | **Pass**: 14.2 against 11.4 |
| F4 | Learned ≤ hand-written rule (`stress`) | **Pass**: 175.0 against 187.2; ratio 0.94 [0.87–1.03] |
| F5 | One radio not made worse in either condition | **Pass** (expected to fail): 366.7 against 383.9; 2,530.6 against 2,745.8 |

**Missed control packets per million**

| Arm | Realistic | Stress |
|---|---|---|
| Private 5G model | 148.9 [131.9–166.7] | 148.9 [131.9–166.7] |
| One radio | 383.9 [316.7–455.0] | 2,745.8 [2,230.0–3,289.2] |
| One radio + perfect warning | 189.2 [123.9–267.8] | 1,396.4 [1,225.3–1,587.5] |
| One radio + learned warning | 366.7 [259.4–493.6] | 2,530.6 [2,061.4–3,061.9] |
| Two AP paths | 11.4 [5.8–16.9] | 229.7 [187.2–268.6] |
| Two AP paths + perfect warning | 10.6 [5.8–15.3] | 136.7 [99.7–186.7] |
| Two AP paths + learned warning | 14.2 [6.1–25.3] | 175.0 [136.7–221.9] |
| Two AP paths + hand-written warning | 14.7 [5.6–28.6] | 187.2 [138.9–245.0] |

## What the results say
1. **A learned model passed a pre-registered control test for the first time in this project.** With two AP paths in the stress condition, warnings from the learned predictor cut misses by 24%, in 5 of 6 scenes, and the paired interval excludes no change.
2. **The gain is modest and local.** It captures a little over half of what a perfect warning gives (24% of a possible 41%). It does not make two AP paths beat the 5G model in stress on their own (175.0 against 148.9); the relay still does that.
3. **In the realistic condition it does nothing measurable.** 14.2 against 11.4 is within noise (ratio 1.24 [0.77–2.06]), and one scene accounts for the difference. There is too little to prevent there.
4. **The learned predictor ties the hand-written rule.** 175.0 against 187.2, with an interval that includes no difference. The hand-written rule was given the site's blocking geometry and the learned one was given none. *Correction, later the same day:* that geometry turned out not to matter for control ([SITE_TRANSFER.md](SITE_TRANSFER.md)), so needing none is not an advantage. A hand-written rule is as good as the learned predictor here.
5. **One radio gains little:** 4% and 8% fewer misses, against 51% and 49% with a perfect warning. For one radio the limit is sensing range: a robot cannot vouch for an AP more than 20 m away.

## What it does not show
- That it works with a real lidar or camera. The sensing here is a model.
- That it transfers to another site. Training and test scenes share one layout and one blocking geometry, which is exactly the thing the learned predictor is supposed to learn per site.
- That it beats a hand-written rule when that rule is given the right geometry. It ties it.

## Next
- The largest remaining headroom is sensing range and quality, not the predictor: perfect warnings are worth 41–51%.
- Fleet-shared sensing (a robot is told about forklifts other robots can see) would extend the range without new sensors. Not built.
- A site-transfer test: train on one layout, test on another, against the hand-written rule with the wrong geometry.

# Extension: fleet-shared sensing (pre-registered)

The limit above was sensing range: a robot only knows about forklifts within
10 m. Here every robot is told about each forklift that any robot detects.

## What is modelled (`foresight.sense_fleet`)
- A shared report arrives 200 ms late and is moved forward along its estimated velocity.
- A robot's own fresh detection wins over a shared one.
- No robot knows about a forklift that no robot detects.
- The predictor is retrained on this sensing (scenes 400–401, 1,411,776 rows) and only looks at forklifts within 25 m of the robot.

## Development (tuning scenes 402–403)
- **What a robot knows:** with its own sensing, a given forklift is known 1.6–1.8% of the time; with sharing, 64–65%.
- **Blockages foreseen across all APs:** about 30% → 60%, at the same share of right warnings (80%).
- **In the loop, two AP paths, stress** (missed per million): no warning 196; own sensing 148; fleet-shared 127–133; perfect 110.
- **One radio:** still no gain in the realistic condition (294 against 285).
- **The 20 m move limit still matters:** without it, fleet-shared warnings were worse (146–169).

These scenes were used to choose settings and support no claim.

## Test
Fresh scenes 600–605, 100 robots, 300 s, moving blockers, two AP paths.
Frozen: both predictors (`foresight_model.pt`, `foresight_model_fleet.pt`),
threshold 0.8, move limit 20 m, report latency 200 ms.

**Arms:** `nr_30ms`, `shared`, `shared+foresee[own]`, `shared+foresee[fleet]`, `shared+foresee[truth]`.

**Hypotheses** (`stress`, rates pooled over six scenes):
- **S1:** fleet-shared warnings give no more misses than own-sensing warnings, and no more in at least 4 of 6 scenes.
- **S2:** fleet-shared warnings cut misses by at least 25% against no warning.
- **S3 (`main`, no harm):** within 5 per million of no warning.

Run: `python -m wirelessjev.native_ext.run_fleet shared-sensing --workers 4`.
Results are appended below after the run.


## Results: fleet-shared sensing (15 Nov 2026)
Scenes 600–605, 100 robots, 300 s, moving blockers, two AP paths. Raw rows:
`artifacts/native/foresight_shared_sensing.json`. Intervals are 95%
bootstrap over six scenes; ratio intervals are paired by scene.

**Verdicts**

| # | Hypothesis | Result |
|---|---|---|
| S1 | Fleet-shared warnings give no more misses than own-sensing warnings (`stress`) | **Fail**: 190.0 against 179.7; ratio 1.06 [0.97–1.16]; no more in only 2 of 6 scenes |
| S2 | Fleet-shared warnings cut misses ≥ 25% against no warning (`stress`) | **Pass**: 41% fewer (320.0 → 190.0) |
| S3 | No harm in `main` | **Pass**: 20.3 against 26.9 |

**Missed control packets per million**

| Arm | Realistic | Stress |
|---|---|---|
| Private 5G model | 160.8 [142.8–184.5] | 160.8 [142.8–184.5] |
| Two AP paths | 26.9 [17.8–36.4] | 320.0 [261.9–377.2] |
| + warnings, own sensing | 15.8 [11.7–20.0] | 179.7 [146.4–218.6] |
| + warnings, fleet-shared sensing | 20.3 [10.6–32.5] | 190.0 [153.9–226.1] |
| + perfect warnings | 11.4 [4.4–20.0] | 131.1 [98.3–167.8] |

## What these results say
1. **Sharing sensing across the fleet adds nothing.** Fleet-shared warnings were not better than a robot's own, in stress or in the realistic condition. The lead on the tuning scenes (127 against 148) did not hold on fresh scenes.
2. **Why:** the response only acts on forklifts close to the robot (it only moves to APs within 20 m, and only when both paths are about to be blocked). A robot's own sensing already covers those. What sharing adds is knowledge of far forklifts, which this response cannot use, delivered 200 ms late.
3. **Own-sensing foresight replicated, more strongly.** On this second, independent set of scenes it cut two-path misses in stress by 44% (320.0 → 179.7; ratio 0.56 [0.47–0.66]), against 24% in the first test. It also helped in the realistic condition here (26.9 → 15.8), where the first test saw nothing. Two tests now agree on the direction; the size is somewhere between a quarter and under a half.
4. **Still short of the 5G model without the relay** in stress: 179.7 against 160.8, though the intervals now overlap.

## Where this leaves foresight
- Keep: warnings from a robot's own obstacle sensing, for robots with two AP paths.
- Drop: fleet-shared sensing for this purpose.
- The gap to a perfect warning (179.7 against 131.1) is now mostly sensing noise and missed detections within range, not range itself.
