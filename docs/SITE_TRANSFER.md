# Site transfer for foresight: the premise did not hold (development only)

**Status:** development scenes only. The fresh-scene test was **not run**,
because the tuning result removed its premise. No pass marks were frozen and
nothing here is a confirmation.

## The question
Foresight ([FORESIGHT.md](FORESIGHT.md)) warns a robot that a path will be
blocked, from what its obstacle sensing sees. A hand-written rule needs the
site's blocking geometry (how much of the line to an AP a forklift can
block, which depends on AP height). A learned predictor does not. So the
expected advantage of learning was: move to a site with different AP
heights, and the learned predictor adapts from the robots' own data while
the hand-written rule is left with the wrong geometry.

## What was built
- **Site geometry as a setting** (`reach_frac`): APs at 10 m instead of 6 m, so a forklift blocks the first 23% of a line instead of 39%.
- **Training with no ground truth** (`foresight.radio_label_rows`): the label is the robot's own radio rule firing within the next second, for the AP it is using. Five minutes of driving at the new site on scenes 400–401 gave 114,383 rows.
- **Arms** (two AP paths, the 20 m move limit, tuning scenes 402–403, 120 s): no warning; the old site's learned predictor, not retrained; the hand-written rule with the old geometry; the same with the new geometry; a predictor trained on the new site's radio labels; one trained on the new site's true blockage; perfect warnings.

## Results (tuning scenes 402–403, new site)
Missed control packets per million, with the share of blockages each source
foresees and how often its warnings are right:

| Warning source | Realistic | Stress | Foreseen | Right |
|---|---|---|---|---|
| None | 8.3 | 172.9 | | |
| Perfect | 10.4 | 112.5 | 100% | 100% |
| Old site's learned predictor, not retrained | 8.3 | 147.9 | 38% | 89% |
| Hand-written rule, old geometry | 2.1 | 156.2 | 44% | 79% |
| Hand-written rule, new geometry | 8.3 | 154.2 | 43% | 83% |
| Learned from the new site's radio rule | 8.3–10.4 | 150.0–154.2 | 42–48% | 50–53% |
| Learned from the new site's true blockage | 8.3 | 112.5 | 36% | 94% |

The realistic column has 0 to 5 misses per arm and cannot be read.

## Why the premise fails
AP height barely changes the blockage that matters. Scene 402, share of
time blocked:

| AP height | Any robot-AP line | Robot's best AP | Second AP | Both at once |
|---|---|---|---|---|
| 6 m (old site) | 6.22% | 1.64% | 3.14% | 0.460% |
| 10 m (new site) | 4.00% | 1.42% | 2.60% | 0.441% |
| 20 m | 2.28% | 1.17% | 1.79% | 0.422% |

Raising the APs from 6 m to 20 m cuts blockage on distant lines by almost
two thirds, but the two paths a robot actually uses are blocked together
almost as often. Those events come from a forklift right beside the robot,
which blocks whatever the AP height.

## What this says
1. **Nothing to transfer.** The hand-written rule with the wrong geometry does as well as with the right one (156.2 against 154.2), and the old site's learned predictor works unchanged at the new site (147.9).
2. **The learned predictor's supposed advantage is not supported.** FORESIGHT.md said the learned predictor "recovered the site geometry from data". That geometry turns out not to matter for control, so recovering it is not an advantage. On the evidence so far a hand-written rule is as good as the learned predictor for foresight.
3. **Training from the robot's own radio works, at a cost.** A predictor trained only on radio labels foresees as many blockages as the others, but its warnings are right about half the time, against 80–90%. In the loop it lands with the rest (150–154).
4. **One lead, probably noise.** The predictor trained on the new site's true blockage matched perfect warnings here (112.5). Its warning statistics are close to the old predictor's, and two scenes give about 55 misses per arm, so this is more likely chance than a real gain. It was not followed up.

## Where this leaves the learned model
- Foresight works in simulation (two pre-registered tests, 24% and 44% fewer misses in the hardest case). What makes it work is the sensing and the cheap response.
- The predictor in the middle can be a small learned network or a few lines of geometry; the results do not separate them.
- A learned model would earn its place here only if real obstacle sensing is messy in ways a geometric rule cannot handle. That is a question for real sensor data, not for this simulator.

Code kept: `reach_frac` (site geometry), `radio_label_rows` (training with
no ground truth), and the unrun `transfer` campaign in `native_ext/run_fleet.py`.
Raw tuning rows: `artifacts/native/transfer_dev.jsonl`.
