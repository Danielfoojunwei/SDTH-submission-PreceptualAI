# Learned blockage detection versus the hand-written rule (development only)

**Status:** development scenes only. Nothing here is a confirmation, and no
learned model drives anything.

**Result:** on development scenes the hand-written rule
([BLOCKAGE_DETECTOR.md](BLOCKAGE_DETECTOR.md)) is not beaten as an alarm by
a logistic model or a small neural network trained on the same telemetry.
The learned models give better probabilities. All three get much worse
when asked to look ahead.

## Why this comparison
The headroom ladder ([JEPA_HEADROOM.md](JEPA_HEADROOM.md)) put the value in
knowing which path is blocked now. A learned detector, JEPA or otherwise, is
only worth building if it beats the simplest thing that works. This sets
that bar before any larger model is trained.

## Setup
- **Data:** `simulate(..., telemetry=[])` exports, every 100 ms and per robot, what the robot can see (its 100 ms-old reading per band, the map's expectation for its AP, delivery counters, whether it is mid-move) next to the hidden truth (serving path blocked now, in 0.5 s, in 1 s). The export does not change the run.
- **Policy generating the data:** one radio, no detector in the loop.
- **Scenes:** train on 400–401, test on 402–403; 100 robots, 60 s each; 119,000 rows each side. About 0.7% of rows are blocked.
- **Features** (`outage_risk/dataset.py`): shortfall against the map and drop below the robot's own running level (mean, lowest and highest band), delivery miss rate, move state; the last 0.5 s (5 readings).
- **Arms:**
  - the rule, version 3, as a yes/no on the same rows;
  - logistic regression on the 0.5 s window, and on the last reading only;
  - a small network (two hidden layers of 32) on the 0.5 s window, 3 training seeds.
- **Measures:**
  - *caught*: share of blocked rows flagged, with each model's threshold set to the rule's own false-alarm rate;
  - *Brier score*: error of the stated probability (lower is better). The rule's probability is its precision when it fires and the base rate when it does not.

## Results (test scenes 402–403)

**Blocked now**

| Condition | Arm | Caught | Brier ×1,000 |
|---|---|---|---|
| Realistic | Rule | **61.0%** | 3.24 |
| Realistic | Logistic, 0.5 s window | 55.2% | 2.57 |
| Realistic | Logistic, last reading | 27.3% | 3.77 |
| Realistic | Small network (3 seeds) | 39.7%, 63.1%, 60.3% | 2.41, 2.16, **2.00** |
| Stress | Rule | **67.8%** | 2.67 |
| Stress | Logistic, 0.5 s window | 30.2% | 2.34 |
| Stress | Logistic, last reading | 27.2% | 2.73 |
| Stress | Small network (3 seeds) | 14.9%, 61.4%, 32.0% | 2.95, 2.09, **2.01** |

The rule's false-alarm rate is 0.06% of clear rows (realistic) and 0.03%
(stress); when it fires it is right 87% and 94% of the time.

**Looking ahead** (share of rows blocked at that time that were flagged)

| Condition | Arm | Now | +0.5 s | +1 s |
|---|---|---|---|---|
| Realistic | Rule | 61.0% | 41.0% | 29.1% |
| Realistic | Logistic, window | 55.2% | 41.6% | 30.3% |
| Realistic | Small network (median) | 60.3% | 38.9% | 29.8% |
| Stress | Rule | 67.8% | 47.1% | 33.3% |
| Stress | Logistic, window | 30.2% | 37.7% | 28.1% |
| Stress | Small network (median) | 32.0% | 31.0% | 27.2% |

## What this says
1. **The rule is a strong alarm.** At a false-alarm rate this low, neither learned model beats it in either condition, and the network's catch rate swings widely between seeds (15% to 61% in stress).
2. **Learned models state better probabilities.** The network's Brier score is lower than the rule's in both conditions (2.0 against 3.2 and 2.7). That matters for graded actions (which path to try first, when to send on both), not for an avoid-and-move trigger.
3. **History matters; the last reading alone is weak.** The logistic model on one reading catches 27%.
4. **Nobody predicts onset.** Everything falls to about 30% at one second. In this simulator blockage starts at random and lasts about 2 s, so what looks like prediction is mostly "blocked now, still blocked soon".

## Limits
- The alarm comparison rests on few false alarms: 72 (realistic) and 36 (stress) clear rows for the rule. The catch-rate differences are noisy; the seed spread shows how much.
- The first run of this comparison scored version 2 of the rule (baseline from the first reading after a move). It caught 45.8% and 50.6%, and the logistic model then looked better in the realistic condition (52.6%). The rule was corrected to match the simulator's version 3; the numbers above are from the corrected run.
- The simulator's readings are noise-free. Real readings are not; `hil.analyse` replays the rule on real laps ([REAL_SITE_PROTOCOL.md](REAL_SITE_PROTOCOL.md), item 7).
- The telemetry comes from runs with no detector acting. Once a detector moves the robot, the data it sees changes.

## What a learned model would need to beat the rule
- **Information the rule does not have.** Each robot here sees only its own link. Blockage that is shared across robots (a forklift blocks an aisle) would make other robots' detections informative. The simulator does not model that yet: blockage is independent per robot and AP.
- **Stable training at very low false-alarm rates**, for example a loss weighted towards the alarm's operating point.
- **Real telemetry** from the site visit, where the rule's noise-free assumption no longer holds.

Run: `python -m wirelessjev.outage_risk.run_dev --cache <dir>`.
Results: `artifacts/native/outage_risk_dev.json`.
