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

## Under moving blockers (development only, added later the same day)
[MOVING_BLOCKERS.md](MOVING_BLOCKERS.md) replaces independent blockage with
forklifts that block several robots and several AP paths at once. The same
comparison was rerun under it, with two additions: each robot can see its
neighbours' alarms, and a predictor that uses the robot's own obstacle
sensing instead of radio.

Train on scenes 400–401, test on 402–403, 120 s each, 239,000 rows a side;
about 1.1% of rows are blocked. One radio, no detector acting.

**Blocked now** (caught at the rule's own false-alarm rate, 0.13% and 0.12%)

| Arm | Realistic: caught | Realistic: Brier ×1,000 | Stress: caught | Stress: Brier ×1,000 |
|---|---|---|---|---|
| Rule | 64.9% | 4.94 | **68.7%** | 4.83 |
| Logistic, own link | 64.2% | 4.54 | 68.2% | 3.91 |
| Small network, own link (3 seeds) | **69.6%, 70.6%, 73.1%** | 3.68, 3.88, **3.60** | 64.2%, 68.8%, 68.4% | 3.60, 3.42, **3.34** |
| Logistic + neighbours' alarms | 64.5% | 4.47 | 66.9% | 3.82 |
| Small network + neighbours' alarms | 69.6%, 68.5%, 71.7% | 3.98, 4.02, 3.83 | 67.0%, 66.9%, 67.3% | 3.87, 3.74, 3.68 |

**Looking ahead** (share of rows blocked at that time that were flagged)

| Arm | Realistic +0.5 s | Realistic +1 s | Stress +0.5 s | Stress +1 s |
|---|---|---|---|---|
| Radio rule | 28.9% | 15.8% | 33.8% | 19.8% |
| Small network, own link (median) | 28.6% | 13.6% | 30.8% | 16.6% |
| Small network + neighbours' alarms (median) | 27.8% | 12.0% | 30.7% | 12.9% |
| **Sensed blockers within 10 m** | **98.2%** | **95.1%** | **98.5%** | **95.3%** |

The sensing predictor's false-alarm rate is 0.02% (+0.5 s) and 0.05%
(+1 s), and it is right 99% and 96% of the time it fires. The radio rule's
false-alarm rate at those horizons is 0.5% and 0.7%.

### What this adds
1. **A learned model now beats the rule in one place.** With moving blockers, the small network on the robot's own link catches more "blocked now" than the rule in the realistic condition in all three seeds (70–73% against 65%), and ties it in stress. Its probabilities are better in both.
2. **Neighbours' alarms improve ranking but not the alarm.** With them, ranking quality (AUC) rises at every horizon, most at one second ahead (0.64 → 0.71 realistic, 0.67 → 0.72 stress). At the alarm's operating point there is no gain, and the probabilities are slightly worse. The reason is geometric: a forklift only blocks a robot it is close to, and with 100 robots on 9,600 m² a blocked robot's neighbour within 10 m is itself blocked only 5–9% of the time.
3. **Radio cannot see a blockage coming; the robot's obstacle sensing can.** No radio-only arm flags more than 34% of blockages half a second ahead or 20% one second ahead. A predictor that knows where nearby forklifts are and assumes they keep their velocity flags 98% and 95%.
4. **Racks do not hide the forklifts that matter.** Limiting the predictor to forklifts in line of sight changed nothing: in a sample, none of 425 blocking forklifts was behind a rack. A robot's serving AP is almost always within 15.5 m across the aisles, so only a forklift in the robot's own aisle or cross aisle can sit on the vulnerable part of the line.

### Limits of the sensing result
This is a headroom figure, not a detector.
- The predictor is given every forklift's exact position and velocity within 10 m. A real lidar or camera pipeline has misses, false objects and delay.
- It is given the AP's position and uses the same blocking geometry that generates the truth. On a real site that mapping is unknown and has to be learned.
- It ignores people, pallets and anything that is not one of the 13 forklifts.

The misses it does have (5% at one second) come from forklifts that stop,
turn or enter the 10 m range late.

## What a learned model would need to beat the rule
- **For "blocked now":** it already can, narrowly, when blockage is shared (above). The remaining job is stable training at very low false-alarm rates.
- **For "blocked soon":** radio telemetry alone is not enough, and other robots' alarms add little. The input that carries the information is the robot's own perception. The learned part is then the mapping from sensed obstacles and AP direction to a coming signal drop, which is site-specific and not known in advance. That is a concrete job for a JEPA-style model over perception and radio together, and the first target where a learned model has information the hand-written rule lacks.
- **Real telemetry** from the site visit, where the rule's noise-free assumption no longer holds. One car with no obstacle sensing logged cannot test the perception idea on Tuesday.

Run: `python -m wirelessjev.outage_risk.run_dev --cache <dir>`; add `--blockage-model moving --seconds 120`
for the moving-blocker run and `--sensing-only` for the sensing predictor.
Results: `artifacts/native/outage_risk_dev*.json`.
