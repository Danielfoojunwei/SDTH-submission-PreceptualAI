# Live blockage detector (pre-registered)

The headroom ladder (`docs/JEPA_HEADROOM.md`) showed that a perfect map is
worth 9% on a single radio, and that knowing which paths are blocked *right
now* is worth a further 63%. This is the first attempt to get that knowledge
from signals the robot already has, with no learned model.

## What is modelled (`src/wirelessjev/native_ext/sim_fleet.py`, `block_detect`)
Every 100 ms the robot compares the level of its serving AP, averaged across
that AP's bands, with a baseline.

1. **Signature.** Fading is independent per band; a blocker is not. A drop
   of more than 10 dB (`block_drop_db`) on the band average, in 2 consecutive
   readings (`block_confirm`), is treated as a blocked path.
2. **Baseline.** A running average of the robot's own readings. Just after a
   move, the baseline is the higher of the first reading and what the map
   expects there, so arriving at a still-blocked AP looks like a drop.
3. **Reaction.** The path is avoided for 2 s (`block_hold_s`). The robot
   moves at once to the best AP that is not being avoided, without waiting
   for a quiet moment. A path flagged again soon after the robot returns is
   avoided for twice as long, up to 16 s.
4. **Second path.** With two AP paths, the same test runs on the second
   path, which is re-pointed in the same way.

The detector sees only the robot's own readings, 100 ms old. It is not told
the blockage state.

## How it was developed (development scenes only)
| Version | Rule | Result on scene 400, one radio, realistic |
|---|---|---|
| 1 | 8 dB drop, one reading | 563 false alarms; misses rose |
| 2 | 10 dB drop, two readings, baseline from first reading after a move | 126 detections; no benefit (robots returned to blocked paths) |
| 3 | Version 2 + map-expected baseline + longer avoidance on repeat | Misses fell (below) |

Version 3, 60 s, 100 robots, missed control packets per million:

| Scene, condition | One radio | + detector | + told the truth (oracle) | Two paths | Two paths + detector |
|---|---|---|---|---|---|
| 400, realistic | 616.7 | 200.0 | 66.7 | 0.0 | 0.0 |
| 401, realistic | 241.7 | 191.7 | 75.0 | 0.0 | 0.0 |
| 400, stress | 4,175.0 | 2,108.3 | 1,650.0 | 150.0 | 125.0 |

Scene 401 stress was not run. The detector's cost is visible in the miss
causes on scene 400: "no radio up" rose from 0 to 9, because each reaction
is a move.

## Limits
- Readings in the simulator have a 100 ms delay but no measurement noise and
  no 1 dB quantisation. Real RSSI has both. The 10 dB threshold leaves
  margin, but this is untested.
- Blockage in the simulator is a 15 dB step on all bands of one AP path.
  Real blockers attenuate bands differently and ramp in.
- Three thresholds were tuned on two development scenes.

## Test
Fresh scenes 560–565, 100 robots, 300 s, learned map from a 60 s survey.
Detector settings are frozen at the defaults in this commit.

**Conditions:** `main` (realistic) and `stress` (3× collisions, 2.4 GHz
noise +10 dB), as in `docs/DIRECTIONS_123.md`.

**Arms:** `nr_30ms`; `single`; `single+detect`; `single+oracle` (same
controller, told the current blockage state); `shared` (second association
on the same card); `shared+detect`.

**Hypotheses** (rates are pooled over the six scenes):
- **B1 (`main`):** `single+detect` has fewer misses than `single` in at
  least 5 of 6 scenes, and the pooled reduction is at least 30%.
- **B2 (`stress`):** the same, with a pooled reduction of at least 25%.
- **B3 (`main`):** the detector recovers at least 40% of the oracle's
  headroom: (`single` − `single+detect`) / (`single` − `single+oracle`).
- **B4 (no harm with two paths):** `shared+detect` ≤ `shared` + 5 per
  million in `main`, and ≤ 1.15 × `shared` in `stress`.
- **B5 (stretch, `main`):** `single+detect` misses ≤ `nr_30ms` in at least
  5 of 6 scenes. The development scenes suggest this will fail (about 200
  against about 50 per million); it is tested so that the answer to "is one
  radio with a detector enough to beat 5G" is on record.

**Reported, not tested:** detections, roams, miss causes and stopped time
for every arm.

Run: `python -m wirelessjev.native_ext.run_fleet detect --workers 4`.
Results are appended below after the run.

## Results (15 Nov 2026)
Scenes 560–565, 100 robots, 300 s. Raw rows:
`artifacts/native/blockage_detector.json`. Intervals are 95% bootstrap over
the six scenes. All figures are simulation.

**Verdicts**

| # | Hypothesis | Result |
|---|---|---|
| B1 | One radio, `main`: fewer misses in ≥ 5 scenes and ≥ 30% pooled | **Pass, 6/6**, 30.3% (200.8 → 140.0) |
| B2 | One radio, `stress`: fewer misses in ≥ 5 scenes and ≥ 25% pooled | **Pass, 6/6**, 49.1% (2,045.6 → 1,041.9) |
| B3 | Recovers ≥ 40% of the oracle's headroom (`main`) | **Pass by the letter, not informative** (see below) |
| B4 | No harm with two AP paths | **Pass** (3.1 against 1.9 in `main`; 81.4 against 86.4 in `stress`) |
| B5 | One radio + detector ≤ 5G (`main`) | **Fail, 0/6** (140.0 against 42.5) |

**Missed control packets per million**

| Arm | Realistic | Stress |
|---|---|---|
| Private 5G model | 42.5 [33.6–52.5] | 42.5 [33.6–52.5] |
| One radio | 200.8 [161.4–232.8] | 2,045.6 [1,785.3–2,293.3] |
| One radio + detector | 140.0 [102.2–177.5] | 1,041.9 [963.1–1,128.1] |
| One radio + told the truth (`single+oracle`) | 180.3 [118.3–243.9] | 951.1 [780.3–1,181.1] |
| Two AP paths | 1.9 [0.3–4.4] | 86.4 [61.7–110.3] |
| Two AP paths + detector | 3.1 [1.4–4.7] | 81.4 [65.0–96.7] |

**What the detector costs** (one radio, totals over six scenes)

| | Realistic: without | Realistic: with | Stress: without | Stress: with |
|---|---|---|---|---|
| Roams per scene | 794 | 1,416 | 781 | 1,380 |
| Misses while the path was blocked | 507 | 139 | 5,647 | 850 |
| Misses on a weak link | 215 | 310 | 1,699 | 2,609 |
| Misses with no radio up (mid-move) | 1 | 55 | 18 | 292 |

## What the results say
1. **The rule works on fresh scenes.** It cut one-radio misses in all 12 scene-condition pairs: 30% in the realistic condition and 49% in stress. The realistic result clears its 30% mark by 0.3 points, so treat it as "about 30%", not as a comfortable pass.
2. **It trades blocked-path misses for other misses.** Misses on a blocked path fall by 73% and 85%. In exchange the robot roams almost twice as often and lands on weaker APs, so weak-link and mid-move misses rise.
3. **It is not enough to make one radio beat 5G.** 140 against 42.5 per million, in none of six scenes.
4. **With two AP paths it changes little.** The second path already covers a blocked first path. The detector's effect there is within the noise in both conditions.

## Why B3 is not informative
The reference arm `single+oracle` turned out to be a weak reference, so the
ratio (296%) should not be read as "the rule beats an oracle".

- That arm is told which paths are blocked, but it keeps the default roam timing: it waits for the next delivered packet before moving. On a blocked path that packet may not arrive.
- It also uses the learned map and the trust gate. The ladder's rung 3 (perfect map, no gate) got 91.7 against 244.8 on development scenes, a far larger gain than this arm's 180.3 against 200.8.
- The detector moves at once and avoids the flagged path for 2 s or more. Those two behaviours, not better information, are why it beat this arm in the realistic condition. In stress the oracle arm is still ahead (951 against 1,042).

The useful reading: how the robot acts on blockage matters as much as
knowing about it. A fair oracle reference would move at once and hold the
avoidance; that comparison has not been run on fresh scenes.

## Later result: the rule does not survive moving blockers
Under forklift-style shared blockage ([MOVING_BLOCKERS.md](MOVING_BLOCKERS.md), scenes 570–575) the rule made one-radio misses 27% worse in the realistic condition (406.4 → 516.1) and made two-path misses worse in both conditions. It still helped one radio in stress (27% fewer). Blockages there last under a second, so roaming away costs more than waiting. The rule is off by default; the results above hold only for blockage that lasts about 2 s.

## Next
- Real signal: `hil.analyse` replays this rule over the laps recorded on 17 Nov ([REAL_SITE_PROTOCOL.md](REAL_SITE_PROTOCOL.md), item 7).
- A learned detector has to beat this rule; on development scenes neither a logistic model nor a small network does ([OUTAGE_RISK.md](OUTAGE_RISK.md)).
