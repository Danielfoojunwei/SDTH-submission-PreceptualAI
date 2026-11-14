# Live blockage detector (pre-registered)

The headroom ladder (`docs/JEPA_HEADROOM.md`) showed that a perfect map is
worth 9% on a single radio, and that knowing which paths are blocked *right
now* is worth a further 63%. This is the first attempt to get that knowledge
from signals the robot already has, with no learned model.

## What is modelled (`src/wirelessjev/native/sim.py`, `block_detect`)
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

Run: `python -m wirelessjev.native.run_fleet detect --workers 4`.
Results are appended below after the run.
