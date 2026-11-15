# Site coverage report from robot measurements (pre-registered check)

The fleet's RadioMap exists to plan robot links. The same map can answer
what a site owner asks about every other device on the floor: where is
coverage weak, where is there no backup AP, and what happens if an AP fails.
This is direction 8, "site-wide wireless intelligence".

## What the report contains (`native/coverage.py`)
From the learned map, on a 2 m grid over the floor the fleet drives:
- **Coverage** at two service levels: basic (−75 dBm) and robot control (−67 dBm).
- **Backup:** the share of floor where a second AP also reaches the level.
- **Holes:** connected weak areas, with size, centre and level.
- **What if an AP fails:** for each AP, how much covered floor is lost.
- **Not enough data:** cells where the map's own uncertainty is above 6 dB.

It also draws two maps (strongest AP, backup AP).

## What "level" means
The report states the level a device typically **measures**. That is about
1.9 dB below the mean signal power in this simulator, because readings are
averaged in dB over fading and the occasional blocked scan. Accuracy is
scored against that typical level. Against mean power the map reads 2.4–3.0
dB low, and the report would look over-cautious; both scores are saved.

## Development (scenes 400–403, 60 s survey)
| Measure | Result |
|---|---|
| Error of the strongest-AP level | 3.4 dB RMSE |
| "No backup AP" map (control level) | 87% of flagged cells right; 95% of true cells found |
| Clearly weak cells (3 dB or more below the line) found | 73% |
| Clearly fine cells wrongly flagged | 3.3% |
| Exact weak-cell map | 53% of flagged cells right; 59% of true cells found |
| Worst AP failure ranked in the report's top three | 4 of 4 scenes |

A five-times longer survey barely helps (3.4 → 3.0 dB), so the limit is the
map model, not the amount of data.

## Limits
- Cell-exact hole boundaries are not reliable: most weak cells are within 3 dB of the line, where a 3.4 dB error decides either way. The report is trustworthy for area-level statements and for clearly weak spots.
- It covers only the floor the fleet drives. Racks, offices and docks the robots never visit are outside it.
- Ground truth exists only in the simulator. On a real site the report can be spot-checked with a handheld survey.

## Test
Fresh scenes 610–615, 100 robots, 60 s survey. No simulation run is needed:
the report is built from the map and scored against ground truth.

**Hypotheses** (pooled over the six scenes):
- **K1:** strongest-AP level error ≤ 4.0 dB RMSE.
- **K2:** "no backup AP" map: at least 80% of flagged cells right and at least 90% of true cells found.
- **K3:** at least 60% of clearly weak cells found, and at most 5% of clearly fine cells flagged.
- **K4 (expected to fail):** exact weak-cell map with at least 70% right and 70% found. Development gave 53% and 59%. It is tested so the limit is on record.
- **K5:** the AP whose failure loses the most covered floor is in the report's top three, in at least 5 of 6 scenes.

Run: `python -m wirelessjev.native.coverage --campaign`.
Results are appended below after the run.
