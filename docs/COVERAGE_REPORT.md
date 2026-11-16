# Site coverage report from robot measurements (pre-registered check)

The fleet's RadioMap exists to plan robot links. The same map can answer
what a site owner asks about every other device on the floor: where is
coverage weak, where is there no backup AP, and what happens if an AP fails.
This is direction 8, "site-wide wireless intelligence".

## What the report contains (`native_ext/coverage.py`)
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

Run: `python -m wirelessjev.native_ext.coverage --campaign`.
Results are appended below after the run.

## Results (15 Nov 2026)
Scenes 610–615, 100 robots, 60 s survey. Raw rows:
`artifacts/native/coverage_report.json`. All figures are simulation.

**Verdicts** (pooled over six scenes)

| # | Hypothesis | Result |
|---|---|---|
| K1 | Strongest-AP level error ≤ 4.0 dB | **Pass**: 3.6 dB RMSE |
| K2 | "No backup AP" map: ≥ 80% right, ≥ 90% found | **Pass**: 88% right, 95% found |
| K3 | ≥ 60% of clearly weak cells found; ≤ 5% of clearly fine cells flagged | **Pass**: 77% found, 2.5% flagged |
| K4 | Exact weak-cell map: ≥ 70% right and found (expected to fail) | **Fail**: 51% right, 59% found |
| K5 | Worst AP failure in the report's top three, ≥ 5 of 6 scenes | **Pass**: 6 of 6 (ranked first in 4) |

**Per scene**

| Scene | Level error (dB) | Floor covered, control level | Floor with a backup AP | Largest hole (m²) | Report's rank of the worst AP failure |
|---|---|---|---|---|---|
| 610 | 3.40 | 89.0% | 12.3% | 92 | 1 |
| 611 | 3.24 | 85.3% | 9.0% | 116 | 1 |
| 612 | 3.58 | 84.8% | 9.7% | 152 | 1 |
| 613 | 3.38 | 89.3% | 9.2% | 92 | 1 |
| 614 | 4.46 | 87.6% | 13.7% | 120 | 3 |
| 615 | 3.18 | 86.7% | 8.6% | 124 | 2 |

Example map (scene 610): `docs/figures/coverage_scene610.png`.

## What the results say
1. **The report is reliable for the questions a site owner asks first.** Where there is no backup AP, where coverage is clearly weak, and which AP failure would hurt most: all three are answered correctly on fresh scenes from one minute of robot driving.
2. **It is not a cell-exact survey.** The exact weak-cell map is right half the time, because most weak cells sit within 3 dB of the line and the map's error is 3.6 dB. The report should show a "marginal" band, not a sharp edge.
3. **This is the place where a more accurate radio map pays.** Control did not improve with lower signal error; a coverage report does, directly. Codex's learned radio models cut signal error by about a quarter in its tests. Using one here is the natural next step, and the first product use where that accuracy matters.
4. **A finding about the simulated site itself:** only about one tenth of the floor has a second AP at the control level (−67 dBm). Basic coverage (−75 dBm) is complete.

## Added after the test (16 Nov)
**A "marginal" band.** A cell whose estimate is within 3 dB of a threshold,
or within the map's own uncertainty if that is larger, is reported as
marginal instead of weak or fine. The report now gives, per service level,
the share of floor that is clearly weak, marginal and covered, and lists
"clearly weak" holes separately. The map hatches the marginal band.

On scenes 610–611 at the control level: clearly weak 2.0% and 3.1% of the
floor; marginal 27% and 29%. So about a quarter of the floor cannot be
called either way from one minute of driving, which matches the failed
cell-exact test (K4).

**A version for real laps** (`coverage.from_laps`). It takes the lap logs,
`site.json` and `waypoints.json` from the site visit. Each reading is placed
by interpolating between waypoint marks; a cell's level for an AP is the
mean of the nearest readings within 3 m, and its uncertainty is their
spread plus a term for position error. It covers only floor within 1.5 m of
where the car drove. There is no ground truth on a real site, so this
version has not been scored; it is tested on fixture laps and runs as part
of the runbook dry run.
```bash
.venv/bin/python -m wirelessjev.native_ext.coverage --laps artifacts/onecar/*-lap-*.jsonl --site site.json --waypoints waypoints.json --out artifacts/onecar/coverage
```

## Not done
- Trying a learned radio model in place of RadioMap for this report.
- Scoring the real-lap version against a handheld survey.
