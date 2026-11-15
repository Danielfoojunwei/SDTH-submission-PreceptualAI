# Moving blockers: do the headline results survive shared blockage? (pre-registered)

Every result so far used blockage drawn independently per robot and AP. That
flatters path diversity: the second AP path, the relay and the Bluetooth
path all rely on failing independently of the first path. Real blockers are
objects. One forklift blocks several robots, and can block a robot's paths
to two APs at once. This test puts the headline claims under that.

## What is modelled (`src/wirelessjev/native/blockers.py`)
- **Blockers:** 13 forklifts. Each repeats: drive along its aisle to a random spot (1.0–2.5 m/s), stop there for 0–20 s half the time, drive to a cross aisle, drive along it to another aisle.
- **Blocking rule:** a robot's path to an AP is blocked (15 dB, all bands, as before) when a forklift is within 0.75 m of the straight line between them, ahead of the robot, inside the first 38.6% of that line from the robot. That fraction is where the line runs below the top of a 2.5 m forklift for a 0.3 m robot antenna and a 6 m AP.
- **Relay links:** the line between two robots is blocked by a forklift anywhere along it.
- **Bluetooth path:** blocked with the AP site its gateway is at, as before.
- **5G and Wi-Fi 8 models:** blocked by the same forklifts, by the same rule.
- **Everything else is identical:** fading and all other random draws are the same under both blockage models (`blockage_model="independent"` is the default and is unchanged).

## Calibration (development scenes 400–401, 120 s)
The number of forklifts was chosen so a robot's best AP is blocked for the
same share of time as under the independent model.

| Measure | Independent | Moving, 13 forklifts |
|---|---|---|
| Best AP blocked (share of time) | 1.00%, 1.49% | 0.97%, 1.58% |
| Both of the top two APs blocked at once | 0.008%, 0.05% | 0.25%, 0.49% |
| A neighbour within 10 m blocked, given this robot is | 1.0%, 2.1% | 5.0%, 8.7% |
| Blockage length on the best AP: mean; 90th percentile | 1.5–2.1 s; 4.1–4.6 s | 0.8–1.0 s; 1.7–2.0 s |
| Blockage events per robot-minute on the best AP | 0.39–0.43 | 0.74–1.00 |

So blockage is as common as before, but the top two APs fail together about
10 to 30 times more often, neighbours fail together, and events are shorter
and more frequent.

## Assumptions and limits
- AP height (6 m), forklift height (2.5 m) and width (1.5 m) are assumed, not measured. A higher AP shortens the vulnerable part of the line and weakens the effect.
- Forklifts do not avoid robots and robots do not avoid forklifts.
- Blockage is still a 15 dB step. People, pallets and doors are not modelled.
- The relay model keeps its other optimistic assumptions (same-slot forwarding, helpers always cooperate); see `docs/DIRECTIONS_123.md`.

## Development check (scene 400, 60 s, 100 robots)
Missed control packets per million, moving blockers:

| Arm | Realistic | Stress |
|---|---|---|
| Private 5G model | 108.3 | 108.3 |
| Wi-Fi 8 model | 441.7 | 1,075.0 |
| One radio | 850.0 | 3,608.3 |
| One radio + blockage rule | 741.7 | 3,233.3 |
| Two AP paths | 8.3 | 558.3 |
| Two AP paths + blockage rule | 100.0 | 325.0 |
| Two AP paths + relay | 0 | 0 |
| Two AP paths + Bluetooth | 0 | 125.0 |
| One radio + relay + Bluetooth | 0 | 0 |
| Two AP paths + relay + Bluetooth | 0 | 0 |

One scene and 120,000 packets per arm: a lead, not a result. Under the
independent model on the same scene, two AP paths gave 0 (realistic) and 150
(stress), and the 5G model about 50.

## Test
Fresh scenes 570–575, 100 robots, 300 s, learned map from a 60 s survey,
moving blockers with the settings above frozen.

**Conditions:** `main` and `stress`, as in `docs/DIRECTIONS_123.md`.

**Arms** (all under moving blockers): `nr_30ms`, `wifi8`, `single`,
`single+detect`, `shared`, `shared+detect`, `shared+relay`,
`shared+relay+ble`, `single+relay+ble`. For a paired reference, `shared` and
`nr_30ms` are also run under the independent model on the same scenes.

**Hypotheses** (M1–M5 pass if they hold in at least 5 of 6 scenes):
- **M1 (`main`):** `shared` misses ≤ `nr_30ms`. The headline claim.
- **M2 (`main`):** `shared` misses ≤ `wifi8`.
- **M3 (`stress`):** `shared+relay` misses ≤ `nr_30ms`.
- **M4 (`stress`):** `shared+relay+ble` misses ≤ `nr_30ms`.
- **M5 (`stress`):** `single+relay+ble` misses ≤ `nr_30ms`.
- **M6:** the blockage rule still helps one radio: `single+detect` cuts pooled misses by at least 10% in both conditions. The development scene gave 13% and 10%, so this is expected to be marginal.

**Expected to fail, reported not tested:** `shared` against `nr_30ms` in
`stress` (it already loses under independent blockage).

**Reported, not tested:** `shared+detect` against `shared`; how much each
arm degrades from the independent model; rescue counts; miss causes.

Run: `python -m wirelessjev.native.run_fleet moving --workers 4`.
Results are appended below after the run.

## Results (15 Nov 2026)
Scenes 570–575, 100 robots, 300 s, 600,000 control packets per scene per
arm. Raw rows: `artifacts/native/moving_blockers.json`. Intervals are 95%
bootstrap over the six scenes. All figures are simulation.

**Verdicts**

| # | Hypothesis | Result |
|---|---|---|
| M1 | Two AP paths ≤ 5G (`main`) | **Pass, 6/6** (14.4 against 171.4) |
| M2 | Two AP paths ≤ Wi-Fi 8 (`main`) | **Pass, 6/6** (14.4 against 439.2) |
| M3 | Two AP paths + relay ≤ 5G (`stress`) | **Pass, 6/6** (11.7 against 171.4) |
| M4 | Two AP paths + relay + Bluetooth ≤ 5G (`stress`) | **Pass, 6/6** (2.2 against 171.4) |
| M5 | One radio + relay + Bluetooth ≤ 5G (`stress`) | **Pass, 6/6** (4.4 against 171.4) |
| M6 | Blockage rule cuts one-radio misses ≥ 10% in both conditions | **Fail**: 27% *more* misses in `main` (406.4 → 516.1); 27% fewer in `stress` (2,932.2 → 2,128.1) |

**Missed control packets per million, moving blockers**

| Arm | Realistic | Stress |
|---|---|---|
| Private 5G model | 171.4 [143.3–199.4] | 171.4 [143.3–199.4] |
| Wi-Fi 8 model | 439.2 [361.7–525.0] | 1,096.9 [993.3–1,199.7] |
| One radio | 406.4 [318.3–501.9] | 2,932.2 [2,420.3–3,685.6] |
| One radio + blockage rule | 516.1 [432.2–620.0] | 2,128.1 [1,947.1–2,278.6] |
| Two AP paths | 14.4 [8.1–21.1] | 255.0 [183.9–328.6] |
| Two AP paths + blockage rule | 76.9 [53.6–119.2] | 458.9 [370.0–546.2] |
| Two AP paths + relay | 0 | 11.7 [4.7–18.1] |
| Two AP paths + relay + Bluetooth | 0 | 2.2 [0.6–3.9] |
| One radio + relay + Bluetooth | 0 | 4.4 [1.7–7.5] |

**How much each arm degrades** (same six scenes, independent → moving)

| Arm | Realistic | Stress |
|---|---|---|
| Private 5G model | 52.2 → 171.4 (3.3×) | 52.2 → 171.4 (3.3×) |
| Two AP paths | 3.1 → 14.4 (4.6×) | 123.6 → 255.0 (2.1×) |

## What the results say
1. **The headline claim survives shared blockage.** Two AP paths still beat the 5G model in the realistic condition in all six scenes, by 12× (it was 17× on these scenes under independent blockage). Uptime is 99.986% against 99.813%.
2. **Path diversity is worth less than the independent model said.** Two-path misses rose 4.6× when the two paths could be blocked together; 45 of the 52 two-path misses happened with both paths blocked.
3. **Everyone gets worse, including 5G.** The 5G model's misses rose 3.3×: shorter, more frequent blockages mean more handovers and more time on a blocked cell.
4. **The stress condition needs the relay, and the relay is no longer perfect.** Two AP paths alone lose to 5G (255.0 against 171.4). With relay it is 11.7; with relay and Bluetooth 2.2. These are not zeros any more, because the same forklifts block relay links. One radio with relay and Bluetooth is at 4.4.
5. **The blockage rule is not safe to ship as it is.** Under moving blockers it made one-radio misses 27% worse in the realistic condition, and made two-path misses worse in both conditions (14.4 → 76.9; 255.0 → 458.9). Blockages now last under a second, so leaving the AP costs more than waiting, and the rule's roams show up as "no radio up" and "weak link" misses. It helped only one radio in stress. It stays off by default.

## What this changes
- The rule was tuned and confirmed under independent blockage (`docs/BLOCKAGE_DETECTOR.md`), where events last about 2 s. Its benefit depends on blockage length, which is a property of the site. Tuesday's blockage trials measure that length on real signal.
- A response that fits short blockages is to switch the packet to the other path, not to roam. The two-path arm already does this per packet, which is why it needs no detector.
- Knowing a blockage is coming (`docs/OUTAGE_RISK.md`) is only useful with a response that is cheap when wrong.

## Limits
As listed above: AP and forklift dimensions are assumed, the relay keeps its
same-slot and always-cooperates assumptions, and nothing has run on
hardware.
