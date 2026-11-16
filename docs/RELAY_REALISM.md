# Relay realism: does the stress win survive a slower, less reliable relay? (pre-registered)

The stress condition is won only when robots relay for each other
([DIRECTIONS_123.md](DIRECTIONS_123.md), [MOVING_BLOCKERS.md](MOVING_BLOCKERS.md)).
The relay model was optimistic in two ways: a forwarded packet arrived in
the same 10 ms slot, and a helper was always free to forward. This test
removes both, in steps.

## What changes (`native_ext/sim_fleet.py`)
- **Delay** (`relay_delay_slots`): forwarding takes extra 10 ms slots. A control packet must arrive within 50 ms, so a forwarded packet that would land later does not count. With a 20 ms delay the relay has three tries per packet, not five; with 30 ms, two.
- **Helper availability** (`relay_coop`): in each 100 ms block a helper is free to forward with this probability; otherwise it is busy with its own traffic.

Unchanged: helpers are the two nearest robots within 15 m with a working
link, relay links are blocked by the same forklifts, and both hops use the
most robust 5 GHz rate.

| Level | Delay | Helper free |
|---|---|---|
| Optimistic (as before) | 0 ms | 100% |
| Realistic | 20 ms | 70% |
| Pessimistic | 30 ms | 50% |

The levels were set before any run. There was no development check, so
nothing was tuned.

## Still not modelled
- The helper must receive the packet from the AP in the first place; this is approximated by the helper's own link quality.
- The channel-switch cost for a robot and helper on different channels.
- Any protocol for choosing a helper. The model assumes the best of the two nearest forwards.

## Test
Fresh scenes 590–595, 100 robots, 300 s, learned map from a 60 s survey,
moving blockers.

**Conditions:** `main` and `stress`.

**Arms:** `nr_30ms`; `shared` (two AP paths); `shared+relay[optimistic]`,
`shared+relay[realistic]`, `shared+relay[pessimistic]`.

**Hypotheses** (each passes if it holds in at least 5 of 6 scenes, `stress`):
- **R1:** `shared+relay[realistic]` misses ≤ `nr_30ms`.
- **R2:** `shared+relay[pessimistic]` misses ≤ `nr_30ms`. This may fail; it marks how much relay quality the stress win needs.
- **R3:** `shared+relay[pessimistic]` misses ≤ `shared`. Even a poor relay should not hurt.

**Reported, not tested:** the `main` condition for every arm; rescue counts; miss causes.

Run: `python -m wirelessjev.native_ext.run_fleet relay --workers 4`.
Results are appended below after the run.

## Results (15 Nov 2026)
Scenes 590–595, 100 robots, 300 s, 600,000 control packets per scene per
arm, moving blockers. Raw rows: `artifacts/native/relay_realism.json`.
Intervals are 95% bootstrap over the six scenes. All figures are simulation.

**Verdicts** (`stress`)

| # | Hypothesis | Result |
|---|---|---|
| R1 | Realistic relay ≤ 5G | **Pass, 6/6** (21.4 against 136.4) |
| R2 | Pessimistic relay ≤ 5G | **Pass, 6/6** (56.7 against 136.4) |
| R3 | Pessimistic relay ≤ two AP paths alone | **Pass, 6/6** (56.7 against 234.2) |

**Missed control packets per million**

| Arm | Realistic condition | Stress |
|---|---|---|
| Private 5G model | 136.4 [116.4–156.4] | 136.4 [116.4–156.4] |
| Two AP paths | 12.8 [6.4–19.2] | 234.2 [172.7–308.9] |
| + relay, optimistic (0 ms, always free) | 0 | 1.1 [0.0–2.8] |
| + relay, realistic (20 ms, free 70%) | 1.7 [0.0–4.4] | 21.4 [11.9–31.4] |
| + relay, pessimistic (30 ms, free 50%) | 3.1 [1.1–6.1] | 56.7 [40.0–73.9] |

## What the results say
1. **The stress win does not depend on the optimistic relay.** With a 20 ms forwarding delay and helpers free 70% of the time, two AP paths plus relay still beat the 5G model by 6.4× in stress. With 30 ms and 50%, by 2.4×, in all six scenes, and the intervals do not overlap.
2. **Relay quality matters a lot.** Misses go from 1.1 to 21.4 to 56.7 as the relay gets slower and less available. The honest figure to quote is the realistic one, not the optimistic one.
3. **Even a poor relay helps.** The pessimistic relay cuts two-path misses by 76% in stress and by 76% in the realistic condition (12.8 → 3.1).
4. **How often it is used:** about 200–250 rescues per scene in stress out of 600,000 packets, and about 12 in the realistic condition. The relay is a rare fallback, so its airtime cost is small.

## Limits
The items under "Still not modelled" stand, and nothing has run on hardware.
Whether the SmartCar's radio can hold a robot-to-robot link alongside its AP
connections is the first thing to test once a second device is available.
