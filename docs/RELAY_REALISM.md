# Relay realism: does the stress win survive a slower, less reliable relay? (pre-registered)

The stress condition is won only when robots relay for each other
([DIRECTIONS_123.md](DIRECTIONS_123.md), [MOVING_BLOCKERS.md](MOVING_BLOCKERS.md)).
The relay model was optimistic in two ways: a forwarded packet arrived in
the same 10 ms slot, and a helper was always free to forward. This test
removes both, in steps.

## What changes (`native/sim.py`)
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

Run: `python -m wirelessjev.native.run_fleet relay --workers 4`.
Results are appended below after the run.
