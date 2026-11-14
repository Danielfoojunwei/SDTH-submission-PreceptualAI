# Fleet-only capabilities: safe plan, robot relays, Bluetooth control path (pre-registered)

A second AP path is our strongest lever but can be copied. These three need
control of the robot or a fleet, which 5G and Wi-Fi 8 do not have.

## What is modelled (`src/wirelessjev/native/sim.py`)
1. **Safe plan.** The robot carries `safe_plan_s` seconds of trajectory. It
   stops when the plan runs out, not after 200 ms of silence. With
   link-aware speed it halves its speed where the link is predicted weak, so
   the plan lasts twice as long; slow time costs half throughput.
   `task_throughput` = 1 − stopped time − 0.5 × slow time.
2. **Robot relay.** When a robot's own paths fail in a slot, one of its two
   nearest robots within 15 m that has a working link forwards the packet:
   two hops at the most robust 5 GHz rate, charged as airtime. The
   robot-to-robot link uses log-distance loss, rack rows between the two
   robots and its own fading.
3. **Bluetooth control path.** One LE 1M attempt per 10 ms slot to a gateway
   at an AP site other than the Wi-Fi primary (8 dBm, 11 dB SNR needed,
   independent fade per attempt from hopping, the same blockage as that
   site, and the same 2.4 GHz noise rise as Wi-Fi in the stress condition).
   **This assumes a Bluetooth gateway at each AP site**, which is extra
   infrastructure.

## Optimistic simplifications (limits)
- A relayed packet arrives in the same 10 ms slot.
- Relay helpers always cooperate.
- Bluetooth airtime is not charged against Wi-Fi.
- The safe plan assumes the robot's local obstacle avoidance makes following a 2 s plan without contact acceptable; that is a safety-case question this simulation cannot answer.

## Test
Fresh scenes 550–555, 100 robots, 300 s, learned map from a 60 s survey.

**Conditions:**
- `main`: realistic collisions and interference.
- `stress`: 3× collisions, interference, 2.4 GHz noise +10 dB.

**Arms:**
- **Baselines:** `nr_30ms`, `wifi8`, `single` (one radio), `shared` (second association on the same card).
- **Safe plan:** `single+plan` (2 s plan), `single+plan+speed`.
- **Relay and Bluetooth, one radio:** `single+relay`, `single+ble`, `single+relay+ble`.
- **Relay and Bluetooth, two AP paths:** `shared+relay`, `shared+ble`, `shared+relay+ble`.

**Hypotheses** (each passes if it holds in at least 5 of 6 scenes):
- **D1 (`main`):** `single+plan` stopped time ≤ `nr_30ms` stopped time under the 200 ms rule.
- **D2 (`main`):** `single+relay` misses ≤ half of `single`.
- **D3 (`main`):** `single+ble` misses ≤ half of `single`.
- **D4 (`stress`):** `shared+relay+ble` misses ≤ `nr_30ms`.
- **D5 (`stress`):** `single+relay+ble` (existing radio only) misses ≤ `nr_30ms`.

**Reported, not tested:** task throughput with and without link-aware speed; stopped time for every arm at 0.2, 1 and 2 s thresholds (so "5G with a safe plan" can be read off); relay and Bluetooth rescue counts; miss causes.

Results are appended below after the run.
