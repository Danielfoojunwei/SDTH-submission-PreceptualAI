# Fleet-only capabilities: safe plan, robot relays, Bluetooth control path (pre-registered)

A second AP path is our strongest lever but can be copied. These three need
control of the robot or a fleet, which 5G and Wi-Fi 8 do not have.

## What is modelled (`src/wirelessjev/native_ext/sim_fleet.py`)
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

## Results (15 Nov 2026)
Scenes 550–555, 100 robots, 300 s, 600,000 control packets per scene per arm.
Raw rows: `artifacts/native/fleet_directions.json`. Intervals are 95%
bootstrap over the six scenes. All figures are simulation.

**Verdicts**

| # | Hypothesis | Result |
|---|---|---|
| D1 | Safe plan: stopped time ≤ 5G's under the 200 ms rule (`main`) | **Pass, 6/6** (0 against 0.0012% of time) |
| D2 | Relay halves one-radio misses (`main`) | **Pass, 6/6** (6.9 against 197.5) |
| D3 | Bluetooth path halves one-radio misses (`main`) | **Pass, 6/6** (0 against 197.5) |
| D4 | Two AP paths + relay + Bluetooth ≤ 5G (`stress`) | **Pass, 6/6** (0 against 57.8) |
| D5 | One radio + relay + Bluetooth ≤ 5G (`stress`) | **Pass, 6/6** (0 against 57.8) |

**Missed control packets per million**

| Arm | Realistic | Stress |
|---|---|---|
| Private 5G model (`nr_30ms`) | 57.8 [41.4–74.2] | 57.8 [41.4–74.2] |
| Wi-Fi 8 model | 293.1 [219.4–380.6] | 861.9 [778.9–961.9] |
| One radio (`single`) | 197.5 [134.2–270.8] | 2,150.6 [1,991.9–2,303.6] |
| Two AP paths, same card (`shared`) | 2.5 [0.6–4.7] | 96.7 [81.9–110.3] |
| One radio + relay | 6.9 [0.8–13.6] | 34.4 [16.4–52.8] |
| One radio + Bluetooth | 0 | 16.9 [8.3–29.4] |
| One radio + relay + Bluetooth | 0 | 0 |
| Two AP paths + relay | 0 | 1.7 [0.0–3.9] |
| Two AP paths + Bluetooth | 0 | 8.1 [2.8–15.3] |
| Two AP paths + relay + Bluetooth | 0 | 0 |

A zero is 0 misses in 3.6 million packets, which bounds the rate below about
1 per million at 95% confidence within this model.

**Safe plan and link-aware speed** (one radio; share of robot-time stopped)

| Arm | Realistic: stopped | Realistic: throughput | Stress: stopped | Stress: throughput |
|---|---|---|---|---|
| Stop after 200 ms silence | 0.0253% | 0.99975 | 0.3927% | 0.99607 |
| 2 s safe plan | 0 | 1.00000 | 0.0252% | 0.99975 |
| 2 s safe plan + link-aware speed | 0 | 0.99848 | 0.0055% | 0.99234 |

The safe plan does not change which packets are missed (197.5 and 2,150.6
in all three rows); it changes what a miss costs.

**How often the extra paths were used** (rescues per scene, of 600,000 packets)

| Arm | Realistic | Stress |
|---|---|---|
| One radio + relay | 227 | 2,245 |
| One radio + Bluetooth | 244 | 2,169 |
| Two AP paths + relay | 6 | 180 |
| Two AP paths + Bluetooth | 6 | 146 |

## What the results say
1. **The stress condition is no longer lost in simulation.** Two AP paths alone lose to the 5G model there (96.7 against 57.8). Adding robot relay takes it to 1.7; adding relay and Bluetooth takes it to 0.
2. **Relay and Bluetooth do on one radio what the second AP path does.** One radio with either extra path beats the 5G model in both conditions (6.9 and 0 realistic; 34.4 and 16.9 stress). The one-radio relay result in stress is the weakest of these: its interval (16.4–52.8) overlaps the 5G model's (41.4–74.2).
3. **The safe plan removes almost all stops.** A 2 s plan cuts stopped time by 94% in stress and to zero in the realistic condition.
4. **Link-aware speed is a loss.** It removes a few more stops but slows the robot 0.3% to 1.5% of the time, so throughput is lower than with the plan alone in both conditions. It should stay off.

## How far to trust the zeros
The zeros come from the optimistic simplifications listed above, not only
from the idea. In particular:
- a relayed packet arrives in the same 10 ms slot, and helpers always cooperate;
- Bluetooth airtime is free and there is a gateway at every AP site, which is extra infrastructure;
- relay and Bluetooth fades are independent of the Wi-Fi fades that caused the failure.

A real system will not reach zero. The supported claim is narrower: an
independent extra path, even a slow one, removes most of the misses the AP
paths leave, including in the noisiest condition. Whether the SmartCar's
radios can hold a robot-to-robot link or a Bluetooth heartbeat alongside
Wi-Fi is untested; the Bluetooth and relay tests are not in Tuesday's
runbook.
