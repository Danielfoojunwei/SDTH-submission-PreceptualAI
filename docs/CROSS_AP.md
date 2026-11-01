# Cross-AP diversity without extra hardware: pre-registered protocol

Follows [WIFI_VS_5G_RESULTS.md](WIFI_VS_5G_RESULTS.md), where only a second
radio beat private 5G and the other capabilities barely moved the result.

## Why the other capabilities did not move the needle
A miss-cause breakdown added to the simulator tags every missed control packet with the most serious condition it met while waiting:
1. no radio up;
2. every connected AP path blocked;
3. control link below the MCS 0 threshold (weak link);
4. contention;
5. other.

On development scene 400 (main condition, 120 s):

| Arm | Misses per million | No radio up | All paths blocked | Weak link | Contention |
|---|---|---|---|---|---|
| Private 5G | 46 | 0 | 5 | 6 | 0 |
| Wi-Fi 8, generous (seamless switching every 100 ms) | 304 | 0 | 7 | 66 | 0 |
| Vertical, single radio | 454 | 34 | 20 | 55 | 0 |
| Vertical + onboard chip as second radio | 21 | 1 | 0 | 4 | 0 |
| Vertical + second BE200 | 12.5 | 1 | 0 | 2 | 0 |

**What the breakdown shows:**
- **Contention never causes a miss.** Multi-rate retries rescue collisions, so scheduled control and power politeness target a failure mode that does not occur at this load.
- **Roam cost is not binding.** Setting it to zero changed nothing.
- **The binding failure is a moment when the single AP path is dead,** from a deep fade, a blockage or a noisy band. Only a second simultaneous path through *another AP* removes it.
- **A self-inflicted cause.** The "no radio up" misses of the single radio came from the trust gate's fallback roam, which used a 60 ms 802.11r cost that is longer than the 50 ms deadline and not cadence-timed. It is now our own 30 ms pre-keyed roam. This is a bug fix made before the test run.

## New arms
- **`vertical_shared`:** one BE200 holding a second association to another AP, time-shared. It is tried only after the primary retry chain fails, at an extra 2 ms channel-switch cost. A scan blocks both associations.
  - **What it needs:** either BE200/iwlwifi support for a second station interface (unverified; the capability probe will check), or Wi-Fi 8 multi-AP infrastructure running our per-packet cross-AP policy.
- **`vertical_onboard`:** the second radio is the Pi's onboard CYW43455, modelled as 1×1 Wi-Fi 5 with no 6 GHz and 5 dB weaker. On the car this means moving management off the hotspot.
- **`wifi8`:** generous Wi-Fi 8 seamless mobility. One radio, any AP, switched every 100 ms on lagged measurements, no outage, no per-packet cross-AP retry.
- **`vertical_shared-levers`:** the shared association with the gate only. Online map, power politeness, cadence and adaptive video are all off, to measure what those levers add once diversity exists.

## Test
- **Scenes:** fresh 540–545, 300 s, 60 s survey, learned map, interference on.
- **Conditions:**
  - `main`: λ = 0.01.
  - `stress`: λ = 0.03 and a +10 dB 2.4 GHz noise rise.
  - `heavy`: 15% of robots stream 6 Mb/s video and bulk uploads are doubled.

**Hypotheses** (each passes if it holds in at least 5 of 6 scenes):
- **X1 (`main`):** `vertical_shared` misses ≤ `nr_30ms`.
- **X2 (`main`):** `vertical_onboard` misses ≤ `nr_30ms`.
- **X3 (`main`):** `vertical_shared` misses ≤ `wifi8`.
- **X4 (`stress`):** `vertical_shared` misses ≤ `nr_30ms`.
- **X5 (`stress`):** `vertical_onboard` misses ≤ `nr_30ms`.
- **X6 (`heavy`):** `vertical_shared` misses ≤ `nr_30ms`.

**Reported, not tested:** miss causes per arm; the difference between `vertical_shared` and `vertical_shared-levers` in each condition; uptime and safety stops.

## Development check before freezing (scene 401, 60 s)
- **Bug fix confirmed.** The fallback-roam fix removed the single radio's "no radio up" misses.
- **Main and heavy conditions:** `vertical_shared`, `vertical_onboard` and `vertical_client` had zero misses.
- **Stress condition:** `vertical_shared` with the levers had 100 per million; without them (`vertical_shared-levers`) it had 25. The levers may *hurt* under stress, and the test reports this ablation in every condition.

No thresholds or arms were changed after this check.

## 10-robot replication (added before it ran)
The same arms, conditions and hypotheses X1–X6 are run with 10 robots on the same warehouse scenes 540–545 (`--robots 10`, output `artifacts/native/cross_ap_10robots.json`).
- **Purpose:** check whether the conclusions hold for a 10-car fleet, the SmartCar milestone size.
- **Difference from the original 10-car twin:** this is the warehouse twin with 10 robots, not the earlier 3-AP 10-car twin.
- **Fewer fixed video robots:** the warehouse's three fixed video robots (indices 16, 41 and 77) do not exist in a 10-robot fleet, so video comes only from the random 4%. In the `heavy` condition, 15% of robots stream.
