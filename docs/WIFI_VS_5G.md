# Vertical Wi-Fi stack vs private 5G: pre-registered protocol

**Goal:** indoors, our Wi-Fi system should match or beat private 5G on the
metrics 5G and its SLAs are bought for: control reliability within a deadline,
uptime, worst-robot tails and safety stops.

## Simulator additions (all in `src/wirelessjev/native/`)
- **Private-5G baseline (`simulate_nr`).** Deliberately generous to 5G:
  - n78 100 MHz TDD on the same 16 sites;
  - licensed, clean spectrum, with no inter-cell interference modelled;
  - scheduled uplink (no contention);
  - UE at 23 dBm (Wi-Fi client 15 dBm);
  - 3.5 GHz path loss;
  - CQI link adaptation on the current fading block;
  - HARQ retries;
  - A3 handover with ~200 ms time-to-trigger;
  - handover interruption of 30 ms (`nr_30ms`) or 0 ms make-before-break (`nr_daps`).
- **Wi-Fi realism:**
  - **Co-channel interference.** Uplink airtime from stations in *other* contention domains on the same channel raises the noise at the AP.
  - **CSMA collisions.** Per-attempt loss is 1 − exp(−λ·(stations − 1)) per domain.
- **New native capabilities:**
  - **Online map learning:** robots' beacon measurements update RadioJEV's map every 10 s.
  - **Transmit-power politeness:** lower power when there is headroom above the top MCS.
  - **Cadence-timed roams:** planned roams fire right after a control delivery.
  - **Adaptive video:** encoded at 60% of the predicted goodput.
  - **Dual radio:** a second client radio on another AP, giving make-before-break and blockage diversity. *This needs extra hardware.*
  - **Scheduled control:** needs AP cooperation.
- **Metrics:**
  - control misses per million (50 ms deadline);
  - 1 s uptime (robot-seconds with no miss);
  - safety-stop fraction (stop after 200 ms of control silence, 1 s to resume);
  - worst-decile robot;
  - p99 longest gap;
  - video, bulk and roams.

## Development (scene 400 only, 120 s)
| Arm | Misses per million | Uptime |
|---|---|---|
| nr_30ms | 46 | 0.99958 |
| mlo_split | 7988 | 0.958 |
| native_gated | 292 | 0.996 |
| vertical (dual radio) | 12.5 | 0.99983 |

## Test (fresh scenes 530–535, 300 s, 60 s survey, learned map)
**Conditions:**
- `main`: λ = 0.01 with interference.
- `stress`: λ = 0.03, interference, and a +10 dB 2.4 GHz noise rise.
- `lenient`: no collisions and no interference (the earlier regime).

**Arms:**
- **5G baselines:** `nr_30ms`, `nr_daps`.
- **Earlier Wi-Fi arms:** `mlo_split`, `native_gated`.
- **Vertical arms:**
  - `vertical_single`: every capability except the second radio. Existing SmartCar hardware.
  - `vertical_client`: adds the second radio. No AP cooperation.
  - `vertical_full`: adds scheduled control.
- **Ablations (`main` only), each `vertical_client` minus one capability:** `-online`, `-power`, `-cadence`, `-video`.

**Hypotheses** (each passes if it holds in at least 5 of 6 scenes):
- **V1 (`main`):** `vertical_client` control misses ≤ `nr_30ms`.
- **V2 (`main`):** `vertical_client` 1 s uptime ≥ `nr_30ms`.
- **V3 (`main`):** `vertical_client` safety-stop fraction ≤ `nr_30ms`.
- **V4 (`main`):** `vertical_single` (existing hardware) control misses ≤ `nr_30ms`.
- **V5 (`stress`):** `vertical_client` control misses ≤ `nr_30ms`.

Results go in `WIFI_VS_5G_RESULTS.md`.
