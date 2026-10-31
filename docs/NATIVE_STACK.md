# Native cross-layer stack: pre-registered protocol

Written before any main-campaign run. Thresholds below are not to be changed
after results are seen; any later change is reported as a deviation.

## Setup
- **Fleet and world:** 100 robots on the warehouse twin (`make_warehouse`) for 300 s per episode, in 10 ms slots.
  - Each of the 16 AP sites is a Wi-Fi 7 multi-link AP (MLD) with three links: 2.4 GHz (20 MHz), 5 GHz (80 MHz) and 6 GHz (160 MHz).
- **Link layer:** packet error from the ns-3 NIST curves (`artifacts/warehouse_v2/studies/wifi_link_curves.json`). 6 GHz reuses the 5 GHz curves with double the rate, +1.9 dB path loss and +1 dB per rack crossed.
- **Fading and blockage:**
  - Nakagami-m=2 block fading every 100 ms, independent per link.
  - Forklift/person blockage of −15 dB, shared by all links of one robot–site path (mean 2 s, 1.5 events per robot-minute).
  - Neither is known to any policy.
- **Traffic:**
  - Control: 20 Hz, 600 B, 50 ms deadline.
  - Teleop video: the warehouse video robots, 30 fps, 150 ms deadline.
  - Bulk: elastic, from the robots' map-upload flows.
- **Airtime:** shared per AP link inside contention domains, minus external occupancy, with priority control > video > bulk.
- **Roam costs** (assumed, swept in the sensitivity runs):

  | Roam type | Cost |
  |---|---|
  | Legacy | 4 s |
  | 802.11r with the same IP | 60 ms |
  | Pre-keyed, planned | 30 ms |

  Reactive policies pay a 30 ms background scan every 5 s.
- **Forecasts:** route-based, true future positions plus a spatially correlated map-error field with σ = 4 dB (matching corrected E7). Fading and blockage are never forecast.

## Policies
1. `legacy`: single 5 GHz link (2.4 GHz fallback), roam on −75 dBm with an 8 dB better candidate, full roam cost, reactive MCS.
2. `ft11r`: as legacy, with 802.11r and the same IP.
3. `mlo_default`: 3-link MLD, EMLSR on the link with the best *lagged* measurement, all traffic on all links, reactive roam with 802.11r, reactive MCS.
4. `native`: WirelessJEV + RadioJEV planned stack, with these components:
   - route-forecast link choice;
   - control mapped to the predicted-best link;
   - risk-gated STR duplication of control;
   - predictive MCS;
   - pre-keyed, route-timed, fleet-staggered roams;
   - map-based scanning every 30 s;
   - data timing (bulk deferred while predicted control risk is high).
5. **Ablations:** `native` minus each component; `native_blind` (no route, assumes stationary); `native_oracle` (σ_map = 0).
6. `native_layered` (simulation/SDR only, M1): control rides the base layer of bulk frames, using the Sionna-measured SNR shift and no separate channel access.

## Metrics
Per robot, then fleet:
- control-deadline miss fraction (primary);
- worst-decile robot miss fraction;
- p99 longest control gap;
- video on-time fraction;
- bulk throughput;
- airtime used;
- roams.

## Hypotheses and thresholds
- **H1:** `native` cuts the fleet control-miss fraction by at least 50% vs `mlo_default`, using at most 15% more airtime, in at least 5 of 6 scenes.
- **H2:** `native` improves worst-decile control misses vs `mlo_default` in at least 5 of 6 scenes.
- **H3:** at σ_map = 8 dB, `native` still beats `mlo_default` on control misses in at least 5 of 6 scenes.
- **H4:** `native_blind` loses at least half of `native`'s gain over `mlo_default`. If not, route knowledge is not the source of the edge.
- **H5:** with the sensitivity setting of a 1 s legacy roam cost, `native` still beats `ft11r` on control misses in at least 5 of 6 scenes.
- **M1 (exploratory):** the layered constellation lowers the critical-bit SNR needed for 1% block error, at equal symbols and power, compared with time sharing. Bulk loss is reported, not hidden.

Scenes are warehouse seeds 500–505. All policies see identical truth, fading
and blockage draws per scene. Results go in `NATIVE_STACK_RESULTS.md`.
