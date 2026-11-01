# Vertical Wi-Fi stack vs private 5G: results

- **Protocol:** [WIFI_VS_5G.md](WIFI_VS_5G.md), frozen in `f958d97` before the test run.
- **Runs:** fresh scenes 530–535, 100 robots, 300 s each, learned map from a 60 s survey.
- **Raw data:** [`artifacts/native/wifi_vs_5g.json`](../artifacts/native/wifi_vs_5g.json).
- **Evidence class:** simulation only; the 5G model is deliberately generous to 5G.

## Verdicts
| | Claim | Result |
|---|---|---|
| V1 | Dual-radio stack has ≤ 5G control misses (`main`) | **Pass, 6/6** |
| V2 | Dual-radio stack 1 s uptime ≥ 5G (`main`) | **Pass, 6/6** |
| V3 | Dual-radio stack safety-stop time ≤ 5G (`main`) | **Pass, 6/6** |
| V4 | Existing SmartCar hardware (single radio) ≤ 5G misses (`main`) | **Fail, 0/6** |
| V5 | Dual-radio stack ≤ 5G misses under `stress` | **Fail, 0/6** |

## Main condition (realistic collisions and interference)
Fleet means over six scenes. Misses are per million control packets with a 50 ms deadline.

| Arm | Misses per million | Worst-decile robot | p99 gap (ms) | 1 s uptime | Safety-stop time | Airtime | Roams | Per scene |
|---|---|---|---|---|---|---|---|---|
| Private 5G, 30 ms handover | 52.8 | 0.041% | 185 | 0.99933 | 0.004% | 2.39 | 3018 | 33–68 |
| Private 5G, make-before-break | 45.0 | 0.034% | 167 | 0.99934 | 0.003% | 2.40 | 3018 | 28–58 |
| Fixed rule (`mlo_split`) | 5092.8 | 1.93% | 1658 | 0.96022 | 0.90% | 8.05 | 690 | 3205–6178 |
| Gated native, single radio | 313.1 | 0.16% | 325 | 0.99589 | 0.030% | 6.71 | 783 | 207–497 |
| Vertical, single radio (existing hardware) | 293.9 | 0.15% | 318 | 0.99609 | 0.030% | 6.87 | 777 | 225–463 |
| **Vertical, dual radio** | **3.1** | **0.003%** | **89** | **0.99995** | **0** | 7.27 | 777 | 0–8.3 |
| Vertical, dual radio + scheduled control | 3.1 | 0.003% | 89 | 0.99995 | 0 | 7.26 | 777 | 0–8.3 |

**Ablations of the dual-radio stack, each minus one capability:**

| Removed | Misses per million |
|---|---|
| Online map learning | 3.6 |
| Power politeness | 3.6 |
| Cadence-timed roams | 3.1 |
| Adaptive video | 3.1 |

## Stress (heavier collisions, +10 dB 2.4 GHz noise) and lenient (no collisions or interference)
| Arm | Stress: misses per million | Stress: uptime | Lenient: misses per million | Lenient: uptime |
|---|---|---|---|---|
| Private 5G, 30 ms | 52.8 | 0.99933 | 52.8 | 0.99933 |
| Fixed rule | 10466 | 0.91765 | 693 | 0.98756 |
| Vertical, single radio | 1642 | 0.98791 | 147.5 | 0.99742 |
| **Vertical, dual radio** | **91.4** | **0.99903** | **0.6** | **0.99999** |

5G is unchanged across conditions because its licensed spectrum is unaffected by Wi-Fi collisions and 2.4 GHz noise.

## What this means
1. **With a second client radio, Wi-Fi beats a generous private-5G model indoors.**
   - About 17× fewer missed control packets.
   - Uptime: 99.995% of robot-seconds clean, against 99.933%.
   - Zero safety stops.
   - Half the worst-case gap.

   This holds in realistic and clean conditions.
2. **The second radio is the capability that matters.**
   - Each other added layer moves the result by a few percent or nothing: online learning, power politeness, cadence-timed roams, adaptive video, scheduled control.
   - Without the second radio, the full vertical stack (294 per million) is no better than the earlier gated planner (313), and about 6× *worse* than 5G.
   - The decisive edge is make-before-break across APs plus diversity against blockage that hits one AP path. Every other layer optimises the margins of a single link, which blockage can still cut.
3. **Under heavy contention and a noisy 2.4 GHz band, even the dual-radio stack loses to 5G** (91 vs 53 per million). Unlicensed contention is the regime private 5G is genuinely built for.
4. **Existing SmartCar hardware alone cannot match private 5G** in this simulation. The claim needs a second Wi-Fi radio per robot: a USB adapter, or a dual-radio module.

## Limits
- Simulation only.
- The 5G model is generous: no inter-cell interference, the same 16 sites, and scheduled access treated as perfect.
- The Wi-Fi collision model is a one-parameter approximation.
- Both radios share the robot's mobility and blockage process per AP path.
- The second radio's cost, power draw and driver support are not modelled.
- The 5G A3 settings ping-pong (about 3,000 handovers in 300 s); with 30 ms interruptions inside a 50 ms deadline this costs 5G little.
- Six scenes, one realisation each.
