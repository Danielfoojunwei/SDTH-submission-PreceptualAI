# Native cross-layer stack: results

- **Protocol:** [NATIVE_STACK.md](NATIVE_STACK.md), committed in `00173d6` before any test-scene run.
- **Runs:** test scenes 500–505, 100 robots, 300 s each, 10 ms slots. All policies share identical truth, fading and blockage per scene.
- **Raw data:** [`artifacts/native/campaign.json`](../artifacts/native/campaign.json) and [`artifacts/native/phy_layered.json`](../artifacts/native/phy_layered.json).
- **Evidence class:** simulation only. Assumed parameters: roam costs, blockage process, a 6 GHz link derived from 5 GHz curves, and contention domains from the twin.

## Development history (before the freeze)
Development used scenes 400–401 only. Four early modelling errors were found and fixed. Each one favoured or handicapped a side unfairly:
1. **No rate-fallback retries.** Control frames had no multi-rate retry, which made control artificially fragile.
2. **A weak multi-link baseline.** Default multi-link chose links by SNR instead of throughput.
3. **Wrong reference rate for the layered gain.** It was measured against MCS 0 instead of 16-QAM r=1/2 (MCS 3).
4. **Deferred bulk was discarded instead of queued.**

`mlo_split`, a fixed rule (control pinned to 2.4 GHz, bulk on the fastest link), was added during development as a tougher baseline. It is reported as a post-hoc baseline.

## Fleet means over six test scenes
| Arm | Control misses | Worst-decile robot | p99 longest gap (ms) | Video on time | Bulk (Mb/s) | Airtime | Roams |
|---|---|---|---|---|---|---|---|
| legacy (4 s roam) | 11.295% | 16.687% | 9669 | 0.888 | 487 | 7.06 | 850 |
| legacy, 1 s roam | 2.933% | 4.510% | 2336 | 0.970 | 488 | 7.29 | 872 |
| ft11r (60 ms roam) | 0.203% | 0.570% | 901 | 0.997 | 488 | 7.39 | 872 |
| mlo_default | 1.391% | 2.679% | 516 | 0.973 | 500 | 7.82 | 702 |
| mlo_split (post hoc) | 0.064% | 0.215% | 453 | 0.973 | 500 | 7.82 | 702 |
| **native** | **0.019%** | **0.146%** | 431 | 0.976 | 500 | **6.61** | 790 |
| native − route forecast | 0.023% | 0.181% | 576 | 0.976 | 500 | 7.22 | 813 |
| native − duplication | **0.010%** | **0.078%** | **269** | 0.977 | 500 | 6.55 | 790 |
| native − predictive MCS | 0.019% | 0.145% | 431 | 0.976 | 500 | 6.61 | 790 |
| native − planned roam | 0.065% | 0.219% | 471 | 0.973 | 500 | 7.80 | 702 |
| native − stagger | 0.019% | 0.145% | 431 | 0.976 | 500 | 6.61 | 791 |
| native − data timing | 0.019% | 0.146% | 431 | 0.976 | 500 | 7.14 | 790 |
| native − reflex | 0.023% | 0.166% | 476 | 0.976 | 500 | 6.62 | 790 |
| native + layered PHY | 0.019% | 0.146% | 431 | 0.976 | 500 | 6.35 | 790 |
| native, map error σ = 8 dB | 0.202% | 1.104% | 2475 | 0.971 | 498 | 7.70 | 929 |
| native, perfect map | 0.010% | 0.075% | 242 | 0.977 | 500 | 6.37 | 768 |
| native, no route (blind) | 0.023% | 0.181% | 576 | 0.976 | 500 | 7.22 | 813 |

Airtime is summed over contention domains (lower means more capacity left for others).

**Bulk saturation.** Bulk delivery reaches the offered load for every multi-link arm, so the arms differ in airtime used, not throughput.

## Pre-registered hypotheses
| | Result | Verdict |
|---|---|---|
| H1 | Control misses −96.5% to −99.4% vs `mlo_default`, with 12–19% *less* airtime, in 6/6 scenes | **Pass** |
| H2 | Worst-decile robot better in 6/6 | **Pass** |
| H3 | At σ_map = 8 dB, still beats `mlo_default` in 6/6 | **Pass**, but it loses to `mlo_split` in 6/6 (below) |
| H4 | The blind policy keeps 99.7% of the gain | **Fail.** Route knowledge is not the source of the control edge. |
| H5 | Beats `ft11r` and 1 s legacy in 6/6 | **Pass** |

**Against the post-hoc fixed-rule baseline:**
- `native` beats `mlo_split` on control misses in all 6 scenes: 0.9–4.6 vs 5.2–7.4 per 10,000 packets.
- It also uses less airtime in all 6.

## What the ablations say
1. **Roam cost dominates everything.** A 4 s roam gives 11.3% control misses; 1 s gives 2.9%; 60 ms (802.11r with the same IP) gives 0.2%. No multi-link or AI feature comes close. Measuring and cutting the real handover cost remains the top priority, and that requires the car.
2. **Splitting control and bulk across links is the second lever, and most of it is a fixed rule.** `mlo_split` cuts default multi-link's misses by 22×.
   - Native planning cuts a further 3.4× (6.4 → 1.9 per 10,000), plus 15% airtime.
   - Planned roaming is the component that delivers it: without it, native falls back to `mlo_split` levels.
3. **Map accuracy is where a learned model pays, and also where it fails.**

   | Map error | Control misses | p99 longest gap |
   |---|---|---|
   | Perfect map | 0.010% | – |
   | σ = 4 dB (corrected E7) | 0.019% | – |
   | σ = 8 dB | 0.202% | 2.5 s |

   At σ = 8 dB the planned roams are *worse than the fixed rule*. Site choice uses absolute map values, so a bad map makes bad roams.
4. **Route as a pilot (M2) is falsified for control at cheap roam costs.** Blind is nearly as good. Route knowledge saves about 8% airtime and shortens the worst gap; it does not reduce misses.
5. **Risk-gated STR duplication hurts (M5 as designed fails).**
   - Halving spatial streams and losing 3 dB costs more than link diversity gains.
   - Removing duplication halves misses (0.010%) and the p99 gap (269 ms).
   - Cross-link retry under EMLSR is the cheaper form of diversity.
6. **Predictive MCS and roam staggering have no measurable effect here.** Robots move about 10 cm per 100 ms, and roam congestion is not binding.
7. **Data timing (M6) saves airtime** (6.61 vs 7.14) with no loss of bulk or misses.
8. **Layered PHY (M1).**
   - **Link level (Sionna, LDPC, equal symbols and power):**

     | Channel | Critical bits | Bulk bits |
     |---|---|---|
     | AWGN | need 5 dB less | need 4 dB more (α = 3) |
     | Rayleigh block fading | need 8 dB less | similar at 1% block error (2 dB grid) |

   - **System level:** it saves only 4% airtime and changes no misses, because rate-fallback retries already rescue control.
   - **Untested lead:** broadcast and multicast frames, such as a fleet-wide safety stop, get no acknowledgement or retries. A base-layer safety channel could matter there.
9. **Exploratory heavy blockage (3× rate).**
   - `native` still has fewer mean misses than `mlo_split` (0.060% vs 0.087%).
   - But its worst-decile robot and p99 gap are worse. Blockage hits every link to a site together, and the forecasts do not see it.

## What this means for building RadioJEV and WirelessJEV
- **RadioJEV's main job is an accurate, calibrated radio map, not route extrapolation.** Moving σ_map from 8 → 4 → 0 dB moved misses 0.20% → 0.019% → 0.010%. The learning target that matters is the map *where roam decisions are made*, trained on decision regret; this is the reality-gap residual from M9.
- **WirelessJEV must know when not to trust RadioJEV.** With a poor map, the fixed rule wins. The planner needs uncertainty-gated autonomy: plan when the map is confident, fall back to robust fixed rules when it is not. This is the next experiment to run.
- **Push lower in the stack only where the stack has no retries.** Unicast control is already protected by rate-fallback retries, so the layered PHY adds little there. Fleet broadcasts, deadline-bound video and the roam itself are where lower layers can still add value.
- **Kill or redesign:**
  - STR duplication: kill.
  - Route-as-pilot for control: kill at cheap roam costs; re-test only if real roams are slow.
  - Predictive MCS: no effect here.
  - Roam staggering: no effect here.
- **Keep:**
  - the control/bulk split across links;
  - planned, pre-keyed roaming;
  - data timing;
  - cross-link retry;
  - the layered safety-broadcast idea, as an SDR lead.

## Limits
- All results are synthetic.
- The roam stage costs are assumptions.
- 6 GHz is extrapolated from the 5 GHz ns-3 curves.
- Airtime contention is fluid, per domain, not per-frame CSMA.
- Fading is independent per band; blockage is shared across one robot–site path.
- One realisation per scene; six scenes.
- The PHY study uses flat block fading with perfect CSI and a 2 dB grid in Rayleigh.
- None of this has run on the SmartCar.
