# One radio, one mind: a vertical, symbiotic WirelessJEV + RadioJEV

Status: design and research programme. The cross-layer simulator in
`src/wirelessjev/native/` is the first executable piece. Everything here is
simulation unless a section says otherwise; no claim is made about real
hardware until the one-car test runs.

> **Naming (current system).** The current prototype default and archived 5G comparison use **RadioMap + WirelessJEV**. RadioMap is the Bayesian radio map learned from robot scans, with uncertainty and online updates. WirelessJEV is the d3-lite JEV reading the mission, numerical planning and the trust gate. **RadioJEV** names both the retained E1–E5 decision-model experiments and the experimental device-scan JEPA now implemented in `wirelessjev.contracted`. Its [20-fit, 366-rollout study](CONTRACTED_RESULTS.md) is complete. It has not earned replacement of RadioMap; neither a universal contract benefit nor a robust regret-training gain was established. All of this remains simulation, not a physical deployment.

## Why we are not seeing new capabilities yet
The learned models have so far only been asked to do these things:
- re-learn the simulator that the numerical planner already encodes;
- rank about four association plans;
- act at a single layer.

Better forecasts (−62% RMSE) produced no better decisions, because the only lever available, the choice of AP, was already saturated. New capability needs new levers at every layer, plus a model that can pull them together.

## The two halves
- **RadioJEV is the body:** a learned radio world model and reflex controller. It predicts the future of every link (signal, loss, queue, busy time), conditioned on the robot's route and the actions under consideration. It acts in milliseconds to seconds: MCS margin, duplication, link mapping, power save.
- **WirelessJEV is the mind:** intent, the fleet plan and the verifier. It turns missions into constraints and objectives, plans across robots and layers over seconds to minutes, and checks every action against exact invariants.

## How they become one model
1. **WirelessJEV plans inside RadioJEV's imagination.** It queries RadioJEV as a simulator, MuZero-style: if robot 17 waits 2 s here and maps control to 6 GHz, what is the miss probability? Search happens over the learned world model, and the result is verified exactly in the twin.
2. **WirelessJEV defines what RadioJEV must get right.** The intent fixes which errors matter, and RadioJEV trains on decision regret, not on RMSE. The mind tells the body where to pay attention.
3. **The fleet route is an input, not a forecast.** Committed routes enter RadioJEV as tokens, so the channel is predicted from where the robot *will be*. This acts like a pilot signal from the future.
4. **An envelope, then reflexes.** WirelessJEV sets a budget envelope every second: airtime, switches, energy, safety floors. RadioJEV acts freely inside it every 10–100 ms. The exact verifier enforces invariants, for example that critical traffic always has a link predicted healthy for the next 5 s.
5. **One token space.** Every layer's state becomes tokens in a single sequence: bit/symbol statistics, link, MAC queues, association, transport, application demand, pose and route, mission text. Heads run at different timescales: fast heads on the robot, slow heads at the edge.
   - **Path:** start as two models sharing a latent state; merge into one only when data from many sites justifies it.
6. **Shared training loop:**
   - **Self-play** in the cross-layer simulator, with exact search as the teacher (AlphaZero distillation).
   - **Reality-gap residuals** once real data exists: learn only where the car disagrees with the twin, because that is the only place a learned model can beat the planner.

## Layer map
| Layer | State the models see | Actions | Owner (timescale) | Where it can run |
|---|---|---|---|---|
| L0 Bits and constellation | Per-layer bit reliability | Layered (hierarchical) mapping; mission-aware bit priority | RadioJEV (per frame) | Simulation, SDR |
| L1 Coding / retransmission | Block errors, retries | Code rate per layer, retry limit | RadioJEV (ms) | Simulation, SDR, partly driver |
| L2 PHY link | SNR, MCS, spatial streams, width | Predictive MCS margin, width, streams, power | RadioJEV (10–100 ms) | Simulation; car if `iw` allows (probe) |
| L3 MAC | Queues, busy time, contention | Access categories, TWT slots, retry policy | Both (10 ms–1 s) | Simulation; partly client |
| L4 Multi-link | Per-link health | Traffic-to-link mapping, EMLSR/STR mode, risk-gated duplication | RadioJEV in WirelessJEV's envelope (100 ms–1 s) | Simulation; car needs a Wi-Fi 7 multi-link AP |
| L5 Association / roam | Neighbour map, keys | Pre-keyed fast transition, route-timed and staggered roams, AP choice | WirelessJEV (1–30 s) | Simulation; car today (BSSID) |
| L6 Network | IP, DHCP | Keep the same IP; prepare the next attachment | WirelessJEV (s) | Simulation; site configuration |
| L7 Transport | Round-trip time, losses | QUIC migration, multipath, pre-warning before a switch | Both (100 ms–s) | Simulation; car software |
| L8 Application / data | Demand, deadlines, buffers | Data timing (upload before a dead zone), prefetch, bitrate | WirelessJEV (s) | Simulation; car software |
| L9 Robot motion / tasks | Route, dwell, task queue | Wait at good spots, connected aisle choice, relay parking | WirelessJEV (s–min) | Simulation; fleet manager |
| L10 Site / environment | Radio map, layout | AP placement, pallet/reflector placement, channel plan proposals | WirelessJEV (days) | Simulation; operator approval |
| L11 Intent / governance | Mission text, contracts | Compile constraints, audit, explain | WirelessJEV | Built |

## Moonshots
Each moonshot goes against normal thinking and has a kill criterion fixed in advance.

| # | Moonshot | Against the grain because | Simulator test (kill if it fails) |
|---|---|---|---|
| M1 | **Mission-aware bits:** safety and pose bits ride the base layer of a layered constellation under bulk data | Wi-Fi treats all bits equally and serves control as separate frames | Sionna: control-bit block-error rate at equal power and bandwidth vs time sharing. Kill if no SNR gain at equal bulk throughput. |
| M2 | **The route as a pilot:** predict the channel from future position, not from past CSI | Rate control and roaming are reactive by design | Predictive vs reactive MCS and link choice. Kill if gain disappears at 2× E7 map error. |
| M3 | **Zero-contention control:** deterministic TWT / trigger-based slots computed from the fleet plan | Wi-Fi is contention-based; determinism is "for cellular" | Control-deadline misses at equal airtime |
| M4 | **Make-before-need roaming:** pre-keyed, route-timed, staggered across the fleet | The industry optimises the break, not when it happens | Interruption p99 vs 802.11r and default multi-link |
| M5 | **Risk-gated diversity:** duplicate only the seconds that matter | Duplication is either always on (5G) or off | Misses vs airtime overhead |
| M6 | **Radio weather drives data:** move bulk to good zones, prefetch before dead zones | Applications ignore the radio | Critical-deadline misses and bulk throughput at equal airtime |
| M7 | **Mobility is a radio resource:** wait, reroute, park a relay robot | The network adapts to robots, never the reverse | Delivery vs task delay. Kill if task cost exceeds radio gain. |
| M8 | **The robot tells the network its future:** route and intent in a vendor element; the infrastructure pre-steers | Clients never disclose intent | Client-only vs fleet-coordinated |
| M9 | **Learn the gap, not the world** | Learned models are usually trained to imitate the simulator | Needs real data |
| M10 | **Evolve the algorithms:** an LLM mutates planner code, scored by the exact simulator (AlphaEvolve-style) | Learn policies → discover code | Held-out sites; kill if no evolved planner beats the hand-built one |
| M11 | **Shape the environment:** reflector or pallet placement and AP moves proposed from the radio map | The environment is treated as given | Coverage and miss reduction per change |
| M12 | **The radio senses for the robot:** link anomalies flag blockage, people or forklifts for the motion planner | Radio serves robots only as a pipe | Detection rate on simulated blockage |
| M13 | **Robot-to-robot learned link** (SDR pairs): an emergent PHY/MAC between two robots | Only standard waveforms are used | SDR only; not in this branch |
| M14 | **Immunity by prediction:** world-model surprise detects jamming and spoofing | Security sits apart from radio management | Detection vs false alarms |

## What this branch executes now
The cross-layer slot simulator (`wirelessjev.native`) exposes:
- L1–L2 using ns-3 NIST packet-error curves;
- L0 using the Sionna layered-constellation study;
- L3–L5 (multi-link operation, traffic-to-link mapping, EMLSR/STR, roam stages, 802.11r);
- L6–L8 (IP retention, a transport pre-warning flag, data timing).

It runs over the 100-robot warehouse, with 6 GHz links and correlated blockage. It compares legacy, 802.11r, default multi-link and the native stack, each with ablations. Hypotheses and thresholds are fixed in `docs/NATIVE_STACK.md` before the main runs.
