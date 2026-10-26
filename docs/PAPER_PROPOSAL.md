# Paper proposal

## Working title
**Typed Intent, Exact Plans: Frozen System-One Decision Models as Objective Readers for Route-Aware Wi-Fi Association of Robot Fleets**

## One-paragraph abstract (draft; numbers are from the digital twin)
Warehouse robot fleets change BSSID roughly once every few seconds across the fleet, each switch costing about 4 s of link. Operators state priorities in words ("amr-017 is being teleoperated"), but today's roaming logic sees only signal strength.

We study how to put a frozen, low-latency typed decision model (a "System One" model: d3-lite, 0.85B parameters) in that loop. Recent work finds such models unreliable as direct numerical optimisers and proposes, without evaluating it, a hierarchy in which the model steers an optimiser. We build and evaluate that hierarchy for client-side BSSID association of 100 AMRs on fixed customer infrastructure:

- The model answers nine typed score questions, framed relative to normal operations, that map the mission text to the weight vector of a multi-term objective.
- An exact batched planner then optimises this objective over the robots' planned positions, with the measured reconnect cost charged once.
- Only the model's answer-code readout is distilled, from exact oracles; the backbone stays frozen and no language-model teacher is used.

**Results.** On held-out scenes and reworded missions:
- Shortlist regret falls from 0.114 (the same model choosing plans directly) to 0.027.
- The gap to the best known plan falls from 0.179 to 0.016, and is 0.028 on missions never seen in adaptation.
- Each re-plan takes about 134 ms instead of about 15 s.

In a 10-minute closed-loop shift the system delivers 27% more mission utility than standard roaming, with fewer reconnections and fewer safety-link violations. We release the twin, the oracles and the mission set as a benchmark.

## What is new and what is not (from the 2026-10-26 prior-art search)

| Claim | Status | Closest prior work (must cite) |
|---|---|---|
| LLM handles semantics, solver handles arithmetic | **Not new** | Language to Rewards (arXiv 2306.08647); Wadayama, LLM-steered power allocation (2604.21316); BeamAgent (2603.18855); ParetoChat (2610.23062); LanguageMPC (2310.03026) |
| Language intent → objective/weights for an O-RAN optimiser | **Not new** | Shokouhi & Wong (2602.22539); LLM-OptiRA (2505.02091); LLM-RAN operator (2509.10478) |
| Jev / System One for wireless intent → controller | **Not new** | Li et al., Jev intent at RIC timescales (2609.23136); Jev edge orchestration (2609.22753) |
| System One models weak at numerics, strong at semantics | **Not new** (we confirm and quantify) | Rahimi et al. (2610.04345); medicine (2610.18529); Jev as Q/policy (2610.11692) |
| **A frozen typed decision model emits a continuous weight vector, via relative-scale score questions, consumed by an exact optimiser; evaluated by plan regret, held-out paraphrases and never-seen missions** | **Appears new** | 2610.04345 proposes the hierarchy (Sec. III-C) without evaluating it; 2609.23136 uses discrete policy fields and a heuristic xApp |
| **Controlled within-model contrast:** same frozen model, same scenes, choosing plans directly vs reading the objective | **Appears new** | none found |
| **Readout-only distillation of a frozen decision model from exact-oracle / known-weight labels (no LLM teacher)** | **Appears new** | decision heads with LoRA (2610.17205); generic frozen probes with LLM labels |
| **Fleet-level, client-side-only association scored over known planned trajectories, with a once-charged measured reconnect cost** | **Appears new** | single-AGV LSTM AP selection (Ohori et al., Sensors 2023); digital-twin roaming needing AP cooperation (WiTwin 2404.18313; WifiRoam 2026); forecast-based cellular association (2301.09294); SDN Wi-Fi Odin/OpenSDWN; Bejerano et al. (MobiCom 2004) |
| Batched exact neighbourhood evaluation (~110 ms per 100-robot re-plan) | **New as an application**, not as an algorithm | GPU local search (Luong et al. 2009); Sionna / Sionna SYS |
| Swarm release gated against the live model; reputation quarantine | **Not new** as a technique | champion-challenger MLOps; incumbent fallback (2609.13714); reputation FL (IEEE IoT-J 9428537); Li/Shen/Lam swarm DSA |
| Public AMR association benchmark (planned trajectories + oracles + missions) | **Appears new** | none found; nearest are CRAWDAD robot RSSI sets and the WifiRoam simulator |

The search can't prove that nothing else exists. The field moved fast in October 2026, and listings after about 20 October may be under-indexed. The claims above are scoped so each one stands even if a near neighbour turns up.

## Contributions (as the paper would state them)
1. **Problem.** Route-aware, client-side Wi-Fi association for robot fleets on infrastructure the operator does not control, formalised with known planned positions, a measured reconnect cost and contention with priority classes.
2. **Method.** A frozen System One model acts as the *objective reader*: relative-scale typed score questions turn the mission into weights, cached per mission. An exact batched planner then optimises; its batch evaluator is verified against the reference evaluator to 1e-15.
3. **Adaptation.** Readout-only distillation from exact oracles, with the backbone frozen. Accuracy on reworded missions rises from 36% to 71% and on never-seen missions from 29% to 69%, from a 5×1024-parameter change.
4. **Evidence.**
   - The controlled contrast with direct plan selection.
   - Plan regret, paraphrase generalisation and never-seen-mission generalisation.
   - A closed-loop shift.
   - A latency breakdown, including the GPU-vs-CPU batching trade-off.
5. **Benchmark.** The twin, hardware profile, oracles, missions and paraphrases.

## Experiments already run (digital twin)
- 10-car: 400 decisions; distilled option-wise regret 0.080 vs MILP 0.215 vs legacy 1.30.
- 100-robot warehouse: 250 decisions; option-wise distilled 0.113; 10-car adapter transfer fails (0.527).
- Intent-to-objective: shortlist regret 0.027 on held-out paraphrases; plan gap 0.016; never-seen missions 0.028.
- Closed loop, 3 seeds × 10 min:

  | Policy | Mean utility | Reconnections | Critical violations per step |
  |---|---|---|---|
  | New (intent + planner) | 2.43 | 2,891 | 9.39 |
  | Previous WirelessJEV | 2.17 | 3,313 | 10.7 |
  | MILP | 2.01 | 3,121 | — |
  | Legacy | 1.92 | 3,107 | 10.44 |

- Ablations already in hand:
  - relative vs absolute score framing: zero-shot accuracy 13.5% → 28.6%;
  - zone decomposition gains about 0;
  - torch CPU vs MPS vs numpy batching.

## What reviewers will require before submission
1. **Baselines we have not run:**
   - a generative LLM emitting the weight vector at the same backbone size (Qwen3.5-0.8B/4B instruct, JSON output; the Wadayama/ParetoChat setup);
   - Ohori-style per-robot predictive AP selection;
   - a Bejerano-style snapshot centralised association;
   - association from *predicted* rather than *planned* trajectories (`MobilityPredictor`), which isolates the value of known plans;
   - 802.11k/v-assisted roaming;
   - an exact MIP optimum on small instances, to bound the search gap.
2. **Ablations:**
   - horizon length;
   - no reconnect cost;
   - distillation data size;
   - number of score levels;
   - per-dimension error vs downstream regret.
3. **Statistics:** confidence intervals and significance tests. The closed loop has 3 seeds, and the intent test set has 10 missions plus 4 never-seen ones; scale both up.
4. **Mission set validity.** I wrote all the paraphrases and the never-seen missions myself, so they carry author bias. We need operator-written or independently generated phrasings (checked by people), and ideally 30 or more distinct missions.
5. **Weight scale.** The 5-level weight table was set from the known missions' weights. Report this, and test a scale fixed before seeing the test missions.
6. **Second simulator.** Repeat the core result on WifiRoam or Sionna SYS so it isn't specific to our twin.
7. **Real hardware.**
   - At minimum: run the one-car test (`docs/ONE_CAR_TEST.md`) and calibrate the twin's RSSI and reconnect time.
   - For a top venue: a small multi-robot run (a few Raspbots) on a real AP set.

## Venue options
- **arXiv preprint first**, soon. Three of the closest papers appeared in the last month, so a timestamp matters.
- **Simulation plus one-car calibration:** ML-for-networking workshops (e.g. NDSS FutureG, NetAI), IEEE ICC/Globecom.
- **With multi-robot field results:** IEEE INFOCOM, ACM MobiCom/MobiSys, or ICRA/IROS for the robotics framing.
- **Benchmark-focused version:** NeurIPS Datasets & Benchmarks track.

## Honest limitations to state in the paper
- Results are from a digital twin; RF, AP and mobility parameters are partly assumed (see `hardware.py`).
- Intent reading costs about 2.5 s per mission change on a Mac GPU without the model's GPU kernels.
- The planner is local search, so optimality is not guaranteed; the gap is still to be measured against an exact MIP.
- Mission texts and the weight scale were authored by the system builders.

## Credit
The reward families, swarm protocol, reputation rule, mobility features and traffic-label framework come from Bowen Shen, Feng Li and Kwok-Yan Lam's papers (see `docs/PAPER_MAP.md`). Cite them, and consider inviting them to collaborate.
