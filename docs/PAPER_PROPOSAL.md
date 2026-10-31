# Paper proposal

The warehouse-connectivity E1–E5 and corrected E7 experiments are complete
for one synthetic training seed. E6 is excluded. The consolidated results and completion status
are in `docs/WAREHOUSE_RESULTS.md`. The subsequently authorised warehouse v2
campaign is described below; general foundation-model, custom on-air PHY,
robotics and space extensions remain deferred.

## Thirty-second horizon evidence

The separate protocol in `WAREHOUSE_HORIZON.md` uses six route-conditioned
five-second forecasts, an observation-only numerical belief, hard switch
budgets and three independently trained seeds for both compact architectures.
Every model seed is evaluated on every one of three held-out environments.
Forecast error improves, but learned closed-loop mission-utility intervals
include zero improvement and worst-decile service declines. These results
support the numerical planning baseline, not a learned-controller headline.

A selector designed after that holdout rejects all six residuals using six
validation environments. On three fresh audit environments the numerical
fallback raises mean service from 0.237 to 0.326 against budgeted strongest
signal, with fewer critical-rate-proxy violations. Eighteen identical
fallback evaluations do not constitute eighteen independent learned fleets.
The twelve-switch cap is an operator-policy scenario; unconstrained roaming
has higher service in the original horizon holdout. This experiment does not
establish superiority under unrestricted switching or physical reliability.
See `WAREHOUSE_HORIZON_RESULTS.md` and `WAREHOUSE_HORIZON_SELECTOR.md` for
the negative results, full comparisons, provenance and limits.

## Subsequent warehouse v2 evidence

The user later authorised a separate warehouse research campaign, documented
in `WAREHOUSE_V2.md` and `WAREHOUSE_V2_RESULTS.md`. It changes how the historical
E2 comparison should be interpreted: a structured controlled-language reader
and the existing numerical candidate evaluator solve this narrow historical
mission set at zero candidate regret; the actual d3 family reader also reaches
zero regret on answered cases, with incomplete new-mission coverage. The E2
RadioJEV result therefore establishes superiority to the original fixed-weight
interface, not to structured numerical planning.

The v2 sensed-input and compact multitask models have three training seeds;
the original native-radio backbone adaptations do not. The first v2 learned
controllers fail closed-loop qualification, and their separately recorded
handover-factorization correction is compared against the strongest-scan
baseline on new rollout seeds. Neither greater model complexity nor better
fleet mean service alone establishes an advantage. All v2 studies remain
simulations/model inference, not physical-car or general foundation-model
evidence. Keep these protocols distinct from the paper's original experiments.

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
The cloud **RadioJEV E1** evaluation now includes the paired intent baseline.
Native-input LoRA reaches held-out regret 0.1068 ± 0.0240 (200 cases) and
new-mission regret 0.0721 ± 0.0136 (240 cases), versus intent + planning's
0.0140 ± 0.0066 and 0.0251 ± 0.0074. CPU latency is 4.31 s for LoRA versus
468 ms for intent. Frozen native inputs reach 0.1069 ± 0.0228 / 0.0837 ± 0.0194.
The observed E1 result favours intent + planning. The original Mac LoRA result
(0.0928 ± 0.0222 / 0.0744 ± 0.0196) remains separately preserved; the earlier
shortlist benchmark is also a separate experiment.

In **E2**, radio-Q reaches normalized regret 0.1170 ± 0.0370 on held-out
missions and 0.0779 ± 0.0253 on new missions (120 cases each). Intent reaches
0.1987 ± 0.0461 / 0.3482 ± 0.0674; E1 zero-shot reaches 0.2905 ± 0.0594 /
0.3288 ± 0.0693. Radio-Q takes 6.42 s per held-out decision versus intent's
591 ms. E1 and E2 use different regret definitions. These are one-seed runs;
95% intervals describe evaluation cases, not variation across training seeds.
Raw results and provenance are in `artifacts/radiojev_cloud/`; the consolidated
E1–E5 completion record is in `docs/WAREHOUSE_RESULTS.md`. Keep native-input representation learning
an exploratory result rather than a principal paper claim.

The **mission-capability router** design in `docs/ROUTER_EXPERIMENT.md` is
deferred; it contributes
no measured result to this paper. E2 compares RadioJEV against the current fixed-weight intent reader;
it does not show superiority to a planner with explicit constraints or a
structured mission parser. Its "nonlinear" mission set also includes a
linear safety penalty and a spatial mean expressible through a focus set.
The move budget is implemented as a soft penalty, and the video threshold
uses a horizon-averaged minimum. State these actual semantics before making
claims about hard constraints, lexicographic priorities or mission classes.

E3 provides mixed evidence: association normalized regret improves from
0.1170 to 0.1046, and DSA greedy reward is 1.000 over a single 500-step episode
(MLP baselines: 0.263 / 0.384). Traffic accuracy after training is only
0.369 / 0.231 / 0.338, below the rule baselines of 0.708 / 0.738 / 0.923.
Admission accuracy is 100%, but all 106 training and 54 test cases have the
"lower" label. A constant prediction also reaches 100%; the result is not
evidence of learning a general admission policy. The raw report is retained
alongside the reproducible label audit; a new dataset must be separately
versioned rather than replacing this evidence.
Independent training seeds and additional DSA layouts remain necessary.

**E4/E5 are complete but do not support a swarm or unlearning advantage.**
E4 mean normalized regret is 0.1820 for local-only training, 0.1781 for
FedAvg and 0.1820 for personalised swarm, versus 0.1729 for the initial E3
model. FedAvg's worst-site regret is 0.2220, versus local/swarm's 0.2023.
All three shared merge proposals failed validation; the swarm matched local
training and its shared component incorporated no E4 site updates.
Consequently E5 does not test removal of an accepted site contribution from
that shared model. Retained-data distillation raised membership-attack AUC
from 0.6167 to 0.6489 and retained normalized regret from 0.1801 to 0.2068.
Retraining without the site yielded AUC 0.6231 (identical to the initial-model
baseline) and retained regret 0.1962. These results do not establish forgetting;
they must remain negative/limited evidence in the paper.

**E7 environment learning** also needs careful qualification. The historical
15-environment report reduced mean plan gap from 1.239 (legacy) to 0.174
(15-minute learned map), but audible-path RMSE increased from 4.98 to 7.90 dB.
This is not evidence that more data improves signal prediction. The original
scan censoring forced omitted APs below −90 dBm even when the eight-result cap
caused their omission; the fitter then treated inequalities as point samples.
The corrected pipeline uses measured top-eight scans, Gaussian censored-data
likelihoods and inverse-sampling weights. Its separately rerun evidence is in
`artifacts/envlearn_corrected/report.json`, with the original report retained.
Audible-path, inaudible-path and all-path errors must be reported separately;
in particular, improvements on audible links do not validate weakly measured
interference paths. Plan gap is relative to a truth-model local search, not a
certified optimum. All E7 results remain synthetic, with no field calibration.

The corrected **15-environment × 5-mission sweep** completed with audible-path
RMSE 4.04 / 3.95 / 3.99 dB at the 1 / 5 / 15-minute survey budgets. Mean plan
gap is 0.202 / 0.182 / 0.159; the 5-minute pessimistic planner reaches 0.144.
This removes the severe historical accuracy regression but does not establish
monotonic learning: 5 of 15 environments still worsen from 1 to 15 minutes.
The 15-minute model's inaudible-path RMSE is 15.03 dB (all paths: 12.23 dB).
These paths are defined by current truth RSSI below −90 dBm, not by absence
from every survey scan. Do not generalise audible-path accuracy to distant
interference links or claim that additional driving always improves the map.

An additional **map-only study on 15 new seeds (15–29)** compares the corrected
v2 calibration with a censored global loss-law fit that uses nominal rack
geometry. Inaudible-path RMSE falls from 18.14 / 16.10 / 15.71 dB to 14.86 /
13.78 / 14.23 dB for 1 / 5 / 15-minute budgets, with nearly unchanged audible
error at 15 minutes (3.96 → 3.94 dB). All 15 environments improve on the
inaudible subset. The 15-minute paired mean improvement is 1.48 dB, with an
exploratory, unadjusted 95% paired t interval of [0.98, 1.99] dB. This is an
extrapolation improvement under known layout assumptions, not a downstream
planning result or a fix for unobserved walls/interference. It remains an
optional study implementation; the default planner is unchanged.

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
