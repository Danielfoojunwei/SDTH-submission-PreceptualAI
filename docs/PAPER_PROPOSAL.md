# Paper proposal

The warehouse-connectivity E1–E5 and corrected E7 experiments are complete
for one synthetic training seed. E6 is excluded. The consolidated results and completion status
are in `docs/WAREHOUSE_RESULTS.md`. The subsequently authorised warehouse v2
campaign is described below; general foundation-model, custom on-air PHY,
robotics and space extensions remain deferred.

## Compact fleet JEPA: completed fixes, no confirmed objective advantage

[SELECTED_JEPA_RESULTS.md](SELECTED_JEPA_RESULTS.md) completes 12 fits over
four matched objectives and three paired training seeds, 112 development
runs, 128 fresh confirmation runs and eight exact replays. Each evaluation
has 100 simulated robots. The new 13,970-inference-parameter model retains
AP/band identity, causal delivered scan context, fleet memory and route
queries. It decodes each horizon directly from RadioMap plus a bounded
correction, avoiding recursive residual summation. This does not bound
cumulative missed commands or provide physical stability guarantees.

The original privileged semantic-JEPA objective loses to matched controls on
development, so its reserved confirmation sites stay unopened. A separately
published, explicitly post-development selection chooses context JEPA before
eight entirely new sites, with fixed weights, controls and timing. Supervised
remains a primary reference rather than being dropped when it wins.

Context JEPA's fresh command loss is 1.081806% / 4.789965% at 30 ms / four-second
handover assumptions, versus RadioMap's 1.097604% / 4.849583%. Every registered
statistical comparison fails, including comparisons with matched supervised
and original JEPA, whose means are lower. Engineering margins pass against
RadioMap/supervised and fail against the original ensemble. A measured RadioMap
warm-time overrun also triggers the prespecified global qualification veto.
Neither a broad JEPA win nor a cost-specific qualified benefit is established.

An all-1,152-row, six-validation-site post-hoc audit finds finite, nonconstant
context/radio/predictor features and active context teachers across all three
paired members. This is evidence against constant representations on those
already exposed inputs, not a causal diagnosis, useful-abstraction guarantee
or independent service result. The common model still lacks a typed future
MAC resource/busy-airtime forecast; its role in the observed failure has not
been isolated. No label-free, EMA-necessity, architecture-novelty, SOTA,
physical-car or production claim follows. The application remains RadioMap.
All negative results and immutable source/evidence are preserved.

## Costed RF learning: no confirmed packet-service advantage

[CONTINUAL_PACKET_RESULTS.md](CONTINUAL_PACKET_RESULTS.md) completes 32
development and 192 confirmation evaluations, plus six exact replays, using
100 simulated robots per run. Eight fixed arms isolate reciprocal-map priors,
probe schedules and response assimilation with paid sensing, reporting and
compute. Six procedural confirmation sites, each with two paired packet seeds,
are the independent analysis units. Source and criteria were committed before
data access; development selected only the registered 100 ms extra head charge.

Neither numerical response-memory nor Bayesian correction passes the service
gate. All twelve primary and four information contrasts fail on observed
outcomes as well as conservative endpoint bounds. At assumed four-second
handovers, canonical command loss is 13.6070%, staggered memory 13.8309% and
staggered Bayesian 13.8321%. Worst-decile margins fail. Integrity, replay and
timing checks pass. This does not establish that improved RF identification
improves packet decisions after costs, and supplies no new JEPA result.

The tested adapter represents RF correction but exposes no predicted busy
airtime/resource state to the planner. A known public 100 ms activity plan is
represented through sparse integer-second, delayed forecasts. These source-level
limitations motivate a new time/resource interface; their causal contribution
is not isolated by this experiment. Dedicated monitoring, AP-assisted sensing,
reliable reporting and slot-average PHY are simulation assumptions. The study
uses procedural propagation and archived ns-3 curves, not full-fleet RT or
physical-car measurements. No model promotion, SOTA, novelty or production
claim follows. Opt-in execution journaling and process containment are separately
software-tested engineering contributions.

## Continual response learning: a measurement-design result

`CONTINUAL_RADIO_RESULTS.md` reports six registered response JEPA/supervised
fits over 24 actual Sionna RT contexts in one configured warehouse. The target
is an action-conditioned received-power response, with geometry and available
probe histories as input; true coupling is only a label/oracle. Receiver gains,
noise and feedback loss are explicitly constructed measurement assumptions.

JEPA fails the prespecified advantage gate. At staggered K2, mean clean errors
are 0.146 nW (JEPA), 0.0894 nW (supervised), 0.0637 nW (LS) and 0.0309 nW
(Bayesian). The useful result is a 77.1% response-MAE reduction for Bayesian
identification when synchronized feedback becomes staggered at equal total
transmitter-on airtime. This does not establish equal packet-service cost,
deadline improvement, online backbone learning or transfer across warehouses.

`CONTINUAL_MEMORY.md` further tests action-specific identifiability: a joint
action effect can be identified while individual coupling coefficients remain
unknown. This is established linear system identification applied to the
warehouse control question, not a first-of-kind architecture claim.

Execution reliability fixes are versioned separately. Historical metrics are
not silently recomputed with corrected transaction semantics. Neither unit
tests, a SOTA paper nor a passing auxiliary loss establishes production use;
the scope remains research until independent physical timing, risk and recovery
evidence satisfy the documented qualification contract.

## Physical representation and hierarchy: diagnosis before a claim

`VISUAL_RADIO_STATUS.md` describes the frozen-model collapse audit, six new
preregistered projection fits, real pretrained DINOv3 frontend, receiver-centric
RF factors and a proposal-only hierarchy. The old action latents are
concentrated, but remain varying and action-sensitive. A separate auxiliary
projection substantially improves normalized future matching while retaining
teacher variance/rank; two of three seeds worsen service prediction. Better
latent matching therefore does not establish better packet decisions.

The physical hypothesis is to preserve the joint association between transmitter
activity, receiver-specific coupling and spectral overlap, grounded to local
visual structure with actual geometry and event timing. A single scene summary
cannot be assumed to preserve those factors. The new fusion encoder retains
AP/band axes and measured-feature bypasses, but is untrained. DINOv3 spatial
pretraining does not make RF material, hidden activity or service risk observable.

The [primary-source review](VISUAL_RADIO_RESEARCH.md) places the design beside
established visual beam/blockage prediction and coupled wireless world models.
The contribution is still a hypothesis requiring paired data, matched
representations/planners, site-level holdouts and measured compute. No unique
architecture, collapse guarantee, SOTA or physical warehouse gain is claimed.

## Service-native candidate: implemented mechanics, unestablished advance

`SERVICE_WORLD_RESULTS.md` reports a simulation-only capability/headroom
diagnostic. A new 68,021-parameter prototype preserves AP/band structure,
measurement masks and event clocks; predicts action-conditioned Gaussian latent
transitions; and initializes service corrections to zero around an explicit
numerical reference. It is untrained and inactive. Unit identities and
hindsight action labels do not establish learned prediction/control quality.

`SERVICE_WORLD_FIRST_PRINCIPLES.md` derives why packet age, known unavailability,
receiver freshness and canonical recovery matter for deadline-sensitive
warehouse control. Exact guaranteed misses during fully unavailable service
windows are lower bounds, not a full service model. The decoder anchors each
output separately; latent rollout error can still accumulate. No theorem or
measurement currently establishes bounded long-horizon error.

The 18-source audit in `SERVICE_WORLD_PRIOR_ART.md` finds that RAN-JEPA already
combines action-conditioned wireless JEPA, mechanistic residuals and matched
counterfactual evaluation. Related work covers coupled robot/wireless models,
delivered memory, graph stochastic dynamics and baseline fallback. Broad
architectural novelty is unsupported. Any contribution must be precisely
defined and established against that literature rather than asserted from a
combination of established components.

Official component execution in `SERVICE_WORLD_BASELINES.md` verifies limited
upstream code access, not a common warehouse leaderboard. The present study
does not establish SOTA, superiority over RadioMap, a new foundation model or
physical 100-car qualification. Matched trained baselines, independently
trained seeds, sealed sites, calibration and end-to-end control/compute
measurements remain outstanding.

## Packet-action JEPA: negative development evidence

`PACKET_VALUE_RESULTS.md` trains three actual JEPA and three matched supervised
packet-value heads on 930 same-state native counterfactual action labels. The
models share RadioMap assistance and identical inference graphs. A commitment
ledger, immediate-link checks and empirical uncertainty corrections do not
establish safe control: the replacement interface suppresses necessary roaming
when uncertainty causes rejection.

Across 64 development evaluations, JEPA loses 24.598% versus canonical
RadioMap's 1.045% at 30 ms handovers, and 12.780% versus 7.515% at four seconds.
The common-controller numerical arm also degrades. Raw learned action values
trail the numerical estimator on validation. These findings reject this
controller configuration and provide no JEPA-necessity claim. Independent
confirmation was not run; all eight reserved sites remain untouched.

The useful system finding is that rejecting an untrusted proposal must retain
baseline recovery. It must not impose a persistent stay. The new shadow-only
interface preserves baseline decisions and blocks activation of the rejected
profile. It does not establish an active learned-controller improvement. A
baseline-preserving intervention changes policy continuation, requiring new
counterfactual labels and development validation before any confirmation.

## Delivered observations and outage-aware control

`CLOSED_LOOP_RESULTS.md` confirms a narrower control result after coupling
fleet-model observations to actual simulated scanning and packet deliveries.
Local neighbor knowledge comes from completed noisy scans, current-link pilots
freeze during radio outages, and only strictly past report deliveries reach
the queued forecaster. Four development sites fix one shared utility horizon
before 96 tests on eight new 100-robot warehouse seeds. Models remain frozen.

At assumed four-second handovers, a switch-loss objective including the full
outage reduces compact-model command loss 5.849% → 5.635%, with paired-site
interval [-3044.79, -1162.50] ppm. With the same utility controller RadioMap is
5.603%; the model interval spans zero. Better forecast RMSE (6.70 vs 8.54 dB)
still does not establish superior control. At 30 ms handovers the selected
utility harms every site for both models, raising compact loss 1.035% → 1.318%.
Its main transferable finding is the dependence of useful predictive control
on physical interruption cost and observation delivery. It does not establish
the necessity of JEPA or a universally better learned wireless controller.

The study adds 120 native evaluations and four exact replays. All 192 software
tests pass. Shared warm-start history, out-of-band pose/routes, reserved
control-frame telemetry, fixed CAD/motion and assumed hardware timing limit
transfer. High residual loss fails deployment qualification. Cost-dependent
routing, immediate-link guards and packet-supervised risk are future tests,
not demonstrated improvements.

## Operational-cost and handover-admission frontier

The broader follow-up (`WORLD_FRONTIER_RESULTS.md`) tests distilled,
route/reliability-loss and temporal models across three seeds each, with
matched runtime optimisation of RadioMap. It charges a measured asynchronous
compute queue and assumed scan outages in native packet evaluation. The
single student uses 53 ms versus 233 ms for the three-member hybrid on this
CPU, but its independent confirmation (228.12 vs 251.04 ppm) has interval
[-85.94, 39.06]. Do not turn the compute gain into a control-superiority claim.

The separate hardware-budget and admission studies expose a stronger system
finding. Under assumed four-second handovers, tuning sensing cadence and
planned admission reduces distilled loss from 14.849% to 7.419% on four fresh
sites. On six other fresh sites, applying the same outage rule to both planned
and reactive proposals reduces 8.851% to 5.920%. This supports separating
confidence/fallback routing from admission of a physical outage. The effect
also benefits RadioMap, so it is not evidence of JEPA necessity.

The 432 additional native runs and 42 fresh sites preserve model failures and
throughput trade-offs. Remaining loss is unacceptable for physical deployment.
Scanner phases, telemetry delivery and motion remain idealised; hardware timing
is assumed. The earlier offline-confirmation cohort remains evidence for its
own conditions, not a guarantee under the stricter frontier. Stronger claims
need live observation coupling, real handovers, independent CAD and native
counterfactual reward learning.

## Fleet-memory world models: confirmed warehouse comparison

`FLEET_WORLD_MODEL_RESULTS.md` and `FLEET_CONFIRMATION_RESULTS.md` record nine
three-seed fits and 760 native runs. The richer models pool causal fleet scans,
known CAD and routes, with mean-field and AP-score supervision. A matched
rolling RadioMap and the existing learned-prior reference get independent
development tuning. The supervised ablation keeps the architecture and removes
the JEPA objective.

Keep the two studies distinct. The first study's development-selected standalone
model improves forecast RMSE but its primary command-miss interval spans zero.
The best hybrid on that discovery cohort was frozen before eight additional
warehouses. Its preregistered independent primary test reduces misses from
167.19 to 133.33 per million, a 20.2% reduction, with paired-site 95% interval
[-68.75, -5.73] ppm. Forecast RMSE falls from 7.09 to 4.57 dB. This supports a
narrow simulated-warehouse comparison against the chosen RadioMap reference.

The secondary hybrid JEPA reaches 128.12 ppm, while previous JEPA reaches
126.04. Neither a generic JEPA-necessity claim nor universal learned-controller
superiority follows. The hybrid retains RadioMap fitting and uses about 0.4 s
per 100-robot forecast on this CPU; the standalone avoids per-BSS map fitting
and takes about 0.19 s. These timings are measured separately from the offline
packet simulations and are not charged to their deadlines.

The layout family, simulated supervision, causal-but-free scan streams,
precomputed motion and 30 ms handover assumption limit transfer claims.
Independent CAD layouts, measured device channels, real scan/inference timing
and field qualification are still needed for a broader foundation-model paper.

## Contracted device-JEPA evidence

The subsequent `JEPA_GATE_REPAIR_RESULTS.md` isolates and repairs the gate's
uncertainty mismatch without changing its five JEPA members. Development-only
selection reduces misses from 329.17 to 111.11 per million on six fresh scenes;
the gain-bound alternative reaches 133.33. A reactive-rescue ablation reaches
113.19, so these results do not establish the necessity of a learned uncertainty
filter. They also do not rehabilitate generic coupling novelty or regret-loss
superiority. All original study evidence below remains unchanged.

The separate study in `CONTRACTED_RESULTS.md` completed 20 five-seed model fits
and 366 native rollouts across 24 fresh simulated scenes. It implements the
censored, route-conditioned world model and finite AP-regret objective. Both
gated and ungated JEPA beat the fixed rule in all six primary scenes; the gate
raises JEPA's mean misses from 114.58 to 151.39 per million. Regret training
changes the total by one command in 1.44 million, with the crossed training-seed
and scene interval spanning zero. RadioMap remains default at 92.36 per million.

Do not lead either proposed paper with contract necessity, ranking-only
guarantees, regret-training superiority or generic JEPA/JEV coupling novelty.
`CONTRACTED_THEORY.md` and `CONTRACTED_PRIOR_ART.md` explain the counterexample
and existing uncertainty/fallback/world-model literature. These new compact
fits do not add independent seeds to the original large-model E1–E5 evidence.

## Compositional-mission and radio audit

The separately versioned v3 protocol and results are in `WAREHOUSE_V3.md` and
`WAREHOUSE_V3_RESULTS.md`. It adds complete composite contracts, group-aware
full-priority scoring, actual masked/JEPA-style training, scene-disjoint packet
telemetry prediction and a matched-profile native Wi-Fi audit. It does not
turn previous exploratory results into a foundation-model claim.

Keep four distinctions explicit: a compiler's canonical-language correctness
versus neural interpretation coverage; offline forecast error versus control
quality; critical-rate proxies versus measured simulator endpoint deadlines;
and an assumed application mute versus actual Wi-Fi authentication. The
whole-family reader failures, superseded PHY pilot, negative pretraining
comparisons and rejected shared releases remain evidence. Approximate
unlearning is compared with actual retained-data retraining, not presented as
certified erasure. Physical-car qualification remains a prerequisite for fleet
deployment claims, and the implementation remains Wi-Fi-specific.

## Full-window native packet learning

The completed v4 study (`WAREHOUSE_NATIVE_RESULTS.md`) uses 58 actual ns-3
runs and six independently fitted compact models over a matched thirty-second
window. Native supervision cuts forecast RMSE from about 0.413 to 0.158.
Mean learned utility is 0.7010 versus numerical 0.6980, with a difference
interval [-0.0078, 0.0139] crossing zero. The linear specialist reaches 0.7015.
No learned candidate passes frozen validation. Numerical remains default;
do not lead the paper with a learned-control superiority claim.

All aggregate critical-floor labels are misses, so zero neural Brier/ECE does
not establish event discrimination or meaningful risk calibration. Preserve
this negative finding, the original reports and the separate post-run audit.
This known-map, fixed-association simulation does not qualify physical robots,
actual handovers or persistent roaming.

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
- An exact batched planner then optimises this objective over the robots' planned positions, with the assumed reconnect cost charged once; physical timing calibration remains pending.
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
| **Fleet-level, client-side-only association scored over known planned trajectories, with a once-charged assumed reconnect cost** | **Appears new** | single-AGV LSTM AP selection (Ohori et al., Sensors 2023); digital-twin roaming needing AP cooperation (WiTwin 2404.18313; WifiRoam 2026); forecast-based cellular association (2301.09294); SDN Wi-Fi Odin/OpenSDWN; Bejerano et al. (MobiCom 2004) |
| Batched exact neighbourhood evaluation (~110 ms per 100-robot re-plan) | **New as an application**, not as an algorithm | GPU local search (Luong et al. 2009); Sionna / Sionna SYS |
| Swarm release gated against the live model; reputation quarantine | **Not new** as a technique | champion-challenger MLOps; incumbent fallback (2609.13714); reputation FL (IEEE IoT-J 9428537); Li/Shen/Lam swarm DSA |
| Public AMR association benchmark (planned trajectories + oracles + missions) | **Appears new** | none found; nearest are CRAWDAD robot RSSI sets and the WifiRoam simulator |

The search can't prove that nothing else exists. The field moved fast in October 2026, and listings after about 20 October may be under-indexed. The claims above are scoped so each one stands even if a near neighbour turns up.

## Contributions (as the paper would state them)
1. **Problem.** Route-aware, client-side Wi-Fi association for robot fleets on infrastructure the operator does not control, formalised with known planned positions, an assumed reconnect cost and contention with priority classes. Actual switch-stage calibration remains required.
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
