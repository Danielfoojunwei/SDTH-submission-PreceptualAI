# Service-world model: prior art and defensible claim boundary

**As of 3 November 2026, neither “nobody has done this” nor “state of the art” is
established.** Live primary sources overlap directly with action-conditioned
wireless JEPA, mechanistic residuals, robot/wireless coupled models, matched
counterfactual rollouts, stochastic graph dynamics, delivery-aware packet
decisions and uncertainty-triggered conventional fallback. Those ingredients
are useful engineering choices; their combination is not by itself a new
scientific contribution.

This review freshly retrieved 16 arXiv metadata pages and PDFs, one publisher
full text and one RFC. The [audit and claim matrix](../artifacts/service_world/literature/audit.json)
record source URLs, exact review scopes, fetch hashes, search failures and
remaining coverage gaps. “Reviewed” means the identified sections were read,
not that every proof was checked or each baseline was reproduced. CQL received
an abstract/introduction review only. No training or external testbed control
was performed for this audit.

## Closest primary sources

| Verified source | Sections actually inspected | Relevant mechanism and claim consequence |
|---|---|---|
| [RAN-JEPA](https://arxiv.org/abs/2610.17584), 2026 | §3.1–3.3, PDF pp. 4–5; §4.1–4.2, pp. 5–6; §4.4–4.5, pp. 7–8 | Observation-only JEPA pretraining, deployment adaptation and frozen-encoder action-conditioned learning. The latent transition is residual; the numeric head combines a mechanistic action-aware reference with a learned correction. A calibrated twin evaluates SLEEP/KEEP from the **same pre-action history**, and predictive models share the downstream controller. Thus “physics residual + action JEPA + matched counterfactuals” is already explicit. Real-equipment four-cell 5G energy/congestion results operate over minutes, rather than warehouse Wi-Fi packet deadlines. |
| [Telecom World Models](https://arxiv.org/abs/2604.06882), 2026 | §V-A/B, pp. 9–10; §VII-A–F, pp. 12–13 | The framework names queues, latency, packet loss, handover success/interruption, interference graphs, policy counterfactuals, multi-timescale planning, uncertainty, shadow deployment and rollback. It also distinguishes observation-policy bias from interventions. These are architectural prior art even though the paper does not implement every proposed capability in our task. |
| [Coupled Control and Wireless World Models](https://arxiv.org/abs/2609.04851), 2026 | §III-A–F, pp. 3–7; §IV-E, pp. 10–11 | Coupled robot-control and wireless JEPAs use visual and RF histories for predictive communication and resilient control. RF inputs include CSI, spectrograms and persistence images; synchronized Gazebo/ROS and material-aware Sionna RT supply experience. This already combines robot embodiment, wireless prediction, physics and missing-observation resilience. Its closed-loop evidence is simulation, not our hardware evidence. |
| [Guess and Send / WorldGT](https://arxiv.org/abs/2610.29587), 2026 | §III-A–C, pp. 3–4; §IV-A, pp. 4–5; §IV-F, p. 7 | An ACK-synchronized receiver twin tracks what was **actually delivered**. Matched all-refresh/prediction-only counterfactuals share receiver memory; packet choices use QPSK and Sionna-calibrated 5G-NR LDPC success/erasure curves. Complete frame outage has an explicit fallback. Delivery-conditioned belief, packet counterfactuals and recovery behavior therefore have direct prior art. The task is semantic video perception, not AP roaming. |
| [Self-Improving Interference Management with UQ](https://arxiv.org/abs/2401.13206), 2024 preprint | §III-A–D, pp. 2–4 | Ensemble mean/variance estimates feed a credibility test based on wireless system performance. An untrusted learned answer invokes conventional optimization; the improved answer feeds later learning. A wireless uncertainty gate with conventional fallback is already implemented. This review accessed the open preprint, not a later journal typeset version. |
| [Safe Policy Improvement with Baseline Bootstrapping](https://arxiv.org/abs/1712.06924), ICML 2019 | Abstract/introduction; §2.3, pp. 3–4 | Insufficiently supported state-actions retain baseline behavior under an explicit policy constraint. General baseline-preserving safe improvement is established. Its guarantees depend on its assumptions; a finite empirical gate in a partially observed warehouse does not inherit them automatically. |
| [Beyond Static Graph World Models / GDM](https://arxiv.org/abs/2609.28670), 2026 | Abstract/related work; §3, pp. 4–5 | Sparse recurrent adjacency, message passing and an RSSM-style stochastic latent model predict changing topology and node features. Prior imagination is corrected by observations; reward, continuation and action-mask heads support decisions. A graph plus stochastic recurrent world model is already a published architecture. This is a generic graph study, not a common Wi-Fi benchmark result. |

## Native wireless representations and established decision interfaces

| Verified source | Reviewed scope and implication |
|---|---|
| [LatentWave](https://arxiv.org/abs/2606.06373), 2026 | §III-A–D and task/linear-probe comparisons: antenna-shared patches and time-frequency structure support heterogeneous wireless JEPA pretraining, including over-the-air RF and Wi-Fi/5G CSI. “Wireless-native JEPA” is not new; supervised comparators still win several tasks. |
| [Field-layer JEPA](https://arxiv.org/abs/2608.10222), 2026 | §III and §IV-B/C: physical path/delay/angle/visibility structure and format-specific tokenizers feed a shared predictive propagation field. Channel/field structure is already explicit; it does not establish a packet-service controller. |
| [Homomorphic Wireless World Models](https://arxiv.org/abs/2603.20048), 2026 | §III–IV: magnitude/phase CSI and velocity-conditioned matrix-exponential latent dynamics, evaluated on real industrial DICHASUS measurements. Structured physical wireless dynamics are prior art; predominantly static-reflector assumptions limit applicability to abrupt blockers. |
| [JEPA-kMMVAE](https://arxiv.org/abs/2610.13315), 2026 | Abstract, §2.1–2.2 and §3.1: stochastic finite-memory latent CSI, noisy historical inputs, frozen pretrained encoder, predictive head and optional residual correction. JEPA plus a stochastic latent plus a residual is already studied. |
| [JEPA for AI-Native 6G](https://arxiv.org/abs/2607.09798), 2026 | §III-A/B and §VII-B–E: native time-frequency/beam/KPI/graph tokens, missing/asynchronous masks, action conditioning, probabilistic uncertainty and fallback. Many elements are roadmap proposals rather than complete empirical implementations, but they preclude claiming the broad framing as new. |
| [System One Models for Wireless Decision-Making](https://arxiv.org/abs/2610.04345), 2026 | §II, §IV-A/B and §V-B–D: query-conditioned probabilities over finite admissible decisions, proper scoring, wireless applications, temporal hysteresis and safety/fallback. A typed decision instead of generated text is explicit prior art. Normalization alone does not establish calibration. |
| [Against the Monolithic Wireless World Model](https://arxiv.org/abs/2605.16689), 2026 | Abstract/introduction and Appendix C: configuration-dependent wireless data, classical processing, digital twins and composable intelligence. This supports careful scope, not a claim that a single model must generalize to any radio or robot. |
| [AutoLab](https://arxiv.org/abs/2609.14854), 2026 | Abstract/introduction, §2–3 and targeted execution/measurement passages: validated robot/AP/scan/traffic experiments, scheduling and operational records. Its contribution is physical experimentation infrastructure, explicitly not a completed autonomous world model. We read the paper without accessing its testbed. |
| [Conservative Q-Learning](https://arxiv.org/abs/2006.04779), 2020 | Abstract/introduction only: conservative values address offline policy shift. Detailed proofs and variants were not audited here; this is background, not a reproduced baseline. |
| [Individual packet deadline scheduling](https://doi.org/10.1186/1687-1499-2014-65), 2014 | Publisher full text, §2/2.3, §4 and §9/endnote c: packet waiting/deadline state and channel-aware threshold scheduling are established. Its many-user/capacity-region/sufficient-power assumptions do not remove real outage floors; the authors explicitly note shadowing/power can make deadline guarantees infeasible. |
| [RFC 9034](https://www.rfc-editor.org/rfc/rfc9034.html), 2021 | §1, §4–5: per-packet origination/deadline metadata, industrial delay requirements and expiry behavior already have a protocol specification. It is primarily synchronized 6LoWPAN, not proof that a given Wi-Fi/5G device exposes the same fields. |

## What first principles justify

The relevant physical object for a control packet is delivered service **before
its deadline**, rather than a reconstructed RSSI image. Queue age, packet size,
association state, contention, serialization, retry limits and handover outage
jointly determine that event. A strong SNR predictor can still choose an action
that provides no service while the radio reconnects. Under a link-exclusive
outage, queued packets with less remaining slack than the unavailable service
interval cannot be rescued by changing the predictor alone. Alternative links,
redundancy or a different action change that premise and must be modeled
explicitly.

That motivates a conditional service-time/deadline-failure distribution, with
an explicit conditioning history and feasible action set. It does **not** make
survival distributions, queue models or deadline-aware scheduling new. It also
does not make JEPA the uniquely correct architecture: a calibrated numerical
queue model, Markov model or recurrent stochastic model may solve the same
decision with less compute.

Radio measurements have physical identity, frequency, timestamp, age and a
measurement mechanism. “Not observed” can mean below sensitivity, displaced by
a capped scan, not requested, or not delivered. Those cases must not silently
become invented low-SNR observations. Delivered-state synchronization and
explicit missingness are causal requirements, with substantial prior art above.
Conditioning on simulator future states, unreceived scans or the eventual
outage outcome would invalidate an online prediction claim.

A model must separate predictable propagation/traffic structure from
irreducible uncertainty under its observations. More parameters, diffusion or
a self-supervised objective cannot guarantee elimination of that uncertainty.
Likewise, predicting absolute anchored state rather than repeatedly adding an
unbounded residual can remove one accumulation mechanism, but does not prove
that long-horizon errors or closed-loop losses are bounded.

## A narrow hypothesis worth testing

A defensible candidate is:

> Under a specified causal warehouse observation interface, can a learned
> deadline-service estimator improve bounded interventions while preserving
> the original controller's observable recovery behavior and continuation
> state, at a matched measurement and inference budget?

This is a **research hypothesis with novelty unestablished**. RAN-JEPA covers
mechanistic residuals and matched interventions; WorldGT covers delivered
memory and packet counterfactuals; SPIBB covers baseline-preserving constraints;
deadline scheduling is older still. A contribution would need a precisely new
estimator, identified intervention/continuation invariant, theorem under stated
assumptions, or convincing new experimental evidence. Renaming the components
or claiming their combination is “native” is insufficient.

The contract must state which actions can change, for how long, which baseline
state continues to evolve, and what observations both branches may consume.
A rejected intervention must not suppress the baseline's recovery reflex or
resume from stale controller state. Passing a replay/invariant check establishes
that implementation property for the tested conditions; it is not a universal
non-regression or physical safety theorem. Paired simulator branches estimate
counterfactuals within the simulator, not automatically real-world causal effects.

## Evidence required before claiming an advance

1. Compare against the canonical RadioMap/controller and a numerical
   queue/service model, plus a matched non-JEPA recurrent/stochastic model.
   Distinguish a pooled fleet-map reference from a local-history comparator;
   equalize information for the architectural ablation. Closest-paper
   adaptations require faithful mechanisms and code/data rights, not their
   names on a different model.
2. Hold actions, planning horizon, observation delivery, recovery logic and
   tuning budget fixed when isolating the estimator. Ablate service targets,
   JEPA loss, uncertainty and the intervention contract separately. An oracle
   may diagnose headroom but is not a deployable competitor.
3. Seal held-out warehouses, traffic/blockage regimes and evaluation seeds
   before selecting the model or thresholds. Use independently trained seeds;
   evaluate paired outcomes at the episode/site level to avoid treating
   correlated packets as independent replications. Keep every result,
   including regressions and abstentions.
4. Measure deadline misses, delivery, outage bursts, tail delay, recovery and
   handovers alongside proper scoring/calibration of the service distribution.
   Include end-to-end decision latency, memory and compute. Forecast MSE alone
   does not establish a control benefit; selected-case gains do not establish
   non-regression.
5. Limit any resulting claim to the validated task and data regime. A 100-car
   simulation is evidence about that simulation. Physical warehouse transfers,
   PHY behavior and multi-protocol operation each need their own measurements.
   Results from unrelated channel, video and RAN papers do not form a shared
   leaderboard on which this model can be declared SOTA.

## Search coverage and remaining uncertainty

The [search receipt](../artifacts/service_world/literature/index_search.json)
contains six queries each to OpenAlex and Crossref, at most six ranked results
per query, filtered to publication dates no later than 3 November 2026. Ten
requests returned HTTP 200. Crossref queries for **“wireless graph stochastic
latent world model”** and **“wireless world model recovery fallback”** returned
HTTP 429; those searches are incomplete. Index results were used as leads,
not as evidence that methods were inspected.

A primary IEEE service-curve lead returned HTTP 202 with zero bytes and remains
[unreviewed](../artifacts/service_world/literature/manet_service2009.json).
An SSRN counterfactual-queue lead resolved to a primary title about financial
limit-order execution and was [excluded from wireless claims](../artifacts/service_world/literature/ssrn7299999.json).
This correction illustrates why search titles alone were not trusted.

Comprehensive IEEE/ACM full-text coverage, forward/backward citation closure,
patent searching and code-license/reproduction audits remain incomplete.
Fresh arXiv retrieval verifies availability and the accessed version, not peer
review or independent replication. PDFs and extracted full texts stay in the
execution workspace; the repository contains factual metadata, original
paraphrases and fetch receipts. No sealed earlier evidence was changed.

The present review is sufficient to reject broad architectural novelty. It is
not sufficient to certify uniqueness of the narrow hypothesis, and it does not
establish that any model beats RadioMap or the strongest relevant competitors.
