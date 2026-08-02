# DSA research mapping: what was taken, adapted, and deliberately left

This records how the dynamic-spectrum-access research programme of **Feng Li**,
**Bowen Shen** and **Prof. Kwok-Yan Lam** informs this system, mechanism by
mechanism, with citations.

## Read this first: what "reproduce" can and cannot mean here

**Their results are not reproduced, and nothing here should be read as
reproducing them.** Reproduction means re-running an experiment and matching
reported numbers. That would require their datasets, simulation environments,
and baselines, none of which are available. What is implemented is the
*mechanisms* their papers describe.

**A second, larger gap matters more.** That corpus solves **transmit-side**
problems: which channel a secondary user should transmit on, at what price, at
what power. Kval is **receive-only** and has no transmit path — a constraint
enforced by CI, not merely documented. So the *problem formulations* do not
transfer at all. What transfers is the **coordination architecture**: how a
fleet of edge nodes federates learning without pooling raw data, handles nodes
in dissimilar environments, and stays useful when some nodes are unreliable or
hostile.

Where a paper's core contribution is transmit-side, that is stated below and
nothing is built.

## Coverage against Bowen Shen's own publication list

Checked against his Google Scholar profile (`user=9a_OyvEAAAAJ`, 18 works, 136
citations, h-index 4). Of those 18:

* **12 are mapped below** — the dynamic-spectrum-access line, including paper
  #12, which this cross-check surfaced and the earlier searches had missed.
* **1 is his master's thesis** (*Privacy-preserving resource management in LEO
  satellite Internet-of-Things*, 2025), which aggregates the LEO papers already
  mapped.
* **1 is federated wireless traffic prediction** (ICAIIC 2025) — same theme as
  #12; the per-flow mechanism implemented for #12 covers it, and no separate
  claim is made.
* **4 sit outside the DSA line.** One of them, *MemMorph: Tool Hijacking in
  LLM Agents via Memory Poisoning* (arXiv 2026), is now implemented against —
  see below. The other three remain unbuilt, with reasons rather than silence:
  * *A low-cost mobile manipulator* (ICRAS 2021) — robotics. Nothing transfers
    beyond the general framing of capable behaviour on cheap embedded hardware,
    which is the sub-node premise this repository already had. Building
    anything from it would be invention, not reproduction.
  * *Encrypted-database search* (ICISS 2023) — searchable encryption over data
    at rest. Genuinely adjacent to the captured-drone problem, and the custody
    half of that problem is now answered by Shamir sharing in
    `threshold_ca.py`, so a captured node's share yields nothing. Searchable
    encryption itself is **not** built: every practical scheme leaks access and
    search patterns, and adding that leakage to a fleet whose value depends on
    siting discretion needs a threat analysis this repository has not done.
  * *A certified LEO satellite management system* (ICISS 2024) — the
    certification-under-partition idea it shares with the MANET papers is what
    `threshold_ca.py` implements; the satellite management system itself is out
    of scope.

**MemMorph resolves an earlier ambiguity, and is now built against.** A prior
search could not determine whether its "Bowen Shen" was the same person; his own
profile confirms it is. Its subject — poisoning an LLM agent's memory to hijack
its tool use — lands directly on this system, which runs an agent that selects
bounded actions from recorded history. An earlier revision of this document
recorded it as a threat but not an implementation, because the arXiv page
returned HTTP 503 and building from a title would have been fabrication. The
abstract was subsequently retrieved and confirms the mechanism: records
disguised as "technical facts, incident reports, and operational policies", up
to **85.9% attack success from three injected records**, surviving three
representative defences, with the paper concluding that memory-level integrity
safeguards are needed.

`memory_integrity.py` implements those safeguards, and `audit.py` now binds each
ledger record to its author. The mapping is direct: Kval's long-term memory *is*
the audit ledger, and `analysis.py` derives baselines from it, so appending a
handful of plausible records shifts what the node believes normal is. The hash
chain detects *modification* of history but not *appending* to it — a gap the
`audit-memory` command now closes. Kval's pre-existing mitigations still apply:
the closed `LeaseAction` vocabulary, single-use leases, and the rule that no
learned component may widen its own authority.

One design note worth recording, because the first attempt was wrong. An
injection detector keyed on "records arriving close together" or "records
resembling each other" fires constantly on a correct node: a healthy link
produces the same decision every cycle, and one loop iteration writes several
records within the same second. That version was caught by an end-to-end test
against a real ledger before it shipped. The detector now requires a group to
resemble **each other** *and* be distinct from **the rest of the history** —
a template inserted into a history, rather than a system doing the same thing
twice. Its honest limit is stated in the module: it catches an outsider
injecting from a template, not a compromised node holding the signing key and
writing poisoned records that resemble its own normal output. Bounded influence
and fleet reputation are the backstop for that case.

## Author disambiguation

`Feng Li` and `Bowen Shen` are common names. The individuals here are
identified from the author footnote of the open-access DR-NTU copy of the 2022
IEEE TVT paper: *"F. Li, B. Shen and G. Wei are with School of Information and
Electronic Engineering, Zhejiang Gongshang University… F. Li is also at School
of Computer Science and Engineering, Nanyang Technological University… K. Y.
Lam [is] with School of Computer Science and Engineering, Nanyang Technological
University."* Feng Li is DBLP `Feng Li 0008`. DBLP's `Bowen Shen` profile
conflates at least six distinct researchers; only the NTU/ZJGSU spectrum cluster
is relevant here.

## The mapping

| # | Paper | Mechanism | Status in Kval |
|---|---|---|---|
| 1 | **Federated Deep RL for DSA**, IEEE TVT 71(7):7952-7956, 2022, [10.1109/TVT.2022.3166535](https://doi.org/10.1109/TVT.2022.3166535) | FL over edge nodes; cohorts of non-interfering users with similar conditions | **Implemented** — `cohorts.kmeans_cohorts` groups nodes by RF-environment similarity before aggregation. The DRL action-selection half is transmit-side; not built. |
| 2 | **Hierarchical federated DRL for DSA**, Ad Hoc Networks 149:103257, 2023, [10.1016/j.adhoc.2023.103257](https://doi.org/10.1016/j.adhoc.2023.103257) | Hierarchical, personalised aggregation for heterogeneous devices | **Implemented** — `cohorts.HierarchicalAggregator` aggregates per cohort, then across cohorts, so a large cohort cannot outvote a small one node-by-node. |
| 3 | **GNN + DQN for DSA**, Ad Hoc Networks 163:103596, 2024, [10.1016/j.adhoc.2024.103596](https://doi.org/10.1016/j.adhoc.2024.103596) | Graph model of the spectrum environment feeding a decision network | **Implemented, without the learned half** — `topology.py`. The fleet is a graph and evidence propagates over it by deterministic message passing, so a node whose radio is down inherits its neighbours' view. Belief weakens per hop by Shafer discounting and is attributed by shortest path to each original observer, so a cycle cannot amplify one observation. The GNN's *trained* weights and the DQN are not built: no model ships and the action half is transmit-side. |
| 4 | **Improved FL for DSA**, JNCA 239:104189, 2025, [10.1016/j.jnca.2025.104189](https://doi.org/10.1016/j.jnca.2025.104189) | Not established — full text unavailable | **Nothing built.** The abstract could not be retrieved (publisher returns 403 to automated fetch). Implementing from a title would be guessing. |
| 5 | **Swarm learning DSA**, IEEE IoT-J 12(18):37720-37733, 2025, [10.1109/JIOT.2025.3583710](https://doi.org/10.1109/JIOT.2025.3583710) | Decentralised P2P learning, no central aggregator; blockchain audit | **Adapted** — `HierarchicalAggregator.select_aggregator` rotates the aggregator role instead of fixing it, removing the single point of failure. The blockchain layer is **not** implemented; Kval already has a SHA-256 hash-chained audit ledger serving the same provenance purpose without a distributed consensus dependency. |
| 6 | **Swarm DSA**, IEEE WCNC 2025, [10.1109/WCNC61545.2025.10978419](https://doi.org/10.1109/WCNC61545.2025.10978419) | Conference precursor of #5 | Covered by #5. |
| 7 | **Machine unlearning for secure DSA**, IEEE ICAIIC 2025, [10.1109/ICAIIC64266.2025.10920793](https://doi.org/10.1109/ICAIIC64266.2025.10920793) | Zero-shot removal of a contributor's influence without retraining | **Implemented, and simplified** — `cohorts.excise_contributor`. Because aggregation is over *sufficient statistics* rather than gradients, excision is **exact**: drop the statistics and re-aggregate, with no residual influence to approximate away. Their harder problem does not arise here. |
| 8 | **Privacy-aware spectrum pricing & power control for LEO IoT**, IEEE TWC 24(11):9569-9584, 2025, [10.1109/TWC.2025.3574172](https://doi.org/10.1109/TWC.2025.3574172), preprint [arXiv:2407.00814](https://arxiv.org/abs/2407.00814) | Reputation-weighted selection of the aggregating node | **Implemented** — `cohorts.ReputationLedger`. Reputation is *earned* from agreement with the cohort, discounts a node's evidence weight, and selects the round's aggregator. Spectrum pricing and power control are transmit-side; not built. |
| 9 | **Federated DSA for LEO satellite IoT**, IEEE ICCT 2024, [10.1109/ICCT62411.2024.10946453](https://doi.org/10.1109/ICCT62411.2024.10946453) | Bi-level K-Means client clustering; federated PPO | **Partially implemented** — deterministic K-Means cohort clustering is in `cohorts.py`. Federated PPO is a policy learner for transmit-side allocation; not built. |
| 10 | **LEO privacy-preserving framework**, IEEE MSN 2024, [10.1109/MSN63567.2024.00079](https://doi.org/10.1109/MSN63567.2024.00079) | Conference precursor of #8 | Covered by #8. |
| 11 | **Dynamic spectrum optimisation for IoV with mobility prediction**, Wireless Personal Communications 137, 2024, [10.1007/s11277-024-11449-w](https://doi.org/10.1007/s11277-024-11449-w) | Extends the **Exploration and Preferential Return** model to D-EPR, uses it to synthesise vehicle traces, trains an LSTM on those traces to predict node locations, then optimises spectrum reuse graph-theoretically | **Mobility model implemented, LSTM deliberately not** — `mobility.py`. Abstract retrieved on the second attempt (see below). EPR itself is implemented and applied to emitter tracks; the LSTM is skipped because in their pipeline EPR *is* the ground truth the network approximates, so shipping the approximation instead of the model would add a training dependency for a worse predictor. The optimisation half is transmit-side. |
| 12 | **Network Traffic Prediction Using PSO-LightGBM-TM**, IEEE INFOCOM 2024 Workshops, [10.1109/INFOCOMWKSHPS61880.2024.10620828](https://doi.org/10.1109/INFOCOMWKSHPS61880.2024.10620828); journal version *Network traffic prediction based on PSO-LightGBM-TM*, Computer Networks, [10.1016/j.comnet.2024.110838](https://doi.org/10.1016/j.comnet.2024.110838) | Particle swarm optimisation tunes LightGBM parameters **per network flow**, so one model does not have to fit every flow. Reported to beat deep-learning baselines while being structurally simpler; evaluated on Abilene, GÉANT and CERNET | **Mechanism implemented, base learner deliberately not** — `forecast.py`. See below. |

### Two independent literature searches agree on a negative finding

Both searches, run separately, reached the same conclusion and it is the most
important result here:

> **Prof. Lam has no published work on spectrum sensing in the signal-detection
> sense** — no energy or cyclostationary detection, no matched filtering, no
> cooperative-sensing fusion rules, no P_d/P_fa analysis, no RF fingerprinting,
> no wideband or sub-Nyquist sensing. Spectrum sensing appears in the corpus
> only as a *cited prerequisite*, never as the contribution.

> **Nor is there adversarial-RF work** — no jamming, anti-jamming,
> spectrum-sensing data falsification, primary-user emulation, or GNSS
> spoofing. "Resilient" in these titles means *dropout-resilient or
> Byzantine-robust in the machine-learning sense*, not RF-resilient. Conflating
> the two would misrepresent the record.

This is why the mapping above transfers *coordination architecture* and nothing
else. The detection front-end, the RF threat model, and hardware-rooted
attestation are outside this corpus and are sourced elsewhere in this
repository — the EW threat register, the receive-only adapter, and the signed
capability leases are Kval's own, not theirs.

### Two further mechanisms adopted, with Lam as an author

| Paper | Mechanism | Status |
|---|---|---|
| Liu, Guo, **Lam**, Zhao, **Efficient Dropout-Resilient Aggregation for Privacy-Preserving Machine Learning**, IEEE TIFS 18:1839-1854, 2023, [10.1109/TIFS.2022.3163592](https://doi.org/10.1109/TIFS.2022.3163592) | Secure aggregation that stays correct when participants drop out mid-protocol | **Implemented as a property, not a protocol.** Aggregation here tolerates nodes appearing and disappearing between rounds and features missing from individual updates, and is tested for it. Kval does not implement their cryptographic secure-aggregation protocol — statistics are already non-sensitive by construction, so the masking machinery has nothing to protect. |
| **Lam** (first author), Mitra, Gondesen, Yi, **ANT-Centric IoT Security Reference Architecture**, IEEE IoT-J 9(8):5895-5908, 2022, [10.1109/JIOT.2021.3073734](https://doi.org/10.1109/JIOT.2021.3073734) | Security managed around critical activities within micro-perimeters rather than inherited enterprise models | **Framing adopted.** It is the best published articulation — and it is his own — of why receive-only should be a *structural* property rather than a policy promise. Kval enforces exactly that: no transmit path exists, and CI fails the build if one appears. |

## Beyond the DSA line

The dynamic-spectrum-access papers are a small slice of Prof. Lam's output
(~437 DBLP records). His secure-distributed-ML and trust-architecture work is
where he leads as first or senior author, and it is more directly useful to a
sensing fleet than the spectrum-access line.

| Paper | Mechanism | Status |
|---|---|---|
| Zhao, Zhao, Yang, Wang, Wang, Lyu, Niyato, **Lam**, **Local Differential Privacy based Federated Learning for IoT**, IEEE IoT-J 2021, [arXiv:2004.08856](https://arxiv.org/abs/2004.08856) | Perturbs each participant's shared values *locally, before release*, via the Piecewise Mechanism and variants, so the aggregator never sees an individual's true contribution | **Implemented** — `privacy.py`. See below. |
| Liu, Guo, **Lam**, Zhao, **Efficient Dropout-Resilient Aggregation**, IEEE TIFS 18:1839-1854, 2023, [10.1109/TIFS.2022.3163592](https://doi.org/10.1109/TIFS.2022.3163592) | Secure aggregation correct under mid-protocol dropout | **Property implemented, protocol not.** Aggregation tolerates nodes appearing/disappearing and features missing from individual updates, and is tested for it. Their cryptographic masking is not implemented — with LDP applied, statistics are no longer sensitive in transit, so the masking machinery has less to protect. |
| **Lam** (first author), Mitra, Gondesen, Yi, **ANT-Centric IoT Security Reference Architecture**, IEEE IoT-J 9(8), 2022, [10.1109/JIOT.2021.3073734](https://doi.org/10.1109/JIOT.2021.3073734) | Security managed around critical activities within micro-perimeters | **Framing adopted.** Best published articulation — his own — of why receive-only should be structural rather than promised. Kval enforces exactly that: no transmit path exists and CI fails the build if one appears. |
| **Toward Efficient and Certified Recovery From Poisoning Attacks in FL**, IEEE TIFS 2025, [arXiv:2401.08216](https://arxiv.org/abs/2401.08216); **Privacy-Preserving Federated Unlearning With Certified Client Removal**, TIFS 2025; **A Survey on Federated Unlearning**, ACM CSUR 2025, [arXiv:2310.20448](https://arxiv.org/abs/2310.20448) | Certified removal of a client's influence, and certified *recovery* after poisoning, with a **bound** on residual influence | **Implemented, with a stronger guarantee** — `cohorts.certified_excision`. Their bound exists because gradient updates smear a client's contribution across every later round. Aggregating sufficient statistics makes removal **exact**, and the certificate proves it rather than asserting it: the recovered baseline is compared byte-for-byte against one built from a fleet the node never joined. Residual influence is zero, not merely bounded. Certified recovery of a *trained model* remains unimplemented; no model ships. |
| **Effective Intrusion Detection in Heterogeneous IoT Networks via Ensemble Knowledge Distillation-Based FL**, IEEE ICC 2024, [arXiv:2401.11968](https://arxiv.org/abs/2401.11968) | Nodes are heterogeneous, so forcing one architecture on all of them fits none. Each keeps **its own** detector and the fleet shares *outputs* rather than *parameters* | **Architecture implemented, distillation not** — `intrusion.py`. Each node judges its readings against *its own* baseline, and the fleet combines verdicts. The distinction only a fleet can draw: one anomalous node is a local condition; the same anomaly across *different cohorts*, simultaneously, in the same direction, is not. Distillation machinery needs trained students and teachers; none ship. |
| **Self-initialized Distributed Certificate Authority for MANET** (ISA 2009); **A robust certification service for highly dynamic MANET in emergency tasks**, Int. J. Communication Systems 2009, [10.1002/dac.1015](https://doi.org/10.1002/dac.1015) | Issuing and validating credentials among intermittently-connected wireless nodes with **no reachable central authority** | **Implemented, and narrowed on purpose** — `threshold_ca.py`. Was flagged here as the most under-appreciated find and a real gap: a sub-node that lost its operator link ran until its policy expired and then failed closed, so an adversary who cut the backhaul could disarm the fleet by waiting. Now `k` of `n` roster peers can attest to *extending an existing grant's lifetime* — never widening scope, never resurrecting a revocation, bounded per renewal and in total so an isolated partition winds down rather than running forever. Their threshold *signing* is deliberately not reproduced; see "Not built". Shamir `(k,n)` sharing over GF(2^8) is implemented for capture-resistant secret custody, verified against the FIPS-197 field vectors. |
| Yang, Zhao, **Lam**, Xiong, Wu, Xiao, **Distributed DRL-Based Spectrum and Power Allocation for HetNets**, IEEE TWC 21(9), 2022, [10.1109/TWC.2022.3153175](https://doi.org/10.1109/TWC.2022.3153175) | Per-cell decisions from **locally observed information only**, with light coordination | **Framing adopted.** Evidence that local-observation autonomy plus light coordination converges — the degraded/partitioned-backhaul mode Kval already implements deterministically. The allocation half is transmit-side. |

### Local differential privacy: closing a threat this repository documented

The machine-unlearning paper's premise is that a spectrum model leaks the
location data it was trained on. Applied here: **a fleet baseline built from
each node's exact feature means encodes where the sensors are and what each one
sees.** Sharing only sufficient statistics narrows that exposure; it does not
close it, because a node's exact mean is still its exact mean.

`privacy.py` implements the Piecewise Mechanism so each node perturbs its
contribution *before release*. The perturbation is unbiased, so the fleet mean
still converges to the truth while no individual value is revealed —
demonstrated in tests over 4,000 releases recovering a 200 ms truth within
15 ms.

Three things are stated plainly because they are real costs:

* **Randomness here is genuine, unlike elsewhere.** `forecast.py` seeds its
  particle swarm deterministically so a forecast is reconstructable from the
  audit ledger. Privacy inverts that requirement: a deterministic
  "perturbation" is none, and anyone knowing the seed recovers the true value.
  This module uses `secrets.SystemRandom`. Auditability is kept at the right
  level — the ledger records *that* perturbation happened and with what budget,
  never the unperturbed value.
* **The budget composes.** Releasing k features at full budget spends
  k·epsilon, so the budget is divided across features released together. Repeat
  rounds over the same data compose too; `epsilon` is per-release.
* **LDP and consensus filtering are in tension.** Trimming or consensus
  clustering removes injected noise along with outliers, biasing an unbiased
  estimator. An operator must choose privacy or poisoning-resistance for a
  given release; applying both to the same values weakens each. This is
  documented in the module rather than silently combined.

### Related work by the same authors, outside the three-way collaboration

| Paper | Mechanism | Status |
|---|---|---|
| Liu, Jia, Na, Lu, **Li**, **Multi-Modal Cooperative Spectrum Sensing Based on Dempster-Shafer Fusion**, IEEE Access 6:199-208, 2018, [10.1109/ACCESS.2017.2761910](https://doi.org/10.1109/ACCESS.2017.2761910) | Per-node multi-detector fusion, then credibility-weighted Dempster-Shafer fusion across nodes, down-weighting malicious users | **Implemented** — `evidence.py`. See below. **Note: Lam and Shen are not authors on this paper.** |
| Hua, Yin, Lu, Zhang, **Li**, **NLOS Identification and Positioning via Localization Residual**, Sensors 18(9):2991, 2018, [10.3390/s18092991](https://doi.org/10.3390/s18092991) | Residual-consistency test identifies and excludes non-line-of-sight anchors | **Implemented** — `geolocate.py`. Was previously computing residuals without using them. The naive form does not work: judging each anchor by its own residual under the full solve fails because the solve is already dragged by the blocked anchor, so every anchor looks mildly wrong and none looks decisively wrong. The test now asks which single anchor, if removed, reconciles the rest, and requires a threefold improvement before blaming one. Sign is retained as evidence — obstruction can only lengthen an apparent range, so a *shorter*-than-implied reading is reported separately as a repeater, reflection or fabricated report. Two structural limits fell out and are enforced: three anchors always fit exactly, so a fleet of `n` survives at most `n-4` blocked nodes; and when no single removal reconciles the rest the estimate is refused rather than returned kilometres wrong. |
| **Li**, **Lam**, Li, Sheng, Hua, Wang, **Advances and Emerging Challenges in Cognitive IoT**, IEEE TII 16(8):5489-5496, 2020, [10.1109/TII.2019.2953246](https://doi.org/10.1109/TII.2019.2953246) | Survey; frames sensing / access / perceiving / self-learning | Conceptual framing only. |

## Per-flow forecasting: the mechanism without the base learner

Paper #12's contribution is not LightGBM. It is the argument that **a single
model fitted across all flows is wrong** — flows differ — and that each flow
deserves its own parameters, found by particle swarm optimisation. That
transfers exactly: a Wi-Fi link, a cellular link and an NTN link have different
latency regimes, and one shared smoothing constant fits none of them.

`forecast.py` implements **their per-flow PSO tuning strategy over a different
base learner**: a damped Holt linear-trend smoother, which is standard-library
arithmetic. LightGBM is a PyPI package, and this repository has a deliberate
no-PyPI-runtime-dependency policy (`docs/DEPENDENCIES.md`) — a node installs
only from signed Ubuntu/NVIDIA repositories. Breaking that for a forecaster
would be the wrong trade.

Consequences to be clear about:

* **None of their reported accuracy is inherited.** Different base learner,
  different data. The forecaster's own fit error is reported with every result
  so its quality is visible rather than assumed.
* The forecaster is **descriptive, not a learned policy**: fitted on a
  profile's own recent history at call time, holding no persistent weights,
  requiring no signed model manifest. It can warn; it does not select profiles.
  Selection stays with the deterministic optimizer and the Safety Shield.
* PSO is **seeded deterministically**, not randomly. Every other decision this
  node makes is reconstructable from its audit ledger; a forecast should be no
  different.

## The single most valuable import: Dempster-Shafer fusion

The fabric's first aggregator counted votes. Counting cannot distinguish
**"this node measured the band and found it clear"** from **"this node could
not measure the band at all"** — both are merely "not a vote for
interference". A fleet of blind nodes therefore looked like a fleet reporting
quiet.

`evidence.py` fixes that. Each node's assessment becomes a mass assignment over
{clear, busy}, and `INSUFFICIENT_EVIDENCE` puts **all** mass on Θ — ignorance,
not disbelief. Demonstrated:

| Fleet | Fused belief |
|---|---|
| 3 nodes that cannot measure the band | `unknown: 1.0` — no claim either way |
| 3 nodes measuring clear | `clear: 1.0` |
| jamming vs clear vs blind | `unknown: 1.0`, `conflict: 1.0`, flagged not trustworthy |
| low-credibility node claiming jamming vs trusted node claiming clear | `clear: 1.0` — the liar is discounted |

Two deliberate deviations from textbook Dempster-Shafer:

* **Conflict is retained, not normalised away.** Dempster's rule divides out
  the conflict mass K, which is exactly how it produces confident nonsense from
  irreconcilable sources. K is surfaced, and a fused belief above the conflict
  threshold is marked untrustworthy. High conflict between nodes on the same
  band is itself intelligence: localised jamming affects some nodes and not
  others, and so does a lying node.
* **Total conflict returns ignorance**, not a division by ~zero.

## Not built, and why

* **Any transmit-side capability** — spectrum pricing, power control, channel
  allocation, access scheduling. This is the bulk of the corpus. Kval has no
  transmit path and CI enforces that. Not a gap; a boundary.
* **Deep RL / DQN / PPO policies.** No model ships, and the learned path
  abstains until a signed manifest verifies. The corpus's DRL is for choosing
  transmit actions.
* **Blockchain.** The audit ledger already provides tamper-evident provenance
  without a consensus protocol or its availability failure modes.
* **Learned GNN weights and the DQN.** The graph and its message passing are
  now built (`topology.py`); the *trained* parts are not, and would need a
  model that does not exist here.
* **The LSTM in paper #11.** The EPR model it is trained to approximate is
  implemented directly instead; see the row above.
* **Paper #4**, whose full text could not be retrieved — the publisher returns
  403 to automated fetch. Implementing from a title would be fabrication.
* **Threshold signing.** Lam's MANET distributed-CA papers share the CA private
  key across nodes so a quorum can mint new certificates. `threshold_ca.py`
  deliberately does not reproduce that: distributing signing authority means a
  quorum of *captured* nodes can issue credentials the operator never approved.
  Quorum renewal gets the availability benefit without ever letting the fleet
  grant itself something new.

## A threat their work implies

Paper #7's premise is that a spectrum model trained on user behaviour and
location data leaks that location. Applied here: **a federated model trained
across the fleet implicitly encodes where the sensors are.** For a defensive
receive-only fleet whose value depends partly on siting discretion, that is a
counter-intelligence exposure, not merely a privacy one.

This is why federation here shares only sufficient statistics (count, mean, sum
of squares), never raw observations or per-node positions — and why
`excise_contributor` exists, so a compromised or relocated node can be removed
exactly.

## Honest limits

* The mechanisms are implemented and unit-tested; **none has been validated
  against the source papers' experimental results**, and no performance claim
  from those papers is inherited.
* **NLOS rejection is verified against synthetic measurements generated from
  the same path-loss model the locator inverts.** That is a fair test of the
  residual logic — it recovers the true position to under a metre with one
  anchor obstructed by 18 dB, and refuses when two are — but it is *not* a
  measurement of real accuracy, which depends on multipath, antenna patterns
  and calibration quality in the actual deployment.
* **EPR is a population model applied to a single track.** Song et al. measured
  it across many agents; using it to predict one emitter is a reasonable
  extension, not a validated one. The predictor reports its exploration
  probability and declines to name a place when exploration dominates, which
  bounds the harm but does not make the extension rigorous.
* **The fleet anomaly detector's false-alarm estimate assumes independent,
  normally distributed features.** Real RF measurements are neither; they are
  autocorrelated and cross-correlated, so the true rate is higher. The function
  is exposed for setting thresholds deliberately and no detection result quotes
  it.
* Cohort clustering, reputation, and hierarchical aggregation are verified in
  software across simulated fleets. **No result has been measured across
  physically separate nodes.**
* Reputation is a heuristic. It resists a minority of misbehaving nodes; it is
  not a Byzantine-agreement protocol and does not claim those guarantees.
