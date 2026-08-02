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
| 3 | **GNN + DQN for DSA**, Ad Hoc Networks 163:103596, 2024, [10.1016/j.adhoc.2024.103596](https://doi.org/10.1016/j.adhoc.2024.103596) | Graph model of the spectrum environment feeding a decision network | **Partially adapted** — the fleet is modelled as a graph for fusion purposes, but message-passing GNN inference is *not* implemented: no model ships, and the DQN half is transmit-side. See "Not built". |
| 4 | **Improved FL for DSA**, JNCA 239:104189, 2025, [10.1016/j.jnca.2025.104189](https://doi.org/10.1016/j.jnca.2025.104189) | Not established — full text unavailable | **Nothing built.** The abstract could not be retrieved (publisher returns 403 to automated fetch). Implementing from a title would be guessing. |
| 5 | **Swarm learning DSA**, IEEE IoT-J 12(18):37720-37733, 2025, [10.1109/JIOT.2025.3583710](https://doi.org/10.1109/JIOT.2025.3583710) | Decentralised P2P learning, no central aggregator; blockchain audit | **Adapted** — `HierarchicalAggregator.select_aggregator` rotates the aggregator role instead of fixing it, removing the single point of failure. The blockchain layer is **not** implemented; Kval already has a SHA-256 hash-chained audit ledger serving the same provenance purpose without a distributed consensus dependency. |
| 6 | **Swarm DSA**, IEEE WCNC 2025, [10.1109/WCNC61545.2025.10978419](https://doi.org/10.1109/WCNC61545.2025.10978419) | Conference precursor of #5 | Covered by #5. |
| 7 | **Machine unlearning for secure DSA**, IEEE ICAIIC 2025, [10.1109/ICAIIC64266.2025.10920793](https://doi.org/10.1109/ICAIIC64266.2025.10920793) | Zero-shot removal of a contributor's influence without retraining | **Implemented, and simplified** — `cohorts.excise_contributor`. Because aggregation is over *sufficient statistics* rather than gradients, excision is **exact**: drop the statistics and re-aggregate, with no residual influence to approximate away. Their harder problem does not arise here. |
| 8 | **Privacy-aware spectrum pricing & power control for LEO IoT**, IEEE TWC 24(11):9569-9584, 2025, [10.1109/TWC.2025.3574172](https://doi.org/10.1109/TWC.2025.3574172), preprint [arXiv:2407.00814](https://arxiv.org/abs/2407.00814) | Reputation-weighted selection of the aggregating node | **Implemented** — `cohorts.ReputationLedger`. Reputation is *earned* from agreement with the cohort, discounts a node's evidence weight, and selects the round's aggregator. Spectrum pricing and power control are transmit-side; not built. |
| 9 | **Federated DSA for LEO satellite IoT**, IEEE ICCT 2024, [10.1109/ICCT62411.2024.10946453](https://doi.org/10.1109/ICCT62411.2024.10946453) | Bi-level K-Means client clustering; federated PPO | **Partially implemented** — deterministic K-Means cohort clustering is in `cohorts.py`. Federated PPO is a policy learner for transmit-side allocation; not built. |
| 10 | **LEO privacy-preserving framework**, IEEE MSN 2024, [10.1109/MSN63567.2024.00079](https://doi.org/10.1109/MSN63567.2024.00079) | Conference precursor of #8 | Covered by #8. |
| 11 | **Dynamic spectrum optimisation for IoV with mobility prediction**, Wireless Personal Communications 137, 2024, [10.1007/s11277-024-11449-w](https://doi.org/10.1007/s11277-024-11449-w) | Deep-learning mobility prediction driving spectrum optimisation | **Not built.** Full text unavailable, and the optimisation half is transmit-side. Emitter-track prediction across fleet reports is a reasonable future analogue. |

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

### Related work by the same authors, outside the three-way collaboration

| Paper | Mechanism | Status |
|---|---|---|
| Liu, Jia, Na, Lu, **Li**, **Multi-Modal Cooperative Spectrum Sensing Based on Dempster-Shafer Fusion**, IEEE Access 6:199-208, 2018, [10.1109/ACCESS.2017.2761910](https://doi.org/10.1109/ACCESS.2017.2761910) | Per-node multi-detector fusion, then credibility-weighted Dempster-Shafer fusion across nodes, down-weighting malicious users | **Implemented** — `evidence.py`. See below. **Note: Lam and Shen are not authors on this paper.** |
| Hua, Yin, Lu, Zhang, **Li**, **NLOS Identification and Positioning via Localization Residual**, Sensors 18(9):2991, 2018, [10.3390/s18092991](https://doi.org/10.3390/s18092991) | Residual-consistency test identifies and excludes non-line-of-sight anchors | **Partially implemented** — `geolocate.py` computes per-node residuals for trilateration. Residual-based NLOS *rejection* is not yet wired. |
| **Li**, **Lam**, Li, Sheng, Hua, Wang, **Advances and Emerging Challenges in Cognitive IoT**, IEEE TII 16(8):5489-5496, 2020, [10.1109/TII.2019.2953246](https://doi.org/10.1109/TII.2019.2953246) | Survey; frames sensing / access / perceiving / self-learning | Conceptual framing only. |

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
* **GNN message passing.** Warrants a trained model, which does not exist here.
* **Papers #4 and #11**, whose full texts could not be retrieved. Implementing
  from a title would be fabrication.

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
* Cohort clustering, reputation, and hierarchical aggregation are verified in
  software across simulated fleets. **No result has been measured across
  physically separate nodes.**
* Reputation is a heuristic. It resists a minority of misbehaving nodes; it is
  not a Byzantine-agreement protocol and does not claim those guarantees.
