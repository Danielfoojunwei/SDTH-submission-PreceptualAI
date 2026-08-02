# Learning in autonomous networks — problems first

Kval's first learner picks a signed connectivity profile from live evidence.
That is one problem out of many. Before adding more learning, this document
states the problems that the standards and research literature actually raise —
quoted — and only then describes the bounded learning Kval adds for each. Where
a quote could not be confirmed verbatim, it is marked as a paraphrase.

The governing principle throughout: **the deterministic optimizer and the
independent Safety Shield stay in control; every learned component is advisory
until it earns trust, and abstains the moment its inputs leave the distribution
it was trained on.**

## The problems, quoted

**1. Intent is not inferable — operators must supply goals and constraints.**
> "Networks, even when they are autonomic, are not clairvoyant and have no way
> of automatically knowing particular operational goals nor which instances of
> networking services to support."
> — IETF RFC 9315, *Intent-Based Networking — Concepts and Definitions*, §3.1
> (IRTF NMRG, Informational, October 2022).

Consequence for Kval: goals live in signed policy (service-class objectives,
power limits, spectrum allowlists), never inferred by a model. A learner may
rank options *within* an operator's stated intent; it may not invent the intent.

**2. Competing control loops conflict — coordination is required.**
> "[CL-general-16] The ZSM framework should support capabilities to avoid
> conflicts between Closed Loops."
> — ETSI GS ZSM 009-1 V1.1.1 (2021-06), *Closed-Loop Automation; Part 1:
> Enablers*, clause 6 (with the related [CL-general-17]: "…to minimize the
> negative effects of conflicts between Closed Loops"). Note these are "should"
> recommendations, not "shall" requirements.

Consequence for Kval: the connectivity loop, power governor, spectrum sensing,
and recovery state machine run at different timescales and are ordered by a
deterministic priority (safety → thermal/power → connectivity → optimization),
so two loops cannot drive the radio in opposite directions in the same tick.

**3. Autonomous decisions must be explainable and auditable after the fact.**
> "Ability to explain and interpret the results produced by AI/ML Models in a
> closed loop at a later point in time — in anomaly management this may be
> required to revisit or audit the decisions or analytics results produced by an
> AI/ML model and validate associated control flow of the closed loop."
> — ETSI GS ZSM 012 V1.1.1 (2022-12), *Enablers for AI-based Network and Service
> Automation*, Annex A, clause A.4.2, item 6. The spec also carries the general
> normative requirement (clause 5, Req-1) that the framework "shall support the
> capability to explain and interpret the outputs and decisions" of AI/ML
> applications.

Consequence for Kval: every decision is a transparent, weighted score today, and
every cycle is written to the SHA-256 hash-chained audit ledger with its inputs,
scores, reason codes, and outcome. A future learned policy must emit the same
explainable `Recommendation` contract and the same reason codes.

**4. Fragmented, multi-layer data is hard to fuse.**
> "The challenges of ingesting large amounts of disparate data across multiple
> layers, and the corresponding requirements on data modeling, formatting and
> representation are discussed."
> — O-RAN nGRG Research Report RR-2023-03 v1.1, *Research Report on Native and
> Cross-domain AI*, Executive summary. (This is an nGRG research report reflecting
> its contributor-authors' views, not a normative O-RAN specification.)

Consequence for Kval: the multimodal state (`multimodal.py`) binds every
modality observation to a source identity, timestamp, quality, missingness map,
and provenance hash, and fails closed when a mandatory modality is stale or
absent — so "multimodal" does not simply mean "more inputs for an attacker to
spoof".

**5. The AI/ML pipeline is itself an attack surface.**
> "Without protections on the AI and ML, the xApps and rApps can be compromised,
> leading to data leakage and degraded, or even denied, service."
> — O-RAN ALLIANCE Security Update 2025 (WG11), 6 March 2025. The O-RAN security
> specification suite (Security Requirements and Controls Specifications v11.0,
> Feb 2025, and the Zero-Trust study) covers, in paraphrase, confidentiality,
> integrity, availability, authenticity/authentication, replay prevention, and
> continuous monitoring via the NIST zero-trust tenets.

Consequence for Kval: models, policies, catalogues, and leases are all signed;
inputs carry provenance; leases are single-use and expiring; and the audit chain
is tamper-evident. See `docs/SECURE_TRANSPORT_AND_EW_THREAT_MODEL.md`.

**6. The specific attacks are evasion, poisoning, and privacy.**
> "Sections 2.2, 2.3, and 2.4 discuss the major classes of attacks: evasion,
> poisoning, and privacy, respectively."
> — NIST AI 100-2e2025, *Adversarial Machine Learning: A Taxonomy and
> Terminology of Attacks and Mitigations*, March 2025 (for Predictive AI
> systems). NIST separately names the attacker *objectives* as availability
> breakdown, integrity violation, and privacy compromise — do not conflate the
> attack classes with the objective categories.

Consequence for Kval: poisoning is countered by signed model manifests, signed
feature baselines, drift/OOD abstention, and quarantine of suspect updates;
privacy by never ingesting packet payloads; evasion by keeping the Safety Shield
independent of any learned component.

**7. Exploration must be made safe — a shield around the learner.**
The paraphrase: a reactive *shield* synthesised from a safety specification
either restricts the agent to a set of safe actions before each decision or
monitors and corrects a chosen action that would violate the specification, so
safety holds during both learning and execution. The paper's own words:
> "Reinforcement learning algorithms discover policies that maximize reward, but
> do not necessarily guarantee safety during learning or execution phases."
> — Alshiekh, Bloem, Ehlers, Könighofer, Niekum, Topcu, *Safe Reinforcement
> Learning via Shielding*, AAAI-18 (arXiv:1708.08611). (The phrasing "prevent
> disastrous decisions while exploring" is a common characterisation, not a
> verbatim quote from this paper.)

Consequence for Kval: the Safety Shield **is** the shield — it is deterministic,
independent of the learner, and can only ever return an action from the approved
set or reject/abstain. A learned policy proposes; the shield disposes.

**8. Autonomy is broader than link selection.**
3GPP Release 18 SA5 defines an AI/ML *management* framework — TS 28.105 (AI/ML
management) and the study TR 28.908 — while the concrete radio use cases of
AI/ML-based network energy saving, mobility optimization, and load balancing are
RAN3-led (studied in TR 37.817 and specified across TS 38.300 / 38.401 / 38.413 /
38.423 / 38.473). A survey of this work summarises the intent as to "define
enhancements for data collection and signaling support to facilitate AI/ML-based
network energy saving, load balancing, and mobility optimization" (Nikbakht et
al., *Artificial Intelligence in 3GPP 5G-Advanced: A Survey*, IEEE ComSoc).

Consequence for Kval: the learning roadmap covers energy/thermal forecasting,
congestion/mobility prediction, and root-cause classification — not only "which
profile". Kval selects a pre-provisioned S-NSSAI/DNN; it does not create a slice
or allocate spectrum (see `docs/STANDARDS_INTEROPERABILITY.md`).

## The learning Kval adds for each problem

| Problem | Learning capability | Bounded controls it may inform |
|---|---|---|
| Link degradation (§8) | Temporal failure forecasting (roadmap; fail-closed contract today) | earlier failover, probe frequency, backoff — via `recovery.py` |
| Repeated outages | Root-cause classification (roadmap) | interface reset, modem reconnect, profile quarantine, escalate |
| Power / heat (§8) | Energy & thermal governance (implemented, deterministic) | `nvpmodel` selection, inference duty, scan/telemetry interval — `power.py` |
| Congestion / mobility (§8) | Traffic & link-lifetime forecasting (roadmap) | route preference, pre-authentication, service budgets |
| Spectrum contention (§4) | Passive occupancy analysis (implemented) | band-availability ranking, jamming *suspicion* — `spectrum.py` |
| Unknown conditions (§6,§7) | Drift / OOD / change-point detection (implemented) | force ABSTAIN, return to baseline — `learning.py` |
| Action effectiveness (§3) | Causal-attribution gate (implemented) | learn only from improvements that clear an effect+t gate — `learning.py` |
| Multi-loop conflict (§2) | Deterministic priority ordering (implemented) | prevents simultaneous destabilising actions |
| Explainability (§3) | Transparent scores + hash-chained audit (implemented) | every decision reconstructable |

"Roadmap" items ship as **fail-closed contracts**: a predictive model is loaded
only when its signed manifest (engine, feature schema, calibration report,
lineage) verifies (`learning.verify_model_manifest`); until then the learned path
abstains and the deterministic optimizer decides.

## Timescales

| Horizon | Loop | Nature |
|---|---|---|
| ms–s | radio, thermal, safety limits | deterministic |
| seconds | link-failure detection, reconnect, failover, QoS | deterministic + recovery FSM |
| minutes | energy, congestion, mobility optimization | forecasting (advisory) |
| hours | drift analysis, evaluation, canary promotion | offline + gated |
| days | operator intent, policy changes, capacity planning | human-authored, signed |

## What the node agent may and may not control

**May control (only under signed policy + Safety Shield +, for actuation, a
capability lease):** connection selection/failover; reconnect backoff; Wi-Fi and
modem power-save; Jetson power mode from an allowlist; inference duty; telemetry
and spectrum-scan intervals; queue/rate/service budgets; approved DNN/S-NSSAI
selection; candidate-model shadow/canary/rollback; local federated participation;
quarantine recommendations.

**Never controls:** arbitrary frequency, waveform, gain, EIRP, or transmit power;
regulatory or Safety-Shield limits; trust anchors or its own authorization; audit
deletion; unbounded shell; its own promotion criteria; permanent credential
revocation without an independent authority. These are enforced by the closed
`LeaseAction` set in `lease.py` — there is no lease that authorises them.

## Capability leases

Every actuation beyond passive observation requires an operator-signed
`CapabilityLease`: exactly one approved action, its numeric bounds, the node it
is bound to, the governing policy hash, a single-use nonce, a validity window, a
rollback action, and a required post-condition. The node holds only the operator
public key; it cannot mint, widen, renew, or replay a lease (`lease.py`,
`schemas/capability-lease-v1.schema.json`).
