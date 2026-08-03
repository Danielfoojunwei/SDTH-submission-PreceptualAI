# Architecture

## Fast local loop

1. `ClockMonitor` runs first, because it gates everything that compares
   timestamps. Its verdict travels with the cycle.
2. `TelemetryCollector` obtains interface, NetworkManager, WiFi, cellular, byte
   counter, and probe measurements from the operating system.
3. `PowerGovernor` bounds power mode and duty against signed policy;
   `LinkStateMachine` advances link health; `assess_emcon` resolves which link
   roles a standing emission-control order suppresses.
4. `ProfileOptimizer` removes profiles with missing mandatory evidence and
   computes a normalized service utility, **within one link role only**.
   Required 3GPP or SDR standards evidence is part of eligibility, not merely a
   score bonus.
5. **`plan_allocation` places each declared flow across slices.** This is the
   decision the loop exists to make. Ranking still runs, because switching is
   still how a dead bearer is replaced, but a flow is carried by a *spread* of
   slices rather than committed to whichever single link ranked highest — and a
   spread that could not survive losing its largest placement is refused rather
   than delivered short. See `allocation.py`.
6. `SafetyShield` checks catalogue signature/expiry, observation freshness,
   profile authorization, confidence, dwell time, and switching rate.
7. `NetworkManagerExecutor` either records an advisory or activates the exact
   approved connection UUID and confirms that it became active.
8. `AuditLedger` persists input hashes, scores, decision, execution result, and
   previous-record hash.

**What the loop no longer does.** It does not sense the radio spectrum. That
stack, the cross-node band-report fabric, and the weather-discrimination tier
built on both were removed: together they outweighed the decision they served
and never reached a flying node's choice of link. Degradation is detected the
way the vehicle experiences it — by probing the links themselves.

**What the allocator does not do.** It computes a placement and the reasons for
it. It does not move packets. Enforcement belongs to a data-plane component
under an operator-signed lease, and no plan authorises anything on its own.

The service writes its latest bounded result atomically to
`/run/kval/latest.json`. No raw packets or application payloads enter Kval.

## Why the first optimizer is deterministic

A genuine GNN-DQN requires representative traces, a frozen objective, training,
held-out evaluation, an exported model, and a signed model lineage. None can be
replaced honestly with random weights or synthetic claims. The first release
therefore uses a transparent constrained optimizer and stores real outcomes for
the future research model.

A TensorRT policy may later be added behind the same `Recommendation` contract.
It must fail closed unless its engine, feature schema, calibration report, and
signature are all present. The Safety Shield remains independent of the model.

## Capability loops

Four additional loops run alongside the connectivity loop. Each is optional, each
is governed by its own operator-signed policy, and each fails closed when that
policy or its evidence is missing.

1. **Power governance** (`power.py`) — a signed thermal/battery/watt/duty policy
   and a governor with thermal hysteresis bound the node's power mode, inference
   duty, and scan/telemetry intervals.
2. **Connectivity recovery** (`recovery.py`) — a link-health state machine with
   exponential backoff, failover, and a flap ceiling that escalates rather than
   thrashing the radio.
3. **Secure-transport verification** (`transport_security.py`) — verifies the
   WireGuard tunnel, both default routes, the nftables ruleset, and WPA3 link
   security against signed policy. See
   [the secure-transport and EW threat model](SECURE_TRANSPORT_AND_EW_THREAT_MODEL.md).

Loops are ordered deterministically — safety → thermal/power → connectivity →
optimization — so two loops cannot drive the radio in opposite directions in the
same tick. `multimodal.py` fuses their observations with per-modality provenance;
`learning.py` holds the fail-closed guardrails (signed model manifest, OOD/drift
abstention, change-point detection, causal attribution) that any future
predictive model must pass. See
[learning in autonomous networks](AUTONOMOUS_NETWORK_LEARNING.md).

## Trust boundaries

- The operator signs the profile catalogue **and every capability policy**
  offline; the node contains only the operator public key.
- NetworkManager owns actual device configuration and activation.
- Advanced 3GPP slicing and SDR profiles require a root-owned local standards
  agent that attests capability and confirms the exact approved profile.
- Kval can select only a catalogue UUID; it cannot construct arbitrary radio
  settings, and it has no transmit path at all.
- Every actuation beyond local measurement requires an operator-signed,
  node-bound, single-use, expiring capability lease (`lease.py`). The closed
  `LeaseAction` set contains no action that transmits, alters trust anchors,
  deletes audit records, or widens the node's own authority.
- The host may keep Kval in advisory-only mode indefinitely.
- Fleet coordination and aggregate decryption are separate from the node loop.
