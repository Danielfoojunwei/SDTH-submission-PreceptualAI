# Architecture

## Fast local loop

1. `TelemetryCollector` obtains interface, NetworkManager, WiFi, cellular, byte
   counter, and probe measurements from the operating system.
2. `ProfileOptimizer` removes profiles with missing mandatory evidence and
   computes a normalized service utility.
   Required 3GPP or SDR standards evidence is part of eligibility, not merely a
   score bonus.
3. `SafetyShield` checks catalogue signature/expiry, observation freshness,
   profile authorization, confidence, dwell time, and switching rate.
4. `NetworkManagerExecutor` either records an advisory or activates the exact
   approved connection UUID and confirms that it became active.
5. `AuditLedger` persists input hashes, scores, decision, execution result, and
   previous-record hash.

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

## Trust boundaries

- The operator signs the profile catalogue offline.
- The node contains only the catalogue public key.
- NetworkManager owns actual device configuration and activation.
- Advanced 3GPP slicing and SDR profiles require a root-owned local standards
  agent that attests capability and confirms the exact approved profile.
- Kval can select only a catalogue UUID; it cannot construct arbitrary radio
  settings.
- The host may keep Kval in advisory-only mode indefinitely.
- Fleet coordination and aggregate decryption are separate from the node loop.
