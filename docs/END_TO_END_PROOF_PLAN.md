# End-to-end proof plan

The system is proven only when evidence exists for every boundary below. Unit
tests alone prove software behavior, not radio, learning, or fleet performance.

| Layer | Build | Passing evidence |
|---|---|---|
| Node identity and context | Signed node manifest and runtime context digest | Tampering, expiry, role mismatch, platform mismatch, and cohort mismatch all fail closed |
| Real observation pipeline | Live NetworkManager, ModemManager, counters and bound probes | Time-aligned field traces from every physical node, with provenance and missing-data flags |
| Deterministic baseline | Existing constrained optimizer | Beats a fixed-profile baseline on held-out real traces without violating the Safety Shield |
| Local learning | GNN-DQN training pipeline plus signed ONNX/TensorRT artifact | Reproducible training run, frozen feature schema, held-out evaluation, calibration and Jetson latency |
| Experience capture | Append-only context/action/outcome records | Every sample binds node context hash, catalogue hash, action, outcome window and audit-chain record |
| Fleet learning | Authenticated cohort enrollment and robust aggregation | Three or more real nodes complete rounds despite dropout; poisoned/outlier updates are rejected |
| Privacy layer | Threshold secure aggregation first; FHE only for the small agreed computation | No coordinator can inspect an individual update; dropout and key-rotation tests pass |
| Model release | Signed model manifest, compatibility gate, canary and rollback | Wrong cohort/schema/hardware/signature is rejected; regression triggers deterministic rollback |
| Radio actuation | Root-only radio agent and vendor driver | Exact approved profile activates on real modem/SDR and is independently verified |
| Whole-system HIL | Minimum three Jetson Orin Nano nodes and a controlled network impairment setup | Priority traffic recovery, bounded switching, learning improvement, isolation behavior and audit reconstruction all meet targets |

## Required physical demonstration

Use at least three independently provisioned Jetson nodes with different signed
roles. Each node must show its identity/context, collect real link outcomes, run
the same signed baseline or learned model compatible with its cohort, and retain
local authority when disconnected. The fleet coordinator may distribute signed
models and aggregate authorized updates, but it may never bypass a node's Safety
Shield or radio-agent allowlist.

## Learning acceptance gates

Do not claim a learning subsystem works until all of the following are retained:

1. Real, consented, time-aligned traces and a versioned feature schema.
2. Train/validation/test split by time and deployment, not random rows.
3. Deterministic baseline comparison and ablation results.
4. Safety-violation rate of zero after the independent shield.
5. Calibration, abstention and out-of-distribution tests.
6. Signed model artifact, training-code commit, dataset manifest and metrics.
7. TensorRT accuracy parity and latency/power measurements on every node class.
8. Canary rollout, rollback and fleet-partition recovery evidence.
