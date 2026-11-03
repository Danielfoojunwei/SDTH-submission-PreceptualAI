# Measured RF response to fleet service

The earlier 77.1% Bayesian response-error reduction did not establish better packet delivery. This study tests that missing link in complete 100-robot trajectories using the native procedural warehouse and archived ns-3 NIST curves. It is not 100-robot ray tracing or physical SmartCar evidence.

```mermaid
flowchart LR
  Public[Public source activity and geometry] --> Prior[Reciprocal RadioMap prior]
  Scans[Completed delivered scans] --> Prior
  Truth[Private physical state] --> Kernel[Versioned packet kernel]
  Public --> Kernel
  Kernel --> Capture[Raw AP power captures]
  Capture --> Transport[Paid delayed reporting]
  Transport --> Memory[Public mailbox and local response update]
  Prior --> Memory
  Memory --> RF[Estimated concurrent RF correction]
  RF --> FIFO[Charged forecast FIFO]
  FIFO --> Control[Canonical planning and recovery]
  Control --> Kernel
  Kernel --> Outcomes[Control, video, bulk and backlog]
```

Private coupling and true sensing masks never enter the predictor. Individual gains need identifiable individual exposure directions; a known synchronized sum cannot invent them. Unknown/stale gains retain the prior. Dedicated monitoring, reciprocal channels, stationary managed coupling and reliable reserved reporting are explicit simulator assumptions.

Sensed sources cause domain deferral. Hidden power is conditioned on the remaining transmission window, avoiding duplicate charging for deferral and RF overlap. All managed-study slots use the same corrected per-band retry allocator. Only managed_probes=None preserves predecessor reproduction. AP-assisted sensing and average-power error curves remain approximations.

The [registered protocol](../artifacts/continual_packet/registration/protocol.json) fixes eight arms, 32 development and 192 confirmation runs, 100 robots, 80 seconds and two handover costs. The scanner-protection mission preserves native power-control politeness. Development selects only the registered compute charge. No neural fitting or confirmation tuning occurs.

No-probe and schedule-matched ignored-response controls separate information from physical scheduling. Whole-site uncertainty and fixed service/timing vetoes determine claims. Original requested video, bulk backlog/overflow and pending endpoint commands remain visible. Null or harmful outcomes are retained.

The separate opt-in [durable journal](DURABLE_EXECUTION.md) and [spawned proposal worker](PROCESS_ISOLATION.md) address local crashes and stuck inference. They change no physical-radio default or authorization. Hardware capability, physical deadlines, risk calibration and deployment qualification remain separate requirements.
