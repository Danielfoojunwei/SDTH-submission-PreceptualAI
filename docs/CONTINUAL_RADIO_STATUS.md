# Continual radio learning, execution recovery and research monitoring

The completed response experiment favors **independent observations and fast
Bayesian calibration** over the registered JEPA model. The runtime now handles
specific software faults more safely, while every current learned candidate
remains unqualified for production execution.

## What the experiment establishes

The [registered study](CONTINUAL_RADIO_RESULTS.md) collected actual Sionna RT
couplings for 24 configured blocker contexts in one warehouse. Three missing
links stopped the initial collection before fitting. A separately registered
uniform higher-depth collection found paths for all 96 entries. Both
collections and source capsules remain intact.

Six fixed 30-epoch fits compare response JEPA with a matched supervised model.
At two staggered feedback observations, their clean response errors average
**0.146 and 0.0894 nW**; least squares reaches **0.0637 nW** and Bayesian
identification **0.0309 nW**. The JEPA-specific gate fails. Teachers retained
variance and response sensitivity, which did not create a predictive advantage.

The Bayesian method reduces response error **77.1%** when synchronized feedback
is replaced with staggered feedback at the same total transmitter-on airtime.
That is a measurement-design result, not a JEPA improvement, packet-deadline
result or proof of equal service cost. Probe measurements, receiver gains,
noise and missing feedback are explicitly constructed from the RT kernels.
Five held-out contexts are blocker poses in the same warehouse; they are not
independent warehouses or measured robot trajectories.

## Learning only what a decision needs

[`ContinualResponseMemory`](../src/wirelessjev/continual_memory.py) retains bounded
device/context/radio-epoch statistics. It updates after observations actually
arrive and separates causal-provenance claims from ordinary associations.

An observation `[1,1]β=5` identifies joint query `[2,2]β=10`, while individual
query `[1,0]β` remains unknown. The whole kernel need not be identified for a
particular action effect to be identifiable. Conversely, more repeated
synchronized observations cannot identify the missing contrast direction.
Missing or expired evidence returns unknown. No numerical rank or declared
provenance creates a calibrated risk bound. See [CONTINUAL_MEMORY.md](CONTINUAL_MEMORY.md).

This is continual local-state assimilation, not live backbone retraining.
Shared representation updates, cross-robot teacher learning and generalization
to new devices/warehouses still require separate evidence. The
[design and falsification criteria](CONTINUAL_RADIO_DESIGN.md) retain those limits.

## Software recovery and production limits

[Execution semantics v2](PRODUCTION_EXECUTION_FIXES.md) adds atomic admission,
single-use issue, owned/idempotent reservation settlement and stricter UNKNOWN
reconciliation. New closed-loop outputs identify their execution version;
historical results retain their original snapshots and semantics.

[Runtime fixes](PRODUCTION_RUNTIME_FIXES.md) bound work, histories and telemetry,
reject stale/invalid forecasts, isolate mutable buffers, and preserve baseline
admission through shadow-worker errors and closure. Learned control remains
inactive. A stuck Python thread cannot be forcibly terminated by this API.

The [qualification contract](MODEL_PROMOTION_CONTRACT.md) checks hash-bound
receipts rather than trusting a paper title or success flag. Actual candidates
are listed in [qualification_status.json](../artifacts/ran_research/qualification_status.json).
None is production eligible. The later opt-in [durable execution wrapper](DURABLE_EXECUTION.md)
and [spawned proposal worker](PROCESS_ISOLATION.md) add tested local obligation
recovery and supervised inference-process termination. They are separate from
the original in-memory ledger and thread services. Physical timing,
rare-event/correlation evidence, capability checks and deployment integration
remain requirements; software unit tests do not supply them.

## Continuous research, without automatic model promotion

The [research watch](RAN_RESEARCH_WATCH.md) completed a real 10-request run and
retained 47 unreviewed leads. [Fresh primary review](RAN_RESEARCH_UPDATE.md)
separately checks methods, public releases, licenses and compatibility.

A real pinned [LWM-Temporal checkpoint](LWM_TEMPORAL_ACCESS.md) with 241,634
parameters ran on CPU after source/license review. Its complex-CSI input and
future-offset attention are incompatible with treating our present RSSI logs as
a causal production input. This establishes pretrained executability only.

The workflow requests daily discovery at **08:00 Asia/Singapore** once merged
to the default branch with Actions enabled. Discovery retains evidence and
requires scientific review. It never installs discovered code, changes weights,
posts comments or promotes a controller. Cache loss or incompatible history
must not silently restart accumulated research from the bootstrap seed.

## Verification

The final suite passes **582 tests in 91.74 seconds**, without failures or
skips, including the cached pretrained DINOv3 integration. The 44 metadata and
history tests also pass with only pytest installed. Independent review verifies
those tests plus all 40 qualification tests and closes the four identified
history/exposure/calibration blockers. Actionlint 1.7.7 validates the workflow.

The [verification receipt](../artifacts/ran_research/verification.json) records
source and evidence hashes and the retained full test log. The archived study
also reproduces its summary/completion bytes without additional fits. These
checks establish the stated software and offline evidence, not physical-device
reliability, SOTA or a live scheduled GitHub run.

GitHub rejected the push and PR jobs before any steps due to an **account
billing lock**. The [CI receipt](../artifacts/ran_research/ci/status.json) retains
the exact annotations and a separate successful local initial-restore check
against actual GitHub metadata. Daily discovery requires billing resolution
and default-branch merge; it is not currently running on GitHub.
