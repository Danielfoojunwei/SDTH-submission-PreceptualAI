# Service-world native diagnostic results

**This is a completed simulation diagnostic with an untrained prototype.** It is not a learned-model result, a physical 100-car trial, SOTA validation or a novelty certification.

Each case contains 100 simulated robots over 80 seconds. The four diagnostic sites are 360100, 360101, 360102, 360103. Original held-out evaluation sites were not accessed. These are descriptive diagnostic counts; packets and sampled robots are not independent training replications.

The completed [sampling pilot](../artifacts/service_world/pilot_sampling_v1_exclusion.json) is preserved separately: strict query-time freshness omitted the intended weak-link cases. The present run uses fresh diagnostic sites and ranks the latest actually observed best-band SNR at the currently associated AP. Saved observation timestamps and decision ages make the lag explicit; current availability remains binding. Neither run trains a model.

## Actual factual and legacy packet counts

Both arms use the numerical RadioMap controller. The factual arm expires pending proposals after one second; the legacy arm has no added proposal lifetime (`None`). Their difference isolates a classical pending-command-expiry change. It is not a JEPA gain. Rates below are calculated from total integer misses/sent, not averages of rounded robot rates.

| Handover | Cases | Factual 1 s TTL misses / sent | Legacy `None` misses / sent | Factual − legacy misses | Factual / legacy ppm |
|---|---:|---:|---:|---:|---:|
| 0.03 s | 4 | 6,640 / 640,000 | 6,640 / 640,000 | +0 | 10,375.00 / 10,375.00 |
| 4 s | 4 | 40,173 / 640,000 | 40,173 / 640,000 | +0 | 62,770.31 / 62,770.31 |

## Independent one-second interventions

Each recorded stay/move intervention is compared with the actual unforced factual continuation from the same checkpoint. The intervention lasts one second, then canonical recovery resumes; miss differences are measured over ten seconds. Negative means fewer focal-robot misses. Focal robots are selected from public pre-intervention measurements, rather than future outcomes.

| Handover | Recorded actions | Beneficial | Harmful | Zero | Checkpoint/robot groups with a negative action |
|---|---:|---:|---:|---:|---:|
| 0.03 s | 72 | 3 | 3 | 66 | 3 / 36 |
| 4 s | 72 | 5 | 31 | 36 | 5 / 36 |

**Best-action selection is NONCAUSAL hindsight.** The [full summary](../artifacts/service_world/summary.json) records the best negative action and tied best actions for every checkpoint/robot group, including groups with none. Each action starts an independent fork of a factual checkpoint. Combining these choices or adding their gains does not produce a joint fleet-policy result. No trained model selected these actions.

## Where factual losses occur

| Handover | No radio available | All paths blocked | Weak link | Contention | Other |
|---|---:|---:|---:|---:|---:|
| 0.03 s | 6,391 | 246 | 3 | 0 | 0 |
| 4 s | 27,364 | 1,836 | 10,973 | 0 | 0 |

These are the kernel's diagnostic attribution categories. They do not establish counterfactual inevitability. Exact outage floors require the actual packet phase, pending state and merged unavailable intervals; a learned predictor cannot restore service during a link-exclusive unavailable interval by itself.

## Executed work and source linkage

The audit contains **8 complete cases, 184 native simulator runs, 24 factual suffix replay checks, 72 recorded input windows and 144 intervention branches**. The replay assertion checks per-robot cumulative miss counters against the matching full factual endpoint; it does not establish equality of every simulator state field.

The reporter verifies every case/input checksum and agreement between [completion](../artifacts/service_world/completion.json), [executed-source manifest](../artifacts/service_world/executed_source.json), [protocol](../artifacts/service_world/protocol.json), archived source copies and current source. 67 source files were verified. The JSON summary preserves the individual hashes and reporter hash. This reporting step does not rerun the simulator or train a model.

## What the prototype establishes

There are **zero trained models, zero learned-policy evaluations and no learned accuracy/control score**. The inactive prototype has 68,021 inference parameters in the recorded probes. Real causal input windows exercise its forward path; a zero residual preserves the supplied physical reference exactly, and an unknown hazard remains `None`. This checks an implementation identity. The physical reference is a known-unavailability lower bound, rather than an estimated or calibrated expected service value. It is not evidence of improvement over RadioMap.

## Scientific claim boundary

The [implemented model](SERVICE_WORLD_MODEL.md), [primary-source prior-art review](SERVICE_WORLD_PRIOR_ART.md) and [official baseline access audit](SERVICE_WORLD_BASELINES.md) describe the current evidence. Action-conditioned wireless JEPA, mechanistic residuals, robot/wireless coupling, matched counterfactuals and stochastic graph dynamics overlap directly with existing literature. No broad first-of-kind claim is supported. Official upstream component forwards are not matched warehouse experiments; unavailable code/data and unmatched tasks prevent a global SOTA conclusion.

The [first-principles service bound](SERVICE_WORLD_FIRST_PRINCIPLES.md) explains why packets whose slack is shorter than a link-exclusive unavailable interval cannot be rescued by a better predictor alone. Separate links, redundancy or a different intervention change that premise and require their own validated model and measurements. Training independent seeds, matched non-JEPA/classical comparators, sealed evaluation and real warehouse validation remain necessary before claiming a learned advance.
