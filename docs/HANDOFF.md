# WirelessJEV research handoff

## Latest: compact JEPA training and fresh confirmation complete

[SELECTED_JEPA_RESULTS.md](SELECTED_JEPA_RESULTS.md) records **12 fits across
four families and three paired training seeds, 112 development evaluations,
128 fresh confirmation evaluations and eight exact replays**. Each evaluation
has 100 simulated robots. All owned training/evaluation stages are complete.

The compact model preserves AP/band structure, uses genuine delivered native
simulator observations, and predicts bounded corrections directly from a
RadioMap anchor at each horizon. Its 13,970 inference parameters and separate
training projection/EMA teacher are implemented and exercised. No recursively
summed residual exists; cumulative control loss can still grow.

Fresh results do not confirm superiority. At 30 ms / four-second handovers,
context-JEPA command loss is 1.081806% / 4.789965%; RadioMap is
1.097604% / 4.849583%; matched supervised is 1.079757% / 4.777951%;
original frozen JEPA is 1.071562% / 4.691771%. All six statistical gates fail.
Engineering margins pass versus RadioMap/supervised and fail versus the old
ensemble. One actual RadioMap warm call, 284.938 ms against 120 ms, triggers
the registered global timing veto. No run, budget or criterion was replaced.

[SEMANTIC_JEPA_RESULTS.md](SEMANTIC_JEPA_RESULTS.md) retains the original
semantic candidate's rejection on all 112 development runs: zero original
confirmation/replays, with `2104000`–`2104007` unopened. Training source was
published before collection in `9ff5fa5`. The separate context selection and
230-file source/input capsule were published in `f5c69d5` before the eight
new `2114000`–`2114007` sites. It reuses fixed weights/timing and does not
rehabilitate the semantic rejection or select a new winner after confirmation.

The post-hoc representation audit uses all 1,152 existing validation rows,
six sites and all three context-JEPA/supervised members, without new data,
perturbation or subsampling. All measured streams are finite/nonconstant;
rank is not a guarantee of useful RF information or absence of every collapse.
Both scientific source closures and earlier PR18 source/evidence remain intact.

Full software validation: **895 passed, one expected warning**. The original
pre-data suite's 838-pass receipt is retained. Final portable checks:

```bash
uv run --frozen python scripts/verify_semantic_jepa.py
uv run --frozen python scripts/verify_selected_jepa.py
```

The original verifier optionally accepts `--verify-caches` for existing local
NPZ files. Neither verifier simulates, trains or retimes. Derived figures and
source/input/output receipts accompany both studies. Raw cases, all member
outcomes, negative/null comparisons, actual timing and replay traces are kept.

Draft [PR #19](https://github.com/Danielfoojunwei/SDTH-submission-PreceptualAI/pull/19)
is stacked on #18, not merged. RadioMap remains the application default.
Hardware timing, cross-layout transfer, calibrated risk and physical recovery
remain unqualified. The results establish no SOTA, EMA necessity, universal
JEPA benefit or indefinite packet-loss bound. All remaining physical/external
publication requirements below still apply.

## Previous: completed costed RF-to-packet confirmation

[CONTINUAL_PACKET_RESULTS.md](CONTINUAL_PACKET_RESULTS.md) records all **32
development + 192 confirmation evaluations and six exact replays**. Each run
has 100 simulated robots for 80 seconds. Six confirmation sites are procedural
instances sharing CAD, with two paired packet seeds and two handover costs.
No neural models were fitted in this follow-up. The earlier six response fits
and original RT evidence remain unchanged.

Neither response memory nor Bayesian updating passes the registered service
gate. All 12 primary and four information contrasts fail even before endpoint
sensitivity. At four-second handovers, canonical command loss is 13.6070%,
staggered memory 13.8309% and staggered Bayesian 13.8321%. Worst-decile margins
fail; silence/video/bulk margins pass. Source/accounting checks and all six
replays pass, with no development/confirmation runtime overrun. No candidate
is promoted. RadioMap remains the application default.

The frozen adapter corrects RF power but passes no typed future busy-airtime
state to the planner. Public source activity changes every 100 ms while the
forecast samples integer seconds and is delayed. These are representation
limitations, not proven causal explanations or grounds for post-test tuning.
A replacement needs a separate resource/time interface and fresh validation.

The opt-in [durable journal](DURABLE_EXECUTION.md) and
[spawned worker](PROCESS_ISOLATION.md) are implemented and tested separately.
They retain ambiguous command obligations and support supervised inference
termination, respectively. They do not persist learned memory, migrate old
thread services, qualify hardware or activate physical commands.

Final software validation: **697 passed, one expected warning, no skips**.
[The verification manifest](../artifacts/continual_packet/verification/manifest.json)
binds source, documents and evidence, including all 284 source and 148 evidence
hashes retained from PR #17. The supervisor exited before report generation;
only deterministic aggregation was restarted from completed archives. No
scientific run or model was rerun because of that interruption. No owned
training/evaluation job needs to remain running.

Draft PR #18 is stacked on #17; neither is merged into `main`. No GitHub checks
are reported for #18 under the current workflow path filters. The last recorded
Actions attempts were blocked before job execution by account billing. Daily
research-watch activation still requires working Actions and default-branch
merge. Hardware timing, capability, risk calibration and field recovery remain
open deployment requirements.

## Completed response experiment, execution recovery and research watch

`CONTINUAL_RADIO_STATUS.md` indexes the completed work. Six fixed 30-epoch
response JEPA/supervised fits use 24 actual RT contexts in one warehouse,
with explicit constructed probe/noise/device episodes and five held-out poses.
The JEPA gate fails: staggered K2 errors are 0.146 nW (JEPA), 0.0894 nW
(supervised), 0.0637 nW (LS), and 0.0309 nW (Bayesian). Bayesian staggered
feedback improves response MAE by 77.1% over equal-transmitter-airtime
synchronized feedback. No packet-control/hardware improvement follows.

The initial depth-two collection failed coverage and is preserved. A separately
committed uniform higher-depth amendment yielded all 96 positive found-path
couplings before training. Source capsules, both collections, all checkpoints
and archived-source byte-identical evaluation are retained under
`artifacts/continual_radio/`. No post-test extra fits or model selection ran.

`CONTINUAL_MEMORY.md` implements bounded per-device/context/radio-epoch state
updates and action-specific identifiability. It does not fine-tune a backbone
online or create risk guarantees. `PRODUCTION_EXECUTION_FIXES.md` identifies
active execution semantics v2; old snapshots/results retain v1 behavior.
`PRODUCTION_RUNTIME_FIXES.md` covers bounded/age-checked forecasts and
baseline-preserving shadow faults. The later opt-in
[SQLite journal](DURABLE_EXECUTION.md) recovers local execution obligations;
the [spawned proposal worker](PROCESS_ISOLATION.md) supports supervised
terminate/kill/reap. Existing thread services remain unchanged. Deployment
integration, physical timing and hardware recovery remain open requirements.

`RAN_RESEARCH_WATCH.md` records a successful live run: 10/10 requests and
47 unreviewed leads. Daily 08:00 Singapore discovery is configured, with
default-branch merge/Actions required for scheduled activation. Discovery
does not execute candidate code or promote models. `LWM_TEMPORAL_ACCESS.md`
records actual licensed pretrained execution and its incompatible complex-CSI,
future-attention input assumptions. `MODEL_PROMOTION_CONTRACT.md` remains
fail-closed: all actual candidates are unqualified for production.

Final validation: **582 tests pass** with no skips, including cached pretrained
DINOv3 integration. All 44 watch/history tests also pass with pytest alone;
40 qualification tests reject the short-exposure and rare-underprediction cases.
Independent review closes the four identified qualification/history blockers.
Actionlint 1.7.7 validates the workflow. `artifacts/ran_research/verification.json`
records source/evidence hashes and the final test log. Training is complete;
no training job needs to remain running.

Draft PR #17 is published. GitHub rejected push run `41671278827` and PR run
`41671318938` before any job step because the account is locked for billing.
`artifacts/ran_research/ci/status.json` and annotation bodies retain this blocker.
The initial restore path did succeed locally against the actual GitHub API and
verified all ten retained response bodies. That is bootstrap evidence, not a
live recovery of a prior workflow artifact. Resolve billing before expecting
Actions to run; daily scheduling still needs default-branch merge.

## Latest: representation diagnosis, actual DINOv3 and physical hierarchy

`VISUAL_RADIO_STATUS.md` indexes the implementation and evidence. The frozen
critic audit finds low effective rank but nonconstant, action-sensitive and
informative latents. Its probes use prior training scenes; they are not model
holdouts. Six new preregistered fits compare the original shared latent with
a training-only future projection. Future MSE relative to a training-target
constant improves from 1.623 to 0.580 without target collapse, while mean
validation miss MAE worsens from 23.322 to 23.710 commands. The service gate
fails; the new critic is not qualified for control. Original held-out sites
and previous evidence remain untouched.

`DINOV3_ACCESS.md` records real pretrained weights, their license/checksums and
actual dense-feature extraction on two licensed warehouse/robot photographs.
Original Facebook access returned 401; the public timm DINOv3 conversion was
downloaded legitimately. `timm==1.0.30` is isolated outside the root environment.
Images with unknown capture times cannot become fresh temporal RF inputs.

`VISUAL_RADIO_PHYSICS.md` and `VISUAL_RADIO_HIERARCHY.md` document receiver-centric
interference factors, calibrated geometry, per-link spatial fusion and finite
option planning. Computation completion, expiry and observation freshness gate
proposals; unknown service preserves canonical continuation. Fusion is
untrained, and no physical radio execution bridge is supplied. Synchronized
warehouse visual/RF logs and matched held-out comparisons remain necessary.
`VISUAL_RADIO_RT.md` adds a real CPU Sionna RT/Mitsuba fixture with all 53
warehouse objects, two explicit blocker poses, actual 2 × 2 channel couplings
and DINOv3 encoding of the same rendered images. All desired/cross couplings
are nonzero. The first zero-path placement is preserved and excluded. These
are static simulated inputs, not a timed robot dataset or a learned RF result.

## Latest: physics-anchored service prototype and causal diagnostic

`SERVICE_WORLD_RESULTS.md` records the finite diagnostic, with source/input
checksums and independent integrity checks. It evaluates unforced canonical
continuation in eight cases (four sites, two handover costs), with 184 simulator
runs, 24 exact miss-counter suffix replays and 144 interventions. All 318
software tests pass; `artifacts/service_world/verification.json` records scope.
It compares that continuation against one-second stay/move interventions, followed by canonical
recovery. A separate numerical comparator retains legacy pending-command
expiry, so that a classical fix cannot be attributed to JEPA. Original reserved
confirmation sites remain unused.

The 68,021-parameter prototype has **no trained weights, learned accuracy
score or active control qualification**. Public AP/band events retain their
measurement masks and physical timestamps. Power/SINR helpers and exact
known-outage deadline floors are independently tested. `SERVICE_WORLD_MODEL.md`
and `SERVICE_WORLD_FIRST_PRINCIPLES.md` describe implementation and limitations.
The completed first sampling run is retained under
`artifacts/service_world/pilot_sampling_v1/`; its query-time freshness check
omitted intended weak-link cases. The corrected protocol uses last-observed
measurements with explicit ages and fresh diagnostic sites.

`SERVICE_WORLD_PRIOR_ART.md` reviews 18 primary sources and rejects broad
first-of-kind claims. `SERVICE_WORLD_BASELINES.md` pins official upstream code,
licenses and real CPU component forwards, including missing code/data and
runtime gaps. Those forwards are executability checks, not benchmark scores.
Training independent seeds, matched numerical/non-JEPA/external comparisons,
sealed evaluation and physical transfer remain necessary for an advance claim.
The new diagnostics do not overturn the earlier rejected packet-JEPA evidence.

Reproduce with `uv run --frozen python -m wirelessjev.service_world.study --workers 4`,
then `uv run --frozen python scripts/report_service_world.py` and
`uv run --frozen python scripts/audit_service_world.py`. Completed cases resume
only when source/input checksums match. Do not modify sealed source or silently
reuse a study after changing its policy/observation contract.

## Packet-action JEPA: six fits complete, rejected on development

`PACKET_VALUE_DESIGN.md` and `PACKET_VALUE_RESULTS.md` record actual JEPA
EMA future-latent training against a matched supervised ablation. Each has
three independent initialisations, the same 23,860 inference parameters and
the same RadioMap route assistance. Forty epochs per fit completed. The
counterfactual corpus contains 744 training and 186 naturally sampled validation
examples from 1,370 native packet runs, with 80 exact factual suffix replays.
State forks preserve full-duration channel draws, packet RNG, queues, pending
scans/reports, forecaster and controller state. Only public observations enter
the models; future observable context is a training target.

Training crosses 24/100-robot densities with stationary/drifting conditions and
samples both five-second and 65-second states. A superseded TTL-ten pilot
(30 completed cases, 1,021 recorded native runs) remains under
`artifacts/packet_value/pilot_ttl10/` and is excluded from fitting and validation.
The corrected forced action uses the same one-second expiry as runtime.

All 64 development runs finished on four sites, 100 robots, 120 seconds, both
30 ms and four-second handovers. At 30 ms, JEPA averages 24.598% command loss,
supervised 26.576%, common-controller RadioMap 12.698%, canonical RadioMap
1.045%. At four seconds: 12.780%, 15.719%, 14.158%, 7.515%. These are descriptive
development results. All eight confirmation sites remain unused; no primary
confirmation interval was computed and no JEPA superiority is claimed.

The replacement controller turns rejected uncertain proposals into ten-second
stays and suppresses canonical recovery. Unconditional calibration corrections
of 65–113 commands eliminate many useful fast moves. Raw learned packet values
also trail numerical values on validation, so relaxing the gate alone is not
a validated remedy. Cumulative losses and silence bursts worsen substantially.
The modeled-risk ledger is not a bound on factual regret.

`wirelessjev.packet_control.ShadowPacketAdmission` preserves baseline admission
and evaluates critic advice in one bounded background worker; it has no active
override mode. `require_confirmed_packet_profile()` rejects this candidate.
These safeguards retain canonical recovery without claiming learned benefit
or physical deadline qualification. See `PACKET_VALUE_SAFETY.md`.

Reproduce the development evidence with `bash scripts/run_packet_value.sh`.
It deliberately does not invoke the rejected profile's reserved confirmation.
All 221 software tests and the independent evidence audit pass. The original
frozen package and negative evidence remain intact. Further active
critic research needs baseline-continuation counterfactuals and a new controller
seal before the untouched tests. No owned training/evaluation jobs remain.

## Completed delivered-telemetry confirmation

`CLOSED_LOOP_DESIGN.md` and `CLOSED_LOOP_RESULTS.md` describe 24 development
evaluations and 96 fresh tests on eight seeds, with 100 simulated robots per
run. Four sites are stationary and four have the existing hidden drift process.
No weights were retrained. Scan measurements now share the native packet
channel; completed local scans inform neighbor selection, and fleet memory
receives only reports carried by a successful existing control exchange.
Current-link pilots freeze during scan/handover outages. Model inputs use
strictly past delivery bins and original measurement positions. Missing reports
remain missing. Forecasts become visible only after a charged FIFO completes.

Four development sites selected one shared 30-second outage-utility horizon
across RadioMap and the compact model. Under assumed four-second handovers,
compact loss falls 5.849% → 5.635% (3.67% fewer misses), paired-site 95% interval
[-3044.79, -1162.50] ppm, six wins and two ties. The identical utility controller
with RadioMap reaches 5.603%; the model comparison spans zero. Compact forecast
RMSE is 6.70 versus 8.54 dB, which still does not establish better control.

At 30 ms handovers utility harms all eight sites for both models: compact loss
rises 1.035% → 1.318%, with more roams and lower bulk throughput. Of its 2,715
additional misses, 2,704 are attributed to weak-link/blockage conditions.
Do not promote the slow-cost controller as a universal default. Its primary
confirmation is a within-model controller comparison. An automatic cost router,
immediate-link constraints and native packet-supervised risk need independent
development/test cohorts before any improvement claim.

All 192 tests pass. Four selected test arms replay exactly, including all 60
forecasts, delivered report events and per-robot packet counters. Protocol,
source/weight identities, latency, development selection, FIFO ledger, raw
causes and figures are archived in `artifacts/closed_loop/`. Reproduce with
`bash scripts/run_closed_loop.sh`; completed cases are reused with identity
checks. All owned evaluation jobs finished. Earlier archives are unchanged.

Loss remains too high for deployment. Initial history is a shared conditional
warm start, and pose/routes remain out-of-band. Reports reserve 128 bytes in
an existing 600-byte bidirectional control exchange; separate uplink/ACK loss
is not modeled. CAD and motion remain fixed/precomputed. CPU costs are sealed
microbenchmarks; handover/scan timings are assumed. Scanner encoding and
controller scoring still lack separate CPU deadlines. Physical-car trials,
independent layouts, PHY transfer and safety qualification remain open.

## Completed broader optimisation frontier

The new work is in `WORLD_FRONTIER_RESULTS.md`, `FRONTIER_CONFIRMATION_RESULTS.md`,
`HARDWARE_BUDGET_RESULTS.md` and `HANDOVER_ADMISSION_RESULTS.md`. It adds nine
18-epoch fits across three compact families and three seeds: distilled,
route/reliability-loss and temporal/innovation. Density and scan cadence vary
during training. Fleet sufficient statistics are incremental; RadioMap uses
the same posterior with faster BLAS covariance evaluation. New inputs include
absolute report count and AP-local cross-band context.

The main frontier has 96 selection runs and 160 tests on 24 fresh sites.
Independent compact-model confirmation adds 32 runs on eight sites. A
hardware-budget search adds 64 development and 32 fresh tests on four sites.
All-handover admission adds 48 tests on six sites. Total: 432 additional native
runs, 42 fresh sites. All 184 tests and the evidence audits pass. All owned
training/evaluation workers have exited; previous evidence is unchanged.

The single distilled model's measured warm service budget is 53.05 ms for
100 robots/nine horizons, versus 232.74 ms for the frozen three-member hybrid.
Optimised RadioMap is 67.20 ms. The temporal model won mixed-condition
development selection but failed its primary control-superiority interval.
The separately fixed distilled model reaches 228.12 versus 251.04 ppm on eight
additional sites, interval [-85.94, 39.06]; this does not establish superiority.
The primary compute improvement is real, but is not a universal control win.

With assumed 4 s handovers, 90 ms scans and 500 ms extra processing delay,
development selects five-second scans and outage-budget gating for every
reference. On four fresh sites the distilled model drops from 14.849% to
7.419% command loss. That experiment keeps the old reactive fallback. A new,
separate handover-admission hook then filters both planned and reactive
proposals: on six other sites, distilled loss drops from 8.851% to 5.920%.
The cohorts differ; do not chain these percentages into one improvement claim.
High remaining loss fails deployment qualification. Four-second outages are
not removed by more accurate mean-field prediction.

`wirelessjev.frontier_control.OptimizedWorldModel` and `AsyncForecastService`
provide a bounded, nonblocking proposal service with real background compute,
forecast-age interpolation and explicit past-only inputs. No radio actions are
performed. The main application default remains unchanged. Source-sealed
research kernels preserve absent-hook packet counters and separate the Boolean
confidence/fallback decision from physical admission of an outage.

Reproduce with `bash scripts/run_world_frontier.sh`; it verifies and reuses
sealed timing without recalibration. Each experiment has source capsules,
checkpoints/weights checksums, raw counters and an independent read-only audit.
The measured CPU ledger includes tensor packing and periodic processing costs;
the refit envelope also includes a catch-up probe. These are microbenchmarks.
Scan/handover durations remain assumptions. Scanner/telemetry availability is
not fully coupled to radio outages, motion is precomputed, and arbitrary CAD
or physical smartcars are not qualified. Native counterfactual reward training,
joint uncertainty and real observation/actuator timing remain open options.

## Completed fleet world-model improvement and independent confirmation

The implementation and results are in `docs/FLEET_WORLD_MODEL_DESIGN.md`,
`docs/FLEET_WORLD_MODEL_RESULTS.md` and `docs/FLEET_CONFIRMATION_RESULTS.md`.
Nine 24-epoch fits completed: three independent initialisations each for
hybrid JEPA, supervised hybrid and standalone fleet JEPA. Training uses 32
sites and six separate validation sites. Twelve further sites support 432
native controller-selection runs. The main study contains 296 tests on 32
fresh sites; independent confirmation adds 32 runs on eight new sites.
Total new native runs: 760. No historical study was rewritten.

The first study chose standalone Fleet JEPA on development data. Its primary
100.00 versus 115.10 ppm difference has interval [-48.45, 19.27], so its control
improvement was not established. It did improve forecast RMSE, 5.63 versus
8.34 dB, and avoids per-BSS RadioMap fitting. Its measured full CPU forecast
call is about 191 ms for 100 robots and nine horizons.

The best discovered hybrid was fixed BEFORE opening eight additional sites.
The independent primary test passes: supervised hybrid has 256 misses versus
321 for learned-prior RadioMap across 1.92 million commands per arm, or 133.33
versus 167.19 ppm. This is 20.2% fewer misses; paired 95% interval for the
difference is [-68.75, -5.73] ppm. Forecast RMSE is 4.57 versus 7.09 dB. The
secondary JEPA hybrid reaches 128.12 ppm; previous JEPA reaches 126.04. The
new hybrids improve forecasting but do not establish universal controller
superiority or the necessity of the JEPA objective.

The explicit `FleetWorldModel(geometry, family="confirmed")` profile keeps the
preregistered supervised candidate even though the secondary JEPA arm has fewer
confirmation misses. It checks the evidence/checkpoint identities and refuses
the profile without a passing independent primary result. The main application
default remains RadioMap. The hybrid CPU call takes roughly 0.4 seconds per
100-robot forecast plus periodic map refits; standalone is about 0.19 seconds.

All training/evaluation workers have exited. Source capsules, raw counters,
checkpoints, selection seals, CPU benchmarks and independent forecast replay
are in `artifacts/fleet_world_model/` and `artifacts/fleet_confirmation/`.
Reproduce sequentially with `bash scripts/run_fleet_research.sh`; audit with
`scripts/audit_fleet_world_model.py` and `scripts/audit_fleet_confirmation.py`.
Training caches are rebuilt under `/workspace/training/fleet-world-model`.

Limits: simulated mean-field supervision, shared canonical warehouse layout,
known geometry/localisation/noise floors, precomputed robot trajectories,
idealised reactive observations and 30 ms native handovers. Forecasts are causal
but supplied offline; measured inference time is not charged to packet deadlines.
Real scan costs, asynchronous inference, arbitrary-layout transfer and physical
smartcar qualification remain open. Stress/yard/drift use fresh seeds from
families represented in training. The new models are not universal winners.

## Completed JEPA gate repair

The follow-up in `docs/JEPA_GATE_REPAIR_RESULTS.md` completed 270 native
simulation runs (12 fitting traces, 90 closed-loop selection runs, 168 fresh
tests on 24 sites) and six additional diagnostic replays. All five JEPA members
are frozen; no world model was retrained. The six old primary counters reproduce
exactly and their archived evidence remains unchanged.

The old gate vetoed 19,012 of 19,331 eligible decision opportunities. 18,214
had positive true AP-score gain without reactive rescue. Median pair uncertainty
was 14.49 dB against 3.01 dB median actual gain error: the gate treated noisy
individual-reading dispersion as uncertainty of relative, route-averaged gain.

Development selected a 0.25 gate penalty. On six NEW primary scenes, old gate,
tuned scale, gain-bound gate, ungated JEPA and reactive rescue produce 329.17,
111.11, 133.33, 209.72 and 113.19 misses/million respectively. The 66.2% reduction
versus the old gate wins all six paired scenes. The extra gain model is not
superior to the simple rescue ablation on the main condition. The repaired
methods still lose to fixed split under stress.

`wirelessjev.jepa_control.controller()` exposes the development-best research
profile, checks the frozen model/calibration identity and restricts the validated
mission to `bulk_uploads`. RadioMap remains default. Native handovers assume
30 ms; real-car qualification remains pending. Workers have exited. Audit with
`scripts/audit_jepa_gate_repair.py`. The old report writer now refuses a changed
execution kernel before mutating historical outputs; use its frozen source
capsule for exact historical regeneration.

## Completed contracted world-model study

On 2 November 2026 (Singapore time), the device-scan JEPA continuation completed
all 20 twelve-epoch fits and 366 native rollouts across 24 fresh scenes.
`docs/CONTRACTED_RESULTS.md` is the result record; `docs/CONTRACTED_MODEL.md`
describes the implemented censoring, route conditioning, EMA and regret losses.
The portable archive is `artifacts/contracted/`; regenerate only its analysis
with `scripts/report_contracted_results.py`, and audit without retraining with
`scripts/audit_contracted_results.py`. Training/evaluation workers have exited.

The 100-robot primary comparison has 622.92 fixed-rule, 92.36 Bayesian-gated,
114.58 JEPA-ungated, 151.39 JEPA-gated and 150.69 regret-JEPA-gated misses per
million commands. The regret difference is one command across 1.44 million;
the five-seed/six-scene interval spans zero. Both JEPA gate arms beat fixed in
six scenes, so necessity is rejected. Scaling uncertainty changes every scene
despite preserved ranks; a rank-only guarantee is mathematically insufficient.
Removing route conditioning improves this particular regret family. Stress
generalisation fails to beat fixed. These results do not justify replacing RadioMap.

Eighteen open full texts were inspected in the relevant method/evaluation
sections. Prior art already covers uncertainty-triggered learned/conventional
switching, intent-aware world-model guardrails and coupled robot/wireless JEPA
models. `docs/CONTRACTED_PRIOR_ART.md` records the overlap and the failed direct
IEEE index request. The proposed broad paper claims are not ready.
RadioMap + WirelessJEV remains the default. No physical car was accessed.

The starting point for this continuation was GitHub `main` at `3c5ce3f` in
<https://github.com/Danielfoojunwei/SDTH-submission-PreceptualAI>.
The cloud development environment has Python 3.12, locked dependencies and a
CPU-only test/simulation workflow. See Git history and the associated review
branch for commit/push status; external publications are tracked separately.

## Completed native packet-supervision campaign

The v4 campaign at `/workspace/training/warehouse-v4-native` completed on
31 October 2026 at 17:50 Singapore time. All 37 development and 21 test native
runs, all six model fits, frozen selection, evaluation and verified archival
finished. The collector, supervisor and archiver have exited. Do not restart
completed runs. See `WAREHOUSE_NATIVE_RESULTS.md` for the final findings.

The portable `artifacts/warehouse_native/` archive contains 214 verified report
files, six finite checkpoints and three source capsules. The separate
`postrun_audit.json` recomputes all endpoint targets and records constant risk
labels. The original results remain unchanged. Verify without retraining:

```sh
.venv/bin/python scripts/audit_warehouse_native_results.py
```

The full thirty-second known-map study uses eight training scenes, three
validation scenes and six test scenes, with three seeds each for MLP and graph.
Neural forecast RMSE is about 0.158 versus numerical 0.413. Learned utility is
0.7010 versus numerical 0.6980, with difference interval [-0.0078, 0.0139].
All six choose identical test plans. The linear specialist reaches 0.7015.
No neural candidate or linear specialist passed validation; numerical remains
the default. Every aggregate critical-floor label is a miss, so the zero
neural Brier score does not establish discrimination or meaningful risk.

Executed source capsules remain immutable, including the pre-training
supervisor recovery attempt. Current reporting/export code may differ from
the collection capsule; do not regenerate caches with changed sources.
Studio bundle and standalone publication are separate from native completion.
No physical car, actual WPA/DHCP handover or persistent native roaming loop
was qualified. Both JEVs remain Wi-Fi-specific.

## Compositional mission and radio audit

The user's pasted six-direction request is mapped to executed evidence and
remaining acceptance boundaries in `WAREHOUSE_SCOPE_CHECKLIST.md`. The v3
protocol/results are in `WAREHOUSE_V3.md` and `WAREHOUSE_V3_RESULTS.md`.
All controller data remains simulated; actual pretrained inference and native
packet programs executed. The default proposal interface uses the complete
controlled grammar and numerical planning. Neural verification/scoring is
explicit and never sends a car command.

Retained completed runs are under `/workspace/training/`:

- `warehouse-v3`: twelve compact models, three seeds per architecture, four
  transfer domains; all model/test selection receipts retained.
- `warehouse-v3-packets-matched`: 48 native runs, 40 unique input/seed/RTS
  conditions. Fixed channels, matched NIST/VHT80/HT20/two-stream powers/profile,
  actual counters/deadlines; eight-second route prefix, assumed switch muting.
- `warehouse-v3-hidden-matched`: six constructed two-client native runs.
- `warehouse-v3-sensing-corrected`: sensing with real simulated previous-service
  counters and an explicit distinction between no scan and unheard APs.
- `warehouse-v3-studies`: completed shared/federation/removal stage. Its idle
  obsolete packet dependency was superseded; this root is not labelled a full
  three-stage completion. The sensing pilot was superseded separately.
- `warehouse-v3-packet-learning-unconditional` and `warehouse-v3-packet-calibration`:
  all time bins included; actual late-arrival-event calibration, not loss risk.
- `warehouse-v3-generative` and `warehouse-v3-schema-audit`: actual Qwen outputs,
  strict and default-aware schema audits; no label-guided repair.
- `warehouse-v3-clauses`: actual d3 clause verification on twelve fresh canonical
  statements, designed after the whole-family gate answered only four of ten.
- `warehouse-v3-ranker` and `warehouse-v3-ranker-record-audit`: full-priority
  group-aware text/no-text heads on fresh environments. The first raw case
  field `gold_choice` was an indicator; the fixed-model audit recovers actual
  indices and verifies all quality fields unchanged. Both versions are retained.
- `warehouse-v3-frontier-matched` and `warehouse-v3-cost-audit`: actual semantic,
  int8 and routing comparisons, per-domain complete local cost and crossed
  model-seed/environment intervals on frozen models.

Do not restart completed training. Superseded 20-MHz/one-stream packet runs,
conditional-on-arrival packet learning and reset-counter sensing are diagnostic
pilots, not completed matched primary evidence. Source/receipt lineage explains
the corrections. The native two-to-four-worker operational retry preserves
completed cases; physical inputs, seeds and binary remain unchanged.

Forecasting and native-event prediction improve in some comparisons; masked
and JEPA-style pretraining do not consistently beat scratch. Cached text does
not establish a general advantage. Pooled/FedAvg release gates reject all
candidates, and approximate removal differs from clean retraining. Retain the
negative outcomes. Historical large-model E1–E5 still have one training seed.

No physical car or network credential changed. Switch duration remains an
assumption; stage capture and field qualification need real hardware. Both JEVs
remain Wi-Fi-specific. External Claude publication and environment publication
remain separate from verified local serving and saved configuration drafts.

## Route-aware horizon continuation

The subsequent "continue building" request adds the protocol in
`docs/WAREHOUSE_HORIZON.md`, with results in `WAREHOUSE_HORIZON_RESULTS.md`
and `WAREHOUSE_HORIZON_SELECTOR.md`. The six independently trained compact
checkpoints and exact executed sources are portable in
`artifacts/warehouse_horizon/`; original E1–E5/E7 and v2 evidence is untouched.
The CPU campaign completed on 31 October 2026 at 09:48 Singapore time; its
fresh-scene selector audit completed at 10:00. No training worker needs restarting.

`/workspace/training/warehouse-horizon` contains the full training campaign
(18 training, six validation, three held-out environments; three seeds per
architecture; 30 full-cross control rollouts). The selector at
`/workspace/training/warehouse-horizon-selector` contains 42 validation and
24 fresh audit rollouts. It was designed after the original holdout; its
policy was sealed before audit environments 143000–143002 were evaluated.
An earlier validation-only selector attempt in `warehouse-horizon-gate`
was stopped before audit access; `superseded.json` explains the whole-schedule
replacement. That failed attempt is not a completed experiment.

All six residuals fail the closed-loop checks. The proposal CLI therefore
uses numerical planning, including when a rejected residual is requested.
An explicit `--research-residual` option permits simulated-input research
only. Actual d3 inference interprets the declared controlled mission grammar;
caller-supplied contracts are labelled separately. No car command is sent.
The observation schema does not authenticate a self-reported hardware origin.

On the fresh audit, numerical mean service is 0.326 versus 0.237 for budgeted
strongest-signal roaming. The twelve-switch cap is a declared operator-policy
scenario, not the existing concurrent transaction limit or a physical rate
limit. Unconstrained roaming has higher service in the original holdout.
These results do not qualify all robots, actual hardware or SOTA performance.

## Warehouse v2 continuation

The user subsequently authorised all six warehouse research directions and
requested actual dependencies and execution. The v2 source, protocols and
reproduction commands are in `docs/WAREHOUSE_V2.md`; interpreted results are in
`docs/WAREHOUSE_V2_RESULTS.md`. This continuation uses the separate branch
`codex/warehouse-v2-research`, preserving the original review branch below.
The implementation adapts pinned ns-3 examples, executes Sionna, and runs real
d3-lite, Qwen3.5-0.8B and Chronos-Bolt weights. It does not reproduce ANSpect
proprietary code or use physical SmartCar measurements. Hardware requirements
are in `docs/WAREHOUSE_HARDWARE_DATA.md`.

Workspace runs are under `/workspace/training/warehouse-v2-{core,studies,factorized,generative}`.
Use their status/receipts, archived report hashes and exact source snapshots
before resuming anything; completed compact-model training should not restart.
The historical large-model experiments still have one training seed. Three
new compact-model seeds must not be reported as three original E1–E5 runs.

## Original completion scope

The user returned the scope to warehouse connectivity: finish WirelessJEV and
RadioJEV E1–E5 and corrected E7. E6 remains excluded. Router, E8, JEPA/world
models, general robotics, custom PHY and satellite work are deferred.
`docs/WAREHOUSE_RESULTS.md` is the consolidated completion/result record.

**Completed:** all nine E1–E5 controller stages, validation, report archiving,
Studio rebuild and review-branch push finished on 30 October 2026 at 20:16
Singapore time. The automated archive commit is `bc63d21`; no training worker
needs restarting. Corrected E7 is complete. Only the specified one-seed
synthetic run is complete; independent seeds and physical qualification remain
separate work.

## E7 environment learning

Two censoring defects were identified in `src/wirelessjev/envlearn.py`:

1. A full eight-result scan used `min(reports + [-90])` as the upper bound for
   every omitted BSSID. This always produced −90 dBm, even when a BSSID was
   omitted solely because eight stronger APs occupied the scan slots.
2. The fitter treated violated bounds as point observations at bound − 3 dB.
   These invented measurements gained influence as surveys grew.

Scans now rank noisy measurements before truncating to eight APs, distinguish
the scan-capacity bound from the detection floor, and fit a censored Gaussian
likelihood with the same measurement-noise model as the detected samples.
Inverse-sampling weights account for retaining non-reports every third scan;
otherwise detections would be overrepresented in the fit.
Regression tests cover the scan bounds, the likelihood against an independent
scalar solution, recovery of a known censored signal and numerical stability.

The historical `artifacts/envlearn/report.json` is retained unchanged. Its
audible-path RMSE was 4.98 / 6.19 / 7.90 dB for 1 / 5 / 15 minutes. Corrected
results belong in `artifacts/envlearn_corrected/report.json`; see README for
the measured results and reproduction command. Current inaudible-path RMSE
is reported separately so it cannot be confused with audible-path accuracy.
This is not a guarantee that each environment improves monotonically with
more driving, or that never-observed interference paths are well calibrated.
The truth planner is a local-search reference, not a proven global optimum.

The corrected full sweep completed: 15 environments, five missions each.
Audible RMSE is 4.04 / 3.95 / 3.99 dB at 1 / 5 / 15-minute budgets; plan gap
is 0.202 / 0.182 / 0.159. Five environments still worsen from 1 to 15 minutes,
so the result does not support a monotonic-more-data claim. Currently
inaudible-path RMSE is 15.03 dB for the 15-minute map (all paths: 12.23 dB).
Those paths are not identical to paths never measured during the survey.

A subsequent matched-survey, map-only calibration study on new seeds 15–29
combines censored global calibration with nominal rack geometry. Inaudible
RMSE falls from 18.14 / 16.10 / 15.71 dB to 14.86 / 13.78 / 14.23 dB at
1 / 5 / 15 minutes; all 15 environments improve on that subset. Audible error
is nearly unchanged. This is not a downstream planner result, so the default
planner remains v2. The experiment, raw evidence and paired confidence
intervals are in `docs/experiments/e7_calibration_study.py` and
`artifacts/envlearn_calibration/report.json`. Distant-link error remains large.

Validation includes the full corrected E7 sweep, the additional calibration
study and regression tests. The original E1 LoRA and E7 reports remain unchanged.

## RadioJEV E1–E5

- **Complete:** E1 `radio_lora`, in `artifacts/radiojev/e1_radio_lora.json`:
  held-out regret 0.0928 ± 0.0222 (200 cases), new-mission regret 0.0744 ± 0.0196
  (240 cases), about 3.35–3.99 s per decision on the original machine.
- The numeric-only E1 result is also retained in `e1_no_llm.json`. It has no
  new-mission evaluation. Its small known-mission regret is not evidence of
  language generalisation.
- Those files describe the historical Mac run. The separate cloud run has
  completed all E1 and E2 arms, including the previously interrupted frozen
  arm and paired intent baseline. E6 remains excluded.

The new paired E1 intent baseline has held-out / new-mission regret
0.0140 ± 0.0066 / 0.0251 ± 0.0074, versus LoRA's 0.1068 ± 0.0240 /
0.0721 ± 0.0136. E2 radio-Q reaches normalized regret 0.1170 ± 0.0370 /
0.0779 ± 0.0253 versus intent's 0.1987 ± 0.0461 / 0.3482 ± 0.0674.
Do not compare E1 regret with E2 normalized regret directly. These are
single-seed results, with confidence intervals over evaluation cases.
See `artifacts/radiojev_cloud/README.md` for the full tables and provenance.

The resume runner is portable Bash and resolves the checkout from its own path:

```bash
bash artifacts/radiojev/run_all.sh --dry-run
bash artifacts/radiojev/run_all.sh
```

It skips the completed LoRA arm and smoke chain, skips completed stage reports,
preserves checkpoints, waits for competing jobs and stops on the first error.
An optional positional argument selects the competing-job `pgrep` pattern.
Read logs under `artifacts/radiojev/`; logs and data caches remain ignored.

**Full rerun:** the user authorized all training, including
regenerating the missing LoRA checkpoint. The isolated run directory is
`/workspace/training/radiojev/full`; inspect its `status.json`, per-stage logs
and `manifest.json`. All 330 E1/E2/E4-E5 scenes are prepared. Completed stages
are tracked in `status.json` and the consolidated warehouse results. At the
scope-reset checkpoint, E1–E3 were complete and E4/E5's local arm had finished;
the remaining comparisons were running. Those comparisons have now completed.
The four E1–E3 radio checkpoints were loaded
and checked for finite tensors. Completed reports and hashes are archived in
`artifacts/radiojev_cloud/`.
The model downloaded and loaded successfully after network access changed;
no Hugging Face token is required for the public model. The worker has no GPU.
The old Mac estimate of 1–2 days does not estimate cloud CPU runtime.

The original controller stopped when E4/E5 passed an E3 `heads` checkpoint to
the E2 `encoder` warm-start loader. `Bank` now initialises the multitask model
without an E2 warm start when E3 exists, then restores all E3 heads, LoRA and
20 readout rows. Three regression cases cover E3 with/without an older E2
checkpoint and the E2-only fallback.
The failed attempt was preserved under the run directory's `attempts/` and
the remaining stage resumed with the versioned `scripts/train_radiojev.py`.
Its source hashes and predecessor checkpoint are recorded separately in
`artifacts/radiojev_cloud/e45_resume_provenance.json`.

Do not restart a healthy job. Future full runs can use the versioned
`scripts/train_radiojev.py --run-dir <new-or-stopped-run-directory>`.
Its lock prevents duplicate controllers. Completed scenes/stages are retained;
an interrupted arm restarts because optimizer/epoch continuation is not
implemented. Results in the new run directory do not overwrite the published
reports until explicitly validated/promoted.

The independent `scripts/finalize_radiojev.py --run-dir <run> --watch` validates
all nine completed controller stages, full evaluation counts, finite report
metrics, E4 signed logs, checkpoint tensors and recorded training-source hashes.
It preserves archived raw reports, generates `docs/WAREHOUSE_RESULTS.md`, saves
an exportable cloud snapshot and rebuilds Studio. It never starts or changes
training. With `--publish-branch codex/e7-censoring-training-progress`, it commits
and pushes only validated generated evidence on that existing branch, provided
the checkout is clean. It never merges. Inspect `finalization_status.json` and
`finalizer.log` under the run directory. A stopped worker or failed validation
is reported as failure rather than completion.

E3 association normalized regret improved from 0.1170 to 0.1046. DSA greedy
reward is 1.000 in one 500-step evaluation episode; MLP baselines reach
0.263 / 0.384. Traffic accuracy falls below the rule baselines on all three
questions. Keep these mixed results separate; one episode does not establish
cross-layout DSA robustness. The admission label audit found only "lower"
labels in all 106 training and 54 test cases; 100% accuracy matches a constant
answer and does not establish admission-policy generalisation. The audit is
retained in `artifacts/radiojev_cloud/e3_label_audit.json`, reproduced by
`docs/experiments/e3_label_audit.py`. A separately versioned admission dataset
with meaningful class coverage is needed before a learning claim.
Independent training seeds remain necessary for stronger paper claims.
The router design in `docs/ROUTER_EXPERIMENT.md` is deferred.

E4's local / FedAvg / personalised-swarm mean normalized regrets are
0.1820 / 0.1781 / 0.1820. All three shared merges were rejected, so the shared
model accepted no E4 site contributions. E5 cannot establish removal of an
accepted contribution from that shared model. Its retained-data distillation
also increased attack AUC (0.6167 → 0.6489) and retained regret
(0.1801 → 0.2068). Preserve the negative result; do not claim validated
unlearning. Full per-site comparisons and the retraining baseline are in
`docs/WAREHOUSE_RESULTS.md`.

## Studio and paper

The exporter and Evidence page now include E1 status and E7 reports, with
historical/corrected pipeline labels. Rebuild after completing any new run:

```bash
uv run --frozen python -m wirelessjev.studio_data
python3 studio/build_artifact.py
```

The distributable is `studio/dist/wirelessjev-studio.html` (ignored by Git).
Use `studio_data --run-dir /workspace/training/radiojev/full` to include the
active rerun's completed tests and epoch-validation snapshot. It keeps those
results separate from published benchmarks and does not claim live telemetry.
The published Studio at
<https://claude.ai/artifact/NiohoyLYJsy5vEKE5BBswC> needs a separate update by
someone with access to that artifact. This environment has no tool to update
that Claude artifact; rebuilding locally does not republish it. The old update
watch is not active. README and the paper proposal describe the current limits.

## Hardware and remaining research

- Run `docs/ONE_CAR_TEST.md` on actual hardware: preflight, read-only status,
  capability probe, scan, proposal, supervised switch and switch-back. Only
  fake-car checks have run. Change the factory SSH password on the car.
- Switch-stage timestamps (association/authentication/DHCP/first packet) remain
  proposed. No physical car, radio or credential was changed here.
- PHY realism with Sionna, route-aware rate/width/band/link adaptation, and E8
  sensed inputs remain proposals awaiting a separate implementation decision.
- Further Wi-Fi 7, joint motion/radio, cross-layer, safety/task-assignment,
  Bluetooth, spectrum-image and embodied-AI ideas remain unstarted. CSI/SDR/
  openwifi/5G work needs hardware. Multi-car trials need the other nine cars.
- The didjital R6.1 branch is restored at `/workspace/didjital-r6.1`, commit
  `a458a90`. Its root, reference and R6.1 software suites passed (3 + 67 + 117
  tests, no skips), plus a frozen synthetic demo and native gate. See
  `docs/R6_COMPATIBILITY.md`. A real runtime bridge, private SmartCar source
  audit and hardware integration remain outstanding.
- The earlier token-expiry estimate and `refs/backup/sdth/` were Mac-specific
  handoff notes, not verified cloud state. Recheck them on that machine. GitHub
  reads here work through the platform's existing authentication.
