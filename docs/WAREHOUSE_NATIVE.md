# Native packet supervision for warehouse decisions (v4)

This follow-up targets two limitations of v3: simplified-twin supervision did
not establish a native controller advantage, and the native audit observed only
eight seconds of a thirty-second decision. Historical reports stay unchanged.
The executable protocol and frozen source are in the run's `protocol.json` and
`training-protocol.json`. A running experiment is not a completed result.

## Question and common comparison

Can a small action-conditioned model trained on actual ns-3 packet outcomes
choose a better warehouse association plan than a strong public-map numerical
planner? This is a separately declared **known-map simulation**. It is not an
RF scan experiment or a physical 100-car trial.

Every environment has 100 simulated robots and 32 APs. Each decision commits
one association for the full thirty-second route. The twelve-switch limit is
an operator policy. A four-second application mute is charged once when an
assignment changes. This is an assumption, not simulated WPA/DHCP or a measured
handover distribution. Each counterfactual starts a separate simulator process;
the experiment does not establish persistent closed-loop roaming.

Inputs are public current positions, declared AP geometry/channel/power,
current associations, application demand and planned route. No twin scan,
previous twin service, hidden material state or future native telemetry enters
the model. Link eligibility comes from a declared log-distance map. It does not
prove an AP was measured or can be used by physical hardware.

The numerical baseline uses that same log-distance geometry, a seven-dB noise
figure and the executed NIST VHT80/HT20 two-stream curves. Approximate MAC sharing
remains a modelling limitation. The native simulator uses MinstrelHt, WMM,
bursty bulk traffic, endpoint timestamps, actual queues/retries and RTS/CTS.
RTS is fixed on for every arm, selected from the completed earlier audit rather
than this test set. Propagation still lacks calibrated racks/materials.

A larger numerical candidate search supplies a four-slot common shortlist:
stay, budgeted map signal, numerical service/critical objective and numerical
worst-decile service. Identical assignments are executed once with explicit
aliases. Numerical planning, learned models and the candidate oracle use the
same shortlist. The oracle is a retrospective upper bound within those
candidates, not an executable method or a global optimum.

## Verifiable reward

The predeclared objective is half mean delivered-demand service plus half mean
critical demand delivered within fifty milliseconds. Both are fractions of
nominal demand over the whole thirty-second window, including assumed muting.
No absent packet is invented to calculate a generated-packet miss fraction.
Critical flows are aggregated into one class, using the maximum declared
critical floor ratio; this does not measure each original flow separately.
Critical packets are 240 bytes in this simulation. Late arrivals do not count
as timely service. Arrivals after the decision window are excluded from this
window's service target.

Per-robot service, worst-decile service and critical-floor violations accompany
the fleet objective. A better average does not prove every robot improves.
The risk head predicts a specific native event: critical timely-demand fraction
below the caller's declared floor. Its temperature uses validation scenes only;
Brier/ECE are measured on test scenes. The numerical risk is an explicitly
uncalibrated 0.02/0.98 proxy. These probabilities do not describe
hardware safety or authentication success.

## Training and separation

The immutable splits are eight training scenes (190000–190007), three validation
scenes (191000–191002), and six new test scenes (192000–192005). The same building
and route generator are used. Robots within scenes are correlated observations.
There is one native RNG run per unique scene/assignment, not repeated hardware
fleets or repeated independent packet-realisation trials.

Three independent training seeds each fit a plain MLP and an AP/channel graph.
They predict per-robot native service and timely critical demand, with a
counterfactual advantage loss relative to staying. Both retain a numerical
prior. A training-fitted linear residual with validation-selected ridge penalty
is an additional specialist baseline. No pretrained foundation-model or
JEPA-pretraining advantage is implied by this experiment.

Models, normalisation, calibration, source hashes and default-method selection
are sealed before the test collector is allowed to run. Selection requires a
positive scene-bootstrap lower bound on utility improvement and no validation
scene worsening worst-decile or critical service. Otherwise the research
default remains numerical. Test checks are diagnostic; they cannot reopen
model selection. Crossed training-seed/scene intervals do not count robots as
independent trials. Six test scenes still make this an exploratory experiment.

## Running and recovery

The durable root is `/workspace/training/warehouse-v4-native`. Run collection
and supervision separately; each holds its own advisory lock:

```sh
.venv/bin/python scripts/collect_warehouse_native.py \
  --out /workspace/training/warehouse-v4-native --workers 4
.venv/bin/python scripts/run_warehouse_native.py \
  --out /workspace/training/warehouse-v4-native --workers 4
```

The supervisor waits for completed development collection, trains all six
models, freezes selection, collects fresh test outcomes, then writes
`results.json` and a hashed completion receipt. `status.json`,
`collection-development.json`, `collection-test.json`, and `progress-*.json`
show live state. Check PID/process state before restarting. Completed native
outputs are reused only when input, binary, protocol and output hashes match;
completed training requires matching checkpoint hashes and frozen sources.

Physical association, scan interruption, field RF calibration, hardware trials
and deployment qualification remain separate. No car command is generated by
this campaign. Both JEV implementations remain Wi-Fi-specific.
