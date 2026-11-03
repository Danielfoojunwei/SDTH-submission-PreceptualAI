# Packet-action JEPA and cumulative service loss

The primary question is whether JEPA reduces actual missed-command cost relative
to both RadioMap and the same supervised inference graph. The frozen protocol
is [artifacts/packet_value/protocol.json](../artifacts/packet_value/protocol.json).
An improvement in average RSSI reconstruction or a controller-only gain does
not answer that question.

## Exact native counterfactual labels

The native packet kernel captures the complete factual fleet state before an
action. It clones packet ages, per-robot counters, current and pending radio
associations, scan/report queues, local observations, forecast completion FIFO,
controller memory and packet RNG state. Branches retain the original complete
channel draws and release phase. A shorter branch uses a stop slot; changing
the original draw duration would change the random experiment.

One focal robot either stays or proposes a switch with the same one-second
expiry as runtime, then holds its association for the ten-second label window.
Other robots continue the same declared controller. The label is its actual
post-fork missed-command difference, including packets already pending at the
fork, plus the branch-window maximum-silence difference. Shared prefixes and
random draws support paired outcomes. Future branch observations provide the
JEPA latent target only during training; hidden channels never enter inference
features. Factual suffix replay and action-intervention checks precede use of
these labels.

Training uses 16 sites with robot-count cycle 24, 24, 100, 100, crossing both
densities with stationary and drifting conditions. Four validation sites have
100 robots. Eighty-second factual trajectories provide forks at seconds 5 and
65, covering early and mature delivered-history states. Training selects two robots with the
lowest observable current margins and two other uniformly sampled available
robots. Validation samples naturally. Up to three candidates come from actual
completed local scans: the two strongest and one other visible AP. Unavailable
robots or missing candidates remain missing.

Before fitting, the diagnostic publishes beneficial, harmful and tied actions
at both physical handover costs. If either training cost has no beneficial
action, the campaign stops and reports insufficient decision evidence. An
all-stay head on all-zero labels cannot qualify as learned decision success.

An excluded pre-fitting pilot is preserved under
`artifacts/packet_value/pilot_ttl10/`: 30 completed label cases, 1,021 recorded
native runs and 691 action examples. Its forced fast actions could wait ten
seconds for a control delivery, whereas runtime proposals expire after one
second. Its alternating 24/100 robot counts also correlated density perfectly
with drift parity, and its forks at seconds 5/15 sampled only early history.
The explicit pre-fitting protocol amendment corrects all three issues before
development or holdout access. Pilot sources and data stay unchanged, remain
auditable and contribute no final training or calibration examples. The report
counts this excluded work separately from the corrected corpus.

## Matched model and controller

JEPA and supervised models share an identical inference graph, observable
history, action query, packet/gap regression heads and three matched random
initialisations. JEPA alone adds a masked EMA future-latent objective and a
variance-collapse regulariser during training. Validation selects each seed's
checkpoint; every trained seed is evaluated. Existing RadioMap route forecasts
provide public physics assistance, so this is an assisted packet-action model
rather than a standalone-map replacement claim.

The common controller compares corrected switch-minus-stay packet costs.
Candidate links must appear in completed local scans and pass the same
immediate-margin checks. Admission requires more than one predicted command
of improvement. Both stay and switch choices commit for ten seconds, matching
the label continuation policy. Fast local link/MCS control continues.

Each executed handover books its nominal physical interruption and accepted
predicted reservation once. The ledger cannot recover a lost packet or bound
the unknown factual RadioMap trajectory after actions diverge. Validation-only
95th-percentile residual corrections are empirical heuristics, not certified
conditional or sequential risk bounds.

Full RadioMap forecast costs and action-head CPU service are sealed before
development. Matched neural heads receive the same conservative measured
charge. Separate FIFO completion gates prevent forecasts or local decisions
from becoming visible early; stale origin associations, scans and outages
invalidate action execution. Proposals and pending cadence moves expire one
second after their original issue. Actual scan completion and successful control
exchange determine report delivery. Positions/routes remain out-of-band public
inputs and initial history is a shared conditional warm start.

## Independent confirmation and acceptance

Four new development sites run for 120 seconds. Eight untouched confirmation
sites run for 300 seconds with 100 robots, both assumed 30 ms and four-second
handovers, 90 ms scans and five-second scan cadence. Each site/cost evaluates
all three JEPA seeds, all three supervised seeds, common-controller RadioMap
and canonical RadioMap: 64 development and 128 confirmation runs.

The four primary contrasts are terminal cumulative missed-command rates for
JEPA minus common-controller RadioMap and JEPA minus supervised learning, at
both physical costs. A 20,000-replicate bootstrap independently resamples sites
and paired training seeds. Every primary upper 95% endpoint must be below
zero. This conjunction prevents selecting one favorable seed, comparator or
handover assumption. Robot packets are not independent experimental units.

A stricter acceptance supplement was committed as `352c093` before any model
fitting, development or confirmation access. The four original primaries stay
unchanged. Broad promotion additionally requires JEPA to beat the canonical
RadioMap controller at both physical costs, using the same crossed-bootstrap
upper-endpoint criterion. These two comparisons are explicitly additional
promotion requirements. A win over the new shared-controller numerical
reference alone cannot support a broad better-than-RadioMap claim. The audit
checks the committed supplement blob and its timestamp against the fitting,
development and test source receipts.

Development can reject a controller before the independent confirmation stage.
In that case all completed development cases and fitted models remain archived,
the report marks the original primaries **NOT EVALUATED**, and no confirmation
interval is calculated. A development rejection must not be described as a
failed independent confirmation. The original frozen controller is not repaired
in place. Any rescue uses a separate versioned controller, development-only
selection and a new pre-holdout seal. Unknown or rejected learned actions must
not silently replace the canonical controller's recovery behavior with a long
physical stay; that coupling is itself a hypothesis requiring an ablation.

The report also retains each trained seed and cumulative excess trajectories
at 60, 120, 180, 240 and 300 seconds, maximum observed positive excess,
final-third excess rate, worst-decile robot loss, global maximum silence and
p99 of per-robot maximum silence. A harmful primary interval or worse observed
global maximum silence or mean worst-decile service against common-controller
RadioMap, canonical RadioMap or the matched supervised reference vetoes broad
research promotion. All six comparison requirements and the expanded tail
veto must pass. Physical deployment remains
unqualified even if this finite research criterion passes.

Absolute loss and excess loss are different quantities. With one radio, a
90 ms scan spans an entire 50 ms command-deadline window and forces about one
miss per isolated scan. At five-second cadence this is approximately a 1%
loss floor. A complete four-second interruption forces 79–80 missed commands.
Overlapping interruptions count once. No fixed floor is subtracted from the
results, and no predictor can undo a command's already-expired deadline.

These tests cannot guarantee that realized excess never grows or that JEPA
wins universally. The fixed CAD family, simulated PHY, precomputed motion,
shared warm start, telemetry assumptions, controller continuation policy and
limited fitted seeds constrain transfer. Source capsules, checkpoints, branch
receipts, packet trajectories, exact label arithmetic and independent audits
preserve both positive and negative outcomes.

Generate and verify the completed report with
`uv run --frozen python scripts/report_packet_value.py` and the same command
with `--audit`. Results are in [PACKET_VALUE_RESULTS.md](PACKET_VALUE_RESULTS.md).
