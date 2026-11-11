# Registered RF and warehouse results

**Complete and independently reviewed:** all 186 actual Sionna RT cases,
15 neural fits and 34 warehouse simulations are complete. The canonical seal
and first independent actual-data audit pass. The retained
[field report](../../artifacts/hierarchical_radio/field_report.json) and
[native report](../../artifacts/hierarchical_radio/native/report.json) supply
these numbers. The [interactive results viewer](../../artifacts/hierarchical_radio_review/viewer/index.html)
shows the fields, comparisons and every native episode.

The selected hierarchical model **does not improve the primary RF endpoint**.
Its sparse heard-RSSI RMSE is **3.088402 dB**, versus **2.995352 dB** for the
parameter-matched global model and **2.951064 dB** for the selected numerical
map. Both predeclared adjusted intervals support higher error for the selected
hierarchy. Predictive agreement improves its matched regression control by a
small amount, and retaining RF amplitudes clearly matters; neither result
establishes superiority over the stronger references.

## Fixed experiment and selection

The published [protocol](PLAN.md) fixes 24 TRAIN, six validation, four technical
DEV and 24 TEST geometries, plus four eight-AP transfer geometries. Each has
three static material states, giving 186 cases. These are ray-traced physical
fields with simulated receiver reports, not measured radios or temporal
episodes. Source carriers are 2.4/5.2 GHz; 6.0 GHz is held out. Full/sparse
contexts contain 192/96 source tokens in total.

Five cells each completed three seeded fits, with 480 committed optimizer
updates per member. Each allocates 1,546 trainable parameters, but the main
four-AP hierarchy infers 72 scene coefficients (`8 + 4 × 16`), versus 24 in
the global-24 control; equal parameter counts do not mean equal state capacity.
All 15 selected epoch 40, the final registered checkpoint;
all fixed validation checks at epochs 10/20/30/40 decrease. This establishes
selection at the budget boundary, not convergence or the cause of the losses.
No fit was extended or reselected after TEST.

The [locked selection](../../artifacts/hierarchical_radio/selection.json)
chooses `hier_reg`: its mean sparse validation interval NLL is 1.321477,
versus 1.322214 for `hier_joint`. The rule chooses between these two hierarchy
branches using all three members, rather than choosing the best TEST method.
The numerical map selects its predeclared 20 m spatial scale by validation
NLL. Its [selection receipt](../../artifacts/hierarchical_radio/numerical/selection.json)
retains all three candidates.
This numerical map is the registered RadioMap operator, rather than a
bit-identical implementation of full-history native RadioMap. Both numerical
interval EM/Laplace and learned interval Newton/Laplace are approximate.

Each member's metric is computed first; member metrics and three states are
then averaged within each TEST geometry. The primary bootstrap resamples 24
whole geometries 20,000 times. Central 98.75% intervals account for the four
predeclared comparisons. They are conditional on the three trained members;
queries, APs, states and training seeds are not independent sample units.
RMSE scores heard reported-RSSI targets. Censored events enter the composite
likelihood; their unknown amplitudes do not become point-error targets.

## All four primary contrasts

Differences are candidate minus reference RMSE; lower is better.

| Candidate | Reference | Difference, dB | Adjusted 98.75% interval, dB | Result |
| --- | --- | ---: | --- | --- |
| Selected `hier_reg` | Global 24-state | +0.093050 | [+0.030280, +0.162483] | Hierarchy has higher error. |
| Selected `hier_reg` | Numerical map | +0.137338 | [+0.049191, +0.230937] | Hierarchy has higher error. |
| Fixed `hier_joint` | Fixed `hier_reg` | −0.034739 | [−0.052060, −0.020327] | Small agreement benefit. |
| Fixed `hier_reg` | Value-erased hierarchy | −2.705794 | [−3.986364, −1.510921] | RF amplitudes improve this model. |

The selected hierarchy's mean harm against both main references occurs in all
three training-member contrasts. The joint-versus-regression mean difference
is negative for all three members. These are descriptive consistency checks,
not a separate increase in independent sample size. The joint cell remains
worse than the numerical map in mean sparse RMSE; it is not substituted for
the validation-selected branch after seeing TEST.

## Controls, context density and material changes

Every number is an equal-geometry mean. Only the four comparisons above carry
the registered adjusted primary inference. The material-change endpoint uses
heard targets common to the base and changed state, and remains descriptive.

| Model | Sparse RMSE, dB | Full RMSE, dB | Localized material-delta RMSE, dB |
| --- | ---: | ---: | ---: |
| Global eight-state control | 2.809131 | 2.783939 | 0.933923 |
| Global 24-state control | 2.995352 | 2.839783 | 1.008329 |
| Selected hierarchical regression | 3.088402 | 2.979055 | 0.965668 |
| Hierarchy with joint agreement | 3.053663 | 2.955278 | 0.962421 |
| Value-erased hierarchy | 5.794196 | 5.681698 | 1.234620 |
| Fixed-feature numerical hierarchy | 2.859271 | 2.861883 | 0.909541 |
| Selected numerical map | 2.951064 | 2.875992 | 0.953367 |

The small global control and fixed-feature hierarchy are strong. Adding
AP-local inferred state and learned features did not establish a better field
model under this budget. The hierarchy's lower material-delta error than the
global-24 model is a limited descriptive pattern: the numerical map, fixed
hierarchy and global-eight control all have lower delta error still. Static
material deltas do not validate learned transitions or stable long rollouts.

The [source-backed interpretation](INTERPRETATION.md) identifies a structural
constraint: the local coefficient prior has a 2 dB SD floor and fixed nonzero
Fourier features, so this hierarchy cannot shrink its local prior channel to
the exact global-eight submodel. This is not a 2 dB floor on predictive spread,
nor evidence that the constraint caused the losses. A zero/off residual branch
and shared physical factors are hypotheses for a new protocol, not a rescue
claim for these results.

The selected hierarchy's sparse p95 absolute error is 5.851632 dB, versus
5.653562 for global-24 and 5.680276 for the numerical map. Its empirical
heard-target 95% coverage is 95.68%, versus 94.81% and 95.66%, with all-query
mean predictive spreads of 3.536572, 3.232541 and 3.286226 dB respectively.
Coverage on selectively heard targets and approximate field spread do not
qualify packet risk, calibrated tails or arbitrary drift.

## All registered transfer groups

These secondary RMSE values are descriptive. Eight-AP transfer has four fresh
geometries and retains the same total token cap, reducing reports per AP.
Stale contexts use state-zero reports; state-zero targets are unchanged
anchors. Held-band and stale groups contain only the predeclared selected
hierarchy, global-24 and numerical references.

| Group | Selected hierarchy | Global 24-state | Numerical map | Fixed hierarchy |
| --- | ---: | ---: | ---: | ---: |
| Held 6.0 GHz, sparse | 3.136908 | 3.055472 | 3.017447 | 2.876906 |
| State-zero stale context, sparse | 3.190986 | 3.102640 | 3.016531 | 2.942190 |
| Eight APs, sparse | 2.554502 | 2.474529 | 2.505165 | 2.380558 |
| Eight APs, full | 2.507110 | 2.294206 | 2.417201 | 2.371106 |

The implementation accepts a different AP count without new learned AP IDs.
These groups do not rescue the primary hierarchy-superiority claim. Fresh
context has lower mean error than stale context for each displayed method;
that is a static observation-conditioning comparison, not temporal learning.
The complete report retains the supplementary global-eight, joint and erased
eight-AP results rather than selecting a new model from them.

## Warehouse control outcomes

All 32 fresh runs and two numerical round-trip controls are complete. Exact
forecast, spread, completion and packet/outcome parity passes for the wiring
pair. This is a wiring check, not scientific improvement. The warehouse model
is the fixed validation-selected seed-zero hierarchy; the four fresh
procedural episodes per handover cost do not replicate all three trained
members or four measured buildings. The following outcomes remain descriptive.

Every row is a four-episode mean. Maximum silence is the mean of each episode's
largest per-robot gap, not a bound on the fleet. The p99 is also computed within
each episode before averaging.

| Handover, s | Forecaster | Command miss, % | Worst-decile miss, % | p99 gap, ms | Mean episode maximum silence, ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.03 | Full-history RadioMap | 1.165000 | 2.556250 | 545.875 | 1,127.5 |
| 0.03 | Matched numerical map | 1.600000 | 6.431250 | 1,551.600 | 2,452.5 |
| 0.03 | Selected hierarchy | 1.278125 | 3.806250 | 1,051.925 | 2,480.0 |
| 0.03 | Global 24-state | 1.265000 | 3.656250 | 290.625 | 2,580.0 |
| 4 | Full-history RadioMap | 2.372500 | 11.425000 | 2,186.250 | 3,300.0 |
| 4 | Matched numerical map | 2.817500 | 15.793750 | 2,933.125 | 5,222.5 |
| 4 | Selected hierarchy | 2.778125 | 14.900000 | 3,300.275 | 3,575.0 |
| 4 | Global 24-state | 2.659375 | 14.093750 | 2,580.200 | 3,590.0 |

The hierarchy has lower mean misses than the matched numerical map in both
handover conditions, but higher mean misses than global-24 and full-history
RadioMap. Lower mean maximum silence than global-24 coexists with more misses.
The hierarchy and both matched controls each retain an **8.89 s** gap in seed
90512002 under both handover conditions. Full-history RadioMap's corresponding
gaps are 3.51/7.73 s. These results do not establish dependable connectivity for
100 physical cars.

All registered paired contrasts retain harms. Positive differences mean more
misses or longer silence for the candidate; percentage-point changes must not
be read as relative percentages.

| Handover, s | Candidate versus reference | Mean miss change, percentage points | Mean maximum-silence change, ms | Episodes with more misses, /4 | Episodes with longer maximum silence, /4 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.03 | Hierarchy / matched numerical | −0.321875 | +27.5 | 0 | 1 |
| 0.03 | Hierarchy / global-24 | +0.013125 | −100.0 | 3 | 0 |
| 0.03 | Hierarchy / full-history Map | +0.113125 | +1,352.5 | 4 | 2 |
| 0.03 | Matched numerical / full-history Map | +0.435000 | +1,325.0 | 4 | 2 |
| 4 | Hierarchy / matched numerical | −0.039375 | −1,647.5 | 2 | 0 |
| 4 | Hierarchy / global-24 | +0.118750 | −15.0 | 4 | 0 |
| 4 | Hierarchy / full-history Map | +0.405625 | +275.0 | 1 | 1 |
| 4 | Matched numerical / full-history Map | +0.445000 | +1,922.5 | 3 | 3 |

The slow hierarchy/full-history mean harm is concentrated: one episode has
691 extra misses, one has 42 fewer, and two tie. An average alone would hide
that pattern. Slow hierarchy/matched-numerical mean improvement still harms
two of four episodes on misses.

RF evaluation supplies learned and numerical predictors the same archived
view-A context. In the warehouse, the three matched methods share a 192-report
budget, selector and support rules; their actions can change subsequent scans
and delivered histories. Native context bytes are not held identical across
arms. Full-history native RadioMap is a separate stronger-information
reference. Actual compute, scans, transport, pending packets and both handover
costs remain in the complete native report.

## Paid work and end-of-horizon uncertainty

The table gives episode means. Forecast-work timings are sums of elapsed wall
time for 20 scheduled jobs, not process CPU consumption. Paid compute applies
the declared floors and actual overruns. Each job also carries 0.5 s shared
work: **10 s of scheduled shared service per episode is additional** to the
paid-compute column, including work not consumed before the horizon. These
queue charges are not an end-to-end or energy benchmark.

| Handover, s | Forecaster | Actual forecast work, s | Paid forecast compute, s | End-pending control packets | Observed + pending miss upper, % |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.03 | Full-history RadioMap | 1.398146 | 5.750000 | 2.00 | 1.170000 |
| 0.03 | Matched numerical map | 9.586610 | 10.124255 | 2.50 | 1.606250 |
| 0.03 | Selected hierarchy | 4.827286 | 6.058760 | 2.00 | 1.283125 |
| 0.03 | Global 24-state | 4.081605 | 5.895686 | 2.25 | 1.270625 |
| 4 | Full-history RadioMap | 1.373408 | 5.750000 | 3.75 | 2.381875 |
| 4 | Matched numerical map | 9.500389 | 10.038887 | 5.50 | 2.831250 |
| 4 | Selected hierarchy | 4.365969 | 5.844043 | 4.50 | 2.789375 |
| 4 | Global 24-state | 4.165126 | 5.894317 | 4.75 | 2.671250 |

The raw receipt calls this end-pending count `end_pending_commands`, but its
source is `spatial_execution.pending_control_packets`: pending control packets,
not the number of central command proposals. The upper fraction treats each
as an additional possible miss. It is an end-of-horizon accounting bound, not
a confidence bound, calibrated risk probability or SLA.

The hierarchy records less forecast work than the matched numerical map, while
full-history RadioMap is faster and has lower mean misses. Actual support
preparation is included in issue/completion accounting; it is not added twice
to forecast work. All methods retain approximately 397–400 fleet scans per
episode, around 35.7–36 summed seconds of radio interruption. Matched numerical
forecasts are consumed 19.5/19 times on average at fast/slow handovers; all
other methods consume 20. The slow hierarchy has two actual central-planning
overruns across four episodes. Video, bulk, scan, preparation, handover and
per-episode cost records remain in the complete native report.

## Integrity status

The [completed report](../../artifacts/hierarchical_radio/report.json) and
[canonical manifest](../../artifacts/hierarchical_radio/verification/manifest.json)
bind **567 source files and 4,900 evidence files**, with all 19,589 named
repository files from the eight prior closures preserved. All registered
scientific execution and canonical verification are complete.
The [first independent actual-data audit](../../artifacts/hierarchical_radio_review/independent.json)
passes: 24,513 unique named repository files, all 186 RT cases, 15 checkpoints,
60 raw validation archives, 4,416 field prediction records, four primary
intervals and all 34 native runs are checked. It performs zero scientific
solves, fits, model constructions, forwards or packet simulations; retained
source and evidence bytes are unchanged. No failed actual-audit attempt,
parser adjustment or scientific retry was needed.

The source preflight passes 438 software tests. The
[whole-repository receipt](../../artifacts/hierarchical_radio_review/software_suite_importlib.json)
records 3,416 passing tests across 149 modules, with 564 source/config/test
files unchanged. Software checks and the passive actual-data audit establish
different parts of verification. The published source, raw validation predictions,
checkpoints, reports and earlier evidence remain retained.
The [viewer browser receipt](../../artifacts/hierarchical_radio_review/viewer/browser_review.json)
passes on Chromium desktop and mobile, including 576 geometry/AP/band
combinations, all 32 individual native episodes and unchanged scientific inputs.

RadioMap remains the default, and original SERVICE holdouts stay closed.
This result does not diagnose collapse, prove optimization exhausted, identify
scatterers, establish classic JEPA or SOTA superiority, qualify production, or
complete the broader foundation-model program. The implemented architecture
and [remaining validation](REMAINING.md) define the narrower scope.
