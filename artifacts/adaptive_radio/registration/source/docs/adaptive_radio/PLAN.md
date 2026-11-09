# Adaptive physical RF-condition study

This finite experiment tests whether sensed RF helps a physics-structured model
adapt when public geometry stays the same. It preserves the completed spectral
study (475 source files and 5,161 evidence files) and the earlier 452/864 closure.
The original 24 robot-SERVICE confirmation geometries remain unopened.

The evidence here concerns simulated static, isolated RF field probes. It does
not establish packet deadlines, interference/MAC performance, network delivery,
real receiver calibration, fleet safety, or generality to arbitrary robots and
environments. A separately registered native-contract service development probe
is documented in [SERVICE.md](SERVICE.md); field error is not packet risk.

## Physical task and permitted information

There are 24 TRAIN, six VAL, four DEV and 24 conditional TEST geometries, each
with three RF conditions: metal, wood or brick obstacle boundary response.
All three conditions retain identical CAD vertices, AP/client coordinates,
known transmitter power and RT sampling seed. Changing material changes the
actual Sionna propagation solve; no synthetic channel is substituted. Every new
CAD is disjoint from the prior 120 registered geometries and the other splits.
The geometry, rather than its three conditions, is the independent unit.

Known transmitter power cycles across layouts through −20, −10, 0 and 10 dBm;
it is constant within a layout's conditions. Device sensitivity/noise-floor
thresholds create actual censored observations from the rendered gain. The raw
reference-20-dBm CIR/CFR, material settings, public reports, transforms and
receipts remain available for passive audit. A missing solved path remains
unknown; it never becomes an invented low-power point measurement.

The learner receives public CAD coordinates and genuine reported observations.
Material, condition/category IDs, source filenames, private channel gain/CFR
and true receiver offsets never enter model features. Relative front-end
calibration uses paired TRAIN context reports in the device-0 gauge.

Both context views use observed bands 2.4/5.2 GHz at six even client poses.
View A uses direction 0/device 0; B uses direction 1/device 1. Sparse contexts
retain the same two planned poses, 0 and 10, across every RF condition.
This is a planned probe-report availability assay with eight scheduled tokens
per view, not a claimed network delivery or timing simulation. Main queries
use the six odd poses and known devices 0–2. The supplementary 6 GHz assay
queries the same even physical links; 6 GHz never enters fitting or the gate.

## Models and fixed training budget

Twelve independently trained models compare four cells with seeds 0, 1 and 2:

- `joint`: an eight-coefficient approximate Gaussian physical-field posterior,
  shared geometry/set-attention prior, censored likelihood updates, and
  uncertainty-weighted agreement between simultaneous A/B posterior means.
- `regression`: exactly the same model and training recipe, with agreement off.
- `rf_erased`: erases amplitude columns 0/1, retaining heard/censored status
  columns 2/3 and reported availability. This is a strong RF-value-erased
  comparison, not a strictly CAD-only control.
- `pure_geometry`: erases all four RF/status columns and uses the planned
  observation schedule as its mask, retaining public geometry/known metadata.

The decoder uses physical coefficients and public query geometry. Its local
correction is bounded, geometry-only, and cannot forward raw RF tokens. The
posterior/spread is an approximate static RF field forecast; it is not an
identified-material posterior or packet-risk head. There is no extra marginal
latent-prior penalty or post-DEV architecture search.

Every fit uses 60 epochs, eight updates per epoch, AdamW at learning rate 0.003
and weight decay 0.0001, width 64, and the same checkpoint rule. Every update
uses all 24 distinct training layouts, sampling one of the three conditions
uniformly per layout. The seeded condition stream is matched across cells;
update parity alternates full and sparse schedules, four each per epoch.
Thus there are 480 updates per member, rather than treating 72 conditions as
72 independent buildings.

Sparse VAL mean within-geometry condition-delta RMSE selects each checkpoint,
with the first epoch breaking exact ties. Only `joint` versus `regression`
participate in the subsequent branch choice: average the three member VAL
scores and choose the lower score, with `joint` breaking an exact tie. This
choice is frozen before DEV. No test data or inference ensemble is used.

Native RadioMap-operator and pooled Bayesian-operator controls use the same
permitted A-context observations, known TX and TRAIN-fitted priors. Literal
frozen old `joint_none` weights provide three supplementary transfer members;
they are neither retrained nor recalibrated for this task. An unsupported
frozen input produces an explicit abstention and denominator, never substitute
RF data or a change to a primary model's cohort.

## Endpoints and practical gate

Condition pairs are fixed at 0→1 and 0→2 at identical query indices. Delta
error compares predicted and observed reported-RSSI changes only where both
reports are heard. Pool their squared errors within a geometry and take the
root; each pair must retain at least one common heard query for the geometry
to be usable. All three case records remain, including unavailable outcomes.
Pair coverage records both-heard, one-heard, both-censored and unknown cases.
Censored values receive likelihood scoring, never invented point labels.

Scalar RMSE and within-geometry 95th-percentile absolute error pool heard
queries across all three conditions. Detector Brier scoring pools all known
heard/censored events, using the explicit detection threshold available on
both types of report. Heard quantization likelihood intersects each bin with
the above-detection event; the censored event uses the threshold CDF. Detection
thresholds are scoring labels and cannot enter model inputs. Missing physical
paths remain excluded and counted.

The primary regime is sparse. The chosen branch must pass every DEV condition:

- At least three usable geometries, actual censored query events, and finite
  primary forecasts and scores.
- Mean condition-delta RMSE at least 5% below both RF-value-erased and RadioMap
  controls; every trained member's delta RMSE no worse than RadioMap.
- Scalar RMSE and p95 absolute error no more than 2% above RadioMap, an explicit
  engineering noninferiority margin rather than a statistical safety claim.
- Holding condition-0 RF context stale for conditions 1/2 must worsen delta
  RMSE by at least 5%. Public query geometry/known TX stay current, and
  condition 0 itself remains unchanged.

A veto is a completed outcome; it cannot trigger automatic retuning, new
training seeds, replacement geometries or the opening of TEST. Full contexts,
6 GHz, pure geometry and frozen transfer remain descriptive conditions.

## Conditional TEST and claims

Four fixed two-sided comparisons use sparse inputs:

1. VAL-selected branch versus RF-value-erased condition-delta RMSE.
2. VAL-selected branch versus RadioMap condition-delta RMSE.
3. Fixed `joint` versus fixed `regression` condition-delta RMSE, regardless of
   which branch VAL selected.
4. VAL-selected branch versus RadioMap detector Brier score.

Each trained member's metric is computed first, then the three metrics are
averaged within a whole geometry. Predictions are not ensembled. The paired
whole-geometry bootstrap uses 20,000 samples, seed 21749 and central 98.75%
intervals (quantiles 0.00625/0.99375) for the four-comparison budget. Conditions,
links, receiver views and training members do not inflate the independent
geometry count. These intervals are conditional on the three trained members;
training-seed spread is separate and not a posterior over retraining.

At least 12 of 24 TEST geometries must support the delta endpoint; otherwise
the study is inconclusive and supports no RF-superiority/alignment claim.
Every unavailable geometry retains its reason and cohort counts. RF adaptation
requires adjusted wins in comparisons 1 and 2; an alignment-specific benefit
requires the separate fixed joint-versus-regression comparison 3. A detector
benefit requires comparison 4. None of these permits original SERVICE
confirmation or production activation.

## Publication, cache and passive closure

Code, protocol, registry, execution order, source capsule and exact passing
preflight tests/stdout are remotely verified before any official RT or fit.
Stage receipts bind every raw case and all source/registry identities. The
previous two sealed closures are checked before and after campaign phases.

The source-published gate cache is part of this initial implementation. Before
opening TEST it runs the full original numerical/raw DEV replay and binds its
permission, callback identities, source/capsule and all relevant dependency
bytes and membership in an immutable receipt. Every reuse rechecks those
bindings and returns a deep copy; SERVICE and production permissions must
remain false. It repeats the original full replay before final auditing and
freezes operational hit logging before the final byte snapshot. Restarts must
replay the original algorithm before reusing the same immutable permission.

The final passive auditor reconstructs permitted public inputs, targets,
thresholds, schedules and paired-condition cohorts, validates raw RT receipts
and learned tensors, and recomputes selection, gates, metrics and bootstrap
comparisons without neural constructors/forwards, baseline fitting or tracing.
A final closure detects changed source, evidence membership or bytes.
