# Robot 7: what the frozen context actually retained

The completed `dev-90473001-ho0.03` archive confirms a specific input problem:
the candidate discarded nearby **AP3 / 5.2 GHz** evidence while retaining two
reports from distant robots for that same target link. This is concrete
spatial-support loss. It does not prove that context selection alone caused the
AP4→AP3 failure, that eight coefficients are insufficient, or that JEPA
representational collapse occurred.

This is one passive pass over saved inputs, forecasts and delivered history.
No fitting, model forward, context reselection, simulation or counterfactual
episode was run. The source script and full row-level evidence are in
[`artifacts/native_context_review`](../../artifacts/native_context_review/diagnosis.json).
All sealed **512 source / 56 evidence** hashes matched before and after.

## Retained versus available target-link reports

Indices below are zero-based. Distances use each forecast's public planned
robot-7 position at horizon zero. Heard values and censor bounds are kept
distinct: an upper bound is not an invented RSSI measurement.

| Snapshot | AP3/5.2 GHz tokens retained | Distances to robot 7 | Available target reports within 5 m / 10 m | Nearby target reports retained |
|---|---|---|---:|---:|
| Decision 0, issued at 0 s | Robot 3: −78.854 dBm heard; robot 12: −71.072 dBm heard |54.81 / 74.26 m |16 / 31 |0 |
| Decision 1, issued at 1 s | Robot 2: below −85.020 dBm; robot 6: below −77.695 dBm |53.73 / 27.83 m |15 / 32 |0 |

At decision 0, both retained target reports came from warm-history bin 55,
five seconds before issue. Robot 7's own latest target report was **below
−89.294 dBm**, 3.86 m from its issue-time query position, also five seconds old.
It was available and omitted. The full available prefix held 1,141 scan rows
for this target: 129 heard values and 1,012 upper bounds. The nearest available
row was 0.40 m away but 45 seconds old; this illustrates why proximity alone
would also be an incomplete selection rule.

At decision 1, robot 7's new target report had been measured at **0.35 s** and
delivered at **0.45 s**. It was available for this forecast, 0.65 seconds old
and only **0.617 m** from the query position. The AP3/5.2 GHz result was **below
−85.757 dBm**. That row was omitted for this target group. The retained robot-2
and robot-6 reports were measured at 0.20 and 0.50 s, delivered at 0.30 and
0.60 s, and were 0.80 and 0.50 seconds old at issue. The available prefix then
held 1,164 target reports: 129 heard values and 1,035 upper bounds.

The exclusion is **target-specific**. Robot 7 was not absent from the whole
context: it contributed five tokens at decision 0 and six at decision 1, for
other AP/band groups. None conveyed its AP3/5.2 GHz report. Both snapshots had
exactly 64 tokens: two for each of 16 APs × two source bands, with 32 tokens at
2.4 GHz and 32 at 5.2 GHz. No 6.5 GHz observation was encoded, as registered.
The JSON lists every selected robot, coordinate, age, AP, band and report state.

## Which evidence could affect the 1.50 s move

The initiating move consumed the **decision-0** forecast, ready at slot 150
(1.50 s). It therefore could not incorporate a report delivered after that
forecast was issued. At the public planned position interpolated to 1.50 s,
its two selected AP3/5.2 GHz reports were still 54.35 and 74.17 m away. This
interpolation uses only the saved public route, not a hidden RF field.

Decision 1 could access robot 7's delivered report but did not select it for
the target group. Its forecast was not ready until slot 230 (**2.30 s**), so
it cannot explain the initial decision-0 forecast or undo the 1.50 s move in
this trace. The two issues—initial spatial support and later report selection—
must not be collapsed into one timing claim.

The stored horizon-zero AP3 forecast was −87.205 dBm with 5.844 dB predicted
standard deviation at decision 0, then −84.044 dBm with 7.398 dB standard
deviation at decision 1. These predictions are at different positions and
times from the omitted reports, so their differences from censor bounds are
not direct same-link/time residuals. The earlier
[outlier diagnosis](../../artifacts/native_bridge_review/outlier_diagnosis.json)
separately establishes a 20.946 dB optimism relative to the archived native
**base** field; that base excludes instantaneous fading and blockage.

## Why selection loses support

The frozen selector receives the past history and public AP geometry, but no
query route or target-robot identity. For each AP/source-band pair it sorts by
newest history bin, heard before censored at equal bin, then ascending robot
index. Round-robin truncation to 64 rows leaves exactly two per group here.
It does not select by distance to the forecast query, preserve the querying
robot's report for every target, or balance local coverage and age.

Time bins and robot IDs are retained in audit `selected_rows`, not in the
20-coordinate model token. There is no explicit observation-age feature or
temporal positional embedding in this adapter. Post-warm packet times can be
reconciled from the delivery log; warm reports have only their one-second
history-bin ages in this diagnosis.

The Map comparison has a different input budget and adaptation mechanism.
Its first cold fit uses the full 60-bin native history and separate per-AP,
per-band spatial random-Fourier models. For AP3/5.2 GHz, that fitter's actual
heard/censor-thinning rules admit **129 heard + 261 censored rows**, with
nearest eligible support **1.01 m** away. Robot 7's omitted five-second-old
target bound is among its eligible censored rows. This is a source-grounded
eligibility count, not a new Map fit.

Map's later warm calls reuse the initial fit until its 20-second refit
interval. Although the decision-1 history contains the new robot-7 report,
it would be wrong to claim that the existing warm Map forecast had freshly
assimilated that report. The archived candidate and Map also have different
forecast paths and calibration assumptions, so this comparison is not a
controlled context-only ablation.

## What the eight-coordinate state establishes

The source defines the predicted mean as
`FSPL(q) + offset(q) + 10 φ(q)ᵀ β(history) + local_db(q)`.
All report-specific mean changes pass through one **eight-coordinate global
coefficient vector**. The local correction is a fixed learned function of
public query geometry, bounded to ±5 dB; it does not retrieve or update from
nearby measurements for that particular query. The stored 8×8 covariance
affects forecast uncertainty. These are regression statistics, not identified
materials, scatterers or semantic scene tokens.

Both archived coefficient vectors are present in the JSON; their Euclidean
change between decisions 0 and 1 is 2.066. They are not identical. Two saved
states cannot diagnose population-level collapse, determine the needed latent
dimension, or prove that the global bottleneck caused this outage. What is
established is that a target-specific nearby report can be absent from the
input, and that subsequent RF mean adaptation uses a global rather than
query-local history state.

There are substantial competing transfer differences. The source Sionna
TRAIN registry has 24 layouts, two APs, widths 15.137–36.918 m and depths
20.283–52.874 m. Native uses a 120×80×6 m hall, 16 APs and 100 robots, with
33 racks extruded from 2D CAD using declared height/depth approximations.
It is not RT geometry truth. Source TRAIN targets use 2.4/5.2 GHz; native's
6.5 GHz is a new extrapolation beyond the supplementary 6.0 GHz endpoint.
Known transmit powers are supplied to FSPL, but native zero device bias is an
assumption, not identified hardware calibration. The native protocol explicitly
has no packet/silence-risk calibration or production qualification.

These findings motivate a separately registered comparison of coverage/age-
aware retrieval and a query-local residual state against the frozen adapter,
with matched observations, timing and independent action safety. They do not
constitute such a comparison or establish that either change will repair the
failure.
