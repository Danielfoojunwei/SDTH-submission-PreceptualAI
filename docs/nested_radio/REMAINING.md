# What remains after the completed RF experiment

The [registered RF experiment](PLAN.md) is complete: 15 fits with 1,920 committed
updates each, 84 fresh actual Sionna RT cases, all four primary comparisons and
the first independent actual-data replay. The selected model's 2.637330 dB
sparse heard-RSSI RMSE improves on the strongest numerical reference's
2.710522 dB under the declared assay. All controls and descriptive harms remain
in [RESULTS.md](RESULTS.md). These findings justify further validation; they do
not qualify a production controller. RadioMap remains the application default,
original SERVICE holdouts remain closed and no hardware is actuated.

## Next finite experiment: transfer RF gains into warehouse decisions

1. Publish a separate protocol before execution. Fix exactly which validated
   checkpoints supply forecasts, numerical references, report budgets, carrier
   assumptions, episodes, seed count, endpoints and stop rules. New TEST or
   SERVICE outcomes must not select a checkpoint, guard or controller.
2. Wire the nested model into the existing causal report selector, paid FIFO
   forecast queue and typed forecast boundary. Preserve actual report-arrival
   clocks, source-band support checks, missing heads, command creation/delivery
   costs, command expiry and handover interruption. Test numerical round-trip
   parity and frozen-checkpoint identity before comparative runs.
3. Compare full-history RadioMap, matched-budget numerical models, global-eight,
   nested regression and selected nested joint under declared budgets. Report
   all robot episodes, command/deadline misses, long-silence tails, actual and
   charged forecast work, scans, switches and pending-command bounds. Retain
   matched information and deployment-default references as distinct comparisons.
4. Use enough independent warehouse episodes and training members for the
   planned uncertainty claim. The current three-member/24-geometry RF intervals
   are conditional on these members. The separate four-episode native assay
   cannot establish a service improvement for these new weights or 100 real cars.

## Explain the representation's benefits and weaknesses

The current nested-versus-unit-scale contrast changes initialization, learned
scales and effective-prior regularization together. A newly registered ablation
would separate these effects, compare fixed small scales, and hold training
budgets and the RMSE selection rule constant. Additional independent training
members and learning curves can test robustness; selecting epoch 80 or 120 is
not a convergence certificate. No change to the frozen experiment can answer
these questions retrospectively.

The joint-view contribution is only 0.002226 dB in the primary assay and loses
to nested regression on the held carrier and both AP-transfer RMSE summaries.
Its inference cost and practical service value need measurement. A simple
regression field remains a serious reference; neither a JEPA name nor an extra
objective establishes the preferred deployment architecture.

The learned eight geometry features have mean VAL participation rank about
1.056 in the selected cell; effective local sixteen about 3.759. These are
public-geometry feature diagnostics, not a collapse or physical-observability
certificate. All eight learned local feature gates are exactly zero in all six
selected nested regression/joint members; the active residual uses 5/5/6 fixed
Fourier channels per cell. The current gain does not show that learned geometry
features help. Pruning may reflect the declared regularizer; it is not proof of
whole-model collapse or a diagnosed cause.

Before adding a larger encoder or anti-collapse loss, test whether local feature
directions predict recoverable residual structure beyond fixed Fourier features.
The coefficient-prior RMS penalty depends on feature scale, so a separately
registered TRAIN/VAL-only feature-scale or function-prior ablation is a useful
candidate. Its final comparison requires fresh TEST geometry, not tuning on the
current results. Keep unused global kernels explicitly inactive and report
effective prior products, not just gate size.

## Dynamics, sensing and uncertainty

The static material-difference diagnostic slightly favors fixed hierarchy over
the selected model. Timed material changes, moving blockers, interference,
missing/delayed scans and devices revisiting places need separately collected
sequences with chronological information boundaries. Build an observation-driven
state transition or update rule only with measurable temporal targets. Predicted
RSSI must never be recycled as new measurement evidence. Evaluate long-horizon
drift, recovery after change and explicit unknown/OOD behavior.

RF means and approximate Laplace spreads do not provide qualified packet risk.
Evaluate censored/weak paths, heard and unheard uncertainty, device offsets,
source-to-target carrier transport, and tails on independent sites/devices.
Protocol-supervised service heads need real or separately validated PHY/retry,
traffic and queue evidence for their supported actions and horizons. Missing
latency or silence heads remain unavailable. No RF calibration receipt grants
independent admission or physical actuation authority.

## External and physical validation

The next external comparison needs a named specialist baseline, rights-cleared
data, matched task/information budgets and measured inference compute. The
current actual Sionna cases are simulated propagation with finite ray-search
settings; they establish neither a measured-site win nor specialist SOTA.
Existing complex-CSI and DINO/camera work are separate scoped readers/studies,
not a validated fused representation in this RSSI model.

The [one-car procedure](../ONE_CAR_TEST.md) still requires actual safe hardware
availability and an approved reachable route. Archive access/preflight,
read-only capability observations, scans, supervised switch and rollback
evidence. The cloud experiment has not supplied the unavailable private network
route or run that procedure. Real fleet tests, password maintenance and external
Studio/deck publication remain separate tasks with their own access needs.

Repository-side work should preserve all ten scientific closures, negative
evidence and the independent audit. Any new scientific phase gets a new source
publication, registration and evidence namespace. The completed RF phase must
not be rerun or rewritten to strengthen its outcome.
