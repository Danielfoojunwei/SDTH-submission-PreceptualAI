# JEPA repair diagnosis: which predicted information reaches control?

**Repair the shared control/runtime path before introducing another model.** The compact forecasters improve absolute mean-field prediction substantially, but their route-change improvement is much smaller and context JEPA is effectively tied with matched supervised on these exposed validation inputs. This supports a targeted factorization hypothesis, not a conclusion that another JEPA target will fix delivery.

## Actual data and scope

This is a derived, post-hoc audit of **all 1,152 existing validation rows across six actual site IDs**, for the three frozen context-JEPA members and three matched supervised members. The same validation data selected their checkpoints. No new sites, worlds, model fitting, controller execution, RF perturbations or model selection occurred. Frozen source and input hashes were checked before and after forwards.

[The derived results](../artifacts/jepa_repair_diagnosis/architecture.json), [input/source receipt](../artifacts/jepa_repair_diagnosis/architecture.receipt.json) and [audit script](../scripts/diagnose_jepa_repair_architecture.py) retain every site, horizon, band and member. The existing validation cache remains unchanged.

The old Fleet JEPA has compatible tensor fields, so its three weights were also evaluated as an ensemble on the **new cached strong-map and feature pipeline**. This is explicitly a weight-transfer diagnostic. It is **not the old forecaster's genuine map prior/runtime/observation pipeline**, and must not replace its independently archived native-service results.

## Quantitative diagnosis

All quantities below are in dB; lower is better. Route error excludes the mathematically zero horizon-zero difference and covers one through five seconds. AP-pair gain error uses all ordered off-diagonal pairs of the native pre-cost score, `mean_h max_band SNR`. The cache has no actual current association or load state; these are score proxies, not actual switch gains or packet counterfactual values.

| Frozen forecaster | Absolute RMSE | Current/h0 RMSE | 1–5 s route-delta RMSE | AP-pair gain RMSE | Zero-load AP regret |
|---|---:|---:|---:|---:|---:|
| Strong RadioMap in cached inputs | 7.4439 | 7.2540 | 3.2317 | 9.0144 | 0.5692 |
| Compact context JEPA: mean of three members | 6.2502 | 5.8852 | 3.1789 | 7.3122 | 0.3724 |
| Compact supervised: mean of three members | 6.2537 | 5.8740 | 3.1785 | 7.3324 | 0.3729 |
| Old Fleet JEPA weights on NEW common inputs | 6.5040 | 6.2340 | 3.1964 | 7.4567 | 0.4100 |

Context JEPA improves absolute RMSE over the cached numerical prior by **16.04%**, but route-delta RMSE by only **1.63%**. Its route error is **3.1789 dB**, versus supervised's **3.1785 dB**. These tiny differences on exposed validation do not establish an objective advantage. All six validation sites show a modest context-JEPA route-error reduction relative to the cached map; this remains descriptive, not an independent confirmation.

The gains are not solely global common-mode calibration: context JEPA's remaining per-band AP-centered error is **5.7945 dB**, versus an AP-common error component of **2.3399 dB**. AP-relative structure still matters and remains imperfect. Absolute error, route error and actual packet loss are different endpoints.

Only **16.62%** of the local AP/band histories contain a heard entry in their last six bins. Bounded fleet memory and the map provide additional information, but absent local readings are not observed contrasts. This limits what a new latent target can identify.

No compact context-JEPA or supervised correction reaches `abs(residual) >= 11.9 dB` on this cache. Widening the 12 dB cap is therefore not the first observed saturation remedy. Some **10.92%** of true pointwise map residuals exceed 12 dB, but their hindsight representational floor is **1.7844 dB RMSE**, far below the models' approximately 6.25 dB error. This does not prove the cap is harmless; it rejects claiming active near-bound saturation as the demonstrated main bottleneck.

The slow admission controller integrates 30 seconds, but the common route-delta loss covers only the first six horizons and long-horizon signal weights fall to 0.5/0.3/0.2. At 30 seconds the three context members have **8.315–8.353 dB** route-delta error. Planned-versus-realized motion and hidden dynamics contribute; a different target cannot make unobservable futures deterministic. This is an additional objective/interface mismatch to test after the shared control repair, not a validated remedy.

The script also verifies the identity `future_error_MSE = current_error_MSE + increment_error_MSE + 2*cross_term`. The cross term is strongly negative. Absolute and increment errors therefore cannot be interpreted by subtracting RMSEs or assuming independent current/future errors.

## Why current-level accuracy can fail to improve control

The frozen native control-band, data-timing and MCS path forms

`near[h] = lagged_current_pilot + forecast[h] - forecast[0]`.

A time-constant error for the same AP/band cancels from that forecast difference. The current pilot is a lagged/noisy observable, not clean mean-field truth, so these metrics are not a perfect-service oracle. Nevertheless, the cancellation is an exact property of the decoder/controller interface. Better static calibration does not automatically improve this branch.

AP planning uses a route-averaged maximum-band score, while outage admission applies nonlinear failure curves to **absolute** SNR. Absolute levels must remain available. Removing a common shift separately in each band is not a complete controller symmetry: `max_band` does not commute with that centering, and physical SNR shifts affect admission and packet success.

Learned spread is not the current direct control bottleneck either: `reactive_rescue(ctx)` overwrites the planned confidence mask with `ctx.eligible`; the band-risk calculation uses a separate fixed 5 dB spread. Better uncertainty calibration alone would not repair those fixed-profile choices.

Two shared mechanisms deserve explicit control experiments before retraining:

- The fast `OutageBudgetGate` integrates a candidate only at horizons at least the handover cost. For 30 ms, its first retained horizon is one second; the missing 30 ms-to-one-second interval understates candidate failure exposure.
- A planned move rejected by separate handover admission can leave `confident=True`, while reactive recovery is gated on `~confident`. That suppresses recovery for reasons unrelated to the JEPA objective.

These are source-level observations, not outcome estimates for a repair. Existing studies, guards and negative outcomes must remain unchanged. A separately versioned controller/runtime with frozen old JEPA and RadioMap can isolate repair benefit using fresh sites and charged completion times.

## Conditional next architecture: current correction plus direct route increments

If a repaired shared controller still leaves useful predictive headroom, test

`forecast[h,s,b] = map[h,s,b] + current_bias[s,b] + route_innovation[h,s,b]`,

with `route_innovation[0] = 0` by construction. Compute `current_bias` once from current public history. Predict every future innovation directly from public route/query features and memory; do not recursively add prior model outputs. This separates the current-level gradient from the route-change gradient that reaches native band/MCS control.

A mere algebraic rewrite of the existing independent per-horizon outputs is not a new model. The architectural change must actually share the current correction and give the increment head a zero origin. Bounds must be specified honestly: independently bounding both heads to 12 dB permits a 24 dB total correction and cannot be advertised as retaining the old total bound.

A contrast-target JEPA could predict `e[h]-e[0]` and per-band centered AP residuals, where `e = truth - current_public_map`. Those use the same privileged simulator labels as supervised controls. Do not force all future corrections to be centered: a real future common-mode decline remains important to absolute admission risk.

An EMA encoder also needs consistent physical channels. Past slopes in dB/second cannot silently share a channel with future increments in dB. Capture timestamps, masks, ages and a defined AP reference are needed; delivery-bin time and differing heard subsets do not establish a physical relative measurement. Unknown AP contrasts remain beliefs, not invented observations.

The current common loss already supervises route changes and centered AP scores. New contrast coordinates add no information. Their possible value is gradient routing, parameter sharing or optimization, and must be tested against a matched factorized supervised model and direct-contrast auxiliary control. Retain a matched future-context JEPA when testing whether an RF-only target helps. Existing absolute-semantic JEPA did not win; that negative result remains relevant.

No new model implementation is justified by this validation audit alone. Use the control/runtime repair study first, then preregister any factorized/contrast objective study on fresh development and confirmation sites. No EMA-necessity, novelty, SOTA, universal transfer or indefinite cumulative-loss guarantee follows.
