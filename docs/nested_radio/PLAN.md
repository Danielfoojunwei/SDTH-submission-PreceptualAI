# Zero-capable residual and fixed-budget RF experiment

This finite follow-up tests the concrete restriction identified in the completed
shared/AP-local study. That model could not recover the exact global-eight
submodel by switching its local channel off. Its primary losses remain frozen.
The new experiment does not assume that a larger latent or a JEPA label wins.

The new model reuses the unchanged physical basis and bounded interval-Newton /
Laplace inference. Sixteen shared feature scales gate the local design in both
context conditioning and query rendering. Their legal range is [0,1], initialized
at 0.1 and projected after every optimizer step; forward requires valid finite
parameters. All-zero design recovers the global-eight function with the same
remaining weights, up to floating-point roundoff. All-one design recovers the
legacy hierarchy. This inclusion property is not learned noninferiority or an
exact match to independently trained global-eight weights.

The local prior retains its original positive SD floor and trainable mean/SD.
A gate-only penalty can be evaded by growing those scales, so the declared
regularizer is 0.01 times the mean of
`alpha * sqrt(local_prior_mean**2 + local_prior_SD**2) / 0.5`.
It measures gated coefficient-prior RMS, not query predictive error or SD.
Rescaling alpha and the prior cannot reduce that penalty while preserving the
same effective coefficient prior. Gates alone are not identified physical
factors; log their effective mean/SD/RMS products. A zero channel may be
stationary under optimization; reactivation is not guaranteed.

Five cells—global-eight regression, global-eight joint agreement, legacy
hierarchical regression, nested regression and nested joint agreement—receive
three independent training seeds each. Nested cells add sixteen trainable
gates, so they allocate 1,562 parameters versus 1,546 for controls. A four-AP
hierarchy still infers 72 coefficients versus eight for the small global model.
Nested-versus-legacy comparisons test the bundled gate/init/regularizer package,
not an isolated removal of the prior floor. Joint cells add the same fixed
0.1 variance-normalized predictive agreement to the same two-view supervised
interval/censor likelihood; there is no EMA teacher.

Reuse exactly the previous immutable 24 TRAIN / six validation geometries,
three states each, public input groups and TRAIN-only relative calibration.
No old TEST or DEV outcome selects weights or gates. Every fit commits 160
epochs × twelve updates = 1,920 updates, with four distinct layouts per batch,
one random state each, alternating full/sparse contexts. AdamW uses lr 0.003,
weight decay 1e-4 and norm clip 5. Three spawned CPU workers use one Torch/BLAS
thread each. Atomic optimizer/RNG snapshots preserve every committed update;
nonfinite failures do not trigger replacement seeds or numerical retries.

Archive weights and actual A-only validation predictions at epochs 40/80/120/160.
Each member chooses the first minimum equal-state/equal-layout sparse heard-RSSI
RMSE. Model selection takes the mean of all three members, with eligible order
(global-eight regression, global-eight joint, nested regression, nested joint)
for exact ties. Legacy hierarchy remains a control. This aligns selection with
the primary point metric; composite interval NLL remains a reported secondary
criterion. Neither epoch 160 nor the minimum certifies convergence.

Select the strongest numerical reference by the same validation RMSE from all
four frozen TRAIN-fit priors: RadioMap-operator 5/10/20 m and fixed-feature
hierarchy. No numerical prior is refitted. Retain all four references. The three
parent map validation archives are reused exactly; fixed-hierarchy validation
predictions are computed once after source publication. This is not a
bit-identical claim about the native full-history RadioMap implementation.

After all fifteen fits, epoch archives and validation reductions pass the
technical selection gate, collect 84 fresh actual Sionna RT cases: 24 TEST
geometries plus four eight-AP transfer geometries, each with three static
material states. Public geometry fingerprints must differ from all 240 earlier
layouts and each other. Materials and TX use independent streams; within a
geometry only one obstacle material changes in each counterfactual state.
Geometry, TX, solver seed and receiver-noise draws remain shared across states.
Use the unchanged 48 receiver poses, checkerboard context/query partition,
192/96 full/sparse source tokens, 2.4/5.2 GHz source carriers and held 6.0 GHz.
Eight-AP transfer preserves total token count. No query reporting cutoff or
private channel/material becomes a model input. No unheard AP becomes a fake
floor measurement: top-K bounds remain upper events and unresolved paths unknown.
The actual LLVM Sionna solver uses four DrJit threads, depth three, 32,768
samples/source and 4,096 path cap. Finite search is not claimed converged.

The four primary sparse heard-RSSI RMSE contrasts are selected model versus
best numerical reference; selected versus global-eight regression; nested
regression versus legacy hierarchy; and nested joint versus nested regression.
Compute member metrics first, then average members/states within each of 24
whole geometries. Use 20,000 paired whole-geometry bootstrap draws, seed 21771,
and four-family adjusted central 98.75% intervals. If selection chooses
exactly global-eight regression, its base comparison is an exact alias/null,
not evidence of residual improvement. Uncertainty is conditional on three
trained members, not arbitrary retraining.

Full context, localized material deltas, heard p95/coverage, held band,
eight-AP transfer and stale-state contexts are descriptive. All outcomes are
retained regardless of efficacy. Predeclared validation-only feature diagnostics
separate fixed Fourier eight, ungated learned geometry eight and effective gated
sixteen features; unused global-control kernel features are labeled inactive.
Their rank is not a deep RF-encoder, collapse or physical-identification
certificate. Point/interval uncertainty remains approximate and unqualified
for packet-risk decisions.

Publish complete source, protocol and case registry before any scientific fit,
new solve or diagnostic forward. Retain raw inputs, targets, predictions,
checkpoints, attempts and reductions, seal exact membership, then independently
replay raw arrays without scientific reruns. All previous studies stay frozen.
This experiment is RF-only: zero new warehouse simulations or hardware actions.
Packet transfer, real devices, event-time dynamics, supported action calibration
and specialist-SOTA comparisons remain separate requirements. RadioMap stays the
application default and original SERVICE holdouts remain closed.
