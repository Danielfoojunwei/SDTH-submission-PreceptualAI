# Real complex-CSI completion: completed results

The registered experiment completed all six neural fits, three numerical
references and its raw audit. **The joint objective did not provide a useful
gain over its matched supervised control or the strongest simple reference.**
This is a negative completion result, not a successful physical radio world
model, a service qualification or a SOTA result.

## Results

Lower packet NMSE is better. NMSE 1 is the error from predicting zero channel
on every target bin. Neural entries are the mean ±sample SD across three
independently trained seeds 17/43/89; deterministic references have one fixed
fit/configuration. Both 16-pilot views are averaged. Packet quantiles and
antenna observations are not independent-trial confidence intervals.

| Recording domain | Joint Fourier-field consistency | Matched supervised control | Complex linear interpolation | Fourier ridge | TRAIN LMMSE |
|---|---:|---:|---:|---:|---:|
| cf06 spatial test, 155 packets ×32 antennas |1.08333 ±0.00090 |1.08349 ±0.00086 |1.65917 |1.00000 |**0.99965** |
| ad00 acquired prefix, 1,024 packets ×64 antennas |1.08018 ±0.00095 |1.08033 ±0.00087 |1.63786 |**1.00000** |1.00022 |
| dc01 acquired prefix, 1,024 packets ×64 antennas |1.06879 ±0.03277 |1.06791 ±0.03350 |**0.08764** |1.00000 |1.00870 |

The tiny joint/control differences reverse direction on dc01. Both learned
arms remain worse than the zero-target-energy reference on all three domains.
On dc01, straightforward complex interpolation is over 12 times better than the
joint model. The full64-tap Fourier reference selected on VAL yields almost
zero on unobserved bins: with this regularly sampled pilot mask, the complete
circular Fourier dictionary's minimum-norm/ridge solution has no useful
cross-bin prior. Its NMSE 1 is a zero-fill behavior, not successful path recovery.
Every fixed Fourier/LMMSE candidate score is retained, including the failures.

Encoder collapse diagnostics on VAL show effective ranks 17.14–17.53 out of 24
and mean feature standard deviations 0.497–0.503. These outputs are not constant
embeddings. The result demonstrates that avoiding obvious total collapse is
insufficient to learn useful radio completion or transfer it between recordings.
It does not establish why every individual failure occurred.

## What was actually tested

Real DICHASUS raw complex measurements were FFT-shifted and corrected using the
publisher's archived STO/CPO calibration. The 64 model frequencies are actual
original samples at shifted indices `0,16,...,1008`, with their true Hz spacing.
Context A contains indices `0,4,...,60`; context B contains `2,6,...,62`; the 32 common
targets are all odd indices. Each row represents the **same antenna/link and
packet**, with digitally withheld frequencies. This is not an on-air sparse
sounding deployment or cross-array phase-coherence demonstration.

cf06 uses a pose-only guarded partition: 561 TRAIN, 130 VAL, 155 TEST and 178
excluded packets. ad00 and dc01 each use all 1,024 previously acquired prefix
packets, rather than their complete recordings. The 3 recording domains contain
2,203 test packets and 136,032 antenna/link observations; those observations are
not 136,032 independent sites. cf06 and ad00 are industrial recordings at
different frequencies/configurations; dc01 is a hallway. Frequency, environment
and receiver changes are confounded. These source prefixes were inspected for
earlier tasks, so this is not a globally unseen confirmation set.

The matched neural arms share architecture, initialization, batch order,
optimizer, 40 epochs per fit, validation checkpoint selection and anti-collapse
regularizers. Both branches differentiate; there is no EMA or stop-gradient
target. The sole controlled difference is a symmetric consistency loss on
predicted-field IFFT coefficients. Per-view scales/phases come only from that
view's 16 observed pilots and are undone before alignment in the common
same-packet receiver gauge. A fixed TRAIN-pilot RMS sets the alignment scale.
By Parseval's identity, this full-basis alignment is energy-scaled field
consistency. It is not independently identified scattering structure, physical
causality, materials or absolute delay. The residual decoder can modify pilot
bins and is a completion predictor, not measurement-preserving interpolation.

The 12 Fourier and 9 LMMSE configurations were selected only on cf06 VAL, with
deterministic first-minimum selection. The selected Fourier settings are 64
taps/ridge 0.01; LMMSE uses diagonal shrinkage 0.5/noise fraction 0.1. Its covariance
was fitted only on complete TRAIN spectra and persisted before test access.
Prediction/audit load this state and never retune after test access. LMMSE
transfers its covariance by normalized frequency-bin index; dc01's bandwidth
differs slightly. Actual physical frequency metadata remains explicit in the
neural and Fourier queries.

## Evidence and reproducibility

- [Exact protocol](../../artifacts/channel_scene/protocol.json) and
  [pose-only inventory](../../artifacts/channel_scene/inventory.json).
- [All fit/VAL selections](../../artifacts/channel_scene/fits/completed.json),
  [frozen prediction receipt](../../artifacts/channel_scene/predictions/completed.json)
  and [complete numerical report](../../artifacts/channel_scene/report.json).
- [Raw replay audit](../../artifacts/channel_scene/verification/raw.json) and
  [byte closure](../../artifacts/channel_scene/verification/manifest.json).

The 46 preflight fixtures passed before publication. Each neural fit completed
the fixed 40 epoch budget. Recorded fit wall times total 165.8 seconds, including
their validation work; this is not an inference-latency benchmark.

The raw audit reconstructed the pose split, TRAIN-pilot RMS, TRAIN covariance,
all fixed VAL candidate scores, selected neural checkpoint VAL scores and
every held prediction from actual pilots and frozen numeric state. All 54
model/view/domain predictions reproduced with maximum absolute difference 0.0.
The final byte closure passes: 491 source files, 24 repository evidence files and
57 external prediction shards. The older 475-source/5,161-evidence closure
remains byte-identical.

Complete predictions occupy 583,835,728 bytes in the external cache. Their
compressed transport capsule is 584,102,112 bytes at
`/workspace/training/channel-scene/deliverables/channel-scene-predictions.tar.gz`,
SHA256 `d7c408a2f8195457a6f133d726e364fab6193b1df1f5e37368d8214a255dba87`.
Git contains strict hashes/shapes and receipts; **the PR without these cache
bytes is not a self-contained passive numerical audit**. Missing shards are
fatal to closure/replay, never silently skipped. Follow the relocation steps
in [the experiment README](../channel_scene/README.md); actual raw acquisition
files are additionally required for full pilot/checkpoint replay.

The source datasets are CC BY 4.0. The derived predictions are adaptations.
Credit Florian Euchner and Marc Gauger for
[cf0x](https://doi.org/10.18419/darus-2854); Florian Euchner, Phillip Stephan,
Marc Gauger and Stephan ten Brink for
[adxx](https://doi.org/10.18419/darus-4062); Florian Euchner, Phillip Stephan and
Marc Gauger for [dcxx](https://doi.org/10.18419/darus-3831), University of
Stuttgart. Retain their release citations and
[CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/). No author
endorsement is implied.
