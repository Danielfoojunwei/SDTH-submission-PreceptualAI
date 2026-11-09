# Why the simultaneous-array pilot did not qualify

The completed pilot did not improve on its simple controls. It is sealed and
must not be retuned or rewritten. The next experiment must change the physical
identification problem and information contract, rather than rename the same
cross-array prediction task or increase the regularization weight until its
already observed test score improves.

This is a post-hoc interpretation. It does not change registered criteria or
establish a new gain. The sealed manifest SHA-256 is
`b9114d4cf4beb65ddd4af63c87e7e154edba5c8f1bba4a56117e1e3f369acf31`.

## What the recorded results establish

All neural values below are means of three independently fitted members, not
prediction ensembles. The factory endpoint is one constructed blocked split
of the first 1,024 records, with 519 common training sources and 155 test
sources. The other recordings are whole held-out domains; none of these
sources becomes 24 independent sites.

| Factory prediction method | RMSE, referenced channel-power dB |
|---|---:|
| Training template | 2.108273 |
| Raw RF ridge | 2.165329 |
| Template plus observed input-level shift | 2.331416 |
| Joint VICReg cell | 2.649784 |
| Matched regression, SIGReg | 2.731788 |
| Chronological auxiliary, SIGReg | 2.747548 |
| Joint without prior | 2.900105 |
| Joint SIGReg | 2.908253 |
| Unstructured decoder, SIGReg | 3.158238 |

The VICReg cell's frozen XY probe averages **2.397191 m**, versus **2.013017 m**
for a ridge probe of the raw visible RF. Its better RF score than other neural
cells does not establish a useful representation advantage over the controls.

In `ad00`, the simple input-level control reaches **4.448623 dB**; neural
bidirectional means lie around **5.809–5.949 dB**. In `dc01`, the fixed template
is **6.395957 dB**; the lowest neural mean is **7.253307 dB**. The input-level
control itself degrades to **8.928401 dB** in that hall. There is no basis for
model promotion, a RadioMap superiority claim, or a robot-service claim.

## The physical mismatch comes before the regularizer

The factory's A and B views observe different receiver arrays and therefore
different physical links. They see the same transmission event, but do not
contain interchangeable measurements of the same channel. A common scene
could explain them; it is not uniquely recoverable from a few averaged powers
without transmitter localization, public CAD and additional structural
assumptions. Simultaneity alone supplies neither injectivity nor the spectral
uniqueness assumptions of the cited representation theorem.

The model's additive power coordinate is the average dB level of the visible
arrays. This is a learned heuristic anchor across different links, not a known
frequency/device/reciprocity operator. A change in receiver placement, number
of source arrays or obstruction can change the relationship between that mean
and a target array. Even a single-path example needs the target link's
distance/antenna response. The decoder can learn compensating offsets, so the
anchor is not a proof that the model class cannot fit the data. It creates a
shortcut and an extrapolation burden that must not be called physical
transport.

The data also average sixteen frequency bins and multiple array members in
linear power. That is a defensible power endpoint, but removes phase and fine
aperture information. A Fourier power renderer on those observations does not
identify CIR phase, path positions or materials. The pilot correctly withheld
those claims and rejected PPR on unidentified factors.

## Excess compression is observed, not merely inferred from rank

Additional diagnostics used only the completed checkpoints and their TRAIN and
VALIDATION observations. No new fit, test-set candidate selection or reserved
scene was opened. The original manifest bytes remained unchanged.

Using the same entropy-of-centered-singular-value-energy definition:

| Coordinates | TRAIN effective rank | VALIDATION effective rank |
|---|---:|---:|
| Raw visible RF, 128 coordinates | 4.583892 | 9.889484 |
| Raw RF after removing each array's mean spectrum level | 30.003855 | 13.496193 |
| Joint SIGReg pooled `z`, seed mean | 1.014027 | 1.025593 |
| Joint VICReg pooled `z`, seed mean | 1.051807 | 1.099029 |

Joint SIGReg puts **99.834%** of TRAIN `z` variation and **99.660%** of
VALIDATION variation in its first principal component. One scalar, the visible
global power level, explains about **74.5%** and **59.6%** of its coordinate
variation respectively. The raw spectrum contains substantially more
variation. The weak probe result corroborates information loss; this is more
specific evidence than declaring every low-rank representation collapsed.

Receiver tokens retain more variation than pooled `z`, but their mean
VALIDATION effective rank is only about **2.43** for joint SIGReg. The aligned
projector has rank about **1.09** and coordinate standard deviation about
**0.079**. With no regularizer the projector has rank about **4.40** but standard
deviation only **0.00091**: a numerically higher rank can still describe an
almost constant output.

### Loss magnitudes did not imply balanced learning pressure

The following is a **frozen-checkpoint replay on the same fixed 32 TRAIN rows**,
not a historical training-gradient log. Gradients are L2 norms over encoder
backbone parameters, excluding the projector. The original training history
recorded total loss, not these component gradients.

| Seed-0 checkpoint | Measurement gradient | Weighted prior gradient | Weighted alignment gradient |
|---|---:|---:|---:|
| Joint SIGReg | 58.4816 | 0.21805 | 0.02127 |
| Joint VICReg | 9.6981 | 0.12166 | 0.03128 |
| Joint without prior | 47.6579 | 0 | 0.0000154 |

For joint SIGReg, measurement loss is 2.13744, weighted prior 0.24669 and
weighted alignment 0.002339 on that batch. The measurement gradient is about
268 times the prior gradient and 2,749 times the alignment gradient. This does
not prove every historical update had those ratios. It demonstrates why a
nonzero regularizer value and the presence of an alignment term were not
evidence that those objectives controlled representation learning.

Most selected epochs are near the 40-epoch budget boundary, and validation
improved substantially from initialization. The optimizer did learn; this was
not simply a non-running loss. More optimization was not exhausted, but the
observed failure does not justify tuning that sealed test cohort or assuming
that more epochs repair the identification problem.

## What interventions do and do not show

Intervention scores must be compared to the **A-to-B** score, rather than the
bidirectional aggregate. For joint SIGReg:

| Domain | Unperturbed A-to-B | Shuffle `z` and tokens | Zero `z` | Zero tokens | Zero both |
|---|---:|---:|---:|---:|---:|
| Factory | 3.134977 | 3.657418 | 3.518530 | 4.827325 | 5.376467 |
| `ad00` | 5.309159 | 5.690030 | 4.304245 | 4.740867 | 4.763911 |

These preserve the measured power coordinate. They show that the learned
paths carry useful in-domain information, yet can harm the transferred
prediction. They do not demonstrate that every pathway is ignored. The hall's
bad power-level control is also consistent with cross-link anchoring failure.

Zeroing or shuffling creates combinations of latent, tokens and power that the
decoder may not have seen in training. These are sensitivity diagnostics, not
randomized causal attribution or proof that the anchor alone caused the loss.
A new predeclared model/control comparison is necessary.

## The next 2a experiment

Use a separate `spectral_scene` namespace. Keep the failed pilot and its source
closure intact. The new question is static, referenced RSSI-field prediction
from public CAD/GPS and physically corresponding reported views; it is not
yet packet service, dynamic interference, physical scatterer reconstruction or
an identified absolute-SNR posterior.

1. **Match physical links.** Each view contains both observed source bands
   0/1 at the same context poses and APs. A uses forward direction/device 0;
   B uses calibrated reverse direction/device 1. Query poses are disjoint.
   Band 2 and device 3 have explicitly separated development/test roles.
2. **Fix the measurement gauge.** Fit relative device offsets from allowed
   TRAIN context pairs, with device 0 as the chosen zero reference. Do not read
   registry truth offsets or query RF. RSSI equals physical link gain plus
   transmit power and front-end effects; those components retain a common-shift
   gauge without an absolute reference.
3. **Use link-specific operators.** FSPL is conditional on the same link and
   fixed reference gains. Public direct-path CAD features and frequency terms
   form an explicit field basis. A small inferred scene state represents its
   reference-gauge coefficients. Never broadcast a different AP's mean power
   as a known physical transformation.
4. **Preserve the information contract.** A strong RadioMap/RFF model, direct
   frequency transport and matched neural regression receive the same public
   CAD, GPS, allowed context observations, device calibration and sensor costs.
   Private RT channels, path lists and material labels are not encoder inputs.
5. **Make the bottleneck real.** The decoder receives only the scene code and
   public query geometry, with no receiver-token or measured-power bypass.
   Include geometry-only and exact matched-regression controls and zero/shuffle
   interventions on the scene state.
6. **Use enough distinct scenes.** The proposed split is 48 TRAIN, eight
   VALIDATION, four DEVELOPMENT and 24 fresh FIELD layouts. Two nearly
   identical views of B layouts do not provide 2B independent scene states.
   An eight-dimensional state with a full batch of 48 distinct TRAIN layouts
   avoids an impossible 48-dimensional whitening target in a 16-layout batch.
7. **Keep the two confirmation tasks separate.** These 24 FIELD specifications
   are newly reserved RF layouts. They must not consume the existing 24
   unopened robot-service confirmation specifications or be pooled with them.
   Robot service still needs its own paid-cost, sensing, tail and replay proof.

The current implementation uses a differentiable regularized inverse of the
eight public basis functions from heard context reports, plus a learned scene
correction. Censors remain inequalities/features and likelihood terms; they
are not inserted into least squares as point measurements. The inverse is a
reference-gauge field estimate, not identified material or scatterer truth.
An independent RadioMap remains necessary: deriving a learned model from a
radio-field estimator does not itself establish superiority over one.

### Scale and anti-collapse implementation

Use quantization-interval and actual-censor likelihoods, excluding genuinely
missing reports and averaging over attempted query observations. Noise figure
determines a thermal detection floor; it is not an observed RSSI jitter sigma.
The new code declares its fixed four-dB likelihood scale as unresolved field
error, **not** sensor-noise truth or calibrated safety uncertainty.

A useful analytic scale convention is to divide measurement loss by a fixed
TRAIN-only reference scale, and to normalize each prior by its zero-code
penalty. For the existing VCReg defaults that penalty is 12.375; for SIGReg it
is `N * sum((1-phi)**2 * weights)` in the actual upstream buffers. That makes
the meaning of an order-one coefficient explicit and removes accidental batch
scaling. Scalar normalization still does not guarantee balanced gradients.

The new implementation instead uses a declared TRAIN-only rule at every
update: detached scalar weights target each auxiliary's scene-code gradient
norm at one quarter of the measurement gradient, capped to `[0.01, 100]`.
Both views receive gradients; there is no EMA or stopped target. It records
achieved norms, caps and exact-zero nonzero-prior stationarity. On diagnostic
updates it also records actual backbone norms and cosines: equal scene-code
gradient norms do not imply equal backbone gradients or compatible directions.

Generic Gaussian/VIC priors are still hypotheses. A physically meaningful
field-coefficient state need not have independent unit-Gaussian coordinates.
An informative low-dimensional inverse should not be condemned solely for
failing that arbitrary marginal. Keep the no-prior inverse and strong
non-neural controls, and decide using fresh predictive quality and useful
state interventions, not a demanded rank.

### Remaining identification limits

Two bands and reciprocal calibrated reports reduce nuisance mismatch but do
not uniquely identify an entire scene, unseen material dispersion, an unseen
band's response or a shadow field away from the context. After exact simulated
calibration, the two views can be almost redundant. Their alignment loss may
add no value; the matched regression control must be allowed to win.

The existing RT data are finite-path-search static simulations with isolated
channel probes, not measured hardware or a complete Maxwell/interference
solution. Simulated receiver offsets and quantizers are a limited device
transfer test. Field error and held-device onboarding must retain that scope.

Only after a new development gate passes should fresh FIELD scenes be opened.
Only after separate robot-service qualification should learned service heads
replace any RadioMap fallback or independent trust gate.
