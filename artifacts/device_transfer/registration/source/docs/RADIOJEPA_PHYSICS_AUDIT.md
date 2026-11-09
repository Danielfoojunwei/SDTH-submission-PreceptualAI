# RadioJEPA: primary-source and physical-assumption audit

The SCENE-JEPA direction is a testable research design. Simultaneous views,
calibrated devices and physical operators are useful structure; they do not
alone establish representation identifiability, improved service, or safety.
This audit changes no previous experiment or result.

## Verified source and reusable code

The cited paper exists: Randall Balestriero and Yann LeCun,
[Proving and Improving the Platonic Representation Hypothesis with JEPAs,
arXiv:2610.39302v1](https://arxiv.org/abs/2610.39302v1), submitted 29 October 2026.
Its PDF and HTML were downloaded from arXiv. Its official
[prh-proof repository](https://github.com/galilai-group/prh-proof) is pinned to
`a2b9949fa907d4fbe562742d4b1ffa7205ab7427` and carries the MIT license.

The paper's training dependency, [stable-pretraining](https://github.com/galilai-group/stable-pretraining),
is also MIT. The actual paper revision resolves to
`ab836bf699a2be712dfcf980a5eb70d36391e876`; that source was downloaded separately.
Useful original components are:

| Source at the pinned revision | Reuse |
|---|---|
| `prh/theory.py` | Finite-world pair kernels, whitening, spectral solutions, rotation distance, nuisance and kernel-mismatch checks |
| `prh/align.py`, `prh/metrics.py` | Procrustes alignment and mutual-neighbor metrics; fit alignment on development pairs only |
| `stable_pretraining/methods/lejepa.py` | Actual `EppsPulley`, `SlicedEppsPulley` and `LeJEPA` implementations |
| `stable_pretraining/losses/joint_embedding.py` | Actual VICReg comparison loss |
| `stable_pretraining.Module`, `Manager` and data/callback APIs | Existing training lifecycle if this framework is adopted; use a thin radio forward/data adapter |

The upstream documentation warns that effective SIGReg strength at this revision
depends on per-GPU batch size. The paper's `lambda=0.02` is not an automatic
default for a different batch/distributed setup. Preserve both copyright notices
when redistributing copied or modified code. The code licenses do not license
downloaded models or datasets.

Actual upstream finite-world tests were run on CPU: **37 passed in 4.03 s**.
The test file was copied byte-for-byte and the actual NumPy `prh.theory` module
was loaded directly to avoid unrelated captioning/tokenizer imports. This is
dependency-isolated mathematical verification, not a full package installation,
paper training reproduction, wireless benchmark, or new model result.

Downloaded sources, Git checkouts, receipt JSON, the unchanged test copy and its
stdout are under `/workspace/upstream/scene-radio/theory/`. Primary hashes are:

| Item | SHA-256 |
|---|---|
| Paper v1 PDF | `083cc4228dffbf946af1a82d9f5817633eb4bb2a38d923b8320478453a3a6838` |
| `prh-proof` MIT license | `824b459e9de14e9d64c193a115ba9d015f0d1c95ce57fdbf6251d02db4f135c6` |
| Pinned `prh/theory.py` | `80bec5c85ee83141960704817af0ff77d72b413e1acfb565a5385a5bab4fb458` |
| Unchanged upstream `test_theory.py` | `48072701f7cce4fc0b8a640d861a44ea711126e69a7b2f8be882ac4629fd0593` |
| `stable-pretraining` MIT license at the paper revision | `8fa02eaa4e67a5455e8ab9268b29e0c283d4b4a5ae239f88282c00392f373bfd` |

## What the PRH result establishes

Theorem 1 explicitly assumes a finite world, injective modality renderings,
the same positive-pair distribution, population optima and minimizers unique
up to rotation. Proposition 1 proves that uniqueness for whitened squared-loss
JEPAs when the relevant spectral gap exists. The paper explicitly leaves
uniqueness under SIGReg's full-distribution constraint as an assumption.
The nuisance extension also requires conditions on the pair kernel and
independent nuisance variables; it does not remove all observability conditions.

RSSI is many-to-one: different scenes, materials, path phases and device gains
can produce the same reading. Real device nuisance often depends on scene and
orientation. We therefore cannot import this theorem as a guarantee that two
radio encoders will converge to the same identifiable physical scene.

There is a particularly relevant boundary case. If every positive pair refers
only to exactly the same state, then its world pair distribution can be
`pi = diag(p)`. Its Markov operator is the identity: every eigenvalue is one,
so the required gap is absent. Using the actual upstream loss/rotation code,
an analytic six-state, two-coordinate example has:

- two centered, whitened embeddings;
- zero positive-pair loss for each;
- best-rotation RMSE **2.0** between them.

The saved `identity-positive-counterexample.json` records this calculation.
It is a finite mathematical example, not fabricated wireless data and not a
counterexample to the paper's qualified theorem. It shows why *simultaneity
alone* does not imply the missing uniqueness assumption. Define and inspect
the shared state-neighborhood kernel where appropriate, and measure alignment
across independent fits instead of assuming it.

The reported **42-fold** sample reduction concerns a particular 109M text
encoder, corpus and DINOv2-alignment target: 21.6M versus 900M samples. It is
not a universal penalty for temporal prediction, nor a predicted wireless gain.
Temporal views can still serve dynamics learning; same-scene views serve a
different, explicitly tested role.

## Corrected physical contracts

| Proposed ingredient | Contract suitable for implementation |
|---|---|
| Multi-band operator | Free-space loss has a known frequency term under fixed antenna-gain/reference assumptions. Indoor material penetration, antenna response, polarization and diffuse paths remain frequency dependent. Predict a conditional operator with uncertainty; do not assert an exact 2.4-to-6 GHz transform. |
| Bandwidth operator | `k*T*B` transforms integrated thermal noise, conditional on temperature and receiver noise figure. It does not determine received signal power, frequency-selective fading, interference, PER or airtime. |
| Reciprocity | A reciprocal propagation matrix reverses by **transpose**, at the same frequency and sufficiently close time. TX/RX front ends require calibration. Raw uplink/downlink RSSI or CSI are not interchangeable. |
| Superposition | Coherent paths add as complex amplitudes before taking squared magnitude. Independent/incoherent interferer powers may add after appropriate averaging. Two equal, opposite-phase paths must be able to cancel. |
| Passivity | Use normalized-port/scattering-energy constraints with explicit antenna and normalization conventions. A naive universal `received power <= transmit power` check can be misleading when gains, combining, near field or measurement references differ. |
| Causal delay | Physical path delay is at least the straight-line propagation time under the declared propagation model. Measured delay also includes clock/STO offsets; do not reject uncalibrated CSI using a physical-delay bound. |
| Spatial shadowing | A Euclidean spatial field can be shared across robots and revisits. OU evolution in accumulated route length is only a local trajectory approximation: after a loop it need not return to the same static shadow. |
| Fading dynamics | Jakes `J0(2*pi*f_D*dt)` is an isotropic-scattering model, not every warehouse's spectrum. An AR(1) fitted at one lag is not an exact Jakes process. Use and test the approximation explicitly. |
| Coherence masking | Coherence depends on carrier, velocity, scattering, delay spread and the correlation convention. `lambda/2` is a useful scale, not a universal hard independence threshold. Nearby independently noisy views can provide useful denoising supervision; far-future fast fading may be inherently unpredictable. |
| Device inversion | Offsets and some phase gauges can be calibrated. Clipping, missing subcarriers and quantization destroy information and have no unique inverse. Return an uncertainty/missing-data contract rather than reconstructing unobserved evidence as fact. |
| Phase sanitization | Cross-antenna conjugate products cancel common phase, but not arbitrary chain offsets or all timing effects. Removing subcarrier-linear phase can also remove useful ToF information. Declare the retained gauge and observables. |
| Airtime conservation | Apply occupancy constraints per channel/receiver contention domain and include idle time. Spatial reuse permits multiple transmitters at once; STR MLO permits parallel activity on distinct links. Summing every transmitter's busy fraction into one global simplex is incorrect. |
| Graph attention | Sets with shared attention can be permutation equivariant. They are not automatically SE(2)/SE(3) equivariant. Use relative geometric quantities/equivariant operations and rotate antennas, velocities and polarization consistently. |
| Scatterer tokens | Path delay/angle/gain tokens can be useful without uniquely identifying physical objects. RSSI alone cannot establish path phase, material or occluded-person identity. Reserve those interpretations for CSI/geometry validation. |

Useful exact numeric references at `T=290 K`, before noise figure, are
`10*log10(k*T/1 mW) ~= -173.975 dBm/Hz` and the bandwidth noise shifts
20-to-80 MHz **+6.0206 dB**, 20-to-160 MHz **+9.0309 dB**, and
20-to-320 MHz **+12.0412 dB**. A free-space frequency shift is
`20*log10(f_target/f_source)` dB under the stated fixed-gain assumptions.

The [ITU-R P.525-5 recommendation](https://www.itu.int/rec/R-REC-P.525/en)
was downloaded from the ITU and verifies the free-space reference.
[3GPP TR 38.901 v18.0.0](https://www.3gpp.org/ftp/Specs/archive/38_series/38.901/38901-i00.zip)
was downloaded from 3GPP. Clauses 7.4.4 and 7.6.3 specify spatial shadowing
correlation and spatial consistency; its large-scale parameters also have
cross-correlations. These references support conditional physical models,
not blanket independence of geometry, shadowing and fast fading.

For reciprocity, the verified original reference is Vieira et al.,
[Reciprocity Calibration for Massive MIMO: Proposal, Modeling and Validation](https://arxiv.org/abs/1606.05156v3),
DOI [10.1109/TWC.2017.2674659](https://doi.org/10.1109/TWC.2017.2674659).
The paper distinguishes reciprocal propagation from nonreciprocal RF chains.
Gudmundson's original reference is
[Correlation model for shadow fading in mobile radio systems](https://doi.org/10.1049/el:19911328),
verified through publisher DOI metadata. The full Gudmundson article was not
retrieved here; the downloadable 3GPP standard supplies the inspected spatial
correlation specification.

## Anti-collapse and physical priors

Non-Gaussian physical variables do **not** prove Gaussian regularization of
learned coordinates is wrong. An invertible change of coordinates can represent
such variables in Gaussian coordinates. PPR has a stronger argument when it
regularizes **explicit, identifiable, gauge-fixed physical factors**, not when
it assigns physical names to arbitrary latent dimensions.

Use conditional priors where physics requires them. Rician K depends on
geometry/blockage; Doppler depends on velocity and angles; antenna effects
depend on orientation. Geometry and fast fading are not generally independent.
A uniform position prior is also not the warehouse's empirical route occupancy.
Applying a Dirichlet prior blindly can invent activity on truly inactive links.

Characteristic-function matching can be generalized to other target
distributions, but the metric and target must match the space. Circular phases
use periodic harmonics; positive power and sparse path amplitude are distinct
from signed Laplace coordinates. Per-coordinate marginal matching alone can
still leave a low-rank joint representation. Check joint structure and
decoder use, not just each marginal.

The renderer does not by itself prevent collapse. Query geometry, an inherited
map, free decoder parameters or a large residual can explain observations while
the inferred latent remains constant. Bound residual paths and test:

1. latent shuffle and constant-latent replacement at fixed query geometry;
2. fixed-geometry RF perturbations with documented observation support;
3. held-out-scene view prediction against constant, geometry-only and linear
   predictors;
4. effective rank and conditional RF-dependent rank, separately from the
   geometric variation;
5. independent-fit alignment, with rotation fitted on development pairs only.

Effective rank above 7/48 alone is not a capability test. A representation may
have high rank from geometry, device identity or noise while missing the
wireless information used by the decoder.

The initial implementation is `radiojepa/priors.py`, with exact extracted
upstream SIGReg and VCReg component/helper bodies under `radiojepa/vendor/`.
Their class/function source-segment hashes, full original-file hashes, commit
and license are recorded in `vendor/provenance.json`. Imports are isolated;
the VCReg square-shape diagnostic uses Python logging. This comparison uses
64 SIGReg projections, versus the upstream default of 1024. The wrapper
explicitly matches the default arithmetic dtype used by upstream random
projections, preserves its checkpointed projection step and rejects unqualified
distributed execution.

The stateful API is `PriorRegularizer(kind)(z, factors=None)`. PPR requires
named `PhysicalFactor` objects with declared semantics, target parameters and
`identified=True`; trainable prior targets and arbitrary latent-position or
material labels are rejected. A power-only CIR basis has unidentifiable phase
and path factors, so PPR is **ineligible for that experiment** until the relevant
factor is identified. Its tests do not authorize an invented physical-factor
claim. The VICReg arm uses the actual variance/covariance regularizer component;
the common model supplies the separate view-prediction objective.

**19 analytic software fixtures pass** for the implemented regularizers,
including original-source hashes, collapse penalties, wrong marginals,
conditional fixed scales, finite gradients, phase and airtime domains, projection
checkpoint state, RNG isolation and rejection of unidentifiable factors. These
are unit checks, not trained wireless outcomes. The positive penalty at an exact
collapsed point is not proof that gradient optimization escapes that point.

## Likelihood, service and safety boundaries

RSSI detection, top-K list censoring, quantization and missing reports are
different events. Use interval mass for a quantized observation,
`F(upper)-F(lower)`; use the actual censor bound for a non-detection; leave
unattempted or undelivered measurements unknown. Compute tail interval masses
stably in log space. A crowded-out AP is not automatically below the noise
floor.

Rician/Nakagami, log-normal and burst-state models are useful candidates,
not established causes or solutions for every previous tail failure. In a
consistent parameterization, Rician `K=0` and Nakagami `m=1` recover Rayleigh
amplitude. Nakagami power is Gamma-distributed with shape `m` and scale
`Omega/m`. Validate these limits, units and sufficient sampling before fitting
fading statistics from clipped/selected/averaged RSSI. Also retain direct
threshold-CDF/quantile comparisons when the parametric family is uncertain.

PER is not command-miss probability. End-to-end misses and silence also depend
on contention, retries, queue age, switching, forecast latency and sensing
interruptions. A per-action service distribution must include those mechanisms
or declare its narrower link-level endpoint. Value-of-information scanning
requires a posterior update and the sensing cost; entropy reduction alone need
not improve mission utility.

Privileged simulator truth used **only during training** is supervised
distillation, not automatically evaluation leakage. It is incompatible with a
claim of label-free self-supervision, and it must never enter inference or the
held-out evaluation-selection path. Keep these distinctions in both attribution
tests and reporting.

Split conformal prediction provides marginal coverage under exchangeability;
it does not automatically give conditional coverage for each device, action or
time. Weighted covariate-shift methods require assumptions such as appropriate
density ratios. Adaptive conformal inference's general guarantee concerns
long-run coverage frequency, not per-tick safety under arbitrary drift.
The original sources inspected were:

- [Angelopoulos and Bates, arXiv:2107.07511](https://arxiv.org/abs/2107.07511);
- [Tibshirani et al., Conformal Prediction Under Covariate Shift, arXiv:1904.06019](https://arxiv.org/abs/1904.06019);
- [Gibbs and Candes, Adaptive Conformal Inference Under Distribution Shift, arXiv:2106.00170](https://arxiv.org/abs/2106.00170).

All three PDFs and metadata were downloaded. Calibration on chosen actions
does not validate never-taken counterfactuals. Logged propensities and support
checks are necessary for off-policy claims; they do not repair zero action
support. Multiple SLA events, robots and horizons need an explicit joint-risk
budget. A barrier function can certify invariance only under its stated model,
uncertainty bounds and nonempty feasible action set. No available radio link
and no safe reachable stopping state means a guaranteed fallback may not exist.

## First experimental slice and meaningful comparisons

Start with measured or explicitly simulated simultaneous multi-band/device
views whose provenance proves the grouping. Match all arms on encoder,
physical renderer, source information, data budget, optimizer policy, head
training, planner, gate and evaluation split. The initial ladder is:

| Arm | Changed factor | Question |
|---|---|---|
| Measurement reconstruction + physical renderer, no extra prior regularizer | Base hybrid | Is the renderer/observation objective already sufficient? |
| Same model + actual SIGReg | Regularizer only | Does generic Gaussian regularization help? |
| Same model + actual VICReg | Regularizer only | Is a second-order constraint sufficient? |
| Same model + PPR | Regularizer only | Do explicit conditional physical-factor priors improve transfer/calibration? |
| Matched supervised model | Remove the cross-view latent objective, retain the same allowed supervised targets and physics | Does the JEPA objective add value? |
| Same model with temporal pairs | Pair construction only | Do demonstrated scene views improve this radio task? |

Do not quietly change the numerical prior or give only one arm clean labels.
Freeze encoders for the declared downstream-head comparison. Publish negative
results and include head/readout compute, adaptation examples and sensing cost.
Disjoint scenes, devices, collection sessions and spatial regions are more
important than random row splits. Co-located measurements from different
times/devices are not simultaneous positives without supporting metadata.

Operator tests should cover phase cancellation, power/noise units, reciprocal
transpose after calibration, missing/censored/quantized likelihoods, positive
and normalized energy, zero-speed/coherence limits, spatial loop closure,
permutation/equivariant transformations, and per-domain airtime. Add negative
tests for unsupported counterfactual actions, expired observations and
unidentifiable delay/device gauges.

Only then promote a narrowly scoped capability under the unchanged independent
trust contract. Keep RadioMap as the default and fallback unless the measured
service, tail, calibration and total-compute qualification passes. This audit
establishes source availability and assumptions; it establishes no new wireless
accuracy, SOTA, production or safety result.
