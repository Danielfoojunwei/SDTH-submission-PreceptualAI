# Complex-channel scene slice

This is new infrastructure for the complex-CSI portion of the RadioJEPA plan.
It does not modify the completed power-spectrum study, its 475 source files or
5,161 evidence files. No training, new measured-CSI holdout evaluation, learned
prediction or application promotion has been executed for this slice.

## Observation contract

`channel_scene.data.ObservationAdmission` names a **new** protocol, split,
purpose and exact source hashes. Its role is explicitly
`complex_cfr_observations`. That distinction matters: previous experiments
classified Sionna CFR as private simulator truth. It becomes an input here only
under a new declared observation task, never by retroactively changing an old
experiment's information budget.

| Reader | Observable input | Default withheld information |
|---|---|---|
| `load_dichasus` | Actual full complex I/Q, original capture times, physical array IDs and public array centers, carrier/grid, actual STO/CPO estimates | Transmitter pose labels; any claim of an absolute RF clock or globally phase-coherent arrays |
| `load_sionna` | Only aggregate `cfr` from the admitted channel file, public AP CAD, carriers and link-solved masks | Per-ray coefficients/delays, material parameters/classes, clean RSSI targets, and mobile poses |

The public encoder whitelist contains only `cfr`, `frequency_offsets_hz`,
`carrier_hz`, `observed`, `infrastructure_xyz_m` and `group_ids`.
`private_labels(..., purpose=...)` is separate and allows only explicit
partition, frozen-probe or evaluation uses. A protocol may declare mobile CAD
poses public; even then, they are not silently inserted into the default
encoder arguments.

Inspection mode rejects the measured ad00/dc01 recordings and RT field-test
folders before opening their observation arrays. The permitted read-only
training eligibility checks completed on:

- Acquired DICHASUS cf06: calibrated complex64 `[1024,32,1024]`, real source
  capture clocks, four bound source files, no pose input or coherence proof.
- Existing `spectral-training-00`: complex64 `[72,2,64]` after preserving
  pose/carrier/direction contexts, three bound files, no fabricated acquisition
  clock, no pose input and no implicitly admitted coherent-phase proof.

These are schema/calibration eligibility checks, not scientific scores. The
reader never creates or downloads replacement observations, retraces a scene,
or loads a model. New experiments must additionally enforce the main runner's
source-publication and phase guards; an admission object alone is not a
cryptographic authorization or a promotion decision.

## Calibration and physical gauges

The measured reader shifts raw FFT order into increasing frequency, then uses
the archived DICHASUS convention:

`H_corrected[k] = H_raw_shifted[k] * exp(j*(2*pi*STO*k/N + CPO))`.

Subselected bins must retain their original shifted-bin identities and original
FFT size. Estimated STO/CPO is not converted to a Doppler label, an absolute
ToF observation or a measured inter-array phase-error bound. Receiver-referenced
CSI may have magnitude above one; it is not received mW or a normalized passive
power-wave scattering parameter. No unit-magnitude clipping is applied.

`PhaseCoherenceProof` binds source identities, covered groups, an explicit
reference-error bound and either instrument evidence or an expressly admitted
ideal simulation assumption. Actual measured readers supply no such proof by
default. The dcxx source specifically warns that between-array offsets are
less reliable. Zero reference error in an ideal RT proof is an oscillator/clock
assumption of that simulation, not a bound on float arithmetic, truncation or
real propagation-model error. Coherent use also requires a separate named
`PhaseCoherenceBudget` selected by the protocol. Evidence must meet both its
phase-radian and timing-second tolerances. There is no permissive default
budget, and a proof cannot set its own acceptance threshold.

The default complex encoder therefore uses phase curvature:

`u[k+2l] * u[k] * conj(u[k+l])^2`, with `u=H/abs(H)`.

On a uniform grid this cancels both an arbitrary per-link constant phase and
an arbitrary linear frequency-phase slope. It still contains complex phase
information that power spectra discard. Deep-null or unknown triplets retain
an explicit validity mask. It does not require unproved cross-array coherence.

Three encoder modes have different contracts:

1. `gauge_invariant`: amplitude plus complex phase curvature; default for
   measured data with unknown phase/sampling origins.
2. `relative_coherent`: also uses within-group relative complex phase;
   requires bound coherence evidence covering those groups.
3. `absolute_coherent`: also exposes reference-aligned I/Q; requires global
   phase evidence and an explicitly admitted absolute reference.

The shared link encoder preserves the referenced power level through a
separate bypass. Its reported link/scene power summaries average masked linear
`abs(H)^2` before converting to dB; they are not means of dB values and do not
acquire dBm units. Alongside carrier it retains actual frequency-bin spacing
and sampled bandwidth, so identical discrete spectra on different bandwidths
remain distinguishable and keep their physical delay scales. It uses set
attention and public pairwise distances in
wavelength units, with group IDs compared only for equality. Its scene output
is permutation invariant and invariant to rigid transformations of the
infrastructure coordinate frame. This is a geometry prior, not a measured
interference graph or a learned 3D scatterer-identification result.

## Fourier and CIR prediction

`CausalFourierOperator` renders
`H(f) = sum_p a_p * exp(-j*2*pi*f*tau_p)` and checks finite coefficients,
nonnegative delays and any explicitly supplied public lower bound. Relative
unknown clocks cannot be given a physical `d/c` bound. A later learned head may
output these coefficients/delays; this operator itself has no learned
parameters or training side effects.

Finite-band IFFT output is exposed with resolution `1/(N*delta_f)` and delay
ambiguity period `1/delta_f`. Its taps use the first sampled frequency as phase
reference. A causal off-grid path produces circular-boundary sidelobes; these
are not evidence of negative physical delay. The same finite samples can also
be generated by delays displaced by an ambiguity period with adjusted complex
coefficients. Consequently a finite-band CIR alone never certifies causality.

For measured data, predicted nonnegative taps are a **canonical relative-gauge
representation**, not recovered absolute ToF. For an explicitly admitted ideal
RT reference, absolute modeled delay bounds can be tested without feeding
private ray-delay labels to the encoder.

## What is identifiable here

| Quantity | Status |
|---|---|
| Within-band amplitude shape and referenced power | Observed, subject to device/reference gain and noise |
| Complex spectral phase curvature | Observed modulo the declared constant-phase/linear-delay gauge |
| Absolute phase and absolute ToF in measured DICHASUS | Not identified by these files/calibration estimates |
| Relative inter-array wavefront phase/AoA | Needs independently justified coherence and array orientation; not enabled by default |
| Absolute RT phase/time reference | An explicit ideal-simulation convention, not hardware evidence |
| Resolved multipath delay components | Finite-band, finite-resolution and generally non-unique; uncertainty/aliasing must be reported |
| Geometric scatterer positions, materials and object identities | Not uniquely identified by the current observation contract |
| Doppler/mover velocities | Not obtained by relabeling CFO or treating static RT contexts as a motion sequence |
| Real device invariance, ACK/PER, action outcomes or service SLAs | Require separate captures, heads, calibration and native/hardware qualification |

## Offline probe recipe after source publication

The bounded first measured experiment is now specified by
`artifacts/channel_scene/protocol.json` and the independent
`channel_scene.benchmark` runner. It digitally withholds32 of64 actual sampled
complex-frequency bins on the **same antenna/link/packet**, comparing two
disjoint16-pilot contexts. The64 bins are original shifted indices
`0,16,...,1008`, with their actual Hz spacing, not power-averaged replacements.
cf06 has guarded pose-only TRAIN/VAL/TEST partitions; all previously acquired
ad00/dc01 prefix frames are whole cross-recording tests. These recordings were
inspected for earlier tasks. Neither globally unseen data nor full-recording
coverage is claimed.

Both neural arms have the identical initialization, sparse-pilot encoder,
finite circular Fourier residual decoder, optimizer,40epochs, shuffled batches
and anti-collapse regularizers. **No EMA or stop-gradient branch exists.** The
joint arm adds only symmetric differentiable coefficient-space alignment.
Independent context RMS/strongest-pilot phase transforms are undone before
aligning predicted-field IFFT coefficients in the common same-packet receiver
gauge. A frozen TRAIN-pilot global RMS sets that loss's scale. No target-bin
value determines an encoder scale, phase, curvature or clock estimate.

The decoder starts from complex linear interpolation; learned finite Fourier
residuals have no identified physical path interpretation. Matched supervised
regression, genuine complex linear interpolation/extrapolation, VAL-selected
Fourier ridge and a TRAIN-only complex LMMSE covariance estimator are mandatory
references. Deterministic references are not presented as independent seeds.
dc01 has a slightly different bandwidth; LMMSE covariance transfers by
normalized bin index, while the neural encoder and Fourier queries retain
actual frequency metadata. This is a declared covariance-transfer assumption.

Three independent neural seeds use the same fixed40epoch budget each. The
primary NMSE first aggregates target energy over antennas/bins within a packet,
then averages packets and the two pilot contexts. Reports retain every seed,
every reference and all negative results. Seed SD and descriptive packet
quantiles are distinguished; correlated packets are not treated as independent
deployment trials. All fits are bound before cross-recording evaluation, and
predictions are persisted before task scores. Source registration and observed
Git publication must precede the fit stage. Example commands use
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` followed by
`uv run --frozen python -m wirelessjev.channel_scene.benchmark PHASE`, where
`PHASE` progresses through `inventory`, `preflight`, `register`, then (only
after publication) `run`.

1. Publish source, exact observation admissions, calibration files and split
   identities before scientific preprocessing/training. Whole layouts and
   physical receiver/device families remain grouped. Previously inspected data
   is labeled as such, not relabeled blind confirmation.
2. Define disjoint whole-array/band views using measured coherence scales.
   Keep all views of one source event in one split. Do not pair asynchronous
   receiver rows merely by file index or treat static RT context indices as
   time.
3. Match calibration, encoder capacity, observations, numerical priors and
   decoder budgets across supervised and joint-view models. Include power-only,
   linear/sparse Fourier prediction and phase-curvature ablations. A coherent
   mode is evaluated only where its proof is admitted.
4. Train on measurement-space likelihoods or identifiable complex invariants.
   Raw complex MSE is not a valid target when the target's phase/time gauge is
   unobserved. Do not supervise pretraining with private positions, materials
   or path-delay tensors. Use separate projector heads when needed.
5. Freeze the encoder before pose/protocol linear probes. Open label access
   explicitly and keep protocol labels separate from pretraining views. Check
   constant/linear predictors, effective rank, RF-versus-geometry sensitivity,
   phase-only sensitivity, held-out device/layout/band error and calibrated
   tails. Rank alone is not a non-collapse or utility proof.
6. Measure full calibration + encoding + query-head work. Only then connect a
   qualified outcome head through the existing independent WirelessJEV trust
   contract, using the same planner and RadioMap fallback. No current code
   changes the shipped application default or asserts service/production SOTA.

All current tests are analytic/negative software fixtures: calibration
convention, gauge invariance, finite-band ambiguity, causal rendering,
gradient finiteness, set/geometry symmetries, privacy, admission and fail-closed
coherence. No official training or held-out complex-CSI score is reported.
