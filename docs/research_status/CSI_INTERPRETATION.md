# What the measured-CSI result establishes

The registered complex-CSI completion experiment is a **negative result for this model and task**. Adding same-packet field consistency to the matched supervised model produces almost no gain. Neither neural arm beats the approximately zero-output reference, and an ordinary interpolation baseline is much better on the third recording. This does not compare a JEPA world model against RadioMap: neither future radio state, interference planning, nor robot service outcomes are the target here.

The independent read-only audit reconstructed all **54 domain/model/view scores and 15 summary rows** from the original measured I/Q and stored predictions, with **zero model forwards, zero fits and zero new scientific cases**. It also reconstructed the pose split, TRAIN-pilot scale and saved TRAIN covariance, checked all registered training budgets and validation-selection metadata, and verified the portable prediction archive. The separate owner audit replays the saved checkpoints and classical estimators; the independent audit does not claim to repeat those forwards.

Evidence: [frozen protocol](../../artifacts/channel_scene/protocol.json), [report](../../artifacts/channel_scene/report.json), [owner raw replay](../../artifacts/channel_scene/verification/raw.json), [independent review](../../artifacts/channel_scene_review/review.json), and [independent script](../../artifacts/channel_scene_review/script.py). Source publication: `9bd34ab688d8b74dc2e05df8a6c55fed87257f20`. The sealed closure contains 491 source files, 24 evidence files and 57 external prediction shards; the older 475-source/5,161-evidence closure remains unchanged.

## Results without a promotion claim

Mean packet NMSE is lower when better. Each packet aggregates its antennas and 32 held-out frequency bins before averaging. Neural results average the two pilot views and three independent training seeds; the ± values are sample SD across seeds 17, 43 and 89, not confidence intervals over independent environments. Classical references are deterministic fits, not three training runs.

| Method | cf06 spatial test | ad00 recording | dc01 recording |
|---|---:|---:|---:|
| Linear interpolation | 1.659171 | 1.637859 | **0.087644** |
| Fourier ridge, selected 64 taps | 1.000000 | **1.000000** | 1.000000 |
| TRAIN covariance LMMSE | **0.999649** | 1.000216 | 1.008705 |
| Matched supervised regression | 1.083490 ± 0.000861 | 1.080332 ± 0.000873 | 1.067912 ± 0.033498 |
| Regression + common-gauge consistency | 1.083330 ± 0.000899 | 1.080180 ± 0.000947 | 1.068786 ± 0.032769 |

Consistency improves the first two aggregate means by only about 0.00016 and 0.00015 NMSE, and worsens the third by about 0.00087. These are descriptive differences on this fixed development study. They do not establish a useful consistency advantage. On dc01, linear interpolation has about twelve times lower NMSE than either neural arm. A score near one is the error of predicting zero for a nonzero target; it does not demonstrate successful channel reconstruction.

Both arms have the same architecture, seed-specific initialization, batch ordering, 40 epochs, supervised targets, variance/covariance regularization and coefficient regularization. The sole controlled difference is the added consistency term. Checkpoints use the first lowest cf06 validation NMSE over the complete 40-epoch history. Fourier and LMMSE choices use the registered finite validation grids. The independent audit checks these recorded selections; it does not independently replay validation predictions.

## The pilot geometry has an exact ambiguity

The experiment first keeps 64 uniformly spaced bins from the original 1,024-bin spectrum: raw shifted bins `0,16,…,1008`. Within this 64-bin grid:

- View A observes `0,4,…,60`: 16 pilots.
- View B observes `2,6,…,62`: 16 different pilots.
- Both views are scored on `1,3,…,63`: 32 missing odd bins.

The decoder uses the complete 64-coefficient circular Fourier representation

\[
H_k=\sum_{m=0}^{63}a_m\exp(-2\pi i km/64).
\]

The first physical frequency supplies the coefficient phase reference; subtracting that reference leaves this exact DFT grid. A single view supplies a rank-16 linear operator on 64 unconstrained complex coefficients, leaving a 48-dimensional nullspace. In view A, the columns for taps \(m\) and \(m+16\) are identical. View B changes their relative sign, but its individual operator still has rank 16.

Even observing **both** views cannot identify an arbitrary field in this decoder class. Their union contains only the 32 even bins and has rank 32. For any \(m\in\{0,\ldots,31\}\), let

\[
\delta a=e_m-e_{m+32}.
\]

On every observed even bin, \(\delta H_k=0\). On every target odd bin,

\[
\delta H_k=2\exp(-2\pi i km/64)\ne0.
\]

Thus two different allowed coefficient vectors produce exactly the same observations in both views and different answers on every target bin. This is an analytic property of the frozen observation operator, not a newly generated benchmark case. It establishes nonidentifiability in the declared unconstrained model class; it does not establish that real channels are unpredictable once an independently justified physical prior or additional measurements are available.

### Why the selected Fourier baseline outputs zero

Let \(A\) contain the 16 observed rows of the complete 64-point DFT and \(B\) the target rows. Distinct DFT rows are orthogonal:

\[
AA^H=64I,\qquad BA^H=0.
\]

For the registered isotropic ridge objective with \(P=16\) pilots,

\[
\hat a=A^H(AA^H+\lambda P I)^{-1}y,
\qquad
\hat H_{\rm target}=B\hat a=0.
\]

The minimum-norm solution at zero ridge has the same missing-row behavior. Consequently the selected 64-tap, ridge-0.01 model is effectively a zero-fill comparator on the scored bins. The independent stored-array audit finds its predicted target energy is only about \(1.3\times10^{-27}\) to \(2.1\times10^{-27}\) of target energy. Its NMSE of one is the expected algebraic result. Tiny floating differences between the full-basis ridge candidates near one do not establish a meaningful preference among their ridge strengths.

Truncating the basis changes the prior and the ambiguity; the registered 8-, 16- and 32-tap candidates were already worse on cf06 validation. Merely imposing a shorter circular tap vector cannot be promised to fix this result. Physical delay support, reference timing and truncation need justification from acquisition metadata or TRAIN-only measurements.

### Timing and gauge matter

The 64-bin grid spacing is about 0.782 MHz. Each individual pilot comb is about 3.128 MHz apart, so its delay ambiguity period is about 0.320 μs. Combining the two combs changes the ambiguity to about 0.639 μs; it does not recover a unique unconstrained 64-coefficient field. These numbers describe the sampled operator, not a measured maximum propagation delay.

Publisher STO/CPO corrections are applied identically to all methods. They do not supply a bounded absolute delay origin or global phase-coherence certificate. An unverified assumption that the physical response lives in the first few circular taps can therefore remove valid signal. Nonnegative DFT indices alone do not identify causal paths, scatterer positions or materials.

Irregular placement can break these particular repeated-column aliases, but **16 measurements still cannot identify 64 arbitrary complex coefficients**. It helps only together with a defensible lower-dimensional, sparse, correlated or probabilistic prior, or an explicit increase in measurement budget.

## This is field consistency, not a demonstrated JEPA advantage

The joint arm compares predicted fields from two pilot subsets of the same packet after undoing their separate pilot-only RMS and phase gauges. That gauge handling is appropriate: no target-derived gain or phase alignment is used. However, both branches differentiate, both have supervised missing-bin targets, and there is no EMA teacher or stopped-gradient target encoder. The tested addition is symmetric field-consistency regularization.

The loss uses a full-length IFFT of the field difference. With the registered normalization, Parseval gives

\[
\operatorname{mean}_m\left|\operatorname{IFFT}(H^A-H^B)_m/s\right|^2
=\frac{1}{64s^2}\operatorname{mean}_k|H^A_k-H^B_k|^2,
\]

where \(s\) is the fixed TRAIN-pilot RMS. It is therefore a rescaled spectral MSE consensus term. A complete Fourier change of coordinates does not by itself impose sparse paths, bounded delay support, propagation dynamics or material structure. Agreement between two underdetermined predictions can leave both wrong in the same target nullspace.

There is also no evidence of complete representation collapse in the saved validation diagnostics: effective ranks are about 17.1–17.5 of 24 dimensions and mean latent standard deviations about 0.497–0.503. Those diagnostics do not prove that the representation carries useful physical information. Here the scored prediction task supplies the more consequential evidence, and its result is poor. Increasing latent variance alone is not an established remedy.

## What the data do and do not cover

cf06 provides 1,024 acquired packets with 32 antennas: 561 TRAIN, 130 validation, 155 spatial-test and 178 guard-excluded rows. ad00 and dc01 each supply 1,024 acquired prefix packets with 64 antennas. These are measured DICHASUS data, licensed CC BY 4.0 as documented in the [source audit](../RADIOJEPA_CSI_DATA_AUDIT.md). The experiment digitally masks already measured spectra; it does not demonstrate reduced on-air sounding cost.

The acquisition prefixes and cf06 partition were inspected in earlier work. This task's source was published before its fitting and scoring, but the study is not globally unseen-data confirmation. The cf06 carrier is 1.272 GHz; the other recordings are approximately 3.44 GHz and also differ in environment and receiver geometry. Band, site and hardware effects are confounded. A carrier feature constant throughout TRAIN does not establish learned frequency-transfer physics.

The raw dc01 frame timestamps are **not monotone in stored source order**. The independent review's field named `capture_elapsed_s` records the literal last-minus-first timestamp, approximately −1,551.5 seconds for dc01; it must not be read as a negative acquisition duration or a chronological trajectory. The completion model uses neither time nor future frames, so this does not change the verified scores. It rules out treating these source indices as evidence of temporal prediction without a separate validated ordering protocol.

Antennas, frequency bins and nearby packets are correlated. No iid-antenna confidence interval is appropriate. Receiver-referenced complex magnitudes are not RSSI dBm or calibrated passive-channel power. The evaluated direction is mobile transmitter to receiving array; this is not bidirectional reciprocity evidence. No warehouse, 100-car, Wi-Fi/5G protocol-transfer, BLER, throughput or service-reliability result follows from this completion benchmark.

## A justified next protocol, before any additional fitting

The negative result justifies improving the observation design and physical prior before enlarging the network. It does not justify retuning on the three reported test outcomes and calling the next score confirmation.

1. **Freeze a new acquisition and evaluation contract.** Use only eligible TRAIN/validation records to choose windows, calibration rules, representations and pilot patterns. These already reported test domains are development evidence for future design; reserve an uninspected recording or acquisition block for confirmation. Bind source, split, objective, finite search and budget before fitting.
2. **Certify the actual observation operator.** Report rank, target-visible nullspace and conditioning on the true frequency grid. At a fixed 16-cell budget, compare a predeclared irregular design with the regular comb; represent unresolved directions as uncertainty. An optional posterior must name its prior and noise assumptions, rather than convert a nullspace into certainty.
3. **Use native frequency resolution deliberately.** The current experiment discards 15 of every 16 measured bins before creating its pilot views. A local window of 64 consecutive native bins has a different bandwidth, resolution and delay ambiguity; it is a different prediction task and must be reported that way. A full-1,024-bin or denser-pilot experiment must charge its additional observations explicitly. Keeping 16 widely spaced pilots over the same physical span leaves the sampling problem largely intact even if the output grid grows.
4. **Choose a prior with an identifiable physical role.** Candidate controls include acquisition-justified delay support, structured/sparse or learned covariance, and representations invariant to unverified common phase or linear timing phase. A gauge-invariant representation cannot by itself restore a coherent receiver-gauge channel: the reference must be supplied or estimated from permitted observations, and uncertainty retained. Score the intended invariant or physical target explicitly; do not use hidden target bins to select phase/gain alignment.
5. **Separate learning objectives.** Compare the same architecture and compute with supervised regression, current field consensus, and a precisely defined predictive representation objective. Keep anti-collapse controls identical. Measure whether each representation adds useful held-out information, rather than treating nonzero latent variance as the outcome.
6. **Build temporal and robot claims on validated acquisition.** A world model needs timestamped causal context, motion/actions, device identities and sensor visibility appropriate to the forecast. The warehouse claim then needs matched-budget interference/planning and service measurements. PHY deployment additionally needs waveform, coding, synchronization and front-end validation; channel NMSE alone does not establish link adaptation or production reliability.

These are proposed separately registered follow-ups. No such additional fits or scientific cases were performed by this interpretation or its independent audit.
