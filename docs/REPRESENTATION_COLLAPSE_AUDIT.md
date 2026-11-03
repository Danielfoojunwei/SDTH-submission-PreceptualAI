# Representation audit: concentration, not constant collapse

The six trained packet critics do **not** produce constant representations on
their 744 archived training actions. Their action latents are concentrated in
a few dominant directions, but they retain action sensitivity and linearly
accessible packet-value information. The stronger warning is that the JEPA
action predictor matches its future EMA targets poorly, despite a diagnostic
linear readout recovering substantially more of those targets.

This is a post-hoc **training-only diagnosis**. It does not overturn the sealed
controller results, demonstrate unseen-site generalization, qualify learned
control, establish SOTA, or quantify an irreducible prediction-error floor.
All inputs and labels come from the warehouse simulator; this audit uses no
physical SmartCar recordings or measured CSI.

## What ran

- All six frozen `packet_value/checkpoints/{jepa,supervised}-{0,1,2}.pt`
  checkpoints, verified against their original training manifest.
- All 744 action examples from the sixteen registered packet **training** sites
  `340000–340015`, at both handover costs. Native label receipts and input-file
  SHA-256 hashes were checked. No official validation, development, or test
  input archive was opened.
- All 72 saved causal windows from the four registered service-world diagnostic
  sites `360100–360103`. This exercises the **untrained** 68,021-parameter service
  model with seed 0; there is no trained service-model checkpoint.
- Fixed ridge probes (`alpha=10`), with four folds containing consecutive blocks
  of four training sites. Every preprocessing statistic and readout fit excludes
  the scored fold. **The frozen encoders were already trained on all sixteen
  sites**, so these folds isolate the readout fit only; they are not model
  holdouts. No hyperparameter was selected from the diagnostic scores.

The machine-readable [report](../artifacts/representation_audit/report.json)
contains the source, checkpoint, protocol, receipt and data hashes; full spectra;
per-cost results; masks; local derivatives; and probe errors.

## The trained representations vary

Effective rank below uses the entropy of the **centered covariance eigenvalues**.
It measures how widely variance is distributed, rather than counting nonzero
coordinates. Nominal hidden width is 48. Numerical rank is 47 throughout the
context and action latents, using a float32-aware SVD tolerance: LayerNorm
itself imposes an affine constraint. The much smaller effective ranks indicate
dominant directions, not literally rank-zero or constant codes.

| Frozen model | Context effective rank | Action effective rank | Action top-PC share | Action centered RMS |
|---|---:|---:|---:|---:|
| JEPA 0 | 7.06 | 3.30 | 61.5% | 0.930 |
| JEPA 1 | 7.74 | 4.26 | 52.5% | 0.898 |
| JEPA 2 | 7.50 | 5.31 | 47.4% | 0.915 |
| Supervised 0 | 7.33 | 3.27 | 63.6% | 1.004 |
| Supervised 1 | 6.93 | 3.04 | 64.7% | 0.962 |
| Supervised 2 | 6.21 | 2.79 | 69.9% | 0.966 |

Both families have similar concentration. JEPA's three context effective ranks
are 7.06–7.74, compared with 6.21–7.33 for supervised training. This does not show
a unique JEPA collapse. Action rank remains 2.48–3.85 for JEPA even within the
4-second-cost stratum, so cost mixture alone does not explain all concentration.
Low rank can also be appropriate for two correlated service targets; usefulness
must be tested separately.

The audit compares **361 pairs of actual candidate queries at each cost**,
holding the recorded scene, fork and focal robot fixed. No unrelated histories
or queries were swapped. Every such pair changes the action latent. JEPA's
average absolute miss-prediction difference is 14.0–19.7 commands at 30 ms and
10.3–14.7 commands at 4 s. These are meaningful responses, though responsiveness
alone does not establish correct preferences.

Derivatives are measured at actual public input points for pilot, cached-link,
pose/route, packet-outcome and action-query feature groups. They test local
response without pretending that arbitrary perturbed inputs are real rollouts.
The full report retains these quantities in native output units.

## The action latents contain packet-value information

The following missed-command MAEs concern the diagnostic training examples and
four-fold **readout** protocol above. They must not be described as new held-out
performance. Gap errors and both cost strata are in the JSON report.

| Frozen representation | Action-latent ridge MAE | Action latent + numerical prior MAE |
|---|---:|---:|
| JEPA 0 | 24.39 | 23.84 |
| JEPA 1 | 21.19 | 20.97 |
| JEPA 2 | 18.48 | 18.45 |
| Supervised 0 | 22.16 | 22.03 |
| Supervised 1 | 25.64 | 24.44 |
| Supervised 2 | 25.06 | 23.93 |

For context, the same diagnostic examples give MAE 49.49 for fold-training
per-cost means, 32.75 for the raw numerical estimator, and 47.87 for a fixed
linear ridge on flattened public histories and queries. The numerical estimator
predicts **miss deltas only**; its gap column is explicitly a fold-training
per-cost constant, never an invented numerical gap prediction.

The action-latent readouts retain useful task information. This does not show
that all useful RF information survived, that linear probes exhaust nonlinear
information, or that better action predictions translate to better closed-loop
operation. The stored trained heads were fit on all these examples; their lower
in-sample errors are not directly comparable to a probe's excluded-site readout.

## The JEPA future objective is weakly realized

The action latent is shared between the packet-value head and the auxiliary
future-latent objective. On these examples, the actual action latent has worse
future-target MSE **and Smooth L1 loss** than a constant global target mean.
The EMA future representations themselves are nonconstant, with effective rank
7.67–8.11. Therefore, this is not a constant-target collapse.

| JEPA seed | Actual future-latent MSE / constant MSE | Diagnostic context+query future readout MSE / constant MSE |
|---|---:|---:|
| 0 | 2.03 | 0.47 |
| 1 | 1.79 | 0.41 |
| 2 | 1.91 | 0.47 |

The linear readout predicts the same checkpoint's EMA targets, using public
context latents and action queries under the same site-fold protocol. This is
evidence that some future-target information is accessible while the deployed
joint predictor underuses it. It supports investigating objective competition,
the future-prediction readout and explicit action-conditioned dynamics. It does
not identify which intervention will improve control, establish that all future
targets are predictable, or estimate the true uncertainty floor. A separate
auxiliary prediction projection is a reasonable controlled experiment; assuming
that a different backbone or larger variance penalty solves the problem is not.

## High cosine in the untrained service encoder is not constant collapse

The service encoder's pooled belief has average raw pairwise cosine **0.9924**.
Yet its centered RMS is **0.0868**, its effective rank is **5.45**, and its first
principal component explains **46.4%** of centered variation. Removing cached
and pilot measurements changes its belief RMS by **0.259**, about 2.98 times its
ordinary between-window centered RMS. Removing global observations changes it
by **0.270**. Shared initialization and biases give a large common component;
the encoder still responds to its evidence.

Across actual alternative APs, its action-transition latent changes by RMS
0.0265. The residual prediction remains exactly zero because the final residual
head is intentionally initialized to zero. That is an initialization contract,
not a learned failure or learned success. Its zero action vector is an explicit
structural probe; no semantic action adapter or controller was evaluated here.

## Missing observations and unpredictability are separate limits

In the native saved windows, cached RSSI is observed for **17.5%** of all AP-band
tokens. Fresh pilot SNR is observed for **5.68%** of all AP-band tokens, but for
**90.9% of the currently configured AP-band tokens**. These denominators matter:
the receiver does not measure every AP simultaneously. Unobserved neighbor ages
are excluded with masks; they are not measured zeros. Completed map predictions,
geometry and configured load remain available beyond the heard subset.

The old packet API aggregates per-AP structure into 32 history features and 24
query features. It has coverage/known-band summaries, rather than the native
model's explicit per-token observation masks and freshness. No raw CSI,
neighbor-age measurements or measured channel busy time exists in that archive.
The recorded input features vary; there is no whole-input constant collapse.
Aggregation can nevertheless remove information before any encoder sees it.

Independent future fades and unobserved blockage events remain potential
uncertainty sources. This audit has one realized branch outcome per recorded
action; it cannot separate irreducible stochasticity from missing sensors,
finite-data error, objective mismatch or approximation error. Estimating that
floor requires repeated conditional rollouts or a registered information-oracle
study, not a representation cosine or a poor forecast score.

## How to use this evidence

Retain the physics prior and the explicit public observation contract. Before
changing the controller, isolate future-prediction learning from service-value
learning using the same backbone and information, then measure whether it
improves matched future prediction, calibration and actual native control.
Compare concentrated representations by downstream utility and centered
spectra, without forcing every coordinate to vary. Preserve local AP/band
features through action evaluation so that global pooling does not hide which
link changed. DINO-style dense representations or a larger architecture remain
hypotheses until they improve these measurements on a new registered protocol.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --frozen python scripts/audit_representations.py
uv run --frozen python -m pytest tests/test_representation_audit.py -q
```

The report has no wall-clock timestamps or stochastic sampling. Its manifest
hashes all 96 accessed source/data/receipt/configuration files. No original
checkpoint, archived source, simulator, data file or control policy is changed.
