# Optional AP-local radio field: completed RF results

The validation-selected `nested_joint` model achieves **2.637330 dB** sparse
heard-RSSI RMSE against **2.710522 dB** for the strongest validation-selected
numerical reference, `hierarchy-fixed`: **2.70% lower error**. All four
predeclared, multiplicity-adjusted primary intervals favor their candidates.
The joint-view addition contributes only **0.002226 dB** over nested regression;
that small statistical effect does not establish a useful service improvement.
All eight learned local geometry channels are gated off in the selected nested
members. The improvement supports a shrunk fixed-feature local residual around
the shared physical model, not a rich learned geometry representation.

This is a completed, independently replayed **RF-only** experiment. Fifteen fits
with three seeds per cell, 84 fresh actual Sionna RT cases and all 5,928 field
prediction records are retained. It runs **zero new warehouse simulations**
and performs **zero hardware actions**. RadioMap remains the application default;
original SERVICE holdouts remain closed.

The [architecture](ARCHITECTURE.md), [registered plan](PLAN.md),
[machine-readable report](../../artifacts/nested_radio/report.json),
[selection](../../artifacts/nested_radio/selection.json),
[independent replay](../../artifacts/nested_radio_review/independent.json) and
[interactive 3D viewer](../../artifacts/nested_radio_review/viewer/index.html)
provide the implementation, raw references and complete comparisons.

## Training and evaluation boundary

| Item | Completed scope |
| --- | --- |
| Fitting data | Exactly 24 TRAIN and six VAL geometries, three static states each: 90 reused physical RT cases, plus TRAIN-only relative calibration. No previous TEST/DEV fitting or selection. |
| Neural cells | `global8_reg`, `global8_joint`, unit-scale `hier_reg`, `nested_reg`, `nested_joint`; independent seeds 0/1/2 per cell. |
| Fixed fitting budget | 160 epochs × 12 updates = 1,920 committed updates per fit; 28,800 total. Four distinct layouts per batch, alternating 192/96-token contexts. |
| Archived validation | Weights and actual sparse view-A predictions at epochs 40/80/120/160: 60 weight snapshots and 60 raw prediction archives. First minimum equal-state/equal-layout heard-RSSI RMSE selects each member. |
| Cell selection | Mean all three selected member VAL RMSEs; fixed tie order among the two global and two nested cells. Unit-scale hierarchy is a retained control. |
| Numerical selection | All four frozen TRAIN-fit priors retained, with no refitting: RadioMap operators 5/10/20 m and fixed-feature hierarchy. Same sparse VAL RMSE criterion selects `hierarchy-fixed`. These operators are not the bit-identical native full-history RadioMap. |
| Fresh physical cases | 24 TEST + four eight-AP transfer geometries, each with three static material states: 84 actual CPU LLVM Sionna RT cases. Geometry fingerprints differ from all 240 previous public layouts. |
| Observation budget | 48 receiver poses, checkerboard 24-context/24-query split; full/sparse source contexts 192/96 total tokens. Eight-AP transfer keeps the total cap, reducing tokens per AP. |
| Frequencies and reporting | 2.4/5.2 GHz source carriers; 6.0 GHz held target. Two receiver offset/quantizer profiles, 0.75 dB receiver noise and genuine measured top-K=2 reporting per band. |
| Physical solve | Depth 3, 32,768 samples/source, 4,096 path cap and four DrJit threads. Finite search is not a convergence certificate. |
| Retained predictions | All 15 trained members plus all four numerical references; 5,928 records across full/sparse, held-band, stale-context and AP-transfer groups. |

Heard quantization intervals, ranked-unheard upper bounds and unknown unresolved
paths remain different events. Unheard APs are never invented floor-valued
measurements. Query cutoffs, private materials and true channel coefficients are
not model inputs. Material changes are static counterfactuals, with geometry,
TX and solver/noise streams shared within a geometry; they are not timed events.

Every fit completes the full budget. Selected epochs are 120/120/80 for both
global cells and both nested cells, and 120/120/120 for `hier_reg`. Validation
selects an earlier checkpoint without establishing convergence or identifying
which budget, selection or architectural choice caused a gain. This cohort and
selection rule differ from other studies; cross-study point estimates are not
paired improvements.

## Four primary comparisons

RMSE is computed per member and state, then averaged within each of 24 TEST
geometries. The 20,000 paired whole-geometry bootstrap draws use seed 21771.
Central **98.75%** intervals adjust the four-family comparison; uncertainty is
conditional on these three trained members, rather than arbitrary retraining.
Negative candidate-minus-reference values favor the candidate.

| Candidate − reference | Candidate RMSE | Reference RMSE | Difference, dB | Adjusted interval, dB |
| --- | --- | --- | --- | --- |
| `selected` − `best_numerical` | 2.637330 | 2.710522 | -0.073192 | [-0.126150, -0.023013] |
| `selected` − `global8_reg` | 2.637330 | 2.651669 | -0.014339 | [-0.023057, -0.005348] |
| `nested_reg` − `hier_reg` | 2.639556 | 2.827782 | -0.188225 | [-0.233787, -0.147843] |
| `nested_joint` − `nested_reg` | 2.637330 | 2.639556 | -0.002226 | [-0.004345, -0.000883] |

The nested-versus-unit-scale comparison tests the **gate initialization,
trainable scales and effective-prior regularizer together**. The coefficient
prior floor has not been removed. The 0.002226 dB agreement gain is about 0.084%
of nested regression's RMSE. This supports a small matched effect under this
protocol, not a practical JEPA, packet-delivery or specialist-SOTA claim.

## Every control and descriptive stress group

All entries below are mean heard-RSSI RMSE in dB. Only the sparse four-AP
comparisons above have the registered adjusted intervals. Four-AP groups contain
24 geometries; eight-AP groups contain four. `nested_joint` is selected on VAL,
not because of any TEST or secondary result.

| Method | 4 AP full | 4 AP sparse | Held 6 GHz sparse | Stale sparse | 8 AP full | 8 AP sparse |
| --- | --- | --- | --- | --- | --- | --- |
| `global8_reg` | 2.621399 | 2.651669 | 2.803276 | 2.785601 | 2.427498 | 2.506870 |
| `global8_joint` | 2.621343 | 2.649823 | 2.806534 | 2.780272 | 2.428204 | 2.507175 |
| `hier_reg` | 2.765341 | 2.827782 | 3.071886 | 2.991998 | 2.613203 | 2.797628 |
| `nested_reg` | 2.609300 | 2.639556 | 2.789258 | 2.773804 | 2.407956 | 2.496387 |
| `nested_joint` | 2.609102 | 2.637330 | 2.792086 | 2.767937 | 2.408953 | 2.496645 |
| `hierarchy-fixed` | 2.710049 | 2.710522 | 2.872619 | 2.828914 | 2.475336 | 2.562213 |
| `radiomap-5m` | 2.757280 | 2.793917 | 3.029139 | 2.879191 | 2.648606 | 2.744583 |
| `radiomap-10m` | 2.745917 | 2.812673 | 3.042474 | 2.902275 | 2.641657 | 2.762689 |
| `radiomap-20m` | 2.703904 | 2.788426 | 3.000088 | 2.876511 | 2.630079 | 2.773594 |

The strongest primary numerical reference is fixed hierarchy, although the
20 m RadioMap operator has slightly lower full-context error. The selected
model's sparse error is also 5.42% below that 20 m operator, but the primary
comparison uses the stronger validation-selected numerical reference.
Joint agreement is not uniformly better than nested regression: its mean error
is higher in both eight-AP groups and the held-6-GHz group. These are descriptive
outcomes; neither a universal transfer win nor a separate confirmatory test is
claimed.

## Tail error, likelihood, coverage and material differences

Sparse four-AP results below retain all methods. MAE and p95 are computed inside
member/state/geometry before reduction; p95 is **not** a pooled fleet tail or
maximum. Coverage is the heard target-center fraction within predicted mean
±1.959964 SD; it does not cover unknown links or certify command risk. Mean
predicted SD includes all query links. Composite NLL uses genuine known interval
and censor events, omitting unknowns; it is not a joint link likelihood.

| Method | MAE, dB | Mean p95, dB | Composite NLL | Heard 95% coverage | Mean SD, dB |
| --- | --- | --- | --- | --- | --- |
| `global8_reg` | 2.141794 | 5.042079 | 1.294557 | 95.671% | 2.836671 |
| `global8_joint` | 2.141368 | 5.035784 | 1.294848 | 95.758% | 2.832035 |
| `hier_reg` | 2.271963 | 5.414480 | 1.314256 | 96.473% | 3.254675 |
| `nested_reg` | 2.128196 | 5.034243 | 1.293857 | 95.651% | 2.824651 |
| `nested_joint` | 2.127533 | 5.027842 | 1.294127 | 95.705% | 2.819641 |
| `hierarchy-fixed` | 2.181721 | 5.159271 | 1.312998 | 97.515% | 3.294990 |
| `radiomap-5m` | 2.245163 | 5.409600 | 1.340439 | 98.182% | 3.603368 |
| `radiomap-10m` | 2.254536 | 5.459985 | 1.334075 | 97.614% | 3.427184 |
| `radiomap-20m` | 2.234894 | 5.372624 | 1.326303 | 96.990% | 3.278146 |

The numerical references have wider predicted spreads and higher heard coverage.
Greater coverage alone is not better calibration. Joint agreement has slightly
worse composite NLL than nested regression despite its lower primary RMSE.
No packet latency, outage, silence probability or SLA is calibrated here.

The material-difference diagnostic measures prediction-difference error against
reported differences for links heard in both baseline and changed states; it
combines both static interventions within each member and geometry.

| Method | Material-difference RMSE, dB |
| --- | --- |
| `global8_reg` | 0.910491 |
| `global8_joint` | 0.909797 |
| `hier_reg` | 0.931754 |
| `nested_reg` | 0.907007 |
| `nested_joint` | 0.906549 |
| `hierarchy-fixed` | 0.906270 |
| `radiomap-5m` | 0.950159 |
| `radiomap-10m` | 0.946799 |
| `radiomap-20m` | 0.954476 |

Fixed hierarchy's 0.906270 dB difference error is slightly below the selected
model's 0.906549 dB. The primary point-error improvement therefore does not
establish superior material-change tracking. No event-time transition, autonomous
novelty detector or stable multistep rollout is trained in this experiment.

## Gates, effective priors and representation diagnostics

Sixteen static feature scales are shared across APs/queries in each trained
member. They are fixed at inference, not a context-dependent admission gate.
Nested cells allocate 1,562 parameters versus 1,546 for controls; four-AP
hierarchical inference uses 72 scene coefficients versus eight for global-eight.
The local design is scaled in both context conditioning and query rendering.
Zero scales reproduce the global-eight function **with the same remaining
weights**, up to roundoff. Function-class inclusion is not equivalence to an
independently trained global model or a guarantee of held-out noninferiority.

Scales are direct parameters projected into [0,1] after optimizer steps. The
unchanged positive local prior SD remains trainable. The fixed nested penalty is
`0.01 × mean(alpha × sqrt(mu_local² + sigma_local²) / 0.5)`; effective mean,
SD and RMS products prevent reporting small gates as shrinkage while silently
compensating with large priors. This coefficient-space penalty depends on basis
scale; it is not a physical field-energy penalty. Zero-channel reactivation is
not guaranteed.

| Cell | Gate range | Mean gate | Zero gates by seed | Mean effective prior RMS, dB | Ungated learned-8 rank | Effective local-16 rank |
| --- | --- | --- | --- | --- | --- | --- |
| `global8_reg` | 0.000000…0.000000 | 0.000000 | 16/16/16 | 0.000000 | Inactive | 0.000 |
| `global8_joint` | 0.000000…0.000000 | 0.000000 | 16/16/16 | 0.000000 | Inactive | 0.000 |
| `hier_reg` | 1.000000…1.000000 | 1.000000 | 0/0/0 | 5.020256 | 1.003 | 7.442 |
| `nested_reg` | 0.000000…0.119827 | 0.026338 | 11/11/10 | 0.140684 | 1.056 | 3.731 |
| `nested_joint` | 0.000000…0.113951 | 0.025662 | 11/11/10 | 0.138098 | 1.056 | 3.759 |

Gate range spans all 48 gate values across three members; means average members
and features. RMS is the effective **coefficient-prior** RMS rendered in dB,
before query features, conditioning and numerical jitter. It is neither physical
RF energy, an identified material, a predictive-SD floor nor a query uncertainty
bound. The mean fixed-Fourier-eight participation rank is 6.734 for every cell.

Diagnostics use archived 1,152×16 public query-geometry features per selected
member from six VAL geometries in state zero; no TEST inputs or selection use.
The displayed ranks are centered participation ranks averaged over three members.
The eight learned geometry features remain close to rank one in the trained
local cells; the record does not justify a general "collapse solved" claim.
More decisively, **all last eight learned-feature gates are exactly zero in
all six selected nested regression/joint members**. Their active fixed-Fourier
channel counts are 5/5/6 by seed in both cells. The raw learned features therefore
do not contribute to these final nested predictions. This pruning can reflect
the regularizer and does not establish its cause or whole-model collapse: the
remaining physical/Fourier field still improves the primary prediction metric.
Feature variance and rank do not measure RF observability or identified physical
states. Global controls' local kernels are unused and untrained; their effective
zero rank is by design and their raw kernel variation is not a trained RF
representation. [Raw diagnostics](../../artifacts/nested_radio/validation_representation.json)
retain singular values, feature variance, all gate/prior products and checkpoint
hashes. Joint agreement uses no EMA teacher and is not classic JEPA.

## Reproducibility and release boundary

Scientific source commit `328c9d1292c216eb973862988f5656b10651bcc6` was published
before fitting and new scientific solves. The seal binds **578 source** and
**2,770 evidence** files. Its manifest SHA-256 is
`f5e7d7c2947119b5c282c626dbce0a4cef387a47a01110dd63415b6b643978cd`.
This is hash-bound provenance, not a cryptographic signature or external
attestation.

The first actual independent replay passes without scientific reruns, retries
or parser changes. It rejoins raw reports/predictions, recomputes selections,
all reductions and four intervals, and checks every source/evidence hash in
all ten study closures. It checks 27,295 unique named files, all 15 fit schedules,
60 epoch predictions and weights, 15 representation archives, and all 5,928
field records. Deserializing checkpoints does not retrain or execute the model.
The [software receipt](../../artifacts/nested_radio_review/software_suite_importlib.json)
records **3,633 passing tests**, 153 modules and 575 source/config/test files
unchanged across execution; this is distinct from the 578-file scientific seal.

The [viewer review](../../artifacts/nested_radio_review/viewer/browser_review.json)
checks its 217 hash-bound inputs and all 576 geometry/carrier/AP/view choices,
four primary contrasts, nine method summaries, 45 stress values, 60 checkpoint
scores and 240 gate/prior products, without scientific calls or external network
requests. It displays actual reported RF targets and public geometry.

These results support an optional, strongly shrunk local field for the stated
simulated RF assay. They do not qualify the new weights for warehouse service.
The separate [native-system evidence](../hierarchical_radio/RESULTS.md) uses
different fixed checkpoints, retains service harms and cannot be inherited by
this model. [Remaining validation](REMAINING.md) sets out the next finite checks.
