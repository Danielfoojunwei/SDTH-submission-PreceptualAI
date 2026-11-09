# Real TUJI1 phone and location transfer

With 16 labeled target-phone positions, AP-specific calibration reduced mean
localization error from **4.516 m to 4.225 m** versus unadapted source kNN,
a **6.45%** reduction. A single global correction reached **4.233 m**.
The small difference between those calibration methods is not an established
advantage, and the best mean did not give the best tail behavior.

These are **classical real-data baselines**, not RadioJEPA encoder-transfer
results. No warehouse service, production, hardware-inverse or global SOTA
claim follows from this experiment.

## What was measured

The licensed [TUJI1 dataset](https://doi.org/10.5281/zenodo.7641701) supplies
6,752 training and 2,147 testing fingerprints over 310 APs on one mapped floor.
Each of five folds excludes one phone from source fitting and evaluates that
phone on the official test reference coordinates. Global train/test coordinate
disjointness was checked. These are new reference points within a densely
mapped floor, not a new building or floorplan.

N=0/1/4/16 counts additional **labeled target-phone positions**, with one actual
fingerprint per position. Three fixed seeds choose nested support prefixes
from the target phone's official training positions. Each fold already has
5,344–5,439 source fingerprints from the other four phones. The few-shot count
does not describe the cost of building that source map.

The source and [protocol](../../artifacts/device_transfer/protocol.json) were
published at `95a70de`, with the publication proof at `5e18fea`, before fitting.
All fits were saved before evaluation RF opened. Predictions were saved before
the separate reference-XYZ reader computed localization errors. No synthetic
RSSI shifts or artificial devices were used.

## Complete mean-error curve

Values are meters: each phone's mean error, averaged over the three support
draws, then equally averaged over the five phones. Source controls are computed
once per phone and reused; their repeated cells are not additional fits.

| Method | N=0 | N=1 | N=4 | N=16 |
|---|---:|---:|---:|---:|
| Source masked kNN | 4.516 | 4.516 | 4.516 | 4.516 |
| Source ridge | 4.413 | 4.413 | 4.413 | 4.413 |
| Source kNN + target support | 4.516 | 4.516 | 4.513 | 4.503 |
| Global-offset kNN | 4.516 | 4.257 | 4.248 | 4.233 |
| AP-shrinkage-offset kNN | 4.516 | 4.279 | 4.256 | 4.225 |

Adding 16 target fingerprints directly to kNN changed its mean by only
**0.013 m**; this is a small descriptive result. Global calibration obtained
most of its gain with one labeled position. Moving from N=1 to N=16 changed
that mean by only another **0.024 m**. AP-specific calibration was worse than
the global correction at N=1 and N=4, and better by only **0.008 m** at N=16.
The support-draw standard deviations of the balanced means at N=16 were
0.012 m for global correction and 0.014 m for AP shrinkage; these are sampling
spread across three draws, not confidence bounds or independent training runs.

## Tails and device differences

The following values average each phone's **95th-percentile** position error
over draws and phones. They are neither pooled 95th percentiles nor confidence
limits.

| Method | N=0 | N=1 | N=4 | N=16 |
|---|---:|---:|---:|---:|
| Source masked kNN | 10.180 | 10.180 | 10.180 | 10.180 |
| Source ridge | 9.301 | 9.301 | 9.301 | 9.301 |
| Source kNN + target support | 10.180 | 10.180 | 10.180 | 10.180 |
| Global-offset kNN | 10.180 | 9.644 | 9.750 | 9.507 |
| AP-shrinkage-offset kNN | 10.180 | 9.871 | 9.773 | 9.654 |

More support did not improve every endpoint monotonically: global correction's
N=4 tail was worse than its N=1 tail. Source ridge had the lowest average of
phone-level 95th percentiles, despite its weaker overall mean than calibrated
kNN. Ridge does not use the kNN coverage fallback, so its zero fallback count
does not establish radio coverage.

Mean errors by held-out phone, with calibrated methods at N=16:

| Held-out phone | Source kNN | Source ridge | Global correction | AP shrinkage |
|---|---:|---:|---:|---:|
| Galaxy S20 | 4.233 | 3.909 | 4.116 | 4.163 |
| Galaxy S7 | 4.358 | 3.389 | 4.357 | 4.323 |
| POCO X3 | 4.131 | 3.610 | 4.089 | 4.022 |
| Galaxy Tab S7 | 5.105 | 3.862 | 3.877 | 3.900 |
| Galaxy A12 | 4.753 | 7.297 | 4.725 | 4.716 |

Ridge had lower mean error on four phones and failed badly on A12. About **83%**
of the AP-calibration mean reduction came from the Tab S7 fold. The global
correction's S7 mean was effectively unchanged. AP shrinkage also worsened
two phone-level tails relative to source kNN: POCO **8.355 → 8.759 m** and
A12 **11.958 → 12.104 m**. Its positive average therefore does not establish
uniformly better reliability. One of A12's 458 test fingerprints used the
declared insufficient-shared-AP centroid fallback; calibration retained that
missingness and fallback.

## Interpretation and verification

The result supports a narrow engineering observation: a few labeled,
co-located observations can improve a classical fingerprint localizer for
some device shifts, while the choice of baseline and tail metric matters.
It does not isolate analog-front-end calibration. The phones were surveyed
on consecutive days; source/target residuals combine device, body, day and
scene changes. The formatted corpus has no session timestamps, so this is
held-device and held-reference-location evaluation, not a session-held-out
experiment or simultaneous-view representation learning.

All five folds share one physical site and overlapping source references.
The three seeds vary support selection; they are not independently pretrained
models or independent sites. No significance, universal-device or new-layout
claim is made. No RadioJEPA encoder was evaluated in this track, and no
command loss, silence, handover timing or real-time compute budget was measured.

The [sealed report](../../artifacts/device_transfer/report.json) retains all
300 case rows and 20 aggregate rows. Its [closure](../../artifacts/device_transfer/verification/manifest.json)
binds 487 source and 514 evidence files. A separate
[read-only review](../../artifacts/device_transfer_review/review.json), stored
outside the frozen study, passed:

- Independent actual-XYZ metric checks for all 300 cases and all 20 aggregates.
- All 60 calibration/identity checks, including the 45 nonzero-support fits.
- 420 independent scalar kNN inference spots and five ridge normal-equation checks.
- The canonical complete prediction-arithmetic audit and final byte closure.

No new fits or experiment prediction streams were generated by the review;
all frozen study bytes stayed unchanged. The
[review script](../../artifacts/device_transfer_review/script.py) and its input
hashes make these checks inspectable.
