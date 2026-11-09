# Real device and location transfer

This finite real-data track evaluates all five TUJI1 phones. It complements
the simulated spectral model and does not qualify warehouse control.

The licensed source is [TUJI1](https://doi.org/10.5281/zenodo.7641701), acquired
from Zenodo with its publisher checksum and an immutable SHA256. It contains
6,752 official training fingerprints and 2,147 testing fingerprints, with 310
AP columns and explicit device IDs. The acquisition audit already inspected
public shapes and aggregate test properties; no localization scores were
computed. The new protocol is registered before any fit or new test view.

For each held-out phone, train on the other four phones' official training
rows. Calibration receives N=0/1/4/16 distinct labeled positions from that
phone's training rows. Three fixed seeds select nested support prefixes.
Evaluation uses only the target phone's official testing locations. Predictions
are saved before the separate test-XYZ API opens labels; the audit rejects any
train/test coordinate overlap.

| Method | What it tests |
|---|---|
| Masked source kNN | Unadapted cross-phone fingerprint localization |
| Source ridge | A deterministic learned linear reference with train-only scaling |
| Source kNN plus N target fingerprints | Direct use of the same calibration budget |
| Global-offset kNN | Whether a robust additive RSSI correction helps |
| AP-shrinkage-offset kNN | Whether AP-specific corrections improve on one global shift |

The +100 marker remains unknown. kNN uses shared observed APs, with a declared
coverage requirement and an explicitly counted centroid fallback. Calibration
uses real AP observations at exact co-located training coordinates; no noise,
RSSI shifts or devices are generated to make the result favorable.

The devices were surveyed over consecutive days. Co-location references are
therefore supervised calibration examples, not simultaneous radio-scene
positive pairs. Estimated corrections combine device, body, day and scene
effects. This experiment cannot identify a pure hardware inverse.

There is one physical site. Five overlapping leave-device-out folds and three
support draws do not become15 independent sites. Source kNN/ridge are
deterministic; the seeds quantify support selection, not pretrained neural
run variance. Report every fold, source baseline, support curve, tail error and
coverage failure, including negative results. Do not promote this endpoint to
a warehouse packet-service, RadioJEPA-necessity or global SOTA claim.

The machine-readable [protocol](../../artifacts/device_transfer/protocol.json)
fixes all methods and hyperparameters before fits. All prior475 source and5161
spectral evidence entries stay unchanged. New source and input hashes must be
published before the runner can open evaluation fingerprints or fit baselines.

`DEVICE_TRANSFER_DATA_ROOT` can relocate the directory containing
`7641701-DATASET.zip`, or the CLI accepts `--archive`. The exact source SHA256
is always enforced. Cached fits are checked against current source, training
rows, normal equations and support orders before test RF opens; cached
predictions are rebound to phone, fit and actual test rows before XYZ opens.
