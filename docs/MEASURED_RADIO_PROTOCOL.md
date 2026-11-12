# Frozen-model transfer to measured RSSI

This study tests whether the three released RadioJEPA members transfer to
measured Wi-Fi RSSI after sparse support conditioning. It compares them with
172 numerical configurations using the same support, reference and query
identities. It does not train on these measurements, choose a neural member,
change the SmartCar deployment default, or establish SOTA. This is a newly
registered assay on a previously audited public corpus; it is not a globally
blind dataset or an independent replication by another laboratory.

## Data rights and endpoint

Source: Miguel Matey-Sanz and colleagues, **UVIndoorLoc-RTT&RSSI**, v1 (2026),
[DOI 10.5281/zenodo.20051265](https://doi.org/10.5281/zenodo.20051265),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The official record,
attribution, archive SHA256 and six coordinate-catalogue SHA256 values are
retained with registration. The downloadable archive is 32,899,133 bytes,
SHA256 `0ff469b8bd9d58a7d37c77423fc40450916bf26bbe208010bfc2bdf075ccf9f5`.
Prepared summaries and selected reported values are derived from that source;
their measurement data retain its attribution and license.

The endpoint is prediction of an arithmetic mean of up to eight
identity-selected **reported RSSI dB values** for one file/AP pair. Each source
result is already a hardware-reported average. These are not eight independent
packets, a linear-power average or a standard error. Eligibility requires
successful status, positive successful-measurement count, finite non-sentinel
RSSI and explicit 5180 MHz metadata. Failed or unheard APs supply no target.
There is no invented detection threshold. The source's POCO frequency metadata
are unsupported, so the eligible official cohort uses Pixel 10 and S24 Ultra.

The audit identified 1,644,814 source rows, including 55,686 failed results and
one status-success result with zero successful measurements. The planned
eligibility checks require 758,456 eligible reports, 4,255 file/AP summaries
and 33,901 selected reports. These are pre-execution audit counts; actual
preparation must match them or stop.

The first published reader stopped during preparation because it additionally
assumed successful counts could not exceed attempted counts. Eight source
rows report positive successes and zero attempts; five otherwise satisfy
eligibility. No summary was saved and no measured-data fit/prediction ran.
The original protocol/source/failure are preserved under `artifacts/measured_radio`.
The separately published counter-contract repair uses
`artifacts/measured_radio_counterfix`, preserves the raw counters and flags
these rows rather than inferring that they are invalid RSSIs. The registered
positive-success eligibility, split namespace, support, models, grid and metric
are unchanged. These counters are not interpreted as independent packet counts.

## Support, splits and unknown references

Provider TEST locations remain TEST. A metadata-only SHA256 rule takes 29 of
the 141 provider TRAIN locations for VAL, leaving 112 TRAIN locations. Every
phone, AP and session at a location stays together. No RSSI chooses a location.

Budgets are 4, 8 and 16 receiver locations per scenario/phone, selected as nested
maximin prefixes from TRAIN coordinates with fixed hash initialization/ties.
All phones use the same planned locations. No missing file is replaced;
S24 Ultra has 15 actual support locations in scenario S03 at budget 16.
Sixteen locations can supply up to 128 summary contexts and 1,024 selected
reported results. Support always comes from session `01`.

Absolute TX power and device calibration are unknown. For each logical AP and
phone, the common support-only reference is

```text
c_a = mean_support(reported_RSSI + free_space_loss)
residual_i = reported_RSSI_i + free_space_loss_i - c_a
query reference = c_a - free_space_loss_query
```

This absorbs mean environmental attenuation as well as device/TX offsets.
It does not identify either calibration. The same reference is used by every
arm, and its estimation uncertainty is absent from the working model spread.
An AP lacking admitted support remains explicitly unsupported.

The model receives AP/receiver XYZ and a computational frame from the full
public coordinate catalogue, including held-out query geometry. It has no CAD,
material, detector, TX or noise-floor measurements. Two active CAD features
are explicitly zero-imputed; the frozen model has no CAD-unknown indicator.
Inactive unknown input slots use compatibility padding. At one carrier and
zero CAD, the global physical basis has rank at most two. The Fourier and
discrepancy components remain active. This is **partial-input transfer**.
The frame preserves distances; the full predictor is not translation invariant.

## Comparators and frozen decisions

The numerical grid contains a support-reference-only arm, fitted log-distance,
ordinary kriging with RBF/Matérn-1/2/Matérn-3/2 kernels, and IDW. Kriging uses
six lengths, three amplitudes and three noise values per kernel. IDW uses four
neighbor counts and two powers. There are 172 configurations in total.

For each budget, one global configuration wins by VAL/session01 score, with
registered-order tie breaking. Six family winners are retained for TEST.
All numerical VAL predictions and all three frozen neural members are archived.
TEST runs even when neural VAL results are unfavorable. There is no neural
weight, member or hyperparameter selection.

Primary score: RMSE across observed AP summaries within phone/location,
equal phones within location, equal locations within scenario, then equal three
scenarios. RadioJEPA computes RMSE **for each frozen member first**, then
averages those scores. It does not score an ensemble-mean prediction.

Primary contrast is budget16/session01 RadioJEPA minus the VAL-selected best
numerical configuration. A paired 20,000-draw location bootstrap within fixed
scenarios uses seed 40517. A separate 5 m XY block sensitivity uses seed 40518,
preserves block sizes/multiplicities and requires four occupied blocks per
scenario. Both upper interval bounds must be below zero to support the narrow
conditional lower-error label. Weights, support and members stay fixed. These
are empirical resampling intervals, not independent-site population evidence.

Budgets 4/8, individual members/scenarios and session02 are descriptive.
Session02 uses the original session01 map/reference. Its phone-specific dates
are not simultaneous, and the measurements are not an autonomous trajectory.
Coverage reports every planned TEST location/phone and observed/supported AP
count. Working-spread coverage is descriptive and is not a calibrated hardware
confidence probability.

## Reproduction and interruption safety

The machine-readable protocol, public catalogue, code capsule, exact three
weight exports and source receipt are committed and pushed **before** the
official source is aggregated or any measured-data fit/prediction runs. A Git
publication receipt binds the registered source commit. Preparation,
predictions and TEST selection each bind the protocol and dataset. Resume
replays support/query identities and selection without refitting existing
groups, checks hashes and verifies deterministic report/seal contents.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --frozen python scripts/run_measured_radio.py register
# Commit and push exactly the new protocol/source before proceeding.
PYTHONPATH=src uv run --frozen python scripts/run_measured_radio.py \
  publish-proof --source-commit <published-source-commit>
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --frozen python scripts/run_measured_radio.py run
PYTHONPATH=src uv run --frozen python scripts/run_measured_radio.py verify
```

The source archive/metadata paths are pinned in `measured_radio/common.py`.
Raw source data remain outside the repository; derived evidence goes under
`artifacts/measured_radio_counterfix`. Existing study seals, weights, deployment ZIP and
unrelated deferred work stay unchanged. An independent audit must replay the
raw results before the outcome is promoted into documentation.

Three scenarios reuse eight physical AP units across 20 logical placements.
They are not three independent warehouses. No outcome here establishes
SmartCar switching, fleet reliability, packet performance, interference-map
accuracy, cross-protocol operation or SOTA. Those need their own controlled
measured endpoints and deployment trials.
