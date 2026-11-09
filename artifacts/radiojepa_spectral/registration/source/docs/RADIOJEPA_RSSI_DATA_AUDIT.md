# RadioJEPA real RSSI, RTT and multisensor data audit

Acquisition date: 9 November 2026. This audit inspected official data and primary
descriptions before training. It generated no model fit, localization result or
warehouse service result. Public HTTPS sufficed; configured `gh` access was used
for repository metadata without displaying credentials.

The local mirror is `/workspace/upstream/scene-radio/rssi/`. Its `manifest.json`
binds **17 downloaded files / 921,696,230 bytes** with SHA256. Fourteen files also
match their publisher's Zenodo MD5. `inspection.json`, `inspection-extra.json`
and `inspection-archives.json` record actual shapes and quality findings. Large
archives remain compressed; TUJI1 RAW alone expands to about 2.48 GB.

## Acquired sources and permitted uses

| Source | Actual acquired content | Rights verified | Suitable role and limit |
|---|---|---|---|
| [UJIIndoorLoc, UCI 310](https://archive.ics.uci.edu/dataset/310/ujiindoorloc) | 19,937 training + 1,111 validation rows, 520 AP columns; phone/user IDs, locations, timestamps | Official UCI page: CC BY 4.0 | Real cross-device/time localization probe; not a synchronized cross-device channel corpus |
| [TUJI1, Zenodo 7641701](https://doi.org/10.5281/zenodo.7641701) | Raw, formatted and final datasets; 6,752 training + 2,147 test fingerprints, 310 APs, five labeled devices | CC BY 4.0 | Device inversion, detection bias and location-held-out transfer. Phones were surveyed across consecutive days, not simultaneously |
| [Tampere campus 2017, Zenodo 5174851](https://doi.org/10.5281/zenodo.5174851) | TIE1: 10,633/50 train/test, 613 APs. SAH1: 9,291/156, 775 APs | CC BY 4.0 | Additional real buildings and localization protocols; no device labels or synchronized multiband pairs in these CSVs |
| [Long-term UJI library, Zenodo 3748719](https://doi.org/10.5281/zenodo.3748719) | Version 2.2, 25 monthly folders, 170 RSS files, 104,160 parsed rows; supporting shelf geometry and nearby-AP information | Internal README: **data CC BY 4.0**, code MIT, documentation CC0; repository metadata alone says MIT | Drift and environment persistence. Do not call all contents MIT. The parsed count differs from the record's advertised 103,584; retain campaign membership explicitly |
| [UVIndoorLoc-RTT&RSSI, Zenodo 20051265](https://doi.org/10.5281/zenodo.20051265) | Raw AP ranging results, 244,748 combined rows, three phones, three scenarios, 20 logical APs and 228 reference locations; baseline package | CC BY 4.0 in record and internal README | Strongest acquired RSSI/FTM source: both quantities come from the **same AP ranging result**, with request/completion clocks and device information |
| [RTT versus RSSI reproducibility package, Zenodo 21128007](https://doi.org/10.5281/zenodo.21128007) | Actual algorithm source and published result tables, 99.8 MB | Code Apache 2.0. Results README specifies ODC Attribution; its ODbL badge is inconsistent, so do not relabel all contents Apache | Reproduce named kNN/MLP/trilateration comparisons. This package is **not** the raw UV dataset |
| [Hybrid Wi-Fi/Bluetooth/magnetometer, UCI 402](https://archive.ics.uci.edu/dataset/402/hybrid+indoor+positioning+dataset+from+wifi+rssi+bluetooth+and+magnetometer) | 1,539 rows, 67 semicolon-separated fields: 32 Wi-Fi RSSI columns and 22 Bluetooth presence columns, position and magnetic fields | Official UCI page: CC BY 4.0 | Multimodal slow-scene observations; Bluetooth fields are presence, not BLE RSSI or CSI. Same-kind phones limit device-transfer evidence |
| [Wristband/smartphone, UCI 377](https://archive.ics.uci.edu/dataset/377/geo+magnetic+field+and+wlan+dataset+for+indoor+localisation+from+wristband+and+smartphone) | Two 325-row phone Wi-Fi tables, phone/watch inertial and magnetic streams, position intervals | Official UCI page: CC BY 4.0 | Timestamped complementary sensing, **not two radio receivers**. The watch stream is byte-identical across the two campaign filenames; avoid duplicate-stream leakage |
| [IPIN 2026 Track 3, Zenodo 23037701](https://doi.org/10.5281/zenodo.23037701) | 177.9 MB archive: 72 training logs, four validation and four scoring trials; two phones, Wi-Fi/BLE/inertial/magnetic observations; technical annex | CC BY 4.0 | Real irregular mobile sensing and held-out trajectories. Reuses 2025 captures, so do not count the two editions as independent data. Floorplans require a separate request |

The UTSIndoorLoc author repository,
[XudongSong/UTSIndoorLoc-dataset](https://github.com/XudongSong/UTSIndoorLoc-dataset),
was found and inspected at the metadata level. GitHub reports no license and its
README supplies no grant. It remains a catalogued candidate, excluded from the
training-ready pool pending clarification of rights. No author was contacted.

Additional official searches covered UCI indoor datasets and Zenodo RTT,
Tampere and localization records. Those responses are retained locally,
including metadata for RTT Android comparison 15391797, spatially averaged
Tampere 1161525, and floor-detection 15384613. This is a bounded, recorded source
search, not a claim that the web or every possible benchmark was exhausted.

## View eligibility comes from measurements

**UV RSSI and FTM:** the raw rows contain `Status`, distance and standard
deviation in millimeters, attempted/successful measurements, average RSSI,
request start/completion UNIX times, and time since system boot. Android 14+
devices additionally expose frequency and bandwidth metadata; unsupported
fields stay unknown. The combined CSV's per-AP completion columns use the
request callback time; do not confuse that with exact RF capture time.

These are genuine paired modalities for one AP measurement. They do not prove
cross-phone simultaneity, channel reciprocity, or simultaneous 2.4/5/6 GHz
reception. The raw sampling must retain clocks and acquisition age. The
nominal app setting was 10 Hz, but the provider explicitly warns that actual
rates vary substantially.

All 944 raw files were parsed for acquisition metadata: 1,644,814 AP rows,
1,589,128 status-success results and 55,686 failure rows. Every known positive
carrier frequency is **5,180 MHz** (758,457 rows); 830,669 rows require Android
14+ metadata support and two return a zero frequency. Unknowns stay unknown.
This source cannot validate real 5-to-6 GHz transfer. One success-status row
has zero successful measurements, so the accepted paired-measurement filter is
explicitly status zero **and** a positive success count, not status alone.

**TUJI1:** its [open primary descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC11004521/)
states that each device was manually surveyed on consecutive days, stopping
at positions for at least ten seconds and selecting readings from the three
seconds before a location tag. Equal coordinates therefore support a
location-conditioned or slow-geometry comparison, not positive pairs for an
identical instantaneous radio scene. The A12 lacks 5 GHz support; its missing
APs must not become invented weak-signal observations.

**IPIN:** the technical annex distinguishes delivery `AppTimestamp` from
capture `SensorTimestamp`; Wi-Fi scans arrive roughly every 3.95 seconds.
Matching trial names across phones are insufficient for fast-scene pairing.
One inspected pair starts five seconds apart. Cross-device clock and pose
alignment must be established before using it as a synchronized positive pair.

For all localization probes, coordinates, device-specific position tags and
test labels stay outside a claimed radio-only self-supervised encoder.
Public AP coordinates may be used only when that evaluation protocol allows
them. Namespace AP identities by dataset and physical deployment.

## Quality issues that change implementation

- **Missing is not a measured power.** UJI/TUJI/Tampere use `+100` as a missing
  marker; UV uses RSSI `-200`, distance `-1000000`, and separate measurement
  counts. UCI 402 uses string `null`. A detection threshold was not supplied
  for every device, so a universal censor bound must not be invented.
- **Device detection bias is substantial.** UJI training heard fraction is
  3.46%; TUJI is about 7.16%; Tampere SAH1 training is 4.22%. These quantities
  are real sparsity, not evidence that absent links were all below a shared
  threshold. Keep observed masks and device capabilities.
- **Raw hardware ranges are not physical path lengths.** UV has 11,058
  negative returned ranges in training, 5,772 in test day one and 6,640 in
  test day two, excluding the missing sentinel. The minimum returned value
  is −255,703 mm. A test record also has zero successful measurements despite
  a populated distance. Preserve status, bias and an outlier model; imposing
  propagation-delay constraints on the latent channel does not justify
  clamping or fabricating the hardware measurements.
- **Delivery can be old.** UV median request-to-callback time is about 3.6 s,
  with maxima around 18.6 s. Neither that latency nor a common callback time
  establishes that all AP captures were simultaneous.
- **Audit actual ranges.** TUJI's downloaded CSV contains heard values from
  −101 to −29 dBm, slightly beyond the descriptor's illustrative −100 to −35.
  The adapter should validate and record actual values rather than silently
  clip them to prose examples.
- **Do not align UCI 377 by row number.** Each position-interval file has 324
  intervals but 317 unique IDs; eight IDs from 0–324 are absent and some are
  revisited. Join explicit IDs and time windows, accounting for the shared
  continuous watch recording.
- **Exclude demonstration estimates.** IPIN's archive explicitly contains
  synthetic sample position estimates and example team rankings to exercise
  the evaluator. The sensor logs and reference positions are real; those
  demonstration outputs are not measured competitor results or SOTA evidence.

None of these sources supplies the warehouse's user-command deadline labels,
handover recovery stages, or a complete CSMA airtime trace. FTM success counts
are ranging-protocol observations, not application command ACKs. Native paid
service qualification and real-car measurements remain separate requirements.

## Pinned reference numbers, with their protocols

These are published references to reproduce, not a verified global leaderboard
or a result of our model.

| Reference | Reported endpoint | Protocol qualification |
|---|---|---|
| TUJI1 descriptor, Table 3 | 1NN: **3.34 m** mean 3D positioning error; tuned kNN: **2.27 m** | Original provided train/test spatial grid split. Best coefficient uses exponential RSSI representation, Sørensen distance, seven neighbors. This is not a held-out-device score |
| UV 2026 comparison package, `02_RESULTS/averages/mlp.tex` | All-device RTT MLP: **1.011 / 0.804 / 1.482 m** for S01/S02/S03 | Table groups all test devices and both test days, trained on all devices; it does not isolate unseen-device transfer |
| Same package, `knn.tex` | All-device, k=5 RTT: **1.193 / 0.928 / 1.572 m** | Same scenario-specific protocol. RSSI-only k=5: **2.981 / 2.819 / 2.269 m**. Sensor modality and allowed geometry must match before comparing |

The actual source tables and algorithm implementations are retained in the
downloaded archive. Its normalization fits training data and transforms test
data. Its environment file is a large notebook/application lock; install any
needed algorithms in an isolated, reviewed environment rather than replacing
the application's dependency lock wholesale.

Recommended first real-data slice: same-result UV RSSI/FTM views with explicit
device and age operators, followed by a predeclared held-out-phone and
held-out-location probe. Use TUJI1 for an independent device-transfer probe,
the library data for chronological drift, and IPIN for irregular sensing after
its clock/trajectory alignment is established. Keep unverified pair types out
of the pretext objective.

The implementation is `radiojepa/rssi_data.py`. It streams bounded immutable
batches, retains signed ranges and separate UNIX/boot clocks, provides distinct
RSSI and FTM measurement views, and opens reference XYZ only through the
separate `private_reference_labels` API. No official prepared corpus or model
fit was generated during this source implementation; explicit unit fixtures
verify parsing, corruption rejection and label separation.
