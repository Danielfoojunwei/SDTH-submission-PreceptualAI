# Measured-channel acquisition for RadioJEPA

Acquired on 9 November 2026. This is a data/source audit, not a training result
or evidence of warehouse service improvement. Raw files and acquisition receipts
are under `/workspace/upstream/scene-radio/csi/`; `manifest.json` binds their
bytes. No private credential was required for these public dataset downloads.
GitHub metadata queries used the configured CLI route.

## Acquired measurements

| Source | Actual acquired measurements | Physical scene and useful views | License |
|---|---|---|---|
| [DICHASUS cf0x](https://doi.org/10.18419/darus-2854), `cf06` | First 1,024 complete TFRecords; CSI `[1024,32,1024,2]`, SNR, CFO, time, transmitter pose | Robot in ARENA2036 factory; four simultaneous eight-element receiver arrays; 1.272 GHz, 50.05586592 MHz | Official DaRUS **CC BY 4.0** |
| [DICHASUS 015x](https://doi.org/10.18419/darus-2202), `0152` | First 1,024 complete TFRecords; CSI `[1024,32,1024,2]`, time and pose | Lab LoS, one 32-element array; 1.272 GHz, 50 MHz | Official DaRUS **CC BY 4.0** |
| [DICHASUS adxx](https://doi.org/10.18419/darus-4062), `ad00` | First 1,024 complete TFRecords; CSI `[1024,64,1024,2]`, SNR, CFO, time, pose; complete 188,334,454-byte environment video | ARENA2036 factory; four simultaneous 16-element arrays; 3.44 GHz, 50.05586592 MHz; receiver-gain variants documented | Official DaRUS **CC BY 4.0** |
| [DICHASUS dcxx](https://doi.org/10.18419/darus-3831), `dc01` | First 1,024 complete TFRecords; CSI `[1024,64,1024,2]`, SNR, CFO, time and pose | Hallway with known CAD, two simultaneous 32-element arrays; 3.438 GHz, 50 MHz; blocker-wall variants documented | Official DaRUS **CC BY 4.0** |
| [UCI 480](https://doi.org/10.24432/C5T60D) | Complete 202,845,041-byte archive, all 7,840 CSV sweeps parsed to complex S11/S21 `[7840,601]` | Four rooms × 196 grid locations × ten stationary VNA sweeps; actual frequency column 2.400–2.500 GHz | Official UCI page **CC BY 4.0** |

Each DICHASUS frame's length and payload CRC32C passed. HTTP 206 responses,
`Content-Range`, full-file sizes and acquired-prefix SHA256s are retained. A
prefix is a partial recording and is never reported as a complete campaign.
The UCI archive passed every ZIP member CRC. The complete `ad00` video matches
the publisher MD5 `62bd1e4764e0b6330aa2bc40e53fe68c`.

Acquisition uses the official DaRUS APIs. The primary DICHASUS website returned
503 during inspection; the official
[GitHub website source](https://github.com/dichasus/dichasus.github.io) was pinned
at `0927cdc49b8144b9f5dd9332041a96692aff811d` for metadata and calibration files.
DaRUS license responses, source metadata, reference STO/CPO corrections and
official calibration notebooks are archived. The parser uses an explicit
TensorFlow v2.20 protobuf wire subset and `google-crc32c==1.7.1`; TensorFlow
itself is unnecessary. Official schema files and Apache-2.0 license are retained.

## Quality and allowed interpretations

| Recording | Source-time range | Source order chronological? | Distinct 0.25 m XY cells in acquired prefix |
|---|---:|---|---:|
| cf06 | 52.80 s | Yes | 63 |
| 0152 | 66.58 s | No | 25 |
| ad00 | 59.33 s | Yes | 77 |
| dc01 | 1,754.93 s | No | 155 |

The `cf06` prefix has 1,021 distinct poses; its tachymeter interpolation age is
0.0358 s at the 95th percentile and 0.0755 s at maximum. Exact per-recording
statistics are saved in `dichasus/*-quality.json`. Nonchronological file order
must never become an implicit time axis.

For the declared cf06 Y-quantile partition, geometry-only inspection gives
561 training, 130 validation and 155 test rows, with 178 boundary exclusions.
The smallest training/test XY separation is 0.9416 m, exceeding the 0.25 m
guard. The two-second temporal training rule yields 519 source/partner pairs;
its median gap is 2.0156 s, but blocked-trajectory holes increase the 95th
percentile to 15.317 s. Training comparisons must disclose sample-count and
time-gap differences or use common eligible source rows.

The [loader](../src/wirelessjev/radiojepa/data.py) averages sixteen contiguous
**linear powers** into each of 64 bins before converting to dB. It shifts raw
FFT order to ascending frequency and preserves absolute referenced scale.
Its numerical floor, zero-power mask and clipped-bin count are explicit.
Private transmitter-pose labels are a separate read for partition/probe
evaluation. Public geometry contains receiver-array centers; absent element
orientation is never replaced by invented global coordinates.

These measurements support same-frame antenna/array views. They do **not**
establish reciprocal uplink/downlink pairs, held-out commodity-device transfer,
or simultaneous 2.4/5/6 GHz views. The lab recording has only one physical
array, so its cross-array pairing API rejects the request. CSI amplitudes are
referenced receiver coefficients, can exceed one, and are not dBm; imposing
`abs(reported_CSI) <= 1` would be an incorrect passivity constraint. Cross-array
phase uncertainty is explicitly documented for dcxx. Carrier-frequency offset
must not be relabelled as Doppler ground truth.

The UCI source describes a band "centered at 2.4 GHz", but the actual CSV
frequency column is 2.400–2.500 GHz. The parser retains that exact column.
Grid folder IDs encode `100*row+column`, rather than consecutive integers.
S11 is reflection, not the reverse S12 channel; S11/S21 are not a reciprocity
pair. All ten sweeps at one location must remain in one split. No packet
timestamps, ACK/PER, moving-scatterer labels or device variants are provided.

The official video documentation states that the first frame corresponds to
the **earliest CSI timestamp in the full recording**, with one-to-two-frame
synchronization accuracy. A prefix's first timestamp does not prove that
global minimum. The video is acquired and publisher-verified, but paired
camera/CSI training remains pending verification of the full-recording time
origin. Its 50 fps, 1920×1080 stream is not presented as paired ground truth yet.

## Additional sources inspected

| Source | Verified finding | Current disposition |
|---|---|---|
| [CSI-Bench authors' repository](https://github.com/guozhen-jenn-zhu/CSI-Bench-Real-WiFi-Sensing-Benchmark), [Kaggle](https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench) | Official API reports version 12, 81,519,442,469 bytes, **CC BY-NC-ND 4.0**. Authors document corrected missing-data/reshape issues and rerun benchmarks. GitHub search metadata's MIT label conflicts with the README/data terms. | Strong commodity-WiFi benchmark candidate; not downloaded or trained. Do not infer adaptation/commercial rights from a code badge. |
| [Widar 3.0 official page](https://tns.thss.tsinghua.edu.cn/widar3.0/), [IEEE DataPort](https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset) | Official DataPort JSON-LD grants CC BY 4.0. Known mislabeled and empty files, packet losses and unequal receiver-frame counts are documented. | Not acquired. Receiver rows cannot be treated as synchronous pairs without timestamp verification. DataPort downloads require its account route or author cloud bulk access. |
| [SenseFi / NTU-Fi](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark) | Authors link processed Google Drive archives; HAR `[3,114,500]`, HumanID and UT-HAR/BVP formats are documented. | Metadata inspected; dataset-specific rights and amplitude/phase semantics require review. A code license alone is insufficient. |
| [UT-HAR original repository](https://github.com/ermongroup/Wifi_Activity_Recognition) | Author README documents 30 subcarriers × three antenna amplitudes and phases; original archive is about 4 GB. | Not acquired; not confused with the separate WiAR release. |
| [WiAR original repository](https://github.com/linteresa/WiAR) | Raw Intel CSI-tool activity data and phase-preprocessing requirements documented; no dataset license pinned. | Catalogued, not admitted for training. |
| [RFChallenge](https://github.com/RFChallenge/icassp2024rfchallenge) | Measured interference frames and generated IQ-mixture challenge tasks; not warehouse channel/geometry data. | Candidate for later superposition/interference studies; dataset license not pinned. |
| “Wi3Rooms”, “WidarGait” | Exact release names could not be pinned in the bounded source search. | Unverified names, not assumed datasets. |

No global SOTA score or exhaustive-web claim follows from this catalogue.
Dataset version, task, split, units and rights must be matched before reporting
a comparison. RadioMap remains the application default; no hardware or
production capability is established by these files.
