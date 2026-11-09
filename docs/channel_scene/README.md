# Measured complex-CSI pilot completion

This bounded experiment compares supervised completion plus symmetric
Fourier-field consistency with its exactly matched supervised control and
three numerical references. By Parseval's identity, the full-basis IFFT
coefficient alignment is an energy-scaled field-consistency loss. It is not
evidence of a separately identified latent scene, scatterers, physical
causality, materials or absolute propagation delays. Neither branch uses an
EMA teacher or a stop-gradient target.

The data are real CRC-verified DICHASUS measurements, licensed CC BY4.0 by their
official DaRUS releases. Credit the DICHASUS researchers at the University of
Stuttgart and the individual release authors; original source URLs, release
metadata, acquisition receipts and calibration provenance are recorded in
[the acquisition audit](../RADIOJEPA_CSI_DATA_AUDIT.md). Published CSI-derived
predictions are adaptations, not original measurements. Their receipts retain
the exact acquired-source/calibration hashes. These are receiver-referenced
coefficients, not dBm. Publisher STO/CPO corrections are shared by every
method, with their broader calibration-data limitation declared.

In particular, cf0x credits Florian Euchner and Marc Gauger
([DOI10.18419/darus-2854](https://doi.org/10.18419/darus-2854)); adxx credits
Florian Euchner, Phillip Stephan, Marc Gauger and Stephan ten Brink
([DOI10.18419/darus-4062](https://doi.org/10.18419/darus-4062)). The dcxx original
release credits Florian Euchner, Phillip Stephan and Marc Gauger
([DOI10.18419/darus-3831](https://doi.org/10.18419/darus-3831)). Retain
the individual release citations and the
[CC BY4.0 license](https://creativecommons.org/licenses/by/4.0/) when sharing
these derived predictions; no original author endorsement is implied.

The three evaluation domains are cf06's spatial test partition and previously
acquired ad00/dc01 prefixes. Tens of thousands of antenna observations are not
tens of thousands of independent sites. No100-car performance, end-to-end
warehouse service, isolated device/band generalization or SOTA claim follows.
All numbers, including losses against simple baselines, belong in the report.

The exact protocol and source capsule are in `artifacts/channel_scene`.
Before source publication, only inventory, software preflight and registration
are permitted. The `run` phase verifies publication, fits all six neural runs
and deterministic references, freezes their numeric state, writes predictions,
scores, replays every prediction from raw pilots/frozen state, and seals the
byte closure. Prediction and audit phases load fitted covariance; they never
retune or refit after test access. A residual may alter observed pilot bins;
this is completion prediction, not measurement-preserving interpolation.

The complete prediction matrices total about630MB uncompressed. Individual
external cache shards are below17MB. Git holds strict SHA256/shape/source
receipts and reports. A complete compressed scientific prediction capsule and
its manifest are created under
`/workspace/training/channel-scene/deliverables/` after the run. **The PR alone
is not a self-contained passive audit without those cache bytes.** Missing
shards make replay and closure verification fail; no missing input is skipped.

To relocate, extract `channel-scene-predictions.tar.gz`, then set
`CHANNEL_SCENE_CACHE_ROOT` to the extracted `channel-scene-predictions` folder.
Retain the downloaded archive itself at that folder's
`deliverables/channel-scene-predictions.tar.gz` as well, so its transport hash
can be checked alongside every extracted shard.
Use the registered single-thread environment and run
`uv run --frozen python -m wirelessjev.channel_scene.benchmark check-closure`.
`audit` additionally requires the actual acquired DICHASUS files under the
documented source paths and replays every held prediction. Do not silently
regenerate a missing prediction shard or repair partial outputs after viewing
test results.
