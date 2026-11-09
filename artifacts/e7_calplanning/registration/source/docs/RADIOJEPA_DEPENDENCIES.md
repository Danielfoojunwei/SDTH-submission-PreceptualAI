# RadioJEPA dependency and ray-data audit

This audit downloaded and decoded published ray-tracing data and executed the
actual Sionna RT and DeepMIMO implementations. It is preparation for SCENE-JEPA,
not a model training run or a warehouse service result. Existing scientific
sources, results, project dependencies and locks were not changed.

## Available implementations

| Component | Actual available version/source | License and use |
|---|---|---|
| Sionna RT | Installed `sionna-rt==2.2.0`; official tag `v2.2.0` resolves to `15b5ee036917a4c2a9b6420e05570b77717e2fcc` | Apache-2.0; installed license retained and hashed |
| Mitsuba / Dr.Jit | `mitsuba==3.9.1`, `drjit==1.5.0` | BSD-3-Clause; actual LLVM polarized CPU execution works |
| DeepMIMO | Isolated `DeepMIMO==4.0.5`; source checkout `d1a34658a52640f69aa46594963029c2a83036d8` | Apache-2.0 code; dataset conditions are separate |
| NumPy for DeepMIMO | `numpy==2.2.6` in the isolated environment | DeepMIMO requires NumPy `<2.3`; the canonical project uses `2.5.4`, so its environment was preserved |
| PyTorch | Existing `2.14.1`; no CUDA device available | Current neural implementation dependency; no model training performed here |
| DINOv3 | Existing pinned ViT-S/16/timm adapter and actual cached pretrained checkpoint | Custom DINOv3 model license; see [the access audit](DINOV3_ACCESS.md). The photographs have no paired RF or calibrated sensor timestamps |
| LWM-Temporal | Existing MIT source and strict-loaded pretrained checkpoint at `df32f4beb6c7fe11d83031edbe5d6312798b81c3` | See [the checkpoint audit](LWM_TEMPORAL_ACCESS.md). Its upstream `forward_features()` returns reconstruction, so a latent-feature baseline needs an explicit encoder adapter |
| Earlier LWM | Existing `wi-lab/lwm` review found no declared reuse license | Not silently substituted for the licensed temporal model |

The isolated environment and exact lock are under
`/workspace/upstream/scene-radio/deps/project/`. It installs the data package
without adding another PyTorch or altering the application environment.
DeepMIMO's optional Sionna extra pins older Mitsuba/Dr.Jit versions; installing
that extra into the canonical environment would conflict with Sionna RT 2.2.0.
The two working paths are kept separate.

Primary sources: [Sionna RT](https://github.com/NVlabs/sionna-rt/tree/v2.2.0),
[DeepMIMO code](https://github.com/DeepMIMO/DeepMIMO),
[DeepMIMO 4.0.5 release metadata](https://pypi.org/pypi/DeepMIMO/4.0.5/json),
[official scenario catalog](https://deepmimo.net/api/scenarios),
[DeepMIMO license/citation page](https://deepmimo.net/license).

## Data actually downloaded and inspected

| Published scenario | Archive bytes | Extracted bytes | Decoded RX grid | Physical scope |
|---|---:|---:|---:|---|
| `asu_campus_3p5` | 30,558,838 | 138,848,484 | 131,931 positions, 85,157 with paths | Outdoor, 3.5 GHz, Wireless InSite ray tracing |
| `i2_28b` | 124,680,875 | 283,511,095 | 140,901 positions with paths | Indoor, 28 GHz, Wireless InSite ray tracing |

Archive SHA-256 values:

- `asu_campus_3p5`: `80e4a4983847023158ed61a5a489025d72f4edcdac89edf4ee1323699e1b7da3`
- `i2_28b`: `1d70b482d560f675b56742d960a0fcf5babd8256bb80a907ccc48beba6ba8597`

The official loader decoded powers, phases, delays, angles, transmitter/receiver
positions and interaction positions. Ten path slots were loaded per position.
For each scenario, the actual channel generator produced finite complex
`[256, 1, 4, 32]` CSI: 256 deterministic active receivers, one RX element, a
2 × 2 synthetic TX array and 32 selected OFDM subcarriers across 20 MHz.
All inspected nonzero path delays exceeded straight-line propagation delay
within a 1 ps tolerance. Raw files and generated samples have individual hashes.

These are simulated propagation records. The generated antenna views are not
independent commodity-device captures, and the static grid is not a time series.
The 28 GHz indoor scenario cannot establish 5/6 GHz Wi-Fi transfer on its own.
No weakly supervised ACK, retry, airtime or command-deadline outcomes are present.

The local data directory is
`/workspace/upstream/scene-radio/deps/deepmimo-data/`; `inspection.json` and
per-scenario receipts contain exact shapes, source endpoints and file hashes.

## Quality and rights checks

The fetched official catalog has 233 records and 228 distinct link names.
Some scenario names occur twice with contradictory indoor/outdoor and TX-count
metadata. The downloaded `params.json`, file contents and actual loader output
must determine the experiment configuration, rather than the website filter.

DeepMIMO documentation also describes path-power units inconsistently: one API
page says dBm; its quickstart describes dBW received power for 0 dBW transmission,
equivalently a normalized channel-power ratio in dB. Preserve raw path-power
values and validate the reference transmission before calling them absolute
RSSI dBm. Device power calibration must not absorb an unnoticed 30 dB shift.

The official website's license component states that use of the datasets,
scripts or modified parts requires citation of DeepMIMO, the ray tracer and any
associated scenario publication. The inspected archive contains no separate
dataset license file, and the page supplies no SPDX grant or explicit commercial
scope. These citation conditions are retained; Apache-2.0 on the software is not
presented as a dataset redistribution license. Research access succeeded without
new credentials. Release of derived assets needs the applicable data conditions
recorded explicitly.

`i1_2p5` is advertised as a 2,445,139,772-byte archive and was not fetched under
this agent's 3 GB total budget. A later dynamic-scenario URL request returned
HTTP 429 without `Retry-After`; it was not repeatedly retried. Existing downloaded
data was retained. No authentication or rate-limit bypass was attempted.

## Actual CPU physics probe

The official `floor_wall` scene was solved independently in both directions at
2.5 and 5.2 GHz using `llvm_ad_mono_polarized`, 20,000 ray samples per source,
maximum depth 3 and a 128-path cap. Four calls completed, approximately
0.48–0.88 seconds each in this run, with finite CIR/CFR and causal propagation
delays. Raw path arrays and the input XML/mesh hashes are retained in
`/workspace/upstream/scene-radio/deps/sionna-cpu-probe/`.

The independently reversed CFR differed by roughly `2.6e-7` and `4.1e-7` relative
maximum error. This checks the ideal calibrated, reciprocal array configuration;
it says nothing about uncalibrated uplink/downlink radio electronics or FDD.

The same scene's gain changed by **−7.7396 dB** between 2.5 and 5.2 GHz; free-space
frequency scaling alone predicts **−6.3613 dB**. The additional **1.3783 dB**
comes from the scene's frequency-dependent propagation/material response. Thus
the proposed frequency operator needs material dependence and uncertainty or a
learned residual; a universal fixed cross-band RSSI shift is insufficient.

This analytic scene is an executability and physics check, not a new warehouse
dataset or proof of real-time operation. The previously collected 24 warehouse
RT contexts remain available as already observed development data; they are one
warehouse and one 5.8 GHz configuration, not independent unseen layouts.

## Reproduction

```bash
uv run --project /workspace/upstream/scene-radio/deps/project python /workspace/upstream/scene-radio/deps/inspect_deepmimo.py
DRJIT_CACHE_DIR=/workspace/upstream/scene-radio/deps/drjit-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python /workspace/upstream/scene-radio/deps/cpu_rt_probe.py
```

The first command reads the already downloaded archives through the official
DeepMIMO package; it does not regenerate the ray tracer. The second runs actual
Sionna RT on its packaged analytic scene. Neither command trains a model,
changes a wireless interface or updates the canonical dependency lock.

## Registered multi-view RT generator

The new `radiojepa/rt_data.py` implements a separate, source-authorized collection
path. `artifacts/radiojepa/rt_registry.json` currently contains specifications
only: eight training, two validation, two development and 24 reserved confirmation
geometries. Every geometry has two APs, twelve receiver poses and a rack-aisle or
partitioned-room layout. Dimensions, material and obstacle height vary; the
physical geometry fingerprints are disjoint across all 36 specifications,
independently of their scene IDs and random seeds.

The registry fingerprint is
`fed8d7df9cb9852c464e47365a24e1b459e968dcb889951d2ce1b23714a1f34d`.
No registered geometry has been traced as part of implementation verification.

After source publication, `collect_stage(stage, output, authorize=...)` checks
the source authorization callback before and during collection. The callback
must return literal `authorized=True`, the matching stage and registry hash.
Confirmation additionally requires literal `confirmation_permitted=True`.
The root `common.authorize_stage` binds that permission to publication and the
qualification receipt. Existing completed scenes are checked passively against
exact file membership and hashes, not regenerated during validation.

Collection performs independent forward and reverse Sionna solves at 2.4, 5.2
and 6 GHz. It preserves raw complex CIR coefficients, absolute delays, 64-tone
CFR, scene XML, meshes, frequency-updated material parameters and solver settings.
Public simulated RSSI is derived from total reference transmit power:
`20 dBm + 10 log10(mean(abs(CFR)**2))`, followed by the explicit simulated receiver
offset and quantizer. Noise figure affects noise floor and detection, not the
incident signal power. Numerical links with no found path are unresolved; they
are not falsely encoded as verified weak measurements.

Three receiver profiles appear in training and validation. The fourth profile
appears only in development/confirmation. Its offset and other generator
constants are audit truth, not measured features to feed a model claiming
few-shot device inversion. These profiles model only RSSI offset, quantization,
noise figure and sensitivity; they do not synthesize CSI front-end impairments
or claim actual device measurements.
The per-link powers describe isolated/orthogonal channel probes; this collection
does not simulate cochannel interference, channel access or application queues.

Eleven analytic/negative/integration fixtures pass, including four actual CPU RT
solves for an explicitly nonreserved two-pose geometry. That small fixture took
roughly three seconds with warmed software in this environment. Full registered
contexts use 32,768 samples per source, twelve receiver poses and six solves;
their execution time has not yet been measured. The first authorized context
should establish the collection budget rather than extrapolating a production
timing guarantee from the small fixture.
