# LWM-Temporal: verified access and bounded pretrained execution

The original public **LWM-Temporal pretrained reconstruction backbone runs on
CPU** at the pinned revision below. Its complete checkpoint loaded with strict
tensor coverage. The input was an explicitly constructed shape fixture, so this
establishes executability only. It supplies no evidence of CSI prediction,
receiver-response estimation, QoS ranking, or continual-learning improvement.

The [probe receipt](../artifacts/ran_research/lwm_temporal/probe_report.json),
[download manifest](../artifacts/ran_research/lwm_temporal/download_manifest.json),
and [review record](../artifacts/ran_research/lwm_temporal/source_review.json)
record the exact access, license, implementation, weights, configuration and
CPU outputs. The [probe script](../scripts/probe_lwm_temporal.py) reproduces the
bounded check. Source and weights remain outside this application at
`/workspace/upstream/continual-radio/lwm-temporal`.

## Source, license and integrity

| Item | Verified value |
|---|---|
| Publisher repository | [wi-lab/lwm-temporal](https://huggingface.co/wi-lab/lwm-temporal/tree/df32f4beb6c7fe11d83031edbe5d6312798b81c3) |
| Revision | `df32f4beb6c7fe11d83031edbe5d6312798b81c3` |
| Repository license | MIT; copyright (c) 2025 Sadjad Alikhani |
| Checkpoint | `checkpoints/pytorch_model.bin`, 1,017,487 bytes |
| Checkpoint SHA-256 | `90be5447ee15f13547f1d0819de4d80fc1b49c28737378824513cb293335e0be` |
| Executed source | `LWMTemporal/models/lwm.py`, 25,446 bytes |
| Executed source SHA-256 | `c8ef6e6ccbc5124edf29d79d190d7c065fd98e668af251ff09979ec1d0d39f72` |
| Checkpoint config SHA-256 | `556fa34bbb1e5649385a97e576ffba81c1c6d4fa21e1635664f72a42d208c392` |
| License SHA-256 | `db8d07cbbc61a50354d22437da225c2803af9b7eae56a4e0baf89d2f37c63df2` |

All 38 non-notebook repository files were downloaded (1,355,842 bytes total).
Their exact SHA-256 hashes are in the manifest. Regular-file Git blob identities
and checkpoint LFS SHA-256 were verified against metadata for the pinned
revision. The 5.45 MB tutorial notebook was omitted. Public source and checkpoint
access succeeded through the configured HTTPS path; no new credentials were
needed. The original [MIT notice](../artifacts/ran_research/lwm_temporal/LICENSE)
is retained verbatim. Repository metadata and its root license state MIT; no
separate checkpoint-specific restrictions were found. This does not license
unrelated datasets or other model repositories.

Before execution, the license, model card, package metadata, import initializers,
complete `models/lwm.py`, Hugging Face wrapper, and reconstruction example were
read. The executed source imports standard Python, NumPy and PyTorch; its
forward path contains tensor operations and a local checkpoint loader. The
probe imports that exact reviewed file directly, without package installation
or `trust_remote_code`. It does not run the upstream training CLI, scenario
generator, dataset pickle loader, notebook, or examples. No WiFo material was
reused and no LWM-CDE execution was needed for this check.

The original loader uses `strict=False`; the probe follows it with an explicit
strict load and exact comparison of every loaded tensor. Both reads enforce
PyTorch weights-only deserialization. All 139 checkpoint tensors matched,
covering **241,634 parameters**, with no missing/unexpected keys and no randomly
initialized parameters retained. No upstream implementation or checkpoint was
patched.

## What actually ran

The original configuration uses 12 blocks, embedding width 32, eight heads,
`(1,1)` patches, real/imaginary token channels, sparse spatio-temporal attention
and learned positions. The checkpoint supplies 2,816 positions. The probe used
one deterministic analytical complex tensor with shape `[1,11,32,8]`, matching
the upstream reconstruction example's dimensions. It masked 1,690 of 2,816
tokens. This fixture is neither measured CSI nor a ray-traced dataset and has no
physical calibration.

On Python 3.12.14, PyTorch 2.14.1+cu130 and NumPy 2.5.4, using **CPU and two
PyTorch threads**, the frozen original forward returned finite float32
reconstructions with shape `[1,2816,2]`. First forward took 4.217 seconds and the
repeat took 1.636 seconds; maximum absolute repeat difference was zero. Model
loading plus strict coverage checks took 0.024 seconds. These are single-fixture
execution timings, not production latency estimates. The serialized input,
mask and reconstruction are retained in
[fixture_forward.npz](../artifacts/ran_research/lwm_temporal/fixture_forward.npz).

The public `forward_features()` method in this source returns reconstruction
values and optional CLS, despite its name. This checkpoint disables CLS. The
probe therefore makes no claim to have exported a semantic latent embedding.

## Compatibility with the current radio task

| Contract | Available input / output | Consequence |
|---|---|---|
| LWM-Temporal | Complex antenna-by-subcarrier channel history, transformed to angle–delay grids; real/imaginary reconstruction | Requires phase-bearing channel measurements and a documented transform/normalization |
| Current public radio history | RSSI/SNR, observation masks and ages, link/load metadata, packet outcomes, queues and finite control state | No measured antenna-by-subcarrier complex CSI tensor is present |
| Existing synchronized Sionna fixture | Two static poses, single antenna per device, complex multipath coefficients and delays for a 2×2 receiver/transmitter layout | Useful physical provenance, but no timed multi-antenna CSI sequence matching this pretrained input |

The application's public contract is explicit in
[service_world/features.py](../src/wirelessjev/service_world/features.py) and
[`_public_state`](../src/wirelessjev/service_world/sim.py); RSSI powers cannot
recover missing phase or per-subcarrier/antenna structure. A receiver/transmitter
coupling matrix must not be relabeled as antenna/subcarrier bins. The
[two-pose RT fixture](VISUAL_RADIO_RT.md) preserves complex CIR values but is not
a temporal channel dataset. No such conversion or integration was attempted.

The default pretrained configuration includes future-frame offsets
`(-4,-3,-2,-1,1,2,3)`. It is a masked reconstruction configuration; using it as an
online causal predictor requires a separately specified past-only inference
and adaptation protocol. The upstream prediction example references
`checkpoints/m18_cp.pth`, which is absent from this pinned repository. The
released `pytorch_model.bin` must not be described as that downstream checkpoint.

A useful downstream comparison still needs rights-cleared timed complex CSI,
antenna/subcarrier/frequency metadata, causal masking, transform/normalization,
an adaptation split, and held-out prediction or action-effect targets. RSSI and
packet-level control tests are separate evaluations. The pretrained backbone
is available as a candidate channel prior; present evidence does not establish
that it solves the receiver-response or service-control task.

## Reproduce

From `/workspace/SDTH-submission-PreceptualAI`, using the existing environment:

```bash
# Downloads and verifies pinned public assets without executing upstream code.
timeout 90s .venv/bin/python scripts/probe_lwm_temporal.py --download-only

# Offline, bounded execution of the exact reviewed implementation and weights.
timeout 90s env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 .venv/bin/python scripts/probe_lwm_temporal.py
```

Both commands completed successfully. No dependencies were installed or
downgraded, no lockfile changed, and no hardware or live radio/network probe was
performed. Download traffic was limited to public model-hosting access. Output
records identify execution scope explicitly and do not claim task accuracy.
