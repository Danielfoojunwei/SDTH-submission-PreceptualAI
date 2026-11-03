# DINOv3: actual pretrained visual observations

The warehouse visual frontend now runs **real pretrained DINOv3 ViT-S/16** on
CPU, retaining a 16 × 16 spatial grid of 384-dimensional patch features per
image. The weights are frozen. This supplies a concrete visual observation
interface for later geometric grounding and radio-model fusion; it does not
establish a visual/RF prediction gain.

The [access and execution receipt](../artifacts/visual_radio/dinov3_access.json)
contains immutable weight/config/license hashes, actual model class,
preprocessing, image attribution, output shapes and checks. The
[adapter](../src/wirelessjev/visual_radio/dino.py) and
[probe](../scripts/probe_dinov3.py) load every checkpoint tensor strictly. A
missing or mismatched checkpoint raises an error; random weights are never
reported as pretrained DINOv3.

## Sources, access and license

| Item | Verified source |
|---|---|
| Primary paper | [DINOv3, arXiv:2508.10104](https://arxiv.org/abs/2508.10104); abstract, introduction, §4 on dense-feature degradation/Gram anchoring and distillation passages inspected |
| Original implementation | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751), pinned commit `6876159a11b4df116f30f667f8c9888617df0751` |
| Original HF model | `facebook/dinov3-vits16-pretrain-lvd1689m`, revision `114c1379950215c8b35dfcd4e90a5c251dde0d32`; configured HF SDK access returned **HTTP 401 / GatedRepoError**, with the repository marked manually gated. The observed status was 401, not 403. No gate was bypassed. |
| Public conversion actually used | [timm/vit_small_patch16_dinov3.lvd1689m](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m/tree/3bf4720a82ec2066db88137180ff1f83a675cef0), revision `3bf4720a82ec2066db88137180ff1f83a675cef0`; public, ungated, actual 86,362,376-byte Safetensors checkpoint downloaded |
| Weight SHA-256 | `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040` |
| Runtime | Isolated `timm==1.0.30`, actual class `timm.models.eva.Eva`; 21,586,944 parameters. Existing Torch/Torchvision were retained. |
| License | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md), also distributed with the timm model. Original and converted license files have identical SHA-256 `25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e`. This is a custom license; timm's code license does not replace the model license. |

The public model card explicitly identifies the original Meta model and
LVD-1689M distillation. It also documents conversion differences: omitted
zero-valued QKV biases and float32 RoPE periods instead of the original
bfloat16 buffer. We preserve that provenance and do not claim bit-identical
outputs to Meta's implementation. No upstream implementation was copied into
our application. Model redistribution requires the accompanying license, and
published research using DINO Materials must acknowledge their use.

## What the adapter preserves

```python
encoder = FrozenDinoV3.from_pretrained(local_files_only=True)
observed = encoder(delivered_frames, patch_valid=processed_patch_mask)
# patch_tokens: [B, 256, 384]; spatial_tokens: [B, 16, 16, 384]
# patch_boxes_original_pixels: [B, 256, 4]; patch_valid: [B, 256]
```

Each `DinoFrame` requires a delivery/ingestion time and explicit frame, camera,
source and license provenance. Capture time can be unknown. Output carries a
separate capture-time validity mask; the public photographs' ingestion times
are never presented as robot sensor capture times or freshness measurements.

Preprocessing is pinned: RGB, bicubic resize of the shorter side to 256,
256 × 256 center crop, and the model's ImageNet mean/std. Its output matches
the pinned timm evaluation transform exactly in the actual probe. Resize
dimensions and crop offsets are recorded, and patch boxes map back to original
image pixels. A pixel coordinate is not calibrated metric 3D geometry; camera
intrinsics, extrinsics, depth and pose remain separate required observations.

The adapter removes the five CLS/register prefix tokens from the dense grid.
It retains CLS separately for diagnostics, rather than reducing the warehouse
to CLS alone. An optional mask applies to the processed patch grid: masked
patch pixels become normalized mean color before the backbone sees them, and
their output tokens become zero. This is explicit input sanitization followed
by downstream masking; it is not native attention masking. DINO still mixes
the remaining visible context globally. Switching a parent model into training
mode keeps this backbone in evaluation mode with gradients disabled.

## Actual image probe and validation

Two real public photographs were downloaded and visually checked:

| Image | Author and license | Scope |
|---|---|---|
| [Modern warehouse with pallet rack storage system](https://commons.wikimedia.org/wiki/File:Modern_warehouse_with_pallet_rack_storage_system.jpg) | Axisadman, [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) | Warehouse interior with racks, pallets and forklifts |
| [BRS2 Amazon fulfilment centre drive unit robot](https://commons.wikimedia.org/wiki/File:BRS2_Amazon_fulfilment_centre_drive_unit_robot.jpg) | Auledas, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) | Static photograph of a warehouse robot on display |

Images and feature arrays remain in
`/workspace/training/visual-radio/dinov3/`; they are not bundled into the repo.
`probe_features.npz` contains actual `[2,256,384]` patch tokens, validity,
original-pixel boxes, CLS tokens and explicit time fields for integration.
The receipt records file checksums and attribution. These photographs provide
no paired RF, material/depth labels, calibrated camera pose or temporal robot
sequence. They establish executability on relevant imagery, not warehouse
connectivity accuracy.

Actual checks passed: finite dense outputs, frozen parameters, exactly
repeatable outputs on the same CPU input, exact preprocessing agreement with
timm, and zeroed invalid patch tokens. Five tests, including integration with
the actual cached pretrained checkpoint, passed. Within-image variation and
effective-rank statistics are descriptive probes; two images cannot certify
representation quality or freedom from collapse. Recorded timing is one CPU
worker observation and does not establish a fleet inference budget.

Reproduce with the existing project environment, without changing its locks:

```bash
uv pip install --python .venv/bin/python --target /workspace/upstream/dinov3-python-deps --no-deps timm==1.0.30
PYTHONPATH=/workspace/upstream/dinov3-python-deps:src .venv/bin/python scripts/probe_dinov3.py --offline
PYTHONPATH=/workspace/upstream/dinov3-python-deps:src .venv/bin/python -m pytest tests/test_visual_radio_dino.py -q
```

`--offline` requires the already downloaded assets and exact HF cache entries.
Omit it for the initial legitimate public downloads and a fresh configured-SDK
check of the original gated model. Tests never download a checkpoint or
substitute a fake backbone; integration checks skip if their optional runtime
or actual cached assets are absent.

## What DINOv3 does and does not resolve about collapse

The paper describes a specific failure: global image metrics can improve while
patch similarity becomes noisy and dense-task performance degrades over long,
large-model training. Gram anchoring regularizes pairwise patch similarities
against a Gram teacher, helping preserve local consistency. That differs from
proving every latent has useful variance or every downstream predictor has
avoided a trivial constant solution.

A frozen pretrained encoder avoids changing these visual weights during our
initial radio-model experiments. It does not prevent a fusion network from
ignoring vision, collapsing its own latent, learning shortcut correlations or
accumulating rollout error. Those require measured ablations: radio-only versus
vision-plus-radio; dense versus pooled visual input; correct versus shuffled
time/pose correspondence; geometry-only versus semantics; matched inference
and training resources; and calibration/control outcomes on sealed paired data.
Semantic similarity also cannot uniquely identify metal, wall thickness,
permittivity or moving-blocker attenuation.

The grounded hierarchy must retain separate facts: visual semantics from this
encoder; geometry/pose with their own validity; delivered RF observations;
uncertainty about unobserved materials and future interference; and service or
control consequences. No hidden simulator blockage label is rendered into
these images, and no claim about a complete hierarchical world model follows
from this frontend probe.
