"""Frozen, checksum-pinned DINOv3 dense visual observations.

These image tokens are semantic observations, not calibrated depth, radio
measurements or a complete world model. No randomly initialized fallback exists.
"""

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn


REPOSITORY = 'timm/vit_small_patch16_dinov3.lvd1689m'
REVISION = '3bf4720a82ec2066db88137180ff1f83a675cef0'
WEIGHTS_SHA256 = '2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040'
CONFIG_SHA256 = '8eb1eeb3a94a3faddfcdd9ae693607a9cd9c8b91c34381578f7f0a48ce454eef'
LICENSE_SHA256 = '25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e'


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class DinoFrame:
    image: object  # PIL.Image; imported only when preprocessing is requested.
    observation_time_seconds: float
    provenance: Mapping[str, str]
    capture_time_seconds: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.observation_time_seconds):
            raise ValueError('Observation time must be finite')
        if self.capture_time_seconds is not None:
            if not math.isfinite(self.capture_time_seconds):
                raise ValueError('Known capture time must be finite')
            if self.capture_time_seconds > self.observation_time_seconds:
                raise ValueError('Capture cannot follow observation delivery')
        for name in ('frame_id', 'camera_id', 'source_uri', 'license'):
            if not self.provenance.get(name):
                raise ValueError(f'Missing provenance: {name}')


@dataclass(frozen=True)
class DinoPatchBatch:
    patch_tokens: torch.Tensor  # B,N,D; no prefix/register tokens.
    patch_valid: torch.Tensor  # B,N; downstream observation validity.
    grid_shape: tuple[int, int]
    patch_boxes_original_pixels: torch.Tensor  # B,N,4: x0,y0,x1,y1.
    observation_time_seconds: torch.Tensor  # B, delivery/ingestion time.
    capture_time_seconds: torch.Tensor  # B, placeholder zero where unknown.
    capture_time_known: torch.Tensor  # B; never treat unknown time as fresh.
    provenance: tuple[Mapping[str, str], ...]
    image_geometry: tuple[Mapping[str, object], ...]
    cls_token: torch.Tensor  # Diagnostic only; spatial tokens are retained.
    model_metadata: Mapping[str, object]

    @property
    def spatial_tokens(self):
        return self.patch_tokens.reshape(self.patch_tokens.shape[0], *self.grid_shape, -1)

    def masked_mean(self):
        """A diagnostic pool; all-invalid frames return zero with a false mask."""
        weights = self.patch_valid.unsqueeze(-1).to(self.patch_tokens.dtype)
        return (self.patch_tokens * weights).sum(1) / weights.sum(1).clamp_min(1)


class FrozenDinoV3(nn.Module):
    """Pinned real pretrained ViT-S/16; 16×16 dense grid at 256px.

    Partial validity is supplied on the *processed* patch grid. Invalid patches
    are replaced with normalized mean-color pixels before global attention and
    their returned tokens are zeroed. This is input sanitization, not DINO's
    native attention masking or a learned occlusion model.
    """

    def __init__(self, backbone, metadata):
        super().__init__()
        if metadata.get('weights_sha256') != WEIGHTS_SHA256:
            raise ValueError('Adapter requires the verified pretrained checkpoint')
        self.backbone = backbone
        self.metadata = dict(metadata)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.train(False)

    @classmethod
    def from_pretrained(cls, *, cache_dir=None, local_files_only=False, device='cpu'):
        # Use configured HF SDK authentication; never inspect or print tokens.
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        import timm

        if timm.__version__ != '1.0.30':
            raise RuntimeError('Pinned adapter requires timm==1.0.30; install it in an isolated dependency target')
        paths = {name: Path(hf_hub_download(REPOSITORY, filename=name, revision=REVISION,
                                          cache_dir=cache_dir, local_files_only=local_files_only))
                 for name in ('config.json', 'LICENSE.md', 'model.safetensors')}
        expected = {'config.json': CONFIG_SHA256, 'LICENSE.md': LICENSE_SHA256,
                    'model.safetensors': WEIGHTS_SHA256}
        for name, checksum in expected.items():
            if _sha256(paths[name]) != checksum:
                raise ValueError(f'Pinned DINOv3 {name} checksum mismatch')
        config = json.loads(paths['config.json'].read_text())
        if config['architecture'] != 'vit_small_patch16_dinov3':
            raise ValueError('Unexpected DINOv3 architecture')
        backbone = timm.create_model(config['architecture'], pretrained=False,
                                     num_classes=0, pretrained_cfg=config['pretrained_cfg'])
        # Strict loading means missing/extra weights fail rather than silently
        # leaving random layers in a purported pretrained representation.
        backbone.load_state_dict(load_file(str(paths['model.safetensors']), device='cpu'), strict=True)
        metadata = {'repository': REPOSITORY, 'revision': REVISION,
                    'weights_sha256': WEIGHTS_SHA256, 'config_sha256': CONFIG_SHA256,
                    'license_sha256': LICENSE_SHA256, 'license': 'DINOv3 License',
                    'model_class': type(backbone).__module__ + '.' + type(backbone).__name__,
                    'implementation_file_sha256': _sha256(inspect.getfile(type(backbone))),
                    'timm_version': timm.__version__, 'preprocess': config['pretrained_cfg'],
                    'parameter_count': sum(p.numel() for p in backbone.parameters()),
                    'patch_size': list(backbone.patch_embed.patch_size),
                    'prefix_tokens_excluded': backbone.num_prefix_tokens,
                    'strict_checkpoint_load': True, 'frozen': True,
                    'radio_or_geometry_accuracy_established': False}
        return cls(backbone, metadata).to(device)

    def train(self, mode=True):
        # A parent module switching to train mode cannot activate this backbone.
        super().train(False)
        return self

    def preprocess(self, frames: Sequence[DinoFrame]):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as functional

        if not frames:
            raise ValueError('At least one delivered frame is required')
        cfg = self.metadata['preprocess']
        if (cfg['input_size'] != [3, 256, 256] or cfg['crop_pct'] != 1.0
                or cfg['interpolation'] != 'bicubic'):
            raise ValueError('Unsupported preprocessing; do not silently change spatial geometry')
        tensors, geometries, provenance = [], [], []
        for frame in frames:
            image = frame.image.convert('RGB')
            original_w, original_h = image.size
            resized = functional.resize(image, 256, interpolation=InterpolationMode.BICUBIC, antialias=True)
            resized_w, resized_h = resized.size
            left, top = round((resized_w - 256) / 2), round((resized_h - 256) / 2)
            cropped = functional.center_crop(resized, [256, 256])
            value = functional.normalize(functional.to_tensor(cropped), cfg['mean'], cfg['std'])
            tensors.append(value)
            geometries.append({'original_size_wh': [original_w, original_h],
                               'resized_size_wh': [resized_w, resized_h],
                               'crop_box_resized_xyxy': [left, top, left + 256, top + 256],
                               'processed_size_wh': [256, 256],
                               'calibrated_camera_geometry': False})
            provenance.append({**frame.provenance,
                               'canonical_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
        return torch.stack(tensors), tuple(geometries), tuple(provenance)

    @torch.inference_mode()
    def forward(self, frames: Sequence[DinoFrame], *, patch_valid=None):
        pixels, geometry, provenance = self.preprocess(frames)
        device = next(self.backbone.parameters()).device
        pixels = pixels.to(device)
        patch_h, patch_w = self.backbone.patch_embed.patch_size
        height, width = pixels.shape[-2] // patch_h, pixels.shape[-1] // patch_w
        shape = (len(frames), height * width)
        if patch_valid is None:
            valid = torch.ones(shape, device=device, dtype=torch.bool)
        else:
            if patch_valid.dtype != torch.bool or tuple(patch_valid.shape) != shape:
                raise ValueError(f'patch_valid must be boolean with shape {shape}')
            valid = patch_valid.to(device)
        pixel_valid = valid.reshape(len(frames), height, width).repeat_interleave(patch_h, 1).repeat_interleave(patch_w, 2)
        pixels = pixels.masked_fill(~pixel_valid[:, None], 0.)
        features = self.backbone.forward_features(pixels)
        prefix = self.backbone.num_prefix_tokens
        tokens = features[:, prefix:]
        if tokens.shape[:2] != shape or not torch.isfinite(tokens).all():
            raise RuntimeError('Unexpected or nonfinite DINOv3 dense output')
        tokens = tokens.masked_fill(~valid[..., None], 0.)
        boxes = []
        for transform in geometry:
            ow, oh = transform['original_size_wh']
            rw, rh = transform['resized_size_wh']
            left, top, _, _ = transform['crop_box_resized_xyxy']
            boxes.append([[(left + x * patch_w) * ow / rw,
                           (top + y * patch_h) * oh / rh,
                           (left + (x + 1) * patch_w) * ow / rw,
                           (top + (y + 1) * patch_h) * oh / rh]
                          for y in range(height) for x in range(width)])
        known = torch.tensor([f.capture_time_seconds is not None for f in frames], device=device)
        return DinoPatchBatch(
            patch_tokens=tokens, patch_valid=valid, grid_shape=(height, width),
            patch_boxes_original_pixels=torch.tensor(boxes, device=device, dtype=torch.float64),
            observation_time_seconds=torch.tensor([f.observation_time_seconds for f in frames], device=device, dtype=torch.float64),
            capture_time_seconds=torch.tensor([f.capture_time_seconds if f.capture_time_seconds is not None else 0.
                                              for f in frames], device=device, dtype=torch.float64),
            capture_time_known=known, provenance=provenance, image_geometry=geometry,
            cls_token=features[:, 0].masked_fill(~valid.any(1, keepdim=True), 0.),
            model_metadata=dict(self.metadata))
