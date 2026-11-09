"""Contract checks plus actual cached-pretrained DINOv3 integration checks.

The integration fixture never downloads weights or substitutes a fake model.
Run with the isolated timm target on PYTHONPATH after probe_dinov3.py.
"""

import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from wirelessjev.visual_radio.dino import DinoFrame, FrozenDinoV3, WEIGHTS_SHA256


PROVENANCE = {'frame_id': 'contract-only', 'camera_id': 'unknown',
              'source_uri': 'contract-validation-fixture', 'license': 'test fixture'}


def test_frame_rejects_future_capture_and_missing_provenance():
    with pytest.raises(ValueError, match='Capture cannot follow'):
        DinoFrame(None, 2., PROVENANCE, capture_time_seconds=3.)
    with pytest.raises(ValueError, match='Missing provenance: camera_id'):
        DinoFrame(None, 2., {k: v for k, v in PROVENANCE.items() if k != 'camera_id'})
    with pytest.raises(ValueError, match='finite'):
        DinoFrame(None, float('nan'), PROVENANCE)


def test_static_photo_capture_time_remains_explicitly_unknown():
    frame = DinoFrame(None, 7., PROVENANCE)
    assert frame.capture_time_seconds is None
    assert frame.observation_time_seconds == 7.


@pytest.fixture(scope='module')
def pretrained():
    pytest.importorskip('timm', reason='Optional real-pretrained integration needs isolated timm dependency')
    from huggingface_hub.errors import LocalEntryNotFoundError

    torch.set_num_threads(2)
    try:
        model = FrozenDinoV3.from_pretrained(local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip('Pinned pretrained DINOv3 is absent from the local HF cache; no fake fallback')
    assert model.metadata['weights_sha256'] == WEIGHTS_SHA256
    return model


@pytest.fixture(scope='module')
def real_frame():
    root = Path('/workspace/training/visual-radio/dinov3')
    if not (root / 'image_access.json').exists():
        pytest.skip('Real licensed probe image not downloaded')
    record = json.loads((root / 'image_access.json').read_text())[0]
    with Image.open(record['path']) as image:
        image = image.convert('RGB')
    return DinoFrame(image, 7., {'frame_id': 'real-warehouse-photo', 'camera_id': 'unknown-public-camera',
                               'source_uri': record['descriptionurl'], 'license': record['license']})


def test_actual_pretrained_dense_spatial_and_time_contract(pretrained, real_frame):
    pretrained.train(True)
    assert not pretrained.training and not pretrained.backbone.training
    assert all(not parameter.requires_grad for parameter in pretrained.parameters())
    output = pretrained([real_frame])
    assert output.patch_tokens.shape == (1, 256, 384)
    assert output.spatial_tokens.shape == (1, 16, 16, 384)
    assert output.patch_valid.all() and torch.isfinite(output.patch_tokens).all()
    assert not output.patch_tokens.requires_grad
    assert output.capture_time_known.tolist() == [False]
    assert output.observation_time_seconds.tolist() == [7.]
    assert output.provenance[0]['source_uri'] == real_frame.provenance['source_uri']
    boxes = output.patch_boxes_original_pixels[0].reshape(16, 16, 4)
    assert (boxes[..., 2:] > boxes[..., :2]).all()
    torch.testing.assert_close(boxes[:, :-1, 2], boxes[:, 1:, 0], rtol=0., atol=1e-10)
    torch.testing.assert_close(boxes[:-1, :, 3], boxes[1:, :, 1], rtol=0., atol=1e-10)
    assert output.image_geometry[0]['calibrated_camera_geometry'] is False


def test_actual_pretrained_masks_missing_patches_and_empty_frame(pretrained, real_frame):
    mask = torch.ones((1, 256), dtype=torch.bool)
    mask[:, 128:] = False
    output = pretrained([real_frame], patch_valid=mask)
    assert torch.equal(output.patch_valid, mask)
    assert torch.count_nonzero(output.patch_tokens[:, 128:]) == 0
    assert torch.isfinite(output.patch_tokens[:, :128]).all()
    empty = pretrained([real_frame], patch_valid=torch.zeros_like(mask))
    assert torch.count_nonzero(empty.patch_tokens) == 0
    assert torch.count_nonzero(empty.masked_mean()) == 0
    assert torch.count_nonzero(empty.cls_token) == 0
    with pytest.raises(ValueError, match='boolean with shape'):
        pretrained([real_frame], patch_valid=torch.ones((1, 256)))


def test_preprocessing_matches_pinned_timm_transform(pretrained, real_frame):
    import timm

    official = timm.data.create_transform(
        **timm.data.resolve_model_data_config(pretrained.backbone), is_training=False)
    pixels, _, _ = pretrained.preprocess([real_frame])
    torch.testing.assert_close(pixels[0], official(real_frame.image), rtol=0., atol=0.)
