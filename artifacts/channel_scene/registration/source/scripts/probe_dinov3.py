"""Run a bounded, real-pretrained DINOv3 dense-feature CPU probe.

Public warehouse/robot photos establish executability, not paired visual/RF
accuracy. Downloaded assets and feature arrays remain outside the repository.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import urllib.request

import numpy as np
from PIL import Image
import torch

from wirelessjev.visual_radio.dino import DinoFrame, FrozenDinoV3, REVISION


ROOT = Path(__file__).resolve().parents[1]
IMAGES = [
    {'title': 'Modern warehouse with pallet rack storage system',
     'url': 'https://upload.wikimedia.org/wikipedia/commons/a/a2/Modern_warehouse_with_pallet_rack_storage_system.jpg',
     'descriptionurl': 'https://commons.wikimedia.org/wiki/File:Modern_warehouse_with_pallet_rack_storage_system.jpg',
     'sha256': 'a79d4c296058bfe72f01351f1bfe46f2b239cd380f2d9d70206880c53edc760e',
     'author': 'Axisadman', 'license': 'CC BY-SA 3.0',
     'license_url': 'https://creativecommons.org/licenses/by-sa/3.0'},
    {'title': 'BRS2 Amazon fulfilment centre drive unit robot',
     'url': 'https://upload.wikimedia.org/wikipedia/commons/e/e8/BRS2_Amazon_fulfilment_centre_drive_unit_robot.jpg',
     'descriptionurl': 'https://commons.wikimedia.org/wiki/File:BRS2_Amazon_fulfilment_centre_drive_unit_robot.jpg',
     'sha256': '264aeb7fce4e00053ad90915d95ad65515c8ce55c66469a4935e24bb3c01a42e',
     'author': 'Auledas', 'license': 'CC BY 4.0',
     'license_url': 'https://creativecommons.org/licenses/by/4.0'},
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def http_get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'WirelessJEV-public-research/1.0'})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read(), response.status, response.url


def official_access():
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

    repo = 'facebook/dinov3-vits16-pretrain-lvd1689m'
    receipt = {'repository': repo, 'sdk': 'configured huggingface_hub; no token inspection/printing'}
    try:
        info = HfApi().model_info(repo)
        receipt.update(revision=info.sha, gated=info.gated)
        result = get_hf_file_metadata(hf_hub_url(repo, 'model.safetensors', revision=info.sha))
        receipt['weights_access'] = {'status': 'accessible', 'etag': result.etag,
                                     'size': result.size, 'commit_hash': result.commit_hash}
    except Exception as error:
        response = getattr(error, 'response', None)
        receipt['weights_access'] = {'status': 'error', 'exception': type(error).__name__,
                                     'http_status': getattr(response, 'status_code', None),
                                     'message': str(error)[:1000]}
    return receipt


def spatial_statistics(tokens):
    values = tokens.float()
    centered = values - values.mean(0, keepdim=True)
    energy = torch.linalg.svdvals(centered).square()
    probabilities = energy / energy.sum().clamp_min(1e-30)
    nonzero = probabilities > 0
    rank = torch.exp(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
    unit = torch.nn.functional.normalize(values, dim=-1)
    cosine = unit @ unit.T
    off_diagonal = ~torch.eye(len(values), dtype=torch.bool, device=values.device)
    return {'finite': bool(torch.isfinite(values).all()),
            'mean_patch_l2_norm': float(values.norm(dim=-1).mean()),
            'mean_spatial_feature_std': float(values.std(0).mean()),
            'centered_spatial_effective_rank': float(rank),
            'mean_off_diagonal_patch_cosine': float(cosine[off_diagonal].mean()),
            'scope': 'Descriptive within-image variation only; not a collapse, semantics, geometry or RF-accuracy benchmark'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=Path('/workspace/training/visual-radio/dinov3'))
    parser.add_argument('--offline', action='store_true', help='Require cached assets/weights and reuse prior HTTP receipts')
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    frames, image_receipts = [], []
    for index, source in enumerate(IMAGES):
        path = args.work_dir / f'public_image_{index}.jpg'
        access = {'cached': path.exists()}
        if not path.exists():
            if args.offline:
                raise FileNotFoundError(f'Cached real image is missing: {path}')
            data, status, resolved = http_get(source['url'])
            path.write_bytes(data)
            access.update(http_status=status, resolved_url=resolved)
        if digest(path) != source['sha256']:
            raise ValueError(f'Public photo changed: {source["title"]}; do not silently substitute a new image')
        with Image.open(path) as image:
            image = image.convert('RGB')
        observed = time.time()
        frame = DinoFrame(image, observed,
                          {'frame_id': f'public-probe-{index}', 'camera_id': 'public-photo-camera-unknown',
                           'source_uri': source['descriptionurl'], 'license': source['license'],
                           'source_file_sha256': source['sha256']}, capture_time_seconds=None)
        frames.append(frame)
        image_receipts.append({**source, 'local_path': str(path), 'bytes': path.stat().st_size,
                               'image_size_wh': list(image.size), 'access': access,
                               'capture_time_known': False, 'observation_time_semantics': 'Probe ingestion only',
                               'calibrated_camera_or_pose': False, 'paired_radio_labels': False})
    start = time.perf_counter()
    model = FrozenDinoV3.from_pretrained(local_files_only=args.offline)
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = model(frames)
    forward_seconds = time.perf_counter() - start
    repeated = model(frames)
    repeat_difference = float((result.patch_tokens - repeated.patch_tokens).abs().max())
    mask = torch.ones((1, 256), dtype=torch.bool)
    mask[:, 128:] = False
    masked = model(frames[:1], patch_valid=mask)
    import timm
    official_transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model.backbone), is_training=False)
    pixels, _, _ = model.preprocess(frames[:1])
    preprocess_difference = float((pixels[0] - official_transform(frames[0].image)).abs().max())
    feature_path = args.work_dir / 'probe_features.npz'
    np.savez_compressed(feature_path,
                        patch_tokens=result.patch_tokens.cpu().numpy(),
                        patch_valid=result.patch_valid.cpu().numpy(),
                        patch_boxes_original_pixels=result.patch_boxes_original_pixels.cpu().numpy(),
                        observation_time_seconds=result.observation_time_seconds.cpu().numpy(),
                        capture_time_seconds=result.capture_time_seconds.cpu().numpy(),
                        capture_time_known=result.capture_time_known.cpu().numpy(),
                        cls_token=result.cls_token.cpu().numpy(),
                        grid_shape=np.asarray(result.grid_shape))
    prior_hf = args.work_dir / 'hf_access.json'
    previous = json.loads(prior_hf.read_text()) if prior_hf.exists() else []
    official = next((r for r in previous if r.get('repo_id', '').startswith('facebook/')), None)
    if not args.offline:
        official = official_access()
    source_receipt = args.work_dir / 'official_sources.json'
    source_metadata = json.loads(source_receipt.read_text()) if source_receipt.exists() else []
    for row in source_metadata:
        if row.get('repository') == 'facebookresearch/dinov3':
            license_file = args.work_dir / 'official_LICENSE.md'
            if license_file.exists():
                row['files']['LICENSE.md'] = {'url': f'https://raw.githubusercontent.com/facebookresearch/dinov3/{row["commit"]}/LICENSE.md',
                                               'sha256': digest(license_file), 'bytes': license_file.stat().st_size}
    paper = args.work_dir / 'dinov3.pdf'
    package = args.work_dir / 'timm_pypi.json'
    metadata = {'completed': True, 'scope': 'Actual pretrained dense visual encoder executability only',
                'model': model.metadata, 'official_facebook_access': official,
                'public_timm_access': next((r for r in previous if r.get('repo_id', '').startswith('timm/')), None),
                'official_paper': {'url': 'https://arxiv.org/abs/2508.10104',
                                   'pdf_sha256': digest(paper) if paper.exists() else None,
                                   'reviewed_scope': 'Abstract, introduction, section 4 dense-feature degradation/Gram anchoring and distillation passages'},
                'official_source_metadata': source_metadata, 'images': image_receipts,
                'timm_package_release_metadata': json.loads(package.read_text()) if package.exists() else None,
                'output': {'patch_tokens_shape': list(result.patch_tokens.shape),
                           'grid_shape': list(result.grid_shape), 'cls_shape': list(result.cls_token.shape),
                           'patch_boxes_shape': list(result.patch_boxes_original_pixels.shape),
                           'capture_time_known': result.capture_time_known.tolist(),
                           'dense_statistics_by_image': [spatial_statistics(v) for v in result.patch_tokens],
                           'feature_array_path': str(feature_path), 'feature_array_sha256': digest(feature_path),
                           'feature_array_bytes': feature_path.stat().st_size,
                           'image_geometry': list(result.image_geometry),
                           'provenance': list(result.provenance)},
                'checks': {'strict_pretrained_checkpoint_load': True,
                           'all_parameters_frozen': all(not p.requires_grad for p in model.parameters()),
                           'repeat_max_absolute_token_difference': repeat_difference,
                           'preprocess_max_absolute_difference_vs_timm': preprocess_difference,
                           'masked_tokens_exactly_zero': bool(torch.count_nonzero(masked.patch_tokens[:, 128:]) == 0),
                           'no_model_gradient': not result.patch_tokens.requires_grad},
                'cpu_execution': {'torch_version': torch.__version__, 'timm_version': timm.__version__,
                                  'platform': platform.platform(), 'torch_threads': torch.get_num_threads(),
                                  'device': 'cpu', 'unique_real_images': len(frames), 'forward_calls': 3,
                                  'model_load_seconds': load_seconds, 'first_two_image_batch_forward_seconds': forward_seconds,
                                  'scope': 'One worker diagnostic timing; no repeated latency benchmark or fleet control timing'},
                'claim_boundary': {'random_weights_used_as_dinov3': False, 'trained_here': False,
                                   'warehouse_visual_rf_accuracy_established': False,
                                   'hierarchical_world_model_validated': False, 'sota_established': False,
                                   'collapse_eliminated_or_impossible': False,
                                   'notes': ['No paired RF, depth/pose, material labels or temporal robot sequence is supplied by these static photographs.',
                                             'Retaining a spatial grid preserves information needed for grounding; pixels are not metric 3D geometry.',
                                             'DINOv3 Gram anchoring addresses dense-feature degradation during its pretraining; it does not guarantee a downstream wireless learner cannot collapse.',
                                             'Public timm conversion retains the DINOv3 license and documents RoPE precision/QKV-bias differences from the original implementation.']},
                'reproduction': {'command': 'PYTHONPATH=/workspace/upstream/dinov3-python-deps:src .venv/bin/python scripts/probe_dinov3.py --offline',
                                 'isolated_dependency': 'timm==1.0.30 at /workspace/upstream/dinov3-python-deps; no core package downgrade',
                                 'adapter_sha256': digest(ROOT / 'src/wirelessjev/visual_radio/dino.py'),
                                 'probe_script_sha256': digest(Path(__file__))}}
    if repeat_difference != 0 or preprocess_difference != 0 or not metadata['checks']['masked_tokens_exactly_zero']:
        raise RuntimeError('DINOv3 interface verification failed; do not publish a successful probe')
    output = ROOT / 'artifacts/visual_radio/dinov3_access.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'completed': True, 'actual_pretrained_weights': True, 'revision': REVISION,
                      'patch_tokens_shape': list(result.patch_tokens.shape),
                      'paired_visual_rf_result': False, 'receipt': str(output),
                      'feature_arrays': str(feature_path)}, indent=2))


if __name__ == '__main__':
    main()
