"""Verified DICHASUS video/CSI pairing and optional, frozen DINOv3 features.

An earliest timestamp from a downloaded prefix is never a video clock origin.
Pairing requires the complete-recording CRC/MD5 verification receipt and actual
extracted video PTS. The publisher's 1--2 frame tolerance is retained; this is
not proof of instantaneous fading, camera calibration or object safety.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time

import numpy as np

RECORDING = 'dichasus-ad00'
SOURCE_BYTES = 6606876848
SOURCE_RECORDS = 12584
SOURCE_MD5 = '84377dbb29df6035b1483b490d2591f9'
VIDEO_MD5 = '62bd1e4764e0b6330aa2bc40e53fe68c'
SOURCE_DOI = '10.18419/darus-4062'
WEIGHTS_SHA256 = '2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040'
CONFIG_SHA256 = '8eb1eeb3a94a3faddfcdd9ae693607a9cd9c8b91c34381578f7f0a48ce454eef'
LICENSE_SHA256 = '25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e'
SHA256 = re.compile(r'[0-9a-f]{64}\Z')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _member(root, name):
    p = Path(name)
    _require(not p.is_absolute() and '..' not in p.parts and p.parts,
             'A relative, nonredirected video evidence member is required')
    root = Path(root)
    full = root / p
    _require(full.resolve().is_relative_to(root.resolve()) and full.is_file(),
             'Missing or escaped video evidence member')
    for part in (full, *full.parents):
        if part == root:
            break
        _require(not part.is_symlink(), 'Video evidence symlinks are not accepted')
    return full


def _number(value, name):
    _require(not isinstance(value, bool) and isinstance(value, (int, float))
             and math.isfinite(value), 'Invalid finite numeric '+name)
    return float(value)


@dataclass(frozen=True)
class AlignedFrame:
    source_csi_row: int
    source_csi_time_s: float
    actual_video_pts_s: float
    selected_frame_number: int
    image_path: Path
    image_sha256: str
    extraction_time_mismatch_s: float
    publisher_uncertainty_seconds: tuple[float, float]
    proof_sha256: str
    video_sha256: str

    @property
    def conservative_alignment_error_seconds(self):
        return self.publisher_uncertainty_seconds[-1] + self.extraction_time_mismatch_s


@dataclass(frozen=True)
class VerifiedAlignment:
    proof_path: Path
    proof_sha256: str
    video_path: Path
    video_sha256: str
    global_earliest_csi_time_s: float
    global_latest_csi_time_s: float
    full_recording_sha256: str
    metadata_sha256: str
    frames: tuple[AlignedFrame, ...]


def load_verified_alignment(path):
    """Validate real full-source evidence; never download or generate a frame."""
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(),
             'Verified video_alignment.json is required; prefix origin is insufficient')
    root = path.parent
    proof_bytes = path.read_bytes()
    proof_sha = hashlib.sha256(proof_bytes).hexdigest()
    proof = json.loads(proof_bytes)
    _require(proof.get('schema') == 1
        and proof.get('status') == 'verified_full_recording_stream'
        and proof.get('full_recording_checksum_verified') is True
        and proof.get('all_record_length_and_payload_crc32c_pass') is True
        and type(proof.get('received_bytes')) is int and proof['received_bytes'] == SOURCE_BYTES
        and type(proof.get('complete_records')) is int and proof['complete_records'] == SOURCE_RECORDS
        and proof.get('publisher_md5') == proof.get('received_md5') == SOURCE_MD5
        and proof.get('official_dataset_doi') == SOURCE_DOI
        and SHA256.fullmatch(proof.get('received_sha256', '')),
        'Full recording identity/size/CRC/publisher MD5 proof is incomplete')
    # Bind the received identities to the archived official publisher metadata.
    publisher = _member(root, 'research/dichasus/darus-4062.json')
    _require(_digest(publisher) == proof.get('publisher_metadata_sha256'),
             'Official dataset metadata changed')
    files = {f['dataFile']['filename']: f['dataFile'] for f in
             json.loads(publisher.read_text())['data']['latestVersion']['files']}
    recorded, video_info = files[RECORDING+'.tfrecords'], files[RECORDING+'.mkv']
    _require(recorded['filesize'] == SOURCE_BYTES
        and recorded['checksum'] == {'type':'MD5', 'value':SOURCE_MD5}
        and video_info['checksum'] == {'type':'MD5', 'value':VIDEO_MD5},
        'Official source identities differ from the supported recording')
    contract = _member(root, 'research/dichasus/dataset-details.html')
    _require(_digest(contract) == proof.get('official_sync_contract_sha256'),
             'Official synchronization contract changed')
    text = contract.read_text()
    _require('earliest CSI datapoint' in text and '1-2 video frames' in text,
             'Global-origin and synchronization tolerance must be documented')
    metadata = _member(root, proof['full_recording_time_pose_metadata'])
    _require(_digest(metadata) == proof.get('metadata_sha256'), 'Full-source timing metadata changed')
    with np.load(metadata, allow_pickle=False) as saved:
        times = saved['time_s'].astype('f8', copy=True)
        lengths = saved['record_length_bytes'].copy()
    _require(times.shape == lengths.shape == (SOURCE_RECORDS,) and np.isfinite(times).all()
        and lengths.dtype.kind in 'iu' and (lengths > 16).all()
        and sum(map(int, lengths)) == SOURCE_BYTES,
        'Full recording timing/count/byte coverage is incomplete')
    origin = _number(proof['global_earliest_csi_time_s'], 'global origin')
    latest = _number(proof['global_latest_csi_time_s'], 'global last time')
    _require(origin == float(times.min()) and latest == float(times.max())
        and origin == _number(proof['video_frame_zero_csi_time_s'], 'video clock origin')
        and proof['earliest_source_row'] == int(times.argmin())
        and proof['latest_source_row'] == int(times.argmax()),
        'A prefix timestamp or inconsistent global video origin was substituted')
    video = _member(root, 'dichasus/video/'+RECORDING+'.mkv')
    _require(video.stat().st_size == video_info['filesize']
        and _digest(video, 'md5') == proof.get('publisher_video_md5') == VIDEO_MD5
        and _digest(video) == proof.get('video_sha256'), 'Actual video differs from publisher/proof')
    paired = proof['paired_frames']
    _require(paired.get('publisher_sync_uncertainty_frames') == [1,2]
        and paired.get('publisher_sync_uncertainty_seconds') == [.02,.04]
        and paired.get('not_guaranteed_subframe_simultaneity') is True,
        'The publisher synchronization uncertainty cannot be discarded')
    log = _member(root, 'dichasus/video/aligned-samples/ffmpeg.log')
    _require(_digest(log) == paired.get('ffmpeg_stderr_sha256'), 'Actual extracted-frame PTS log changed')
    pts = [float(x) for x in re.findall(rb'\[Parsed_showinfo[^\n]*?\bpts_time:([0-9.+-]+)',log.read_bytes())]
    rows = paired['actual_frame_samples']
    _require(rows and len(rows) == len(pts), 'Actual frame/PTS evidence is absent or incomplete')
    frames, seen = [], set()
    duration = _number(proof['video_duration_s'], 'video duration')
    for item, actual_pts in zip(rows, pts):
        index = item['source_csi_row']
        _require(type(index) is int and 0 <= index < SOURCE_RECORDS and index not in seen,
                 'Duplicate or invalid full-source CSI row')
        seen.add(index)
        source_time = _number(item['source_csi_time_s'], 'source timestamp')
        offset = _number(item['video_target_offset_s'], 'target PTS')
        mismatch = _number(item['extraction_time_mismatch_s'], 'extraction mismatch')
        _require(source_time == times[index] and math.isclose(offset, source_time-origin,abs_tol=1e-8,rel_tol=0)
            and actual_pts == item['actual_video_pts_s'] and 0 <= actual_pts <= duration
            and mismatch >= 0 and math.isclose(mismatch,abs(actual_pts-offset),abs_tol=1e-8,rel_tol=0)
            and mismatch <= .020001, 'CSI row, actual video PTS and global clock mapping disagree')
        number = item['selected_frame_number']
        _require(type(number) is int and number >= 0 and number == int(np.rint(offset*50)),
                 'Nominal frame selection differs from the recorded 50fps source')
        image = _member(root, item['file'])
        _require(_digest(image) == item['sha256'], 'Aligned source image changed')
        frames.append(AlignedFrame(index, source_time, actual_pts, number, image,
            item['sha256'], mismatch, (.02,.04), proof_sha, proof['video_sha256']))
    _require(_digest(path) == proof_sha, 'Alignment receipt changed during verification')
    return VerifiedAlignment(path, proof_sha, video, proof['video_sha256'], origin, latest,
        proof['received_sha256'], proof['metadata_sha256'], tuple(frames))


def pair_csi_rows(alignment, indices=None, *, csi_times_s=None):
    """Use only actually extracted/proven frames; never infer missing images."""
    if not isinstance(alignment,VerifiedAlignment):
        raise TypeError('Use a verified full-recording video alignment')
    if indices is None:
        rows = np.asarray([frame.source_csi_row for frame in alignment.frames],dtype='i8')
    else:
        rows = np.asarray(indices)
    _require(rows.ndim == 1 and rows.dtype.kind in 'iu' and len(rows)
             and len(np.unique(rows)) == len(rows), 'Use nonempty unique integral source rows')
    mapping = {frame.source_csi_row:frame for frame in alignment.frames}
    _require(all(int(row) in mapping for row in rows), 'Requested CSI row has no verified extracted video frame')
    result = tuple(mapping[int(row)] for row in rows)
    if csi_times_s is not None:
        times = np.asarray(csi_times_s,dtype='f8')
        _require(times.shape == rows.shape and np.isfinite(times).all()
            and np.array_equal(times,np.array([p.source_csi_time_s for p in result])),
            'Caller timestamps differ from the full-source CSI row identities')
    return result


def open_frame(pair):
    """Read/copy an actual aligned image without changing its source file."""
    from PIL import Image
    if not isinstance(pair,AlignedFrame) or _digest(pair.image_path) != pair.image_sha256:
        raise ValueError('Actual aligned frame bytes/provenance are required')
    with Image.open(pair.image_path) as image:
        return image.convert('RGB').copy()


def extract_frozen_features(alignment_path, output, *, csi_indices=None):
    """Explicit, source-gated actual CPU DINOv3 extraction; never auto-download."""
    from . import common as c
    c.check_sources()  # Must precede any backbone construction or forward.
    output = Path(output)
    _require(not output.resolve().is_relative_to(c.OUT.resolve()),
             'Secondary vision output must stay outside the main RF experiment closure')
    alignment = load_verified_alignment(alignment_path)
    pairs = pair_csi_rows(alignment,csi_indices)
    if output.exists() and any(output.rglob('*')):
        raise RuntimeError('Feature destination is nonempty; retain immutable prior evidence')
    from ..visual_radio.dino import FrozenDinoV3,DinoFrame
    encoder = FrozenDinoV3.from_pretrained(local_files_only=True,device='cpu')
    _require(encoder.metadata.get('weights_sha256') == WEIGHTS_SHA256
        and encoder.metadata.get('config_sha256') == CONFIG_SHA256
        and encoder.metadata.get('license_sha256') == LICENSE_SHA256
        and encoder.metadata.get('strict_checkpoint_load') is True
        and encoder.metadata.get('frozen') is True, 'Actual licensed frozen DINOv3 checkpoint is required')
    frames=[]
    for pair in pairs:
        # Ingestion monotonic time is not a source camera capture timestamp.
        frames.append(DinoFrame(open_frame(pair),time.monotonic(),{
            'frame_id':f'{RECORDING}:frame{pair.selected_frame_number}:pts{pair.actual_video_pts_s}',
            'camera_id':RECORDING+'-video','source_uri':'https://doi.org/'+SOURCE_DOI,
            'license':'CC-BY-4.0','capture_clock':'video PTS, publisher approximate alignment'},
            capture_time_seconds=None))
    import torch
    with torch.inference_mode():
        observed=encoder(frames)
    _require(_digest(alignment.proof_path) == alignment.proof_sha256
        and all(_digest(p.image_path) == p.image_sha256 for p in pairs),
        'Video alignment/image source changed during frozen extraction')
    c.check_sources()
    arrays={name:getattr(observed,name).detach().cpu().numpy() for name in
        ('patch_tokens','patch_valid','patch_boxes_original_pixels','cls_token',
         'observation_time_seconds','capture_time_seconds','capture_time_known')}
    _require(arrays['patch_tokens'].shape == (len(pairs),256,384)
        and np.isfinite(arrays['patch_tokens']).all() and not arrays['capture_time_known'].any(),
        'Unexpected actual DINOv3 feature axes or imputed capture timestamps')
    arrays.update(source_csi_row=np.array([p.source_csi_row for p in pairs],dtype='i8'),
        source_csi_time_s=np.array([p.source_csi_time_s for p in pairs]),
        source_video_pts_s=np.array([p.actual_video_pts_s for p in pairs]))
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays)
    c.write_bytes_once(output/'features.npz',buffer.getvalue())
    result={'schema':1,'completed':True,'actual_frozen_dinov3':True,'model':encoder.metadata,
        'source_receipt_sha256':c.digest(c.OUT/'registration/source.json'),
        'alignment_sha256':alignment.proof_sha256,'video_sha256':alignment.video_sha256,
        'full_recording_sha256':alignment.full_recording_sha256,
        'full_recording_timing_metadata_sha256':alignment.metadata_sha256,
        'features_sha256':c.digest(output/'features.npz'),
        'frames':[{'source_csi_row':p.source_csi_row,'source_csi_time_s':p.source_csi_time_s,
            'actual_video_pts_s':p.actual_video_pts_s,'frame_number':p.selected_frame_number,
            'image_sha256':p.image_sha256,'extraction_time_mismatch_s':p.extraction_time_mismatch_s,
            'publisher_uncertainty_frames':[1,2],'publisher_uncertainty_seconds':[.02,.04],
            'conservative_alignment_error_seconds':p.conservative_alignment_error_seconds,
            'unpaired':False,'paired_with_publisher_tolerance':True} for p in pairs],
        'image_geometry':list(observed.image_geometry),
        'ingestion_clock':'observer monotonic seconds; not source video capture UTC',
        'source_capture_time_imputed':False,'exact_simultaneity_proven':False,
        'camera_geometry_calibrated':False,'object_or_robot_safety_established':False,
        'RF_accuracy_gain_established':False,'visual_overlay_label_leakage_audited':False}
    c.write_once(output/'receipt.json',result)
    return result
