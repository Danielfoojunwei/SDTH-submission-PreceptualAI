"""Measured DICHASUS scene views; pose labels never enter encoder features.

Acquisition receipts bind real, complete TFRecord prefix records. The parsed
NPZ preserves raw FFT-order complex coefficients and source timestamps. This
module returns only measured power and public receiver metadata. It never
downloads, writes prepared artifacts, creates synthetic channels, or trains.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re

import numpy as np

DEFAULT_ROOT = Path('/workspace/upstream/scene-radio/csi/dichasus')
POWER_FLOOR = 1.e-12
BINS = 64
RAW_BINS = 1024
REFERENCED_POWER_UNIT = 'dB of referenced linear channel power; not dBm'


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _readonly(a, dtype=None):
    a = np.asarray(a, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _path(root, name):
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Dataset member must stay inside the configured data root')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@dataclass(frozen=True)
class Recording:
    name: str
    log_power_db: np.ndarray
    bin_frequency_hz: np.ndarray
    carrier_hz: float
    bandwidth_hz: float
    antenna_array_centers_m: np.ndarray
    antenna_group_ids: np.ndarray
    time_s: np.ndarray
    positive_power_mask: np.ndarray
    source_sha256: dict[str, str]
    clipped_power_bin_count: int
    power_unit: str = REFERENCED_POWER_UNIT
    _npz_path: Path = field(repr=False, compare=False, default=Path())

    @property
    def samples(self):
        return len(self.time_s)

    def public_features(self):
        """Explicit encoder whitelist: no source pose, SNR target or label."""
        return {'log_power_db': self.log_power_db,
                'bin_frequency_hz': self.bin_frequency_hz,
                'carrier_hz': self.carrier_hz,
                'bandwidth_hz': self.bandwidth_hz,
                'antenna_array_centers_m': self.antenna_array_centers_m,
                'antenna_group_ids': self.antenna_group_ids,
                'time_s': self.time_s,
                'positive_power_mask': self.positive_power_mask}


def load_recording(name, root=DEFAULT_ROOT):
    """Read actual measured CSI, keeping original frame order and timestamps.

    Frequency is shifted into ascending order; sixteen contiguous *linear*
    powers are averaged before converting to dB. No amplitude/phase/position
    labels are invented. An explicit -120 dB referenced-unit floor handles
    numerical zeros; its count and the positive-power mask remain visible.
    """
    if not isinstance(name, str) or not re.fullmatch(r'dichasus-[a-z0-9]+', name):
        raise ValueError('Use a recording identifier, not a path')
    root = Path(root)
    receipt_path = _path(root, name + '-first1024.json')
    binding_path = _path(root, name + '-binding.json')
    npz_path = _path(root, name + '-first1024.npz')
    receipt = json.loads(receipt_path.read_text())
    binding = json.loads(binding_path.read_text())
    npz_sha = _sha(npz_path)
    if (receipt.get('source_recording') != name
            or receipt.get('all_length_and_payload_crc32c_pass') is not True
            or receipt.get('parsed_npz_sha256') != npz_sha
            or binding.get('recording') != name):
        raise ValueError('Acquired recording identity/CRC/NPZ bytes differ')
    with np.load(npz_path, allow_pickle=False) as data:
        raw_csi = data['csi']
        if raw_csi.dtype != np.float32:
            raise ValueError('Acquired CSI must retain source float32 representation')
        csi = np.array(raw_csi, copy=True)
        times = np.array(data['time'], dtype='f8', copy=True).reshape(-1)
    if (csi.ndim != 4 or csi.shape[2:] != (RAW_BINS, 2)
            or len(csi) != len(times) or len(csi) != receipt.get('acquired_records')
            or not len(csi) or not np.isfinite(csi).all()
            or not np.isfinite(times).all()):
        raise ValueError('Unexpected measured CSI/time axes or nonfinite values')
    n, antennas = csi.shape[:2]
    centers = np.asarray(binding['array_centers_by_antenna_m'], dtype='f8')
    raw_groups = np.asarray(binding['antenna_group_ids'])
    if (centers.shape != (antennas, 3) or not np.isfinite(centers).all()
            or raw_groups.shape != (antennas,) or raw_groups.dtype.kind not in 'iu'
            or (raw_groups < 0).any()):
        raise ValueError('Public array geometry/group identities disagree with CSI')
    groups = raw_groups.astype('i8')
    for group in np.unique(groups):
        points = centers[groups == group]
        if not np.array_equal(points, np.broadcast_to(points[0], points.shape)):
            raise ValueError('One physical array cannot have conflicting array centers')
    carrier = float(binding['frequency_hz'])
    bandwidth = float(binding['bandwidth_hz'])
    if (not np.isfinite(carrier) or not np.isfinite(bandwidth)
            or bandwidth <= 0 or carrier <= bandwidth / 2
            or carrier != receipt.get('frequency_hz')
            or bandwidth != receipt.get('bandwidth_hz')):
        raise ValueError('Frequency metadata differs from acquired source')
    # Float64 accumulation prevents squaring/averaging artifacts; outputs are
    # float32 for the encoder, while physical frequency/clock metadata is f64.
    shifted = np.fft.fftshift(csi, axes=2).astype('f8')
    linear = np.square(shifted).sum(-1).reshape(n, antennas, BINS, RAW_BINS // BINS).mean(-1)
    positive = linear > 0
    clipped = int((linear < POWER_FLOOR).sum())
    power = (10. * np.log10(np.maximum(linear, POWER_FLOOR))).astype('f4')
    raw_frequency = carrier + bandwidth * (np.arange(RAW_BINS, dtype='f8') - RAW_BINS // 2) / RAW_BINS
    frequencies = raw_frequency.reshape(BINS, RAW_BINS // BINS).mean(-1)
    if not np.isfinite(power).all() or not (np.diff(frequencies) > 0).all():
        raise ValueError('Measured-power transform produced invalid features')
    return Recording(name=name, log_power_db=_readonly(power),
        bin_frequency_hz=_readonly(frequencies), carrier_hz=carrier,
        bandwidth_hz=bandwidth, antenna_array_centers_m=_readonly(centers),
        antenna_group_ids=_readonly(groups), time_s=_readonly(times),
        positive_power_mask=_readonly(positive),
        source_sha256={receipt_path.name: _sha(receipt_path),
                       binding_path.name: _sha(binding_path), npz_path.name: npz_sha},
        clipped_power_bin_count=clipped, _npz_path=npz_path)


def private_pose_labels(recording):
    """Explicit labels for partition/probe/evaluation, never encoder inputs."""
    if not isinstance(recording, Recording):
        raise TypeError('Use a bound public Recording')
    if _sha(recording._npz_path) != recording.source_sha256[recording._npz_path.name]:
        raise ValueError('Source NPZ changed before private-label access')
    with np.load(recording._npz_path, allow_pickle=False) as data:
        pose = np.array(data['pos-tachy'], dtype='f8', copy=True)
    if pose.shape != (recording.samples, 3) or not np.isfinite(pose).all():
        raise ValueError('Private measured poses have invalid axes/values')
    return _readonly(pose)


@dataclass(frozen=True)
class SpatialSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    excluded: np.ndarray
    report: dict


def _near(query, reference, guard):
    if not len(query) or not len(reference):
        return np.zeros(len(query), dtype=bool)
    result = np.zeros(len(query), dtype=bool)
    for start in range(0, len(query), 128):
        difference = query[start:start + 128, None] - reference[None]
        result[start:start + 128] = (np.square(difference).sum(-1) < guard * guard).any(-1)
    return result


def spatial_split(recording, guard_m=.25):
    """Deterministic cf06 blocked split using private pose solely for indexing."""
    if recording.name != 'dichasus-cf06':
        raise ValueError('Other recordings are whole held-out domains, not training splits')
    if not np.isfinite(guard_m) or guard_m <= 0:
        raise ValueError('Use a positive spatial guard in meters')
    positions = private_pose_labels(recording)[:, :2]
    lower, upper = np.quantile(positions[:, 1], [.6, .8])
    half = guard_m / 2.
    train = np.flatnonzero(positions[:, 1] < lower - half)
    validation = np.flatnonzero((positions[:, 1] > lower + half) & (positions[:, 1] < upper - half))
    test = np.flatnonzero(positions[:, 1] > upper + half)
    initial = {'train': len(train), 'validation': len(validation), 'test': len(test)}
    # Test membership is fixed; remove nearby training/validation measurements.
    train = train[~_near(positions[train], positions[test], guard_m)]
    validation = validation[~_near(positions[validation], positions[test], guard_m)]
    train = train[~_near(positions[train], positions[validation], guard_m)]
    if not len(train) or not len(validation) or not len(test):
        raise ValueError('Physical blocked split has an empty cohort')
    chosen = np.concatenate((train, validation, test))
    if len(np.unique(chosen)) != len(chosen) or _near(positions[train], positions[test], guard_m).any():
        raise AssertionError('Spatial split leaked a source row or close location')
    excluded = np.setdiff1d(np.arange(recording.samples), chosen)
    distances = np.linalg.norm(positions[train, None] - positions[None, test], axis=-1)
    report = {'rule': 'Y quantiles60/80, half-guard on both sides of each boundary; fixed test, remove close train/validation rows',
        'private_pose_used_for_partition_only': True, 'guard_m': float(guard_m),
        'y_cutoffs_m': [float(lower), float(upper)], 'initial_counts': initial,
        'counts': {'train': len(train), 'validation': len(validation), 'test': len(test), 'excluded': len(excluded)},
        'minimum_train_test_xy_distance_m': float(distances.min()),
        'removed_close_rows': sum(initial.values()) - len(chosen),
        'whole_heldout_domains': ['dichasus-ad00', 'dichasus-dc01', 'dichasus-0152']}
    return SpatialSplit(*(_readonly(x, 'i8') for x in (train, validation, test, excluded)), report=report)


def view_indices(recording):
    """Disjoint complete physical receive arrays, observed in the same frame."""
    groups = recording.antenna_group_ids
    identities = np.unique(groups)
    if len(identities) < 2:
        raise ValueError('This recording has one physical array; cross-array views are unavailable')
    first = np.flatnonzero(np.isin(groups, identities[::2]))
    second = np.flatnonzero(np.isin(groups, identities[1::2]))
    if not len(first) or not len(second) or np.intersect1d(first, second).size:
        raise AssertionError('Static scene views overlap or are empty')
    return _readonly(first, 'i8'), _readonly(second, 'i8')


def temporal_partners(recording, trainidx, minimum_gap_s=2.):
    """Explicit chronological baseline; both endpoints stay in training only."""
    if recording.name != 'dichasus-cf06':
        raise ValueError('Whole held-out recordings cannot supply temporal training pairs')
    raw = np.asarray(trainidx)
    if (raw.ndim != 1 or raw.dtype.kind not in 'iu' or len(np.unique(raw)) != len(raw)
            or not len(raw) or (raw < 0).any() or (raw >= recording.samples).any()
            or not np.isfinite(minimum_gap_s) or minimum_gap_s < 2.):
        raise ValueError('Use unique valid training indices and at least a2s gap')
    allowed = spatial_split(recording).train
    if not np.isin(raw, allowed).all():
        raise ValueError('Temporal endpoints must stay inside the spatial training split')
    indices = raw[np.argsort(recording.time_s[raw], kind='stable')]
    times = recording.time_s[indices]
    partner = np.searchsorted(times, times + minimum_gap_s, side='left')
    # A rounded addition can select a timestamp whose *actual subtraction*
    # is one ULP below the required gap. Advance, never relax the physical gap.
    while True:
        available = np.flatnonzero(partner < len(indices))
        early = available[times[partner[available]] - times[available] < minimum_gap_s]
        if not len(early):
            break
        partner[early] += 1
    valid = partner < len(indices)
    pairs = np.stack((indices[valid], indices[partner[valid]]), -1)
    if not len(pairs) or not np.all(recording.time_s[pairs[:, 1]] - recording.time_s[pairs[:, 0]] >= minimum_gap_s):
        raise ValueError('No valid chronological training pairs at the registered gap')
    return _readonly(pairs, 'i8')
