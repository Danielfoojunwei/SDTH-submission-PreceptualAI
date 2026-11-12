"""Actual TUJI1 fingerprints with separate test-feature and XYZ-label reads.

The dataset is a sequential, multi-device site survey. Equal coordinates are
supervised co-location references; they are not simultaneous channel views.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import zipfile

import numpy as np

ARCHIVE_SHA256 = 'bc943587c7d6c55f7a4e9e946b72dcad476bfadcb99935ea35c3ba4084dbf008'
SOURCE_URL = 'https://doi.org/10.5281/zenodo.7641701'
DEFAULT_ARCHIVE = Path('/workspace/upstream/scene-radio/rssi/downloads/7641701-DATASET.zip')
DEVICE_NAMES = {1: 'S20', 2: 'S7', 3: 'POCO', 4: 'tabS7', 5: 'A12'}
MEMBERS = {'train_rssi': 'DATASET/RSS_training.csv', 'train_labels': 'DATASET/Coordinates_training.csv',
           'test_rssi': 'DATASET/RSS_testing.csv', 'test_labels': 'DATASET/Coordinates_testing.csv'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def archive_path(path=None):
    """Relocate identical public data without changing protocol/source identity."""
    if path is not None:
        return Path(path)
    directory = os.environ.get('DEVICE_TRANSFER_DATA_ROOT')
    return Path(directory) / DEFAULT_ARCHIVE.name if directory else DEFAULT_ARCHIVE


def seal(value, dtype=None):
    value = np.asarray(value, dtype=dtype)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def verify_archive(path, expected=ARCHIVE_SHA256):
    path = archive_path(path)
    if not path.is_file() or digest(path) != expected:
        raise RuntimeError('TUJI1 archive is missing or its SHA256 differs')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or not set(MEMBERS.values()) <= set(names):
            raise RuntimeError('TUJI1 archive membership is malformed')
    return path


def _array(archive, member, *, usecols=None):
    with archive.open(member) as stream:
        return np.loadtxt(stream, delimiter=',', ndmin=2, usecols=usecols)


@dataclass(frozen=True)
class Fingerprints:
    rssi_dbm: np.ndarray
    heard: np.ndarray
    device_ids: np.ndarray
    row0: np.ndarray

    def subset(self, index):
        return Fingerprints(*(seal(getattr(self, name)[index]) for name in
                              ('rssi_dbm', 'heard', 'device_ids', 'row0')))


@dataclass(frozen=True)
class TrainingData:
    fingerprints: Fingerprints
    xyz_m: np.ndarray
    building: np.ndarray
    floor: np.ndarray


def _fingerprints(rssi, devices, *, links=None):
    rssi, devices = np.asarray(rssi), np.asarray(devices).reshape(-1)
    if (rssi.ndim != 2 or not len(rssi) or rssi.shape[0] != len(devices)
            or (links is not None and rssi.shape[1] != links)
            or not np.isfinite(rssi).all() or not np.isfinite(devices).all()
            or not np.isin(devices, list(DEVICE_NAMES)).all()):
        raise ValueError('Malformed fingerprint/device matrix')
    heard = rssi != 100
    if np.any(rssi[heard] != np.floor(rssi[heard])):
        raise ValueError('TUJI1 contains a noninteger RSSI measurement')
    # Missing rows have inert numeric content plus an explicit mask. No censor
    # threshold or weak received-power value is invented for the +100 marker.
    return Fingerprints(seal(np.where(heard, rssi, 0.), 'f8'), seal(heard, '?'),
                        seal(devices, 'i8'), seal(np.arange(len(devices)), 'i8'))


def load_training(path=None, *, expected=ARCHIVE_SHA256, expected_links=310):
    path = verify_archive(path, expected)
    with zipfile.ZipFile(path) as archive:
        rssi = _array(archive, MEMBERS['train_rssi'])
        labels = _array(archive, MEMBERS['train_labels'])
    if labels.shape != (len(rssi), 6) or not np.isfinite(labels).all():
        raise ValueError('Training labels differ from the TUJI1 six-column schema')
    if not np.array_equal(labels[:, 3:], np.floor(labels[:, 3:])):
        raise ValueError('Floor/building/device identities must be integral')
    features = _fingerprints(rssi, labels[:, 5], links=expected_links)
    return TrainingData(features, seal(labels[:, :3], 'f8'),
                        seal(labels[:, 4], 'i8'), seal(labels[:, 3], 'i8'))


def _authorized():
    from . import benchmark
    benchmark.check_sources(require_published=True)


def load_test_features(path=None, *, expected=ARCHIVE_SHA256, expected_links=310):
    """Test phone IDs are metadata; XYZ columns are not decoded by this API."""
    _authorized()
    from . import benchmark
    benchmark.require_fits_before_test(path)
    path = verify_archive(path, expected)
    with zipfile.ZipFile(path) as archive:
        rssi = _array(archive, MEMBERS['test_rssi'])
        devices = _array(archive, MEMBERS['test_labels'], usecols=(5,))[:, 0]
    return _fingerprints(rssi, devices, links=expected_links)


def private_test_xyz(path=None, *, expected=ARCHIVE_SHA256):
    """Explicit label read, only after registered forecasts are persisted."""
    _authorized()
    from . import benchmark
    benchmark.require_predictions_before_labels(path)
    path = verify_archive(path, expected)
    with zipfile.ZipFile(path) as archive:
        xyz = _array(archive, MEMBERS['test_labels'], usecols=(0, 1, 2))
    if xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Malformed private test reference coordinates')
    return seal(xyz, 'f8')


def position_keys(xyz):
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Finite reference XYZ positions required')
    return [tuple(float(x) for x in row) for row in xyz]


def support_inventory(training, target, *, minimum_shared=3):
    """Exact co-location/support eligibility using masks and training labels only."""
    fp, xyz = training.fingerprints, training.xyz_m
    source_rows = np.flatnonzero(fp.device_ids != target)
    target_rows = np.flatnonzero(fp.device_ids == target)
    keys = position_keys(xyz)
    source_masks = {}
    for row in source_rows:
        key = keys[row]
        source_masks[key] = source_masks.get(key, np.zeros(fp.heard.shape[1], bool)) | fp.heard[row]
    eligible = {}
    for row in target_rows:
        key = keys[row]
        shared = fp.heard[row] & source_masks.get(key, np.zeros(fp.heard.shape[1], bool))
        if int(shared.sum()) >= minimum_shared and key not in eligible:
            eligible[key] = int(row)  # One genuine fingerprint per distinct support position.
    rows = [eligible[key] for key in sorted(eligible)]
    return {'target_device': int(target), 'source_rows': source_rows.tolist(),
            'target_training_rows': target_rows.tolist(), 'eligible_support_rows': rows,
            'eligible_support_positions': len(rows), 'minimum_shared_observed_aps': minimum_shared}


def support_order(training, target, seed, maximum, *, minimum_shared=3):
    rows = support_inventory(training, target, minimum_shared=minimum_shared)['eligible_support_rows']
    if type(seed) is not int or maximum > len(rows) or maximum < 0:
        raise ValueError('Not enough real support positions or invalid seed')
    rng = np.random.default_rng(np.random.SeedSequence([20261109, int(target), seed]))
    return np.asarray(rows, dtype='i8')[rng.permutation(len(rows))[:maximum]]


def metadata_inventory(training):
    """Read no test RF/XYZ and fit no baseline or calibration parameter."""
    return {'training_rows': len(training.xyz_m), 'links': training.fingerprints.heard.shape[1],
            'devices': {str(device): {'name': DEVICE_NAMES[device],
                'training_rows': int((training.fingerprints.device_ids == device).sum()),
                'eligible_support_positions': support_inventory(training, device)['eligible_support_positions']}
                for device in DEVICE_NAMES},
            'one_physical_site': True, 'sequential_device_campaigns': True,
            'test_rf_or_xyz_opened_by_inventory': False}
