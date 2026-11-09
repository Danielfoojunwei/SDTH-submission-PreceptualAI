"""Classical localization and supervised co-location RSSI calibration.

Calibration references are the same AP at the same tagged position, measured
on other device campaigns. Device/day/body effects are not identifiable here;
the fitted offset is not a proven hardware inverse or an instantaneous pair.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .data import Fingerprints, TrainingData, position_keys, seal


@dataclass(frozen=True)
class OffsetCalibration:
    global_offset_db: float
    ap_offset_db: np.ndarray
    per_ap_pair_counts: np.ndarray
    support_rows: np.ndarray
    paired_ap_observations: int


def fit_offsets(training: TrainingData, target: int, support_rows, *, shrinkage=4.):
    if not np.isfinite(shrinkage) or shrinkage <= 0:
        raise ValueError('A positive predeclared shrinkage pseudocount is required')
    fp = training.fingerprints
    raw_rows = np.asarray(support_rows)
    if raw_rows.size and raw_rows.dtype.kind not in 'iu':
        raise ValueError('Support row identities must be integers')
    support_rows = np.asarray(support_rows, dtype='i8')
    if (support_rows.ndim != 1 or len(set(support_rows.tolist())) != len(support_rows)
            or (support_rows < 0).any() or (support_rows >= len(fp.row0)).any()
            or (fp.device_ids[support_rows] != target).any()):
        raise ValueError('Support must contain unique real rows of the held-out device')
    keys = position_keys(training.xyz_m)
    if len({keys[row] for row in support_rows}) != len(support_rows):
        raise ValueError('N counts distinct measured support positions')
    source = np.flatnonzero(fp.device_ids != target)
    by_position = {}
    for row in source:
        by_position.setdefault(keys[row], []).append(row)
    differences = [[] for _ in range(fp.heard.shape[1])]
    for row in support_rows:
        reference = by_position.get(keys[row], [])
        if not reference:
            raise ValueError('A support position has no source-device co-location reference')
        paired = 0
        for ap in np.flatnonzero(fp.heard[row]):
            observed = [fp.rssi_dbm[index, ap] for index in reference if fp.heard[index, ap]]
            if observed:
                differences[ap].append(float(fp.rssi_dbm[row, ap] - np.median(observed)))
                paired += 1
        if paired < 3:
            raise ValueError('A selected support fingerprint lacks three observed AP pairs')
    all_values = [value for values in differences for value in values]
    global_offset = float(np.median(all_values)) if all_values else 0.
    count = np.array([len(values) for values in differences], dtype='i8')
    local = np.array([np.median(values) if values else global_offset for values in differences])
    weight = count / (count + shrinkage)
    offsets = weight * local + (1. - weight) * global_offset
    return OffsetCalibration(global_offset, seal(offsets, 'f8'), seal(count, 'i8'),
                             seal(support_rows, 'i8'), len(all_values))


def corrected(features: Fingerprints, offset):
    offset = np.asarray(offset, dtype='f8')
    if offset.ndim > 1 or (offset.ndim == 1 and offset.shape != (features.heard.shape[1],)) or not np.isfinite(offset).all():
        raise ValueError('Calibration must be one finite offset or one per AP')
    return Fingerprints(seal(np.where(features.heard, features.rssi_dbm - offset, 0.)),
                        features.heard, features.device_ids, features.row0)


def _validate(features, xyz):
    if (features.rssi_dbm.shape != features.heard.shape or len(features.device_ids) != len(features.row0)
            or len(features.row0) != len(xyz) or np.asarray(xyz).shape != (len(features.row0), 3)
            or not len(xyz) or not np.isfinite(features.rssi_dbm).all() or not np.isfinite(xyz).all()):
        raise ValueError('Finite matching reference features/XYZ are required')


class MaskedKNN:
    """RMSE over genuinely shared AP powers; no missing-power imputation."""
    def __init__(self, features: Fingerprints, xyz, *, neighbors=5, minimum_shared=3):
        _validate(features, xyz)
        if type(neighbors) is not int or neighbors < 1 or type(minimum_shared) is not int or minimum_shared < 1:
            raise ValueError('Positive integer neighborhood/coverage settings required')
        self.features, self.xyz = features, seal(xyz, 'f8')
        self.neighbors, self.minimum_shared = neighbors, minimum_shared
        self.mask = np.asarray(features.heard, dtype='f8')
        self.power = np.where(features.heard, features.rssi_dbm, 0.)
        self.power_squared = self.power * self.power
        self.prior_xyz = self.xyz.mean(0)

    def predict(self, query: Fingerprints, *, chunk_size=256):
        if (query.rssi_dbm.shape != query.heard.shape or query.heard.shape[1] != self.features.heard.shape[1]
                or not np.isfinite(query.rssi_dbm).all() or type(chunk_size) is not int or chunk_size < 1):
            raise ValueError('Malformed query fingerprints/chunk setting')
        predictions, supported, neighbor_rows = [], [], []
        for start in range(0, len(query.row0), chunk_size):
            h = np.asarray(query.heard[start:start+chunk_size], dtype='f8')
            q = np.where(h, query.rssi_dbm[start:start+chunk_size], 0.)
            common = h @ self.mask.T
            squared = (q*q) @ self.mask.T + h @ self.power_squared.T - 2. * q @ self.power.T
            distance = np.sqrt(np.maximum(squared, 0.) / np.maximum(common, 1.))
            distance[common < self.minimum_shared] = np.inf
            nearest = np.argsort(distance, axis=1, kind='stable')[:, :min(self.neighbors, len(self.xyz))]
            best = np.take_along_axis(distance, nearest, axis=1)
            good = np.isfinite(best)
            weight = np.where(good, 1. / (best + 1.), 0.)
            total = weight.sum(1)
            estimate = (self.xyz[nearest] * weight[..., None]).sum(1) / np.maximum(total[:, None], 1.e-300)
            estimate[total == 0] = self.prior_xyz
            predictions.append(estimate)
            supported.append(total > 0)
            neighbor_rows.append(np.where(good, self.features.row0[nearest], -1))
        return {'xyz_m': seal(np.concatenate(predictions), 'f8'),
                'has_shared_ap_support': seal(np.concatenate(supported), '?'),
                'neighbor_source_rows': seal(np.concatenate(neighbor_rows), 'i8')}


class RidgeLocalizer:
    """Train-only per-AP scaling plus missing masks; deterministic closed-form fit."""
    def __init__(self, features: Fingerprints, xyz, *, alpha=10.):
        _validate(features, xyz)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError('A positive predeclared ridge coefficient is required')
        heard = features.heard
        count = heard.sum(0)
        self.mean = np.where(count, (features.rssi_dbm * heard).sum(0) / np.maximum(count, 1), 0.)
        variance = (np.where(heard, features.rssi_dbm-self.mean, 0.)**2).sum(0) / np.maximum(count, 1)
        self.scale = np.maximum(np.sqrt(variance), 1.)
        self.target_mean = np.asarray(xyz).mean(0)
        self.alpha = float(alpha)
        x = self._features(features)
        self.feature_mean = x.mean(0)
        x = x-self.feature_mean
        penalty = alpha * np.eye(x.shape[1])
        self.coefficient = np.linalg.solve(x.T @ x + penalty, x.T @ (xyz-self.target_mean))

    def _features(self, features):
        if features.rssi_dbm.shape[1] != len(self.mean):
            raise ValueError('Ridge AP membership changed')
        powers = np.where(features.heard, (features.rssi_dbm-self.mean)/self.scale, 0.)
        return np.concatenate((powers, features.heard.astype('f8')), 1)

    def predict(self, query):
        return seal((self._features(query)-self.feature_mean) @ self.coefficient + self.target_mean, 'f8')
