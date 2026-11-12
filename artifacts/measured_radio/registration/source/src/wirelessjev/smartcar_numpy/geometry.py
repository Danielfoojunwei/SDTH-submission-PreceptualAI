"""Public radio geometry operators for the portable NumPy inference path.

These reproduce the frozen public AABB features and eight physical regressors.
They do not identify materials, promise radio accuracy, or implement a new PHY.
No training, simulator or tensor framework is imported here.
"""
from __future__ import annotations

import numpy as np


REFERENCE_CARRIER_HZ = 2.4e9
GEOMETRY_FEATURE_NAMES = (
    'rx_x_over_width', 'rx_y_over_depth', 'rx_z_over_height',
    'ap_x_over_width', 'ap_y_over_depth', 'ap_z_over_height',
    'log10_distance_over_2', 'log_carrier_over_2p4GHz', 'direction_sign',
    'cad_box_crossings_over_8', 'cad_box_path_fraction',
    'calibration_pair_count_clip16_over16', 'calibration_unknown',
    'context_estimated_quantization_over4dB', 'receiver_noise_floor_over100dBm',
    'known_transmit_power_over30dBm',
)
CONTEXT_FEATURE_NAMES = (
    'observed_residual_over30dB', 'censor_residual_over30dB', 'heard', 'censored',
) + GEOMETRY_FEATURE_NAMES
PHYSICS_BASIS_NAMES = (
    'intercept', 'log10_distance_over2', 'box_crossings_over8',
    'inside_length_over10m', 'log_frequency_ratio', 'logf_times_logdistance',
    'logf_times_box_crossings', 'logf_times_inside_length',
)


def _real_array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(name + ' must be real')
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name + ' must be finite real numeric values') from error
    if not np.isfinite(result).all():
        raise ValueError(name + ' must be finite')
    return result


def segment_box_features(rx_xyz, ap_xyz, cad_boxes_m):
    """Count crossed public AABBs and measure their union path fraction.

    The count retains overlapping boxes separately; overlapping path intervals
    are merged for the fraction. A single-point face contact has zero path.
    Endpoints and output arithmetic remain float64, as in the frozen operator.
    """
    rx = _real_array(rx_xyz, 'RX coordinates')
    ap = _real_array(ap_xyz, 'AP coordinates')
    boxes = _real_array(cad_boxes_m, 'CAD boxes')
    if rx.shape != ap.shape or rx.ndim != 2 or rx.shape[1] != 3:
        raise ValueError('Use matching finite Q x 3 endpoint arrays')
    if boxes.size == 0:
        boxes = np.empty((0, 2, 3), dtype=np.float64)
    if (boxes.ndim != 3 or boxes.shape[1:] != (2, 3)
            or np.any(boxes[:, 1] < boxes[:, 0])):
        raise ValueError('CAD boxes require ordered finite lower/upper XYZ corners')
    counts = np.zeros(len(rx), dtype=np.float64)
    fractions = np.zeros(len(rx), dtype=np.float64)
    for row, (start, finish) in enumerate(zip(ap, rx)):
        with np.errstate(over='ignore', invalid='ignore'):
            delta = finish - start
        if not np.isfinite(delta).all():
            raise ValueError('Endpoint differences must remain finite')
        intervals = []
        for low, high in boxes:
            lo, hi, valid = 0., 1., True
            for axis in range(3):
                if abs(delta[axis]) < 1e-14:
                    if start[axis] < low[axis] or start[axis] > high[axis]:
                        valid = False
                        break
                else:
                    a = (low[axis] - start[axis]) / delta[axis]
                    b = (high[axis] - start[axis]) / delta[axis]
                    lo = max(lo, min(a, b))
                    hi = min(hi, max(a, b))
            if valid and hi > lo + 1e-12:
                intervals.append((lo, hi))
        counts[row] = len(intervals)
        if intervals:
            intervals.sort()
            lo, hi = intervals[0]
            length = 0.
            for a, b in intervals[1:]:
                if a <= hi:
                    hi = max(hi, b)
                else:
                    length += hi - lo
                    lo, hi = a, b
            fractions[row] = length + hi - lo
    return counts, fractions


def link_geometry_features(rx_xyz, ap_xyz, carrier_hz, direction, bbox_dims_m, cad_boxes_m, *,
                           calibration_pair_count=0, calibration_unknown=True,
                           quantization_width_db=0., noise_floor_dbm=-100., tx_power_dbm=20.):
    """Return the frozen sixteen-column public schema as float32.

    Coordinates keep their supplied physical reference frame. AP identities,
    private material labels and private channel coefficients are not features.
    """
    rx = _real_array(rx_xyz, 'RX coordinates')
    ap = _real_array(ap_xyz, 'AP coordinates')
    dims = _real_array(bbox_dims_m, 'Room dimensions')
    if (rx.shape != ap.shape or rx.ndim != 2 or rx.shape[1] != 3
            or dims.shape != (3,) or np.any(dims <= 0)):
        raise ValueError('Geometry requires matching Q x XYZ and positive room dimensions')
    n = len(rx)

    def vector(value, name):
        raw = _real_array(value, name)
        try:
            result = np.broadcast_to(raw, (n,))
        except ValueError as error:
            raise ValueError(name + ' must broadcast per query') from error
        return result

    frequency = vector(carrier_hz, 'Carrier')
    which = vector(direction, 'Direction')
    count = vector(calibration_pair_count, 'Calibration count')
    unknown = vector(calibration_unknown, 'Calibration status')
    quant = vector(quantization_width_db, 'Context-derived quantization')
    noise = vector(noise_floor_dbm, 'Receiver noise')
    tx = vector(tx_power_dbm, 'Known transmit power')
    if (np.any(frequency <= 0) or not np.isin(which, (0., 1.)).all()
            or np.any(count < 0) or np.any(count != np.floor(count))
            or not np.isin(unknown, (0., 1.)).all() or np.any(quant < 0)):
        raise ValueError('Invalid physical view/calibration metadata')
    with np.errstate(over='ignore', invalid='ignore'):
        distance = np.linalg.norm(rx - ap, axis=-1)
    if np.any(distance <= 0) or not np.isfinite(distance).all():
        raise ValueError('AP and RX distances must be positive and finite')
    crossings, fraction = segment_box_features(rx, ap, cad_boxes_m)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        features = np.column_stack((rx / dims, ap / dims, np.log10(distance) / 2,
            np.log(frequency / REFERENCE_CARRIER_HZ), 2 * which - 1, crossings / 8, fraction,
            np.minimum(count, 16) / 16, unknown, quant / 4, noise / 100, tx / 30))
        result = features.astype(np.float32)
    if result.shape != (n, len(GEOMETRY_FEATURE_NAMES)) or not np.isfinite(result).all():
        raise ValueError('Malformed public geometry features')
    return result


def query_physics_basis(geometry_features, *, context=False):
    """Render eight regressors with the frozen context/query arithmetic split.

    ``context=False`` explicitly promotes real inputs to float64 for arithmetic
    before a float32 result, matching the archived NumPy query adapter even when
    its input geometry is float32. ``context=True`` requires a float32 NumPy
    array and keeps every operation float32, matching the tensor context path.
    The arithmetic mode is never inferred implicitly from the input dtype.
    """
    if type(context) is not bool:
        raise ValueError('Context arithmetic mode must be an explicit boolean')
    if context:
        if not isinstance(geometry_features, np.ndarray) or geometry_features.dtype != np.float32:
            raise ValueError('Context geometry must be an explicit float32 NumPy array')
        x = geometry_features
        dtype = np.float32
    else:
        x = _real_array(geometry_features, 'Geometry features')
        dtype = np.float64
    if x.ndim < 2 or x.shape[-1] != len(GEOMETRY_FEATURE_NAMES):
        raise ValueError('Use the sixteen-column public geometry schema')
    if not np.isfinite(x).all():
        raise ValueError('Geometry features must be finite')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        distance = np.power(dtype(10.), dtype(2.) * x[..., 6])
        length = x[..., 10] * distance / dtype(10.)
        frequency = x[..., 7]
        result = np.stack((np.ones_like(frequency), x[..., 6], x[..., 9], length, frequency,
            frequency * x[..., 6], frequency * x[..., 9], frequency * length), axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Derived physical basis must remain finite')
    return result
