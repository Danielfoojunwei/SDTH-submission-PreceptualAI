"""Finite numerical controls for the separately registered measured-RSSI assay.

Every control consumes the same prepared episode as the frozen field model.
The reference is an observed-gauge nuisance estimate, not measured TX power.
No missing report, detector threshold, CAD or packet probability is invented.
GP standard deviations are working Gaussian predictive spreads, uncalibrated
for hardware or service. This module does not select a configuration or score
any validation/test target.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular

LENGTHSCALES_M = (1., 2., 5., 10., 20., 40.)
AMPLITUDES_DB = (3., 6., 12.)
NOISE_STD_DB = (1., 2., 4.)
KERNELS = ('rbf', 'matern12', 'matern32')
NEIGHBORS = (1, 3, 5, 9)
POWERS = (1, 2)
JITTER = 1.e-8
MAX_SUPPORT = 192


def configuration_grid():
    """Return 172 independent, deterministically ordered configuration dicts.

    The caller must freeze/select these on TRAIN/VAL only. Candidate order is
    also a deterministic tie-breaker, never an ordering learned from TEST.
    """
    result = [{'id': 'reference', 'method': 'reference'},
              {'id': 'logdistance', 'method': 'logdistance'}]
    for kernel in KERNELS:
        for length in LENGTHSCALES_M:
            for amplitude in AMPLITUDES_DB:
                for noise in NOISE_STD_DB:
                    result.append({'id': f'gp_{kernel}_l{length:g}_a{amplitude:g}_n{noise:g}',
                        'method': 'gp', 'kernel': kernel, 'lengthscale_m': length,
                        'amplitude_db': amplitude, 'noise_std_db': noise})
    for neighbors in NEIGHBORS:
        for power in POWERS:
            result.append({'id': f'idw_k{neighbors}_p{power}', 'method': 'idw',
                           'neighbors': neighbors, 'power': power})
    return result


def _configuration(value):
    if not isinstance(value, dict):
        raise ValueError('Use a registered numerical configuration dictionary')
    for expected in configuration_grid():
        if value.get('id') == expected['id']:
            if set(value) != set(expected):
                break
            # Python equates True with 1; booleans are not numerical settings.
            if any(isinstance(value[key], (bool, np.bool_)) for key in expected):
                break
            if value == expected:
                return expected
            break
    raise ValueError('Numerical configuration is outside the registered finite grid')


def _float(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu' or raw.shape != shape:
        raise ValueError(name + ' must have the declared real numeric axes')
    result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(name + ' must be finite')
    return result


def _public(rows, sites, name):
    if rows is None:
        # The common adapter represents an entirely unsupported axis as None.
        # Identity/membership checks below still require its index axis empty.
        return {'ap': np.empty(0, dtype=np.int64), 'rx': np.empty((0, 3)),
                'reference': np.empty(0)}
    if not isinstance(rows, dict):
        raise ValueError(name + ' must be a prepared public row dictionary')
    raw = np.asarray(rows.get('ap'))
    if raw.ndim != 1 or raw.dtype.kind not in 'iu':
        raise ValueError(name + ' AP indices must be a one-dimensional integer array')
    ap = raw.astype(np.int64)
    if np.any(ap < 0) or np.any(ap >= sites):
        raise ValueError(name + ' AP index is outside the common AP table')
    n = len(ap)
    rx = _float(rows.get('rx_xyz'), (n, 3), name + ' RX coordinates')
    fspl = _float(rows.get('fspl'), (n,), name + ' free-space operator')
    offset = _float(rows.get('offset'), (n,), name + ' nuisance reference')
    return {'ap': ap, 'rx': rx, 'reference': fspl + offset}


def _episode(prepared):
    if not isinstance(prepared, dict):
        raise ValueError('Use the common prepared measured episode')
    ap_xyz = np.asarray(prepared.get('ap_xyz'))
    if ap_xyz.ndim != 2 or ap_xyz.shape[1:] != (3,) or not 1 <= len(ap_xyz) <= 128:
        raise ValueError('Use a nonempty bounded common AP XYZ table')
    ap_xyz = _float(ap_xyz, ap_xyz.shape, 'AP XYZ coordinates')
    support = _public(prepared.get('view'), len(ap_xyz), 'Support')
    query = _public(prepared.get('query'), len(ap_xyz), 'Query')
    n, m = len(support['ap']), len(query['ap'])
    if n > MAX_SUPPORT:
        raise ValueError('Support exceeds the common frozen-field context budget')
    ids = prepared.get('support_ids')
    if (not isinstance(ids, list) or len(ids) != n
            or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != n):
        raise ValueError('Support must retain unique common source identities')
    view = prepared['view']
    if view is None:
        view = {'known': np.empty(0, dtype=bool), 'scheduled': np.empty(0, dtype=bool),
                'lower': np.empty(0), 'upper': np.empty(0)}
    for key in ('known', 'scheduled'):
        mask = np.asarray(view.get(key))
        if mask.dtype != np.bool_ or mask.shape != (n,) or not mask.all():
            raise ValueError('Prepared RSSI support must contain actual known scheduled reports')
    lo = _float(view.get('lower'), (n,), 'Support lower residual')
    hi = _float(view.get('upper'), (n,), 'Support upper residual')
    if not np.array_equal(lo, hi):
        raise ValueError('Measured aggregate support must remain continuous point reports')
    support['residual'] = lo
    support['rssi'] = support['reference'] + lo
    for name, rows in (('Support', support), ('Query', query)):
        distance = np.linalg.norm(rows['rx'] - ap_xyz[rows['ap']], axis=1)
        if not np.isfinite(distance).all() or np.any(distance <= 0):
            raise ValueError(name + ' AP/receiver distances must be positive and finite')
        rows['logdistance'] = np.log10(distance)
    statuses = prepared.get('statuses')
    indices = prepared.get('query_indices')
    if (not isinstance(statuses, list) or not isinstance(indices, list)
            or len(indices) != m or any(type(i) is not int for i in indices)
            or len(set(indices)) != m or indices != sorted(indices)
            or any(i < 0 or i >= len(statuses) for i in indices)):
        raise ValueError('Query axes must preserve common original query identities')
    query_ids = []
    for row in statuses:
        if (not isinstance(row, dict) or not isinstance(row.get('query_id'), str)
                or not row['query_id'] or row.get('status') not in ('prediction', 'unsupported')):
            raise ValueError('Use common prediction/unsupported query statuses')
        query_ids.append(row['query_id'])
    if (len(set(query_ids)) != len(query_ids)
            or indices != [i for i, row in enumerate(statuses) if row['status'] == 'prediction']):
        raise ValueError('Supported query identity/membership differs from the common episode')
    if any(not np.any(support['ap'] == ap) for ap in np.unique(query['ap'])):
        raise ValueError('A common supported query has no actual AP support')
    return support, query, statuses, indices


def _kernel(a, b, configuration):
    distance = np.linalg.norm(a[:, None] - b[None, :], axis=-1)
    scaled = distance / configuration['lengthscale_m']
    kind = configuration['kernel']
    if kind == 'rbf':
        shape = np.exp(-.5 * scaled ** 2)
    elif kind == 'matern12':
        shape = np.exp(-scaled)
    elif kind == 'matern32':
        z = math.sqrt(3.) * scaled
        shape = (1. + z) * np.exp(-z)
    else:
        raise ValueError('Unknown registered covariance kernel')
    return configuration['amplitude_db'] ** 2 * shape


def _gp(rx, residual, query_xyz, configuration):
    """Ordinary kriging, integrating its unknown constant by GLS.

    Positive observation variance permits repeated actual coordinates. No
    duplicate is transformed into an extra noise-free measurement. Jitter is
    numerical regularization and is not counted as query observation noise.
    """
    noise = configuration['noise_std_db']
    covariance = _kernel(rx, rx, configuration)
    covariance.flat[::len(rx) + 1] += noise ** 2 + JITTER
    factor = cholesky(covariance, lower=True, check_finite=True)
    ones = np.ones(len(rx))
    inverse_one = cho_solve((factor, True), ones, check_finite=True)
    precision = float(ones @ inverse_one)
    intercept = float(inverse_one @ residual / precision)
    alpha = cho_solve((factor, True), residual - intercept, check_finite=True)
    cross = _kernel(rx, query_xyz, configuration)
    mean = intercept + cross.T @ alpha
    projected = solve_triangular(factor, cross, lower=True, check_finite=True)
    variance = configuration['amplitude_db'] ** 2 - np.sum(projected ** 2, axis=0)
    variance += (1. - inverse_one @ cross) ** 2 / precision
    # Predict the reported RSSI, not a noise-free latent field.
    std = np.sqrt(np.maximum(variance, 0.) + noise ** 2)
    return mean, std


def _idw(rx, residual, query_xyz, configuration):
    values = []
    for point in query_xyz:
        distance = np.linalg.norm(rx - point, axis=1)
        exact = distance == 0.
        if exact.any():
            # Coincident actual reports share the prediction; their input
            # identities remain separate and no pseudo-observation is created.
            values.append(float(residual[exact].mean()))
            continue
        order = np.argsort(distance, kind='stable')[:configuration['neighbors']]
        # Normalize before exponentiation, avoiding overflow near support.
        weight = (distance[order].min() / distance[order]) ** configuration['power']
        values.append(float(weight @ residual[order] / weight.sum()))
    return np.asarray(values, dtype=np.float64)


def predict_baseline(prepared, configuration):
    """Predict all original queries from exactly the common admitted support.

    No independent eligibility filter, support selection, calibration corpus,
    validation labels, test labels or random seed is consulted. Unsupported
    rows copy the field adapter's identities/statuses/reasons with null output.
    """
    config = _configuration(configuration)
    support, query, statuses, indices = _episode(prepared)
    mean = query['reference'].copy()
    std = np.full(len(mean), np.nan)
    method = config['method']
    if method != 'reference':
        for ap in np.unique(query['ap']):
            seen = support['ap'] == ap
            ask = query['ap'] == ap
            if method == 'logdistance':
                x = support['logdistance'][seen]
                y = support['rssi'][seen]
                centered = x - x.mean()
                denominator = float(centered @ centered)
                if len(np.unique(x)) >= 2 and denominator > np.finfo(np.float64).eps:
                    slope = float(np.clip(centered @ (y - y.mean()) / denominator, -60., 0.))
                    mean[ask] = y.mean() + slope * (query['logdistance'][ask] - x.mean())
            elif method == 'idw':
                mean[ask] += _idw(support['rx'][seen], support['residual'][seen], query['rx'][ask], config)
            elif method == 'gp':
                correction, spread = _gp(support['rx'][seen], support['residual'][seen], query['rx'][ask], config)
                mean[ask] += correction
                std[ask] = spread
    if not np.isfinite(mean).all() or (method == 'gp' and (not np.isfinite(std).all() or np.any(std <= 0))):
        raise FloatingPointError('Nonfinite numerical RSSI prediction')
    result = [dict(row, mean_dbm=None, std_db=None) for row in statuses]
    for j, i in enumerate(indices):
        result[i].update(mean_dbm=float(mean[j]), std_db=float(std[j]) if method == 'gp' else None)
    return result
