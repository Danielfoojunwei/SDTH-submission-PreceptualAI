"""Read-only field estimates from position-tagged, actually delivered scans.

The numerical adapter is NumPy/SciPy only. Torch and the frozen field model
are imported only for an explicit RadioJEPA shadow call. Omitted APs remain
unknown; no detector threshold or below-floor measurement is fabricated.
Neither estimator supplies calibrated packet risk or permission to actuate.
"""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular

from .contracts import (canonical_sha256, records_sha256, site_sha256,
                        validate_query, validate_record, validate_site)

MAX_CONTEXT = 192
MAX_MAP_POINTS_PER_AP = 128
MAX_QUERY_POSES = 256
MAX_RECORDS = 10000
MAX_SUPPORT_DISTANCE_M = 5.
MAX_SUPPORT_AGE_SECONDS = 300.
WORKING_OBSERVATION_STD_DB = 2.
WORKING_DISCREPANCY_STD_DB = 2.
JITTER = 1.e-8
C_M_S = 299792458.


def _real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + ' must be finite')
    return float(value)


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(name + ' must be a lowercase SHA-256')
    return value


def _inputs(site, records, query_xyz=None, *, now_ns=None):
    now = time.time_ns() if now_ns is None else now_ns
    if type(now) is not int or now <= 0:
        raise ValueError('A positive desktop query clock is required')
    declared = validate_site(site)
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_RECORDS:
        raise ValueError('A nonempty bounded list of genuine survey captures is required')
    rows = [validate_record(declared, value, now_ns=now) for value in records]
    if len({r['capture_id'] for r in rows}) != len(rows):
        raise ValueError('A capture was replayed into the survey')
    if [r['pc_received_unix_ns'] for r in rows] != sorted(r['pc_received_unix_ns'] for r in rows):
        raise ValueError('Keep actual desktop delivery order')
    queries = None
    if query_xyz is not None:
        if not isinstance(query_xyz, list) or not 1 <= len(query_xyz) <= MAX_QUERY_POSES:
            raise ValueError('Use one to 256 explicit query positions')
        queries = [validate_query(declared, q) for q in query_xyz]
    return declared, rows, queries, now


def _heard_rows(site, records):
    configured = {ap['bssid']: i for i, ap in enumerate(site['aps'])}
    groups = [[] for _ in site['aps']]
    foreign = 0
    for record in records:
        for heard in record['aps']:
            index = configured.get(heard['bssid'])
            if index is None:
                foreign += 1
                continue
            point = {'capture_id': record['capture_id'], 'bssid': heard['bssid'],
                'signal_dbm': heard['signal_dbm'], 'xyz_m': list(record['pose']['xyz_m']),
                'pc_request_started_unix_ns': record['pc_request_started_unix_ns'],
                'pc_received_unix_ns': record['pc_received_unix_ns']}
            if np.linalg.norm(np.asarray(point['xyz_m']) - site['aps'][index]['xyz_m']) <= 0:
                raise ValueError('AP and observed receiver positions must differ')
            groups[index].append(point)
    return groups, foreign


def _unique_latest(points):
    # Retain an actual sample, not a fabricated median or extra independent
    # observation. Repeated positions do not acquire artificial precision.
    latest = {}
    for point in points:
        latest[tuple(round(x, 6) for x in point['xyz_m'])] = point
    return sorted(latest.values(), key=lambda p: p['pc_received_unix_ns'], reverse=True)


def _coverage(points, maximum):
    if len(points) <= 1:
        return list(points)
    xyz = np.asarray([p['xyz_m'] for p in points], dtype=float)
    chosen = [0]  # most recently delivered actual point, before geometry-only coverage
    distance = np.sum((xyz - xyz[0]) ** 2, axis=1)
    while len(chosen) < min(maximum, len(points)):
        distance[chosen] = -1.
        index = int(np.argmax(distance))
        chosen.append(index)
        distance = np.minimum(distance, np.sum((xyz - xyz[index]) ** 2, axis=1))
    return [points[i] for i in chosen]


def _fspl(rx, ap):
    positions = np.asarray(rx, dtype=float)
    distance = np.linalg.norm(positions - np.asarray(ap['xyz_m']), axis=-1)
    if np.any(distance <= 0):
        raise ValueError('Positive AP/receiver distance is required')
    frequency = ap['frequency_mhz'] * 1.e6
    return ap['tx_power_dbm'] - 20. * np.log10(4. * math.pi * frequency * distance / C_M_S)


def _kernel(a, b, length, amplitude):
    squared = np.sum((np.asarray(a)[:, None] - np.asarray(b)[None]) ** 2, axis=-1)
    return amplitude ** 2 * np.exp(-.5 * squared / length ** 2)


def _condition(points, ap, offset, length, amplitude):
    xyz = np.asarray([p['xyz_m'] for p in points], dtype=float)
    value = np.asarray([p['signal_dbm'] for p in points]) - _fspl(xyz, ap) - offset
    covariance = _kernel(xyz, xyz, length, amplitude)
    covariance.flat[::len(points) + 1] += WORKING_OBSERVATION_STD_DB ** 2 + JITTER
    factor = cholesky(covariance, lower=True, check_finite=True)
    ones = np.ones(len(points))
    inverse_one = cho_solve((factor, True), ones)
    inverse_value = cho_solve((factor, True), value)
    precision = float(ones @ inverse_one)
    intercept = float(ones @ inverse_value / precision)
    residual = value - intercept
    alpha = cho_solve((factor, True), residual)
    # Restricted likelihood integrates the constant residual intercept. These
    # are working Gaussian assumptions, not measured/qualified noise levels.
    score = .5 * float(residual @ alpha) + float(np.log(np.diag(factor)).sum())
    score += .5 * math.log(precision) + .5 * (len(points) - 1) * math.log(2. * math.pi)
    return {'lengthscale_m': length, 'amplitude_db': amplitude, 'intercept_db': intercept,
        'factor': factor.tolist(), 'alpha': alpha.tolist(), 'inverse_one': inverse_one.tolist(),
        'intercept_precision': precision, 'restricted_nll': score}


def fit_map(site, records):
    """Fit a bounded numerical RSSI map using only genuine heard samples.

    A physical FSPL mean plus an unknown constant and RBF residual is fitted
    per configured BSSID. Candidate hyperparameters use restricted likelihood;
    their uncertainty is not a hardware-validated coverage certificate.
    """
    began = time.perf_counter()
    site, captures, _, now = _inputs(site, records)
    heard, foreign = _heard_rows(site, captures)
    groups = {}
    for ap, points in zip(site['aps'], heard):
        actual = _coverage(_unique_latest(points), MAX_MAP_POINTS_PER_AP)
        group = {'raw_heard_observations': len(points), 'points': actual,
                 'unreported_observations_created': 0}
        if actual:
            candidates = [_condition(actual, ap, site['receiver']['offset_db'], length, amplitude)
                for length in (5., 2., 10., 20.) for amplitude in (6., 3., 12.)] if len(actual) > 1 else [
                    _condition(actual, ap, site['receiver']['offset_db'], 5., 6.)]
            group['gp'] = min(candidates, key=lambda item: item['restricted_nll'])
        else:
            group['gp'] = None
        groups[ap['bssid']] = group
    model = {'schema': 1, 'model_kind': 'ordinary_kriging_RBF_with_FSPL', 'site': site,
        'site_sha256': site_sha256(site), 'records_sha256': records_sha256(captures),
        'capture_ids': [r['capture_id'] for r in captures], 'fit_available_unix_ns': now,
        'foreign_heard_rows_ignored': foreign, 'groups': groups,
        'support_radius_m': MAX_SUPPORT_DISTANCE_M, 'support_max_age_seconds': MAX_SUPPORT_AGE_SECONDS,
        'working_observation_std_db': WORKING_OBSERVATION_STD_DB,
        'working_discrepancy_std_db': WORKING_DISCREPANCY_STD_DB,
        'fit_elapsed_seconds': time.perf_counter() - began,
        'real_hardware_accuracy_validated': False, 'packet_risk_qualified': False,
        'measurement_semantics': 'Heard RSSI only; latest actual sample per micrometre coordinate bucket; unreported APs unknown.'}
    model['model_sha256'] = canonical_sha256(model)
    return model


def _support(points, position, now):
    if not points:
        return {'observations': 0, 'nearest_m': None, 'within_support': False,
                'latest_observation_age_upper_s': None, 'reason': 'unobserved_ap'}
    xyz = np.asarray([p['xyz_m'] for p in points])
    distance = np.linalg.norm(xyz - np.asarray(position), axis=1)
    age = np.asarray([(now - p['pc_request_started_unix_ns']) / 1.e9 for p in points])
    if (age < 0).any() or any(p['pc_received_unix_ns'] > now for p in points):
        raise ValueError('An observation was not causally available at query time')
    nearby = distance <= MAX_SUPPORT_DISTANCE_M
    fresh = age <= MAX_SUPPORT_AGE_SECONDS
    eligible = nearby & fresh
    reason = 'supported' if eligible.any() else 'stale_observations' if not fresh.any() else 'outside_spatial_support'
    return {'observations': len(points), 'nearest_m': float(distance.min()),
        'within_support': bool(eligible.any()),
        'latest_observation_age_upper_s': float(age[nearby].min() if nearby.any() else age.min()),
        'reason': reason}


def _row(site, query_index, position, ap, support, model_kind):
    return {'site_id': site['site_id'], 'query_index': query_index, 'query_xyz_m': list(position),
        'bssid': ap['bssid'], 'ssid': ap['ssid'], 'frequency_mhz': ap['frequency_mhz'],
        'status': 'unknown', 'mean_dbm': None, 'std_db': None, 'support': support,
        'model_kind': model_kind, 'real_hardware_accuracy_validated': False,
        'packet_risk_qualified': False}


def _model(model, now):
    if not isinstance(model, dict) or type(model.get('schema')) is not int or model['schema'] != 1:
        raise ValueError('Numerical map schema 1 required')
    body = {key: value for key, value in model.items() if key != 'model_sha256'}
    if _sha(model.get('model_sha256'), 'Map identity') != canonical_sha256(body):
        raise ValueError('Numerical map bytes/parameters changed')
    site = validate_site(model['site'])
    if model.get('model_kind') != 'ordinary_kriging_RBF_with_FSPL' or model['site_sha256'] != site_sha256(site):
        raise ValueError('Numerical map site/model identity changed')
    _sha(model['records_sha256'], 'Survey identity')
    if type(model['fit_available_unix_ns']) is not int or not 0 < model['fit_available_unix_ns'] <= now:
        raise ValueError('Numerical map fit is not causally available')
    if set(model['groups']) != {ap['bssid'] for ap in site['aps']}:
        raise ValueError('Numerical map AP membership changed')
    if (model.get('support_radius_m') != MAX_SUPPORT_DISTANCE_M
            or model.get('support_max_age_seconds') != MAX_SUPPORT_AGE_SECONDS
            or model.get('working_observation_std_db') != WORKING_OBSERVATION_STD_DB
            or model.get('working_discrepancy_std_db') != WORKING_DISCREPANCY_STD_DB):
        raise ValueError('Numerical map working support/noise assumptions changed')
    for ap in site['aps']:
        points = model['groups'][ap['bssid']]['points']
        if not isinstance(points, list) or len(points) > MAX_MAP_POINTS_PER_AP:
            raise ValueError('Numerical map point budget changed')
        for point in points:
            validate_query(site, point['xyz_m'])
            _real(point['signal_dbm'], 'Stored heard RSSI')
            a, b = point['pc_request_started_unix_ns'], point['pc_received_unix_ns']
            if type(a) is not int or type(b) is not int or not 0 < a <= b <= model['fit_available_unix_ns']:
                raise ValueError('Numerical map observation clocks changed')
    return site


def predict_map(model, query_xyz):
    """Render supported observed-gauge RSSI marginals; no radio action occurs."""
    began, now = time.perf_counter(), time.time_ns()
    site = _model(model, now)
    if not isinstance(query_xyz, list) or not 1 <= len(query_xyz) <= MAX_QUERY_POSES:
        raise ValueError('Use one to 256 explicit query positions')
    positions = [validate_query(site, p) for p in query_xyz]
    rows = []
    for query_index, position in enumerate(positions):
        for ap in site['aps']:
            group = model['groups'][ap['bssid']]
            support = _support(group['points'], position, now)
            row = _row(site, query_index, position, ap, support, model['model_kind'])
            if np.linalg.norm(np.asarray(position) - ap['xyz_m']) <= 0:
                row['support'] = dict(support, within_support=False, reason='coincident_ap_query')
            elif support['within_support'] and group['gp'] is not None:
                gp, points = group['gp'], group['points']
                n = len(points)
                factor, alpha, inverse_one = (np.asarray(gp[k], dtype=float) for k in ('factor', 'alpha', 'inverse_one'))
                if (factor.shape != (n, n) or alpha.shape != (n,) or inverse_one.shape != (n,)
                        or not all(np.isfinite(x).all() for x in (factor, alpha, inverse_one))
                        or np.any(np.diag(factor) <= 0)):
                    raise ValueError('Numerical map factor/coefficients malformed')
                length = _real(gp['lengthscale_m'], 'Kernel scale')
                amplitude = _real(gp['amplitude_db'], 'Kernel amplitude')
                precision = _real(gp['intercept_precision'], 'Mean precision')
                if min(length, amplitude, precision) <= 0:
                    raise ValueError('Positive kernel and mean precision required')
                x = np.asarray([p['xyz_m'] for p in points])
                cross = _kernel(x, np.asarray([position]), length, amplitude)[:, 0]
                solved = solve_triangular(factor, cross, lower=True, check_finite=True)
                variance = amplitude ** 2 - float(solved @ solved)
                variance += (1. - float(cross @ inverse_one)) ** 2 / precision
                row.update(status='prediction', mean_dbm=float(_fspl([position], ap)[0]
                    + site['receiver']['offset_db'] + _real(gp['intercept_db'], 'Residual intercept') + cross @ alpha),
                    std_db=math.sqrt(max(variance, 0.) + WORKING_DISCREPANCY_STD_DB ** 2))
            row['assumptions'] = {'stationary_local_RBF': True, 'FSPL_plus_fitted_residual': True,
                'observed_receiver_gauge_only': True, 'working_noise_std_db': WORKING_OBSERVATION_STD_DB,
                'query_discrepancy_floor_db': WORKING_DISCREPANCY_STD_DB,
                'support_rule_is_engineering_heuristic_not_calibration': True,
                'capture_age_uses_desktop_request_window_not_cross_host_clock_subtraction': True,
                'unreported_APs_are_unknown': True}
            row['model_sha256'] = model['model_sha256']
            row['records_sha256'] = model['records_sha256']
            rows.append(row)
    elapsed = time.perf_counter() - began
    for row in rows:
        if row['mean_dbm'] is not None and not all(math.isfinite(row[key]) for key in ('mean_dbm', 'std_db')):
            raise FloatingPointError('Nonfinite numerical RSSI forecast')
        row['inference_elapsed_seconds'] = elapsed
    return rows


def prepare_radiojepa_inputs(site, records, query_xyz, *, now_ns=None):
    """Build heard-only public tensors; no model, training or tracing is called.

    The site frame is floor-origin. XY is translated to the centered RT frame;
    AP/RX distances and CAD intersections are unchanged. Declared quantization
    supplies bins, otherwise continuous RSSI supplies genuine point events.
    There is no observed hardware detector floor in this contract. The finite
    internal ``floor`` equals each supplied event's lower endpoint solely for
    the inherited heard-event validator; it contributes no extra observation.
    """
    from ..spectral_scene.operators import link_geometry_features, query_physics_basis
    site, captures, positions, now = _inputs(site, records, query_xyz, now_ns=now_ns)
    if site['receiver']['reference_gauge_assumed'] is not True:
        raise ValueError('RadioJEPA requires an explicitly assumed receiver reference gauge')
    heard, foreign = _heard_rows(site, captures)
    groups = []
    for ap, points in zip(site['aps'], heard):
        if not 2400. <= ap['frequency_mhz'] <= 5200.:
            # A refused target carrier must not silently re-enter as an
            # out-of-domain source observation affecting supported links.
            groups.append([])
            continue
        fresh = [p for p in _unique_latest(points)
                 if (now - p['pc_request_started_unix_ns']) / 1.e9 <= MAX_SUPPORT_AGE_SECONDS]
        groups.append(_coverage(fresh, MAX_CONTEXT))
    selected = []
    for depth in range(MAX_CONTEXT):
        for ap_index, points in enumerate(groups):
            if depth < len(points) and len(selected) < MAX_CONTEXT:
                selected.append((ap_index, points[depth]))
        if len(selected) >= MAX_CONTEXT:
            break
    selected_groups = [[] for _ in site['aps']]
    for index, point in selected:
        selected_groups[index].append(point)
    rows, query_rows = [], []
    for query_index, position in enumerate(positions):
        for ap_index, ap in enumerate(site['aps']):
            support = _support(selected_groups[ap_index], position, now)
            support['available_heard_observations'] = len(heard[ap_index])
            if not selected_groups[ap_index] and heard[ap_index]:
                support['reason'] = 'stale_observations'
            row = _row(site, query_index, position, ap, support, 'RadioJEPA_NestedField_shadow')
            if not 2400. <= ap['frequency_mhz'] <= 5200.:
                row['support'] = dict(support, within_support=False, reason='carrier_outside_trained_source_interval')
            elif np.linalg.norm(np.asarray(position) - ap['xyz_m']) <= 0:
                row['support'] = dict(support, within_support=False, reason='coincident_ap_query')
            if row['support']['within_support']:
                query_rows.append((len(rows), ap_index, position))
            rows.append(row)
    origin = np.asarray((site['dimensions_m'][0] / 2., site['dimensions_m'][1] / 2., 0.))
    aps = np.asarray([ap['xyz_m'] for ap in site['aps']], dtype=float) - origin
    boxes = np.asarray(site['cad_boxes_m'], dtype=float).reshape(-1, 2, 3) - origin
    quant, offset = site['receiver']['quantization_db'], site['receiver']['offset_db']
    def features(physical_rx, ids):
        rx = np.asarray(physical_rx, dtype=float).reshape(-1, 3) - origin
        tx = aps[ids]
        frequency = np.asarray([site['aps'][i]['frequency_mhz'] * 1.e6 for i in ids])
        power = np.asarray([site['aps'][i]['tx_power_dbm'] for i in ids])
        q = link_geometry_features(rx, tx, frequency, 0, site['dimensions_m'], boxes,
            calibration_pair_count=0, calibration_unknown=True, quantization_width_db=quant,
            noise_floor_dbm=site['receiver']['noise_floor_dbm'], tx_power_dbm=power)
        fspl = power - 20. * np.log10(4. * math.pi * frequency * np.linalg.norm(rx - tx, axis=-1) / C_M_S)
        return rx.astype('f4'), q, fspl.astype('f4')
    view = query = None
    if selected and query_rows:
        if (len(selected) + len(query_rows)) * max(1, len(boxes)) > 1000000:
            raise ValueError('Requested CAD/query combination exceeds the bounded runtime adapter')
        ids = np.asarray([index for index, _ in selected], dtype='i8')
        rx, q, fspl = features([p['xyz_m'] for _, p in selected], ids)
        value = np.asarray([p['signal_dbm'] for _, p in selected], dtype='f8')
        reference = fspl.astype('f8') + offset
        lo, hi = value - quant / 2. - reference, value + quant / 2. - reference
        if quant > 0 and np.any(lo >= hi):
            raise ValueError('Declared quantization bins are below supported endpoint precision')
        view = {'context': np.column_stack(((value - reference) / 30., np.zeros(len(ids)),
                np.ones(len(ids)), np.zeros(len(ids)), q)).astype('f4'),
            'known': np.ones(len(ids), dtype=bool), 'scheduled': np.ones(len(ids), dtype=bool),
            'ap': ids, 'rx_xyz': rx, 'lower': lo, 'upper': hi,
            'floor': lo.copy(), 'fspl': fspl, 'offset': np.full(len(ids), offset, dtype='f8')}
        ids = np.asarray([index for _, index, _ in query_rows], dtype='i8')
        rx, q, fspl = features([position for _, _, position in query_rows], ids)
        query = {'q': q, 'ap': ids, 'rx_xyz': rx, 'basis': query_physics_basis(q),
                 'fspl': fspl, 'offset': np.full(len(ids), offset, dtype='f8')}
    return {'site': site, 'view': view, 'query': query, 'ap_xyz': aps.astype('f4'), 'rows': rows,
        'query_row_indices': [row for row, _, _ in query_rows], 'now_unix_ns': now,
        'records_sha256': records_sha256(captures),
        'selected_context': [{'ap_index': i, **p} for i, p in selected],
        'foreign_heard_rows_ignored': foreign, 'maximum_context': MAX_CONTEXT,
        'model_origin_xyz_m': origin.tolist()}


def _checkpoint(path, expected_sha256):
    expected = _sha(expected_sha256, 'Checkpoint identity')
    source = Path(path)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('A bounded regular checkpoint file is required')
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Checkpoint SHA-256 differs from the approved identity')
    import torch
    from ..nested_radio.model import NestedField
    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    if not isinstance(value, dict) or not isinstance(value.get('settings'), dict):
        raise ValueError('A published NestedField checkpoint schema is required')
    settings = value['settings']
    if (settings != {'cell': 'nested_joint', 'width': 32, 'inference_steps': 3}
            or any(type(settings.get(k)) is not int for k in ('width', 'inference_steps'))
            or value.get('cell') != settings['cell']
            or type(value.get('seed')) is not int or value['seed'] not in (0, 1, 2)
            or type(value.get('selected_epoch')) is not int or value['selected_epoch'] not in (40, 80, 120, 160)
            or type(value.get('selected_updates')) is not int or value['selected_updates'] != value['selected_epoch'] * 12
            or type(value.get('committed_updates')) is not int or value['committed_updates'] != 1920):
        raise ValueError('Checkpoint settings or registered fitting budget differ')
    state = value.get('state')
    if (not isinstance(state, dict) or not state
            or not all(isinstance(v, torch.Tensor) and v.dtype == torch.float32 and torch.isfinite(v).all()
                       for v in state.values())):
        raise ValueError('Checkpoint must contain only finite inert floating tensors')
    if not isinstance(value.get('identity'), dict):
        raise ValueError('Checkpoint source identity must be an object')
    _sha(value['identity'].get('source_receipt_sha256'), 'Checkpoint source receipt')
    with torch.random.fork_rng(devices=[]):
        model = NestedField(**settings).to(device='cpu', dtype=torch.float32)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, {'cell': value['cell'], 'seed': value['seed'],
        'selected_epoch': value['selected_epoch'], 'checkpoint_sha256': expected,
        'source_receipt_sha256': value.get('identity', {}).get('source_receipt_sha256')}


def predict_radiojepa(site, records, query_xyz, checkpoint_path, expected_sha256):
    """Run an exact pinned model in shadow mode on admitted real scan records."""
    began = time.perf_counter()
    prepared = prepare_radiojepa_inputs(site, records, query_xyz)
    model, identity = _checkpoint(checkpoint_path, expected_sha256)
    if prepared['query'] is not None:
        import torch
        def tensor(values):
            return {name: torch.from_numpy(np.array(value, copy=True))[None] for name, value in values.items()}
        with torch.inference_mode():
            belief = model.encode_view(tensor(prepared['view']),
                torch.from_numpy(np.array(prepared['ap_xyz'], copy=True))[None])
            mean, std = model.render_queries(belief, tensor(prepared['query']))
        mean, std = mean[0].numpy(), std[0].numpy()
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise FloatingPointError('Nonfinite RadioJEPA RSSI shadow forecast')
        for row_index, mu, sigma in zip(prepared['query_row_indices'], mean, std):
            prepared['rows'][row_index].update(status='prediction', mean_dbm=float(mu), std_db=float(sigma))
    elapsed = time.perf_counter() - began
    for row in prepared['rows']:
        row.update(identity)
        row['records_sha256'] = prepared['records_sha256']
        row['context_observations'] = len(prepared['selected_context'])
        row['model_origin_xyz_m'] = prepared['model_origin_xyz_m']
        row['inference_elapsed_seconds'] = elapsed
        row['as_of_pc_unix_ns'] = prepared['now_unix_ns']
        row['assumptions'] = {'shadow_only': True, 'reference_gauge_assumed': True,
            'front_end_calibration_identified': False, 'public_geometry_and_declared_TX_required': True,
            'observations_are_downlink_scan_RSSI_not_SINR_or_packet_risk': True,
            'source_carrier_interval_mhz': [2400., 5200.],
            'carrier_and_hardware_transfer_unvalidated': True,
            'unreported_APs_are_unknown': True, 'invented_censor_bounds': False,
            'quantization_is_declared_receiver_assumption': True,
            'internal_floor_is_supplied_event_lower_not_measured_detector_threshold': True,
            'support_rule_is_engineering_heuristic_not_calibration': True,
            'capture_age_uses_desktop_request_window_not_cross_host_clock_subtraction': True}
        row['assumptions']['elapsed_includes_checkpoint_load_and_input_preparation'] = True
    return prepared['rows']
