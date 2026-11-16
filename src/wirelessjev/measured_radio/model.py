"""Support-only reference adapter for a frozen measured-RSSI transfer assay.

This is not a SmartCar capture adapter. Provider locations remain auditable,
and missing transmit power, noise and CAD are not turned into measurements.
Only the reported RSSI point statistic is modelled. Frozen coefficient priors
are never changed; conditioning is the released field's existing operation.
"""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

from ..smartcar_numpy.geometry import link_geometry_features, query_physics_basis
from ..smartcar_numpy.model import MAX_CONTEXT, NumpyField

C_M_S = 299792458.
CARRIER_HZ = 5.18e9
FEATURE_KNOWN_MASK = np.array(
    [True] * 9 + [False, False, True, True, False, False, False], dtype=bool)


def _xyz(value, name):
    if np.iscomplexobj(value):
        raise ValueError(name + ' must be finite real XYZ')
    try:
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(name + ' must be finite real XYZ') from error
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(name + ' must be finite real XYZ')
    return result


def _text(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError('A nonempty public ' + key + ' is required')
    return value


def _real(value, name):
    if isinstance(value, (bool, str, bytes)) or np.iscomplexobj(value):
        raise ValueError(name + ' must be finite real')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name + ' must be finite real') from error
    if not math.isfinite(number):
        raise ValueError(name + ' must be finite real')
    return number


def _loss(rx, ap):
    """Positive-distance isotropic free-space loss; never a TX measurement."""
    distance = np.linalg.norm(np.asarray(rx, dtype=np.float64) - ap, axis=-1)
    if np.any(distance <= 0) or not np.isfinite(distance).all():
        raise ValueError('AP and receiver must have distinct finite positions')
    return 20. * np.log10(4. * math.pi * CARRIER_HZ * distance / C_M_S)


def _reject_query_labels(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError('Public query keys must be strings')
            lowered = key.lower()
            if ('rssi' in lowered or 'received_power' in lowered
                    or 'signal_db' in lowered or lowered in {'target', 'label', 'y'}):
                raise ValueError('Query RSSI/target labels must not enter the adapter')
            _reject_query_labels(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_query_labels(item)


def _catalogue(metadata, scenario):
    try:
        catalogue = metadata['scenarios'][scenario]
        frame = catalogue['frame']
        origin = _xyz(frame['origin_provider_xyz'], 'Provider-frame origin')
        dimensions = _xyz(frame['dimensions_m'], 'Computational dimensions')
        raw_aps, raw_locations = catalogue['aps'], catalogue['locations']
    except (KeyError, TypeError) as error:
        raise ValueError('Audited scenario AP/location/frame metadata is required') from error
    if (dimensions <= 0).any():
        raise ValueError('Computational dimensions must be positive')
    if not isinstance(raw_aps, list) or not 1 <= len(raw_aps) <= 128:
        raise ValueError('Between1 and128 public AP placements are required')
    if not isinstance(raw_locations, list) or not raw_locations:
        raise ValueError('Public location catalogue is required')
    aps, locations = {}, {}
    for raw, output, name in ((raw_aps, aps, 'AP'), (raw_locations, locations, 'location')):
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError('Public ' + name + ' catalogue rows must be mappings')
            identifier = _text(item, 'id')
            if identifier in output:
                raise ValueError('Duplicate public ' + name + ' identifier')
            provider = _xyz(item.get('xyz_provider'), name + ' provider XYZ')
            model = _xyz(item.get('xyz_model'), name + ' model XYZ')
            if not np.allclose(provider - origin, model, rtol=0., atol=1.e-7):
                raise ValueError('Provider/model coordinates disagree with audited translation')
            output[identifier] = (item, model)
    return dimensions, aps, locations, deepcopy(frame)


def prepare_episode(support, queries, metadata):
    """Build one scenario/phone episode without reading any query labels.

    Support must already be the common selected TRAIN/session01 rows, at most
    192; no hidden truncation or extra rows may inform the nuisance reference.
    Each row is the reader's point statistic from actual successful reports.
    TEST/session02 uses the same session01 nuisance, a declared transfer
    assumption rather than newly calibrated hardware. Unknown AP references
    produce explicit unsupported rows.
    """
    if (not isinstance(support, list) or not isinstance(queries, list)
            or len(support) > MAX_CONTEXT or not support + queries
            or not isinstance(metadata, dict)):
        raise ValueError('Explicit support/query lists and at most192 common support rows required')
    for row in support + queries:
        if not isinstance(row, dict):
            raise ValueError('Episode records must be mappings')
    for row in queries:
        _reject_query_labels(row)
    scenario, phone = _text((support + queries)[0], 'scenario'), _text((support + queries)[0], 'phone')
    android_versions = [row.get('android_version') for row in support + queries]
    if any(value is not None for value in android_versions):
        if (any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in android_versions) or len(set(android_versions)) != 1):
            raise ValueError('An episode must preserve one known receiver Android calibration group')
    android_version = android_versions[0]
    dimensions, aps, locations, frame = _catalogue(metadata, scenario)
    ap_ids = list(aps)
    ap_index = {identifier: index for index, identifier in enumerate(ap_ids)}
    ap_xyz64 = np.stack([aps[identifier][1] for identifier in ap_ids])
    support_ids, query_ids, seen, support_locations = [], [], set(), set()

    def validate(row, is_support):
        identifier = _text(row, 'id')
        if identifier in seen:
            raise ValueError('Duplicate support/query record identifier')
        seen.add(identifier)
        if _text(row, 'scenario') != scenario or _text(row, 'phone') != phone:
            raise ValueError('An episode must contain one scenario and receiver phone')
        session = _text(row, 'session')
        if session not in {'01', '02'} or (is_support and session != '01'):
            raise ValueError('Support is session01; public queries may use session01 or02')
        identifier_ap, location = _text(row, 'ap_id'), _text(row, 'location_id')
        if identifier_ap not in aps or location not in locations:
            raise ValueError('Record AP/location is absent from public catalogue')
        partition = _text(row, 'partition')
        if partition != locations[location][0].get('partition'):
            raise ValueError('Record partition disagrees with public whole-location partition')
        if (is_support and partition != 'train') or (not is_support and partition not in {'validation', 'test'}):
            raise ValueError('Support must be TRAIN and queries must be held-out locations')
        position = _xyz(row.get('xyz_model'), 'Record model XYZ')
        if not np.allclose(position, locations[location][1], rtol=0., atol=1.e-7):
            raise ValueError('Record coordinates disagree with public location catalogue')
        if not math.isclose(_real(row.get('carrier_hz'), 'Known carrier'), CARRIER_HZ, rel_tol=0., abs_tol=1.):
            raise ValueError('This measured assay admits known5180MHz reports only')
        key = (session, location, identifier_ap)
        return identifier, identifier_ap, location, position, key

    support_keys, query_keys, support_ap_ids, support_positions, values = set(), set(), [], [], []
    for row in support:
        identifier, identifier_ap, location, position, key = validate(row, True)
        if key in support_keys:
            raise ValueError('Duplicate support file/location/AP point statistic')
        support_keys.add(key); support_locations.add(location)
        value = _real(row.get('rssi_dbm'), 'Reported support RSSI')
        if value == -200.:
            raise ValueError('A missing RSSI sentinel is not a support measurement')
        support_ids.append(identifier); support_ap_ids.append(identifier_ap)
        support_positions.append(position); values.append(value)
    query_positions, query_ap_ids = [], []
    for row in queries:
        identifier, identifier_ap, location, position, key = validate(row, False)
        if key in query_keys:
            raise ValueError('Duplicate query file/location/AP point statistic')
        if location in support_locations:
            raise ValueError('Support and query whole locations must be disjoint')
        query_keys.add(key); query_ids.append(identifier)
        query_positions.append(position); query_ap_ids.append(identifier_ap)

    support_values = np.asarray(values, dtype=np.float64)
    support_groups = np.asarray([ap_index[a] for a in support_ap_ids], dtype=np.int64)
    support_rx64 = np.asarray(support_positions, dtype=np.float64).reshape(-1, 3)
    support_loss = _loss(support_rx64, ap_xyz64[support_groups])
    reference_by_ap = {a: float(np.mean((support_values + support_loss)[support_groups == index]))
                       for a, index in ap_index.items() if np.any(support_groups == index)}
    statuses, query_indices = [], []
    for index, (identifier, identifier_ap, position) in enumerate(zip(query_ids, query_ap_ids, query_positions)):
        if identifier_ap not in reference_by_ap:
            status, reason = 'unsupported', 'no_admitted_support_for_ap_device_reference'
        elif np.linalg.norm(position - ap_xyz64[ap_index[identifier_ap]]) <= 0:
            status, reason = 'unsupported', 'coincident_ap_query'
        else:
            status, reason = 'prediction', 'support_only_reference'
            query_indices.append(index)
        statuses.append({'query_id': identifier, 'status': status, 'reason': reason})

    def geometry(rx, groups):
        # Zero CAD is an explicitly missing active covariate, not an empty
        # measured room. TX/noise/quantization are inactive padding, not data.
        return link_geometry_features(rx, ap_xyz64[groups], CARRIER_HZ, 0,
            dimensions, np.empty((0, 2, 3)), calibration_pair_count=0,
            calibration_unknown=True, quantization_width_db=0.,
            noise_floor_dbm=0., tx_power_dbm=0.)

    view = query = None
    if support:
        q = geometry(support_rx64, support_groups)
        offsets = np.asarray([reference_by_ap[a] for a in support_ap_ids], dtype=np.float64)
        residual = support_values - (-support_loss + offsets)
        view = {'context': np.column_stack((residual / 30., np.zeros(len(support)),
                    np.ones(len(support)), np.zeros(len(support)), q)).astype(np.float32),
            'known': np.ones(len(support), dtype=bool), 'scheduled': np.ones(len(support), dtype=bool),
            'ap': support_groups, 'rx_xyz': support_rx64.astype(np.float32),
            'lower': residual, 'upper': residual.copy(), 'floor': residual.copy(),
            'fspl': -support_loss, 'offset': offsets,
            'weights': np.ones(len(support), dtype=np.float64)}
    if query_indices:
        groups = np.asarray([ap_index[query_ap_ids[i]] for i in query_indices], dtype=np.int64)
        rx64 = np.asarray([query_positions[i] for i in query_indices], dtype=np.float64)
        q = geometry(rx64, groups)
        query = {'q': q, 'ap': groups, 'rx_xyz': rx64.astype(np.float32),
            'basis': query_physics_basis(q), 'fspl': -_loss(rx64, ap_xyz64[groups]),
            'offset': np.asarray([reference_by_ap[query_ap_ids[i]] for i in query_indices], dtype=np.float64)}
    result = {'view': view, 'query': query, 'ap_xyz': ap_xyz64.astype(np.float32),
        'query_indices': query_indices, 'statuses': statuses, 'reference_by_ap': reference_by_ap,
        'support_ids': support_ids, 'support_ap_ids': support_ap_ids, 'support_rssi_dbm': support_values,
        'query_ids': query_ids, 'ap_ids': ap_ids, 'feature_known_mask': FEATURE_KNOWN_MASK.copy(),
        'assumptions': {'scenario': scenario, 'phone': phone, 'android_version': android_version, 'frame': frame,
            'frame_is_computational_not_a_measured_floor': True,
            'full_prediction_translation_invariant': False,
            'reference': 'mean_support(reported_RSSI_plus_FSPL_loss)_per_AP_and_phone',
            'reference_identifies_tx_or_device_calibration': False,
            'reference_absorbs_mean_environmental_loss': True,
            'support_session': '01', 'session02_reuses_session01_reference': True,
            'point_target': 'reader_reported_RSSI_point_statistic',
            'detector_floor_observed': False, 'floor_equals_point_for_internal_validator_only': True,
            'unknown_cad_active_features_zero_imputed': True,
            'inactive_unknown_tx_noise_quantization_slots_zero_padded': True,
            'single_carrier_hz': CARRIER_HZ, 'global_physics_basis_rank_at_most': 2,
            'spread_includes_nuisance_reference_uncertainty': False,
            'packet_risk_qualified': False, 'hardware_accuracy_qualified': False,
            'query_labels_used': False, 'frozen_prior_or_weights_changed': False}}
    _validate_redundant_fields(result)
    return result


def _validate_redundant_fields(prepared):
    """The released core cannot detect conflicting redundant event fields."""
    view = prepared['view']
    if view is None:
        if len(prepared['support_ids']):
            raise ValueError('Missing view for admitted support')
        return
    count = len(prepared['support_ids'])
    if (not 1 <= count <= MAX_CONTEXT or len(set(prepared['support_ids'])) != count
            or view['context'].shape != (count, 20)
            or not np.array_equal(view['lower'], view['upper'])
            or not np.array_equal(view['floor'], view['lower'])
            or not np.array_equal(view['context'][:, 0], (view['lower'] / 30.).astype(np.float32))
            or np.any(view['context'][:, 1] != 0.) or np.any(view['context'][:, 2] != 1.)
            or np.any(view['context'][:, 3] != 0.)
            or not np.all(view['known']) or not np.all(view['scheduled'])
            or not np.array_equal(view['weights'], np.ones(count))
            or not np.allclose(view['lower'] + view['fspl'] + view['offset'],
                               prepared['support_rssi_dbm'], atol=1.e-12, rtol=0.)):
        raise ValueError('Redundant reported RSSI/reference/point-event fields disagree')


def predict(model, prepared):
    """Condition an already loaded frozen field; preserve unsupported queries.

    Returns the common numerical/neural row schema. This function is inference
    only and does not load, select, fit or mutate checkpoint arrays.
    """
    if not isinstance(model, NumpyField):
        raise ValueError('An explicitly loaded frozen NumPy field is required')
    from ..smartcar_numpy.export import RELEASED_STATES, state_sha256
    if state_sha256(model.arrays) not in RELEASED_STATES.values():
        raise ValueError('Only the three exact released frozen member states are admitted')
    _validate_redundant_fields(prepared)
    rows = [{**row, 'mean_dbm': None, 'std_db': None} for row in prepared['statuses']]
    if prepared['query'] is None:
        return rows
    if prepared['view'] is None:
        raise ValueError('Supported queries require actual admitted support')
    belief = model.encode_view(prepared['view'], prepared['ap_xyz'])
    mean, std = model.render_queries(belief, prepared['query'])
    if len(mean) != len(prepared['query_indices']):
        raise ValueError('Prediction/query axes disagree')
    for index, value, spread in zip(prepared['query_indices'], mean, std):
        rows[index].update(mean_dbm=float(value), std_db=float(spread))
    return rows
