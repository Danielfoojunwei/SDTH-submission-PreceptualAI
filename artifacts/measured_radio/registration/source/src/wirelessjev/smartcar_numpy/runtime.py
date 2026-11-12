"""Torch-free shadow adapter preserving the existing real-survey boundaries.

This prepares the same admitted reports and geometry as the reference backend.
Prediction equivalence is a software claim; hardware accuracy remains unknown.
"""
from __future__ import annotations
import math
import time
import numpy as np
from ..smartcar_runtime.contracts import records_sha256
from ..smartcar_runtime.field import (
    _inputs, _heard_rows, _unique_latest, _coverage, _support, _row,
    MAX_CONTEXT, MAX_SUPPORT_AGE_SECONDS, C_M_S,
)

def prepare_numpy_inputs(site, records, query_xyz, *, now_ns=None):
    """Build heard-only public tensors; no model, training or tracing is called.

    The site frame is floor-origin. XY is translated to the centered RT frame;
    AP/RX distances and CAD intersections are unchanged. Declared quantization
    supplies bins, otherwise continuous RSSI supplies genuine point events.
    There is no observed hardware detector floor in this contract. The finite
    internal ``floor`` equals each supplied event's lower endpoint solely for
    the inherited heard-event validator; it contributes no extra observation.
    """
    from .geometry import link_geometry_features, query_physics_basis
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


def predict_numpy(site, records, query_xyz, export_path, expected_sha256, expected_source_sha256):
    """Render the pinned numerical model in shadow, without importing Torch."""
    from .export import load_export
    from .model import NumpyField
    began = time.perf_counter()
    prepared = prepare_numpy_inputs(site, records, query_xyz)
    arrays, metadata = load_export(export_path, expected_sha256, expected_source_sha256)
    model = NumpyField(arrays)
    if prepared['query'] is not None:
        belief = model.encode_view(prepared['view'], prepared['ap_xyz'])
        mean, std = model.render_queries(belief, prepared['query'])
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise FloatingPointError('Nonfinite numerical RadioJEPA shadow forecast')
        for row_index, mu, sigma in zip(prepared['query_row_indices'], mean, std):
            prepared['rows'][row_index].update(status='prediction', mean_dbm=float(mu), std_db=float(sigma))
    elapsed = time.perf_counter()-began
    identity = {key: metadata[key] for key in ('cell', 'seed', 'source_checkpoint_sha256', 'export_sha256')}
    for row in prepared['rows']:
        row.update(identity)
        row.update(records_sha256=prepared['records_sha256'],
                   context_observations=len(prepared['selected_context']),
                   model_origin_xyz_m=prepared['model_origin_xyz_m'],
                   inference_elapsed_seconds=elapsed,
                   backend='numpy_scipy', experimental_shadow_only=True)
        row['assumptions'] = {'reference_gauge_assumed': True,
            'front_end_calibration_identified': False,
            'carrier_and_hardware_transfer_unvalidated': True,
            'observations_are_downlink_scan_RSSI_not_SINR_or_packet_risk': True,
            'source_carrier_interval_mhz': [2400., 5200.],
            'unreported_APs_are_unknown': True, 'invented_censor_bounds': False,
            'quantization_is_declared_receiver_assumption': True,
            'internal_floor_is_supplied_event_lower_not_measured_detector_threshold': True,
            'support_rule_is_engineering_heuristic_not_calibration': True,
            'capture_age_uses_desktop_request_window_not_cross_host_clock_subtraction': True}
    return prepared['rows']

