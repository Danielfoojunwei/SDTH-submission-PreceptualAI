"""Matched public-report adapters for the new spatial field models.

This module does not create worlds, train models or execute radio actions.
The research runner must bind its models/priors and cases before calling it.
The warehouse is an explicit public rack extrusion, not RT ground truth.
Native 6.5 GHz is a separate, unqualified extrapolation beyond RT's 6.0 GHz.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time

import numpy as np
import torch

from ..adaptive_radio.service import (
    RSSIForecast, RSSIOutcomeAdapter, BridgedPredictor, NATIVE_CARRIERS_HZ,
)
from ..native.world import HORIZONS
from ..native_bridge.predictor import (
    AP_HEIGHT_M, RX_HEIGHT_M, WIDTH, HEIGHT, public_features, radio_assumptions,
)
from ..native_bridge_ops.operators import PublicQueryGeometry, query_operands
from ..spatial_control.memory import (
    NATIVE_GAUGE, _native_arrays, _native_events, native_history_memory,
)
from ..spatial_control.runtime import SupportAdmission
from ..spectral_scene.operators import fspl_received_power_dbm

MAX_CONTEXT = 192
ENCODER_MAX_AGE_S = 60.0
AGE_DISTANCE_M_PER_S = 2.0
SUPPORT_RETENTION_S = 6.0
SOURCE_BANDS = (0, 1)
METHODS = ('hierarchy', 'global_reg', 'matched_radiomap', 'full_radiomap')
MODEL_ORIGIN_M = (WIDTH/2., HEIGHT/2., 0.)


def _model_xyz(physical_xyz):
    """Express native public positions in the centered RT model frame."""
    return (np.asarray(physical_xyz, dtype=float)-np.asarray(MODEL_ORIGIN_M)).astype('f4')


def _model_features(physical_features):
    result = np.array(physical_features, copy=True)
    # Rack intersections and all physical operators were evaluated in the
    # original native frame. Translation changes only absolute XY features.
    result[:, [0, 1, 3, 4]] -= .5
    return result


def _real(value, name, *, minimum=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value)) or (minimum is not None and value < minimum)):
        raise ValueError(name+' must be a finite real within its declared range')
    return float(value)


def _sha(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)):
        raise ValueError(name+' must be a lowercase SHA-256 identity')
    return value


def _geometry(geometry):
    if getattr(geometry, 'yard', False):
        raise ValueError('Public warehouse rack extrusion does not describe a yard')
    public = PublicQueryGeometry(geometry.sites, geometry.tx, geometry.noise)
    if len(public.sites) > 96:
        raise ValueError('The 192-row matched budget requires at least one row per AP/source band')
    return public


def _plan(plan, robots):
    value = np.asarray(plan, dtype=float)
    if value.shape != (robots, len(HORIZONS), 2) or not np.isfinite(value).all():
        raise ValueError('A finite public robot-by-horizon XY plan is required')
    return value


def _payload_index(payloads):
    result = {}
    for item in payloads:
        if not isinstance(item, dict):
            raise ValueError('Actual completed scan payloads are required')
        try:
            key = (item['robot'], item['measurement_slot'], item['complete_slot'], item['payload_sha256'])
        except KeyError as error:
            raise ValueError('Incomplete completed-payload identity') from error
        if any(type(value) is not int or value < 0 for value in key[:3]):
            raise ValueError('Completed-payload clocks and robot identity must be nonnegative integers')
        _sha(key[3], 'Completed-payload hash')
        if key in result and result[key] is not item:
            # Header indexing must never compare local-only RF dictionaries.
            raise ValueError('Ambiguous local scan payload identity')
        result[key] = item
    return result


def _check_delivered_row(event, payload, row_value, row_heard, row_bound, row_position):
    """A server row may use only the payload that was actually delivered."""
    if payload is None:
        raise ValueError('A post-warm server report lacks its actual completed payload')
    try:
        processed = payload['processed_slot']
        values = np.asarray(payload['value_dbm'])
        heard = np.asarray(payload['heard'])
        position = np.asarray(payload['position_m'], dtype=float)
        bound = _real(payload['upper_dbm'], 'Actual censor bound')
        gauge = payload['reference_id']
    except KeyError as error:
        raise ValueError('Incomplete completed report semantics') from error
    if (type(processed) is not int or not event['complete_slot'] <= processed <= event['delivery_slot']
            or gauge != NATIVE_GAUGE or values.shape != row_value.shape or heard.shape != row_heard.shape
            or heard.dtype != np.bool_ or values.dtype.kind not in 'fi'
            or not np.isfinite(values).all() or position.shape != (3,) or not np.isfinite(position).all()
            or position[2] != RX_HEIGHT_M):
        raise ValueError('Delivered payload clocks, axes, gauge or position disagree')
    wire = values.astype('f4')  # ClockedScans' actual serialized receiver values
    if (not np.isfinite(wire).all() or not np.array_equal(wire.astype(float), values.astype(float))
            or hashlib.sha256(wire.tobytes()+heard.tobytes()).hexdigest() != event['payload_sha256']
            or not np.array_equal(wire.astype(row_value.dtype), row_value) or not np.array_equal(heard, row_heard)
            or not np.array_equal(position[:2].astype(row_position.dtype), row_position)
            or not np.all(row_bound == np.asarray(bound, dtype=row_bound.dtype))):
        raise ValueError('Delivered payload does not match the actual server-history row')


def _coverage_selection(positions, ages, plan, count):
    """Greedy facility coverage using only public positions and known ages.

    A two-metre-per-second age penalty is a fixed retrieval heuristic, not a
    wireless coherence law. Equal scores use the existing chronological/robot
    ordering. No RSSI value or heard-before-censored preference enters choice.
    """
    near = np.asarray(HORIZONS) <= 5.0
    queries = np.unique(plan[:, near].reshape(-1, 2), axis=0)
    with np.errstate(over='ignore', invalid='ignore'):
        distances = np.sum((positions[:, None]-queries[None])**2, axis=-1)
        costs = distances+(ages[:, None]*AGE_DISTANCE_M_PER_S)**2
    if not np.isfinite(costs).all():
        raise ValueError('Public retrieval coordinates exceed finite supported geometry')
    chosen = []
    current = None
    for _ in range(min(count, len(positions))):
        score = costs.mean(1) if current is None else np.minimum(costs, current).mean(1)
        if chosen:
            score[chosen] = np.inf
        index = int(np.argmin(score))
        chosen.append(index)
        current = costs[index].copy() if current is None else np.minimum(current, costs[index])
    return np.asarray(chosen, dtype=np.int64)


def select_context(past, geometry, plan, *, issue_time_s, events=(),
                   local_completed_payloads=(), maximum=MAX_CONTEXT):
    """Select the same genuine reports for all matched native predictors.

    With 16 APs, six real full scans provide 192 AP/source-band rows. Warm
    captures retain one-second intervals; post-warm rows require exact
    payload-matched measurement/completion/delivery clocks. A missing scan is
    not an observation, and an omitted AP's actual upper bound stays a bound.
    """
    public = _geometry(geometry)
    positions, values, heard, bound, scanned = _native_arrays(past)
    sites = len(public.sites)
    issue = _real(issue_time_s, 'Forecast issue clock', minimum=0)
    if len(values) != 60+math.floor(issue) or values.shape[2] != 3*sites:
        raise ValueError('Use the exact strictly earlier native history prefix and current AP axes')
    plan = _plan(plan, values.shape[1])
    if type(maximum) is not int or not 1 <= maximum <= MAX_CONTEXT or maximum % (2*sites):
        raise ValueError('The finite matched budget must divide evenly across AP/source-band groups')
    per_group = maximum//(2*sites)
    event_rows = _native_events(events, issue=issue, warm_bins=60, bin_seconds=1.,
        slots_per_second=100, n_robots=values.shape[1], history_length=len(values))
    payloads = _payload_index(local_completed_payloads)
    candidates = []
    for t, robot in zip(*np.nonzero(scanned)):
        t, robot = int(t), int(robot)
        if t < 60:
            measured = (float(t-60), float(t-59))
            arrival = (np.nan, np.nan)
            provenance = 0  # explicitly unknown arrival, known available by zero
        else:
            event = event_rows.get((t, robot))
            if event is None:
                raise ValueError('Post-warm scanned rows require exact delivered measurement clocks')
            payload = payloads.get((robot, event['measurement_slot'], event['complete_slot'], event['payload_sha256']))
            _check_delivered_row(event, payload, values[t, robot], heard[t, robot], bound[t, robot], positions[t, robot])
            measured = (event['measurement_slot']/100.,)*2
            arrival = (event['delivery_slot']/100.,)*2
            provenance = 1
        age = issue-measured[0]
        if 0 <= age <= ENCODER_MAX_AGE_S:
            candidates.append((t, robot, measured, arrival, provenance, age))
    if not candidates:
        raise ValueError('No actual causally timed scan can supply the context; do not invent one')
    # Latest timestamps, then stable robot/history order, define deterministic
    # tie handling for the geometry/age-only facility heuristic.
    candidates.sort(key=lambda row: (row[5], row[1], row[0]))
    candidate_positions = np.asarray([positions[row[0], row[1]] for row in candidates])
    candidate_ages = np.asarray([row[5] for row in candidates])
    chosen = _coverage_selection(candidate_positions, candidate_ages, plan, per_group)
    scans = [candidates[index] for index in chosen]
    selected = np.asarray([(row[0], row[1], site, band)
                           for row in scans for site in range(sites) for band in SOURCE_BANDS], dtype=np.int64)
    t, robot, ap_index, band_index = selected.T
    link = ap_index*3+band_index
    rx = np.column_stack((positions[t, robot], np.full(len(t), RX_HEIGHT_M)))
    aps = np.column_stack((public.sites, np.full(sites, AP_HEIGHT_M)))
    frequency = np.asarray(NATIVE_CARRIERS_HZ)[band_index]
    q = _model_features(public_features(rx, aps[ap_index], frequency, public.noise[band_index], public.tx[band_index]))
    fspl = fspl_received_power_dbm(rx, aps[ap_index], frequency, public.tx[band_index])
    h = heard[t, robot, link]
    point = values[t, robot, link]
    upper = bound[t, robot, link]
    if np.any(point[h] < -90.-1.e-5):
        raise ValueError('A heard native report cannot be below its declared detector floor')
    lower_residual = np.where(h, point-fspl, -np.inf)
    upper_residual = np.where(h, point-fspl, upper-fspl)
    context = np.column_stack((np.where(h, (point-fspl)/30., 0.),
        np.where(~h, (upper-fspl)/30., 0.), h, ~h, q)).astype('f4')
    repeat = sites*2
    measurement = np.repeat(np.asarray([row[2] for row in scans]), repeat, axis=0)
    arrival = np.repeat(np.asarray([row[3] for row in scans]), repeat, axis=0)
    view = {'context': context, 'known': np.ones(len(t), bool), 'scheduled': np.ones(len(t), bool),
        'ap': ap_index.copy(), 'rx_xyz': _model_xyz(rx), 'lower': lower_residual.astype('f8'),
        'upper': upper_residual.astype('f8'), 'floor': (-90.-fspl).astype('f8'),
        'fspl': fspl.astype('f8'), 'offset': np.zeros(len(t), dtype='f8')}
    metadata = {'selected_rows': selected, 'selected_scan_rows': np.asarray([(x[0], x[1]) for x in scans], dtype='i8'),
        'measurement_intervals_s': measurement, 'arrival_intervals_s': arrival,
        'clock_provenance': np.repeat(np.asarray([row[4] for row in scans], dtype='i1'), repeat),
        'conservative_age_s': np.repeat(np.asarray([row[5] for row in scans]), repeat),
        'candidate_scans': len(candidates), 'selected_scans': len(scans), 'maximum_tokens': maximum,
        'actual_tokens': len(t), 'tokens_per_ap_source_band': len(scans),
        'source_history_length': len(values), 'selection_uses_rssi_amplitudes_or_heard_preference': False,
        'age_distance_m_per_s': AGE_DISTANCE_M_PER_S,
        'maximum_observation_age_s': ENCODER_MAX_AGE_S,
        'model_coordinate_origin_m': MODEL_ORIGIN_M,
        'model_coordinate_frame': 'centered public warehouse XY; subtract origin from native positions',
        'physical_context_rx_xyz_m': rx.astype('f4'), 'physical_ap_xyz_m': aps.astype('f4'),
        'warm_clock_scope': 'one-second measurement interval; arrival unknown but available by zero',
        'position_scope': 'public planned positions at horizons up to five seconds; no future RF',
        'point_scope': 'heard continuous serialized report treated as a point with working model discrepancy'}
    return {'view': view, 'ap_xyz': _model_xyz(aps), 'metadata': metadata}


def make_query(geometry, plan):
    """Carrier-specific physical query arrays in native robot/horizon/AP order."""
    public = _geometry(geometry)
    plan = np.asarray(plan, dtype=float)
    if plan.ndim != 3:
        raise ValueError('A public robot/horizon XY plan is required')
    plan = _plan(plan, len(plan))
    operands = query_operands(public, plan)
    shape = operands['shape']
    rx = np.broadcast_to(np.concatenate((plan, np.full((*plan.shape[:2], 1), RX_HEIGHT_M)), axis=-1)[:, :, None, None],
                         shape+(3,)).reshape(-1, 3)
    groups = np.broadcast_to(np.arange(len(public.sites))[None, None, :, None], shape).reshape(-1)
    query = {'q': _model_features(operands['query']), 'ap': groups.astype('i8'), 'rx_xyz': _model_xyz(rx),
             'basis': operands['basis'], 'fspl': operands['fspl'], 'offset': operands['offset']}
    return query, shape


def _batched(values):
    result = {}
    for name, array in values.items():
        a = np.array(array, copy=True)
        if a.dtype.kind in 'iu': a = a.astype('i8')
        elif a.dtype.kind == 'b': pass
        elif a.dtype.kind == 'f': a = a.astype('f4')
        else: raise ValueError('Only numeric public input arrays can enter a field model')
        result[name] = torch.from_numpy(a)[None]
    return result


class NativeFieldPredictor:
    """One matched-context field model; no World or private RF argument."""
    def __init__(self, method, geometry, sensor, *, model=None, model_sha256=None,
                 prior=None, prior_sha256=None):
        self.public_geometry = _geometry(geometry)
        if method not in ('hierarchy', 'global_reg', 'matched_radiomap'):
            raise ValueError('A registered matched field method is required')
        if not isinstance(getattr(sensor, 'events', None), list) or not isinstance(getattr(sensor, 'local_completed_payloads', None), list):
            raise ValueError('Actual server-delivery and local-completion ledgers are required')
        if method == 'matched_radiomap':
            from .baselines import Prior
            if model is not None or model_sha256 is not None or not isinstance(prior, Prior) or prior.method != 'radiomap_operator':
                raise ValueError('Matched numerical forecasts require the selected TRAIN/VAL RadioMap prior')
            identity = _sha(prior_sha256, 'Bound numerical-prior identity')
        else:
            if prior is not None or prior_sha256 is not None or not callable(getattr(model, 'encode_view', None)) or not callable(getattr(model, 'render_queries', None)):
                raise ValueError('A frozen field model with encode/render APIs is required')
            cells = ('hier_reg', 'hier_joint') if method == 'hierarchy' else ('global_reg',)
            if getattr(model, 'config', {}).get('cell') not in cells:
                raise ValueError('Field branch differs from the registered native method')
            identity = _sha(model_sha256, 'Bound checkpoint identity')
            model.eval()
        self.method, self.geometry, self.sensor = method, geometry, sensor
        self.model, self.prior = model, prior
        self.source_id = 'hierarchical_radio_native_'+method
        self.identity_sha256 = identity
        self.records = []; self.memories = {}; self.support_preparation = []
        self.last_history = None

    @torch.no_grad()
    def step(self, past, plan):
        began = time.perf_counter()
        decision = len(self.records)
        if len(past.value) != 60+decision:
            raise ValueError('Native forecasts require their exact FIFO past prefix')
        self.last_history = {name: np.array(getattr(past, name), copy=True)
            for name in ('positions', 'value', 'heard', 'bound', 'scanned')}
        support_started = time.perf_counter()
        self.memories[decision] = native_history_memory(past, issue_time_s=float(decision),
            events=self.sensor.events, retention_age_s=SUPPORT_RETENTION_S)
        self.support_preparation.append({'decision': decision,
            'elapsed_seconds': time.perf_counter()-support_started,
            'reports': len(self.memories[decision]), 'retention_age_seconds': SUPPORT_RETENTION_S,
            'snapshot_metadata': dict(self.memories[decision].snapshot_metadata),
            'matched_forecast_context_budget': MAX_CONTEXT,
            'guard_modifies_issued_forecast': False, 'packet_risk_qualified': False})
        context = select_context(past, self.geometry, plan, issue_time_s=float(decision),
            events=self.sensor.events, local_completed_payloads=self.sensor.local_completed_payloads)
        view = context['view']; query, shape = make_query(self.geometry, plan)
        state = None
        if self.method == 'matched_radiomap':
            from .baselines import predict
            mean, std = predict(self.prior, {'a': view, 'query': query, 'ap_xyz': context['ap_xyz']}, view='a')
        else:
            ap_xyz = torch.from_numpy(np.array(context['ap_xyz'], copy=True))[None]
            state = self.model.encode_view(_batched(view), ap_xyz)
            mean, std = self.model.render_queries(state, _batched(query))
            mean, std = mean[0].cpu().numpy(), std[0].cpu().numpy()
        forecast = RSSIForecast(HORIZONS, np.zeros(len(plan), dtype='i8'),
            np.asarray(mean).reshape(shape), np.asarray(std).reshape(shape), NATIVE_CARRIERS_HZ,
            self.source_id, self.identity_sha256)
        record = {'decision': decision, 'history_length': len(past.value),
            'context': view['context'].copy(), 'known': view['known'].copy(),
            'scheduled': view['scheduled'].copy(), 'selected_rows': context['metadata']['selected_rows'].copy(),
            'selected_scan_rows': context['metadata']['selected_scan_rows'].copy(),
            'measurement_intervals_s': context['metadata']['measurement_intervals_s'].copy(),
            'arrival_intervals_s': context['metadata']['arrival_intervals_s'].copy(),
            'clock_provenance': context['metadata']['clock_provenance'].copy(),
            'context_age_s': context['metadata']['conservative_age_s'].copy(),
            'context_ap': view['ap'].copy(), 'context_rx_xyz': view['rx_xyz'].copy(),
            'context_lower': view['lower'].copy(), 'context_upper': view['upper'].copy(),
            'context_floor': view['floor'].copy(), 'context_fspl': view['fspl'].copy(),
            'context_offset': view['offset'].copy(), 'ap_xyz': context['ap_xyz'].copy(),
            'physical_context_rx_xyz_m': context['metadata']['physical_context_rx_xyz_m'].copy(),
            'physical_ap_xyz_m': context['metadata']['physical_ap_xyz_m'].copy(),
            'physical_query_rx_xyz_m': np.broadcast_to(
                np.concatenate((np.asarray(plan), np.full((*np.asarray(plan).shape[:2], 1), RX_HEIGHT_M)), axis=-1)[:, :, None, None],
                shape+(3,)).reshape(-1, 3).astype('f4'),
            'plan': np.array(plan, copy=True), 'mean_dbm': forecast.mean_dbm.copy(), 'std_db': forecast.std_db.copy(),
            'context_metadata': {key: value for key, value in context['metadata'].items() if not isinstance(value, np.ndarray)}}
        for key, value in query.items():
            record['query_'+key] = np.array(value, copy=True)
        if state is not None:
            for key in ('global_mean', 'global_covariance', 'local_mean', 'local_cholesky', 'local_coupling'):
                if isinstance(state.get(key), torch.Tensor):
                    record['belief_'+key] = state[key][0].cpu().numpy().copy()
            inference = state.get('inference_log', ())
            for key in ('objective_before', 'objective_after', 'step_size', 'null_step'):
                if inference:
                    record['inference_'+key] = np.asarray([
                        step[key][0].detach().cpu().numpy().item() for step in inference])
        self.records.append(record)
        return forecast, 'cold' if decision == 0 else 'warm', time.perf_counter()-began


class MatchedSupportAdmission(SupportAdmission):
    """The frozen support algorithm with accurate new-context provenance."""
    def __call__(self, decision):
        before = len(self.records)
        result = super().__call__(decision)
        tokens = len(self.raw.records[decision.decision]['context'])
        for record in self.records[before:]:
            record['support_information'] = 'extra delivered history; matched forecast context unchanged at admission'
            record['forecast_context_tokens'] = tokens
            record['forecast_context_budget'] = MAX_CONTEXT
        return result


@dataclass(frozen=True)
class PredictorBundle:
    raw: NativeFieldPredictor | None
    predictor: object
    support_enabled: bool


def build_predictor(method, geometry, sensor, *, model=None, model_sha256=None,
                    prior=None, prior_sha256=None):
    """Build a non-actuating predictor after the caller's publication guard.

    Full-history RadioMap remains a separate stronger-information reference.
    The other three arms share exactly the same source-report selection and
    support admission. None of these paths qualifies RF or packet uncertainty.
    """
    if method not in METHODS:
        raise ValueError('Unknown finite native field method')
    _geometry(geometry)
    if method == 'full_radiomap':
        if any(value is not None for value in (model, model_sha256, prior, prior_sha256)):
            raise ValueError('The full-history reference does not consume learned field artifacts')
        from ..closed_loop.runtime import SinglePredictor
        return PredictorBundle(None, SinglePredictor(geometry, 'radiomap'), False)
    raw = NativeFieldPredictor(method, geometry, sensor, model=model, model_sha256=model_sha256,
        prior=prior, prior_sha256=prior_sha256)
    adapter = RSSIOutcomeAdapter(radio_assumptions(geometry), raw.source_id,
        site_id='native_public_warehouse', device_id='native_zero_gauge_simulation')
    return PredictorBundle(raw, BridgedPredictor(raw, adapter), True)
