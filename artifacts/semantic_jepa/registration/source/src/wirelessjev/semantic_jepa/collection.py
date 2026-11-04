"""Collect shared training observations through the actual delivered packet path.

The collector uses one fixed numerical policy for every learned family. Its
forecast service profile is a declared training-data assumption, not measured
hardware qualification. Only the first sixty timestamps are a conditional
simulated warm start; all subsequent RF reports must complete and arrive over
the native simulated control transport.

The caller owns scientific registration and archives the returned receipt.
This module neither registers experiments nor writes training/evaluation data.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np

from ..closed_loop.control import OutageUtility
from ..closed_loop.telemetry import DeliveredScans
from ..contracted.data import READINGS, runtime_world
from ..fleetwm.data import episode
from ..frontier.study import mask_scans
from ..gate_repair.gate import reactive_rescue
from ..hardware_budget import OutageBudgetGate
from ..native.intent_policy import intent_policy
from .runtime import ChargedForecasts, Predictor
from .sim import simulate


COLLECTION_PROFILE_SECONDS = {'cold': 1., 'warm': .2, 'refit': 1.}
EXTRA_COMPUTE_SECONDS = .5
SCAN_PERIOD_SECONDS = 5
SCAN_OUTAGE_SECONDS = .09
FUTURE_CONTEXT_SECONDS = 30
OBSERVATION_FIELDS = ('positions', 'value', 'heard', 'bound', 'scanned')
STABLE_JOB_FIELDS = (
    'decision', 'history_cutoff_exclusive', 'ready_slot', 'charged_seconds',
    'kind', 'forecast_sha256', 'heard_entries', 'scans_last_bin',
)


def _value_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_digest(value):
    value = np.ascontiguousarray(value)
    header = json.dumps({'dtype': value.dtype.str, 'shape': list(value.shape)},
                        sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(header + b'\n' + value.tobytes()).hexdigest()


def _validate_request(seed, n, query_duration, condition, index):
    for name, value, minimum in [('seed', seed, 0), ('n', n, 1),
                                 ('query_duration', query_duration, 1),
                                 ('index', index, 0)]:
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if condition not in ('main', 'stress', 'drift'):
        raise ValueError('Use a declared main, stress or drift collection')


def collect(seed, n, query_duration, condition, index):
    """Return ``(full, geometry, delivered_observations, stable_receipt)``.

    ``full`` contains private training labels and is kept separate from the
    observed-history object. The prediction boundary receives only public
    geometry, a strictly past delivered prefix and the public planned route.
    Training queries last ``query_duration`` seconds; collection continues for
    thirty more seconds so every future-context label comes from a completed
    delivered rollout rather than the old independent scan stream.

    Actual wall-clock timings are deliberately omitted from this deterministic
    receipt. They do not determine the assumed training collection queue and
    must not be mistaken for measured service-budget or runtime qualification.
    """
    _validate_request(seed, n, query_duration, condition, index)
    duration = query_duration + FUTURE_CONTEXT_SECONDS
    full, _, geometry, original = episode(seed, n, duration, condition)
    warm = mask_scans(original, SCAN_PERIOD_SECONDS).before(60)
    delivered = DeliveredScans(warm, duration, seed)
    predictor = Predictor(geometry, 'radiomap')
    profile = dict(COLLECTION_PROFILE_SECONDS)
    live = ChargedForecasts(predictor, delivered, duration, profile,
                           EXTRA_COMPUTE_SECONDS)
    cost = .03 if (index // 4) % 2 == 0 else 4.
    utility = OutageUtility(live.mean, live, cost, 30) if cost == 4. else None
    admission = utility if utility is not None else OutageBudgetGate(live.mean, live, cost)
    reading = READINGS['bulk_uploads']
    policy = replace(intent_policy(reading['weights'], reading['focus'], dual_hardware='none'),
                     name='collection_radiomap', online_map=False, gate=True,
                     scan_period_s=SCAN_PERIOD_SECONDS, cadence_scan=True,
                     roam_cost_s=cost)
    world = runtime_world(full, duration, live.mean, live.std[:, :, 0])
    result = simulate(world, policy, roam_gate=reactive_rescue,
                      handover_admission=admission, forecast_queue=live,
                      forecast_source=live, scan_observer=delivered,
                      scan_duration_s=SCAN_OUTAGE_SECONDS,
                      roam_scorer=utility.scores if utility is not None else None)

    observed = delivered.history
    jobs = [{name: row[name] for name in STABLE_JOB_FIELDS} for row in live.records]
    if len(jobs) != duration or any(row['decision'] != d or
                                  row['history_cutoff_exclusive'] != 60 + d
                                  for d, row in enumerate(jobs)):
        raise RuntimeError('Collection forecast prefix is not causal and complete')
    telemetry = delivered.summary()
    events = telemetry.pop('events')
    if any(event['delivery_slot'] >= event['first_usable_decision'] * 100 or
           not event['measurement_slot'] < event['complete_slot'] <= event['delivery_slot']
           for event in events):
        raise RuntimeError('Collection contains a premature RF report')
    postwarm = observed.scanned[60:]
    # Delayed cadence can place two reports in one delivery bin. The history
    # retains the latest one; received-event count is not occupied-bin count.
    delivered_bins = sorted({(event['delivery_slot'] // 100, event['robot'])
                             for event in events})
    observed_bins = [tuple(map(int, index)) for index in np.argwhere(postwarm)]
    if observed_bins != delivered_bins:
        raise RuntimeError('Delivered history does not match received reports')
    if np.any(observed.heard[60:] & ~postwarm[..., None]):
        raise RuntimeError('A missing post-warm scan became a measurement')
    observations_sha256 = {name: _array_digest(getattr(observed, name))
                           for name in OBSERVATION_FIELDS}
    receipt = {
        'schema': 1,
        'completed': True,
        'seed': seed,
        'robots': n,
        'query_duration_seconds': query_duration,
        'collection_duration_seconds': duration,
        'condition': condition,
        'index': index,
        'collection_handover_seconds': cost,
        'collector_policy': {
            'family': 'radiomap', 'mission': 'bulk_uploads',
            'controller': 'outage_utility_30' if utility is not None else 'outage_budget',
            'reactive_rescue': True, 'dual_hardware': 'none',
            'scan_period_seconds': SCAN_PERIOD_SECONDS,
            'scan_outage_seconds': SCAN_OUTAGE_SECONDS,
            'profile_seconds': profile,
            'extra_compute_seconds': EXTRA_COMPUTE_SECONDS,
        },
        'collection_profile_is_training_assumption': True,
        'collection_profile_is_measured_qualification': False,
        'warm_start_seconds': 60,
        'warm_start_is_conditional_simulated_survey': True,
        'postwarm_observations_use_actual_deliveries_only': True,
        'future_truth_is_predictor_input': False,
        'latest_observation_strictly_before_decision': True,
        'training_collection_shared_by_all_families': True,
        'actual_wall_clock_timings_omitted_from_stable_receipt': True,
        'forecast_jobs': jobs,
        'telemetry': telemetry,
        'delivered_history_bins': [list(index) for index in delivered_bins],
        'delivery_bin_collision_events': len(events) - len(delivered_bins),
        'telemetry_events': events,
        'telemetry_events_sha256': _value_digest(events),
        'observations_sha256': observations_sha256,
        'mean_forecast_sha256': _array_digest(live.mean),
        'std_forecast_sha256': _array_digest(live.std),
        'action_trace_sha256': _value_digest(result['packet_detail']['action_trace']),
        'packet_result_sha256': _value_digest(result),
        'simulation_only': True,
    }
    return full, geometry, observed, receipt
