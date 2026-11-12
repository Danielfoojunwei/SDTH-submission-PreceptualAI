"""Finite, source-bound warehouse transfer; passive verification never forwards.

The 32 comparisons and two wiring controls are fixed before native data exists.
No result chooses a model, seed, threshold, default or hardware action.
"""
from __future__ import annotations

from copy import copy
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import time
from types import SimpleNamespace

import numpy as np

from . import common as c

FIELDS = ('positions', 'value', 'heard', 'bound', 'scanned')
WORLD_FIELDS = ('sites', 'snr', 'rssi_dl', 'domain', 'avail', 'video_mbps',
                'bulk_mbps', 'noise_db', 'channel', 'plan', 'pos', 'tx_dl')
METHODS = ('full_radiomap', 'matched_radiomap', 'hierarchy', 'global_reg')
PARITY_METHODS = ('full_radiomap', 'radiomap_roundtrip')


def native_spec():
    p = c.protocol_spec()['native']
    if (p['seeds'] != list(c.NATIVE_SEEDS) or p['robots'] != 100 or p['duration_seconds'] != 20
            or p['methods'] != list(METHODS) or p['maximum_simulations'] != 34
            or p['handover_seconds'] != [.03, 4.]):
        raise RuntimeError('Native finite protocol differs from its implementation')
    cases = [{'id': f'fresh-{seed}-ho{ho:g}', 'seed': seed, 'handover_seconds': ho,
              'order': list(METHODS[(i+j)%4:]+METHODS[:(i+j)%4])}
             for i, seed in enumerate(c.NATIVE_SEEDS) for j, ho in enumerate((.03, 4.))]
    return dict(p, cases=cases,
                parity_case={'id': 'wiring-parity', 'seed': c.NATIVE_SEEDS[0], 'handover_seconds': .03})


def admit_case(case, method, *, parity=False):
    p = native_spec()
    if type(parity) is not bool or type(case) is not dict or type(method) is not str:
        raise ValueError('Exact registered native case/method required')
    expected = p['parity_case'] if parity else next((x for x in p['cases'] if x['id'] == case.get('id')), None)
    if expected is None or c.encoded(case) != c.encoded(expected) or method not in (PARITY_METHODS if parity else METHODS):
        raise ValueError('Unregistered native case/method')
    return expected


def _checked_reference(item):
    if type(item) is not dict or set(item) != {'path', 'sha256'} or c.reference(c.safe_path(item['path'])) != item:
        raise RuntimeError('Bound native dependency changed')
    return item


def _prerequisites():
    """Require completed field evidence, never create missing prerequisites."""
    c.check_sources()  # publication is mandatory for direct APIs as well as CLI
    selection_path, field_path = c.OUT/'selection.json', c.OUT/'field_report.json'
    if not selection_path.is_file() or not field_path.is_file():
        raise RuntimeError('Immutable VAL selection and completed field report must precede native transfer')
    from . import study
    selected = study.verify_selection()
    field = study.field_report()  # existing immutable report; archived predictions only
    if (field.get('complete') is not True or field.get('primary_layouts') != 24
            or field.get('actual_RT_cases') != 186 or field.get('actual_neural_fits') != 15
            or field.get('source_receipt_sha256') != c.source_receipt_sha256()
            or field.get('selection_sha256') != c.digest(selection_path)
            or field.get('selected_hierarchy') != selected['selected_hierarchy']
            or not {'ap_transfer-sparse', 'ap_transfer-full'} <= set(field.get('secondary_descriptive', {}))):
        raise RuntimeError('Completed fixed TEST/AP-transfer report binding required')
    numerical = study.verify_numerical()
    weights = {'hierarchy': _checked_reference(selected['checkpoints'][selected['selected_hierarchy']+'-seed0']),
               'global_reg': _checked_reference(selected['checkpoints']['global_reg-seed0']),
               'matched_radiomap': _checked_reference(numerical['selected_prior'])}
    return {'source_receipt_sha256': c.source_receipt_sha256(), 'protocol_sha256': c.protocol_sha256(),
            'selection': c.reference(selection_path), 'field_report': c.reference(field_path),
            'selected_hierarchy': selected['selected_hierarchy'], 'weights': weights}


def _world_path(seed): return c.OUT/'native/worlds'/str(seed)
def _case_path(case, method): return c.OUT/'native/cases'/case['id']/method


def _world_receipt(seed, binding):
    if type(seed) is not int or seed not in c.NATIVE_SEEDS:
        raise ValueError('Unregistered native episode seed')
    folder = _world_path(seed); receipt = c.read(folder/'receipt.json')
    if (receipt.get('seed') != seed or receipt.get('binding') != binding
            or receipt.get('robots') != 100 or receipt.get('duration_seconds') != 20):
        raise RuntimeError('Native world binding changed')
    for key, filename in (('world', 'world.npz'), ('warm', 'warm.npz'), ('attempt', 'attempt.json')):
        if receipt[key]['path'] != str((folder/filename).relative_to(c.ROOT)):
            raise RuntimeError('Noncanonical native world archive path')
        _checked_reference(receipt[key])
    world, warm = (c.arrays(c.safe_path(receipt[key]['path'])) for key in ('world', 'warm'))
    if (set(world) != set(WORLD_FIELDS) or set(warm) != set(FIELDS)
            or {key: c.array_digest(value) for key, value in world.items()} != receipt['world_array_sha256']
            or {key: c.array_digest(value) for key, value in warm.items()} != receipt['warm_array_sha256']):
        raise RuntimeError('Native world/report array identity changed')
    _require(len(receipt['APs']) == 16 and receipt['public_geometry'] == {
        'sites': world['sites'].tolist(), 'tx': world['tx_dl'].tolist(),
        'noise': world['noise_db'].tolist(), 'yard': False}, 'native public geometry/source axes')
    attempt = c.read(c.safe_path(receipt['attempt']['path'])); created = attempt.pop('created_unix_ns', None)
    if type(created) is not int or created <= 0 or attempt != {'seed': seed, 'binding': binding}:
        raise RuntimeError('World attempt binding changed')
    return receipt, world, warm


def prepare_world(seed):
    binding = _prerequisites()
    if type(seed) is not int or seed not in c.NATIVE_SEEDS:
        raise ValueError('Unregistered native episode seed')
    from ..fleetwm.data import episode, Observations, Geometry
    from ..contracted.data import runtime_world
    from ..native.world import World, HORIZONS
    from ..twin.radio import AP
    folder = _world_path(seed)
    if not (folder/'receipt.json').exists():
        if folder.exists(): raise RuntimeError('Incomplete native world; regeneration forbidden')
        c.attempt(folder/'attempt.json', {'seed': seed, 'binding': binding})
        full, _, geometry, observed = episode(seed, 100, 20, 'main')
        actual = observed.before(60); allowed = (np.arange(60)%5 == 0)[:, None]
        heard = actual.heard & allowed[..., None]
        warm = Observations(actual.positions, np.where(heard, actual.value, 0).astype('f4'),
                            heard, actual.bound, actual.scanned & allowed)
        empty = np.zeros((20, 100, len(HORIZONS), len(geometry.sites), 3), dtype='f4')
        world = runtime_world(full, 20, empty, empty[:, :, 0])
        receipt = {'seed': seed, 'robots': 100, 'duration_seconds': 20, 'binding': binding,
            'world': c.npz(folder/'world.npz', {name: np.asarray(getattr(world, name)) for name in WORLD_FIELDS}),
            'warm': c.npz(folder/'warm.npz', {name: np.asarray(getattr(warm, name)) for name in FIELDS}),
            'attempt': c.reference(folder/'attempt.json'), 'APs': [asdict(ap) for ap in geometry.aps],
            'public_geometry': {'sites': geometry.sites.tolist(), 'tx': geometry.tx.tolist(),
                                'noise': geometry.noise.tolist(), 'yard': False},
            'world_array_sha256': {name: c.array_digest(getattr(world, name)) for name in WORLD_FIELDS},
            'warm_array_sha256': {name: c.array_digest(getattr(warm, name)) for name in FIELDS},
            'warm_scan_stride_seconds': 5, 'warm_clock_scope': 'one-second measurement intervals; available by zero',
            'physical_truth_not_model_input': True}
        c.write(folder/'receipt.json', receipt)
    receipt, a, o = _world_receipt(seed, binding)
    empty = np.zeros((20, 100, len(HORIZONS), len(a['sites']), 3), dtype='f4')
    world = World(seed, 100, a['sites'], a['snr'], a['rssi_dl'], empty, a['domain'], a['avail'],
        a['video_mbps'], a['bulk_mbps'], 20., 0., False, empty[:, :, 0], a['noise_db'], None,
        a['channel'], None, a['plan'], a['pos'], a['tx_dl'])
    aps = tuple(AP(**(item | {'allowed_channels': tuple(item['allowed_channels'])})) for item in receipt['APs'])
    return world, Observations(*(o[name] for name in FIELDS)), Geometry(a['sites'], a['tx_dl'], a['noise_db'], aps, False)


def _policy(ho):
    from ..spatial_control.kernel import Policy
    return Policy('hierarchical_radio_transfer', 'native', roam_cost_s=ho, scan_period_s=5.,
                  gate=True, cadence_scan=True, cadence_roam=True, online_map=False, dual_radio=False)


def _simulation_config():
    return {'seed': 0, 'duration_seconds': 20, 'collision_lambda': 0., 'interference': False,
            'scan_duration_seconds': .09, 'observation_period_slots': 10,
            'native_proposal_ttl_slots': 100, 'planner': 'native', 'causal_central': True,
            'local_recovery': False}


def _run_arm(case, method, world, warm, geometry, binding, *, parity=False):
    # The durable exclusive start also prevents calling this worker twice.
    if binding != _prerequisites(): raise RuntimeError('Native execution binding changed')
    admit_case(case, method, parity=parity)
    folder = _case_path(case, method)
    attempt = c.read(folder/'attempt.json')
    if attempt.get('identity') != _identity(case, method, parity, binding):
        raise RuntimeError('A matching durable simulation attempt is required')
    c.write(folder/'started.json', {'attempt': c.reference(folder/'attempt.json'), 'created_unix_ns': time.time_ns()})
    from ..adaptive_radio.service import TypedRoamUtility
    from ..jepa_repair.runtime import CompletionSafeForecasts
    from ..worldmodel.radiomap_adapter import RadioMapAdapter
    from ..gate_repair.gate import reactive_rescue
    from ..spatial_control.runtime import ClockedScans
    from ..spatial_control import kernel
    from .native import build_predictor, MatchedSupportAdmission
    sensor = ClockedScans(warm, 20, world.seed); raw = None; support = False
    if parity and method == 'radiomap_roundtrip':
        from ..native_bridge.predictor import MapRoundTripPredictor, bridged
        raw = MapRoundTripPredictor(geometry, binding['source_receipt_sha256']); predictor = bridged(raw)
    else:
        kwargs = {}
        if method in ('hierarchy', 'global_reg'):
            from .training import load
            cell = binding['selected_hierarchy'] if method == 'hierarchy' else 'global_reg'
            kwargs = {'model': load(cell, 0), 'model_sha256': binding['weights'][method]['sha256']}
        elif method == 'matched_radiomap':
            from .baselines import load_prior
            item = binding['weights'][method]
            kwargs = {'prior': load_prior(c.arrays(c.safe_path(item['path']))), 'prior_sha256': item['sha256']}
        bundle = build_predictor(method, geometry, sensor, **kwargs)
        raw, predictor, support = bundle.raw, bundle.predictor, bundle.support_enabled
    profile = ({kind: 4. for kind in ('cold', 'warm', 'refit')} if parity
               else {'cold': 1., 'warm': .25, 'refit': 1.})
    live = CompletionSafeForecasts(predictor, sensor, 20, profile, .5)
    physical = copy(world); physical.forecast = live.mean; physical.sigma_belief = live.std[:, :, 0]; physical.rmap = None
    utility = TypedRoamUtility(RadioMapAdapter(), method, live, handover_seconds=case['handover_seconds'],
                              horizon=10., max_candidates=64, deadline_seconds=.1)
    guard = MatchedSupportAdmission(utility, raw, live, sensor) if support else utility
    decisions = []; actions = []
    def capture(decision, confident, site):
        decisions.append({'forecast': int(decision.decision), 'current': decision.current.tolist(),
            'target': decision.candidate.tolist(), 'eligible': decision.eligible.tolist(),
            'gate': confident.tolist(), 'gain': decision.gain.tolist()})
    def executed(slot, robots, targets):
        actions.append({'slot': int(slot), 'robots': np.asarray(robots).tolist(), 'targets': np.asarray(targets).tolist()})
    policy = _policy(case['handover_seconds'])
    native = kernel.simulate(physical, policy, seed=0, duration_s=20, collision_lambda=0., interference=False,
        roam_gate=reactive_rescue, handover_admission=guard, roam_scorer=guard.scores,
        forecast_queue=live, forecast_source=live, scan_observer=sensor, scan_duration_s=.09,
        decision_trace=capture, actual_action_hook=executed, detailed=True,
        observation_period_slots=10, native_proposal_ttl_slots=100, causal_central=True, spatial_runtime=None)
    saved = {'forecast_mean_snr': live.mean, 'forecast_std_db': live.std}
    saved.update({'history_'+name: np.asarray(getattr(sensor.history, name)) for name in FIELDS})
    model_metadata = []
    if raw is not None:
        for j, record in enumerate(raw.records):
            for key, value in record.items():
                if isinstance(value, np.ndarray): saved[f'job{j}_{key}'] = value
            model_metadata.append({key: value for key, value in record.items() if not isinstance(value, np.ndarray)})
    result = {'method': method, 'seed': world.seed, 'handover_seconds': case['handover_seconds'],
        'parity': parity, 'native': native, 'forecast_jobs': live.records,
        'forecast_consumption': live.consumption_records, 'telemetry': sensor.summary(),
        'scan_started_events': sensor.started_events, 'local_completed_payloads': sensor.local_completed_payloads,
        'planner_and_admission': utility.records, 'decisions': decisions, 'actual_actions': actions,
        'policy_config': asdict(policy), 'simulation_config': _simulation_config(),
        'support_enabled': support, 'support': guard.records if support else [],
        'support_refresh': guard.refresh_records if support else [],
        'support_preparation': getattr(raw, 'support_preparation', []), 'model_records': model_metadata,
        'causal_central': True, 'local_runtime': None, 'production_qualified': False}
    return result, saved


def _identity(case, method, parity, binding):
    return {'case': case, 'method': method, 'parity': parity, 'binding': binding,
            'world_receipt': c.reference(_world_path(case['seed'])/'receipt.json')}


def _require(condition, reason):
    if not condition: raise ValueError('Canonical hierarchical native result: '+reason)


def _num(value):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'finite nonnegative cost required')
    return float(value)


def _eq(a, b): return math.isclose(_num(a), float(b), abs_tol=1e-10, rel_tol=0)
def _sha(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def metrics(result):
    native = result['native']; detail = native['packet_detail']; jobs = result['forecast_jobs']
    misses = np.asarray(detail['misses']); sent = np.asarray(detail['sent']); gap = np.asarray(detail['max_gap'])
    _require(misses.shape == sent.shape == gap.shape == (100,) and sent.sum() > 0, 'packet counter axes')
    derived = {'control_miss': float(misses.sum()/sent.sum()),
        'control_miss_worst_decile': float(np.sort(misses/np.maximum(sent, 1))[-10:].mean()),
        'control_gap_p99_ms': float(np.percentile(gap, 99)*10), 'maximum_silence_ms': float(gap.max()*10)}
    for key in ('control_miss', 'control_miss_worst_decile', 'control_gap_p99_ms'):
        _require(_eq(native[key], derived[key]), 'headline differs from raw packet counters')
    spatial = native['spatial_execution']; pending = spatial['pending_control_packets']
    _require(type(pending) is int and 0 <= pending <= 100, 'pending packet bound')
    return derived | {key: native[key] for key in ('video_on_time', 'bulk_mbps', 'roams', 'safety_stop_fraction')} | {
        'control_misses': int(misses.sum()), 'control_sent': int(sent.sum()), 'end_pending_commands': pending,
        'observed_plus_pending_miss_fraction': float((misses.sum()+pending)/sent.sum()),
        'paid_forecast_compute_seconds': sum(j['charged_compute_seconds'] for j in jobs),
        'actual_forecast_compute_seconds': sum(j['actual_compute_seconds'] for j in jobs),
        'forecast_jobs_consumed': len(result['forecast_consumption']),
        'scans_started': result['telemetry']['scans_started'],
        'scan_interruption_seconds': sum(e['interruption_slots'] for e in spatial['scan_events'])*.01,
        'central_planning_cpu_seconds': sum(e['elapsed_seconds'] for e in spatial['central_decisions']),
        'central_planning_overruns': sum(e['overrun'] for e in spatial['central_decisions']),
        'central_busy_forecast_skips': len(spatial['central_compute_events']),
        'support_preparation_cpu_seconds': sum(e['elapsed_seconds'] for e in result['support_preparation']),
        'support_vetoes': sum(not x['permitted'] for x in result['support']),
        'central_handovers': sum(e['event'] == 'executed' for e in spatial['central_events'])}


def _validate_clocks(result, saved, *, parity, sites):
    shape = (20, 100, 9, sites, 3)
    for name in ('forecast_mean_snr', 'forecast_std_db'):
        a = saved[name]
        _require(a.shape == shape and a.dtype == np.float32 and np.isfinite(a).all(), 'forecast shape/type/finiteness')
    _require((saved['forecast_std_db'] >= 0).all(), 'negative forecast deviation')
    jobs = result['forecast_jobs']; _require(type(jobs) is list and len(jobs) == 20, 'exact twenty forecast jobs')
    previous = 0.; ready = []
    profile = {k: 4. for k in ('cold', 'warm', 'refit')} if parity else {'cold': 1., 'warm': .25, 'refit': 1.}
    for j, job in enumerate(jobs):
        _require(type(job['decision']) is int and job['decision'] == j and job['history_cutoff_exclusive'] == 60+j, 'job/history cutoff')
        kind = job['kind']; _require(kind == ('cold' if j == 0 else 'warm'), 'timing kind')
        actual = max(_num(job['actual_prediction_seconds']), _num(job['actual_issue_seconds']))
        charged = max(profile[kind], actual); service = charged+.5
        for key, value in (('actual_compute_seconds', actual), ('charged_compute_seconds', charged),
                ('profile_seconds', profile[kind]), ('shared_extra_seconds', .5), ('charged_seconds', service)):
            _require(_eq(job[key], value), 'paid completion cost accounting')
        previous = max(float(j), previous)+service; slot = math.ceil(previous*100-1e-10); ready.append(slot)
        _require(type(job['ready_slot']) is int and job['ready_slot'] == slot, 'FIFO forecast completion')
        _require(job['completion_uses_actual_elapsed'] is True and job['offline_recorded_schedule'] is False
            and job['accounting_mode'] == 'actual_cost' and job['runtime_overrun'] is (actual > profile[kind])
            and job['predictor_runtime_overrun'] is (job['actual_prediction_seconds'] > profile[kind]), 'completion semantics')
        _require(job['forecast_shape'] == list(shape[1:]) and job['forecast_dtype'] == 'float32'
            and job['forecast_sha256'] == _sha(saved['forecast_mean_snr'][j])
            and job['forecast_std_sha256'] == _sha(saved['forecast_std_db'][j]), 'forecast array hash')
        _require(job['heard_entries'] == int(saved['history_heard'][59+j].sum())
            and job['scans_last_bin'] == int(saved['history_scanned'][59+j].sum()), 'history mask identity')
    last = -1; last_slot = -1
    for row in result['forecast_consumption']:
        j, slot = row['forecast_decision'], row['slot']
        _require(type(j) is int and type(slot) is int and last < j < 20 and last_slot <= slot < 2000
            and row['ready_slot'] == ready[j] <= slot and j == np.searchsorted(ready, slot, side='right')-1
            and _eq(row['age_seconds'], slot/100-j), 'forecast used before newest FIFO completion')
        last, last_slot = j, slot
    return ready


def _validate_actions(result, sites):
    spatial = result['native']['spatial_execution']
    _require(spatial['causal_central'] is True and spatial['local_recovery_enabled'] is False
        and result['local_runtime'] is None and not spatial['local_events'] and not spatial['local_decisions']
        and not spatial['local_validation_batches'], 'unregistered local authority')
    for key, value in (('central_creation_delay_slots', 10), ('central_ttl_slots', 100),
                       ('scan_minimum_spacing_slots', 500), ('local_decision_floor_slots', 1)):
        _require(spatial[key] == value, 'action budget')
    _require(_eq(spatial['central_decision_budget_seconds'], .1), 'central compute budget')
    computations = {}; busy = 0
    for row in spatial['central_decisions']:
        slot = row['slot']; elapsed = _num(row['elapsed_seconds'])
        _require(type(slot) is int and busy <= slot < 2000 and slot not in computations
            and _eq(row['budget_seconds'], .1) and row['overrun'] is (elapsed > .1)
            and row['created_slot'] == slot+10
            and row['compute_ready_slot'] == slot+max(10, math.ceil(elapsed/.01-1e-12))
            and any(e['slot'] == slot and e['forecast_decision'] == row['forecast_decision']
                    for e in result['forecast_consumption']), 'central CPU occupancy')
        computations[slot] = row; busy = row['compute_ready_slot']
    commands = {}; terminal = set(); interruption = math.ceil(result['handover_seconds']/.01-1e-12)
    for event in spatial['central_events']:
        key, kind, slot = event['command_id'], event['event'], event['slot']
        _require(type(slot) is int and 0 <= slot <= 2000 and type(event['robot']) is int and 0 <= event['robot'] < 100
            and 0 <= event['current'] < sites and 0 <= event['target'] < sites and event['current'] != event['target'], 'central axes/no-op')
        base = {k: event[k] for k in ('command_id', 'robot', 'current', 'target', 'proposed_slot', 'created_slot', 'expires_slot', 'interruption_slots')}
        _require(event['created_slot'] == event['proposed_slot']+10 and event['expires_slot'] == event['created_slot']+100
            and event['interruption_slots'] == interruption, 'central creation/expiry/cost')
        if kind == 'queued':
            _require(key not in commands and slot == event['proposed_slot'] and slot in computations
                and computations[slot]['overrun'] is False, 'unpaid/duplicate central proposal')
            commands[key] = base
        else:
            _require(key in commands and base == commands[key] and key not in terminal
                and kind in ('cancelled', 'executed', 'right_censored'), 'central lifecycle identity')
            terminal.add(key)
            if kind == 'executed':
                _require(event['created_slot'] < event['delivery_slot'] == event['execution_slot'] == slot < event['expires_slot']
                    and event['created_slot'] <= event['packet_created_slot'] <= slot and slot-event['packet_created_slot'] < 5
                    and event['outage_until_slot'] == slot+1+interruption, 'central execution lacks causal paid packet')
            elif kind == 'right_censored': _require(slot == 2000, 'early command censor')
    _require(set(commands) == terminal, 'unfinished central command')
    initial = spatial['initial_scan_start_upper_slot']; _require(len(initial) == 100, 'initial scan clocks')
    previous = {robot: slot for robot, slot in enumerate(initial)}
    for event in spatial['scan_events']:
        robot, slot = event['robot'], event['slot']
        _require(type(robot) is int and 0 <= robot < 100 and type(slot) is int and 0 <= slot < 2000
            and slot-previous[robot] >= 500 and event['measurement_slot'] == slot
            and event['interruption_start_slot'] == slot+1 and event['complete_slot'] == slot+10
            and event['interruption_slots'] == 9 and event['reason'] == 'background_after_delivery'
            and event['action_id'] is None, 'unpaid/unauthorized scan')
        previous[robot] = slot
    ages = np.asarray(spatial['pending_packet_age_end'])
    _require(ages.shape == (100,) and ages.dtype.kind in 'iu'
        and spatial['pending_control_packets'] == int((ages >= 0).sum()), 'pending packet upper bound')
    return len(commands)


def _validate_sensor(result, saved, sites):
    from ..fleetwm.data import Observations
    from ..spatial_control.memory import _native_events
    from .native import _payload_index, _check_delivered_row
    events = result['telemetry']['events']; completed = result['local_completed_payloads']
    spatial = result['native']['spatial_execution']
    _require(events == result['native']['packet_detail']['delivered_report_events']
        and completed == spatial['local_completed_payloads'], 'raw sensor ledgers disagree')
    _require(result['telemetry']['reports_delivered'] == len(events)
        and result['telemetry']['scans_started'] == len(result['scan_started_events']), 'sensor summary counts')
    past = Observations(*(saved['history_'+name] for name in FIELDS))
    rows = _native_events(events, issue=20., warm_bins=60, bin_seconds=1., slots_per_second=100,
                          n_robots=100, history_length=80)
    payloads = _payload_index(completed)
    _require(all(0 <= event['processed_slot'] < 2000 for event in completed), 'completion outside observation window')
    for t, robot in zip(*np.nonzero(past.scanned[60:])):
        t = int(t)+60; robot = int(robot); event = rows.get((t, robot))
        _require(event is not None, 'post-warm scan lacks delivered clock')
        payload = payloads.get((robot, event['measurement_slot'], event['complete_slot'], event['payload_sha256']))
        _check_delivered_row(event, payload, past.value[t, robot], past.heard[t, robot], past.bound[t, robot], past.positions[t, robot])
    _require(all(past.scanned[t, robot] for t, robot in rows), 'delivered event absent from server history')
    paid_scans = spatial['scan_events']; starts = result['scan_started_events']
    _require(len(paid_scans) == len(starts), 'unpaid or omitted scan attempt')
    for start, paid in zip(starts, paid_scans):
        _require(start['robot'] == paid['robot'] and start['slot'] == paid['measurement_slot']
            and start['complete_slot'] == paid['complete_slot'], 'sensor capture differs from paid scan')
    return past


def _validate_contexts(result, saved, world_receipt, past):
    from ..fleetwm.data import Observations
    from ..spatial_control.memory import native_history_memory
    from ..native_bridge.predictor import radio_assumptions
    from .native import select_context, make_query
    raw_geometry = world_receipt['public_geometry']
    geometry = SimpleNamespace(**{name: np.asarray(raw_geometry[name]) for name in ('sites', 'tx', 'noise')}, yard=False)
    sites = len(geometry.sites); all_events = result['telemetry']['events']; all_payloads = result['local_completed_payloads']
    neural = result['method'] in ('hierarchy', 'global_reg')
    expected_array_keys = set()
    _require(len(result['model_records']) == 20 and len(result['support_preparation']) == 20, 'exact model/preparation jobs')
    for j in range(20):
        prefix = f'job{j}_'; record = result['model_records'][j]
        snapshot = Observations(*(np.asarray(getattr(past, name))[:60+j] for name in FIELDS))
        events = [e for e in all_events if e['delivery_slot'] <= j*100]
        payloads = [e for e in all_payloads if e['processed_slot'] <= j*100]
        plan = saved[prefix+'plan']; _require(plan.shape == (100, 9, 2) and np.isfinite(plan).all(), 'public plan shape')
        selected = select_context(snapshot, geometry, plan, issue_time_s=float(j),
            events=events, local_completed_payloads=payloads)
        view, metadata = selected['view'], selected['metadata']; query, shape = make_query(geometry, plan)
        expected = {'context': view['context'], 'known': view['known'], 'scheduled': view['scheduled'],
            'selected_rows': metadata['selected_rows'], 'selected_scan_rows': metadata['selected_scan_rows'],
            'measurement_intervals_s': metadata['measurement_intervals_s'], 'arrival_intervals_s': metadata['arrival_intervals_s'],
            'clock_provenance': metadata['clock_provenance'], 'context_age_s': metadata['conservative_age_s'],
            'context_ap': view['ap'], 'context_rx_xyz': view['rx_xyz'], 'context_lower': view['lower'],
            'context_upper': view['upper'], 'context_floor': view['floor'], 'context_fspl': view['fspl'],
            'context_offset': view['offset'], 'ap_xyz': selected['ap_xyz'],
            'physical_context_rx_xyz_m': metadata['physical_context_rx_xyz_m'],
            'physical_ap_xyz_m': metadata['physical_ap_xyz_m']}
        physical_rx = np.broadcast_to(np.concatenate((plan, np.ones((*plan.shape[:2], 1))), axis=-1)[:, :, None, None],
                                      shape+(3,)).reshape(-1, 3).astype('f4')
        expected['physical_query_rx_xyz_m'] = physical_rx
        expected.update({'query_'+key: value for key, value in query.items()})
        for name, value in expected.items():
            observed = saved[prefix+name]
            _require(observed.dtype == value.dtype and np.array_equal(observed, value, equal_nan=True), 'changed causal context/query '+name)
        expected_array_keys.update(prefix+name for name in (*expected, 'plan', 'mean_dbm', 'std_db'))
        _require(record['decision'] == j and record['history_length'] == 60+j
            and c.encoded(record['context_metadata']) == c.encoded({k: v for k, v in metadata.items() if not isinstance(v, np.ndarray)}),
            'model context metadata binding')
        mean, std = saved[prefix+'mean_dbm'], saved[prefix+'std_db']
        _require(mean.shape == std.shape == shape and np.isfinite(mean).all() and np.isfinite(std).all() and (std >= 0).all(), 'raw marginal RSSI forecast')
        # Exactly the same declared device0/UL transfer used by the actual bridge.
        converted = radio_assumptions(geometry).convert(mean).astype('f4')
        _require(np.array_equal(converted, saved['forecast_mean_snr'][j])
            and np.array_equal(std.astype('f4'), saved['forecast_std_db'][j]), 'RSSI-to-SNR forecast wiring')
        preparation = result['support_preparation'][j]
        memory = native_history_memory(snapshot, issue_time_s=float(j), events=events, retention_age_s=6.)
        _require(preparation['decision'] == j and preparation['reports'] == len(memory)
            and c.encoded(preparation['snapshot_metadata']) == c.encoded(memory.snapshot_metadata)
            and preparation['matched_forecast_context_budget'] == 192
            and preparation['guard_modifies_issued_forecast'] is False and preparation['packet_risk_qualified'] is False
            and _eq(preparation['retention_age_seconds'], 6.)
            and _num(preparation['elapsed_seconds']) <= result['forecast_jobs'][j]['actual_prediction_seconds'], 'unpaid/wrong support preparation')
        if neural:
            global_dim = 24 if result['method'] == 'global_reg' else 8
            state_shapes = {'belief_global_mean': (global_dim,), 'belief_global_covariance': (global_dim, global_dim)}
            if result['method'] == 'hierarchy':
                state_shapes.update(belief_local_mean=(sites, 16), belief_local_cholesky=(sites, 16, 16),
                                    belief_local_coupling=(sites, 16, 8))
            for key, axes in state_shapes.items():
                _require(saved[prefix+key].shape == axes and np.isfinite(saved[prefix+key]).all(), 'posterior state block axes')
                expected_array_keys.add(prefix+key)
            for key in ('objective_before', 'objective_after', 'step_size', 'null_step'):
                name = prefix+'inference_'+key; a = saved[name]
                _require(a.shape == (3,) and np.isfinite(a).all(), 'finite three-step context inference ledger')
                expected_array_keys.add(name)
            before = saved[prefix+'inference_objective_before']; after = saved[prefix+'inference_objective_after']
            _require((after <= before+1e-10*np.maximum(1., np.abs(before))).all(), 'accepted Newton step increased context objective')
    return expected_array_keys


def validate_result(receipt, result, saved, world_receipt, *, warm_arrays=None):
    """Passive identity, raw-counter, cost, FIFO and causal-action verifier."""
    case, method, parity = receipt['case'], receipt['method'], receipt['parity']
    admit_case(case, method, parity=parity); sites = len(world_receipt['APs'])
    _require(result['method'] == method and type(result['seed']) is int and result['seed'] == case['seed']
        and _eq(result['handover_seconds'], case['handover_seconds']) and result['parity'] is parity
        and result['causal_central'] is True and result['production_qualified'] is False, 'raw case/authority binding')
    _require(c.encoded(result['policy_config']) == c.encoded(asdict(_policy(case['handover_seconds'])))
        and result['simulation_config'] == _simulation_config(), 'actual policy/simulator configuration')
    native = result['native']; detail = native['packet_detail']
    _require(native['policy'] == 'hierarchical_radio_transfer' and native['scene'] == case['seed']
        and _eq(native['duration_s'], 20.) and _eq(native['collision_lambda'], 0.) and native['interference'] is False
        and detail['stop_slot'] == 2000 and detail['resume_slot'] == 0 and detail['observation_period_slots'] == 10
        and detail['native_proposal_ttl_slots'] == 100 and detail['focal_hold_slots'] is None, 'native episode/configuration')
    for name, shape, kind in (('positions', (80, 100, 2), 'f'), ('value', (80, 100, sites*3), 'f'),
            ('heard', (80, 100, sites*3), 'b'), ('bound', (80, 100, sites*3), 'f'), ('scanned', (80, 100), 'b')):
        a = saved['history_'+name]
        _require(a.shape == shape and a.dtype.kind in kind and np.isfinite(a).all(), 'final sensor history axes/type')
    if warm_arrays is None:
        _checked_reference(world_receipt['warm'])
        warm_arrays = c.arrays(c.safe_path(world_receipt['warm']['path']))
    _require(set(warm_arrays) == set(FIELDS), 'bound warm observation membership')
    for name in FIELDS:
        prefix, expected = saved['history_'+name][:60], warm_arrays[name]
        _require(prefix.dtype == expected.dtype and np.array_equal(prefix, expected), 'history warm prefix differs from bound world')
    _require(not (saved['history_heard'] & ~saved['history_scanned'][..., None]).any(), 'heard without actual scan')
    for key in ('misses', 'sent', 'gap', 'max_gap', 'window_max_gap'):
        a = np.asarray(detail[key]); _require(a.shape == (100,) and a.dtype.kind in 'iu' and (a >= 0).all(), 'integer packet counters')
    misses, sent = np.asarray(detail['misses']), np.asarray(detail['sent'])
    _require((misses <= sent).all() and native['control_misses'] == int(misses.sum())
        and native['control_sent'] == int(sent.sum()), 'raw packet totals')
    ready = _validate_clocks(result, saved, parity=parity, sites=sites)
    commands = _validate_actions(result, sites)
    past = _validate_sensor(result, saved, sites)
    matched = not parity and method != 'full_radiomap'
    _require(result['support_enabled'] is matched and (matched or not result['support']), 'support method scope')
    expected_keys = {'forecast_mean_snr', 'forecast_std_db'} | {'history_'+name for name in FIELDS}
    if matched: expected_keys |= _validate_contexts(result, saved, world_receipt, past)
    elif method == 'radiomap_roundtrip':
        _require(len(result['model_records']) == 20, 'roundtrip model record count')
        for j in range(20):
            for name in ('source_snr', 'source_std', 'plan', 'mean_dbm'):
                key = f'job{j}_{name}'; _require(np.isfinite(saved[key]).all(), 'roundtrip archive'); expected_keys.add(key)
            _require(np.array_equal(saved[f'job{j}_source_snr'], saved['forecast_mean_snr'][j])
                and np.array_equal(saved[f'job{j}_source_std'], saved['forecast_std_db'][j]), 'roundtrip internal source forecast')
    else: _require(not result['model_records'] and not result['support_preparation'], 'full-history reference has learned metadata')
    _require(set(saved) == expected_keys, 'extra/missing inert prediction/history/context arrays')
    recomputed = metrics(result)
    if 'metrics' in receipt: _require(receipt['metrics'] == recomputed, 'receipt metrics differ from counters')
    return {'passes': True, 'jobs': len(ready), 'causal_central': True, 'central_commands': commands,
            'scope': 'passive identity, counter, cost and clock arithmetic; no service qualification'}


def execute(case, method, world, warm, geometry, *, parity=False):
    binding = _prerequisites(); admit_case(case, method, parity=parity)
    receipt_world, a, o = _world_receipt(case['seed'], binding)
    if (world.seed != case['seed'] or world.n != 100 or world.duration_s != 20
            or {k: c.array_digest(getattr(world, k)) for k in WORLD_FIELDS} != receipt_world['world_array_sha256']
            or {k: c.array_digest(getattr(warm, k)) for k in FIELDS} != receipt_world['warm_array_sha256']
            or not np.array_equal(geometry.sites, a['sites']) or not np.array_equal(geometry.tx, a['tx_dl'])
            or not np.array_equal(geometry.noise, a['noise_db']) or geometry.yard is not False
            or c.encoded([asdict(ap) for ap in geometry.aps]) != c.encoded(receipt_world['APs'])):
        raise RuntimeError('Exact registered native world/history/public geometry required')
    identity = _identity(case, method, parity, binding); folder = _case_path(case, method)
    if (folder/'receipt.json').exists():
        receipt = c.read(folder/'receipt.json')
        if receipt.get('identity') != identity: raise RuntimeError('Completed native attempt identity changed')
        load_result(receipt, binding=binding); return receipt
    if folder.exists(): raise RuntimeError('Partial native simulation attempt; replacement forbidden')
    c.attempt(folder/'attempt.json', {'identity': identity})
    result, saved = _run_arm(case, method, world, warm, geometry, binding, parity=parity)
    receipt = dict(identity, identity=identity, simulations=1, simulation_only=True, default_changed=False,
        original_SERVICE_opened=False, production_qualified=False,
        arrays=c.npz(folder/'predictions_history.npz', saved),
        attempt=c.reference(folder/'attempt.json'), started=c.reference(folder/'started.json'))
    raw_path = folder/'raw.json.gz'; c.write_bytes(raw_path, gzip.compress(c.encoded(result), mtime=0))
    receipt.update(raw=c.reference(raw_path), metrics=metrics(result))
    receipt['canonical_validation'] = validate_result(receipt, result, saved, receipt_world, warm_arrays=o)
    c.write(folder/'receipt.json', receipt); return receipt


def load_result(receipt, *, binding=None):
    if binding is None: binding = _prerequisites()
    case, method, parity = receipt['case'], receipt['method'], receipt['parity']
    admit_case(case, method, parity=parity)
    expected = _identity(case, method, parity, binding)
    _require(receipt['identity'] == expected and all(receipt[key] == value for key, value in expected.items()), 'receipt identity')
    _require(receipt['simulations'] == 1 and receipt['simulation_only'] is True and receipt['default_changed'] is False
        and receipt['original_SERVICE_opened'] is False and receipt['production_qualified'] is False, 'receipt authority')
    folder = _case_path(case, method)
    for key, filename in (('raw', 'raw.json.gz'), ('arrays', 'predictions_history.npz'),
                          ('attempt', 'attempt.json'), ('started', 'started.json')):
        _require(receipt[key]['path'] == str((folder/filename).relative_to(c.ROOT)), 'canonical evidence path')
        _checked_reference(receipt[key])
    attempt = c.read(c.safe_path(receipt['attempt']['path'])); created = attempt.pop('created_unix_ns', None)
    _require(type(created) is int and created > 0 and attempt == {'identity': expected}, 'durable attempt identity')
    started = c.read(c.safe_path(receipt['started']['path']))
    _require(set(started) == {'attempt', 'created_unix_ns'} and started['attempt'] == receipt['attempt']
        and type(started['created_unix_ns']) is int and started['created_unix_ns'] >= created, 'exclusive simulation start binding')
    world_receipt, _, warm = _world_receipt(case['seed'], binding)
    result = json.loads(gzip.decompress(c.safe_path(receipt['raw']['path']).read_bytes()))
    proof = validate_result(receipt, result, c.arrays(c.safe_path(receipt['arrays']['path'])), world_receipt, warm_arrays=warm)
    _require(proof == receipt['canonical_validation'], 'canonical passive validation changed')
    return result


def _parity(binding):
    case = native_spec()['parity_case']; values = []
    for method in PARITY_METHODS:
        receipt = c.read(_case_path(case, method)/'receipt.json')
        values.append((receipt, load_result(receipt, binding=binding), c.arrays(c.safe_path(receipt['arrays']['path']))))
    a, b = values
    checks = {'forecast_snr_exact': np.array_equal(a[2]['forecast_mean_snr'], b[2]['forecast_mean_snr']),
        'forecast_std_exact': np.array_equal(a[2]['forecast_std_db'], b[2]['forecast_std_db']),
        'ready_slots_exact': [j['ready_slot'] for j in a[1]['forecast_jobs']] == [j['ready_slot'] for j in b[1]['forecast_jobs']],
        'packet_counters_exact': c.encoded(a[1]['native']['packet_detail']) == c.encoded(b[1]['native']['packet_detail']),
        'native_outcomes_exact': c.encoded({k: v for k, v in a[1]['native'].items() if k != 'spatial_execution'})
            == c.encoded({k: v for k, v in b[1]['native'].items() if k != 'spatial_execution'})}
    return {'passes': all(checks.values()), 'checks': checks,
        'receipts': [c.reference(_case_path(case, method)/'receipt.json') for method in PARITY_METHODS],
        'both_causal_central': True, 'profile_seconds': 4., 'scientific_gain_claimed': False}


def _summary(binding):
    p = native_spec(); parity = _parity(binding)
    _require(parity['passes'], 'wiring parity failed')
    rows = []
    for case in p['cases']:
        for method in METHODS:
            receipt = c.read(_case_path(case, method)/'receipt.json'); load_result(receipt, binding=binding)
            rows.append({'case': case['id'], 'seed': case['seed'], 'handover_seconds': case['handover_seconds'],
                         'method': method, 'metrics': receipt['metrics'],
                         'receipt': c.reference(_case_path(case, method)/'receipt.json')})
    means = []; contrasts = []
    for ho in (.03, 4.):
        for method in METHODS:
            items = [r['metrics'] for r in rows if r['handover_seconds'] == ho and r['method'] == method]
            means.append({'handover_seconds': ho, 'method': method, 'episodes': 4,
                          'mean': {key: float(np.mean([x[key] for x in items])) for key in items[0]}})
        for candidate, reference in (('hierarchy', 'matched_radiomap'), ('hierarchy', 'global_reg'),
                                     ('hierarchy', 'full_radiomap'), ('matched_radiomap', 'full_radiomap')):
            pairs = []
            for seed in c.NATIVE_SEEDS:
                a = next(r['metrics'] for r in rows if r['seed'] == seed and r['handover_seconds'] == ho and r['method'] == candidate)
                b = next(r['metrics'] for r in rows if r['seed'] == seed and r['handover_seconds'] == ho and r['method'] == reference)
                pairs.append({'seed': seed, 'delta': {key: a[key]-b[key] for key in a}})
            contrasts.append({'handover_seconds': ho, 'candidate': candidate, 'reference': reference,
                'paired_episodes': pairs, 'mean_delta': {key: float(np.mean([x['delta'][key] for x in pairs])) for key in pairs[0]['delta']},
                'episodes_with_more_control_misses': sum(x['delta']['control_misses'] > 0 for x in pairs),
                'episodes_with_larger_maximum_silence': sum(x['delta']['maximum_silence_ms'] > 0 for x in pairs)})
    return {'schema': 1, 'complete': True, 'binding': binding, 'fresh_simulations': 32, 'parity_simulations': 2,
        'maximum_simulations': 34, 'rows': rows, 'method_means': means, 'contrasts': contrasts, 'parity': parity,
        'simulation_only': True, 'descriptive_four_paired_episodes_per_handover': True,
        'full_radiomap_has_more_information': True, 'unqualified_6point5GHz_extrapolation': True,
        'real_hardware_actions': 0, 'default_changed': False, 'production_qualified': False,
        'original_SERVICE_opened': False}


def run_native():
    binding = _prerequisites(); p = native_spec()
    world, warm, geometry = prepare_world(c.NATIVE_SEEDS[0])
    for method in PARITY_METHODS: execute(p['parity_case'], method, world, warm, geometry, parity=True)
    proof = _parity(binding); path = c.OUT/'native/parity.json'
    if path.exists(): _require(c.read(path) == proof, 'parity receipt changed')
    else: c.write(path, proof)
    _require(proof['passes'], 'wiring parity failed; no fresh comparisons allowed')
    for case in p['cases']:
        world, warm, geometry = prepare_world(case['seed'])
        for method in case['order']:
            receipt = execute(case, method, world, warm, geometry)
            print('native', case['id'], method, receipt['metrics']['control_miss'], flush=True)
    report = _summary(binding); path = c.OUT/'native/report.json'
    if path.exists(): _require(c.read(path) == report, 'native report changed')
    else: c.write(path, report)
    return verify_native()


def verify_native():
    binding = _prerequisites(); report = _summary(binding)
    _require(c.read(c.OUT/'native/report.json') == report, 'native report differs from fixed raw membership')
    _require(c.read(c.OUT/'native/parity.json') == report['parity'], 'parity proof differs from raw evidence')
    expected = {_case_path(case, method)/'receipt.json' for case in native_spec()['cases'] for method in METHODS}
    expected |= {_case_path(native_spec()['parity_case'], method)/'receipt.json' for method in PARITY_METHODS}
    actual = set((c.OUT/'native/cases').rglob('receipt.json'))
    _require(actual == expected and len(actual) == 34, 'extra/missing native simulation membership')
    attempts = set((c.OUT/'native/cases').rglob('attempt.json'))
    _require(attempts == {path.with_name('attempt.json') for path in expected}, 'unregistered native simulation attempts')
    return report
