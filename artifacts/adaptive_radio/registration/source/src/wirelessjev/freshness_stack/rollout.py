"""Registered sparse forecast scheduling and conservative native stack stages."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from ..closed_loop.control import OutageUtility
from ..jepa_repair.telemetry import TimedDeliveredScans
from ..contracted.data import READINGS, runtime_world
from ..fleetwm.data import episode
from ..frontier.study import mask_scans
from ..gate_repair.gate import reactive_rescue
from ..native.intent_policy import intent_policy
from ..native.world import HORIZONS
from ..semantic_jepa import common as frozen_common
from ..semantic_jepa.runtime import Predictor
from . import runtime
from .sim import simulate

FAMILIES = ('radiomap', 'frozen_fleet_jepa')
VARIANTS = ('fifo', 'latest', 'latest_single', 'latest_single_sense')
STAGES = ('development', 'confirmation')
REPLAY_FIELDS = ('result', 'forecast_mean_sha256', 'forecast_std_sha256',
                 'telemetry', 'decision_trace', 'admission_trace',
                 'local_scan_observability', 'action_trace_sha256', 'arrivals', 'queue_mode',
                 'accepted_decisions', 'forecast_metric_decisions')


def _common():
    return importlib.import_module('.common', __package__)


def _value_digest(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def queue_mode(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown stack stage')
    return 'fifo' if variant == 'fifo' else 'latest'


def _parts(key):
    if not isinstance(key, (tuple, list)) or len(key) != 5:
        raise ValueError('Malformed repair case key')
    stage, site, cost, family, variant = key
    if (stage not in STAGES or type(site) is not int or site < 0
            or type(cost) not in (int, float) or not math.isfinite(cost)
            or cost != 4. or family not in FAMILIES or variant not in VARIANTS):
        raise ValueError('Unregistered repair case key')
    return stage, site, float(cost), family, variant


def arms(cost, p=None):
    p = _common().protocol() if p is None else p
    if type(cost) not in (int, float) or cost not in p['handover_seconds']:
        raise ValueError('Unregistered handover cost')
    if cost != 4.:
        raise ValueError('Freshness stack is registered only for four-second handovers')
    variants = VARIANTS
    return [(family, variant) for family in FAMILIES for variant in variants]


def validate_case_key(key, p=None):
    stage, site, cost, family, variant = _parts(key)
    p = _common().protocol() if p is None else p
    if (site not in p[stage + '_sites'] or cost not in p['handover_seconds']
            or (family, variant) not in arms(cost, p)):
        raise ValueError('Unregistered repair case key')


def arm_name(family, variant):
    if family not in FAMILIES or variant not in VARIANTS:
        raise ValueError('Unknown fixed-model/controller identity')
    return f'{family}-{variant}'


def case_name(key):
    stage, site, cost, family, variant = _parts(key)
    return f'{stage}-{site}-h{round(cost * 1000):04d}-{arm_name(family, variant)}'


def case_keys(stage, passed_costs=None, p=None):
    if stage not in STAGES:
        raise ValueError('Unknown repair stage')
    p = _common().protocol() if p is None else p
    costs = p['handover_seconds'] if passed_costs is None else passed_costs
    if not set(costs) <= set(p['handover_seconds']):
        raise ValueError('Unregistered cost subset')
    return [(stage, site, float(cost), family, variant)
            for site in p[stage + '_sites'] for cost in costs
            for family, variant in arms(cost, p)]


def _mode(replay=False, live_replication=False):
    if type(replay) is not bool or type(live_replication) is not bool or (replay and live_replication):
        raise ValueError('Replication flags must be exclusive booleans')
    return 'recorded_schedule_offline' if replay else 'actual_cost'


def _replication_guard(key, p, replay=False, live_replication=False):
    _mode(replay, live_replication)
    if (replay or live_replication) and (key[0] != 'confirmation' or key[1] != p['confirmation_sites'][0]):
        raise ValueError('Replications require the first registered confirmation site')


def _folder(c, replay=False, live_replication=False):
    _mode(replay, live_replication)
    return c.OUT / ('replays' if replay else 'live_replications' if live_replication else 'cases')


def _identities():
    c = _common()
    frozen = c.check_sources()
    original = frozen_common.check_sources()
    latency_path = c.ROOT / 'artifacts/semantic_jepa/latency.json'
    latency = json.loads(latency_path.read_text())
    if (latency.get('sealed') is not True or latency.get('outcomes_used') is not False
            or latency['source_sha256'] != original['source_sha256']
            or not set(FAMILIES) <= set(latency['profiles'])):
        raise RuntimeError('Inherited sealed timing identity changed')
    checkpoints = c.checkpoint_hashes()
    if len(checkpoints) != 3:
        raise RuntimeError('Require all three frozen JEPA ensemble members')
    for name, expected in checkpoints.items():
        if c.digest(c.ROOT / name) != expected or frozen['source_sha256'].get(name) != expected:
            raise RuntimeError('Frozen JEPA checkpoint differs from registration')
    return {'source_sha256': frozen['source_sha256'],
            'registration_sha256': c.digest(c.OUT / 'registration/source.json'),
            'protocol_sha256': c.digest(c.OUT / 'protocol.json'),
            'checkpoint_sha256': checkpoints, 'latency_sha256': c.digest(latency_path)}, latency


def _archive_paths(folder, key):
    return tuple(Path(folder) / (case_name(key) + suffix) for suffix in ('.json.gz', '.receipt.json'))


def _read_record(folder, key, *, replay=False, live_replication=False):
    mode = _mode(replay, live_replication)
    path, receipt_path = _archive_paths(folder, key)
    if not path.exists() and not receipt_path.exists():
        return None
    if not path.exists() or not receipt_path.exists():
        raise RuntimeError('Partial repair archive; refuse silent rerun')
    receipt = json.loads(receipt_path.read_text())
    flags = {'case': list(key), 'replay': replay, 'live_replication': live_replication,
             'accounting_mode': mode}
    if receipt.get('archive_sha256') != frozen_common.digest(path):
        raise RuntimeError('Repair archive hash changed')
    if any(receipt.get(name) != value for name, value in flags.items()):
        raise RuntimeError('Repair receipt identity changed')
    value = json.loads(gzip.decompress(path.read_bytes()))
    if (value.get('completed') is not True or value.get('stage') != key[0]
            or any(value.get(name) != expected for name, expected in flags.items())):
        raise RuntimeError('Repair archive body identity changed')
    if receipt.get('result_sha256') != _value_digest(value['result']):
        raise RuntimeError('Result receipt changed')
    return value


def _write_record(folder, key, value):
    path, receipt = _archive_paths(folder, key)
    frozen_common.write_once(path, value, compressed=True)
    frozen_common.write_once(receipt, {'archive_sha256': frozen_common.digest(path),
        'result_sha256': _value_digest(value['result']), 'case': list(key),
        'replay': value['replay'], 'live_replication': value['live_replication'],
        'accounting_mode': value['accounting_mode']})


def load_case(key, *, replay=False, live_replication=False):
    c = _common()
    validate_case_key(key)
    _replication_guard(key, c.protocol(), replay, live_replication)
    value = _read_record(_folder(c, replay, live_replication), key,
                         replay=replay, live_replication=live_replication)
    if value is None:
        return None
    identity, latency = _identities()
    if any(value.get(name) != expected for name, expected in identity.items()):
        raise RuntimeError('Repair archive source/checkpoint/timing identity changed')
    stage, site, cost, family, variant = _parts(key)
    expected = {'stage': stage, 'site': site, 'handover_seconds': cost, 'family': family,
                'controller_variant': variant, 'training_seed': None,
                'queue_mode': queue_mode(variant),
                'hardware_qualified': False, 'production_qualified': False, 'simulation_only': True}
    if any(value.get(name) != expected for name, expected in expected.items()):
        raise RuntimeError('Repair method or qualification identity changed')
    recorded, recorded_arrivals = None, None
    if replay or live_replication:
        main = load_case(key)
        if main is None:
            raise RuntimeError('Verified main case is required before any replication')
        main_path, _ = _archive_paths(c.OUT / 'cases', key)
        if value.get('source_main_archive_sha256') != c.digest(main_path):
            raise RuntimeError('Replication main-case binding changed')
        if replay:
            recorded, recorded_arrivals = main['jobs'], main['arrivals']
            if (value.get('recorded_schedule_sha256') != _value_digest({'jobs': recorded, 'arrivals': recorded_arrivals})
                    or value.get('recorded_arrivals_sha256') != _value_digest(recorded_arrivals)):
                raise RuntimeError('Offline replay schedule binding changed')
    audit = timing_audit(value['jobs'], value['forecast_consumption'], latency['profiles'][family],
                         c.protocol()['extra_compute_seconds'], c.protocol()['duration_seconds'],
                         arrivals=value['arrivals'], mode=value['queue_mode'],
                         accounting_mode=value['accounting_mode'], recorded_jobs=recorded,
                         recorded_arrivals=recorded_arrivals)
    if (value.get('paid_cost_and_consumption_pass') is not audit['passes']
            or value.get('timing_safety_pass') is not audit['passes']):
        raise RuntimeError('Timing summary differs from actual paid/consumption ledger')
    accepted = [job['decision'] for job in value['jobs']]
    if value.get('accepted_decisions') != accepted or value.get('forecast_metric_decisions') != accepted:
        raise RuntimeError('Accepted source timestamps differ from forecast diagnostics')
    return value


class DecisionTrace:
    def __init__(self):
        self.rows = []

    def __call__(self, decision, confident, after):
        # Confidence is a planning gate, not executed-acceptance ground truth.
        self.rows.append({'decision': int(decision.decision), 'current': decision.current.tolist(),
            'candidate': decision.candidate.tolist(), 'gain': decision.gain.tolist(),
            'eligible': decision.eligible.tolist(), 'confidence_mask': confident.tolist(),
            'after_ap_moves': after.tolist()})


class AdmissionTrace:
    """Observe real hook masks; approval is not proof of an executed move."""
    def __init__(self, admission, *, separate):
        self.admission, self.separate = admission, separate
        self.rows, self.calls = [], {}

    def __call__(self, decision):
        index = self.calls.get(decision.decision, 0)
        if index >= (2 if self.separate else 1):
            raise RuntimeError('Frozen admission-hook call order changed')
        self.calls[decision.decision] = index + 1
        mask = np.asarray(self.admission(decision))
        if mask.dtype != np.bool_ or mask.shape != decision.eligible.shape:
            raise RuntimeError('Malformed real admission mask')
        self.rows.append({'decision': int(decision.decision),
            'hook_role': 'reactive' if self.separate and index == 1 else 'planned',
            'current': decision.current.tolist(), 'candidate': decision.candidate.tolist(),
            'gain': decision.gain.tolist(), 'eligible': decision.eligible.tolist(),
            'admission_mask': mask.tolist()})
        return mask


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _paid_job_check(row, profile, extra, offline, original):
    kind = row.get('kind')
    fields = ('actual_prediction_seconds', 'actual_issue_seconds', 'actual_compute_seconds',
              'profile_seconds', 'charged_compute_seconds', 'charged_seconds', 'shared_extra_seconds')
    if kind not in profile or any(not _number(row.get(field)) for field in fields):
        return False
    equal = lambda a, b: _number(a) and math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12)
    actual = max(row['actual_prediction_seconds'], row['actual_issue_seconds'])
    allowance = profile[kind]
    if (not equal(row['actual_compute_seconds'], actual) or not equal(row['profile_seconds'], allowance)
            or not equal(row['charged_compute_seconds'], max(allowance, actual))
            or not equal(row['shared_extra_seconds'], extra)
            or not equal(row['charged_seconds'], row['charged_compute_seconds'] + extra)
            or row.get('runtime_overrun') is not (actual > allowance)
            or row.get('predictor_runtime_overrun') is not (row['actual_prediction_seconds'] > allowance)
            or row.get('accounting_mode') != ('recorded_schedule_offline' if offline else 'actual_cost')
            or row.get('offline_recorded_schedule') is not offline
            or row.get('completion_uses_actual_elapsed') is not (not offline)):
        return False
    if not offline:
        return True
    inherited = fields + ('decision', 'source_decision', 'job_index', 'issue_seconds',
                          'start_seconds', 'mode', 'kind', 'history_cutoff_exclusive', 'ready_slot')
    if any(row.get(field) != original.get(field) for field in inherited):
        return False
    fresh_fields = ('fresh_actual_prediction_seconds', 'fresh_actual_issue_seconds', 'fresh_actual_compute_seconds')
    if any(not _number(row.get(field)) for field in fresh_fields):
        return False
    fresh = max(row['fresh_actual_prediction_seconds'], row['fresh_actual_issue_seconds'])
    return (equal(row['fresh_actual_compute_seconds'], fresh)
        and row.get('fresh_runtime_overrun') is (fresh > allowance)
        and row.get('exceeds_original_recorded_compute') is (fresh > original['charged_compute_seconds'])
        and row.get('fresh_exceeds_original_actual_compute') is (fresh > original['actual_compute_seconds'])
        and row.get('recorded_completion_used_actual_elapsed') is True
        and row.get('recorded_forecast_sha256') == original['forecast_sha256']
        and row.get('recorded_forecast_std_sha256') == original['forecast_std_sha256']
        and row.get('matches_recorded_forecast') is (row['forecast_sha256'] == original['forecast_sha256']
            and row['forecast_std_sha256'] == original['forecast_std_sha256']))


def timing_audit(jobs, consumption, profile, extra, duration, *, arrivals, mode,
                 accounting_mode='actual_cost', recorded_jobs=None, recorded_arrivals=None):
    """Reconstruct every arrival, paid sparse job and source-time consumption."""
    offline = accounting_mode == 'recorded_schedule_offline'
    fail = {'passes': False, 'real_time_contract_checked': not offline, 'offline_schedule_only': offline}
    if accounting_mode not in ('actual_cost', 'recorded_schedule_offline') or mode not in ('fifo', 'latest'):
        raise ValueError('Unknown accounting or queue mode')
    if len(arrivals) != duration or (offline and (arrivals != recorded_arrivals or recorded_jobs is None or len(jobs) != len(recorded_jobs))):
        return fail
    previous, ordinal, ready, sources = 0., 0, [], []
    equal = lambda a, b: _number(a) and math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12)
    for decision, arrival in enumerate(arrivals):
        accepted = mode == 'fifo' or float(decision) >= previous
        if (arrival.get('decision') != decision or arrival.get('arrival_slot') != decision * 100
                or arrival.get('mode') != mode or arrival.get('accepted') is not accepted
                or arrival.get('job_index') != (ordinal if accepted else None)
                or arrival.get('reason') != ('accepted' if accepted else 'busy_previous_job')
                or not equal(arrival.get('previous_ready_seconds'), previous)
                or arrival.get('previous_ready_slot') != math.ceil(previous * 100 - 1e-10)):
            return fail
        if not accepted:
            continue
        if ordinal >= len(jobs):
            return fail
        job = jobs[ordinal]
        if (job.get('decision') != decision or job.get('source_decision') != decision
                or job.get('job_index') != ordinal or job.get('mode') != mode
                or job.get('history_cutoff_exclusive') != 60 + decision
                or not equal(job.get('issue_seconds'), float(decision))
                or not equal(job.get('start_seconds'), max(float(decision), previous))
                or not _paid_job_check(job, profile, extra, offline,
                                       recorded_jobs[ordinal] if offline else None)):
            return fail
        previous = max(float(decision), previous) + job['charged_seconds']
        slot = math.ceil(previous * 100 - 1e-10)
        if type(job.get('ready_slot')) is not int or job['ready_slot'] != slot:
            return fail
        ready.append(slot); sources.append(decision); ordinal += 1
    if ordinal != len(jobs):
        return fail
    expected = []
    for slot in sorted(set(ready)):
        if slot < duration * 100:
            index = int(np.searchsorted(ready, slot, side='right')) - 1
            expected.append({'slot': slot, 'forecast_decision': sources[index],
                'ready_slot': ready[index], 'age_seconds': slot / 100 - sources[index]})
    if consumption != expected:
        return fail
    held = [(expected[i + 1]['slot'] - 1 if i + 1 < len(expected) else duration * 100 - 1) / 100 - event['forecast_decision']
            for i, event in enumerate(expected)]
    return {'passes': True, 'envelope_exceedances': sum(job['runtime_overrun'] for job in jobs),
        'predictor_envelope_exceedances': sum(job['predictor_runtime_overrun'] for job in jobs),
        'real_time_contract_checked': not offline, 'offline_schedule_only': offline,
        'consumed_forecasts': len(consumption), 'arrivals': duration, 'accepted_jobs': len(jobs),
        'skipped_arrivals': duration - len(jobs),
        'maximum_consumed_forecast_age_seconds': max((event['age_seconds'] for event in expected), default=None),
        'maximum_held_forecast_age_seconds': max(held, default=None)}


def _evaluate(key, p, profile, *, recorded_jobs=None, recorded_arrivals=None):
    """Execute real runtime/kernel components; source checks belong to run_case.

    Small software fixtures call this helper on known non-study IDs. Scientific
    entry points always validate registration and immutable identities first.
    """
    stage, site, cost, family, variant = _parts(key)
    torch.set_num_threads(1)
    started = time.perf_counter()
    full, _, geometry, observations = episode(site, p['robots'], p['duration_seconds'],
                                             'drift' if site % 2 else 'main')
    warm = mask_scans(observations, p['scan_period_seconds']).before(60)
    delivered = TimedDeliveredScans(warm, p['duration_seconds'], site)
    # Snapshot genuine past local observations BEFORE any scan or control
    # action. Later cache updates must not rewrite this initial evidence.
    local_observability = {
        'initial_capture_slots': delivered.initial_local_capture_slots.copy().tolist(),
        'initial_heard': delivered.initial_local_capture_heard.copy().tolist(),
        'initial_rssi_dbm': delivered.local.copy().tolist(),
    }
    predictor = Predictor(geometry, family)
    queue_type = runtime.FreshForecasts if recorded_jobs is None else runtime.RecordedScheduleReplay
    queue_args = {'mode': queue_mode(variant)}
    if recorded_jobs is not None:
        queue_args.update(recorded_jobs=recorded_jobs, recorded_arrivals=recorded_arrivals)
    live = queue_type(predictor, delivered, p['duration_seconds'], profile,
                      p['extra_compute_seconds'], **queue_args)
    # Ordinary planned AND reactive moves retain the original cost-aware
    # admission in both variants. Only the sustained local watchdog can act
    # independently after a full outage plus the stop threshold has elapsed.
    utility = OutageUtility(live.mean, live, cost, 30.)
    admission_trace = AdmissionTrace(utility, separate=True)
    reading = READINGS[p.get('mission', 'bulk_uploads')]
    policy = replace(intent_policy(reading['weights'], reading['focus'], dual_hardware='none'),
        name=arm_name(family, variant), online_map=False, gate=True,
        scan_period_s=p['scan_period_seconds'], cadence_scan=True, roam_cost_s=cost,
        duplication=variant not in ('latest_single', 'latest_single_sense'))
    trace = DecisionTrace()
    world = runtime_world(full, p['duration_seconds'], live.mean, live.std[:, :, 0])
    result = simulate(world, policy,
        roam_gate=reactive_rescue,
        handover_admission=admission_trace,
        forecast_queue=live, forecast_source=live, scan_observer=delivered,
        decision_trace=trace, scan_duration_s=p['scan_outage_seconds'],
        roam_scorer=utility.scores,
        emergency_recovery=True, risk_gated_scans=(variant == 'latest_single_sense'))
    local_observability['completion_events'] = deepcopy(delivered.local_completion_events)

    # Private labels are read only after the completed control rollout.
    future = np.arange(60, 60 + p['duration_seconds'])[:, None] + np.asarray(HORIZONS, dtype=int)[None]
    truth = full.snr[future * 5].transpose(0, 2, 1, 3, 4)
    accepted = [job['decision'] for job in live.records]
    indices = np.asarray(accepted, dtype=int)
    computed, computed_truth = live.mean[indices], truth[indices]
    difference = computed - computed_truth
    predicted_ap = computed[:, :, :6].max(-1).mean(2)
    true_ap = computed_truth[:, :, :6].max(-1).mean(2)
    chosen = predicted_ap.argmax(-1)
    regret = true_ap.max(-1) - np.take_along_axis(true_ap, chosen[..., None], -1)[..., 0]
    forecast_metrics = {'rmse_db': float(np.sqrt(np.mean(difference ** 2))),
        'route_delta_rmse_db': float(np.sqrt(np.mean(((computed - computed[:, :, :1]) -
                                                     (computed_truth - computed_truth[:, :, :1])) ** 2))),
        'ap_choice_regret_db': float(regret.mean())}
    events = delivered.events
    causal = (len(live.arrivals) == p['duration_seconds']
        and all(e['measurement_slot'] < e['complete_slot'] <= e['delivery_slot']
                and e['first_usable_decision'] == e['delivery_slot'] // 100 + 1 for e in events)
        and all(row['decision'] == i for i, row in enumerate(live.arrivals))
        and all(row['job_index'] == i and row['source_decision'] == row['decision']
                and row['history_cutoff_exclusive'] == 60 + row['decision']
                for i, row in enumerate(live.records)))
    accounting_mode = 'actual_cost' if recorded_jobs is None else 'recorded_schedule_offline'
    audit = timing_audit(live.records, live.consumption_records, profile,
                         p['extra_compute_seconds'], p['duration_seconds'],
                         arrivals=live.arrivals, mode=queue_mode(variant),
                         accounting_mode=accounting_mode, recorded_jobs=recorded_jobs,
                         recorded_arrivals=recorded_arrivals)
    return {'completed': True, 'stage': stage, 'case': list(key), 'site': site,
        'handover_seconds': cost, 'family': family, 'controller_variant': variant,
        'training_seed': None, 'new_fits': 0,
        'result': result, 'forecast_metrics': forecast_metrics,
        'forecast_metric_scope': 'Accepted forecasts only at their actual source-time queries; not latency-adjusted consumed forecasts or matched-query accuracy.',
        'arrivals': live.arrivals, 'queue_mode': queue_mode(variant),
        'accepted_decisions': accepted, 'forecast_metric_decisions': accepted,
        'telemetry': delivered.summary(), 'jobs': live.records, 'decision_trace': trace.rows,
        'admission_trace': admission_trace.rows, 'forecast_consumption': live.consumption_records,
        'local_scan_observability': local_observability,
        'action_trace_sha256': _value_digest(result['packet_detail']['action_trace']),
        'forecast_mean_sha256': hashlib.sha256(computed.tobytes()).hexdigest(),
        'forecast_std_sha256': hashlib.sha256(live.std[indices].tobytes()).hexdigest(),
        'runtime_overrun': any(row['runtime_overrun'] for row in live.records),
        'predictor_runtime_overrun': any(row['predictor_runtime_overrun'] for row in live.records),
        'paid_cost_and_consumption_pass': audit['passes'], 'timing_safety_pass': audit['passes'],
        'timing_audit': audit, 'accounting_mode': accounting_mode,
        'causality_pass': causal, 'predictor_startup_seconds': predictor.startup_seconds,
        'elapsed_seconds': time.perf_counter() - started,
        'hardware_qualified': False, 'production_qualified': False, 'simulation_only': True}


def run_case(key, *, replay=False, live_replication=False):
    """Run one registered fresh case, or consume verified immutable evidence."""
    c = _common()
    validate_case_key(key)  # Before registration loading or any world construction.
    _replication_guard(key, c.protocol(), replay, live_replication)
    if key[0] == 'confirmation':
        c.check_confirmation_permission(key[2])
    key = tuple(key)
    folder = _folder(c, replay, live_replication)
    with c.lock(folder / ('.' + case_name(key) + '.lock')):
        existing = load_case(key, replay=replay, live_replication=live_replication)
        if existing is not None:
            return existing
        identity, latency = _identities()
        main, main_path = None, None
        if replay or live_replication:
            main = load_case(key)
            if main is None:
                raise RuntimeError('Complete and verify the main case before any replication')
            main_path, _ = _archive_paths(c.OUT / 'cases', key)
        value = _evaluate(key, c.protocol(), latency['profiles'][key[3]],
                          recorded_jobs=main['jobs'] if replay else None,
                          recorded_arrivals=main['arrivals'] if replay else None)
        value.update(identity, replay=replay, live_replication=live_replication,
            source_main_archive_sha256=c.digest(main_path) if main_path is not None else None,
            recorded_schedule_sha256=_value_digest({'jobs': main['jobs'], 'arrivals': main['arrivals']}) if replay else None,
            recorded_arrivals_sha256=_value_digest(main['arrivals']) if replay else None)
        after, _ = _identities()
        if identity != after:
            raise RuntimeError('Scientific identity changed during repair rollout')
        _write_record(folder, key, value)
        print(json.dumps({'phase': 'freshness_stack', 'stage': key[0], 'site': key[1], 'cost': key[2],
            'arm': arm_name(key[3], key[4]), 'misses': value['result']['control_misses'],
            'runtime_overrun': value['runtime_overrun'],
            'paid_cost_and_consumption_pass': value['paid_cost_and_consumption_pass'],
            'accounting_mode': value['accounting_mode'], 'replay': replay,
            'live_replication': live_replication}), flush=True)
        return value
