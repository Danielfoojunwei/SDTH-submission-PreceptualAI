"""Fresh selected-JEPA confirmation using frozen, real runtime components.

The semantic experiment's own veto and reserved sites remain untouched. This
separate runner requires its own registered sites, selected checkpoint family
and source identity before opening a confirmation world.
"""
from __future__ import annotations

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
from ..closed_loop.telemetry import DeliveredScans
from ..contracted.data import READINGS, runtime_world
from ..fleetwm.data import episode
from ..frontier.study import mask_scans
from ..gate_repair.gate import reactive_rescue
from ..hardware_budget import OutageBudgetGate
from ..native.intent_policy import intent_policy
from ..native.world import HORIZONS
from ..semantic_jepa import common as frozen_common
from ..semantic_jepa.runtime import ChargedForecasts, Predictor
from ..semantic_jepa.sim import simulate
from ..semantic_jepa.study import training_receipt

BASELINES = ('radiomap', 'frozen_fleet_jepa')
REPLAY_FIELDS = ('result', 'forecast_mean_sha256', 'forecast_std_sha256',
                 'telemetry', 'decision_trace', 'action_trace_sha256')


def _common():
    return importlib.import_module('.common', __package__)


def _value_digest(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def selected_family(p=None, selection=None):
    c = _common() if p is None or selection is None else None
    p = c.protocol() if p is None else p
    selection = c.selection() if selection is None else selection
    family = selection.get('selected_family', selection.get('candidate_family'))
    if family not in ('context_jepa', 'semantic_jepa'):
        raise ValueError('Selection must identify an actual preregistered JEPA family')
    if p.get('candidate_family', family) != family:
        raise ValueError('Protocol and selected family differ')
    if selection.get('sealed') is not True:
        raise ValueError('Selected family must be sealed before evaluation')
    return family


def arms(p=None, selection=None):
    c = _common() if p is None or selection is None else None
    p = c.protocol() if p is None else p
    selection = c.selection() if selection is None else selection
    family = selected_family(p, selection)
    seeds = p['training_seeds']
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed not in (0, 1, 2) for seed in seeds)):
        raise ValueError('Use distinct frozen paired training seeds')
    return [(name, None) for name in BASELINES] + [
        (name, seed) for name in (family, 'supervised') for seed in seeds]


def arm_name(family, seed):
    return family if seed is None else f'{family}-s{seed}'


def validate_case_key(key, p=None, selection=None):
    if not isinstance(key, (tuple, list)) or len(key) != 4:
        raise ValueError('Malformed scientific case key')
    c = _common() if p is None or selection is None else None
    p = c.protocol() if p is None else p
    selection = c.selection() if selection is None else selection
    site, cost, family, seed = key
    if (type(site) is not int or site not in p['confirmation_sites']
            or type(cost) not in (float, int) or not math.isfinite(cost)
            or cost not in p['handover_seconds']
            or (seed is not None and type(seed) is not int)
            or (family, seed) not in arms(p, selection)):
        raise ValueError('Unregistered selected-JEPA case key')
    if not math.isclose(float(cost) * 1000, round(float(cost) * 1000), abs_tol=1e-9, rel_tol=0):
        raise ValueError('Case filename requires an exact millisecond cost')


def case_name(key):
    site, cost, family, seed = key
    return f'confirmation-{site}-h{round(cost * 1000):04d}-{arm_name(family, seed)}'


def case_keys(p=None, selection=None):
    c = _common() if p is None or selection is None else None
    p = c.protocol() if p is None else p
    selection = c.selection() if selection is None else selection
    return [(site, float(cost), family, seed) for site in p['confirmation_sites']
            for cost in p['handover_seconds'] for family, seed in arms(p, selection)]


def _identities():
    c = _common()
    frozen = c.check_sources()
    p, selection = c.protocol(), c.selection()
    selected_family(p, selection)
    training = training_receipt()  # Verifies every original frozen checkpoint.
    original = frozen_common.check_sources()
    latency_path = c.ROOT / 'artifacts/semantic_jepa/latency.json'
    latency = json.loads(latency_path.read_text())
    if (latency.get('sealed') is not True or latency.get('outcomes_used') is not False
            or latency['checkpoint_sha256'] != training['checkpoint_sha256']
            or latency['source_sha256'] != original['source_sha256']):
        raise RuntimeError('Original sealed timing/checkpoint identity changed')
    required_profiles = {arm_name(family, seed) for family, seed in arms(p, selection)}
    if not required_profiles <= set(latency['profiles']):
        raise RuntimeError('A selected arm has no pre-existing sealed timing profile')
    return {'source_sha256': frozen['source_sha256'],
            'registration_sha256': c.digest(c.OUT / 'registration/source.json'),
            'selection_sha256': c.digest(c.OUT / 'selection.json'),
            'checkpoint_sha256': training['checkpoint_sha256'],
            'parent_training_sha256': c.digest(c.ROOT / 'artifacts/semantic_jepa/training.json'),
            'latency_sha256': c.digest(latency_path)}, latency


def _archive_paths(folder, key):
    name = case_name(key)
    folder = Path(folder)
    return folder / (name + '.json.gz'), folder / (name + '.receipt.json')


def _read_record(folder, key, *, replay=False):
    """Verify passive bytes/identity before any provenance-aware consumption."""
    path, receipt_path = _archive_paths(folder, key)
    if not path.exists() and not receipt_path.exists():
        return None
    if not path.exists() or not receipt_path.exists():
        raise RuntimeError('Partial selected-JEPA archive; refuse silent rerun')
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('archive_sha256') != frozen_common.digest(path)
            or receipt.get('case') != list(key) or receipt.get('replay') is not replay):
        raise RuntimeError('Archive receipt hash or case identity changed')
    value = json.loads(gzip.decompress(path.read_bytes()))
    if (value.get('case') != list(key) or value.get('replay') is not replay
            or value.get('stage') != 'confirmation' or value.get('completed') is not True):
        raise RuntimeError('Archive body identity changed')
    if receipt.get('result_sha256') != _value_digest(value['result']):
        raise RuntimeError('Result receipt changed')
    return value


def _write_record(folder, key, value):
    """Reuse the immutable byte writer, never replace existing evidence."""
    path, receipt = _archive_paths(folder, key)
    frozen_common.write_once(path, value, compressed=True)
    frozen_common.write_once(receipt, {'archive_sha256': frozen_common.digest(path),
        'result_sha256': _value_digest(value['result']), 'case': list(key),
        'replay': value['replay']})


def load_case(key, *, replay=False):
    c = _common()
    validate_case_key(key)
    if type(replay) is not bool:
        raise ValueError('Replay flag must be boolean')
    value = _read_record(c.OUT / ('replays' if replay else 'cases'), key, replay=replay)
    if value is None:
        return None
    identity, _ = _identities()
    if any(value.get(name) != expected for name, expected in identity.items()):
        raise RuntimeError('Archive source, selection, checkpoint or timing identity changed')
    site, cost, family, seed = key
    expected = {'site': site, 'handover_seconds': cost, 'family': family, 'training_seed': seed,
                'hardware_qualified': False, 'simulation_only': True}
    if any(value.get(name) != expected for name, expected in expected.items()):
        raise RuntimeError('Archive method or qualification identity changed')
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


def _evaluate(key, p, profile):
    """Execute real runtime/kernel components; source checks belong to run_case.

    Small software fixtures call this helper on known non-study IDs. Scientific
    entry points always validate registration and immutable identities first.
    """
    site, cost, family, seed = key
    torch.set_num_threads(1)
    started = time.perf_counter()
    full, _, geometry, observations = episode(site, p['robots'], p['duration_seconds'],
                                             'drift' if site % 2 else 'main')
    warm = mask_scans(observations, p['scan_period_seconds']).before(60)
    delivered = DeliveredScans(warm, p['duration_seconds'], site)
    predictor = Predictor(geometry, family, seed)
    live = ChargedForecasts(predictor, delivered, p['duration_seconds'], profile,
                           p['extra_compute_seconds'])
    utility = OutageUtility(live.mean, live, cost, 30) if cost == 4. else None
    admission = utility if utility is not None else OutageBudgetGate(live.mean, live, cost)
    reading = READINGS[p['mission']]
    policy = replace(intent_policy(reading['weights'], reading['focus'], dual_hardware='none'),
        name=arm_name(family, seed), online_map=False, gate=True,
        scan_period_s=p['scan_period_seconds'], cadence_scan=True, roam_cost_s=cost)
    trace = DecisionTrace()
    world = runtime_world(full, p['duration_seconds'], live.mean, live.std[:, :, 0])
    result = simulate(world, policy, roam_gate=reactive_rescue, handover_admission=admission,
        forecast_queue=live, forecast_source=live, scan_observer=delivered,
        decision_trace=trace, scan_duration_s=p['scan_outage_seconds'],
        roam_scorer=utility.scores if utility is not None else None)

    # Private labels are read only after the completed control rollout.
    future = np.arange(60, 60 + p['duration_seconds'])[:, None] + np.asarray(HORIZONS, dtype=int)[None]
    truth = full.snr[future * 5].transpose(0, 2, 1, 3, 4)
    difference = live.mean - truth
    predicted_ap = live.mean[:, :, :6].max(-1).mean(2)
    true_ap = truth[:, :, :6].max(-1).mean(2)
    chosen = predicted_ap.argmax(-1)
    regret = true_ap.max(-1) - np.take_along_axis(true_ap, chosen[..., None], -1)[..., 0]
    forecast_metrics = {'rmse_db': float(np.sqrt(np.mean(difference ** 2))),
        'route_delta_rmse_db': float(np.sqrt(np.mean(((live.mean - live.mean[:, :, :1]) -
                                                     (truth - truth[:, :, :1])) ** 2))),
        'ap_choice_regret_db': float(regret.mean())}
    events = delivered.events
    causal = (len(live.records) == p['duration_seconds']
        and all(e['measurement_slot'] < e['complete_slot'] <= e['delivery_slot']
                and e['first_usable_decision'] == e['delivery_slot'] // 100 + 1 for e in events)
        and all(row['decision'] == i and row['history_cutoff_exclusive'] == 60 + i
                for i, row in enumerate(live.records)))
    return {'completed': True, 'stage': 'confirmation', 'case': list(key), 'site': site,
        'handover_seconds': cost, 'family': family, 'training_seed': seed,
        'result': result, 'forecast_metrics': forecast_metrics,
        'forecast_metric_scope': 'Issued forecasts at issue-time horizons; not latency-adjusted consumed forecasts.',
        'telemetry': delivered.summary(), 'jobs': live.records, 'decision_trace': trace.rows,
        'action_trace_sha256': _value_digest(result['packet_detail']['action_trace']),
        'forecast_mean_sha256': hashlib.sha256(live.mean.tobytes()).hexdigest(),
        'forecast_std_sha256': hashlib.sha256(live.std.tobytes()).hexdigest(),
        'runtime_overrun': any(row['runtime_overrun'] for row in live.records),
        'causality_pass': causal, 'predictor_startup_seconds': predictor.startup_seconds,
        'elapsed_seconds': time.perf_counter() - started,
        'hardware_qualified': False, 'simulation_only': True}


def run_case(key, *, replay=False):
    """Run one registered fresh case, or consume its verified existing archive."""
    c = _common()
    validate_case_key(key)  # Must happen before any world construction.
    if type(replay) is not bool:
        raise ValueError('Replay flag must be boolean')
    key = tuple(key)
    folder = c.OUT / ('replays' if replay else 'cases')
    with c.locked(folder / ('.' + case_name(key) + '.lock')):
        existing = load_case(key, replay=replay)
        if existing is not None:
            return existing
        identity, latency = _identities()
        p = c.protocol()
        site, cost, family, seed = key
        profile = latency['profiles'][arm_name(family, seed)]
        value = _evaluate(key, p, profile)
        value.update(identity, replay=replay)
        # Fail closed if source/configuration changed while this case ran.
        after, _ = _identities()
        if identity != after:
            raise RuntimeError('Scientific identity changed during selected-JEPA rollout')
        _write_record(folder, key, value)
        print(json.dumps({'phase': 'selected_confirmation', 'site': site, 'cost': cost,
            'arm': arm_name(family, seed), 'misses': value['result']['control_misses'],
            'runtime_overrun': value['runtime_overrun'], 'replay': replay}), flush=True)
        return value
