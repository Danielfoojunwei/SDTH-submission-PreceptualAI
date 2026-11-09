"""Register, fit, charge, vet development, then conditionally confirm JEPA."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import gzip
import hashlib
import json
import math
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
from . import common as c
from .runtime import Predictor, ChargedForecasts
from .sim import simulate


def arms():
    p = c.protocol()
    return [('radiomap', None), ('frozen_fleet_jepa', None)] + [
        (family, seed) for family in p['families'] for seed in p['training_seeds']]


def arm_name(family, seed):
    return family if seed is None else f'{family}-s{seed}'


def training_receipt():
    value = json.loads((c.OUT / 'training.json').read_text())
    p = c.protocol()
    expected = {str((c.OUT / 'checkpoints' / f'{family}-{seed}.pt').relative_to(c.ROOT))
                for family in p['families'] for seed in p['training_seeds']}
    if (value.get('completed') is not True or value['fits'] != 12
            or set(value['checkpoint_sha256']) != expected
            or value['identity']['source_sha256'] != c.source_hashes()
            or value['identity']['protocol'] != p):
        raise RuntimeError('Complete all twelve frozen fits first')
    for name, expected in value['checkpoint_sha256'].items():
        if c.digest(c.ROOT / name) != expected:
            raise RuntimeError(f'Trained checkpoint changed: {name}')
    return value


def calibrate():
    frozen = c.check_sources()
    training = training_receipt()
    path = c.OUT / 'latency.json'
    if path.exists():
        value = json.loads(path.read_text())
        if value['source_sha256'] != frozen['source_sha256'] or value['checkpoint_sha256'] != training['checkpoint_sha256']:
            raise RuntimeError('Sealed latency identity changed')
        return value
    if (c.OUT / 'development.json').exists() or any((c.OUT / 'cases').glob('development-*.json.gz')):
        raise RuntimeError('Cannot retime after development outcomes')
    torch.set_num_threads(1)
    p = c.protocol()
    full, _, geometry, observations = episode(p['calibration_site'], p['robots'], 24)
    observations = mask_scans(observations, p['scan_period_seconds'])
    raw, startup = {}, {}
    for family, seed in arms():
        name = arm_name(family, seed)
        values = {'cold': [], 'warm': [], 'refit': []}
        startup[name] = []
        for repeat in range(3):
            predictor = Predictor(geometry, family, seed)
            startup[name].append(predictor.startup_seconds)
            for decision in range(24):
                _, _, kind, elapsed = predictor.step(observations.before(60 + decision), full.plan[60 + decision])
                values[kind].append(elapsed)
        if any(not x for x in values.values()):
            raise RuntimeError('Missing timing regime')
        raw[name] = values
        print(json.dumps({'phase': 'calibration', 'arm': name,
                          'p95_seconds': {k: float(np.quantile(v, .95)) for k, v in values.items()}}), flush=True)
    envelope = {kind: max(float(np.quantile(raw[arm_name(f, s)][kind], .95))
                         for f in p['families'] for s in p['training_seeds'])
                for kind in ('cold', 'warm', 'refit')}
    charge = lambda values: {kind: max(.01, math.ceil(2 * value * 100) / 100) for kind, value in values.items()}
    profiles = {}
    for family, seed in arms():
        name = arm_name(family, seed)
        profiles[name] = charge(envelope if family in p['families'] else {
            kind: float(np.quantile(values, .95)) for kind, values in raw[name].items()})
    result = {'sealed': True, 'source_sha256': frozen['source_sha256'],
              'checkpoint_sha256': training['checkpoint_sha256'], 'profiles': profiles,
              'raw_seconds': raw, 'startup_seconds_separate_from_request': startup,
              'neural_common_p95_seconds': envelope, 'threads': 1,
              'rule': 'Common neural max-arm p95 by regime; double, round up10ms, floor10ms. Comparators own p95.',
              'outcomes_used': False, 'calibration_site': p['calibration_site']}
    c.write_once(path, result)
    return result


def case_keys(stage, costs=None):
    p = c.protocol()
    if stage not in ('development', 'confirmation'):
        raise ValueError('Unknown evaluation stage')
    costs = p['handover_seconds'] if costs is None else costs
    return [(stage, site, float(cost), family, seed) for site in p[stage + '_sites']
            for cost in costs for family, seed in arms()]


def case_name(key):
    stage, site, cost, family, seed = key
    return f'{stage}-{site}-h{round(cost * 1000):04d}-{arm_name(family, seed)}'


def validate_case_key(key):
    if not isinstance(key, (tuple, list)) or len(key) != 5:
        raise ValueError('Malformed scientific case key')
    stage, site, cost, family, seed = key
    p = c.protocol()
    if (stage not in ('development', 'confirmation') or type(site) is not int
            or site not in p[stage + '_sites'] or type(cost) not in (float, int)
            or cost not in p['handover_seconds'] or (seed is not None and type(seed) is not int)
            or (family, seed) not in arms()):
        raise ValueError('Unregistered scientific case key')


def load_case(key, *, replay=False):
    validate_case_key(key)
    path = c.OUT / ('replays' if replay else 'cases') / (case_name(key) + '.json.gz')
    receipt_path = path.with_name(case_name(key) + '.receipt.json')
    if not path.exists() and not receipt_path.exists():
        return None
    if not path.exists() or not receipt_path.exists():
        raise RuntimeError('Partial archive identity; refuse silent rerun')
    receipt = json.loads(receipt_path.read_text())
    if receipt['archive_sha256'] != c.digest(path):
        raise RuntimeError('Archive hash changed')
    value = json.loads(gzip.decompress(path.read_bytes()))
    if value['case'] != list(key) or value['replay'] is not replay:
        raise RuntimeError('Archive case identity mismatch')
    if value['source_sha256'] != c.check_sources()['source_sha256']:
        raise RuntimeError('Archive source changed')
    if value['checkpoint_sha256'] != training_receipt()['checkpoint_sha256']:
        raise RuntimeError('Archive checkpoint identity changed')
    if value['latency_sha256'] != c.digest(c.OUT / 'latency.json'):
        raise RuntimeError('Archive latency identity changed')
    if receipt['result_sha256'] != c.value_digest(value['result']):
        raise RuntimeError('Result receipt changed')
    return value


class DecisionTrace:
    def __init__(self):
        self.rows = []

    def __call__(self, decision, confident, after):
        self.rows.append({'decision': int(decision.decision), 'current': decision.current.tolist(),
                          'candidate': decision.candidate.tolist(), 'gain': decision.gain.tolist(),
                          'eligible': decision.eligible.tolist(), 'confidence_mask': confident.tolist(),
                          'after_ap_moves': after.tolist()})


def case(task):
    key, replay = task
    validate_case_key(key)  # Reject before creating a world or opening new site data.
    stage, site, cost, family, seed = key
    if stage == 'confirmation':
        development = json.loads((c.OUT / 'development.json').read_text())
        qualification = json.loads((c.OUT / 'qualification.json').read_text())
        if (qualification['development_sha256'] != c.digest(c.OUT / 'development.json')
                or qualification['passed_costs'] != development['passed_costs']
                or qualification['source_sha256'] != c.source_hashes()
                or cost not in qualification['passed_costs']):
            raise RuntimeError('Development-vetoed confirmation is prohibited')
    name = case_name(key)
    folder = c.OUT / ('replays' if replay else 'cases')
    with c.locked(folder / ('.' + name + '.lock')):
        old = load_case(key, replay=replay)
        if old is not None:
            return old
        frozen = c.check_sources()
        training = training_receipt()
        p = c.protocol()
        torch.set_num_threads(1)
        started = time.perf_counter()
        full, _, geometry, observations = episode(site, p['robots'], p['duration_seconds'], 'drift' if site % 2 else 'main')
        warm = mask_scans(observations, p['scan_period_seconds']).before(60)
        delivered = DeliveredScans(warm, p['duration_seconds'], site)
        predictor = Predictor(geometry, family, seed)
        profile = json.loads((c.OUT / 'latency.json').read_text())['profiles'][arm_name(family, seed)]
        live = ChargedForecasts(predictor, delivered, p['duration_seconds'], profile, p['extra_compute_seconds'])
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
        # All private labels are read only AFTER the completed control rollout.
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
        causal = (all(e['measurement_slot'] < e['complete_slot'] <= e['delivery_slot'] and
                      e['first_usable_decision'] == e['delivery_slot'] // 100 + 1 for e in events)
                  and all(row['history_cutoff_exclusive'] == 60 + i for i, row in enumerate(live.records)))
        value = {'completed': True, 'case': list(key), 'replay': replay, 'stage': stage, 'site': site,
                 'handover_seconds': cost, 'family': family, 'training_seed': seed,
                 'result': result, 'forecast_metrics': forecast_metrics, 'telemetry': delivered.summary(),
                 'jobs': live.records, 'decision_trace': trace.rows,
                 'forecast_mean_sha256': hashlib.sha256(live.mean.tobytes()).hexdigest(),
                 'forecast_std_sha256': hashlib.sha256(live.std.tobytes()).hexdigest(),
                 'source_sha256': frozen['source_sha256'], 'checkpoint_sha256': training['checkpoint_sha256'],
                 'latency_sha256': c.digest(c.OUT / 'latency.json'),
                 'runtime_overrun': any(row['runtime_overrun'] for row in live.records),
                 'causality_pass': causal, 'predictor_startup_seconds': predictor.startup_seconds,
                 'elapsed_seconds': time.perf_counter() - started, 'hardware_qualified': False}
        c.check_sources()
        c.write_once(folder / (name + '.json.gz'), value, compressed=True)
        c.write_once(folder / (name + '.receipt.json'), {
            'archive_sha256': c.digest(folder / (name + '.json.gz')),
            'result_sha256': c.value_digest(result), 'case': list(key), 'replay': replay})
        print(json.dumps({'phase': stage, 'site': site, 'cost': cost, 'arm': arm_name(family, seed),
                          'misses': result['control_misses'], 'runtime_overrun': value['runtime_overrun'],
                          'replay': replay}), flush=True)
        return value


def execute(keys, workers=2, *, replay=False):
    if workers not in (1, 2):
        raise ValueError('One or two evaluation workers are registered')
    tasks = [(key, replay) for key in keys]
    if workers == 1:
        return [case(task) for task in tasks]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(case, tasks))


def development(workers=2):
    from .report import development_gate
    c.check_sources()
    training_receipt()
    calibrate()
    rows = execute(case_keys('development'), workers)
    verdict = development_gate(rows, c.protocol(), integrity=all(r['causality_pass'] for r in rows))
    verdict.update(source_sha256=c.check_sources()['source_sha256'],
                   checkpoint_sha256=training_receipt()['checkpoint_sha256'],
                   latency_sha256=c.digest(c.OUT / 'latency.json'), completed=True)
    c.write_once(c.OUT / 'development.json', verdict)
    return verdict


def replay_keys(costs):
    site = c.protocol()['confirmation_sites'][0]
    return [('confirmation', site, float(cost), family, seed) for cost in costs
            for family, seed in [('radiomap', None), ('supervised', 0), ('semantic_jepa', 0)]]


def compare_replays(originals, replays, expected_keys):
    expected_keys = list(expected_keys)
    original_index = {tuple(row['case']): row for row in originals}
    replay_index = {tuple(row['case']): row for row in replays}
    if len(replay_index) != len(replays) or set(replay_index) != set(expected_keys):
        raise ValueError('Replay membership differs from registration')
    if not set(expected_keys) <= set(original_index):
        raise ValueError('Replay has no original case')
    fields = ('result', 'forecast_mean_sha256', 'forecast_std_sha256', 'telemetry', 'decision_trace')
    return [{'case': list(key), 'exact': all(original_index[key][field] == replay_index[key][field]
                                           for field in fields)} for key in expected_keys]


def validate_development_seal(dev, rows):
    """Recompute the veto before any reserved confirmation world is opened."""
    from .report import development_gate
    expected = development_gate(rows, c.protocol(), integrity=all(row['causality_pass'] for row in rows))
    for name, value in expected.items():
        if dev.get(name) != value:
            raise RuntimeError(f'Development verdict differs from raw archives: {name}')
    if (dev.get('completed') is not True or dev['source_sha256'] != c.source_hashes()
            or dev['checkpoint_sha256'] != training_receipt()['checkpoint_sha256']
            or dev['latency_sha256'] != c.digest(c.OUT / 'latency.json')):
        raise RuntimeError('Development source/checkpoint/latency seal changed')
    return expected['passed_costs']


def confirmation(workers=2):
    c.check_sources()
    dev = json.loads((c.OUT / 'development.json').read_text())
    development_rows = [load_case(key) for key in case_keys('development')]
    if any(row is None for row in development_rows):
        raise RuntimeError('Complete development archives before confirmation')
    costs = validate_development_seal(dev, development_rows)
    c.write_once(c.OUT / 'qualification.json', {'development_sha256': c.digest(c.OUT / 'development.json'),
                 'passed_costs': costs, 'source_sha256': c.source_hashes(),
                 'recomputed_from_all_development_archives': True})
    rows = execute(case_keys('confirmation', costs), workers) if costs else []
    replays = execute(replay_keys(costs), 1, replay=True) if costs else []
    comparisons = compare_replays(rows, replays, replay_keys(costs))
    value = {'completed': True, 'passed_costs': costs, 'confirmation_cases': len(rows),
             'source_sha256': c.check_sources()['source_sha256'],
             'replays': comparisons, 'all_exact': all(row['exact'] for row in comparisons),
             'confirmation_not_run_if_development_rejected': not costs}
    c.write_once(c.OUT / 'confirmation.json', value)
    return value


def report():
    from .report import build_report, markdown
    c.check_sources()
    dev = json.loads((c.OUT / 'development.json').read_text())
    completion = json.loads((c.OUT / 'confirmation.json').read_text())
    devrows = [load_case(key) for key in case_keys('development')]
    rows = [load_case(key) for key in case_keys('confirmation', dev['passed_costs'])]
    if any(row is None for row in devrows + rows):
        raise RuntimeError('Incomplete case matrix')
    validate_development_seal(dev, devrows)
    expected_replays = replay_keys(dev['passed_costs'])
    replays = [load_case(key, replay=True) for key in expected_replays]
    if any(row is None for row in replays):
        raise RuntimeError('Missing required replay archive')
    comparisons = compare_replays(rows, replays, expected_replays)
    exact = all(row['exact'] for row in comparisons)
    if (completion.get('completed') is not True or completion['source_sha256'] != c.source_hashes() or completion['passed_costs'] != dev['passed_costs']
            or completion['confirmation_cases'] != len(rows) or completion['replays'] != comparisons
            or completion['all_exact'] is not exact):
        raise RuntimeError('Confirmation/replay receipt differs from actual archives')
    value = build_report(devrows, rows, c.protocol(), validation={
        'source_and_archive_integrity': True,
        'causality_and_kernel_checks': all(row['causality_pass'] for row in devrows + rows),
        'exact_replay': exact})
    value.update(source_sha256=c.source_hashes(), training_sha256=c.digest(c.OUT / 'training.json'),
                 development_sha256=c.digest(c.OUT / 'development.json'),
                 confirmation_sha256=c.digest(c.OUT / 'confirmation.json'),
                 latency_sha256=c.digest(c.OUT / 'latency.json'))
    c.write_once(c.OUT / 'report.json', value)
    c.write_once(c.OUT / 'report_markdown.json', {'markdown': markdown(value)})
    text = markdown(value)
    c.write_text_once(c.OUT / 'report.md', text)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'train', 'calibrate', 'development', 'confirmation', 'report', 'all'))
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.phase == 'register':
        c.register()
    else:
        c.check_sources()
    if args.phase in ('train', 'all'):
        from .training import train
        train(args.workers)
    if args.phase in ('calibrate', 'all'):
        calibrate()
    if args.phase in ('development', 'all'):
        development(args.workers)
    if args.phase in ('confirmation', 'all'):
        confirmation(args.workers)
    if args.phase in ('report', 'all'):
        report()


if __name__ == '__main__':
    main()
