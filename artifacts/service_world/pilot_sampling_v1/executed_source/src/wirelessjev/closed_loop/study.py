"""Preregistered, restartable delivered-telemetry confirmation; no model fitting."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from ..frontier.study import ROOT, OUT as FRONT, sources, checkpoint_hashes, digest, mask_scans, READINGS, runtime_world, write
from ..fleetwm.data import episode
from ..native.intent_policy import intent_policy
from ..gate_repair.gate import reactive_rescue
from ..hardware_budget import OutageBudgetGate
from .runtime import SinglePredictor, LiveForecasts
from .telemetry import DeliveredScans
from .control import OutageUtility
from .sim import simulate

OUT = ROOT/'artifacts/closed_loop'
P = json.loads((OUT/'protocol.json').read_text())


def source_hashes():
    files = list(Path(__file__).parent.glob('*.py')) + [ROOT/'src/wirelessjev/hardware_budget.py', OUT/'protocol.json']
    return sources() | {str(f.relative_to(ROOT)): digest(f) for f in files}


def verify_seeds():
    current = set(P['development_sites']+P['test_sites'])
    assert len(current) == len(P['development_sites'])+len(P['test_sites'])
    def numbers(x):
        if isinstance(x, dict):
            return set().union(*(numbers(v) for v in x.values())) if x else set()
        if isinstance(x, list):
            return set().union(*(numbers(v) for v in x)) if x else set()
        return {x} if isinstance(x, int) else set()
    for path in (ROOT/'artifacts').glob('*/protocol.json'):
        if path.parent == OUT:
            continue
        assert not current & numbers(json.loads(path.read_text())), str(path)


def snapshot(stage):
    receipt = OUT/(stage+'_source.json')
    hashes = source_hashes()
    if receipt.exists():
        old = json.loads(receipt.read_text())
        assert old['source_sha256'] == hashes and old['checkpoint_sha256'] == checkpoint_hashes()
        return
    for name in hashes:
        path = OUT/(stage+'_source')/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, path)
    write(receipt, {'source_sha256': hashes, 'checkpoint_sha256': checkpoint_hashes(), 'time_unix': time.time(),
                    'latency_sha256': digest(OUT/'latency.json')})


class IdealHistory:
    def __init__(self, history):
        self.history = history
        self.n = history.value.shape[1]

    def before(self, decision):
        return self.history.before(60+decision)


def calibrate():
    dest = OUT/'latency.json'
    if dest.exists():
        assert json.loads(dest.read_text())['source_sha256'] == source_hashes()
        return
    if (OUT/'selection.json').exists():
        raise RuntimeError('Never recalibrate after development selection')
    torch.set_num_threads(1)
    full, _, g, obs = episode(301000, 100, 44)
    obs = mask_scans(obs, 5)
    old = json.loads((FRONT/'latency.json').read_text())['p95_seconds']['5']
    measured = {}; profile = {}
    for family in P['models']:
        pred = SinglePredictor(g, family)
        raw = {'cold': [], 'warm': [], 'refit': []}
        for d in range(44):
            _, _, kind, elapsed = pred.step(obs.before(60+d), full.plan[60+d])
            raw[kind].append(elapsed)
        profile[family] = {kind: max(old[family][kind], float(np.quantile(v, .95)) if v else old[family][kind]) for kind, v in raw.items()}
        measured[family] = raw
    write(dest, {'sealed': True, 'time_unix': time.time(), 'source_sha256': source_hashes(),
                 'checkpoint_sha256': checkpoint_hashes(), 'p95_seconds': profile, 'raw_seconds': measured,
                 'prior_timing_sha256': digest(FRONT/'latency.json'), 'test_accessed': False,
                 'calibration_site': 301000, 'threads': 1})
    print('CLOSED_LOOP_LATENCY_SEALED', flush=True)


def case(task):
    stage, seed, cost, family, variant, horizon = task
    torch.set_num_threads(1)
    key = f'{stage}-{seed}-{cost}-{family}-{variant}-{horizon}'
    dest = OUT/'cases'/(key+'.json')
    if dest.exists():
        r = json.loads(dest.read_text())
        assert r['source_sha256'] == source_hashes()
        assert r['checkpoint_sha256'] == checkpoint_hashes()
        assert r['latency_sha256'] == digest(OUT/'latency.json')
        return r
    started = time.time(); D = P['duration_seconds']; N = P['robots']
    full, _, g, original = episode(seed, N, D, 'drift' if seed%2 else 'main')
    original = mask_scans(original, 5)
    delivered = DeliveredScans(original.before(60), D, seed)
    history = IdealHistory(original) if variant == 'ideal_reports' else delivered
    profile = json.loads((OUT/'latency.json').read_text())['p95_seconds'][family]
    live = LiveForecasts(SinglePredictor(g, family), history, D, profile, P['extra_compute_seconds'])
    utility = OutageUtility(live.mean, live, cost, horizon) if variant == 'delivered_utility' else None
    gate = utility if utility is not None else OutageBudgetGate(live.mean, live, cost)
    reading = READINGS['bulk_uploads']
    policy = replace(intent_policy(reading['weights'], reading['focus'], dual_hardware='none'),
                     name=family+'_'+variant, online_map=False, gate=True, scan_period_s=5., cadence_scan=True, roam_cost_s=cost)
    world = runtime_world(full, D, live.mean, live.std[:, :, 0])
    result = simulate(world, policy, roam_gate=reactive_rescue, handover_admission=gate,
                      forecast_queue=live, forecast_source=live, scan_observer=delivered,
                      scan_duration_s=P['scan_outage_seconds'], roam_scorer=utility.scores if utility else None)
    # The evaluator alone receives hidden mean-field labels, after control ends.
    future = np.arange(60, 60+D)[:, None]+np.asarray([0,1,2,3,4,5,10,20,30])[None]
    truth = full.snr[future*5].transpose(0,2,1,3,4)
    forecast = {'rmse_db': float(np.sqrt(np.mean((live.mean-truth)**2))),
                'current_rmse_db': float(np.sqrt(np.mean((live.mean[:,:,0]-truth[:,:,0])**2))),
                'route_delta_rmse_db': float(np.sqrt(np.mean(((live.mean-live.mean[:,:,:1])-(truth-truth[:,:,:1]))**2)))}
    r = {'completed': True, 'stage': stage, 'scene': seed, 'condition': 'drift' if seed%2 else 'main',
         'family': family, 'variant': variant, 'horizon_seconds': horizon, 'handover_seconds': cost,
         'result': result, 'forecast': forecast, 'telemetry': delivered.summary(), 'jobs': live.records,
         'forecasts_consumed': live.last+1, 'last_forecast_age_seconds': live.age,
         'source_sha256': source_hashes(), 'checkpoint_sha256': checkpoint_hashes(),
         'latency_sha256': digest(OUT/'latency.json'), 'elapsed_seconds': time.time()-started,
         'no_truth_in_predictor_or_admission_inputs': True, 'simulation_only': True}
    write(dest, r)
    print(json.dumps({'closed_loop': key, 'misses': result['control_misses'], 'roams': result['roams'],
                      'reports': r['telemetry']['reports_delivered'], 'elapsed': round(time.time()-started, 2)}), flush=True)
    return r


def development(workers):
    if (OUT/'selection.json').exists():
        assert json.loads((OUT/'selection.json').read_text())['source_sha256'] == source_hashes()
        return
    snapshot('development')
    tasks = [('development', s, 4., f, v, h) for s in P['development_sites'] for f in P['models']
             for v,h in [('delivered_reports',30)]+[('delivered_utility',h) for h in P['horizon_candidates_seconds']]]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(case, tasks))
    options = []
    for h in P['horizon_candidates_seconds']:
        rs = [r['result'] for r in rows if r['variant']=='delivered_utility' and r['horizon_seconds']==h]
        options.append({'horizon_seconds': h, 'misses': sum(r['control_misses'] for r in rs),
                        'sum_bulk_mbps': sum(r['bulk_mbps'] for r in rs)})
    selected = min(options, key=lambda r:(r['misses'], -r['sum_bulk_mbps'], r['horizon_seconds']))
    write(OUT/'selection.json', {'sealed': True, 'sealed_at_unix': time.time(), 'selected': selected,
                                'candidates': options, 'source_sha256': source_hashes(),
                                'checkpoint_sha256': checkpoint_hashes(), 'latency_sha256': digest(OUT/'latency.json'),
                                'test_accessed': False})
    print('CLOSED_LOOP_DEVELOPMENT_SEALED', selected, flush=True)


def test(workers):
    choice = json.loads((OUT/'selection.json').read_text()); assert choice['sealed']
    assert choice['source_sha256'] == source_hashes() and choice['latency_sha256'] == digest(OUT/'latency.json')
    snapshot('test'); h = choice['selected']['horizon_seconds']
    tasks = [('test', s, cost, f, v, h if v=='delivered_utility' else 30)
             for s in P['test_sites'] for cost in P['handover_seconds'] for f in P['models'] for v in P['test_variants']]
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(case, tasks))
    write(OUT/'completion.json', {'completed': True, 'development_runs': 24, 'fresh_sites': len(P['test_sites']),
                                 'test_runs': len(tasks), 'native_runs': 24+len(tasks)})
    print('CLOSED_LOOP_CONFIRMATION_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['calibrate','development','test','all'], default='all')
    parser.add_argument('--workers', type=int, default=4); args = parser.parse_args()
    verify_seeds()
    if args.phase in ('calibrate','all'): calibrate()
    if args.phase in ('development','all'): development(args.workers)
    if args.phase in ('test','all'): test(args.workers)


if __name__ == '__main__':
    main()
