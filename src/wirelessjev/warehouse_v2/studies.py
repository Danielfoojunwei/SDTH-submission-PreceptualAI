"""Paired compatibility, external simulator, and extension study entry points."""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np
import torch

from ..fastplan import FastPlanner
from ..radiojev.e2 import build_scene, outcomes
from ..radiojev.nonlinear import nl_texts, nl_utility
from ..twin.warehouse import make_warehouse
from .experiment import atomic_json, sha256
from .missions import parse_mission, evaluate_mission
from .metrics import cluster_interval


def core_root(root):
    configuration = root/'core_run.json'
    return Path(json.loads(configuration.read_text())['path']) if configuration.exists() else root


def load_model(root, seed=0, arm='temporal_graph'):
    from .model import TemporalRadioModel
    path = core_root(root)/'checkpoints'/f'{arm}-{seed}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = TemporalRadioModel(**checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


def run_compatibility(root, data, embeddings, seeds):
    records = []
    semantic = json.loads((core_root(root)/'semantic_reader.json').read_text())
    readings = {item['text']: item for item in semantic['readings']}
    for seed in range(53000, 53020):
        # These are our own, provenance-recorded caches from the completed run.
        cached = Path(f'/workspace/training/radiojev/full/scene_cache/e2/test/{seed}.pkl')
        if cached.exists():
            with cached.open('rb') as f:
                scene_data = pickle.load(f)
        else:
            scene_data = build_scene(seed)
        scene = make_warehouse(seed)
        fp = FastPlanner(scene)
        o = outcomes(fp, scene_data['plans'], scene)
        for split in ['test', 'novel']:
            for mid, texts in nl_texts(split).items():
                for text in texts:
                    start = time.perf_counter()
                    spec = parse_mission(text, semantics='historical_e2')
                    value, _ = evaluate_mission(spec, o)
                    choice = int(value.argmax())
                    latency = (time.perf_counter()-start)*1000
                    gold = nl_utility(mid, o)  # Labels are used only after selection.
                    span = max(float(np.ptp(gold)), 1e-9)
                    item = {'scene': seed, 'split': split, 'text': text, 'reader': 'controlled_language',
                            'choice': choice, 'norm_regret': float((gold.max()-gold[choice])/span),
                            'parse_and_select_ms': latency}
                    records.append(item)
                    actual_reading = readings[text]
                    typed = {'scene': seed, 'split': split, 'text': text, 'reader': 'd3_typed_family',
                             'model_executed': actual_reading.get('receipt', {}).get('model_executed', True)}
                    if 'spec' in actual_reading:
                        from .missions import MissionSpec
                        score, _ = evaluate_mission(MissionSpec(**actual_reading['spec']), o)
                        picked = int(score.argmax())
                        typed.update(choice=picked, norm_regret=float((gold.max()-gold[picked])/span))
                    else:
                        typed.update(choice=None, norm_regret=None, status='deferred', error=actual_reading.get('error'))
                    records.append(typed)
        print(f'Structured E2 compatibility scene {seed}', flush=True)
    summary = {}
    for reader in ['controlled_language', 'd3_typed_family']:
        summary[reader] = {}
        for split in ['test', 'novel']:
            subset = [r for r in records if r['reader'] == reader and r['split'] == split]
            answered = [r for r in subset if r['norm_regret'] is not None]
            summary[reader][split] = {'n': len(subset), 'answered': len(answered),
                'coverage': len(answered)/len(subset),
                'norm_regret_answered': cluster_interval([r['norm_regret'] for r in answered], [r['scene'] for r in answered]) if answered else None}
    atomic_json(root/'compatibility.json', {'source': 'historical_e2_simulation_replay_with_new_readers',
        'semantics': 'historical_e2_soft', 'summary': summary, 'records': records,
        'limits': ['Uses the same simulator outcomes/candidate plans, not learned sensed-input radio predictions',
                   'Controlled-language parser has public warehouse glossary and rules, not learned language generalisation',
                   'Regret on answered missions must be read together with coverage',
                   'No global optimisation or hard-constraint guarantee']})


def ns3_binary(name):
    path = Path('/workspace/upstream/warehouse-v2/ns3/build-warehouse/scratch')/f'ns3.49-{name}-optimized'
    if not path.is_file():
        raise FileNotFoundError(f'Build the actual ns-3 executable first: {path}')
    return path


def write_topology(path, sample, association):
    inputs = sample['inputs']
    robot, ap = inputs['robot'], inputs['ap']
    lines = [f'{len(robot)} {len(ap)}']
    for a in ap:
        lines.append(f'{a[0]*120:.8f} {a[1]*80:.8f} {round(float(a[3])*165)} {int(a[2]>.5)}')
    for i, r in enumerate(robot):
        lines.append(f'{r[0]*120:.8f} {r[1]*80:.8f} {sample["demand"][i]:.8f} {int(association[i])}')
    path.write_text('\n'.join(lines)+'\n')


def run_ns3(root, data, embeddings, seeds):
    from concurrent.futures import ThreadPoolExecutor
    from .model import tensor_inputs
    binary = ns3_binary('warehouse-wifi')
    curve_binary = ns3_binary('warehouse-link-curves')
    cache = root/'cache'/'ns3'
    cache.mkdir(parents=True, exist_ok=True)
    curves = root/'wifi_link_curves.json'
    subprocess.run([str(curve_binary), '--output='+str(curves.resolve())], check=True)
    previous = json.loads((root/'ns3.json').read_text()) if (root/'ns3.json').exists() else {}
    report = {'source': 'ns3_packet_simulation', 'binary_sha256': sha256(binary),
              'curve_binary_sha256': sha256(curve_binary), 'curve_sha256': sha256(curves), 'snapshots': [],
              'limits': ['Static candidate-association snapshots, not mobility or handover validation',
                         '20 MHz channels; MinstrelHt; log-distance propagation without warehouse racks',
                         'Synthetic offered load; no physical-car measurement',
                         'Packets received during the drain period count toward application-duration goodput']}
    jobs, comparisons = {}, []
    for scene in sorted({s['scene'] for s in data['test']})[:3]:
        sample = next(s for s in data['test'] if s['scene'] == scene)
        model, checkpoint = load_model(root)
        with torch.inference_mode():
            prediction = model(tensor_inputs(sample), torch.tensor(sample['plans']))
        # Selection uses predicted fleet throughput, without truth labels.
        utility = (prediction['service'].numpy()*sample['demand']).sum(1)
        selected = int(utility.argmax())
        choices = {'stay': 0, 'strongest_scan': 1, 'observed_load_balance': 2, 'temporal_graph': selected}
        for policy, choice in choices.items():
            topology = cache/f'{scene}-{policy}.txt'
            metrics_path = cache/f'{scene}-{policy}.json'
            write_topology(topology, sample, sample['plans'][choice])
            key = (scene, sha256(topology))
            if key not in jobs:
                jobs[key] = (topology, metrics_path, scene)
            comparisons.append((key, scene, policy, choice))

    def execute_job(job):
        topology, metrics_path, scene = job
        reused = next((row for row in previous.get('snapshots', [])
                       if row['scene'] == scene and row['topology_sha256'] == sha256(topology)
                       and previous.get('binary_sha256') == report['binary_sha256']
                       and Path(row['raw_report']).is_file()
                       and sha256(Path(row['raw_report'])) == row['raw_sha256']), None)
        if reused:
            return Path(reused['raw_report']), reused['wall_time_s'], True
        start = time.perf_counter()
        subprocess.run([str(binary), '--input='+str(topology.resolve()), '--output='+str(metrics_path.resolve()),
                        '--seed=37', f'--run={scene}', '--duration=8'], check=True, timeout=600)
        return metrics_path, time.perf_counter()-start, False

    # Separate native processes and separate output paths; no shared ns-3 state.
    # Identical topology/association/seed comparisons explicitly share one run.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {key: pool.submit(execute_job, job) for key, job in jobs.items()}
        used = set()
        for key, scene, policy, choice in comparisons:
            metrics_path, elapsed, reused = futures[key].result()
            measured = json.loads(metrics_path.read_text())
            if measured['robots'] != 100 or measured['seed'] != 37 or measured['run'] != scene or measured['duration_s'] != 8:
                raise RuntimeError('Native simulation configuration/report mismatch')
            rows = measured['per_robot']
            throughput = [r['throughput_mbps'] for r in rows]
            topology = jobs[key][0]
            summary = {'scene': scene, 'policy': policy, 'candidate': choice,
                       'wall_time_s': elapsed, 'resumed_verified_report': reused,
                       'shares_identical_simulation': key in used,
                       'total_throughput_mbps': float(sum(throughput)),
                       'worst_decile_throughput_mbps': float(np.sort(throughput)[:10].mean()),
                       'robots_receiving_no_packets': sum(r['rx_packets'] == 0 for r in rows),
                       'raw_report': str(metrics_path), 'raw_sha256': sha256(metrics_path),
                       'topology_sha256': sha256(topology), 'per_robot': rows}
            used.add(key)
            report['snapshots'].append(summary)
            report['unique_native_simulations'] = len(used)
            atomic_json(root/'ns3.json', report)
            print(f'ns-3 scene={scene} policy={policy} Mbps={summary["total_throughput_mbps"]:.3f}', flush=True)


def run_sensing(root, data, embeddings, seeds):
    from .sensing import study
    atomic_json(root/'sensing.json', study(root))


def run_transfer(root, data, embeddings, seeds):
    from .adaptation import transfer_study
    atomic_json(root/'transfer.json', transfer_study(root, embeddings, seeds))


def run_federation(root, data, embeddings, seeds):
    from .adaptation import federation_study
    atomic_json(root/'federation.json', federation_study(root, embeddings, seeds))


def run_multitask(root, data, embeddings, seeds):
    from .multitask import study
    atomic_json(root/'multitask.json', study(root, seeds))
