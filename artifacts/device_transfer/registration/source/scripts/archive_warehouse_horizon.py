"""Validate and archive completed horizon research without inventing results."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean
import tarfile

import numpy as np
import torch

from wirelessjev.warehouse_v2.experiment import atomic_json, sha256


def load(path):
    return json.loads(Path(path).read_text())


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise RuntimeError('Nonfinite model checkpoint')
    elif isinstance(value, dict):
        for child in value.values(): finite_tree(child)
    elif isinstance(value, list):
        for child in value: finite_tree(child)
    elif isinstance(value, float) and not np.isfinite(value):
        raise RuntimeError('Nonfinite research report')


def paired_interval(records, method, metric, *, draws=4000):
    """Full-cross paired model-seed/environment bootstrap, not nine IID rows."""
    baseline = {r['scene']: r for r in records if r['method'] == 'numerical_planner'}
    scenes = sorted(baseline)
    selected = [r for r in records if r['method'] == method]
    seeds = sorted({r['model_seed'] for r in selected})
    def metric_value(row):
        return row[metric] if metric in row else row['metrics'][metric]
    matrix = np.array([[metric_value(next(r for r in selected if r['scene'] == scene and r['model_seed'] == seed))
                        -metric_value(baseline[scene]) for scene in scenes] for seed in seeds])
    rng = np.random.default_rng(0)
    samples = [matrix[np.ix_(rng.integers(len(seeds), size=len(seeds)),
                             rng.integers(len(scenes), size=len(scenes)))].mean() for _ in range(draws)]
    return {'difference': float(matrix.mean()), 'ci95': np.quantile(samples, [.025, .975]).tolist(),
            'model_seeds': seeds, 'environments': scenes, 'paired_matrix': matrix.tolist(),
            'method': 'crossed_model_seed_and_environment_percentile_bootstrap'}


def robot_comparison(records, method, baseline_method='budgeted_strongest', threshold=.01):
    baseline = {r['scene']: r for r in records if r['method'] == baseline_method}
    selected = [r for r in records if r['method'] == method]
    rows = []
    for row in selected:
        old = baseline[row['scene']]['metrics']['per_robot']
        new = row['metrics']['per_robot']
        differences = {robot: value['mean_service']-old[robot]['mean_service'] for robot, value in new.items()}
        worst = min(differences, key=differences.get)
        rows.append({'scene': row['scene'], 'model_seed': row['model_seed'],
                     'higher_by_more_than_threshold': sum(v > threshold for v in differences.values()),
                     'lower_by_more_than_threshold': sum(v < -threshold for v in differences.values()),
                     'within_threshold': sum(abs(v) <= threshold for v in differences.values()),
                     'worst_robot': worst, 'worst_change': differences[worst]})
    return {'baseline': baseline_method, 'service_fraction_threshold': threshold, 'records': rows,
            'descriptive_only': True, 'independent_physical_robots': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('/workspace/training/warehouse-horizon'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/warehouse_horizon'))
    args = parser.parse_args()
    root, out = args.run, args.out
    status = load(root/'status.json')
    if status['state'] != 'completed':
        raise RuntimeError('Horizon campaign is not complete')
    for name, digest in status['reports'].items():
        if sha256(root/name) != digest:
            raise RuntimeError('Completion receipt/report mismatch')
    provenance = load(root/'provenance.json')
    for relative, digest in provenance['source_sha256'].items():
        if sha256(root/'source_snapshot'/relative) != digest:
            raise RuntimeError('Executed source capsule changed')
    selection = load(root/'selection.json')
    checkpoints = {}
    for name, digest in selection['checkpoint_sha256'].items():
        path = root/'checkpoints'/name
        if sha256(path) != digest:
            raise RuntimeError('Checkpoint changed after selection')
        finite_tree(torch.load(path, map_location='cpu', weights_only=True))
        checkpoints[name] = digest
    report = load(root/'closedloop.json')
    records = report['records']
    for row in records:
        if row['method'] != 'unconstrained_strongest' and row['constraint_violations']:
            raise RuntimeError('A primary controller violated its declared contract')
    out.mkdir(parents=True, exist_ok=True)
    for name in checkpoints:
        target = out/'checkpoints'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/'checkpoints'/name, target)
    files = {}
    for name in ['config.json', 'provenance.json', 'status.json', 'reader.json', 'selection.json', 'offline.json', 'closedloop.json']:
        source = root/name
        finite_tree(load(source))
        shutil.copyfile(source, out/name)
        files[name] = sha256(source)
    for path in sorted(root.glob('train-*.json')):
        shutil.copyfile(path, out/path.name)
        files[path.name] = sha256(path)
    capsule = out/'executed_sources.tar.gz'
    with capsule.open('wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as archive:
            for path in sorted((root/'source_snapshot').rglob('*')):
                if path.is_file():
                    info = archive.gettarinfo(str(path), arcname=str(path.relative_to(root/'source_snapshot')))
                    info.mtime = info.uid = info.gid = 0; info.uname = info.gname = ''
                    with path.open('rb') as data: archive.addfile(info, data)
    methods = ['stay', 'budgeted_strongest', 'unconstrained_strongest', 'numerical_planner', 'residual_mlp', 'residual_graph']
    summary = []
    for method in methods:
        rows = [r for r in records if r['method'] == method]
        summary.append({'method': method, 'runs': len(rows),
                        'mean_mission_utility': mean(r['mean_mission_utility'] for r in rows),
                        'mean_service': mean(r['metrics']['mean_service'] for r in rows),
                        'worst_decile_service': mean(r['metrics']['worst_decile_mean_service'] for r in rows),
                        'critical_violation_robot_steps': mean(r['metrics']['critical_violation_robot_steps'] for r in rows),
                        'reconnections': mean(r['metrics']['reconnections'] for r in rows),
                        'constraint_violations': sum(r['constraint_violations'] for r in rows),
                        'compute_ms_p95_mean': mean(r['compute_ms_p95'] for r in rows)})
    comparisons = {method: {metric: paired_interval(records, method, metric) for metric in
                            ['mean_mission_utility', 'critical_violation_robot_steps', 'worst_decile_mean_service']}
                   for method in ['residual_mlp', 'residual_graph']}
    snapshot = {'source': 'multi_step_warehouse_simulation', 'model_seeds': [0, 1, 2],
                'environment_count': 3, 'full_cross': True, 'horizon_s': 30, 'offsets_s': [5, 10, 15, 20, 25, 30],
                'switch_budget': 12, 'summary': summary, 'paired_comparisons': comparisons,
                'per_robot_comparisons': {method: robot_comparison(records, method) for method in
                                         ['numerical_planner', 'residual_mlp', 'residual_graph']},
                'hardware_qualified': False, 'sota_established': False}
    atomic_json(out/'snapshot.json', snapshot)
    atomic_json(out/'manifest.json', {'reports': files, 'checkpoint_sha256': checkpoints,
                'executed_source_capsule_sha256': sha256(capsule), 'run': str(root),
                'source': 'simulation', 'hardware_experiments_executed': False})
    lines = ['# Route-aware horizon results', '',
             'This separately versioned campaign forecasts six five-second outcomes across each thirty-second commitment. Inputs are past censored scans, previous delivery telemetry, public AP metadata/CAD and committed routes; actual future positions and hidden propagation remain evaluation-only.', '',
             'Training uses 18 ten-minute environments, validation six, and held-out evaluation three. Both compact architectures have three independently trained seeds; each is evaluated on all three held-out environments. The original E1–E5/E7 and warehouse v2 reports are unchanged.', '',
             'The mission changes from fleet data to worst-decile service halfway through each ten-minute rollout. Both include an explicit critical-rate-violation penalty and a hard twelve-switch budget. d3 interprets the two mission families with actual pinned weights; numeric fields are compiled from the declared controlled language. These two statements do not establish broad language understanding.', '',
             '| Controller | Runs | Mission utility ↑ | Service fraction ↑ | Worst decile ↑ | Critical-violation robot-steps ↓ | Reconnections ↓ | Budget violations | Mean p95 compute (ms) |',
             '|---|---|---|---|---|---|---|---|---|']
    for row in summary:
        lines.append(f'| {row["method"]} | {row["runs"]} | {row["mean_mission_utility"]:.4f} | {row["mean_service"]:.4f} | {row["worst_decile_service"]:.4f} | {row["critical_violation_robot_steps"]:.1f} | {row["reconnections"]:.1f} | {row["constraint_violations"]} | {row["compute_ms_p95_mean"]:.1f} |')
    lines += ['', 'The twelve-switch cap is an operator-policy scenario, not a hardware rate limit or the existing maximum concurrent transaction count. Batched execution could permit more switches across thirty seconds. The unconstrained strongest-scan arm violates this declared budget and has higher service in these scenes. Primary comparisons therefore concern budgeted missions only. Every planner uses the same candidate construction and local-search round limit, with independently validated visible targets and move counts.', '',
              '## Paired learned-minus-numerical differences', '']
    for method, metrics in comparisons.items():
        for metric, value in metrics.items():
            low, high = value['ci95']
            lines.append(f'- {method}, {metric}: {value["difference"]:.4f}, crossed-bootstrap 95% interval [{low:.4f}, {high:.4f}].')
    lines += ['', 'The bootstrap resamples model seeds and environments separately. Three environments remain a small evaluation set. Intervals do not establish physical-fleet reliability or universal benefit.', '',
              '## Individual simulated robots', '',
              'Against budgeted strongest-scan, descriptive comparisons count a change only when mean service differs by more than one percentage point. Each row is one environment and one model seed, not another independent physical fleet. Some robots can regress even when the fleet objective improves. Full counts and worst examples are in `snapshot.json`.', '',
              '## Validation-only selection', '']
    for key, recipe in selection['models'].items():
        lines.append(f'- Model {key}: residual strength {recipe["selected_strength"]}, risk temperature {recipe["temperature"]:.3f}.')
    lines += ['', 'A zero residual strength means the numerical baseline was selected; it is not a learned advantage. Raw learned forecasts and validated blends are both retained in `offline.json`. Selection was sealed before accessing test outcomes. Checkpoint epoch selection includes the zero-residual initial state as a declared candidate.', '',
              '## Limits', '',
              'All radio observations, delivery counters and actions are simulated. The NIST payload-error curves do not constitute hardware calibration or a packet-level warehouse validation. Four seconds of switch interruption is assumed and charged only across affected intervals; scan interruption and actuation failures remain unmodelled. Critical-rate violations are not application packet deadlines. Local compute timing includes belief construction and planning, but excludes physical sensing, first-use language inference and actuation.', '',
              'There is no physical-car, ANSpect-waveform, general foundation-model or SOTA claim. Exact source capsules, report hashes and finite checkpoint hashes are archived with the results.', '',
              'The separately versioned post-holdout selector and fresh-environment audit are documented in [WAREHOUSE_HORIZON_SELECTOR.md](WAREHOUSE_HORIZON_SELECTOR.md). They preserve these original outcomes. Software and actual proposal-interface execution receipts are recorded in `artifacts/warehouse_horizon/verification.json`.']
    Path('docs/WAREHOUSE_HORIZON_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(f'Archived completed horizon campaign: {len(records)} rollouts and {len(checkpoints)} verified checkpoints')


if __name__ == '__main__':
    main()
