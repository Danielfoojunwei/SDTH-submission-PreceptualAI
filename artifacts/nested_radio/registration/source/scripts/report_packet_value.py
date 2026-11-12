"""Report and independently audit packet-counterfactual JEPA evidence.

The reporter never loads a model or opens a simulator. All estimates are
recomputed from the archived native packet counters and training receipts.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/packet_value'
LABELS = {'jepa': 'Packet JEPA', 'supervised': 'Matched supervised',
          'radiomap': 'Common-controller RadioMap', 'canonical': 'Canonical RadioMap'}
PRIOR = ('contracted', 'gate_repair', 'fleet_world_model', 'fleet_confirmation',
         'world_frontier', 'frontier_confirmation', 'hardware_budget',
         'handover_admission', 'closed_loop')
SUPPLEMENT_COMMIT = '352c0938bb5903f669ce50af4ec50b9633408431'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def cost_key(value):
    return str(float(value))


def crossed(difference, protocol):
    """Resample complete sites and paired training seeds independently.

    Inputs are S x K, with K=3 paired fits. A deterministic reference is
    broadcast to those same three fitted policies before forming differences.
    No packet or robot is treated as an independent experiment.
    """
    d = np.asarray(difference, dtype=float)
    if d.ndim != 2 or not d.size or not np.isfinite(d).all():
        raise ValueError('Need finite site-by-training-seed contrasts')
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    count = protocol['bootstrap_replicates']
    site = rng.integers(d.shape[0], size=(count, d.shape[0]))
    seed = rng.integers(d.shape[1], size=(count, d.shape[1]))
    sampled = d[site[:, :, None], seed[:, None, :]].mean(axis=(1, 2))
    return {'mean_difference_ppm': float(d.mean()),
            'crossed_bootstrap_95_ppm': np.quantile(sampled, [.025, .975]).tolist(),
            'site_means_ppm': d.mean(1).tolist(), 'training_seed_means_ppm': d.mean(0).tolist(),
            'site_seed_differences_ppm': d.tolist(),
            'site_wins': int((d.mean(1) < 0).sum()), 'site_ties': int((d.mean(1) == 0).sum()),
            'sites': d.shape[0], 'paired_training_seeds': d.shape[1]}


def ci_text(contrast):
    lo, hi = contrast['crossed_bootstrap_95_ppm']
    return f'[{lo:.2f}, {hi:.2f}] ppm'


def label_diagnostic(receipts, protocol):
    groups = {}
    for split in ('training', 'validation'):
        groups[split] = {}
        for cost in protocol['handover_seconds']:
            values = np.array([r['delta_misses'] for c in receipts
                               if c['split'] == split and float(c['cost']) == float(cost)
                               for r in c['records']], dtype=float)
            groups[split][cost_key(cost)] = {
                'examples': int(len(values)), 'beneficial': int((values < 0).sum()),
                'harmful': int((values > 0).sum()), 'ties': int((values == 0).sum()),
                'minimum_delta_misses': float(values.min()) if len(values) else None,
                'maximum_delta_misses': float(values.max()) if len(values) else None,
                'standard_deviation': float(values.std()) if len(values) else None}
    gate = all(g['beneficial'] > 0 for g in groups['training'].values())
    return {'gate_passed': gate, 'by_split_and_cost': groups,
            'native_label_runs': sum(int(c['native_runs']) for c in receipts),
            'label_examples': sum(int(c['examples']) for c in receipts),
            'factual_replays': sum(int(c['factual_replays']) for c in receipts)}


def excluded_pilot():
    root = OUT/'pilot_ttl10'
    receipts = [read(f) for f in sorted((root/'labels').glob('*.json'))]
    return {'excluded': True, 'path': 'artifacts/packet_value/pilot_ttl10',
            'completed_label_cases': len(receipts),
            'recorded_native_runs': sum(c['native_runs'] for c in receipts),
            'recorded_examples': sum(c['examples'] for c in receipts),
            'reason': 'Forced fast actions could remain pending for ten seconds, while runtime proposals expire after one second',
            'other_limitations': ['Training density was confounded with drift parity',
                                  'Both fork times sampled early history only'],
            'used_for_final_training_calibration_or_confirmation': False}


def supplement_record():
    path = OUT/'acceptance_supplement.json'
    name = str(path.relative_to(ROOT))
    committed = subprocess.check_output(['git', 'show', f'{SUPPLEMENT_COMMIT}:{name}'], cwd=ROOT)
    assert committed == path.read_bytes(), 'Acceptance supplement differs from preregistered Git blob'
    subprocess.run(['git', 'merge-base', '--is-ancestor', SUPPLEMENT_COMMIT, 'HEAD'], cwd=ROOT, check=True)
    timestamp = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', SUPPLEMENT_COMMIT], cwd=ROOT))
    supplement = read(path)
    assert supplement['original_primary_unchanged'] and supplement['no_controller_or_data_change']
    assert supplement['no_test_results_accessed']
    return {'path': name, 'sha256': digest(path), 'preregistration_commit': SUPPLEMENT_COMMIT,
            'commit_time_unix': timestamp, 'four_original_primary_comparisons_unchanged': True,
            'additional_comparisons': 'JEPA minus canonical RadioMap at both costs',
            'canonical_included_in_tail_veto': True}


def row_metrics(case):
    result = case['result']; packet = result['packet_detail']
    misses = np.asarray(packet['misses'], dtype=float)
    sent = np.asarray(packet['sent'], dtype=float)
    maxima = np.asarray(packet['max_gap'], dtype=float)*.01
    fractions = misses/np.maximum(sent, 1)
    worst = np.sort(fractions)[-max(1, len(fractions)//10):].mean()
    cumulative = packet['cumulative']
    return {'misses': int(misses.sum()), 'commands': int(sent.sum()),
            'ppm': float(misses.sum()/max(sent.sum(), 1)*1e6),
            'worst_decile_fraction': float(worst),
            'maximum_silence_seconds': float(maxima.max()),
            'p99_robot_maximum_silence_seconds': float(np.quantile(maxima, .99)),
            'bulk_mbps': float(result['bulk_mbps']), 'video_on_time': float(result['video_on_time']),
            'safety_stop_fraction': float(result['safety_stop_fraction']), 'roams': int(result['roams']),
            'seconds': [int(c['slot_end'])/100 for c in cumulative],
            'cumulative_misses': [int(c['misses']) for c in cumulative],
            'cumulative_sent': [int(c['sent']) for c in cumulative]}


def family_arrays(cases, family, cost, protocol, *, sites=None):
    sites = protocol['test_sites'] if sites is None else sites
    seeds = protocol['training_seeds'] if family in protocol['families'] else [None]
    rows = [[next(c for c in cases if c['scene'] == scene and c['family'] == family
                  and float(c['cost']) == float(cost) and c['model_seed'] == seed)
             for seed in seeds] for scene in sites]
    metrics = [[row_metrics(c) for c in row] for row in rows]
    return rows, metrics


def summarise_arm(metrics, seeds, protocol):
    flattened = [v for row in metrics for v in row]
    matrix = np.array([[v['ppm'] for v in row] for row in metrics])
    misses = np.asarray([[v['cumulative_misses'] for v in row] for row in metrics], dtype=float)
    sent = np.asarray([[v['cumulative_sent'] for v in row] for row in metrics], dtype=float)
    out = {'mean_ppm': float(matrix.mean()), 'mean_percent': float(matrix.mean()/10000),
           'pooled_misses_across_evaluated_policies': sum(v['misses'] for v in flattened),
           'pooled_commands_across_evaluated_policies': sum(v['commands'] for v in flattened),
           'site_seed_ppm': matrix.tolist(), 'mean_worst_decile_fraction': float(np.mean([v['worst_decile_fraction'] for v in flattened])),
           'observed_global_maximum_silence_seconds': max(v['maximum_silence_seconds'] for v in flattened),
           'mean_p99_robot_maximum_silence_seconds': float(np.mean([v['p99_robot_maximum_silence_seconds'] for v in flattened])),
           'mean_bulk_mbps': float(np.mean([v['bulk_mbps'] for v in flattened])),
           'mean_video_on_time': float(np.mean([v['video_on_time'] for v in flattened])),
           'mean_safety_stop_fraction': float(np.mean([v['safety_stop_fraction'] for v in flattened])),
           'mean_roams': float(np.mean([v['roams'] for v in flattened])),
           'mean_cumulative_misses_per_fleet': misses.mean(axis=(0, 1)).tolist(),
           'mean_cumulative_commands_per_fleet': sent.mean(axis=(0, 1)).tolist(),
           'per_training_seed': {str(seed): {'mean_ppm': float(matrix[:, j].mean()),
                                           'site_ppm': matrix[:, j].tolist(),
                                           'mean_worst_decile_fraction': float(np.mean([row[j]['worst_decile_fraction'] for row in metrics])),
                                           'maximum_silence_seconds': max(row[j]['maximum_silence_seconds'] for row in metrics)}
                                 for j, seed in enumerate(seeds)}}
    return out, matrix, misses, sent


def describe_development(cases, protocol, diagnostic, completion):
    """Negative development evidence; never compute confirmation intervals."""
    conditions = {}; duration = protocol['development_duration_seconds']
    for cost in protocol['handover_seconds']:
        arms = {}; matrices = {}; curves = {}; counts = {}
        for family in LABELS:
            rows, metrics = family_arrays(cases, family, cost, protocol, sites=protocol['development_sites'])
            seeds = protocol['training_seeds'] if family in protocol['families'] else [None]
            arms[family], matrices[family], curves[family], counts[family] = summarise_arm(metrics, seeds, protocol)
            if family != 'canonical':
                controllers = [c['controller'] for row in rows for c in row]
                committed = sum(len(j['committed_robots']) for c in controllers for j in c['decisions'])
                eligible = sum(sum(j['eligible']) for c in controllers for j in c['decisions'])
                arms[family]['controller_diagnostic'] = {
                    'committed_robot_decisions': committed, 'eligible_switch_proposals': eligible,
                    'explicit_stays': committed-eligible,
                    'executed_switches': sum(len(c['executions']) for c in controllers),
                    'validation_corrections_commands': sorted(set(c['correction_commands'] for c in controllers))}
        contrasts = {}; trajectories = {}
        for reference in ('radiomap', 'supervised', 'canonical'):
            delta = matrices['jepa']-matrices[reference]
            contrasts[reference] = {'mean_difference_ppm': float(delta.mean()),
                                    'site_seed_differences_ppm': delta.tolist(),
                                    'training_seed_means_ppm': delta.mean(0).tolist(),
                                    'descriptive_development_only': True}
            excess = curves['jepa']-curves[reference]
            start = (2*duration)//3
            j_miss = curves['jepa'][:, :, -1]-curves['jepa'][:, :, start-1]
            r_miss = curves[reference][:, :, -1]-curves[reference][:, :, start-1]
            j_sent = counts['jepa'][:, :, -1]-counts['jepa'][:, :, start-1]
            r_sent = counts[reference][:, :, -1]-counts[reference][:, :, start-1]
            trajectories[reference] = {
                'mean_cumulative_excess_misses_per_fleet': excess.mean(axis=(0, 1)).tolist(),
                'maximum_observed_positive_excess_misses': float(np.maximum(excess, 0).max()),
                'final_third_mean_excess_ppm': float(((j_miss/np.maximum(j_sent, 1)-r_miss/np.maximum(r_sent, 1))*1e6).mean()),
                'checkpoints': {str(t): {'mean_excess_misses': float(excess[:, :, t-1].mean()),
                                       'site_seed_excess_misses': excess[:, :, t-1].tolist()}
                                for t in protocol['cumulative_metrics_seconds'] if t <= duration},
                'descriptive_development_only': True}
        conditions[cost_key(cost)] = {'arms': arms, 'jepa_contrasts': contrasts, 'cumulative_excess': trajectories}
    return {'completed': True, 'status': 'development_rejected', 'simulation_only': True,
            'independent_training_fits': 6, 'development_sites': len(protocol['development_sites']),
            'fresh_sites': 0, 'duration_seconds': duration, 'development_runs': len(cases), 'test_runs': 0,
            'native_label_runs': diagnostic['native_label_runs'], 'diagnostic': diagnostic,
            'conditions': conditions, 'primary_evaluated': False, 'primary_success': None,
            'primary_comparisons': {}, 'additional_promotion_evaluated': False,
            'additional_promotion_success': None, 'additional_promotion_comparisons': {},
            'promotion_vetoes': [], 'research_promotion_criterion_passed': False,
            'physical_deployment_qualified': False, 'universal_or_zero_loss_guarantee': False,
            'original_study_disposition': 'Rejected on development before confirmation',
            'rejection_reason': completion.get('reason', 'Controller rejected after development degradation; untouched confirmation cohort not accessed')}


def aggregate(cases, protocol, diagnostic):
    conditions = {}; primary = {}; additional = {}; vetoes = []
    duration = protocol['confirmation_duration_seconds']
    for cost in protocol['handover_seconds']:
        arms = {}; arrays = {}; curves = {}; sent_curves = {}
        for family in LABELS:
            _, metrics = family_arrays(cases, family, cost, protocol)
            seeds = protocol['training_seeds'] if family in protocol['families'] else [None]
            arms[family], arrays[family], curves[family], sent_curves[family] = summarise_arm(metrics, seeds, protocol)
        contrasts = {}; trajectories = {}
        for reference in ('radiomap', 'supervised', 'canonical'):
            delta = arrays['jepa']-arrays[reference]
            contrast = crossed(delta, protocol)
            contrasts[reference] = contrast
            excess = curves['jepa']-curves[reference]
            start = (2*duration)//3
            j_miss = curves['jepa'][:, :, -1]-curves['jepa'][:, :, start-1]
            r_miss = curves[reference][:, :, -1]-curves[reference][:, :, start-1]
            j_sent = sent_curves['jepa'][:, :, -1]-sent_curves['jepa'][:, :, start-1]
            r_sent = sent_curves[reference][:, :, -1]-sent_curves[reference][:, :, start-1]
            final_third = crossed((j_miss/np.maximum(j_sent, 1)-r_miss/np.maximum(r_sent, 1))*1e6, protocol)
            trajectory = {'mean_cumulative_excess_misses_per_fleet': excess.mean(axis=(0, 1)).tolist(),
                          'maximum_observed_positive_excess_misses': float(np.maximum(excess, 0).max()),
                          'mean_of_site_seed_maximum_positive_excess_misses': float(np.maximum(excess, 0).max(2).mean()),
                          'site_seed_maximum_positive_excess_misses': np.maximum(excess, 0).max(2).tolist(),
                          'final_third_excess_rate': final_third,
                          'checkpoints': {str(t): {'mean_excess_misses': float(excess[:, :, t-1].mean()),
                                                 'site_seed_excess_misses': excess[:, :, t-1].tolist()}
                                          for t in protocol['cumulative_metrics_seconds']}}
            trajectories[reference] = trajectory
            key = f'{cost_key(cost)}:jepa_minus_{reference}'
            is_primary = reference in ('radiomap', 'supervised')
            if is_primary:
                primary[key] = contrast
            else:
                additional[key] = contrast
            lo, _ = contrast['crossed_bootstrap_95_ppm']
            if lo > 0:
                reason = 'Harmful primary interval excludes zero' if is_primary else 'Harmful additional promotion interval excludes zero'
                vetoes.append({'cost': cost, 'reference': reference, 'reason': reason})
            for metric in ('observed_global_maximum_silence_seconds', 'mean_worst_decile_fraction'):
                if arms['jepa'][metric] > arms[reference][metric]+1e-12:
                    vetoes.append({'cost': cost, 'reference': reference, 'reason': 'Observed tail service worsens',
                                   'metric': metric, 'jepa': arms['jepa'][metric], 'reference_value': arms[reference][metric]})
        conditions[cost_key(cost)] = {'arms': arms, 'jepa_contrasts': contrasts, 'cumulative_excess': trajectories}
    success = all(c['crossed_bootstrap_95_ppm'][1] < 0 for c in primary.values())
    supplementary_success = all(c['crossed_bootstrap_95_ppm'][1] < 0 for c in additional.values())
    return {'completed': True, 'status': 'evaluated', 'simulation_only': True,
            'independent_training_fits': len(protocol['families'])*len(protocol['training_seeds']),
            'fresh_sites': len(protocol['test_sites']), 'duration_seconds': duration,
            'development_runs': len(protocol['development_sites'])*len(protocol['handover_seconds'])*8,
            'test_runs': len(cases), 'native_label_runs': diagnostic['native_label_runs'],
            'diagnostic': diagnostic, 'conditions': conditions, 'primary_comparisons': primary,
            'primary_evaluated': True, 'primary_success': success,
            'additional_promotion_evaluated': True, 'additional_promotion_comparisons': additional,
            'additional_promotion_success': supplementary_success, 'promotion_vetoes': vetoes,
            'research_promotion_criterion_passed': bool(success and supplementary_success and not vetoes),
            'physical_deployment_qualified': False, 'universal_or_zero_loss_guarantee': False}


def figures(summary):
    dest = OUT/'figures'; dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    d = summary['diagnostic']['by_split_and_cost']
    for ax, (cost, data) in zip(axes, d['training'].items()):
        ax.bar(['Beneficial', 'Tie', 'Harmful'], [data['beneficial'], data['ties'], data['harmful']],
               color=['#269b82', '#a7b3c2', '#c36458'])
        ax.set_title(f'Native training labels · {float(cost):g} s interruption')
        ax.set_ylabel('Observed stay/switch counterfactuals')
    fig.tight_layout(); fig.savefig(dest/'packet_label_balance.png', dpi=180); plt.close(fig)
    if summary['status'] not in ('evaluated', 'development_rejected'):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    colors = {'jepa': '#239b8c', 'supervised': '#9576b4', 'radiomap': '#3e79ad', 'canonical': '#c8a16b'}
    for ax, (cost, group) in zip(axes, summary['conditions'].items()):
        for j, (family, arm) in enumerate(group['arms'].items()):
            ax.bar(j, arm['mean_percent'], color=colors[family])
            values = np.asarray(arm['site_seed_ppm']).ravel()/10000
            ax.scatter(j+np.linspace(-.12, .12, len(values)), values, color='#1c3545', s=7, alpha=.55)
        ax.set_xticks(range(4), [LABELS[f] for f in group['arms']], rotation=15, ha='right')
        ax.set_ylabel('Commands missing deadline (%)'); ax.set_title(f'{float(cost):g} s assumed handover')
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    scope = 'development only' if summary['status'] == 'development_rejected' else 'independent confirmation'
    fig.suptitle(f'100 simulated robots · {summary["duration_seconds"]} seconds · {scope}')
    filename = 'packet_development.png' if summary['status'] == 'development_rejected' else 'packet_confirmation.png'
    fig.tight_layout(); fig.savefig(dest/filename, dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, (cost, group) in zip(axes, summary['conditions'].items()):
        for reference in ('radiomap', 'supervised', 'canonical'):
            y = group['cumulative_excess'][reference]['mean_cumulative_excess_misses_per_fleet']
            ax.plot(np.arange(1, len(y)+1), y, label='JEPA − '+LABELS[reference], color=colors[reference])
        ax.axhline(0, color='#666', lw=.8); ax.set_title(f'{float(cost):g} s assumed handover')
        ax.set_xlabel('Simulated seconds'); ax.set_ylabel('Mean cumulative excess misses / fleet')
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(dest/'cumulative_excess.png', dpi=180); plt.close(fig)


def result_document(summary, protocol):
    lines = ['# Packet-action JEPA: native counterfactual evidence', '',
             'This study tests an actual JEPA auxiliary objective against an otherwise matched supervised model and RadioMap. '
             'Native packet-state forks label one focal robot staying or switching once and committing for ten seconds. '
             'All methods share the same observable controller, scan mechanism and physical interruption costs.', '',
             '![Native counterfactual label balance](../artifacts/packet_value/figures/packet_label_balance.png)', '']
    diagnostic = summary['diagnostic']
    lines += [f'The label corpus contains {diagnostic["label_examples"]:,} observed action differences from '
              f'{diagnostic["native_label_runs"]:,} native packet runs, including {diagnostic["factual_replays"]:,} factual suffix replays. '
              'Label counts are reported separately for training and naturally sampled validation.', '',
              '| Split | Handover seconds | Beneficial | Tie | Harmful |', '|---|---:|---:|---:|---:|']
    for split, costs in diagnostic['by_split_and_cost'].items():
        for cost, values in costs.items():
            lines.append(f'| {split} | {float(cost):g} | {values["beneficial"]} | {values["ties"]} | {values["harmful"]} |')
    lines += ['']
    pilot = summary['excluded_pilot']
    lines += [f'An excluded pre-fitting pilot is preserved separately: **{pilot["completed_label_cases"]} completed label cases, '
              f'{pilot["recorded_native_runs"]:,} recorded native runs and {pilot["recorded_examples"]:,} action examples**. '
              'Its forced fast actions could remain pending ten seconds, while runtime proposals expire after one second. '
              'Its alternating robot counts also confounded density with drift, and both fork times used early history. '
              'These records are excluded from the final label, calibration, fit and confirmation counts. '
              'Before fitting or development/holdout access, an explicit protocol amendment aligned action expiry, balanced '
              'density across conditions and added a mature-history fork. Original pilot sources/data remain untouched.', '']
    if summary['status'] == 'diagnostic_failed':
        lines += ['The prespecified diagnostic gate failed: at least one physical cost has no beneficial training action. '
                  'Training and confirmation stopped. No JEPA superiority, decision discrimination or all-stay-controller success is claimed.', '']
    elif summary['status'] == 'development_rejected':
        lines += ['## Rejected on development; confirmation not run', '',
                  'The label diagnostic passed and all six matched fits completed. The frozen controller was then rejected '
                  f'after all {summary["development_runs"]} development runs on {summary["development_sites"]} sites, '
                  f'100 robots each and {summary["duration_seconds"]} seconds per run. '
                  '**The original four primary comparisons and two additional promotion comparisons were NOT EVALUATED.** '
                  'No confirmation confidence intervals are reported, no test source capsule exists and no holdout cases ran. '
                  'This is a development rejection, not a failed independent confirmation.', '',
                  '![Descriptive development results](../artifacts/packet_value/figures/packet_development.png)', '',
                  str(summary['rejection_reason']), '']
        for cost, group in summary['conditions'].items():
            lines += [f'### Development: {float(cost):g} second handover', '',
                      '| Method | Loss % | Mean roams / fleet | Worst-decile loss % | Max silence seconds |',
                      '|---|---:|---:|---:|---:|']
            for family, arm in group['arms'].items():
                lines.append(f'| {LABELS[family]} | {arm["mean_percent"]:.3f} | {arm["mean_roams"]:.2f} | '
                             f'{100*arm["mean_worst_decile_fraction"]:.3f} | {arm["observed_global_maximum_silence_seconds"]:.2f} |')
            lines += ['']
            for family in ('jepa', 'supervised'):
                seed_rates = ', '.join(f'{seed}: {stats["mean_ppm"]:.2f} ppm' for seed, stats in group['arms'][family]['per_training_seed'].items())
                lines += [f'{LABELS[family]}, all fitted seeds: {seed_rates}.', '']
            for family in ('jepa', 'supervised', 'radiomap'):
                d = group['arms'][family]['controller_diagnostic']
                lines += [f'{LABELS[family]}: {d["eligible_switch_proposals"]:,} eligible switch proposals among '
                          f'{d["committed_robot_decisions"]:,} committed robot decisions; '
                          f'{d["executed_switches"]:,} executed handovers. Validation corrections: '
                          +', '.join(f'{v:.2f}' for v in d['validation_corrections_commands'])+' commands.', '']
        lines += ['![Development cumulative excess](../artifacts/packet_value/figures/cumulative_excess.png)', '',
                  'The failure exposes a control-interface problem: when empirical uncertainty corrections or the '
                  'immediate-link guard reject a proposal, the new controller commits to staying and suppresses the '
                  'canonical roaming policy. Its conservative rejection does not restore canonical recovery. '
                  'Large missed-opportunity costs can therefore accumulate despite few executed switches. '
                  'The same issue affects the numerical common-controller arm; development degradation cannot be '
                  'attributed to JEPA alone. This mechanism is consistent with the observed counters; its individual '
                  'causal contributions need a separate ablation.', '',
                  'The original controller, fits, calibration, sources and negative development trajectories remain '
                  'unchanged evidence. A baseline-preserving rescue requires a separately versioned controller and '
                  'acceptance record, development-only selection and a new pre-holdout seal. '
                  'The unaccessed confirmation seeds are not successful tests of this rejected profile.', '']
    else:
        lines += [f'Six independent fits completed: three JEPA and three matched supervised initialisations. '
                  f'{summary["development_runs"]} development runs precede {summary["test_runs"]} confirmation runs on '
                  f'{summary["fresh_sites"]} untouched sites, 100 robots each, {summary["duration_seconds"]} seconds per run. '
                  'Every fitted seed is retained; repeated seeds are not additional independent sites.', '',
                  '![Packet confirmation](../artifacts/packet_value/figures/packet_confirmation.png)', '']
        for cost, group in summary['conditions'].items():
            lines += [f'## Assumed handover: {float(cost):g} seconds', '',
                      '| Method | Misses / million | Worst-decile loss | Max silence seconds | Bulk Mbps |',
                      '|---|---:|---:|---:|---:|']
            for family, arm in group['arms'].items():
                lines.append(f'| {LABELS[family]} | {arm["mean_ppm"]:.2f} | {100*arm["mean_worst_decile_fraction"]:.3f}% | '
                             f'{arm["observed_global_maximum_silence_seconds"]:.2f} | {arm["mean_bulk_mbps"]:.2f} |')
            lines += ['']
            for reference in ('radiomap', 'supervised', 'canonical'):
                c = group['jepa_contrasts'][reference]
                role = 'additional promotion comparison' if reference == 'canonical' else 'original primary comparison'
                lines += [f'JEPA minus {LABELS[reference]} ({role}): **{c["mean_difference_ppm"]:.2f} ppm**, '
                          f'crossed site/training-seed 95% interval **{ci_text(c)}**.', '']
        lines += ['![Cumulative excess misses](../artifacts/packet_value/figures/cumulative_excess.png)', '',
                  'The four prespecified primary comparisons '+('all pass.' if summary['primary_success'] else 'do not all pass.'), '',
                  'The two additional canonical-RadioMap promotion comparisons '
                  +('both pass.' if summary['additional_promotion_success'] else 'do not both pass.'), '',
                  f'The observed tail-service veto records **{len(summary["promotion_vetoes"])}** violations. '
                  'Research promotion requires all four original primary comparisons, both additional canonical-RadioMap '
                  'comparisons and no tail veto against any of the three references. '
                  f'That combined criterion **{"passes" if summary["research_promotion_criterion_passed"] else "does not pass"}**. '
                  'Physical deployment remains unqualified.', '',
                  'All seed-specific rates, cumulative checkpoints at 60/120/180/240/300 seconds, maximum positive cumulative excess, '
                  'final-third excess rates, and p99 of robot maximum silence are in `summary.json`. '
                  'The p99 statistic describes per-robot maximum silence, not a per-packet latency distribution.', '']
    lines += ['The stricter canonical-RadioMap criterion was committed before any fitting, development or holdout access '
              'in `acceptance_supplement.json` (Git commit `352c093`). It adds claim requirements without changing '
              'the controller, data or four original primary comparisons. Beating only the new shared-controller '
              'RadioMap cannot support a broad better-than-RadioMap claim.', '']
    if summary['status']=='development_rejected' and (OUT/'diagnosis.json').exists():
        diagnosis=read(OUT/'diagnosis.json')
        lines += ['## Validation diagnosis', '',
                  'This is a post-hoc explanation of development failure, not a newly selected controller or confirmation result.', '',
                  '| Handover seconds | Numeric MAE, commands | JEPA mean MAE | Supervised mean MAE |',
                  '|---|---:|---:|---:|']
        for cost,rows in diagnosis['conditions'].items():
            j=np.mean([v['validation_mae_commands'] for k,v in rows.items() if k.startswith('jepa-')])
            s=np.mean([v['validation_mae_commands'] for k,v in rows.items() if k.startswith('supervised-')])
            lines.append(f'| {float(cost):g} | {rows["radiomap"]["validation_mae_commands"]:.2f} | {j:.2f} | {s:.2f} |')
        row=diagnosis['conditions']['0.03']['jepa-0']
        lines += ['',f'For JEPA seed zero at 30 ms, raw values and the immediate-link check admit {row["raw_feasible_admissions"]} validation actions '
                  f'({row["raw_admission_beneficial"]} beneficial, {row["raw_admission_neutral"]} neutral, {row["raw_admission_harmful"]} harmful). '
                  f'The unconditional +{row["correction_commands"]:.2f}-command correction leaves {row["corrected_feasible_admissions"]}. '
                  'This demonstrates suppression on those validation examples. It does not validate removing the correction or overriding canonical control.', '',
                  'The separate `ShadowPacketAdmission` interface returns the baseline mask unchanged and uses a bounded background worker. '
                  'The qualification guard refuses this development-rejected profile. Neither safeguard activates learned switching or establishes a learned benefit; '
                  'physical runtime deadlines remain unqualified. See `PACKET_VALUE_SAFETY.md`.', '']
    lines += ['## Interpretation and physical limits', '',
              'The one-radio, 90 ms scan and 50 ms command-deadline assumptions impose about one forced missed command per '
              'isolated scan. At five-second scanning this is approximately a 1% loss floor. Four-second handovers force '
              '79–80 command misses when a complete interruption lies inside the evaluation interval. Overlapping scan and '
              'handover outages must not be counted twice. Raw absolute losses remain visible; this report subtracts no fixed floor.', '',
              'Cumulative excess is a matched simulation comparison, not an online-observable counterfactual after policies diverge. '
              'A reservation ledger can account for modeled risk and known interruption exposure; it cannot undo earlier losses '
              'or guarantee realized regret. Ten-second commitment and immediate-link constraints are common to the numerical '
              'and neural methods, so their benefits alone do not establish JEPA necessity.', '',
              'JEPA and supervised heads share the same inference graph and RadioMap route/physics assistance. '
              'The JEPA auxiliary target uses future observable local context only during training. Validation-only empirical '
              'corrections are heuristics, not proven conditional risk bounds. Inference and action-head queues are charged '
              'before proposals become visible. All physical scan/handover times remain assumptions.', '',
              'These are native Python packet simulations using archived ns-3 PHY curves. The CAD family, precomputed motion, '
              'shared warm start, out-of-band pose/routes, reserved control-frame telemetry and modeled channels limit transfer. '
              'The corrected training corpus crosses stationary/drifting conditions with 24/100-robot densities and samples '
              'early and mature histories; it still covers only those two densities and the existing CAD family. '
              'No arbitrary-robot, protocol-agnostic, hardware, universal-JEPA, certified-risk or zero-loss claim follows.', '',
              'The frozen design is in `PACKET_VALUE_DESIGN.md`; protocol, checkpoints, labels, per-robot trajectories, source '
              'capsules and weight identities are under `artifacts/packet_value/`. Earlier evidence trees are unchanged.', '']
    return '\n'.join(lines)


def archive(summary):
    write(OUT/'summary.json', summary)
    figures(summary)
    protocol = read(OUT/'protocol.json')
    (ROOT/'docs/PACKET_VALUE_RESULTS.md').write_text(result_document(summary, protocol))
    source = OUT/'analysis_source'; source.mkdir(exist_ok=True)
    shutil.copyfile(__file__, source/'report_packet_value.py')
    for name in ('diagnose_packet_value.py','review_packet_development.py','run_packet_value.sh'):
        path=ROOT/'scripts'/name
        if path.exists():shutil.copyfile(path,source/name)
    manifest = {'files_sha256': {str(f.relative_to(OUT)): digest(f) for f in sorted(OUT.rglob('*'))
                                if f.is_file() and f.name != 'manifest.json'},
                'previous_manifests': {str((ROOT/'artifacts'/name/'manifest.json').relative_to(ROOT)):
                                       digest(ROOT/'artifacts'/name/'manifest.json') for name in PRIOR},
                'document_sha256': digest(ROOT/'docs/PACKET_VALUE_RESULTS.md'),
                'design_sha256': digest(ROOT/'docs/PACKET_VALUE_DESIGN.md'),
                'checkpoint_sha256': read(OUT/'selection.json')['checkpoint_sha256'] if summary['status'] != 'diagnostic_failed' else {},
                'acceptance_supplement': supplement_record(),
                'simulation_only': True}
    write(OUT/'manifest.json', manifest)


def report():
    protocol = read(OUT/'protocol.json'); completion = read(OUT/'completion.json')
    receipts = [read(f) for f in sorted((OUT/'labels').glob('*.json'))]
    diagnostic = label_diagnostic(receipts, protocol)
    if not diagnostic['gate_passed']:
        assert completion.get('diagnostic_failed') or completion.get('status') == 'diagnostic_failed'
        summary = {'completed': True, 'status': 'diagnostic_failed', 'diagnostic': diagnostic,
                   'primary_success': False, 'additional_promotion_success': False,
                   'research_promotion_criterion_passed': False,
                   'physical_deployment_qualified': False, 'simulation_only': True}
    elif completion.get('development_rejected') or completion.get('status') == 'development_rejected':
        assert completion.get('completed')
        assert not list((OUT/'cases').glob('test-*.json')) and not (OUT/'test_source.json').exists()
        cases = [read(f) for f in sorted((OUT/'cases').glob('development-*.json'))]
        summary = describe_development(cases, protocol, diagnostic, completion)
    else:
        assert completion.get('completed')
        cases = [read(f) for f in sorted((OUT/'cases').glob('test-*.json'))]
        summary = aggregate(cases, protocol, diagnostic)
    summary['excluded_pilot'] = excluded_pilot()
    archive(summary)
    print(json.dumps({'completed': True, 'status': summary['status'], 'primary_success': summary['primary_success'],
                      'research_promotion_criterion_passed': summary['research_promotion_criterion_passed']}, indent=2))


def audit_labels(protocol, receipts, receipt, *, root=None, require_complete=True):
    root = OUT if root is None else root
    expected = {(split, scene, float(cost)) for split in ('training', 'validation')
                for scene in protocol[split+'_sites'] for cost in protocol['handover_seconds']}
    actual = {(c['split'], c['scene'], float(c['cost'])) for c in receipts}
    assert (actual == expected if require_complete else actual <= expected) and len(actual) == len(receipts)
    for c in receipts:
        assert c['completed'] and c['source_sha256'] == receipt['source_sha256']
        data = root/'data'/f'{c["split"]}-{c["scene"]}-{c["cost"]}.npz'
        assert digest(data) == c['data_sha256']
        assert c['examples'] == len(c['records'])
        assert c['factual_replays'] == len(protocol['fork_seconds'])
        stay_runs = len({(r['fork_slot'], r['robot']) for r in c['records']})
        assert c['native_runs'] == 1+len(c['records'])+stay_runs+c['factual_replays']
        with np.load(data) as values:
            if c['examples'] == 0:
                assert not values.files
                continue
            assert len(values['context']) == c['examples']
            assert values['context'].shape[1:] == (6, 32) and values['query'].shape[1:] == (24,)
            for index, record in enumerate(c['records']):
                assert record['fork_slot'] in {100*t for t in protocol['fork_seconds']}
                assert record['current'] != record['candidate']
                assert record['stay_misses'] >= 0 and record['move_misses'] >= 0
                assert record['delta_misses'] == record['move_misses']-record['stay_misses']
                assert np.isclose(record['delta_gap_seconds'], .01*(record['move_window_max_gap']-record['stay_window_max_gap']))
                assert record['delta_misses'] == float(values['delta_misses'][index])
                assert np.isclose(record['delta_gap_seconds'], values['delta_gap'][index])
                assert float(values['cost'][index]) == float(c['cost'])
                assert int(values['scene'][index]) == c['scene']
                slots = record['future_slots']
                assert slots and all(record['fork_slot'] <= s < record['fork_slot']+100*protocol['label_horizon_seconds'] for s in slots)
                assert len(set(slots)) == len(slots) and slots == sorted(slots)


def audit_diagnostic_record(diagnostic, reconstructed):
    assert diagnostic['completed'] and diagnostic['no_all_stay_success_claim']
    assert diagnostic['passed'] == reconstructed['gate_passed']
    assert diagnostic['native_runs'] == reconstructed['native_label_runs']
    for split, costs in reconstructed['by_split_and_cost'].items():
        for cost, values in costs.items():
            observed = diagnostic['groups'][split][cost]
            for key in ('examples', 'beneficial', 'harmful'):
                assert observed[key] == values[key]
            assert observed['zero'] == values['ties']
            assert observed['minimum_delta'] == values['minimum_delta_misses']
            assert observed['maximum_delta'] == values['maximum_delta_misses']


def audit_case(case, protocol, receipt, checkpoint_hashes, latency_hash, calibration_hash):
    assert case['simulation_only'] and case['source_sha256'] == receipt['source_sha256']
    assert case['checkpoint_sha256'] == checkpoint_hashes
    assert case['latency_sha256'] == latency_hash and case['calibration_sha256'] == calibration_hash
    duration = protocol['confirmation_duration_seconds'] if case['stage'] == 'test' else protocol['development_duration_seconds']
    n = protocol['confirmation_robots']; result = case['result']; packet = result['packet_detail']
    misses = np.asarray(packet['misses']); sent = np.asarray(packet['sent']); maximum = np.asarray(packet['max_gap'])
    assert misses.shape == sent.shape == maximum.shape == (n,)
    assert (misses >= 0).all() and (misses <= sent).all()
    assert int(misses.sum()) == result['control_misses']
    assert int(sent.sum()) == result['control_sent'] == n*duration*20
    assert result['control_miss'] == result['control_misses']/result['control_sent']
    assert sum(result['miss_causes'].values()) == result['control_misses']
    previous_miss = previous_sent = previous_max = 0
    cumulative = packet['cumulative']; assert len(cumulative) == duration
    for second, row in enumerate(cumulative, 1):
        assert row['slot_end'] == 100*second
        cm, cs, cg, cx = (row[k] for k in ('misses', 'sent', 'gap', 'max_gap'))
        assert cm >= previous_miss and cs >= previous_sent and cx >= previous_max
        assert cm <= cs and 0 <= cg <= cx
        previous_miss, previous_sent, previous_max = cm, cs, cx
    assert previous_miss == int(misses.sum()) and previous_sent == int(sent.sum())
    assert previous_max == int(maximum.max())
    assert packet['cumulative_scope'] == 'Fleet totals; final per-robot counters remain explicit'
    telemetry = OUT/case['telemetry_file']
    assert digest(telemetry) == case['telemetry_sha256']
    events = json.loads(gzip.decompress(telemetry.read_bytes()))
    for event in events:
        assert event['measurement_slot'] < event['complete_slot'] <= event['delivery_slot'] < 100*event['first_usable_decision']
        assert event['first_usable_decision'] == event['delivery_slot']//100+1
    if case['family'] != 'canonical':
        controller = case['controller']
        assert controller['not_factual_guarantee']
        executions = controller['executions']; keys = [(e['slot'], e['robot'], e['target']) for e in executions]
        assert len(keys) == len(set(keys))
        assert len(executions) == result['roams']
        assert np.isclose(controller['known_outage_seconds'], sum(e['nominal_outage_seconds'] for e in executions))
        assert np.isclose(controller['nominal_command_equivalent_exposure'], controller['known_outage_seconds']*20)
        booked = [e['booked_predicted_delta_misses'] for e in executions if e['booked_predicted_delta_misses'] is not None]
        assert all(v < -1 for v in booked)
        assert np.isclose(controller['booked_predicted_delta_misses'], sum(booked))
        last = {}
        for event in executions:
            assert event['slot'] >= last.get(event['robot'], -1000)+1000
            last[event['robot']] = event['slot']
            assert event['decision_id'] is not None and event['booked_predicted_delta_misses'] is not None
            job = controller['decisions'][event['decision_id']]
            assert event['slot'] >= job['ready_slot']
            assert job['eligible'][event['robot']] and job['target'][event['robot']] == event['target']
        ready = 0.
        committed_until = {}
        for job in controller['decisions']:
            ready = max(job['issued_slot']/100, ready)+job['charged_head_seconds']
            assert job['ready_slot'] == int(np.ceil(ready*100-1e-10))
            for robot in job['committed_robots']:
                assert job['issued_slot'] >= committed_until.get(robot, 0)
                committed_until[robot] = job['ready_slot']+1000
        assert controller['jobs_issued'] == len(controller['decisions'])
        assert controller['jobs_published']+controller['jobs_pending'] == controller['jobs_issued']


def audit():
    protocol = read(OUT/'protocol.json'); manifest = read(OUT/'manifest.json'); summary = read(OUT/'summary.json')
    for name, expected in manifest['files_sha256'].items(): assert digest(OUT/name) == expected, name
    for name, expected in manifest['previous_manifests'].items(): assert digest(ROOT/name) == expected, name
    assert digest(ROOT/'docs/PACKET_VALUE_RESULTS.md') == manifest['document_sha256']
    assert digest(ROOT/'docs/PACKET_VALUE_DESIGN.md') == manifest['design_sha256']
    assert manifest['acceptance_supplement'] == supplement_record()
    if (OUT/'diagnosis.json').exists():
        diagnosis=read(OUT/'diagnosis.json')
        assert diagnosis['posthoc_validation_diagnostic'] and not diagnosis['new_controller_selected']
        assert diagnosis['calibration_sha256']==digest(OUT/'calibration.json')
    pilot_root = OUT/'pilot_ttl10'; pilot = read(pilot_root/'data_source.json')
    for name, expected in pilot['source_sha256'].items():
        # Compare historical copies only: these were superseded before fitting.
        assert digest(pilot_root/'data_source'/name) == expected, name
    pilot_receipts = [read(f) for f in sorted((pilot_root/'labels').glob('*.json'))]
    audit_labels(read(pilot_root/'protocol.json'), pilot_receipts, pilot, root=pilot_root, require_complete=False)
    assert summary['excluded_pilot'] == excluded_pilot()
    assert summary['excluded_pilot']['completed_label_cases'] == 30
    assert summary['excluded_pilot']['recorded_native_runs'] == 1021
    assert not list((pilot_root/'fits').glob('*.json')) and not list((pilot_root/'cases').glob('test-*.json'))
    stages = ['data'] if summary['status'] == 'diagnostic_failed' else ['data', 'training', 'development']
    if summary['status'] == 'evaluated':
        stages.append('test')
    snapshots = {}
    for stage in stages:
        receipt = snapshots[stage] = read(OUT/(stage+'_source.json'))
        for name, expected in receipt['source_sha256'].items():
            assert digest(OUT/(stage+'_source')/name) == expected, name
            assert digest(ROOT/name) == expected, name
    receipts = [read(f) for f in sorted((OUT/'labels').glob('*.json'))]
    audit_labels(protocol, receipts, snapshots['data'])
    recomputed_diagnostic = label_diagnostic(receipts, protocol)
    assert recomputed_diagnostic == summary['diagnostic']
    audit_diagnostic_record(read(OUT/'diagnostic.json'), recomputed_diagnostic)
    for i, split in enumerate(('training', 'validation', 'development', 'test')):
        for other in ('training', 'validation', 'development', 'test')[i+1:]:
            assert not set(protocol[split+'_sites']) & set(protocol[other+'_sites'])
    new_sites = set().union(*(set(protocol[split+'_sites']) for split in ('training', 'validation', 'development', 'test')))
    for old_path in (ROOT/'artifacts').glob('*/protocol.json'):
        if old_path.parent == OUT:
            continue
        for key, values in read(old_path).items():
            if key.endswith('_sites') and isinstance(values, list):
                assert not new_sites & set(values), str(old_path)
    completion = read(OUT/'completion.json')
    if summary['status'] == 'diagnostic_failed':
        assert not recomputed_diagnostic['gate_passed']
        assert completion.get('diagnostic_failed') or completion.get('status') == 'diagnostic_failed'
        assert not list((OUT/'fits').glob('*.json')) and not list((OUT/'cases').glob('test-*.json'))
        return {'passed': True, 'status': 'diagnostic_failed', 'native_label_runs': recomputed_diagnostic['native_label_runs'],
                'no_unsupported_training_or_superiority_claim': True, 'simulation_only': True}
    assert recomputed_diagnostic['gate_passed'] and completion['completed']
    for stage in stages[1:]:
        assert manifest['acceptance_supplement']['commit_time_unix'] < snapshots[stage]['time_unix']
    selected = read(OUT/'selection.json')
    assert selected['sealed'] and not selected['test_accessed']
    assert selected['no_development_tuning'] and selected['all_training_seeds_retained'] == protocol['training_seeds']
    assert snapshots['development']['time_unix'] < selected['sealed_at_unix']
    assert selected['source_sha256'] == snapshots['development']['source_sha256']
    if summary['status'] == 'evaluated':
        assert selected['sealed_at_unix'] < snapshots['test']['time_unix']
        assert selected['source_sha256'] == snapshots['test']['source_sha256']
    checkpoint_hashes = selected['checkpoint_sha256']; assert len(checkpoint_hashes) == 6
    assert checkpoint_hashes == manifest['checkpoint_sha256']
    for name, expected in checkpoint_hashes.items(): assert digest(ROOT/name) == expected, name
    fits = [read(p) for p in sorted((OUT/'fits').glob('*.json'))]
    assert len(fits) == 6
    for family in protocol['families']:
        for seed in protocol['training_seeds']:
            path = OUT/'fits'/f'{family}-{seed}.json'; fit = read(path)
            assert fit['completed'] and len(fit['history']) == protocol['epochs']
            assert not fit['test_accessed']
            assert [r['epoch'] for r in fit['history']] == list(range(1, protocol['epochs']+1))
            assert np.isclose(fit['best_validation_score'], min(r['validation_score'] for r in fit['history']))
            assert fit['source_sha256'] == snapshots['training']['source_sha256']
            cp = ROOT/next(name for name in checkpoint_hashes if Path(name).name == f'{family}-{seed}.pt')
            assert fit['checkpoint_sha256'] == digest(cp)
    assert {fit['inference_parameters'] for fit in fits} == {23860}
    training = read(OUT/'training.json')
    assert training['completed'] and training['fits'] == 6 and not training['test_accessed']
    assert training['checkpoint_sha256'] == checkpoint_hashes
    latency_hash = digest(OUT/'latency.json'); calibration_hash = digest(OUT/'calibration.json')
    assert latency_hash == selected['latency_sha256'] and calibration_hash == selected['calibration_sha256']
    latency = read(OUT/'latency.json')
    assert all(np.isfinite(v) and v >= 0 for v in latency['head_seconds'].values())
    calibration = read(OUT/'calibration.json')
    for record in (latency, calibration):
        assert record['sealed'] and not record['test_accessed']
        assert record['source_sha256'] == snapshots['training']['source_sha256']
        assert record['checkpoint_sha256'] == checkpoint_hashes
        assert record['time_unix'] < snapshots['development']['time_unix']
    for method, costs in calibration['raw'].items():
        for cost, raw in costs.items():
            residuals = np.asarray(raw['residuals'])
            expected = recomputed_diagnostic['by_split_and_cost']['validation'][cost]['examples']
            assert raw['examples'] == len(residuals) == expected
            assert np.isfinite(residuals).all()
            assert calibration['corrections'][method][cost] == max(0., float(np.quantile(residuals, .95, method='higher')))
    head_raw = latency['model_head_raw_seconds']
    assert np.isclose(latency['head_seconds']['radiomap'], np.quantile(head_raw['radiomap'], .95))
    assert np.isclose(latency['head_seconds']['neural'], max(np.quantile(v, .95) for key, v in head_raw.items() if key != 'radiomap'))
    old_forecast = read(ROOT/'artifacts/closed_loop/latency.json')['p95_seconds']['radiomap']
    for kind, old in old_forecast.items():
        raw = latency['forecast_raw_seconds'][kind]
        assert latency['forecast_seconds'][kind] == max(old, float(np.quantile(raw, .95)) if raw else old)
    cases = [read(f) for f in sorted((OUT/'cases').glob('*.json'))]
    for stage in ('development', 'test') if summary['status'] == 'evaluated' else ('development',):
        expected = {(scene, float(cost), family, seed)
                    for scene in protocol[stage+'_sites'] for cost in protocol['handover_seconds']
                    for family in LABELS for seed in (protocol['training_seeds'] if family in protocol['families'] else [None])}
        cohort = [c for c in cases if c['stage'] == stage]
        keys = {(c['scene'], float(c['cost']), c['family'], c['model_seed']) for c in cohort}
        assert len(cohort) == len(keys) and keys == expected
        for case in cohort:
            audit_case(case, protocol, snapshots[stage], checkpoint_hashes, latency_hash, calibration_hash)
            if case['family'] != 'canonical':
                key = f'{case["family"]}-{case["model_seed"]}' if case['family'] in protocol['families'] else 'radiomap'
                assert case['controller']['correction_commands'] == calibration['corrections'][key][cost_key(case['cost'])]
                head = 'neural' if case['family'] in protocol['families'] else 'radiomap'
                assert case['controller']['charged_head_seconds'] == latency['head_seconds'][head]
    tests = [c for c in cases if c['stage'] == 'test']
    if summary['status'] == 'development_rejected':
        assert completion['development_rejected'] and completion['test_runs'] == 0 and not tests
        assert not (OUT/'test_source.json').exists() and len(cases) == 64
        assert completion['fits'] == 6 and completion['development_runs'] == 64
        assert completion['label_native_runs'] == recomputed_diagnostic['native_label_runs']
        recomputed = describe_development(cases, protocol, recomputed_diagnostic, completion)
        recomputed['excluded_pilot'] = excluded_pilot()
        assert recomputed == summary
        assert summary['primary_success'] is None and not summary['primary_evaluated']
        return {'passed': True, 'status': 'development_rejected', 'independent_fits': 6,
                'development_runs': 64, 'test_runs': 0, 'fresh_sites': 0,
                'original_primary_not_evaluated': True, 'simulation_only': True}
    assert len(tests) == 128 and len(cases)-len(tests) == 64
    assert completion['fits'] == 6 and completion['development_runs'] == 64 and completion['test_runs'] == 128
    assert completion['fresh_sites'] == 8 and completion['label_native_runs'] == recomputed_diagnostic['native_label_runs']
    recomputed = aggregate(tests, protocol, recomputed_diagnostic)
    assert len(recomputed['primary_comparisons']) == 4 and len(recomputed['additional_promotion_comparisons']) == 2
    recomputed['excluded_pilot'] = excluded_pilot()
    assert recomputed == summary
    return {'passed': True, 'status': 'evaluated', 'independent_fits': 6, 'development_runs': 64,
            'test_runs': 128, 'fresh_sites': 8, 'crossed_training_seed_and_site_intervals': True,
            'physical_floor_not_subtracted': True, 'simulation_only': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--audit', action='store_true'); args = parser.parse_args()
    if args.audit: print(json.dumps(audit(), indent=2))
    else: report()
