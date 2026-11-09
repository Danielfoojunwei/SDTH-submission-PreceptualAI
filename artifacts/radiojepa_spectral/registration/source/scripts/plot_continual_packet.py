"""Render recorded full-fleet results; no new model evaluation or selection."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'artifacts/continual_packet/report.json'
    report = json.loads(source.read_text())
    out = source.parent / 'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={'width_ratios': [1.35, 1]})
    fig.subplots_adjust(left=.065, right=.98, bottom=.30, top=.79, wspace=.29)
    arms = ['canonical', 'reciprocal_radiomap', 'synchronized_ignored', 'staggered_ignored',
            'synchronized_memory', 'staggered_memory', 'synchronized_bayesian', 'staggered_bayesian']
    labels = ['Canonical', 'Reciprocal\nmap', 'Sync\nignored', 'Stagger\nignored',
              'Sync\nmemory', 'Stagger\nmemory', 'Sync\nBayes', 'Stagger\nBayes']
    indexed = {(row['arm'], row['handover_seconds']): row for row in report['means']}
    colors = ['#465b7c', '#009b92']
    for offset, cost, color in zip([-.19, .19], [.03, 4.], colors):
        values = [100 * indexed[arm, cost]['command_loss'] for arm in arms]
        axes[0].bar(np.arange(len(arms)) + offset, values, width=.36,
                    color=color, label=f'{cost:g} s switch')
    axes[0].set_xticks(range(len(arms)), labels, fontsize=8)
    axes[0].set_ylabel('Observed missed commands (%)')
    axes[0].set_ylim(0, max(100 * row['command_loss'] for row in report['means']) * 1.25)
    axes[0].set_title('Complete-run service after charged probe costs', loc='left', weight='bold')
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_axisbelow(True)
    axes[0].grid(axis='y', alpha=.15)
    for family, color in [('memory', '#7961a8'), ('bayesian', '#009b92')]:
        for cost, style in [(.03, '-'), (4., '--')]:
            rows = sorted((r for r in report['cumulative_excess'] if r['estimator'] == family
                           and r['handover_seconds'] == cost and r['reference'] == 'canonical'),
                          key=lambda r: r['second'])
            x = [r['second'] for r in rows]
            y = [r['mean_difference'] for r in rows]
            axes[1].plot(x, y, style, marker='o', color=color, label=f'{family}, {cost:g} s')
    axes[1].axhline(0, color='#89909a', linewidth=1)
    axes[1].set_xlabel('Elapsed trial time (s)')
    axes[1].set_ylabel('Additional fleet misses versus canonical')
    axes[1].set_title('Observed cumulative excess; lower is better', loc='left', weight='bold')
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=.15)
    fig.suptitle('Costed RF learning on the warehouse fleet', x=.065, ha='left', weight='bold', fontsize=19)
    fig.text(.065, .86, '100 robots · six procedural confirmation sites · two paired packet seeds · 80 seconds per run',
             color='#465b7c', fontsize=11)
    fig.text(.065, .17, 'Observed means shown. Registered qualification also applies conservative endpoint-pending bounds, multiplicity correction,\n'
             'silence/worst-decile/video/bulk margins, timing and exact-replay checks. Full comparison matrices and intervals are in report.json.\n'
             'Managed-source monitoring, AP-assisted carrier sense and slot-average PHY are simulator assumptions.\n'
             'No physical-car, JEPA-superiority, SOTA or production-readiness claim.', fontsize=9, color='#465b7c', linespacing=1.6, va='top')
    outputs = []
    for extension in ['svg', 'pdf', 'png']:
        path = out / ('fleet-service.' + extension)
        fig.savefig(path, dpi=180, facecolor='white')
        outputs.append(path)
    plt.close(fig)
    paths = [source, Path(__file__), *outputs]
    (out / 'receipt.json').write_text(json.dumps({'new_evaluation_or_training': False,
        'sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}, indent=2) + '\n')


if __name__ == '__main__':
    main()
