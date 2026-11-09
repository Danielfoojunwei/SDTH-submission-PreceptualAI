"""Plot recorded response errors and registered excitation; no new evaluation."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    summary_path = ROOT / 'artifacts/continual_radio/summary.json'
    protocol_path = ROOT / 'artifacts/continual_radio/registration/protocol.json'
    summary = json.loads(summary_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    destination = ROOT / 'artifacts/continual_radio/figures'
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig = plt.figure(figsize=(11, 6.6), facecolor='white')
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 2.3], wspace=.38, hspace=.65,
                            left=.08, right=.97, top=.79, bottom=.23)
    colors = ['#425779', '#009a93']
    designs = ['synchronized', 'staggered']
    for row, design in enumerate(designs):
        ax = fig.add_subplot(grid[row, 0])
        matrix = np.asarray(protocol['probe_patterns'][design])
        ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha='center', va='center',
                    color='white' if value else '#24364b', weight='bold')
        ax.set_xticks([0, 1], ['TX 1', 'TX 2'])
        ax.set_yticks([0, 1], ['Slot 1', 'Slot 2'])
        ax.set_title(f'{design.capitalize()}: rank {np.linalg.matrix_rank(matrix)}',
                     loc='left', color=colors[row], weight='bold')
        ax.tick_params(length=0)
    ax = fig.add_subplot(grid[:, 1])
    families = ['jepa', 'supervised', 'least_squares', 'bayesian']
    names = ['Response\nJEPA', 'Matched\nsupervised', 'Least\nsquares', 'Bayesian\nidentification']
    for offset, design, color in zip([-.19, .19], designs, colors):
        methods = summary['groups'][design + '-K2']['methods']
        values = [np.mean([methods[f'{family}-{seed}']['clean_mae_mw'] for seed in range(3)])
                  if family in ('jepa', 'supervised') else methods[family]['clean_mae_mw']
                  for family in families]
        bars = ax.bar(np.arange(4) + offset, np.asarray(values) * 1e6, .36,
                      label=design.capitalize(), color=color)
        ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
    ax.set_xticks(range(4), names)
    ax.set_ylim(0, .192)
    ax.set_ylabel('Clean response MAE (nW); lower is better')
    ax.set_title('Recorded errors at K = 2 feedback observations', loc='left', weight='bold')
    ax.legend(frameon=False, ncols=2, loc='upper right', fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.15)
    bayes_sync = summary['groups']['synchronized-K2']['methods']['bayesian']['clean_mae_mw']
    bayes_staggered = summary['groups']['staggered-K2']['methods']['bayesian']['clean_mae_mw']
    reduction = 100 * (1 - bayes_staggered / bayes_sync)
    fig.suptitle('Independent measurements reveal action effects', x=.08, ha='left',
                 fontsize=19, weight='bold', color='#24364b')
    fig.text(.08, .87, f'{reduction:.1f}% lower Bayesian response error at equal transmitter-on airtime',
             fontsize=12, color='#007d77')
    fig.text(.08, .13, 'Left: registered nominal activity patterns, before missing reports. Both K2 designs use 2 transmitter-airtime units.\n'
             'Right: recorded means; three training seeds per learned family. Numerical controls use the same measurement episodes.\n'
             'Actual Sionna RT kernels; constructed probe/device/noise episodes. One warehouse, five held-out blocker poses.\n'
             'Response JEPA fails its advantage gate. No packet-delivery, hardware or cross-warehouse claim.',
             fontsize=9, linespacing=1.6, color='#425779', va='top')
    outputs = []
    for extension in ('svg', 'png', 'pdf'):
        path = destination / ('response-identification.' + extension)
        fig.savefig(path, dpi=180, facecolor='white')
        outputs.append(path)
    plt.close(fig)
    def hashed(path):
        return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt = {'new_evaluation_or_training': False,
               'inputs': [hashed(summary_path), hashed(protocol_path), hashed(Path(__file__))],
               'outputs': [hashed(path) for path in outputs], 'bayesian_error_reduction_percent': reduction}
    (destination / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
