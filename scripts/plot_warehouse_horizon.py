"""Export scientific plots from completed, archived horizon evidence."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=Path('artifacts/warehouse_horizon'))
    args = p.parse_args()
    offline = json.loads((args.archive/'offline.json').read_text())
    summary = json.loads((args.archive/'snapshot.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    offsets = summary['offsets_s']
    axes[0].plot(offsets, offline['baseline']['service_rmse_by_offset'], 'o-', label='Numerical belief')
    for graph, label in [(False, 'Validated residual MLP'), (True, 'Validated residual graph')]:
        values = np.array([offline['models'][f'{seed}-{graph}']['validated_blend']['service_rmse_by_offset'] for seed in [0, 1, 2]])
        axes[0].plot(offsets, values.mean(0), 'o-', label=label)
        axes[0].fill_between(offsets, values.min(0), values.max(0), alpha=.12)
    axes[0].set(xlabel='Forecast offset (s)', ylabel='Service-fraction RMSE', title='Held-out forecast error')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    axes[1].axvline(0., color='grey', linewidth=1)
    for index, (method, label) in enumerate([('residual_mlp', 'MLP'), ('residual_graph', 'Graph')]):
        result = summary['paired_comparisons'][method]['mean_mission_utility']
        value = result['difference']; low, high = result['ci95']
        axes[1].errorbar(value, index, xerr=[[max(0., value-low)], [max(0., high-value)]],
                         fmt='o', capsize=4, label=label)
    axes[1].set(yticks=[0, 1], yticklabels=['Residual MLP', 'Residual graph'],
                xlabel='Mission utility difference versus numerical planner')
    axes[1].set_title('Closed-loop effect\nCrossed bootstrap 95% interval', fontsize=11)
    axes[1].xaxis.set_major_locator(MaxNLocator(5))
    axes[1].grid(axis='x', alpha=.2)
    fig.suptitle('Simulated 100-robot warehouse · three model seeds × three environments', fontsize=11)
    fig.savefig(args.archive/'forecast_and_control.png', dpi=180)
    fig.savefig(args.archive/'forecast_and_control.svg')
    plt.close(fig)
    print('Exported forecast_and_control.png and .svg from archived observations')


if __name__ == '__main__':
    main()
