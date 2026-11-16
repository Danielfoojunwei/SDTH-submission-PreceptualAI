"""Render sealed results; no scientific fit, model call or new selection."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT/'artifacts/measured_radio_counterfix'
seal = json.loads((SOURCE/'verification/manifest.json').read_text())
for name, expected in seal['files_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
report = json.loads((SOURCE/'report.json').read_text())

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), gridspec_kw={'width_ratios':[1.1, 1.1, 1.]})
fig.subplots_adjust(left=.06, right=.985, top=.75, bottom=.25, wspace=.42)
fig.suptitle('Measured Wi-Fi transfer: numerical mapping wins the registered primary comparison',
             x=.06, y=.97, ha='left', fontsize=15, weight='bold')
fig.text(.06, .875, 'Frozen RadioJEPA · same support and observed queries · validation-selected numerical controls',
         fontsize=10, color='#475569')
blue, red = '#2563a6', '#b94732'

budgets = [4, 8, 16]
neural, numerical = [], []
for budget in budgets:
    row = report['budgets'][str(budget)]
    scores = row['scores_by_session']['01']
    neural.append(scores['radiojepa'])
    numerical.append(scores[row['best_numerical_id']])
ax = axes[0]
ax.plot(budgets, neural, 'o-', color=red, label='Frozen RadioJEPA', lw=2)
ax.plot(budgets, numerical, 's-', color=blue, label='VAL-selected numerical', lw=2)
ax.set(title='A  Support efficiency', xlabel='Surveyed receiver locations', ylabel='Conditional RSSI RMSE (dB)')
ax.set_xticks(budgets)
ax.set_ylim(3., 6.2)
ax.grid(axis='y', color='#e2e8f0')
ax.legend(loc='lower left', frameon=False, fontsize=9)
for x, y in ((16, neural[-1]), (16, numerical[-1])):
    ax.annotate(f'{y:.3f}', (x, y), xytext=(-5, 9), textcoords='offset points', ha='right', fontsize=9)

primary = report['budgets']['16']
scenarios = ['S01', 'S02', 'S03']
x = np.arange(3)
ax = axes[1]
ax.bar(x-.18, [primary['session01_scenario_scores']['radiojepa'][s] for s in scenarios],
       width=.34, color=red, label='RadioJEPA')
ax.bar(x+.18, [primary['session01_scenario_scores'][primary['best_numerical_id']][s] for s in scenarios],
       width=.34, color=blue, label='Numerical')
ax.set(title='B  Fixed-scenario strata', ylabel='Conditional RSSI RMSE (dB)', xlabel='Descriptive; 16 support locations')
ax.set_xticks(x, scenarios)
ax.set_ylim(0, 9.1)
ax.set_axisbelow(True)
ax.grid(axis='y', color='#e2e8f0')

contrast = report['primary_contrast']
point = contrast['candidate_minus_reference_db']
ax = axes[2]
for y, key, label in ((1, 'location_resampling_95_interval', 'Locations'),
                       (0, 'spatial_5m_block_95_sensitivity', '5 m blocks')):
    low, high = contrast[key]
    ax.errorbar(point, y, xerr=[[point-low],[high-point]], fmt='o', capsize=5, color=red, lw=2)
    ax.text((low+high)/2, y+.18, f'[{low:+.3f}, {high:+.3f}]', ha='center', fontsize=9)
ax.axvline(0, color='#64748b', ls='--', lw=1)
ax.set_yticks([0, 1], ['5 m blocks', 'Locations'])
ax.set(title='C  Primary paired difference', xlabel='RadioJEPA − numerical RMSE (dB)', xlim=(-.25, 2.3), ylim=(-.5, 1.5))
ax.text(.5, -.28, 'Positive values favor numerical mapping', transform=ax.transAxes,
        ha='center', fontsize=9, color=red)
fig.text(.06, .075,
    'UVIndoorLoc-RTT&RSSI v1 (Matey-Sanz et al., CC BY 4.0) · 87 TEST locations / 174 phone-location units\n'
    'Observed successful 5180 MHz reports only. Fixed support/members; empirical 95% resampling intervals. No hardware or SOTA qualification.',
    fontsize=9, color='#475569', linespacing=1.5)
for extension in ('png', 'svg', 'pdf'):
    path = OUT/f'results.{extension}'
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=200, facecolor='white', metadata={'Creator':'Sealed measured-RSSI result renderer'})
plt.close(fig)
proof = {'schema':1, 'report_sha256':hashlib.sha256((SOURCE/'report.json').read_bytes()).hexdigest(),
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'files_sha256':{f'results.{ext}':hashlib.sha256((OUT/f'results.{ext}').read_bytes()).hexdigest() for ext in ('png','svg','pdf')},
    'scientific_fits_or_model_forwards':0, 'all_labels_derive_from_sealed_report':True}
with (OUT/'plot_receipt.json').open('x') as stream:
    json.dump(proof, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(proof, indent=2))
