"""Plot completed registered semantic-JEPA evidence without evaluating a model.

Requires all twelve fits, sealed latency and the final development/conditional
confirmation report. Three-member ranges are descriptive, not confidence
intervals. This script is outside the frozen scientific source closure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('supervised', 'context_jepa', 'semantic_aux', 'semantic_jepa')
REFERENCES = ('radiomap', 'frozen_fleet_jepa')
ORDER = (*REFERENCES, *FAMILIES)
LABELS = {'radiomap': 'RadioMap', 'frozen_fleet_jepa': 'Frozen\nFleetJEPA',
          'supervised': 'Supervised', 'context_jepa': 'Context\nJEPA',
          'semantic_aux': 'Semantic\nregression', 'semantic_jepa': 'Semantic\nJEPA'}
COLORS = {'supervised': '#536C8C', 'context_jepa': '#9670AD',
          'semantic_aux': '#BE843C', 'semantic_jepa': '#008D83'}
COST_COLORS = ('#536C8C', '#008D83')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def name(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def number(value, *, nonnegative=True):
    if isinstance(value, bool):
        raise ValueError('Boolean supplied where a measured number is required')
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError('Nonfinite or invalid measured value')
    return result


def load_inputs(folder):
    """Refuse partial receipts, partial fits or undocumented source mixtures."""
    filenames = ('protocol.json', 'training.json', 'development.json',
                 'confirmation.json', 'latency.json', 'report.json')
    paths = [folder / filename for filename in filenames]
    contents = {path.name: json.loads(path.read_text()) for path in paths}
    protocol, training = contents['protocol.json'], contents['training.json']
    report, latency = contents['report.json'], contents['latency.json']
    if (tuple(protocol['families']) != FAMILIES or protocol['training_seeds'] != [0, 1, 2]
            or training.get('completed') is not True or training['fits'] != 12
            or latency.get('sealed') is not True or contents['confirmation.json'].get('completed') is not True
            or 'development' not in report or 'confirmation' not in report):
        raise ValueError('Finish all registered fits, latency and final reporting first')
    if (training['identity']['protocol'] != protocol
            or report['source_sha256'] != training['identity']['source_sha256']
            or latency['source_sha256'] != report['source_sha256']
            or latency['checkpoint_sha256'] != training['checkpoint_sha256']):
        raise ValueError('Source, checkpoint or protocol identities differ')
    for stem in ('training', 'development', 'confirmation', 'latency'):
        if report[stem + '_sha256'] != digest(folder / (stem + '.json')):
            raise ValueError(f'Final report does not bind {stem} receipt')
    registered_protocol = report['source_sha256'].get('artifacts/semantic_jepa/protocol.json')
    if registered_protocol != digest(folder / 'protocol.json'):
        raise ValueError('Protocol differs from registered source')
    expected = {f'{family}-{seed}.json' for family in FAMILIES for seed in protocol['training_seeds']}
    actual = {path.name for path in (folder / 'fits').glob('*.json')}
    if actual != expected:
        raise ValueError('Require the exact twelve-fit membership, without selecting members')
    fits = {}
    for family in FAMILIES:
        fits[family] = []
        for seed in protocol['training_seeds']:
            path = folder / 'fits' / f'{family}-{seed}.json'
            fit = json.loads(path.read_text())
            if (fit.get('completed') is not True or fit['family'] != family or fit['seed'] != seed
                    or fit['identity'] != training['identity']
                    or training['fit_receipt_sha256'][f'{family}-{seed}'] != digest(path)
                    or len(fit['history']) != protocol['epochs']
                    or [row['epoch'] for row in fit['history']] != list(range(1, protocol['epochs'] + 1))):
                raise ValueError(f'Incomplete or changed fit {family}/{seed}')
            for row in fit['history']:
                for field in ('validation_weighted_rmse', 'validation_ap_regret'):
                    number(row[field])
            if not isinstance(fit['diagnostics'], dict):
                raise ValueError('Actual fitted encoder diagnostics are required')
            for section in ('online_radio_by_band', 'teacher_radio_by_band'):
                if set(fit['diagnostics'][section]) != {'0', '1', '2'}:
                    raise ValueError('Incomplete physical-band diagnostics')
                for entry in fit['diagnostics'][section].values():
                    number(entry['std_mean']); number(entry['effective_rank'])
            number(fit['diagnostics']['fixed_geometry_plus1db_radio_l2'])
            number(fit['diagnostic_sample_rows'])
            fits[family].append(fit)
            paths.append(path)
    costs = protocol['handover_seconds']
    if len(costs) != 2:
        raise ValueError('This figure requires the registered two handover costs')
    development, confirmation = report['development'], report['confirmation']
    if development['passed_costs'] != confirmation['passed_development_costs']:
        raise ValueError('Development/conditional confirmation seals differ')
    for section, measured_costs in ((development, costs), (confirmation, development['passed_costs'])):
        expected_rows = {(family, float(cost)) for family in ORDER for cost in measured_costs}
        rows = {(row['family'], float(row['handover_seconds'])) for row in section['means']}
        if rows != expected_rows or len(rows) != len(section['means']):
            raise ValueError('Incomplete or duplicated plotted service membership')
        for row in section['means']:
            number(row['command_loss'])
    for family in ORDER:
        members = [None] if family in REFERENCES else protocol['training_seeds']
        for seed in members:
            key = family if seed is None else f'{family}-s{seed}'
            if not latency['raw_seconds'][key]['warm']:
                raise ValueError('Warm request measurements missing')
            for value in latency['raw_seconds'][key]['warm']:
                number(value)
            number(latency['profiles'][key]['warm'])
    return protocol, report, latency, fits, paths


def label(family):
    return LABELS[family].replace('\n', ' ')


def curve(axis, fits, field, ylabel, np):
    for family in FAMILIES:
        data = np.array([[number(row[field]) for row in fit['history']] for fit in fits[family]])
        epochs = np.arange(1, data.shape[1] + 1)
        axis.fill_between(epochs, data.min(0), data.max(0), color=COLORS[family], alpha=.13, linewidth=0)
        axis.plot(epochs, data.mean(0), color=COLORS[family], linewidth=1.8, label=label(family))
    axis.set_xlabel('Training epoch')
    axis.set_ylabel(ylabel)
    axis.grid(alpha=.15)
    axis.set_axisbelow(True)


def service(axis, section, costs, *, confirmation, np, maximum):
    if not costs:
        axis.set_axis_off()
        axis.text(.5, .62, 'CONFIRMATION NOT RUN', transform=axis.transAxes,
                  ha='center', weight='bold', color='#45505E', fontsize=13)
        axis.text(.5, .39, 'Both handover costs failed development.\nReserved test sites remain unused.\nNo held-out service outcome is shown.',
                  transform=axis.transAxes, ha='center', va='center', color='#536C8C', linespacing=1.7)
        return
    index = {(row['family'], float(row['handover_seconds'])): row for row in section['means']}
    position = np.arange(len(ORDER))
    width = .34 if len(costs) == 2 else .5
    for ci, cost in enumerate(costs):
        offset = (ci - (len(costs) - 1) / 2) * width
        means, low, high = [], [], []
        for family in ORDER:
            value = 100 * number(index[family, float(cost)]['command_loss'])
            members = [100 * number(row['command_loss']) for row in section['per_seed_means']
                       if row['family'] == family and float(row['handover_seconds']) == float(cost)]
            expected = 1 if family in REFERENCES else 3
            if len(members) != expected:
                raise ValueError('Incomplete per-member service ranges')
            means.append(value)
            low.append(max(0., value - min(members)))
            high.append(max(0., max(members) - value))
        # Match cost colors by registered physical cost, even if one cost was vetoed.
        color = COST_COLORS[0 if float(cost) == .03 else 1]
        axis.bar(position + offset, means, width * .92, color=color,
                 label=f'{float(cost):g} s handover', yerr=np.array([low, high]),
                 capsize=2, error_kw={'elinewidth': .8, 'alpha': .8})
    axis.set_xticks(position, [LABELS[family] for family in ORDER], fontsize=8)
    axis.set_ylabel('Observed missed commands (%)')
    axis.set_ylim(0, max(maximum, 1e-3) * 1.35)
    axis.legend(frameon=False, fontsize=8, loc='upper left')
    axis.grid(axis='y', alpha=.15)
    axis.set_axisbelow(True)
    flags = section['per_cost'] if confirmation else section['by_cost']
    summaries = []
    for row in flags:
        if confirmation and row['status'] == 'not_run_development_veto':
            summaries.append(f"{row['handover_seconds']:g}s: NOT RUN (development veto)")
        else:
            summaries.append(f"{row['handover_seconds']:g}s: {'PASS' if row['passes'] else 'FAIL'}")
    axis.text(0., 1.04, '  ·  '.join(summaries), transform=axis.transAxes, fontsize=8,
              color='#45505E', va='bottom')


def learning_and_service(protocol, report, fits, plt, np):
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.8))
    fig.subplots_adjust(left=.07, right=.97, top=.83, bottom=.25, hspace=.55, wspace=.23)
    fig.suptitle('Semantic JEPA: learning and complete fleet service', x=.07, y=.97,
                 ha='left', fontsize=19, weight='bold')
    fig.text(.07, .925, 'All four matched training families · three independently trained members each · no seed selection',
             fontsize=11, color='#536C8C')
    curve(axes[0, 0], fits, 'validation_weighted_rmse', 'Weighted validation SNR RMSE (dB)', np)
    curve(axes[0, 1], fits, 'validation_ap_regret', 'Validation AP-choice regret (dB)', np)
    axes[0, 0].set_title('A  Common validation error', loc='left', weight='bold')
    axes[0, 1].set_title('B  Common decision diagnostic', loc='left', weight='bold')
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    values = [100 * number(row['command_loss']) for section in ('development', 'confirmation')
              for field in ('means', 'per_seed_means') for row in report[section][field]]
    maximum = max(values)
    service(axes[1, 0], report['development'], protocol['handover_seconds'],
            confirmation=False, np=np, maximum=maximum)
    service(axes[1, 1], report['confirmation'], report['development']['passed_costs'],
            confirmation=True, np=np, maximum=maximum)
    axes[1, 0].set_title(f"C  DEVELOPMENT · {report['development']['independent_sites']} sites",
                         loc='left', weight='bold', pad=29)
    axes[1, 1].set_title(f"D  CONDITIONAL CONFIRMATION · {report['confirmation']['independent_sites']} tested sites",
                         loc='left', weight='bold', pad=29)
    fig.text(.07, .155, 'Curves show three-member means with min–max seed bands; service whiskers show ranges of member means, not confidence intervals.\n'
             'Service includes charged forecast queues and scan/handover costs. All fixed members and both registered costs remain visible.\n'
             'Confirmation is opened only after the development gate; report.json contains site intervals, pending-command sensitivity and every veto.\n'
             'Privileged simulator labels and shared procedural CAD; no physical-car, SOTA or production-readiness claim.',
             fontsize=9, color='#536C8C', linespacing=1.5, va='top')
    return fig


def encoder_and_cost(protocol, latency, fits, plt, np):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.8))
    fig.subplots_adjust(left=.07, right=.97, top=.83, bottom=.25, hspace=.55, wspace=.23)
    fig.suptitle('Physical-band representation and measured request cost', x=.07, y=.97,
                 ha='left', fontsize=19, weight='bold')
    sample_counts = sorted({fit['diagnostic_sample_rows'] for group in fits.values() for fit in group})
    fig.text(.07, .925, f'Fixed validation-slice diagnostics ({", ".join(map(str, sample_counts))} rows) · descriptive three-member ranges',
             fontsize=11, color='#536C8C')
    for axis, field in ((axes[0, 0], 'std_mean'), (axes[0, 1], 'effective_rank')):
        maximum_diagnostic = 0.
        for fi, family in enumerate(FAMILIES):
            offset = (fi - 1.5) * .035
            for section, style in (('online_radio_by_band', '-'), ('teacher_radio_by_band', '--')):
                values = np.array([[number(fit['diagnostics'][section][str(band)][field]) for band in range(3)]
                                   for fit in fits[family]])
                maximum_diagnostic = max(maximum_diagnostic, float(values.max()))
                x = np.arange(3) + offset
                axis.fill_between(x, values.min(0), values.max(0), color=COLORS[family], alpha=.07, linewidth=0)
                axis.plot(x, values.mean(0), style, color=COLORS[family], marker='o', markersize=3,
                          linewidth=1.6, label=label(family) if section.startswith('online') else None)
        axis.set_xticks(range(3), ['2.4 GHz', '5.2 GHz', '6 GHz'])
        axis.grid(alpha=.15)
        axis.set_axisbelow(True)
        axis.set_ylim(0, max(maximum_diagnostic, 1e-6) * 1.3)
    axes[0, 0].set_title('A  Within-band radio embedding variation', loc='left', weight='bold')
    axes[0, 0].set_ylabel('Mean coordinate standard deviation')
    axes[0, 1].set_title('B  Centered effective rank', loc='left', weight='bold')
    axes[0, 1].set_ylabel('Entropy effective rank')
    first = axes[0, 0].legend(frameon=False, fontsize=8, ncol=2, loc='upper left')
    axes[0, 0].add_artist(first)
    axes[0, 1].legend(handles=[Line2D([0], [0], color='#45505E', label='Online radio encoder'),
                             Line2D([0], [0], color='#45505E', linestyle='--', label='EMA radio encoder')],
                         frameon=False, fontsize=8, loc='upper left')
    sensitivity = []
    for fi, family in enumerate(FAMILIES):
        values = np.array([number(fit['diagnostics']['fixed_geometry_plus1db_radio_l2']) for fit in fits[family]])
        sensitivity.append(values)
        axes[1, 0].bar(fi, values.mean(), color=COLORS[family], width=.55,
                       yerr=np.array([[values.mean() - values.min()], [values.max() - values.mean()]]), capsize=3)
        axes[1, 0].scatter(fi + np.linspace(-.09, .09, 3), values, s=12, color='#27384A', zorder=3)
    axes[1, 0].set_xticks(range(4), [LABELS[f] for f in FAMILIES], fontsize=8)
    axes[1, 0].set_title('C  RF sensitivity with geometry held fixed', loc='left', weight='bold')
    axes[1, 0].set_ylabel('Mean link-latent L2 response to +1 dB')
    axes[1, 0].grid(axis='y', alpha=.15)
    axes[1, 0].set_axisbelow(True)
    actual_costs, charges, lows, highs = [], [], [], []
    for family in ORDER:
        members = [None] if family in REFERENCES else protocol['training_seeds']
        keys = [family if seed is None else f'{family}-s{seed}' for seed in members]
        values = np.array([1000 * np.quantile(latency['raw_seconds'][key]['warm'], .95) for key in keys])
        envelopes = [1000 * number(latency['profiles'][key]['warm']) for key in keys]
        actual_costs.append(values.mean()); charges.append(np.mean(envelopes))
        lows.append(values.mean() - values.min()); highs.append(values.max() - values.mean())
    x = np.arange(len(ORDER))
    axes[1, 1].bar(x, actual_costs, color=['#64717F', '#3F7596', *[COLORS[f] for f in FAMILIES]],
                   width=.55, yerr=np.array([lows, highs]), capsize=2, label='Measured arm p95')
    axes[1, 1].scatter(x, charges, marker='D', s=24, color='#27384A', label='Sealed charged envelope', zorder=3)
    axes[1, 1].set_xticks(x, [LABELS[f] for f in ORDER], fontsize=8)
    axes[1, 1].set_title('D  Warm full-predictor CPU cost', loc='left', weight='bold')
    axes[1, 1].set_ylabel('Request time (ms)')
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].set_ylim(0, max(charges + actual_costs) * 1.3)
    axes[1, 1].grid(axis='y', alpha=.15)
    axes[1, 1].set_axisbelow(True)
    fig.text(.07, .155, 'Solid/dashed lines describe online/EMA radio encoders; only semantic JEPA trains against that radio teacher. Other families retain it as an unused EMA diagnostic.\n'
             'The teacher input has two physical coordinates: higher latent rank or RF sensitivity does not establish useful prediction or better service.\n'
             'Warm timings include predictor feature/packing work; checkpoint startup is measured separately. Ranges are member variation, not uncertainty guarantees.\n'
             'Simulation-only descriptive evidence; no collapse-elimination, physical latency, SOTA or production qualification.',
             fontsize=9, color='#536C8C', linespacing=1.5, va='top')
    return fig


def render(folder, destination, dpi):
    # Load and validate every required actual input before creating any output.
    protocol, report, latency, fits, inputs = load_inputs(folder)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'semantic-jepa-registered-figures',
                         'axes.spines.top': False, 'axes.spines.right': False})
    destination.mkdir(parents=True, exist_ok=True)
    figures = [('learning-and-service', learning_and_service(protocol, report, fits, plt, np)),
               ('representation-and-cost', encoder_and_cost(protocol, latency, fits, plt, np))]
    outputs = []
    for stem, figure in figures:
        for extension in ('svg', 'pdf', 'png'):
            path = destination / f'{stem}.{extension}'
            metadata = ({'CreationDate': None, 'ModDate': None} if extension == 'pdf'
                        else {'Date': None} if extension == 'svg' else {'Software': 'Matplotlib'})
            figure.savefig(path, dpi=dpi, facecolor='white', metadata=metadata)
            outputs.append(path)
        plt.close(figure)
    receipt = {'new_training_or_model_evaluations': False, 'new_model_or_seed_selection': False,
               'ranges_are_confidence_intervals': False,
               'uncertainty_scope': 'Descriptive min-max across all three registered independently trained members.',
               'confirmation_costs_actually_tested': report['confirmation']['passed_development_costs'],
               'confirmation_status_by_cost': report['confirmation']['per_cost'],
               'diagnostics_are_service_qualification': False,
               'teacher_scope': 'Two-coordinate privileged simulator semantics; EMA radio diagnostics are unused teachers in other families.',
               'software': {'python': platform.python_version(), 'matplotlib': matplotlib.__version__, 'numpy': np.__version__},
               'dpi': dpi, 'script_sha256': digest(Path(__file__)),
               'inputs_sha256': {name(path): digest(path) for path in inputs},
               'outputs_sha256': {name(path): digest(path) for path in outputs}}
    (destination / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'figures': [str(path) for path in outputs], 'new_evaluations': False}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts/semantic_jepa',
                        help='Completed study artifact directory')
    parser.add_argument('--output-directory', type=Path, help='Defaults to ARTIFACTS/figures')
    parser.add_argument('--dpi', type=int, default=180)
    args = parser.parse_args()
    if not 72 <= args.dpi <= 600:
        parser.error('--dpi must be between 72 and 600')
    destination = args.output_directory or args.artifacts / 'figures'
    try:
        render(args.artifacts.resolve(), destination.resolve(), args.dpi)
    except (KeyError, ValueError, FileNotFoundError) as error:
        parser.exit(1, f'Cannot render completed study evidence: {error}\n')


if __name__ == '__main__':
    main()
