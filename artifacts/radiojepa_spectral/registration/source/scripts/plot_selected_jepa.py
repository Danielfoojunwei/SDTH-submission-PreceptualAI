"""Render completed selected-JEPA evidence using passive receipts/archives only.

This derived script is outside the frozen source closure. It imports no model,
simulator or study runner, and refuses incomplete confirmation results.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ('radiomap', 'frozen_fleet_jepa')
REPLAY_FIELDS = ('result', 'forecast_mean_sha256', 'forecast_std_sha256',
                 'telemetry', 'decision_trace', 'action_trace_sha256')
LABELS = {'radiomap': 'RadioMap', 'frozen_fleet_jepa': 'Original FleetJEPA',
          'supervised': 'Matched supervised', 'context_jepa': 'Context JEPA',
          'semantic_jepa': 'Semantic JEPA'}
COLORS = {'radiomap': '#697B8C', 'frozen_fleet_jepa': '#A07B58',
          'supervised': '#8568A8', 'context_jepa': '#007E83', 'semantic_jepa': '#007E83'}
BLUE, ORANGE, RED, INK = '#2675A8', '#C18320', '#BD4A57', '#203343'


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def value_digest(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def relative(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def number(value):
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError('A finite measured number is required')
    return float(value)


def member_names(p, family):
    return [family] if family in BASELINES else [f'{family}-s{seed}' for seed in p['training_seeds']]


def case_name(key):
    site, cost, family, seed = key
    suffix = '' if seed is None else f'-s{seed}'
    return f'confirmation-{site}-h{round(cost * 1000):04d}-{family}{suffix}'


def load_inputs(folder):
    """Bind final analysis, exact case membership, archives and timing provenance."""
    names = ('protocol.json', 'selection.json', 'registration/source.json',
             'registration/git.json', 'completion.json', 'report.json')
    paths = [folder / name for name in names]
    if any(not path.exists() for path in paths):
        raise ValueError('Finish all 128 registered cases, eight replays and final reporting before rendering')
    values = {name: json.loads((folder / name).read_text()) for name in names}
    p, report, selection = (values[name] for name in ('protocol.json', 'report.json', 'selection.json'))
    registration, completion = values['registration/source.json'], values['completion.json']
    families = ('radiomap', 'frozen_fleet_jepa', 'supervised', p['candidate_family'])
    costs = [float(cost) for cost in p['handover_seconds']]
    if (len(set(families)) != 4 or costs != [.03, 4.] or p['training_seeds'] != [0, 1, 2]
            or len(p['confirmation_sites']) != 8 or len(set(p['confirmation_sites'])) != 8
            or report['candidate_family'] != p['candidate_family']
            or selection['selected_family'] != p['candidate_family']
            or completion['completed'] is not True or report['case_count'] != 128 or completion['case_count'] != 128
            or len(report['primary']) != 6 or report['primary_multiplicity_budget'] != 6):
        raise ValueError('Final report differs from the registered eight-site/eight-arm matrix')
    if (report['completion_sha256'] != digest(folder / 'completion.json')
            or report['selection_sha256'] != digest(folder / 'selection.json')
            or completion['selection_sha256'] != report['selection_sha256']
            or report['source_sha256'] != completion['source_sha256']
            or report['source_sha256'] != registration['source_sha256']):
        raise ValueError('Report, completion and registration identities differ')
    for name, expected in report['source_sha256'].items():
        path = ROOT / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Changed frozen source or input: ' + name)
    latency_path = ROOT / p['source_study'] / 'latency.json'
    latency = json.loads(latency_path.read_text())
    if latency.get('sealed') is not True or latency.get('outcomes_used') is not False:
        raise ValueError('Require the outcome-independent sealed timing profile')
    paths.append(latency_path)
    for family in families:
        for name in member_names(p, family):
            if selection['reused_timing_profiles'][name] != latency['profiles'][name]:
                raise ValueError('Reused request charges changed')
    seeds_for = lambda family: [None] if family in BASELINES else p['training_seeds']
    keys = [(site, cost, family, seed) for site in p['confirmation_sites'] for cost in costs
            for family in families for seed in seeds_for(family)]
    replay_keys = [(p['confirmation_sites'][0], cost, family, seed) for cost in costs
                   for family, seed in [('radiomap', None), ('frozen_fleet_jepa', None),
                                        (p['candidate_family'], 0), ('supervised', 0)]]
    rows, replay_rows = [], []
    for replay, expected, target in ((False, keys, rows), (True, replay_keys, replay_rows)):
        subfolder = folder / ('replays' if replay else 'cases')
        if {path.name for path in subfolder.glob('*.json.gz')} != {case_name(key) + '.json.gz' for key in expected}:
            raise ValueError('Incomplete or unregistered archive membership')
        for key in expected:
            path = subfolder / (case_name(key) + '.json.gz')
            receipt_path = subfolder / (case_name(key) + '.receipt.json')
            receipt = json.loads(receipt_path.read_text())
            row = json.loads(gzip.decompress(path.read_bytes()))
            if (receipt['archive_sha256'] != digest(path) or receipt['case'] != list(key)
                    or row['case'] != list(key) or row['completed'] is not True
                    or row['stage'] != 'confirmation' or row['replay'] is not replay
                    or receipt['replay'] is not replay or receipt['result_sha256'] != value_digest(row['result'])
                    or row['source_sha256'] != report['source_sha256']
                    or row['registration_sha256'] != digest(folder / 'registration/source.json')
                    or row['selection_sha256'] != report['selection_sha256']
                    or row['latency_sha256'] != digest(latency_path)):
                raise ValueError('Archive receipt, result or provenance changed: ' + path.name)
            if len(row['jobs']) != p['duration_seconds']:
                raise ValueError('Incomplete actual request timing trace')
            arm = family_name = key[2]
            if key[3] is not None:
                arm += f'-s{key[3]}'
            profile = latency['profiles'][arm]
            previous_ready = 0.
            for index, job in enumerate(row['jobs']):
                elapsed = number(job['actual_prediction_seconds'])
                charge = number(job['charged_seconds'])
                expected_charge = profile[job['kind']] + p['extra_compute_seconds']
                previous_ready = max(index, previous_ready) + expected_charge
                if (elapsed < 0 or not math.isclose(charge, expected_charge, abs_tol=1e-12, rel_tol=0)
                        or job['ready_slot'] != math.ceil(previous_ready * 100 - 1e-10)
                        or job['runtime_overrun'] is not (elapsed > profile[job['kind']])):
                    raise ValueError('Actual timing, charged FIFO or runtime veto differs: ' + path.name)
            actual_overrun = any(job['runtime_overrun'] for job in row['jobs'])
            causal = (all(event['measurement_slot'] < event['complete_slot'] <= event['delivery_slot']
                          and event['first_usable_decision'] == event['delivery_slot'] // 100 + 1
                          for event in row['telemetry']['events'])
                      and all(job['decision'] == index and job['history_cutoff_exclusive'] == 60 + index
                              for index, job in enumerate(row['jobs'])))
            if row['runtime_overrun'] is not actual_overrun or row['causality_pass'] is not causal:
                raise ValueError('Raw causal/runtime flags disagree with actual trace')
            for job in row['jobs']:
                number(job['actual_prediction_seconds']); number(job['charged_seconds'])
            number(row['predictor_startup_seconds'])
            paths.extend((path, receipt_path))
            # Full action/telemetry exports stay in the hashed archives rather
            # than accumulating 136 large decoded copies while plotting.
            target.append({'case': row['case'], 'family': family_name,
                           'predictor_startup_seconds': row['predictor_startup_seconds'],
                           'causality_pass': causal, 'runtime_overrun': actual_overrun,
                           'replay_field_sha256': {field: value_digest(row[field]) for field in REPLAY_FIELDS},
                           'jobs': [{field: job[field] for field in
                                     ('actual_prediction_seconds', 'charged_seconds', 'kind', 'runtime_overrun')}
                                    for job in row['jobs']],
                           'result': {field: row['result'][field] for field in ('control_misses', 'control_sent')}})
    original = {tuple(row['case']): row for row in rows}
    actual_replays = [{'case': row['case'], 'exact': row['replay_field_sha256'] ==
                      original[tuple(row['case'])]['replay_field_sha256']} for row in replay_rows]
    exact = all(row['exact'] for row in actual_replays)
    causal = all(row['causality_pass'] for row in rows + replay_rows)
    runtime = any(row['runtime_overrun'] for row in rows + replay_rows)
    validation = report['integrity_validation']
    if (completion['replays'] != actual_replays or completion['all_exact'] is not exact
            or validation['source_and_archive_integrity'] is not True
            or validation['causality_and_kernel_checks'] is not causal
            or validation['exact_replay'] is not exact or validation['runtime_failure'] is not runtime
            or report['all_row_causality_pass'] is not all(row['causality_pass'] for row in rows)
            or report['integrity_pass'] is not (causal and exact)
            or report['any_runtime_overrun'] is not runtime):
        raise ValueError('Final replay/integrity/runtime flags differ from current raw archives')
    metrics = {(row['site'], row['handover_seconds'], row['family'], row['training_seed']): row
               for row in report['every_case']}
    if len(metrics) != len(keys) or set(metrics) != set(keys):
        raise ValueError('Final report differs from raw confirmation membership')
    for row in rows:
        metric = metrics[tuple(row['case'])]
        if any(metric[field] != row['result'][field] for field in ('control_misses', 'control_sent')):
            raise ValueError('Report differs from native command counters')
    def site_values(cost, family):
        result = []
        for site in p['confirmation_sites']:
            group = [metrics[(site, cost, family, seed)] for seed in seeds_for(family)]
            result.append(sum(row['control_misses'] for row in group) / sum(row['control_sent'] for row in group))
        return result
    plotted = {'family_site_observed_command_loss': [], 'comparisons': report['primary']}
    for cost in costs:
        for family in families:
            observed = site_values(cost, family)
            mean = next(row['command_loss'] for row in report['means']
                        if row['family'] == family and row['handover_seconds'] == cost)
            if not math.isclose(sum(observed) / len(observed), mean, abs_tol=1e-15, rel_tol=0):
                raise ValueError('Plotted site means differ from final report')
            plotted['family_site_observed_command_loss'].append(
                {'family': family, 'handover_seconds': cost, 'sites': p['confirmation_sites'],
                 'values': observed, 'mean': mean})
        for reference in p['primary_references']:
            comparison = next(row for row in report['primary']
                              if row['reference'] == reference and row['handover_seconds'] == cost)
            paired = [a - b for a, b in zip(site_values(cost, p['candidate_family']), site_values(cost, reference))]
            if any(not math.isclose(a, b, abs_tol=1e-15, rel_tol=0)
                   for a, b in zip(paired, comparison['paired_site_differences'])):
                raise ValueError('Intervals differ from plotted raw site contrasts')
    return p, report, latency, families, rows, replay_rows, paths, plotted


def style(plt):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.titleweight': 'bold',
                         'axes.labelcolor': INK, 'text.color': INK, 'axes.edgecolor': '#BDCAD3',
                         'xtick.color': INK, 'ytick.color': INK, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'figure.facecolor': '#FAFCFD', 'axes.facecolor': 'white',
                         'savefig.facecolor': '#FAFCFD'})


def tidy(axis):
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='x', color='#E4EAEE', linewidth=.7)
    axis.set_axisbelow(True)


def evidence_figure(plt, np, p, report, families, plotted):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 3, figsize=(17.3, 10.2),
                             gridspec_kw={'width_ratios': [1.15, 1.65, 1.0]})
    fig.subplots_adjust(left=.10, right=.98, top=.84, bottom=.22, wspace=.48, hspace=.64)
    fig.suptitle('Prospective confirmation of the development-selected JEPA', x=.055, ha='left',
                 y=.975, fontsize=19, fontweight='bold')
    fig.text(.055, .931, f"{p['robots']} simulated robots × {p['duration_seconds']} s · "
             f"{len(p['confirmation_sites'])} new sites · 3 independently trained members per new family · 128 runs + 8 replays",
             fontsize=11)
    fig.text(.055, .895, 'Every registered reference remains visible; qualification is evaluated separately for each comparison.',
             fontsize=10, color='#516574')
    for row_index, cost in enumerate(p['handover_seconds']):
        loss_ax, delta_ax, status_ax = axes[row_index]
        title = f'{cost:g} s handover assumption'
        loss_ax.set_title(('A' if row_index == 0 else 'D') + '  Observed command loss\n' + title,
                          loc='left', pad=12)
        for y, family in enumerate(families):
            record = next(value for value in plotted['family_site_observed_command_loss']
                          if value['family'] == family and value['handover_seconds'] == cost)
            values = 100 * np.asarray(record['values'])
            loss_ax.scatter(values, y + np.linspace(-.11, .11, len(values)), s=23,
                            color=COLORS[family], alpha=.35, edgecolors='none')
            mean = 100 * record['mean']
            loss_ax.scatter([mean], [y], s=67, marker='D', color=COLORS[family],
                            edgecolors='white', zorder=4)
            loss_ax.annotate(f'{mean:.5f}%', (mean, y), xytext=(0, 12), textcoords='offset points',
                             ha='center', fontsize=8, color=COLORS[family])
        loss_ax.set_yticks(range(4), [LABELS[family] for family in families], fontsize=9)
        loss_ax.set_ylim(3.5, -.65); loss_ax.set_xlabel('Command deadline misses (%)')
        loss_ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        loss_ax.margins(x=.18); tidy(loss_ax)

        delta_ax.set_title(('B' if row_index == 0 else 'E') + '  Paired service improvement\n' + title,
                           loc='left', pad=12)
        references = p['primary_references']
        for y, reference in enumerate(references):
            result = next(value for value in report['primary']
                          if value['reference'] == reference and value['handover_seconds'] == cost)
            paired = -1e5 * np.asarray(result['paired_site_differences'])
            lower, upper = -1e5 * np.asarray(result['descriptive_95_interval'])[::-1]
            delta_ax.scatter(paired, y + np.linspace(-.09, .09, len(paired)), s=19,
                             color='#AEBBC4', alpha=.65, zorder=2)
            delta_ax.plot([lower, upper], [y, y], color=BLUE, lw=3,
                          solid_capstyle='round', zorder=3)
            delta_ax.scatter([-1e5 * result['mean_difference']], [y], marker='D', s=42,
                             color=INK, zorder=6)
            delta_ax.scatter([-1e5 * result['multiplicity_upper']], [y - .15],
                             marker='^', s=49, color=ORANGE, zorder=5)
            delta_ax.scatter([-1e5 * result['endpoint_sensitivity']['multiplicity_upper']], [y + .15],
                             marker='v', s=49, color=RED, zorder=5)
        delta_ax.axvline(0, color=INK, linestyle='--', linewidth=1)
        delta_ax.set_yticks(range(3), [f'vs {LABELS[reference]}' for reference in references], fontsize=9)
        delta_ax.set_ylim(2.5, -.6)
        delta_ax.set_xlabel('Avoided misses per 100,000 commands\n(reference − selected JEPA; positive favors JEPA)')
        delta_ax.margins(x=.12); tidy(delta_ax)

        status_ax.set_title(('C' if row_index == 0 else 'F') + '  Registered qualification', loc='left', pad=12)
        status_ax.set_xlim(-.45, 2.65); status_ax.set_ylim(2.5, -.6); status_ax.axis('off')
        for x, name in enumerate(('Statistics', 'Engineering', 'Qualified')):
            status_ax.text(x, -.45, name, ha='center', va='center', fontsize=8, fontweight='bold')
        for y, reference in enumerate(references):
            result = next(value for value in report['primary']
                          if value['reference'] == reference and value['handover_seconds'] == cost)
            for x, field in enumerate(('statistical_pass', 'engineering_pass', 'qualified')):
                passed = result[field]
                status_ax.text(x, y, 'PASS' if passed else 'FAIL', ha='center', va='center', fontsize=9,
                               color='#00776F' if passed else RED, fontweight='bold',
                               bbox={'boxstyle': 'round,pad=.32', 'facecolor': '#E5F3EF' if passed else '#FBECEE',
                                     'edgecolor': 'none'})
    handles = [Line2D([], [], marker='o', ls='', color='#AEBBC4', label='Eight paired sites'),
               Line2D([], [], marker='D', ls='', color=INK, label='Mean reduction'),
               Line2D([], [], color=BLUE, lw=3, label='Descriptive 95% site-bootstrap interval'),
               Line2D([], [], marker='^', ls='', color=ORANGE, label='Multiplicity-adjusted lower bound'),
               Line2D([], [], marker='v', ls='', color=RED, label='Conservative endpoint lower bound')]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.055, .112),
               ncol=3, frameon=False, fontsize=9)
    integrity = 'PASS' if report['integrity_pass'] else 'FAIL'
    runtime = 'YES' if report['any_runtime_overrun'] else 'NO'
    footer = (f"Statistics: both one-sided {100 * report['primary_upper_quantile']:.4f}% lower bounds must exceed zero; "
              f"{report['bootstrap_replicates']:,} whole-site bootstrap draws, six-comparison budget.\n"
              'Pending commands are misses for the candidate and successes for the reference in the endpoint check. '
              'Intervals condition on the three frozen members.\n'
              f'Integrity/causality/replay gate: {integrity}. Actual runtime overrun: {runtime}. '
              'Engineering covers worst-decile command loss, maximum silence, original video and bulk delivery.\n'
              'Shared procedural warehouse CAD, stationary and hidden-drift sites; simulation only. '
              'Research gates do not establish physical-car performance, production safety, EMA necessity or SOTA.')
    fig.text(.055, .104, footer, fontsize=8.1, linespacing=1.6, color='#516574', va='top')
    return fig


def timing_summary(np, p, latency, families, rows, replays):
    result = {}
    for family in families:
        names = member_names(p, family)
        selected = [row for row in rows if row['family'] == family]
        replayed = [row for row in replays if row['family'] == family]
        profiles = [latency['profiles'][name] for name in names]
        if any(profile != profiles[0] for profile in profiles):
            raise ValueError('Members do not share the registered timing envelope')
        actual = {kind: [number(job['actual_prediction_seconds']) for row in selected for job in row['jobs']
                         if job['kind'] == kind] for kind in ('cold', 'warm', 'refit')}
        if any(not values for values in actual.values()):
            raise ValueError('Missing actual confirmation timing regime')
        result[family] = {
            'calibration_max_member_p95_seconds': {kind: max(float(np.quantile(latency['raw_seconds'][name][kind], .95))
                for name in names) for kind in ('cold', 'warm', 'refit')},
            'compute_charge_seconds': profiles[0], 'shared_fifo_delay_seconds': p['extra_compute_seconds'],
            'confirmation_max_request_seconds': {kind: max(values) for kind, values in actual.items()},
            'confirmation_request_counts': {kind: len(values) for kind, values in actual.items()},
            'calibration_max_startup_seconds': max(number(value) for name in names
                for value in latency['startup_seconds_separate_from_request'][name]),
            'confirmation_max_startup_seconds': max(number(row['predictor_startup_seconds']) for row in selected),
            'replay_max_request_seconds': max(number(job['actual_prediction_seconds'])
                for row in replayed for job in row['jobs']),
            'confirmation_overruns': sum(job['runtime_overrun'] for row in selected for job in row['jobs']),
            'replay_overruns': sum(job['runtime_overrun'] for row in replayed for job in row['jobs'])}
    return result


def timing_figure(plt, np, p, report, families, timings):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.1), gridspec_kw={'width_ratios': [1.12, 1.]})
    fig.subplots_adjust(left=.11, right=.97, top=.84, bottom=.17, wspace=.27, hspace=.68)
    fig.suptitle('Measured compute, charged availability and the control path', x=.055, y=.975,
                 ha='left', fontsize=19, fontweight='bold')
    fig.text(.055, .929, 'Frozen CPU calibration is reused without retiming; actual wall times come from the completed confirmation request traces.', fontsize=11)
    fig.text(.055, .891, 'One request forecasts for all 100 robots. Startup is separate. Every family pays another 0.50 s FIFO delay.',
             fontsize=10, color='#516574')
    ax = axes[0, 0]
    y = np.arange(len(families)); height = .23
    calibration = [timings[family]['calibration_max_member_p95_seconds']['warm'] for family in families]
    observed = [timings[family]['confirmation_max_request_seconds']['warm'] for family in families]
    compute = [timings[family]['compute_charge_seconds']['warm'] for family in families]
    ax.barh(y - height, calibration, height, color=BLUE, label='Calibration p95 (max member)')
    ax.barh(y, observed, height, color=ORANGE, label='Confirmation maximum')
    ax.barh(y + height, compute, height, color='#8295A4', label='Sealed compute charge')
    ax.barh(y + height, p['extra_compute_seconds'], height, left=compute,
            color='#C8DDD9', label='Shared FIFO delay')
    ax.set_yticks(y, [LABELS[family] for family in families], fontsize=9); ax.invert_yaxis()
    ax.set_xlabel('Seconds per warm request / charged availability')
    ax.set_title('A  Warm requests: timing and charge', loc='left', pad=11); tidy(ax)
    ax.legend(loc='upper left', bbox_to_anchor=(-.02, -.23), ncol=2, fontsize=8, frameon=False)

    ax = axes[0, 1]; ax.axis('off')
    ax.set_title('B  Actual maxima and charges in every regime', loc='left', pad=11)
    values = [[LABELS[family]] + [f"{timings[family]['confirmation_max_request_seconds'][kind]:.3f} / "
               f"{timings[family]['compute_charge_seconds'][kind]:.2f}"
               for kind in ('cold', 'refit', 'warm')] for family in families]
    table = ax.table(cellText=values, colLabels=['Family', 'Cold', 'Refit', 'Warm'],
                     cellLoc='center', loc='upper center', colWidths=[.40, .20, .20, .20])
    table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1, 1.9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#E1E8ED'); cell.set_facecolor('#E9F0F4' if row == 0 else 'white')
    ax.text(0, -.04, 'Cell = actual confirmation maximum / sealed compute charge (seconds).\n'
            'Add 0.50 s FIFO delay to the charge for ready time; native slots round up.\n'
            'New members share the original twelve-arm calibration envelope.\n'
            f"Global actual-overrun veto: {'TRIGGERED' if report['any_runtime_overrun'] else 'not triggered'}.",
            transform=ax.transAxes, va='top', fontsize=9, linespacing=1.5)

    ax = axes[1, 0]
    ax.barh(y - .14, [1000 * timings[family]['calibration_max_startup_seconds'] for family in families],
            height=.27, color=BLUE, label='Calibration maximum')
    ax.barh(y + .14, [1000 * timings[family]['confirmation_max_startup_seconds'] for family in families],
            height=.27, color=ORANGE, label='Confirmation maximum')
    ax.set_yticks(y, [LABELS[family] for family in families], fontsize=9); ax.invert_yaxis()
    ax.set_xlabel('Predictor initialization / checkpoint loading (ms)')
    ax.set_title('C  Startup is separate from request compute', loc='left', pad=11); tidy(ax)
    ax.legend(loc='upper left', bbox_to_anchor=(-.02, -.20), frameon=False, fontsize=8, ncol=2)

    ax = axes[1, 1]; ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('D  Direct-horizon control path (schematic)', loc='left', pad=11)
    boxes = [(.01, .62, .28, .24, 'Delivered scans\npublic geometry\nplanned routes', '#E9F0F4'),
             (.37, .62, .28, .24, 'Map anchor +\nAP/band context\nJEPA correction', '#DDEFEA'),
             (.73, .62, .25, .24, 'Direct horizons\nmean + std\nall robots', '#DDEFEA'),
             (.73, .13, .25, .25, 'Charged FIFO\ntime-stamped\navailability', '#F7ECD9'),
             (.37, .13, .28, .25, 'Shared fixed\ncost-specific\ncontroller', '#E6E1F1'),
             (.01, .13, .28, .25, 'Native packets\nqueues, scans\nhandover costs', '#E9F0F4')]
    for x, bottom, width, height, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, bottom), width, height, boxstyle='round,pad=.015',
                                   facecolor=color, edgecolor='#A7BBC5', linewidth=.8))
        ax.text(x + width / 2, bottom + height / 2, label, ha='center', va='center', fontsize=8.2, linespacing=1.4)
    for start, end in [((.30, .74), (.35, .74)), ((.66, .74), (.71, .74)),
                       ((.855, .60), (.855, .40)), ((.71, .255), (.67, .255)),
                       ((.35, .255), (.31, .255)), ((.15, .40), (.15, .60))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=11,
                                    color='#587684', linewidth=1.2))
    ax.text(.50, .49, 'Absolute map anchor + bounded ±12 dB correction\nEach horizon is predicted directly.',
            ha='center', va='center', fontsize=7.7, color='#516574')
    fig.text(.055, .10, 'Timing summaries are descriptive maxima and calibration quantiles, not confidence intervals or device benchmarks. '
             'Calibration: one CPU thread, three repeated 24-step sequences per arm.\n'
             'Confirmation maxima use 128 scientific rollouts; eight exact replays also enter the runtime veto. '
             'Predictor startup includes initialization/loading outside request charges.\n'
             'Schematic only: the previously trained model uses simulator future-target supervision and an EMA context teacher, absent at inference. '
             'Original-ensemble comparisons cover the entire package.\n'
             'Direct horizons avoid recursive summation of model corrections; they do not guarantee bounded cumulative packet loss. '
             'Simulation only; no physical-car or production qualification.',
             fontsize=8.1, color='#516574', linespacing=1.6, va='top')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=ROOT / 'artifacts/selected_jepa')
    parser.add_argument('--output', type=Path, help='Default: FOLDER/figures')
    args = parser.parse_args()
    p, report, latency, families, rows, replays, inputs, plotted = load_inputs(args.folder)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    style(plt)
    timings = timing_summary(np, p, latency, families, rows, replays)
    figures = [('confirmation', evidence_figure(plt, np, p, report, families, plotted)),
               ('timing-and-control', timing_figure(plt, np, p, report, families, timings))]
    output = args.output or args.folder / 'figures'
    output.mkdir(parents=True, exist_ok=True)
    outputs = []
    for stem, figure in figures:
        for extension in ('svg', 'pdf', 'png'):
            path = output / f'{stem}.{extension}'
            figure.savefig(path, dpi=190, metadata={'Creator': 'plot_selected_jepa.py'})
            outputs.append(path)
        plt.close(figure)
    receipt = {'schema': 1, 'derived_only': True, 'scientific_execution': False,
               'script_sha256': digest(Path(__file__)),
               'inputs_sha256': {relative(path): digest(path) for path in sorted(set(inputs))},
               'outputs_sha256': {relative(path): digest(path) for path in outputs},
               'software': {'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
               'observed_loss_units': 'percent', 'paired_reduction_units': 'avoided misses per100000 generated commands',
               'reduction_sign': 'reference minus selected JEPA; positive favors selected JEPA',
               'intervals': {'descriptive': 'two-sided95% whole-site bootstrap conditional on the three fitted members',
                             'decision': 'one-sided1-.05/6 bounds for observed and conservative endpoint outcomes'},
               'plotted_data': plotted, 'timing_data': timings,
               'report_sha256': digest(args.folder / 'report.json'), 'limits': report['limits']}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'figures': [relative(path) for path in outputs],
                      'receipt': relative(output / 'receipt.json'), 'scientific_execution': False}))


if __name__ == '__main__':
    main()
