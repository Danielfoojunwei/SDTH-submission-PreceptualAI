"""Render completed, independently audited pilot/fast-loop reports only.

No model, simulator, dataset or scientific study package is imported. Run
after the measured study and raw audit; before the final plot-binding manifest.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('radiomap', 'frozen_fleet_jepa')
VARIANTS = ('base', 'pilot', 'fast', 'pilot_fast')
SHORT = {'radiomap': 'Map', 'frozen_fleet_jepa': 'JEPA'}
ARM = {'base': 'Base', 'pilot': '+ Delivered pilots',
       'fast': '+ Cached forecast control', 'pilot_fast': 'Both · fixed final'}
SHORT_ARM = {'base': 'Base', 'pilot': 'Pilots', 'fast': 'Fast', 'pilot_fast': 'Both'}
COLORS = {'radiomap': '#4477AA', 'frozen_fleet_jepa': '#158C85'}
INK, MUTED, ORANGE, RED, PALE = '#203343', '#657987', '#C08A36', '#B84F66', '#FAFCFD'
RF_METRIC = 'short_route_delta_rmse_db'
AUDIT_FLAGS = ('independent_pilot_payload_provenance_verified',
    'independent_fast_local_arithmetic_verified', 'independent_four_primary_bootstrap_verified',
    'report_recomputed', 'exact_native_and_matched_replays_verified',
    'no_worlds_models_training_or_timing_executed')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def number(value):
    require(not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value),
            'Figure inputs must be finite measured numbers')
    return float(value)


def close(left, right):
    return math.isclose(number(left), number(right), rel_tol=1e-10, abs_tol=1e-12)


def load_inputs(folder):
    names = ('protocol.json', 'registration/source.json', 'qualification.json',
        'completion.json', 'development.json', 'execution_order.json',
        'report.json', 'verification/raw_audit.json')
    paths = [folder / name for name in names]
    require(all(path.is_file() for path in paths), 'Complete the study and independent raw audit before plotting')
    p, reg, qual, done, dev, order, report, audit = [read(path) for path in paths]
    require(report['schema'] == 1 and report['stage'] == 'confirmation'
        and report['study'] == p['study'] and done['completed'] is True,
        'Require the completed registered confirmation report')
    require(tuple(p['families']) == FAMILIES and p['handover_seconds'] == [4.]
        and p['variants_by_cost'] == {'4.0': list(VARIANTS)}
        and p['robots'] == 100 and p['duration_seconds'] == 60,
        'Families, factorial arms or physical budget differ from registration')
    require(report['primary_multiplicity_budget'] == 4 and len(report['primary']) == 4
        and close(report['primary_upper_quantile'], 1 - .05 / 4), 'Keep all four primary slots')
    require(all(report[name] is False for name in ('hardware_qualified', 'production_qualified',
        'activated', 'model_or_controller_selection_performed', 'jepa_objective_necessity_claimed',
        'ema_necessity_claimed')) and report['new_fits'] == 0, 'Experimental claim scope changed')
    require(report['source_sha256'] == reg['source_sha256'] == done['source_sha256']
        and report['completion_sha256'] == digest(folder / 'completion.json')
        and report['qualification_sha256'] == done['qualification_sha256'] == digest(folder / 'qualification.json')
        and report['development_sha256'] == qual['development_sha256'] == digest(folder / 'development.json')
        and report['execution_order_sha256'] == done['execution_order_sha256'] == digest(folder / 'execution_order.json')
        and report['passed_development_costs'] == done['passed_costs'] == qual['passed_costs'] == dev['gate']['passed_costs'],
        'Report, source, completion or development binding differs')
    for name, expected in reg['source_sha256'].items():
        require(digest(ROOT / name) == expected and digest(folder / 'registration/source' / name) == expected,
                'Frozen source/capsule changed: ' + name)
    require(audit['schema'] == 1 and audit['report_sha256'] == digest(folder / 'report.json')
        and audit['script_sha256'] == digest(ROOT / 'scripts/audit_pilot_fastloop.py')
        and all(audit.get(flag) is True for flag in AUDIT_FLAGS), 'Independent raw audit is missing or changed')
    costs = report['passed_development_costs']
    require(costs in ([], [4.]), 'Unregistered development-passed cost')
    sites = p['confirmation_sites'] if costs else []
    require(report['sites'] == sites and report['independent_sites'] == len(sites), 'Whole-site units changed')
    cases = report['every_case']
    data = {(r['site'], r['family'], r['controller_variant']): r for r in cases}
    expected = {(site, f, v) for site in sites for f in FAMILIES for v in VARIANTS}
    require(len(cases) == len(data) == report['case_count'] and set(data) == expected, 'Incomplete factorial matrix')
    for row in cases:
        require(row['stage'] == 'confirmation' and row['handover_seconds'] == 4., 'Stage or outage differs')
        loss = number(row['control_misses']) / number(row['control_sent'])
        endpoint = (number(row['control_misses']) + number(row['pending_control_packets'])) / number(row['control_sent'])
        require(close(loss, row['command_loss']) and close(endpoint, row['command_loss_pending_as_misses'])
            and 0 <= loss <= endpoint <= 1., 'Command counters and plotted loss differ')
        require(row['pilot_reporting']['new_rf_reads'] == row['pilot_reporting']['extra_control_frames'] == 0,
                'Intervention added RF sampling or control frames')
    matched = {(r['site'], r['family']): r for r in report['matched_every_case']}
    require(len(matched) == len(report['matched_every_case']) == report['matched_case_count']
        and set(matched) == {(s, f) for s in sites for f in FAMILIES}, 'Incomplete matched RF matrix')
    for site in sites:
        left, right = [matched[site, family] for family in FAMILIES]
        require(left['matched_inputs'] == right['matched_inputs']
            and left['reference_archive_sha256'] == right['reference_archive_sha256'], 'Matched RF inputs differ')
        require(all(number(r['metrics'][RF_METRIC]) >= 0. for r in (left, right)), 'Negative matched RF error')
    return p, report, data, matched, audit, paths


def style(plt):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.labelcolor': INK, 'axes.edgecolor': '#CED9E0', 'axes.titleweight': 'bold',
        'axes.titlesize': 12, 'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': PALE,
        'savefig.facecolor': PALE, 'axes.facecolor': 'white', 'svg.fonttype': 'none',
        'pdf.fonttype': 42, 'grid.color': '#E5ECF0', 'grid.linewidth': .7})


def tidy(ax, horizontal=False):
    ax.set_axisbelow(True);ax.grid(axis='x' if horizontal else 'y', alpha=.8)
    ax.tick_params(length=0, pad=7)


def no_estimate(ax):
    ax.axis('off')
    ax.text(.5, .5, 'Confirmation not run\nRegistered development veto retained',
            ha='center', va='center', color=MUTED, fontsize=13, linespacing=1.6)


def footer(fig, report, text):
    fig.text(.055, .035, text + '\nSource / causal / paid / exact replay integrity: '
        + str(report['proof']['experimental_claim_integrity_pass'])
        + '. Simulation evidence; fixed neural weights, no hardware or production qualification.',
        color=MUTED, fontsize=8.2, linespacing=1.6, va='bottom')


def packet_figure(plt, np, p, report, data):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.3), sharey=True)
    fig.subplots_adjust(left=.085, right=.98, top=.855, bottom=.17, hspace=.38, wspace=.22)
    fig.suptitle('Do delivered pilots and cached forecasts improve packet delivery?',
        x=.055, y=.965, ha='left', fontsize=17, weight='bold')
    fig.text(.055, .925, 'Fixed 2 × 2 experiment · original forecast packages · '
             + str(report['independent_sites']) + ' paired whole-site observations', color=MUTED)
    sites = report['sites'];offsets = np.linspace(-.13, .13, len(sites)) if sites else []
    maximum = max((r['command_loss_pending_as_misses'] * 1e5 for r in data.values()), default=1.)
    for index, (variant, ax) in enumerate(zip(VARIANTS, axes.flat)):
        ax.set_title(f'{"ABCD"[index]}  {ARM[variant]}', loc='left', pad=14)
        if not sites:
            no_estimate(ax);continue
        for x, family in enumerate(FAMILIES):
            observed = np.array([data[s, family, variant]['command_loss'] for s in sites]) * 1e5
            endpoint = np.array([data[s, family, variant]['command_loss_pending_as_misses'] for s in sites]) * 1e5
            ax.bar(x, observed.mean(), width=.5, color=COLORS[family], alpha=.75, edgecolor=COLORS[family])
            ax.bar(x, endpoint.mean() - observed.mean(), bottom=observed.mean(), width=.5,
                   facecolor='white', edgecolor=INK, hatch='////', linewidth=.9, zorder=3)
            ax.scatter(x + offsets, observed, s=30, facecolor='white', edgecolor=COLORS[family], zorder=5)
        for offset, site in zip(offsets, sites):
            ax.plot([offset, 1 + offset], [data[site, f, variant]['command_loss'] * 1e5 for f in FAMILIES],
                    color='#B8C5CD', alpha=.55, linewidth=.65, zorder=1)
        ax.set_xticks([0, 1], ['RadioMap', 'Frozen JEPA']);ax.set_xlim(-.55, 1.55)
        ax.set_ylim(0, max(maximum * 1.12, .01));ax.set_ylabel('Misses per 100,000 generated commands');tidy(ax)
    fig.legend(handles=[Patch(facecolor='#8296A2', label='Observed loss · mean over sites'),
        Patch(facecolor='white', edgecolor=INK, hatch='////', label='Pending commands counted as misses'),
        Line2D([], [], color=MUTED, marker='o', markerfacecolor='white', linestyle='', label='Whole site; gray lines pair models')],
        loc='lower left', bbox_to_anchor=(.055, .102), ncol=3, frameon=False, fontsize=8.5)
    footer(fig, report, 'All arms retain latest request admission, STR off, full 4 s AP handovers, the same due-scan rule and sustained-recovery watchdog.\n'
        'Fixed arms are shown without selecting an intermediate controller. Dots represent whole sites, not robots or independently trained models.')
    return fig


def primary_figure(plt, np, report, matched):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(16.3, 7.8), gridspec_kw={'width_ratios': [1.35, 1.]})
    fig.subplots_adjust(left=.24, right=.97, top=.80, bottom=.26, wspace=.58)
    fig.suptitle('Packet benefit and forecast accuracy answer separate questions',
        x=.055, y=.965, ha='left', fontsize=18, weight='bold')
    fig.text(.055, .915, 'Four prespecified primary slots · whole-site pairing · observed and conservative endpoint bounds', color=MUTED)
    ax = axes[0];ax.set_title('A  Registered packet contrasts', loc='left', pad=16)
    packet = [r for r in report['primary'] if r['kind'] == 'packet']
    if report['sites']:
        ax.axvline(0., color=INK, linewidth=1., linestyle='--');labels = []
        for i, row in enumerate(packet):
            color = COLORS[row['candidate_family']]
            low, high = (-np.asarray(row['descriptive_95_interval']) * 1e5)[::-1]
            ax.scatter(-np.asarray(row['paired_site_differences']) * 1e5,
                i + np.linspace(-.05, .05, len(report['sites'])), color='#AABBC5', s=20)
            ax.plot([low, high], [i, i], color=color, linewidth=3., solid_capstyle='round')
            ax.scatter(-row['mean_difference'] * 1e5, i, marker='D', color=INK, s=42, zorder=4)
            ax.scatter(-row['multiplicity_upper'] * 1e5, i - .11, marker='^', color=ORANGE, s=48, zorder=4)
            ax.scatter(-row['endpoint_sensitivity']['multiplicity_upper'] * 1e5, i + .11, marker='v', color=RED, s=48, zorder=4)
            labels.append(f"{SHORT[row['candidate_family']]} {SHORT_ARM[row['candidate_variant']]} vs "
                f"{SHORT[row['reference_family']]} {SHORT_ARM[row['reference_variant']]}\nPacket benefit: {row['experimental_benefit']}")
        ax.set_yticks(range(len(packet)), labels, fontsize=9.5);ax.set_ylim(len(packet) - .5, -.5)
        ax.set_xlabel('Avoided misses per 100,000 generated commands\n(reference − candidate; positive favors candidate)', labelpad=12);tidy(ax, True)
    else:
        no_estimate(ax)
    ax = axes[1];ax.set_title('B  Matched route-change prediction', loc='left', pad=16)
    if report['sites']:
        for site in report['sites']:
            values = [matched[site, f]['metrics'][RF_METRIC] for f in FAMILIES]
            ax.plot([0, 1], values, color='#ABBBC5', alpha=.8, linewidth=1.2, zorder=1)
            for x, f in enumerate(FAMILIES):
                ax.scatter(x, values[x], color=COLORS[f], s=30, zorder=3)
        means = [np.mean([matched[s, f]['metrics'][RF_METRIC] for s in report['sites']]) for f in FAMILIES]
        ax.scatter([0, 1], means, marker='D', color=INK, s=62, zorder=4)
        rf = next(r for r in report['primary'] if r['kind'] == 'matched_rf')
        ax.set_xticks([0, 1], ['RadioMap', 'Frozen JEPA']);ax.set_xlim(-.35, 1.35);ax.set_ylim(bottom=0)
        ax.set_ylabel('Short route-delta RMSE (dB) · lower is better')
        ax.text(.02, -.21, f"Mean Map − JEPA: {-rf['mean_difference']:+.3f} dB\n"
            f"Multiplicity lower bound: {-rf['multiplicity_upper']:+.3f} dB\nMatched accuracy flag: {rf['matched_forecast_advantage']}",
            transform=ax.transAxes, fontsize=8.8, color=MUTED, va='top', linespacing=1.55);tidy(ax)
    else:
        no_estimate(ax)
    fig.legend(handles=[Line2D([], [], color=INK, marker='D', linestyle='', label='Mean reduction'),
        Line2D([], [], color=COLORS[FAMILIES[1]], linewidth=3, label='Descriptive 95% site-bootstrap interval'),
        Line2D([], [], color=ORANGE, marker='^', linestyle='', label='Multiplicity lower bound · observed'),
        Line2D([], [], color=RED, marker='v', linestyle='', label='Multiplicity lower bound · pending-as-misses')],
        loc='lower left', bbox_to_anchor=(.055, .115), frameon=False, ncol=2, fontsize=8.5)
    footer(fig, report, 'All four primary slots retain 1 − 0.05/4 bounds. Packet benefit also requires unchanged tail, silence, video and bulk engineering gates.\n'
        'Matched RF uses identical public histories/queries, horizons 1–5 s, and no additional sites. Forecast accuracy alone does not qualify a controller.')
    return fig


def resources_figure(plt, np, p, report, data):
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 3, figsize=(18., 11.6))
    fig.subplots_adjust(left=.09, right=.97, top=.85, bottom=.19, wspace=.5, hspace=.52)
    fig.suptitle('Compute, sensing and reporting costs remain visible', x=.055, y=.965, ha='left', fontsize=18, weight='bold')
    fig.text(.055, .92, 'Means per 100-robot, 60-second site · actual-cost main rollouts · repetitions add no sites', color=MUTED)
    if not report['sites']:
        for ax in axes.flat:
            no_estimate(ax)
        footer(fig, report, 'Registered development veto prevents confirmation estimates. No substitute data or intermediate arm is selected.')
        return fig
    arms = [(f, v) for f in FAMILIES for v in VARIANTS];y = np.arange(len(arms))
    labels = [f'{SHORT[f]} · {SHORT_ARM[v]}' for f, v in arms];colors = [COLORS[f] for f, v in arms]
    def values(getter):
        return np.asarray([np.mean([getter(data[s, f, v]) for s in report['sites']]) for f, v in arms])
    def horizontal(ax, value, title, xlabel, secondary=None, secondary_label=None):
        if secondary is not None:
            ax.barh(y, secondary, color='#DDE6EB', edgecolor='#A8BAC6', height=.65, label=secondary_label)
        ax.barh(y, value, color=colors, alpha=.85, height=.35 if secondary is not None else .6,
                label='Measured / consumed' if secondary is not None else None)
        ax.set_yticks(y, labels, fontsize=8.3);ax.invert_yaxis();ax.set_xlabel(xlabel, fontsize=9.)
        ax.set_title(title, loc='left', pad=13);tidy(ax, True)
    horizontal(axes[0, 0], values(lambda r: r['timing']['actual_compute_seconds_total']),
        'A  Forecast compute is paid', 'Seconds per site; includes every accepted job',
        values(lambda r: r['timing']['charged_service_seconds_total']), 'Charged forecast service')
    horizontal(axes[0, 1], values(lambda r: r['timing']['consumptions']),
        'B  Completed jobs drive planning', 'Forecast jobs per site',
        values(lambda r: r['timing']['forecast_jobs']), 'Accepted / computed')
    horizontal(axes[0, 2], values(lambda r: r['timing']['maximum_held_forecast_age_seconds']
        if r['timing']['maximum_held_forecast_age_seconds'] is not None else np.nan),
        'C  Held forecasts keep aging', 'Mean per-site maximum source age (s)\nDashed: 5 s local support; missing: no consumed forecast')
    axes[0, 2].axvline(5., color=ORANGE, linestyle='--', linewidth=1.)
    horizontal(axes[1, 0], values(lambda r: r['sensing']['scan_starts'] * p['scan_outage_seconds']),
        'D  Full scans interrupt the radio', 'Scheduled scan-outage robot-seconds per site')
    ax = axes[1, 1];scan = values(lambda r: r['pilot_reporting']['full_scan_reserved_bytes_delivered']) / 1024.
    pilot = values(lambda r: r['pilot_reporting']['pilot_bytes_delivered']) / 1024.
    ax.barh(y, scan, height=.6, color='#AEBEC8', label='Delivered full-scan reservation')
    ax.barh(y, pilot, left=scan, height=.6, color=colors, label='Delivered 43-byte pilots')
    ax.set_yticks(y, labels, fontsize=8.3);ax.invert_yaxis()
    ax.set_xlabel('Delivered telemetry KiB per site · no extra frames', fontsize=9.)
    ax.set_title('E  Existing control-report budget', loc='left', pad=13)
    tidy(ax, True)
    horizontal(axes[1, 2], values(lambda r: r['fast_local_timing']['total_seconds']) * 1000.,
        'F  Local helper work is measured', 'Helper ms/site · not independently scheduled')
    maximum = max(r['fast_local_timing']['max_seconds'] for r in data.values()) * 1000.
    calls = sum(r['fast_local_timing']['calls'] for r in data.values())
    reserved = sum(r['pilot_reporting']['reserved_bytes_attempted'] for r in data.values())
    used = sum(r['pilot_reporting']['telemetry_budget_bytes_attempted'] for r in data.values())
    controls = sum(r['pilot_reporting']['control_exchanges_attempted'] for r in data.values())
    retained = sum(r['pilot_reporting']['pilot_rows_retained'] for r in data.values())
    fig.legend(handles=[Patch(facecolor='#DDE6EB', edgecolor='#A8BAC6', label='A / B light bars: charged service / accepted jobs'),
        Patch(facecolor='#607D90', label='A / B colored bars: measured compute / consumed jobs'),
        Patch(facecolor='#AEBEC8', label='E gray: delivered full-scan reservation'),
        Patch(facecolor=COLORS[FAMILIES[1]], label='E colored: delivered 43-byte pilots')],
        loc='lower left', bbox_to_anchor=(.055, .105), frameon=False, ncol=2, fontsize=8.)
    footer(fig, report, f'{controls:,} attempted original 600-byte control exchanges; {used / 1024:,.1f} KiB attempted reports / '
        f'{reserved / 1024:,.1f} KiB existing reservation. {retained:,} pilot rows retained.\n'
        f'Fast helper: {calls:,} robot updates; largest measured call {maximum:.3f} ms. '
        'No extra RF or model calls; pilot refresh <100 ms and capture <200 ms, with association guards.\n'
        'Public pose/clock and zero intra-AP switch cost remain assumed. '
        'Forecast charge includes actual work and shared allowance. Unscheduled helper/observer CPU and modeled airtime do not establish complete-stack real-time operation.')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=ROOT / 'artifacts/pilot_fastloop')
    parser.add_argument('--output', type=Path, help='Default: FOLDER/figures')
    args = parser.parse_args();p, report, data, matched, audit, inputs = load_inputs(args.folder)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    style(plt)
    figures = [('pilot-fastloop-packet-factorial', packet_figure(plt, np, p, report, data)),
        ('pilot-fastloop-primary-and-rf', primary_figure(plt, np, report, matched)),
        ('pilot-fastloop-resources', resources_figure(plt, np, p, report, data))]
    output = args.output or args.folder / 'figures';output.mkdir(parents=True, exist_ok=True);outputs = []
    for name, fig in figures:
        for extension in ('svg', 'pdf', 'png'):
            path = output / f'{name}.{extension}'
            fig.savefig(path, dpi=190, metadata={'Creator': 'plot_pilot_fastloop.py'});outputs.append(path)
        plt.close(fig)
    receipt = {'schema': 1, 'derived_only': True, 'scientific_execution': False,
        'new_fitting_or_selection': False, 'script_sha256': digest(Path(__file__)),
        'report_sha256': digest(args.folder / 'report.json'),
        'inputs_sha256': {relative(path): digest(path) for path in inputs},
        'outputs_sha256': {relative(path): digest(path) for path in outputs},
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'primary_comparisons': report['primary'], 'all_four_primary_supported': report['all_four_primary_supported'],
        'secondary_components_descriptive_only': True, 'independent_unit': 'whole site',
        'independent_sites': report['independent_sites'], 'matched_forecast_advantage': report['matched_forecast_advantage'],
        'uncertainty': 'Four-slot multiplicity; observed AND pending endpoint for packets; matched RF is fully observed.',
        'audit_flags': {name: audit[name] for name in AUDIT_FLAGS},
        'no_physical_or_production_qualification': True, 'limits': report['limits']}
    receipt_path = output / 'receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'figures': [relative(p) for p in outputs], 'receipt': relative(receipt_path), 'scientific_execution': False}))


if __name__ == '__main__':
    main()
