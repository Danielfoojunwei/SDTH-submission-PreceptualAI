"""Render completed cost-aware recovery evidence from passive archives only.

No scientific package, model, simulator, dataset builder or training runner is
imported. Missing/changed evidence fails closed. Figures are derived artifacts.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right
from fractions import Fraction
import gzip
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('radiomap', 'frozen_fleet_jepa')
LABELS = {'radiomap': 'RadioMap', 'frozen_fleet_jepa': 'Frozen JEPA ensemble'}
VARS = {'legacy': 'Legacy', 'repair': 'Cost-aware recovery'}
COLORS = {'legacy': '#8495A5', 'repair': '#007E83'}
INK, BLUE, TEAL, ORANGE, RED, PALE = '#203343', '#3278A5', '#007E83', '#BB862F', '#BC4C5E', '#F8FBFD'
REPLAY_FIELDS = ('result', 'forecast_mean_sha256', 'forecast_std_sha256', 'telemetry',
                 'decision_trace', 'admission_trace', 'local_scan_observability', 'action_trace_sha256')
SCHEDULE_FIELDS = ('decision', 'kind', 'history_cutoff_exclusive', 'ready_slot',
    'actual_prediction_seconds', 'actual_issue_seconds', 'actual_compute_seconds',
    'profile_seconds', 'charged_compute_seconds', 'charged_seconds', 'shared_extra_seconds')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value: raise ValueError(message)


def relative(path):
    try: return str(Path(path).resolve().relative_to(ROOT))
    except ValueError: return str(Path(path).resolve())


def finite(value):
    require(not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value), 'Expected a finite measured number')
    return float(value)


def close(a, b):
    return math.isclose(finite(a), finite(b), rel_tol=1e-11, abs_tol=1e-12)


def variants(cost):
    require(cost == 4., 'Cost-aware recovery is a four-second-only study')
    return ('legacy', 'repair')


def keys(p, stage, costs):
    return [(stage, site, float(cost), family, variant) for site in p[stage + '_sites']
            for cost in costs for family in FAMILIES for variant in variants(cost)]


def case_name(key):
    stage, site, cost, family, variant = key
    return f'{stage}-{site}-h{round(cost * 1000):04d}-{family}-{variant}'


def audit_jobs(row, p, profile):
    """Check actual paid completion and every expected consumption event."""
    jobs = row['jobs']; require(len(jobs) == p['duration_seconds'], 'Incomplete request ledger')
    previous = 0.; ready = []
    for index, job in enumerate(jobs):
        require(job['decision'] == index and job['history_cutoff_exclusive'] == 60 + index,
                'Forecast consumes a noncausal prefix')
        actual = max(finite(job['actual_prediction_seconds']), finite(job['actual_issue_seconds']))
        compute = max(profile[job['kind']], actual)
        require(actual >= 0 and close(job['actual_compute_seconds'], actual)
                and close(job['charged_compute_seconds'], compute)
                and close(job['charged_seconds'], compute + p['extra_compute_seconds'])
                and job['runtime_overrun'] is (actual > profile[job['kind']]),
                'Actual forecast work was not paid or capacity flag changed')
        previous = max(float(index), previous) + job['charged_seconds']
        require(job['ready_slot'] == math.ceil(previous * 100 - 1e-10), 'Actual-cost FIFO differs')
        ready.append(job['ready_slot'])
    expected, last = [], -1
    for slot in range(p['duration_seconds'] * 100):
        index = bisect_right(ready, slot) - 1
        if index > last:
            expected.append({'slot': slot, 'forecast_decision': index, 'ready_slot': ready[index],
                             'age_seconds': slot / 100 - index})
            last = index
    require(row['forecast_consumption'] == expected, 'Consumed forecast precedes completion or is omitted')
    require(row['runtime_overrun'] is any(job['runtime_overrun'] for job in jobs)
            and row['paid_cost_and_consumption_pass'] is True, 'Paid/capacity summary changed')


def packet_metrics(row, p):
    detail, flow = row['result']['packet_detail'], row['result']['packet_accounting']
    sent, misses, gap = detail['sent'], detail['misses'], detail['max_gap']
    pending = [int(age >= 0) for age in detail['pending_control_packet_age_slots']]
    require(len(sent) == len(misses) == len(gap) == len(pending) == p['robots']
            and all(type(x) is int and x == 20 * p['duration_seconds'] for x in sent)
            and all(type(x) is int and 0 <= x <= sent[i] for i, x in enumerate(misses)),
            'Native per-robot command counters differ')
    total_sent, total_miss = sum(sent), sum(misses)
    require(row['result']['control_sent'] == total_sent and row['result']['control_misses'] == total_miss,
            'Native global counters differ from robot counters')
    decile = max(1, p['robots'] // 10)
    rates = [m / s for m, s in zip(misses, sent)]
    worst = sum(sorted(rates)[-decile:]) / decile
    pending_worst = sum(sorted((m + q) / s for m, q, s in zip(misses, pending, sent))[-decile:]) / decile
    return {'control_sent': total_sent, 'control_misses': total_miss,
        'command_loss': total_miss / total_sent,
        'command_loss_pending_as_misses': (total_miss + sum(pending)) / total_sent,
        'worst_decile_loss': worst, 'worst_decile_pending_as_misses': pending_worst,
        'max_silence_seconds': max(gap) / 100,
        'video_vs_original': flow['video_delivered_bits'] / max(flow['video_requested_bits'], 1e-9),
        'bulk_mbps': flow['bulk_delivered_bits'] / p['duration_seconds'] / 1e6,
        'roams': row['result']['roams']}


def load_inputs(folder):
    required = ('protocol.json', 'registration/source.json', 'registration/git.json',
                'development.json', 'qualification.json', 'completion.json', 'report.json')
    paths = [folder / name for name in required]
    require(all(path.is_file() for path in paths), 'Finish the registered pipeline before rendering')
    p, reg, meta, dev, qualification, completion, report = [read(path) for path in paths]
    costs = qualification['passed_costs']
    require(tuple(p['families']) == FAMILIES and p['handover_seconds'] == [4.]
            and p['variants_by_cost'] == {'4.0': ['legacy', 'repair']}
            and p['robots'] == 100 and p['duration_seconds'] == 60 and p['bootstrap_replicates'] == 20000
            and close(p['extra_compute_seconds'], .5)
            and len(p['confirmation_sites']) == 8 and len(p['development_sites']) == 2
            and len(set(p['confirmation_sites'] + p['development_sites'])) == 10
            and set(costs) <= {4.} and len(costs) == len(set(costs)), 'Protocol/site/cost identity changed')
    require(report['source_sha256'] == completion['source_sha256'] == reg['source_sha256']
            and report['completion_sha256'] == digest(folder / 'completion.json')
            and report['development_sha256'] == qualification['development_sha256'] == digest(folder / 'development.json')
            and completion['qualification_sha256'] == report['qualification_sha256'] == digest(folder / 'qualification.json')
            and report['passed_development_costs'] == completion['passed_costs'] == costs
            and dev['gate']['passed_costs'] == costs and report['development_gate'] == dev['gate']
            and completion['completed'] is True,
            'Final source/permission/report/completion identities differ')
    require(report['primary_multiplicity_budget'] == 3 and len(report['primary']) == 3
            and close(report['primary_upper_quantile'], 1 - .05 / 3)
            and report['production_qualified'] is False and report['hardware_qualified'] is False
            and report['ema_necessity_claimed'] is False and report['activated'] is False,
            'Final claim scope changed')
    execution_order = folder / 'execution_order.json'
    require(execution_order.is_file()
            and completion['execution_order_sha256'] == report['execution_order_sha256']
            == dev['execution_order_sha256'] == digest(execution_order), 'Registered execution order changed')
    paths.append(execution_order)
    for name, expected in reg['source_sha256'].items():
        require(digest(ROOT / name) == expected
                and digest(folder / 'registration/source' / name) == expected, 'Frozen source/capsule/input changed: ' + name)
    latency_path = ROOT / 'artifacts/semantic_jepa/latency.json'; latency = read(latency_path)
    paths.append(latency_path)
    profiles = latency['profiles']
    main_keys = keys(p, 'development', p['handover_seconds']) + keys(p, 'confirmation', costs)
    repeats = [key for key in keys(p, 'confirmation', costs) if key[1] == p['confirmation_sites'][0]]
    row_data, main_fingerprints, offline_checks, fresh = {}, {}, [], []
    metrics = {(r['stage'], r['site'], r['handover_seconds'], r['family'], r['controller_variant']): r
               for r in report['every_case']}
    require(set(metrics) == set(keys(p, 'confirmation', costs)), 'Final metric matrix incomplete')
    for subfolder, expected, replay, live in [('cases', main_keys, False, False),
            ('replays', repeats, True, False), ('live_replications', repeats, False, True)]:
        directory = folder / subfolder
        actual = {path.name for path in directory.iterdir() if path.is_file()
                  and not path.name.startswith('.') and not path.name.endswith(('.lock', '.part'))} if directory.exists() else set()
        require(actual == {case_name(k) + suffix for k in expected for suffix in ('.json.gz', '.receipt.json')},
                'Unregistered/incomplete archive matrix: ' + subfolder)
        for key in expected:
            path = directory / (case_name(key) + '.json.gz'); receipt_path = path.with_name(case_name(key) + '.receipt.json')
            receipt = read(receipt_path); row = json.loads(gzip.decompress(path.read_bytes()))
            mode = 'recorded_schedule_offline' if replay else 'actual_cost'
            require(receipt['archive_sha256'] == digest(path) and receipt['result_sha256'] == value_digest(row['result'])
                    and receipt['case'] == row['case'] == list(key) and row['completed'] is True
                    and row['replay'] is replay and row['live_replication'] is live
                    and receipt['replay'] is replay and receipt['live_replication'] is live
                    and row['accounting_mode'] == receipt['accounting_mode'] == mode
                    and row['stage'] == key[0] and row['site'] == key[1] and row['handover_seconds'] == key[2]
                    and row['family'] == key[3] and row['controller_variant'] == key[4]
                    and row['training_seed'] is None and row['simulation_only'] is True
                    and row['production_qualified'] is False and row['hardware_qualified'] is False
                    and row['checkpoint_sha256'] == reg['checkpoint_sha256']
                    and row['source_sha256'] == reg['source_sha256']
                    and row['registration_sha256'] == digest(folder / 'registration/source.json')
                    and row['protocol_sha256'] == digest(folder / 'protocol.json')
                    and row['latency_sha256'] == digest(latency_path), 'Raw archive identity changed: ' + path.name)
            profile = profiles[key[3]]
            if not replay: audit_jobs(row, p, profile)
            require(row['causality_pass'] is True and all(
                e['measurement_slot'] < e['complete_slot'] <= e['delivery_slot']
                and e['first_usable_decision'] == e['delivery_slot'] // 100 + 1 for e in row['telemetry']['events']),
                'Delivered telemetry is noncausal')
            fingerprint = {field: value_digest(row[field]) for field in REPLAY_FIELDS}
            if not replay and not live:
                main_fingerprints[key] = {'fields': fingerprint, 'jobs': row['jobs'],
                    'consumption': row['forecast_consumption'], 'archive_sha': digest(path)}
                values = packet_metrics(row, p)
                if key[0] == 'confirmation':
                    require(all(close(metrics[key][field], value) for field, value in values.items() if field != 'roams'),
                            'Report differs from native counters')
                execution = row['result']['repair_execution']
                require(execution['emergency_recovery_enabled'] is (key[4] == 'repair'), 'Recovery flag differs from registered variant')
                for event in execution['emergency_handovers']:
                    require(event['available_silence_slots'] >= event['handover_threshold_slots'] == 420
                        and event['available_silence_slots_after'] == 0
                        and event['silence_slots'] >= 20 and event['current_rssi_dbm'] < -75
                        and event['neighbor_gain_db'] >= 8 and event['neighbor_rssi_dbm'] >= -90
                        and close(event['neighbor_gain_db'], event['neighbor_rssi_dbm'] - event['current_rssi_dbm'])
                        and event['neighbor_heard'] is True
                        and 0 <= event['neighbor_capture_age_slots'] <= event['neighbor_max_age_slots']
                        and event['neighbor_capture_age_slots'] == event['slot'] - event['neighbor_capture_slot']
                        and event['outage_until_slot'] - event['slot'] == round(key[2] * 100)
                        and event['pending_cadence_target_after'] == -1, 'A logged recovery violates its observed trigger or outage')
                illustration = None
                if key[4] == 'repair':
                    require(execution['handover_threshold_slots'] == 420, 'Available-radio threshold differs')
                    for check in execution['watchdog_checks']:
                        require(type(check['available_silence_slots']) is int and check['available_silence_slots'] >= 0
                            and check['handover_threshold_slots'] == 420
                            and (not check['recovery_triggered'] or check['available_silence_slots'] >= 420),
                            'Watchdog used insufficient available-radio evidence')
                    snapshots = row['result']['packet_detail']['cumulative']
                    require([s['slot_end'] for s in snapshots] == list(range(100, p['duration_seconds']*100+1, 100)),
                        'Available-radio snapshots incomplete')
                    actions = {a['slot_end']: a for a in row['result']['packet_detail']['action_trace']}
                    for snap in snapshots:
                        require(len(snap['available_silence_slots']) == p['robots'], 'Available-evidence snapshot axes differ')
                        action = actions[snap['slot_end']]
                        for robot, available in enumerate(snap['available_silence_slots']):
                            require(type(available) is int and available >= 0
                                and available <= snap['current_gap'][robot]
                                and (action['primary_unavailable_until_slot'][robot] <= snap['slot_end']-1 or available == 0),
                                'Known handover outage was counted as available-radio failure')
                    if execution['emergency_handovers']:
                        first = min(execution['emergency_handovers'], key=lambda e:(e['slot'], e['robot']))
                        robot = first['robot']
                        illustration = {'case': list(key), 'robot': robot, 'first_event': first,
                            'sample_scope': 'actual one-second snapshots, first emergency in canonical case order; illustrative only',
                            'snapshots': [{'second':snap['slot_end']/100,
                                'available_silence_slots':snap['available_silence_slots'][robot],
                                'raw_control_silence_slots':snap['current_gap'][robot]} for snap in snapshots],
                            'emergency_events':[e for e in execution['emergency_handovers'] if e['robot']==robot],
                            'watchdog_scans':[e for e in execution['emergency_scans'] if e['robot']==robot]}
                row_data[key] = {**values, 'illustration': illustration, 'jobs': [{name: job[name] for name in
                    ('kind', 'actual_compute_seconds', 'charged_compute_seconds', 'charged_seconds', 'profile_seconds', 'runtime_overrun')}
                    for job in row['jobs']], 'startup_seconds': row['predictor_startup_seconds'],
                    'forecast_age_max': max((e['age_seconds'] for e in row['forecast_consumption']), default=None),
                    'emergency_handovers': len(execution['emergency_handovers']),
                    'emergency_scans': len(execution['emergency_scans']),
                    'watchdog_checks': len(execution['watchdog_checks']),
                    'scans_started': row['telemetry']['scans_started']}
            elif replay:
                original = main_fingerprints[key]
                schedule = all(all(a.get(field) == b.get(field) for field in SCHEDULE_FIELDS)
                               for a, b in zip(row['jobs'], original['jobs'], strict=True))
                exact = (fingerprint == original['fields'] and schedule
                    and row['forecast_consumption'] == original['consumption']
                    and row['source_main_archive_sha256'] == original['archive_sha']
                    and row['recorded_schedule_sha256'] == value_digest(original['jobs'])
                    and all(j['matches_recorded_forecast'] is True and j['offline_recorded_schedule'] is True
                            and j['completion_uses_actual_elapsed'] is False for j in row['jobs']))
                offline_checks.append({'case': list(key), 'exact': exact})
            else:
                require(row['source_main_archive_sha256'] == main_fingerprints[key]['archive_sha'],
                        'Fresh replication changed its main-case binding')
                original = row_data[key]; now = packet_metrics(row, p)
                fresh.append({'case': list(key), 'miss_difference': now['control_misses'] - original['control_misses'],
                    'changed_ready_jobs': sum(a['ready_slot'] != b['ready_slot'] for a, b in
                                              zip(main_fingerprints[key]['jobs'], row['jobs'], strict=True))})
            paths.extend((path, receipt_path))
    expected_costs = []
    for cost in p['handover_seconds']:
        severe = False
        for family in FAMILIES:
            rates = {}
            for variant in ('legacy', 'repair'):
                part = [row_data['development', site, cost, family, variant] for site in p['development_sites']]
                rates[variant] = Fraction(sum(r['control_misses'] for r in part), sum(r['control_sent'] for r in part))
            severe |= rates['repair'] - rates['legacy'] > Fraction(str(p['development_severe_regression_absolute']))
        require(next(r['severe_regression'] for r in dev['gate']['per_cost'] if r['handover_seconds'] == cost) is bool(severe),
                'Development regression status differs from exact native counters')
        if not severe: expected_costs.append(cost)
    require(costs == expected_costs, 'Confirmation subset differs from mechanically valid development veto')
    require(completion['case_count'] == report['case_count'] == len(keys(p, 'confirmation', costs))
            and completion['offline_replay_count'] == completion['fresh_live_replication_count'] == len(repeats)
            and completion['all_offline_exact'] is all(r['exact'] for r in offline_checks)
            and {tuple(r['case']): r['exact'] for r in completion['offline_comparisons']} ==
                {tuple(r['case']): r['exact'] for r in offline_checks}, 'Completion/replay claims differ from raw archives')
    require(completion['all_fresh_paid_cost_checks_pass'] is True
            and report['proof']['source_and_archive_pass'] is True
            and report['proof']['mechanical_integrity_pass'] is True
            and report['proof']['paid_cost_and_consumption_pass'] is True
            and report['proof']['causality_pass'] is True
            and report['proof']['exact_offline_replay_pass'] is all(r['exact'] for r in offline_checks)
            and report['proof']['experimental_claim_integrity_pass'] is all(r['exact'] for r in offline_checks),
            'Report proof differs from actual namespaces')
    parent_folder = ROOT / 'artifacts/jepa_repair'
    parent_paths = [parent_folder/'protocol.json', parent_folder/'report.json', parent_folder/'figures/confirmation.svg']
    require(all(path.is_file() for path in parent_paths), 'Retain the prior failed controller evidence and figure')
    parent_p, parent_report = read(parent_paths[0]), read(parent_paths[1])
    require(not set(p['development_sites']+p['confirmation_sites']) &
            set(parent_p['development_sites']+parent_p['confirmation_sites']), 'New study reuses parent sites')
    prior = next(g for g in parent_report['development_gate']['per_cost'] if g['handover_seconds']==4.)
    require(prior['status']=='development_severe_regression_veto' and 4. not in parent_report['passed_development_costs'],
            'Prior aggressive slow-cost failure is no longer preserved')
    parent = {'figure':relative(parent_paths[2]), 'figure_sha256':digest(parent_paths[2]),
        'report_sha256':digest(parent_paths[1]), 'status':prior['status'],
        'loss_difference_pp':{r['family']:100*r['repair_minus_legacy'] for r in prior['family_checks']},
        'new_sites_disjoint_from_parent':True}
    paths.extend(parent_paths)
    return p, report, profiles, row_data, paths, offline_checks, fresh, parent


def validate_statistics(np, p, report, data):
    """Derive all plotted means/contrasts from native whole-site counters."""
    sites = p['confirmation_sites']; rng = np.random.default_rng(p['bootstrap_seed'])
    indices = rng.integers(0, len(sites), size=(p['bootstrap_replicates'], len(sites)))
    values = lambda cost, arm, field: np.asarray([data['confirmation', site, cost, *arm][field] for site in sites])
    for row in report['means']:
        arm = row['family'], row['controller_variant']; cost = row['handover_seconds']
        require(close(row['command_loss'], values(cost, arm, 'command_loss').mean()), 'Plotted mean differs')
    expected = {(float(cost), label) for cost in p['handover_seconds'] for label in
                ('repair_minus_legacy:radiomap', 'repair_minus_legacy:frozen_fleet_jepa', 'repaired_jepa_minus_repaired_radiomap')}
    require({(r['handover_seconds'], r['contrast_id']) for r in report['primary']} == expected,
            'A registered contrast is missing or substituted')
    require(report['all_three_experimental_benefits'] is all(r['experimental_benefit'] for r in report['primary']),
            'Whole-study three-comparison flag differs')
    for row in report['primary']:
        label = row['contrast_id']
        if label.startswith('repair_minus_legacy:'):
            family = label.split(':', 1)[1]
            require((row['candidate_family'], row['candidate_variant'], row['reference_family'], row['reference_variant'])
                    == (family, 'repair', family, 'legacy'), 'Repair contrast changed arms')
        else:
            require((row['candidate_family'], row['candidate_variant'], row['reference_family'], row['reference_variant'])
                    == ('frozen_fleet_jepa', 'repair', 'radiomap', 'repair'), 'Model contrast changed arms')
        require((row['status'] == 'tested') is (row['handover_seconds'] in report['passed_development_costs']),
                'Tested versus development-vetoed cost differs')
        if row['status'] != 'tested':
            require(row['mean_difference'] is None and row['experimental_benefit'] is False,
                    'Unrun contrast is represented as an outcome'); continue
        cost = row['handover_seconds']; candidate = row['candidate_family'], row['candidate_variant']
        reference = row['reference_family'], row['reference_variant']
        difference = values(cost, candidate, 'command_loss') - values(cost, reference, 'command_loss')
        endpoint = values(cost, candidate, 'command_loss_pending_as_misses') - values(cost, reference, 'command_loss')
        draws, pending_draws = difference[indices].mean(1), endpoint[indices].mean(1)
        interval = np.quantile(draws, [.025, .975]); upper = np.quantile(draws, p['primary_upper_quantile'])
        endpoint_upper = np.quantile(pending_draws, p['primary_upper_quantile'])
        require(len(row['paired_site_differences']) == len(sites) and len(row['descriptive_95_interval']) == 2,
            'Paired observation/interval axes differ')
        require(close(row['mean_difference'], difference.mean()) and all(close(a, b) for a, b in zip(row['paired_site_differences'], difference))
            and all(close(a, b) for a, b in zip(row['descriptive_95_interval'], interval))
            and close(row['multiplicity_upper'], upper)
            and close(row['endpoint_sensitivity']['multiplicity_upper'], endpoint_upper)
            and row['statistical_pass'] is bool(upper < 0 and endpoint_upper < 0), 'Primary bootstrap differs from paired native counters')
        engineering = next(g for g in report['engineering_gates'] if g['contrast_id'] == row['contrast_id'] and g['handover_seconds'] == cost)
        candidate_worst = values(cost, candidate, 'worst_decile_pending_as_misses').mean()
        reference_worst = values(cost, reference, 'worst_decile_loss').mean()
        checks = {'worst_decile': candidate_worst <= reference_worst + p['worst_decile_margin'],
            'maximum_silence': values(cost, candidate, 'max_silence_seconds').max() <= values(cost, reference, 'max_silence_seconds').max() + p['silence_margin_seconds'],
            'original_video': values(cost, candidate, 'video_vs_original').mean() >= values(cost, reference, 'video_vs_original').mean() - p['video_margin'],
            'bulk_delivery': values(cost, candidate, 'bulk_mbps').mean() >= p['bulk_ratio'] * values(cost, reference, 'bulk_mbps').mean()}
        require(engineering['checks'] == checks and row['engineering_pass'] is bool(all(checks.values()))
            and row['experimental_benefit'] is bool(row['statistical_pass'] and row['engineering_pass']
                  and report['proof']['experimental_claim_integrity_pass']), 'Engineering/benefit flag differs')


def style(plt):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'text.color': INK,
        'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#C4D2DD',
        'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'axes.facecolor': 'white',
        'figure.facecolor': PALE, 'savefig.facecolor': PALE, 'svg.fonttype': 'none'})


def tidy(ax):
    ax.grid(axis='x', color='#DCE5EB', alpha=.7, linewidth=.7); ax.set_axisbelow(True)
    ax.spines['left'].set_color('#D3DFE7'); ax.tick_params(length=3)


def primary_label(row):
    if row['contrast_id'] == 'repaired_jepa_minus_repaired_radiomap': return 'Repaired JEPA vs repaired Map'
    return 'Repair benefit: ' + ('JEPA' if row['candidate_family'] == 'frozen_fleet_jepa' else 'RadioMap')


def status_panel(ax, records):
    ax.axis('off'); ax.set_xlim(0, 3.3); ax.set_ylim(-.5, 3.2)
    for index, title in enumerate(('Statistics', 'Engineering', 'Benefit')):
        ax.text(index + .42, 2.95, title, ha='center', fontsize=9, fontweight='bold')
    for index, row in enumerate(records):
        for column, field in enumerate(('statistical_pass', 'engineering_pass', 'experimental_benefit')):
            unrun = row['status'] != 'tested'; passed = row[field] is True
            label, color, face = ('UNRUN', '#6F7F8B', '#EDF1F4') if unrun else (
                ('PASS', TEAL, '#E3F2ED') if passed else ('FAIL', RED, '#F9EDEF'))
            ax.text(column + .42, 2.45 - index * .95, label, ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color,
                    bbox={'boxstyle': 'round,pad=.22', 'facecolor': face, 'edgecolor': 'none'})


def development_veto_note(p, report):
    gate = report['development_gate']['per_cost'][0]
    rates = {r['family']: r['repair_minus_legacy'] * 100 for r in gate['family_checks']}
    reason = 'SEVERE-REGRESSION VETO' if gate['severe_regression'] else 'MECHANICAL VETO'
    return (reason + '\nTwo development sites; repair − legacy loss\n'
            + f"RadioMap {rates['radiomap']:+.3f} pp · JEPA {rates['frozen_fleet_jepa']:+.3f} pp\n"
            + f"Registered stop threshold: +{p['development_severe_regression_absolute'] * 100:.3f} pp\n"
            + '4 s confirmation cases were not run.')


def parent_reference(fig, parent, y, fontsize=8.5):
    difference = parent['loss_difference_pp']
    fig.text(.055, y, 'PRIOR AGGRESSIVE 4 s REPAIR FAILED DEVELOPMENT: '
             f"RadioMap {difference['radiomap']:+.3f} pp · JEPA {difference['frozen_fleet_jepa']:+.3f} pp. "
             'That result remains unchanged.', fontsize=fontsize, color=RED)
    fig.text(.055, y - .024, 'This independent cost-aware study uses disjoint new site IDs. '
             'Prior failed study: jepa_repair/figures/confirmation.svg',
             fontsize=fontsize, color=BLUE, url='../../jepa_repair/figures/confirmation.svg')


def evidence_figure(plt, np, p, report, data, parent):
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(17.4, 8.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.10, 1.55, .85], left=.12, right=.98,
                        top=.78, bottom=.40, wspace=.56)
    fig.suptitle('Cost-aware local recovery with a paid 4 s handover', x=.055, y=.977,
                 ha='left', fontsize=19, fontweight='bold')
    fig.text(.055, .926, f"100 simulated robots × 60 s · {report['independent_sites']} tested confirmation sites · "
             f"{report['case_count']} factual runs · original fixed weights", fontsize=11.5)
    fig.text(.055, .876, 'Ordinary legacy scoring, admission and neighbor caches are preserved in both arms. '
             'The recovery gate requires 420 available-radio silent slots.', fontsize=10.2, color='#516574')
    records = report['primary']; tested = 4. in report['passed_development_costs']
    ax = fig.add_subplot(gs[0, 0]); ax.set_title('A  Observed command loss\nFour-second handover assumption', loc='left', pad=12)
    if tested:
        arm_list = [(family, variant) for family in FAMILIES for variant in variants(4.)]
        for index, arm in enumerate(arm_list):
            values = np.asarray([data['confirmation', site, 4., *arm]['command_loss'] * 100 for site in p['confirmation_sites']])
            color = COLORS[arm[1]]
            ax.scatter(values, index + np.linspace(-.06, .06, len(values)), s=23, alpha=.45, color=color, edgecolors='none')
            ax.scatter(values.mean(), index, s=58, marker='D', color=color, edgecolors=PALE, linewidths=.6, zorder=4)
            ax.annotate(f'{values.mean():.4f}%', (values.mean(), index), xytext=(0, 9), textcoords='offset points', ha='center', color=color, fontsize=8.5)
        ax.set_yticks(range(4), [('Map' if f == 'radiomap' else 'JEPA') + ' · ' + ('Legacy' if v == 'legacy' else 'Recovery') for f, v in arm_list], fontsize=9)
        ax.set_ylim(3.65, -.65); ax.set_xlabel('Commands missing deadline (%)'); tidy(ax)
    else:
        ax.axis('off'); ax.text(.5, .5, development_veto_note(p, report), transform=ax.transAxes,
            ha='center', va='center', color='#657684', fontsize=9.6, linespacing=1.7)
    ax = fig.add_subplot(gs[0, 1]); ax.set_title('B  Paired service reduction\nPositive favors the named candidate', loc='left', pad=12)
    if tested:
        ax.axvline(0, color=INK, linestyle='--', linewidth=1.)
        for index, row in enumerate(records):
            difference = -np.asarray(row['paired_site_differences']) * 1e5
            low, high = (-np.asarray(row['descriptive_95_interval']) * 1e5)[::-1]
            color = TEAL if row['candidate_family'] == 'frozen_fleet_jepa' else BLUE
            ax.scatter(difference, index + np.linspace(-.055, .055, len(difference)), s=23, color='#AAB9C2', alpha=.7, zorder=2)
            ax.plot([low, high], [index, index], color=color, linewidth=3, solid_capstyle='round')
            ax.scatter(-row['mean_difference'] * 1e5, index, marker='D', s=48, color=INK, zorder=4)
            ax.scatter(-row['multiplicity_upper'] * 1e5, index - .11, marker='^', s=50, color=ORANGE, zorder=4)
            ax.scatter(-row['endpoint_sensitivity']['multiplicity_upper'] * 1e5, index + .11, marker='v', s=50, color=RED, zorder=4)
        ax.set_yticks(range(3), [primary_label(r) for r in records], fontsize=8.6)
        ax.set_ylim(2.6, -.6); tidy(ax)
        ax.set_xlabel('Avoided misses per 100,000 generated commands\n(reference − candidate)')
    else:
        ax.set_yticks(range(3), [primary_label(r) for r in records], fontsize=8.6)
        ax.set_ylim(2.6, -.6); ax.set_xticks([])
        ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False)
        ax.text(.5, .5, 'No held-out effect estimates\n4 s confirmation cases were not run.',
            transform=ax.transAxes, ha='center', color='#657684', linespacing=1.7)
        ax.set_xlabel('All three registered primary contrasts remain untested')
    ax = fig.add_subplot(gs[0, 2]); status_panel(ax, records)
    ax.set_title('C  Registered gates', loc='left', pad=12)
    handles = [Line2D([], [], color='#AAB9C2', marker='o', linestyle='', label='Paired whole-site observations'),
        Line2D([], [], color=INK, marker='D', linestyle='', label='Mean reduction'),
        Line2D([], [], color=TEAL, linewidth=3, label='Descriptive 95% site-bootstrap interval'),
        Line2D([], [], color=ORANGE, marker='^', linestyle='', label='Multiplicity-adjusted lower bound'),
        Line2D([], [], color=RED, marker='v', linestyle='', label='Conservative endpoint lower bound')]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.055, .275), frameon=False, ncol=3, fontsize=8.3)
    fig.text(.055, .234, 'All three one-sided primary comparisons retain the 1 − 0.05/3 bound; both reduction lower bounds must exceed zero. '
             'Intervals: 20,000 whole-site bootstrap draws, conditional on fixed weights.\n'
             'Endpoint sensitivity counts pending commands as candidate misses and reference successes. '
             'Engineering gates cover worst-decile loss, silence, original video and bulk.\n'
             f"Source / causal / paid completion / offline integrity: {report['proof']['experimental_claim_integrity_pass']}. "
             'Measured profile exceedances remain paid capacity diagnostics.\n'
             'Whole controller-package effects do not establish JEPA-objective necessity, hardware safety, production readiness or SOTA.',
             fontsize=8.1, color='#516574', linespacing=1.65, va='top')
    parent_reference(fig, parent, .078)
    return fig


def architecture(ax):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    boxes = [(.01, .71, .22, .22, 'Delivered RF reports\npublic routes + geometry', '#E9F0F4'),
        (.30, .71, .26, .22, 'Fixed RadioMap / JEPA\nactual-cost FIFO forecast', '#E3E9F4'),
        (.64, .71, .34, .22, 'Ordinary legacy scores / caches\nall-handover admission preserved', '#DDF0EC'),
        (.01, .15, .25, .26, 'Current pilots + fresh heard scan\n20-slot silence: due scan\n90 ms actual scan outage', '#E9F0F4'),
        (.32, .15, .28, .26, '420 AVAILABLE-radio silent slots\nreset: delivery, association, own outage\npause: scans', '#DDF0EC'),
        (.66, .15, .32, .26, 'Fresh neighbor ≥8 dB better\npaid full 4 s emergency move\nclear stale cadence target', '#F6EDDA')]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',facecolor=color,edgecolor='#ADC5D3',linewidth=1.))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8.5,linespacing=1.5)
    for start, end in [((.235,.82),(.295,.82)),((.565,.82),(.635,.82)),
            ((.267,.28),(.315,.28)),((.607,.28),(.655,.28)),((.82,.415),(.82,.70))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=10,color='#567A8A',linewidth=1.4))
    ax.text(.49,.54,'Local completion is required before neighbor use.\nKnown own downtime cannot justify another move.',
        ha='center',va='center',fontsize=8.5,color='#516574',linespacing=1.5)


def operations_figure(plt, np, p, report, profiles, data, offline, fresh, parent):
    fig = plt.figure(figsize=(17.4, 11.1))
    gs = fig.add_gridspec(2, 3, left=.07, right=.98, top=.84, bottom=.205, hspace=.66, wspace=.47)
    fig.suptitle('Available-radio evidence, actual recovery and paid completion', x=.055, y=.978,
                 ha='left', fontsize=19, fontweight='bold')
    fig.text(.055, .936, 'A four-second-only, independent prospective study. '
             'Own handover downtime resets the watchdog; scans pause its evidence counter.', fontsize=11)
    fig.text(.055, .892, 'Native packet and action traces are descriptive diagnostics; registered statistical and engineering decisions remain in the primary figure.',
             fontsize=10.2, color='#516574')
    ax = fig.add_subplot(gs[0, :2]); architecture(ax)
    ax.set_title('A  Preserve ordinary policy; gate emergency action on available-radio evidence',loc='left',pad=10)
    confirmed = {key:value for key,value in data.items() if key[0]=='confirmation'}
    ax = fig.add_subplot(gs[0, 2]); ax.set_title('B  Warm forecast work and service charge',loc='left',pad=10)
    warm = {}; labels=[]
    for index,family in enumerate(FAMILIES):
        jobs=[j for key,row in confirmed.items() if key[3]==family for j in row['jobs'] if j['kind']=='warm']
        if not jobs: continue
        measured=max(j['actual_compute_seconds'] for j in jobs)
        charged=max(j['charged_seconds'] for j in jobs); allowance=profiles[family]['warm']
        warm[family]={'requests':len(jobs),'actual_compute_max':measured,'charged_service_max':charged,'old_warm_allowance':allowance}
        ax.barh(index-.14,measured,height=.23,color=ORANGE,label='Measured compute maximum' if index==0 else None)
        ax.barh(index+.14,charged,height=.23,color=TEAL,label='Charged service maximum (+0.5 s)' if index==0 else None)
        ax.plot([allowance,allowance],[index-.27,index+.02],color=INK,linewidth=1.7,label='Reused warm allowance' if index==0 else None)
        labels.append(index)
    if confirmed:
        ax.set_yticks(labels,[LABELS[FAMILIES[index]].replace('Frozen JEPA ensemble','JEPA ensemble') for index in labels])
        ax.invert_yaxis(); ax.set_xlabel('Seconds per warm forecast request'); tidy(ax)
        ax.legend(loc='upper left',bbox_to_anchor=(-.02,-.28),fontsize=7.4,frameon=False)
    else:
        ax.axis('off');ax.text(.5,.5,'No confirmation timing\n4 s cases not run: development veto',transform=ax.transAxes,ha='center',color='#657684',linespacing=1.7)
    ax = fig.add_subplot(gs[1,0]);ax.axis('off');ax.set_title('C  Actual recovery, sensing and audit counts',loc='left',pad=12)
    table=[]
    for family in FAMILIES:
        legacy=[row for key,row in confirmed.items() if key[3:]==(family,'legacy')]
        repair=[row for key,row in confirmed.items() if key[3:]==(family,'repair')]
        label='Map' if family=='radiomap' else 'JEPA'
        table.append([label,'unrun','unrun','unrun'] if not repair else [label,
            f"{np.mean([r['emergency_handovers'] for r in repair]):.2f}",
            f"{np.mean([r['emergency_scans'] for r in repair]):.2f}",
            f"{np.mean([r['roams'] for r in legacy]):.1f} → {np.mean([r['roams'] for r in repair]):.1f}"])
    artist=ax.table(cellText=table,colLabels=['Family','Watchdog\nmoves','Due scans\nwhile silent','All moves\nlegacy → repair'],
        cellLoc='center',colLoc='center',colWidths=[.22,.23,.24,.31],bbox=[0,.64,1,.31])
    artist.auto_set_font_size(False);artist.set_fontsize(7.6)
    for (r,c),cell in artist.get_celld().items():
        cell.set_edgecolor('#D3DFE7');cell.set_facecolor('#E9F0F4' if r==0 else 'white')
    ages=[r['forecast_age_max'] for r in confirmed.values() if r['forecast_age_max'] is not None]
    lines=[f"Confirmation cases: {len(confirmed)}; source / paid / causal: PASS",
        f"Actual jobs: {sum(len(r['jobs']) for r in confirmed.values()):,}; profile-exceeded cases: {sum(any(j['runtime_overrun'] for j in r['jobs']) for r in confirmed.values())}",
        f"Local watchdog checks: {sum(r['watchdog_checks'] for r in confirmed.values())}",
        'Largest consumed forecast age: '+(f'{max(ages):.2f} s' if ages else 'unrun'),
        f"Exact offline replays: {sum(r['exact'] for r in offline)} / {len(offline)}",
        f"Fresh-cost repetitions: {len(fresh)}; changed availability: {sum(r['changed_ready_jobs']>0 for r in fresh)}",
        f"Fresh max absolute miss change: {max((abs(r['miss_difference']) for r in fresh),default=0)}" if fresh else 'Fresh miss sensitivity: unrun',
        'Counts above: means per 100-robot, 60 s trial.']
    ax.text(0,.57,'\n'.join(lines),transform=ax.transAxes,va='top',fontsize=8.2,color='#516574',linespacing=1.65)
    ax=fig.add_subplot(gs[1,1:]);ax.set_title('D  An actual local evidence counter beside raw silence',loc='left',pad=12)
    illustrative=next((row['illustration'] for key,row in sorted(confirmed.items())
        if key[4]=='repair' and row['illustration'] is not None),None)
    if illustrative:
        event=illustrative['first_event'];center=event['slot']/100
        x=np.asarray([s['second'] for s in illustrative['snapshots']])
        raw=np.asarray([s['raw_control_silence_slots'] for s in illustrative['snapshots']])/100
        counter_points=[(s['second'],-3,s['available_silence_slots']/100) for s in illustrative['snapshots']]
        for e in illustrative['emergency_events']:
            counter_points.extend([(e['slot']/100,-2,e['available_silence_slots']/100),
                (e['slot']/100,-1,e['available_silence_slots_after']/100)])
        counter_points.sort()
        counter_x=np.asarray([point[0] for point in counter_points])
        counter_y=np.asarray([point[2] for point in counter_points])
        low,high=max(0,center-7),min(p['duration_seconds'],center+7)
        mask=(x>=low)&(x<=high)
        counter_mask=(counter_x>=low)&(counter_x<=high)
        ax.plot(x[mask],raw[mask],color='#8B9EAC',linewidth=1.5,marker='o',markersize=2.5,label='Raw control silence (1 s snapshots)')
        ax.plot(counter_x[counter_mask],counter_y[counter_mask],color=TEAL,linewidth=2,marker='o',markersize=2.5,
            label='Available silence (1 s snapshots + logged reset)')
        ax.axhline(4.2,color=ORANGE,linestyle='--',linewidth=1.1,label='Emergency threshold: 4.20 available seconds')
        for number,e in enumerate(illustrative['emergency_events']):
            start,end=e['slot']/100,e['outage_until_slot']/100
            if end>=low and start<=high:
                ax.axvspan(start,end,color=RED,alpha=.10,label='Actual paid emergency handover outage' if number==0 else None)
                ax.scatter([start],[e['available_silence_slots']/100],color=ORANGE,s=36,zorder=5)
                ax.scatter([start],[0],color=RED,s=36,zorder=5)
        ax.set_xlim(low,high);ax.set_ylim(bottom=-.12)
        ax.set_xlabel(f"Trial seconds · site {illustrative['case'][1]}, {LABELS[illustrative['case'][3]]}, robot {illustrative['robot']}\n"
            'First emergency in canonical case order; illustrative, not a primary outcome')
        ax.set_ylabel('Seconds of silence');ax.grid(color='#DCE5EB',alpha=.7,linewidth=.7);ax.set_axisbelow(True)
        ax.legend(loc='upper left',fontsize=7.4,frameon=False,ncol=2,bbox_to_anchor=(0,1.01))
    else:
        ax.axis('off');ax.text(.5,.5,'No emergency handover trace in confirmation cases.\n'
            'No invented recovery curve is shown.',transform=ax.transAxes,ha='center',color='#657684',linespacing=1.7)
    fig.text(.055,.139,'Measured history-to-forecast issue work determines actual FIFO completion. Startup/loading is separate; shared +0.5 s is a scheduling assumption. '
        'Ordinary admission/scoring CPU is not separately scheduled.\n'
        'A move resets available silence to zero; delivery / association / own handover reset it, scans pause it. '
        'The full 4 s handover and 90 ms scan costs are paid.\n'
        'Offline replay retains the original paid schedule; fresh-cost repetitions measure timing sensitivity without extra sites. '
        'Shared procedural CAD, finite trials and warm history do not establish physical reliability.',
        fontsize=8.2,color='#516574',linespacing=1.65,va='top')
    parent_reference(fig,parent,.052,fontsize=8.2)
    return fig,warm,illustrative


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=ROOT/'artifacts/costaware_recovery')
    parser.add_argument('--output',type=Path,help='Default: FOLDER/figures')
    args=parser.parse_args()
    p,report,profiles,data,inputs,offline,fresh,parent=load_inputs(args.folder)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    style(plt);validate_statistics(np,p,report,data)
    second,warm,illustrative=operations_figure(plt,np,p,report,profiles,data,offline,fresh,parent)
    figures=[('confirmation',evidence_figure(plt,np,p,report,data,parent)),('available-radio-recovery-and-cost',second)]
    output=args.output or args.folder/'figures';output.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for name,fig in figures:
        for extension in ('svg','pdf','png'):
            path=output/f'{name}.{extension}'
            fig.savefig(path,dpi=190,metadata={'Creator':'plot_costaware_recovery.py'});outputs.append(path)
        plt.close(fig)
    receipt={'schema':1,'derived_only':True,'scientific_execution':False,'new_fitting_or_selection':False,
        'script_sha256':digest(Path(__file__)),'report_sha256':digest(args.folder/'report.json'),
        'inputs_sha256':{relative(path):digest(path) for path in sorted(set(inputs))},
        'outputs_sha256':{relative(path):digest(path) for path in outputs},
        'software':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
        'observed_loss_units':'percent','paired_reduction_units':'avoided misses per100000 generated commands',
        'reduction_sign':'reference minus candidate; positive favors candidate',
        'uncertainty':'20k whole-site bootstrap;95% descriptive,1-.05/3 observed AND conservative endpoint lower bounds for three primary decisions',
        'primary_comparisons':report['primary'],'all_three_experimental_benefits':report['all_three_experimental_benefits'],
        'passed_development_costs':report['passed_development_costs'],'offline_replay_checks':offline,
        'fresh_live_cost_sensitivity':fresh,'warm_request_summary':warm,'illustrative_actual_counter_trace':illustrative,
        'prior_failed_parent_study':parent,'no_physical_or_production_qualification':True,'limits':report['limits']}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'figures':[relative(path) for path in outputs],'receipt':relative(output/'receipt.json'),'scientific_execution':False}))


if __name__=='__main__':main()
