"""Verify completed repair archives without worlds, inference or retiming.

--write-manifest writes only derived final metadata. Plain verification checks
the existing source/document/evidence manifest. Older science stays immutable.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/jepa_repair'
sys.path.insert(0, str(ROOT / 'src'))
PR19_COMMIT = '30e4a518601d9cca3d4ab417f93a379967dc565e'
OLD_COMMITS = {'semantic_jepa': '9ff5fa56e7327bccde15a10a1a3952a82c699e02',
               'selected_jepa': 'f5c69d55cf110578f7c48de96899a5879bb9f106'}
DOCUMENTS = ('README.md', 'docs/HANDOFF.md', 'docs/PAPER_PROPOSAL.md',
             'docs/JEPAREPAIR_DESIGN.md', 'docs/JEPAREPAIR_RESULTS.md',
             'docs/JEPAREPAIR_DIAGNOSIS_ARCHITECTURE.md',
             'docs/JEPAREPAIR_RUNTIME_DIAGNOSIS.md')


def require(value, message):
    if not value: raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def committed(commit, name):
    return subprocess.check_output(['git', 'show', commit + ':' + name], cwd=ROOT)


def git_tree(commit):
    return subprocess.check_output(['git', 'rev-parse', commit + '^{tree}'], cwd=ROOT, text=True).strip()


def visible(folder):
    return {p.name for p in folder.iterdir() if not p.name.startswith('.') and not p.name.endswith(('.lock', '.part'))} if folder.exists() else set()


def registration(c):
    from wirelessjev.semantic_jepa import common as old
    from wirelessjev.selected_jepa import common as selected
    closures = {}
    for name, module in (('semantic_jepa', old), ('selected_jepa', selected)):
        receipt = module.check_sources(); commit = OLD_COMMITS[name]
        expected_count = 80 if name == 'semantic_jepa' else 230
        require(len(receipt['source_sha256']) == expected_count, 'Original source closure count differs')
        require((module.OUT / 'registration/source.json').read_bytes() == committed(commit, 'artifacts/' + name + '/registration/source.json'), 'Original registration changed')
        for path, expected in receipt['source_sha256'].items():
            raw = committed(commit, path)
            require(hashlib.sha256(raw).hexdigest() == expected and (module.OUT / 'registration/source' / path).read_bytes() == raw, 'Original source/capsule/Git differs: ' + path)
        closures[name] = expected_count
    frozen = c.check_sources(); meta = read(OUT / 'registration/git.json'); commit = meta['scientific_source_commit']
    require(frozen['new_evaluation_data_accessed'] is False and frozen['source_training_and_results_already_observed'] is True and frozen['new_fits'] == 0, 'Repair registration scope differs')
    require(re.fullmatch('[0-9a-f]{40}', commit) and meta['scientific_source_tree'] == git_tree(commit) and meta['before_new_evaluations'] is True, 'Repair pre-evaluation Git metadata differs')
    require((OUT / 'registration/source.json').read_bytes() == committed(commit, 'artifacts/jepa_repair/registration/source.json'), 'Repair registration differs from Git')
    for path, expected in frozen['source_sha256'].items():
        raw = committed(commit, path)
        require(digest(ROOT / path) == expected == hashlib.sha256(raw).hexdigest() and (OUT / 'registration/source' / path).read_bytes() == raw, 'Repair current/capsule/Git differs: ' + path)
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, 'artifacts'], cwd=ROOT, text=True).splitlines()
    prefixes = tuple('artifacts/jepa_repair/' + folder + '/' for folder in ('cases', 'replays', 'live_replications'))
    require(not any(name.startswith(prefixes) for name in names), 'Preregistration contains new scientific cases')
    c.verify_fresh_sites(); p = c.protocol()
    require(p['development_sites'] == [2123000, 2123001] and p['confirmation_sites'] == list(range(2124000, 2124008)) and p['robots'] == 100 and p['duration_seconds'] == 60 and p['new_fits'] == 0, 'Prospective identities differ')
    new_sites = set(p['development_sites'] + p['confirmation_sites'])
    for name in names:
        if re.fullmatch(r'artifacts/[^/]+/protocol\.json', name) and name != 'artifacts/jepa_repair/protocol.json':
            require(not new_sites & c._integers(json.loads(committed(commit, name))), 'New sites overlap an existing protocol')
    require(read(OUT / 'execution_order.json') == c.execution_order(), 'Captured order differs')
    return frozen, meta, closures


def preserve_prior():
    result = {}
    for experiment in ('semantic_jepa', 'selected_jepa'):
        name = 'artifacts/' + experiment + '/verification/manifest.json'; path = ROOT / name
        raw = committed(PR19_COMMIT, name); require(path.read_bytes() == raw, 'Prior manifest changed')
        manifest = json.loads(raw); counts = {}
        for group in ('sources_sha256', 'evidence_sha256'):
            for filename, expected in manifest[group].items():
                require(digest(ROOT / filename) == expected, 'Prior source/evidence changed: ' + filename)
            counts[group] = len(manifest[group])
        for filename, expected in manifest['documents_sha256'].items():
            require(hashlib.sha256(committed(PR19_COMMIT, filename)).hexdigest() == expected, 'Historical publication document differs')
        result[experiment] = {'manifest_sha256': digest(path), 'preserved_entries': counts}
    old = read(ROOT / 'artifacts/selected_jepa/report.json')
    require(old['any_runtime_overrun'] is True and all(row['qualified'] is False for row in old['primary']), 'Previous global veto/qualification changed')
    return result


def membership(folder, keys, rollout):
    names = {rollout.case_name(key) + suffix for key in keys for suffix in ('.json.gz', '.receipt.json')}
    require(visible(folder) == names, 'Incomplete/extra archive member: ' + str(folder))


def jobs_check(row, p, profile, original=None):
    offline = original is not None; ready = []; finish = 0.
    require(len(row['jobs']) == p['duration_seconds'], 'Incomplete forecast ledger')
    for i, job in enumerate(row['jobs']):
        require(type(job['decision']) is int and job['decision'] == i and job['history_cutoff_exclusive'] == 60 + i, 'Forecast crosses past-history boundary')
        require(job['kind'] == ('cold' if i == 0 else 'refit' if i % 20 == 0 else 'warm'), 'Unexpected timing regime')
        fields = ('actual_prediction_seconds', 'actual_issue_seconds', 'actual_compute_seconds', 'profile_seconds', 'charged_compute_seconds', 'charged_seconds', 'shared_extra_seconds')
        require(all(type(job[f]) in (int, float) and math.isfinite(job[f]) and job[f] >= 0 for f in fields), 'Invalid measured timing')
        actual = max(job['actual_prediction_seconds'], job['actual_issue_seconds']); compute = max(profile[job['kind']], actual)
        expected = {'actual_compute_seconds': actual, 'profile_seconds': profile[job['kind']], 'charged_compute_seconds': compute, 'shared_extra_seconds': p['extra_compute_seconds'], 'charged_seconds': compute + p['extra_compute_seconds']}
        require(all(math.isclose(job[f], v, rel_tol=0, abs_tol=1e-12) for f, v in expected.items()), 'Measured forecast issue not fully paid')
        require(job['runtime_overrun'] is (actual > profile[job['kind']]) and job['predictor_runtime_overrun'] is (job['actual_prediction_seconds'] > profile[job['kind']]), 'Capacity diagnostic differs')
        require(job['offline_recorded_schedule'] is offline and job['completion_uses_actual_elapsed'] is (not offline) and job['accounting_mode'] == ('recorded_schedule_offline' if offline else 'actual_cost'), 'Offline/fresh namespaces mixed')
        if offline:
            before = original['jobs'][i]
            require(all(job[f] == before[f] for f in set(fields) | {'decision', 'kind', 'history_cutoff_exclusive', 'ready_slot'}), 'Offline replay changed paid schedule')
            fresh_fields = ('fresh_actual_prediction_seconds', 'fresh_actual_issue_seconds', 'fresh_actual_compute_seconds')
            require(all(type(job[f]) in (int, float) and math.isfinite(job[f]) and job[f] >= 0 for f in fresh_fields), 'Bad fresh offline diagnostic')
            fresh = max(job[fresh_fields[0]], job[fresh_fields[1]])
            require(job['fresh_actual_compute_seconds'] == fresh and job['exceeds_original_recorded_compute'] is (fresh > before['charged_compute_seconds']) and job['recorded_completion_used_actual_elapsed'] is True and job['matches_recorded_forecast'] is True and job['forecast_sha256'] == before['forecast_sha256'] and job['forecast_std_sha256'] == before['forecast_std_sha256'], 'Offline forecasts/diagnostics differ')
        finish = max(float(i), finish) + job['charged_seconds']
        require(type(job['ready_slot']) is int and job['ready_slot'] == math.ceil(finish * 100 - 1e-10), 'Actual-cost FIFO not honored')
        require(all(re.fullmatch('[0-9a-f]{64}', job[f]) for f in ('forecast_sha256', 'forecast_std_sha256')) and job['forecast_dtype'] == 'float32' and job['forecast_shape'][0] == p['robots'], 'Forecast shape/dtype/hash differs')
        ready.append(job['ready_slot'])
    consumed = []
    for slot in sorted(set(ready)):
        if slot < p['duration_seconds'] * 100:
            index = max(i for i, end in enumerate(ready) if end <= slot)
            consumed.append({'slot': slot, 'forecast_decision': index, 'ready_slot': ready[index], 'age_seconds': slot / 100 - index})
    require(row['forecast_consumption'] == consumed, 'Premature/missing forecast consumption')
    require(row['runtime_overrun'] is any(j['runtime_overrun'] for j in row['jobs']) and row['predictor_runtime_overrun'] is any(j['predictor_runtime_overrun'] for j in row['jobs']), 'Case capacity diagnostic differs')


def telemetry_check(row, p):
    events = row['telemetry']['events']
    for event in events:
        times = [event[f] for f in ('measurement_slot', 'complete_slot', 'delivery_slot')]
        require(type(event['robot']) is int and 0 <= event['robot'] < p['robots'] and all(type(v) is int for v in times) and 0 <= times[0] < times[1] <= times[2] < p['duration_seconds'] * 100 and event['first_usable_decision'] == times[2] // 100 + 1 and re.fullmatch('[0-9a-f]{64}', event['payload_sha256']), 'RF report violates capture/delivery causality')
    require(row['telemetry']['reports_delivered'] == len(events) and row['causality_pass'] is True, 'Telemetry/causality summary differs')


def local_freshness_check(row, p):
    """Reconstruct genuine local captures and every chosen neighbor's evidence."""
    local = row['local_scan_observability']
    captures = np.asarray(local['initial_capture_slots'], dtype=np.int64)
    heard = np.asarray(local['initial_heard'])
    values = np.asarray(local['initial_rssi_dbm'], dtype=np.float32)
    require(captures.ndim == 2 and captures.shape[0] == p['robots'] and captures.shape[1] % 3 == 0
            and heard.dtype == np.bool_ and heard.shape == values.shape == captures.shape
            and np.isfinite(values).all() and (captures[heard] < 0).all(), 'Initial local capture schema/history differs')
    n, links = captures.shape; events = local['completion_events']; by_robot = [[] for _ in range(n)]
    payloads = {}; previous_processed = -1
    for event in events:
        robot, measured, complete, processed = [event[f] for f in ('robot', 'measurement_slot', 'complete_slot', 'processed_slot')]
        require(all(type(v) is int for v in (robot, measured, complete, processed)) and 0 <= robot < n
                and 0 <= measured < complete <= processed < 6000 and processed >= previous_processed,
                'Local completion occurred before a genuine capture/completion or outside trial')
        previous_processed = processed; ids = event['heard_links']; rssi = event['heard_rssi_dbm']
        require(ids == sorted(set(ids)) and all(type(i) is int and 0 <= i < links for i in ids)
                and len(ids) == len(rssi) == event['heard_entries'] and len(ids) <= 8
                and all(type(v) in (float, int) and math.isfinite(v) and v >= -90 for v in rssi),
                'Local report heard mask/RSSI differs from receiver rules')
        report_value = np.zeros(links, dtype='f4'); report_heard = np.zeros(links, dtype=bool)
        report_value[ids] = rssi; report_heard[ids] = True
        payload = hashlib.sha256(report_value.tobytes() + report_heard.tobytes()).hexdigest()
        require(event['payload_sha256'] == payload, 'Local report bytes/hash differ')
        identity = robot, measured, complete
        require(identity not in payloads, 'Duplicate local completion')
        payloads[identity] = payload, processed
        by_robot[robot].append(event)
    for event in row['telemetry']['events']:
        identity = tuple(event[f] for f in ('robot', 'measurement_slot', 'complete_slot'))
        require(identity in payloads and payloads[identity][0] == event['payload_sha256']
                and payloads[identity][1] <= event['delivery_slot'], 'Central report lacks its completed local payload')
    audit = row['result']['repair_execution']; checked = 0
    for event in audit['watchdog_checks'] + audit['emergency_handovers']:
        target = event.get('to_site', event.get('best_other_site')); band = event['selected_band']
        if target is None:
            require(band is None and event['neighbor_capture_slot'] is None
                    and event['neighbor_capture_age_slots'] is None and event['neighbor_heard'] is False,
                    'Missing neighbor gained invented capture metadata')
            continue
        slot, robot = event['slot'], event['robot']; require(type(band) is int and 0 <= band < 3, 'Bad selected observation band')
        link = 3 * target + band; require(0 <= link < links, 'Observed neighbor outside AP/band axes')
        prior = [capture for capture in by_robot[robot] if capture['processed_slot'] <= slot]
        if prior:
            capture = prior[-1]
            require(link in capture['heard_links'], 'Chosen neighbor was not heard in most recent complete local scan')
            index = capture['heard_links'].index(link); measured = capture['measurement_slot']; value = capture['heard_rssi_dbm'][index]
        else:
            require(bool(heard[robot, link]), 'Chosen neighbor absent from initial observed history')
            measured, value = int(captures[robot, link]), float(values[robot, link])
        age = slot - measured
        rssi = event.get('neighbor_rssi_dbm', event.get('best_other_rssi_dbm'))
        require(event['neighbor_capture_slot'] == measured and event['neighbor_capture_age_slots'] == age
                and event['neighbor_heard'] is True and 0 <= age <= round(p['scan_period_seconds'] * 100)
                and rssi == value, 'Chosen neighbor capture/RSSI is stale, unheard, future or inconsistent')
        checked += 1
    return checked


def recovery_check(row, p):
    result = row['result']; audit = result['repair_execution']; enabled = row['controller_variant'] == 'repair'
    require(audit['emergency_recovery_enabled'] is enabled and audit['watchdog_threshold_slots'] == 20 and audit['fresh_central_reports_still_require_control_delivery'] is True, 'Local sensing/threshold scope differs')
    if not enabled:
        require(not any(audit[f] for f in ('watchdog_checks', 'emergency_handovers', 'emergency_scans')), 'Comparator executed repair-only recovery')
    checks = {}
    for check in audit['watchdog_checks']:
        slot, robot = check['slot'], check['robot']
        require(type(slot) is int and slot % 10 == 0 and 0 <= slot < 6000 and type(robot) is int and 0 <= robot < 100 and check['silence_slots'] >= 20 and check['neighbor_max_age_slots'] == round(p['scan_period_seconds'] * 100), 'Watchdog cadence/silence/freshness bound differs')
        neighbor = check['best_other_rssi_dbm']
        trigger = check['current_rssi_dbm'] < -75 and neighbor is not None and neighbor >= -90 and neighbor - check['current_rssi_dbm'] >= 8
        require(check['recovery_triggered'] is trigger and (slot, robot) not in checks, 'Watchdog RF trigger differs/duplicates')
        checks[slot, robot] = check
    for action in audit['emergency_handovers']:
        check = checks.get((action['slot'], action['robot']))
        require(check is not None and check['recovery_triggered'] is True and action['from_site'] == check['current_site'] and action['to_site'] == check['best_other_site'] and action['from_site'] != action['to_site'] and action['current_rssi_dbm'] < -75 and action['neighbor_gain_db'] >= 8 and action['neighbor_rssi_dbm'] >= -90 and action['outage_until_slot'] == action['slot'] + round(row['handover_seconds'] * 100) and action['pending_cadence_target_after'] == -1, 'Emergency move thresholds/paid outage/pending cancellation differ')
        sample = next(e for e in result['packet_detail']['action_trace'] if e['slot_end'] > action['slot'])
        require(sample['primary_unavailable_until_slot'][action['robot']] >= action['outage_until_slot'], 'Emergency outage absent from action state')
    require(len(audit['emergency_handovers']) == sum(e['recovery_triggered'] for e in checks.values()), 'Triggered recovery lacks action')
    for event in audit['emergency_scans']:
        require(event['silence_slots'] >= 20 and 0 <= event['slot'] < 6000 and event['scan_complete_slot'] == event['slot'] + 1 + round(p['scan_outage_seconds'] * 100), 'Emergency scan threshold/paid duration differs')
    return {f: len(audit[f]) for f in ('watchdog_checks', 'emergency_handovers', 'emergency_scans')}


def load_rows(c, rollout, keys, *, replay=False, live=False, originals=None):
    p = c.protocol(); folder = OUT / ('replays' if replay else 'live_replications' if live else 'cases')
    if replay or live: membership(folder, keys, rollout)
    rows = []; counts = {f: 0 for f in ('watchdog_checks', 'emergency_handovers', 'emergency_scans')}
    for key in keys:
        name = rollout.case_name(key); path = folder / (name + '.json.gz'); receipt = read(folder / (name + '.receipt.json'))
        require(receipt['archive_sha256'] == digest(path), 'Archive bytes changed: ' + name)
        row = json.loads(gzip.decompress(path.read_bytes()))
        require(receipt['result_sha256'] == value_digest(row['result']) and row == rollout.load_case(key, replay=replay, live_replication=live), 'Raw result/loader differs: ' + name)
        require(row['replay'] is replay and row['live_replication'] is live and row['new_fits'] == 0 and row['hardware_qualified'] is False and row['production_qualified'] is False and row['simulation_only'] is True, 'Archive training/qualification scope differs')
        original = originals[tuple(key)] if replay else None; profile = c.latency()['profiles'][row['family']]
        jobs_check(row, p, profile, original)
        audit = rollout.timing_audit(row['jobs'], row['forecast_consumption'], profile, p['extra_compute_seconds'], p['duration_seconds'], accounting_mode=row['accounting_mode'], recorded_jobs=original['jobs'] if replay else None)
        require(audit == row['timing_audit'] and audit['passes'] is True and audit['offline_schedule_only'] is replay and audit['real_time_contract_checked'] is (not replay), 'Actual timing audit differs')
        if not replay: require(c.verify_paid_row(row) is True, 'Fresh paid-cost proof failed')
        telemetry_check(row, p); local_freshness_check(row, p)
        for f, value in recovery_check(row, p).items(): counts[f] += value
        require(row['action_trace_sha256'] == value_digest(row['result']['packet_detail']['action_trace']) and all(re.fullmatch('[0-9a-f]{64}', row[f]) for f in ('forecast_mean_sha256', 'forecast_std_sha256')), 'Forecast/action digest differs')
        rows.append(row)
    return rows, counts


def statistics_check(report, rows, p, costs):
    """Independent bootstrap arithmetic from actual native command counters."""
    indexed = {tuple(row['case']): row for row in rows}
    require(len(report['primary']) == report['primary_multiplicity_budget'] == 6 and report['primary_upper_quantile'] == 1 - .05 / 6, 'Six-test budget changed')
    indices = np.random.default_rng(p['bootstrap_seed']).integers(0, 8, size=(p['bootstrap_replicates'], 8))
    for contrast in report['primary']:
        cost = contrast['handover_seconds']
        if cost not in costs:
            require(contrast['status'] == 'not_run_development_veto' and contrast['mean_difference'] is None and contrast['statistical_pass'] is False and contrast['experimental_benefit'] is False, 'Unrun cost gained invented evidence')
            continue
        observed, endpoint = [], []
        for site in p['confirmation_sites']:
            a = indexed['confirmation', site, cost, contrast['candidate_family'], contrast['candidate_variant']]['result']
            b = indexed['confirmation', site, cost, contrast['reference_family'], contrast['reference_variant']]['result']
            require(a['control_sent'] == b['control_sent'] == 120000, 'Paired command counts differ')
            observed.append(a['control_misses'] / a['control_sent'] - b['control_misses'] / b['control_sent'])
            endpoint.append((a['control_misses'] + a['packet_detail']['pending_control_packets']) / a['control_sent'] - b['control_misses'] / b['control_sent'])
        for values, archived in ((observed, contrast), (endpoint, contrast['endpoint_sensitivity'])):
            values = np.asarray(values); draws = values[indices].mean(1)
            require(archived['paired_site_differences'] == values.tolist() and archived['mean_difference'] == float(values.mean()) and archived['multiplicity_upper'] == float(np.quantile(draws, p['primary_upper_quantile'])) and archived['descriptive_95_interval'] == np.quantile(draws, [.025, .975]).tolist(), 'Bootstrap/endpoint differs from native counters')
        require(contrast['statistical_pass'] is bool(contrast['multiplicity_upper'] < 0 and contrast['endpoint_sensitivity']['multiplicity_upper'] < 0), 'Primary superiority flag differs')
    require(report['hardware_qualified'] is False and report['production_qualified'] is False and report['activated'] is False and report['ema_necessity_claimed'] is False, 'Research scope promoted')


def tests_check():
    path = OUT / 'logs/preflight-tests.txt'; text = path.read_text()
    matches = re.findall(r'(?m)^([0-9]+) passed(?:, ([0-9]+) warnings?)? in ([0-9.]+)s', text)
    require(len(matches) == 1 and not re.search(r'(?m)^=+.*(?:failed|ERRORS).*?=+$', text), 'No unique successful full-test log summary')
    passed, warnings, seconds = matches[0]
    return {'passed': int(passed), 'warnings': int(warnings or 0), 'seconds': float(seconds), 'log': str(path.relative_to(ROOT)), 'sha256': digest(path)}


def verify(*, write_manifest=False):
    require((OUT / 'report.json').is_file() and (OUT / 'report.md').is_file(), 'Complete the registered pipeline/report first')
    from wirelessjev.jepa_repair import common as c, rollout, study, report as stats
    frozen, meta, closures = registration(c); prior = preserve_prior(); p = c.protocol()
    devkeys = study.ordered_keys('development'); require(len(devkeys) == 20, 'Development matrix changed')
    dev, devrecovery = load_rows(c, rollout, devkeys); study._fresh(dev, devkeys)
    gate = stats.development_gate(dev, p, validation=c.development_validation(dev))
    require(gate == c.verified_development(), 'Development permission differs from actual rows')
    for entry in gate['per_cost']:
        for check in entry['family_checks']:
            selected = lambda variant: [r['result'] for r in dev if r['family'] == check['family'] and r['controller_variant'] == variant and r['handover_seconds'] == entry['handover_seconds']]
            rate = lambda rows: Fraction(sum(r['control_misses'] for r in rows), sum(r['control_sent'] for r in rows))
            difference = rate(selected('repair')) - rate(selected('legacy'))
            require(check['exact_difference_numerator'] == difference.numerator and check['exact_difference_denominator'] == difference.denominator and check['severe_regression'] is (difference > Fraction(str(p['development_severe_regression_absolute']))), 'Development guard differs from exact counters')
    costs = gate['passed_costs']; keys = study.ordered_keys('confirmation', costs); repeats = study.replication_keys(costs)
    membership(OUT / 'cases', devkeys + keys, rollout)
    require(len(keys) <= 80 and len(repeats) <= 10 and all(key[1] == p['confirmation_sites'][0] for key in repeats), 'Replication scope changed')
    main, mainrecovery = load_rows(c, rollout, keys); originals = {tuple(row['case']): row for row in main}
    offline, offlinerecovery = load_rows(c, rollout, repeats, replay=True, originals=originals)
    live, liverecovery = load_rows(c, rollout, repeats, live=True)
    study._fresh(main, keys); study._fresh(live, repeats, live_replication=True)
    completion = study._completion(main, offline, live, keys, repeats, costs)
    require(completion == read(OUT / 'completion.json') and completion['all_offline_exact'] is True and completion['all_fresh_paid_cost_checks_pass'] is True, 'Completion/offline/fresh proof differs')
    validation = {'source_and_archive_integrity': True, 'causality_and_kernel_checks': all(study._causality(row) for row in dev + main + offline + live), 'paid_cost_and_consumption_checks': True, 'exact_offline_replay': completion['all_offline_exact']}
    expected = stats.aggregate(main, p, validation=validation, passed_costs=costs)
    first = [row for row in main if row['site'] == p['confirmation_sites'][0]]
    expected.update(development_gate=gate, offline_replay_comparisons=completion['offline_comparisons'], fresh_live_cost_comparisons=completion['fresh_live_cost_comparisons'], offline_replay_summary=stats.replication_summary(offline, p, 'recorded_schedule_offline', passed_costs=costs, original_rows=first), fresh_live_cost_summary=stats.replication_summary(live, p, 'fresh_live_cost', passed_costs=costs, original_rows=first), source_sha256=c.source_hashes(), completion_sha256=digest(OUT / 'completion.json'), development_sha256=digest(OUT / 'development.json'), qualification_sha256=digest(OUT / 'qualification.json'), execution_order_sha256=digest(OUT / 'execution_order.json'))
    archived = read(OUT / 'report.json'); require(expected == archived, 'Final report differs from complete raw reproduction')
    statistics_check(archived, main, p, costs)
    markdown = stats.markdown(expected) + ('\n## Replication scope\n\n' + f'{len(offline)} exact-schedule OFFLINE replays; {len(live)} fresh actual-cost repetitions. ' + 'The former establish only replay under recorded availability; the latter are descriptive runtime sensitivity and add no independent sites.\n')
    require((OUT / 'report.md').read_text() == markdown, 'Final Markdown differs from raw report')
    tests = tests_check(); proof = OUT / 'verification'; manifest = proof / 'manifest.json'
    sources = [path for folder in ('src', 'scripts', 'tests') for path in (ROOT / folder).rglob('*.py') if '__pycache__' not in path.parts] + [ROOT / 'pyproject.toml', ROOT / 'uv.lock']
    documents = [ROOT / name for name in DOCUMENTS]
    evidence = [path for path in OUT.rglob('*') if path.is_file() and path != manifest and not any(part.startswith('.') for part in path.relative_to(OUT).parts) and not path.name.endswith(('.lock', '.part'))]
    hashes = lambda paths: {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}
    value = {'schema': 1, 'scientific_source_commit': meta['scientific_source_commit'], 'scientific_source_tree': meta['scientific_source_tree'], 'registration_precedes_only_new_development_and_confirmation': True, 'registered_source_files': len(frozen['source_sha256']), 'original_source_closures': closures, 'source_registration_sha256': digest(OUT / 'registration/source.json'), 'prior_scientific_source_and_evidence': prior, 'previous_global_document_hashes_apply_to_their_original_commits': True, 'development_cases': len(dev), 'passed_development_costs': costs, 'confirmation_cases': len(main), 'offline_exact_replays': len(offline), 'fresh_live_replications': len(live), 'independent_confirmation_sites': 8 if costs else 0, 'new_fits': 0, 'full_tests': tests, 'recovery_counts': {'development': devrecovery, 'confirmation': mainrecovery, 'offline_replay': offlinerecovery, 'fresh_live': liverecovery}, 'actual_cost_contract_preserves_previous_overrun_veto': True, 'profile_exceedance_diagnostic_only_under_new_paid_actual_cost_contract': True, 'actual_jobs_timing_paid_fifo_consumption_and_telemetry_verified': True, 'conditional_matrix_complete_and_no_unrun_cost_fabricated': True, 'all_six_primary_tests_and_pending_endpoint_bootstraps_recomputed': True, 'frozen_report_reproduced_from_actual_archives': True, 'offline_replays_make_no_fresh_measured_deadline_claim': True, 'fresh_replications_add_no_independent_sites': True, 'primary': archived['primary'], 'timing': archived['timing'], 'all_six_experimental_benefits': archived['all_six_experimental_benefits'], 'report_sha256': digest(OUT / 'report.json'), 'hardware_qualified': False, 'production_qualified': False, 'activated': False, 'new_worlds_forwards_training_or_timing_executed_by_verifier': False, 'sources_sha256': hashes(sources), 'documents_sha256': hashes(documents), 'evidence_sha256': hashes(evidence)}
    if write_manifest:
        proof.mkdir(exist_ok=True); value['verified_at_utc'] = datetime.now(timezone.utc).isoformat()
        fd, temporary = tempfile.mkstemp(prefix='.manifest-', dir=proof)
        try:
            with os.fdopen(fd, 'w') as stream:
                stream.write(json.dumps(value, indent=2) + '\n'); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, manifest)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    else:
        existing = read(manifest); existing.pop('verified_at_utc'); require(existing == value, 'Final source/document/evidence manifest differs')
    print(json.dumps({key: value[key] for key in ('registered_source_files', 'development_cases', 'passed_development_costs', 'confirmation_cases', 'offline_exact_replays', 'fresh_live_replications', 'all_six_experimental_benefits', 'production_qualified')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true', help='Write derived final metadata explicitly')
    args = parser.parse_args(); verify(write_manifest=args.write_manifest)
