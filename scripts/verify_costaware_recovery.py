"""Verify the separate slow cost-aware study without scientific execution.

--audit-only defers global source/document hashing. Final --write-manifest
writes derived metadata only in this experiment's own output directory.
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
import verify_jepa_repair as base

ROOT = base.ROOT
OUT = ROOT / 'artifacts/costaware_recovery'
require, read, digest, value_digest = base.require, base.read, base.digest, base.value_digest
committed, git_tree, visible = base.committed, base.git_tree, base.visible
membership = base.membership
jobs_check, telemetry_check = base.jobs_check, base.telemetry_check
local_freshness_check = base.local_freshness_check
DOCUMENTS = ('README.md', 'docs/HANDOFF.md', 'docs/PAPER_PROPOSAL.md',
             'docs/COSTAWARE_RECOVERY_DESIGN.md', 'docs/COSTAWARE_RECOVERY_RESULTS.md')
SCIENTIFIC_SOURCE_COMMIT = '6f931a01a5bbc68ffeb128418a4762cce4198bc1'
EXPECTED_TESTS = (1245, 1, 172.87)
EXPECTED_TEST_LOG_SHA256 = '69205b224a1c3f7b0ce128689aa5881e5a9fabf2a849cebaaa8cc72aa73fb834'


def registration(c):
    """Verify this capsule and pinned 386/230/80 predecessor closures."""
    parent_frozen, _, old_closures = base.registration(c.parent)
    require(len(parent_frozen['source_sha256']) == 386, 'Parent frozen closure differs')
    frozen = c.check_sources(); meta = read(OUT / 'registration/git.json'); commit = meta['scientific_source_commit']
    require(commit == SCIENTIFIC_SOURCE_COMMIT and len(frozen['source_sha256']) == 422, 'Frozen child source identity/count differs')
    require(re.fullmatch('[0-9a-f]{40}', commit) and meta['scientific_source_tree'] == git_tree(commit)
            and meta['before_new_evaluations'] is True, 'Child pre-evaluation Git metadata differs')
    require(frozen['new_evaluation_data_accessed'] is False and frozen['new_fits'] == 0
            and frozen['parent_development_already_observed'] is True
            and frozen['source_training_and_results_already_observed'] is True, 'Child registration scope differs')
    require((OUT / 'registration/source.json').read_bytes() == committed(commit,
            'artifacts/costaware_recovery/registration/source.json'), 'Child registration differs from Git')
    for path, expected in frozen['source_sha256'].items():
        raw = committed(commit, path)
        require(digest(ROOT / path) == expected == hashlib.sha256(raw).hexdigest()
                and (OUT / 'registration/source' / path).read_bytes() == raw, 'Child current/capsule/Git differs: ' + path)
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, 'artifacts'], cwd=ROOT, text=True).splitlines()
    prefixes = tuple('artifacts/costaware_recovery/' + folder + '/' for folder in ('cases', 'replays', 'live_replications'))
    require(not any(name.startswith(prefixes) for name in names), 'Preregistration contains new child evaluations')
    c.verify_fresh_sites(); p = c.protocol()
    require(p['development_sites'] == [2133000, 2133001] and p['confirmation_sites'] == list(range(2134000, 2134008))
            and p['handover_seconds'] == [4.0] and p['variants_by_cost'] == {'4.0': ['legacy', 'repair']}
            and p['robots'] == 100 and p['duration_seconds'] == 60 and p['new_fits'] == 0, 'Child prospective identities differ')
    new_sites = set(p['development_sites'] + p['confirmation_sites'])
    for name in names:
        if re.fullmatch(r'artifacts/[^/]+/protocol\.json', name) and name != 'artifacts/costaware_recovery/protocol.json':
            require(not new_sites & c._integers(json.loads(committed(commit, name))), 'Child sites overlap an older protocol')
    require(read(OUT / 'execution_order.json') == c.execution_order(), 'Captured child order differs')
    require(c.parent.verified_development()['passed_costs'] == [.03], 'Parent slow-cost veto changed')
    require(frozen['parent_development_sha256'] == digest(c.parent.OUT / 'development.json')
            and frozen['parent_qualification_sha256'] == digest(c.parent.OUT / 'qualification.json'), 'Observed parent development binding changed')
    for receipt_path in c.parent_development_receipts():
        receipt = read(receipt_path); archive = receipt_path.with_name(receipt_path.name.removesuffix('.receipt.json') + '.json.gz')
        require(digest(archive) == receipt['archive_sha256'], 'Observed parent development archive changed')
    from wirelessjev.costaware_recovery import runtime, telemetry
    from wirelessjev.jepa_repair import runtime as old_runtime, telemetry as old_telemetry
    require(runtime.CompletionSafeForecasts is old_runtime.CompletionSafeForecasts
            and runtime.RecordedScheduleReplay is old_runtime.RecordedScheduleReplay
            and telemetry.TimedDeliveredScans is old_telemetry.TimedDeliveredScans, 'Frozen paid-runtime/telemetry aliases differ')
    return frozen, meta, dict(old_closures, jepa_repair=386)


def preserve_prior():
    return base.preserve_prior()


def tests_check():
    require(EXPECTED_TESTS is not None and EXPECTED_TEST_LOG_SHA256 is not None, 'Fill actual complete preflight metadata before final audit')
    path = OUT / 'logs/preflight-tests.txt'; text = path.read_text()
    matches = re.findall(r'(?m)^([0-9]+) passed(?:, ([0-9]+) warnings?)? in ([0-9.]+)s', text)
    require(len(matches) == 1, 'No unique complete test summary')
    passed, warnings, seconds = matches[0]; actual = int(passed), int(warnings or 0), float(seconds)
    require(actual == EXPECTED_TESTS and digest(path) == EXPECTED_TEST_LOG_SHA256, 'Complete child preflight log differs')
    return {'passed': actual[0], 'warnings': actual[1], 'seconds': actual[2], 'log': str(path.relative_to(ROOT)), 'sha256': digest(path)}


def recovery_check(row, p):
    """Check logged evidence values; full 10 ms recurrence is source/test scope."""
    result = row['result']; audit = result['repair_execution']; enabled = row['controller_variant'] == 'repair'
    require(audit['emergency_recovery_enabled'] is enabled and audit['watchdog_threshold_slots'] == 20
            and audit['fresh_central_reports_still_require_control_delivery'] is True, 'Child local sensing/scan threshold differs')
    checks = {}; snapshots = {e['slot_end']: e for e in result['packet_detail']['cumulative']}
    if not enabled:
        require(not any(audit[f] for f in ('watchdog_checks', 'emergency_handovers', 'emergency_scans')), 'Legacy executed child watchdog')
    else:
        require(audit['handover_threshold_slots'] == 420
                and audit['available_silence_slots_final'] == snapshots[6000]['available_silence_slots'], 'Cost-derived threshold/final counter differs')
        previous = np.zeros(100, dtype=np.int64)
        action_at = {e['slot_end']: e for e in result['packet_detail']['action_trace']}
        for slot, sample in sorted(snapshots.items()):
            values = np.asarray(sample['available_silence_slots'])
            require(values.shape == (100,) and values.dtype.kind in 'iu' and (values >= 0).all()
                    and (values <= np.asarray(sample['current_gap'])).all() and ((values - previous) <= 100).all(), 'Sampled available counter grows too fast/exceeds silence')
            down = np.asarray(action_at[slot]['primary_unavailable_until_slot']) >= slot
            require((values[down] == 0).all(), 'Known primary handover outage counted as evidence')
            previous = values
    for check in audit['watchdog_checks']:
        slot, robot = check['slot'], check['robot']; available = check['available_silence_slots']
        require(type(slot) is int and slot % 10 == 0 and 0 <= slot < 6000 and type(robot) is int
                and 0 <= robot < 100 and check['silence_slots'] >= 20 and type(available) is int
                and 0 <= available <= check['silence_slots'] and check['handover_threshold_slots'] == 420
                and check['neighbor_max_age_slots'] == 500, 'Cost-aware cadence/counter/freshness bound differs')
        if slot in snapshots:
            require(available == snapshots[slot]['available_silence_slots'][robot]
                    and check['silence_slots'] == snapshots[slot]['current_gap'][robot], 'Watchdog counter differs from preceding snapshot boundary')
        require(slot in action_at and action_at[slot]['primary_unavailable_until_slot'][robot] <= slot
                and action_at[slot]['primary_site'][robot] == check['current_site'],
                'Watchdog used an unavailable primary radio or inconsistent association')
        neighbor = check['best_other_rssi_dbm']
        trigger = available >= 420 and check['current_rssi_dbm'] < -75 and neighbor is not None and neighbor >= -90 and neighbor - check['current_rssi_dbm'] >= 8
        require(check['recovery_triggered'] is trigger and (slot, robot) not in checks, 'Cost-aware RF/evidence trigger differs')
        checks[slot, robot] = check
    previous_end = {}
    for action in audit['emergency_handovers']:
        slot, robot = action['slot'], action['robot']; check = checks.get((slot, robot))
        require(check is not None and check['recovery_triggered'] is True
                and action['available_silence_slots'] == check['available_silence_slots'] >= 420
                and action['handover_threshold_slots'] == 420 and action['available_silence_slots_after'] == 0
                and action['silence_slots'] >= action['available_silence_slots']
                and action['from_site'] == check['current_site'] and action['to_site'] == check['best_other_site']
                and action['from_site'] != action['to_site'] and action['current_rssi_dbm'] < -75
                and action['neighbor_gain_db'] >= 8 and action['neighbor_rssi_dbm'] >= -90
                and action['outage_until_slot'] == slot + 400 and action['pending_cadence_target_after'] == -1,
                'Cost-aware action reset/RF thresholds/paid outage differs')
        require(slot >= 420 and (robot not in previous_end or slot >= previous_end[robot] + 420), 'Own handover outage justified another emergency')
        previous_end[robot] = action['outage_until_slot']
        sample = next(e for e in result['packet_detail']['action_trace'] if e['slot_end'] > slot)
        require(sample['primary_unavailable_until_slot'][robot] >= slot + 400, 'Emergency outage absent from action state')
    require(len(audit['emergency_handovers']) == sum(e['recovery_triggered'] for e in checks.values()), 'Triggered recovery lacks action')
    for event in audit['emergency_scans']:
        require(event['silence_slots'] >= 20 and 0 <= event['slot'] < 6000 and event['scan_complete_slot'] == event['slot'] + 10, 'Child scan threshold/90 ms outage differs')
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
    require(len(report['primary']) == report['primary_multiplicity_budget'] == 3 and report['primary_upper_quantile'] == 1 - .05 / 3, 'Three-test budget changed')
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


def verify(*, write_manifest=False, audit_only=False):
    require((OUT / 'report.json').is_file() and (OUT / 'report.md').is_file(), 'Complete the registered pipeline/report first')
    from wirelessjev.costaware_recovery import common as c, rollout, study, report as stats
    frozen, meta, closures = registration(c); prior = preserve_prior(); p = c.protocol()
    devkeys = study.ordered_keys('development'); require(len(devkeys) == 8, 'Development matrix changed')
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
    require(len(keys) <= 32 and len(repeats) <= 4 and all(key[1] == p['confirmation_sites'][0] for key in repeats), 'Replication scope changed')
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
    if audit_only:
        require(not write_manifest, 'Independent audit and final manifest are separate operations')
        proof.mkdir(exist_ok=True)
        audit = {'schema': 1, 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
            'scientific_source_commit': meta['scientific_source_commit'], 'scientific_source_tree': meta['scientific_source_tree'],
            'registered_source_files': len(frozen['source_sha256']), 'old_source_closures': closures,
            'prior_manifest_source_and_evidence_preserved': prior, 'full_tests': tests,
            'development_cases': len(dev), 'development_gate': gate,
            'passed_development_costs': costs, 'confirmation_cases': len(main),
            'offline_exact_replays': len(offline), 'fresh_live_replications': len(live),
            'all_source_case_receipt_actual_paid_fifo_consumption_checks_pass': True,
            'all_raw_local_payload_capture_age_threshold_outage_checks_pass': True,
            'three_test_bootstrap_and_conservative_pending_endpoint_reproduced': True,
            'completion_report_and_markdown_reproduced': True,
            'recovery_counts': {'development': devrecovery, 'confirmation': mainrecovery,
                                'offline_replay': offlinerecovery, 'fresh_live': liverecovery},
            'primary': archived['primary'], 'means': archived['means'],
            'engineering_gates': archived['engineering_gates'], 'timing': archived['timing'],
            'offline_comparisons': completion['offline_comparisons'],
            'fresh_live_cost_comparisons': completion['fresh_live_cost_comparisons'],
            'all_three_experimental_benefits': archived['all_three_experimental_benefits'],
            'production_qualified': False, 'hardware_qualified': False, 'activated': False,
            'report_sha256': digest(OUT / 'report.json'), 'verifier_sha256': digest(Path(__file__)),
            'no_worlds_forwards_fits_or_timing_executed': True,
            'global_document_and_all_current_source_manifest_deferred': True}
        audit.update(parent_global_manifest_is_not_a_dependency=True,
            available_counter_recorded_event_and_snapshot_checks_pass=True,
            full_10ms_counter_recurrence_verified_from_archives=False,
            timing_by_namespace={'development': gate['timing'], 'confirmation': archived['timing'],
                'offline_replay': archived['offline_replay_summary']['timing'],
                'fresh_live': archived['fresh_live_cost_summary']['timing']},
            counter_update_proof_scope='Frozen source and explicit software tests; sampled archives cannot reconstruct every 10ms delivery reset/scan pause.')
        (proof / 'independent-audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
        print(json.dumps({key: audit[key] for key in ('registered_source_files', 'development_cases',
            'passed_development_costs', 'confirmation_cases', 'offline_exact_replays',
            'fresh_live_replications', 'recovery_counts', 'all_three_experimental_benefits')}, indent=2))
        return audit
    sources = [path for folder in ('src', 'scripts', 'tests') for path in (ROOT / folder).rglob('*.py') if '__pycache__' not in path.parts] + [ROOT / 'pyproject.toml', ROOT / 'uv.lock']
    documents = [ROOT / name for name in DOCUMENTS]
    evidence = [path for path in OUT.rglob('*') if path.is_file() and path != manifest and not any(part.startswith('.') for part in path.relative_to(OUT).parts) and not path.name.endswith(('.lock', '.part'))]
    hashes = lambda paths: {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}
    value = {'schema': 1, 'scientific_source_commit': meta['scientific_source_commit'], 'scientific_source_tree': meta['scientific_source_tree'], 'registration_precedes_only_new_development_and_confirmation': True, 'registered_source_files': len(frozen['source_sha256']), 'original_source_closures': closures, 'source_registration_sha256': digest(OUT / 'registration/source.json'), 'prior_scientific_source_and_evidence': prior, 'previous_global_document_hashes_apply_to_their_original_commits': True, 'development_cases': len(dev), 'passed_development_costs': costs, 'confirmation_cases': len(main), 'offline_exact_replays': len(offline), 'fresh_live_replications': len(live), 'independent_confirmation_sites': 8 if costs else 0, 'new_fits': 0, 'full_tests': tests, 'recovery_counts': {'development': devrecovery, 'confirmation': mainrecovery, 'offline_replay': offlinerecovery, 'fresh_live': liverecovery}, 'actual_cost_contract_preserves_previous_overrun_veto': True, 'profile_exceedance_diagnostic_only_under_new_paid_actual_cost_contract': True, 'actual_jobs_timing_paid_fifo_consumption_and_telemetry_verified': True, 'conditional_matrix_complete_and_no_unrun_cost_fabricated': True, 'all_three_primary_tests_and_pending_endpoint_bootstraps_recomputed': True, 'frozen_report_reproduced_from_actual_archives': True, 'offline_replays_make_no_fresh_measured_deadline_claim': True, 'fresh_replications_add_no_independent_sites': True, 'primary': archived['primary'], 'timing': archived['timing'], 'all_three_experimental_benefits': archived['all_three_experimental_benefits'], 'report_sha256': digest(OUT / 'report.json'), 'hardware_qualified': False, 'production_qualified': False, 'activated': False, 'new_worlds_forwards_training_or_timing_executed_by_verifier': False, 'sources_sha256': hashes(sources), 'documents_sha256': hashes(documents), 'evidence_sha256': hashes(evidence)}
    value.update(parent_global_manifest_is_not_a_dependency=True,
        available_counter_recorded_event_and_snapshot_checks_pass=True,
        full_10ms_counter_recurrence_verified_from_archives=False,
        counter_update_proof_scope='Frozen source and explicit software tests; sampled archives cannot reconstruct every 10ms delivery reset/scan pause.')
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
    print(json.dumps({key: value[key] for key in ('registered_source_files', 'development_cases', 'passed_development_costs', 'confirmation_cases', 'offline_exact_replays', 'fresh_live_replications', 'all_three_experimental_benefits', 'production_qualified')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true', help='Write derived final metadata explicitly')
    parser.add_argument('--audit-only', action='store_true', help='Write independent raw-evidence audit; defer final global document/source manifest')
    args = parser.parse_args(); verify(write_manifest=args.write_manifest, audit_only=args.audit_only)
