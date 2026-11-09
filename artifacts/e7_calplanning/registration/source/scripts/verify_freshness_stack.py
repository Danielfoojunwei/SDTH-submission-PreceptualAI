"""Independent passive verification of the registered freshness stack.

No worlds, forwards, training or timing run in any mode. Source/test metadata
must be pinned after publication. --audit-only defers global document hashing;
--write-manifest writes only this study's derived verification manifest.
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
import verify_costaware_recovery as prior

ROOT = base.ROOT
OUT = ROOT / 'artifacts/freshness_stack'
PR20_COMMIT = '1441f68089173279536afc11fc17a3926115cd5c'
SCIENTIFIC_SOURCE_COMMIT = '5493d425538046b79bff6c4b06202892fe35d8dc'
EXPECTED_SOURCE_COUNT = 446
EXPECTED_TESTS = (1489, 1, 184.62)
EXPECTED_TEST_LOG_SHA256 = '0abe5be89ebe5fd53686eecd6ff5f5d826377ba71cfb490c488bc9c6aa4ef02a'
EXPECTED_TESTED_SOURCE_SHA256 = '0e652402527c2d7963f4455b2a0167d86f6554ecf97e1a67d58a01f2997c8b0b'
DOCUMENTS = ('README.md', 'docs/HANDOFF.md', 'docs/PAPER_PROPOSAL.md',
             'docs/FRESHNESS_STACK_DESIGN.md', 'docs/FRESHNESS_STACK_RESULTS.md')
require, read, digest, value_digest = base.require, base.read, base.digest, base.value_digest
committed, git_tree, visible = base.committed, base.git_tree, base.visible
membership, telemetry_check = base.membership, base.telemetry_check
local_freshness_check = base.local_freshness_check
HORIZONS = np.asarray([0, 1, 2, 3, 4, 5, 10, 20, 30], dtype=float)


def preserve_prior():
    result = prior.preserve_prior()
    for experiment in ('jepa_repair', 'costaware_recovery'):
        name = 'artifacts/' + experiment + '/verification/manifest.json'; raw = committed(PR20_COMMIT, name)
        require((ROOT / name).read_bytes() == raw, 'Published PR20 manifest changed')
        manifest = json.loads(raw); counts = {}
        require(len(manifest['sources_sha256']) == 360, 'PR20 source count differs')
        for group in ('sources_sha256', 'evidence_sha256'):
            for path, expected in manifest[group].items():
                require(digest(ROOT / path) == expected, 'PR20 source/evidence changed: ' + path)
            counts[group] = len(manifest[group])
        for path, expected in manifest['documents_sha256'].items():
            require(hashlib.sha256(committed(PR20_COMMIT, path)).hexdigest() == expected, 'Historical PR20 document differs')
        result[experiment] = {'manifest_sha256': digest(ROOT / name), 'preserved_entries': counts}
    return result


def registration(c):
    require(SCIENTIFIC_SOURCE_COMMIT is not None and EXPECTED_SOURCE_COUNT is not None, 'Pin published scientific source before verification')
    _, _, closures = prior.registration(c.parent)
    frozen = c.check_sources(); meta = read(OUT / 'registration/git.json'); commit = meta['scientific_source_commit']
    require(commit == SCIENTIFIC_SOURCE_COMMIT and len(frozen['source_sha256']) == EXPECTED_SOURCE_COUNT
            and meta['scientific_source_tree'] == git_tree(commit) and meta['before_new_evaluations'] is True, 'Freshness source/count/tree metadata differs')
    require(frozen['new_evaluation_data_accessed'] is False and frozen['new_fits'] == 0
            and frozen['parent_results_already_observed'] is True, 'Freshness registration scope differs')
    require((OUT / 'registration/source.json').read_bytes() == committed(commit,
            'artifacts/freshness_stack/registration/source.json'), 'Freshness registration differs from Git')
    for path, expected in frozen['source_sha256'].items():
        raw = committed(commit, path)
        require(digest(ROOT / path) == expected == hashlib.sha256(raw).hexdigest()
                and (OUT / 'registration/source' / path).read_bytes() == raw, 'Freshness current/capsule/Git differs: ' + path)
    require(frozen['previous_source_and_evidence_preserved'] == c.verify_previous_evidence()
            and frozen['passive_diagnosis_source_inputs_and_outputs_verified'] == c.verify_diagnosis_inputs()
            and frozen['observed_input_sha256'] == {str(path.relative_to(ROOT)): digest(path) for path in c.prior_inputs()}, 'Observed prior/diagnosis bindings differ')
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, 'artifacts'], cwd=ROOT, text=True).splitlines()
    prefixes = tuple('artifacts/freshness_stack/' + f + '/' for f in ('cases', 'replays', 'live_replications'))
    require(not any(name.startswith(prefixes) for name in names), 'Pre-data commit contains freshness cases')
    c.verify_fresh_sites(); p = c.protocol()
    require(p['development_sites'] == [2143000, 2143001] and p['confirmation_sites'] == list(range(2144000, 2144008))
            and p['handover_seconds'] == [4] and p['variants_by_cost'] == {'4.0': ['fifo', 'latest', 'latest_single', 'latest_single_sense']}
            and p['robots'] == 100 and p['duration_seconds'] == 60 and p['new_fits'] == 0, 'Freshness identities differ')
    new = set(p['development_sites'] + p['confirmation_sites'])
    for name in names:
        if re.fullmatch(r'artifacts/[^/]+/protocol\.json', name) and name != 'artifacts/freshness_stack/protocol.json':
            require(not new & c._integers(json.loads(committed(commit, name))), 'Freshness site overlaps an old protocol')
    require(read(OUT / 'execution_order.json') == c.execution_order(), 'Freshness captured order differs')
    return frozen, meta, dict(closures, costaware_recovery=422)


def tests_check():
    require(EXPECTED_TESTS is not None and EXPECTED_TEST_LOG_SHA256 is not None, 'Pin complete preflight test metadata')
    path = OUT / 'logs/preflight-tests.txt'; matches = re.findall(r'(?m)^([0-9]+) passed(?:, ([0-9]+) warnings?)? in ([0-9.]+)s', path.read_text())
    require(len(matches) == 1, 'No unique full-test summary')
    passed, warnings, seconds = matches[0]; actual = int(passed), int(warnings or 0), float(seconds)
    require(actual == EXPECTED_TESTS and digest(path) == EXPECTED_TEST_LOG_SHA256, 'Full-test log differs')
    receipt_path = OUT / 'logs/tested-source-sha256.json'; receipt = read(receipt_path)
    require(digest(receipt_path) == EXPECTED_TESTED_SOURCE_SHA256 and receipt['source_count'] == 616
            and receipt['before_sha256'] == receipt['after_sha256'] and len(receipt['before_sha256']) == 616
            and receipt['unchanged'] is True and receipt['new_evaluation_data_accessed'] is False
            and receipt['tests'] == {'passed': actual[0], 'warnings': actual[1], 'seconds': actual[2]}
            and receipt['test_log_sha256'] == EXPECTED_TEST_LOG_SHA256, 'Full tested-source snapshot differs')
    require(all(digest(ROOT/name) == expected for name, expected in receipt['before_sha256'].items()), 'Source/input changed since actual full suite')
    launch = read(OUT / 'logs/launch.json')
    require(launch['scientific_source_commit'] == SCIENTIFIC_SOURCE_COMMIT and launch['workers'] == 2
            and launch['module'] == 'wirelessjev.freshness_stack.study'
            and launch['phase_order'] == ['development', 'confirmation', 'report']
            and launch['thread_limits'] == {name: '1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
            'Actual launch/source/worker/thread contract differs')
    return {'passed': actual[0], 'warnings': actual[1], 'seconds': actual[2], 'log': str(path.relative_to(ROOT)),
            'sha256': digest(path), 'tested_source_entries': 616, 'tested_source_receipt_sha256': digest(receipt_path),
            'launch_sha256': digest(OUT / 'logs/launch.json')}


def sparse_jobs_check(row, p, profile, original=None):
    """Independent admission/payment arithmetic, source timestamps and held age."""
    offline = original is not None; jobs = row['jobs']; arrivals = row['arrivals']
    mode = 'fifo' if row['controller_variant'] == 'fifo' else 'latest'
    require(row['queue_mode'] == mode and len(arrivals) == 60 and 1 <= len(jobs) <= 60, 'Sparse request membership differs')
    if offline: require(arrivals == original['arrivals'], 'Offline arrivals changed')
    previous = 0.; fresh_previous = 0.; ordinal = 0; ready = []; sources = []
    for source, arrival in enumerate(arrivals):
        accepted = mode == 'fifo' or source >= previous
        expected_arrival = {'decision': source, 'arrival_slot': source * 100, 'accepted': accepted,
            'job_index': ordinal if accepted else None, 'reason': 'accepted' if accepted else 'busy_previous_job',
            'previous_ready_seconds': previous, 'previous_ready_slot': math.ceil(previous * 100 - 1e-10), 'mode': mode}
        require(arrival == expected_arrival and type(arrival['accepted']) is bool, 'Busy arrival admitted or assigned hidden work')
        if not accepted: continue
        require(ordinal < len(jobs), 'Accepted request lacks a measured job')
        job = jobs[ordinal]; kind = job['kind']
        require(job['decision'] == job['source_decision'] == source and job['job_index'] == ordinal
                and job['history_cutoff_exclusive'] == 60 + source and job['mode'] == mode
                and job['issue_seconds'] == float(source) and job['start_seconds'] == max(float(source), previous), 'Source/job ordinal/history/start differs')
        fields = ('actual_prediction_seconds', 'actual_issue_seconds', 'actual_compute_seconds', 'profile_seconds',
                  'charged_compute_seconds', 'charged_seconds', 'shared_extra_seconds')
        require(kind in profile and all(type(job[f]) in (int, float) and math.isfinite(job[f]) and job[f] >= 0 for f in fields), 'Invalid actual paid timing')
        actual = max(job['actual_prediction_seconds'], job['actual_issue_seconds']); compute = max(profile[kind], actual)
        expected = {'actual_compute_seconds': actual, 'profile_seconds': profile[kind], 'charged_compute_seconds': compute,
                    'shared_extra_seconds': p['extra_compute_seconds'], 'charged_seconds': compute + p['extra_compute_seconds']}
        require(all(math.isclose(job[f], value, rel_tol=0, abs_tol=1e-12) for f, value in expected.items()), 'Sparse actual issue work undercharged')
        require(job['runtime_overrun'] is (actual > profile[kind]) and job['predictor_runtime_overrun'] is (job['actual_prediction_seconds'] > profile[kind])
                and job['accounting_mode'] == ('recorded_schedule_offline' if offline else 'actual_cost')
                and job['offline_recorded_schedule'] is offline and job['completion_uses_actual_elapsed'] is (not offline), 'Capacity/offline accounting boundary differs')
        if offline:
            before = original['jobs'][ordinal]
            inherited = fields + ('decision', 'source_decision', 'job_index', 'issue_seconds', 'start_seconds', 'mode', 'kind', 'history_cutoff_exclusive', 'ready_slot')
            require(all(job[f] == before[f] for f in inherited) and job['forecast_sha256'] == before['forecast_sha256']
                    and job['forecast_std_sha256'] == before['forecast_std_sha256'] and job['matches_recorded_forecast'] is True
                    and job['recorded_completion_used_actual_elapsed'] is True, 'Offline original schedule/forecasts changed')
            fresh = max(job['fresh_actual_prediction_seconds'], job['fresh_actual_issue_seconds'])
            require(math.isfinite(fresh) and fresh >= 0 and job['fresh_actual_compute_seconds'] == fresh
                    and job['exceeds_original_recorded_compute'] is (fresh > before['charged_compute_seconds']), 'Offline fresh diagnostic differs')
            fresh_start = max(float(source), fresh_previous); fresh_compute = max(profile[kind], fresh)
            fresh_service = fresh_compute+p['extra_compute_seconds']; fresh_previous = fresh_start+fresh_service
            require(job['fresh_start_seconds'] == fresh_start and job['fresh_charged_compute_seconds'] == fresh_compute
                    and job['fresh_charged_seconds'] == fresh_service
                    and job['fresh_ready_slot'] == math.ceil(fresh_previous*100-1e-10),
                    'Offline diagnostic fresh FIFO differs from newly measured work')
        previous = max(float(source), previous) + job['charged_seconds']
        require(type(job['ready_slot']) is int and job['ready_slot'] == math.ceil(previous * 100 - 1e-10), 'Sparse completion before fully paid FIFO finish')
        require(job['forecast_dtype'] == 'float32' and len(job['forecast_shape']) == 4
                and job['forecast_shape'][0:2] == [100, 9] and job['forecast_shape'][-1] == 3
                and all(re.fullmatch('[0-9a-f]{64}', job[f]) for f in ('forecast_sha256', 'forecast_std_sha256')), 'Forecast axes/dtype/hash differs')
        ready.append(job['ready_slot']); sources.append(source); ordinal += 1
    require(ordinal == len(jobs) and row['accepted_decisions'] == sources
            and row['forecast_metric_decisions'] == sources, 'Extra job or uncomputed accuracy query included')
    expected_consumption = []
    for slot in sorted(set(ready)):
        if slot < 6000:
            index = max(i for i, end in enumerate(ready) if end <= slot)
            expected_consumption.append({'slot': slot, 'forecast_decision': sources[index], 'ready_slot': ready[index], 'age_seconds': slot / 100 - sources[index]})
    require(row['forecast_consumption'] == expected_consumption, 'Early/missing/source-ordinal forecast consumption')
    held = [(expected_consumption[i+1]['slot'] - 1 if i+1 < len(expected_consumption) else 5999) / 100 - event['forecast_decision']
            for i, event in enumerate(expected_consumption)]
    snapshots = row['result']['freshness_execution']['held_forecast_age']
    require(len(snapshots) == 60, 'Missing held-age boundary snapshots')
    for second, snapshot in enumerate(snapshots, 1):
        boundary = second * 100; consumed = [e for e in expected_consumption if e['slot'] < boundary]
        source = consumed[-1]['forecast_decision'] if consumed else None
        require(snapshot == {'slot_end': boundary, 'source_decision': source,
            'age_seconds': second - source if source is not None else None,
            'phase': 'right_boundary_before_next_slot_consumption'}, 'Held snapshot uses ordinal or wrong boundary phase')
    require(row['runtime_overrun'] is any(j['runtime_overrun'] for j in jobs)
            and row['predictor_runtime_overrun'] is any(j['predictor_runtime_overrun'] for j in jobs), 'Case profile diagnostic differs')
    require(row['timing_audit']['maximum_held_forecast_age_seconds'] == max(held, default=None), 'All-slot held age differs')
    return {'arrivals': 60, 'accepted_jobs': len(jobs), 'skipped_arrivals': 60-len(jobs),
            'consumptions': len(expected_consumption), 'maximum_held_age_seconds': max(held, default=None)}


def sensing_check(row, p):
    """Recompute logged guards/arithmetic; never reconstruct private RF truth."""
    execution = row['result']['freshness_execution']; enabled = row['controller_variant'] == 'latest_single_sense'
    require(execution['risk_gated_scans_enabled'] is enabled and execution['scan_guard_rssi_dbm'] == -67.
            and execution['maximum_forecast_age_seconds'] == execution['lookahead_seconds'] == execution['reassessment_seconds'] == 5.
            and execution['emergency_due_scan_bypasses_postponement'] is True
            and execution['postponement_preserves_due_and_capture_timestamps'] is True, 'Sensing contract differs')
    assessments = execution['scan_assessments']; starts = execution['scan_starts']
    if not enabled: require(not assessments, 'Non-sensing arm ran adaptive assessment')
    start_map = {}
    for event in starts:
        identity = event['slot'], event['robot']
        require(identity not in start_map and 0 <= event['slot'] < 6000 and 0 <= event['robot'] < 100
                and event['complete_slot'] == event['slot'] + 10, 'Scan start/paid90ms completion differs')
        start_map[identity] = event
    require(row['telemetry']['scans_started'] == len(starts), 'Started scan count differs')
    for event in row['local_scan_observability']['completion_events']:
        start = start_map.get((event['measurement_slot'], event['robot']))
        require(start is not None and start['complete_slot'] == event['complete_slot'], 'Local observation lacks a genuine paid scan start')
    blocked = {}; observed = set(); suppressed = 0; consumed = row['forecast_consumption']
    jobs = {job['decision']: job for job in row['jobs']}
    for event in sorted(assessments, key=lambda e: (e['slot'], e['robot'])):
        slot, robot = event['slot'], event['robot']; identity = slot, robot
        require(identity not in observed and 0 <= slot < 6000 and 0 <= robot < 100, 'Duplicate/invalid sensing assessment')
        observed.add(identity)
        if event['reason'] == 'emergency_silence_bypass':
            require(event['suppress_scan'] is False and event['control_gap_slots'] >= 20
                    and event['primary_available'] is True and identity in start_map,
                    'Emergency silence failed to bypass postponement')
            blocked[robot] = 0; continue
        require(slot >= blocked.get(robot, 0), 'Ordinary reassessment before fixed5s delay')
        measured, noise, tx = [np.asarray(event[name], dtype=float) for name in ('measured_snr_db', 'noise_dbm', 'ap_tx_dbm')]
        require(all(a.shape == (3,) and np.isfinite(a).all() for a in (measured, noise, tx)), 'Invalid actual pilot inputs')
        rssi = measured + noise + tx - 15
        require(np.allclose(rssi, event['current_rssi_dbm'], rtol=0, atol=1e-10)
                and math.isclose(float(rssi.max()), event['current_best_rssi_dbm'], rel_tol=0, abs_tol=1e-10), 'Pilot RSSI arithmetic differs')
        capture, refresh = event['pilot_capture_slot'], event['pilot_refresh_slot']
        valid_pilot = (event['pilot_site'] == event['current_site'] and type(capture) is int and type(refresh) is int
            and 0 <= slot-refresh < 10 and 0 <= slot-capture < 20)
        reason = None; suppress = False
        if not valid_pilot: reason = 'stale_or_wrong_association_pilot'
        elif not event['primary_available'] or event['control_gap_slots'] >= 20: reason = 'control_recovery_required'
        elif rssi.max() < -67: reason = 'weak_current_pilot'
        else:
            require(refresh % 10 == 0 and capture == max(0, refresh-10), 'Pilot capture/refresh cadence differs')
            latest = [e for e in consumed if e['slot'] <= slot]
            if not latest:
                require(event['source_decision'] is None, 'Invented consumed forecast during fallback')
                reason = 'missing_forecast'
            else:
                source = latest[-1]['forecast_decision']; age = slot / 100-source
                require(event['source_decision'] == source and source in jobs
                        and event['forecast_age_seconds'] == age, 'Sensing source timestamp/held age differs')
                if not 0 <= age <= 5: reason = 'stale_or_future_forecast'
                elif age+5 > HORIZONS[-1]: reason = 'unsupported_forecast_horizon'
                else:
                    original = np.asarray(event['original_forecast_snr_db'], dtype=float)
                    require(original.shape == (9, 3) and np.isfinite(original).all()
                            and event['forecast_ready_slot'] == jobs[source]['ready_slot'] <= slot
                            and event['forecast_sha256'] == jobs[source]['forecast_sha256'], 'Consumed9x3 evidence/completion/hash binding differs')
                    shifted = np.stack([np.interp(age+np.arange(6), HORIZONS, original[:, b]) for b in range(3)], -1)
                    innovation = shifted-shifted[:1]; score = float((rssi[None]+innovation).min(0).max())
                    require(event['prediction_sample_seconds'] == list(range(6))
                            and np.allclose(event['forecast_at_age_plus_samples_snr_db'], shifted, rtol=0, atol=1e-10)
                            and np.allclose(event['route_innovation_db'], innovation, rtol=0, atol=1e-10)
                            and math.isclose(event['predicted_best_worst_rssi_dbm'], score, rel_tol=0, abs_tol=1e-10), 'Fixed-band route-risk arithmetic differs')
                    suppress = score >= -67
                    reason = 'strong_current_and_predicted_link' if suppress else 'predicted_link_risk'
        require(event['reason'] == reason and event['suppress_scan'] is suppress, 'Scan guard reason/threshold differs')
        require((identity not in start_map) if suppress else (identity in start_map), 'Suppressed scan created work or permitted scan omitted')
        if suppress: blocked[robot] = slot+500; suppressed += 1
        else: blocked[robot] = 0
    if enabled: require(set(start_map) <= observed, 'Adaptive scan start lacks its assessment/bypass')
    if row['controller_variant'] in ('latest_single', 'latest_single_sense'):
        require(row['result']['dup_fraction'] == 0
                and all(all(band == -1 for band in event['secondary_band']) for event in row['result']['packet_detail']['action_trace']),
                'Single-radio arm retained simultaneous duplication')
    return {'assessments': len(assessments), 'suppressed_assessments': suppressed, 'paid_scan_starts': len(starts)}


def load_rows(c, rollout, keys, *, replay=False, live=False, originals=None):
    p = c.protocol(); folder = OUT / ('replays' if replay else 'live_replications' if live else 'cases')
    if replay or live: membership(folder, keys, rollout)
    rows = []; counts = {f: 0 for f in ('watchdog_checks', 'emergency_handovers', 'emergency_scans')}
    for key in keys:
        name = rollout.case_name(key); path = folder / (name+'.json.gz'); receipt = read(folder / (name+'.receipt.json'))
        require(receipt['archive_sha256'] == digest(path), 'Freshness archive bytes changed')
        row = json.loads(gzip.decompress(path.read_bytes()))
        require(receipt['result_sha256'] == value_digest(row['result'])
                and row == rollout.load_case(key, replay=replay, live_replication=live), 'Freshness raw result/identity differs')
        require(row['replay'] is replay and row['live_replication'] is live and row['new_fits'] == 0
                and row['hardware_qualified'] is False and row['production_qualified'] is False and row['simulation_only'] is True,
                'Freshness namespace/training/qualification differs')
        original = originals[tuple(key)] if replay else None; profile = c.latency()['profiles'][row['family']]
        sparse_jobs_check(row, p, profile, original)
        audit = rollout.timing_audit(row['jobs'], row['forecast_consumption'], profile, p['extra_compute_seconds'], p['duration_seconds'],
            arrivals=row['arrivals'], mode=row['queue_mode'], accounting_mode=row['accounting_mode'],
            recorded_jobs=original['jobs'] if replay else None, recorded_arrivals=original['arrivals'] if replay else None)
        require(audit == row['timing_audit'] and audit['passes'] is True and audit['offline_schedule_only'] is replay
                and audit['real_time_contract_checked'] is (not replay), 'Sparse paid/FIFO/held-age audit differs')
        if not replay: require(c.verify_paid_row(row) is True, 'Fresh sparse paid work failed')
        require(c.causal_history(row) is True, 'Sparse source/history causality failed')
        used_sources = [event['forecast_decision'] for event in row['forecast_consumption']]
        require([event['decision'] for event in row['decision_trace']] == used_sources,
                'Native controller consumed an ordinal/skipped forecast row')
        require([(event['decision'], event['hook_role']) for event in row['admission_trace']]
                == [(source, role) for source in used_sources for role in ('planned', 'reactive')],
                'Ordinary controller admission/source identities changed')
        telemetry_check(row, p); local_freshness_check(row, p); sensing_check(row, p)
        # Every nested arm uses the frozen cost-aware recovery controller.
        for field, value in prior.recovery_check(dict(row, controller_variant='repair'), p).items(): counts[field] += value
        require(row['action_trace_sha256'] == value_digest(row['result']['packet_detail']['action_trace'])
                and all(re.fullmatch('[0-9a-f]{64}', row[f]) for f in ('forecast_mean_sha256', 'forecast_std_sha256')),
                'Sparse forecast/action digest differs')
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
    secondary = report['secondary_component_contrasts']
    require(len(secondary) == 6, 'Six nested descriptive contrasts missing')
    for contrast in secondary:
        require(contrast['statistical_pass'] is None and contrast['statistical_superiority_claimed'] is False
                and contrast['experimental_benefit'] is False, 'Intermediate arm received an inferential qualification')
        if not costs:
            require(contrast['status'] == 'not_run_development_veto' and contrast['mean_difference'] is None,
                    'Unrun intermediate arm gained evidence')
            continue
        values = []
        for site in p['confirmation_sites']:
            a = indexed['confirmation', site, 4., contrast['candidate_family'], contrast['candidate_variant']]['result']
            b = indexed['confirmation', site, 4., contrast['reference_family'], contrast['reference_variant']]['result']
            values.append(a['control_misses']/a['control_sent']-b['control_misses']/b['control_sent'])
        draws = np.asarray(values)[indices].mean(1)
        require(contrast['upper_quantile'] == .975 and contrast['paired_site_differences'] == values
                and contrast['mean_difference'] == float(np.mean(values))
                and contrast['descriptive_95_interval'] == np.quantile(draws, [.025, .975]).tolist(),
                'Descriptive nested bootstrap differs')


def verify(*, write_manifest=False, audit_only=False):
    require((OUT / 'report.json').is_file() and (OUT / 'report.md').is_file(), 'Complete the registered pipeline/report first')
    from wirelessjev.freshness_stack import common as c, rollout, study, report as stats
    frozen, meta, closures = registration(c); prior = preserve_prior(); p = c.protocol()
    devkeys = study.ordered_keys('development'); require(len(devkeys) == 16, 'Development matrix changed')
    dev, devrecovery = load_rows(c, rollout, devkeys); study._fresh(dev, devkeys)
    gate = stats.development_gate(dev, p, validation=c.development_validation(dev))
    require(gate == c.verified_development(), 'Development permission differs from actual rows')
    for entry in gate['per_cost']:
        for check in entry['family_checks']:
            selected = lambda variant: [r['result'] for r in dev if r['family'] == check['family'] and r['controller_variant'] == variant and r['handover_seconds'] == entry['handover_seconds']]
            rate = lambda rows: Fraction(sum(r['control_misses'] for r in rows), sum(r['control_sent'] for r in rows))
            difference = rate(selected('latest_single_sense')) - rate(selected('fifo'))
            require(check['exact_difference_numerator'] == difference.numerator and check['exact_difference_denominator'] == difference.denominator and check['severe_regression'] is (difference > Fraction(str(p['development_severe_regression_absolute']))), 'Development guard differs from exact counters')
    costs = gate['passed_costs']; keys = study.ordered_keys('confirmation', costs); repeats = study.replication_keys(costs)
    membership(OUT / 'cases', devkeys + keys, rollout)
    require(len(keys) <= 64 and len(repeats) <= 8 and all(key[1] == p['confirmation_sites'][0] for key in repeats), 'Replication scope changed')
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
        audit.update(frozen_prior_manifests_are_dependencies_current_global_docs_may_change=True,
            available_counter_recorded_event_and_snapshot_checks_pass=True,
            full_10ms_counter_recurrence_verified_from_archives=False,
            timing_by_namespace={'development': gate['timing'], 'confirmation': archived['timing'],
                'offline_replay': archived['offline_replay_summary']['timing'],
                'fresh_live': archived['fresh_live_cost_summary']['timing']},
            counter_update_proof_scope='Frozen source and explicit software tests; sampled archives cannot reconstruct every 10ms delivery reset/scan pause.')
        audit.update(sparse_accepted_sources_and_all_arrivals_independently_verified=True,
            exact_all_slot_interval_held_age_verified=True, logged_sensing_guards_and_arithmetic_verified=True,
            no_intermediate_arm_selected_or_qualified=True,
            sensing_proof_limits=['Logged9x3 extraction is frozen-source/exact-replay backed; the slice does not reconstruct its full-forecast hash.',
                'Pilot capture/AP/refresh and arithmetic are checked; private physical pilot values are not resimulated.',
                'Postponement/start consistency is checked; sampled logs do not reconstruct every due opportunity/control delivery.'])
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
    value.update(frozen_prior_manifests_are_dependencies_current_global_docs_may_change=True,
        available_counter_recorded_event_and_snapshot_checks_pass=True,
        full_10ms_counter_recurrence_verified_from_archives=False,
        counter_update_proof_scope='Frozen source and explicit software tests; sampled archives cannot reconstruct every 10ms delivery reset/scan pause.')
    value.update(sparse_accepted_sources_and_all_arrivals_independently_verified=True,
        exact_all_slot_interval_held_age_verified=True, logged_sensing_guards_and_arithmetic_verified=True,
        no_intermediate_arm_selected_or_qualified=True,
        sensing_proof_limits=['9x3 extraction is frozen-source/exact-replay backed, not a separate full-frame cryptographic reconstruction.',
            'Pilot values are not privately resimulated; sampled logs do not reconstruct every due/control-delivery transition.'])
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
