"""Nested stack orchestration, sparse request histories and actual-cost permission boundaries.

All rows below are tiny explicit arithmetic fixtures, not research-world runs
or measured performance evidence. Native runtime execution is tested separately.
"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from wirelessjev.pilot_fastloop import common as c, rollout as r, study as s, matched as m


def packet_row(key, p, *, replay=False, live=False, severe=False, profile=.01, actual=None):
    stage, site, cost, family, variant = key
    n, duration = p['robots'], p['duration_seconds']
    miss = 2 if variant == 'base' else 1
    if severe and variant == 'pilot_fast': miss = 10
    if live: miss += 1
    missed, sent, gaps = [miss] * n, [duration * 20] * n, [5] * n
    detail = {'misses': missed, 'sent': sent, 'max_gap': gaps,
        'pending_control_packet_age_slots': [-1] * n, 'pending_control_packets': 0,
        'cumulative': [{'slot_end': second * 100, 'misses': missed,
            'sent': [second * 20] * n, 'max_gap': gaps} for second in range(1, duration + 1)]}
    flow = {'video_requested_bits': 1000., 'video_generated_bits': 1000.,
        'video_delivered_bits': 900., 'video_pending_bits': 50., 'video_expired_bits': 50.,
        'bulk_generated_bits': 1000., 'bulk_delivered_bits': 900.,
        'final_bulk_backlog_bits': 100., 'bulk_overflow_drop_bits': 0.}
    jobs, arrivals, consumed, previous = [], [], [], 0.
    actual = (.02 if live else .002) if actual is None else actual
    mode = 'latest'
    for source in range(duration):
        accepted = mode == 'fifo' or source >= previous
        arrivals.append({'decision': source, 'arrival_slot': source * 100,
            'accepted': accepted, 'job_index': len(jobs) if accepted else None,
            'reason': 'accepted' if accepted else 'busy_previous_job',
            'previous_ready_seconds': previous,
            'previous_ready_slot': __import__('math').ceil(previous * 100 - 1e-10), 'mode': mode})
        if not accepted: continue
        ordinal = len(jobs)
        charge = max(profile, actual)
        start = max(float(source), previous)
        previous = start + charge + .5
        ready = __import__('math').ceil(previous * 100 - 1e-10)
        jobs.append({'decision': source, 'source_decision': source, 'job_index': ordinal,
            'issue_seconds': float(source), 'start_seconds': start, 'mode': mode,
            'history_cutoff_exclusive': 60 + source,
            'kind': 'cold' if ordinal == 0 else 'warm', 'actual_prediction_seconds': actual / 2,
            'actual_issue_seconds': actual, 'actual_compute_seconds': actual,
            'profile_seconds': profile, 'charged_compute_seconds': charge,
            'shared_extra_seconds': .5, 'charged_seconds': charge + .5, 'ready_slot': ready,
            'runtime_overrun': actual > profile, 'predictor_runtime_overrun': actual / 2 > profile,
            'forecast_sha256': f'{source + 1:064x}', 'forecast_std_sha256': f'{source + 21:064x}',
            'accounting_mode': 'actual_cost', 'offline_recorded_schedule': False,
            'completion_uses_actual_elapsed': True})
        if ready < duration * 100:
            consumed.append({'slot': ready, 'forecast_decision': source,
                'ready_slot': ready, 'age_seconds': ready / 100 - source})
    # Deterministic local-cache provenance, with the same schema as native
    # rollout exports. Unknown bands have neither a capture nor an observation.
    unknown = int(np.iinfo(np.int64).min)
    heard = np.array([True, True, True, False, False, False], dtype=bool)
    values = np.array([-68., -69., -71., 0., 0., 0.], dtype='f4')
    local_observability = {
        'initial_capture_slots': [[-100, -100, unknown, -500, unknown, unknown]
                                  for _ in range(n)],
        'initial_heard': [[True, True, False, True, False, False] for _ in range(n)],
        'initial_rssi_dbm': [[-70. - i, -72. - i, -100., -80. - i, -100., -100.]
                             for i in range(n)],
        'completion_events': [{'robot': 0, 'measurement_slot': 10,
            'complete_slot': 19, 'processed_slot': 20, 'heard_entries': 3,
            'heard_links': [0, 1, 2], 'heard_rssi_dbm': values[heard].tolist(),
            'payload_sha256': hashlib.sha256(values.tobytes() + heard.tobytes()).hexdigest()}]}
    return {'completed': True, 'case': list(key), 'stage': stage, 'site': site,
        'handover_seconds': cost, 'family': family, 'controller_variant': variant,
        'training_seed': None, 'replay': replay, 'live_replication': live,
        'accounting_mode': 'actual_cost', 'causality_pass': True,
        'paid_cost_and_consumption_pass': True, 'timing_safety_pass': True,
        'runtime_overrun': actual > profile, 'predictor_runtime_overrun': actual / 2 > profile,
        'arrivals': arrivals, 'queue_mode': mode,
        'accepted_decisions': [job['decision'] for job in jobs],
        'forecast_metric_decisions': [job['decision'] for job in jobs],
        'jobs': jobs, 'forecast_consumption': consumed,
        'forecast_mean_sha256': 'a' * 64, 'forecast_std_sha256': 'b' * 64,
        'pilot_enabled': p['variants'][variant]['pilot_reporting'],
        'fast_local_enabled': p['variants'][variant]['fast_local'],
        'access_points': 2, 'pilot_reporting': {'unit_fixture': True,
            'enabled': p['variants'][variant]['pilot_reporting'], 'new_rf_reads': 0, 'extra_control_frames': 0},
        'fast_local': {'fast_local_enabled': p['variants'][variant]['fast_local'],
            'extra_RF_or_model_calls': False, 'helper_work_independently_scheduled': False},
        'fast_local_timing': {'calls': 0, 'total_seconds': 0., 'max_seconds': 0.}, 'matched_inputs': None,
        'telemetry': {'events': []}, 'decision_trace': [{'decision': 0}],
        'local_scan_observability': local_observability,
        'admission_trace': [{'decision': 0, 'mask': [True] * n}], 'action_trace_sha256': 'c' * 64,
        'result': {'control_misses': sum(missed), 'control_sent': sum(sent),
            'control_miss': sum(missed) / sum(sent), 'packet_detail': detail,
            'packet_accounting': flow}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    p = json.loads(c.PROTOCOL.read_text())
    p.update(development_sites=[619001, 619002], confirmation_sites=[619101, 619102],
             robots=2, duration_seconds=3, cumulative_metrics_seconds=[1, 2, 3],
             bootstrap_replicates=32, bootstrap_seed=7)
    monkeypatch.setattr(c, 'OUT', tmp_path)
    monkeypatch.setattr(c, 'protocol', lambda: deepcopy(p))
    monkeypatch.setattr(c, 'source_hashes', lambda: {'unit_fixture_only': 'fixed'})
    monkeypatch.setattr(c, 'check_sources', lambda: {'source_sha256': c.source_hashes()})
    monkeypatch.setattr(c, 'latency', lambda: {'profiles': {
        family: {'cold': .01, 'warm': .01, 'refit': .01} for family in p['families']}})
    c.write_once(tmp_path / 'execution_order.json', c.execution_order())
    state = SimpleNamespace(p=p, calls=[], blocks=[], severe_costs=set(), active=False)

    def load(key, *, replay=False, live_replication=False):
        folder = tmp_path / ('replays' if replay else 'live_replications' if live_replication else 'cases')
        return r._read_record(folder, key, replay=replay, live_replication=live_replication)

    def run(key, *, replay=False, live_replication=False):
        existing = load(key, replay=replay, live_replication=live_replication)
        if existing is not None: return existing
        state.calls.append((tuple(key), replay, live_replication))
        row = packet_row(key, p, live=live_replication, severe=key[2] in state.severe_costs)
        if replay:
            original = load(key)
            row = deepcopy(original)
            row.update(replay=True, live_replication=False, accounting_mode='recorded_schedule_offline',
                recorded_schedule_sha256=c.value_digest({'jobs': original['jobs'], 'arrivals': original['arrivals']}),
                recorded_arrivals_sha256=c.value_digest(original['arrivals']),
                source_main_archive_sha256=c.digest(tmp_path / 'cases' / (r.case_name(key) + '.json.gz')))
            for job in row['jobs']:
                job.update(accounting_mode='recorded_schedule_offline', offline_recorded_schedule=True,
                    completion_uses_actual_elapsed=False, recorded_completion_used_actual_elapsed=True,
                    recorded_forecast_sha256=job['forecast_sha256'], recorded_forecast_std_sha256=job['forecast_std_sha256'],
                    fresh_actual_prediction_seconds=.001, fresh_actual_issue_seconds=.002,
                    fresh_actual_compute_seconds=.002, fresh_runtime_overrun=False,
                    exceeds_original_recorded_compute=False, fresh_exceeds_original_actual_compute=False,
                    matches_recorded_forecast=True)
        folder = tmp_path / ('replays' if replay else 'live_replications' if live_replication else 'cases')
        r._write_record(folder, key, row)
        return row

    class Pool:
        def __init__(self, *, max_workers):
            assert max_workers == 2
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def map(self, fn, tasks):
            assert not state.active, 'A previous site was not completed before the next submission'
            state.active = True
            tasks = list(tasks)
            assert len({task[0][1] for task in tasks}) == 1
            state.blocks.append([task[0] for task in tasks])
            for task in tasks: yield fn(task)
            state.active = False
    monkeypatch.setattr(r, 'load_case', load)
    monkeypatch.setattr(r, 'run_case', run)
    monkeypatch.setattr(s, 'ProcessPoolExecutor', Pool)
    state.matched_calls, state.phase_trace = [], []
    original_run = r.run_case
    def traced_run(key, **kwargs):
        state.phase_trace.append(('native', key[0], kwargs.get('replay', False), kwargs.get('live_replication', False)))
        return original_run(key, **kwargs)
    monkeypatch.setattr(r, 'run_case', traced_run)
    def paths(key, *, replay=False):
        folder = tmp_path / 'matched' / ('replays' if replay else 'cases')
        name = '-'.join(map(str, key))
        return folder / (name + '.json'), folder / (name + '.receipt.json')
    def matched_load(key, *, replay=False):
        archive, receipt = paths(key, replay=replay)
        if not archive.exists() and not receipt.exists(): return None
        if not archive.exists() or not receipt.exists(): raise RuntimeError('Partial matched archive')
        value = json.loads(archive.read_text())
        if json.loads(receipt.read_text())['archive_sha256'] != c.digest(archive):
            raise RuntimeError('Matched source/archive identity changed')
        return value
    def matched_validate(row):
        if row['metrics']['short_route_delta_rmse_db'] != (1. if row['family'] == 'frozen_fleet_jepa' else 2.):
            raise RuntimeError('Unit raw-array metric integrity mismatch')
        if row.get('unit_source') != 'fixed': raise RuntimeError('Matched source changed')
        if row.get('unit_query') != [0, 1, 2]: raise RuntimeError('Matched causality changed')
        for flag in ('source_and_archive_pass', 'causality_pass', 'matched_inputs_pass', 'prediction_integrity_pass'):
            if row.get(flag) is not True: raise RuntimeError('Matched proof flag changed')
        return True
    def matched_run(key, *, replay=False):
        state.phase_trace.append(('matched', key[0], replay))
        before = matched_load(key, replay=replay)
        if before is not None: return before
        state.matched_calls.append((tuple(key), replay))
        row = {'case': list(key), 'stage': key[0], 'site': key[1], 'family': key[2],
            'completed': True, 'replay': replay, 'offline_only': True, 'new_fits': 0, 'training_seed': None,
            'reference_case': [key[0], key[1], 4., 'radiomap', 'pilot_fast'],
            'matched_inputs': {'fixture_reference': key[1]}, 'reference_archive_sha256': 'a' * 64,
            'fresh_measured_deadline_claim': False,
            'metrics': {'short_route_delta_rmse_db': 1. if key[2] == 'frozen_fleet_jepa' else 2.,
                'signal_rmse_db': 3., 'all_route_delta_rmse_db': 2., 'ap_choice_regret_db': .5},
            'source_and_archive_pass': True, 'causality_pass': True,
            'matched_inputs_pass': True, 'prediction_integrity_pass': True,
            'unit_source': 'fixed', 'unit_query': [0, 1, 2]}
        archive, receipt = paths(key, replay=replay)
        archive.parent.mkdir(parents=True, exist_ok=True)
        c.write_once(archive, row)
        c.write_once(receipt, {'archive_sha256': c.digest(archive)})
        return row
    def compare(a, b):
        checks = {'source': a['unit_source'] == b['unit_source'],
                  'query': a['unit_query'] == b['unit_query'], 'metrics': a['metrics'] == b['metrics']}
        return {'case': a['case'], 'exact': all(checks.values()), 'checks': checks,
                'fresh_measured_deadline_claim': False}
    monkeypatch.setattr(m, 'archive_paths', paths)
    monkeypatch.setattr(m, 'load_case', matched_load)
    monkeypatch.setattr(m, 'run_case', matched_run)
    monkeypatch.setattr(m, 'validate_row', matched_validate)
    monkeypatch.setattr(m, 'compare_replay', compare)
    # Native typed-payload mechanics have separate real-kernel tests. Here the
    # explicit arithmetic fixtures isolate request ledgers and phase permissions.
    monkeypatch.setattr(r, '_causal_telemetry', lambda *a, **k: True)
    return state


def test_capture_order_and_complete_site_blocks(setup):
    keys = s.ordered_keys('development')
    assert keys == [tuple(k) for k in c.execution_order()['development']]
    assert len(keys) == 16
    s.execute(keys)
    assert [block[0][1] for block in setup.blocks] == setup.p['development_sites']
    assert [key for block in setup.blocks for key in block] == keys


@pytest.mark.parametrize('workers', [1, 3, 0, True])
def test_only_registered_two_workers_allowed(setup, workers):
    with pytest.raises(ValueError, match='two registered'):
        s.execute(s.ordered_keys('development'), workers)
    assert not setup.calls


@pytest.mark.parametrize('mutation', ['swap', 'drop', 'duplicate'])
def test_captured_order_tampering_rejected_before_case(setup, mutation):
    path = c.OUT / 'execution_order.json'; value = c.read_json(path)
    if mutation == 'swap': value['development'][0], value['development'][1] = value['development'][1], value['development'][0]
    elif mutation == 'drop': value['development'].pop()
    else: value['development'][1] = value['development'][0]
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='execution order'):
        s.ordered_keys('development')
    assert not setup.calls


def test_development_seal_matches_independent_common_guard(setup):
    result = s.development()
    assert set(result) == {'gate', 'validation', 'source_sha256', 'execution_order_sha256'}
    assert result['gate'] == c.verified_development()
    assert result['gate']['passed_costs'] == [4.]
    assert c.read_json(c.OUT / 'qualification.json')['recomputed_from_raw_archives'] is True


@pytest.mark.parametrize('target', ['qualification', 'development'])
def test_forged_permissions_do_not_open_confirmation(setup, target):
    s.development(); setup.calls.clear()
    path = c.OUT / (target + '.json'); value = c.read_json(path)
    if target == 'qualification': value['passed_costs'] = []
    else: value['gate']['passed_costs'] = []
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='seal|permission'):
        s.confirmation()
    assert not setup.calls


def test_permission_failure_happens_before_executor_or_world(setup, monkeypatch):
    def forbidden(): raise RuntimeError('Raw development permission rejected')
    monkeypatch.setattr(c, 'verified_development', forbidden)
    monkeypatch.setattr(s, 'ProcessPoolExecutor', lambda **k: pytest.fail('Executor opened before permission'))
    with pytest.raises(RuntimeError, match='permission rejected'):
        s.confirmation()
    assert not setup.calls


def test_full_conditional_pipeline_keeps_offline_and_fresh_separate(setup):
    s.development(); result = s.confirmation(); report = s.report()
    assert result['case_count'] == 16
    assert result['offline_replay_count'] == result['fresh_live_replication_count'] == 8
    assert result['all_offline_exact'] and result['all_fresh_paid_cost_checks_pass']
    assert report['case_count'] == 16
    assert report['proof']['experimental_claim_integrity_pass']
    assert report['offline_replay_summary']['independent_sites_added'] == 0
    assert report['fresh_live_cost_summary']['independent_sites_added'] == 0
    assert all(row['sensitivity_only'] and not row['bitwise_equality_required'] for row in result['fresh_live_cost_comparisons'])
    assert any(row['changed_ready_jobs'] for row in result['fresh_live_cost_comparisons'])
    assert report['timing']['profile_exceeded_cases'] == 0
    assert report['fresh_live_cost_summary']['timing']['profile_exceeded_cases'] == 8




def test_slow_veto_creates_no_confirmation_or_replication_cases(setup):
    setup.severe_costs = {4.}
    assert s.development()['gate']['passed_costs'] == []
    result = s.confirmation(); report = s.report()
    assert result['case_count'] == result['offline_replay_count'] == result['fresh_live_replication_count'] == 0
    assert report['independent_sites'] == 0
    assert not any(key[0] == 'confirmation' for key, _, _ in setup.calls)


@pytest.mark.parametrize('field', ['ready_slot', 'charged_compute_seconds', 'charged_seconds', 'kind'])
def test_offline_schedule_changes_cannot_earn_exactness(setup, field):
    s.development(); s.confirmation()
    key = s.replication_keys([4.])[0]
    main, replay = r.load_case(key), r.load_case(key, replay=True)
    replay = deepcopy(replay)
    replay['jobs'][0][field] = 'refit' if field == 'kind' else replay['jobs'][0][field] + 1
    comparison = s._offline_comparisons([main], [replay], [key])[0]
    assert not comparison['exact'] and not comparison['checks']['original_schedule_exact']


@pytest.mark.parametrize('mutation', ['matches_flag', 'actual_hash', 'offline_metadata', 'consumption', 'binding', 'native_result',
                                     'local_capture', 'local_payload', 'local_completion'])
def test_replay_booleans_cannot_replace_actual_equality(setup, mutation):
    s.development(); s.confirmation()
    key = s.replication_keys([4.])[0]
    main, replay = r.load_case(key), deepcopy(r.load_case(key, replay=True))
    if mutation == 'matches_flag': replay['jobs'][0]['matches_recorded_forecast'] = False
    elif mutation == 'actual_hash': replay['jobs'][0]['forecast_sha256'] = 'f' * 64
    elif mutation == 'offline_metadata': replay['jobs'][0]['completion_uses_actual_elapsed'] = True
    elif mutation == 'consumption': replay['forecast_consumption'][0]['slot'] -= 1
    elif mutation == 'binding': replay['recorded_schedule_sha256'] = '0' * 64
    elif mutation == 'native_result': replay['result']['control_misses'] += 1
    elif mutation == 'local_capture': replay['local_scan_observability']['initial_capture_slots'][0][0] -= 100
    elif mutation == 'local_payload': replay['local_scan_observability']['completion_events'][0]['heard_rssi_dbm'][0] += 1
    else: replay['local_scan_observability']['completion_events'][0]['processed_slot'] += 1
    assert not s._offline_comparisons([main], [replay], [key])[0]['exact']


def test_fresh_recomputed_time_is_diagnostic_during_offline_replay(setup):
    s.development(); s.confirmation()
    key = s.replication_keys([4.])[0]
    main, replay = r.load_case(key), deepcopy(r.load_case(key, replay=True))
    job = replay['jobs'][0]
    job.update(fresh_actual_prediction_seconds=1., fresh_actual_issue_seconds=2.,
        fresh_actual_compute_seconds=2., fresh_runtime_overrun=True,
        exceeds_original_recorded_compute=True, fresh_exceeds_original_actual_compute=True)
    comparison = s._offline_comparisons([main], [replay], [key])[0]
    assert comparison['exact']
    assert not comparison['fresh_measured_deadline_claim']
    assert comparison['fresh_compute_is_diagnostic_only']
    assert comparison['fresh_jobs_exceeding_recorded_charge'] == 1


@pytest.mark.parametrize('flag', ['replay', 'live_replication', 'accounting_mode'])
def test_replication_namespace_is_not_interchangeable(setup, flag):
    s.development(); s.confirmation()
    key = s.replication_keys([4.])[0]
    main, replay = r.load_case(key), deepcopy(r.load_case(key, replay=True))
    replay[flag] = False if flag == 'replay' else True if flag == 'live_replication' else 'actual_cost'
    with pytest.raises(RuntimeError, match='namespace'):
        s._offline_comparisons([main], [replay], [key])


@pytest.mark.parametrize('mutation', ['undercharge', 'premature_ready', 'premature_consumption', 'missing_consumption'])
def test_fresh_cost_safety_rechecked_from_jobs(setup, mutation):
    s.development(); s.confirmation()
    key = s.replication_keys([4.])[0]
    row = deepcopy(r.load_case(key, live_replication=True))
    if mutation == 'undercharge': row['jobs'][0]['charged_compute_seconds'] = .001
    elif mutation == 'premature_ready': row['jobs'][0]['ready_slot'] -= 1
    elif mutation == 'missing_consumption': row['forecast_consumption'].pop()
    else: row['forecast_consumption'][0]['slot'] -= 1
    with pytest.raises(RuntimeError):
        s._fresh([row], [key], live_replication=True)


@pytest.mark.parametrize('field', ['all_offline_exact', 'all_fresh_paid_cost_checks_pass', 'case_count', 'offline_comparisons'])
def test_report_recomputes_completion_instead_of_trusting_flags(setup, field):
    s.development(); s.confirmation()
    path = c.OUT / 'completion.json'; value = c.read_json(path)
    value[field] = [] if field == 'offline_comparisons' else 0 if field == 'case_count' else False
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='Completion flags'):
        s.report()


def test_missing_replication_cannot_be_called_complete(setup):
    s.development(); s.confirmation()
    key = s.replication_keys([4.])[0]
    (c.OUT / 'live_replications' / (r.case_name(key) + '.receipt.json')).unlink()
    with pytest.raises(RuntimeError, match='Partial'):
        s.report()


def test_immutable_report_reproduction_creates_no_worlds(setup):
    s.development(); s.confirmation(); first = s.report()
    count = len(setup.calls)
    assert s.report() == first
    assert len(setup.calls) == count
    assert c.read_json(c.OUT / 'report.json') == first


def test_registered_scientific_matrix_has_only_new_slow_sites():
    p = c.protocol()
    assert p['handover_seconds'] == [4.]
    assert p['development_sites'] == [2153000, 2153001]
    assert p['confirmation_sites'] == list(range(2154000, 2154008))
    assert len(r.case_keys('development')) == 16
    assert len(r.case_keys('confirmation')) == 64
    assert not set(p['development_sites'] + p['confirmation_sites']) & set(
        c.parent.protocol()['development_sites'] + c.parent.protocol()['confirmation_sites'])


@pytest.mark.parametrize('mutation', ['job_ordinal', 'source_timestamp', 'history_cutoff', 'arrivals_missing', 'accepted_list', 'skipped_fifo'])
def test_source_timestamp_and_request_membership_cannot_be_forged(setup, mutation):
    key = s.ordered_keys('development')[0]
    row = packet_row(key, setup.p)
    if mutation == 'job_ordinal': row['jobs'][1]['job_index'] += 1
    elif mutation == 'source_timestamp': row['jobs'][1]['source_decision'] += 1
    elif mutation == 'history_cutoff': row['jobs'][1]['history_cutoff_exclusive'] += 1
    elif mutation == 'arrivals_missing': row['arrivals'].pop()
    elif mutation == 'accepted_list': row['accepted_decisions'].pop()
    else:
        row['controller_variant'] = 'fifo'; row['queue_mode'] = 'fifo'
        row['arrivals'][1]['accepted'] = False
    assert not c.causal_history(row)
    with pytest.raises(RuntimeError, match='Causality'):
        s._causality(row)


def test_sparse_sources_keep_real_timestamps_and_consumption_age(setup, monkeypatch):
    key = ('development', setup.p['development_sites'][0], 4., 'frozen_fleet_jepa', 'base')
    monkeypatch.setattr(c, 'latency', lambda: {'profiles': {
        family: dict.fromkeys(('cold', 'warm', 'refit'), .6) for family in setup.p['families']}})
    row = packet_row(key, setup.p, profile=.6)
    assert row['accepted_decisions'] == [0, 2]
    assert [j['job_index'] for j in row['jobs']] == [0, 1]
    assert row['jobs'][1]['history_cutoff_exclusive'] == 62
    assert row['arrivals'][1]['job_index'] is None
    assert c.causal_history(row) and c.verify_paid_row(row)
    row['jobs'][1]['decision'] = 1
    assert not c.causal_history(row)


def test_fresh_repetition_can_change_membership_without_truncated_ordinal_pairing(setup, monkeypatch):
    key = ('confirmation', setup.p['confirmation_sites'][0], 4., 'frozen_fleet_jepa', 'base')
    monkeypatch.setattr(c, 'latency', lambda: {'profiles': {
        family: dict.fromkeys(('cold', 'warm', 'refit'), .49) for family in setup.p['families']}})
    original = packet_row(key, setup.p, profile=.49)
    fresh = packet_row(key, setup.p, live=True, profile=.49, actual=.51)
    assert original['accepted_decisions'] == [0, 1, 2]
    assert fresh['accepted_decisions'] == [0, 2]
    comparison = s._fresh_comparisons([original], [fresh], [key])[0]
    assert comparison['matched_source_decisions'] == [0, 2]
    assert comparison['removed_source_decisions'] == [1]
    assert comparison['added_source_decisions'] == []
    assert comparison['original_accepted_jobs'] == 3 and comparison['fresh_accepted_jobs'] == 2
    assert comparison['paid_cost_and_consumption_pass']
    assert not comparison['bitwise_equality_required']


@pytest.mark.parametrize('mutation', ['arrival_flag', 'source_job', 'arrival_digest', 'combined_digest'])
def test_offline_arrival_and_sparse_schedule_are_both_bound(setup, mutation):
    s.development(); s.confirmation()
    key = s.replication_keys([4.])[0]
    original, replay = r.load_case(key), deepcopy(r.load_case(key, replay=True))
    if mutation == 'arrival_flag': replay['arrivals'][1]['accepted'] = False
    elif mutation == 'source_job': replay['jobs'][1]['source_decision'] += 1
    elif mutation == 'arrival_digest': replay['recorded_arrivals_sha256'] = '0' * 64
    else: replay['recorded_schedule_sha256'] = c.value_digest(original['jobs'])
    comparison = s._offline_comparisons([original], [replay], [key])[0]
    assert not comparison['exact']



def test_native_and_matched_groups_have_strict_separate_order(setup):
    s.development(); setup.phase_trace.clear()
    result = s.confirmation()
    native = [row for row in setup.phase_trace if row[0] == 'native']
    assert len(native) == 16 + 8 + 8
    assert all(row[0] == 'native' for row in setup.phase_trace[:32])
    assert all(row[0] == 'matched' for row in setup.phase_trace[32:])
    assert result['matched_sequence_count'] == 4
    assert result['matched_offline_replay_count'] == 2
    assert result['all_matched_offline_exact']
    assert all(not row['fresh_measured_deadline_claim'] for row in result['matched_offline_comparisons'])


def test_matched_confirmation_permission_precedes_pool_and_cache(setup, monkeypatch):
    def rejected(cost): raise RuntimeError('Raw matched permission rejected')
    monkeypatch.setattr(c, 'check_confirmation_permission', rejected)
    monkeypatch.setattr(s, 'ProcessPoolExecutor', lambda **k: pytest.fail('Opened before permission'))
    monkeypatch.setattr(m, 'run_case', lambda *a, **k: pytest.fail('Cache/model opened before permission'))
    with pytest.raises(RuntimeError, match='permission rejected'):
        s.execute_matched(s.ordered_matched_keys('confirmation'))
    assert not setup.matched_calls


@pytest.mark.parametrize('mutation', ['drop', 'duplicate', 'reorder'])
def test_matched_incomplete_or_reordered_matrix_fails_before_execution(setup, mutation):
    keys = s.ordered_matched_keys('development')
    if mutation == 'drop': keys.pop()
    elif mutation == 'duplicate': keys[-1] = keys[0]
    else: keys[0], keys[1] = keys[1], keys[0]
    with pytest.raises(RuntimeError, match='complete registered'):
        s.execute_matched(keys)
    assert not setup.matched_calls


@pytest.mark.parametrize('field', ['unit_source', 'unit_query', 'metrics', 'prediction_integrity_pass'])
def test_raw_matched_tamper_revokes_development_permission(setup, field):
    s.development(); setup.calls.clear(); setup.matched_calls.clear()
    key = s.ordered_matched_keys('development')[0]
    archive, receipt = m.archive_paths(key)
    row = c.read_json(archive)
    if field == 'metrics': row[field]['short_route_delta_rmse_db'] += 1
    elif field == 'unit_query': row[field] = [0, 2, 3]
    elif field == 'prediction_integrity_pass': row[field] = False
    else: row[field] = 'changed'
    archive.write_text(json.dumps(row)); receipt.write_text(json.dumps({'archive_sha256': c.digest(archive)}))
    with pytest.raises(RuntimeError, match='Matched|integrity'):
        s.confirmation()
    assert not setup.calls and not setup.matched_calls


@pytest.mark.parametrize('field', ['matched_sequence_count', 'matched_offline_replay_count', 'all_matched_offline_exact', 'matched_offline_comparisons'])
def test_report_recomputes_matched_completion_fields(setup, field):
    s.development(); s.confirmation()
    path = c.OUT / 'completion.json'; value = c.read_json(path)
    value[field] = [] if field == 'matched_offline_comparisons' else False if field.startswith('all_') else 0
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='Completion flags'):
        s.report()


def test_veto_opens_no_matched_holdouts_and_retains_four_primary_slots(setup):
    setup.severe_costs = {4.}
    s.development(); result = s.confirmation(); value = s.report()
    assert result['matched_sequence_count'] == result['matched_offline_replay_count'] == 0
    assert not any(key[0] == 'confirmation' for key, replay in setup.matched_calls)
    assert len(value['primary']) == 4
    assert all(row['upper_quantile'] == .9875 for row in value['primary'])


def test_all_variants_are_latest_and_feature_flags_are_bound(setup):
    for key in s.ordered_keys('development'):
        row = packet_row(key, setup.p)
        assert c.causal_history(row)
        row['pilot_enabled'] = not row['pilot_enabled']
        assert not c.causal_history(row)
    row = packet_row(s.ordered_keys('development')[0], setup.p)
    row['queue_mode'] = 'fifo'
    assert not c.causal_history(row)

@pytest.fixture
def source_layout(tmp_path, monkeypatch):
    """Tiny file-byte closure; no scientific data or held-out model is opened."""
    p = json.loads(c.PROTOCOL.read_text())
    root, out = tmp_path / 'root', tmp_path / 'root/artifacts/pilot_fastloop'
    out.mkdir(parents=True)
    for name, text in {
        'src/wirelessjev/pilot_fastloop/common.py': 'unit new source\n',
        'tests/test_pilot_fastloop_fixture.py': 'unit source test\n',
        'parent_frozen.py': 'frozen parent\n',
        'prior_evidence.json': '{"completed":true}\n',
        'README.md': 'Current narrative may change.\n',
    }.items():
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
    (out / 'protocol.json').write_text(json.dumps(p))
    monkeypatch.setattr(c, 'ROOT', root); monkeypatch.setattr(c, 'OUT', out)
    monkeypatch.setattr(c, 'PROTOCOL', out / 'protocol.json')
    for path in c.prior_inputs():
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{}\n')
    historic_doc = b'Original published narrative.\n'
    manifest = {'sources_sha256': {'parent_frozen.py': c.digest(root / 'parent_frozen.py')},
        'evidence_sha256': {'prior_evidence.json': c.digest(root / 'prior_evidence.json')},
        'documents_sha256': {'README.md': hashlib.sha256(historic_doc).hexdigest()}}
    manifest_name = 'artifacts/freshness_stack/verification/manifest.json'
    raw_manifest = json.dumps(manifest).encode(); (root / manifest_name).write_bytes(raw_manifest)
    historical = {manifest_name: raw_manifest, 'README.md': historic_doc}
    monkeypatch.setattr(c, 'historical_bytes', lambda name: historical[name])
    folder = root / 'artifacts/jepa_advantage_diagnosis'
    receipt = {'base_commit': c.BASE_COMMIT, 'passive_only': True,
        'no_worlds_forwards_fits_or_timing': True, 'source_and_capsules_verified_before_and_after': True,
        'script_sha256': c.digest(root / 'scripts/audit_jepa_advantages.py'),
        'inputs_sha256': {'prior_evidence.json': c.digest(root / 'prior_evidence.json')},
        'outputs_sha256': {'artifacts/jepa_advantage_diagnosis/diagnosis.json': c.digest(folder / 'diagnosis.json')}}
    (folder / 'receipt.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(c.parent, 'source_hashes', lambda: {'parent_frozen.py': c.digest(root / 'parent_frozen.py')})
    monkeypatch.setattr(c.parent, 'check_sources', lambda: {'unit_fixture': True})
    monkeypatch.setattr(c.parent, 'verify_previous_evidence', lambda: {'older': {'sources_sha256': 1}})
    monkeypatch.setattr(c.parent, 'verify_diagnosis_inputs', lambda: {'older': {'inputs_sha256': 1}})
    monkeypatch.setattr(c, 'checkpoint_hashes', lambda: {'unit-checkpoint': 'fixed'})
    monkeypatch.setattr(c, 'versions', lambda: {'fixture': '1'})
    c.write_once(out / 'execution_order.json', c.execution_order())
    return SimpleNamespace(root=root, out=out, p=p, historical=historical)


def test_source_closure_inherits_old_stack_and_binds_advantage_diagnosis(source_layout):
    value = c.source_hashes()
    assert 'parent_frozen.py' in value
    assert 'tests/test_pilot_fastloop_fixture.py' in value
    for path in c.prior_inputs(): assert str(path.relative_to(c.ROOT)) in value
    assert len(c.execution_order()['matched_development']) == 4
    assert len(c.execution_order()['matched_confirmation']) == 16


def test_registration_precedes_worlds_models_and_caches(source_layout):
    value = c.register()
    assert value['new_evaluation_data_accessed'] is False
    assert value['new_fits'] == 0 and value['parent_results_already_observed'] is True
    assert value['previous_global_document_hashes_apply_to_original_commits'] is True
    assert value['observed_input_sha256'] == {str(path.relative_to(c.ROOT)): c.digest(path) for path in c.prior_inputs()}
    assert c.check_sources() == value
    assert not (c.OUT / 'matched').exists()


@pytest.mark.parametrize('folder', ['cases', 'replays', 'live_replications', 'matched'])
def test_registration_refuses_any_preexisting_native_or_matched_namespace(source_layout, folder):
    (c.OUT / folder).mkdir()
    with pytest.raises(RuntimeError, match='precede every'):
        c.register()


@pytest.mark.parametrize('mutation', ['capsule', 'new_source', 'new_test_membership', 'prior_report', 'prior_evidence', 'prior_source', 'manifest', 'diagnosis', 'parent_guard'])
def test_source_or_frozen_evidence_mutation_fails_closed(source_layout, monkeypatch, mutation):
    c.register()
    if mutation == 'capsule': path = c.OUT / 'registration/source/src/wirelessjev/pilot_fastloop/common.py'
    elif mutation == 'new_source': path = c.ROOT / 'src/wirelessjev/pilot_fastloop/common.py'
    elif mutation == 'new_test_membership': path = c.ROOT / 'tests/test_pilot_fastloop_added.py'
    elif mutation == 'prior_report': path = c.ROOT / 'artifacts/freshness_stack/report.json'
    elif mutation == 'prior_evidence': path = c.ROOT / 'prior_evidence.json'
    elif mutation == 'prior_source': path = c.ROOT / 'parent_frozen.py'
    elif mutation == 'manifest': path = c.ROOT / 'artifacts/freshness_stack/verification/manifest.json'
    elif mutation == 'diagnosis': path = c.ROOT / 'artifacts/jepa_advantage_diagnosis/diagnosis.json'
    else:
        def rejected(): raise RuntimeError('Frozen parent identity changed')
        monkeypatch.setattr(c.parent, 'check_sources', rejected); path = None
    if path is not None: path.write_text('changed bytes\n')
    with pytest.raises(RuntimeError): c.check_sources()


def test_historical_docs_allow_current_narrative_change(source_layout):
    value = c.register()
    (c.ROOT / 'README.md').write_text('New claims and honest limits.\n')
    assert c.check_sources() == value
    source_layout.historical['README.md'] = b'changed history'
    with pytest.raises(RuntimeError, match='Historical published document'):
        c.check_sources()


def test_site_overlap_guard_includes_arbitrary_previous_protocol(source_layout):
    path = c.ROOT / 'artifacts/older/protocol.json'; path.parent.mkdir()
    path.write_text(json.dumps({'hidden': {'site': source_layout.p['confirmation_sites'][0]}}))
    with pytest.raises(RuntimeError, match='overlap'): c.verify_fresh_sites()


def test_historical_manifest_cannot_traverse_outside_repository(source_layout):
    name = 'artifacts/freshness_stack/verification/manifest.json'
    value = json.loads(source_layout.historical[name])
    value['sources_sha256'] = {'../outside.py': '0' * 64}
    raw = json.dumps(value).encode(); source_layout.historical[name] = raw
    (c.ROOT / name).write_bytes(raw)
    with pytest.raises(RuntimeError, match='Previously completed'):
        c.verify_previous_evidence()


@pytest.mark.parametrize('field', ['base_commit', 'passive_only', 'no_worlds_forwards_fits_or_timing', 'source_and_capsules_verified_before_and_after'])
def test_passive_advantage_receipt_requires_historical_base_and_literal_scope(source_layout, field):
    path = c.ROOT / 'artifacts/jepa_advantage_diagnosis/receipt.json'
    value = c.read_json(path); value[field] = 'other' if field == 'base_commit' else 1
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='historical base or passive scope'):
        c.verify_diagnosis_inputs()


def test_common_rechecks_joined_full_scan_and_pilot_ledger(setup, monkeypatch):
    row = packet_row(s.ordered_keys('development')[0], setup.p)
    calls = []
    def reject_join(telemetry, pilots, **dimensions):
        calls.append((telemetry, pilots, dimensions))
        return False
    monkeypatch.setattr(r, '_causal_telemetry', reject_join)
    assert not c.causal_history(row)
    assert calls == [(row['telemetry'], row['pilot_reporting'],
        {'robots': setup.p['robots'], 'duration': setup.p['duration_seconds'], 'ap_count': 2})]
