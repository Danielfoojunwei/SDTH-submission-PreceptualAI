"""Slow-only orchestration, recorded replay and actual-cost permission boundaries.

All rows below are tiny explicit arithmetic fixtures, not research-world runs
or measured performance evidence. Native runtime execution is tested separately.
"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from wirelessjev.costaware_recovery import common as c, rollout as r, study as s


def packet_row(key, p, *, replay=False, live=False, severe=False):
    stage, site, cost, family, variant = key
    n, duration = p['robots'], p['duration_seconds']
    miss = 2 if variant == 'legacy' else 1
    if severe and variant == 'repair': miss = 10
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
    jobs, consumed, previous = [], [], 0.
    actual = .02 if live else .002
    for index in range(duration):
        charge = max(.01, actual)
        previous = max(float(index), previous) + charge + .5
        ready = __import__('math').ceil(previous * 100 - 1e-10)
        jobs.append({'decision': index, 'history_cutoff_exclusive': 60 + index,
            'kind': 'cold' if index == 0 else 'warm', 'actual_prediction_seconds': actual / 2,
            'actual_issue_seconds': actual, 'actual_compute_seconds': actual,
            'profile_seconds': .01, 'charged_compute_seconds': charge,
            'shared_extra_seconds': .5, 'charged_seconds': charge + .5, 'ready_slot': ready,
            'runtime_overrun': actual > .01, 'predictor_runtime_overrun': actual / 2 > .01,
            'forecast_sha256': f'{index + 1:064x}', 'forecast_std_sha256': f'{index + 21:064x}',
            'accounting_mode': 'actual_cost', 'offline_recorded_schedule': False,
            'completion_uses_actual_elapsed': True})
        if ready < duration * 100:
            consumed.append({'slot': ready, 'forecast_decision': index,
                'ready_slot': ready, 'age_seconds': ready / 100 - index})
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
        'runtime_overrun': actual > .01, 'predictor_runtime_overrun': actual / 2 > .01,
        'jobs': jobs, 'forecast_consumption': consumed,
        'forecast_mean_sha256': 'a' * 64, 'forecast_std_sha256': 'b' * 64,
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
                recorded_schedule_sha256=c.value_digest(original['jobs']),
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
    return state


def test_capture_order_and_complete_site_blocks(setup):
    keys = s.ordered_keys('development')
    assert keys == [tuple(k) for k in c.execution_order()['development']]
    assert len(keys) == 8
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
    assert result['case_count'] == 8
    assert result['offline_replay_count'] == result['fresh_live_replication_count'] == 4
    assert result['all_offline_exact'] and result['all_fresh_paid_cost_checks_pass']
    assert report['case_count'] == 8
    assert report['proof']['experimental_claim_integrity_pass']
    assert report['offline_replay_summary']['independent_sites_added'] == 0
    assert report['fresh_live_cost_summary']['independent_sites_added'] == 0
    assert all(row['sensitivity_only'] and not row['bitwise_equality_required'] for row in result['fresh_live_cost_comparisons'])
    assert any(row['changed_ready_jobs'] for row in result['fresh_live_cost_comparisons'])
    assert report['timing']['profile_exceeded_cases'] == 0
    assert report['fresh_live_cost_summary']['timing']['profile_exceeded_cases'] == 4




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
    assert p['development_sites'] == [2133000, 2133001]
    assert p['confirmation_sites'] == list(range(2134000, 2134008))
    assert len(r.case_keys('development')) == 8
    assert len(r.case_keys('confirmation')) == 32
    assert not set(p['development_sites'] + p['confirmation_sites']) & set(
        c.parent.protocol()['development_sites'] + c.parent.protocol()['confirmation_sites'])


@pytest.fixture
def source_layout(tmp_path, monkeypatch):
    # A separate miniature filesystem tests pinning/capsule behavior only.
    p = json.loads(c.PROTOCOL.read_text())
    root = tmp_path / 'root'
    out = root / 'artifacts/costaware_recovery'
    old = root / 'artifacts/jepa_repair'
    out.mkdir(parents=True); (old / 'cases').mkdir(parents=True)
    (root / 'src/wirelessjev/costaware_recovery').mkdir(parents=True)
    (root / 'src/wirelessjev/costaware_recovery/common.py').write_text('unit source\n')
    (root / 'tests').mkdir()
    (root / 'tests/test_costaware_recovery_fixture.py').write_text('unit tests\n')
    (root / 'parent_frozen.py').write_text('unit frozen parent\n')
    (out / 'protocol.json').write_text(json.dumps(p))
    (old / 'development.json').write_text('{"observed":true}\n')
    (old / 'qualification.json').write_text('{"observed_veto":true}\n')
    for key in c.parent_rollout.case_keys('development'):
        (old / 'cases' / (c.parent_rollout.case_name(key) + '.receipt.json')).write_text('{}\n')
    monkeypatch.setattr(c, 'ROOT', root)
    monkeypatch.setattr(c, 'OUT', out)
    monkeypatch.setattr(c, 'PROTOCOL', out / 'protocol.json')
    monkeypatch.setattr(c.parent, 'OUT', old)
    monkeypatch.setattr(c.parent, 'source_hashes', lambda: {'parent_frozen.py': c.digest(root / 'parent_frozen.py')})
    monkeypatch.setattr(c.parent, 'check_sources', lambda: {'unit_fixture': True})
    monkeypatch.setattr(c.parent, 'verified_development', lambda: {'passed_costs': [.03]})
    monkeypatch.setattr(c, 'checkpoint_hashes', lambda: {'unit-checkpoint': 'fixed'})
    monkeypatch.setattr(c, 'versions', lambda: {'fixture': '1'})
    c.write_once(out / 'execution_order.json', c.execution_order())
    return SimpleNamespace(root=root, out=out, old=old, p=p)


def test_source_closure_pins_twenty_parent_dev_receipts_not_live_confirmation(source_layout):
    a = c.source_hashes()
    prior = [name for name in a if name.startswith('artifacts/jepa_repair/cases/')]
    assert len(prior) == 20
    assert all('/development-' in name for name in prior)
    assert 'artifacts/jepa_repair/development.json' in a
    assert 'artifacts/jepa_repair/qualification.json' in a
    assert 'tests/test_costaware_recovery_fixture.py' in a
    assert not any(name.startswith('tests/test_jepa_repair') for name in a)
    # Parent confirmation keeps running; these files must not move this closure.
    path = source_layout.old / 'cases/confirmation-in-progress.receipt.json'
    path.write_text('{"still_running":true}\n')
    assert c.source_hashes() == a
    path.write_text('{"now_complete":true}\n')
    assert c.source_hashes() == a
    (source_layout.old / 'development.json').write_text('{"observed":"changed"}\n')
    assert c.source_hashes() != a


@pytest.mark.parametrize('mutation', ['missing', 'extra'])
def test_incomplete_or_extra_parent_development_receipts_refused(source_layout, mutation):
    if mutation == 'missing': next((source_layout.old / 'cases').glob('development-*.receipt.json')).unlink()
    else: (source_layout.old / 'cases/development-extra.receipt.json').write_text('{}')
    with pytest.raises(RuntimeError, match='twenty completed'):
        c.source_hashes()


def test_registration_explicitly_records_prior_observed_choice_and_no_training(source_layout):
    value = c.register()
    assert value['parent_development_already_observed'] is True
    assert value['source_training_and_results_already_observed'] is True
    assert value['new_evaluation_data_accessed'] is False and value['new_fits'] == 0
    assert value['parent_development_sha256'] == c.digest(source_layout.old / 'development.json')
    assert 'parent-contract change' in value['selection_scope']
    assert c.check_sources() == value
    assert not (source_layout.out / 'cases').exists()


@pytest.mark.parametrize('mutation', ['capsule', 'new_source', 'parent_development', 'parent_guard'])
def test_registered_source_changes_fail_closed(source_layout, monkeypatch, mutation):
    c.register()
    if mutation == 'capsule':
        path = source_layout.out / 'registration/source/src/wirelessjev/costaware_recovery/common.py'
        path.write_text('changed capsule')
    elif mutation == 'new_source':
        (source_layout.root / 'src/wirelessjev/costaware_recovery/common.py').write_text('changed source')
    elif mutation == 'parent_development':
        (source_layout.old / 'development.json').write_text('changed outcome')
    else:
        def rejected(): raise RuntimeError('Frozen parent identity changed')
        monkeypatch.setattr(c.parent, 'check_sources', rejected)
    with pytest.raises(RuntimeError): c.check_sources()


def test_registration_cannot_follow_any_new_case_namespace(source_layout):
    (source_layout.out / 'live_replications').mkdir()
    with pytest.raises(RuntimeError, match='precede every'):
        c.register()


def test_fresh_site_guard_rejects_overlap_with_parent_protocol(source_layout):
    (source_layout.old / 'protocol.json').write_text(json.dumps({
        'confirmation_sites': [source_layout.p['confirmation_sites'][0]]}))
    with pytest.raises(RuntimeError, match='overlap'):
        c.verify_fresh_sites()
