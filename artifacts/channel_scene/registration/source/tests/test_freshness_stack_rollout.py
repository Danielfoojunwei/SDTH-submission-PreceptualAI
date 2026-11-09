"""Real tiny native rollouts, replication boundaries and immutable archives."""
from copy import deepcopy
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import pytest
import torch
from wirelessjev.fleetwm.data import episode
from wirelessjev.frontier.study import mask_scans
from wirelessjev.freshness_stack import rollout as r
from wirelessjev.costaware_recovery.telemetry import TimedDeliveredScans
from wirelessjev.semantic_jepa.common import OUT as OLD_OUT
from wirelessjev.semantic_jepa.runtime import Predictor


@pytest.fixture
def protocol():
    # Software-fixture IDs, not the prospective scientific sites.
    return {'development_sites': [743, 744], 'confirmation_sites': [745, 746],
            'handover_seconds': [4.], 'robots': 2, 'duration_seconds': 3,
            'mission': 'bulk_uploads', 'scan_period_seconds': 5,
            'scan_outage_seconds': .09, 'extra_compute_seconds': .5}


def test_matrix_preserves_models_and_control_ablations(protocol):
    assert len(r.arms(4., protocol)) == 8
    with pytest.raises(ValueError):
        r.arms(.03, protocol)
    for stage in r.STAGES:
        keys = r.case_keys(stage, p=protocol)
        assert len(keys) == 16 and len({r.case_name(key) for key in keys}) == 16
        for key in keys:
            r.validate_case_key(key, protocol)
    assert len(r.case_keys('confirmation', [4.], protocol)) == 16
    assert r.case_name(('development', 743, 4., 'radiomap', 'latest_single_sense')) == 'development-743-h4000-radiomap-latest_single_sense'


@pytest.mark.parametrize('key', [
    ('confirmation', 2134000, 4., 'radiomap', 'latest_single_sense'),
    ('confirmation', 745, .06, 'radiomap', 'latest_single_sense'),
    ('confirmation', 745, .03, 'radiomap', 'latest_single_sense'),
    ('confirmation', 745, float('nan'), 'radiomap', 'latest_single_sense'),
    ('confirmation', 745, 4., '../escape', 'latest_single_sense'),
    ('confirmation', True, 4., 'radiomap', 'latest_single_sense'),
    ('../escape', 745, 4., 'radiomap', 'latest_single_sense'),
    ('development', 743, 4., 'radiomap', 'exact_only'),
    ('development', 743, 4., 'context_jepa', 'latest_single_sense'),
    ('development', 743, 4., 'radiomap', 'unknown'),
])
def test_keys_rejected_before_world(protocol, monkeypatch, key):
    monkeypatch.setattr(r, '_common', lambda: SimpleNamespace(protocol=lambda: protocol))
    monkeypatch.setattr(r, 'episode', lambda *a, **k: pytest.fail('World must stay unopened'))
    with pytest.raises(ValueError):
        r.run_case(key)


@pytest.mark.parametrize('replay,live', [(1, False), (False, 1), (True, True), ('yes', False)])
def test_replication_flags_exclusive_booleans(replay, live):
    with pytest.raises(ValueError, match='exclusive booleans'):
        r._mode(replay, live)


@pytest.mark.parametrize('key', [
    ('development', 743, 4., 'radiomap', 'latest_single_sense'),
    ('confirmation', 746, 4., 'radiomap', 'latest_single_sense'),
])
def test_replications_cannot_open_other_sites(protocol, monkeypatch, key):
    monkeypatch.setattr(r, '_common', lambda: SimpleNamespace(protocol=lambda: protocol))
    monkeypatch.setattr(r, 'episode', lambda *a, **k: pytest.fail('World must stay unopened'))
    for flags in ({'replay': True}, {'live_replication': True}):
        with pytest.raises(ValueError, match='first registered'):
            r.run_case(key, **flags)


def test_confirmation_permission_before_world(protocol, monkeypatch):
    def veto(cost):
        raise RuntimeError('Development veto remains active')
    c = SimpleNamespace(protocol=lambda: protocol, check_confirmation_permission=veto)
    monkeypatch.setattr(r, '_common', lambda: c)
    monkeypatch.setattr(r, 'episode', lambda *a, **k: pytest.fail('World must stay unopened'))
    with pytest.raises(RuntimeError, match='Development veto'):
        r.run_case(('confirmation', 745, 4., 'radiomap', 'latest_single_sense'))


@pytest.fixture(scope='module')
def actual_case():
    torch.set_num_threads(1)
    p = {'robots': 2, 'duration_seconds': 3, 'mission': 'bulk_uploads',
         'scan_period_seconds': 5, 'scan_outage_seconds': .09,
         'extra_compute_seconds': .5}
    profile = json.loads((OLD_OUT / 'latency.json').read_text())['profiles']['radiomap']
    row = r._evaluate(('confirmation', 745, 4., 'radiomap', 'latest_single_sense'), p, profile)
    row.update(replay=False, live_replication=False)
    return row, p, profile


def test_actual_native_packets_and_paid_ledger(actual_case):
    row, p, profile = actual_case
    assert row['stage'] == 'confirmation' and row['controller_variant'] == 'latest_single_sense'
    assert row['family'] == 'radiomap' and row['training_seed'] is None and row['new_fits'] == 0
    assert row['accounting_mode'] == 'actual_cost'
    assert row['causality_pass'] and row['paid_cost_and_consumption_pass']
    assert row['timing_safety_pass'] and not row['hardware_qualified'] and not row['production_qualified']
    assert len(row['jobs']) == 2 and row['accepted_decisions'] == [0, 2]
    assert row['result']['control_sent'] == sum(row['result']['packet_detail']['sent']) == 120
    assert row['result']['control_misses'] == sum(row['result']['packet_detail']['misses'])
    assert row['result']['packet_detail']['pending_control_right_censored'] is True
    assert [entry['hook_role'] for entry in row['admission_trace']] == ['planned', 'reactive'] * len(row['decision_trace'])
    assert r.timing_audit(row['jobs'], row['forecast_consumption'], profile, .5, 3,
                              arrivals=row['arrivals'], mode=row['queue_mode'])['passes']
    for event in row['telemetry']['events']:
        assert event['measurement_slot'] < event['complete_slot'] <= event['delivery_slot']
        assert event['first_usable_decision'] == event['delivery_slot'] // 100 + 1
    assert all(np.isfinite(value) for value in row['forecast_metrics'].values())


def test_actual_recorded_replay_exact_and_explicitly_offline(actual_case):
    main, p, profile = actual_case
    replay = r._evaluate(tuple(main['case']), p, profile, recorded_jobs=main['jobs'], recorded_arrivals=main['arrivals'])
    assert replay['accounting_mode'] == 'recorded_schedule_offline'
    assert replay['paid_cost_and_consumption_pass']
    assert replay['timing_audit']['real_time_contract_checked'] is False
    assert all(job['offline_recorded_schedule'] and 'fresh_actual_compute_seconds' in job
               for job in replay['jobs'])
    assert 'local_scan_observability' in r.REPLAY_FIELDS
    for field in r.REPLAY_FIELDS:
        assert replay[field] == main[field]
    assert replay['forecast_consumption'] == main['forecast_consumption']


def test_archived_local_scans_preserve_past_snapshots_capture_times_and_payloads(actual_case):
    _, p, profile = actual_case
    # A one-second cadence ensures this tiny real-PHY fixture actually
    # completes local scans; it is not a scientific study site or result.
    p = dict(p, scan_period_seconds=1)
    key = ('confirmation', 745, 4., 'radiomap', 'latest_single_sense')
    row = r._evaluate(key, p, profile)
    proof = row['local_scan_observability']
    _, _, _, observations = episode(745, 2, 3, 'drift')
    warm = mask_scans(observations, 1).before(60)
    initial = TimedDeliveredScans(warm, 3, 745)
    np.testing.assert_array_equal(proof['initial_capture_slots'], initial.initial_local_capture_slots)
    np.testing.assert_array_equal(proof['initial_heard'], initial.initial_local_capture_heard)
    np.testing.assert_array_equal(proof['initial_rssi_dbm'], initial.local)
    events = proof['completion_events']
    assert events
    central = {(event['robot'], event['measurement_slot'], event['complete_slot']): event
               for event in row['telemetry']['events']}
    links = initial.links
    matched_central = 0
    for event in events:
        assert 0 <= event['measurement_slot'] < event['complete_slot'] <= event['processed_slot'] < 300
        assert event['heard_links'] == sorted(set(event['heard_links']))
        assert len(event['heard_links']) == len(event['heard_rssi_dbm']) == event['heard_entries']
        heard = np.zeros(links, dtype=bool)
        heard[event['heard_links']] = True
        values = np.zeros(links, dtype='f4')
        values[heard] = event['heard_rssi_dbm']
        assert event['payload_sha256'] == hashlib.sha256(values.tobytes() + heard.tobytes()).hexdigest()
        identity = (event['robot'], event['measurement_slot'], event['complete_slot'])
        if identity in central:
            assert central[identity]['payload_sha256'] == event['payload_sha256']
            matched_central += 1
    assert matched_central


def test_actual_live_replication_pays_new_measurements(actual_case):
    main, p, profile = actual_case
    live = r._evaluate(tuple(main['case']), p, profile)
    assert live['accounting_mode'] == 'actual_cost'
    assert live['timing_audit']['real_time_contract_checked'] is True
    assert live['paid_cost_and_consumption_pass']
    assert all(job['completion_uses_actual_elapsed'] and not job['offline_recorded_schedule']
               for job in live['jobs'])


@pytest.mark.parametrize('cost,variant', [(4., variant) for variant in r.VARIANTS])
def test_real_control_wiring_and_roles(protocol, cost, variant):
    torch.set_num_threads(1)
    profile = json.loads((OLD_OUT / 'latency.json').read_text())['profiles']['radiomap']
    row = r._evaluate(('development', 744, cost, 'radiomap', variant), protocol, profile)
    assert row['causality_pass'] and row['paid_cost_and_consumption_pass']
    assert row['result']['control_sent'] == 120
    roles = [entry['hook_role'] for entry in row['admission_trace']]
    expected = ['planned', 'reactive']
    assert roles == expected * len(row['decision_trace'])


def test_actual_frozen_jepa_uses_same_past_public_input(protocol):
    torch.set_num_threads(1)
    profile = json.loads((OLD_OUT / 'latency.json').read_text())['profiles']['frozen_fleet_jepa']
    row = r._evaluate(('development', 745, 4., 'frozen_fleet_jepa', 'latest_single_sense'), protocol, profile)
    full, _, geometry, observations = episode(745, 2, 3, 'drift')
    past = mask_scans(observations, 5).before(60)
    mean, _, _, _ = Predictor(geometry, 'frozen_fleet_jepa').step(past, full.plan[60])
    assert row['jobs'][0]['forecast_sha256'] == hashlib.sha256(mean.tobytes()).hexdigest()
    assert row['family'] == 'frozen_fleet_jepa' and row['training_seed'] is None
    assert row['causality_pass'] and row['paid_cost_and_consumption_pass']


@pytest.mark.parametrize('mutation', ['undercharge', 'early_ready', 'early_consumption', 'drop_consumption', 'mask_overrun'])
def test_paid_audit_rejects_forged_accounting(actual_case, mutation):
    row, _, profile = actual_case
    row = deepcopy(row)
    if mutation == 'undercharge':
        row['jobs'][0]['charged_compute_seconds'] = 0.
    elif mutation == 'early_ready':
        row['jobs'][0]['ready_slot'] -= 1
    elif mutation == 'early_consumption':
        row['forecast_consumption'][0]['slot'] -= 1
    elif mutation == 'drop_consumption':
        row['forecast_consumption'].pop()
    else:
        row['jobs'][0]['runtime_overrun'] = not row['jobs'][0]['runtime_overrun']
    assert not r.timing_audit(row['jobs'], row['forecast_consumption'], profile, .5, 3,
                              arrivals=row['arrivals'], mode=row['queue_mode'])['passes']


def test_archives_reject_partial_corrupt_or_wrong_mode(tmp_path, actual_case):
    row, _, _ = actual_case; key = tuple(row['case'])
    r._write_record(tmp_path, key, row)
    assert r._read_record(tmp_path, key) == row
    r._write_record(tmp_path, key, row)
    changed = deepcopy(row); changed['result']['control_misses'] += 1
    with pytest.raises(RuntimeError, match='differs'):
        r._write_record(tmp_path, key, changed)
    receipt = tmp_path / (r.case_name(key) + '.receipt.json')
    data = json.loads(receipt.read_text()); data['accounting_mode'] = 'recorded_schedule_offline'
    receipt.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match='receipt identity'):
        r._read_record(tmp_path, key)
    receipt.unlink()
    with pytest.raises(RuntimeError, match='Partial'):
        r._read_record(tmp_path, key)


def test_archive_hash_checked_before_decoding(tmp_path, actual_case):
    row, _, _ = actual_case; key = tuple(row['case'])
    r._write_record(tmp_path, key, row)
    path = tmp_path / (r.case_name(key) + '.json.gz')
    data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    with pytest.raises(RuntimeError, match='hash'):
        r._read_record(tmp_path, key)


@pytest.mark.parametrize('mutation', ['skip_work', 'source_ordinal', 'duplicate_job', 'wrong_previous_ready', 'lost_arrival'])
def test_sparse_audit_rejects_forged_arrival_or_source_identity(actual_case, mutation):
    row, _, profile = actual_case
    row = deepcopy(row)
    if mutation == 'skip_work':
        row['arrivals'][1]['accepted'] = True
    elif mutation == 'source_ordinal':
        row['jobs'][1]['source_decision'] = 1
    elif mutation == 'duplicate_job':
        row['jobs'].append(deepcopy(row['jobs'][-1]))
    elif mutation == 'wrong_previous_ready':
        row['arrivals'][1]['previous_ready_seconds'] = 0.
    else:
        row['arrivals'].pop()
    assert not r.timing_audit(row['jobs'], row['forecast_consumption'], profile, .5, 3,
        arrivals=row['arrivals'], mode=row['queue_mode'])['passes']


def test_fifo_native_rollout_matches_predecessor_controller_packets():
    from wirelessjev.costaware_recovery.rollout import _evaluate as old_evaluate
    torch.set_num_threads(1)
    p = dict(robots=2, duration_seconds=3, mission='bulk_uploads',
        scan_period_seconds=5, scan_outage_seconds=.09, extra_compute_seconds=.5)
    profile = json.loads((OLD_OUT / 'latency.json').read_text())['profiles']['radiomap']
    old = old_evaluate(('development', 744, 4., 'radiomap', 'repair'), p, profile)
    new = r._evaluate(('development', 744, 4., 'radiomap', 'fifo'), p, profile)
    a, b = deepcopy(old['result']), deepcopy(new['result'])
    b.pop('freshness_execution'); a.pop('policy'); b.pop('policy')
    assert a == b
    for field in ['telemetry', 'forecast_mean_sha256', 'forecast_std_sha256',
                  'forecast_consumption', 'decision_trace', 'admission_trace', 'local_scan_observability']:
        assert old[field] == new[field], field


def test_real_sparse_rf_diagnostics_hash_only_computed_source_queries(monkeypatch):
    captured = []
    class TracedPredictor(Predictor):
        def step(self, past, plan):
            mean, std, kind, elapsed = super().step(past, plan)
            captured.append((len(past.value) - 60, mean.copy(), std.copy()))
            return mean, std, kind, elapsed
    monkeypatch.setattr(r, 'Predictor', TracedPredictor)
    p = dict(robots=2, duration_seconds=3, mission='bulk_uploads',
        scan_period_seconds=5, scan_outage_seconds=.09, extra_compute_seconds=.5)
    profile = json.loads((OLD_OUT / 'latency.json').read_text())['profiles']['radiomap']
    row = r._evaluate(('development', 744, 4., 'radiomap', 'latest'), p, profile)
    assert row['accepted_decisions'] == row['forecast_metric_decisions'] == [x[0] for x in captured] == [0, 2]
    mean, std = np.stack([x[1] for x in captured]), np.stack([x[2] for x in captured])
    assert row['forecast_mean_sha256'] == hashlib.sha256(mean.tobytes()).hexdigest()
    assert row['forecast_std_sha256'] == hashlib.sha256(std.tobytes()).hexdigest()
    full, _, _, _ = episode(744, 2, 3, 'main')
    times = 60 + np.array(row['accepted_decisions'])[:, None] + np.array([0, 1, 2, 3, 4, 5, 10, 20, 30])[None]
    truth = full.snr[times * 5].transpose(0, 2, 1, 3, 4)
    assert row['forecast_metrics']['rmse_db'] == pytest.approx(float(np.sqrt(np.mean((mean - truth) ** 2))))
