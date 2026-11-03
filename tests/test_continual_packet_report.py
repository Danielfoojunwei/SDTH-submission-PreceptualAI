"""Arithmetic fixtures only: no site generation, model fitting or reserved data."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import gzip
import json
import math
import numpy as np
import pytest
from wirelessjev.continual_packet import report, study


@pytest.fixture
def registration():
    p = study.protocol()
    # Unit fixture identities are unrelated to the reserved scientific sites.
    p['confirmation_sites'] = list(range(10, 16))
    p['development_sites'] = [1, 2]
    return p


@pytest.fixture
def rows(registration):
    p = registration
    base = {'canonical': 400, 'reciprocal_radiomap': 350,
        'synchronized_ignored': 400, 'staggered_ignored': 400,
        'synchronized_memory': 250, 'staggered_memory': 100,
        'synchronized_bayesian': 300, 'staggered_bayesian': 150}
    output = []
    for _, site, packet, cost, arm in study.case_keys('confirmation', p):
        missed = base[arm] + site - 10 + packet * 5
        counts, sent = [missed] * 100, [1600] * 100
        trace = [{'slot_end': second * 100, 'misses': [missed * second // 80] * 100,
                  'sent': [second * 20] * 100} for second in p['cumulative_metrics_seconds']]
        output.append({'site': site, 'packet_seed': packet, 'handover_seconds': cost, 'arm': arm,
            'result': {'control_misses': sum(counts), 'control_sent': sum(sent), 'roams': 3,
                'miss_causes': {'fixture': sum(counts)},
                'packet_detail': {'misses': counts, 'sent': sent, 'max_gap': [10 if arm.startswith('staggered_') and arm not in ('staggered_ignored',) else 20] * 100,
                    'cumulative': trace, 'delivered_report_events': []},
                'managed_packet_accounting': {'video_requested_bits': 1e6, 'video_generated_bits': 8e5,
                    'video_delivered_bits': 7e5, 'bulk_generated_bits': 1e6, 'bulk_delivered_bits': 8e5,
                    'final_bulk_backlog_bits': 2e5, 'bulk_overflow_drop_bits': 0.,
                    'bulk_flow_conservation_error_bits': 0., 'pending_control_packets': 2,
                    'pending_control_packet_age_slots': [1, 2] + [-1] * 98},
                'managed_probe_summary': {'captured_slots': 0 if p['arms'][arm][0] == 'none' else 48,
                    'control_allocator_slots': 8000, 'control_allocator': 'per_retry_band_domain_every_managed_slot',
                    'cycles_delivered': 0, 'report_bytes_transmitted': 0, 'report_transport_slots': [],
                    'requested_domain_slot_airtime': 2., 'allocated_domain_slot_airtime': 2.,
                    'pending_transport_cycles': 0, 'pending_delivery_cycles': 0}},
            'predictor_records': [{'decision': 0, 'local_compute_seconds': .001, 'used_observations': 0}],
            'probe_reports': [], 'runtime': {'local_head_p95_seconds': .001,
                'head_overrun_decisions': [], 'total_prediction_overrun_decisions': []}})
    return output


VALID = {'source_and_archive_integrity': True, 'causality_and_kernel_checks': True, 'exact_replay': True}


def test_exact_registered_matrix_and_no_data_access(registration):
    assert len(study.case_keys('development', registration)) == 32
    assert len(study.case_keys('confirmation', registration)) == 192
    assert len(study.verification_keys()) == 6
    assert len(set(study.case_name(k) for k in study.case_keys('confirmation', registration))) == 192


def test_all_primary_information_and_engineering_tests_reported(rows, registration):
    result = report.aggregate(rows, registration, validation=VALID)
    assert len(result['primary']) == 12
    assert len(result['information_effect']) == 4
    assert len(result['engineering_gates']) == 8
    assert len(result['every_case']) == 192 and len(result['means']) == 16
    assert len(result['cumulative_excess']) == 48
    assert all(result['primary_pass_by_estimator'].values())
    assert all(result['independent_information_pass_by_estimator'].values())
    assert all(result['service_qualified_by_estimator'].values())
    assert all(r['upper_quantile'] == 1 - .05 / 12 for r in result['primary'])
    assert all(r['upper_quantile'] == 1 - .05 / 4 for r in result['information_effect'])
    assert result['activated'] is False and result['hardware_qualified'] is False
    assert result['bootstrap_replicates'] == 20000
    assert len(result['primary'][0]['paired_site_differences']) == 6
    assert all(r['pending_control_right_censored'] for r in result['every_case'])
    assert 'negative and null' in report.markdown(result)


def test_null_and_negative_outcomes_not_hidden(rows, registration):
    for row in rows:
        if row['arm'] == 'staggered_memory':
            row['result']['packet_detail']['misses'] = [500] * 100
            row['result']['control_misses'] = 50000
    result = report.aggregate(rows, registration, validation=VALID)
    assert not result['primary_pass_by_estimator']['memory']
    assert result['primary_pass_by_estimator']['bayesian']
    assert len([r for r in result['primary'] if r['estimator'] == 'memory']) == 6
    assert all(r['mean_difference'] > 0 for r in result['primary'] if r['estimator'] == 'memory')
    assert not result['service_qualified_by_estimator']['memory']


def test_seed_average_precedes_whole_site_bootstrap(rows, registration):
    # At site10 one packet seed improves and one regresses: the site average is zero.
    for row in rows:
        if row['site'] == 10 and row['arm'] == 'staggered_memory':
            missed = 200 if row['packet_seed'] == 0 else 605
            row['result']['packet_detail']['misses'] = [missed] * 100
            row['result']['control_misses'] = missed * 100
    result = report.aggregate(rows, registration, validation=VALID)
    comparison = next(r for r in result['primary'] if r['estimator'] == 'memory' and
                      r['handover_seconds'] == .03 and r['reference'] == 'canonical')
    assert comparison['paired_site_differences'][0] == pytest.approx(0.)
    assert len(comparison['paired_site_differences']) == 6
    indices = np.random.default_rng(2004000).integers(0, 6, (20000, 6))
    manual = np.asarray(comparison['paired_site_differences'])[indices].mean(1)
    assert comparison['multiplicity_upper'] == pytest.approx(np.quantile(manual, 1 - .05 / 12))


@pytest.mark.parametrize('change,gate', [('video', 'original_video'), ('bulk', 'bulk_delivery'),
    ('gap', 'maximum_silence'), ('decile', 'worst_decile')])
def test_secondary_service_regression_vetoes_primary_gain(rows, registration, change, gate):
    for row in rows:
        if row['arm'] != 'staggered_memory':
            continue
        if change == 'video':
            row['result']['managed_packet_accounting']['video_delivered_bits'] = 680000.
        elif change == 'bulk':
            row['result']['managed_packet_accounting']['bulk_delivered_bits'] = 790000.
        elif change == 'gap':
            row['result']['packet_detail']['max_gap'][0] = 21
        else:
            # Lower overall loss while concentrating failures in ten robots.
            row['result']['packet_detail']['misses'] = [900] * 10 + [0] * 90
            row['result']['control_misses'] = 9000
    result = report.aggregate(rows, registration, validation=VALID)
    assert result['primary_pass_by_estimator']['memory']
    assert not result['engineering_pass_by_estimator']['memory']
    assert not result['service_qualified_by_estimator']['memory']
    assert any(not r['checks'][gate] for r in result['engineering_gates'] if r['estimator'] == 'memory')


def test_information_effect_is_distinct_from_service_primary(rows, registration):
    lookup = {(r['site'], r['packet_seed'], r['handover_seconds'], r['arm']): r for r in rows}
    for row in rows:
        if row['arm'] == 'synchronized_memory':
            source = lookup[(row['site'], row['packet_seed'], row['handover_seconds'], 'staggered_memory')]
            row['result']['packet_detail']['misses'] = source['result']['packet_detail']['misses'].copy()
            row['result']['control_misses'] = source['result']['control_misses']
    result = report.aggregate(rows, registration, validation=VALID)
    assert result['service_qualified_by_estimator']['memory']
    assert not result['independent_information_pass_by_estimator']['memory']
    assert not result['independent_information_service_qualified_by_estimator']['memory']
    assert all(r['mean_difference'] == 0 for r in result['information_effect'] if r['estimator'] == 'memory')


@pytest.mark.parametrize('fault', ['head', 'total', 'replay', 'kernel', 'source', 'missing_validation'])
def test_integrity_or_uncharged_compute_veto(rows, registration, fault):
    validation = VALID.copy()
    if fault in ('head', 'total'):
        field = 'head_overrun_decisions' if fault == 'head' else 'total_prediction_overrun_decisions'
        rows[0]['runtime'][field] = [7]
    elif fault == 'missing_validation':
        validation = None
    else:
        field = {'replay': 'exact_replay', 'kernel': 'causality_and_kernel_checks', 'source': 'source_and_archive_integrity'}[fault]
        validation[field] = False
    result = report.aggregate(rows, registration, validation=validation)
    assert not any(result['service_qualified_by_estimator'].values())
    assert all(result['primary_pass_by_estimator'].values())


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'unexpected', 'sent', 'fractional'])
def test_incomplete_or_unpaired_data_fail_closed(rows, registration, fault):
    if fault == 'missing':
        rows.pop()
    elif fault == 'duplicate':
        rows.append(rows[0])
    elif fault == 'unexpected':
        rows[0]['site'] = 999
    elif fault == 'sent':
        rows[0]['result']['packet_detail']['sent'][0] = 1599
        rows[0]['result']['control_sent'] -= 1
    else:
        rows[0]['result']['packet_detail']['misses'][0] += .5
    with pytest.raises(ValueError):
        report.aggregate(rows, registration, validation=VALID)


def test_public_schedule_fixed_rng_cost_matching_and_finite_horizon(registration):
    geometry = SimpleNamespace(aps=[SimpleNamespace(channel=52), SimpleNamespace(channel=36)])
    specs = [study.public_spec(73, geometry, design, registration) for design in ('none', 'synchronized', 'staggered')]
    assert all(s.positions_m == specs[0].positions_m and s.activity_plan == specs[0].activity_plan for s in specs)
    assert len(specs[0].activity_plan) == 11100
    expected_rng = np.random.default_rng(np.random.SeedSequence([73, 202021]))
    assert np.array_equal(specs[0].positions_m, expected_rng.uniform([10, 6], [110, 74], (2, 2)))
    assert specs[0].channel == expected_rng.choice([36, 52])
    assert specs[0].power_mw == pytest.approx((10 ** 1.5,) * 2)
    plan = np.asarray(specs[0].activity_plan)
    for start in specs[0].probe_starts:
        assert np.array_equal(plan[start:start + 3], np.zeros((3, 2)))
    assert all(start % 10 <= 7 for start in specs[0].probe_starts)
    assert len([s for s in specs[0].probe_starts if s < 8000]) == 16
    assert specs[1].activity(1) == pytest.approx((.02, .02))
    assert specs[2].activity(1) == pytest.approx((.02, 0.))
    assert specs[2].activity(2) == pytest.approx((0., .02))
    assert specs[0].phase(1) is None


def test_atomic_gzip_evidence_is_immutable(tmp_path):
    path = tmp_path / 'case.json.gz'
    study.write_once(path, {'truth': 'unit fixture', 'count': 7}, compressed=True)
    original = path.read_bytes()
    assert json.loads(gzip.decompress(original))['count'] == 7
    study.write_once(path, {'truth': 'unit fixture', 'count': 7}, compressed=True)
    assert path.read_bytes() == original
    with pytest.raises(RuntimeError, match='Immutable'):
        study.write_once(path, {'count': 8}, compressed=True)
    assert path.read_bytes() == original


def test_registration_is_source_only_and_rejects_modified_source(tmp_path, monkeypatch, registration):
    out = tmp_path / 'artifacts/continual_packet'
    proto = out / 'registration/protocol.json'
    proto.parent.mkdir(parents=True)
    proto.write_text(json.dumps(registration))
    source = tmp_path / 'module.py'
    source.write_text('source only\n')
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(study, 'OUT', out)
    monkeypatch.setattr(study, 'PROTOCOL', proto)
    monkeypatch.setattr(study, 'sources', lambda: {'module.py': study.digest(source)})
    receipt = study.register()
    assert receipt['data_accessed'] is False
    assert receipt['main_case_counts'] == {'development': 32, 'confirmation': 192}
    assert study.register() == receipt
    sealed = out / 'registration/source/module.py'
    assert sealed.read_text() == 'source only\n'
    source.write_text('changed\n')
    with pytest.raises(RuntimeError, match='implementation changed'):
        study.register()
    assert sealed.read_text() == 'source only\n'


def full_fixture(row, p):
    row = deepcopy(row)
    row.update(case=['confirmation', row['site'], row['packet_seed'], row['handover_seconds'], row['arm']],
        source_sha256={'unit': 'fixture'}, data_sha256={'physical': 'unit data'},
        charged_latency={'forecast_seconds': {'cold': .1, 'warm': .1, 'refit': .1},
            'existing_extra_seconds': .5, 'shared_head_seconds': .1}, extra_seconds=.5)
    row['forecast_timing'] = []
    row['predictor_records'] = []
    ready = 0.
    for d in range(80):
        ready = max(d, ready) + .6
        row['forecast_timing'].append({'decision': d, 'history_cutoff_exclusive': 60 + d,
            'kind': 'cold' if d == 0 else 'warm', 'charged_seconds': .6,
            'ready_slot': math.ceil(ready * 100 - 1e-10), 'actual_prediction_seconds': .01})
        row['predictor_records'].append({'decision': d, 'local_compute_seconds': .001})
    counts = row['result']['packet_detail']['misses']
    row['result']['packet_detail']['cumulative'] = [
        {'slot_end': t * 100, 'misses': [m * t // 80 for m in counts], 'sent': [20 * t] * 100}
        for t in range(1, 81)]
    return row


def test_case_resume_checks_source_data_charge_archive_and_causal_ledger(rows, registration, tmp_path, monkeypatch):
    row = full_fixture(rows[0], registration)
    key = tuple(row['case'])
    monkeypatch.setattr(study, 'OUT', tmp_path)
    monkeypatch.setattr(study, 'protocol', lambda: registration)
    monkeypatch.setattr(study, 'check_sources', lambda: {'source_sha256': {'unit': 'fixture'}})
    path = tmp_path / 'cases' / (study.case_name(key) + '.json.gz')
    study.write_once(path, row, compressed=True)
    # Crash between atomic archive and receipt can be recovered after complete validation.
    loaded = study.load_case(key, data=row['data_sha256'], charged=row['charged_latency'])
    assert loaded == row
    receipt = path.with_suffix('').with_suffix('.receipt.json')
    assert receipt.exists()
    for kwargs in ({'data': {'physical': 'changed'}}, {'charged': {'changed': True}}):
        with pytest.raises(RuntimeError, match='mismatch'):
            study.load_case(key, **kwargs)
    old_bytes = path.read_bytes()
    bad = deepcopy(row)
    bad['forecast_timing'][1]['history_cutoff_exclusive'] += 1
    path.write_bytes(gzip.compress(study.encoded(bad), mtime=0))
    with pytest.raises(RuntimeError, match='noncausal'):
        study.load_case(key)
    path.write_bytes(old_bytes)
    receipt.write_text('{}')
    with pytest.raises(RuntimeError, match='Immutable'):
        study.load_case(key)


def test_latency_seal_uses_only_maximum_corrected_development_case_p95(tmp_path, monkeypatch, registration):
    monkeypatch.setattr(study, 'OUT', tmp_path)
    monkeypatch.setattr(study, 'protocol', lambda: registration)
    monkeypatch.setattr(study, 'check_sources', lambda: {'source_sha256': {'fixture': 'only'}})
    rows = []
    for key in study.case_keys('development', registration):
        arm = key[-1]
        value = 100. if arm == 'canonical' else .051
        rows.append({'case': list(key), 'arm': arm, 'runtime': {'local_head_p95_seconds': value}})
        study.write_once(tmp_path / 'cases' / (study.case_name(key) + '.receipt.json'), {'unit_fixture': True})
    with pytest.raises(RuntimeError, match='every development'):
        study.seal_latency(rows[:-1])
    study.seal_latency(rows)
    seal = json.loads((tmp_path / 'latency.json').read_text())
    assert seal['shared_head_seconds'] == .11
    assert seal['packet_outcomes_used'] is False
    assert len(seal['development_receipt_sha256']) == 32


def test_one_miss_apparent_gain_cannot_qualify_from_two_pending_commands(rows, registration):
    for row in rows:
        if row['arm'] not in ('canonical', 'reciprocal_radiomap', 'staggered_ignored',
                              'synchronized_ignored', 'synchronized_memory', 'staggered_memory'):
            continue
        counts = [400] * 100
        account = row['result']['managed_packet_accounting']
        account['pending_control_packet_age_slots'] = [-1] * 100
        account['pending_control_packets'] = 0
        if row['arm'] == 'staggered_memory':
            counts[0] -= 1
            account['pending_control_packet_age_slots'][:2] = [1, 2]
            account['pending_control_packets'] = 2
        row['result']['packet_detail']['misses'] = counts
        row['result']['control_misses'] = sum(counts)
    result = report.aggregate(rows, registration, validation=VALID)
    comparisons = [r for r in result['primary'] if r['estimator'] == 'memory']
    assert all(r['observed_passes'] for r in comparisons)
    assert all(not r['endpoint_passes'] for r in comparisons)
    assert not result['primary_pass_by_estimator']['memory']
    information = [r for r in result['information_effect'] if r['estimator'] == 'memory']
    assert all(r['observed_passes'] and not r['endpoint_passes'] for r in information)
    assert not result['service_qualified_by_estimator']['memory']


@pytest.mark.parametrize('fault', ['shape', 'negative', 'deadline', 'fractional', 'count', 'already_missed'])
def test_pending_endpoint_vector_is_validated(rows, registration, fault):
    row = rows[0]
    account = row['result']['managed_packet_accounting']
    if fault == 'shape':
        account['pending_control_packet_age_slots'].pop()
    elif fault == 'negative':
        account['pending_control_packet_age_slots'][0] = -2
    elif fault == 'deadline':
        account['pending_control_packet_age_slots'][0] = 5
    elif fault == 'fractional':
        account['pending_control_packet_age_slots'][0] = .5
    elif fault == 'count':
        account['pending_control_packets'] += 1
    else:
        row['result']['packet_detail']['misses'][0] = 1600
        row['result']['control_misses'] = sum(row['result']['packet_detail']['misses'])
    with pytest.raises(ValueError, match='right-censored'):
        report.aggregate(rows, registration, validation=VALID)
