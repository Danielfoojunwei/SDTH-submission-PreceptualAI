"""Pure packet/ledger arithmetic fixtures; no study worlds or model forwards."""
from copy import deepcopy
import random
import pytest
from wirelessjev.freshness_stack.report import (
    FINAL, VARIANTS, aggregate, arms, development_gate, markdown, replication_summary)

MECHANICAL = dict(source_and_archive_integrity=True, causality_and_kernel_checks=True,
                  paid_cost_and_consumption_checks=True)
VALIDATION = dict(MECHANICAL, exact_offline_replay=True)


@pytest.fixture
def p():
    return dict(study='Fixture only', families=['radiomap', 'frozen_fleet_jepa'],
        handover_seconds=[4.], variants_by_cost={'4.0': list(VARIANTS)},
        primary_upper_quantile=1-.05/3, robots=10, duration_seconds=2,
        development_sites=[11, 12], confirmation_sites=[21, 22, 23],
        cumulative_metrics_seconds=[1, 2], bootstrap_replicates=500, bootstrap_seed=2146000,
        worst_decile_margin=1/40, silence_margin_seconds=.05, video_margin=.01,
        bulk_ratio=.99, development_severe_regression_absolute=.01)


def one(p, stage, site, cost, family, variant, miss):
    n, total = p['robots'], 20*p['duration_seconds']
    m, sent, ages = [miss]*n, [total]*n, [-1]*n
    detail = {'misses': m, 'sent': sent, 'max_gap': [2]*n,
        'pending_control_packet_age_slots': ages, 'pending_control_packets': 0,
        'cumulative': [
            {'slot_end': 100, 'misses': [miss//2]*n, 'sent': [20]*n, 'max_gap': [1]*n},
            {'slot_end': 200, 'misses': m.copy(), 'sent': sent.copy(), 'max_gap': [2]*n}]}
    account = {'video_requested_bits': 1000., 'video_generated_bits': 1000.,
        'video_delivered_bits': 900., 'video_pending_bits': 100., 'video_expired_bits': 0.,
        'video_flow_conservation_error_bits': 0., 'bulk_generated_bits': 100.,
        'bulk_delivered_bits': 90., 'final_bulk_backlog_bits': 10., 'bulk_overflow_drop_bits': 0.,
        'bulk_flow_conservation_error_bits': 0., 'pending_control_packet_age_slots': ages.copy(),
        'pending_control_packets': 0}
    mode = 'fifo' if variant == 'fifo' else 'latest'
    jobs = [dict(job_index=i, decision=i, source_decision=i, mode=mode,
                 charged_seconds=.6, charged_compute_seconds=.1,
                 actual_compute_seconds=.04, runtime_overrun=False) for i in range(p['duration_seconds'])]
    arrivals = [dict(decision=i, arrival_slot=i*100, accepted=True, job_index=i,
                     reason='accepted', mode=mode) for i in range(p['duration_seconds'])]
    return {'stage': stage, 'site': site, 'handover_seconds': cost, 'family': family,
        'controller_variant': variant, 'training_seed': None, 'runtime_overrun': False,
        'predictor_runtime_overrun': False, 'causality_pass': True,
        'paid_cost_and_consumption_pass': True, 'replay': False, 'live_replication': False,
        'accounting_mode': 'actual_cost', 'forecast_metrics': {}, 'jobs': jobs, 'arrivals': arrivals,
        'queue_mode': mode, 'accepted_decisions': list(range(p['duration_seconds'])),
        'forecast_consumption': [dict(slot=60, forecast_decision=0, ready_slot=60, age_seconds=.6)],
        'result': {'control_misses': miss*n, 'control_sent': total*n, 'control_miss': miss/total,
            'packet_detail': detail, 'packet_accounting': account,
            'freshness_execution': {'scan_starts': [], 'scan_assessments': []}}}


def matrix(p, stage='confirmation', costs=None):
    costs = p['handover_seconds'] if costs is None else costs
    values = {'radiomap': [10, 8, 6, 4], 'frozen_fleet_jepa': [9, 7, 5, 2]}
    return [one(p, stage, site, cost, family, variant, values[family][VARIANTS.index(variant)])
        for site in p[stage+'_sites'] for cost in costs for family, variant in arms(cost)]


def set_counts(row, counts):
    detail = row['result']['packet_detail']; detail['misses'] = list(counts)
    detail['cumulative'][0]['misses'] = [c//2 for c in counts]
    detail['cumulative'][-1]['misses'] = list(counts)
    row['result']['control_misses'] = sum(counts)
    row['result']['control_miss'] = sum(counts)/row['result']['control_sent']


def test_fixed_three_primary_and_descriptive_nested_components(p):
    report = aggregate(matrix(p), p, VALIDATION)
    assert report['case_count'] == 24 and report['independent_sites'] == 3
    assert len(arms(4.)) == 8 and len(report['primary']) == len(report['engineering_gates']) == 3
    assert report['primary_multiplicity_budget'] == 3 and report['all_three_experimental_benefits']
    assert len(report['secondary_component_contrasts']) == 6
    for contrast in report['secondary_component_contrasts']:
        assert contrast['statistical_pass'] is None and not contrast['experimental_benefit']
        assert contrast['observed_passes'] is None and contrast['endpoint_passes'] is None
        assert not contrast['statistical_superiority_claimed']
    assert all(row['upper_quantile'] == 1-.05/3 for row in report['primary'])
    assert not report['production_qualified'] and not report['hardware_qualified']
    assert not report['activated'] and not report['ema_necessity_claimed']
    assert report['new_fits'] == 0


def test_actual_eight_site_matrix_counts(p):
    p['confirmation_sites'] = list(range(21, 29))
    assert aggregate(matrix(p), p, VALIDATION)['case_count'] == 64
    assert development_gate(matrix(p, 'development'), p, MECHANICAL)['case_count'] == 16


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'foreign_site', 'variant', 'cost', 'stage',
    'invented_seed', 'offline', 'live', 'mode', 'numeric_flag'])
def test_complete_matrix_and_namespace_fail_closed(p, change):
    rows = matrix(p)
    if change == 'missing': rows.pop()
    if change == 'duplicate': rows.append(deepcopy(rows[0]))
    if change == 'foreign_site': rows[0]['site'] = 999
    if change == 'variant': rows[0]['controller_variant'] = 'selected_intermediate'
    if change == 'cost': rows[0]['handover_seconds'] = .03
    if change == 'stage': rows[0]['stage'] = 'development'
    if change == 'invented_seed': rows[0]['training_seed'] = 0
    if change == 'offline': rows[0]['replay'] = True
    if change == 'live': rows[0]['live_replication'] = True
    if change == 'mode': rows[0]['accounting_mode'] = 'sealed_budget'
    if change == 'numeric_flag': rows[0]['replay'] = 0
    with pytest.raises(ValueError): aggregate(rows, p, VALIDATION)


@pytest.mark.parametrize('field', ['causality_pass', 'paid_cost_and_consumption_pass'])
def test_failed_intermediate_arm_vetoes_all_primary_benefits_and_development(p, field):
    for stage in ('development', 'confirmation'):
        rows = matrix(p, stage)
        target = next(row for row in rows if row['controller_variant'] == 'latest_single')
        target[field] = False
        if stage == 'development':
            assert development_gate(rows, p, MECHANICAL)['passed_costs'] == []
        else:
            report = aggregate(rows, p, VALIDATION)
            assert all(row['statistical_pass'] for row in report['primary'])
            assert not any(row['experimental_benefit'] for row in report['primary'])


@pytest.mark.parametrize('validation', [None, {}, MECHANICAL,
    {**VALIDATION, 'exact_offline_replay': False}, {**VALIDATION, 'source_and_archive_integrity': 1}])
def test_validation_flags_require_literal_true(p, validation):
    report = aggregate(matrix(p), p, validation)
    assert not any(row['experimental_benefit'] for row in report['primary'])


def test_pending_endpoint_disqualifies_small_observed_gain(p):
    rows = matrix(p)
    for row in rows:
        if row['family'] == 'radiomap' and row['controller_variant'] == FINAL:
            set_counts(row, [9]*p['robots'])
            for container in (row['result']['packet_detail'], row['result']['packet_accounting']):
                container['pending_control_packet_age_slots'] = [0]*p['robots']
                container['pending_control_packets'] = p['robots']
    contrast = next(row for row in aggregate(rows, p, VALIDATION)['primary']
                    if row['contrast_id'] == 'final_minus_fifo:radiomap')
    assert contrast['observed_passes'] and not contrast['endpoint_passes']
    assert not contrast['experimental_benefit']


@pytest.mark.parametrize('change', ['silence', 'worst', 'video', 'bulk'])
def test_mean_improvement_cannot_hide_engineering_regression(p, change):
    rows = matrix(p)
    for row in rows:
        if row['family'] != 'frozen_fleet_jepa' or row['controller_variant'] != FINAL: continue
        detail, account = row['result']['packet_detail'], row['result']['packet_accounting']
        if change == 'silence':
            detail['max_gap'] = [20]*p['robots']; detail['cumulative'][-1]['max_gap'] = [20]*p['robots']
        if change == 'worst':
            counts = detail['misses'].copy(); counts[0] = 35; set_counts(row, counts)
        if change == 'video': account.update(video_delivered_bits=800., video_pending_bits=200.)
        if change == 'bulk': account.update(bulk_delivered_bits=80., final_bulk_backlog_bits=20.)
    contrast = next(row for row in aggregate(rows, p, VALIDATION)['primary']
                    if row['contrast_id'] == 'final_minus_fifo:frozen_fleet_jepa')
    assert contrast['statistical_pass'] and not contrast['engineering_pass']
    assert not contrast['experimental_benefit']


def test_development_final_only_exact_severe_boundary_no_selection(p):
    rows = matrix(p, 'development')
    for row in rows:
        counts = [1]*p['robots']
        if row['controller_variant'] == FINAL: counts[:4] = [2]*4
        if row['controller_variant'] == 'latest': counts = [30]*p['robots']
        set_counts(row, counts)
    gate = development_gate(rows, p, MECHANICAL)
    assert gate['passed_costs'] == [4.]
    assert all(check['final_minus_fifo'] == .01 and not check['severe_regression']
               for check in gate['per_cost'][0]['family_checks'])
    row = next(row for row in rows if row['controller_variant'] == FINAL)
    counts = row['result']['packet_detail']['misses'].copy(); counts[0] += 1; set_counts(row, counts)
    gate = development_gate(rows, p, MECHANICAL)
    assert gate['passed_costs'] == [] and gate['per_cost'][0]['status'] == 'development_severe_regression_veto'
    assert not gate['model_or_controller_selection_performed'] and not gate['statistical_superiority_claimed']


def test_no_confirmation_retains_three_unrun_primaries_and_six_components(p):
    report = aggregate([], p, VALIDATION, passed_costs=[])
    assert report['case_count'] == report['independent_sites'] == 0
    assert len(report['primary']) == 3 and len(report['secondary_component_contrasts']) == 6
    assert all(row['mean_difference'] is None and not row['experimental_benefit'] for row in report['primary'])
    assert not report['all_three_experimental_benefits'] and 'not measured' in markdown(report)
    with pytest.raises(ValueError): aggregate(matrix(p), p, VALIDATION, passed_costs=[])


def test_sparse_source_ids_and_all_slot_held_age_not_job_ordinals(p):
    rows = matrix(p)
    for row in rows:
        if row['queue_mode'] == 'fifo': continue
        row['arrivals'][1].update(accepted=False, job_index=None, reason='busy')
        row['accepted_decisions'] = [0]; row['jobs'] = row['jobs'][:1]
    report = aggregate(rows, p, VALIDATION)
    assert report['timing']['arrivals'] == 48 and report['timing']['jobs'] == 30
    assert report['timing']['skipped_arrivals'] == 18 and report['timing']['consumptions'] == 24
    assert report['timing']['maximum_forecast_age_seconds'] == .6
    assert report['timing']['maximum_held_forecast_age_seconds'] == 1.99
    assert report['timing']['accepted_jobs_not_consumed'] == 6


@pytest.mark.parametrize('change', ['missing_arrival', 'skipped_fifo', 'job_ordinal', 'decision', 'accepted_ids',
    'consume_ordinal', 'nonfinite_cost', 'bad_mode', 'assigned_skip'])
def test_malformed_sparse_ledgers_fail_closed(p, change):
    rows = matrix(p); row = next(row for row in rows if row['controller_variant'] == 'latest')
    if change == 'missing_arrival': row['arrivals'].pop()
    if change == 'skipped_fifo':
        row = rows[0]; row['arrivals'][1].update(accepted=False, job_index=None)
    if change == 'job_ordinal': row['jobs'][1]['job_index'] = 0
    if change == 'decision': row['jobs'][1]['source_decision'] = 0
    if change == 'accepted_ids': row['accepted_decisions'] = [0]
    if change == 'consume_ordinal': row['forecast_consumption'][0]['age_seconds'] = 1.6
    if change == 'nonfinite_cost': row['jobs'][0]['charged_seconds'] = float('nan')
    if change == 'bad_mode': row['queue_mode'] = 'fifo'
    if change == 'assigned_skip': row['arrivals'][1].update(accepted=False)
    with pytest.raises(ValueError): aggregate(rows, p, VALIDATION)


def test_age_extends_to_before_next_consumption_and_excludes_initial_fallback(p):
    rows = matrix(p)
    for row in rows:
        row['forecast_consumption'] = [dict(slot=60, forecast_decision=0, ready_slot=60, age_seconds=.6),
            dict(slot=160, forecast_decision=1, ready_slot=160, age_seconds=.6)]
    report = aggregate(rows, p, VALIDATION)
    assert report['timing']['maximum_held_forecast_age_seconds'] == 1.59
    assert report['timing']['fallback_slots_before_first_forecast'] == 60*len(rows)


def test_paid_profile_exceedance_is_diagnostic_but_kept_visible(p):
    rows = matrix(p); rows[0].update(runtime_overrun=True, predictor_runtime_overrun=True)
    for job in rows[0]['jobs']:
        job.update(runtime_overrun=True, actual_compute_seconds=.25, charged_compute_seconds=.25, charged_seconds=.75)
    report = aggregate(rows, p, VALIDATION)
    assert report['timing']['profile_exceeded_cases'] == 1
    assert report['timing']['profile_exceeded_jobs'] == 2
    assert report['all_three_experimental_benefits']


def test_execution_order_does_not_change_analysis_or_development_seal(p):
    for stage in ('development', 'confirmation'):
        rows = matrix(p, stage); shuffled = deepcopy(rows); random.Random(5).shuffle(shuffled)
        analysis = development_gate if stage == 'development' else aggregate
        validation = MECHANICAL if stage == 'development' else VALIDATION
        assert analysis(rows, p, validation) == analysis(shuffled, p, validation)


@pytest.mark.parametrize('mode', ['recorded_schedule_offline', 'fresh_live_cost'])
def test_replications_keep_eight_arms_and_add_no_independent_sites(p, mode):
    main = matrix(p); rows = [deepcopy(row) for row in main if row['site'] == p['confirmation_sites'][0]]
    for row in rows:
        row.update(replay=mode == 'recorded_schedule_offline', live_replication=mode == 'fresh_live_cost',
                   accounting_mode=mode if mode == 'recorded_schedule_offline' else 'actual_cost')
    if mode == 'fresh_live_cost': set_counts(rows[0], [12]*p['robots'])
    summary = replication_summary(rows, p, mode, original_rows=main)
    assert summary['case_count'] == 8 and summary['independent_sites_added'] == 0
    assert not summary['statistical_superiority_claimed'] and not summary['fresh_measured_deadline_claim']
    assert any(not row['packet_result_exactly_matches_original'] for row in summary['differences_from_main']) is (mode == 'fresh_live_cost')
    with pytest.raises(ValueError): aggregate(rows, p, VALIDATION)


def test_primary_fixed_final_does_not_select_better_intermediate(p):
    rows = matrix(p)
    for row in rows:
        if row['controller_variant'] == 'latest': set_counts(row, [0]*p['robots'])
        if row['controller_variant'] == FINAL: set_counts(row, [12]*p['robots'])
    report = aggregate(rows, p, VALIDATION)
    assert all(row['candidate_variant'] == FINAL for row in report['primary'])
    assert not any(row['experimental_benefit'] for row in report['primary'])
    assert all(row['statistical_pass'] is None for row in report['secondary_component_contrasts'])
    assert 'not a factorial' in markdown(report) and 'No training' in markdown(report)


def test_previous_report_globals_unchanged(p):
    from wirelessjev.costaware_recovery import report as old
    assert old.variants(4.) == ('legacy', 'repair')
    aggregate(matrix(p), p, VALIDATION)
    assert old.variants(4.) == ('legacy', 'repair')


def test_nonconsecutive_source_timestamps_survive_ordinal_one(p):
    p.update(duration_seconds=4, cumulative_metrics_seconds=[1, 4])
    rows = matrix(p)
    for row in rows:
        detail = row['result']['packet_detail']
        detail['cumulative'][-1]['slot_end'] = 400
        if row['queue_mode'] == 'fifo':
            row['forecast_consumption'] = [dict(slot=d*100+60, forecast_decision=d,
                ready_slot=d*100+60, age_seconds=.6) for d in range(4)]
            continue
        row['accepted_decisions'] = [0, 2]
        row['jobs'] = [row['jobs'][0], row['jobs'][2]]
        row['jobs'][1]['job_index'] = 1
        for d, arrival in enumerate(row['arrivals']):
            arrival.update(accepted=d in (0, 2), job_index={0: 0, 2: 1}.get(d))
        row['forecast_consumption'] = [dict(slot=150, forecast_decision=0, ready_slot=150, age_seconds=1.5),
            dict(slot=350, forecast_decision=2, ready_slot=350, age_seconds=1.5)]
    report = aggregate(rows, p, VALIDATION)
    latest = next(row for row in report['every_case'] if row['controller_variant'] == 'latest')
    assert latest['timing']['maximum_forecast_age_seconds'] == 1.5
    assert latest['timing']['maximum_held_forecast_age_seconds'] == 3.49
    assert latest['timing']['forecast_jobs'] == latest['timing']['skipped_arrivals'] == 2
    target = next(row for row in rows if row['controller_variant'] == 'latest')
    target['forecast_consumption'][1]['forecast_decision'] = 1
    with pytest.raises(ValueError): aggregate(rows, p, VALIDATION)


def test_no_consumed_forecast_has_no_claimed_freshness(p):
    rows = matrix(p)
    for row in rows: row['forecast_consumption'] = []
    report = aggregate(rows, p, VALIDATION)
    assert report['timing']['maximum_forecast_age_seconds'] is None
    assert report['timing']['maximum_held_forecast_age_seconds'] is None
    assert report['timing']['consumptions'] == 0
    assert report['timing']['accepted_jobs_not_consumed'] == 2*len(rows)
    assert report['timing']['fallback_slots_before_first_forecast'] == 200*len(rows)


def test_actual_sensing_decision_key_counts_suppressed_assessments(p):
    rows = matrix(p)
    for row in rows:
        row['result']['freshness_execution'] = {
            'scan_assessments': [dict(suppress_scan=True, reason='low_risk'),
                dict(suppress_scan=False, reason='weak_current'),
                dict(suppress_scan=False, reason='emergency_silence')],
            'scan_starts': [dict(slot=100), dict(slot=150)]}
    report = aggregate(rows, p, VALIDATION)
    assert all(row['sensing'] == dict(scan_assessments=3, scan_starts=2,
        suppressed_scan_assessments=1) for row in report['every_case'])
    assert all(row['scan_starts_per_trial'] == 2 and row['suppressed_scan_assessments_per_trial'] == 1
               for row in report['means'])
