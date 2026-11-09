"""Pure packet/ledger arithmetic fixtures; no study worlds or model forwards."""
from copy import deepcopy
import random
import json
import numpy as np
import pytest
from wirelessjev.pilot_fastloop.report import (
    FINAL, VARIANTS, aggregate as raw_aggregate, arms, development_gate as raw_development_gate, markdown, replication_summary)

MECHANICAL = dict(source_and_archive_integrity=True, causality_and_kernel_checks=True,
                  paid_cost_and_consumption_checks=True, matched_source_and_archive_integrity=True,
                  matched_causality_checks=True, matched_metric_integrity=True)
VALIDATION = dict(MECHANICAL, exact_offline_replay=True, exact_matched_replay=True)


_AUTO = object()


def matched(p, stage='confirmation'):
    return [dict(case=[stage, site, family], stage=stage, site=site, family=family,
        completed=True, replay=False, offline_only=True, fresh_measured_deadline_claim=False,
        new_fits=0, training_seed=None, reference_case=[stage, site, 4., 'radiomap', FINAL],
        matched_inputs={'fixture_reference': site}, reference_archive_sha256='a'*64,
        source_and_archive_pass=True, causality_pass=True, matched_inputs_pass=True,
        prediction_integrity_pass=True,
        metrics=dict(short_route_delta_rmse_db=2. if family == 'radiomap' else 1.,
            signal_rmse_db=3., all_route_delta_rmse_db=2., ap_choice_regret_db=.5))
        for site in p[stage+'_sites'] for family in p['families']]


def aggregate(rows, p, validation=None, passed_costs=None, matched_rows=_AUTO):
    if matched_rows is _AUTO:
        matched_rows = [] if passed_costs == [] else matched(p)
    return raw_aggregate(rows, p, validation, passed_costs, matched_rows)


def development_gate(rows, p, validation=None, matched_rows=_AUTO):
    if matched_rows is _AUTO:
        matched_rows = matched(p, 'development')
    return raw_development_gate(rows, p, validation, matched_rows)


@pytest.fixture
def p():
    return dict(study='Fixture only', families=['radiomap', 'frozen_fleet_jepa'],
        handover_seconds=[4.], variants_by_cost={'4.0': list(VARIANTS)},
        primary_upper_quantile=1-.05/4, robots=10, duration_seconds=2,
        development_sites=[11, 12], confirmation_sites=[21, 22, 23],
        cumulative_metrics_seconds=[1, 2], bootstrap_replicates=500, bootstrap_seed=2156000,
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
    mode = 'latest'
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
        'pilot_reporting': dict(enabled=variant in ('pilot', FINAL), new_rf_reads=0, extra_control_frames=0),
        'fast_local': dict(fast_local_enabled=variant in ('fast', FINAL),
            extra_RF_or_model_calls=False, helper_work_independently_scheduled=False),
        'fast_local_timing': dict(calls=0, total_seconds=0., max_seconds=0.),
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


def test_fixed_four_primary_and_descriptive_factorial_components(p):
    report = aggregate(matrix(p), p, VALIDATION)
    assert report['case_count'] == 24 and report['independent_sites'] == 3
    assert len(arms(4.)) == 8 and len(report['primary']) == 4 and len(report['engineering_gates']) == 3
    assert report['primary_multiplicity_budget'] == 4 and report['all_three_packet_benefits']
    assert len(report['secondary_component_contrasts']) == 14
    for contrast in report['secondary_component_contrasts']:
        assert contrast['statistical_pass'] is None and not contrast['experimental_benefit']
        assert contrast['observed_passes'] is None and contrast['endpoint_passes'] is None
        assert not contrast['statistical_superiority_claimed']
    assert all(row['upper_quantile'] == 1-.05/4 for row in report['primary'])
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
        target = next(row for row in rows if row['controller_variant'] == 'fast')
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
                    if row['contrast_id'] == 'combined_minus_base:radiomap')
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
                    if row['contrast_id'] == 'combined_minus_base:frozen_fleet_jepa')
    assert contrast['statistical_pass'] and not contrast['engineering_pass']
    assert not contrast['experimental_benefit']


def test_development_final_only_exact_severe_boundary_no_selection(p):
    rows = matrix(p, 'development')
    for row in rows:
        counts = [1]*p['robots']
        if row['controller_variant'] == FINAL: counts[:4] = [2]*4
        if row['controller_variant'] == 'pilot': counts = [30]*p['robots']
        set_counts(row, counts)
    gate = development_gate(rows, p, MECHANICAL)
    assert gate['passed_costs'] == [4.]
    assert all(check['combined_minus_base'] == .01 and not check['severe_regression']
               for check in gate['per_cost'][0]['family_checks'])
    row = next(row for row in rows if row['controller_variant'] == FINAL)
    counts = row['result']['packet_detail']['misses'].copy(); counts[0] += 1; set_counts(row, counts)
    gate = development_gate(rows, p, MECHANICAL)
    assert gate['passed_costs'] == [] and gate['per_cost'][0]['status'] == 'development_severe_regression_veto'
    assert not gate['model_or_controller_selection_performed'] and not gate['statistical_superiority_claimed']


def test_no_confirmation_retains_four_unrun_primaries_and_fourteen_components(p):
    report = aggregate([], p, VALIDATION, passed_costs=[])
    assert report['case_count'] == report['independent_sites'] == 0
    assert len(report['primary']) == 4 and len(report['secondary_component_contrasts']) == 14
    assert all(row['mean_difference'] is None and not row['experimental_benefit'] for row in report['primary'])
    assert not report['all_three_packet_benefits'] and 'not measured' in markdown(report)
    with pytest.raises(ValueError): aggregate(matrix(p), p, VALIDATION, passed_costs=[])


def test_sparse_source_ids_and_all_slot_held_age_not_job_ordinals(p):
    rows = matrix(p)
    for row in rows:
        row['arrivals'][1].update(accepted=False, job_index=None, reason='busy')
        row['accepted_decisions'] = [0]; row['jobs'] = row['jobs'][:1]
    report = aggregate(rows, p, VALIDATION)
    assert report['timing']['arrivals'] == 48 and report['timing']['jobs'] == 24
    assert report['timing']['skipped_arrivals'] == 24 and report['timing']['consumptions'] == 24
    assert report['timing']['maximum_forecast_age_seconds'] == .6
    assert report['timing']['maximum_held_forecast_age_seconds'] == 1.99
    assert report['timing']['accepted_jobs_not_consumed'] == 0


@pytest.mark.parametrize('change', ['missing_arrival', 'job_ordinal', 'decision', 'accepted_ids',
    'consume_ordinal', 'nonfinite_cost', 'bad_mode', 'assigned_skip'])
def test_malformed_sparse_ledgers_fail_closed(p, change):
    rows = matrix(p); row = next(row for row in rows if row['controller_variant'] == 'pilot')
    if change == 'missing_arrival': row['arrivals'].pop()
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
    assert report['all_three_packet_benefits']


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
        if row['controller_variant'] == 'pilot': set_counts(row, [0]*p['robots'])
        if row['controller_variant'] == FINAL: set_counts(row, [12]*p['robots'])
    report = aggregate(rows, p, VALIDATION)
    assert all(row['candidate_variant'] == FINAL for row in report['primary'])
    assert not any(row['experimental_benefit'] for row in report['primary'])
    assert all(row['statistical_pass'] is None for row in report['secondary_component_contrasts'])
    assert '2x2 factorial' in markdown(report) and 'No training' in markdown(report)


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
        row['accepted_decisions'] = [0, 2]
        row['jobs'] = [row['jobs'][0], row['jobs'][2]]
        row['jobs'][1]['job_index'] = 1
        for d, arrival in enumerate(row['arrivals']):
            arrival.update(accepted=d in (0, 2), job_index={0: 0, 2: 1}.get(d))
        row['forecast_consumption'] = [dict(slot=150, forecast_decision=0, ready_slot=150, age_seconds=1.5),
            dict(slot=350, forecast_decision=2, ready_slot=350, age_seconds=1.5)]
    report = aggregate(rows, p, VALIDATION)
    latest = next(row for row in report['every_case'] if row['controller_variant'] == 'pilot')
    assert latest['timing']['maximum_forecast_age_seconds'] == 1.5
    assert latest['timing']['maximum_held_forecast_age_seconds'] == 3.49
    assert latest['timing']['forecast_jobs'] == latest['timing']['skipped_arrivals'] == 2
    target = next(row for row in rows if row['controller_variant'] == 'pilot')
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


def test_matched_primary_is_short_delta_only_fully_observed_and_separate(p):
    streams = matched(p)
    # Better short motion-conditioned prediction can coexist with worse static
    # calibration and long-horizon prediction; neither replaces the endpoint.
    for row in streams:
        if row['family'] == 'frozen_fleet_jepa':
            row['metrics'].update(signal_rmse_db=50., all_route_delta_rmse_db=70.)
    report = aggregate(matrix(p), p, VALIDATION, matched_rows=streams)
    rf = report['primary'][-1]
    assert report['matched_case_count'] == 6
    assert report['matched_independent_sites_added'] == rf['independent_sites_added'] == 0
    assert rf['kind'] == 'matched_rf' and rf['units'] == 'dB'
    assert rf['metric'] == 'short_route_delta_rmse_db'
    assert rf['paired_site_differences'] == [-1.]*3 and rf['fully_observed']
    assert rf['endpoint_sensitivity'] is rf['engineering_pass'] is None
    assert rf['matched_forecast_advantage'] and report['matched_forecast_advantage']
    assert not rf['experimental_benefit'] and not rf['packet_benefit']
    assert not rf['fresh_measured_deadline_claim']


def test_matched_accuracy_does_not_promote_packet_controller(p):
    rows = matrix(p)
    for row in rows:
        if row['family'] == 'frozen_fleet_jepa' and row['controller_variant'] == FINAL:
            row['result']['packet_accounting'].update(bulk_delivered_bits=20., final_bulk_backlog_bits=80.)
    report = aggregate(rows, p, VALIDATION)
    assert report['matched_forecast_advantage']
    assert not report['all_three_packet_benefits'] and not report['all_four_primary_supported']
    assert not report['production_qualified'] and not report['hardware_qualified']


def test_matched_null_does_not_erase_supported_packet_contrasts(p):
    streams = matched(p)
    for row in streams: row['metrics']['short_route_delta_rmse_db'] = 2.
    report = aggregate(matrix(p), p, VALIDATION, matched_rows=streams)
    assert report['all_three_packet_benefits']
    assert not report['matched_forecast_advantage'] and not report['all_four_primary_supported']
    assert report['primary'][-1]['multiplicity_upper'] == 0.


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'foreign_site', 'family', 'stage',
    'case', 'replay', 'completed', 'online', 'deadline', 'fit', 'bool_fit',
    'nan', 'negative', 'boolean_metric', 'trace', 'reference', 'no_receipt', 'bad_sha',
    'reference_variant', 'reference_family', 'training_seed'])
def test_matched_matrix_identity_and_shared_reference_fail_closed(p, change):
    streams = matched(p)
    if change == 'missing': streams.pop()
    if change == 'duplicate': streams.append(deepcopy(streams[0]))
    if change == 'foreign_site': streams[0]['site'] = 999
    if change == 'family': streams[0]['family'] = 'oracle'
    if change == 'stage': streams[0]['stage'] = 'development'
    if change == 'case': streams[0]['case'][-1] = 'other'
    if change == 'replay': streams[0]['replay'] = True
    if change == 'completed': streams[0]['completed'] = 1
    if change == 'online': streams[0]['offline_only'] = False
    if change == 'deadline': streams[0]['fresh_measured_deadline_claim'] = True
    if change == 'fit': streams[0]['new_fits'] = 1
    if change == 'bool_fit': streams[0]['new_fits'] = False
    if change == 'nan': streams[0]['metrics']['short_route_delta_rmse_db'] = float('nan')
    if change == 'negative': streams[0]['metrics']['short_route_delta_rmse_db'] = -.1
    if change == 'boolean_metric': streams[0]['metrics']['short_route_delta_rmse_db'] = True
    if change == 'trace': streams[0]['matched_inputs'] = {'fixture_reference': 'different'}
    if change == 'reference': streams[0]['reference_archive_sha256'] = 'b'*64
    if change == 'no_receipt': streams[0]['matched_inputs'] = {}
    if change == 'bad_sha': streams[0]['reference_archive_sha256'] = 'x'*64
    if change == 'reference_variant': streams[0]['reference_case'][-1] = 'base'
    if change == 'reference_family': streams[0]['reference_case'][-2] = 'frozen_fleet_jepa'
    if change == 'training_seed': streams[0]['training_seed'] = 10
    with pytest.raises(ValueError): aggregate(matrix(p), p, VALIDATION, matched_rows=streams)


@pytest.mark.parametrize('flag', ['source_and_archive_pass', 'causality_pass',
    'matched_inputs_pass', 'prediction_integrity_pass'])
@pytest.mark.parametrize('value', [False, 1, None])
def test_any_matched_mechanical_failure_stops_development_and_claims(p, flag, value):
    for stage in ('development', 'confirmation'):
        streams = matched(p, stage); streams[-1][flag] = value
        if stage == 'development':
            gate = development_gate(matrix(p, stage), p, MECHANICAL, matched_rows=streams)
            assert gate['passed_costs'] == []
            assert gate['per_cost'][0]['status'] == 'development_mechanical_veto'
        else:
            report = aggregate(matrix(p), p, VALIDATION, matched_rows=streams)
            assert all(row['statistical_pass'] for row in report['primary'])
            assert not report['matched_forecast_advantage']
            assert not report['all_three_packet_benefits']


@pytest.mark.parametrize('flag', ['matched_source_and_archive_integrity',
    'matched_causality_checks', 'matched_metric_integrity', 'exact_matched_replay'])
def test_matched_runner_flags_must_be_literal_true(p, flag):
    validation = dict(VALIDATION, **{flag: 1})
    report = aggregate(matrix(p), p, validation)
    assert not report['proof']['experimental_claim_integrity_pass']
    assert not report['matched_forecast_advantage'] and not report['all_three_packet_benefits']


def test_full_registered_site_counts_and_two_replication_namespaces(p):
    p['confirmation_sites'] = list(range(21, 29))
    main = matrix(p)
    report = aggregate(main, p, VALIDATION)
    gate = development_gate(matrix(p, 'development'), p, MECHANICAL)
    assert (gate['case_count'], gate['matched_case_count']) == (16, 4)
    assert (report['case_count'], report['matched_case_count'], report['independent_sites']) == (64, 16, 8)
    for mode in ('recorded_schedule_offline', 'fresh_live_cost'):
        repeats = [deepcopy(row) for row in main if row['site'] == p['confirmation_sites'][0]]
        for row in repeats:
            row.update(replay=mode == 'recorded_schedule_offline', live_replication=mode == 'fresh_live_cost',
                accounting_mode=mode if mode == 'recorded_schedule_offline' else 'actual_cost')
        summary = replication_summary(repeats, p, mode, original_rows=main)
        assert summary['case_count'] == 8 and summary['independent_sites_added'] == 0


def test_factorial_interaction_and_main_effects_use_balanced_cells(p):
    rows = matrix(p)
    cells = {'base': 10, 'pilot': 8, 'fast': 7, 'pilot_fast': 3}
    for row in rows: set_counts(row, [cells[row['controller_variant']]]*p['robots'])
    report = aggregate(rows, p, VALIDATION)
    contrasts = {row['contrast_id'].split(':')[-1]: row for row in report['secondary_component_contrasts']
        if row['family'] == 'radiomap'}
    expected = {'pilot_at_fast_off': -2/40, 'pilot_at_fast_on': -4/40,
        'fast_at_pilot_off': -3/40, 'fast_at_pilot_on': -5/40,
        'pilot_main_effect': -3/40, 'fast_main_effect': -4/40, 'pilot_by_fast_interaction': -2/40}
    assert set(contrasts) == set(expected)
    for name, difference in expected.items():
        row = contrasts[name]
        assert row['mean_difference'] == pytest.approx(difference)
        assert row['paired_site_differences'] == pytest.approx([difference]*3)
        assert row['statistical_pass'] is None and not row['statistical_superiority_claimed']
        assert not row['experimental_benefit'] and row['engineering_pass'] is None


def test_matched_bootstrap_is_paired_whole_site_with_fixed_separate_seed(p):
    streams = matched(p)
    differences = [-1.2, .4, -.3]
    for row in streams:
        row['metrics']['short_route_delta_rmse_db'] = 2. + (differences[p['confirmation_sites'].index(row['site'])]
            if row['family'] == 'frozen_fleet_jepa' else 0.)
    report = aggregate(matrix(p), p, VALIDATION, matched_rows=streams)
    rf = report['primary'][-1]
    indices = np.random.default_rng(2157000).integers(0, 3, size=(500, 3))
    draws = np.asarray(differences)[indices].mean(1)
    assert report['matched_bootstrap_seed'] == 2157000
    assert rf['paired_site_differences'] == pytest.approx(differences)
    assert rf['multiplicity_upper'] == pytest.approx(np.quantile(draws, .9875))
    assert rf['descriptive_95_interval'] == pytest.approx(np.quantile(draws, [.025, .975]))
    assert not rf['statistical_pass']


def test_veto_reserves_rf_slot_and_rejects_attached_confirmatory_streams(p):
    report = aggregate([], p, VALIDATION, passed_costs=[], matched_rows=[])
    assert len(report['primary']) == 4 and report['primary_multiplicity_budget'] == 4
    assert all(row['upper_quantile'] == .9875 for row in report['primary'])
    rf = report['primary'][-1]
    assert rf['mean_difference'] is None and rf['engineering_pass'] is None
    assert not rf['matched_forecast_advantage'] and not report['matched_forecast_advantage']
    with pytest.raises(ValueError): aggregate([], p, VALIDATION, passed_costs=[], matched_rows=matched(p))


def test_no_matched_streams_never_pass_via_vacuous_proofs(p):
    with pytest.raises(ValueError): raw_aggregate(matrix(p), p, VALIDATION)
    with pytest.raises(ValueError): raw_development_gate(matrix(p, 'development'), p, MECHANICAL)


def test_old_report_contract_remains_unchanged(p):
    from wirelessjev.freshness_stack import report as old
    assert old.VARIANTS == ('fifo', 'latest', 'latest_single', 'latest_single_sense')
    assert len(old.primary_contrasts(4.)) == 3
    aggregate(matrix(p), p, VALIDATION)
    assert old.VARIANTS == ('fifo', 'latest', 'latest_single', 'latest_single_sense')


@pytest.mark.parametrize('change', ['pilot_off', 'numeric_enabled', 'fast_on', 'extra_model_call',
    'scheduled_helper', 'extra_rf_read', 'extra_control', 'nan_time', 'impossible_time'])
def test_factorial_receipts_and_helper_cost_claims_fail_closed(p, change):
    rows = matrix(p)
    row = next(row for row in rows if row['controller_variant'] == 'pilot')
    if change == 'pilot_off': row['pilot_reporting']['enabled'] = False
    if change == 'numeric_enabled': row['pilot_reporting']['enabled'] = 1
    if change == 'fast_on': row['fast_local']['fast_local_enabled'] = True
    if change == 'extra_model_call': row['fast_local']['extra_RF_or_model_calls'] = True
    if change == 'scheduled_helper': row['fast_local']['helper_work_independently_scheduled'] = True
    if change == 'extra_rf_read': row['pilot_reporting']['new_rf_reads'] = 1
    if change == 'extra_control': row['pilot_reporting']['extra_control_frames'] = 1
    if change == 'nan_time': row['fast_local_timing']['total_seconds'] = float('nan')
    if change == 'impossible_time': row['fast_local_timing']['max_seconds'] = .1
    with pytest.raises(ValueError): aggregate(rows, p, VALIDATION)


def test_report_keeps_raw_event_streams_in_archives_and_preserves_counts(p):
    rows = matrix(p)
    # A large public payload and forecast catalog must not multiply into every
    # report, development seal and replication summary.
    large = {'raw_fixture_payload': 'x' * 100000}
    for row in rows:
        row['pilot_reporting'].update(frame_events=[large]*4, row_replacements=[large]*2,
            pilot_reports_delivered=3)
        row['fast_local'].update(fast_control_updates=[large]*6, forecast_slices=[large]*2,
            mode_counts={'predictive': 6, 'myopic': 0, 'hold': 0})
    report = aggregate(rows, p, VALIDATION)
    assert len(json.dumps(report)) < 200000
    for row in report['every_case']:
        pilot, fast = row['pilot_reporting'], row['fast_local']
        assert 'frame_events' not in pilot and 'row_replacements' not in pilot
        assert 'fast_control_updates' not in fast and 'forecast_slices' not in fast
        assert (pilot['frame_event_count'], pilot['row_replacement_count']) == (4, 2)
        assert (fast['fast_control_update_count'], fast['forecast_slice_count']) == (6, 2)
        assert pilot['pilot_reports_delivered'] == 3 and fast['mode_counts']['predictive'] == 6
        assert row['raw_evidence_locator'].startswith('artifacts/pilot_fastloop/cases/')
        assert row['raw_evidence_locator'].endswith('.json.gz')
    assert rows[0]['pilot_reporting']['frame_events'][0] is large
