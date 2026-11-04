"""Pure prospective-confirmation fixtures; no held-out data or model forward."""
from copy import deepcopy

import pytest

from wirelessjev.selected_jepa.report import aggregate, arms, markdown


VALIDATION = dict(source_and_archive_integrity=True, causality_and_kernel_checks=True, exact_replay=True)


@pytest.fixture
def p():
    return {'study': 'Selected JEPA unit fixture', 'candidate_family': 'context_jepa',
        'confirmation_sites': [21, 22, 23], 'training_seeds': [0, 1, 2],
        'handover_seconds': [.03, 4.], 'primary_references': ['radiomap', 'supervised', 'frozen_fleet_jepa'],
        'robots': 10, 'duration_seconds': 2, 'bootstrap_replicates': 500,
        'bootstrap_seed': 2115000, 'primary_upper_quantile': 1 - .05 / 6,
        'cumulative_metrics_seconds': [1, 2], 'worst_decile_margin': 1 / 40,
        'silence_margin_seconds': .05, 'video_margin': .01, 'bulk_ratio': .99}


def one(p, site, cost, family, seed, *, miss=2, pending=False, gap=2, video=900., bulk=90.):
    n, sent_count = p['robots'], 20 * p['duration_seconds']
    misses, sent = [miss] * n, [sent_count] * n
    ages = [0 if pending else -1] * n
    detail = {'misses': misses, 'sent': sent, 'max_gap': [gap] * n,
        'pending_control_packet_age_slots': ages, 'pending_control_packets': n if pending else 0,
        'cumulative': [
            {'slot_end': 100, 'misses': [miss // 2] * n, 'sent': [20] * n, 'max_gap': [min(gap, 1)] * n},
            {'slot_end': 200, 'misses': misses.copy(), 'sent': sent.copy(), 'max_gap': [gap] * n}]}
    account = {'video_requested_bits': 1000., 'video_generated_bits': 1000.,
        'video_delivered_bits': video, 'video_pending_bits': 1000. - video, 'video_expired_bits': 0.,
        'video_flow_conservation_error_bits': 0., 'bulk_generated_bits': 100.,
        'bulk_delivered_bits': bulk, 'final_bulk_backlog_bits': 100. - bulk,
        'bulk_overflow_drop_bits': 0., 'bulk_flow_conservation_error_bits': 0.,
        'pending_control_packet_age_slots': ages.copy(), 'pending_control_packets': n if pending else 0}
    return {'stage': 'confirmation', 'site': site, 'handover_seconds': cost,
        'family': family, 'training_seed': seed, 'runtime_overrun': False, 'causality_pass': True,
        'forecast_metrics': {}, 'result': {'control_misses': miss * n, 'control_sent': sent_count * n,
        'control_miss': miss / sent_count, 'packet_detail': detail, 'packet_accounting': account}}


def matrix(p, *, candidate=2, reference=4):
    return [one(p, site, cost, family, seed,
                miss=candidate if family == p['candidate_family'] else reference)
        for site in p['confirmation_sites'] for cost in p['handover_seconds'] for family, seed in arms(p)]


def replace_count(row, count):
    n = len(row['result']['packet_detail']['misses'])
    detail = row['result']['packet_detail']
    detail['misses'] = [count] * n
    detail['cumulative'][0]['misses'] = [count // 2] * n
    detail['cumulative'][-1]['misses'] = [count] * n
    row['result']['control_misses'] = count * n
    row['result']['control_miss'] = count / detail['sent'][0]


def test_exact_eight_arm_matrix_and_site_units(p):
    result = aggregate(matrix(p), p, VALIDATION)
    assert len(arms(p)) == 8
    assert result['case_count'] == 3 * 2 * 8
    assert result['independent_sites'] == 3
    assert result['candidate_family'] == 'context_jepa'
    assert len(result['primary']) == 6
    assert len(result['engineering_gates']) == 6
    assert len(result['cumulative_excess']) == 12
    assert len(result['means']) == 8
    assert len(result['per_seed_means']) == 16
    assert all(len(r['paired_site_differences']) == 3 for r in result['primary'])
    assert all(r['upper_quantile'] == 1 - .05 / 6 for r in result['primary'])
    assert result['headline_numerical_benefit_at_4s']
    assert result['both_costs_all_references_pass']
    assert not result['production_qualified']
    assert not result['hardware_qualified']
    assert not result['activated']
    assert not result['ema_necessity_claimed']


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'foreign_site', 'foreign_seed', 'extra_cost',
                                 'wrong_stage', 'wrong_candidate', 'baseline_seed'])
def test_exact_confirmation_membership(p, kind):
    rows = matrix(p)
    if kind == 'missing': rows.pop()
    if kind == 'duplicate': rows.append(deepcopy(rows[0]))
    if kind == 'foreign_site': rows[0]['site'] = 999
    if kind == 'foreign_seed': rows[0]['training_seed'] = 99
    if kind == 'extra_cost': rows[0]['handover_seconds'] = 1.
    if kind == 'wrong_stage': rows[0]['stage'] = 'development'
    if kind == 'wrong_candidate': rows[0]['family'] = 'semantic_jepa'
    if kind == 'baseline_seed':
        next(r for r in rows if r['family'] == 'radiomap')['training_seed'] = 0
    with pytest.raises(ValueError): aggregate(rows, p, VALIDATION)


def test_both_costs_are_mandatory_and_budget_cannot_shrink(p):
    rows = matrix(p)
    with pytest.raises(ValueError): aggregate([r for r in rows if r['handover_seconds'] == 4.], p, VALIDATION)
    bad = deepcopy(p); bad['handover_seconds'] = [4.]
    with pytest.raises(ValueError): aggregate(rows, bad, VALIDATION)
    bad = deepcopy(p); bad['primary_references'].pop()
    with pytest.raises(ValueError): aggregate(rows, bad, VALIDATION)
    bad = deepcopy(p); bad['primary_upper_quantile'] = .95
    with pytest.raises(ValueError): aggregate(rows, bad, VALIDATION)


@pytest.mark.parametrize('validation', [None, {}, {'exact_replay': True},
    {**VALIDATION, 'exact_replay': False}, {**VALIDATION, 'source_and_archive_integrity': 1},
    {**VALIDATION, 'causality_and_kernel_checks': False}, {**VALIDATION, 'runtime_failure': True}])
def test_integrity_replay_and_runtime_fail_closed(p, validation):
    result = aggregate(matrix(p), p, validation)
    assert not result['headline_numerical_benefit_at_4s']
    assert all(not r['qualified'] for r in result['primary'])
    assert all(not r['all_references_pass'] for r in result['per_cost'])


@pytest.mark.parametrize('failure', ['causal_false', 'causal_missing', 'runtime'])
def test_one_factual_case_failure_vetoes_every_claim(p, failure):
    rows = matrix(p)
    if failure == 'causal_false': rows[0]['causality_pass'] = False
    if failure == 'causal_missing': del rows[0]['causality_pass']
    if failure == 'runtime': rows[0]['runtime_overrun'] = True
    result = aggregate(rows, p, VALIDATION)
    assert all(r['statistical_pass'] for r in result['primary'])
    assert all(not r['qualified'] for r in result['primary'])
    assert not result['headline_numerical_benefit_at_4s']


def test_per_cost_claims_do_not_require_both_cost_success(p):
    rows = matrix(p)
    for row in rows:
        if row['family'] == p['candidate_family'] and row['handover_seconds'] == .03:
            replace_count(row, 5)
    result = aggregate(rows, p, VALIDATION)
    fast, slow = result['per_cost']
    assert not fast['numerical_benefit']
    assert slow['numerical_benefit']
    assert result['headline_numerical_benefit_at_4s']
    assert not result['both_costs_all_references_pass']


@pytest.mark.parametrize('tied_reference,missing_claim', [
    ('supervised', 'objective_advantage'),
    ('frozen_fleet_jepa', 'original_jepa_improvement'),
    ('radiomap', 'numerical_benefit'),
])
def test_reference_specific_claims_cannot_substitute_for_one_another(p, tied_reference, missing_claim):
    rows = matrix(p)
    for row in rows:
        if row['family'] == tied_reference: replace_count(row, 2)
    result = aggregate(rows, p, VALIDATION)
    for cost in result['per_cost']:
        assert not cost[missing_claim]
        assert sum(cost[name] for name in ('numerical_benefit', 'objective_advantage', 'original_jepa_improvement')) == 2
        assert not cost['all_references_pass']
    assert result['headline_numerical_benefit_at_4s'] is (tied_reference != 'radiomap')
    assert not result['ema_necessity_claimed']


def test_engineering_gate_is_specific_to_reference(p):
    rows = matrix(p)
    for row in rows:
        if row['family'] == 'frozen_fleet_jepa':
            row['result']['packet_accounting'].update(video_delivered_bits=950., video_pending_bits=50.)
    result = aggregate(rows, p, VALIDATION)
    for cost in result['per_cost']:
        assert cost['numerical_benefit']
        assert cost['objective_advantage']
        assert not cost['original_jepa_improvement']
    assert all(row['statistical_pass'] for row in result['primary'])
    assert all(not row['qualified'] for row in result['primary'] if row['reference'] == 'frozen_fleet_jepa')


def test_pending_endpoint_prevents_one_command_apparent_gain(p):
    rows = matrix(p, candidate=3, reference=4)
    for row in rows:
        if row['family'] == p['candidate_family']:
            for container in (row['result']['packet_detail'], row['result']['packet_accounting']):
                container['pending_control_packet_age_slots'] = [0] * p['robots']
                container['pending_control_packets'] = p['robots']
    result = aggregate(rows, p, VALIDATION)
    assert all(row['observed_passes'] for row in result['primary'])
    assert all(not row['endpoint_passes'] for row in result['primary'])
    assert all(not row['qualified'] for row in result['primary'])


def test_reference_pending_is_not_counted_as_failure_for_candidate_gain(p):
    rows = matrix(p, candidate=3, reference=4)
    for row in rows:
        if row['family'] in ('radiomap', 'supervised', 'frozen_fleet_jepa'):
            for container in (row['result']['packet_detail'], row['result']['packet_accounting']):
                container['pending_control_packet_age_slots'] = [0] * p['robots']
                container['pending_control_packets'] = p['robots']
    result = aggregate(rows, p, VALIDATION)
    for contrast in result['primary']:
        assert contrast['endpoint_sensitivity']['mean_difference'] == pytest.approx(-1 / 40)


def test_three_training_members_do_not_invent_independent_sites_or_rounding_gain(p):
    rows = matrix(p, candidate=0, reference=2)
    for row in rows:
        if row['family'] == p['candidate_family']:
            replace_count(row, [0, 0, 6][row['training_seed']])
    result = aggregate(rows, p, VALIDATION)
    assert result['independent_sites'] == 3
    for contrast in result['primary']:
        assert contrast['paired_site_differences'] == [0.] * 3
        assert contrast['descriptive_95_interval'] == [0., 0.]
        lo, hi = contrast['secondary_crossed_site_training_seed_95_interval']
        assert lo < 0 < hi
        assert not contrast['qualified']


@pytest.mark.parametrize('dimension', ['worst', 'silence', 'video', 'bulk'])
def test_mean_gain_cannot_hide_bad_service_tail_or_traffic(p, dimension):
    rows = matrix(p)
    for row in rows:
        if row['family'] != p['candidate_family']: continue
        detail, account = row['result']['packet_detail'], row['result']['packet_accounting']
        if dimension == 'worst':
            detail['misses'][0] = 20; detail['cumulative'][-1]['misses'][0] = 20
            row['result']['control_misses'] = sum(detail['misses'])
            row['result']['control_miss'] = sum(detail['misses']) / row['result']['control_sent']
        if dimension == 'silence':
            detail['max_gap'] = [20] * p['robots']; detail['cumulative'][-1]['max_gap'] = [20] * p['robots']
        if dimension == 'video': account.update(video_delivered_bits=800., video_pending_bits=200.)
        if dimension == 'bulk': account.update(bulk_delivered_bits=80., final_bulk_backlog_bits=20.)
    result = aggregate(rows, p, VALIDATION)
    assert all(row['statistical_pass'] for row in result['primary'])
    assert all(not row['engineering_pass'] for row in result['primary'])
    assert not result['headline_numerical_benefit_at_4s']


def test_zero_margin_sensitivity_remains_visible(p):
    rows = matrix(p)
    for row in rows:
        if row['family'] == p['candidate_family']:
            detail, account = row['result']['packet_detail'], row['result']['packet_accounting']
            detail['max_gap'] = [6] * p['robots']; detail['cumulative'][-1]['max_gap'] = [6] * p['robots']
            account.update(video_delivered_bits=895., video_pending_bits=105.,
                           bulk_delivered_bits=89.5, final_bulk_backlog_bits=10.5)
    result = aggregate(rows, p, VALIDATION)
    assert all(row['passes'] for row in result['engineering_gates'])
    assert all(not row['zero_margin_passes'] for row in result['engineering_gates'])
    assert result['headline_numerical_benefit_at_4s']


def test_raw_counter_validation_is_not_bypassed(p):
    rows = matrix(p)
    rows[0]['result']['control_misses'] += 1
    with pytest.raises(ValueError): aggregate(rows, p, VALIDATION)


def test_actual_candidate_name_retained_without_mutating_prior_study(p):
    from wirelessjev.semantic_jepa import report as prior
    original = prior.CANDIDATE
    result = aggregate(matrix(p), p, VALIDATION)
    assert prior.CANDIDATE == original == 'semantic_jepa'
    assert result['candidate_family'] == 'context_jepa'
    assert all(row['candidate_family'] == 'context_jepa' for row in result['primary'])
    assert 'semantic_jepa' not in {row['family'] for row in result['means']}
    alternate = dict(p, candidate_family='semantic_jepa')
    assert aggregate(matrix(alternate), alternate, VALIDATION)['candidate_family'] == 'semantic_jepa'
    assert prior.CANDIDATE == original


def test_cumulative_units_and_markdown_preserve_separate_claims(p):
    result = aggregate(matrix(p), p, VALIDATION)
    assert all(row['mean_difference'] == -20 for row in result['cumulative_excess'] if row['second'] == 2)
    text = markdown(result)
    assert 'context_jepa' in text
    assert 'Numerical benefit' in text and 'Original JEPA improvement' in text
    assert 'Objective-package advantage' in text
    assert 'not an EMA-necessity' in text
    assert '838' not in text
