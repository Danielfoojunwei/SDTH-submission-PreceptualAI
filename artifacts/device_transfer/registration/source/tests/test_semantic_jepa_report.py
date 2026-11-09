"""Arithmetic and scientific-gate tests; all inputs are explicit fixtures."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from wirelessjev.semantic_jepa.report import (
    aggregate_confirmation, arms, build_report, development_gate, markdown, metrics,
)


@pytest.fixture
def p():
    value = json.loads((Path(__file__).resolve().parents[1] / 'artifacts/semantic_jepa/protocol.json').read_text())
    value.update(robots=10, duration_seconds=2, development_sites=[11, 12],
                 confirmation_sites=[21, 22, 23], cumulative_metrics_seconds=[1, 2],
                 bootstrap_replicates=500, worst_decile_margin=1 / 40)
    return value


VALIDATION = dict(source_and_archive_integrity=True, causality_and_kernel_checks=True, exact_replay=True)


def row(p, *, stage='confirmation', site=21, cost=.03, family='semantic_jepa', seed=0,
        miss=2, pending=False, gap=2, video=900., bulk=90., overrun=False):
    n, total = p['robots'], 20 * p['duration_seconds']
    misses = [miss] * n
    sent = [total] * n
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
    return {'stage': stage, 'site': site, 'handover_seconds': cost, 'family': family,
        'training_seed': seed, 'runtime_overrun': overrun, 'forecast_metrics': {'rmse_db': 3.},
        'result': {'control_misses': miss * n, 'control_sent': total * n,
                   'control_miss': miss / total, 'packet_detail': detail, 'packet_accounting': account}}


def matrix(p, *, stage='confirmation', costs=None, candidate=2, reference=4):
    if costs is None:
        costs = p['handover_seconds']
    return [row(p, stage=stage, site=site, cost=cost, family=family, seed=seed,
                miss=candidate if family == 'semantic_jepa' else reference)
            for site in p[stage + '_sites'] for cost in costs for family, seed in arms(p)]


def replace_count(target, miss):
    n = len(target['result']['packet_detail']['misses'])
    target['result']['packet_detail']['misses'] = [miss] * n
    target['result']['packet_detail']['cumulative'][0]['misses'] = [miss // 2] * n
    target['result']['packet_detail']['cumulative'][-1]['misses'] = [miss] * n
    target['result']['control_misses'] = miss * n
    target['result']['control_miss'] = miss / target['result']['packet_detail']['sent'][0]


def test_raw_metrics_and_pending_bound(p):
    result = metrics(row(p, miss=3, pending=True, gap=17), p)
    assert result['control_misses'] == 30
    assert result['control_sent'] == 400
    assert result['command_loss'] == pytest.approx(.075)
    assert result['command_loss_pending_as_misses'] == pytest.approx(.1)
    assert result['worst_decile_pending_as_misses'] == pytest.approx(.1)
    assert result['max_silence_seconds'] == .17
    assert result['video_vs_original'] == .9
    assert result['bulk_mbps'] == 45e-6
    assert result['cumulative'][-1]['control_misses'] == 30


@pytest.mark.parametrize('kind', ['global', 'fraction', 'negative', 'fractional', 'generation',
                                 'pending_age', 'pending_total', 'pending_export', 'gap'])
def test_invalid_exact_counters_are_rejected(p, kind):
    value = row(p)
    result = value['result']; detail = result['packet_detail']
    if kind == 'global': result['control_misses'] += 1
    if kind == 'fraction': result['control_miss'] += .01
    if kind == 'negative': detail['misses'][0] = -1
    if kind == 'fractional': detail['misses'][0] = 1.5
    if kind == 'generation': detail['sent'][0] += 1
    if kind == 'pending_age': detail['pending_control_packet_age_slots'][0] = 5
    if kind == 'pending_total': detail['pending_control_packets'] += 1
    if kind == 'pending_export': result['packet_accounting']['pending_control_packet_age_slots'][0] = 0
    if kind == 'gap': detail['max_gap'][0] = 201
    with pytest.raises(ValueError): metrics(value, p)


@pytest.mark.parametrize('kind', ['video', 'bulk', 'nonfinite', 'negative', 'reported'])
def test_mass_balance_and_finite_accounting(p, kind):
    value = row(p); account = value['result']['packet_accounting']
    if kind == 'video': account['video_pending_bits'] += 1
    if kind == 'bulk': account['final_bulk_backlog_bits'] += 1
    if kind == 'nonfinite': account['video_delivered_bits'] = float('nan')
    if kind == 'negative': account['bulk_delivered_bits'] = -1
    if kind == 'reported': account['video_flow_conservation_error_bits'] = 100
    with pytest.raises(ValueError): metrics(value, p)


@pytest.mark.parametrize('kind', ['duplicate', 'missing_final', 'wrong_final', 'nonmonotone', 'missing_checkpoint'])
def test_cumulative_trace_validation(p, kind):
    value = row(p, miss=6); trace = value['result']['packet_detail']['cumulative']
    if kind == 'duplicate': trace.insert(1, deepcopy(trace[0]))
    if kind == 'missing_final': trace.pop()
    if kind == 'wrong_final': trace[-1]['misses'][0] += 1
    if kind == 'nonmonotone': trace[0]['misses'][0] = 7
    if kind == 'missing_checkpoint': trace.pop(0)
    with pytest.raises(ValueError): metrics(value, p)


def test_complete_positive_confirmation_uses_whole_sites(p):
    result = aggregate_confirmation(matrix(p), p, p['handover_seconds'], VALIDATION)
    assert result['case_count'] == 3 * 2 * 14
    assert result['independent_sites'] == 3
    assert result['primary_multiplicity_budget'] == 8
    assert len(result['primary']) == 8
    assert len(result['engineering_gates']) == 8
    assert len(result['cumulative_excess']) == 16
    assert all(len(r['paired_site_differences']) == 3 for r in result['primary'])
    assert result['both_costs_pass'] is True
    assert result['production_qualified'] is False
    assert result['activated'] is False
    for comparison in result['primary']:
        assert comparison['mean_difference'] == pytest.approx(-.05)
        assert comparison['multiplicity_upper'] == pytest.approx(-.05)
        assert comparison['secondary_crossed_site_training_seed_95_interval'] == pytest.approx([-.05, -.05])


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'foreign_site', 'foreign_seed', 'wrong_stage', 'extra_cost'])
def test_exact_matrix_membership(p, kind):
    rows = matrix(p)
    if kind == 'missing': rows.pop()
    if kind == 'duplicate': rows.append(deepcopy(rows[-1]))
    if kind == 'foreign_site': rows[0]['site'] = 999
    if kind == 'foreign_seed': rows[0]['training_seed'] = 99
    if kind == 'wrong_stage': rows[0]['stage'] = 'development'
    if kind == 'extra_cost': rows[0]['handover_seconds'] = 1.
    with pytest.raises(ValueError): aggregate_confirmation(rows, p, p['handover_seconds'], VALIDATION)


@pytest.mark.parametrize('validation', [None, {}, {'exact_replay': True},
    {**VALIDATION, 'exact_replay': False}, {**VALIDATION, 'source_and_archive_integrity': 1},
    {**VALIDATION, 'development_runtime_overrun': True}])
def test_integrity_and_development_runtime_fail_closed(p, validation):
    result = aggregate_confirmation(matrix(p), p, p['handover_seconds'], validation)
    assert not result['both_costs_pass']
    assert all(not item['passes'] for item in result['per_cost'])


def test_pending_endpoint_erases_observed_one_command_gain(p):
    rows = matrix(p, candidate=3, reference=4)
    for value in rows:
        if value['family'] == 'semantic_jepa':
            for container in (value['result']['packet_detail'], value['result']['packet_accounting']):
                container['pending_control_packet_age_slots'] = [0] * p['robots']
                container['pending_control_packets'] = p['robots']
    result = aggregate_confirmation(rows, p, p['handover_seconds'], VALIDATION)
    assert all(value['observed_passes'] for value in result['primary'])
    assert all(not value['endpoint_passes'] for value in result['primary'])
    assert not result['both_costs_pass']


def test_runtime_overrun_vetoes_both_costs(p):
    rows = matrix(p)
    rows[0]['runtime_overrun'] = True
    result = aggregate_confirmation(rows, p, p['handover_seconds'], VALIDATION)
    assert result['any_runtime_overrun'] is True
    assert all(not value['passes'] for value in result['per_cost'])


def test_cost_subset_retains_full_multiplicity_and_unrun_status(p):
    result = aggregate_confirmation(matrix(p, costs=[4.]), p, [4.], VALIDATION)
    assert result['primary_multiplicity_budget'] == 8
    assert len(result['primary']) == 4
    assert all(value['upper_quantile'] == 1 - .05 / 8 for value in result['primary'])
    assert result['per_cost'][0]['status'] == 'not_run_development_veto'
    assert result['per_cost'][1]['passes']
    assert not result['both_costs_pass']


def test_no_confirmation_is_explicit_and_cannot_pass(p):
    result = aggregate_confirmation([], p, [], VALIDATION)
    assert result['case_count'] == 0
    assert result['primary'] == []
    assert not result['both_costs_pass']
    assert all(value['status'] == 'not_run_development_veto' for value in result['per_cost'])
    with pytest.raises(ValueError): aggregate_confirmation(matrix(p), p, [], VALIDATION)


def test_seed_average_not_independent_replication(p):
    rows = matrix(p, reference=8)
    for value in rows:
        if value['family'] == 'semantic_jepa': replace_count(value, [0, 0, 12][value['training_seed']])
    result = aggregate_confirmation(rows, p, p['handover_seconds'], VALIDATION)
    contrast = result['primary'][0]
    assert contrast['paired_site_differences'] == pytest.approx([-.1] * 3)
    assert contrast['descriptive_95_interval'] == pytest.approx([-.1, -.1])
    lo, hi = contrast['secondary_crossed_site_training_seed_95_interval']
    assert lo < -.1 < hi
    assert len([r for r in result['per_seed_means'] if r['family'] == 'semantic_jepa']) == 6


def test_engineering_margins_and_zero_margin_report(p):
    rows = matrix(p)
    for value in rows:
        if value['family'] == 'semantic_jepa':
            detail = value['result']['packet_detail']
            detail['max_gap'] = [6] * p['robots']
            detail['cumulative'][-1]['max_gap'] = [6] * p['robots']
            account = value['result']['packet_accounting']
            account['video_delivered_bits'] = 895.
            account['video_pending_bits'] = 105.
            account['bulk_delivered_bits'] = 89.5
            account['final_bulk_backlog_bits'] = 10.5
    result = aggregate_confirmation(rows, p, p['handover_seconds'], VALIDATION)
    assert all(value['passes'] for value in result['engineering_gates'])
    assert all(not value['zero_margin_passes'] for value in result['engineering_gates'])
    assert all(not value['zero_margin_checks']['maximum_silence'] for value in result['engineering_gates'])


@pytest.mark.parametrize('change', ['silence', 'worst', 'video', 'bulk'])
def test_one_bad_engineering_dimension_blocks_qualification(p, change):
    rows = matrix(p)
    for value in rows:
        if value['family'] != 'semantic_jepa': continue
        detail, account = value['result']['packet_detail'], value['result']['packet_accounting']
        if change == 'silence':
            detail['max_gap'] = [20] * p['robots']; detail['cumulative'][-1]['max_gap'] = [20] * p['robots']
        if change == 'worst':
            detail['misses'][0] = 20; detail['cumulative'][-1]['misses'][0] = 20
            value['result']['control_misses'] = sum(detail['misses'])
            value['result']['control_miss'] = sum(detail['misses']) / value['result']['control_sent']
        if change == 'video': account.update(video_delivered_bits=800., video_pending_bits=200.)
        if change == 'bulk': account.update(bulk_delivered_bits=80., final_bulk_backlog_bits=20.)
    result = aggregate_confirmation(rows, p, p['handover_seconds'], VALIDATION)
    assert not result['both_costs_pass']
    assert all(not value['passes'] for value in result['engineering_gates'])


def test_development_strict_vs_nonstrict_rules(p):
    rows = matrix(p, stage='development', candidate=2, reference=3)
    for value in rows:
        if value['family'] in ('context_jepa', 'semantic_aux'): replace_count(value, 2)
    result = development_gate(rows, p)
    assert result['passed_costs'] == p['handover_seconds']
    assert not result['statistical_superiority_claimed']
    for value in rows:
        if value['family'] == 'supervised': replace_count(value, 2)
    rejected = development_gate(rows, p)
    assert rejected['passed_costs'] == []


@pytest.mark.parametrize('veto', ['integrity', 'runtime'])
def test_development_fail_closed(p, veto):
    rows = matrix(p, stage='development')
    if veto == 'runtime': rows[-1]['runtime_overrun'] = True
    result = development_gate(rows, p, integrity=veto != 'integrity')
    assert result['passed_costs'] == []


def test_cumulative_uses_fleet_counts_not_rates(p):
    result = aggregate_confirmation(matrix(p), p, p['handover_seconds'], VALIDATION)
    at_end = [r for r in result['cumulative_excess'] if r['second'] == 2]
    assert all(value['mean_difference'] == -20 for value in at_end)
    assert all('fleet misses' in value['units'] for value in at_end)


def test_fixed_registration_and_finite_runtime_validation(p):
    bad = deepcopy(p); bad['primary_upper_quantile'] = .95
    with pytest.raises(ValueError): aggregate_confirmation(matrix(p), bad, p['handover_seconds'], VALIDATION)
    value = row(p); value['runtime_overrun'] = 0
    with pytest.raises(ValueError): metrics(value, p)
    value = row(p); value['handover_seconds'] = float('nan')
    with pytest.raises(ValueError): metrics(value, p)


def test_end_to_end_pure_report_and_markdown_retain_all_outcomes(p):
    report = build_report(matrix(p, stage='development'), matrix(p), p, VALIDATION)
    text = markdown(report)
    assert report['confirmation']['both_costs_pass']
    assert text.count('## ') == 2
    assert 'radiomap' in text and 'semantic_aux' in text
    assert 'no physical or production qualification' in text
    rejected = build_report(matrix(p, stage='development', candidate=4, reference=4), [], p, VALIDATION)
    assert 'not_run_development_veto' in markdown(rejected)
    assert not rejected['confirmation']['both_costs_pass']


def test_rounding_three_members_cannot_create_false_development_gain(p):
    rows = matrix(p, stage='development', candidate=0, reference=2)
    for value in rows:
        if value['family'] == 'semantic_jepa':
            replace_count(value, [0, 0, 6][value['training_seed']])
    development = development_gate(rows, p)
    assert development['passed_costs'] == []
    assert all(not c['passes'] for cost in development['by_cost']
               for c in cost['comparisons'] if c['requires_strict_improvement'])
    rows = matrix(p, candidate=0, reference=2)
    for value in rows:
        if value['family'] == 'semantic_jepa':
            replace_count(value, [0, 0, 6][value['training_seed']])
    confirmation = aggregate_confirmation(rows, p, p['handover_seconds'], VALIDATION)
    assert all(c['paired_site_differences'] == [0.] * 3 for c in confirmation['primary'])
    assert not confirmation['both_costs_pass']


def test_metrics_accept_actual_instrumented_native_exports(p):
    from wirelessjev.native.sim import policies
    from wirelessjev.native.world import build_world
    from wirelessjev.semantic_jepa.sim import simulate
    world = build_world(12345, duration_s=2, n_robots=p['robots'], sigma_map=4.)
    actual = simulate(world, {policy.name: policy for policy in policies()}['native'])
    value = row(p)
    value['result'] = actual
    checked = metrics(value, p)
    assert checked['control_misses'] == actual['control_misses']
    assert checked['command_loss'] == actual['control_miss']
    assert checked['cumulative'][-1]['control_sent'] == actual['control_sent']
    assert abs(checked['video_flow_error_bits']) < 1e-5
