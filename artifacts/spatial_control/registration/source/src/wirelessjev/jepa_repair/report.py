"""Prospective controller repair under an actual-cost completion contract.

No model is fitted or selected. Soft profile exceedances are diagnostics;
source, causality, paid completion/consumption and exact OFFLINE replay gate
experimental benefits. Older studies retain their original runtime vetoes.
"""
from fractions import Fraction
import math
import numpy as np
from ..semantic_jepa.report import metrics as raw_metrics

FAMILIES = ('radiomap', 'frozen_fleet_jepa')
COSTS = (.03, 4.)
MECHANICAL_FLAGS = ('source_and_archive_integrity', 'causality_and_kernel_checks', 'paid_cost_and_consumption_checks')
REPLAY_FLAG = 'exact_offline_replay'


def variants(cost):
    if cost == .03: return ('legacy', 'exact_only', 'repair')
    if cost == 4.: return ('legacy', 'repair')
    raise ValueError('Unregistered handover cost')


def arms(cost):
    return [(family, variant) for family in FAMILIES for variant in variants(cost)]


def _protocol(p):
    if tuple(p['families']) != FAMILIES or len(p['handover_seconds']) != 2 or set(p['handover_seconds']) != set(COSTS):
        raise ValueError('Both fixed forecast families and costs are required')
    if p['variants_by_cost'] != {str(cost): list(variants(cost)) for cost in COSTS}:
        raise ValueError('Controller variants differ from registration')
    if not math.isclose(p['primary_upper_quantile'], 1 - .05 / 6, rel_tol=0, abs_tol=1e-15):
        raise ValueError('Keep the full six-comparison multiplicity budget')
    for name in ('robots', 'duration_seconds', 'bootstrap_replicates', 'bootstrap_seed'):
        if isinstance(p[name], bool) or not isinstance(p[name], int) or p[name] < (0 if name == 'bootstrap_seed' else 1):
            raise ValueError('Invalid integer protocol field: ' + name)
    for stage in ('development', 'confirmation'):
        sites = p[stage + '_sites']
        if not sites or len(set(sites)) != len(sites) or any(type(site) is not int for site in sites):
            raise ValueError('Invalid scientific site membership')
    if set(p['development_sites']) & set(p['confirmation_sites']):
        raise ValueError('Development and confirmation sites overlap')
    for name in ('worst_decile_margin', 'silence_margin_seconds', 'video_margin', 'bulk_ratio', 'development_severe_regression_absolute'):
        value = p[name]
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid engineering margin: ' + name)
    if p['bulk_ratio'] > 1: raise ValueError('Bulk retention must be a fraction')
    checkpoints = p['cumulative_metrics_seconds']
    if not checkpoints or len(set(checkpoints)) != len(checkpoints) or any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 < t <= p['duration_seconds'] for t in checkpoints):
        raise ValueError('Invalid cumulative checkpoints')


def identity(row):
    if type(row['site']) is not int or type(row['handover_seconds']) not in (float, int) or not math.isfinite(row['handover_seconds']) or row['training_seed'] is not None:
        raise ValueError('Malformed identity or invented training replicate')
    return row['stage'], row['site'], float(row['handover_seconds']), row['family'], row['controller_variant']


def _metrics(row, p):
    identity(row)
    value = raw_metrics(row, p)
    # Reuse packet validation only. This NEW prospective contract pays actual
    # cost; the old required runtime boolean is a capacity diagnostic here.
    value['profile_exceeded'] = value.pop('runtime_overrun')
    value.update(controller_variant=row['controller_variant'],
        causality_pass=row.get('causality_pass') is True,
        paid_cost_and_consumption_pass=row.get('paid_cost_and_consumption_pass') is True,
        accounting_mode=row['accounting_mode'], replay=row['replay'], live_replication=row['live_replication'],
        predictor_profile_exceeded=row.get('predictor_runtime_overrun'))
    jobs, consumed = row['jobs'], row['forecast_consumption']
    value['timing'] = {'forecast_jobs': len(jobs), 'consumptions': len(consumed),
        'profile_exceeded_jobs': sum(j.get('runtime_overrun') is True for j in jobs),
        'charged_service_seconds_total': float(sum(j.get('charged_seconds', 0.) for j in jobs)),
        'charged_compute_seconds_total': float(sum(j.get('charged_compute_seconds', 0.) for j in jobs)),
        'actual_compute_seconds_total': float(sum(j.get('actual_compute_seconds', 0.) for j in jobs)),
        'maximum_forecast_age_seconds': max((e['age_seconds'] for e in consumed), default=None)}
    return value


def _costs(p, passed_costs):
    if passed_costs is None: return [float(cost) for cost in p['handover_seconds']]
    values = [float(cost) for cost in passed_costs]
    if len(values) != len(set(values)) or not set(values) <= set(p['handover_seconds']):
        raise ValueError('Invalid development-passed costs')
    return [float(cost) for cost in p['handover_seconds'] if cost in values]


def _index(rows, p, stage, costs, mode='main'):
    _protocol(p)
    sites = p[stage + '_sites'] if mode == 'main' else [p['confirmation_sites'][0]]
    expected = {(stage, site, float(cost), family, variant) for site in sites for cost in costs for family, variant in arms(cost)}
    data, original = {}, {}
    flags_for_mode = {'main': (False, False, 'actual_cost'), 'fresh_live_cost': (False, True, 'actual_cost'), 'recorded_schedule_offline': (True, False, 'recorded_schedule_offline')}
    if mode not in flags_for_mode: raise ValueError('Unknown accounting namespace')
    for row in rows:
        key = identity(row)
        if key in original or row['stage'] != stage: raise ValueError('Duplicate case or wrong stage')
        flags = row.get('replay'), row.get('live_replication'), row.get('accounting_mode')
        if type(row.get('replay')) is not bool or type(row.get('live_replication')) is not bool or flags != flags_for_mode[mode]:
            raise ValueError('Offline/live/main namespaces must remain separate')
        original[key] = row
    if set(original) != expected: raise ValueError('Incomplete or unregistered case matrix')
    # Randomized execution order must not change the derived development seal.
    for key in sorted(original): data[key] = _metrics(original[key], p)
    for site in sites:
        counts = {tuple(r['result']['packet_detail']['sent']) for k, r in original.items() if k[1] == site}
        if len(counts) != (1 if costs else 0): raise ValueError('Unmatched paired command generation')
    return data, sites


def _proof(data, validation, require_replay):
    validation = {} if validation is None else dict(validation)
    source = validation.get('source_and_archive_integrity') is True
    causal = validation.get('causality_and_kernel_checks') is True and all(r['causality_pass'] for r in data.values())
    paid = validation.get('paid_cost_and_consumption_checks') is True and all(r['paid_cost_and_consumption_pass'] for r in data.values())
    offline = validation.get(REPLAY_FLAG) is True if require_replay else None
    mechanical = bool(source and causal and paid)
    return {'source_and_archive_pass': source, 'causality_pass': causal, 'paid_cost_and_consumption_pass': paid,
        'exact_offline_replay_pass': offline, 'mechanical_integrity_pass': mechanical,
        'experimental_claim_integrity_pass': bool(mechanical and (offline if require_replay else True))}


def _values(data, stage, sites, cost, arm, field):
    return np.asarray([data[stage, site, float(cost), *arm][field] for site in sites], dtype=float)


def _rate(data, stage, sites, cost, arm):
    rows = [data[stage, site, float(cost), *arm] for site in sites]
    return Fraction(sum(r['control_misses'] for r in rows), sum(r['control_sent'] for r in rows))


def _means(data, p, stage, sites, costs):
    fields = ('command_loss', 'command_loss_pending_as_misses', 'worst_decile_loss', 'worst_decile_pending_as_misses', 'video_vs_original', 'bulk_mbps', 'bulk_backlog_bits', 'bulk_overflow_bits', 'pending_control_packets')
    result = []
    for cost in costs:
        for arm in arms(cost):
            rows = [data[stage, site, float(cost), *arm] for site in sites]
            result.append({'stage': stage, 'handover_seconds': float(cost), 'family': arm[0], 'controller_variant': arm[1],
                **{field: float(np.mean([r[field] for r in rows])) for field in fields},
                'global_max_silence_seconds': max(r['max_silence_seconds'] for r in rows),
                'profile_exceeded_cases': sum(r['profile_exceeded'] for r in rows),
                'causality_failed_cases': sum(not r['causality_pass'] for r in rows),
                'paid_cost_proof_failed_cases': sum(not r['paid_cost_and_consumption_pass'] for r in rows),
                'independent_sites': len(sites), 'new_training_members': 0})
    return result


def _timing(data):
    return {'profile_exceedance_is_diagnostic_not_a_veto': True,
        'profile_exceeded_cases': sum(r['profile_exceeded'] for r in data.values()),
        'predictor_profile_exceeded_cases': sum(r['predictor_profile_exceeded'] is True for r in data.values()),
        'paid_cost_proof_failed_cases': sum(not r['paid_cost_and_consumption_pass'] for r in data.values()),
        'causality_failed_cases': sum(not r['causality_pass'] for r in data.values()),
        'profile_exceeded_jobs': sum(r['timing']['profile_exceeded_jobs'] for r in data.values()),
        'jobs': sum(r['timing']['forecast_jobs'] for r in data.values()),
        'consumptions': sum(r['timing']['consumptions'] for r in data.values()),
        'maximum_forecast_age_seconds': max((r['timing']['maximum_forecast_age_seconds'] for r in data.values() if r['timing']['maximum_forecast_age_seconds'] is not None), default=None),
        'charged_compute_seconds_total': sum(r['timing']['charged_compute_seconds_total'] for r in data.values()),
        'charged_service_seconds_total': sum(r['timing']['charged_service_seconds_total'] for r in data.values()),
        'actual_compute_seconds_total': sum(r['timing']['actual_compute_seconds_total'] for r in data.values())}


def development_gate(rows, p, validation=None):
    """Mechanical/severe-regression veto only; never tuning or selection."""
    costs = _costs(p, None); data, sites = _index(rows, p, 'development', costs)
    proof = _proof(data, validation, require_replay=False)
    threshold = Fraction(str(p['development_severe_regression_absolute']))
    per_cost, passed = [], []
    for cost in costs:
        checks = []
        for family in FAMILIES:
            difference = _rate(data, 'development', sites, cost, (family, 'repair')) - _rate(data, 'development', sites, cost, (family, 'legacy'))
            checks.append({'family': family, 'repair_minus_legacy': float(difference), 'exact_difference_numerator': difference.numerator,
                'exact_difference_denominator': difference.denominator, 'severe_regression': difference > threshold})
        severe = any(r['severe_regression'] for r in checks)
        go = bool(proof['mechanical_integrity_pass'] and not severe)
        status = 'go' if go else 'development_mechanical_veto' if not proof['mechanical_integrity_pass'] else 'development_severe_regression_veto'
        per_cost.append({'handover_seconds': cost, 'status': status, 'family_checks': checks, 'passes': go, 'severe_regression': severe})
        if go: passed.append(cost)
    return {'stage': 'development', 'case_count': len(rows), 'independent_sites': len(sites), 'passed_costs': passed,
        'per_cost': per_cost, 'proof': proof, 'timing': _timing(data), 'means': _means(data, p, 'development', sites, costs),
        'every_case': list(data.values()), 'model_or_controller_selection_performed': False,
        'statistical_superiority_claimed': False, 'production_qualified': False, 'activated': False}


def primary_contrasts(cost):
    return [{'contrast_id': 'repair_minus_legacy:' + family, 'candidate': (family, 'repair'), 'reference': (family, 'legacy')}
            for family in FAMILIES] + [{'contrast_id': 'repaired_jepa_minus_repaired_radiomap',
            'candidate': ('frozen_fleet_jepa', 'repair'), 'reference': ('radiomap', 'repair')}]


def _bootstrap(difference, indices, upper):
    difference = np.asarray(difference, dtype=float)
    if difference.shape != (indices.shape[1],) or not np.isfinite(difference).all():
        raise ValueError('Invalid paired whole-site differences')
    draws = difference[indices].mean(1)
    return {'mean_difference': float(difference.mean()), 'paired_site_differences': difference.tolist(),
        'descriptive_95_interval': np.quantile(draws, [.025, .975]).tolist(),
        'upper_quantile': upper, 'multiplicity_upper': float(np.quantile(draws, upper))}


def _engineering(data, p, stage, sites, cost, contrast):
    candidate, reference = contrast['candidate'], contrast['reference']
    mean = lambda arm, field: float(_values(data, stage, sites, cost, arm, field).mean())
    maximum = lambda arm: float(max(_values(data, stage, sites, cost, arm, 'max_silence_seconds')))
    v = {'candidate_worst_decile_pending_as_misses': mean(candidate, 'worst_decile_pending_as_misses'),
        'candidate_worst_decile_observed': mean(candidate, 'worst_decile_loss'),
        'reference_worst_decile_observed': mean(reference, 'worst_decile_loss'),
        'candidate_max_silence_seconds': maximum(candidate), 'reference_max_silence_seconds': maximum(reference),
        'candidate_video_vs_original': mean(candidate, 'video_vs_original'), 'reference_video_vs_original': mean(reference, 'video_vs_original'),
        'candidate_bulk_mbps': mean(candidate, 'bulk_mbps'), 'reference_bulk_mbps': mean(reference, 'bulk_mbps')}
    zero = {'worst_decile': v['candidate_worst_decile_pending_as_misses'] <= v['reference_worst_decile_observed'],
        'maximum_silence': v['candidate_max_silence_seconds'] <= v['reference_max_silence_seconds'],
        'original_video': v['candidate_video_vs_original'] >= v['reference_video_vs_original'],
        'bulk_delivery': v['candidate_bulk_mbps'] >= v['reference_bulk_mbps']}
    checks = {'worst_decile': v['candidate_worst_decile_pending_as_misses'] <= v['reference_worst_decile_observed'] + p['worst_decile_margin'],
        'maximum_silence': v['candidate_max_silence_seconds'] <= v['reference_max_silence_seconds'] + p['silence_margin_seconds'],
        'original_video': v['candidate_video_vs_original'] >= v['reference_video_vs_original'] - p['video_margin'],
        'bulk_delivery': v['candidate_bulk_mbps'] >= p['bulk_ratio'] * v['reference_bulk_mbps']}
    return {'contrast_id': contrast['contrast_id'], 'handover_seconds': float(cost), 'status': 'tested',
        'candidate_family': candidate[0], 'candidate_variant': candidate[1], 'reference_family': reference[0], 'reference_variant': reference[1],
        'checks': checks, 'passes': bool(all(checks.values())), 'zero_margin_checks': zero,
        'zero_margin_passes': bool(all(zero.values())), **v}


def _contrast(data, p, sites, cost, contrast, indices, primary):
    candidate, reference = contrast['candidate'], contrast['reference']
    observed = _values(data, 'confirmation', sites, cost, candidate, 'command_loss') - _values(data, 'confirmation', sites, cost, reference, 'command_loss')
    endpoint = _values(data, 'confirmation', sites, cost, candidate, 'command_loss_pending_as_misses') - _values(data, 'confirmation', sites, cost, reference, 'command_loss')
    upper = p['primary_upper_quantile'] if primary else .975
    value = {'contrast_id': contrast['contrast_id'], 'handover_seconds': float(cost), 'status': 'tested',
        'candidate_family': candidate[0], 'candidate_variant': candidate[1], 'reference_family': reference[0], 'reference_variant': reference[1],
        **_bootstrap(observed, indices, upper), 'endpoint_sensitivity': _bootstrap(endpoint, indices, upper)}
    value['observed_passes'] = value['multiplicity_upper'] < 0
    value['endpoint_passes'] = value['endpoint_sensitivity']['multiplicity_upper'] < 0
    value['statistical_pass'] = bool(value['observed_passes'] and value['endpoint_passes']) if primary else None
    value['statistical_superiority_claimed'] = False if not primary else None
    return value


def _untested(cost, contrast):
    candidate, reference = contrast['candidate'], contrast['reference']
    return {'contrast_id': contrast['contrast_id'], 'handover_seconds': float(cost), 'status': 'not_run_development_veto',
        'candidate_family': candidate[0], 'candidate_variant': candidate[1], 'reference_family': reference[0], 'reference_variant': reference[1],
        'mean_difference': None, 'paired_site_differences': None, 'descriptive_95_interval': None,
        'upper_quantile': None, 'multiplicity_upper': None, 'endpoint_sensitivity': None,
        'observed_passes': False, 'endpoint_passes': False, 'statistical_pass': False,
        'engineering_pass': False, 'experimental_benefit': False}


def aggregate(rows, p, validation=None, passed_costs=None):
    """Confirm fixed contrasts; capacity exceedance alone is never a veto."""
    costs = _costs(p, passed_costs); data, sites = _index(rows, p, 'confirmation', costs)
    proof = _proof(data, validation, require_replay=True)
    rng = np.random.default_rng(p['bootstrap_seed'])
    indices = rng.integers(0, len(sites), size=(p['bootstrap_replicates'], len(sites)))
    primary, engineering, cumulative, secondary, per_cost = [], [], [], [], []
    for cost in p['handover_seconds']:
        records = []
        for contrast in primary_contrasts(cost):
            if cost not in costs:
                value = _untested(cost, contrast); primary.append(value); records.append(value)
                continue
            value = _contrast(data, p, sites, cost, contrast, indices, primary=True)
            gate = _engineering(data, p, 'confirmation', sites, cost, contrast)
            value['engineering_pass'] = gate['passes']
            value['experimental_benefit'] = bool(value['statistical_pass'] and gate['passes'] and proof['experimental_claim_integrity_pass'])
            engineering.append(gate); primary.append(value); records.append(value)
            for second in p['cumulative_metrics_seconds']:
                def missed(site, arm):
                    return next(r['control_misses'] for r in data['confirmation', site, float(cost), *arm]['cumulative'] if r['slot_end'] == second * 100)
                difference = [missed(site, contrast['candidate']) - missed(site, contrast['reference']) for site in sites]
                cumulative.append({'contrast_id': contrast['contrast_id'], 'handover_seconds': float(cost),
                    'second': second, 'units': 'fleet misses per whole site', **_bootstrap(difference, indices, .975)})
        family_benefits = {r['candidate_family']: r['experimental_benefit'] for r in records if r['contrast_id'].startswith('repair_minus_legacy:')}
        model_benefit = next(r['experimental_benefit'] for r in records if r['contrast_id'] == 'repaired_jepa_minus_repaired_radiomap')
        per_cost.append({'handover_seconds': float(cost), 'status': 'tested' if cost in costs else 'not_run_development_veto',
            'repair_benefit_by_family': family_benefits, 'repaired_jepa_vs_radiomap_benefit': model_benefit,
            'all_primary_experimental_benefits': bool(all(r['experimental_benefit'] for r in records))})
    if .03 in costs:
        for family in FAMILIES:
            for label, candidate, reference in [('fractional_integral', (family, 'exact_only'), (family, 'legacy')),
                    ('recovery_policy_package', (family, 'repair'), (family, 'exact_only'))]:
                contrast = {'contrast_id': 'secondary_' + label + ':' + family, 'candidate': candidate, 'reference': reference}
                secondary.append(_contrast(data, p, sites, .03, contrast, indices, primary=False))
    return {'schema': 1, 'study': p['study'], 'stage': 'confirmation', 'case_count': len(rows),
        'sites': sites if costs else [], 'registered_sites': p['confirmation_sites'], 'independent_sites': len(sites) if costs else 0,
        'passed_development_costs': costs, 'primary_multiplicity_budget': 6, 'primary_upper_quantile': p['primary_upper_quantile'],
        'bootstrap_unit': 'whole site; fixed original JEPA ensemble is one forecast treatment',
        'bootstrap_replicates': p['bootstrap_replicates'], 'bootstrap_seed': p['bootstrap_seed'],
        'primary': primary, 'per_cost': per_cost, 'engineering_gates': engineering, 'secondary_fast_contrasts': secondary,
        'cumulative_excess': cumulative, 'means': _means(data, p, 'confirmation', sites, costs) if costs else [],
        'every_case': list(data.values()), 'timing': _timing(data), 'proof': proof,
        'integrity_validation': {} if validation is None else dict(validation),
        'all_six_experimental_benefits': bool(all(r['experimental_benefit'] for r in primary)), 'new_fits': 0,
        'ema_necessity_claimed': False, 'hardware_qualified': False, 'production_qualified': False, 'activated': False,
        'limits': ['New prospective actual-cost contract; older overrun vetoes and results remain unchanged.',
            'Controller-package effects do not establish a JEPA training-objective or architecture advantage.',
            'Offline fixed-schedule replay is not proof of freshly measured deadlines.',
            'Fresh live-cost repetitions are sensitivity observations, not extra independent sites.',
            'Shared procedural CAD, assumed sensing/public routes and fixed weights limit transfer.',
            'Admission/scorer CPU is not separately scheduled; no complete-stack real-time guarantee.',
            'Empirical endpoint/tail margins and finite silence do not establish production or indefinite safety.']}


def replication_summary(rows, p, mode, passed_costs=None, original_rows=None):
    """Keep offline/fresh first-site repetitions outside inferential units."""
    if mode not in ('fresh_live_cost', 'recorded_schedule_offline'): raise ValueError('Use an explicit replication namespace')
    costs = _costs(p, passed_costs); data, sites = _index(rows, p, 'confirmation', costs, mode=mode)
    original = {} if original_rows is None else {identity(r): r for r in original_rows}
    if original_rows is not None and len(original) != len(original_rows):
        raise ValueError('Duplicate original main cases')
    differences = []
    for row in sorted(rows, key=identity):
        key = identity(row)
        if original_rows is not None:
            if key not in original: raise ValueError('Replication has no original main case')
            before, after = _metrics(original[key], p), data[key]
            differences.append({'case': list(key), 'command_loss_difference': after['command_loss'] - before['command_loss'],
                'max_silence_difference_seconds': after['max_silence_seconds'] - before['max_silence_seconds'],
                'packet_result_exactly_matches_original': row['result'] == original[key]['result']})
    return {'mode': mode, 'case_count': len(rows), 'source_site': sites[0] if costs else None,
        'independent_sites_added': 0, 'statistical_superiority_claimed': False, 'fresh_measured_deadline_claim': False,
        'timing': _timing(data), 'every_case': list(data.values()), 'differences_from_main': differences,
        'note': 'Offline uses original paid schedules; fresh repeats may differ legitimately with measured costs.'}


summarize_replications = replication_summary


def markdown(report):
    lines = ['# Prospective controller and actual-cost runtime repair', '',
        f"{report['case_count']} confirmation cases on {report['independent_sites']} whole sites. No training or objective selection occurred.", '',
        '| Handover s | Contrast | Status | Mean loss difference | Statistical | Engineering | Experimental benefit |',
        '|---|---|---|---:|---|---|---|']
    for row in report['primary']:
        difference = 'not measured' if row['mean_difference'] is None else f"{row['mean_difference']:.7f}"
        lines.append(f"| {row['handover_seconds']:g} | {row['contrast_id']} | {row['status']} | {difference} | {row['statistical_pass']} | {row['engineering_pass']} | {row['experimental_benefit']} |")
    lines += ['', 'All six comparisons retain their multiplicity budget. Observed and conservative pending-command bounds must both show superiority. Exact-only fast arms are secondary diagnostics, never a post-test alternative candidate.', '',
        '| Handover s | Family | Controller | Command loss | Worst decile | Max silence s | Video/original | Bulk Mbps |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for row in report['means']:
        lines.append(f"| {row['handover_seconds']:g} | {row['family']} | {row['controller_variant']} | {row['command_loss']:.7f} | {row['worst_decile_loss']:.7f} | {row['global_max_silence_seconds']:.2f} | {row['video_vs_original']:.5f} | {row['bulk_mbps']:.5f} |")
    lines += ['', f"Source/causality/paid-cost/offline replay checks: {report['proof']['experimental_claim_integrity_pass']}. Profile-exceeded cases: {report['timing']['profile_exceeded_cases']} (diagnostic, not an older-study waiver).", '',
        'Zero-margin engineering checks, cumulative losses, every case and secondary fast contrasts remain in the machine-readable report. Repetitions add no independent sites.', '']
    lines += ['- ' + text for text in report['limits']]
    return '\n'.join(lines) + '\n'
