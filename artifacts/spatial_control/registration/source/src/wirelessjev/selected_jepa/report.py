"""Prospective confirmation of a JEPA selected on a completed prior study.

The candidate retains its actual family name. Each comparison has a distinct
claim: numerical baseline benefit, matched objective-package advantage, or
improvement over the old frozen JEPA. No global from an older study is changed.
"""
from __future__ import annotations

import math
import numpy as np

from ..semantic_jepa.report import identity, metrics

BASELINES = ('radiomap', 'frozen_fleet_jepa')
REFERENCES = ('radiomap', 'supervised', 'frozen_fleet_jepa')
INTEGRITY_FLAGS = ('source_and_archive_integrity', 'causality_and_kernel_checks', 'exact_replay')
CLAIM_FOR_REFERENCE = {
    'radiomap': 'numerical_benefit',
    'supervised': 'objective_advantage',
    'frozen_fleet_jepa': 'original_jepa_improvement',
}


def _protocol(p):
    candidate = p['candidate_family']
    if candidate not in ('context_jepa', 'semantic_jepa'):
        raise ValueError('Candidate must retain a registered JEPA family identity')
    refs = p['primary_references']
    if len(refs) != 3 or set(refs) != set(REFERENCES):
        raise ValueError('All three registered references are required')
    costs = p['handover_seconds']
    if len(costs) != 2 or set(costs) != {.03, 4.}:
        raise ValueError('Both registered costs are required; no sensitivity suppression')
    seeds = p['training_seeds']
    if (not seeds or len(set(seeds)) != len(seeds) or
            any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)):
        raise ValueError('Invalid paired training seeds')
    sites = p['confirmation_sites']
    if (len(sites) < 2 or len(set(sites)) != len(sites) or
            any(isinstance(site, bool) or not isinstance(site, int) for site in sites)):
        raise ValueError('Invalid independent confirmation sites')
    q = 1 - .05 / 6
    if not math.isclose(p['primary_upper_quantile'], q, abs_tol=1e-15, rel_tol=0):
        raise ValueError('The full six-comparison multiplicity budget must be retained')
    for name in ('robots', 'duration_seconds', 'bootstrap_replicates', 'bootstrap_seed'):
        if isinstance(p[name], bool) or not isinstance(p[name], int) or p[name] < (0 if name == 'bootstrap_seed' else 1):
            raise ValueError('Invalid integer protocol field: ' + name)
    for name in ('worst_decile_margin', 'silence_margin_seconds', 'video_margin', 'bulk_ratio'):
        value = p[name]
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid empirical engineering margin: ' + name)
    if p['bulk_ratio'] > 1:
        raise ValueError('Bulk margin must be a retained fraction')
    checkpoints = p['cumulative_metrics_seconds']
    if (not checkpoints or len(set(checkpoints)) != len(checkpoints) or
            any(isinstance(t, bool) or not isinstance(t, (int, float)) or
                not math.isfinite(t) or not 0 < t <= p['duration_seconds'] for t in checkpoints)):
        raise ValueError('Invalid cumulative checkpoints')
    return candidate


def arms(p):
    candidate = p['candidate_family']
    return [(family, seed) for family in (candidate, 'supervised') for seed in p['training_seeds']] + [
        (family, None) for family in BASELINES]


def _seeds(p, family):
    return [None] if family in BASELINES else p['training_seeds']


def _index(rows, p):
    _protocol(p)
    expected = {(site, float(cost), family, seed) for site in p['confirmation_sites']
                for cost in p['handover_seconds'] for family, seed in arms(p)}
    original, indexed = {}, {}
    for row in rows:
        key = identity(row)
        if row['stage'] != 'confirmation' or key in original:
            raise ValueError('Duplicate case or wrong stage in prospective confirmation')
        original[key] = row
    if set(original) != expected:
        raise ValueError('Confirmation matrix is incomplete or outside registered membership')
    for key, row in original.items():
        indexed[key] = metrics(row, p)
    for site in p['confirmation_sites']:
        generated = {tuple(row['result']['packet_detail']['sent'])
                     for key, row in original.items() if key[0] == site}
        if len(generated) != 1:
            raise ValueError('Paired methods have unmatched command generation')
    return original, indexed


def _site_values(data, p, cost, family, field):
    values = []
    for site in p['confirmation_sites']:
        rows = [data[site, float(cost), family, seed] for seed in _seeds(p, family)]
        if field in ('command_loss', 'command_loss_pending_as_misses'):
            numerator = sum(row['control_misses'] for row in rows)
            if field == 'command_loss_pending_as_misses':
                numerator += sum(row['pending_control_packets'] for row in rows)
            value = numerator / sum(row['control_sent'] for row in rows)
        else:
            value = float(np.mean([row[field] for row in rows]))
        values.append(value)
    return np.asarray(values, dtype=float)


def _bootstrap(values, indices, upper_quantile):
    values = np.asarray(values, dtype=float)
    if values.shape != (indices.shape[1],) or not np.isfinite(values).all():
        raise ValueError('Invalid independent-site contrast')
    draws = values[indices].mean(1)
    return {'mean_difference': float(values.mean()),
            'descriptive_95_interval': np.quantile(draws, [.025, .975]).tolist(),
            'upper_quantile': upper_quantile,
            'multiplicity_upper': float(np.quantile(draws, upper_quantile)),
            'paired_site_differences': values.tolist()}


def _engineering(data, p, cost, reference):
    candidate = p['candidate_family']
    mean = lambda family, field: float(_site_values(data, p, cost, family, field).mean())
    maximum = lambda family: max(data[site, float(cost), family, seed]['max_silence_seconds']
        for site in p['confirmation_sites'] for seed in _seeds(p, family))
    values = {
        'candidate_mean_worst_decile_pending_as_misses': mean(candidate, 'worst_decile_pending_as_misses'),
        'candidate_mean_worst_decile_observed': mean(candidate, 'worst_decile_loss'),
        'reference_mean_worst_decile_observed': mean(reference, 'worst_decile_loss'),
        'candidate_max_silence_seconds': maximum(candidate),
        'reference_max_silence_seconds': maximum(reference),
        'candidate_mean_video_vs_original': mean(candidate, 'video_vs_original'),
        'reference_mean_video_vs_original': mean(reference, 'video_vs_original'),
        'candidate_mean_bulk_mbps': mean(candidate, 'bulk_mbps'),
        'reference_mean_bulk_mbps': mean(reference, 'bulk_mbps')}
    zero = {
        'worst_decile': values['candidate_mean_worst_decile_pending_as_misses'] <= values['reference_mean_worst_decile_observed'],
        'maximum_silence': values['candidate_max_silence_seconds'] <= values['reference_max_silence_seconds'],
        'original_video': values['candidate_mean_video_vs_original'] >= values['reference_mean_video_vs_original'],
        'bulk_delivery': values['candidate_mean_bulk_mbps'] >= values['reference_mean_bulk_mbps']}
    checks = {
        'worst_decile': values['candidate_mean_worst_decile_pending_as_misses'] <= values['reference_mean_worst_decile_observed'] + p['worst_decile_margin'],
        'maximum_silence': values['candidate_max_silence_seconds'] <= values['reference_max_silence_seconds'] + p['silence_margin_seconds'],
        'original_video': values['candidate_mean_video_vs_original'] >= values['reference_mean_video_vs_original'] - p['video_margin'],
        'bulk_delivery': values['candidate_mean_bulk_mbps'] >= p['bulk_ratio'] * values['reference_mean_bulk_mbps']}
    return {'handover_seconds': float(cost), 'candidate_family': candidate, 'reference': reference,
            'checks': checks, 'passes': bool(all(checks.values())), 'zero_margin_checks': zero,
            'zero_margin_passes': bool(all(zero.values())), **values}


def _means(data, p):
    families = [p['candidate_family'], 'supervised', *BASELINES]
    fields = ('command_loss', 'command_loss_pending_as_misses', 'worst_decile_loss',
              'worst_decile_pending_as_misses', 'video_vs_original', 'bulk_mbps',
              'bulk_backlog_bits', 'bulk_overflow_bits', 'pending_control_packets')
    means, per_seed = [], []
    for cost in p['handover_seconds']:
        for family in families:
            group = [data[site, float(cost), family, seed]
                     for site in p['confirmation_sites'] for seed in _seeds(p, family)]
            means.append({'family': family, 'handover_seconds': float(cost),
                **{field: float(_site_values(data, p, cost, family, field).mean()) for field in fields},
                'global_max_silence_seconds': max(row['max_silence_seconds'] for row in group),
                'independent_sites': len(p['confirmation_sites']), 'training_members': len(_seeds(p, family))})
            for seed in _seeds(p, family):
                rows = [data[site, float(cost), family, seed] for site in p['confirmation_sites']]
                per_seed.append({'family': family, 'handover_seconds': float(cost), 'training_seed': seed,
                    **{field: float(np.mean([row[field] for row in rows])) for field in fields}})
    return means, per_seed


def aggregate(rows, p, validation=None):
    """Analyze all eight arms without allowing one claim to imply another."""
    original, data = _index(rows, p)
    candidate = p['candidate_family']
    sites, seeds = p['confirmation_sites'], p['training_seeds']
    validation = {} if validation is None else dict(validation)
    causal_rows = all(row.get('causality_pass') is True for row in original.values())
    integrity = all(validation.get(name) is True for name in INTEGRITY_FLAGS) and causal_rows
    runtime_overrun = (any(row['runtime_overrun'] for row in data.values()) or
                       validation.get('runtime_failure', False) is not False)
    rng = np.random.default_rng(p['bootstrap_seed'])
    indices = rng.integers(0, len(sites), size=(p['bootstrap_replicates'], len(sites)))
    seed_rng = np.random.default_rng(p['bootstrap_seed'] + 1)
    seed_indices = seed_rng.integers(0, len(seeds), size=(p['bootstrap_replicates'], len(seeds)))
    primary, engineering, cumulative, per_cost = [], [], [], []
    for cost in p['handover_seconds']:
        records = []
        for reference in p['primary_references']:
            observed = (_site_values(data, p, cost, candidate, 'command_loss') -
                        _site_values(data, p, cost, reference, 'command_loss'))
            endpoint = (_site_values(data, p, cost, candidate, 'command_loss_pending_as_misses') -
                        _site_values(data, p, cost, reference, 'command_loss'))
            record = {'candidate_family': candidate, 'handover_seconds': float(cost), 'reference': reference,
                'claim': CLAIM_FOR_REFERENCE[reference],
                **_bootstrap(observed, indices, p['primary_upper_quantile']),
                'endpoint_sensitivity': _bootstrap(endpoint, indices, p['primary_upper_quantile'])}
            record['observed_passes'] = record['multiplicity_upper'] < 0
            record['endpoint_passes'] = record['endpoint_sensitivity']['multiplicity_upper'] < 0
            record['statistical_pass'] = record['observed_passes'] and record['endpoint_passes']
            paired = np.array([[data[site, float(cost), candidate, seed]['command_loss'] -
                data[site, float(cost), reference, None if reference in BASELINES else seed]['command_loss']
                for seed in seeds] for site in sites])
            draws = paired[indices[:, :, None], seed_indices[:, None, :]].mean((1, 2))
            record['secondary_crossed_site_training_seed_95_interval'] = np.quantile(draws, [.025, .975]).tolist()
            gate = _engineering(data, p, cost, reference)
            engineering.append(gate)
            record['engineering_pass'] = gate['passes']
            record['qualified'] = bool(record['statistical_pass'] and gate['passes'] and integrity and not runtime_overrun)
            record['passes'] = record['qualified']
            primary.append(record)
            records.append(record)
            for second in p['cumulative_metrics_seconds']:
                def count(site, family, seed):
                    return next(row['control_misses'] for row in data[site, float(cost), family, seed]['cumulative']
                                if row['slot_end'] == second * 100)
                difference = [np.mean([count(site, candidate, seed) for seed in seeds]) -
                    np.mean([count(site, reference, seed) for seed in _seeds(p, reference)]) for site in sites]
                cumulative.append({'candidate_family': candidate, 'handover_seconds': float(cost),
                    'reference': reference, 'second': second,
                    'units': 'fleet misses, paired training-member mean per whole site',
                    **_bootstrap(difference, indices, .975)})
        qualifications = {record['claim']: record['qualified'] for record in records}
        per_cost.append({'handover_seconds': float(cost), 'status': 'tested', **qualifications,
            'primary_pass': bool(all(record['statistical_pass'] for record in records)),
            'engineering_pass': bool(all(record['engineering_pass'] for record in records)),
            'all_references_pass': bool(all(record['qualified'] for record in records)),
            'passes': bool(all(record['qualified'] for record in records))})
    means, seed_means = _means(data, p)
    slow = next(row for row in per_cost if row['handover_seconds'] == 4.)
    return {'schema': 1, 'study': p.get('study', 'Prospective development-selected JEPA confirmation'),
        'candidate_family': candidate, 'case_count': len(rows), 'sites': sites,
        'independent_sites': len(sites), 'training_seeds': seeds,
        'bootstrap_unit': 'whole site after averaging paired independently trained members',
        'bootstrap_replicates': p['bootstrap_replicates'], 'bootstrap_seed': p['bootstrap_seed'],
        'primary_multiplicity_budget': 6, 'primary_upper_quantile': p['primary_upper_quantile'],
        'primary': primary, 'engineering_gates': engineering, 'per_cost': per_cost,
        'headline_numerical_benefit_at_4s': slow['numerical_benefit'],
        'both_costs_all_references_pass': bool(all(row['all_references_pass'] for row in per_cost)),
        'means': means, 'per_seed_means': seed_means, 'cumulative_excess': cumulative,
        'every_case': list(data.values()), 'integrity_validation': validation,
        'all_row_causality_pass': causal_rows, 'integrity_pass': integrity,
        'any_runtime_overrun': bool(runtime_overrun), 'ema_necessity_claimed': False,
        'activation': False, 'activated': False, 'hardware_qualified': False, 'production_qualified': False,
        'limits': ['Selected on a completed prior development study; its rejected semantic candidate remains rejected.',
            'Privileged simulator training and a shared procedural warehouse CAD; no physical-car qualification.',
            'A numerical-baseline gain does not establish a matched-supervised or EMA-mechanism advantage.',
            'Primary site intervals condition on three fitted members; crossed site/member intervals are secondary.',
            'Pending commands are right censored; empirical tail margins and finite silence are not safety bounds.',
            'No SOTA, novelty, universal transfer or indefinite cumulative-loss guarantee follows.']}


def markdown(report):
    candidate = report['candidate_family']
    lines = ['# Development-selected JEPA: fresh prospective warehouse confirmation', '',
             f"Candidate: `{candidate}`. {report['case_count']} cases on {report['independent_sites']} whole sites; "
             'paired training members are averaged within site, not counted as extra sites.', '',
             'Each reference supports a separate claim. Numerical benefit, improvement over the old JEPA, '
             'and matched objective-package advantage cannot substitute for one another.', '',
             '| Handover s | Numerical benefit | Original JEPA improvement | Objective-package advantage | All references |',
             '|---|---|---|---|---|']
    for row in report['per_cost']:
        lines.append(f"| {row['handover_seconds']:g} | {row['numerical_benefit']} | "
                     f"{row['original_jepa_improvement']} | {row['objective_advantage']} | {row['all_references_pass']} |")
    lines += ['', 'All six comparisons share the registered multiplicity budget. Observed and conservative '
              'endpoint upper bounds must both be negative, engineering margins must pass against that '
              'specific reference, and source/replay/causality/timing vetoes must all clear.', '',
              '| Handover s | Reference | Mean loss difference | Descriptive 95% | Observed upper | Endpoint upper | Statistical | Engineering | Qualified |',
              '|---|---|---:|---|---:|---:|---|---|---|']
    for row in report['primary']:
        lo, hi = row['descriptive_95_interval']
        lines.append(f"| {row['handover_seconds']:g} | {row['reference']} | {row['mean_difference']:.7f} | "
                     f"[{lo:.7f}, {hi:.7f}] | {row['multiplicity_upper']:.7f} | "
                     f"{row['endpoint_sensitivity']['multiplicity_upper']:.7f} | {row['statistical_pass']} | "
                     f"{row['engineering_pass']} | {row['qualified']} |")
    lines += ['', '| Handover s | Family | Command loss | Worst decile | Max silence s | Original video fraction | Bulk Mbps |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in report['means']:
        lines.append(f"| {row['handover_seconds']:g} | {row['family']} | {row['command_loss']:.7f} | "
                     f"{row['worst_decile_loss']:.7f} | {row['global_max_silence_seconds']:.2f} | "
                     f"{row['video_vs_original']:.5f} | {row['bulk_mbps']:.5f} |")
    lines += ['', f"Headline numerical benefit at four seconds: {report['headline_numerical_benefit_at_4s']}. "
              f"Integrity: {report['integrity_pass']}; runtime overrun: {report['any_runtime_overrun']}.", '',
              'Individual-member outcomes, secondary crossed site/member intervals, zero-margin engineering '
              'checks, cumulative losses and every case remain in the machine-readable report. '
              'This is not an EMA-necessity or production qualification claim.', '']
    lines += ['- ' + item for item in report['limits']]
    return '\n'.join(lines) + '\n'
