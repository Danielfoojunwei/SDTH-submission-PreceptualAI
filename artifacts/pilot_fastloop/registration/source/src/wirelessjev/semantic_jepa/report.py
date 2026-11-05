"""Registered whole-site analysis for supervised semantic-JEPA research.

All public analysis helpers are pure. Fixtures can exercise arithmetic, while
actual archives must first be verified by the study runner. Models and robots
are never treated as additional independent confirmation sites.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

CANDIDATE = 'semantic_jepa'
BASELINES = ('radiomap', 'frozen_fleet_jepa')
INTEGRITY_FLAGS = ('source_and_archive_integrity', 'causality_and_kernel_checks', 'exact_replay')


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    return int(value)


def _finite(value, name, *, minimum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be numeric')
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f'Invalid {name}')
    return value


def _vector(value, n, name, *, minimum=0):
    raw = np.asarray(value)
    if raw.dtype.kind == 'b':
        raise ValueError(f'{name} must contain integer counters')
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f'Malformed {name}') from error
    if (result.shape != (n,) or not np.isfinite(result).all() or
            np.any(result < minimum) or np.any(result != np.floor(result))):
        raise ValueError(f'Malformed {name}')
    return result


def identity(row):
    seed = row['training_seed']
    if seed is not None:
        seed = _integer(seed, 'training_seed')
    return (_integer(row['site'], 'site'),
            _finite(row['handover_seconds'], 'handover_seconds', minimum=0),
            row['family'], seed)


def arms(p):
    return [(family, seed) for family in p['families'] for seed in p['training_seeds']] + [
        (family, None) for family in BASELINES]


def _protocol(p):
    if (len(set(p['families'])) != len(p['families']) or CANDIDATE not in p['families'] or
            set(p['families']) & set(BASELINES)):
        raise ValueError('Invalid family registration')
    if (len(set(p['training_seeds'])) != len(p['training_seeds']) or not p['training_seeds'] or
            any(isinstance(seed, bool) or not isinstance(seed, int) for seed in p['training_seeds'])):
        raise ValueError('Invalid paired training seeds')
    costs = [_finite(cost, 'handover_seconds', minimum=0) for cost in p['handover_seconds']]
    if not costs or len(set(costs)) != len(costs):
        raise ValueError('Invalid registered costs')
    refs = p['primary_references']
    if (not refs or len(set(refs)) != len(refs) or CANDIDATE in refs or
            not set(refs) <= set(p['families']) | set(BASELINES)):
        raise ValueError('Invalid primary references')
    expected_q = 1 - .05 / (len(refs) * len(costs))
    if not math.isclose(p['primary_upper_quantile'], expected_q, abs_tol=1e-15, rel_tol=0):
        raise ValueError('Multiplicity budget differs from complete registered family')
    if _integer(p['bootstrap_replicates'], 'bootstrap_replicates') < 1:
        raise ValueError('No bootstrap draws')
    _integer(p['bootstrap_seed'], 'bootstrap_seed')
    if _integer(p['robots'], 'robots') < 1 or _integer(p['duration_seconds'], 'duration_seconds') < 1:
        raise ValueError('Invalid episode dimensions')
    for name in ('worst_decile_margin', 'silence_margin_seconds', 'video_margin', 'bulk_ratio'):
        _finite(p[name], name, minimum=0)
    if p['bulk_ratio'] > 1:
        raise ValueError('Bulk margin must be a retained delivery fraction')
    for stage in ('development', 'confirmation'):
        sites = p[stage + '_sites']
        if not sites or len(set(sites)) != len(sites):
            raise ValueError(f'Invalid {stage} sites')
        for site in sites:
            _integer(site, 'site')
    checkpoints = p['cumulative_metrics_seconds']
    if (len(set(checkpoints)) != len(checkpoints) or
            any(not 0 < _finite(t, 'checkpoint') <= p['duration_seconds'] for t in checkpoints)):
        raise ValueError('Invalid cumulative checkpoints')


def metrics(row, p):
    """Validate exact native counters before computing one run's metrics."""
    site, cost, family, seed = identity(row)
    if not isinstance(row['runtime_overrun'], bool):
        raise ValueError('Missing or nonboolean runtime veto')
    n, duration = p['robots'], p['duration_seconds']
    result = row['result']
    detail, account = result['packet_detail'], result['packet_accounting']
    misses = _vector(detail['misses'], n, 'misses')
    sent = _vector(detail['sent'], n, 'sent')
    gap = _vector(detail['max_gap'], n, 'max_gap')
    pending_age = _vector(detail['pending_control_packet_age_slots'], n, 'pending ages', minimum=-1)
    if (np.any(sent != duration * 20) or np.any(misses > sent) or
            np.any(gap > duration * 100) or np.any(pending_age >= 5)):
        raise ValueError('Counters outside native episode/deadline support')
    pending = pending_age >= 0
    if np.any(misses + pending > sent):
        raise ValueError('Pending commands exceed generated counters')
    if (_integer(result['control_misses'], 'control_misses') != int(misses.sum()) or
            _integer(result['control_sent'], 'control_sent') != int(sent.sum())):
        raise ValueError('Per-robot and global command totals differ')
    if not math.isclose(_finite(result['control_miss'], 'control_miss'), misses.sum() / sent.sum(), abs_tol=1e-12, rel_tol=0):
        raise ValueError('Reported loss differs from raw counters')
    for container in (detail, account):
        if ('pending_control_packets' in container and
                _integer(container['pending_control_packets'], 'pending_control_packets') != int(pending.sum())):
            raise ValueError('Pending command total differs')
        if 'pending_control_packet_age_slots' in container:
            other = _vector(container['pending_control_packet_age_slots'], n, 'pending ages', minimum=-1)
            if not np.array_equal(other, pending_age):
                raise ValueError('Pending age exports differ')

    flow_names = ('video_requested_bits', 'video_generated_bits', 'video_delivered_bits',
                  'video_pending_bits', 'video_expired_bits', 'bulk_generated_bits',
                  'bulk_delivered_bits', 'final_bulk_backlog_bits', 'bulk_overflow_drop_bits')
    flow = {name: _finite(account[name], name, minimum=0) for name in flow_names}
    video_scale = max(1., flow['video_generated_bits'], flow['video_requested_bits'])
    bulk_scale = max(1., flow['bulk_generated_bits'])
    video_error = flow['video_generated_bits'] - sum(flow[name] for name in
        ('video_delivered_bits', 'video_pending_bits', 'video_expired_bits'))
    bulk_error = flow['bulk_generated_bits'] - sum(flow[name] for name in
        ('bulk_delivered_bits', 'final_bulk_backlog_bits', 'bulk_overflow_drop_bits'))
    if (flow['video_generated_bits'] > flow['video_requested_bits'] + video_scale * 1e-9 or
            abs(video_error) > video_scale * 1e-9 or abs(bulk_error) > bulk_scale * 1e-9):
        raise ValueError('Video or bulk conservation failed')
    for name, expected, scale in [('video_flow_conservation_error_bits', video_error, video_scale),
                                  ('bulk_flow_conservation_error_bits', bulk_error, bulk_scale)]:
        if name in account and not math.isclose(_finite(account[name], name), expected,
                abs_tol=scale * 1e-9, rel_tol=0):
            raise ValueError('Reported conservation residual differs')

    cumulative = detail['cumulative']
    if not cumulative:
        raise ValueError('Cumulative counters missing')
    normalized = []
    previous_slot, previous_miss, previous_sent, previous_gap = 0, np.zeros(n), np.zeros(n), np.zeros(n)
    for entry in cumulative:
        slot = _integer(entry['slot_end'], 'slot_end')
        cm = _vector(entry['misses'], n, 'cumulative misses')
        cs = _vector(entry['sent'], n, 'cumulative sent')
        cg = _vector(entry['max_gap'], n, 'cumulative max_gap')
        if (not previous_slot < slot <= duration * 100 or np.any(cm < previous_miss) or
                np.any(cs < previous_sent) or np.any(cg < previous_gap) or
                np.any(cm > cs) or np.any(cs > sent) or np.any(cg > slot)):
            raise ValueError('Malformed/nonmonotone cumulative counters')
        normalized.append({'slot_end': slot, 'control_misses': int(cm.sum()),
                           'control_sent': int(cs.sum()), 'max_silence_seconds': float(cg.max() / 100)})
        previous_slot, previous_miss, previous_sent, previous_gap = slot, cm, cs, cg
    if (previous_slot != duration * 100 or not np.array_equal(previous_miss, misses) or
            not np.array_equal(previous_sent, sent) or not np.array_equal(previous_gap, gap)):
        raise ValueError('Final cumulative endpoint differs from raw counters')
    slots = {entry['slot_end'] for entry in cumulative}
    if any(second * 100 not in slots for second in p['cumulative_metrics_seconds']):
        raise ValueError('Registered cumulative checkpoint missing')

    decile = max(1, n // 10)
    return {'stage': row['stage'], 'site': site, 'handover_seconds': cost, 'family': family,
            'training_seed': seed, 'control_misses': int(misses.sum()), 'control_sent': int(sent.sum()),
            'command_loss': float(misses.sum() / sent.sum()),
            'command_loss_pending_as_misses': float((misses.sum() + pending.sum()) / sent.sum()),
            'pending_control_packets': int(pending.sum()), 'pending_control_right_censored': True,
            'worst_decile_loss': float(np.sort(misses / sent)[-decile:].mean()),
            'worst_decile_pending_as_misses': float(np.sort((misses + pending) / sent)[-decile:].mean()),
            'max_silence_seconds': float(gap.max() / 100),
            'p99_robot_max_silence_seconds': float(np.quantile(gap, .99) / 100),
            'video_vs_original': flow['video_delivered_bits'] / max(flow['video_requested_bits'], 1e-9),
            'bulk_mbps': flow['bulk_delivered_bits'] / duration / 1e6,
            'bulk_backlog_bits': flow['final_bulk_backlog_bits'],
            'bulk_overflow_bits': flow['bulk_overflow_drop_bits'],
            'video_flow_error_bits': video_error, 'bulk_flow_error_bits': bulk_error,
            'runtime_overrun': row['runtime_overrun'], 'cumulative': normalized,
            'forecast_metrics': dict(row.get('forecast_metrics', {}))}


def _index(rows, p, stage, costs):
    _protocol(p)
    sites = p[stage + '_sites']
    expected = {(site, float(cost), family, seed)
                for site in sites for cost in costs for family, seed in arms(p)}
    indexed = {}
    for row in rows:
        key = identity(row)
        if row['stage'] != stage or key in indexed:
            raise ValueError('Duplicated case or wrong scientific stage')
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError('Case matrix incomplete or outside registered membership')
    data = {key: metrics(row, p) for key, row in indexed.items()}
    for site in sites:
        counts = {tuple(indexed[key]['result']['packet_detail']['sent'])
                  for key in indexed if key[0] == site}
        if len(counts) != (1 if costs else 0):
            raise ValueError('Paired arms have unmatched generated commands')
    return data


def _family_seeds(p, family):
    return [None] if family in BASELINES else p['training_seeds']


def _site_values(data, p, sites, cost, family, field):
    values = []
    for site in sites:
        rows = [data[site, float(cost), family, seed] for seed in _family_seeds(p, family)]
        if field in ('command_loss', 'command_loss_pending_as_misses'):
            # Pool equal-size members as exact counters before the single
            # division: rounding a three-member mean cannot invent a gain.
            misses = sum(row['control_misses'] for row in rows)
            if field == 'command_loss_pending_as_misses':
                misses += sum(row['pending_control_packets'] for row in rows)
            values.append(misses / sum(row['control_sent'] for row in rows))
        else:
            values.append(float(np.mean([row[field] for row in rows])))
    return np.asarray(values, dtype=float)


def _total_rate(data, p, sites, cost, family):
    rows = [data[site, float(cost), family, seed]
            for site in sites for seed in _family_seeds(p, family)]
    return Fraction(sum(row['control_misses'] for row in rows),
                    sum(row['control_sent'] for row in rows))


def _engineering(data, p, sites, cost):
    gates = []
    mean = lambda family, field: float(_site_values(data, p, sites, cost, family, field).mean())
    maximum = lambda family: max(data[site, float(cost), family, seed]['max_silence_seconds']
                    for site in sites for seed in _family_seeds(p, family))
    for reference in p['primary_references']:
        values = {'candidate_mean_worst_decile_pending_as_misses': mean(CANDIDATE, 'worst_decile_pending_as_misses'),
            'candidate_mean_worst_decile_observed': mean(CANDIDATE, 'worst_decile_loss'),
            'reference_mean_worst_decile_observed': mean(reference, 'worst_decile_loss'),
            'candidate_max_silence_seconds': maximum(CANDIDATE),
            'reference_max_silence_seconds': maximum(reference),
            'candidate_mean_video_vs_original': mean(CANDIDATE, 'video_vs_original'),
            'reference_mean_video_vs_original': mean(reference, 'video_vs_original'),
            'candidate_mean_bulk_mbps': mean(CANDIDATE, 'bulk_mbps'),
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
        gates.append({'handover_seconds': float(cost), 'reference': reference, 'checks': checks,
                      'passes': bool(all(checks.values())), 'zero_margin_checks': zero,
                      'zero_margin_passes': bool(all(zero.values())), **values})
    return gates


def _means(data, p, sites, costs):
    fields = ('command_loss', 'command_loss_pending_as_misses', 'worst_decile_loss',
              'worst_decile_pending_as_misses', 'video_vs_original', 'bulk_mbps',
              'bulk_backlog_bits', 'bulk_overflow_bits', 'pending_control_packets')
    families = [*p['families'], *BASELINES]
    means, per_seed = [], []
    for cost in costs:
        for family in families:
            group = [data[site, float(cost), family, seed]
                     for site in sites for seed in _family_seeds(p, family)]
            values = {field: float(np.mean([row[field] for row in group])) for field in fields}
            means.append({'family': family, 'handover_seconds': float(cost), **values,
                'global_max_silence_seconds': max(row['max_silence_seconds'] for row in group),
                'independent_sites': len(sites), 'training_members': len(_family_seeds(p, family))})
            for seed in _family_seeds(p, family):
                rows = [data[site, float(cost), family, seed] for site in sites]
                per_seed.append({'family': family, 'handover_seconds': float(cost), 'training_seed': seed,
                    **{field: float(np.mean([row[field] for row in rows])) for field in fields}})
    return means, per_seed


def development_gate(rows, p, integrity=True):
    """Descriptive per-cost veto; it never declares statistical superiority."""
    if not isinstance(integrity, bool):
        raise ValueError('Development integrity must be an explicit boolean')
    data = _index(rows, p, 'development', p['handover_seconds'])
    sites = p['development_sites']
    gates, by_cost, passed = [], [], []
    global_overrun = any(row['runtime_overrun'] for row in data.values())
    for cost in p['handover_seconds']:
        candidate_rate = _total_rate(data, p, sites, cost, CANDIDATE)
        candidate = float(candidate_rate)
        comparisons = []
        for reference in p['primary_references']:
            reference_rate = _total_rate(data, p, sites, cost, reference)
            actual = float(reference_rate)
            strict = reference in ('radiomap', 'supervised')
            comparisons.append({'reference': reference, 'candidate_loss': candidate, 'reference_loss': actual,
                                'requires_strict_improvement': strict,
                                'passes': candidate_rate < reference_rate if strict else candidate_rate <= reference_rate})
        engineering = _engineering(data, p, sites, cost)
        gates.extend(engineering)
        okay = bool(integrity and not global_overrun and all(row['passes'] for row in comparisons)
                    and all(row['passes'] for row in engineering))
        by_cost.append({'handover_seconds': float(cost), 'comparisons': comparisons,
                        'engineering_pass': all(row['passes'] for row in engineering),
                        'passes': okay, 'status': 'go' if okay else 'development_veto'})
        if okay:
            passed.append(float(cost))
    means, seeds = _means(data, p, sites, p['handover_seconds'])
    return {'stage': 'development', 'case_count': len(rows), 'independent_sites': len(sites),
            'passed_costs': passed, 'by_cost': by_cost, 'engineering_gates': gates,
            'integrity_pass': integrity, 'any_runtime_overrun': global_overrun,
            'means': means, 'per_seed_means': seeds, 'every_case': list(data.values()),
            'statistical_superiority_claimed': False, 'production_qualified': False, 'activated': False}


def _bootstrap(values, indices, upper):
    values = np.asarray(values, dtype=float)
    if values.shape != (indices.shape[1],) or not np.isfinite(values).all():
        raise ValueError('Invalid whole-site differences')
    draws = values[indices].mean(1)
    return {'mean_difference': float(values.mean()),
            'descriptive_95_interval': np.quantile(draws, [.025, .975]).tolist(),
            'upper_quantile': upper, 'multiplicity_upper': float(np.quantile(draws, upper)),
            'paired_site_differences': values.tolist()}


def aggregate_confirmation(rows, p, passed_costs, validation=None):
    """Require complete conditional tests and positive integrity/replay flags."""
    costs = [float(cost) for cost in passed_costs]
    if len(set(costs)) != len(costs) or not set(costs) <= set(p['handover_seconds']):
        raise ValueError('Invalid costs passed from sealed development')
    costs = [float(cost) for cost in p['handover_seconds'] if cost in costs]
    data = _index(rows, p, 'confirmation', costs)
    sites = p['confirmation_sites']
    validation = {} if validation is None else dict(validation)
    integrity = all(validation.get(name) is True for name in INTEGRITY_FLAGS)
    overrun = any(row['runtime_overrun'] for row in data.values()) or validation.get('development_runtime_overrun', False) is not False
    rng = np.random.default_rng(p['bootstrap_seed'])
    indices = rng.integers(0, len(sites), size=(p['bootstrap_replicates'], len(sites)))
    seed_rng = np.random.default_rng(p['bootstrap_seed'] + 1)
    seed_indices = seed_rng.integers(0, len(p['training_seeds']),
                    size=(p['bootstrap_replicates'], len(p['training_seeds'])))
    primary, engineering, cumulative, by_cost = [], [], [], []
    for cost in costs:
        for reference in p['primary_references']:
            observed = (_site_values(data, p, sites, cost, CANDIDATE, 'command_loss') -
                        _site_values(data, p, sites, cost, reference, 'command_loss'))
            endpoint = (_site_values(data, p, sites, cost, CANDIDATE, 'command_loss_pending_as_misses') -
                        _site_values(data, p, sites, cost, reference, 'command_loss'))
            record = {'handover_seconds': cost, 'reference': reference,
                      **_bootstrap(observed, indices, p['primary_upper_quantile']),
                      'endpoint_sensitivity': _bootstrap(endpoint, indices, p['primary_upper_quantile'])}
            record['observed_passes'] = record['multiplicity_upper'] < 0
            record['endpoint_passes'] = record['endpoint_sensitivity']['multiplicity_upper'] < 0
            record['passes'] = record['observed_passes'] and record['endpoint_passes']
            # Secondary crossed resampling uses the SAME paired training index
            # across families and whole sites; it creates no extra site units.
            paired = np.array([[data[site, cost, CANDIDATE, seed]['command_loss'] -
                data[site, cost, reference, None if reference in BASELINES else seed]['command_loss']
                for seed in p['training_seeds']] for site in sites])
            draws = paired[indices[:, :, None], seed_indices[:, None, :]].mean((1, 2))
            record['secondary_crossed_site_training_seed_95_interval'] = np.quantile(draws, [.025, .975]).tolist()
            primary.append(record)
            for second in p['cumulative_metrics_seconds']:
                def count(site, family, seed):
                    return next(entry['control_misses'] for entry in data[site, cost, family, seed]['cumulative']
                                if entry['slot_end'] == second * 100)
                difference = [np.mean([count(site, CANDIDATE, seed) for seed in p['training_seeds']]) -
                    np.mean([count(site, reference, seed) for seed in _family_seeds(p, reference)]) for site in sites]
                cumulative.append({'handover_seconds': cost, 'reference': reference, 'second': second,
                    'units': 'fleet misses, paired training-seed mean per whole site',
                    **_bootstrap(difference, indices, .975)})
        gates = _engineering(data, p, sites, cost)
        engineering.extend(gates)
        primary_pass = all(row['passes'] for row in primary if row['handover_seconds'] == cost)
        engineering_pass = all(row['passes'] for row in gates)
        by_cost.append({'handover_seconds': cost, 'status': 'tested',
            'primary_pass': primary_pass, 'engineering_pass': engineering_pass,
            'passes': bool(primary_pass and engineering_pass and integrity and not overrun)})
    for cost in p['handover_seconds']:
        if cost not in costs:
            by_cost.append({'handover_seconds': float(cost), 'status': 'not_run_development_veto',
                            'primary_pass': False, 'engineering_pass': False, 'passes': False})
    by_cost.sort(key=lambda row: p['handover_seconds'].index(row['handover_seconds']))
    means, seeds = _means(data, p, sites, costs) if costs else ([], [])
    return {'stage': 'confirmation', 'case_count': len(rows), 'sites': sites,
        'independent_sites': len(sites) if costs else 0, 'registered_site_count': len(sites),
        'tested_sites': sites if costs else [], 'training_seeds': p['training_seeds'],
        'bootstrap_unit': 'whole site after averaging paired independently trained members',
        'bootstrap_replicates': p['bootstrap_replicates'], 'bootstrap_seed': p['bootstrap_seed'],
        'primary_multiplicity_budget': len(p['primary_references']) * len(p['handover_seconds']),
        'primary_upper_quantile': p['primary_upper_quantile'], 'passed_development_costs': costs,
        'per_cost': by_cost, 'primary': primary, 'engineering_gates': engineering,
        'cumulative_excess': cumulative, 'means': means, 'per_seed_means': seeds,
        'every_case': list(data.values()), 'integrity_validation': validation, 'integrity_pass': integrity,
        'any_runtime_overrun': bool(overrun), 'both_costs_pass': bool(all(row['passes'] for row in by_cost)),
        'hardware_qualified': False, 'production_qualified': False, 'activated': False,
        'limits': ['Privileged simulator mean-field labels; not observation-only/self-supervised learning.',
            'Shared procedural warehouse CAD and assumed scan/handover/report capabilities.',
            'Primary site intervals condition on the three fitted members; crossed-seed intervals are secondary.',
            'Pending commands are right censored; finite observed silence is not a safety bound.',
            'Direct horizon anchors do not bound indefinite closed-loop cumulative loss.',
            'No hardware, universal, SOTA, novelty or production qualification follows.']}


def build_report(development_rows, confirmation_rows, p, validation=None):
    validation = {} if validation is None else dict(validation)
    dev_integrity = all(validation.get(name) is True for name in
                        ('source_and_archive_integrity', 'causality_and_kernel_checks'))
    development = development_gate(development_rows, p, integrity=dev_integrity)
    validation['development_runtime_overrun'] = development['any_runtime_overrun']
    confirmation = aggregate_confirmation(confirmation_rows, p, development['passed_costs'], validation)
    return {'schema': 1, 'study': p['study'], 'development': development,
            'confirmation': confirmation, 'activated': False, 'production_qualified': False}


def markdown(report):
    """Render complete positive, negative and unrun outcomes without selection."""
    development = report.get('development')
    confirmation = report.get('confirmation', report)
    lines = ['# Privileged semantic JEPA: registered warehouse service comparison', '',
             'Simulation-only supervised radio semantics; no physical or production qualification.', '']
    if development is not None:
        lines += [f"Development: {development['case_count']} runs; "
                  f"{development['independent_sites']} whole sites; costs allowed to confirmation: "
                  f"{development['passed_costs']}.", '']
    lines += [f"Confirmation: {confirmation['case_count']} runs; "
              f"{confirmation['independent_sites']} registered whole sites. "
              'Paired training members are averaged within site; robots and members are not extra sites.', '',
              '| Handover s | Status | Primary | Engineering | Service experiment passes |',
              '|---|---|---|---|---|']
    for row in confirmation['per_cost']:
        lines.append(f"| {row['handover_seconds']:g} | {row['status']} | {row['primary_pass']} | "
                     f"{row['engineering_pass']} | {row['passes']} |")
    lines += ['', 'All registered per-cost comparisons retain the full eight-comparison multiplicity budget. '
              'Both observed and conservative pending-command upper bounds must be negative.', '',
              '| Handover s | Reference | Mean loss difference | Descriptive 95% | Observed upper | Endpoint upper | Pass |',
              '|---|---|---:|---|---:|---:|---|']
    for row in confirmation['primary']:
        lo, hi = row['descriptive_95_interval']
        lines.append(f"| {row['handover_seconds']:g} | {row['reference']} | {row['mean_difference']:.7f} | "
                     f"[{lo:.7f}, {hi:.7f}] | {row['multiplicity_upper']:.7f} | "
                     f"{row['endpoint_sensitivity']['multiplicity_upper']:.7f} | {row['passes']} |")
    for label, section in [('Development', development), ('Confirmation', confirmation)]:
        if section is None:
            continue
        lines += ['', f'## {label} means', '',
                  '| Handover s | Family | Command loss | Worst decile | Max silence s | Original video fraction | Bulk Mbps |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for row in section['means']:
            lines.append(f"| {row['handover_seconds']:g} | {row['family']} | {row['command_loss']:.7f} | "
                         f"{row['worst_decile_loss']:.7f} | {row['global_max_silence_seconds']:.2f} | "
                         f"{row['video_vs_original']:.5f} | {row['bulk_mbps']:.5f} |")
    lines += ['', f"Integrity: {confirmation['integrity_pass']}; runtime overrun: "
              f"{confirmation['any_runtime_overrun']}; both costs pass: {confirmation['both_costs_pass']}.", '',
              'Per-seed means, zero-margin engineering checks, cumulative contrasts, crossed site/seed intervals '
              'and all individual metrics remain in the machine-readable report.', '']
    lines += ['- ' + item for item in confirmation['limits']]
    return '\n'.join(lines) + '\n'
