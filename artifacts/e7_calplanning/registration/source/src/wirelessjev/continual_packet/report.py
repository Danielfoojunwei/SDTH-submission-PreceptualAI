"""Paired-site preregistered analysis; no fitting, case selection or activation.

The pure aggregate function accepts fixtures for unit tests. The report entry
point requires every immutable real confirmation case and exact replay receipts.
"""
from __future__ import annotations
import json
import math
import numpy as np

ESTIMATORS = ('memory', 'bayesian')
REFERENCES = ('canonical', 'reciprocal_radiomap', 'staggered_ignored')


def key(row):
    return (int(row['site']), int(row['packet_seed']), float(row['handover_seconds']), row['arm'])


def case_metrics(row, p):
    result = row['result']
    d, account = result['packet_detail'], result['managed_packet_accounting']
    misses, sent = np.asarray(d['misses'], dtype=float), np.asarray(d['sent'], dtype=float)
    if misses.shape != (p['robots'],) or sent.shape != misses.shape or np.any(sent <= 0) or np.any(misses < 0) or np.any(misses > sent):
        raise ValueError('Malformed raw per-robot command counters')
    if np.any(misses != np.floor(misses)) or np.any(sent != np.floor(sent)):
        raise ValueError('Command counters must be integers')
    if int(misses.sum()) != result['control_misses'] or int(sent.sum()) != result['control_sent']:
        raise ValueError('Raw and reported command totals differ')
    if np.any(sent != p['duration_seconds'] * 20):
        raise ValueError('Unexpected command generation count')
    ages = np.asarray(account['pending_control_packet_age_slots'])
    if (ages.shape != sent.shape or not np.isfinite(ages).all() or np.any(ages < -1) or
            np.any(ages >= 5) or np.any(ages != np.floor(ages))):
        raise ValueError('Malformed right-censored command ages')
    pending = ages >= 0
    if int(pending.sum()) != account['pending_control_packets'] or np.any(misses + pending > sent):
        raise ValueError('Malformed right-censored command counters')
    gaps = np.asarray(d['max_gap'], dtype=float)
    if gaps.shape != sent.shape or not np.isfinite(gaps).all() or np.any(gaps < 0):
        raise ValueError('Malformed observed silence counters')
    reports, records = row.get('probe_reports', []), row.get('predictor_records', [])
    summary = result['managed_probe_summary']
    local = [record['local_compute_seconds'] for record in records]
    measured_ages = [(r['delivered_slot'] - r['measured_slots'][-1]) / 100 for r in reports]
    # Availability age is derived only from already delivered public reports.
    decision_ages = []
    for decision in range(p['duration_seconds']):
        available = [r for r in reports if r['delivered_slot'] <= decision * 100]
        if available:
            decision_ages.append(decision - max(r['measured_slots'][-1] for r in available) / 100)
    identified = sum(r.get('identified_queries', 0) for r in records)
    fallback = sum(r.get('fallback_queries', 0) for r in records)
    return {'site': row['site'], 'packet_seed': row['packet_seed'],
        'handover_seconds': row['handover_seconds'], 'arm': row['arm'],
        'control_misses': int(misses.sum()), 'control_sent': int(sent.sum()),
        'command_loss': float(misses.sum() / sent.sum()),
        'command_loss_pending_as_misses': float((misses.sum() + pending.sum()) / sent.sum()),
        'worst_decile_loss': float(np.sort(misses / sent)[-max(1, p['robots'] // 10):].mean()),
        'worst_decile_pending_as_misses': float(np.sort((misses + pending) / sent)[-max(1, p['robots'] // 10):].mean()),
        'max_silence_seconds': float(gaps.max() / 100),
        'p99_robot_max_silence_seconds': float(np.quantile(gaps, .99) / 100),
        'video_requested_bits': account['video_requested_bits'],
        'video_encoded_bits': account['video_generated_bits'],
        'video_delivered_bits': account['video_delivered_bits'],
        'video_vs_original': float(account['video_delivered_bits'] / max(account['video_requested_bits'], 1e-9)),
        'bulk_generated_bits': account['bulk_generated_bits'], 'bulk_delivered_bits': account['bulk_delivered_bits'],
        'bulk_mbps': float(account['bulk_delivered_bits'] / p['duration_seconds'] / 1e6),
        'bulk_backlog_bits': account['final_bulk_backlog_bits'],
        'bulk_overflow_bits': account['bulk_overflow_drop_bits'],
        'pending_control_packets': account['pending_control_packets'],
        'pending_control_right_censored': True, 'roams': result['roams'],
        'captures': summary['captured_slots'], 'report_bytes': summary['report_bytes_transmitted'],
        'probe_and_report_requested_domain_slot_airtime': summary['requested_domain_slot_airtime'],
        'managed_allocated_domain_slot_airtime': summary['allocated_domain_slot_airtime'],
        'pending_probe_transport_cycles': summary['pending_transport_cycles'],
        'pending_probe_delivery_cycles': summary['pending_delivery_cycles'],
        'delivered_measurement_age_mean_seconds': float(np.mean(measured_ages)) if measured_ages else None,
        'latest_available_report_age_mean_seconds': float(np.mean(decision_ages)) if decision_ages else None,
        'identified_basis_queries': identified, 'fallback_basis_queries': fallback,
        'identified_basis_fraction': identified / (identified + fallback) if identified + fallback else None,
        'used_observations': sum(r.get('used_observations', 0) for r in records),
        'local_head_p95_seconds': float(np.quantile(local, .95)) if local else None,
        'local_head_max_seconds': max(local) if local else None,
        'runtime_overrun': bool(row['runtime']['head_overrun_decisions'] or row['runtime']['total_prediction_overrun_decisions'])}


def bootstrap_summary(values, indices, upper_quantile):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != indices.shape[1] or not np.isfinite(values).all():
        raise ValueError('Invalid paired site differences')
    draws = values[indices].mean(axis=1)
    return {'mean_difference': float(values.mean()),
            'descriptive_95_interval': np.quantile(draws, [.025, .975]).tolist(),
            'upper_quantile': upper_quantile, 'multiplicity_upper': float(np.quantile(draws, upper_quantile)),
            'paired_site_differences': values.tolist()}


def aggregate(rows, p, *, validation=None):
    """All confirmation rows, paired seeds inside sites, bootstrap sites only."""
    sites, packets, costs, arms = p['confirmation_sites'], p['confirmation_packet_seeds'], p['handover_seconds'], p['arms']
    expected = {(site, packet, float(cost), arm) for site in sites for packet in packets for cost in costs for arm in arms}
    indexed = {key(row): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError('Confirmation matrix incomplete, duplicated or outside registration')
    metrics = {k: case_metrics(row, p) for k, row in indexed.items()}
    for site in sites:
        for packet in packets:
            counts = {tuple(indexed[(site, packet, float(cost), arm)]['result']['packet_detail']['sent'])
                      for cost in costs for arm in arms}
            if len(counts) != 1:
                raise ValueError('Paired arms have unmatched command generation')
    rng = np.random.default_rng(p['bootstrap_seed'])
    indices = rng.integers(0, len(sites), size=(p['bootstrap_replicates'], len(sites)))
    def site_values(cost, arm, field='command_loss'):
        return np.array([np.mean([metrics[(site, packet, float(cost), arm)][field] for packet in packets]) for site in sites])
    primary, information, gates = [], [], []
    for estimator in ESTIMATORS:
        candidate = 'staggered_' + estimator
        for cost in costs:
            for reference in REFERENCES:
                difference = site_values(cost, candidate) - site_values(cost, reference)
                record = {'estimator': estimator, 'handover_seconds': cost, 'reference': reference,
                          **bootstrap_summary(difference, indices, 1 - .05 / 12)}
                endpoint = site_values(cost, candidate, 'command_loss_pending_as_misses') - site_values(cost, reference)
                record['endpoint_sensitivity'] = bootstrap_summary(endpoint, indices, 1 - .05 / 12)
                record['observed_passes'] = record['multiplicity_upper'] < 0
                record['endpoint_passes'] = record['endpoint_sensitivity']['multiplicity_upper'] < 0
                record['passes'] = record['observed_passes'] and record['endpoint_passes']
                primary.append(record)
            difference = (site_values(cost, candidate) - site_values(cost, 'staggered_ignored') -
                          site_values(cost, 'synchronized_' + estimator) + site_values(cost, 'synchronized_ignored'))
            record = {'estimator': estimator, 'handover_seconds': cost,
                      **bootstrap_summary(difference, indices, 1 - .05 / 4)}
            endpoint = (site_values(cost, candidate, 'command_loss_pending_as_misses') -
                        site_values(cost, 'staggered_ignored') - site_values(cost, 'synchronized_' + estimator) +
                        site_values(cost, 'synchronized_ignored', 'command_loss_pending_as_misses'))
            record['endpoint_sensitivity'] = bootstrap_summary(endpoint, indices, 1 - .05 / 4)
            record['observed_passes'] = record['multiplicity_upper'] < 0
            record['endpoint_passes'] = record['endpoint_sensitivity']['multiplicity_upper'] < 0
            record['passes'] = record['observed_passes'] and record['endpoint_passes']
            information.append(record)
            for reference in ('canonical', 'reciprocal_radiomap'):
                candidate_rows = [metrics[(s, packet, float(cost), candidate)] for s in sites for packet in packets]
                reference_rows = [metrics[(s, packet, float(cost), reference)] for s in sites for packet in packets]
                mean = lambda rs, field: float(np.mean([r[field] for r in rs]))
                values = {'candidate_max_silence_seconds': max(r['max_silence_seconds'] for r in candidate_rows),
                    'reference_max_silence_seconds': max(r['max_silence_seconds'] for r in reference_rows),
                    'candidate_mean_worst_decile_loss': mean(candidate_rows, 'worst_decile_loss'),
                    'candidate_mean_worst_decile_pending_as_misses': mean(candidate_rows, 'worst_decile_pending_as_misses'),
                    'reference_mean_worst_decile_loss': mean(reference_rows, 'worst_decile_loss'),
                    'candidate_mean_video_vs_original': mean(candidate_rows, 'video_vs_original'),
                    'reference_mean_video_vs_original': mean(reference_rows, 'video_vs_original'),
                    'candidate_mean_bulk_mbps': mean(candidate_rows, 'bulk_mbps'),
                    'reference_mean_bulk_mbps': mean(reference_rows, 'bulk_mbps')}
                checks = {'maximum_silence': values['candidate_max_silence_seconds'] <= values['reference_max_silence_seconds'],
                    'worst_decile': values['candidate_mean_worst_decile_pending_as_misses'] <= values['reference_mean_worst_decile_loss'],
                    'original_video': values['candidate_mean_video_vs_original'] >= values['reference_mean_video_vs_original'] - .01,
                    'bulk_delivery': values['candidate_mean_bulk_mbps'] >= .99 * values['reference_mean_bulk_mbps']}
                gates.append({'estimator': estimator, 'handover_seconds': cost, 'reference': reference,
                              'checks': checks, 'passes': all(checks.values()), **values})
    validation = {} if validation is None else dict(validation)
    required_integrity = ('source_and_archive_integrity', 'causality_and_kernel_checks', 'exact_replay')
    integrity_pass = all(validation.get(name) is True for name in required_integrity)
    overrun = any(r['runtime_overrun'] for r in metrics.values()) or validation.get('development_runtime_overrun', False)
    primary_pass = {e: all(r['passes'] for r in primary if r['estimator'] == e) for e in ESTIMATORS}
    information_pass = {e: all(r['passes'] for r in information if r['estimator'] == e) for e in ESTIMATORS}
    engineering_pass = {e: all(r['passes'] for r in gates if r['estimator'] == e) for e in ESTIMATORS}
    qualified = {e: primary_pass[e] and engineering_pass[e] and integrity_pass and not overrun for e in ESTIMATORS}
    # Checkpoint analyses are descriptive; they cannot replace the full-run primary.
    cumulative = []
    for estimator in ESTIMATORS:
        for cost in costs:
            for reference in REFERENCES:
                for second in p['cumulative_metrics_seconds']:
                    def at(site, packet, arm):
                        trace = indexed[(site, packet, float(cost), arm)]['result']['packet_detail']['cumulative']
                        entry = next((r for r in trace if r['slot_end'] == second * 100), None)
                        if entry is None:
                            raise ValueError('Registered cumulative checkpoint missing')
                        return int(sum(entry['misses']))
                    excess = [np.mean([at(s, packet, 'staggered_' + estimator) - at(s, packet, reference)
                                       for packet in packets]) for s in sites]
                    cumulative.append({'estimator': estimator, 'handover_seconds': cost, 'reference': reference,
                        'second': second, 'units': 'integer misses, packet-seed mean per whole site',
                        **bootstrap_summary(excess, indices, .975)})
    averages = []
    for cost in costs:
        for arm in arms:
            group = [r for r in metrics.values() if r['arm'] == arm and r['handover_seconds'] == cost]
            numeric = ('command_loss', 'worst_decile_loss', 'video_vs_original', 'bulk_mbps', 'bulk_backlog_bits',
                       'bulk_overflow_bits', 'pending_control_packets', 'report_bytes', 'managed_allocated_domain_slot_airtime')
            averages.append({'arm': arm, 'handover_seconds': cost,
                **{field: float(np.mean([r[field] for r in group])) for field in numeric},
                'global_max_silence_seconds': max(r['max_silence_seconds'] for r in group)})
    return {'schema': 1, 'scope': p['scope'], 'main_confirmation_cases': len(rows), 'sites': sites,
        'packet_seeds': packets, 'bootstrap_unit': 'whole site after paired packet-seed averaging',
        'bootstrap_replicates': p['bootstrap_replicates'], 'bootstrap_seed': p['bootstrap_seed'],
        'primary': primary, 'information_effect': information, 'engineering_gates': gates,
        'primary_pass_by_estimator': primary_pass, 'independent_information_pass_by_estimator': information_pass,
        'endpoint_sensitivity_rule': 'Unresolved commands in positive-sign arms are misses; negative-sign arms are successes. Primary and information bounds must pass under both observed and conservative outcomes.',
        'engineering_pass_by_estimator': engineering_pass, 'integrity_validation': validation,
        'integrity_pass': integrity_pass, 'any_uncharged_runtime_overrun': overrun,
        'service_qualified_by_estimator': qualified,
        'independent_information_service_qualified_by_estimator': {e: qualified[e] and information_pass[e] for e in ESTIMATORS},
        'means': averages, 'every_case': list(metrics.values()), 'cumulative_excess': cumulative,
        'uncertainty_calibrated': False, 'hardware_qualified': False, 'activated': False,
        'limitations': ['Procedural simulator, shared CAD, not measured 100-robot warehouse evidence.',
            'AP-assisted carrier sense, aligned nested duty, mean RF before nonlinear PER and dedicated capture are assumptions.',
            'Bayesian posterior updates do not establish algebraic identification or calibrated service risk.',
            'Six sites are independent analysis units; packet seeds and robots are not extra sites.',
            'Pending controls are right censored. Observed maximum silence is not a safety guarantee.',
            'No JEPA, physical-world, novelty or SOTA claim follows from this experiment.']}


def markdown(report):
    lines = ['# Costed managed-probe packet study', '', report['scope'], '',
        f"{report['main_confirmation_cases']} confirmation cases; {len(report['sites'])} whole sites; "
        f"{len(report['packet_seeds'])} paired packet seeds; {report['bootstrap_replicates']} site bootstrap draws.", '',
        '| Estimator | Primary (all six) | Information (both costs) | Engineering margins | Integrity | Service qualified |',
        '|---|---|---|---|---|---|']
    for e in ESTIMATORS:
        values = [e, report['primary_pass_by_estimator'][e], report['independent_information_pass_by_estimator'][e],
                  report['engineering_pass_by_estimator'][e], report['integrity_pass'] and not report['any_uncharged_runtime_overrun'],
                  report['service_qualified_by_estimator'][e]]
        lines.append('| ' + ' | '.join(str(v) for v in values) + ' |')
    lines += ['', 'Every primary comparison is reported, including negative and null outcomes. '
              'A negative difference favors the staggered estimator. Upper bounds use 1−0.05/12; '
              'information contrasts use 1−0.05/4. These are empirical margins, not safety certification.', '',
        '| Estimator | Handover s | Reference | Loss difference | Descriptive 95% | Observed upper | Endpoint upper | Pass |',
        '|---|---:|---|---:|---|---:|---:|---|']
    for r in report['primary']:
        lo, hi = r['descriptive_95_interval']
        lines.append(f"| {r['estimator']} | {r['handover_seconds']} | {r['reference']} | {r['mean_difference']:.6f} | "
                     f"[{lo:.6f}, {hi:.6f}] | {r['multiplicity_upper']:.6f} | {r['endpoint_sensitivity']['multiplicity_upper']:.6f} | {r['passes']} |")
    lines += ['', report['endpoint_sensitivity_rule'], '',
              '| Estimator | Handover s | Information difference-in-differences | Observed upper | Endpoint upper | Pass |',
              '|---|---:|---:|---:|---:|---|']
    for r in report['information_effect']:
        lines.append(f"| {r['estimator']} | {r['handover_seconds']} | {r['mean_difference']:.6f} | "
                     f"{r['multiplicity_upper']:.6f} | {r['endpoint_sensitivity']['multiplicity_upper']:.6f} | {r['passes']} |")
    lines += ['', '| Arm | Handover s | Command loss | Worst decile | Max silence s | Original video fraction | Bulk Mbps | Pending controls |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in report['means']:
        lines.append(f"| {r['arm']} | {r['handover_seconds']} | {r['command_loss']:.6f} | {r['worst_decile_loss']:.6f} | "
            f"{r['global_max_silence_seconds']:.2f} | {r['video_vs_original']:.4f} | {r['bulk_mbps']:.4f} | {r['pending_control_packets']:.2f} |")
    lines += ['', 'All per-case metrics, cumulative checkpoint excess, backlog, overflow, paid resources, '
              'measurement ages, identification/fallback counts and timings are in report.json. '
              'Immutable case archives retain every per-robot second-by-second counter trace, raw paid report, '
              'scan delivery event and forecast completion ledger. Endpoint pending controls are right censored.', '']
    lines += ['- ' + item for item in report['limitations']]
    return '\n'.join(lines) + '\n'


def run_report():
    from . import study
    frozen, p = study.check_sources(), study.protocol()
    charged = study.charges('confirmation')
    development = json.loads((study.OUT / 'development.json').read_text())
    development_keys = study.case_keys('development')
    if (development['completed'] is not True or development['source_sha256'] != frozen['source_sha256'] or
            development['main_cases'] != len(development_keys) or
            set(development['receipt_sha256']) != {study.case_name(k) for k in development_keys}):
        raise RuntimeError('Invalid development completion receipt')
    development_rows = []
    for k in development_keys:
        row = study.load_case(k, charged=study.charges('development'))
        if row is None:
            raise RuntimeError('Development timing archive missing')
        path = study.OUT / 'cases' / (study.case_name(k) + '.receipt.json')
        if study.digest(path) != development['receipt_sha256'][study.case_name(k)]:
            raise RuntimeError('Development receipt changed after completion')
        development_rows.append(row)
    completion = json.loads((study.OUT / 'confirmation.json').read_text())
    keys = study.case_keys('confirmation')
    expected = {study.case_name(k) for k in keys}
    if (completion['completed'] is not True or completion['source_sha256'] != frozen['source_sha256'] or
            completion['main_cases'] != len(keys) or set(completion['receipt_sha256']) != expected):
        raise RuntimeError('Invalid or incomplete confirmation completion receipt')
    rows = []
    for k in keys:
        row = study.load_case(k, charged=charged)
        if row is None:
            raise RuntimeError('Registered confirmation archive missing')
        path = study.OUT / 'cases' / (study.case_name(k) + '.receipt.json')
        if study.digest(path) != completion['receipt_sha256'][study.case_name(k)]:
            raise RuntimeError('Confirmation receipt changed after completion')
        rows.append(row)
    replay_receipt = json.loads((study.OUT / 'replay_validation.json').read_text())
    expected_replays = study.verification_keys()
    if (replay_receipt['source_sha256'] != frozen['source_sha256'] or
            {tuple(r['case']) for r in replay_receipt['comparisons']} != set(expected_replays)):
        raise RuntimeError('Verification replay identity differs from registration')
    originals = {tuple(row['case']): row for row in rows}
    actual_replay_passes = []
    replay_hashes = {}
    for k in expected_replays:
        replay = study.load_case(k, charged=charged, replay=True)
        if replay is None:
            raise RuntimeError('Required verification replay archive missing')
        actual_replay_passes.append(all(originals[k][field] == replay[field] for field in
            ('result', 'data_sha256', 'forecast_mean_sha256', 'forecast_std_sha256')))
        path = study.OUT / 'replays' / (study.case_name(k) + '.receipt.json')
        replay_hashes[study.case_name(k)] = study.digest(path)
    replay_pass = (replay_receipt['completed'] is True and replay_receipt['all_exact'] is True and
                   all(r['exact_match'] is True for r in replay_receipt['comparisons']) and all(actual_replay_passes))
    report = aggregate(rows, p, validation={'source_and_archive_integrity': True,
        'causality_and_kernel_checks': all(r['integrity']['causality_and_accounting_pass'] is True for r in rows),
        'exact_replay': replay_pass,
        'development_runtime_overrun': any(r['runtime']['head_overrun_decisions'] or
            r['runtime']['total_prediction_overrun_decisions'] for r in development_rows)})
    report.update(source_sha256=frozen['source_sha256'], protocol_sha256=study.digest(study.PROTOCOL),
        latency_sha256=study.digest(study.OUT / 'latency.json'),
        confirmation_receipt_sha256=study.digest(study.OUT / 'confirmation.json'),
        replay_receipt_sha256=study.digest(study.OUT / 'replay_validation.json'),
        development_receipt_sha256=study.digest(study.OUT / 'development.json'),
        development_runtime=[{'case': r['case'], **r['runtime']} for r in development_rows],
        replay_archive_receipts=replay_hashes)
    study.write_once(study.OUT / 'report.json', report)
    # Markdown is an immutable derived artifact too, stored using the same atomic helper.
    text = markdown(report)
    study.write_once(study.OUT / 'report_markdown.json', {'markdown': text})
    path = study.OUT / 'report.md'
    if path.exists() and path.read_text() != text:
        raise RuntimeError('Existing immutable Markdown report differs')
    if not path.exists():
        import os
        import tempfile
        fd, temporary = tempfile.mkstemp(prefix='.report-', dir=study.OUT)
        try:
            with os.fdopen(fd, 'w') as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_text() != text:
                    raise RuntimeError('Concurrent Markdown report differs')
        finally:
            os.unlink(temporary)
    return report


if __name__ == '__main__':
    run_report()
