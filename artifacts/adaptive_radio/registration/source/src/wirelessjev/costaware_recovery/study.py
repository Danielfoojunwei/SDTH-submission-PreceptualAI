"""Registered slow-cost recovery, actual-cost runs and explicit offline replays.

Fresh runs pay measured forecast/issue work. Offline replay reuses the original
paid schedule and makes no newly measured deadline or production claim.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import math
from . import common as c
from . import report as statistics
from . import rollout

_SCHEDULE_FIELDS = ('decision', 'kind', 'history_cutoff_exclusive', 'ready_slot',
    'actual_prediction_seconds', 'actual_issue_seconds', 'actual_compute_seconds',
    'profile_seconds', 'charged_compute_seconds', 'charged_seconds', 'shared_extra_seconds')


def _workers(workers):
    if type(workers) is not int or workers != 2 or c.protocol()['workers'] != 2:
        raise ValueError('Use exactly the two registered execution workers')


def _costs(passed_costs):
    p = c.protocol()
    if passed_costs is None:
        return [float(cost) for cost in p['handover_seconds']]
    if (not isinstance(passed_costs, (tuple, list))
            or any(type(cost) not in (int, float) or not math.isfinite(cost)
                   or cost not in p['handover_seconds'] for cost in passed_costs)
            or len(set(passed_costs)) != len(passed_costs)):
        raise ValueError('Use a unique registered cost subset')
    return [float(cost) for cost in p['handover_seconds'] if cost in passed_costs]


def ordered_keys(stage, passed_costs=None):
    """Check the complete captured permutation before filtering passed costs."""
    if stage not in ('development', 'confirmation'):
        raise ValueError('Unknown scientific stage')
    c.check_sources()
    captured = c.read_json(c.OUT / 'execution_order.json')
    if captured != c.execution_order():
        raise RuntimeError('Captured execution order differs from preregistration')
    full = [tuple(key) for key in captured[stage]]
    expected = rollout.case_keys(stage)
    if len(full) != len(expected) or len(set(full)) != len(full) or set(full) != set(expected):
        raise RuntimeError('Execution order has incomplete or duplicate membership')
    costs = _costs(passed_costs)
    return [key for key in full if key[2] in costs]


def replication_keys(passed_costs):
    first = c.protocol()['confirmation_sites'][0]
    return [key for key in ordered_keys('confirmation', passed_costs) if key[1] == first]


def _run(task):
    key, replay, live = task
    return rollout.run_case(key, replay=replay, live_replication=live)


def execute(keys, workers=2, *, replay=False, live_replication=False):
    """Finish the current site before submitting another site to two workers."""
    _workers(workers)
    if type(replay) is not bool or type(live_replication) is not bool or (replay and live_replication):
        raise ValueError('Use mutually exclusive boolean replication flags')
    keys = [tuple(key) for key in keys]
    if not keys:
        return []
    stage = keys[0][0]
    for key in keys:
        rollout.validate_case_key(key)
        if key[0] != stage:
            raise ValueError('Do not mix scientific stages')
    costs = _costs(list(dict.fromkeys(key[2] for key in keys)))
    expected = replication_keys(costs) if replay or live_replication else ordered_keys(stage, costs)
    if keys != expected or ((replay or live_replication) and stage != 'confirmation'):
        raise RuntimeError('Execution must follow complete registered site blocks')
    if stage == 'confirmation':
        gate = c.verified_development()  # Before executor/world construction.
        if not set(costs) <= set(gate['passed_costs']):
            raise RuntimeError('Development veto prohibits this confirmation cost')
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        cursor = 0
        while cursor < len(keys):
            site = keys[cursor][1]
            end = cursor + 1
            while end < len(keys) and keys[end][1] == site:
                end += 1
            tasks = [(key, replay, live_replication) for key in keys[cursor:end]]
            rows.extend(list(pool.map(_run, tasks)))
            cursor = end
    return rows


def _membership(rows, keys, *, replay=False, live_replication=False):
    if len(rows) != len(keys):
        raise RuntimeError('Incomplete archive matrix')
    mode = 'recorded_schedule_offline' if replay else 'actual_cost'
    for row, key in zip(rows, keys):
        if (tuple(row['case']) != tuple(key) or row['stage'] != key[0]
                or row['site'] != key[1] or row['handover_seconds'] != key[2]
                or row['family'] != key[3] or row['controller_variant'] != key[4]
                or row['replay'] is not replay or row['live_replication'] is not live_replication
                or row['accounting_mode'] != mode):
            raise RuntimeError('Archive case or replication namespace differs')


def _causality(row):
    actual = (len(row['jobs']) == c.protocol()['duration_seconds']
        and all(job['decision'] == i and job['history_cutoff_exclusive'] == 60 + i
                for i, job in enumerate(row['jobs']))
        and all(event['measurement_slot'] < event['complete_slot'] <= event['delivery_slot']
                and event['first_usable_decision'] == event['delivery_slot'] // 100 + 1
                for event in row['telemetry']['events']))
    if row['causality_pass'] is not actual:
        raise RuntimeError('Causality flag differs from actual history/delivery traces')
    return actual


def _fresh(rows, keys, *, live_replication=False):
    _membership(rows, keys, live_replication=live_replication)
    p, profiles = c.protocol(), c.latency()['profiles']
    for row in rows:
        if c.verify_paid_row(row) is not True:
            raise RuntimeError('A fresh run did not pay its actual forecast cost')
        audit = rollout.timing_audit(row['jobs'], row['forecast_consumption'], profiles[row['family']],
            p['extra_compute_seconds'], p['duration_seconds'])
        if (audit['passes'] is not True or audit['real_time_contract_checked'] is not True
                or audit['offline_schedule_only'] is not False or row['timing_safety_pass'] is not True):
            raise RuntimeError('Fresh paid/completion/consumption ledger is invalid')
        _causality(row)
    return True


def _load(keys, *, replay=False, live_replication=False):
    rows = [rollout.load_case(key, replay=replay, live_replication=live_replication) for key in keys]
    if any(row is None for row in rows):
        raise RuntimeError('Complete all registered archives before analysis')
    _membership(rows, keys, replay=replay, live_replication=live_replication)
    return rows


def development(workers=2):
    c.check_sources(); _workers(workers)
    keys = ordered_keys('development')
    rows = execute(keys, workers)
    _fresh(rows, keys)
    validation = c.development_validation(rows)
    gate = statistics.development_gate(rows, c.protocol(), validation=validation)
    c.check_sources()
    value = {'gate': gate, 'validation': validation, 'source_sha256': c.source_hashes(),
             'execution_order_sha256': c.digest(c.OUT / 'execution_order.json')}
    c.write_once(c.OUT / 'development.json', value)
    c.write_once(c.OUT / 'qualification.json', {
        'development_sha256': c.digest(c.OUT / 'development.json'),
        'source_sha256': c.source_hashes(), 'passed_costs': gate['passed_costs'],
        'recomputed_from_raw_archives': True})
    if c.verified_development() != gate:
        raise RuntimeError('Development permission does not reproduce raw archives')
    return value


def _offline_comparisons(main_rows, replay_rows, keys):
    """Measure exactness from raw fields; fresh compute stays diagnostic."""
    _membership(replay_rows, keys, replay=True)
    main = {tuple(row['case']): row for row in main_rows}
    if len(main) != len(main_rows):
        raise RuntimeError('Duplicate original cases')
    p, profiles = c.protocol(), c.latency()['profiles']
    comparisons = []
    for replay, key in zip(replay_rows, keys):
        original = main.get(tuple(key))
        if original is None:
            raise RuntimeError('Offline replay lacks its original case')
        jobs, recorded = replay['jobs'], original['jobs']
        schedule_equal = (len(jobs) == len(recorded) == p['duration_seconds'] and
            all(all(job.get(field) == before.get(field) for field in _SCHEDULE_FIELDS)
                for job, before in zip(jobs, recorded)))
        matches = (len(jobs) == p['duration_seconds'] and all(
            job.get('matches_recorded_forecast') is True
            and job.get('forecast_sha256') == before.get('forecast_sha256')
            and job.get('forecast_std_sha256') == before.get('forecast_std_sha256')
            for job, before in zip(jobs, recorded)))
        explicit_offline = all(job.get('accounting_mode') == 'recorded_schedule_offline'
            and job.get('offline_recorded_schedule') is True
            and job.get('completion_uses_actual_elapsed') is False
            and job.get('recorded_completion_used_actual_elapsed') is True for job in jobs)
        main_path = c.OUT / 'cases' / (rollout.case_name(key) + '.json.gz')
        binding = (replay.get('recorded_schedule_sha256') == c.value_digest(recorded)
                   and replay.get('source_main_archive_sha256') == c.digest(main_path))
        timing = rollout.timing_audit(jobs, replay['forecast_consumption'], profiles[key[3]],
            p['extra_compute_seconds'], p['duration_seconds'],
            accounting_mode='recorded_schedule_offline', recorded_jobs=recorded)
        checks = {'recomputed_native_fields': all(replay[field] == original[field] for field in rollout.REPLAY_FIELDS),
            'original_schedule_exact': schedule_equal,
            'actual_forecasts_match_recorded': matches,
            'forecast_consumption_exact': replay['forecast_consumption'] == original['forecast_consumption'],
            'explicit_offline_metadata': explicit_offline, 'original_archive_and_schedule_binding': binding,
            'recorded_schedule_audit': (timing['passes'] is True
                and timing['offline_schedule_only'] is True and timing['real_time_contract_checked'] is False)}
        comparisons.append({'case': list(key), 'exact': all(checks.values()), 'checks': checks,
            'fresh_measured_deadline_claim': False, 'fresh_compute_is_diagnostic_only': True,
            'fresh_jobs_exceeding_recorded_charge': sum(job.get('exceeds_original_recorded_compute') is True for job in jobs)})
    return comparisons


def _fresh_comparisons(main_rows, live_rows, keys):
    _fresh(live_rows, keys, live_replication=True)
    main = {tuple(row['case']): row for row in main_rows}
    if len(main) != len(main_rows):
        raise RuntimeError('Duplicate original cases')
    comparisons = []
    for row, key in zip(live_rows, keys):
        original = main.get(tuple(key))
        if original is None:
            raise RuntimeError('Fresh replication lacks its original case')
        c.verify_paid_row(original)
        pairs = list(zip(original['jobs'], row['jobs']))
        comparisons.append({'case': list(key),
            'control_miss_difference': row['result']['control_misses'] - original['result']['control_misses'],
            'packet_result_exactly_matches_original': row['result'] == original['result'],
            'changed_ready_jobs': sum(a['ready_slot'] != b['ready_slot'] for a, b in pairs),
            'changed_timing_regime_jobs': sum(a['kind'] != b['kind'] for a, b in pairs),
            'changed_charged_service_jobs': sum(a['charged_seconds'] != b['charged_seconds'] for a, b in pairs),
            'changed_charged_compute_jobs': sum(a['charged_compute_seconds'] != b['charged_compute_seconds'] for a, b in pairs),
            'forecast_consumption_exact': row['forecast_consumption'] == original['forecast_consumption'],
            'paid_cost_and_consumption_pass': True, 'independent_sites_added': 0,
            'bitwise_equality_required': False, 'sensitivity_only': True})
    return comparisons


def _receipt_hashes(keys, replay_keys):
    value = {}
    for folder, members in (('cases', keys), ('replays', replay_keys), ('live_replications', replay_keys)):
        for key in members:
            name = folder + '/' + rollout.case_name(key) + '.receipt.json'
            value[name] = c.digest(c.OUT / name)
    return value


def _completion(main, offline, live, keys, repeats, costs):
    comparisons = _offline_comparisons(main, offline, repeats)
    sensitivity = _fresh_comparisons(main, live, repeats)
    return {'completed': True, 'source_sha256': c.source_hashes(),
        'qualification_sha256': c.digest(c.OUT / 'qualification.json'),
        'execution_order_sha256': c.digest(c.OUT / 'execution_order.json'),
        'passed_costs': costs, 'case_count': len(main), 'offline_replay_count': len(offline),
        'fresh_live_replication_count': len(live), 'offline_comparisons': comparisons,
        'all_offline_exact': all(row['exact'] for row in comparisons),
        'fresh_live_cost_comparisons': sensitivity,
        'all_fresh_paid_cost_checks_pass': all(row['paid_cost_and_consumption_pass'] for row in sensitivity),
        'archive_receipt_sha256': _receipt_hashes(keys, repeats)}


def confirmation(workers=2):
    c.check_sources(); _workers(workers)
    gate = c.verified_development()  # BEFORE any new confirmation world.
    devkeys = ordered_keys('development'); devrows = _load(devkeys); _fresh(devrows, devkeys)
    costs = _costs(gate['passed_costs'])
    keys, repeats = ordered_keys('confirmation', costs), replication_keys(costs)
    main = execute(keys, workers)
    _fresh(main, keys)
    offline = execute(repeats, workers, replay=True)
    live = execute(repeats, workers, live_replication=True)
    _fresh(live, repeats, live_replication=True)
    value = _completion(main, offline, live, keys, repeats, costs)
    c.check_sources(); c.write_once(c.OUT / 'completion.json', value)
    return value


def report():
    """Reload all namespaces and recompute permission/completion/paid proofs."""
    c.check_sources()
    gate = c.verified_development()
    devkeys = ordered_keys('development'); devrows = _load(devkeys); _fresh(devrows, devkeys)
    costs = _costs(gate['passed_costs'])
    keys, repeats = ordered_keys('confirmation', costs), replication_keys(costs)
    main = _load(keys); _fresh(main, keys)
    offline = _load(repeats, replay=True)
    live = _load(repeats, live_replication=True); _fresh(live, repeats, live_replication=True)
    expected = _completion(main, offline, live, keys, repeats, costs)
    if c.read_json(c.OUT / 'completion.json') != expected:
        raise RuntimeError('Completion flags/counts/replications differ from actual raw archives')
    rows = devrows + main + offline + live
    validation = {'source_and_archive_integrity': True,
        'causality_and_kernel_checks': all(_causality(row) for row in rows),
        'paid_cost_and_consumption_checks': True,
        'exact_offline_replay': expected['all_offline_exact']}
    value = statistics.aggregate(main, c.protocol(), validation=validation, passed_costs=costs)
    first_main = [row for row in main if row['site'] == c.protocol()['confirmation_sites'][0]]
    value.update(development_gate=gate,
        offline_replay_comparisons=expected['offline_comparisons'],
        fresh_live_cost_comparisons=expected['fresh_live_cost_comparisons'],
        offline_replay_summary=statistics.replication_summary(offline, c.protocol(),
            'recorded_schedule_offline', passed_costs=costs, original_rows=first_main),
        fresh_live_cost_summary=statistics.replication_summary(live, c.protocol(),
            'fresh_live_cost', passed_costs=costs, original_rows=first_main),
        source_sha256=c.source_hashes(), completion_sha256=c.digest(c.OUT / 'completion.json'),
        development_sha256=c.digest(c.OUT / 'development.json'),
        qualification_sha256=c.digest(c.OUT / 'qualification.json'),
        execution_order_sha256=c.digest(c.OUT / 'execution_order.json'))
    c.check_sources(); c.write_once(c.OUT / 'report.json', value)
    text = statistics.markdown(value)
    text += ('\n## Replication scope\n\n'
        f"{len(offline)} exact-schedule OFFLINE replays; {len(live)} fresh actual-cost repetitions. "
        'The former establish only replay under recorded availability; the latter are descriptive runtime sensitivity and add no independent sites.\n')
    c.write_text_once(c.OUT / 'report.md', text)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'development', 'confirmation', 'report', 'all'))
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(); _workers(args.workers)
    if args.phase == 'register':
        c.register()
    else:
        if args.phase == 'all' and not (c.OUT / 'registration/source.json').exists():
            c.register()
        if args.phase in ('development', 'all'):
            development(args.workers)
        if args.phase in ('confirmation', 'all'):
            confirmation(args.workers)
        if args.phase in ('report', 'all'):
            report()
    print(f'COSTAWARE_RECOVERY_PHASE_COMPLETE {args.phase}', flush=True)


if __name__ == '__main__':
    main()
