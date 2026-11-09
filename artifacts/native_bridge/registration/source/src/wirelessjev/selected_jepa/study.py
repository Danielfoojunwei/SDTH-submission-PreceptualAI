"""A separate frozen confirmation; no training or retrospective retuning."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from . import common as c
from .rollout import case_keys, run_case, load_case, REPLAY_FIELDS


def replay_keys():
    p = c.protocol()
    site = p['confirmation_sites'][0]
    return [(site, float(cost), family, seed) for cost in p['handover_seconds']
            for family, seed in [('radiomap', None), ('frozen_fleet_jepa', None),
                                 (p['candidate_family'], 0), ('supervised', 0)]]


def execute(keys, workers=2):
    if workers not in (1, 2):
        raise ValueError('Use one or two evaluation workers')
    if workers == 1:
        return [run_case(key) for key in keys]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(run_case, keys))


def compare_replays(rows, replays):
    expected = replay_keys()
    original = {tuple(row['case']): row for row in rows}
    actual = {tuple(row['case']): row for row in replays}
    if len(actual) != len(replays) or set(actual) != set(expected) or not set(expected) <= set(original):
        raise RuntimeError('Replay/original membership differs from registration')
    return [{'case': list(key), 'exact': all(original[key][field] == actual[key][field]
            for field in REPLAY_FIELDS)} for key in expected]


def confirm(workers=2):
    c.check_sources()
    c.verify_selection()  # Derive selected family again before opening fresh worlds.
    rows = execute(case_keys(), workers)
    replays = [run_case(key, replay=True) for key in replay_keys()]
    comparisons = compare_replays(rows, replays)
    value = {'completed': True, 'case_count': len(rows), 'new_fits': 0,
             'source_sha256': c.source_hashes(), 'selection_sha256': c.digest(c.OUT / 'selection.json'),
             'replays': comparisons, 'all_exact': all(row['exact'] for row in comparisons)}
    c.write_once(c.OUT / 'completion.json', value)
    return value


def report():
    from .report import aggregate, markdown
    c.check_sources()
    c.verify_selection()
    rows = [load_case(key) for key in case_keys()]
    replays = [load_case(key, replay=True) for key in replay_keys()]
    if any(row is None for row in rows + replays):
        raise RuntimeError('Complete every registered case and replay before analysis')
    comparisons = compare_replays(rows, replays)
    completion = json.loads((c.OUT / 'completion.json').read_text())
    exact = all(row['exact'] for row in comparisons)
    if (completion.get('completed') is not True or completion['case_count'] != len(rows)
            or completion['source_sha256'] != c.source_hashes()
            or completion['selection_sha256'] != c.digest(c.OUT / 'selection.json')
            or completion['replays'] != comparisons or completion['all_exact'] is not exact):
        raise RuntimeError('Completion/replay flags differ from actual archives')
    value = aggregate(rows, c.protocol(), validation={
        'source_and_archive_integrity': True,
        'causality_and_kernel_checks': all(row['causality_pass'] for row in rows + replays),
        'exact_replay': exact,
        'runtime_failure': any(row['runtime_overrun'] for row in rows + replays)})
    value.update(source_sha256=c.source_hashes(), selection_sha256=c.digest(c.OUT / 'selection.json'),
                 completion_sha256=c.digest(c.OUT / 'completion.json'))
    c.write_once(c.OUT / 'report.json', value)
    c.write_text_once(c.OUT / 'report.md', markdown(value))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'confirm', 'report', 'all'))
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.phase == 'register':
        c.register()
    if args.phase in ('confirm', 'all'):
        confirm(args.workers)
    if args.phase in ('report', 'all'):
        report()


if __name__ == '__main__':
    main()
