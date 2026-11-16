"""Independent arithmetic/provenance replay of the frozen measured-RSSI assay.

Imports no study, data, baseline or model code; performs no fits or forwards.
Run only once the registered prediction groups/report/seal are complete.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np


SCENARIOS = ('S01', 'S02', 'S03')
PHONES = ('GP10', 'S24U')
BUDGETS = (4, 8, 16)
MEMBERS = tuple(f'radiojepa-seed{i}' for i in range(3))
SOURCE_COMMIT = '5faa3d4380eb0773772d407e58f06080e36e9fa7'
PROTOCOL_SHA256 = '37c9520331486a173e948b583278ad160245076a534ab0939d67160b98f9fdb0'
STUDY = 'artifacts/measured_radio_counterfix'
PUBLIC_FIELDS = ('id', 'scenario', 'phone', 'session', 'location_id', 'partition',
                 'ap_id', 'xyz_provider', 'xyz_model', 'orientation', 'carrier_hz',
                 'android_version', 'source_archive_sha256', 'source_member',
                 'source_member_sha256')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def check(condition, label):
    if not condition:
        raise AssertionError(label)


def same(actual, expected, label):
    """Check exact structure with a tight roundoff-only floating tolerance."""
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and actual.keys() == expected.keys(), label+' keys')
        for key in expected:
            same(actual[key], expected[key], label+'/'+str(key))
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), label+' list')
        for i, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, label+'/'+str(i))
    elif isinstance(expected, (float, np.floating)):
        check(isinstance(actual, (float, int)) and math.isfinite(actual)
              and abs(actual-float(expected)) <= 2e-11, label+' floating value')
    else:
        check(actual == expected, label+' exact value')


def grid():
    configurations = [{'id': 'reference', 'method': 'reference'},
                      {'id': 'logdistance', 'method': 'logdistance'}]
    for kernel in ('rbf', 'matern12', 'matern32'):
        for length in (1., 2., 5., 10., 20., 40.):
            for amplitude in (3., 6., 12.):
                for noise in (1., 2., 4.):
                    configurations.append({'id': f'gp_{kernel}_l{length:g}_a{amplitude:g}_n{noise:g}',
                        'method': 'gp', 'kernel': kernel, 'lengthscale_m': length,
                        'amplitude_db': amplitude, 'noise_std_db': noise})
    for neighbors in (1, 3, 5, 9):
        for power in (1, 2):
            configurations.append({'id': f'idw_k{neighbors}_p{power}', 'method': 'idw',
                                   'neighbors': neighbors, 'power': power})
    check(len(configurations) == 172, 'independent numerical grid size')
    return configurations


def family(configuration):
    return ('gp_'+configuration['kernel'] if configuration['method'] == 'gp'
            else configuration['method'])


def cohort(data, partition, budget):
    queries, targets, episodes, supported = [], [], [], []
    for scenario in SCENARIOS:
        locations = data['metadata']['support_location_ids'][scenario][str(budget)]
        order = {loc: i for i, loc in enumerate(locations)}
        for phone in PHONES:
            support = sorted((r for r in data['records'] if r['scenario'] == scenario
                and r['phone'] == phone and r['partition'] == 'train'
                and r['session'] == '01' and r['location_id'] in order),
                key=lambda r: (order[r['location_id']], r['ap_id']))
            held = sorted((r for r in data['records'] if r['scenario'] == scenario
                and r['phone'] == phone and r['partition'] == partition
                and (partition == 'test' or r['session'] == '01')),
                key=lambda r: (r['session'], r['location_id'], r['ap_id']))
            check(len(support) <= 128, 'registered support maximum')
            check(not ({r['location_id'] for r in support} & {r['location_id'] for r in held}),
                  'whole-location separation')
            known = {r['ap_id'] for r in support}
            aps = {a['id']: a['xyz_model'] for a in data['metadata']['scenarios'][scenario]['aps']}
            supported.extend(r['ap_id'] in known and
                np.linalg.norm(np.asarray(r['xyz_model'])-aps[r['ap_id']]) > 0 for r in held)
            queries.extend({key: r[key] for key in PUBLIC_FIELDS} for r in held)
            targets.extend(r['rssi_dbm'] for r in held)
            episodes.append({'scenario': scenario, 'phone': phone,
                'planned_support_locations': locations,
                'actual_support_locations': sorted({r['location_id'] for r in support}),
                'support_ids': [r['id'] for r in support]})
    return queries, np.asarray(targets), episodes, np.asarray(supported, dtype=bool)


def open_group(root, data, receipt, partition, budget, configurations):
    out = root/STUDY
    path = out/'predictions'/f'{partition}-locations{budget}.json'
    evidence = read(path)
    check(evidence['partition'] == partition and evidence['budget_locations'] == budget,
          'group identity')
    check(evidence['protocol_sha256'] == PROTOCOL_SHA256, 'group protocol binding')
    check(evidence['dataset_sha256'] == receipt['dataset_sha256'], 'group dataset binding')
    check(evidence['data_receipt_sha256'] == digest(out/'data/receipt.json'), 'group data receipt binding')
    if partition == 'test':
        check(evidence['selection_sha256'] == digest(out/'selection.json'), 'TEST selection binding')
    methods = [c['id'] for c in configurations]+list(MEMBERS)
    check(evidence['methods'] == methods, 'exact methods and order')
    queries, targets, episodes, admitted = cohort(data, partition, budget)
    check(evidence['queries'] == queries and evidence['episodes'] == episodes, 'exact query/support cohort')
    expected_path = f'{STUDY}/predictions/{partition}-locations{budget}.npz'
    check(evidence['arrays_path'] == expected_path, 'prediction array path')
    npz_path = root/expected_path
    check(digest(npz_path) == evidence['arrays_sha256'], 'prediction array hash')
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    check(set(arrays) == {'target', 'mean', 'std', 'supported', 'std_known'}, 'array keys')
    check(np.array_equal(arrays['target'], targets), 'target identity and order')
    shape = (len(methods), len(queries))
    for key in ('mean', 'std', 'supported', 'std_known'):
        check(arrays[key].shape == shape, key+' axis shape')
    for key in ('supported', 'std_known'):
        check(arrays[key].dtype == bool, key+' Boolean dtype')
    check(np.array_equal(arrays['supported'], np.broadcast_to(admitted, shape)), 'common query eligibility')
    spread_methods = np.asarray([m.startswith('gp_') or m in MEMBERS for m in methods])
    check(np.array_equal(arrays['std_known'], spread_methods[:, None] & admitted), 'spread availability')
    for key, mask in (('mean', arrays['supported']), ('std', arrays['std_known'])):
        check(np.isfinite(arrays[key][mask]).all(), key+' known finite values')
        check(np.all(arrays[key][~mask] == 0), key+' masked finite zero padding')
    check(np.all(arrays['std'][arrays['std_known']] > 0), 'positive known working spreads')
    check(np.isfinite(targets).all(), 'finite observed targets')
    return evidence, arrays


def scores(evidence, arrays, session):
    """Independently reduce each member's AP errors before linear averaging."""
    phone_columns = defaultdict(list)
    for i, q in enumerate(evidence['queries']):
        if q['session'] == session and arrays['supported'][0, i]:
            phone_columns[q['scenario'], q['location_id'], q['phone']].append(i)
    by_method = {}
    for row, method in enumerate(evidence['methods']):
        locations = defaultdict(list)
        for (scenario, location, _), columns in phone_columns.items():
            errors = arrays['mean'][row, columns]-arrays['target'][columns]
            locations[scenario, location].append(math.sqrt(math.fsum(float(x*x) for x in errors)/len(errors)))
        by_method[method] = {key: math.fsum(values)/len(values) for key, values in locations.items()}
    by_method['radiojepa'] = {key: math.fsum(by_method[member][key] for member in MEMBERS)/3
                             for key in by_method[MEMBERS[0]]}
    scalar = {}
    for method, locations in by_method.items():
        means = []
        for scenario in SCENARIOS:
            values = [v for (s, _), v in locations.items() if s == scenario]
            check(bool(values), 'every scenario has scored locations')
            means.append(math.fsum(values)/len(values))
        scalar[method] = math.fsum(means)/3
    return scalar, by_method


def coverage(evidence, arrays, metadata, session):
    expected = {(s, loc['id'], p) for s in SCENARIOS
                for loc in metadata['scenarios'][s]['locations']
                if loc['partition'] == 'test' for p in PHONES}
    observed, supported = defaultdict(int), defaultdict(int)
    for i, q in enumerate(evidence['queries']):
        if q['session'] == session:
            key = q['scenario'], q['location_id'], q['phone']
            observed[key] += 1
            if arrays['supported'][0, i]:
                supported[key] += 1
    return {'planned_phone_location_units': len(expected), 'eligible_phone_location_units': len(observed),
        'supported_phone_location_units': len(supported), 'eligible_AP_summaries': sum(observed.values()),
        'supported_AP_summaries': sum(supported.values()),
        'missing_eligible_phone_location_units': [list(k) for k in sorted(expected-observed.keys())],
        'missing_supported_phone_location_units': [list(k) for k in sorted(expected-supported.keys())],
        'phone_location_AP_counts': [{'scenario': s, 'location_id': loc, 'phone': p,
            'eligible_APs': observed.get((s, loc, p), 0), 'supported_APs': supported.get((s, loc, p), 0)}
            for s, loc, p in sorted(expected)],
        'estimand': 'Conditional on observed successful known-carrier reports and an admitted support reference'}


def spread(evidence, arrays, method, session):
    indices = [evidence['methods'].index(m) for m in MEMBERS] if method == 'radiojepa' else [evidence['methods'].index(method)]
    phone = defaultdict(list)
    for j, q in enumerate(evidence['queries']):
        if q['session'] == session and arrays['std_known'][indices, j].all():
            errors = abs(arrays['mean'][indices, j]-arrays['target'][j])
            widths = arrays['std'][indices, j]
            phone[q['scenario'], q['location_id'], q['phone']].append(
                (sum(errors <= 1.96*widths)/len(indices), sum(widths)/len(indices)))
    if not phone:
        return {'working_spread_available': False}
    locations = defaultdict(list)
    for (scenario, loc, _), values in phone.items():
        locations[scenario, loc].append(np.asarray(values).mean(axis=0))
    result = np.asarray([np.asarray([np.asarray(v).mean(axis=0)
        for (s, _), v in locations.items() if s == scenario]).mean(axis=0)
        for scenario in SCENARIOS]).mean(axis=0)
    return {'working_spread_available': True, 'fraction_inside_1_96_working_std': float(result[0]),
        'mean_working_std_db': float(result[1]), 'phone_location_units': len(phone),
        'scope': 'Equal phone/location/scenario diagnostic, conditional on plug-in support reference; not calibrated hardware confidence'}


def intervals(candidate, reference, metadata):
    check(candidate.keys() == reference.keys(), 'paired location units')
    ordinary_rng, block_rng = np.random.default_rng(40517), np.random.default_rng(40518)
    ordinary, spatial, point, counts = [], [], [], {}
    for scenario in SCENARIOS:
        keys = sorted(key for key in candidate if key[0] == scenario)
        differences = np.asarray([candidate[key]-reference[key] for key in keys])
        point.append(float(differences.mean()))
        draws = ordinary_rng.integers(len(keys), size=(20000, len(keys)))
        ordinary.append(differences[draws].mean(axis=1))
        catalogue = metadata['scenarios'][scenario]['locations']
        xmin = min(row['xyz_provider'][0] for row in catalogue)
        ymin = min(row['xyz_provider'][1] for row in catalogue)
        coordinates = {row['id']: row['xyz_provider'] for row in catalogue}
        blocks = defaultdict(list)
        for key, difference in zip(keys, differences):
            x, y = coordinates[key[1]][:2]
            blocks[math.floor((x-xmin)/5), math.floor((y-ymin)/5)].append(float(difference))
        ordered = [blocks[key] for key in sorted(blocks)]
        counts[scenario] = len(ordered)
        totals = np.asarray([math.fsum(values) for values in ordered])
        sizes = np.asarray([len(values) for values in ordered])
        draws = block_rng.integers(len(ordered), size=(20000, len(ordered)))
        spatial.append(totals[draws].sum(axis=1)/sizes[draws].sum(axis=1))
    interval = np.quantile(np.asarray(ordinary).mean(axis=0), [.025, .975]).tolist()
    enough = min(counts.values()) >= 4
    block_interval = np.quantile(np.asarray(spatial).mean(axis=0), [.025, .975]).tolist() if enough else None
    return {'candidate_minus_reference_db': float(np.mean(point)), 'location_resampling_95_interval': interval,
        'spatial_5m_block_95_sensitivity': block_interval, 'block_counts': counts,
        'locations': len(candidate), 'bootstrap_draws': 20000, 'location_seed': 40517, 'block_seed': 40518,
        'conditional_lower_error_supported': bool(enough and interval[1] < 0 and block_interval[1] < 0),
        'uncertainty_scope': 'Empirical location/block resampling within fixed scenarios; weights/support/members fixed; not independent-site or calibrated spatial inference'}


def audit(root):
    out = root/STUDY
    check(digest(out/'protocol.json') == PROTOCOL_SHA256, 'predeclared protocol fingerprint')
    protocol, source, publication = (read(out/path) for path in
        ('protocol.json', 'registration/source.json', 'registration/git.json'))
    original = root/'artifacts/measured_radio'
    amendment = protocol['preparation_contract_amendment']
    check(digest(original/'protocol.json') == amendment['original_protocol_sha256'], 'preserved aborted protocol')
    check(digest(original/'failure.json') == amendment['failure_receipt_sha256'], 'preserved preparation failure')
    prior = read(original/'protocol.json')
    for key in ('namespace', 'source_sha256', 'metadata_member_sha256', 'validation_fraction',
                'support_budgets', 'rows_per_group', 'admitted_phones', 'eligibility', 'target',
                'partitions', 'support', 'geometry', 'nuisance_reference', 'neural', 'numerical_grid',
                'selection', 'metric', 'coverage', 'primary', 'secondary', 'limitations', 'immutable_outcome_policy'):
        check(protocol[key] == prior[key], 'counter repair preserves design '+key)
    check(publication['source_commit'] == SOURCE_COMMIT, 'predeclared published source')
    check(publication['protocol_sha256'] == PROTOCOL_SHA256, 'publication protocol')
    check(protocol['source_receipt_sha256'] == digest(out/'registration/source.json'), 'protocol source receipt')
    for name, expected in source['files_sha256'].items():
        check(digest(root/name) == expected, 'live frozen source '+name)
        check(digest(out/'registration/source'/name) == expected, 'capsule frozen source '+name)
    for name, expected in publication['published_files_sha256'].items():
        published = subprocess.check_output(['git', 'show', SOURCE_COMMIT+':'+name], cwd=root)
        check(hashlib.sha256(published).hexdigest() == expected and digest(root/name) == expected,
              'published Git/source binding '+name)
    check(digest(out/'registration/catalogue.json') == protocol['catalogue_sha256'], 'public catalogue hash')
    receipt = read(out/'data/receipt.json')
    check(receipt['protocol_sha256'] == PROTOCOL_SHA256, 'prepared data protocol')
    check(receipt['source_archive_sha256'] == protocol['source_sha256'], 'prepared source archive')
    check(digest(out/'data/dataset.json.gz') == receipt['dataset_sha256'], 'prepared data bytes')
    data = json.loads(gzip.decompress((out/'data/dataset.json.gz').read_bytes()))
    check(data['metadata'] == read(out/'registration/catalogue.json'), 'prepared public catalogue')
    for key, expected in protocol['expected_audit'].items():
        check(data['audit'][key] == receipt['audit'][key] == expected, 'registered eligibility '+key)
    configurations = grid()
    check(configurations == protocol['numerical_grid'], 'registered numerical grid independently reconstructed')
    selection, report = read(out/'selection.json'), read(out/'report.json')
    check(selection['protocol_sha256'] == PROTOCOL_SHA256 and selection['dataset_sha256'] == receipt['dataset_sha256'],
          'selection provenance')
    check(report['selection_sha256'] == digest(out/'selection.json'), 'report selection lock')
    groups, validation_winners, primary = [], {}, None
    checked_metrics, checked_spreads, checked_VAL_scores = 0, 0, 0
    records_by_id = {r['id']: r for r in data['records']}
    for budget in BUDGETS:
        ev, ar = open_group(root, data, receipt, 'validation', budget, configurations)
        scalar, _ = scores(ev, ar, '01')
        choices = selection['budgets'][str(budget)]
        val_scores = {config['id']: scalar[config['id']] for config in configurations}
        same(choices['all_VAL_scores'], val_scores, 'all VAL scores')
        same(choices['frozen_radiojepa_VAL_score'], scalar['radiojepa'], 'frozen neural VAL score')
        winner = min(configurations, key=lambda config: val_scores[config['id']])['id']
        families = {}
        for config in configurations:
            key = family(config)
            if key not in families or val_scores[config['id']] < val_scores[families[key]]:
                families[key] = config['id']
        check(choices['best_numerical_id'] == winner and choices['family_best_ids'] == families,
              'global/family VAL winners and registered tie order')
        check(choices['VAL_array_sha256'] == ev['arrays_sha256'], 'selection VAL array lock')
        validation_winners[str(budget)] = {'global': winner, 'families': families}
        checked_VAL_scores += len(val_scores)
        groups.append({'partition': 'validation', 'budget': budget, 'methods': len(ev['methods']),
            'queries': len(ev['queries']), 'arrays_sha256': ev['arrays_sha256'],
            'cohort_sha256': hashlib.sha256(canonical({'queries': ev['queries'], 'episodes': ev['episodes']})).hexdigest()})
        chosen = [config for config in configurations if config['id'] in families.values()]
        ev, ar = open_group(root, data, receipt, 'test', budget, chosen)
        check(len(chosen) == 6, 'six TEST numerical family controls')
        record = report['budgets'][str(budget)]
        check(record['best_numerical_id'] == winner, 'report numerical winner lock')
        check(record['raw_arrays_sha256'] == ev['arrays_sha256'], 'report raw array lock')
        check(record['eligible_query_summaries'] == len(ev['queries'])
              and record['supported_query_summaries'] == int(ar['supported'][0].sum()), 'report query coverage')
        costs = []
        for episode in ev['episodes']:
            costs.append({**episode, 'support_summary_count': len(episode['support_ids']),
                'selected_support_report_count': sum(records_by_id[i]['selected_count'] for i in episode['support_ids']),
                'eligible_support_report_count': sum(records_by_id[i]['eligible_count'] for i in episode['support_ids'])})
        check(record['episodes'] == costs, 'actual support locations and report costs')
        test_methods = [c['id'] for c in chosen]+['radiojepa']
        for session in ('01', '02'):
            scalar, by_method = scores(ev, ar, session)
            same(record['scores_by_session'][session], {method: scalar[method] for method in test_methods}, 'TEST scores')
            same(record['member_scores_by_session'][session], {method: scalar[method] for method in MEMBERS}, 'member TEST scores')
            same(record['coverage_by_session'][session], coverage(ev, ar, data['metadata'], session), 'all phone/location/AP coverage')
            same(record['spread_diagnostics_by_session'][session], {method: spread(ev, ar, method, session)
                for method in test_methods}, 'working spread diagnostics')
            checked_metrics += len(test_methods)+len(MEMBERS)
            checked_spreads += len(test_methods)
            if session == '01':
                strata = {method: {scenario: float(np.mean([v for (s, _), v in by_method[method].items()
                    if s == scenario])) for scenario in SCENARIOS} for method in test_methods}
                same(record['session01_scenario_scores'], strata, 'scenario TEST scores')
                if budget == 16:
                    primary = intervals(by_method['radiojepa'], by_method[winner], data['metadata'])
                    same(report['primary_contrast'], primary, 'both complete paired resampling intervals')
        groups.append({'partition': 'test', 'budget': budget, 'methods': len(ev['methods']),
            'queries': len(ev['queries']), 'arrays_sha256': ev['arrays_sha256'],
            'cohort_sha256': hashlib.sha256(canonical({'queries': ev['queries'], 'episodes': ev['episodes']})).hexdigest()})
    manifest = read(out/'verification/manifest.json')
    actual_files = {str(path.relative_to(root)): digest(path) for path in out.rglob('*')
                    if path.is_file() and 'verification' not in path.relative_to(out).parts}
    check(actual_files == manifest['files_sha256'], 'complete final scientific closure')
    return {'schema': 1, 'passes': True, 'source_commit': SOURCE_COMMIT,
        'protocol_sha256': PROTOCOL_SHA256, 'dataset_sha256': receipt['dataset_sha256'],
        'selection_sha256': digest(out/'selection.json'), 'report_sha256': digest(out/'report.json'),
        'seal_sha256': digest(out/'verification/manifest.json'), 'groups': groups,
        'independently_recomputed_VAL_scores': checked_VAL_scores,
        'independently_recomputed_TEST_member_and_group_scores': checked_metrics,
        'independently_recomputed_working_spread_diagnostics': checked_spreads,
        'independent_VAL_winners': validation_winners, 'independent_primary_contrast': primary,
        'scientific_files_hash_checked': len(actual_files),
        'study_imports': 0, 'model_forwards': 0, 'baseline_fits': 0,
        'scope': 'Arithmetic/cohort/provenance replay from sealed reported-RSSI summaries and raw prediction arrays; no re-fit or model output reproduction; raw-source summary audit is separate',
        'SOTA_or_hardware_qualified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args = parser.parse_args()
    result = audit(args.root.resolve())
    result['audit_script_sha256'] = digest(__file__)
    encoded = canonical(result)
    if args.output.exists():
        check(args.output.read_bytes() == encoded, 'existing independent audit identity')
    else:
        with args.output.open('xb') as stream:
            stream.write(encoded)
    print(json.dumps({'passes': True, 'output': str(args.output), 'primary': result['independent_primary_contrast']},
                     sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
