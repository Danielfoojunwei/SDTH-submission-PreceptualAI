"""Registered offline measured-RSSI transfer; no training or SmartCar action."""
from __future__ import annotations

import gzip
import hashlib
import io
import math
from pathlib import Path
import subprocess
import time

import numpy as np

from . import common as c
from .baselines import configuration_grid, predict_baseline
from .data import prepare_dataset, query_metadata
from .model import prepare_episode, predict
from ..smartcar_numpy.export import load_export
from ..smartcar_numpy.model import NumpyField


def _bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(value)


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _verify_or_write(path, value):
    if Path(path).exists():
        if c.read(path) != value:
            raise RuntimeError('Existing evidence differs from deterministic replay: '+str(path))
    else:
        c.write(path, value)
    return value


def _binding():
    receipt_path = c.OUT/'data/receipt.json'
    receipt = c.read(receipt_path)
    return {'protocol_sha256': c.digest(c.OUT/'protocol.json'),
            'dataset_sha256': receipt['dataset_sha256'],
            'data_receipt_sha256': c.digest(receipt_path)}


def _check_receipt(receipt):
    if (receipt['protocol_sha256'] != c.digest(c.OUT/'protocol.json')
            or receipt['source_archive_sha256'] != c.SOURCE_SHA256
            or receipt['source_metadata_sha256'] != c.METADATA_SHA256):
        raise RuntimeError('Prepared data receipt belongs to another registration')


def guard():
    receipt = c.read(c.OUT/'registration/source.json')
    protocol = c.read(c.OUT/'protocol.json')
    if c.digest(c.OUT/'registration/source.json') != protocol['source_receipt_sha256']:
        raise RuntimeError('Registered source receipt changed')
    for name, expected in receipt['files_sha256'].items():
        if c.digest(c.ROOT/name) != expected or c.digest(c.OUT/'registration/source'/name) != expected:
            raise RuntimeError('Registered implementation changed: '+name)
    publication = c.read(c.OUT/'registration/git.json')
    if (publication['protocol_sha256'] != c.digest(c.OUT/'protocol.json')
            or publication['source_receipt_sha256'] != protocol['source_receipt_sha256']):
        raise RuntimeError('Published protocol identity changed')
    from .registration import verify_published
    verify_published(publication['source_commit'])
    if c.digest(c.SOURCE) != protocol['source_sha256'] or c.digest(c.SOURCE_METADATA) != c.METADATA_SHA256:
        raise RuntimeError('Measured source or official license metadata changed')
    if c.digest(c.OUT/'registration/catalogue.json') != protocol['catalogue_sha256']:
        raise RuntimeError('Registered public geometry catalogue changed')
    return protocol


def prepare():
    protocol = guard()
    path, receipt_path = c.OUT/'data/dataset.json.gz', c.OUT/'data/receipt.json'
    if receipt_path.exists():
        receipt = c.read(receipt_path)
        _check_receipt(receipt)
        if c.digest(path) != receipt['dataset_sha256']:
            raise RuntimeError('Prepared measured summaries changed')
        return receipt
    if path.exists():
        raise RuntimeError('Unsealed prepared data exists; preserve it for review before resuming')
    start = time.perf_counter()
    dataset = prepare_dataset(c.SOURCE, protocol)
    if dataset['metadata'] != c.read(c.OUT/'registration/catalogue.json'):
        raise RuntimeError('Prepared catalogue differs from the metadata-only registration')
    for name, expected in protocol['expected_audit'].items():
        if dataset['audit'].get(name) != expected:
            raise RuntimeError('Measured eligibility differs from registered audit: '+name)
    _bytes(path, gzip.compress(c.encoded(dataset), mtime=0))
    receipt = {'schema': 1, 'dataset_sha256': c.digest(path), 'audit': dataset['audit'],
        'protocol_sha256': c.digest(c.OUT/'protocol.json'), 'elapsed_seconds': time.perf_counter()-start,
        'source_archive_sha256': protocol['source_sha256'], 'source_metadata_sha256': c.METADATA_SHA256,
        'summary_kind': 'mean of up to8 identity-selected per-result reported RSSI dB values',
        'real_SmartCar_measurements': False, 'physical_TX_or_receiver_calibration_identified': False}
    c.write(receipt_path, receipt)
    return receipt


def dataset():
    receipt = c.read(c.OUT/'data/receipt.json')
    _check_receipt(receipt)
    path = c.OUT/'data/dataset.json.gz'
    if c.digest(path) != receipt['dataset_sha256']:
        raise RuntimeError('Prepared dataset identity changed')
    return __import__('json').loads(gzip.decompress(path.read_bytes()))


def _models():
    path = c.ROOT/'artifacts/smartcar_numpy/models/manifest.json'
    manifest = c.read(path)
    result = {}
    for member in manifest['members']:
        arrays, _ = load_export(path.parent/member['path'], member['sha256'], member['source_checkpoint_sha256'])
        result[f"radiojepa-seed{member['seed']}"] = NumpyField(arrays)
    if list(result) != ['radiojepa-seed0', 'radiojepa-seed1', 'radiojepa-seed2']:
        raise RuntimeError('All three released members are required in fixed order')
    return result


def episodes(value, partition, budget):
    if partition not in ('validation', 'test') or budget not in c.BUDGETS:
        raise ValueError('Only registered held-location prediction groups are permitted')
    records, metadata = value['records'], value['metadata']
    result = []
    for scenario in c.SCENARIOS:
        locations = metadata['support_location_ids'][scenario][str(budget)]
        order = {loc: i for i, loc in enumerate(locations)}
        for phone in c.PHONES:
            support = [r for r in records if r['scenario'] == scenario and r['phone'] == phone
                       and r['partition'] == 'train' and r['session'] == '01' and r['location_id'] in order]
            support.sort(key=lambda r: (order[r['location_id']], r['ap_id']))
            queries = [r for r in records if r['scenario'] == scenario and r['phone'] == phone
                       and r['partition'] == partition and (partition == 'test' or r['session'] == '01')]
            queries.sort(key=lambda r: (r['session'], r['location_id'], r['ap_id']))
            public = [query_metadata(r) for r in queries]
            prepared = prepare_episode(support, public, metadata)
            if prepared['support_ids'] != [r['id'] for r in support]:
                raise RuntimeError('A model changed the common support cohort')
            result.append({'scenario': scenario, 'phone': phone, 'planned_support_locations': locations,
                'actual_support_locations': sorted({r['location_id'] for r in support}),
                'support_ids': prepared['support_ids'], 'public_queries': public,
                'private_targets': [r['rssi_dbm'] for r in queries], 'prepared': prepared})
    return result


def predictions(value, partition, budget, configs):
    neural = _models()
    methods = [r['id'] for r in configs]+list(neural)
    prepared_episodes = episodes(value, partition, budget)
    queries = [row for episode in prepared_episodes for row in episode['public_queries']]
    targets = np.asarray([y for episode in prepared_episodes for y in episode['private_targets']], dtype='f8')
    means = np.zeros((len(methods), len(queries)), dtype='f8')
    stds = np.zeros_like(means)
    supported = np.zeros(means.shape, dtype=bool)
    std_known = np.zeros_like(supported)
    calls, column = [], 0
    for episode in prepared_episodes:
        prepared, public = episode['prepared'], episode['public_queries']
        expected_ids = [r['id'] for r in public]
        for index, method in enumerate(methods):
            start = time.perf_counter()
            rows = predict_baseline(prepared, configs[index]) if index < len(configs) else predict(neural[method], prepared)
            elapsed = time.perf_counter()-start
            if [r['query_id'] for r in rows] != expected_ids:
                raise RuntimeError('A method changed the held-out query identities')
            for j, row in enumerate(rows, column):
                if row['status'] == 'prediction':
                    mu, sd = row['mean_dbm'], row['std_db']
                    if not math.isfinite(mu) or (sd is not None and (not math.isfinite(sd) or sd <= 0)):
                        raise FloatingPointError('Nonfinite measured-field prediction')
                    means[index, j], supported[index, j] = mu, True
                    if sd is not None:
                        stds[index, j], std_known[index, j] = sd, True
                elif row['status'] != 'unsupported':
                    raise RuntimeError('Unknown prediction status')
            calls.append({'scenario': episode['scenario'], 'phone': episode['phone'], 'method': method,
                          'elapsed_seconds': elapsed, 'query_rows': len(rows), 'support_rows': len(episode['support_ids'])})
        column += len(public)
    if not np.all(supported == supported[0]):
        raise RuntimeError('Methods must share the predeclared support-based query coverage')
    evidence = {'schema': 1, **_binding(), 'partition': partition, 'budget_locations': budget, 'methods': methods,
        'queries': queries, 'episodes': [{k: e[k] for k in ('scenario', 'phone', 'planned_support_locations',
            'actual_support_locations', 'support_ids')} for e in prepared_episodes], 'calls': calls,
        'array_semantics': 'Unknown means/spreads use finite zero padding ONLY where Boolean masks are false',
        'targets_are_measured_group_statistics': True, 'model_inputs_include_query_RSSI': False}
    if partition == 'test':
        evidence['selection_sha256'] = c.digest(c.OUT/'selection.json')
    arrays = {'target': targets, 'mean': means, 'std': stds, 'supported': supported, 'std_known': std_known}
    return evidence, arrays


def group_path(partition, budget):
    return c.OUT/'predictions'/f'{partition}-locations{budget}'


def save_group(evidence, arrays):
    stem = group_path(evidence['partition'], evidence['budget_locations'])
    path, index_path = stem.with_suffix('.npz'), stem.with_suffix('.json')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _bytes(path, buffer.getvalue())
    c.write(index_path, {**evidence, 'arrays_path': str(path.relative_to(c.ROOT)), 'arrays_sha256': c.digest(path)})


def load_group(partition, budget):
    stem = group_path(partition, budget)
    evidence = c.read(stem.with_suffix('.json'))
    if (any(evidence.get(k) != v for k, v in _binding().items())
            or evidence.get('partition') != partition or evidence.get('budget_locations') != budget
            or len(evidence['methods']) != len(set(evidence['methods']))
            or len(evidence['queries']) != len({q['id'] for q in evidence['queries']})):
        raise RuntimeError('Prediction archive identity differs from this registered group')
    if partition == 'test' and evidence.get('selection_sha256') != c.digest(c.OUT/'selection.json'):
        raise RuntimeError('TEST archive differs from its frozen validation selection')
    path = c.ROOT/evidence['arrays_path']
    if path != stem.with_suffix('.npz'):
        raise RuntimeError('Prediction archive path differs from its registered group')
    if c.digest(path) != evidence['arrays_sha256']:
        raise RuntimeError('Raw measured-field predictions changed')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    shape = (len(evidence['methods']), len(evidence['queries']))
    if (set(arrays) != {'target', 'mean', 'std', 'supported', 'std_known'}
            or arrays['target'].shape != (shape[1],)
            or any(arrays[k].shape != shape for k in ('mean', 'std', 'supported', 'std_known'))
            or arrays['supported'].dtype != bool or arrays['std_known'].dtype != bool
            or not np.isfinite(arrays['target']).all()
            or not np.isfinite(arrays['mean'][arrays['supported']]).all()
            or not np.isfinite(arrays['std'][arrays['std_known']]).all()
            or np.any(arrays['std_known'] & ~arrays['supported'])
            or np.any(arrays['std'][arrays['std_known']] <= 0)
            or not np.all(arrays['supported'] == arrays['supported'][0])):
        raise RuntimeError('Raw prediction axes or known-value masks disagree')
    return evidence, arrays


def verify_cohort(value, evidence, arrays, configs):
    """Replay only public cohort construction; no fit or prediction on resume."""
    expected = episodes(value, evidence['partition'], evidence['budget_locations'])
    public = [q for e in expected for q in e['public_queries']]
    targets = np.asarray([y for e in expected for y in e['private_targets']], dtype='f8')
    cohort = [{k: e[k] for k in ('scenario', 'phone', 'planned_support_locations',
                'actual_support_locations', 'support_ids')} for e in expected]
    supported = np.asarray([r['status'] == 'prediction' for e in expected for r in e['prepared']['statuses']])
    if (evidence['queries'] != public or evidence['episodes'] != cohort
            or not np.array_equal(targets, arrays['target'])
            or not np.array_equal(arrays['supported'][0], supported)
            or evidence['methods'] != [r['id'] for r in configs]+[f'radiojepa-seed{s}' for s in c.SEEDS]):
        raise RuntimeError('Saved predictions differ from the exact prepared support/query cohort')


def location_metrics(evidence, arrays, method, session='01'):
    """RMSE within APs, then equal phones, retaining each whole-location unit."""
    if method == 'radiojepa':
        indices = [evidence['methods'].index(f'radiojepa-seed{s}') for s in c.SEEDS]
    else:
        indices = [evidence['methods'].index(method)]
    phones = {}
    for column, query in enumerate(evidence['queries']):
        if query['session'] == session and arrays['supported'][0, column]:
            phones.setdefault((query['scenario'], query['location_id'], query['phone']), []).append(column)
    locations = {}
    for (scenario, location, phone), columns in phones.items():
        errors = arrays['mean'][indices][:, columns]-arrays['target'][None, columns]
        value = float(np.sqrt(np.mean(errors**2, axis=1)).mean())
        locations.setdefault((scenario, location), []).append(value)
    return {key: float(np.mean(values)) for key, values in locations.items()}


def metric(evidence, arrays, method, session='01'):
    rows = location_metrics(evidence, arrays, method, session)
    if set(s for s, _ in rows) != set(c.SCENARIOS):
        raise RuntimeError('Every declared scenario must retain scored location coverage')
    return float(np.mean([np.mean([value for (s, _), value in rows.items() if s == scenario]) for scenario in c.SCENARIOS]))


def _family(config):
    return 'gp_'+config['kernel'] if config['method'] == 'gp' else config['method']


def validation(value):
    configs = configuration_grid()
    for budget in c.BUDGETS:
        if group_path('validation', budget).with_suffix('.json').exists():
            evidence, arrays = load_group('validation', budget)
            verify_cohort(value, evidence, arrays, configs)
            continue
        if group_path('validation', budget).with_suffix('.npz').exists():
            raise RuntimeError('Unsealed prediction arrays exist; preserve them before resuming')
        save_group(*predictions(value, 'validation', budget, configs))


def select():
    path = c.OUT/'selection.json'
    configs, selections = configuration_grid(), {}
    for budget in c.BUDGETS:
        evidence, arrays = load_group('validation', budget)
        if evidence['methods'][:len(configs)] != [r['id'] for r in configs]:
            raise RuntimeError('Validation must retain the entire registered numerical grid')
        scores = {r['id']: metric(evidence, arrays, r['id']) for r in configs}
        best = min(configs, key=lambda r: scores[r['id']])
        family = {}
        for config in configs:
            name = _family(config)
            if name not in family or scores[config['id']] < scores[family[name]]:
                family[name] = config['id']
        selections[str(budget)] = {'best_numerical_id': best['id'], 'family_best_ids': family,
            'all_VAL_scores': scores, 'VAL_array_sha256': evidence['arrays_sha256'],
            'frozen_radiojepa_VAL_score': metric(evidence, arrays, 'radiojepa')}
    result = {'schema': 1, 'selection_only_from_validation_session01': True,
        'no_TEST_targets_for_selection': True, 'no_neural_weight_or_member_selection': True,
        'dataset_sha256': c.read(c.OUT/'data/receipt.json')['dataset_sha256'],
        'protocol_sha256': c.digest(c.OUT/'protocol.json'), 'budgets': selections}
    return _verify_or_write(path, result)


def testing(value, selection):
    grid = configuration_grid()
    for budget in c.BUDGETS:
        chosen = set(selection['budgets'][str(budget)]['family_best_ids'].values())
        configs = [r for r in grid if r['id'] in chosen]
        if group_path('test', budget).with_suffix('.json').exists():
            evidence, arrays = load_group('test', budget)
            verify_cohort(value, evidence, arrays, configs)
            continue
        if group_path('test', budget).with_suffix('.npz').exists():
            raise RuntimeError('Unsealed prediction arrays exist; preserve them before resuming')
        save_group(*predictions(value, 'test', budget, configs))


def paired_intervals(candidate, reference, metadata):
    if set(candidate) != set(reference):
        raise ValueError('Paired whole-location cohorts must match')
    point = float(np.mean([np.mean([candidate[k]-reference[k] for k in sorted(candidate) if k[0] == scenario])
                          for scenario in c.SCENARIOS]))
    rng, block_rng = np.random.default_rng(40517), np.random.default_rng(40518)
    draws, block_draws, counts = [], [], {}
    for scenario in c.SCENARIOS:
        keys = sorted(k for k in candidate if k[0] == scenario)
        if not keys:
            raise ValueError('Every fixed scenario must retain matched locations')
        values = np.asarray([candidate[k]-reference[k] for k in keys])
        indices = rng.integers(0, len(keys), size=(20000, len(keys)))
        draws.append(values[indices].mean(axis=1))
        locs = metadata['scenarios'][scenario]['locations']
        origin = np.min(np.asarray([r['xyz_provider'][:2] for r in locs]), axis=0)
        coordinates = {r['id']: np.asarray(r['xyz_provider'][:2]) for r in locs}
        blocks = {}
        for key, delta in zip(keys, values):
            cell = tuple(np.floor((coordinates[key[1]]-origin)/5.).astype(int))
            blocks.setdefault(cell, []).append(delta)
        counts[scenario] = len(blocks)
        sums = np.asarray([sum(v) for _, v in sorted(blocks.items())])
        sizes = np.asarray([len(v) for _, v in sorted(blocks.items())])
        indices = block_rng.integers(0, len(sums), size=(20000, len(sums)))
        block_draws.append(sums[indices].sum(axis=1)/sizes[indices].sum(axis=1))
    ordinary = np.mean(draws, axis=0)
    sensitivity = np.mean(block_draws, axis=0)
    interval = [float(x) for x in np.quantile(ordinary, [.025, .975])]
    enough = min(counts.values()) >= 4
    block_interval = [float(x) for x in np.quantile(sensitivity, [.025, .975])] if enough else None
    return {'candidate_minus_reference_db': point, 'location_resampling_95_interval': interval,
        'spatial_5m_block_95_sensitivity': block_interval, 'block_counts': counts,
        'locations': len(candidate), 'bootstrap_draws': 20000, 'location_seed': 40517, 'block_seed': 40518,
        'conditional_lower_error_supported': bool(enough and interval[1] < 0 and block_interval[1] < 0),
        'uncertainty_scope': 'Empirical location/block resampling within fixed scenarios; weights/support/members fixed; not independent-site or calibrated spatial inference'}


def coverage(evidence, arrays, metadata, session):
    """Expose every missing phone/location unit rather than silently dropping it."""
    expected = {(s, loc['id'], p) for s in c.SCENARIOS for loc in metadata['scenarios'][s]['locations']
                if loc['partition'] == 'test' for p in c.PHONES}
    observed, admitted = {}, {}
    for column, q in enumerate(evidence['queries']):
        if q['session'] != session:
            continue
        key = q['scenario'], q['location_id'], q['phone']
        observed[key] = observed.get(key, 0)+1
        if arrays['supported'][0, column]:
            admitted[key] = admitted.get(key, 0)+1
    return {'planned_phone_location_units': len(expected),
        'eligible_phone_location_units': len(observed), 'supported_phone_location_units': len(admitted),
        'eligible_AP_summaries': sum(observed.values()), 'supported_AP_summaries': sum(admitted.values()),
        'missing_eligible_phone_location_units': [list(k) for k in sorted(expected-set(observed))],
        'missing_supported_phone_location_units': [list(k) for k in sorted(expected-set(admitted))],
        'phone_location_AP_counts': [{'scenario': k[0], 'location_id': k[1], 'phone': k[2],
            'eligible_APs': observed.get(k, 0), 'supported_APs': admitted.get(k, 0)} for k in sorted(expected)],
        'estimand': 'Conditional on observed successful known-carrier reports and an admitted support reference'}


def spread_diagnostics(evidence, arrays, method, session):
    """Descriptive point-statistic spread; no plug-in reference uncertainty."""
    indices = ([evidence['methods'].index(f'radiojepa-seed{s}') for s in c.SEEDS]
               if method == 'radiojepa' else [evidence['methods'].index(method)])
    units = {}
    for j, q in enumerate(evidence['queries']):
        if q['session'] != session or not arrays['std_known'][indices, j].all():
            continue
        error = np.abs(arrays['mean'][indices, j]-arrays['target'][j])
        width = arrays['std'][indices, j]
        units.setdefault((q['scenario'], q['location_id'], q['phone']), []).append(
            [float(np.mean(error <= 1.96*width)), float(np.mean(width))])
    if not units:
        return {'working_spread_available': False}
    locations = {}
    for (s, loc, _), rows in units.items():
        locations.setdefault((s, loc), []).append(np.mean(rows, axis=0))
    scenario = [np.mean([np.mean(v, axis=0) for (s, _), v in locations.items() if s == sc], axis=0)
                for sc in c.SCENARIOS]
    result = np.mean(scenario, axis=0)
    return {'working_spread_available': True, 'fraction_inside_1_96_working_std': float(result[0]),
        'mean_working_std_db': float(result[1]), 'phone_location_units': len(units),
        'scope': 'Equal phone/location/scenario diagnostic, conditional on plug-in support reference; not calibrated hardware confidence'}


def report(value, selection):
    output = {'schema': 1, 'scope': 'Offline held-location mean reported-RSSI prediction with partial geometry inputs',
        'no_new_training': True, 'no_hardware_or_SOTA_qualification': True,
        'source_audit': value['audit'], 'selection_sha256': c.digest(c.OUT/'selection.json'), 'budgets': {}}
    for budget in c.BUDGETS:
        evidence, arrays = load_group('test', budget)
        methods = [m for m in evidence['methods'] if not m.startswith('radiojepa-')]+['radiojepa']
        best = selection['budgets'][str(budget)]['best_numerical_id']
        scores = {session: {method: metric(evidence, arrays, method, session) for method in methods}
                  for session in ('01', '02')}
        strata = {}
        for method in methods:
            rows = location_metrics(evidence, arrays, method)
            strata[method] = {scenario: float(np.mean([v for (s, _), v in rows.items() if s == scenario])) for scenario in c.SCENARIOS}
        record_by_id = {r['id']: r for r in value['records']}
        costs = [{**e, 'support_summary_count': len(e['support_ids']),
                  'selected_support_report_count': sum(record_by_id[i]['selected_count'] for i in e['support_ids']),
                  'eligible_support_report_count': sum(record_by_id[i]['eligible_count'] for i in e['support_ids'])}
                 for e in evidence['episodes']]
        member_scores = {session: {f'radiojepa-seed{s}': metric(evidence, arrays, f'radiojepa-seed{s}', session)
            for s in c.SEEDS} for session in ('01', '02')}
        output['budgets'][str(budget)] = {'best_numerical_id': best, 'scores_by_session': scores,
            'session01_scenario_scores': strata, 'eligible_query_summaries': len(evidence['queries']),
            'supported_query_summaries': int(arrays['supported'][0].sum()),
            'raw_arrays_sha256': evidence['arrays_sha256'], 'episodes': costs,
            'member_scores_by_session': member_scores,
            'coverage_by_session': {session: coverage(evidence, arrays, value['metadata'], session) for session in ('01', '02')},
            'spread_diagnostics_by_session': {session: {method: spread_diagnostics(evidence, arrays, method, session)
                for method in methods} for session in ('01', '02')}}
        if budget == 16:
            output['primary_contrast'] = paired_intervals(location_metrics(evidence, arrays, 'radiojepa'),
                location_metrics(evidence, arrays, best), value['metadata'])
    return _verify_or_write(c.OUT/'report.json', output)


def seal():
    guard()
    files = {str(p.relative_to(c.ROOT)): c.digest(p) for p in c.OUT.rglob('*')
             if p.is_file() and 'verification' not in p.relative_to(c.OUT).parts}
    result = {'schema': 1, 'source_archive_sha256': c.SOURCE_SHA256,
        'source_metadata_sha256': c.METADATA_SHA256, 'files_sha256': files,
        'real_SmartCar_actions': 0, 'new_training': 0, 'SOTA_established': False}
    return _verify_or_write(c.OUT/'verification/manifest.json', result)


def run():
    guard()
    prepare()
    value = dataset()
    validation(value)
    selection = select()
    testing(value, selection)
    result = report(value, selection)
    seal()
    return result
