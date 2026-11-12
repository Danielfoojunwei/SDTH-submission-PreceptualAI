"""Finite source-published real-phone transfer benchmark and raw audit.

Three seeds vary the genuine target-support sample. Deterministic kNN/ridge
references are not relabelled as independently pretrained neural runs.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time

import numpy as np

from ..spectral_scene import common as c
from . import data, calibration as models

ROOT = c.ROOT
OUT = ROOT / 'artifacts/device_transfer'
PROTOCOL = OUT / 'protocol.json'
INVENTORY = OUT / 'inventory.json'
PRIOR = ROOT / 'artifacts/radiojepa_spectral/verification/manifest.json'
PRIOR_SHA256 = '53f329a2ce82d6498116fa098b1231ad31c5826be4435b4610c713701bcf598c'
_PUBLISHED = set()
_FIT_AUDITED = set()
FIT_ARRAYS = {'source_rows','source_rssi_dbm','source_heard','source_xyz_m','ridge_mean',
    'ridge_scale','ridge_coefficient','ridge_feature_mean','ridge_target_mean','support_order',
    'global_offsets_db','ap_offsets_db','ap_pair_counts','pair_counts'}


def protocol():
    return c.read_json(PROTOCOL)


def _prior(full=False):
    if c.digest(PRIOR) != PRIOR_SHA256:
        raise RuntimeError('The completed spectral manifest changed')
    value = c.read_json(PRIOR)
    if len(value['source_sha256']) != 475 or len(value['evidence_sha256']) != 5161:
        raise RuntimeError('The completed spectral closure cardinality changed')
    for name, expected in value['source_sha256'].items():
        if c.digest(c.safe_path(name)) != expected:
            raise RuntimeError('Prior scientific source changed: ' + name)
    if full:
        from ..spectral_scene.verification import verify_closure
        if verify_closure().get('passes') is not True:
            raise RuntimeError('Previous spectral closure failed')
    return {'manifest_sha256': PRIOR_SHA256, 'sources': 475, 'evidence': 5161}


def source_hashes():
    prior = c.read_json(PRIOR)
    result = dict(prior['source_sha256'])
    files = list((ROOT / 'src/wirelessjev/device_transfer').glob('*.py'))
    files += list((ROOT / 'tests').glob('test_device_transfer*.py'))
    files += [ROOT / 'docs/device_transfer/PLAN.md', PROTOCOL, INVENTORY,
              OUT / '.gitignore', PRIOR]
    for path in files:
        name = str(path.relative_to(ROOT))
        result[name] = c.digest(c.safe_path(name))
    return dict(sorted(result.items()))


def prepare_inventory(archive=None):
    """Training-only metadata and support ordering; no baseline/calibration fit."""
    p = protocol()
    training = data.load_training(archive, expected=p['archive_sha256'])
    if len(training.xyz_m) != p['expected_training_rows']:
        raise RuntimeError('Official training cardinality differs')
    value = {'schema': 1, 'archive_sha256': p['archive_sha256'],
        'source_url': data.SOURCE_URL, 'metadata': data.metadata_inventory(training),
        'expected_test_rows_from_prior_public_acquisition_audit': p['expected_test_rows_by_device'],
        'support_orders': {str(device): {str(seed): data.support_order(training, device, seed,
            max(p['support_sizes']), minimum_shared=p['minimum_shared_observed_aps']).tolist()
            for seed in p['support_seeds']} for device in p['target_devices']},
        'test_rf_or_xyz_opened': False, 'scientific_fits_performed': 0}
    c.write_once(INVENTORY, value)
    return value


def preflight():
    before = source_hashes()
    prior = _prior(full=True)
    command = ['uv', 'run', '--frozen', 'pytest', '-q'] + sorted(
        str(path.relative_to(ROOT)) for path in (ROOT / 'tests').glob('test_device_transfer*.py'))
    completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    after = source_hashes()
    if completed.returncode or before != after:
        raise RuntimeError('Device-transfer tests failed or source changed during preflight:\n'
                           + completed.stdout.decode(errors='replace'))
    log = OUT / 'preflight/tests.txt'
    c.write_bytes_once(log, completed.stdout)
    value = {'schema': 1, 'exit_code': completed.returncode, 'command': command,
        'sources_before_sha256': before, 'sources_after_sha256': after,
        'test_file_sha256': {name: sha for name, sha in before.items() if name.startswith('tests/test_device_transfer')},
        'thread_environment': {name: os.environ.get(name) for name in protocol()['thread_limits']},
        'prior_evidence': prior, 'stdout_path': str(log.relative_to(ROOT)), 'stdout_sha256': c.digest(log)}
    c.write_once(OUT / 'preflight/tests.json', value)
    return value


def register():
    with c.lock(OUT / 'registration.lock'):
        if (OUT / 'registration/source.json').exists():
            return check_sources(require_published=False)
        prior = _prior(full=True)
        for name in ('fits', 'predictions', 'report.json', 'verification'):
            path = OUT / name
            if path.exists() and (path.is_file() or any(path.rglob('*'))):
                raise RuntimeError('Registration must precede real transfer fits/test outputs')
        sources = source_hashes()
        tests = c.read_json(OUT / 'preflight/tests.json')
        if (tests.get('exit_code') != 0 or tests.get('sources_before_sha256') != sources
                or tests.get('sources_after_sha256') != sources or tests.get('prior_evidence') != prior
                or tests.get('test_file_sha256') != {name: sha for name, sha in sources.items() if name.startswith('tests/test_device_transfer')}
                or tests.get('thread_environment') != protocol()['thread_limits']
                or c.digest(c.safe_path(tests['stdout_path'])) != tests['stdout_sha256']):
            raise RuntimeError('Final fixture proof differs from registered sources')
        capsule = OUT / 'registration/source'
        for name, expected in sources.items():
            target = c.safe_path(name, capsule)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(c.safe_path(name), target)
            if c.digest(target) != expected:
                raise RuntimeError('Source changed during capsule copy')
        value = {'schema': 1, 'source_sha256': sources, 'previous': prior,
            'archive_sha256': protocol()['archive_sha256'], 'inventory_sha256': c.digest(INVENTORY),
            'preflight_sha256': {str((OUT / 'preflight/tests.json').relative_to(ROOT)): c.digest(OUT / 'preflight/tests.json'),
                                  tests['stdout_path']: tests['stdout_sha256']},
            'scientific_fits_or_localization_metrics_before_registration': False,
            'software_unit_fixtures_excluded_from_scientific_fit_count': True,
            'prior_public_data_audit_disclosed_in_protocol': True}
        c.write_once(OUT / 'registration/source.json', value)
        return value


def check_sources(*, require_published=True):
    receipt = c.read_json(OUT / 'registration/source.json')
    if _prior() != receipt['previous'] or receipt['source_sha256'] != source_hashes():
        raise RuntimeError('Registered real-transfer source identity changed')
    capsule = OUT / 'registration/source'
    if {str(path.relative_to(capsule)) for path in capsule.rglob('*') if path.is_file()} != set(receipt['source_sha256']):
        raise RuntimeError('Real-transfer source capsule membership changed')
    for name, expected in receipt['source_sha256'].items():
        if c.digest(c.safe_path(name, capsule)) != expected:
            raise RuntimeError('Real-transfer source capsule bytes changed')
    for name, expected in receipt['preflight_sha256'].items():
        if c.digest(c.safe_path(name)) != expected:
            raise RuntimeError('Preflight proof changed')
    if not require_published:
        return receipt
    publication = c.read_json(OUT / 'registration/git.json')
    commit = publication.get('commit', '')
    if (not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit)
            or publication.get('remote_observed_commit') != commit or not publication.get('branch')
            or not publication.get('remote') or publication.get('source_receipt_sha256') != c.digest(OUT / 'registration/source.json')):
        raise RuntimeError('Publish and observe registered transfer source before scientific execution')
    key = (commit, c.value_digest(publication))
    if key not in _PUBLISHED:
        expected = dict(receipt['source_sha256']) | receipt['preflight_sha256']
        expected[str((OUT / 'registration/source.json').relative_to(ROOT))] = publication['source_receipt_sha256']
        expected.update({str((capsule / name).relative_to(ROOT)): sha for name, sha in receipt['source_sha256'].items()})
        proc = subprocess.Popen(['git', 'archive', '--format=tar', commit, *expected], cwd=ROOT, stdout=subprocess.PIPE)
        actual = set()
        with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
            for member in archive:
                if member.isdir():
                    continue
                if (not member.isfile() or member.name not in expected
                        or hashlib.sha256(archive.extractfile(member).read()).hexdigest() != expected[member.name]):
                    raise RuntimeError('Published transfer source differs')
                actual.add(member.name)
        if proc.wait() or actual != set(expected):
            raise RuntimeError('Published transfer source is incomplete')
        _PUBLISHED.add(key)
    return receipt


def _npz(path, arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    c.write_bytes_once(path, stream.getvalue())
    return {'path': str(path.relative_to(ROOT)), 'sha256': c.digest(path)}


def _load_npz(reference):
    path = c.safe_path(reference['path'])
    if c.digest(path) != reference['sha256']:
        raise RuntimeError('Real-transfer numeric evidence changed')
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def _fold_fit(training, device, p, inventory):
    source_rows = np.flatnonzero(training.fingerprints.device_ids != device)
    source = training.fingerprints.subset(source_rows)
    ridge = models.RidgeLocalizer(source, training.xyz_m[source_rows], alpha=p['ridge_alpha'])
    shape = (len(p['support_seeds']), len(p['support_sizes']))
    state = {'source_rows': source_rows, 'source_rssi_dbm': source.rssi_dbm,
        'source_heard': source.heard, 'source_xyz_m': training.xyz_m[source_rows],
        'ridge_mean': ridge.mean, 'ridge_scale': ridge.scale, 'ridge_coefficient': ridge.coefficient,
        'ridge_feature_mean': ridge.feature_mean,
        'ridge_target_mean': ridge.target_mean, 'support_order': np.array([
            inventory['support_orders'][str(device)][str(seed)] for seed in p['support_seeds']], dtype='i8'),
        'global_offsets_db': np.zeros(shape),
        'ap_offsets_db': np.zeros(shape + (source.heard.shape[1],)),
        'ap_pair_counts': np.zeros(shape + (source.heard.shape[1],), 'i8'),
        'pair_counts': np.zeros(shape, 'i8')}
    for si, seed in enumerate(p['support_seeds']):
        for ni, n in enumerate(p['support_sizes']):
            offset = models.fit_offsets(training, device, state['support_order'][si, :n],
                                        shrinkage=p['ap_shrinkage_pseudocount'])
            state['global_offsets_db'][si, ni] = offset.global_offset_db
            state['ap_offsets_db'][si, ni] = offset.ap_offset_db
            state['ap_pair_counts'][si, ni] = offset.per_ap_pair_counts
            state['pair_counts'][si, ni] = offset.paired_ap_observations
    return state


def fit_all(training, p, inventory):
    fits = []
    for device in p['target_devices']:
        check_sources()
        path = OUT / 'fits' / f'device-{device}.npz'
        receipt_path = OUT / 'fits' / f'device-{device}.json'
        if receipt_path.exists():
            receipt = c.read_json(receipt_path)
            _validate_fit_receipt(receipt, device, training, p, inventory)
            fits.append(receipt)
            continue
        begin = time.perf_counter_ns()
        state = _fold_fit(training, device, p, inventory)
        receipt = {'device': device, 'state': _npz(path, state),
            'elapsed_seconds': (time.perf_counter_ns()-begin)/1e9,
            'source_training_rows': len(state['source_rows']), 'target_test_rf_opened': False,
            'target_test_xyz_opened': False, 'independent_neural_training_runs': 0,
            'source_sha256': source_hashes(), 'inventory_sha256': c.digest(INVENTORY)}
        c.write_once(receipt_path, receipt)
        _validate_fit_receipt(receipt, device, training, p, inventory)
        fits.append(receipt)
    value = {'schema': 1, 'completed': True, 'fits': fits,
        'deterministic_source_ridge_fits': len(p['target_devices']),
        'nonzero_support_calibration_fits': len(p['target_devices'])*len(p['support_seeds'])*sum(n > 0 for n in p['support_sizes']), 'new_neural_fits': 0,
        'test_rf_or_xyz_opened': False, 'source_sha256': source_hashes()}
    c.write_once(OUT / 'fits.json', value)
    return value


def _validate_fit_receipt(receipt, device, training, p, inventory):
    expected = str((OUT / 'fits' / f'device-{device}.npz').relative_to(ROOT))
    if (receipt.get('device') != device or receipt.get('source_sha256') != source_hashes()
            or receipt.get('inventory_sha256') != c.digest(INVENTORY)
            or receipt.get('target_test_rf_opened') is not False
            or receipt.get('target_test_xyz_opened') is not False
            or receipt.get('state', {}).get('path') != expected):
        raise RuntimeError('Cached fit device/source/inventory/scope binding differs')
    state = _load_npz(receipt['state'])
    if set(state) != FIT_ARRAYS:
        raise RuntimeError('Unexpected fit state array membership')
    key = (device, c.value_digest(receipt), c.digest(INVENTORY), c.value_digest(p))
    # Every reuse still rehashes the complete NPZ and current source/input bytes.
    # Only the already-verified normal equations are memoized for this identity.
    if key not in _FIT_AUDITED:
        audit_fit(training, device, state, p, inventory)
        _FIT_AUDITED.add(key)
    return state


def require_fits_before_test(archive=None):
    """No checksum-consistent stale fit may open evaluation fingerprints."""
    check_sources()
    p, inventory = protocol(), c.read_json(INVENTORY)
    training = data.load_training(archive, expected=p['archive_sha256'], expected_links=p['expected_links'])
    value = c.read_json(OUT / 'fits.json')
    if (value.get('completed') is not True or value.get('source_sha256') != source_hashes()
            or value.get('test_rf_or_xyz_opened') is not False
            or [row.get('device') for row in value.get('fits', [])] != p['target_devices']):
        raise RuntimeError('Complete current source fits before reading test fingerprints')
    for row in value['fits']:
        device = row['device']
        if c.read_json(OUT / 'fits' / f'device-{device}.json') != row:
            raise RuntimeError('Cached aggregate/per-fold fit receipts differ')
        _validate_fit_receipt(row, device, training, p, inventory)
    return value


def _predict_fold(training, test, state, p):
    source = training.fingerprints.subset(state['source_rows'])
    knn = models.MaskedKNN(source, state['source_xyz_m'], neighbors=p['neighbors'],
                           minimum_shared=p['minimum_shared_observed_aps'])
    baseline = knn.predict(test)
    normalized = np.where(test.heard, (test.rssi_dbm-state['ridge_mean'])/state['ridge_scale'], 0.)
    ridge_x = np.concatenate((normalized, test.heard.astype('f8')), 1)
    arrays = {'test_rows': test.row0, 'base_xyz': baseline['xyz_m'],
              'base_supported': baseline['has_shared_ap_support'],
              'ridge_xyz': (ridge_x-state['ridge_feature_mean']) @ state['ridge_coefficient'] + state['ridge_target_mean']}
    for si, seed in enumerate(p['support_seeds']):
        for ni, n in enumerate(p['support_sizes']):
            if n == 0:
                continue  # Identical zero-support references are computed once.
            support = state['support_order'][si, :n]
            augmented_rows = np.concatenate((state['source_rows'], support))
            augmented = models.MaskedKNN(training.fingerprints.subset(augmented_rows),
                training.xyz_m[augmented_rows], neighbors=p['neighbors'],
                minimum_shared=p['minimum_shared_observed_aps'])
            for name, predictor, query in (
                ('support', augmented, test),
                ('global', knn, models.corrected(test, state['global_offsets_db'][si, ni])),
                ('ap', knn, models.corrected(test, state['ap_offsets_db'][si, ni]))):
                prediction = predictor.predict(query)
                key = f'{name}_s{seed}_n{n}'
                arrays[key+'_xyz'] = prediction['xyz_m']
                arrays[key+'_supported'] = prediction['has_shared_ap_support']
    return arrays


def predict_all(training, fits, archive, p):
    if require_fits_before_test(archive) != fits:
        raise RuntimeError('Current fitting phase differs from persisted fits')
    test = data.load_test_features(archive, expected=p['archive_sha256'], expected_links=p['expected_links'])
    if len(test.row0) != p['expected_testing_rows']:
        raise RuntimeError('Official testing cardinality differs')
    predictions = []
    for fit in fits['fits']:
        device = fit['device']
        query = test.subset(test.device_ids == device)
        if len(query.row0) != p['expected_test_rows_by_device'][str(device)]:
            raise RuntimeError('Official test phone membership differs')
        path = OUT / 'predictions' / f'device-{device}.npz'
        receipt_path = OUT / 'predictions' / f'device-{device}.json'
        if receipt_path.exists():
            receipt = c.read_json(receipt_path)
            _validate_prediction_receipt(receipt, device, fit, query, p)
            predictions.append(receipt)
            continue
        begin = time.perf_counter_ns()
        arrays = _predict_fold(training, query, _load_npz(fit['state']), p)
        receipt = {'device': device, 'predictions': _npz(path, arrays),
            'elapsed_seconds': (time.perf_counter_ns()-begin)/1e9,
            'test_rows': query.row0.tolist(), 'private_xyz_opened': False,
            'fit_receipt_sha256': c.digest(OUT / 'fits' / f'device-{device}.json'),
            'source_sha256': source_hashes()}
        c.write_once(receipt_path, receipt)
        _validate_prediction_receipt(receipt, device, fit, query, p)
        predictions.append(receipt)
    value = {'schema': 1, 'completed': True, 'private_xyz_opened': False,
        'predictions': predictions, 'fits_sha256': c.digest(OUT / 'fits.json'),
        'archive_sha256': p['archive_sha256'], 'source_sha256': source_hashes()}
    c.write_once(OUT / 'before-labels.json', value)
    return test, value


def _validate_prediction_receipt(receipt, device, fit, query, p):
    expected = str((OUT / 'predictions' / f'device-{device}.npz').relative_to(ROOT))
    if (receipt.get('device') != device or receipt.get('private_xyz_opened') is not False
            or receipt.get('source_sha256') != source_hashes()
            or receipt.get('test_rows') != query.row0.tolist()
            or receipt.get('fit_receipt_sha256') != c.digest(OUT / 'fits' / f'device-{device}.json')
            or receipt.get('predictions', {}).get('path') != expected):
        raise RuntimeError('Cached prediction device/source/fit/test-row binding differs')
    saved = _load_npz(receipt['predictions'])
    keys = {'test_rows','base_xyz','base_supported','ridge_xyz'}
    for seed in p['support_seeds']:
        for n in p['support_sizes']:
            if n:
                for name in ('support','global','ap'):
                    keys.update((f'{name}_s{seed}_n{n}_xyz', f'{name}_s{seed}_n{n}_supported'))
    if set(saved) != keys or not np.array_equal(saved['test_rows'], query.row0):
        raise RuntimeError('Cached prediction membership differs from the current phone')
    for name, array in saved.items():
        if name == 'test_rows':
            continue
        shape = (len(query.row0),3) if name.endswith('xyz') else (len(query.row0),)
        if (array.shape != shape or not np.isfinite(array).all()
                or (name.endswith('supported') and array.dtype.kind != 'b')):
            raise RuntimeError('Cached prediction axes/numerics differ')
    return saved


def require_predictions_before_labels(archive=None):
    fits = require_fits_before_test(archive)
    value = c.read_json(OUT / 'before-labels.json')
    p = protocol()
    if (value.get('completed') is not True or value.get('private_xyz_opened') is not False
            or value['source_sha256'] != source_hashes()
            or value['fits_sha256'] != c.digest(OUT / 'fits.json')
            or sorted(row['device'] for row in value['predictions']) != p['target_devices']):
        raise RuntimeError('Persist the complete registered predictions before opening XYZ')
    test = data.load_test_features(archive, expected=p['archive_sha256'], expected_links=p['expected_links'])
    by_device = {row['device']:row for row in fits['fits']}
    for row in value['predictions']:
        device = row['device']
        if c.read_json(OUT / 'predictions' / f'device-{device}.json') != row:
            raise RuntimeError('Pre-label/per-device prediction receipts differ')
        query = test.subset(test.device_ids == device)
        _validate_prediction_receipt(row, device, by_device[device], query, p)
    return value


def metric(predicted, truth, supported):
    predicted, truth, supported = np.asarray(predicted), np.asarray(truth), np.asarray(supported)
    if (predicted.shape != truth.shape or truth.ndim != 2 or truth.shape[1] != 3
            or supported.shape != (len(truth),) or not len(truth)
            or not np.isfinite(predicted).all() or not np.isfinite(truth).all()):
        raise RuntimeError('Malformed localization prediction/label evidence')
    error = np.linalg.norm(predicted[:, :2]-truth[:, :2], axis=1)
    _, group = np.unique(truth, axis=0, return_inverse=True)
    balanced = np.bincount(group, weights=error)/np.bincount(group)
    return {'mean_2d_error_m': float(error.mean()), 'median_2d_error_m': float(np.median(error)),
        'p95_2d_error_m': float(np.quantile(error, .95)),
        'location_balanced_mean_2d_error_m': float(balanced.mean()),
        'mean_3d_error_m': float(np.linalg.norm(predicted-truth, axis=1).mean()),
        'shared_ap_fallback_fraction': float((~supported.astype(bool)).mean())}


def _prediction_key(model, n, seed):
    if model == 'source_ridge':
        return 'ridge_xyz', None
    if model == 'source_knn' or n == 0:
        return 'base_xyz', 'base_supported'
    prefix = {'source_knn_with_target_support': 'support', 'global_offset_knn': 'global',
              'ap_shrinkage_knn': 'ap'}[model]
    return f'{prefix}_s{seed}_n{n}_xyz', f'{prefix}_s{seed}_n{n}_supported'


def evaluate(training, test, prediction_receipt, xyz, p):
    if set(data.position_keys(training.xyz_m)) & set(data.position_keys(xyz)):
        raise RuntimeError('Official train/test positions overlap; no held-location claim is permitted')
    cases = []
    for receipt in prediction_receipt['predictions']:
        device = receipt['device']; arrays = _load_npz(receipt['predictions'])
        truth = xyz[arrays['test_rows']]
        for si, seed in enumerate(p['support_seeds']):
            for n in p['support_sizes']:
                support = c.read_json(INVENTORY)['support_orders'][str(device)][str(seed)][:n]
                for model in p['models']:
                    key, supported_key = _prediction_key(model, n, seed)
                    supported = np.ones(len(truth), bool) if supported_key is None else arrays[supported_key]
                    cases.append({'device': device, 'model': model, 'support_size': n, 'support_seed': seed,
                        'support_rows': support, 'prediction_key': key, 'test_rows': len(truth),
                        'metrics': metric(arrays[key], truth, supported),
                        'shared_ap_coverage_metric_applicable': model != 'source_ridge',
                        'source_prediction_reused': model in ('source_knn', 'source_ridge') or n == 0})
    return report_from_cases(cases, p)


def report_from_cases(cases, p):
    expected = {(d, m, n, s) for d in p['target_devices'] for m in p['models']
                for n in p['support_sizes'] for s in p['support_seeds']}
    actual = {(r['device'], r['model'], r['support_size'], r['support_seed']) for r in cases}
    if len(cases) != len(expected) or actual != expected:
        raise RuntimeError('Real-transfer case matrix is incomplete or duplicated')
    summaries = []
    for n in p['support_sizes']:
        for model in p['models']:
            rows = [row for row in cases if row['support_size'] == n and row['model'] == model]
            result = {'model': model, 'support_size': n, 'heldout_phones': len(p['target_devices'])}
            for name in p['metrics']:
                result['balanced_phone_'+name] = float(np.mean([row['metrics'][name] for row in rows]))
            per_seed = [np.mean([row['metrics']['mean_2d_error_m'] for row in rows if row['support_seed'] == seed])
                        for seed in p['support_seeds']]
            result['support_draw_std_balanced_mean_m'] = float(np.std(per_seed, ddof=1))
            summaries.append(result)
    baselines = {row['support_size']: row['balanced_phone_mean_2d_error_m'] for row in summaries if row['model'] == 'source_knn'}
    for row in summaries:
        row['mean_change_vs_source_knn_m'] = row['balanced_phone_mean_2d_error_m']-baselines[row['support_size']]
    return {'schema': 1, 'completed': True, 'dataset': 'real TUJI1', 'cases': cases, 'summaries': summaries,
        'case_count': len(cases), 'heldout_phones': len(p['target_devices']), 'physical_sites': 1,
        'official_test_fingerprints': p['expected_testing_rows'], 'support_draws': len(p['support_seeds']),
        'simultaneous_cross_device_pairs_claimed': False, 'pure_hardware_inverse_identified': False,
        'independent_pretrained_neural_runs': 0, 'radiojepa_objective_attribution': False,
        'global_sota_claimed': False, 'warehouse_service_qualified': False, 'activated': False,
        'descriptive_only': True, 'train_test_positions_disjoint': True,
        'source_sha256': source_hashes(), 'archive_sha256': p['archive_sha256'],
        'limits': ['All five device folds share one physical site and overlapping source references.',
            'Support seeds are calibration-sampling variation, not independent neural pretraining runs.',
            'Co-location calibration across campaigns includes device/day/body/environment confounds.',
            'Real localization transfer is not warehouse command-service or RadioJEPA qualification.']}


def markdown(report):
    lines = ['# Real TUJI1 device transfer', '',
        'Five held-out phones, globally held-out official test locations, one physical site. All results are descriptive.', '',
        '| Method | N labeled target positions | Balanced phone mean error (m) | Change vs source kNN (m) | Support-draw SD (m) |',
        '|---|---:|---:|---:|---:|']
    for row in report['summaries']:
        lines.append(f"| {row['model']} | {row['support_size']} | {row['balanced_phone_mean_2d_error_m']:.4f} | {row['mean_change_vs_source_knn_m']:+.4f} | {row['support_draw_std_balanced_mean_m']:.4f} |")
    return '\n'.join(lines + ['', 'No simultaneous-pair, hardware-inverse, warehouse-promotion or global SOTA claim.'])+'\n'


def _close(actual, expected, name, *, atol=1e-9):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, rtol=0., atol=atol):
        raise RuntimeError('Independent real-transfer audit differs: ' + name)


def _audit_offsets(training, device, support, shrinkage):
    """Recompute saved sufficient statistics, without creating a calibration fit."""
    fp, keys = training.fingerprints, data.position_keys(training.xyz_m)
    source = np.flatnonzero(fp.device_ids != device)
    differences = [[] for _ in range(fp.heard.shape[1])]
    for row in support:
        refs = [index for index in source if keys[index] == keys[row]]
        if not refs or fp.device_ids[row] != device:
            raise RuntimeError('Calibration support crossed a phone/location boundary')
        pairs = 0
        for ap in np.flatnonzero(fp.heard[row]):
            values = [fp.rssi_dbm[index, ap] for index in refs if fp.heard[index, ap]]
            if values:
                differences[ap].append(float(fp.rssi_dbm[row, ap]-np.median(values)))
                pairs += 1
        if pairs < 3:
            raise RuntimeError('Calibration support lacks genuine observed co-location pairs')
    flat = [value for values in differences for value in values]
    global_offset = float(np.median(flat)) if flat else 0.
    counts = np.array([len(values) for values in differences])
    median = np.array([np.median(values) if values else global_offset for values in differences])
    weight = counts/(counts+shrinkage)
    return global_offset, weight*median+(1.-weight)*global_offset, counts, len(flat)


def audit_fit(training, device, state, p, inventory):
    expected_rows = np.flatnonzero(training.fingerprints.device_ids != device)
    if not np.array_equal(state['source_rows'], expected_rows):
        raise RuntimeError('Source fit contains target-phone or invented reference rows')
    source = training.fingerprints.subset(expected_rows)
    for name, actual in (('source_rssi_dbm', source.rssi_dbm), ('source_heard', source.heard),
                         ('source_xyz_m', training.xyz_m[expected_rows])):
        _close(state[name], actual, name, atol=0.)
    count = source.heard.sum(0)
    mean = (source.rssi_dbm*source.heard).sum(0)/np.maximum(count, 1)
    variance = (np.where(source.heard, source.rssi_dbm-mean, 0.)**2).sum(0)/np.maximum(count, 1)
    scale = np.maximum(np.sqrt(variance), 1.)
    _close(state['ridge_mean'], mean, 'train-only ridge mean')
    _close(state['ridge_scale'], scale, 'train-only ridge scale')
    x = np.concatenate((np.where(source.heard, (source.rssi_dbm-mean)/scale, 0.), source.heard.astype('f8')), 1)
    feature_mean = x.mean(0)
    _close(state['ridge_feature_mean'], feature_mean, 'ridge intercept normalization')
    x -= feature_mean
    target_mean = training.xyz_m[expected_rows].mean(0)
    _close(state['ridge_target_mean'], target_mean, 'train-only coordinate centroid')
    lhs = (x.T @ x+p['ridge_alpha']*np.eye(x.shape[1])) @ state['ridge_coefficient']
    rhs = x.T @ (training.xyz_m[expected_rows]-target_mean)
    tolerance = 1e-8 * max(1., float(np.abs(rhs).max()))
    _close(lhs, rhs, 'ridge normal-equation residual', atol=tolerance)
    for si, seed in enumerate(p['support_seeds']):
        expected = inventory['support_orders'][str(device)][str(seed)]
        if not np.array_equal(state['support_order'][si], expected):
            raise RuntimeError('Published target-support order changed')
        for ni, n in enumerate(p['support_sizes']):
            values = _audit_offsets(training, device, expected[:n], p['ap_shrinkage_pseudocount'])
            for field, value in zip(('global_offsets_db', 'ap_offsets_db', 'ap_pair_counts', 'pair_counts'), values):
                _close(state[field][si, ni], value, field)
    return {'passes': True, 'device': device, 'source_rows': len(expected_rows),
            'calibration_draws': len(p['support_seeds'])*sum(n > 0 for n in p['support_sizes']),
            'ridge_normal_equation_checked': True}


def _visible():
    excluded = {OUT / 'verification/audit.json', OUT / 'verification/manifest.json'}
    return {str(path.relative_to(ROOT)): c.digest(path) for path in sorted(OUT.rglob('*'))
            if path.is_file() and path not in excluded
            and not any(part.startswith('.') for part in path.relative_to(OUT).parts)
            and not path.name.endswith(('.lock', '.part'))}


def audit(archive=None, *, write=False):
    """Recompute classical inference/arithmetic; never fit or generate new data."""
    check_sources(); _prior(full=True)
    before = _visible()
    p, inventory = protocol(), c.read_json(INVENTORY)
    training = data.load_training(archive, expected=p['archive_sha256'])
    if inventory['metadata'] != data.metadata_inventory(training):
        raise RuntimeError('Actual training metadata differs')
    fits = c.read_json(OUT / 'fits.json')
    predictions = require_predictions_before_labels(archive)
    if (fits['source_sha256'] != source_hashes() or fits.get('test_rf_or_xyz_opened') is not False
            or [item['device'] for item in fits['fits']] != p['target_devices']):
        raise RuntimeError('Fit source/scope membership differs')
    test = data.load_test_features(archive, expected=p['archive_sha256'])
    fit_proofs = []
    for fit, pred in zip(fits['fits'], predictions['predictions']):
        device = fit['device']
        if (pred['device'] != device or fit.get('target_test_rf_opened') is not False
                or fit.get('target_test_xyz_opened') is not False
                or fit['source_sha256'] != source_hashes()
                or fit['inventory_sha256'] != c.digest(INVENTORY)
                or pred['source_sha256'] != source_hashes()):
            raise RuntimeError('Fit/prediction provenance differs')
        state = _load_npz(fit['state'])
        fit_proofs.append(audit_fit(training, device, state, p, inventory))
        query = test.subset(test.device_ids == device)
        saved = _load_npz(pred['predictions'])
        recomputed = _predict_fold(training, query, state, p)
        if set(saved) != set(recomputed) or pred['test_rows'] != query.row0.tolist():
            raise RuntimeError('Prediction array/phone membership differs')
        for name in saved:
            _close(saved[name], recomputed[name], 'saved predictor arithmetic '+name)
    xyz = data.private_test_xyz(archive, expected=p['archive_sha256'])
    expected = evaluate(training, test, predictions, xyz, p)
    if c.read_json(OUT / 'report.json') != expected or (OUT / 'report.md').read_text() != markdown(expected):
        raise RuntimeError('Real-data report differs from raw predictions/XYZ')
    expected_fit = {f'device-{device}{suffix}' for device in p['target_devices'] for suffix in ('.json', '.npz')}
    if ({path.name for path in (OUT / 'fits').iterdir() if path.is_file()} != expected_fit
            or {path.name for path in (OUT / 'predictions').iterdir() if path.is_file()} != expected_fit):
        raise RuntimeError('Missing or extra fold fit/prediction evidence')
    check_sources()
    if before != _visible():
        raise RuntimeError('Real-data evidence changed during audit')
    value = {'schema': 1, 'passes': True, 'archive_sha256': p['archive_sha256'],
        'source_sha256': source_hashes(), 'evidence_sha256': before, 'folds': fit_proofs,
        'case_count': len(expected['cases']), 'reference_fingerprints': p['expected_testing_rows'],
        'new_fits_performed': 0, 'new_synthetic_measurements': 0,
        'classical_prediction_arithmetic_recomputed': True,
        'supervised_co_location_not_simultaneous': True, 'one_physical_site': True}
    if write:
        c.write_once(OUT / 'verification/audit.json', value)
    return value


def seal():
    check_sources()
    proof = c.read_json(OUT / 'verification/audit.json')
    if proof.get('passes') is not True or proof['source_sha256'] != source_hashes() or proof['evidence_sha256'] != _visible():
        raise RuntimeError('Real-transfer raw audit identity changed')
    value = {'schema': 1, 'source_sha256': source_hashes(), 'evidence_sha256': _visible(),
        'audit_sha256': c.digest(OUT / 'verification/audit.json'), 'prior_manifest_sha256': PRIOR_SHA256,
        'archive_sha256': protocol()['archive_sha256'], 'warehouse_service_qualified': False,
        'activated': False}
    c.write_once(OUT / 'verification/manifest.json', value)
    return value


def verify_closure():
    check_sources(); _prior(full=True)
    value = c.read_json(OUT / 'verification/manifest.json')
    if (value['source_sha256'] != source_hashes() or value['evidence_sha256'] != _visible()
            or value['audit_sha256'] != c.digest(OUT / 'verification/audit.json')
            or value['prior_manifest_sha256'] != PRIOR_SHA256):
        raise RuntimeError('Sealed real-transfer source/evidence membership or bytes changed')
    return {'passes': True, 'source_files': len(value['source_sha256']),
            'evidence_files': len(value['evidence_sha256']), 'raw_audit_repeated': False}


def run(archive=None):
    with c.lock(OUT / 'run.lock', nonblocking=True):
        check_sources(); _prior(full=True)
        if (OUT / 'verification/manifest.json').exists():
            verify_closure()
            return c.read_json(OUT / 'report.json')
        p = protocol()
        if any(os.environ.get(name) != value for name, value in p['thread_limits'].items()):
            raise RuntimeError('Use the registered single-thread BLAS/OMP environment')
        training = data.load_training(archive, expected=p['archive_sha256'], expected_links=p['expected_links'])
        inventory = c.read_json(INVENTORY)
        if data.metadata_inventory(training) != inventory['metadata']:
            raise RuntimeError('Training-only inventory changed')
        fits = fit_all(training, p, inventory)
        test, predictions = predict_all(training, fits, archive, p)
        xyz = data.private_test_xyz(archive, expected=p['archive_sha256'])
        value = evaluate(training, test, predictions, xyz, p)
        c.write_once(OUT / 'report.json', value)
        c.write_bytes_once(OUT / 'report.md', markdown(value).encode())
        audit(archive, write=True)
        seal()
        verify_closure()
        return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('inventory', 'preflight', 'register', 'run', 'audit', 'check-closure'))
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args(argv)
    functions = {'inventory': lambda: prepare_inventory(args.archive), 'preflight': preflight,
                 'register': register, 'run': lambda: run(args.archive),
                 'audit': lambda: audit(args.archive, write=True), 'check-closure': verify_closure}
    value = functions[args.phase]()
    print(c.encoded({'phase': args.phase, 'completed': value.get('completed', value.get('passes', True))}).decode())


if __name__ == '__main__':
    main()
