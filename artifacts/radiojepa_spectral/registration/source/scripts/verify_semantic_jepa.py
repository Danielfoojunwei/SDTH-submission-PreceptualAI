"""Verify the completed registered semantic-JEPA archive without new rollouts.

Default: check the existing final manifest and reproduce the frozen report.
--write-manifest explicitly creates/refreshes derived verification metadata.
--verify-caches additionally checks existing local NPZ bytes and every array.
Neither mode collects data, trains models, calibrates timing or runs simulations.
The preregistration Git commit must be present in the local clone.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/semantic_jepa'
REGISTRATION_COMMIT = '9ff5fa56e7327bccde15a10a1a3952a82c699e02'
PRIOR_PR18_COMMIT = '18c294988c242106f4f01e2a85ce1ef55f099e2b'
DOCUMENTS = ('README.md', 'docs/HANDOFF.md', 'docs/PAPER_PROPOSAL.md',
             'docs/SEMANTIC_JEPA_DESIGN.md', 'docs/SEMANTIC_JEPA_RESULTS.md')
sys.path.insert(0, str(ROOT / 'src'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def committed_bytes(commit, name):
    return subprocess.check_output(['git', 'show', commit + ':' + name], cwd=ROOT)


def git_tree(commit):
    return subprocess.check_output(['git', 'rev-parse', commit + '^{tree}'],
                                   cwd=ROOT, text=True).strip()


def hash_value(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def is_hash(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def visible_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return set()
    return {path.name for path in folder.iterdir()
            if not path.name.startswith('.') and not path.name.endswith(('.lock', '.part'))}


def check_jobs(jobs, duration, profiles, extra, *, measured):
    require(len(jobs) == duration, 'Incomplete forecast-job ledger')
    previous = 0.
    overruns = 0
    for decision, job in enumerate(jobs):
        require(job['decision'] == decision and job['history_cutoff_exclusive'] == 60 + decision,
                'Forecast ledger violates the strictly-past boundary')
        kind = 'cold' if decision == 0 else 'refit' if decision % 20 == 0 else 'warm'
        require(job['kind'] == kind, 'Unexpected map-fit regime in forecast ledger')
        charged = profiles[kind] + extra
        require(math.isclose(job['charged_seconds'], charged, rel_tol=1e-13, abs_tol=1e-13),
                'Forecast charge differs from the sealed profile')
        previous = max(float(decision), previous) + charged
        require(job['ready_slot'] == int(math.ceil(previous * 100 - 1e-10)),
                'Forecast availability differs from the registered FIFO')
        require(is_hash(job['forecast_sha256']), 'Malformed forecast-frame hash')
        if measured:
            actual = job['actual_prediction_seconds']
            require(isinstance(actual, (int, float)) and math.isfinite(actual) and actual >= 0,
                    'Invalid actual prediction timing')
            overrun = actual > profiles[kind]
            require(job['runtime_overrun'] is overrun, 'Runtime-overrun ledger differs from measured timing')
            overruns += int(overrun)
    return overruns


def check_events(events, duration, robots):
    for event in events:
        require(type(event['robot']) is int and 0 <= event['robot'] < robots,
                'Report robot identity outside collection')
        measured, complete, delivered = (event[key] for key in
                                        ('measurement_slot', 'complete_slot', 'delivery_slot'))
        require(0 <= measured < complete <= delivered < duration * 100,
                'RF report precedes capture/completion or lies outside collection')
        require(event['first_usable_decision'] == delivered // 100 + 1,
                'Report has premature forecast availability')
        require(is_hash(event['payload_sha256']), 'Malformed delivered-report payload hash')


def check_collections(c, p, data_receipts):
    expected_names = {f'{split}-{site}.json' for split in ('training', 'validation')
                      for site in p[split + '_sites']}
    require(visible_files(OUT / 'collection') == expected_names,
            'Collection receipt membership differs from all 38 registered sites')
    count = 0
    for split in ('training', 'validation'):
        data = data_receipts[split]
        expected_hashes = {f'{split}-{site}' for site in p[split + '_sites']}
        require(set(data['collection_receipt_sha256']) == expected_hashes,
                'Dataset does not bind every registered collection receipt')
        for index, site in enumerate(p[split + '_sites']):
            name = f'{split}-{site}'
            path = OUT / 'collection' / (name + '.json')
            require(digest(path) == data['collection_receipt_sha256'][name],
                    'Collection receipt hash differs: ' + name)
            row = read_json(path)
            require(row['identity'] == data['identity'] and row['split'] == split and row['site'] == site,
                    'Collection receipt source/site/split differs')
            collection = row['collection']
            duration = p['training_decisions'] + 30
            cycle = p['training_conditions'] if split == 'training' else p['validation_conditions']
            condition = cycle[index % len(cycle)]
            require(row['condition'] == condition and row['query_duration_seconds'] == p['training_decisions']
                    and row['delivered_history_length'] == 60 + duration,
                    'Collection condition or history length differs')
            expected = {'completed': True, 'seed': site, 'robots': p['training_robots'],
                'query_duration_seconds': p['training_decisions'], 'collection_duration_seconds': duration,
                'condition': condition, 'index': index,
                'postwarm_observations_use_actual_deliveries_only': True,
                'future_truth_is_predictor_input': False,
                'latest_observation_strictly_before_decision': True,
                'training_collection_shared_by_all_families': True,
                'collection_profile_is_training_assumption': True,
                'collection_profile_is_measured_qualification': False,
                'actual_wall_clock_timings_omitted_from_stable_receipt': True}
            require(all(collection.get(key) == value for key, value in expected.items()),
                    'Collection does not retain the declared observation boundary')
            require(collection['collection_handover_seconds'] == (.03 if (index // 4) % 2 == 0 else 4.),
                    'Training collection handover balance differs')
            events = collection['telemetry_events']
            check_events(events, duration, p['training_robots'])
            require(hash_value(events) == collection['telemetry_events_sha256'],
                    'Collection event digest differs')
            bins = sorted({(event['delivery_slot'] // 100, event['robot']) for event in events})
            require(collection['delivered_history_bins'] == [list(pair) for pair in bins]
                    and collection['delivery_bin_collision_events'] == len(events) - len(bins),
                    'Delivered collection bins differ from actual report events')
            require(collection['telemetry']['reports_delivered'] == len(events),
                    'Collection report count differs')
            # These are deliberately DIFFERENT digest namespaces. The training
            # wrapper hashes contiguous raw bytes; the collector prefixes a
            # canonical dtype/shape header and LF. Full histories are not stored
            # in the archive, so we validate each schema/format and its binding
            # inside the dataset-hashed collection receipt. No cross-digest
            # reconstruction or collection resimulation is performed here.
            observation_names = {'positions', 'value', 'heard', 'bound', 'scanned'}
            for namespace, hashes in [('wrapper raw-array bytes', row['delivered_history_sha256']),
                                      ('collector typed/shaped arrays', collection['observations_sha256'])]:
                require(isinstance(hashes, dict) and set(hashes) == observation_names
                        and all(is_hash(value) for value in hashes.values()),
                        'Malformed observed-history digest namespace: ' + namespace)
            policy = collection['collector_policy']
            check_jobs(collection['forecast_jobs'], duration, policy['profile_seconds'],
                       policy['extra_compute_seconds'], measured=False)
            for key in ('mean_forecast_sha256', 'std_forecast_sha256', 'action_trace_sha256', 'packet_result_sha256'):
                require(is_hash(collection[key]), 'Malformed collection forecast/result hash')
            count += 1
    require(count == 38, 'Expected all 32 training and six validation collections')
    return count


def check_fits(c, p, training, frozen):
    import torch
    tasks = [(family, seed) for family in p['families'] for seed in p['training_seeds']]
    require(len(tasks) == 12 and training['completed'] is True and training['fits'] == 12,
            'Twelve independently trained fits are not complete')
    expected_fit_names = {f'{family}-{seed}.json' for family, seed in tasks}
    expected_checkpoint_names = {f'{family}-{seed}.pt' for family, seed in tasks}
    require(visible_files(OUT / 'fits') == expected_fit_names, 'Fit receipt membership differs')
    require(visible_files(OUT / 'checkpoints') == expected_checkpoint_names, 'Checkpoint membership differs')
    require(set(training['fit_receipt_sha256']) == {f'{family}-{seed}' for family, seed in tasks},
            'Training completion does not bind all fit receipts')
    data = {split: read_json(OUT / 'data' / (split + '.json')) for split in ('training', 'validation')}
    require(set(training['data_receipt_sha256']) == set(data), 'Training data-receipt membership differs')
    base_identity = training['identity']
    require(base_identity['source_sha256'] == frozen['source_sha256'] and base_identity['protocol'] == p,
            'Training source or protocol identity differs')
    for split, receipt in data.items():
        require(digest(OUT / 'data' / (split + '.json')) == training['data_receipt_sha256'][split],
                'Dataset receipt hash differs')
        require(receipt['completed'] is True and receipt['identity'] == dict(base_identity, split=split)
                and receipt['sites'] == p[split + '_sites']
                and receipt['actual_delivered_training_observations'] is True
                and receipt['strictly_past_observable_inputs'] is True
                and receipt['future_states_are_training_labels_only'] is True,
                'Dataset boundary or provenance differs')
        require(is_hash(receipt['npz_sha256']) and all(is_hash(value['sha256'])
                for value in receipt['arrays'].values()), 'Malformed dataset cache/array hashes')
    inference_counts = set()
    for family, seed in tasks:
        name = f'{family}-{seed}'
        path = OUT / 'fits' / (name + '.json')
        require(digest(path) == training['fit_receipt_sha256'][name], 'Fit receipt changed: ' + name)
        fit = read_json(path)
        require(fit['completed'] is True and fit['family'] == family and fit['seed'] == seed
                and fit['identity'] == base_identity
                and fit['data_receipt_sha256'] == training['data_receipt_sha256'],
                'Fit receipt identity differs: ' + name)
        history = fit['history']
        require([row['epoch'] for row in history] == list(range(1, p['epochs'] + 1)),
                'Fit does not contain all registered training epochs')
        require(all(math.isfinite(row['validation_score']) for row in history),
                'Nonfinite checkpoint-selection history')
        best = min(history, key=lambda row: row['validation_score'])
        require(fit['best_epoch'] == best['epoch']
                and fit['best_validation_score'] == best['validation_score'],
                'Selected checkpoint differs from the fixed validation rule')
        require(fit['best_validation_metrics'] == {key: best[key] for key in
                ('validation_weighted_rmse', 'validation_ap_regret', 'validation_score')},
                'Selected validation metrics differ from history')
        checkpoint = OUT / 'checkpoints' / (name + '.pt')
        checkpoint_name = str(checkpoint.relative_to(ROOT))
        require(digest(checkpoint) == fit['checkpoint_sha256'] == training['checkpoint_sha256'][checkpoint_name],
                'Trained checkpoint hash differs: ' + name)
        # Passive weights-only decoding validates metadata; no model forward is run.
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        require(saved['identity'] == base_identity and saved['family'] == family and saved['seed'] == seed
                and saved['epoch'] == fit['best_epoch'] and saved['config']['family'] == family
                and saved['data_receipt_sha256'] == training['data_receipt_sha256'],
                'Checkpoint provenance differs from the fit receipt')
        require(all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
                    for value in saved['state'].values()), 'Nonfinite checkpoint weights')
        inference_counts.add(fit['parameters_inference'])
    require(len(inference_counts) == 1, 'Matched neural families no longer share inference parameter count')
    return data, len(tasks)


def check_latency(p, study, training, frozen):
    """Recompute sealed charges from archived timings; never benchmark again."""
    import numpy as np
    value = read_json(OUT / 'latency.json')
    require(value['sealed'] is True and value['outcomes_used'] is False
            and value['source_sha256'] == frozen['source_sha256']
            and value['checkpoint_sha256'] == training['checkpoint_sha256']
            and value['calibration_site'] == p['calibration_site'] and value['threads'] == 1,
            'Sealed pre-development timing provenance differs')
    names = {study.arm_name(family, seed) for family, seed in study.arms()}
    require(set(value['profiles']) == names and set(value['raw_seconds']) == names,
            'Timing receipt does not cover the registered 14 arms')
    regimes = ('cold', 'warm', 'refit')
    for name in names:
        require(set(value['raw_seconds'][name]) == set(regimes)
                and set(value['profiles'][name]) == set(regimes), 'Timing regime membership differs')
        for kind, expected_count in [('cold', 3), ('warm', 66), ('refit', 3)]:
            raw = value['raw_seconds'][name][kind]
            require(len(raw) == expected_count and all(isinstance(x, (int, float))
                    and math.isfinite(x) and x >= 0 for x in raw),
                    'Timing calibration repetitions differ')
    envelope = {kind: max(float(np.quantile(value['raw_seconds'][study.arm_name(f, s)][kind], .95))
                    for f in p['families'] for s in p['training_seeds']) for kind in regimes}
    require(value['neural_common_p95_seconds'] == envelope, 'Common neural timing envelope differs')
    for family, seed in study.arms():
        name = study.arm_name(family, seed)
        raw = envelope if family in p['families'] else {
            kind: float(np.quantile(value['raw_seconds'][name][kind], .95)) for kind in regimes}
        expected = {kind: max(.01, math.ceil(2 * cost * 100) / 100) for kind, cost in raw.items()}
        require(value['profiles'][name] == expected, 'Sealed charge differs from registered timing rule')
    return value


def check_caches(c, data, *, requested):
    observations = {}
    if requested:
        import numpy as np
        from wirelessjev.semantic_jepa import training
    for split, receipt in data.items():
        path = c.WORK / (split + '.npz')
        observation = {'present': path.is_file(), 'verified_this_invocation': False}
        if requested:
            require(path.is_file(), f'Requested cache absent: {path}; no collection was attempted')
            require(digest(path) == receipt['npz_sha256'], 'Local cache bytes differ: ' + split)
            with np.load(path, allow_pickle=False) as archive:
                require(training.NPZ_IDENTITY in archive, 'Cache has no embedded registered identity')
                require(json.loads(archive[training.NPZ_IDENTITY].tobytes().decode()) == receipt['identity'],
                        'Cache embedded identity differs')
                arrays = {name: archive[name] for name in archive.files if name != training.NPZ_IDENTITY}
            training._validate_arrays(arrays, receipt['identity'])
            require(training._dataset_receipt(path, receipt['identity'], arrays) == receipt,
                    'Local cache array/site/query digests differ: ' + split)
            observation['verified_this_invocation'] = True
        observations[split] = observation
    return observations


def check_case_membership(study, keys, folder):
    expected = {study.case_name(key) + suffix for key in keys for suffix in ('.json.gz', '.receipt.json')}
    require(visible_files(folder) == expected, 'Archive membership differs: ' + str(folder))


def check_cases(c, study, keys, latency, *, replay=False):
    rows, overrun_cases = [], 0
    for key in keys:
        row = study.load_case(key, replay=replay)
        require(row is not None, 'Registered case missing: ' + study.case_name(key))
        folder = OUT / ('replays' if replay else 'cases')
        receipt = read_json(folder / (study.case_name(key) + '.receipt.json'))
        require(receipt['case'] == list(key) and receipt['replay'] is replay,
                'Receipt identity differs from its archive')
        require(row['completed'] is True and row['hardware_qualified'] is False,
                'Incomplete case or unsupported hardware flag')
        p = c.protocol()
        check_events(row['telemetry']['events'], p['duration_seconds'], p['robots'])
        profile = latency['profiles'][study.arm_name(row['family'], row['training_seed'])]
        overruns = check_jobs(row['jobs'], p['duration_seconds'], profile,
                             p['extra_compute_seconds'], measured=True)
        require(row['runtime_overrun'] is bool(overruns), 'Case runtime-veto summary differs')
        require(row['causality_pass'] is True, 'Case does not pass delivered-observation causality')
        require(is_hash(row['forecast_mean_sha256']) and is_hash(row['forecast_std_sha256']),
                'Malformed full-forecast hashes')
        overrun_cases += int(bool(overruns))
        rows.append(row)
    return rows, overrun_cases


def verify(*, write_manifest=False, verify_caches=False):
    for name in ('report.json', 'report.md', 'report_markdown.json'):
        require((OUT / name).is_file(), 'Complete the registered report first: ' + name)
    from wirelessjev.semantic_jepa import common as c, study
    frozen = c.check_sources()
    require(len(frozen['source_sha256']) == 80, 'Registered source membership differs from 80 files')
    require(frozen['data_accessed'] is False and frozen['scientific_source_frozen'] is True,
            'Source registration is not a pre-data freeze')
    git_meta = read_json(OUT / 'registration/git.json')
    tree = git_tree(REGISTRATION_COMMIT)
    require(git_meta['scientific_source_commit'] == REGISTRATION_COMMIT
            and git_meta['scientific_source_tree'] == tree
            and git_meta['before_registered_collection_or_fitting'] is True,
            'Pre-data Git identity differs from the pinned scientific commit')
    for name, expected in frozen['source_sha256'].items():
        value = committed_bytes(REGISTRATION_COMMIT, name)
        require(hashlib.sha256(value).hexdigest() == expected
                and (OUT / 'registration/source' / name).read_bytes() == value,
                'Registered source differs from preregistration Git bytes: ' + name)

    training = study.training_receipt()
    data, fits = check_fits(c, c.protocol(), training, frozen)
    collections = check_collections(c, c.protocol(), data)
    cache_observation = check_caches(c, data, requested=verify_caches)
    latency = check_latency(c.protocol(), study, training, frozen)
    development_keys = study.case_keys('development')
    devrows, dev_overruns = check_cases(c, study, development_keys, latency)
    dev = read_json(OUT / 'development.json')
    costs = study.validate_development_seal(dev, devrows)
    qualification = read_json(OUT / 'qualification.json')
    require(qualification == {'development_sha256': digest(OUT / 'development.json'),
            'passed_costs': costs, 'source_sha256': frozen['source_sha256'],
            'recomputed_from_all_development_archives': True},
            'Conditional confirmation permission differs from recomputed development')
    confirmation_keys = study.case_keys('confirmation', costs)
    replay_keys = study.replay_keys(costs)
    check_case_membership(study, development_keys + confirmation_keys, OUT / 'cases')
    check_case_membership(study, replay_keys, OUT / 'replays')
    confirmation, confirm_overruns = check_cases(c, study, confirmation_keys, latency)
    replays, replay_overruns = check_cases(c, study, replay_keys, latency, replay=True)
    comparisons = study.compare_replays(confirmation, replays, replay_keys)
    require(all(row['exact'] for row in comparisons), 'Required exact replay failed')
    archived = read_json(OUT / 'report.json')
    require(study.report() == archived, 'Frozen derivation differs from the immutable report')
    require(archived['development']['passed_costs'] == costs
            and archived['confirmation']['passed_development_costs'] == costs,
            'Report tested-cost subset differs from the verified development gate')
    for row in archived['confirmation']['per_cost']:
        require((row['status'] == 'tested') is (row['handover_seconds'] in costs),
                'Untested cost is not explicitly development-vetoed')
    require(archived['activated'] is False and archived['production_qualified'] is False
            and archived['confirmation']['hardware_qualified'] is False,
            'Unsupported activation/qualification claim')

    log_path = OUT / 'logs/preflight-tests.txt'
    log = log_path.read_text()
    require(re.search(r'^838 passed, 1 warning in 128\.12s', log, re.M) is not None,
            'Archived full-test execution differs from 838/1/128.12s')
    prior_path = ROOT / 'artifacts/continual_packet/verification/manifest.json'
    require(prior_path.read_bytes() == committed_bytes(PRIOR_PR18_COMMIT,
            str(prior_path.relative_to(ROOT))), 'Prior PR18 verification manifest changed')
    prior = read_json(prior_path)
    prior_counts = {}
    for group in ('sources_sha256', 'evidence_sha256'):
        for name, expected in prior[group].items():
            require(digest(ROOT / name) == expected, 'Prior PR18 source/evidence changed: ' + name)
        prior_counts[group] = len(prior[group])
    # Historical global README/document hashes belong to the prior commit;
    # adding new files and updating the current summaries is permitted.
    sources = [path for folder in ('src', 'scripts', 'tests') for path in (ROOT / folder).rglob('*.py')
               if '__pycache__' not in path.parts] + [ROOT / 'pyproject.toml', ROOT / 'uv.lock']
    documents = [ROOT / name for name in DOCUMENTS]
    proof = OUT / 'verification'
    path = proof / 'manifest.json'
    evidence = [item for item in OUT.rglob('*') if item.is_file() and item != path
                and not any(part.startswith('.') for part in item.relative_to(OUT).parts)
                and not item.name.endswith(('.lock', '.part'))]
    hashes = lambda paths: {str(item.relative_to(ROOT)): digest(item) for item in sorted(paths)}
    value = {'schema': 1, 'scientific_source_commit': REGISTRATION_COMMIT,
        'scientific_source_tree': tree, 'registered_source_files': 80,
        'full_tests': {'passed': 838, 'warnings': 1, 'seconds': 128.12,
                       'log': str(log_path.relative_to(ROOT)), 'sha256': digest(log_path)},
        'independently_trained_fits': fits, 'collection_receipts': collections,
        'collection_history_digest_namespaces': {
            'wrapper': 'SHA256 contiguous raw array bytes',
            'collector': 'SHA256 canonical dtype/shape JSON header + LF + contiguous raw bytes',
            'validated': 'Separate schemas/hash formats and dataset-bound collection-receipt bytes',
            'full_history_cross_reconstruction_performed': False,
            'collection_resimulation_performed': False},
        'main_cases': {'development': len(devrows), 'confirmation': len(confirmation)},
        'exact_replays': len(replays), 'passed_development_costs': costs,
        'experiment_service_gate_by_cost': archived['confirmation']['per_cost'],
        'runtime_overrun_cases': {'development': dev_overruns, 'confirmation': confirm_overruns,
                                  'replay': replay_overruns},
        'all_registered_source_and_receipt_checks_passed': True,
        'report_recomputed_from_actual_archives': True, 'report_sha256': digest(OUT / 'report.json'),
        'prior_PR18_commit': PRIOR_PR18_COMMIT, 'prior_PR18_manifest_sha256': digest(prior_path),
        'prior_PR18_preserved_entry_counts': prior_counts,
        'previous_global_document_hashes_apply_to_their_original_commit': True,
        'cache_receipts': {split: {'npz_sha256': receipt['npz_sha256'], 'array_count': len(receipt['arrays'])}
                           for split, receipt in data.items()},
        'cache_check_optional_and_not_claimed_by_default': True,
        'hardware_qualified': False, 'production_qualified': False, 'activated': False,
        'new_simulations_training_collection_or_timing_calibration_performed': False,
        'sources_sha256': hashes(sources), 'documents_sha256': hashes(documents),
        'evidence_sha256': hashes(evidence)}
    if write_manifest:
        proof.mkdir(exist_ok=True)
        value['verified_at_utc'] = datetime.now(timezone.utc).isoformat()
        value['cache_observation_at_manifest_creation'] = cache_observation
        fd, temporary = tempfile.mkstemp(prefix='.manifest-', dir=proof)
        try:
            with os.fdopen(fd, 'w') as stream:
                stream.write(json.dumps(value, indent=2) + '\n')
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    else:
        stored = read_json(path)
        stored.pop('verified_at_utc')
        stored.pop('cache_observation_at_manifest_creation')
        require(stored == value, 'Final manifest differs from current source/documents/evidence')
    print(json.dumps({'main_cases': value['main_cases'], 'fits': fits,
                      'exact_replays': len(replays), 'passed_development_costs': costs,
                      'cache_observation_this_invocation': cache_observation,
                      'hardware_qualified': False, 'production_qualified': False}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true',
                        help='Explicitly create/refresh the derived verification manifest')
    parser.add_argument('--verify-caches', action='store_true',
                        help='Require existing local NPZ caches and verify all bytes/array digests; never rebuild')
    arguments = parser.parse_args()
    verify(write_manifest=arguments.write_manifest, verify_caches=arguments.verify_caches)
