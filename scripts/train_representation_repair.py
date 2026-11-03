"""Read-only verification/reuse of the completed projection-repair archive.

The new nested registration declares intentional reuse of exposed packet data,
not a new independent cohort. Historical fitting code remains in executed_source
and the original preregistration commit; this current runner never retrains.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from wirelessjev.representation_repair import (
    ProjectedPacketValueModel, data_arrays, digest, evaluate, summarize,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def safe_relative(name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts and str(path) != '.',
            'Unsafe archive-relative evidence path')
    return path


def validate_reuse_registration(protocol, original):
    """Retain actual scene identities; this is no independent-cohort claim."""
    for split in ('training', 'validation'):
        require(protocol[split+'_sites'] == original[split+'_sites'],
                f'Reuse registration must retain the original {split} scene identities')
    reused = set(protocol['training_sites']) | set(protocol['validation_sites'])
    reserved = set(original['development_sites']) | set(original['test_sites'])
    require(not reused & reserved, 'Reuse registration accesses reserved packet holdouts')
    require(protocol['handover_seconds'] == original['handover_seconds'], 'Original physical-cost identities changed')
    require(protocol['reserved_holdouts_accessed'] is False and protocol['native_control_evaluated'] is False,
            'Reuse study must retain its development-only/no-control scope')


def verify_archive(root):
    """Check historical execution, immutable inputs and exact saved metrics.

    No optimizer is created and no archive file is written. Current support
    code/test changes are disclosed separately from frozen execution copies.
    """
    root = Path(root).resolve()
    out = root/'artifacts/representation_repair'
    migration_path = out/'layout_migration.json'
    require(migration_path.exists(),
            'No migrated completed archive. Reproduce fresh training from preregistration commit 74c01c7 in a clean checkout.')
    migration = read(migration_path)
    nested = root/migration['registration_path_after']
    require(migration['registration_path_before'] == 'artifacts/representation_repair/protocol.json'
            and migration['registration_path_after'] == 'artifacts/representation_repair/registration/protocol.json',
            'Unexpected reuse-study registration relocation')
    require(not (root/migration['registration_path_before']).exists(),
            'Reuse study still declares a top-level independent-cohort protocol')
    require(digest(nested) == migration['registration_sha256_before_and_after'],
            'Relocated registration bytes changed')
    protected = migration['immutable_artifacts_sha256']
    for name, expected in protected.items():
        path = out/safe_relative(name)
        require(path.exists() and digest(path) == expected, f'Archived evidence changed: {name}')
    historical_path = out/'verification_at_execution.json'
    require(digest(historical_path) == migration['original_execution_verification_sha256_before_and_after'],
            'Original execution-time verification changed')
    historical = read(historical_path)
    manifest = read(out/'preregistration.json')
    completion = read(out/'completion.json')
    protocol = read(nested)
    require(manifest['commit'] == migration['preregistration_commit'] == completion['preregistration_commit']
            == historical['preregistration_commit'], 'Historical preregistration commits disagree')
    require(digest(out/'preregistration.json') == completion['preregistration_sha256']
            == historical['preregistration_sha256'], 'Historical preregistration digest mismatch')
    require(digest(out/'completion.json') == historical['completion_sha256'], 'Historical completion changed')
    old_registration_name = migration['registration_path_before']
    require(digest(nested) == manifest['source_sha256'][old_registration_name] == completion['protocol_sha256'],
            'Relocated registration differs from the exact fitted protocol')
    changed_support = set(migration['post_execution_support_files_allowed_to_change'])
    require(changed_support == {'scripts/train_representation_repair.py', 'tests/test_representation_repair.py'},
            'Only post-execution runner/tests may differ from registered execution')
    changed = []
    for name, expected in manifest['source_sha256'].items():
        path = safe_relative(name)
        copied = out/'executed_source'/path
        require(copied.exists() and digest(copied) == expected, f'Frozen executed source changed: {name}')
        committed = subprocess.check_output(['git', 'show', f"{manifest['commit']}:{name}"], cwd=root)
        require(hashlib.sha256(committed).hexdigest() == expected, f'Executed source differs from preregistration commit: {name}')
        current = nested if name == old_registration_name else root/path
        if digest(current) != expected:
            require(name in changed_support, f'Current non-support execution source changed: {name}')
            changed.append(name)
    commit_time = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', manifest['commit']], cwd=root, text=True))
    require(commit_time <= manifest['registered_unix'], 'Source commit does not precede registration')
    original = read(root/'artifacts/packet_value/protocol.json')
    validate_reuse_registration(protocol, original)
    original_training = read(root/'artifacts/packet_value/training.json')
    require(all(digest(root/safe_relative(name)) == expected
                for name, expected in original_training['checkpoint_sha256'].items()),
            'An original packet checkpoint changed')
    data_hashes = {}
    train = data_arrays(root, protocol, 'training', data_hashes)
    validation = data_arrays(root, protocol, 'validation', data_hashes)
    require(data_hashes == manifest['data_and_receipt_sha256'], 'Historical exposed-data manifest differs')
    require(completion['source_sha256'] == manifest['source_sha256'], 'Completion execution sources differ')
    torch.set_num_threads(1)
    fits = []
    for seed in protocol['training_seeds']:
        for variant in protocol['variants']:
            key = f'{variant}-{seed}'
            receipt = read(out/'fits'/(key+'.json'))
            checkpoint = out/'checkpoints'/(key+'.pt')
            log = out/'logs'/(key+'.jsonl')
            require(digest(checkpoint) == receipt['checkpoint_sha256'] == completion['checkpoint_sha256'][key],
                    f'Checkpoint identity mismatch: {key}')
            require(digest(log) == receipt['epoch_log_sha256'], f'Epoch log identity mismatch: {key}')
            epochs = [json.loads(line) for line in log.read_text().splitlines()]
            require(epochs == receipt['history'] and [r['epoch'] for r in epochs] == list(range(1, protocol['epochs']+1)),
                    f'Incomplete/different real epoch history: {key}')
            require(receipt['started_unix'] > manifest['registered_unix'], f'Fit precedes preregistration: {key}')
            require(receipt['source_sha256'] == manifest['source_sha256']
                    and receipt['data_and_receipt_sha256'] == data_hashes
                    and receipt['preregistration_commit'] == manifest['commit'], f'Fit provenance differs: {key}')
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            require(state['epoch'] == receipt['epoch'] == protocol['epochs'] and state['seed'] == seed
                    and state['variant'] == variant and state['config'] == protocol['model']
                    and state['preregistration_commit'] == manifest['commit'], f'Fixed-fit state differs: {key}')
            model = ProjectedPacketValueModel(variant, **state['config'])
            model.load_state_dict(state['state'])
            require(model.inference_parameters == receipt['inference_parameters'] == protocol['inference_parameters_both']
                    and model.extra_training_parameters == receipt['extra_training_parameters']
                    == protocol['extra_training_parameters'][variant], f'Parameter accounting differs: {key}')
            metrics = evaluate(model, train, validation)
            require(metrics == receipt['metrics'], f'Exact checkpoint metric recomputation differs: {key}')
            require(epochs[-1]['validation_service'] == metrics['validation']['service'], f'Final epoch metrics differ: {key}')
            require(receipt['reserved_holdouts_accessed'] is False and receipt['native_control_evaluated'] is False,
                    f'Fit exceeds registered reuse scope: {key}')
            fits.append(receipt)
    require(len(fits) == completion['fits'] == 6, 'The original six-fit set changed')
    require(summarize(fits, protocol) == completion['summary'], 'Paired results do not reconcile')
    return {'completed': True, 'mode': 'Read-only verification/reuse of historical fitted archive',
            'preregistration_commit': manifest['commit'], 'verified_fits': len(fits),
            'fixed_final_epoch': protocol['epochs'], 'exact_checkpoint_metric_recomputation': True,
            'execution_source_copies_verified': len(manifest['source_sha256']),
            'immutable_registered_artifacts_verified': len(protected),
            'exposed_input_and_receipt_files_verified': len(data_hashes),
            'frozen_execution_source_copies_unchanged': True,
            'current_model_source_matches_execution': True,
            'current_support_differs_from_execution': sorted(changed),
            'current_support_sha256': {name: digest(root/name) for name in sorted(changed_support)},
            'registration_sha256_preserved': digest(nested), 'registration_path': str(nested.relative_to(root)),
            'layout_migration_sha256': digest(migration_path),
            'verification_at_execution_sha256': digest(historical_path),
            'completion_sha256': digest(out/'completion.json'),
            'reserved_holdout_inputs_opened': False, 'new_optimizer_steps': 0, 'old_archives_modified': False,
            'all_six_original_packet_checkpoints_unchanged': True,
            'training_scenes_reused': protocol['training_sites'], 'validation_scenes_reused': protocol['validation_sites'],
            'summary': completion['summary']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=None, help='Optional new verification JSON; historical evidence remains read-only')
    args = parser.parse_args()
    result = verify_archive(args.root)
    if args.output is not None:
        dest = args.output.resolve()
        out = args.root.resolve()/'artifacts/representation_repair'
        protected = read(out/'layout_migration.json')['immutable_artifacts_sha256']
        reserved = {out/safe_relative(name) for name in protected}
        manifest = read(out/'preregistration.json')
        reserved |= {args.root.resolve()/safe_relative(name)
                     for name in manifest['source_sha256'] | manifest['data_and_receipt_sha256']}
        reserved |= {out/'layout_migration.json', out/'registration/protocol.json', out/'protocol.json'}
        require(dest not in reserved, 'Refusing to overwrite historical archive evidence')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
