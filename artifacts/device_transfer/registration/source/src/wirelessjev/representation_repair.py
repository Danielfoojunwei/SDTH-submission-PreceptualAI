"""Preregistered training-only projection ablation for the frozen packet task.

The service forward graph is unchanged. An optional learned linear projection
is used only by the original auxiliary future-latent objective. This is an
established separation of task/pretext spaces, not a DINO feature experiment.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .packet_value.model import PacketValueModel, objective, update_target


class ProjectedPacketValueModel(PacketValueModel):
    """Keep the value head on the service latent; project only training pretext."""

    def __init__(self, variant='shared', **config):
        if variant not in ('shared', 'projected'):
            raise ValueError('Choose shared or projected')
        super().__init__(**config)
        if self.config['family'] != 'jepa':
            raise ValueError('This ablation is defined for JEPA training')
        self.variant = variant
        hidden = self.config['hidden']
        self.future_projection = nn.Identity() if variant == 'shared' else nn.Linear(hidden, hidden)
        if variant == 'projected':
            nn.init.eye_(self.future_projection.weight)
            nn.init.zeros_(self.future_projection.bias)

    def _predict(self, encoded, query):
        prediction = super()._predict(encoded, query)
        if self.training:
            # The service head has already consumed the unprojected latent.
            prediction['latent'] = self.future_projection(prediction['latent'])
        return prediction

    def projected_future_latent(self, context, query):
        """Explicit diagnostic path; never an extra eval-time service operation."""
        self._check_context(context)
        if query.shape != (len(context), self.config['query_dim']) or not torch.isfinite(query).all():
            raise ValueError('Invalid action query')
        prediction = super()._predict(self.encoder(context), query)
        return self.future_projection(prediction['latent'])

    @property
    def extra_training_parameters(self):
        return sum(p.numel() for p in self.future_projection.parameters())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def latent_stats(latent):
    x = np.asarray(latent, dtype=np.float64)
    centered = x-x.mean(0)
    eig = np.linalg.svd(centered, compute_uv=False)**2 / max(1, len(x)-1)
    total = float(eig.sum())
    p = eig[eig > 0]/total if total else np.array([])
    return {'examples': len(x), 'dimensions': x.shape[1], 'centered_rms': float(np.sqrt(np.mean(centered**2))),
            'covariance_trace': total,
            'entropy_effective_rank': float(np.exp(-np.sum(p*np.log(p)))) if len(p) else 0.,
            'top_pc_variance_fraction': float(eig[0]/total) if total else 0.}


def data_arrays(root, protocol, split, hashes):
    """Only protocol-listed old training and old validation inputs are readable."""
    if split not in ('training', 'validation'):
        raise ValueError('Only development training/validation splits are authorized')
    parts = []
    source = root/'artifacts/packet_value'
    for scene in protocol[split+'_sites']:
        for cost in protocol['handover_seconds']:
            stem = f'{split}-{scene}-{cost}'
            path = source/'data'/(stem+'.npz')
            receipt_path = source/'labels'/(stem+'.json')
            receipt = json.loads(receipt_path.read_text())
            identity = digest(path)
            if receipt['split'] != split or receipt['scene'] != scene or receipt['data_sha256'] != identity:
                raise ValueError('Original data receipt does not reconcile')
            hashes[str(path.relative_to(root))] = identity
            hashes[str(receipt_path.relative_to(root))] = digest(receipt_path)
            with np.load(path, allow_pickle=False) as z:
                part = {k: z[k].copy() for k in ('context', 'query', 'delta_misses', 'delta_gap',
                                                'future_context', 'future_valid', 'cost', 'scene')}
            if set(np.unique(part['scene'])) != {scene}:
                raise ValueError('Data has an unregistered scene')
            parts.append(part)
    arrays = {k: np.concatenate([a[k] for a in parts]) for k in parts[0]}
    if len(arrays['context']) != protocol['expected_examples'][split]:
        raise ValueError('Registered original data count differs')
    return arrays


def batch(arrays, index):
    return {key: torch.as_tensor(value[index], dtype=torch.float32) for key, value in arrays.items()
            if key in ('context', 'query', 'delta_misses', 'delta_gap', 'future_context', 'future_valid')}


@torch.no_grad()
def service_errors(model, arrays):
    model.eval()
    prediction = []
    for start in range(0, len(arrays['context']), 128):
        b = batch(arrays, slice(start, start+128))
        prediction.append(model(b['context'], b['query'])['mean'].numpy())
    prediction = np.concatenate(prediction)
    target = np.c_[arrays['delta_misses'], arrays['delta_gap']]

    def score(mask):
        error = prediction[mask]-target[mask]
        return {'examples': int(mask.sum()), 'miss_mae': float(np.abs(error[:, 0]).mean()),
                'gap_mae_seconds': float(np.abs(error[:, 1]).mean()),
                'miss_rmse': float(np.sqrt(np.mean(error[:, 0]**2))),
                'gap_rmse_seconds': float(np.sqrt(np.mean(error[:, 1]**2)))}
    return {'all': score(np.ones(len(target), dtype=bool)),
            'by_cost': {str(float(cost)): score(arrays['cost'] == cost) for cost in np.unique(arrays['cost'])}}


@torch.no_grad()
def latent_arrays(model, arrays):
    model.eval()
    rows = {key: [] for key in ('context', 'service_action', 'projected_action', 'future_teacher')}
    valid = []
    for start in range(0, len(arrays['context']), 128):
        b = batch(arrays, slice(start, start+128))
        prediction = model(b['context'], b['query'])
        rows['context'].append(prediction['context_latent'].numpy())
        rows['service_action'].append(prediction['latent'].numpy())
        rows['projected_action'].append(model.projected_future_latent(b['context'], b['query']).numpy())
        rows['future_teacher'].append(model.target_encoder(b['future_context'], b['future_valid']).numpy())
        valid.append(b['future_valid'].reshape(len(b['context']), -1).bool().any(-1).numpy())
    return {key: np.concatenate(pieces) for key, pieces in rows.items()}, np.concatenate(valid)


def evaluate(model, train, validation):
    train_latent, train_valid = latent_arrays(model, train)
    validation_latent, validation_valid = latent_arrays(model, validation)
    if not train_valid.any() or not validation_valid.any():
        raise ValueError('No real observed future targets')
    constant = train_latent['future_teacher'][train_valid].mean(0)
    output = {}
    for split, arrays, latents, valid in (
        ('training', train, train_latent, train_valid),
        ('validation', validation, validation_latent, validation_valid),
    ):
        target, prediction = latents['future_teacher'][valid], latents['projected_action'][valid]
        mse = float(np.mean((prediction-target)**2))
        constant_mse = float(np.mean((constant-target)**2))
        absolute = np.abs(prediction-target)
        output[split] = {'service': service_errors(model, arrays),
            'future': {'valid_examples': int(valid.sum()), 'mse': mse,
                'training_teacher_mean_constant_mse': constant_mse,
                'mse_over_training_teacher_mean_constant': mse/constant_mse if constant_mse else None,
                'smooth_l1': float(np.where(absolute < 1, .5*absolute**2, absolute-.5).mean())},
            'latents': {key: latent_stats(value[valid]) for key, value in latents.items()}}
    return output


def register(root, protocol_path, source_paths, data_hashes):
    """Require committed preregistration/code and snapshot their exact bytes."""
    commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', str(protocol_path.relative_to(root))], cwd=root, text=True).strip()
    if not commit:
        raise ValueError('Commit the protocol and exact code before any fit')
    for path in (protocol_path, *source_paths[:3]):
        name = str(path.relative_to(root))
        committed = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root)
        if committed != path.read_bytes():
            raise ValueError('Preregistered code/protocol differs from the committed version')
    hashes = {str(path.relative_to(root)): digest(path) for path in source_paths}
    hashes[str(protocol_path.relative_to(root))] = digest(protocol_path)
    out = root/'artifacts/representation_repair'
    manifest = {'preregistered': True, 'commit': commit, 'registered_unix': time.time(),
        'source_sha256': hashes, 'data_and_receipt_sha256': data_hashes,
        'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'numpy': np.__version__, 'device': 'cpu', 'torch_threads': torch.get_num_threads()},
        'old_archives_immutable': True, 'reserved_holdouts_accessed': False}
    manifest_path = out/'preregistration.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous[key] != manifest[key] for key in ('commit', 'source_sha256', 'data_and_receipt_sha256')):
            raise ValueError('Frozen preregistration differs; do not overwrite results')
        return previous
    if list((out/'fits').glob('*.json')) or list((out/'checkpoints').glob('*.pt')):
        raise ValueError('A fit exists without a preceding preregistration')
    for name in hashes:
        target = out/'executed_source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root/name).read_bytes())
    write(manifest_path, manifest)
    return manifest


def fit_one(root, protocol, manifest, train, validation, variant, seed):
    out = root/'artifacts/representation_repair'
    key = f'{variant}-{seed}'
    report_path, checkpoint_path = out/'fits'/(key+'.json'), out/'checkpoints'/(key+'.pt')
    if report_path.exists():
        old = json.loads(report_path.read_text())
        if old['checkpoint_sha256'] != digest(checkpoint_path) or old['source_sha256'] != manifest['source_sha256']:
            raise ValueError('Completed fit differs from its original receipt')
        return old
    log = out/'logs'/(key+'.jsonl')
    if log.exists() or checkpoint_path.exists():
        raise ValueError('Partial fit exists; inspect it rather than silently restarting')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ProjectedPacketValueModel(variant=variant, **protocol['model'])
    optimizer = torch.optim.AdamW(model.parameters(), **protocol['optimizer'])
    started = time.time()
    if started <= manifest['registered_unix']:
        raise ValueError('Fit must start after frozen preregistration')
    history = []
    log.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(protocol['epochs']):
        model.train()
        order = rng.permutation(len(train['context']))
        losses, components = [], []
        for start in range(0, len(order), protocol['batch_size']):
            b = batch(train, order[start:start+protocol['batch_size']])
            loss, terms = objective(model, b)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite original objective')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['gradient_clip'])
            optimizer.step()
            update_target(model)
            losses.append(float(loss.detach()))
            components.append({name: float(value) for name, value in terms.items()})
        errors = service_errors(model, validation)
        row = {'epoch': epoch+1, 'train_loss_mean_of_batches': float(np.mean(losses)),
            'objective_terms_mean_of_batches': {name: float(np.mean([c[name] for c in components])) for name in components[0]},
            'validation_service': errors, 'elapsed_seconds': time.time()-started}
        history.append(row)
        with log.open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
    # Final epoch is fixed in the committed protocol. Validation scores never
    # choose a checkpoint, epoch, seed, hyperparameter or new run.
    metrics = evaluate(model, train, validation)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'config': model.config, 'variant': variant, 'seed': seed, 'epoch': protocol['epochs'],
                'state': model.state_dict(), 'preregistration_commit': manifest['commit']}, checkpoint_path)
    receipt = {'completed': True, 'variant': variant, 'seed': seed, 'epoch': protocol['epochs'],
        'started_unix': started, 'elapsed_seconds': time.time()-started, 'history': history,
        'metrics': metrics, 'inference_parameters': model.inference_parameters,
        'extra_training_parameters': model.extra_training_parameters,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'checkpoint_sha256': digest(checkpoint_path), 'epoch_log_sha256': digest(log),
        'source_sha256': manifest['source_sha256'], 'data_and_receipt_sha256': manifest['data_and_receipt_sha256'],
        'preregistration_commit': manifest['commit'], 'checkpoint_selection': 'Fixed final epoch, no selection',
        'reserved_holdouts_accessed': False, 'native_control_evaluated': False,
        'dino_features_used': False, 'simulation_data_only': True}
    write(report_path, receipt)
    print(json.dumps({'fit': key, 'seconds': round(receipt['elapsed_seconds'], 2),
        'validation': metrics['validation']['service']['all'],
        'future_ratio': metrics['validation']['future']['mse_over_training_teacher_mean_constant']}), flush=True)
    return receipt


def summarize(fits, protocol):
    by_key = {(r['variant'], r['seed']): r for r in fits}
    rows = []
    for seed in protocol['training_seeds']:
        shared, projected = (by_key[(variant, seed)] for variant in ('shared', 'projected'))
        a, b = shared['metrics'], projected['metrics']
        guard = {}
        for split in ('training', 'validation'):
            left, right = a[split]['latents']['future_teacher'], b[split]['latents']['future_teacher']
            guard[split] = {'target_centered_rms_ratio': right['centered_rms']/left['centered_rms'],
                'target_effective_rank_ratio': right['entropy_effective_rank']/left['entropy_effective_rank']}
        future_better = b['validation']['future']['mse_over_training_teacher_mean_constant'] < a['validation']['future']['mse_over_training_teacher_mean_constant']
        guard_pass = all(value >= protocol['collapse_guard_ratio'] for g in guard.values() for value in g.values())
        delta = {name: b['validation']['service']['all'][name]-a['validation']['service']['all'][name]
                 for name in ('miss_mae', 'gap_mae_seconds')}
        rows.append({'seed': seed, 'validation_projected_minus_shared': delta,
            'future_normalized_mse_improves': future_better, 'target_collapse_guard': guard,
            'target_collapse_guard_pass': guard_pass,
            'both_service_maes_nonincreasing': all(value <= 0 for value in delta.values())})
    return {'paired_seed_results': rows,
        'all_seeds_future_improve_with_target_guard': all(r['future_normalized_mse_improves'] and r['target_collapse_guard_pass'] for r in rows),
        'all_seeds_both_service_maes_nonincreasing': all(r['both_service_maes_nonincreasing'] for r in rows),
        'limits': ['Old validation is exposed development data, never a new confirmation set.',
            'Projection adds training capacity; shared identity control has no capacity-matching auxiliary parameters.',
            'Each variant learns a different EMA target; raw cross-variant latent MSE alone is not comparable.',
            'No DINO visual features, new architecture/novelty claim, reserved holdout or native control evaluation.']}


def run(root):
    root = Path(root).resolve()
    torch.set_num_threads(1)
    out = root/'artifacts/representation_repair'
    protocol_path = out/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    data_hashes = {}
    train = data_arrays(root, protocol, 'training', data_hashes)
    validation = data_arrays(root, protocol, 'validation', data_hashes)
    source_paths = [root/name for name in (
        'src/wirelessjev/representation_repair.py', 'scripts/train_representation_repair.py',
        'tests/test_representation_repair.py', 'src/wirelessjev/packet_value/model.py',
        'src/wirelessjev/packet_value/study.py', 'src/wirelessjev/packet_value/features.py',
        'artifacts/packet_value/protocol.json', 'uv.lock', 'artifacts/packet_value/training.json')]
    original = json.loads((root/'artifacts/packet_value/training.json').read_text())
    if digest(source_paths[3]) != original['source_sha256']['src/wirelessjev/packet_value/model.py']:
        raise ValueError('Original JEPA loss/inference implementation changed')
    manifest = register(root, protocol_path, source_paths, data_hashes)
    fits = [fit_one(root, protocol, manifest, train, validation, variant, seed)
            for seed in protocol['training_seeds'] for variant in protocol['variants']]
    result = {'completed': True, 'fits': len(fits), 'protocol_sha256': digest(protocol_path),
        'preregistration_sha256': digest(out/'preregistration.json'),
        'preregistration_commit': manifest['commit'], 'source_sha256': manifest['source_sha256'],
        'checkpoint_sha256': {f"{r['variant']}-{r['seed']}": r['checkpoint_sha256'] for r in fits},
        'summary': summarize(fits, protocol), 'reserved_holdouts_accessed': False,
        'native_control_evaluated': False, 'sota_or_novelty_established': False,
        'fit_wall_seconds_total': sum(r['elapsed_seconds'] for r in fits)}
    write(out/'completion.json', result)
    return result
