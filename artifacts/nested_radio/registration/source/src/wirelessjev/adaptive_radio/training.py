"""Source-bound, resumable fitting; whole-layout conditional-change validation."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import io
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from . import common as c, data
from .model import AdaptiveScene, CELLS, objective

SEEDS = (0, 1, 2)


def tensor(value):
    value = np.array(value, copy=True)
    return torch.from_numpy(value if value.dtype == bool else value.astype('f4'))


def batch(examples):
    fields = ('context_a', 'context_b', 'mask_a', 'mask_b', 'scheduled_mask_a', 'scheduled_mask_b',
              'query', 'query_basis', 'query_fspl_dbm', 'query_offset_db')
    result = {name: tensor(np.stack([getattr(e, name) for e in examples])) for name in fields}
    result['targets'] = {name: tensor(np.stack([e.target_arrays()[name] for e in examples]))
                         for name in examples[0].target_arrays()}
    return result


def identity():
    c.check_sources()
    return {'source_receipt_sha256': c.digest(c.OUT / 'registration/source.json'),
            'protocol_sha256': c.digest(c.PROTOCOL),
            'data_receipt_sha256': c.digest(c.OUT / 'data/prepared-training.json'),
            'rt_stage_receipts_sha256': {stage: c.digest(c.OUT / 'rt_data' / f'{stage}.json')
                                        for stage in ('training', 'validation')}}


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.resume-', suffix='.pt', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_history(history, *, complete=False):
    if not isinstance(history, list) or len(history) > 60 or (complete and len(history) != 60):
        raise RuntimeError('Training history budget changed')
    for epoch, entry in enumerate(history):
        if (type(entry['epoch']) is not int or entry['epoch'] != epoch or len(entry['updates']) != 8
                or not np.isfinite(entry['validation_mean_layout_delta_rmse_db'])
                or any(not isinstance(u, dict) or 'loss' not in u
                       or not np.isfinite(list(u.values())).all() for u in entry['updates'])):
            raise RuntimeError('Malformed or nonfinite optimization history')


def verify_completed(receipt, ident, cell, seed):
    path = c.OUT / 'checkpoints' / f'{cell}-seed{seed}.pt'
    if (receipt.get('completed') is not True or receipt['cell'] != cell
            or type(receipt['seed']) is not int or receipt['seed'] != seed
            or receipt['run_id'] != f'{cell}-seed{seed}' or receipt['identity'] != ident
            or type(receipt['epochs']) is not int or type(receipt['updates']) is not int
            or type(receipt['selected_epoch']) is not int
            or receipt['epochs'] != 60 or receipt['updates'] != 480
            or receipt['settings'] != {'cell': cell, 'width': 64, 'em_steps': 3}
            or receipt['checkpoint_sha256'] != c.digest(path)):
        raise RuntimeError('Completed fit identity/budget/checkpoint changed')
    validate_history(receipt['history'], complete=True)
    scores = [h['validation_mean_layout_delta_rmse_db'] for h in receipt['history']]
    selected = int(np.argmin(scores))
    if (receipt['selected_epoch'] != selected
            or receipt['validation_mean_layout_delta_rmse_db'] != min(scores)):
        raise RuntimeError('Checkpoint differs from first-minimum sparse VAL rule')
    checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    if (type(checkpoint['seed']) is not int or type(checkpoint['selected_epoch']) is not int
            or any(checkpoint[k] != receipt[k] for k in ('identity', 'cell', 'seed', 'settings', 'selected_epoch'))):
        raise RuntimeError('Checkpoint metadata differs')
    if (not isinstance(checkpoint['state'],dict) or not checkpoint['state']
            or not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in checkpoint['state'].values())):
        raise RuntimeError('Nonfinite checkpoint tensor')
    return checkpoint


@torch.no_grad()
def predictions(model, values):
    belief = model.encode(values['context_a'], values['mask_a'], values['scheduled_mask_a'])
    means, stds = model.render(belief, values['query'], values['query_basis'],
                              values['query_fspl_dbm'], values['query_offset_db'])
    return list(zip(means.numpy(), stds.numpy()))


def fit(task):
    cell, seed = task
    if cell not in CELLS or type(seed) is not int or seed not in SEEDS:
        raise ValueError('Unregistered training member')
    ident = identity()
    name = f'{cell}-seed{seed}'
    receipt_path = c.OUT / 'fits' / f'{name}.json'
    if receipt_path.exists():
        receipt = c.read_json(receipt_path)
        verify_completed(receipt, ident, cell, seed)
        return receipt
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AdaptiveScene(cell)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1.e-4)
    full = data.examples('training', regime='full')
    sparse = data.examples('training', regime='sparse')
    expected = 24 * 3
    if len(full) != expected or len(sparse) != expected:
        raise RuntimeError('Expected 24 TRAIN layouts with all three RF states')
    for group in (full, sparse):
        if any(e.condition != i % 3 or e.layout_id != group[(i // 3) * 3].layout_id
               for i, e in enumerate(group)):
            raise RuntimeError('Training cases are not in registered layout-major order')
    training = {'full': batch(full), 'sparse': batch(sparse)}
    validation_examples = data.examples('validation', regime='sparse')
    validation = batch(validation_examples)
    from .study import validation_score

    history, best, best_epoch, best_state, elapsed = [], float('inf'), None, None, 0.
    resume = c.WORK / f'{name}.last.pt'
    if resume.exists():
        prior = torch.load(resume, weights_only=True, map_location='cpu')
        if (prior['identity'] != ident or prior['cell'] != cell or type(prior['seed']) is not int or prior['seed'] != seed
                or prior['run_id'] != name or prior['settings'] != model.config):
            raise RuntimeError('Wrong source/member resume state')
        validate_history(prior['history'])
        model.load_state_dict(prior['state'])
        optimizer.load_state_dict(prior['optimizer'])
        torch.set_rng_state(prior['torch_rng'])
        rng.bit_generator.state = prior['numpy_rng']
        history, best, best_epoch, best_state, elapsed = (prior[k] for k in
            ('history', 'best', 'best_epoch', 'best_state', 'elapsed_seconds'))
        if history:
            scores = [h['validation_mean_layout_delta_rmse_db'] for h in history]
            if best != min(scores) or best_epoch != int(np.argmin(scores)):
                raise RuntimeError('Resume validation selection changed')
    began = time.perf_counter()
    for epoch in range(len(history), 60):
        model.train()
        logs = []
        for update in range(8):
            selected = np.arange(24) * 3 + rng.integers(0, 3, size=24)
            regime = 'full' if update % 2 == 0 else 'sparse'
            values = training[regime]
            selected_values = {k: (v[selected] if k != 'targets' else
                                   {a: b[selected] for a, b in v.items()}) for k, v in values.items()}
            v = selected_values
            optimizer.zero_grad(set_to_none=True)
            loss, details = objective(model,
                (v['context_a'], v['mask_a'], v['scheduled_mask_a']),
                (v['context_b'], v['mask_b'], v['scheduled_mask_b']),
                v['query'], v['query_basis'], v['query_fspl_dbm'], v['query_offset_db'], v['targets'])
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            logs.append({**{k: float(x) for k, x in details.items()},
                         'total_gradient_before_clip': float(norm)})
        model.eval()
        score = validation_score(predictions(model, validation), validation_examples)
        if not np.isfinite(score):
            raise FloatingPointError('Nonfinite sparse condition-change VAL score')
        if score < best:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({'epoch': epoch, 'validation_mean_layout_delta_rmse_db': score, 'updates': logs})
        atomic_save(resume, {'identity': ident, 'cell': cell, 'seed': seed, 'run_id': name,
            'settings': model.config, 'state': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'torch_rng': torch.get_rng_state(), 'numpy_rng': rng.bit_generator.state,
            'history': history, 'best': best, 'best_epoch': best_epoch, 'best_state': best_state,
            'elapsed_seconds': elapsed + time.perf_counter() - began})
    elapsed += time.perf_counter() - began
    model.load_state_dict(best_state)
    checkpoint_path = c.OUT / 'checkpoints' / f'{name}.pt'
    stream = io.BytesIO()
    torch.save({'identity': ident, 'cell': cell, 'seed': seed, 'settings': model.config,
                'state': best_state, 'selected_epoch': best_epoch}, stream)
    c.write_bytes_once(checkpoint_path, stream.getvalue())
    receipt = {'schema': 1, 'completed': True, 'cell': cell, 'seed': seed, 'run_id': name,
        'identity': ident, 'settings': model.config, 'epochs': 60, 'updates': 480,
        'history': history, 'selected_epoch': best_epoch,
        'validation_mean_layout_delta_rmse_db': best,
        'checkpoint_sha256': c.digest(checkpoint_path), 'elapsed_seconds': elapsed,
        'learned_parameters': sum(p.numel() for p in model.parameters()),
        'batch_distinct_layouts': 24, 'training_bands': [0, 1],
        'uncertainty_calibrated': False, 'packet_or_production_qualified': False}
    c.write_once(receipt_path, receipt)
    return receipt


def train_all(workers=2):
    if workers not in (1, 2):
        raise ValueError('Use one or two bounded CPU workers')
    # Prepare deterministic receipt files before concurrent readers arrive.
    for stage, regime in (('training', 'full'), ('training', 'sparse'), ('validation', 'sparse')):
        data.examples(stage, regime=regime)
    tasks = [(cell, seed) for cell in CELLS for seed in SEEDS]
    if workers == 1:
        return [fit(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fit, tasks))


def load(cell, seed):
    if cell not in CELLS or type(seed) is not int or seed not in SEEDS:
        raise ValueError('Unregistered checkpoint')
    receipt = c.read_json(c.OUT / 'fits' / f'{cell}-seed{seed}.json')
    checkpoint = verify_completed(receipt, identity(), cell, seed)
    model = AdaptiveScene(**checkpoint['settings'])
    model.load_state_dict(checkpoint['state'])
    model.eval()
    return model


@torch.no_grad()
def encode(model, example, view='a'):
    if view not in ('a', 'b'):
        raise ValueError('Unknown source view')
    return model.encode(tensor(getattr(example, 'context_' + view))[None],
                        tensor(getattr(example, 'mask_' + view))[None],
                        tensor(getattr(example, 'scheduled_mask_' + view))[None])


@torch.no_grad()
def predict(model, example, view='a', *, context_override=None, z_override=None):
    source = example if context_override is None else context_override
    belief = encode(model, source, view)
    if z_override is not None:
        belief = {name: tensor(z_override[name]).reshape_as(belief[name]) for name in ('mean', 'covariance')}
    mean, std = model.render(belief, tensor(example.query)[None], tensor(example.query_basis)[None],
                             tensor(example.query_fspl_dbm)[None], tensor(example.query_offset_db)[None])
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise FloatingPointError('Invalid field forecast')
    return mean[0].numpy(), std[0].numpy()
