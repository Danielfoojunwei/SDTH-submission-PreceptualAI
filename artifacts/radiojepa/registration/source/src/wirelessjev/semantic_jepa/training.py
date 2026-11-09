"""Registered matched fits. Future radio states are training labels only."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
import hashlib, importlib, io, json, math, os, tempfile, time
from pathlib import Path
import numpy as np
import torch
from ..fleetwm.data import FleetInputs
from ..native.world import HORIZONS

FIELDS = tuple(f.name for f in fields(FleetInputs))
LABELS = ('truth', 'future_context', 'valid_link')
METADATA = ('site_id', 'decision_time')
FAMILIES = ('supervised', 'context_jepa', 'semantic_aux', 'semantic_jepa')
NPZ_IDENTITY = '__registered_identity_utf8__'


def _common(): return importlib.import_module('.common', __package__)
def _model_module(): return importlib.import_module('.model', __package__)


def _identity(split=None):
    c = _common(); c.check_sources()
    result = {'source_sha256': c.source_hashes(), 'protocol': c.protocol(),
              'software': {'numpy': str(np.__version__), 'torch': str(torch.__version__)}}
    if split is not None:
        if split not in ('training', 'validation'): raise ValueError('Only registered training/validation data')
        result['split'] = split
    return result


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _atomic_bytes(path, data, *, immutable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        if immutable:
            try: os.link(temp, path)
            except FileExistsError:
                if path.read_bytes() != data: raise RuntimeError(f'Immutable bytes differ: {path}')
        else: os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def _torch_bytes(value):
    buffer = io.BytesIO(); torch.save(value, buffer); return buffer.getvalue()


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def training_episode(task):
    """Construct past-only inputs before adding private labels in separate fields."""
    from ..fleetwm.data import scan_context
    from .collection import collect
    from ..frontier.fast import FleetMemory, fit_reference
    split, index, seed = task; identity = _identity(split); p = identity['protocol']
    seeds = p[split + '_sites']
    if not 0 <= index < len(seeds) or seeds[index] != seed: raise ValueError('Unregistered episode')
    torch.set_num_threads(1)
    cycle = p.get('training_conditions', ['main', 'main', 'stress', 'drift'])
    val = p.get('validation_conditions', ['main', 'main', 'stress', 'drift', 'main', 'main'])
    condition = cycle[index % len(cycle)] if split == 'training' else val[index % len(val)]
    n, duration = p['training_robots'], p['training_decisions']
    full, g, obs, collected = collect(seed, n, duration, condition, index)
    expected_collection = {'seed': seed, 'robots': n, 'query_duration_seconds': duration,
                           'collection_duration_seconds': duration + int(max(HORIZONS)),
                           'condition': condition, 'index': index}
    if any(collected.get(key) != value for key, value in expected_collection.items()):
        raise RuntimeError('Collection returned a different site/query identity')
    if (collected.get('postwarm_observations_use_actual_deliveries_only') is not True
            or collected.get('future_truth_is_predictor_input') is not False):
        raise RuntimeError('Collection observation boundary is not validated')
    if len(obs.value) < 60 + duration + int(max(HORIZONS)) or obs.value.shape[1] != n:
        raise RuntimeError('Delivered collection cannot cover all training labels')
    receipt = {'identity': identity, 'split': split, 'site': seed, 'condition': condition,
               'query_duration_seconds': duration, 'collection': collected,
               'delivered_history_length': len(obs.value),
               'delivered_history_sha256': {key: hashlib.sha256(
                   np.ascontiguousarray(getattr(obs, key)).tobytes()).hexdigest()
                   for key in ('positions', 'value', 'heard', 'bound', 'scanned')}}
    _common().check_sources()
    _common().write_once(_common().OUT / 'collection' / f'{split}-{seed}.json', receipt)
    memory = FleetMemory(g)
    rows = []; rmap = None
    for t in range(60, 60 + duration, p['training_query_stride']):
        past = obs.before(t)
        if rmap is None or (t - 60) % p.get('map_refresh_seconds', 20) == 0:
            rmap = fit_reference(g, past, strong=True)
        observed, _ = memory.inputs(past, full.plan[t], rmap)
        row = {k: getattr(observed, k) for k in FIELDS}; future = t + np.asarray(HORIZONS, dtype=int)
        row['truth'] = full.snr[future * 5].transpose(1, 0, 2, 3).reshape(n, len(HORIZONS), -1)
        row['future_context'] = np.stack([scan_context(g, obs.before(int(tau))) for tau in future], 1)
        row['valid_link'] = np.ones(row['context'].shape[:2], dtype=np.float32)
        row['site_id'] = np.full(n, seed, dtype=np.int64); row['decision_time'] = np.full(n, t, dtype=np.int16)
        rows.append(row)
    result = {k: np.concatenate([row[k] for row in rows]) for k in rows[0]}
    for k in FIELDS + LABELS: result[k] = result[k].astype(np.float16)
    _common().check_sources()
    print(json.dumps({'phase': 'semantic_data', 'split': split, 'site': seed,
                      'condition': condition, 'rows': len(result['context'])}), flush=True)
    return result


def _validate_arrays(arrays, identity):
    p, split = identity['protocol'], identity['split']
    if set(arrays) != set(FIELDS + LABELS + METADATA): raise RuntimeError('Dataset schema changed')
    times = list(range(60, 60 + p['training_decisions'], p['training_query_stride']))
    count = len(p[split + '_sites']) * p['training_robots'] * len(times); ctx = arrays['context']
    if ctx.ndim != 3 or ctx.shape[0] != count or ctx.shape[2] != 48: raise RuntimeError('Context dimensions changed')
    b, links, _ = ctx.shape
    if links < 3 or links % 3: raise RuntimeError('Incomplete AP/band axis')
    shapes = {'query': (b, 9, links, 11), 'memory': (b, links, 13), 'truth': (b, 9, links),
              'future_context': (b, 9, links, 48), 'valid_link': (b, links), 'site_id': (b,), 'decision_time': (b,)}
    shapes.update({k: (b, 9, links) for k in ('physics', 'map_mean', 'map_std', 'fleet_mean')})
    for k, expected in shapes.items():
        if arrays[k].shape != expected: raise RuntimeError(f'Dataset dimension changed: {k}')
    for k, v in arrays.items():
        if not np.isfinite(v).all(): raise RuntimeError(f'Nonfinite dataset: {k}')
    if np.any((arrays['valid_link'] != 0) & (arrays['valid_link'] != 1)): raise RuntimeError('Invalid link mask')
    if np.any(arrays['map_std'] < 0): raise RuntimeError('Negative map uncertainty')
    for seed in p[split + '_sites']:
        for decision in times:
            if ((arrays['site_id'] == seed) & (arrays['decision_time'] == decision)).sum() != p['training_robots']:
                raise RuntimeError('Dataset site/query membership changed')
    if set(np.unique(arrays['site_id']).tolist()) != set(p[split + '_sites']): raise RuntimeError('Unregistered site')


def _dataset_receipt(path, identity, arrays):
    collection_hashes = {}
    for seed in identity['protocol'][identity['split'] + '_sites']:
        name = f"{identity['split']}-{seed}"
        receipt_path = _common().OUT / 'collection' / (name + '.json')
        receipt = json.loads(receipt_path.read_text())
        if receipt['identity'] != identity or receipt['site'] != seed or receipt['split'] != identity['split']:
            raise RuntimeError('Collection receipt identity changed')
        collection_hashes[name] = _common().digest(receipt_path)
    return {'completed': True, 'identity': identity, 'npz_sha256': _common().digest(path),
            'collection_receipt_sha256': collection_hashes,
            'rows': len(arrays['context']), 'sites': identity['protocol'][identity['split'] + '_sites'],
            'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                           'sha256': hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()}
                       for k, v in arrays.items()},
            'strictly_past_observable_inputs': True, 'actual_delivered_training_observations': True,
            'future_states_are_training_labels_only': True}


def dataset(split, workers=1, *, rebuild_missing=False):
    """Source-bound cache; missing or changed registered caches fail closed."""
    if workers not in (1, 2, 3, 4): raise ValueError('Use one to four workers')
    identity = _identity(split); c = _common()
    path = c.WORK / (split + '.npz'); receipt_path = c.OUT / 'data' / (split + '.json')
    if receipt_path.exists() and not path.exists() and not rebuild_missing:
        raise RuntimeError('Registered dataset cache missing; use explicit rebuild_missing=True')
    if not path.exists():
        tasks = [(split, i, s) for i, s in enumerate(identity['protocol'][split + '_sites'])]
        if workers == 1: pieces = [training_episode(task) for task in tasks]
        else:
            with ProcessPoolExecutor(workers) as pool: pieces = list(pool.map(training_episode, tasks))
        arrays = {k: np.concatenate([piece[k] for piece in pieces]) for k in pieces[0]}
        _validate_arrays(arrays, identity)
        data = dict(arrays); data[NPZ_IDENTITY] = np.frombuffer(_json_bytes(identity), dtype=np.uint8)
        buffer = io.BytesIO(); np.savez_compressed(buffer, **data)
        _atomic_bytes(path, buffer.getvalue(), immutable=True)
    with np.load(path, allow_pickle=False) as archive:
        if NPZ_IDENTITY not in archive or json.loads(archive[NPZ_IDENTITY].tobytes().decode()) != identity:
            raise RuntimeError('Dataset embedded identity changed')
        arrays = {k: archive[k] for k in archive.files if k != NPZ_IDENTITY}
    _validate_arrays(arrays, identity); receipt = _dataset_receipt(path, identity, arrays)
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt: raise RuntimeError('Dataset receipt/cache changed')
    c.check_sources(); c.write_once(receipt_path, receipt)
    return arrays


def batch(arrays, index):
    return {k: torch.as_tensor(np.array(arrays[k][index], dtype=np.float32, copy=True)) for k in FIELDS + LABELS}


def validation_metrics(model, arrays, batch_size=128):
    """One common score; padded bands/APs cannot win the AP argmax."""
    model.eval(); squared = weight_sum = 0.; regrets = []
    with torch.no_grad():
        for start in range(0, len(arrays['context']), batch_size):
            b = batch(arrays, slice(start, start + batch_size)); mean = model(b)['mean']; truth = b['truth']
            if not torch.isfinite(mean).all(): raise FloatingPointError('Nonfinite validation prediction')
            weight = torch.where(truth > 0, 1., .2) * b['valid_link'][:, None]
            squared += float(((mean - truth).square() * weight).sum()); weight_sum += float(weight.sum())
            n, h, links = mean.shape; valid_band = b['valid_link'].bool().reshape(n, links // 3, 3)
            ap_valid = valid_band.any(-1)
            if not ap_valid.any(-1).all(): raise ValueError('Every row needs a real AP')
            def scores(v):
                v = v[:, :min(6, h)].reshape(n, min(6, h), links // 3, 3)
                return v.masked_fill(~valid_band[:, None], -1e6).max(-1).values.mean(1).masked_fill(~ap_valid, -1e6)
            predicted, actual = scores(mean), scores(truth)
            regrets.extend((actual.max(-1).values - actual.gather(1, predicted.argmax(-1)[:, None])[:, 0]).tolist())
    if weight_sum <= 0 or not regrets: raise ValueError('No validation targets')
    rmse = math.sqrt(squared / weight_sum); regret = float(np.mean(regrets))
    return {'validation_weighted_rmse': rmse, 'validation_ap_regret': regret, 'validation_score': rmse + .5 * regret}


def _checkpoint_matches(saved, expected):
    if set(saved) != set(expected): return False
    for k in saved:
        if k == 'state':
            if set(saved[k]) != set(expected[k]) or any(not torch.equal(saved[k][n], expected[k][n]) for n in saved[k]): return False
        elif saved[k] != expected[k]: return False
    return True


def _jsonable(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    return value


def fit_one(task):
    """Resume a complete source-bound epoch, including optimizer and both RNGs."""
    family, seed = task; identity = _identity(); p = identity['protocol']; c = _common()
    if family not in p['families'] or family not in FAMILIES or seed not in p['training_seeds']:
        raise ValueError('Unregistered fit family/seed')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    checkpoint = c.OUT / 'checkpoints' / f'{family}-{seed}.pt'
    report_path = c.OUT / 'fits' / f'{family}-{seed}.json'
    if report_path.exists():
        result = json.loads(report_path.read_text())
        if (result['identity'] != identity or result['family'] != family or result['seed'] != seed
                or not result['completed'] or not checkpoint.exists()): raise RuntimeError('Completed fit identity changed')
        if result['checkpoint_sha256'] != c.digest(checkpoint): raise RuntimeError('Completed checkpoint changed')
        for split, expected in result['data_receipt_sha256'].items():
            if c.digest(c.OUT / 'data' / (split + '.json')) != expected: raise RuntimeError('Dataset receipt changed after fit')
        return result
    training = dataset('training', 1); validation = dataset('validation', 1)
    data_hashes = {s: c.digest(c.OUT / 'data' / (s + '.json')) for s in ('training', 'validation')}
    module = _model_module(); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = module.SemanticJEPA(family=family, jepa_weight=p.get('auxiliary_weight', p.get('jepa_weight', .1)), hidden=p.get('hidden', 32))
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],
                                 lr=p.get('learning_rate', 7e-4), weight_decay=p.get('weight_decay', 2e-4))
    last = c.WORK / f'{family}-{seed}.last.pt'; history = []
    first = best_epoch = 0; elapsed_before = 0.; best = float('inf'); best_state = best_metrics = None
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=True)
        if (saved['identity'] != identity or saved['family'] != family or saved['seed'] != seed
                or saved['data_receipt_sha256'] != data_hashes): raise RuntimeError('Resume identity changed')
        model.load_state_dict(saved['state']); optimizer.load_state_dict(saved['optimizer'])
        history, first = saved['history'], saved['completed_epochs']
        best, best_state = saved['best_score'], saved['best_state']
        best_epoch, best_metrics = saved['best_epoch'], saved['best_metrics']
        elapsed_before = saved['elapsed_seconds']; rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        if first != len(history) or not 0 <= first <= p['epochs']: raise RuntimeError('Malformed resume history')
    started = time.perf_counter()
    for epoch in range(first, p['epochs']):
        model.train(); order = rng.permutation(len(training['context'])); losses = []
        size = p.get('training_batch_size', 64)
        for start in range(0, len(order), size):
            cost, _ = module.objective(model, batch(training, order[start:start + size]))
            if cost.ndim != 0 or not torch.isfinite(cost): raise FloatingPointError('Invalid training loss')
            optimizer.zero_grad(set_to_none=True); cost.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient')
            optimizer.step(); module.update_target(model); losses.append(float(cost.detach()))
        metrics = validation_metrics(model, validation)
        row = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), **metrics}; history.append(row)
        if metrics['validation_score'] < best:
            best = metrics['validation_score']; best_state = _cpu_state(model)
            best_epoch, best_metrics = epoch + 1, metrics
        c.check_sources()
        resume = {'identity': identity, 'family': family, 'seed': seed, 'data_receipt_sha256': data_hashes,
                  'state': _cpu_state(model), 'optimizer': optimizer.state_dict(), 'history': history,
                  'completed_epochs': epoch + 1, 'best_score': best, 'best_state': best_state,
                  'best_epoch': best_epoch, 'best_metrics': best_metrics, 'numpy_rng': rng.bit_generator.state,
                  'torch_rng': torch.get_rng_state(), 'elapsed_seconds': elapsed_before + time.perf_counter() - started}
        _atomic_bytes(last, _torch_bytes(resume))
        _atomic_bytes(c.WORK / f'{family}-{seed}.status.json', _json_bytes(row))
        print(json.dumps({'phase': 'semantic_fit', 'family': family, 'seed': seed, **row}), flush=True)
    if best_state is None or not history: raise RuntimeError('No completed validated checkpoint')
    model.load_state_dict(best_state); model.eval()
    final = {'config': model.config, 'state': best_state, 'epoch': best_epoch,
             'family': family, 'seed': seed, 'identity': identity, 'data_receipt_sha256': data_hashes}
    if checkpoint.exists():
        if not _checkpoint_matches(torch.load(checkpoint, map_location='cpu', weights_only=True), final):
            raise RuntimeError('Incomplete fit has a different immutable checkpoint')
    else: _atomic_bytes(checkpoint, _torch_bytes(final), immutable=True)
    diagnostics = None
    if hasattr(module, 'diagnostics'):
        with torch.no_grad():
            diagnostics = _jsonable(module.diagnostics(model, batch(validation, slice(0, min(128, len(validation['context']))))))
    c.check_sources()
    result = {'completed': True, 'family': family, 'seed': seed, 'identity': identity, 'history': history,
              'best_epoch': best_epoch, 'best_validation_score': best, 'best_validation_metrics': best_metrics,
              'elapsed_seconds': elapsed_before + time.perf_counter() - started,
              'parameters_total_including_training_only': sum(v.numel() for v in model.parameters()),
              'parameters_trainable': sum(v.numel() for v in model.parameters() if v.requires_grad),
              'parameters_inference': int(getattr(model, 'inference_parameters', sum(v.numel() for v in model.parameters()))),
              'checkpoint_sha256': c.digest(checkpoint), 'data_receipt_sha256': data_hashes,
              'diagnostics': diagnostics, 'diagnostic_sample_rows': min(128, len(validation['context'])),
              'confirmation_accessed': False, 'development_control_accessed': False,
              'privileged_simulator_truth_is_training_only': True}
    _json_bytes(result); c.write_once(report_path, result)
    return result


def train(workers=1):
    if workers not in (1, 2, 3, 4): raise ValueError('Use one to four training workers')
    identity = _identity(); c = _common(); p = c.protocol()
    dataset('training', workers); dataset('validation', workers)
    tasks = [(f, s) for f in p['families'] for s in p['training_seeds']]
    if workers == 1: reports = [fit_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(workers) as pool: reports = list(pool.map(fit_one, tasks))
    c.check_sources()
    result = {'completed': True, 'identity': identity, 'fits': len(reports), 'families': p['families'],
              'training_seeds': p['training_seeds'],
              'checkpoint_sha256': {str((c.OUT / 'checkpoints' / f'{f}-{s}.pt').relative_to(c.ROOT)):
                                    c.digest(c.OUT / 'checkpoints' / f'{f}-{s}.pt') for f, s in tasks},
              'fit_receipt_sha256': {f'{f}-{s}': c.digest(c.OUT / 'fits' / f'{f}-{s}.json') for f, s in tasks},
              'data_receipt_sha256': {s: c.digest(c.OUT / 'data' / (s + '.json')) for s in ('training', 'validation')},
              'total_fit_elapsed_seconds': float(sum(r['elapsed_seconds'] for r in reports)),
              'confirmation_accessed': False, 'development_control_accessed': False}
    c.write_once(c.OUT / 'training.json', result)
    return result


def load(family, seed):
    identity = _identity(); c = _common()
    if family not in identity['protocol']['families'] or family not in FAMILIES or seed not in identity['protocol']['training_seeds']:
        raise ValueError('Unregistered frozen fit family/seed')
    report = json.loads((c.OUT / 'fits' / f'{family}-{seed}.json').read_text())
    checkpoint = c.OUT / 'checkpoints' / f'{family}-{seed}.pt'
    if (report['identity'] != identity or report['family'] != family or report['seed'] != seed
            or not report['completed'] or c.digest(checkpoint) != report['checkpoint_sha256']):
        raise RuntimeError('Frozen fit identity or checkpoint changed')
    for split, expected in report['data_receipt_sha256'].items():
        if c.digest(c.OUT / 'data' / (split + '.json')) != expected:
            raise RuntimeError('Data receipt differs from frozen fit')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['identity'] != identity or saved['family'] != family or saved['seed'] != seed:
        raise RuntimeError('Checkpoint provenance differs from receipt')
    model = _model_module().SemanticJEPA(**saved['config']); model.load_state_dict(saved['state']); model.eval()
    return model


@torch.no_grad()
def predict(model, observed: FleetInputs):
    """A single member's observable-only flat AP/band forecast."""
    if not isinstance(observed, FleetInputs): raise TypeError('Use the explicit label-free FleetInputs boundary')
    n, horizons, links = observed.map_mean.shape
    if n < 1 or horizons != len(HORIZONS) or links % 3: raise ValueError('Unexpected forecast dimensions')
    means, stds = [], []; model.eval()
    for start in range(0, n, 128):
        b = {k: torch.as_tensor(np.array(getattr(observed, k)[start:start + 128], dtype=np.float32, copy=True)) for k in FIELDS}
        b['valid_link'] = torch.ones((min(128, n - start), links), dtype=torch.float32)
        result = model(b); mean, std = result['mean'], result['std']; expected = (min(128, n - start), horizons, links)
        if (tuple(mean.shape) != expected or tuple(std.shape) != expected or not torch.isfinite(mean).all()
                or not torch.isfinite(std).all() or (std <= 0).any()): raise FloatingPointError('Invalid neural forecast')
        means.append(mean.cpu().numpy().copy()); stds.append(std.cpu().numpy().copy())
    return np.concatenate(means), np.concatenate(stds)
