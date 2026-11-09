"""Read-only, training-only diagnostics of existing wireless representations.

Geometry, information accessibility, action response and predictive accuracy
are separate questions. A common direction/high cosine is not constant collapse.
No model fitting, checkpoint selection or control promotion occurs here; the
small linear readouts are explicitly post-hoc diagnostics on model-exposed data.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import torch

from .packet_value.model import PacketValueModel
from .service_world.features import GLOBAL_FEATURES, LINK_FEATURES
from .service_world.model import ServiceWorldModel


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _matrix(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError('Need a finite nonempty examples-by-features matrix')
    return x


def representation_stats(values):
    """Centered covariance spectrum and both raw/centered cosine geometry.

    Entropy rank uses covariance eigenvalues (squared singular values), not
    singular-value entropy. Spectra are invariant to orthogonal changes of
    latent coordinates. Per-axis standard deviations are descriptive only.
    """
    source_dtype = np.asarray(values).dtype
    source_eps = np.finfo(source_dtype).eps if source_dtype.kind == 'f' else np.finfo(float).eps
    x = _matrix(values)
    centered = x - x.mean(0)
    if np.array_equal(x, np.broadcast_to(x[0], x.shape)):
        centered.fill(0.)  # avoid reduction roundoff on an exactly constant batch
    singular = np.linalg.svd(centered, compute_uv=False)
    eigen = singular**2 / max(len(x) - 1, 1)
    total = float(eigen.sum())
    energy = float(np.mean(np.sum(x*x, -1)))
    centered_energy = float(np.mean(np.sum(centered*centered, -1)))
    rms = float(np.sqrt(np.mean(x*x)))
    centered_rms = float(np.sqrt(np.mean(centered*centered)))
    # A numerical diagnostic at the model's float32 observation precision;
    # absolute units and the centered/raw energy ratio remain visible.
    constant = centered_rms <= 1e-6 * max(1., rms)
    if total > np.finfo(float).tiny:
        p = eigen[eigen > 0] / total
        rank = float(np.exp(-(p*np.log(p)).sum()))
        participation = float(total**2 / (eigen*eigen).sum())
        top = float(eigen[0] / total)
    else:
        rank = participation = top = 0.
    rank_tolerance = float(singular.max(initial=0.) * max(x.shape) * source_eps)
    numerical_rank = int((singular > rank_tolerance).sum())
    if constant and total == 0:
        numerical_rank = 0

    def cosine(z):
        norm = np.linalg.norm(z, axis=1)
        z = z[norm > 1e-12]
        norm = norm[norm > 1e-12]
        if len(z) < 2:
            return {'pairs': 0, 'mean': None, 'median': None, 'p05': None, 'p95': None}
        z = z / norm[:, None]
        # Deterministic distinct pairs, no self-pair inflation and no NxN RAM.
        i, j = np.triu_indices(min(len(z), 256), 1)
        if len(z) > 256:
            keep = np.linspace(0, len(z)-1, 256, dtype=int)
            z = z[keep]
        c = np.clip(np.einsum('ij,ij->i', z[i], z[j]), -1, 1)
        return {'pairs': len(c), 'mean': float(c.mean()), 'median': float(np.median(c)),
                'p05': float(np.quantile(c, .05)), 'p95': float(np.quantile(c, .95))}

    std = centered.std(0)
    return {'examples': len(x), 'dimensions': x.shape[1],
            'maximum_estimable_rank': min(len(x)-1, x.shape[1]),
            'numerically_constant_on_sample': bool(constant),
            'centered_rms': centered_rms, 'raw_rms': rms,
            'centered_energy_fraction': centered_energy / energy if energy else 0.,
            'covariance_trace': total, 'entropy_effective_rank': rank,
            'participation_rank': participation, 'numerical_rank': numerical_rank,
            'numerical_rank_singular_value_tolerance': rank_tolerance,
            'input_numeric_precision': str(source_dtype),
            'top_pc_variance_fraction': top, 'covariance_eigenvalues': eigen.tolist(),
            'axis_std_min_median_max': [float(std.min()), float(np.median(std)), float(std.max())],
            'axis_std': std.tolist(), 'raw_cosine': cosine(x), 'centered_cosine': cosine(centered)}


def observed_feature_stats(values, observed, names):
    """Unknown entries, including poisoned ones, are excluded rather than zeroed."""
    x, mask = np.asarray(values), np.asarray(observed, dtype=bool)
    if x.shape != mask.shape or x.shape[-1] != len(names):
        raise ValueError('Feature names/masks must match the observed data')
    rows = {}
    for k, name in enumerate(names):
        z = x[..., k][mask[..., k]]
        if not np.isfinite(z).all():
            raise ValueError('An observed feature is nonfinite')
        rows[name] = {'observed_fraction': float(mask[..., k].mean()), 'observations': int(len(z)),
                      'mean': float(z.mean()) if len(z) else None,
                      'std': float(z.std()) if len(z) else None,
                      'minimum': float(z.min()) if len(z) else None,
                      'maximum': float(z.max()) if len(z) else None}
    return rows


def site_folds(scene):
    """Four fixed blocks; never split multiple actions from a scene across folds."""
    scenes = np.unique(scene)
    if len(scenes) < 4:
        raise ValueError('Need at least four training scenes')
    for sites in np.array_split(scenes, 4):
        test = np.isin(scene, sites)
        yield np.flatnonzero(~test), np.flatnonzero(test)


def crossfit_ridge(features, target, scene, *, alpha=10.):
    """Fixed-alpha readouts: all scaling/centering is fit within each site fold."""
    x, y = _matrix(features), _matrix(target)
    if len(x) != len(y) or len(scene) != len(x) or alpha <= 0:
        raise ValueError('Invalid probe arrays or regularization')
    out = np.empty_like(y)
    for train, test in site_folds(scene):
        mean, scale = x[train].mean(0), x[train].std(0)
        scale = np.where(scale > 1e-8, scale, 1.)
        a = (x[train] - mean) / scale
        offset = y[train].mean(0)
        weight = np.linalg.solve(a.T @ a + alpha*np.eye(a.shape[1]), a.T @ (y[train]-offset))
        out[test] = (x[test]-mean) / scale @ weight + offset
    return out


def prediction_metrics(prediction, target, cost):
    p, y = _matrix(prediction), _matrix(target)
    if p.shape != y.shape or y.shape[1] != 2:
        raise ValueError('Predictions/targets must have misses and gap columns')

    def metric(ids):
        rows = {}
        for k, name in enumerate(('missed_commands', 'gap_seconds')):
            error = p[ids, k]-y[ids, k]
            denominator = float(np.sum((y[ids, k]-y[ids, k].mean())**2))
            rows[name] = {'mae': float(np.abs(error).mean()), 'rmse': float(np.sqrt(np.mean(error**2))),
                          'r2': 1-float(np.sum(error**2))/denominator if denominator else None}
        return {'examples': int(np.sum(ids)), **rows}
    return {'all': metric(np.ones(len(y), dtype=bool)),
            'by_cost': {str(float(c)): metric(cost == c) for c in np.unique(cost)}}


def action_sensitivity(query, latent, means, groups):
    """Only actual alternative queries with the same recorded focal history."""
    q, z, p = map(_matrix, (query, latent, means))
    pairs = []
    grouped = {}
    for i, group in enumerate(groups):
        grouped.setdefault(group, []).append(i)
    for indices in grouped.values():
        pairs.extend((i, j) for i, j in itertools.combinations(indices, 2) if not np.array_equal(q[i], q[j]))
    if not pairs:
        return {'pairs': 0}
    i, j = np.asarray(pairs).T
    spread = float(np.sqrt(np.mean((z-z.mean(0))**2)))
    change = np.sqrt(np.mean((z[i]-z[j])**2, -1))
    return {'pairs': len(pairs), 'query_rms_change': float(np.sqrt(np.mean((q[i]-q[j])**2))),
            'latent_rms_change': float(np.mean(change)),
            'latent_change_over_between_example_centered_rms': float(np.mean(change))/spread if spread else None,
            'exactly_equal_latent_pairs': int((change == 0).sum()),
            'mean_absolute_miss_prediction_change': float(np.abs(p[i, 0]-p[j, 0]).mean()),
            'mean_absolute_gap_prediction_change_seconds': float(np.abs(p[i, 1]-p[j, 1]).mean())}


def _probe_predictions(a, representations):
    y = np.stack([a['delta_misses'], a['delta_gap']], -1)
    cost, scene = a['cost'], a['scene']
    constants = np.empty_like(y)
    for train, test in site_folds(scene):
        for c in np.unique(cost):
            constants[test[cost[test] == c]] = y[train[cost[train] == c]].mean(0)
    numeric = np.stack([a['numeric_delta'], constants[:, 1]], -1)
    probes = {'per_cost_training_fold_mean': constants,
              'numeric_misses_plus_constant_gap': numeric,
              'trained_head_model_exposed_data': representations['mean']}
    inputs = {'public_context_and_query': np.concatenate([a['context'].reshape(len(y), -1), a['query']], -1),
              'numeric_and_cost': np.stack([a['numeric_delta'], cost], -1),
              'context_latent_and_query': np.concatenate([representations['context'], a['query']], -1),
              'action_latent': representations['action'],
              'action_latent_and_numeric': np.c_[representations['action'], a['numeric_delta']]}
    for name, x in inputs.items():
        probes['ridge_' + name] = crossfit_ridge(x, y, scene)
    return {name: prediction_metrics(pred, y, cost) for name, pred in probes.items()}


def _input_gradients(model, a):
    # Derivatives at actual recorded inputs: no unphysical swapped trajectory
    # or invented future observation is supplied to the model.
    ids = np.linspace(0, len(a['context'])-1, min(96, len(a['context'])), dtype=int)
    context = torch.tensor(a['context'][ids], requires_grad=True)
    query = torch.tensor(a['query'][ids], requires_grad=True)
    out = model(context, query)
    groups = {'pilot': (0, 3), 'cached_current_best_coverage': (3, 12),
              'pose_route_and_current_ap_geometry': (12, 22), 'load_and_packet_outcomes': (22, 28),
              'cost_and_switch_history': (28, 32)}
    rows = {}
    for k, label in enumerate(('missed_commands', 'gap_seconds')):
        gc, gq = torch.autograd.grad(out['mean'][:, k].sum(), (context, query), retain_graph=k == 0)
        rows[label] = {name: float(torch.linalg.vector_norm(gc[..., start:end].flatten(1), dim=1).mean())
                       for name, (start, end) in groups.items()}
        rows[label]['action_query'] = float(torch.linalg.vector_norm(gq, dim=1).mean())
    return {'examples': len(ids), 'meaning': 'Mean local Jacobian L2 norms in native output units per normalized input unit; sensitivity is not accuracy',
            'groups': rows}


def _packet_inputs(root, read_json, recorded):
    out = root / 'artifacts/packet_value'
    protocol = read_json(out/'protocol.json')
    parts, groups = [], []
    for seed in protocol['training_sites']:
        for cost in protocol['handover_seconds']:
            stem = f'training-{seed}-{cost}'
            path = out/'data'/(stem+'.npz')
            receipt = read_json(out/'labels'/(stem+'.json'))
            identity = recorded(path)
            if receipt['split'] != 'training' or receipt['scene'] != seed or receipt['data_sha256'] != identity:
                raise ValueError('Training input identity does not match its native receipt')
            with np.load(path, allow_pickle=False) as z:
                a = {k: z[k].copy() for k in z.files}
            if set(np.unique(a['scene'])) != {seed} or len(a['context']) != len(receipt['records']):
                raise ValueError('Training scene/action alignment differs')
            parts.append(a)
            groups.extend(f"{seed}/{cost}/{r['fork_slot']}/{r['robot']}" for r in receipt['records'])
    return {k: np.concatenate([a[k] for a in parts]) for k in parts[0]}, groups, protocol


def audit(root):
    """Read only registered training/diagnostic inputs, never official holdouts."""
    root = Path(root).resolve()
    torch.set_num_threads(1)
    manifest = {}

    def recorded(path):
        manifest[str(path.relative_to(root))] = digest(path)
        return manifest[str(path.relative_to(root))]

    def read_json(path):
        recorded(path)
        return json.loads(path.read_text())

    a, groups, protocol = _packet_inputs(root, read_json, recorded)
    out = root/'artifacts/packet_value'
    training = read_json(out/'training.json')
    source = root/'src/wirelessjev/packet_value/model.py'
    if recorded(source) != training['source_sha256'][str(source.relative_to(root))]:
        raise ValueError('Packet inference source differs from the trained archive')
    packet = {}
    for family in protocol['families']:
        for seed in protocol['training_seeds']:
            path = out/'checkpoints'/f'{family}-{seed}.pt'
            if recorded(path) != training['checkpoint_sha256'][str(path.relative_to(root))]:
                raise ValueError('Trained checkpoint identity differs')
            state = torch.load(path, map_location='cpu', weights_only=True)
            model = PacketValueModel(**state['config']).eval()
            model.load_state_dict(state['state'])
            context, query = torch.from_numpy(a['context']), torch.from_numpy(a['query'])
            with torch.no_grad():
                p = model(context, query)
                raw = model.encoder.gru(context)[0][:, -1]
                pre = model.predictor[:3](torch.cat([p['context_latent'], query], -1))
                reps = {'context_gru_before_norm': raw.numpy(), 'context': p['context_latent'].numpy(),
                        'action_before_norm': pre.numpy(), 'action': p['latent'].numpy(), 'mean': p['mean'].numpy()}
                if family == 'jepa':
                    teacher = model.target_encoder(torch.as_tensor(a['future_context'], dtype=torch.float32),
                                                   torch.from_numpy(a['future_valid']))
                    reps['future_ema_target'] = teacher.numpy()
            rows = {'seed': seed, 'family': family, 'selected_epoch': state['epoch'],
                    'trained_inference_parameters': model.inference_parameters,
                    'layers': {name: representation_stats(z) for name, z in reps.items() if name != 'mean'},
                    'layers_by_cost': {str(float(c)): {name: representation_stats(reps[name][a['cost'] == c])
                        for name in ('context', 'action')} for c in np.unique(a['cost'])},
                    'value_head_in_training_target_units': representation_stats(reps['mean']/np.asarray(model.config['target_scale'])),
                    'action_sensitivity': {str(float(c)): action_sensitivity(a['query'][a['cost'] == c],
                        reps['action'][a['cost'] == c], reps['mean'][a['cost'] == c],
                        np.asarray(groups)[a['cost'] == c]) for c in np.unique(a['cost'])},
                    'input_local_gradients': _input_gradients(model, a),
                    'diagnostic_probes': _probe_predictions(a, reps)}
            if family == 'jepa':
                target = reps['future_ema_target']
                baseline = float(np.mean((target-target.mean(0))**2))
                mse = float(np.mean((reps['action']-target)**2))
                future_probe = crossfit_ridge(np.c_[reps['context'], a['query']], target, a['scene'])
                probe_mse = float(np.mean((future_probe-target)**2))
                absolute = np.abs(reps['action']-target)
                huber = float(np.where(absolute < 1., .5*absolute**2, absolute-.5).mean())
                absolute_constant = np.abs(target-target.mean(0))
                constant_huber = float(np.where(absolute_constant < 1., .5*absolute_constant**2, absolute_constant-.5).mean())
                rows['future_alignment'] = {'latent_mse': mse, 'constant_target_mean_mse': baseline,
                    'mse_over_constant': mse/baseline if baseline else None,
                    'smooth_l1': huber, 'constant_target_mean_smooth_l1': constant_huber,
                    'crossfit_context_plus_query_future_probe_mse': probe_mse,
                    'future_probe_mse_over_constant': probe_mse/baseline if baseline else None,
                    'interpretation': 'Aligned axes belong to one checkpoint and its EMA teacher; not an accuracy or unpredictability bound'}
            packet[f'{family}-{seed}'] = rows

    # Known input aggregation is audited directly rather than diagnosed from
    # a high cosine downstream. No raw CSI, neighbor AP-age or busy-time field
    # exists in this archived 32/24-feature API.
    packet_input = {'examples': len(a['context']), 'training_scenes': np.unique(a['scene']).tolist(),
        'context': representation_stats(a['context'].reshape(len(a['context']), -1)),
        'query': representation_stats(a['query']),
        'context_axis_std': a['context'].reshape(-1, 32).std(0).tolist(),
        'query_axis_std': a['query'].std(0).tolist(),
        'query_known_band_fractions': a['query'][:, 17:20].mean(0).tolist(),
        'context_rows_with_no_temporal_variation_fraction': float((np.ptp(a['context'], axis=1) == 0).all(-1).mean()),
        'future_label_valid_fraction': float(a['future_valid'].mean()),
        'target_miss_std': float(a['delta_misses'].std()), 'target_gap_std_seconds': float(a['delta_gap'].std())}

    service_out = root/'artifacts/service_world'
    sp = read_json(service_out/'protocol.json')
    windows, current, alternate = [], [], []
    for scene in sp['diagnostic_sites']:
        if any(scene in protocol.get(key, []) for key in ('validation_sites', 'development_sites', 'test_sites')):
            raise ValueError('Diagnostic scene overlaps a packet held-out registration')
        for cost in sp['handover_seconds']:
            stem = f'{scene}-{cost:g}'
            case = read_json(service_out/'cases'/(stem+'.json'))
            if not case.get('no_heldout_access') or not case.get('no_learned_policy_evaluated'):
                raise ValueError('Service input receipt does not qualify training-only diagnostics')
            path = service_out/'inputs'/(stem+'.npz')
            if recorded(path) != case['inputs_sha256']:
                raise ValueError('Native diagnostic input digest mismatch')
            with np.load(path, allow_pickle=False) as z:
                w = {k: z[k].copy() for k in ('links', 'globals', 'observed', 'elapsed')}
                keys = list(zip(z['slot'].tolist(), z['robot'].tolist()))
            windows.append(w)
            for slot, robot in keys:
                records = [r for r in case['records'] if (r['slot'], r['robot']) == (slot, robot)]
                stay = next(r for r in records if r['kind'] == 'stay')
                move = next((r for r in records if r['kind'] == 'move'), stay)
                current.append(stay['current']); alternate.append(move['target'])
    native = {k: np.concatenate([w[k] for w in windows]) for k in windows[0]}
    torch.manual_seed(0)
    model = ServiceWorldModel().eval()
    b = {k: torch.as_tensor(v) for k, v in native.items()}
    with torch.no_grad():
        enc = model.encode(**b)
        ablated = {k: v.clone() for k, v in b.items()}
        ablated['observed'][..., :5] = False
        ablated['links'][..., :5] = 0.
        without_radio = model.encode(**ablated)
        without_global = model.encode(**b, global_observed=torch.zeros_like(b['globals'], dtype=torch.bool))
        n = len(current)
        action = torch.zeros(n, 1, 12)
        prior = torch.zeros(n, 1, 2)
        stay = model.rollout(enc, action, torch.tensor(current), prior)
        move = model.rollout(enc, action, torch.tensor(alternate), prior)
    spread = float(enc['belief'].std(0, correction=0).square().mean().sqrt())

    def sensitivity(other):
        distance = float((enc['belief']-other['belief']).square().mean().sqrt())
        return {'belief_rms_change': distance, 'change_over_between_example_centered_rms': distance/spread if spread else None}

    service = {'untrained': True, 'initialization_seed': 0, 'windows': len(current),
        'scenes': sp['diagnostic_sites'], 'inference_parameters': model.inference_parameters,
        'belief': representation_stats(enc['belief'].numpy()),
        'link_tokens': representation_stats(enc['links'].reshape(-1, model.config['hidden']).numpy()),
        'current_ap_band_mean': representation_stats(enc['links'][torch.arange(n), torch.tensor(current)].mean(1).numpy()),
        'radio_feature_masks': observed_feature_stats(native['links'], native['observed'], LINK_FEATURES),
        'pilot_observed_fraction_on_configured_current_ap_band_tokens': float(
            native['observed'][..., 3][native['observed'][..., 10]].mean()),
        'global_features': observed_feature_stats(native['globals'], np.ones_like(native['globals'], dtype=bool), GLOBAL_FEATURES),
        'evidence_ablations': {'without_cached_or_pilot_observations': sensitivity(without_radio),
                              'without_global_observations': sensitivity(without_global)},
        'ap_selection_structural_probe': {'changed_ap_windows': int(np.sum(np.asarray(current) != alternate)),
            'latent_rms_change': float((stay['latent']-move['latent']).square().mean().sqrt()),
            'mean_residual_change': float((stay['residual']-move['residual']).abs().max()),
            'meaning': 'Actual candidate APs with the same unspecified zero action vector; structural selection check, no trained/semantic action adapter or control result'},
        'zero_residual_is_intentional_initialization': bool((stay['residual'] == 0).all() and (move['residual'] == 0).all())}

    for name in ('src/wirelessjev/representation_audit.py', 'src/wirelessjev/service_world/model.py',
                 'src/wirelessjev/service_world/features.py', 'src/wirelessjev/packet_value/features.py',
                 'scripts/audit_representations.py', 'uv.lock'):
        recorded(root/name)
    return {'completed': True, 'protocol_version': 1, 'packet_inputs': packet_input,
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                        'torch': str(torch.__version__), 'device': 'cpu', 'torch_threads': torch.get_num_threads()},
            'packet_models': packet, 'untrained_service_model': service,
            'probe_protocol': {'alpha': 10., 'folds': [sites.tolist() for sites in np.array_split(np.unique(a['scene']), 4)],
                'preprocessing': 'Each fold uses only probe-training-site mean/std and target centering',
                'critical_limit': 'All frozen encoders were already trained on every probe scene. Only the diagnostic readout excludes scored sites; these are NOT held-out model performance results.',
                'numeric_gap_baseline': 'Constant fold-training per-cost gap mean; numeric_delta predicts misses only'},
            'scope': {'official_validation_development_test_inputs_opened': False,
                'old_models_or_archives_modified': False, 'foundation_or_controller_models_trained': False,
                'diagnostic_linear_readouts_fit': True, 'learned_control_evaluated': False,
                'physical_unpredictability_floor_identified': False, 'sota_or_novelty_established': False},
            'file_sha256': manifest,
            'interpretation_limits': ['High raw cosine is not constant collapse.',
                'Low centered effective rank is concentration, not proof of useless information.',
                'Latent axes and per-axis variance are not identifiable across checkpoints.',
                'A failed linear probe does not rule out nonlinear information.',
                'Single realized outcomes per action cannot identify irreducible future uncertainty.',
                'Service encoder is untrained; its zero residual head is an initialization contract.']}
