"""Derived diagnostics on already exposed, immutable validation data only.

No world construction, fitting, controller execution, new selection or scientific
source/evidence modification. Old Fleet JEPA is evaluated on the NEW common
cached feature pipeline: this is a weight-transfer diagnostic, not its genuine
old runtime or its independently measured service performance.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from audit_semantic_representations import existing_validation, digest, name


def average(values, mask):
    mask = np.broadcast_to(mask, values.shape)
    return float(values[mask].mean()) if mask.any() else None


def rmse(values, mask):
    value = average(np.square(values), mask)
    return float(np.sqrt(value)) if value is not None else None


def forecast_metrics(mean, std, truth, base, valid, recent):
    error = mean - truth
    delta_error = (mean - mean[:, :1]) - (truth - truth[:, :1])
    future = valid[:, None]
    error0 = error[:, :1]
    common_error = error.reshape(len(mean), mean.shape[1], -1, 3).mean(2, keepdims=True)
    centered = error.reshape(len(mean), mean.shape[1], -1, 3) - common_error
    correction = mean - base
    true_correction = truth - base
    horizon_mask = np.array([1., 1., 1., 1., 1., 1., .5, .3, .2])[None, :, None]
    weights = np.where(truth > 0, 1., .2) * horizon_mask * future
    mse = float((np.square(error) * weights).sum() / weights.sum())
    values = {
        'absolute_rmse_db': rmse(error, future),
        'training_loss_weighted_rmse_db': float(np.sqrt(mse)),
        'h0_current_error_rmse_db': rmse(error0, future),
        'route_delta_rmse_h1_to_h5_db': rmse(delta_error[:, 1:6], future),
        'route_delta_rmse_all_higher_horizons_db': rmse(delta_error[:, 1:], future),
        'ap_common_error_component_rmse_db': float(np.sqrt(np.mean(common_error ** 2))),
        'ap_centered_per_band_error_rmse_db': float(np.sqrt(np.mean(centered ** 2))),
        'mean_abs_correction_db': average(np.abs(correction), future),
        'max_abs_correction_db': float(np.abs(correction[future.repeat(mean.shape[1], 1)]).max()),
        'fraction_abs_correction_ge_11p9db': average((np.abs(correction) >= 11.9).astype(float), future),
        'pointwise_oracle_12db_cap_rmse_floor_db': rmse(np.maximum(np.abs(true_correction) - 12., 0.), future),
        'pointwise_oracle_12db_cap_exceeded_fraction': average((np.abs(true_correction) > 12.).astype(float), future),
        'recent_heard_link_fraction': float(recent.mean()),
        'recent_heard_absolute_rmse_db': rmse(error, future & recent[:, None]),
        'no_recent_heard_absolute_rmse_db': rmse(error, future & ~recent[:, None]),
        'recent_heard_route_delta_rmse_h1_to_h5_db': rmse(delta_error[:, 1:6], future & recent[:, None]),
        'no_recent_heard_route_delta_rmse_h1_to_h5_db': rmse(delta_error[:, 1:6], future & ~recent[:, None]),
    }
    # Exact error identity, not a counterfactual policy or an oracle prediction.
    d = delta_error[:, 1:]
    static = float(np.mean(np.broadcast_to(error0, d.shape) ** 2))
    route = float(np.mean(d ** 2))
    cross = float(2 * np.mean(error0 * d))
    future_mse = float(np.mean(error[:, 1:] ** 2))
    values['future_mse_identity_db2'] = {'current_error_mse': static,
        'increment_error_mse': route, 'twice_cross_term': cross,
        'future_error_mse': future_mse,
        'identity_residual': future_mse - static - route - cross}
    if std is not None:
        values['mean_field_marginal_90pct_coverage'] = average((np.abs(error) <= 1.645 * std).astype(float), future)
        values['spread_is_not_packet_risk'] = True
    return values


def ap_metrics(mean, truth):
    n, h, links = mean.shape
    predicted = mean[:, :6].reshape(n, 6, links // 3, 3).max(-1).mean(1)
    actual = truth[:, :6].reshape(n, 6, links // 3, 3).max(-1).mean(1)
    error = predicted - actual
    centered = error - error.mean(-1, keepdims=True)
    regret = actual.max(-1) - np.take_along_axis(actual, predicted.argmax(-1)[:, None], -1)[:, 0]
    sites = links // 3
    return {'score_definition': 'Mean first-six-horizon maximum-band SNR, before load/roam/admission costs.',
        'ap_score_rmse_db': float(np.sqrt(np.mean(error ** 2))),
        'centered_ap_score_rmse_db': float(np.sqrt(np.mean(centered ** 2))),
        'all_ordered_offdiagonal_ap_pair_gain_rmse_db': float(np.sqrt(2 * sites / (sites - 1) * np.mean(centered ** 2))),
        'zero_load_ap_choice_regret_db': float(regret.mean()),
        'zero_load_ap_choice_disagreement_fraction': float((predicted.argmax(-1) != actual.argmax(-1)).mean()),
        'actual_current_ap_and_load_not_in_cache': True,
        'not_actual_switch_gain_or_packet_counterfactual': True}


def metrics(mean, std, arrays):
    truth = arrays['truth'].astype('f8')
    base = arrays['map_mean'].astype('f8')
    mean = mean.astype('f8')
    std = std.astype('f8') if std is not None else None
    valid = arrays['valid_link'].astype(bool)
    if not valid.all():
        raise RuntimeError('This AP-contrast diagnostic requires the existing complete physical AP/band grid')
    context = arrays['context'].reshape(len(mean), mean.shape[-1], 6, 8)
    recent = ((context[..., 2] > .5) & (context[..., 3] > .5)).any(-1)
    result = {'overall': forecast_metrics(mean, std, truth, base, valid, recent),
              'ap_scores': ap_metrics(mean, truth), 'by_site': {}, 'by_band': {}, 'by_horizon': []}
    for site in np.unique(arrays['site_id']):
        rows = arrays['site_id'] == site
        result['by_site'][str(int(site))] = {
            **forecast_metrics(mean[rows], std[rows] if std is not None else None,
                               truth[rows], base[rows], valid[rows], recent[rows]),
            'ap_scores': ap_metrics(mean[rows], truth[rows])}
    for band in range(3):
        result['by_band'][str(band)] = {
            'absolute_rmse_db': rmse(mean[:, :, band::3] - truth[:, :, band::3], valid[:, None, band::3]),
            'current_error_rmse_db': rmse((mean[:, :1] - truth[:, :1])[:, :, band::3], valid[:, None, band::3]),
            'route_delta_rmse_h1_to_h5_db': rmse(((mean[:, 1:6] - mean[:, :1]) -
                (truth[:, 1:6] - truth[:, :1]))[:, :, band::3], valid[:, None, band::3])}
    from wirelessjev.native.world import HORIZONS
    for h, second in enumerate(HORIZONS):
        result['by_horizon'].append({'seconds': second,
            'absolute_rmse_db': rmse(mean[:, h] - truth[:, h], valid),
            'route_delta_rmse_db': rmse((mean[:, h] - mean[:, 0]) - (truth[:, h] - truth[:, 0]), valid)})
    return result


@torch.inference_mode()
def predictions(model, arrays, training):
    means, stds = [], []
    for start in range(0, len(arrays['context']), 64):
        batch = training.batch(arrays, slice(start, start + 64))
        public = {key: batch[key] for key in training.FIELDS}
        public['valid_link'] = batch['valid_link']
        result = model(public)
        if not torch.isfinite(result['mean']).all() or not torch.isfinite(result['std']).all():
            raise RuntimeError('Nonfinite frozen prediction')
        means.append(result['mean'].cpu().numpy().copy())
        stds.append(result['std'].cpu().numpy().copy())
    return np.concatenate(means), np.concatenate(stds)


def main():
    from wirelessjev.semantic_jepa import common, training
    from wirelessjev.selected_jepa import common as selected
    from wirelessjev.fleetwm.study import load as load_old
    common.check_sources(); selected.check_sources()
    torch.set_num_threads(1)
    arrays, cache, data_receipt, data = existing_validation(training, common)
    paths = [Path(__file__), ROOT / 'scripts/audit_semantic_representations.py', cache, data_receipt,
             common.OUT / 'registration/source.json', selected.OUT / 'registration/source.json']
    paths += [common.OUT / 'checkpoints' / f'{family}-{seed}.pt'
              for family in ('context_jepa', 'supervised') for seed in (0, 1, 2)]
    paths += [common.OUT / 'fits' / f'{family}-{seed}.json'
              for family in ('context_jepa', 'supervised') for seed in (0, 1, 2)]
    paths += [common.OUT / 'collection' / (key + '.json') for key in data['collection_receipt_sha256']]
    paths += [ROOT / f'artifacts/fleet_world_model/checkpoints/hybrid_jepa-{seed}.pt' for seed in (0, 1, 2)]
    inputs = {name(path): digest(path) for path in paths}
    output = {'schema': 1, 'posthoc_descriptive_only': True, 'validation_already_exposed': True,
        'rows': len(arrays['context']), 'sites': [int(s) for s in np.unique(arrays['site_id'])],
        'new_fits_or_worlds': 0, 'models': {}, 'single_member_metric_means': {},
        'old_fleet_scope': 'Original three hybrid-JEPA weights on the NEW common cached strong-map/features; NOT the original runtime/prior/observation pipeline.',
        'limits': ['Validation selected checkpoints and is not independent evidence.',
                   'No current association, queue or consumed-forecast age is stored in this cache.',
                   'AP score/gain proxies omit load, switching costs and admission; they are not packet values.',
                   'Cap floors are pointwise hindsight representational lower bounds, not realizable controls.',
                   'All derived error coordinates use actual frozen predictions and existing labels, with no RF input perturbation.']}
    output['models']['strong_map_cached'] = metrics(arrays['map_mean'], arrays['map_std'], arrays)
    for family in ('context_jepa', 'supervised'):
        scalar = []
        for seed in (0, 1, 2):
            model = training.load(family, seed)
            mean, std = predictions(model, arrays, training)
            value = metrics(mean, std, arrays)
            output['models'][f'{family}-{seed}'] = value
            scalar.append({**value['overall'], **value['ap_scores']})
            print(f'Diagnosed {family}-{seed} on {len(mean)} existing rows', flush=True)
            del model
        output['single_member_metric_means'][family] = {key: float(np.mean([row[key] for row in scalar]))
            for key in scalar[0] if type(scalar[0][key]) is float}
    old_predictions = [predictions(model, arrays, training) for model in load_old('hybrid_jepa')]
    old_means = np.stack([value[0] for value in old_predictions])
    old_std = np.sqrt(np.mean(np.stack([value[1] ** 2 for value in old_predictions]), axis=0)
                      + np.var(old_means, axis=0, ddof=1))
    output['models']['old_fleet_jepa_common_input_ensemble'] = metrics(old_means.mean(0), old_std, arrays)
    common.check_sources(); selected.check_sources()
    if any(digest(ROOT / path) != expected for path, expected in inputs.items()):
        raise RuntimeError('A frozen input changed during diagnostic forwards')
    destination = ROOT / 'artifacts/jepa_repair_diagnosis/architecture.json'
    common.write_once(destination, output)
    common.write_once(destination.with_name('architecture.receipt.json'), {
        'posthoc_descriptive_only': True, 'script_sha256': digest(Path(__file__)),
        'input_sha256': inputs, 'output_sha256': {name(destination): digest(destination)},
        'software_versions': common.versions(), 'new_fits': 0, 'new_worlds': 0})
    print(name(destination), flush=True)


if __name__ == '__main__':
    main()
