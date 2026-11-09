"""Post-hoc representation diagnostics on the entire EXISTING validation cache.

Run only after the selected-JEPA confirmation and its replays have completed:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      uv run --frozen python scripts/audit_semantic_representations.py

No dataset builder, simulator, optimizer, selection or checkpoint writer is
called. Missing caches fail closed. Streaming moments cover every valid token;
there is no subsampling. The two derived outputs are outside frozen studies'
scientific source/input closures. Numeric variation is not task usefulness or
an anti-collapse guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
BATCH_ROWS = 64
NUMERIC_TOLERANCE = 1e-10
FAMILIES = ('context_jepa', 'supervised')
SEEDS = (0, 1, 2)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def name(path):
    return os.path.relpath(Path(path).resolve(), ROOT)


class Moments:
    """Stable block-Welford population covariance; at most D x D storage."""
    def __init__(self, width):
        self.width = width
        self.tokens = self.finite_tokens = self.nonfinite_values = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros((width, width), dtype=np.float64)

    def add(self, values):
        x = np.asarray(values, dtype=np.float64).reshape(-1, self.width)
        self.tokens += len(x)
        self.nonfinite_values += int((~np.isfinite(x)).sum())
        x = x[np.isfinite(x).all(1)]
        count = len(x)
        if not count:
            return
        mean = x.mean(0)
        centered = x - mean
        m2 = centered.T @ centered
        total = self.finite_tokens + count
        delta = mean - self.mean
        self.m2 += m2 + np.outer(delta, delta) * (self.finite_tokens * count / total)
        self.mean += delta * (count / total)
        self.finite_tokens = total

    def result(self):
        base = {'tokens': self.tokens, 'finite_tokens': self.finite_tokens,
                'nonfinite_values': self.nonfinite_values,
                'all_values_finite': self.nonfinite_values == 0,
                'width': self.width, 'moments_use_only_finite_tokens': True}
        if not self.finite_tokens:
            return {**base, 'centered_rms': None, 'population_std_mean': None,
                    'centered_entropy_effective_rank': None,
                    'nonconstant_at_numeric_tolerance': None}
        covariance = self.m2 / self.finite_tokens
        covariance = (covariance + covariance.T) * .5
        variance = np.maximum(np.diag(covariance), 0.)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.)[::-1]
        energy = float(eigenvalues.sum())
        positive = eigenvalues[eigenvalues > 0]
        probability = positive / energy if energy > 0 else positive
        rank = float(np.exp(-(probability * np.log(probability)).sum())) if energy > 0 else 0.
        rms = float(np.sqrt(variance.mean()))
        std = np.sqrt(variance)
        return {**base, 'centered_rms': rms, 'population_std_mean': float(std.mean()),
                'population_std_min': float(std.min()), 'population_std_max': float(std.max()),
                'population_std_by_coordinate': std.tolist(),
                'centered_covariance_eigenvalues_descending': eigenvalues.tolist(),
                'centered_entropy_effective_rank': rank,
                'nonconstant_coordinates_at_numeric_tolerance': int((std > NUMERIC_TOLERANCE).sum()),
                'nonconstant_at_numeric_tolerance': rms > NUMERIC_TOLERANCE}


def _tokens(value, valid, indices, band):
    """Extract actual site/band tokens; H tokens stay explicitly correlated."""
    if value.ndim == 3:
        group = value[indices][:, band::3]
        keep = valid[indices][:, band::3]
    elif value.ndim == 4:
        group = value[indices][:, :, band::3]
        keep = valid[indices][:, None, band::3].expand(-1, value.shape[1], -1)
    else:
        raise ValueError('Expected B x link x D or B x H x link x D')
    return group[keep].detach().cpu().numpy()


def accumulate(storage, components, valid, site_ids, accumulator_type):
    groups = [('pooled_all_sites', np.arange(len(site_ids)))]
    groups += [(str(int(site)), np.flatnonzero(site_ids == site)) for site in np.unique(site_ids)]
    for site, indices in groups:
        for band in range(3):
            destination = storage.setdefault(site, {}).setdefault(str(band), {})
            for component, value in components.items():
                accumulator = destination.setdefault(component, accumulator_type(value.shape[-1]))
                accumulator.add(_tokens(value, valid, indices, band))


def finish(storage):
    return {site: {band: {component: accumulator.result()
                         for component, accumulator in components.items()}
                   for band, components in bands.items()}
            for site, bands in storage.items()}


def existing_validation(training, common):
    """Read/check cache directly; never call dataset(), which can generate data."""
    path = common.WORK / 'validation.npz'
    receipt_path = common.OUT / 'data/validation.json'
    if not path.is_file() or not receipt_path.is_file():
        raise RuntimeError('Existing validation cache AND receipt are required; this audit never rebuilds data')
    identity = training._identity('validation')
    with np.load(path, allow_pickle=False) as archive:
        if training.NPZ_IDENTITY not in archive:
            raise RuntimeError('Missing embedded validation identity')
        if json.loads(archive[training.NPZ_IDENTITY].tobytes().decode()) != identity:
            raise RuntimeError('Validation cache identity differs from frozen study')
        arrays = {key: archive[key] for key in archive.files if key != training.NPZ_IDENTITY}
    training._validate_arrays(arrays, identity)
    expected = training._dataset_receipt(path, identity, arrays)
    if expected != json.loads(receipt_path.read_text()):
        raise RuntimeError('Existing validation receipt/cache/collection hashes differ')
    if len(arrays['context']) != 1152 or len(np.unique(arrays['site_id'])) != 6:
        raise RuntimeError('This diagnostic is registered to all 1152 existing rows and six validation sites')
    return arrays, path, receipt_path, expected


def completion_gate():
    """Prevent accidental heavy audit execution while confirmation still runs."""
    path = ROOT / 'artifacts/selected_jepa/completion.json'
    report_path = ROOT / 'artifacts/selected_jepa/report.json'
    if not path.is_file() or not report_path.is_file():
        raise RuntimeError('Wait for the confirmation pipeline: completed confirmation and report are required')
    value = json.loads(path.read_text())
    if value.get('completed') is not True or value.get('case_count') != 128 or len(value.get('replays', [])) != 8:
        raise RuntimeError('Wait for all 128 confirmation cases and eight replay records')
    report = json.loads(report_path.read_text())
    if report.get('completion_sha256') != digest(path) or report.get('case_count') != 128:
        raise RuntimeError('Final confirmation report does not bind the completed case matrix')
    return path, value, report_path


@torch.inference_mode()
def audit_model(model, arrays, training):
    health = {}
    for start in range(0, len(arrays['context']), BATCH_ROWS):
        index = slice(start, start + BATCH_ROWS)
        batch = training.batch(arrays, index)
        public = {key: batch[key] for key in training.FIELDS}
        public['valid_link'] = batch['valid_link']
        site_ids = arrays['site_id'][index]
        p = model(public)  # No labels cross the inference boundary.
        valid = p['valid_link']
        online = {'context_input_48': public['context'], 'radio_input_2': p['radio_semantic'],
                  'context_latent': p['context'], 'radio_latent': p['radio_latent'],
                  'link_latent': p['link_latent'], 'predictor_latent': p['latent'],
                  'forecast_mean_db': p['mean'][..., None],
                  'forecast_route_delta_db': (p['mean'] - p['mean'][:, :1])[..., None]}
        if model.config['family'] == 'context_jepa':
            online['training_only_projected_predictor'] = model.projector(p['latent'])
        all_components = dict(online)
        if model.config['family'] == 'context_jepa':
            future = batch['future_context']
            target = model.target_context_encoder(future)
            all_components['training_future_context_input_48'] = future
            all_components['training_active_context_teacher'] = target
        accumulate(health, all_components, valid, site_ids, Moments)
    return {'health_by_actual_site_and_band': finish(health),
            'active_training_teacher': 'EMA context encoder' if model.config['family'] == 'context_jepa' else None,
            'allocated_but_unused_teachers_are_not_reported_as_active': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    completion_path, completion, confirmation_report_path = completion_gate()
    from wirelessjev.semantic_jepa import common, training
    from wirelessjev.selected_jepa import common as selected
    from wirelessjev.selected_jepa.study import replay_keys

    semantic_source = common.check_sources()
    confirmation_source = selected.check_sources()
    audit_script_name = name(Path(__file__))
    if (audit_script_name in semantic_source['source_sha256']
            or audit_script_name in confirmation_source['source_sha256']):
        raise RuntimeError('This derived audit must remain outside frozen scientific source closures')
    if (completion['source_sha256'] != selected.source_hashes()
            or completion['selection_sha256'] != digest(selected.OUT / 'selection.json')
            or {tuple(item['case']) for item in completion['replays']} != set(replay_keys())):
        raise RuntimeError('Completed confirmation identity or replay membership differs')
    if selected.selection()['selected_family'] != 'context_jepa':
        raise RuntimeError('This descriptive audit targets the frozen context-JEPA choice')
    torch.set_num_threads(1)
    arrays, cache_path, receipt_path, data_receipt = existing_validation(training, common)
    paths = [Path(__file__), cache_path, receipt_path, completion_path, confirmation_report_path,
             common.OUT / 'registration/source.json', selected.OUT / 'registration/source.json',
             selected.OUT / 'selection.json', common.OUT / 'training.json']
    paths += [common.OUT / 'collection' / (key + '.json')
              for key in data_receipt['collection_receipt_sha256']]
    for family in FAMILIES:
        for seed in SEEDS:
            paths += [common.OUT / 'checkpoints' / f'{family}-{seed}.pt',
                      common.OUT / 'fits' / f'{family}-{seed}.json']
    inputs = {name(path): digest(path) for path in paths}
    outputs = {}
    for family in FAMILIES:
        for seed in SEEDS:
            model = training.load(family, seed)
            result = audit_model(model, arrays, training)
            result.update(config=dict(model.config), checkpoint_sha256=inputs[
                name(common.OUT / 'checkpoints' / f'{family}-{seed}.pt')],
                inference_parameters=model.inference_parameters)
            outputs[f'{family}-{seed}'] = result
            print(f'Audited {family} seed {seed}: all {len(arrays["context"])} existing validation rows', flush=True)
            del model
    common.check_sources()
    selected.check_sources()
    if any(digest(ROOT / path) != expected for path, expected in inputs.items()):
        raise RuntimeError('An audit input changed during derived analysis')
    report = {
        'schema': 1, 'posthoc_descriptive_only': True,
        'data_role': 'Previously exposed validation used for checkpoint selection; not independent evaluation.',
        'all_existing_validation_rows': True, 'rows': len(arrays['context']), 'sites': {},
        'batch_rows': BATCH_ROWS, 'token_subsampling': False, 'streaming_population_covariance': True,
        'input_precision': 'Existing float16 cache is cast through frozen training.batch to float32; moments use float64.',
        'numeric_variation_tolerance': NUMERIC_TOLERANCE, 'rank_target_imposed': False,
        'statistical_independence_of_tokens_claimed': False,
        'sample_units': 'Correlated AP-band tokens; predictor/teacher streams additionally pool nine horizons.',
        'band_index_to_name': {'0': '2g', '1': '5g', '2': '6g'},
        'input_dimensions': {
            'context': '48 = six scans x eight existing value/mask/geometry/band features',
            'radio_semantic': '2 = last-heard/current-map innovation and fixed absolute 5 dB SNR margin',
            'interpretation': 'Nonlinear radio embeddings can have covariance rank above two despite two scalar inputs.'},
        'rf_or_feature_perturbations_performed': False,
        'perturbation_omission_reason': 'Changing only RSSI would not guarantee consistency with censor bounds, cached maps and fleet features; this audit uses existing observations unchanged.',
        'models': outputs, 'completion_replays_all_exact': completion['all_exact'],
        'scientific_fits_simulations_or_new_data_created': 0,
        'selection_or_qualification_changed': False, 'anti_collapse_guarantee': False,
        'limits': [
            'Finite/nonconstant/high-rank representations need not encode useful wireless decisions.',
            'Pooled geometry, AP identities, horizons and masks can explain variance; rank is not RF knowledge.',
            'The radio path has two scalar semantic inputs; it is not a 48-dimensional scan encoder.',
            'Only context-JEPA has an active future-context teacher; supervised teacher allocations are unused.',
            'No perturbation or RF sensitivity evidence is generated; existing validation inputs remain unchanged.',
            'No new heldouts, control improvement, causal mechanism, safety, SOTA or production claim follows.']}
    for site in np.unique(arrays['site_id']):
        selected_rows = arrays['site_id'] == site
        report['sites'][str(int(site))] = {'rows': int(selected_rows.sum()),
            'existing_decision_times': sorted(int(t) for t in np.unique(arrays['decision_time'][selected_rows]))}
    output = common.OUT / 'diagnostics/full_validation_representations.json'
    common.write_once(output, report)
    common.write_once(output.with_name('full_validation_representations.receipt.json'), {
        'schema': 1, 'derived_posthoc_only': True, 'script_sha256': digest(Path(__file__)),
        'script_outside_frozen_scientific_source_closures': True,
        'input_sha256': inputs, 'software_versions': common.versions(),
        'output_sha256': {name(output): digest(output)},
        'rows': len(arrays['context']), 'models': len(outputs), 'new_fits': 0,
        'new_worlds_or_measurements': 0, 'sampling': 'All existing rows and valid tokens; no subsampling.'})
    print(name(output), flush=True)


if __name__ == '__main__':
    main()
