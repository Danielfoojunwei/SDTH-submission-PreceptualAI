"""TRAIN-only numerical controls for the adaptive RF-condition experiment.

The native RadioMap fitter is reused unchanged, not replaced by a neural proxy.
Shared field designs/posteriors are pure functions from the sealed spectral
experiment; its guards, output directories and scientific artifacts are never
modified. All fitting in this module is protected by the new publication guard.
"""
from __future__ import annotations

from dataclasses import replace
import io
import math

import numpy as np

from ..spectral_scene.baselines import (
    BaselinePrior, MethodPrior, FieldBaseline, METHODS, KNOWN_HZ,
    PROPAGATION_COLUMNS, N_RFF, CENSOR_WEIGHT, _observations,
    _interval_posterior, _truncated_moments, _query,
)

_CACHE = None
_FROZEN = {}
SCALE_FLOOR_DB = .25
SCALE_CEILING_DB = 30.


def _common():
    from . import common
    return common


def _expected_cases(c):
    return [c.case_spec('training', i, k).case_id
            for i in range(c.COUNTS['training']) for k in c.CONDITIONS]


def _fit_interval_scale(x, lower, upper, weights, prior_variance):
    """Joint interval/censor EM with a Gaussian coefficient prior.

    Unlike variance from heard residuals, this includes the conditional second
    moment of every admitted censored observation. It is an approximate fitted
    working discrepancy, not independently calibrated predictive uncertainty.
    Quantization enters as its actual interval, without an added bin variance.
    """
    sigma = 6.
    mass = float(weights.sum())
    if not len(x) or not mass > 0:
        raise ValueError('Field estimation needs actual known TRAIN reports')
    for _ in range(12):
        coef, covariance = _interval_posterior(
            x, lower, upper, weights, sigma, prior_variance)
        mu = x @ coef
        expected, variance = _truncated_moments(mu, lower, upper, sigma)
        estimate = math.sqrt(float(np.sum(weights * (variance + (expected - mu)**2)) / mass))
        updated = float(np.clip(estimate, SCALE_FLOOR_DB, SCALE_CEILING_DB))
        if abs(updated - sigma) < 1e-5:
            sigma = updated
            break
        sigma = updated
    coef, covariance = _interval_posterior(
        x, lower, upper, weights, sigma, prior_variance)
    return coef, covariance, sigma


def learn_prior(training_rows):
    """Fit both controls on exactly the new TRAIN cases, never the held band."""
    c = _common()
    c.check_sources()
    rows = tuple(training_rows)
    expected = _expected_cases(c)
    if (not rows or any(r.stage != 'training' for r in rows)
            or [r.scene_id for r in rows] != expected):
        raise ValueError('Use all registered TRAIN case blocks exactly once in layout/condition order')
    pieces = []
    for r in rows:
        observation = _observations(r)
        q = observation[0]
        if (not np.isin(np.asarray(r.device), (0, 1, 2)).all()
                or not np.isin(np.asarray(r.band_index), (0, 1)).all()
                or not np.isin(q['frequency'], KNOWN_HZ).all()):
            raise ValueError('Held carrier/device reports cannot fit adaptive numerical priors')
        pieces.append(observation)
    q = {key: np.concatenate([part[0][key] for part in pieces]) for key in pieces[0][0]}
    known = np.concatenate([part[1] for part in pieces])
    heard = np.concatenate([part[2] for part in pieces])
    lower = np.concatenate([part[5] for part in pieces])
    upper = np.concatenate([part[6] for part in pieces])
    weight = np.concatenate([part[7] for part in pieces])
    if not heard.any():
        raise ValueError('A field location/scale requires actual heard TRAIN reports')
    rng = np.random.default_rng(2179321)
    omega = rng.normal(0, .1, (N_RFF, 2))
    phase = rng.uniform(0, 2*math.pi, N_RFF)
    models, scales = {}, {}
    for kind in METHODS:
        d = 8 + len(PROPAGATION_COLUMNS) + N_RFF if kind == 'radiomap_operator' else 8
        empty = MethodPrior(kind, np.zeros(d), np.ones(d), np.zeros(d),
                            np.eye(d), np.ones(d), 6., 6., omega, phase)
        raw = empty.raw_design(q)
        mean = np.average(raw[known], axis=0, weights=weight[known])
        deviation = np.sqrt(np.average((raw[known]-mean)**2, axis=0, weights=weight[known]))
        scale = np.where(deviation > 1e-6, deviation, 1.)
        mean[0], scale[0] = 0., 1.  # preserve the intercept
        x = (raw-mean)/scale
        variance = np.full(d, 400.)
        variance[0] = 1600.
        coef, covariance, discrepancy = _fit_interval_scale(
            x[known], lower[known], upper[known], weight[known], variance)
        local = np.full(d, discrepancy**2/max(1, d-1))
        local[0] = discrepancy**2
        # This legacy positive field stores the local inverse working scale.
        # There is NO random sensor Gaussian in the deterministic frontend.
        models[kind] = MethodPrior(kind, mean, scale, coef, covariance, local,
                                   discrepancy, discrepancy, omega, phase)
        scales[kind] = discrepancy
    metadata = {
        'training_case_ids': expected,
        'training_cases': len(rows),
        'training_layouts': c.COUNTS['training'],
        'conditions_per_layout': len(c.CONDITIONS),
        'training_report_rows': len(known),
        'known_report_rows': int(known.sum()),
        'heard_report_rows': int(heard.sum()),
        'censored_report_rows': int((known & ~heard).sum()),
        'effective_physical_link_mass': float(weight[known].sum()),
        'training_carriers_hz': list(KNOWN_HZ),
        'held_carrier_or_device_used': False,
        'validation_or_evaluation_reports_used': False,
        'TRAIN_query_measurements_used_as_permitted_prior_targets': True,
        'field_discrepancy_db': scales,
        'field_scale_estimator': '12 alternating interval/censor EM updates; Gaussian coefficient prior; clipped .25..30dB',
        'sensor_gaussian_sigma_db': 0.,
        'sensor_noise_source': 'deterministic frozen frontend: offset, quantization, threshold; no random receiver noise added',
        'quantization': 'actual intervals in global/pooled likelihood; native center approximation adds mean heard-bin width squared / 12 to local inverse variance only',
        'legacy_measurement_noise_db_slot': 'TRAIN field-discrepancy scale for context inverse, NOT sensor Gaussian sigma',
        'context_inverse_scale_formula': 'pooled: field_sigma; native: sqrt(field_sigma**2 + mean_heard_bin_width**2/12)',
        'duplicate_frontend_weight': 'inverse number of co-recorded frontends/directions per physical link within a case, including unknown attempts',
        'radiomap_core': 'wirelessjev.native.radiomap.RadioMap._fit_one (unchanged source)',
        'native_censor_weight': CENSOR_WEIGHT,
        'propagation_geometry_columns': list(PROPAGATION_COLUMNS),
        'hidden_material_condition_identity_as_features': False,
        'frontend_status_noise_quantization_and_direction_are_not_mean_field_features': True,
        'spectral_transfer': 'known per-link FSPL plus TRAIN geometry/frequency field; native log-frequency correction transport',
        'calibration_or_service_qualified': False,
    }
    return BaselinePrior(models, metadata)


class AdaptiveFieldBaseline(FieldBaseline):
    """Native per-AP/band or pooled inverse with explicit local working noise."""
    def adapt(self, context_rows):
        _, _, heard, _, _, lower, upper, _ = _observations(context_rows)
        extra = float(np.mean((upper[heard]-lower[heard])**2)/12.) if heard.any() else 0.
        sigma = self.prior.noise_db
        if self.method == 'radiomap_operator':
            sigma = math.sqrt(sigma*sigma + extra)
        self.prior = replace(self.prior, measurement_noise_db=sigma)
        return super().adapt(context_rows)


def _read_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def _data_dependencies(c):
    names = ('data/prepared-training.json', 'data/frontend-training.json', 'rt_data/training.json')
    return {name: c.digest(c.OUT/name) for name in names}


def load_prior():
    global _CACHE
    c = _common()
    c.check_sources()
    receipt = c.read_json(c.OUT/'baselines/prior.json')
    if (receipt['source_receipt_sha256'] != c.digest(c.OUT/'registration/source.json')
            or receipt['protocol_sha256'] != c.digest(c.PROTOCOL)
            or receipt['data_dependencies'] != _data_dependencies(c)):
        raise RuntimeError('Adaptive numerical prior source/data identity changed')
    reference = receipt['archive']
    path = c.safe_path(reference['path'])
    if c.digest(path) != reference['sha256']:
        raise RuntimeError('Adaptive numerical prior bytes changed')
    _CACHE = BaselinePrior.from_arrays(_read_arrays(path), receipt['metadata'])
    return _CACHE


def prepare():
    """Fit once after publication; existing receipts only load verified arrays."""
    global _CACHE
    c = _common()
    c.check_sources()
    from . import data
    target = c.OUT/'baselines/prior.json'
    with c.lock(c.OUT/'baselines/prepare.lock'):
        if target.exists():
            load_prior()
            return c.read_json(target)
        data.prepare_training()
        calibration = data.load_calibration()
        scenes = [data.load_scene('training', i, k)
                  for i in range(c.COUNTS['training']) for k in c.CONDITIONS]
        prior = learn_prior([data.training_rows(scene, calibration) for scene in scenes])
        stream = io.BytesIO()
        np.savez_compressed(stream, **prior.as_arrays())
        path = c.OUT/'baselines/prior.npz'
        c.write_bytes_once(path, stream.getvalue())
        receipt = {
            'schema': 1,
            'source_receipt_sha256': c.digest(c.OUT/'registration/source.json'),
            'protocol_sha256': c.digest(c.PROTOCOL),
            'archive': {'path': str(path.relative_to(c.ROOT)), 'sha256': c.digest(path)},
            'data_dependencies': _data_dependencies(c),
            'metadata': prior.metadata,
            'methods': list(METHODS),
            'no_evaluation_query_or_heldband_fit': True,
        }
        c.write_once(target, receipt)
        _CACHE = prior
        return receipt


def predict(name, example, *, prior=None):
    chosen = _CACHE if prior is None else prior
    if chosen is None:
        raise RuntimeError('Prepare/load adaptive numerical priors before timing prediction')
    return AdaptiveFieldBaseline(chosen, name).adapt(example.context_rows_a).predict(example.query_rows)


def verify_prior():
    """Passive receipt/array checks only; no global/local fit or forward pass."""
    c = _common()
    prior = load_prior()
    receipt = c.read_json(c.OUT/'baselines/prior.json')
    m = prior.metadata
    if (m['training_case_ids'] != _expected_cases(c)
            or m['training_cases'] != c.COUNTS['training']*len(c.CONDITIONS)
            or m['training_layouts'] != c.COUNTS['training']
            or m['training_carriers_hz'] != list(KNOWN_HZ)
            or m['held_carrier_or_device_used'] is not False
            or m['validation_or_evaluation_reports_used'] is not False
            or m['hidden_material_condition_identity_as_features'] is not False
            or m['sensor_gaussian_sigma_db'] != 0.
            or receipt['methods'] != list(METHODS)
            or receipt['no_evaluation_query_or_heldband_fit'] is not True):
        raise RuntimeError('Adaptive numerical prior fit eligibility changed')
    return {'passes': True, 'methods': list(METHODS),
            'training_layouts': m['training_layouts'], 'training_cases': m['training_cases'],
            'new_fits_or_forward_passes': False, 'heldband_or_helddevice_used': False}


def frozen_references():
    """Bind literal old weights and fit metadata without loading/running a model."""
    c = _common()
    c.check_sources()
    sealed = c.previous_manifest()['evidence_sha256']
    references = {}
    for seed in (0, 1, 2):
        root = 'artifacts/radiojepa_spectral/'
        names = (root+f'checkpoints/joint_none-seed{seed}.pt', root+f'fits/joint_none-seed{seed}.json')
        for name in names:
            if name not in sealed or c.digest(c.safe_path(name)) != sealed[name]:
                raise RuntimeError('Frozen transfer artifact differs from the completed closure')
        fit = c.read_json(c.safe_path(names[1]))
        if (fit['cell'] != 'joint_none' or fit['seed'] != seed or fit['completed'] is not True
                or fit['checkpoint_sha256'] != sealed[names[0]]):
            raise RuntimeError('Frozen transfer fit identity changed')
        references[str(seed)] = {'checkpoint': {'path': names[0], 'sha256': sealed[names[0]]},
                                 'fit': {'path': names[1], 'sha256': sealed[names[1]]}}
    return {'previous_manifest_sha256': c.PREVIOUS_MANIFEST_SHA256,
            'members': references, 'retrained_or_adapted': False,
            'spread_db': 4., 'spread_is_calibrated': False}


def frozen_eligibility(example):
    """Only supplementary transfer can abstain; primary cohort never changes."""
    values = np.asarray(example.context_a)
    mask = np.asarray(example.mask_a)
    if values.ndim != 2 or values.shape[1] != 20 or mask.shape != (len(values),) or mask.dtype != bool:
        raise ValueError('Malformed frozen transfer context schema')
    _query(example.query_rows)
    if not np.isfinite(values[mask]).all():
        raise ValueError('Nonfinite actual frozen transfer context')
    if not mask.any():
        return {'eligible': False, 'reason': 'no_known_A_context_RF'}
    return {'eligible': True, 'reason': None}


def _load_frozen(seed):
    if seed not in (0, 1, 2):
        raise ValueError('Only the three sealed joint_none transfer seeds are available')
    c = _common()
    references = frozen_references()['members'][str(seed)]
    key = (seed, references['checkpoint']['sha256'])
    if key not in _FROZEN:
        import torch
        from ..spectral_scene.model import SpectralScene
        path = c.safe_path(references['checkpoint']['path'])
        saved = torch.load(path, map_location='cpu', weights_only=True)
        fit = c.read_json(c.safe_path(references['fit']['path']))
        if (saved['cell'] != 'joint_none' or saved['seed'] != seed
                or saved['identity'] != fit['identity'] or saved['settings'] != fit['settings']
                or saved['selected_epoch'] != fit['selected_epoch']):
            raise RuntimeError('Frozen checkpoint metadata does not match its sealed fit')
        model = SpectralScene(**saved['settings'])
        model.load_state_dict(saved['state'], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _FROZEN[key] = model
    return _FROZEN[key]


def predict_frozen(seed, example):
    """Literal old model, known-TX-normalized inputs, no update or adaptation."""
    import torch
    eligible = frozen_eligibility(example)
    if not eligible['eligible']:
        n = len(np.asarray(example.query_rows.carrier_hz))
        return np.full(n, np.nan), np.full(n, np.nan)
    model = _load_frozen(seed)
    def tensor(value):
        a = np.asarray(value).copy()
        return torch.as_tensor(a, dtype=torch.bool if a.dtype == bool else torch.float32)[None]
    with torch.no_grad():
        mean, _ = model(tensor(example.context_a), tensor(example.mask_a), tensor(example.query),
                        tensor(example.query_basis), tensor(example.query_fspl_dbm),
                        tensor(example.query_offset_db))
    result = mean[0].numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite frozen transfer prediction')
    return result, np.full_like(result, 4.)
