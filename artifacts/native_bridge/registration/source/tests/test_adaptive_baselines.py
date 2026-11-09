"""Analytic software fixtures only, never adaptive scientific campaign results."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wirelessjev.adaptive_radio import baselines as b
from wirelessjev.spectral_scene.data import LinkRows
from wirelessjev.spectral_scene.operators import fspl_received_power_dbm, link_geometry_features


def rows(case='fixture-training-0-0', *, stage='training', poses=tuple(range(6)),
         bands=(0, 1), devices=(0,), shift=0., missing=False):
    index = np.array([(p, a, f, d) for p in poses for a in range(2) for f in bands for d in devices])
    p, a, f, d = index.T
    rx = np.column_stack((2.+p*1.4+shift, 2.+p*.3, np.ones(len(p))))
    ap = np.column_stack((1.+a*10, np.ones(len(p)), np.full(len(p), 2.)))
    hz = np.array((2.4e9, 5.2e9, 6e9))[f]
    offset = np.array((0., -3., 4., 0.))[d]
    g = link_geometry_features(rx, ap, hz, np.zeros(len(p)), (24, 20, 5), np.empty((0, 2, 3)),
                              calibration_pair_count=16, calibration_unknown=False,
                              quantization_width_db=.5)
    fspl = fspl_received_power_dbm(rx, ap, hz, tx_power_dbm=-10.)
    report = np.round((fspl+2.5+.4*g[:, 6]-3.*g[:, 7]+offset)*2)/2
    heard = np.full(len(p), not missing, bool)
    return LinkRows(case, stage, rx, ap, hz, np.zeros(len(p), int), d, f, p, a, report,
                    report-.25, report+.25, heard, np.zeros(len(p), bool),
                    np.full(len(p), -95.), fspl, offset, g,
                    tuple(f'{case}-{i}' for i in range(len(p))))


def fixture_common():
    return SimpleNamespace(check_sources=lambda: None, COUNTS={'training': 1}, CONDITIONS=(0, 1, 2),
        case_spec=lambda stage, i, k: SimpleNamespace(case_id=f'fixture-training-{i}-{k}'))


@pytest.fixture
def prior(monkeypatch):
    monkeypatch.setattr(b, '_common', fixture_common)
    return b.learn_prior([rows(f'fixture-training-0-{k}', devices=(0, 1, 2), shift=k*.3)
                          for k in range(3)])


def query_only(r):
    class Query:
        def __getattr__(self, name):
            if name in {'geometry_features', 'rx_xyz', 'ap_xyz', 'carrier_hz', 'fspl_dbm', 'offset_db'}:
                return getattr(r, name)
            raise AssertionError('Query RF/private labels must not be read: '+name)
    return Query()


def test_source_guard_precedes_fit_and_prepare(monkeypatch):
    def blocked():
        raise RuntimeError('not published')
    monkeypatch.setattr(b, '_common', lambda: SimpleNamespace(check_sources=blocked))
    with pytest.raises(RuntimeError, match='not published'):
        b.learn_prior([])
    with pytest.raises(RuntimeError, match='not published'):
        b.prepare()


@pytest.mark.parametrize('violation', ('missing', 'validation', 'heldband', 'helddevice', 'order'))
def test_exact_new_training_admission(monkeypatch, violation):
    monkeypatch.setattr(b, '_common', fixture_common)
    blocks = [rows(f'fixture-training-0-{k}') for k in range(3)]
    if violation == 'missing':
        blocks.pop()
    elif violation == 'validation':
        blocks[0] = replace(blocks[0], stage='validation')
    elif violation == 'heldband':
        blocks[0] = rows(bands=(2,))
    elif violation == 'helddevice':
        blocks[0] = rows(devices=(3,))
    else:
        blocks.reverse()
    with pytest.raises(ValueError):
        b.learn_prior(blocks)


def test_censored_scale_uses_unheard_second_moments():
    rng = np.random.default_rng(421)
    true = rng.normal(-4., 3., 3000)
    report = np.round(true*2)/2
    heard = report >= -3.
    lower = np.where(heard, report-.25, -np.inf)
    upper = np.where(heard, report+.25, -3.25)
    coef, covariance, sigma = b._fit_interval_scale(np.ones((len(true), 1)), lower, upper,
                                                  np.ones(len(true)), np.array((1600.,)))
    assert coef[0] == pytest.approx(-4., abs=.35)
    assert sigma == pytest.approx(3., abs=.25)
    assert sigma > np.std(report[heard])+.8
    assert covariance[0, 0] > 0


def test_roundtrip_and_sensor_discrepancy_separation(prior):
    loaded = b.BaselinePrior.from_arrays(prior.as_arrays(), prior.metadata)
    assert loaded.metadata['sensor_gaussian_sigma_db'] == 0.
    assert loaded.metadata['training_layouts'] == 1
    assert loaded.metadata['training_cases'] == 3
    assert loaded.metadata['native_censor_weight'] == 5.
    assert loaded.metadata['hidden_material_condition_identity_as_features'] is False
    for name in b.METHODS:
        model = loaded.methods[name]
        assert model.measurement_noise_db == model.noise_db
        assert model.noise_db >= b.SCALE_FLOOR_DB
        np.testing.assert_array_equal(model.coef, prior.methods[name].coef)
        with pytest.raises(ValueError):
            model.coef[0] = 1.
    # Three co-recorded frontends are one physical-link likelihood mass.
    assert loaded.metadata['effective_physical_link_mass'] == pytest.approx(
        loaded.metadata['training_report_rows']/3)


def test_actual_native_fitter_and_quantization_scale(prior, monkeypatch):
    from wirelessjev.native.radiomap import RadioMap
    original, calls = RadioMap._fit_one, []
    def spy(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(RadioMap, '_fit_one', spy)
    context = rows(poses=(0, 2, 4))
    baseline = b.AdaptiveFieldBaseline(prior).adapt(context)
    assert len(calls) == 4
    expected = np.sqrt(prior.methods['radiomap_operator'].noise_db**2+.5**2/12.)
    assert baseline.prior.measurement_noise_db == pytest.approx(expected)
    mean, std = baseline.predict(query_only(rows(poses=(1, 3, 5), bands=(2,))))
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()


@pytest.mark.parametrize('method', b.METHODS)
def test_censor_bounds_missing_reports_and_query_whitelist(prior, method):
    context = rows(poses=(0, 2, 4))
    h, c = context.heard.copy(), context.censored.copy()
    h[:2] = False
    c[0] = True
    bound = context.censor_upper_dbm.copy()
    bound[0] = context.reported_rssi_dbm[0]-4
    context = replace(context, heard=h, censored=c, censor_upper_dbm=bound)
    poison = context.reported_rssi_dbm.copy()
    poison[:2] = (99999, -99999)
    changed = replace(context, reported_rssi_dbm=poison)
    query = query_only(rows(poses=(1, 3, 5), bands=(2,)))
    first = b.AdaptiveFieldBaseline(prior, method).adapt(context).predict(query)
    second = b.AdaptiveFieldBaseline(prior, method).adapt(changed).predict(query)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize('method', b.METHODS)
def test_direction_frontend_metadata_not_mean_features(prior, method):
    target = rows(poses=(1, 3, 5), bands=(2,))
    model = b.AdaptiveFieldBaseline(prior, method).adapt(rows(poses=(0, 2, 4)))
    mean, _ = model.predict(query_only(target))
    g = target.geometry_features.copy()
    g[:, 8] = -g[:, 8]
    g[:, 11:15] = (0., 1., .9, -.3)
    mean2, _ = model.predict(query_only(replace(target, geometry_features=g)))
    np.testing.assert_array_equal(mean, mean2)


def test_no_context_frozen_abstention_and_schema_errors():
    target = rows(bands=(2,))
    example = SimpleNamespace(context_a=np.zeros((4, 20)), mask_a=np.zeros(4, bool),
                              query_rows=query_only(target))
    assert b.frozen_eligibility(example) == {'eligible': False, 'reason': 'no_known_A_context_RF'}
    mean, std = b.predict_frozen(0, example)
    assert len(mean) == len(target.heard) and np.isnan(mean).all() and np.isnan(std).all()
    example.mask_a = np.zeros(3, bool)
    with pytest.raises(ValueError, match='schema'):
        b.frozen_eligibility(example)


def test_passive_verification_has_no_fit_or_forward(prior, monkeypatch, tmp_path):
    c = fixture_common()
    c.ROOT = tmp_path
    c.OUT = tmp_path/'artifacts/adaptive_radio'
    c.PROTOCOL = c.OUT/'protocol.json'
    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    c.digest = digest
    c.safe_path = lambda p: tmp_path/p
    c.read_json = lambda p: json.loads(Path(p).read_text())
    for name in ('protocol.json', 'registration/source.json', 'data/prepared-training.json',
                 'data/frontend-training.json', 'rt_data/training.json'):
        p = c.OUT/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{}')
    path = c.OUT/'baselines/prior.npz'
    path.parent.mkdir(parents=True)
    stream = io.BytesIO()
    np.savez_compressed(stream, **prior.as_arrays())
    path.write_bytes(stream.getvalue())
    receipt = {'source_receipt_sha256': digest(c.OUT/'registration/source.json'),
               'protocol_sha256': digest(c.PROTOCOL), 'data_dependencies': b._data_dependencies(c),
               'archive': {'path': str(path.relative_to(tmp_path)), 'sha256': digest(path)},
               'metadata': prior.metadata, 'methods': list(b.METHODS),
               'no_evaluation_query_or_heldband_fit': True}
    (path.parent/'prior.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(b, '_common', lambda: c)
    def forbidden(*args, **kwargs):
        raise AssertionError('passive verification cannot fit or run a model')
    monkeypatch.setattr(b, '_fit_interval_scale', forbidden)
    monkeypatch.setattr(b.AdaptiveFieldBaseline, 'adapt', forbidden)
    monkeypatch.setattr(b, '_load_frozen', forbidden)
    assert b.verify_prior()['new_fits_or_forward_passes'] is False
    path.write_bytes(path.read_bytes()+b'tamper')
    with pytest.raises(RuntimeError, match='bytes changed'):
        b.verify_prior()
