"""Software fixtures only: no reserved site, World, training or real forecaster."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wirelessjev.fleetwm.data import Geometry, Observations
from wirelessjev.pilot_fastloop import matched
from wirelessjev.semantic_jepa import common as frozen_common
from wirelessjev.twin.radio import AP


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    p = {'development_sites': [21, 22], 'confirmation_sites': list(range(31, 39)),
         'families': list(matched.FAMILIES), 'duration_seconds': 3, 'robots': 2}
    out = tmp_path / 'artifacts/pilot_fastloop'
    (out / 'registration').mkdir(parents=True)
    (out / 'registration/source.json').write_text('{}')
    (out / 'protocol.json').write_text(json.dumps(p))
    permission = []
    c = SimpleNamespace(ROOT=tmp_path, OUT=out, protocol=lambda: p,
                        check_sources=lambda: {'source_sha256': {'fixture.py': 'a' * 64}},
                        checkpoint_hashes=lambda: {'fixture.pt': 'b' * 64},
                        digest=frozen_common.digest, value_digest=frozen_common.value_digest,
                        write_once=frozen_common.write_once, lock=frozen_common.locked,
                        check_confirmation_permission=lambda cost: permission.append(cost))
    monkeypatch.setattr(matched, '_common', lambda: c)
    geometry = Geometry(np.array([[0., 0.], [5., 8.]]), np.array([20., 20., 20.]),
                        np.array([-90., -90., -90.]),
                        (AP('a', 0., 0., 36), AP('b', 5., 8., 52)))
    rng = np.random.default_rng(37)
    shape = (63, 2, 6)
    heard = rng.random(shape) < .4
    scanned = rng.random((63, 2)) < .5
    # Hidden inactive buffers deliberately contain sentinel data.
    history = Observations(rng.normal(size=(63, 2, 2)).astype('f4'),
                           rng.normal(-65, 5, size=shape).astype('f4'), heard,
                           np.full(shape, -90., dtype='f4'), scanned)
    plans = rng.normal(size=(3, 2, 9, 2)).astype('f4')
    labels = rng.normal(20, 8, size=(3, 2, 9, 2, 3)).astype('f4')
    collector = {'geometry': geometry, 'history': history, 'plans': plans, 'labels': labels}
    return SimpleNamespace(c=c, p=p, collector=collector, permission=permission,
                           key=('development', 21, 4., 'radiomap', 'pilot_fast'))


def _cache(fixture):
    return matched.save_inputs(fixture.key, fixture.collector)


def _reference(fixture, monkeypatch, receipt):
    monkeypatch.setattr(matched, '_reference', lambda key: ({'matched_inputs': receipt}, 'c' * 64))


def _fake_factory(events=None, malformed=None):
    class FixturePredictor:
        def __init__(self, geometry, family):
            self.calls = 0

        def step(self, history, plan):
            if events is not None:
                events.append(('forward', len(history.value)))
            assert len(history.value) == 60 + self.calls
            assert plan.shape == (2, 9, 2)
            mean = np.broadcast_to(plan[..., :1, None], (2, 9, 2, 3)).copy().astype('f4')
            std = np.ones_like(mean)
            # Mutating the request may not change shared cache or later queries.
            history.value[:] = 10000
            plan[:] = 10000
            kind = 'cold' if not self.calls else 'warm'
            self.calls += 1
            if malformed == 'negative_std':
                std[:] = -1
            if malformed == 'nan':
                mean.flat[0] = np.nan
            if malformed == 'shape':
                mean = mean[:, :1]
            if malformed == 'dtype':
                mean = mean.astype('f8')
            return mean, std, kind, .001
    return FixturePredictor


def _row(fixture, monkeypatch, *, replay=False):
    receipt = _cache(fixture)
    _reference(fixture, monkeypatch, receipt)
    public = matched.verify_inputs(receipt)
    key = ('development', 21, 'radiomap')
    mean, queries = matched._predict(public, key[2], predictor_factory=_fake_factory())
    path = matched._prediction_path(key, replay)
    frozen_common._write_bytes_once(path, matched._npz_bytes({'mean': mean}))
    row = {'schema': 1, 'completed': True, 'case': list(key), 'stage': key[0], 'site': key[1],
           'family': key[2], 'replay': replay, 'offline_only': True,
           'fresh_measured_deadline_claim': False, 'new_fits': 0, 'training_seed': None,
           **matched._identities(), 'reference_case': list(fixture.key),
           'reference_archive_sha256': 'c' * 64, 'matched_inputs': receipt,
           'query_decisions': [0, 1, 2], 'history_cutoffs_exclusive': [60, 61, 62],
           'per_query': queries, 'prediction_path': matched._relative(path),
           'prediction_file_sha256': fixture.c.digest(path), 'forecast_mean_sha256': matched._array_hash(mean),
           'metrics': matched._metrics(mean, fixture.collector['labels']),
           **{name: True for name in matched.PROOF_FIELDS},
           'hardware_qualified': False, 'production_qualified': False, 'activated': False}
    return row


def test_case_manifest_and_replay_boundaries(fixture):
    assert len(matched.case_keys('development')) == 4
    assert len(matched.case_keys('confirmation')) == 16
    assert matched.case_name(('development', 21, 'radiomap')) == 'development-21-radiomap'
    matched._guard(('confirmation', 31, 'radiomap'), True, fixture.p)
    with pytest.raises(ValueError):
        matched._guard(('confirmation', 32, 'radiomap'), True, fixture.p)
    with pytest.raises(ValueError):
        matched.validate_case_key(('development', 31, 'radiomap'))
    with pytest.raises(ValueError):
        matched.validate_case_key(('development', True, 'radiomap'))
    with pytest.raises(ValueError):
        matched._guard(('development', 21, 'radiomap'), 1, fixture.p)


def test_public_cache_separates_labels_and_preserves_visible_partial_pilots(fixture):
    receipt = _cache(fixture)
    public = matched.verify_inputs(receipt, reference_key=fixture.key)
    history = fixture.collector['history']
    assert set(np.load(fixture.c.ROOT / receipt['public_path']).files) == set(matched.PUBLIC_KEYS)
    np.testing.assert_array_equal(public.history.value[history.heard], history.value[history.heard])
    np.testing.assert_array_equal(public.history.value[~history.heard], 0)
    np.testing.assert_array_equal(public.history.bound[~history.scanned], 0)
    pilot = history.heard & ~history.scanned[..., None]
    assert pilot.any()
    np.testing.assert_array_equal(public.history.value[pilot], history.value[pilot])
    assert not public.history.value.flags.writeable
    with pytest.raises(ValueError):
        public.plans.setflags(write=True)
    assert receipt == _cache(fixture)


def test_inactive_rf_buffer_changes_do_not_change_public_cache(fixture):
    receipt = _cache(fixture)
    h = fixture.collector['history']
    newvalue, newbound = h.value.copy(), h.bound.copy()
    newvalue[~h.heard] = 3456
    newbound[~h.scanned] = 7890
    fixture.collector['history'] = replace(h, value=newvalue, bound=newbound)
    assert _cache(fixture) == receipt


def test_verify_inputs_does_not_open_labels(fixture, monkeypatch):
    receipt = _cache(fixture)
    original = matched._load_npz
    def load(path, keys):
        assert '.labels.' not in str(path)
        return original(path, keys)
    monkeypatch.setattr(matched, '_load_npz', load)
    assert matched.verify_inputs(receipt).plans.shape == (3, 2, 9, 2)


def test_confirmation_permission_precedes_model_and_cache_access(fixture, monkeypatch):
    events = []
    def deny(cost):
        events.append('permission')
        raise RuntimeError('fixture blocked')
    fixture.c.check_confirmation_permission = deny
    monkeypatch.setattr(matched, '_identities', lambda: pytest.fail('identity before permission'))
    monkeypatch.setattr(matched, 'load_case', lambda *a, **k: pytest.fail('cache before permission'))
    with pytest.raises(RuntimeError, match='fixture blocked'):
        matched.run_case(('confirmation', 31, 'radiomap'))
    with pytest.raises(RuntimeError, match='fixture blocked'):
        matched.save_inputs(('confirmation', 31, 4., 'radiomap', 'pilot_fast'), fixture.collector)
    assert events == ['permission', 'permission']


def test_strict_past_prefixes_and_request_mutations_are_isolated(fixture):
    public = matched.verify_inputs(_cache(fixture))
    original_values, original_plans = public.history.value.copy(), public.plans.copy()
    events = []
    means, queries = matched._predict(public, 'radiomap', predictor_factory=_fake_factory(events))
    assert events == [('forward', 60), ('forward', 61), ('forward', 62)]
    assert means.shape == (3, 2, 9, 2, 3)
    assert [q['history_cutoff_exclusive'] for q in queries] == [60, 61, 62]
    np.testing.assert_array_equal(public.history.value, original_values)
    np.testing.assert_array_equal(public.plans, original_plans)


@pytest.mark.parametrize('malformed', ['negative_std', 'nan', 'shape', 'dtype'])
def test_invalid_real_output_contract_is_rejected(fixture, malformed):
    public = matched.verify_inputs(_cache(fixture))
    with pytest.raises(RuntimeError, match='Malformed matched model output'):
        matched._predict(public, 'radiomap', predictor_factory=_fake_factory(malformed=malformed))


def test_primary_metric_uses_only_nonzero_short_deltas():
    truth = np.zeros((1, 1, 9, 1, 3), dtype='f4')
    prediction = truth + 100
    assert matched._metrics(prediction, truth)['short_route_delta_rmse_db'] == 0
    prediction[:, :, 1:6] += 2
    prediction[:, :, 6:] += 1000
    metrics = matched._metrics(prediction, truth)
    assert metrics['short_route_delta_rmse_db'] == 2
    assert metrics['all_route_delta_rmse_db'] > 500


def test_npz_serialization_is_deterministic_and_nonpickled():
    a, b = np.arange(6, dtype='f4'), np.ones(4, dtype=bool)
    assert matched._npz_bytes({'a': a, 'b': b}) == matched._npz_bytes({'b': b, 'a': a})
    with pytest.raises(ValueError):
        matched._npz_bytes({'bad': np.array([{}], dtype=object)})


@pytest.mark.parametrize('field', ['public', 'labels', 'metadata'])
def test_changed_cache_bytes_are_rejected(fixture, field):
    receipt = _cache(fixture)
    path = fixture.c.ROOT / receipt[field + '_path']
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(RuntimeError, match='identity changed'):
        matched.verify_inputs(receipt)


def test_path_escape_and_wrong_reference_rejected(fixture):
    receipt = _cache(fixture)
    receipt['public_path'] = '../escape.npz'
    with pytest.raises(RuntimeError, match='Unsafe'):
        matched.verify_inputs(receipt)
    with pytest.raises(ValueError):
        matched.save_inputs(('development', 21, 4., 'radiomap', 'base'), fixture.collector)


def test_partial_inputs_fail_closed(fixture):
    public, _, _ = matched._input_paths(fixture.key)
    public.parent.mkdir(parents=True)
    public.write_bytes(b'partial')
    with pytest.raises(RuntimeError, match='Partial matched input'):
        _cache(fixture)


def test_passive_metrics_recomputed_from_full_saved_prediction(fixture, monkeypatch):
    row = _row(fixture, monkeypatch)
    assert matched.validate_row(row) is True
    row['metrics']['short_route_delta_rmse_db'] += .01
    with pytest.raises(RuntimeError, match='metrics differ'):
        matched.validate_row(row)


@pytest.mark.parametrize('change', ['cutoff', 'hash', 'source', 'reference', 'proof'])
def test_passive_validation_rejects_corruption(fixture, monkeypatch, change):
    row = _row(fixture, monkeypatch)
    if change == 'cutoff':
        row['per_query'][1]['history_cutoff_exclusive'] += 1
    elif change == 'hash':
        row['per_query'][0]['forecast_sha256'] = '0' * 64
    elif change == 'source':
        row['source_sha256'] = {}
    elif change == 'reference':
        row['reference_archive_sha256'] = '0' * 64
    else:
        row['causality_pass'] = 1
    with pytest.raises(RuntimeError):
        matched.validate_row(row)


def test_partial_matched_archive_refuses_rerun(fixture):
    path, _ = matched.archive_paths(('development', 21, 'radiomap'))
    path.parent.mkdir(parents=True)
    path.write_bytes(b'partial')
    with pytest.raises(RuntimeError, match='Partial matched archive'):
        matched.load_case(('development', 21, 'radiomap'))


def test_full_run_opens_labels_after_all_forwards_and_reuses_archive(fixture, monkeypatch):
    receipt = _cache(fixture)
    _reference(fixture, monkeypatch, receipt)
    events = []
    import wirelessjev.semantic_jepa.runtime as runtime
    monkeypatch.setattr(runtime, 'Predictor', _fake_factory(events))
    original = matched._load_labels
    def labels(public, p):
        assert [event for event in events if event[0] == 'forward'] == [('forward', 60), ('forward', 61), ('forward', 62)]
        events.append(('labels', None))
        return original(public, p)
    monkeypatch.setattr(matched, '_load_labels', labels)
    key = ('development', 21, 'radiomap')
    row = matched.run_case(key)
    assert row['offline_only'] is True
    assert row['fresh_measured_deadline_claim'] is False
    assert matched.load_case(key) == row
    assert matched.run_case(key) == row
    assert len([e for e in events if e[0] == 'forward']) == 3


def test_exact_replay_ignores_fresh_elapsed_but_checks_std(fixture, monkeypatch):
    fixture.key = ('confirmation', 31, 4., 'radiomap', 'pilot_fast')
    receipt = _cache(fixture)
    _reference(fixture, monkeypatch, receipt)
    import wirelessjev.semantic_jepa.runtime as runtime
    monkeypatch.setattr(runtime, 'Predictor', _fake_factory())
    key = ('confirmation', 31, 'radiomap')
    original = matched.run_case(key)
    replay = matched.run_case(key, replay=True)
    assert matched.compare_replay(original, replay)['exact'] is True
    changed = deepcopy(replay)
    changed['per_query'][0]['actual_prediction_seconds'] = 55.
    assert matched.compare_replay(original, changed)['exact'] is True
    changed['per_query'][0]['forecast_std_sha256'] = 'e' * 64
    result = matched.compare_replay(original, changed)
    assert result['exact'] is False
    assert result['checks']['per_query_hashes'] is False


def test_archive_bytes_and_receipt_are_verified(fixture, monkeypatch):
    receipt = _cache(fixture)
    _reference(fixture, monkeypatch, receipt)
    import wirelessjev.semantic_jepa.runtime as runtime
    monkeypatch.setattr(runtime, 'Predictor', _fake_factory())
    key = ('development', 21, 'radiomap')
    matched.run_case(key)
    path, _ = matched.archive_paths(key)
    value = json.loads(gzip.decompress(path.read_bytes()))
    value['metrics']['signal_rmse_db'] += .01
    path.write_bytes(gzip.compress(json.dumps(value).encode(), mtime=0))
    with pytest.raises(RuntimeError, match='receipt changed'):
        matched.load_case(key)
