"""Local fixture checks only; these do not run the registered study."""
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from wirelessjev.fleetwm.data import FleetInputs
from wirelessjev.semantic_jepa import training as t
from wirelessjev.semantic_jepa.model import SemanticJEPA


@pytest.fixture(autouse=True)
def one_thread():
    threads = torch.get_num_threads(); deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads); torch.use_deterministic_algorithms(deterministic)


@pytest.fixture
def common(tmp_path, monkeypatch):
    p = {'training_sites': [101], 'validation_sites': [102], 'training_robots': 4,
         'training_decisions': 2, 'training_query_stride': 1, 'training_seeds': [0, 1, 2],
         'families': list(t.FAMILIES), 'epochs': 2, 'training_batch_size': 4,
         'learning_rate': .0007, 'weight_decay': .0002, 'auxiliary_weight': .1,
         'scan_period_seconds': 5, 'map_refresh_seconds': 20}
    def write_once(path, value, **kwargs):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        data = t._json_bytes(value)
        if path.exists():
            if path.read_bytes() != data: raise RuntimeError('Fixture immutable evidence differs')
        else: path.write_bytes(data)
    c = SimpleNamespace(ROOT=tmp_path, OUT=tmp_path/'artifacts', WORK=tmp_path/'work',
                        protocol=lambda: copy.deepcopy(p), check_sources=lambda: None,
                        source_hashes=lambda: {'unit-fixture.py': 'fixed-fixture-source'},
                        digest=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                        write_once=write_once, fixture_protocol=p)
    monkeypatch.setattr(t, '_common', lambda: c)
    return c


def arrays(protocol, split='training'):
    rng = np.random.default_rng(17)
    n = protocol['training_robots']; sites = protocol[split+'_sites']
    times = list(range(60, 60+protocol['training_decisions'], protocol['training_query_stride']))
    b = len(sites)*len(times)*n; links = 6
    base = (rng.normal(size=(b, 9, links)) + 12).astype('f2')
    ctx = (rng.normal(size=(b, links, 6, 8))*.1).astype('f2')
    ctx[..., 0] += .5; ctx[..., 2:4] = 1
    ctx[..., 7] = np.tile([0, .5, 1], 2)[None, :, None]
    query = (rng.normal(size=(b, 9, links, 11))*.1).astype('f2')
    query[..., 5] = np.tile([0, .5, 1], 2)
    result = {'context': ctx.reshape(b, links, 48), 'query': query,
              'memory': (rng.normal(size=(b, links, 13))*.1).astype('f2'),
              'map_mean': base, 'map_std': np.full_like(base, 2),
              'physics': base-1, 'fleet_mean': base+1,
              'truth': (base+rng.normal(size=base.shape)).astype('f2'),
              'valid_link': np.ones((b, links), dtype='f2')}
    result['future_context'] = np.repeat(result['context'][:, None], 9, axis=1)
    result['site_id'] = np.repeat(sites, len(times)*n).astype('i8')
    result['decision_time'] = np.tile(np.repeat(times, n), len(sites)).astype('i2')
    return result


def cache(c, split):
    identity = t._identity(split); value = arrays(c.protocol(), split)
    for seed in c.protocol()[split+'_sites']:
        c.write_once(c.OUT/'collection'/f'{split}-{seed}.json',
                     {'identity': identity, 'split': split, 'site': seed,
                      'collection': {'unit_fixture': True, 'no_scientific_data': True}})
    payload = dict(value); payload[t.NPZ_IDENTITY] = np.frombuffer(t._json_bytes(identity), dtype='u1')
    stream = io.BytesIO(); np.savez_compressed(stream, **payload)
    path = c.WORK/(split+'.npz'); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(stream.getvalue())
    return value


def test_unregistered_data_and_fit_refuse_before_creation(common):
    def reject(): raise RuntimeError('unregistered fixture')
    common.check_sources = reject
    with pytest.raises(RuntimeError, match='unregistered'): t.dataset('training')
    with pytest.raises(RuntimeError, match='unregistered'): t.fit_one(('supervised', 0))
    assert not common.WORK.exists()


def test_cached_data_creates_receipt_and_reuses_without_generating(common, monkeypatch):
    expected = cache(common, 'training')
    monkeypatch.setattr(t, 'training_episode', lambda task: pytest.fail('must use immutable fixture cache'))
    actual = t.dataset('training')
    for key in expected: np.testing.assert_array_equal(actual[key], expected[key])
    receipt = common.OUT/'data/training.json'; original = receipt.read_bytes()
    t.dataset('training')
    assert receipt.read_bytes() == original
    assert json.loads(original)['sites'] == [101]


def test_cache_tampering_and_source_change_fail(common):
    cache(common, 'training'); t.dataset('training')
    path = common.WORK/'training.npz'
    with np.load(path, allow_pickle=False) as z: changed = {k: z[k] for k in z.files}
    changed['truth'][0, 0, 0] += 2
    with path.open('wb') as stream: np.savez_compressed(stream, **changed)
    with pytest.raises(RuntimeError, match='receipt/cache'): t.dataset('training')
    common.source_hashes = lambda: {'unit-fixture.py': 'different'}
    with pytest.raises(RuntimeError, match='identity'): t.dataset('training')


def test_missing_registered_cache_does_not_auto_rebuild(common, monkeypatch):
    cache(common, 'training'); t.dataset('training'); (common.WORK/'training.npz').unlink()
    monkeypatch.setattr(t, 'training_episode', lambda task: pytest.fail('must not rebuild'))
    with pytest.raises(RuntimeError, match='cache missing'): t.dataset('training')


@pytest.mark.parametrize('kind', ['site', 'query', 'mask', 'dimension', 'nan'])
def test_dataset_membership_and_schema_guards(common, kind):
    a = arrays(common.protocol())
    if kind == 'site': a['site_id'][0] = 999
    if kind == 'query': a['decision_time'][0] += 1
    if kind == 'mask': a['valid_link'][0, 0] = .5
    if kind == 'dimension': a['query'] = a['query'][:, :, :, :-1]
    if kind == 'nan': a['truth'][0, 0, 0] = np.nan
    with pytest.raises(RuntimeError): t._validate_arrays(a, t._identity('training'))


def test_batch_excludes_metadata_and_copies_input(common):
    a = arrays(common.protocol()); b = t.batch(a, slice(0, 2))
    assert not set(t.METADATA) & set(b)
    b['context'][0, 0, 0] += 10
    assert b['context'][0, 0, 0] != a['context'][0, 0, 0]


def test_validation_excludes_padded_ap_even_when_its_truth_would_win(common):
    a = arrays(common.protocol()); a['valid_link'][:, 3:] = 0
    a['truth'][:, :, :3] = -10; a['map_mean'][:, :, :3] = -10
    a['truth'][:, :, 3:] = 10000; a['map_mean'][:, :, 3:] = -30000
    class Reference(torch.nn.Module):
        def forward(self, batch): return {'mean': batch['map_mean']}
    assert t.validation_metrics(Reference(), a)['validation_score'] == 0


def test_resume_is_numerically_identical_after_real_epoch_commit(common, monkeypatch):
    for split in ('training', 'validation'): cache(common, split)
    baseline = t.fit_one(('supervised', 0))
    initial_checkpoint = torch.load(common.OUT/'checkpoints/supervised-0.pt', weights_only=True)
    base_history = baseline['history']
    common.OUT = common.ROOT/'resumed-artifacts'; common.WORK = common.ROOT/'resumed-work'
    for split in ('training', 'validation'): cache(common, split)
    original = t._atomic_bytes; tripped = False
    def crash_after_epoch(path, data, **kwargs):
        nonlocal tripped
        original(path, data, **kwargs)
        if str(path).endswith('.last.pt') and not tripped:
            tripped = True
            raise SystemExit('fixture process interruption after committed epoch')
    monkeypatch.setattr(t, '_atomic_bytes', crash_after_epoch)
    with pytest.raises(SystemExit): t.fit_one(('supervised', 0))
    assert torch.load(common.WORK/'supervised-0.last.pt', weights_only=True)['completed_epochs'] == 1
    assert not (common.OUT/'checkpoints/supervised-0.pt').exists()
    monkeypatch.setattr(t, '_atomic_bytes', original)
    resumed = t.fit_one(('supervised', 0))
    resumed_checkpoint = torch.load(common.OUT/'checkpoints/supervised-0.pt', weights_only=True)
    assert resumed['history'] == base_history
    assert resumed['best_epoch'] == baseline['best_epoch']
    for key, value in initial_checkpoint['state'].items():
        torch.testing.assert_close(value, resumed_checkpoint['state'][key], rtol=0, atol=0)
    assert resumed['parameters_inference'] < resumed['parameters_total_including_training_only']
    assert resumed['diagnostics'] is not None
    frozen_bytes = (common.OUT/'fits/supervised-0.json').read_bytes()
    monkeypatch.setattr(t, 'dataset', lambda *args: pytest.fail('completed fit must not train again'))
    assert t.fit_one(('supervised', 0)) == resumed
    assert (common.OUT/'fits/supervised-0.json').read_bytes() == frozen_bytes


def test_resume_source_mismatch_and_checkpoint_mutation_are_refused(common, monkeypatch):
    for split in ('training', 'validation'): cache(common, split)
    result = t.fit_one(('supervised', 0)); checkpoint = common.OUT/'checkpoints/supervised-0.pt'
    checkpoint.write_bytes(checkpoint.read_bytes()+b'changed')
    with pytest.raises(RuntimeError, match='checkpoint changed'): t.fit_one(('supervised', 0))
    assert result['completed']
    with pytest.raises(RuntimeError): t.load('supervised', 0)


def test_prediction_boundary_and_frozen_reload(common):
    for split in ('training', 'validation'): cache(common, split)
    t.fit_one(('semantic_jepa', 1)); model = t.load('semantic_jepa', 1)
    a = arrays(common.protocol()); observed = FleetInputs(*(a[k].astype('f4') for k in t.FIELDS))
    mean, std = t.predict(model, observed)
    assert mean.shape == std.shape == a['truth'].shape
    assert np.isfinite(mean).all() and np.all(std > 0)
    with pytest.raises(TypeError): t.predict(model, a)
    direct = model(t.batch(a, slice(None)))
    np.testing.assert_allclose(mean, direct['mean'].detach().numpy(), rtol=0, atol=0)


def test_atomic_immutable_write_never_overwrites_fixture(tmp_path):
    path = tmp_path/'evidence'
    t._atomic_bytes(path, b'one', immutable=True)
    t._atomic_bytes(path, b'one', immutable=True)
    with pytest.raises(RuntimeError, match='Immutable'): t._atomic_bytes(path, b'two', immutable=True)
    assert path.read_bytes() == b'one'
    assert not list(tmp_path.glob('.evidence*'))


def test_training_split_refuses_confirmation_and_unregistered_fit(common):
    with pytest.raises(ValueError): t.dataset('confirmation')
    with pytest.raises(ValueError): t.fit_one(('semantic_jepa', 999))
    with pytest.raises(ValueError): t.fit_one(('unknown', 0))
    assert not common.WORK.exists()


def test_explicit_missing_cache_rebuild_must_match_archived_digest(common, monkeypatch):
    cache(common, 'training'); t.dataset('training')
    original = (common.OUT/'data/training.json').read_bytes()
    (common.WORK/'training.npz').unlink()
    monkeypatch.setattr(t, 'training_episode', lambda task: arrays(common.protocol(), task[0]))
    t.dataset('training', rebuild_missing=True)
    assert (common.OUT/'data/training.json').read_bytes() == original


def test_actual_small_episode_keeps_future_truth_outside_observable_inputs(common, monkeypatch):
    from dataclasses import replace
    from wirelessjev.semantic_jepa import collection
    original = collection.collect
    first = t.training_episode(('training', 0, 101))
    def shifted_truth(*args, **kwargs):
        full, geometry, observed, receipt = original(*args, **kwargs)
        return replace(full, snr=full.snr+3.), geometry, observed, receipt
    monkeypatch.setattr(collection, 'collect', shifted_truth)
    second = t.training_episode(('training', 0, 101))
    for key in t.FIELDS + ('future_context',):
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_allclose(second['truth'].astype('f4')-first['truth'].astype('f4'), 3., atol=.032)
    assert np.all(first['site_id'] == 101)
    assert set(first['decision_time']) == {60, 61}


def test_collection_receipt_tampering_invalidates_dataset(common):
    cache(common, 'training'); t.dataset('training')
    path = common.OUT/'collection/training-101.json'
    changed = json.loads(path.read_text()); changed['collection']['unit_fixture'] = False
    path.write_bytes(t._json_bytes(changed))
    with pytest.raises(RuntimeError, match='receipt/cache'): t.dataset('training')


def test_collection_identity_mismatch_refuses_before_archiving(common, monkeypatch):
    from wirelessjev.semantic_jepa import collection
    monkeypatch.setattr(collection, 'collect', lambda *args: (None, None, None, {'seed': 999}))
    with pytest.raises(RuntimeError, match='site/query identity'):
        t.training_episode(('training', 0, 101))
    assert not (common.OUT/'collection').exists()
