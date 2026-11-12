"""Disposable analytic data/guard fixtures; no ray solves or neural calls."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wirelessjev.nested_radio import data as d
from wirelessjev.hierarchical_radio import data as old
from wirelessjev.hierarchical_radio import simulation as sim


def calibration():
    return {'schema': 1, 'reference_device': 0, 'estimates': {
        '0': {'count': 16, 'quantization_db': 1., 'offset_db': 0., 'unknown': False},
        '1': {'count': 16, 'quantization_db': .5, 'offset_db': -3., 'unknown': False}}}


def analytic_scene(stage='test', index=0, state=0, *, unknown=False):
    spec = d.case_spec(stage, index, state); g = spec.geometry; aps = len(g.ap_positions_m)
    power = np.broadcast_to((-58. - 3 * np.arange(aps))[None, :, None, None], (48, aps, 3, 2)).copy()
    power += (np.arange(48)[:, None, None, None] % 3) * .3 - state * .8
    gain = 10 ** (power / 10); solved = np.ones(gain.shape, bool)
    if unknown: solved[:, 0] = False
    reports = sim.reports_from_gain(gain, solved, spec.tx_power_dbm, noise_seed=g.seed)
    return old.PublicScene(stage, index, state, spec.layout_id, spec.case_id,
        np.asarray(g.ap_positions_m), np.asarray(g.rx_positions_m),
        np.array((g.width_m, g.depth_m, g.ceiling_m)),
        np.array([[lo, hi] for _, lo, hi in sim.obstacle_boxes(g)]), np.asarray(sim.BANDS_HZ),
        spec.tx_power_dbm, reports['noise_floor_dbm'], reports['rssi_dbm'], reports['heard'],
        reports['censored'], reports['censor_upper_dbm'], reports['quantization_interval_dbm'],
        reports['reported_detection_threshold_dbm'], {})


@pytest.fixture
def common(tmp_path, monkeypatch):
    root = tmp_path / 'software-repository'; out = root / 'artifacts/nested_radio'; out.mkdir(parents=True)
    calls = []
    def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def write(path, value):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as handle: json.dump(value, handle, sort_keys=True, allow_nan=False)
    def reference(path):
        path = Path(path).resolve()
        assert path.is_relative_to(root)
        return {'path': str(path.relative_to(root)), 'sha256': digest(path)}
    def npz(path, values):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle: np.savez_compressed(handle, **values)
        return reference(path)
    def arrays(path):
        with np.load(path, allow_pickle=False) as source: return {k: source[k].copy() for k in source.files}
    def authorize(spec, folder=None):
        calls.append(('authorize', spec.case_id))
        return {'authorized': True, 'case_id': spec.case_id, 'case_path': str((out / 'rt_data/cases' / spec.case_id).resolve()),
                'source_receipt_sha256': 'a' * 64, 'registry_sha256': 'b' * 64}
    c = SimpleNamespace(ROOT=root, OUT=out, REGISTRY=out/'registry.json', digest=digest, write=write,
        read=lambda p: json.loads(Path(p).read_bytes()), reference=reference, arrays=arrays, npz=npz,
        safe_path=lambda p: root/p, source_receipt_sha256=lambda: 'a'*64, registry_sha256=lambda: 'b'*64,
        protocol_sha256=lambda: 'c'*64, check_sources=lambda: calls.append(('source',)),
        verify_parent_data=lambda: {'strict_parent_guard': 'passed', 'selected_data': ['training', 'validation']},
        case_path=lambda s: out/'rt_data/cases'/s.case_id, authorize=authorize, calls=calls)
    monkeypatch.setattr(d, '_common', lambda: c); monkeypatch.setattr(d, '_GROUPS', {})
    write(out/'selection.json', {'scope': 'software fixture, never a fit'})
    return c


def parent_inputs(c):
    base = c.ROOT/'artifacts/hierarchical_radio'; groups = {}; stages = {}
    c.write(base/'data/calibration.json', {'calibration': calibration()})
    for stage in ('training', 'validation'):
        path = base/'rt_data'/f'{stage}.json'; c.write(path, {'stage': stage, 'software_fixture': True}); stages[stage] = c.reference(path)
        for regime in ('full', 'sparse'):
            name = stage+'-'+regime
            ref = c.npz(base/'data/groups'/f'{name}.npz', {'analytic_metadata_fixture': np.zeros((2, 2))})
            path = base/'data/groups'/f'{name}.json'; c.write(path, {'arrays': ref}); groups[name] = c.reference(path)
    c.write(base/'data/prepared.json', {'groups': groups, 'stages': stages,
        'target_band_indices': [0, 1], 'calibration': c.reference(base/'data/calibration.json')})


def public_folder(c, stage='test', index=0, state=0):
    spec = d.case_spec(stage, index, state); folder = c.case_path(spec)
    scene = analytic_scene(stage, index, state); _, meshes = sim.write_scene(spec, folder/'scene')
    views = {'rssi_dbm': scene.rssi_dbm, 'heard': scene.heard, 'censored': scene.censored,
        'censor_upper_dbm': scene.censor_upper_dbm, 'quantization_interval_dbm': scene.quantization_interval_dbm,
        'reported_detection_threshold_dbm': scene.detection_threshold_dbm, 'noise_floor_dbm': scene.noise_floor_dbm,
        'ap_positions_m': scene.ap_xyz.astype('f4'), 'rx_positions_m': scene.rx_xyz.astype('f4'),
        'bbox_dims_m': scene.dimensions, 'carrier_hz': scene.carriers}
    c.npz(folder/'public_views.npz', views)
    receipt = {'case': spec.to_dict(), 'case_id': spec.case_id, 'layout_id': spec.layout_id,
        'public_geometry': sim.public_geometry(spec.geometry), 'public_geometry_sha256': spec.geometry_sha256,
        'known_tx_power_dbm': spec.tx_power_dbm, 'source_receipt_sha256': 'a'*64, 'registry_sha256': 'b'*64,
        'real_device_measurements': False, 'unresolved_paths_are_censored': False,
        'files_sha256': {str(p.relative_to(folder)): sim.digest(p) for p in meshes+[folder/'public_views.npz']}}
    c.write(folder/'receipt.json', receipt)
    return spec, folder


def fresh_stage(c, monkeypatch):
    # Only disposable software schemas are constructed; no trace_case is called.
    monkeypatch.setattr(d, 'COUNTS', {'test': 1, 'ap_transfer': 1})
    parent_inputs(c); d.prepare_training()
    refs = {}
    for state in d.STATES:
        spec, folder = public_folder(c, state=state); refs[spec.case_id] = c.reference(folder/'receipt.json')
    c.write(d.stage_path('test'), {'stage': 'test', 'complete': True, 'source_receipt_sha256': 'a'*64,
        'registry_sha256': 'b'*64, 'layouts': 1, 'cases': 3, 'case_receipts': refs})


def test_registry_contains84_fresh_cases_disjoint_from_all240_prior_layouts():
    value = d.geometry_registry()
    assert value['case_count'] == 84 and value['prior_geometry_records'] == 240
    assert value['all_new_geometries_disjoint_from_prior240'] is True
    rows = [row for stage in value['stages'].values() for row in stage]
    assert len(rows) == 28 and len({r['public_geometry_sha256'] for r in rows}) == 28
    assert all(r['layout_id'].startswith('nested-') and len(r['cases']) == 3 for r in rows)


@pytest.mark.parametrize('stage', list(d.COUNTS))
def test_material_states_retain_geometry_TX_noise_seed_and_change_one_object(stage):
    cases = [d.case_spec(stage, 0, s) for s in d.STATES]
    assert len({s.geometry_sha256 for s in cases}) == len({s.tx_power_dbm for s in cases}) == len({s.geometry.seed for s in cases}) == 1
    assert cases[1].materials == ('metal', cases[0].materials[1])
    assert cases[2].materials == (cases[0].materials[0], 'metal')
    assert len(cases[0].geometry.ap_positions_m) == (8 if stage == 'ap_transfer' else 4)
    assert cases[0] != sim.case_spec(stage, 0, 0)
    for _, lo, hi in sim.obstacle_boxes(cases[0].geometry):
        points = np.asarray(cases[0].geometry.ap_positions_m+cases[0].geometry.rx_positions_m)
        assert not np.all((points > lo) & (points < hi), axis=-1).any()


@pytest.mark.parametrize('args', [('training', 0, 0), ('bad', 0, 0), ('test', True, 0), ('test', 24, 0), ('test', 0, True), ('test', 0, 3)])
def test_unregistered_or_boolean_fresh_case_rejected(args):
    with pytest.raises(ValueError): d.case_spec(*args)


def test_geometry_registry_rejects_any_old_or_repeated_fingerprint(monkeypatch):
    monkeypatch.setattr(d, 'geometry_spec', lambda stage, index: sim.geometry_spec('test', 0))
    with pytest.raises(RuntimeError, match='overlap'): d.geometry_registry()


def test_publication_guard_precedes_solver_and_attempt(common, monkeypatch):
    called = []; spec = d.case_spec('test', 0, 0)
    monkeypatch.setattr(sim, '_trace_case', lambda *a, **kw: called.append(True))
    monkeypatch.setattr(common, 'authorize', lambda *a, **kw: {'authorized': False})
    with pytest.raises(RuntimeError): d.trace_case(spec)
    assert called == [] and not common.case_path(spec).exists()


@pytest.mark.parametrize('change', ['case', 'source', 'registry', 'path', 'boolean_permission'])
def test_changed_authorization_is_rejected_before_solver(common, monkeypatch, change):
    spec = d.case_spec('test', 0, 0); permission = common.authorize(spec)
    if change == 'case': permission['case_id'] = 'other'
    if change == 'source': permission['source_receipt_sha256'] = 'not-a-sha'
    if change == 'registry': permission['registry_sha256'] = ''
    if change == 'path': permission['case_path'] = str(common.OUT/'another')
    if change == 'boolean_permission': permission['authorized'] = 1
    monkeypatch.setattr(common, 'authorize', lambda *a, **kw: permission)
    monkeypatch.setattr(sim, '_trace_case', lambda *a, **kw: pytest.fail('Solver reached after bad permission'))
    with pytest.raises(RuntimeError): d.trace_case(spec)


def test_partial_attempt_cannot_trigger_replacement_trace(common, monkeypatch):
    spec = d.case_spec('test', 0, 0); folder = common.case_path(spec); folder.mkdir(parents=True)
    monkeypatch.setattr(sim, '_trace_case', lambda *a, **kw: pytest.fail('Partial case must not retrace'))
    with pytest.raises(RuntimeError, match='Partial'): d.trace_case(spec)


def test_attempt_is_durable_before_worker_and_changed_postguard_retains_partial(common, monkeypatch):
    spec = d.case_spec('test', 0, 0); folder = common.case_path(spec)
    permission = common.authorize(spec); calls = []
    def authorize(*a, **kw):
        return permission if not calls else {**permission, 'registry_sha256': 'f'*64}
    def analytic_worker(s, p):
        assert common.read(p/'attempt.json')['authorized'] is True
        assert common.read(p/'attempt.json')['case_id'] == spec.case_id
        calls.append('fixture-worker'); return {}
    monkeypatch.setattr(common, 'authorize', authorize); monkeypatch.setattr(sim, '_trace_case', analytic_worker)
    with pytest.raises(RuntimeError, match='changed during'): d.trace_case(spec)
    assert calls == ['fixture-worker'] and (folder/'attempt.json').exists() and not (folder/'receipt.json').exists()
    with pytest.raises(RuntimeError, match='Partial'): d.trace_case(spec)


def test_reader_opens_only_report_NPZ_and_public_vertices(common, monkeypatch):
    spec, folder = public_folder(common); original = Path.open
    def guarded(path, *args, **kwargs):
        if path.name in ('scene.xml', 'physical_channels.npz') or path.name.endswith('-cir.npz'):
            pytest.fail('Private physical/material bytes may not enter the public reader')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    scene = d.load_scene(folder, expected_spec=spec)
    assert scene.layout_id == 'nested-test-00' and scene.ap_xyz.shape == (4, 3)
    assert len(scene.source_files_sha256) == 10


@pytest.mark.parametrize('bad', ['public', 'mesh', 'source', 'registry', 'spec', 'scope', 'symlink'])
def test_loader_refuses_changed_sources_or_scope(common, bad):
    spec, folder = public_folder(common)
    if bad == 'public': (folder/'public_views.npz').write_bytes(b'changed')
    if bad == 'mesh': (folder/'scene/meshes/floor.ply').write_text('changed')
    if bad in ('source', 'registry', 'scope'):
        receipt = common.read(folder/'receipt.json')
        receipt[{'source':'source_receipt_sha256','registry':'registry_sha256','scope':'real_device_measurements'}[bad]] = 'changed'
        (folder/'receipt.json').write_text(json.dumps(receipt))
    if bad == 'spec': spec = d.case_spec('test', 1, 0)
    if bad == 'symlink':
        actual = folder/'receipt-actual.json'; (folder/'receipt.json').rename(actual); (folder/'receipt.json').symlink_to(actual)
    with pytest.raises((ValueError, RuntimeError)): d.load_scene(folder, expected_spec=spec)


def test_training_preparation_is_strict_immutable_and_public_input_only(common):
    parent_inputs(common); value = d.prepare_training()
    assert d.verify_prepared() == value and d.prepare_training() == value
    assert value['training_cases'] == 72 and value['validation_cases'] == 18
    names = str(value['parent_inputs'])
    assert 'test-' not in names and 'development-' not in names
    assert value['reused_parent_TEST_or_DEV_for_fitting'] is False
    path = common.ROOT/'artifacts/hierarchical_radio/data/groups/training-full.npz'
    path.write_bytes(b'changed')
    with pytest.raises(RuntimeError): d.verify_prepared()


def test_parent_dependency_guard_cannot_be_skipped_by_cached_prepared(common, monkeypatch):
    parent_inputs(common); d.prepare_training()
    def refused(): raise RuntimeError('parent input failed byte binding')
    monkeypatch.setattr(common, 'verify_parent_data', refused)
    with pytest.raises(RuntimeError, match='byte binding'): d.verify_prepared()


@pytest.mark.parametrize('stage,regime', [('test','full'),('development','sparse'),('ap_transfer','full'),('training','invalid')])
def test_old_holdouts_never_enter_training_reader(stage, regime, common, monkeypatch):
    from wirelessjev.hierarchical_radio import study
    monkeypatch.setattr(study, 'examples', lambda *a, **kw: pytest.fail('Forbidden parent data accessed'))
    with pytest.raises(ValueError): d.training_examples(stage, regime)


def parent_examples(stage, regime):
    base = old.make_example(analytic_scene(), calibration(), regime=regime); values = []
    for index in range(3*(24 if stage == 'training' else 6)):
        spec = sim.case_spec(stage, index//3, index%3); e = deepcopy(base)
        e['binding'].update(stage=stage, index=index//3, state=index%3, condition=index%3,
            layout_id=spec.layout_id, case_id=spec.case_id, context_case_id=spec.case_id,
            regime=regime, held_band=False, stale_requested=False)
        values.append(e)
    return values


@pytest.mark.parametrize('stage', ['training', 'validation'])
def test_reused_training_reader_preserves_hashes_and_owns_immutable_arrays(common, monkeypatch, stage):
    from wirelessjev.hierarchical_radio import study
    parent_inputs(common); d.prepare_training(); values = parent_examples(stage, 'sparse')
    monkeypatch.setattr(study, 'examples', lambda s, r: values)
    loaded = d.training_examples(stage, 'sparse')
    assert len(loaded) == len(values)
    assert loaded[0]['binding']['model_input_sha256'] == values[0]['binding']['model_input_sha256']
    assert not loaded[0]['a']['context'].flags.writeable
    with pytest.raises(ValueError): loaded[0]['a']['context'].setflags(write=True)
    loaded[0]['binding']['state'] = 99
    assert values[0]['binding']['state'] == 0


@pytest.mark.parametrize('change', ['count', 'stage', 'target_bytes'])
def test_parent_fit_reader_rejects_truncated_wrong_split_or_mutated_arrays(common, monkeypatch, change):
    from wirelessjev.hierarchical_radio import study
    parent_inputs(common); d.prepare_training(); values = parent_examples('validation', 'sparse')
    if change == 'count': values.pop()
    if change == 'stage': values[0]['binding']['stage'] = 'test'
    if change == 'target_bytes': values[0]['target']['center_dbm'][0] += 7
    monkeypatch.setattr(study, 'examples', lambda *a, **kw: values)
    with pytest.raises(RuntimeError): d.training_examples('validation', 'sparse')


@pytest.mark.parametrize('held,stale', [(False, False), (True, False), (False, True)])
def test_fresh_groups_have_exact_membership_units_and_stale_state0_anchor(common, monkeypatch, held, stale):
    fresh_stage(common, monkeypatch)
    values = d.examples('test', 'sparse', held, stale)
    assert len(values) == 3 and len(values[0]['a']['context']) == 96
    assert len(values[0]['query']['q']) == (96 if held else 192)
    assert [e['binding']['stale_requested'] for e in values] == ([False, True, True] if stale else [False]*3)
    for e in values:
        assert e['binding']['case_id'].startswith('nested-test-00-')
        assert e['binding']['query_features_use_actual_target_topK_cutoff'] is False
    receipt = common.read(common.OUT/'data/groups'/f'{d.group_name("test","sparse",held,stale)}.json')
    assert d.verify_group(receipt) == receipt
    assert d.group_reference('test', 'sparse', held, stale)['sha256'] == common.digest(common.OUT/'data/groups'/f'{d.group_name("test","sparse",held,stale)}.json')
    values[0]['binding']['state'] = 999; values[0]['a'].pop('known')
    again = d.examples('test', 'sparse', held, stale)
    assert again[0]['binding']['state'] == 0 and 'known' in again[0]['a']
    with pytest.raises(ValueError): again[0]['a']['context'].setflags(write=True)


def test_cached_fresh_group_cannot_bypass_current_authorization(common, monkeypatch):
    fresh_stage(common, monkeypatch); d.examples('test')
    monkeypatch.setattr(common, 'authorize', lambda *a, **kw: {'authorized': False})
    with pytest.raises(RuntimeError): d.examples('test')


@pytest.mark.parametrize('corrupt', ['truncated', 'reordered', 'methods_source', 'array_bytes', 'extra_state', 'extra_array'])
def test_rehashed_fresh_group_still_requires_exact_binding_and_arrays(common, monkeypatch, corrupt):
    fresh_stage(common, monkeypatch); d.examples('test')
    path = common.OUT/'data/groups/test-sparse.json'; receipt = common.read(path)
    if corrupt == 'truncated': receipt['bindings'].pop()
    if corrupt == 'reordered': receipt['bindings'].reverse()
    if corrupt == 'methods_source': receipt['source_receipt_sha256'] = 'z'*64
    if corrupt == 'extra_state': receipt['bindings'][1]['state'] = 0
    if corrupt in ('array_bytes', 'extra_array'):
        p = common.safe_path(receipt['arrays']['path']); arrays = common.arrays(p)
        if corrupt == 'array_bytes': arrays['a__context'][0, 0, 0] += 8
        else: arrays['unbound_private_label'] = np.zeros(3)
        with p.open('wb') as f: np.savez_compressed(f, **arrays)
        receipt['arrays'] = common.reference(p)
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError): d.examples('test')


@pytest.mark.parametrize('stage,regime,held,stale', [('training','sparse',False,False), ('test','wrong',False,False), ('test','full',1,False), ('test','full',False,0)])
def test_unknown_group_schema_is_rejected(stage, regime, held, stale):
    with pytest.raises(ValueError): d.group_name(stage, regime, held, stale)


def test_empty_or_extra_role_examples_cannot_be_packed():
    with pytest.raises(ValueError): d.pack_examples([])
    value = old.make_example(analytic_scene(), calibration())
    value['query']['private_label'] = np.zeros(192)
    with pytest.raises(ValueError): d.pack_examples([value])
