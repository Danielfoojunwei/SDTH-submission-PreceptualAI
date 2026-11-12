"""Fresh physical cases and immutable reuse of the preceding TRAIN/VAL split.

Only reported NPZ values and public CAD vertices become model inputs. The
frozen ray solver and report semantics are reused through a new publication
guard; preceding TEST/DEV outcomes never enter fitting or selection.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
import re
import time

import numpy as np

from ..hierarchical_radio import data as public
from ..hierarchical_radio import simulation as sim
from ..spectral_scene.data import _read_cad_box

COUNTS = {'test': 24, 'ap_transfer': 4}
STATES = (0, 1, 2)
TOTAL_CASES = 84
_GROUPS = {}
ROLE_KEYS = {
    'a': {'context', 'known', 'scheduled', 'ap', 'rx_xyz', 'lower', 'upper', 'floor', 'fspl', 'offset'},
    'b': {'context', 'known', 'scheduled', 'ap', 'rx_xyz', 'lower', 'upper', 'floor', 'fspl', 'offset'},
    'query': {'q', 'ap', 'rx_xyz', 'basis', 'fspl', 'offset'},
    'target': {'center_dbm', 'lower_dbm', 'upper_dbm', 'heard', 'censored', 'censor_upper_dbm', 'detection_threshold_dbm'},
}


def _common():
    from . import common
    return common


def _stream(stage, index, stream):
    return np.random.default_rng(np.random.SeedSequence([12112026, list(COUNTS).index(stage), index, stream]))


def geometry_spec(stage, index):
    if stage not in COUNTS or type(index) is not int or not 0 <= index < COUNTS[stage]:
        raise ValueError('Unknown fresh nested-radio geometry')
    rng = _stream(stage, index, 11)
    width, depth = float(rng.uniform(22., 43.)), float(rng.uniform(27., 55.))
    ceiling = float(rng.uniform(4.6, 6.4))
    height = float(rng.uniform(2.3, min(3.8, ceiling - .7)))
    positions = ((-.36, -.36), (.36, -.36), (-.36, .36), (.36, .36))
    if stage == 'ap_transfer':
        positions = tuple((x, y) for y in (-.36, -.12, .12, .36) for x in (-.36, .36))
    aps = tuple((float((x + rng.uniform(-.012, .012)) * width),
                 float((y + rng.uniform(-.012, .012)) * depth),
                 float(ceiling - rng.uniform(.5, .8))) for x, y in positions)
    rx = tuple((x * width, y * depth, float(1.04 + rng.uniform(0., .04)))
               for y in sim.Y_FRACTIONS for x in sim.X_FRACTIONS)
    return sim.rt.GeometrySpec(f'nested-{stage}-{index:02d}',
        ('rack_aisles', 'partitioned_room')[int(rng.integers(2))], width, depth,
        ceiling, height, 'wood', aps, rx, 1212000 + 1000 * list(COUNTS).index(stage) + index)


def case_spec(stage, index, state):
    geometry = geometry_spec(stage, index)
    if type(state) is not int or state not in STATES:
        raise ValueError('Unknown nested-radio material state')
    base = tuple(_stream(stage, index, 23).choice(('wood', 'brick', 'concrete'), size=2, replace=False).tolist())
    materials = base if state == 0 else ('metal', base[1]) if state == 1 else (base[0], 'metal')
    power = float(_stream(stage, index, 37).choice((-20., -10., 0., 10.)))
    return sim.CaseSpec(stage, index, state, geometry, materials, power)


def geometry_registry(root=None):
    c = _common(); root = c.ROOT if root is None else Path(root)
    prior, references = [], {}
    for name in ('artifacts/radiojepa/rt_registry.json', 'artifacts/radiojepa_spectral/field_registry.json'):
        path = root / name; value = json.loads(path.read_bytes()); references[name] = sim.digest(path)
        prior.extend(sim.geometry_fingerprint(g) for rows in value['stages'].values() for g in rows)
    name = 'artifacts/adaptive_radio/geometry_registry.json'; path = root / name
    value = json.loads(path.read_bytes()); references[name] = sim.digest(path)
    prior.extend(sim.geometry_fingerprint(row['public_geometry']) for rows in value['stages'].values() for row in rows)
    name = 'artifacts/hierarchical_radio/registry.json'; path = root / name
    value = json.loads(path.read_bytes()); references[name] = sim.digest(path)
    prior.extend(sim.geometry_fingerprint(row['public_geometry']) for rows in value['stages'].values() for row in rows)
    if len(prior) != 240:
        raise RuntimeError('Bind all 240 preceding public geometry records')
    stages, fresh = {}, []
    for stage, count in COUNTS.items():
        stages[stage] = []
        for index in range(count):
            geometry = geometry_spec(stage, index); fingerprint = sim.geometry_fingerprint(geometry)
            fresh.append(fingerprint)
            stages[stage].append({'layout_id': geometry.scene_id,
                'public_geometry': sim.public_geometry(geometry), 'public_geometry_sha256': fingerprint,
                'APs': len(geometry.ap_positions_m),
                'cases': [case_spec(stage, index, state).to_dict() for state in STATES]})
    if len(set(fresh)) != sum(COUNTS.values()) or set(fresh) & set(prior):
        raise RuntimeError('Fresh layouts overlap a preceding or another fresh physical geometry')
    return {'schema': 1, 'counts': dict(COUNTS), 'states_per_layout': 3, 'case_count': 3 * sum(COUNTS.values()),
        'bands_hz': list(sim.BANDS_HZ), 'source_bands': [0, 1], 'held_band': 2,
        'context_poses': list(sim.CONTEXT_POSES), 'query_poses': list(sim.QUERY_POSES),
        'context_tokens_full': 192, 'context_tokens_sparse': 96, 'top_k_per_band': sim.TOP_K,
        'receiver_measurement_sigma_db': sim.MEASUREMENT_SIGMA_DB,
        'all_new_geometries_disjoint_from_prior240': True,
        'prior_registry_sha256': references, 'prior_geometry_records': len(prior),
        'geometry_material_and_power_use_independent_rng_streams': True,
        'material_states_share_public_CAD_TX_solver_seed_and_receiver_noise_draws': True,
        'static_counterfactuals_not_temporal_events': True, 'stages': stages}


def _case_path(spec):
    return _common().case_path(spec)


def _contained_case(folder):
    c = _common(); folder = Path(folder)
    if (c.OUT.is_symlink() or not folder.resolve().is_relative_to(c.OUT.resolve())
            or not folder.resolve().is_relative_to(c.ROOT.resolve())):
        raise RuntimeError('Fresh physical cases must stay inside their own output folder')
    for ancestor in (folder, *folder.parents):
        if ancestor.is_symlink():
            raise RuntimeError('Symlinked fresh physical case paths are forbidden')
        if ancestor == c.OUT:
            break
    return folder


def stage_path(stage):
    if stage not in COUNTS:
        raise ValueError('Unknown physical evaluation stage')
    return _common().OUT / 'rt_data' / f'{stage}.json'


def _permission(spec, folder):
    c = _common()
    if not isinstance(spec, sim.CaseSpec) or spec != case_spec(spec.stage, spec.index, spec.state):
        raise ValueError('Only exact fresh registered cases may be traced')
    permission = c.authorize(spec, folder=folder)
    if (not isinstance(permission, Mapping) or permission.get('authorized') is not True
            or permission.get('case_id') != spec.case_id
            or any(not isinstance(permission.get(key), str) or not re.fullmatch('[0-9a-f]{64}', permission[key])
                   for key in ('source_receipt_sha256', 'registry_sha256'))
            or not isinstance(permission.get('case_path'), str)):
        raise RuntimeError('Fresh RT publication/selection authorization refused')
    _contained_case(folder); expected = Path(permission['case_path'])
    if (not expected.is_absolute() or folder.is_symlink() or folder.resolve() != expected.resolve()
            or folder.resolve() != _case_path(spec).resolve()):
        raise RuntimeError('Only the authorized canonical fresh case path is admitted')
    return dict(permission)


def trace_case(spec, folder=None):
    """One new solve, or passive reuse; the attempt always precedes the solver."""
    c = _common()
    if not isinstance(spec, sim.CaseSpec) or spec != case_spec(spec.stage, spec.index, spec.state):
        raise ValueError('Only exact fresh registered cases may be traced')
    folder = _case_path(spec) if folder is None else Path(folder)
    permission = _permission(spec, folder)
    if (folder / 'receipt.json').exists():
        return sim.validate_case_receipt(spec, folder,
            source_sha256=permission['source_receipt_sha256'], registry_sha256=permission['registry_sha256'])
    if folder.exists():
        raise RuntimeError('Partial fresh RT attempt exists; replacement is forbidden')
    c.write(folder / 'attempt.json', {'case_id': spec.case_id, 'created_unix_ns': time.time_ns(), **permission})
    result = sim._trace_case(spec, folder)
    if _permission(spec, folder) != permission:
        raise RuntimeError('Fresh RT source/selection authorization changed during the trace')
    result.update(source_receipt_sha256=permission['source_receipt_sha256'], registry_sha256=permission['registry_sha256'])
    c.write(folder / 'receipt.json', result)
    return sim.validate_case_receipt(spec, folder, source_sha256=permission['source_receipt_sha256'],
        registry_sha256=permission['registry_sha256'], reconstruct_reports=True)


def collect(stage):
    c = _common(); stage_path(stage); c.check_sources(); refs = {}
    for index in range(COUNTS[stage]):
        for state in STATES:
            spec = case_spec(stage, index, state); folder = _case_path(spec)
            trace_case(spec, folder); refs[spec.case_id] = c.reference(folder / 'receipt.json')
            print('RT', spec.case_id, 'complete', flush=True)
    value = {'schema': 1, 'stage': stage, 'complete': True, 'case_receipts': refs,
        'source_receipt_sha256': c.source_receipt_sha256(), 'registry_sha256': c.registry_sha256(),
        'layouts': COUNTS[stage], 'cases': 3 * COUNTS[stage], 'fresh_after_prior_hierarchy_TEST': True,
        'static_counterfactuals_not_temporal_events': True}
    path = stage_path(stage)
    if path.exists():
        if c.read(path) != value:
            raise RuntimeError('Completed fresh physical stage identity changed')
    else:
        c.write(path, value)
    return value


def verify_stage(stage, *, reconstruct_reports=False):
    """Verify immutable receipt membership; private channels only for raw audit."""
    c = _common(); value = c.read(stage_path(stage))
    if (value.get('stage') != stage or value.get('complete') is not True
            or value.get('source_receipt_sha256') != c.source_receipt_sha256()
            or value.get('registry_sha256') != c.registry_sha256()
            or value.get('layouts') != COUNTS[stage] or value.get('cases') != 3 * COUNTS[stage]):
        raise RuntimeError('Fresh physical stage binding changed')
    specs = [case_spec(stage, i, s) for i in range(COUNTS[stage]) for s in STATES]
    if set(value['case_receipts']) != {s.case_id for s in specs}:
        raise RuntimeError('Fresh physical stage must retain every declared state/layout')
    for spec in specs:
        folder = _case_path(spec); path = folder / 'receipt.json'
        if c.reference(path) != value['case_receipts'][spec.case_id]:
            raise RuntimeError('Bound fresh case receipt changed')
        receipt = c.read(path)
        if (receipt.get('case') != json.loads(sim.encoded(spec.to_dict()))
                or receipt.get('case_id') != spec.case_id or receipt.get('layout_id') != spec.layout_id
                or receipt.get('source_receipt_sha256') != c.source_receipt_sha256()
                or receipt.get('registry_sha256') != c.registry_sha256()):
            raise RuntimeError('Fresh case source or exact identity changed')
        if reconstruct_reports:
            sim.validate_case_receipt(spec, folder, source_sha256=c.source_receipt_sha256(),
                registry_sha256=c.registry_sha256(), reconstruct_reports=True)
    return value


def _source_name(path):
    c = _common(); path = Path(path).resolve()
    if not path.is_relative_to(c.ROOT.resolve()):
        raise ValueError('Public source must stay inside the repository')
    return str(path.relative_to(c.ROOT))


def load_scene(folder, *, expected_spec=None):
    """Fresh public-only reader: no XML, CIR, gain or material tensor read."""
    c = _common(); folder = _contained_case(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError('Actual immutable public scene folder required')
    receipt_path = folder / 'receipt.json'
    if receipt_path.is_symlink():
        raise RuntimeError('A fresh receipt cannot be a symlink')
    receipt = c.read(receipt_path)
    raw = receipt.get('case', {}); spec = case_spec(raw.get('stage'), raw.get('index'), raw.get('state'))
    if (expected_spec is not None and expected_spec != spec or folder.resolve() != _case_path(spec).resolve()
            or receipt.get('case') != json.loads(sim.encoded(spec.to_dict()))
            or receipt.get('case_id') != spec.case_id or receipt.get('layout_id') != spec.layout_id
            or receipt.get('public_geometry_sha256') != spec.geometry_sha256
            or receipt.get('public_geometry') != json.loads(sim.encoded(sim.public_geometry(spec.geometry)))
            or receipt.get('known_tx_power_dbm') != spec.tx_power_dbm
            or receipt.get('source_receipt_sha256') != c.source_receipt_sha256()
            or receipt.get('registry_sha256') != c.registry_sha256()
            or receipt.get('real_device_measurements') is not False
            or receipt.get('unresolved_paths_are_censored') is not False):
        raise RuntimeError('Fresh public scene identity or observation scope changed')
    sources = {_source_name(receipt_path): sim.digest(receipt_path)}
    path = folder / 'public_views.npz'
    if path.is_symlink() or sim.digest(path) != receipt['files_sha256'].get('public_views.npz'):
        raise RuntimeError('Public reported observations changed')
    sources[_source_name(path)] = sim.digest(path)
    mesh_names = sorted(name for name in receipt['files_sha256'] if name.startswith('scene/meshes/') and name.endswith('.ply'))
    if set(mesh_names) != {x for x in sim.expected_case_files(spec) if x.startswith('scene/meshes/')}:
        raise RuntimeError('Exact public CAD mesh membership required')
    boxes = []
    for name in mesh_names:
        relative = Path(name); mesh = folder / relative
        if (relative.is_absolute() or '..' in relative.parts or mesh.is_symlink()
                or sim.digest(mesh) != receipt['files_sha256'][name]):
            raise RuntimeError('Public CAD vertex source changed')
        boxes.append(_read_cad_box(mesh)); sources[_source_name(mesh)] = sim.digest(mesh)
    names = ('rssi_dbm', 'heard', 'censored', 'censor_upper_dbm', 'quantization_interval_dbm',
             'reported_detection_threshold_dbm', 'noise_floor_dbm', 'ap_positions_m', 'rx_positions_m',
             'bbox_dims_m', 'carrier_hz')
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in names}
    if (not np.allclose(values['ap_positions_m'], spec.geometry.ap_positions_m, rtol=0, atol=2e-5)
            or not np.allclose(values['rx_positions_m'], spec.geometry.rx_positions_m, rtol=0, atol=2e-5)
            or not np.array_equal(values['bbox_dims_m'],
                (spec.geometry.width_m, spec.geometry.depth_m, spec.geometry.ceiling_m))):
        raise RuntimeError('Public numeric geometry changed')
    return public.PublicScene(spec.stage, spec.index, spec.state, spec.layout_id, spec.case_id,
        values['ap_positions_m'], values['rx_positions_m'], values['bbox_dims_m'], np.asarray(boxes),
        values['carrier_hz'], spec.tx_power_dbm, values['noise_floor_dbm'], values['rssi_dbm'], values['heard'],
        values['censored'], values['censor_upper_dbm'], values['quantization_interval_dbm'],
        values['reported_detection_threshold_dbm'], sources)


def parent_input_references():
    """Passive paths only: no previous TEST/DEV report or target is read."""
    c = _common(); base = c.ROOT / 'artifacts/hierarchical_radio'
    prepared = c.read(base / 'data/prepared.json')
    expected_groups = {f'{s}-{r}' for s in ('training', 'validation') for r in ('full', 'sparse')}
    if (set(prepared['groups']) != expected_groups or set(prepared['stages']) != {'training', 'validation'}
            or prepared['target_band_indices'] != [0, 1]
            or prepared['calibration'] != c.reference(base / 'data/calibration.json')):
        raise RuntimeError('Preceding fitting split or calibration identity changed')
    groups, arrays = {}, {}
    for name in sorted(expected_groups):
        path = base / 'data/groups' / f'{name}.json'; ref = c.reference(path)
        if prepared['groups'][name] != ref:
            raise RuntimeError('Preceding TRAIN/VAL group differs from prepared split')
        group = c.read(path); expected_path = base / 'data/groups' / f'{name}.npz'
        if group['arrays'] != c.reference(expected_path):
            raise RuntimeError('Preceding TRAIN/VAL array archive changed')
        groups[name], arrays[name] = ref, group['arrays']
    stages = {s: c.reference(base / 'rt_data' / f'{s}.json') for s in ('training', 'validation')}
    if stages != prepared['stages']:
        raise RuntimeError('Preceding TRAIN/VAL physical receipts changed')
    return {'prepared': c.reference(base / 'data/prepared.json'), 'calibration': prepared['calibration'],
            'groups': groups, 'arrays': arrays, 'stages': stages}


def _prepared_value():
    c = _common()
    return {'schema': 1, 'source_receipt_sha256': c.source_receipt_sha256(),
        'registry_sha256': c.registry_sha256(), 'protocol_sha256': c.protocol_sha256(),
        'parent_dependency_identity': c.verify_parent_data(), 'parent_inputs': parent_input_references(),
        'training_cases': 72, 'validation_cases': 18, 'training_layouts': 24, 'validation_layouts': 6,
        'target_band_indices': [0, 1], 'reused_parent_TEST_or_DEV_for_fitting': False,
        'new_RT_or_calibration_fit_required_for_TRAIN_VAL': False}


def prepare_training():
    c = _common(); c.check_sources(); value = _prepared_value(); path = c.OUT / 'data/prepared.json'
    if path.exists():
        if c.read(path) != value:
            raise RuntimeError('Immutable reused fitting data identity changed')
    else:
        c.write(path, value)
    return value


def verify_prepared():
    c = _common(); value = c.read(c.OUT / 'data/prepared.json')
    if value != _prepared_value():
        raise RuntimeError('Prepared TRAIN/VAL data source/reference identity changed')
    return value


def training_examples(stage, regime='full'):
    if stage not in ('training', 'validation') or regime not in ('full', 'sparse'):
        raise ValueError('Only the immutable preceding TRAIN/VAL source bands may enter fitting')
    c = _common(); c.check_sources(); verify_prepared()
    from ..hierarchical_radio import study as parent
    # This existing reader verifies exact input-array hashes and fitting split
    # membership; the allowlist above prevents its TEST/DEV pathways being used.
    values = parent.examples(stage, regime)
    expected_count = 3 * (24 if stage == 'training' else 6)
    if len(values) != expected_count:
        raise RuntimeError('Every immutable parent fitting layout/state is required')
    for index, example in enumerate(values):
        spec = sim.case_spec(stage, index // 3, index % 3)
        expected = {'stage': stage, 'index': index // 3, 'state': index % 3, 'condition': index % 3,
            'layout_id': spec.layout_id, 'case_id': spec.case_id, 'context_case_id': spec.case_id,
            'regime': regime, 'held_band': False, 'stale_requested': False}
        if any(example['binding'].get(key) != value for key, value in expected.items()):
            raise RuntimeError('Only the exact immutable parent TRAIN/VAL source-band groups may enter fitting')
    # Verify and freeze an owned copy, without exposing the frozen parent's
    # mutable in-memory group cache to callers of this new study.
    copies = unpack_examples(pack_examples(values), [e['binding'] for e in values])
    return _copy_examples(copies)


def calibration():
    c = _common(); verify_prepared()
    ref = c.read(c.OUT / 'data/prepared.json')['parent_inputs']['calibration']
    return c.read(c.safe_path(ref['path']))['calibration']


def group_name(stage, regime, held_band=False, stale=False):
    if (stage not in COUNTS or regime not in ('full', 'sparse')
            or type(held_band) is not bool or type(stale) is not bool):
        raise ValueError('Only registered fresh evaluation groups are admitted')
    return stage + '-' + regime + ('-held6GHz' if held_band else '') + ('-stale' if stale else '')


def group_reference(stage, regime='sparse', held_band=False, stale=False):
    c = _common(); name = group_name(stage, regime, held_band, stale)
    return c.reference(c.OUT / 'data/groups' / f'{name}.json')


def pack_examples(values):
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError('Nonempty canonical public examples required')
    if any(set(e) != {'a', 'b', 'query', 'target', 'ap_xyz', 'binding'}
           or any(set(e[role]) != fields for role, fields in ROLE_KEYS.items()) for e in values):
        raise ValueError('Only canonical reported model/target roles may be cached')
    result = {'ap_xyz': np.stack([e['ap_xyz'] for e in values])}
    for role in ('a', 'b', 'query', 'target'):
        for key in values[0][role]:
            result[role + '__' + key] = np.stack([e[role][key] for e in values])
    return result


def unpack_examples(values, bindings):
    expected = {'ap_xyz'} | {role + '__' + key for role, fields in ROLE_KEYS.items() for key in fields}
    if (not bindings or set(values) != expected
            or any(v.ndim == 0 or len(v) != len(bindings) for v in values.values())):
        raise RuntimeError('Exact cached example count and array axes required')
    result = []
    for i, binding in enumerate(bindings):
        e = {'binding': deepcopy(binding), 'ap_xyz': public.immutable(values['ap_xyz'][i])}
        for role in ('a', 'b', 'query', 'target'):
            e[role] = {key.split('__', 1)[1]: public.immutable(value[i]) for key, value in values.items() if key.startswith(role + '__')}
        for role in ('a', 'b', 'query'):
            if {k: public.array_digest(v) for k, v in e[role].items()} != binding['model_input_sha256'][role]:
                raise RuntimeError('Cached reported model input array changed')
        if (public.array_digest(e['ap_xyz']) != binding['model_input_sha256']['ap_xyz']
                or {k: public.array_digest(v) for k, v in e['target'].items()} != binding['target_sha256']):
            raise RuntimeError('Cached geometry or target array changed')
        result.append(e)
    return result


def _copy_examples(values):
    return [{'ap_xyz': e['ap_xyz'], 'binding': deepcopy(e['binding']),
             **{role: dict(e[role]) for role in ('a', 'b', 'query', 'target')}} for e in values]


def verify_group(receipt, name=None):
    c = _common(); expected_name = group_name(receipt['stage'], receipt['regime'], receipt['held_band'], receipt['stale'])
    identity = {'source_receipt_sha256': c.source_receipt_sha256(), 'registry_sha256': c.registry_sha256(),
        'prepared_sha256': c.digest(c.OUT / 'data/prepared.json'), 'selection_sha256': c.digest(c.OUT / 'selection.json'),
        'stage_receipt_sha256': c.digest(stage_path(receipt['stage']))}
    if (name is not None and expected_name != name or any(receipt.get(k) != v for k, v in identity.items())
            or receipt['arrays'] != c.reference(c.OUT / 'data/groups' / f'{expected_name}.npz')):
        raise RuntimeError('Fresh example group source or canonical arrays changed')
    stage = receipt['stage']; bindings = receipt['bindings']
    if len(bindings) != 3 * COUNTS[stage]:
        raise RuntimeError('Every fresh geometry/state binding is required')
    for index, binding in enumerate(bindings):
        spec = case_spec(stage, index // 3, index % 3)
        context = case_spec(stage, index // 3, 0) if receipt['stale'] else spec
        expected = {'stage': stage, 'index': index // 3, 'state': index % 3, 'condition': index % 3,
            'layout_id': spec.layout_id, 'case_id': spec.case_id, 'context_case_id': context.case_id,
            'regime': receipt['regime'], 'held_band': receipt['held_band'],
            'stale_requested': bool(receipt['stale'] and spec.state != 0)}
        if any(binding.get(k) != v for k, v in expected.items()):
            raise RuntimeError('Exact fresh group layout/state order required')
    return receipt


def examples(stage, regime='sparse', held_band=False, stale=False):
    c = _common(); name = group_name(stage, regime, held_band, stale); c.check_sources()
    # Direct readers are also closed until the fixed fit/selection gate admits
    # new evaluation cases, even if somebody copied a data folder beforehand.
    _permission(case_spec(stage, 0, 0), _case_path(case_spec(stage, 0, 0)))
    verify_prepared(); verify_stage(stage)
    path = c.OUT / 'data/groups' / f'{name}.json'
    if path.exists():
        receipt = verify_group(c.read(path), name); identity = c.digest(path)
        if identity not in _GROUPS:
            _GROUPS[identity] = unpack_examples(c.arrays(c.safe_path(receipt['arrays']['path'])), receipt['bindings'])
        return _copy_examples(_GROUPS[identity])
    scenes = [load_scene(_case_path(case_spec(stage, i, s)), expected_spec=case_spec(stage, i, s))
              for i in range(COUNTS[stage]) for s in STATES]
    cal = calibration(); values = []
    for i, scene in enumerate(scenes):
        context = scenes[(i // 3) * 3] if stale else scene
        values.append(public.make_example(scene, cal, regime=regime, held_band=held_band, context_scene=context))
    ref = c.npz(c.OUT / 'data/groups' / f'{name}.npz', pack_examples(values))
    receipt = {'schema': 1, 'stage': stage, 'regime': regime, 'held_band': held_band, 'stale': stale,
        'source_receipt_sha256': c.source_receipt_sha256(), 'registry_sha256': c.registry_sha256(),
        'prepared_sha256': c.digest(c.OUT / 'data/prepared.json'), 'selection_sha256': c.digest(c.OUT / 'selection.json'),
        'stage_receipt_sha256': c.digest(stage_path(stage)), 'bindings': [e['binding'] for e in values],
        'arrays': ref, 'reported_only': True, 'parent_TEST_or_DEV_reused': False}
    c.write(path, receipt); _GROUPS[c.digest(path)] = values
    return _copy_examples(values)
