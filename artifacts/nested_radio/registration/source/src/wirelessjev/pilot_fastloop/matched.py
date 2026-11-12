"""Offline, matched-public-query accuracy of the two frozen forecaster packages.

The native RadioMap/pilot_fast trace is fixed before outcomes. Its public cache
and private mean-field labels are separate files. No World is constructed here;
labels are opened only after the complete sequence of real model forwards.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np

from ..fleetwm.data import Geometry, Observations
from ..semantic_jepa import common as frozen_common
from ..twin.radio import AP

FAMILIES = ('radiomap', 'frozen_fleet_jepa')
STAGES = ('development', 'confirmation')
PUBLIC_KEYS = ('positions', 'value', 'heard', 'bound', 'scanned', 'plans')
CANONICALIZATION = 'Zero value where heard=False and bound where scanned=False; retain all visible values, positive pilots, masks, positions and public plans.'
PROOF_FIELDS = ('source_and_archive_pass', 'causality_pass', 'matched_inputs_pass',
                'prediction_integrity_pass')


def _common():
    return importlib.import_module('.common', __package__)


def _parts(key):
    if not isinstance(key, (list, tuple)) or len(key) != 3:
        raise ValueError('Malformed matched case key')
    stage, site, family = key
    if stage not in STAGES or type(site) is not int or site < 0 or family not in FAMILIES:
        raise ValueError('Unknown matched case')
    return stage, site, family


def validate_case_key(key, p=None):
    stage, site, family = _parts(key)
    p = _common().protocol() if p is None else p
    if site not in p[stage + '_sites'] or family not in p['families']:
        raise ValueError('Unregistered matched case')


def case_keys(stage, p=None):
    if stage not in STAGES:
        raise ValueError('Unknown matched stage')
    p = _common().protocol() if p is None else p
    return [(stage, site, family) for site in p[stage + '_sites'] for family in FAMILIES]


def case_name(key):
    stage, site, family = _parts(key)
    return f'{stage}-{site}-{family}'


def _reference_key(key):
    stage, site, _ = _parts(key)
    return (stage, site, 4., 'radiomap', 'pilot_fast')


def _validate_reference(key, p):
    if not isinstance(key, (list, tuple)) or len(key) != 5:
        raise ValueError('Malformed matched reference')
    stage, site, cost, family, variant = key
    validate_case_key((stage, site, family), p)
    if type(cost) not in (int, float) or cost != 4. or family != 'radiomap' or variant != 'pilot_fast':
        raise ValueError('Only native RadioMap pilot_fast supplies matched inputs')
    return (stage, site, 4., family, variant)


def _guard(key, replay, p):
    if type(replay) is not bool:
        raise ValueError('Replay must be a boolean')
    if replay and (key[0] != 'confirmation' or key[1] != p['confirmation_sites'][0]):
        raise ValueError('Matched replay requires the first confirmation site')


def _identities():
    c = _common()
    registered = c.check_sources()
    return {'source_sha256': registered['source_sha256'],
            'registration_sha256': c.digest(c.OUT / 'registration/source.json'),
            'protocol_sha256': c.digest(c.OUT / 'protocol.json'),
            'checkpoint_sha256': c.checkpoint_hashes()}


def _relative(path):
    return str(Path(path).relative_to(_common().ROOT))


def _path(name):
    c = _common()
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise RuntimeError('Unsafe matched artifact path')
    resolved = (c.ROOT / path).resolve()
    if not resolved.is_relative_to(c.ROOT.resolve()):
        raise RuntimeError('Matched artifact escapes repository')
    return resolved


def _input_paths(key):
    stage, site, *_ = key
    base = _common().OUT / 'matched/inputs' / f'{stage}-{site}'
    return tuple(Path(str(base) + suffix) for suffix in ('.public.npz', '.labels.npz', '.metadata.json'))


def archive_paths(key, replay=False):
    _parts(key)
    if type(replay) is not bool:
        raise ValueError('Replay must be a boolean')
    folder = _common().OUT / 'matched' / ('replays' if replay else 'cases')
    return tuple(folder / (case_name(key) + suffix) for suffix in ('.json.gz', '.receipt.json'))


def _prediction_path(key, replay=False):
    return _common().OUT / 'matched' / ('replay_predictions' if replay else 'predictions') / (case_name(key) + '.npz')


def _npz_bytes(arrays):
    """Deterministic, non-pickled NPZ bytes (including fixed ZIP metadata)."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            value = np.asarray(arrays[name])
            if value.dtype.hasobject:
                raise ValueError('Object arrays cannot cross the matched boundary')
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, value, allow_pickle=False)
            info = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return stream.getvalue()


def _array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _immutable(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _load_npz(path, keys):
    with np.load(path, allow_pickle=False) as values:
        if set(values.files) != set(keys):
            raise RuntimeError('Matched array keys changed')
        return {name: _immutable(values[name]) for name in keys}


def _geometry_record(geometry):
    if not isinstance(geometry, Geometry):
        raise TypeError('Matched input requires public Geometry')
    return {'sites': geometry.sites.tolist(), 'tx': geometry.tx.tolist(),
            'noise': geometry.noise.tolist(), 'aps': [asdict(ap) for ap in geometry.aps],
            'yard': geometry.yard}


def _geometry(value):
    if set(value) != {'sites', 'tx', 'noise', 'aps', 'yard'} or type(value['yard']) is not bool:
        raise RuntimeError('Malformed matched geometry')
    sites, tx, noise = (np.asarray(value[name], dtype=float) for name in ('sites', 'tx', 'noise'))
    if (sites.ndim != 2 or sites.shape[1] != 2 or not len(sites) or tx.shape != (3,)
            or noise.shape != (3,) or not all(np.isfinite(v).all() for v in (sites, tx, noise))
            or len(value['aps']) != len(sites)):
        raise RuntimeError('Malformed matched geometry arrays')
    aps = tuple(AP(**(row | {'allowed_channels': tuple(row['allowed_channels'])})) for row in value['aps'])
    if any(not math.isfinite(ap.x) or not math.isfinite(ap.y) for ap in aps):
        raise RuntimeError('Malformed public AP coordinates')
    return Geometry(*(_immutable(v) for v in (sites, tx, noise)), aps, value['yard'])


def _validate_public(arrays, geometry, p):
    duration, robots, sites = p['duration_seconds'], p['robots'], len(geometry.sites)
    shapes = {'positions': (60 + duration, robots, 2), 'value': (60 + duration, robots, sites * 3),
              'heard': (60 + duration, robots, sites * 3), 'bound': (60 + duration, robots, sites * 3),
              'scanned': (60 + duration, robots), 'plans': (duration, robots, 9, 2)}
    if set(arrays) != set(PUBLIC_KEYS) or any(arrays[k].shape != shape for k, shape in shapes.items()):
        raise RuntimeError('Matched public history or plan shape changed')
    if any(arrays[k].dtype != np.dtype(bool) for k in ('heard', 'scanned')):
        raise RuntimeError('Matched observation masks must be boolean')
    if any(not np.issubdtype(arrays[k].dtype, np.floating) or not np.isfinite(arrays[k]).all()
           for k in ('positions', 'value', 'bound', 'plans')):
        raise RuntimeError('Matched public numbers must be finite floats')
    if np.any(arrays['value'][~arrays['heard']] != 0) or np.any(arrays['bound'][~arrays['scanned']] != 0):
        raise RuntimeError('Unobserved RF buffer values leaked into public cache')


@dataclass(frozen=True)
class PublicInputs:
    geometry: Geometry
    history: Observations
    plans: np.ndarray
    receipt: dict
    metadata: dict


def save_inputs(native_key, collector):
    """Called only after completed native simulation, before its archive write."""
    c, p = _common(), _common().protocol()
    key = _validate_reference(native_key, p)
    # Permission precedes cache access, and identities precede any new evidence.
    if key[0] == 'confirmation':
        c.check_confirmation_permission(4.)
    identity = _identities()
    if set(collector) != {'history', 'plans', 'geometry', 'labels'} or not isinstance(collector['history'], Observations):
        raise TypeError('Expected the completed native public-history collector')
    history = collector['history']
    arrays = {name: np.asarray(getattr(history, name)) for name in PUBLIC_KEYS[:-1]}
    arrays['value'] = np.where(arrays['heard'], arrays['value'], 0)
    arrays['bound'] = np.where(arrays['scanned'][..., None], arrays['bound'], 0)
    arrays['plans'] = np.asarray(collector['plans'])
    geometry = _geometry(_geometry_record(collector['geometry']))
    _validate_public(arrays, geometry, p)
    truth = np.asarray(collector['labels'])
    _validate_labels(truth, p, geometry)
    public_path, labels_path, metadata_path = _input_paths(key)
    metadata = {'schema': 1, 'reference_case': list(key), **identity,
                'geometry': _geometry_record(geometry), 'public_keys': list(PUBLIC_KEYS),
                'canonicalization': CANONICALIZATION,
                'history_rule': 'Only rows strictly before 60+decision cross the model boundary.',
                'label_scope': 'Clean mean-field future SNR on the reference native realized trajectory; offline package accuracy only.',
                'shapes': {name: list(value.shape) for name, value in arrays.items()},
                'dtypes': {name: value.dtype.str for name, value in arrays.items()},
                'labels_shape': list(truth.shape), 'labels_dtype': truth.dtype.str}
    with c.lock(metadata_path.with_suffix('.lock')):
        paths = (public_path, labels_path, metadata_path)
        if any(path.exists() for path in paths):
            if not all(path.exists() for path in paths):
                raise RuntimeError('Partial matched input cache; refuse silent overwrite')
        if identity != _identities():
            raise RuntimeError('Source changed during matched cache capture')
        frozen_common._write_bytes_once(public_path, _npz_bytes(arrays))
        frozen_common._write_bytes_once(labels_path, _npz_bytes({'truth': truth}))
        c.write_once(metadata_path, metadata)
    receipt = {'schema': 1, 'reference_case': list(key)}
    for name, path in zip(('public', 'labels', 'metadata'), paths):
        receipt[name + '_path'] = _relative(path)
        receipt[name + '_sha256'] = c.digest(path)
    verify_inputs(receipt, reference_key=key)
    return receipt


def verify_inputs(receipt, reference_key=None):
    """Validate all identities; never deserialize/open the private label arrays."""
    c, p = _common(), _common().protocol()
    if not isinstance(receipt, dict) or set(receipt) != {'schema', 'reference_case', 'public_path', 'public_sha256', 'labels_path', 'labels_sha256', 'metadata_path', 'metadata_sha256'} or receipt['schema'] != 1:
        raise RuntimeError('Malformed matched input receipt')
    key = _validate_reference(receipt['reference_case'], p)
    if reference_key is not None and list(key) != list(reference_key):
        raise RuntimeError('Matched reference identity changed')
    paths = _input_paths(key)
    for name, path in zip(('public', 'labels', 'metadata'), paths):
        if _path(receipt[name + '_path']) != path.resolve() or c.digest(path) != receipt[name + '_sha256']:
            raise RuntimeError('Matched input file identity changed: ' + name)
    metadata = json.loads(paths[2].read_text())
    identity = _identities()
    if (metadata.get('schema') != 1 or metadata.get('reference_case') != list(key)
            or metadata.get('public_keys') != list(PUBLIC_KEYS) or metadata.get('canonicalization') != CANONICALIZATION
            or any(metadata.get(name) != value for name, value in identity.items())):
        raise RuntimeError('Matched input provenance changed')
    geometry = _geometry(metadata['geometry'])
    arrays = _load_npz(paths[0], PUBLIC_KEYS)
    _validate_public(arrays, geometry, p)
    if (metadata.get('shapes') != {name: list(value.shape) for name, value in arrays.items()}
            or metadata.get('dtypes') != {name: value.dtype.str for name, value in arrays.items()}):
        raise RuntimeError('Matched public array metadata changed')
    history = Observations(*(arrays[name] for name in PUBLIC_KEYS[:-1]))
    return PublicInputs(geometry, history, arrays['plans'], receipt, metadata)


def _validate_labels(truth, p, geometry):
    if (truth.shape != (p['duration_seconds'], p['robots'], 9, len(geometry.sites), 3)
            or not np.issubdtype(truth.dtype, np.floating) or not np.isfinite(truth).all()):
        raise RuntimeError('Malformed private matched labels')


def _load_labels(public, p):
    truth = _load_npz(_path(public.receipt['labels_path']), ('truth',))['truth']
    _validate_labels(truth, p, public.geometry)
    if public.metadata['labels_shape'] != list(truth.shape) or public.metadata['labels_dtype'] != truth.dtype.str:
        raise RuntimeError('Private label metadata changed')
    return truth


def _metrics(mean, truth):
    """Primary evaluates only 1..5-second changes, excluding the zero anchor."""
    if mean.shape != truth.shape or not np.isfinite(mean).all() or not np.isfinite(truth).all():
        raise RuntimeError('Malformed matched metric arrays')
    # float64 arithmetic prevents a float32 square/reduction from obscuring audit differences.
    prediction, target = mean.astype('f8'), truth.astype('f8')
    delta = (prediction - prediction[:, :, :1]) - (target - target[:, :, :1])
    predicted_ap, true_ap = prediction[:, :, :6].max(-1).mean(2), target[:, :, :6].max(-1).mean(2)
    chosen = predicted_ap.argmax(-1)
    regret = true_ap.max(-1) - np.take_along_axis(true_ap, chosen[..., None], -1)[..., 0]
    return {'short_route_delta_rmse_db': float(np.sqrt(np.mean(delta[:, :, 1:6] ** 2))),
            'signal_rmse_db': float(np.sqrt(np.mean((prediction - target) ** 2))),
            'all_route_delta_rmse_db': float(np.sqrt(np.mean(delta ** 2))),
            'ap_choice_regret_db': float(regret.mean())}


def _predict(public, family, *, predictor_factory):
    """Private fixture seam; production run_case always supplies the real Predictor."""
    predictor = predictor_factory(public.geometry, family)
    means, queries = [], []
    shape = (public.plans.shape[1], 9, len(public.geometry.sites), 3)
    for decision in range(len(public.plans)):
        before = time.perf_counter()
        # A copy of a strictly past prefix prevents future history and mutations
        # from reaching this or subsequent calls. Plans are public at source time.
        past = public.history.before(60 + decision)
        plan = np.array(public.plans[decision], copy=True)
        mean, std, kind, elapsed = predictor.step(past, plan)
        actual = time.perf_counter() - before
        mean, std = np.asarray(mean), np.asarray(std)
        if (mean.shape != shape or std.shape != shape or mean.dtype != np.dtype('f4')
                or std.dtype != np.dtype('f4') or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or np.any(std < 0)
                or kind not in ('cold', 'warm', 'refit') or not _number(elapsed)):
            raise RuntimeError('Malformed matched model output')
        means.append(mean.copy())
        queries.append({'decision': decision, 'history_cutoff_exclusive': 60 + decision,
                        'kind': kind, 'actual_prediction_seconds': float(elapsed),
                        'actual_issue_seconds': actual, 'forecast_sha256': _array_hash(mean),
                        'forecast_std_sha256': _array_hash(std)})
    return np.stack(means), queries


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _reference(key):
    from . import rollout
    reference_key = _reference_key(key)
    row = rollout.load_case(reference_key)
    if row is None:
        raise RuntimeError('Complete native reference case before matched evaluation')
    path = _common().OUT / 'cases' / (rollout.case_name(reference_key) + '.json.gz')
    return row, _common().digest(path)


def _check_row(row, public, mean):
    c, p = _common(), _common().protocol()
    key = _parts(row.get('case'))
    validate_case_key(key, p)
    _guard(key, row.get('replay'), p)
    expected = {'schema': 1, 'completed': True, 'case': list(key), 'stage': key[0], 'site': key[1],
                'family': key[2], 'offline_only': True, 'fresh_measured_deadline_claim': False,
                'new_fits': 0, 'training_seed': None, 'reference_case': list(_reference_key(key)),
                'hardware_qualified': False, 'production_qualified': False, 'activated': False}
    if any(row.get(name) != value or (type(value) is bool and type(row.get(name)) is not bool)
           for name, value in expected.items()) or any(row.get(name) is not True for name in PROOF_FIELDS):
        raise RuntimeError('Matched method, proof or qualification identity changed')
    if any(row.get(name) != value for name, value in _identities().items()):
        raise RuntimeError('Matched source/checkpoint identity changed')
    reference, reference_sha = _reference(key)
    if reference.get('matched_inputs') != public.receipt or row.get('reference_archive_sha256') != reference_sha:
        raise RuntimeError('Matched case no longer binds exact native reference')
    decisions = list(range(p['duration_seconds']))
    if (row.get('query_decisions') != decisions or row.get('history_cutoffs_exclusive') != [60 + d for d in decisions]
            or not isinstance(row.get('per_query'), list) or len(row['per_query']) != len(decisions)):
        raise RuntimeError('Matched query ledger changed')
    expected_shape = (p['duration_seconds'], p['robots'], 9, len(public.geometry.sites), 3)
    if mean.shape != expected_shape or mean.dtype != np.dtype('f4') or not np.isfinite(mean).all():
        raise RuntimeError('Malformed saved matched predictions')
    for d, query in enumerate(row['per_query']):
        if (type(query.get('decision')) is not int or query['decision'] != d
                or type(query.get('history_cutoff_exclusive')) is not int or query['history_cutoff_exclusive'] != 60 + d
                or query.get('kind') not in ('cold', 'warm', 'refit')
                or not all(_number(query.get(name)) for name in ('actual_prediction_seconds', 'actual_issue_seconds'))
                or query.get('forecast_sha256') != _array_hash(mean[d])
                or not _sha(query.get('forecast_std_sha256'))):
            raise RuntimeError('Matched query or raw prediction hash changed')
    if row.get('forecast_mean_sha256') != _array_hash(mean):
        raise RuntimeError('Full matched prediction hash changed')
    truth = _load_labels(public, p)
    if row.get('metrics') != _metrics(mean, truth):
        raise RuntimeError('Matched metrics differ from saved raw arrays')
    prediction_path = _prediction_path(key, row['replay'])
    if (_path(row.get('prediction_path', '')) != prediction_path.resolve()
            or row.get('prediction_file_sha256') != c.digest(prediction_path)):
        raise RuntimeError('Matched prediction file identity changed')
    return True


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_row(row):
    if not isinstance(row, dict):
        raise RuntimeError('Missing matched row')
    public = verify_inputs(row.get('matched_inputs'), reference_key=row.get('reference_case'))
    key = _parts(row.get('case'))
    path = _prediction_path(key, row.get('replay'))
    mean = _load_npz(path, ('mean',))['mean']
    return _check_row(row, public, mean)


def load_case(key, *, replay=False):
    c = _common()
    validate_case_key(key)
    _guard(key, replay, c.protocol())
    path, receipt_path = archive_paths(key, replay)
    prediction_path = _prediction_path(key, replay)
    present = [p.exists() for p in (path, receipt_path, prediction_path)]
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError('Partial matched archive; refuse silent rerun')
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('case') != list(key) or receipt.get('replay') is not replay
            or receipt.get('archive_sha256') != c.digest(path)):
        raise RuntimeError('Matched archive receipt changed')
    row = json.loads(gzip.decompress(path.read_bytes()))
    if row.get('case') != list(key) or row.get('replay') is not replay or receipt.get('row_sha256') != c.value_digest(row):
        raise RuntimeError('Matched archive body identity changed')
    validate_row(row)
    return row


def run_case(key, *, replay=False):
    """Real forwards on a registered native cache, strictly after permission."""
    c, p = _common(), _common().protocol()
    validate_case_key(key, p)
    key = tuple(key)
    _guard(key, replay, p)
    if key[0] == 'confirmation':
        c.check_confirmation_permission(4.)
    identity = _identities()  # Before reading any cache or constructing a model.
    path, receipt_path = archive_paths(key, replay)
    with c.lock(path.parent / ('.' + case_name(key) + '.lock')):
        existing = load_case(key, replay=replay)
        if existing is not None:
            return existing
        main = load_case(key) if replay else None
        if replay and main is None:
            raise RuntimeError('Complete main matched case before replay')
        reference, reference_sha = _reference(key)
        public = verify_inputs(reference['matched_inputs'], reference_key=_reference_key(key))
        import torch
        from ..semantic_jepa.runtime import Predictor
        torch.set_num_threads(1)
        mean, queries = _predict(public, key[2], predictor_factory=Predictor)
        # Crucial boundary: every forecast and std hash exists BEFORE labels load.
        truth = _load_labels(public, p)
        metrics = _metrics(mean, truth)
        if identity != _identities():
            raise RuntimeError('Source changed during matched forwards')
        prediction_path = _prediction_path(key, replay)
        frozen_common._write_bytes_once(prediction_path, _npz_bytes({'mean': mean}))
        row = {'schema': 1, 'completed': True, 'case': list(key), 'stage': key[0], 'site': key[1],
               'family': key[2], 'replay': replay, 'offline_only': True,
               'fresh_measured_deadline_claim': False, 'new_fits': 0, 'training_seed': None,
               **identity, 'reference_case': list(_reference_key(key)),
               'reference_archive_sha256': reference_sha, 'matched_inputs': public.receipt,
               'query_decisions': list(range(len(mean))), 'history_cutoffs_exclusive': [60 + d for d in range(len(mean))],
               'per_query': queries, 'prediction_path': _relative(prediction_path),
               'prediction_file_sha256': c.digest(prediction_path), 'forecast_mean_sha256': _array_hash(mean),
               'metrics': metrics, **{name: True for name in PROOF_FIELDS},
               'hardware_qualified': False, 'production_qualified': False, 'activated': False}
        _check_row(row, public, mean)
        if replay and compare_replay(main, row)['exact'] is not True:
            raise RuntimeError('Matched raw forward replay differs')
        c.write_once(path, row, compressed=True)
        c.write_once(receipt_path, {'case': list(key), 'replay': replay,
                     'archive_sha256': c.digest(path), 'row_sha256': c.value_digest(row)})
        return row


def compare_replay(original, replay):
    """Fresh timings are diagnostic; public inputs and actual outputs must match."""
    for row in (original, replay):
        validate_row(row)
    fields = ('case', 'source_sha256', 'registration_sha256', 'protocol_sha256', 'checkpoint_sha256',
              'reference_case', 'reference_archive_sha256', 'matched_inputs', 'query_decisions',
              'history_cutoffs_exclusive', 'forecast_mean_sha256', 'prediction_file_sha256', 'metrics')
    checks = {name: original.get(name) == replay.get(name) for name in fields}
    strip = lambda row: [{k: q[k] for k in ('decision', 'history_cutoff_exclusive', 'kind', 'forecast_sha256', 'forecast_std_sha256')}
                         for q in row['per_query']]
    checks['per_query_hashes'] = strip(original) == strip(replay)
    checks['replication_flags'] = original.get('replay') is False and replay.get('replay') is True
    return {'case': original['case'], 'exact': all(checks.values()), 'checks': checks,
            'fresh_measured_deadline_claim': False}
