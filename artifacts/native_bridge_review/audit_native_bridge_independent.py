"""Independent sealed-native audit: no project imports, models or simulations.

Only JSON/NPZ/Git-archive reads and arithmetic are permitted. The default
output is outside the scientific evidence tree. Run only after all9 cases seal.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tarfile

import numpy as np

ROOT = Path('/workspace/SDTH-submission-PreceptualAI')
OUT = ROOT / 'artifacts/native_bridge'
ADAPTIVE = ROOT / 'artifacts/adaptive_radio'
PREVIOUS = ROOT / 'artifacts/radiojepa_spectral/verification/manifest.json'
PREVIOUS_SHA = '53f329a2ce82d6498116fa098b1231ad31c5826be4435b4610c713701bcf598c'
DEFAULT_OUTPUT = Path('/workspace/.cloud-setup/native-bridge-independent.json')
OBSERVATIONS = ('positions', 'value', 'heard', 'bound', 'scanned')
WORLD_ARRAYS = ('sites', 'snr', 'rssi_dl', 'domain', 'avail', 'video_mbps',
                'bulk_mbps', 'noise_db', 'channel', 'plan', 'pos', 'tx_dl')
HORIZONS = np.array((0., 1., 2., 3., 4., 5., 10., 20., 30.))
CARRIERS = np.array((2.4e9, 5.2e9, 6.5e9))
INPUT_HASHES = {}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def sha(path):
    path = Path(path)
    if path not in INPUT_HASHES:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                digest.update(block)
        INPUT_HASHES[path] = digest.hexdigest()
    return INPUT_HASHES[path]


def safe(name, root=ROOT):
    name, root = Path(name), Path(root)
    check(not name.is_absolute() and '..' not in name.parts, 'Unsafe evidence reference')
    path = root / name
    check(path.resolve().is_relative_to(root.resolve()), 'Evidence escaped its namespace')
    for item in (path, *path.parents):
        if item == root:
            break
        check(not item.is_symlink(), 'Redirected evidence member: ' + str(item))
    check(path.is_file(), 'Missing evidence: ' + str(path))
    return path


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def reference(value, expected):
    check(set(value) == {'path', 'sha256'}, 'Reference schema changed')
    path = safe(value['path'])
    check(path == expected and sha(path) == value['sha256'], 'Canonical reference or bytes differ')
    return path


def arrays(path):
    sha(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.dtype.str, value.shape)).encode() + value.tobytes()).hexdigest()


def equal(actual, expected, name='value'):
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and set(actual) == set(expected), name + ' membership differs')
        for key, value in expected.items():
            equal(actual[key], value, name + '.' + str(key))
    elif isinstance(expected, (list, tuple)):
        check(isinstance(actual, (list, tuple)) and len(actual) == len(expected), name + ' length differs')
        for index, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, name + '[' + str(index) + ']')
    elif isinstance(expected, bool):
        check(type(actual) is bool and actual is expected, name + ' boolean differs')
    elif isinstance(expected, (float, int, np.number)):
        check(not isinstance(actual, bool) and np.isfinite(actual)
              and np.isclose(actual, expected, rtol=1e-12, atol=1e-12), name + ' numeric value differs')
    else:
        check(actual == expected, name + ' differs')


def exact_array(actual, expected, name):
    check(actual.dtype == expected.dtype and actual.shape == expected.shape
          and np.array_equal(actual, expected), name + ' dtype/shape/bytes differ')


def verify_source_and_seal():
    manifest_path = OUT / 'verification/manifest.json'
    check(manifest_path.is_file(), 'Wait for the original9-case native study to seal')
    before_manifest = sha(manifest_path)
    manifest = read(manifest_path)
    registration = read(OUT / 'registration/source.json')
    publication = read(OUT / 'registration/git.json')
    source = registration['source_sha256']
    check(len(source) == 512, 'Expected exact512 native registered sources')
    check(manifest['source_sha256'] == source and manifest['production_qualified'] is False,
          'Native seal source or qualification differs')
    check(manifest['source_receipt_sha256'] == sha(OUT / 'registration/source.json'),
          'Native source receipt differs from seal')
    check(publication['commit'] == publication['remote_observed_commit']
          == 'fa0a3b3070f37ec9398f0124c1d807f474a68e88', 'Native publication differs')
    check(publication['source_receipt_sha256'] == sha(OUT / 'registration/source.json'),
          'Published native source receipt differs')
    capsule = OUT / 'registration/source'
    check({str(p.relative_to(capsule)) for p in capsule.rglob('*') if p.is_file()} == set(source),
          'Native source capsule membership differs')
    for name, digest in source.items():
        check(sha(safe(name)) == digest and sha(safe(name, capsule)) == digest,
              'Native current/capsule source differs: ' + name)
    for name, digest in registration['preflight_sha256'].items():
        check(sha(safe(name)) == digest, 'Native preflight proof differs')
    preflight = read(OUT / 'preflight/tests.json')
    check(preflight['exit_code'] == 0 and preflight['source_sha256'] == source,
          'Native passing preflight does not bind source')
    visible = {str(p.relative_to(ROOT)): sha(p) for p in OUT.rglob('*')
               if p.is_file() and p != manifest_path and not p.is_relative_to(capsule)}
    check(visible == manifest['evidence_sha256'], 'Native sealed bytes or membership differ')
    for path in OUT.rglob('*'):
        check(not path.is_symlink(), 'Native evidence contains a redirected member')
    members = dict(source) | registration['preflight_sha256']
    members[str((OUT / 'registration/source.json').relative_to(ROOT))] = sha(OUT / 'registration/source.json')
    members.update({str((capsule / name).relative_to(ROOT)): digest for name, digest in source.items()})
    process = subprocess.Popen(['git', 'archive', '--format=tar', publication['commit'], *members],
                               cwd=ROOT, stdout=subprocess.PIPE)
    found = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for item in archive:
                if item.isdir():
                    continue
                check(item.isfile() and item.name in members and item.name not in found,
                      'Unexpected published native member')
                check(hashlib.sha256(archive.extractfile(item).read()).hexdigest() == members[item.name],
                      'Published native source bytes differ')
                found.add(item.name)
        check(process.wait() == 0 and found == set(members), 'Incomplete native Git publication')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()
    check(sha(PREVIOUS) == PREVIOUS_SHA, 'Preserved spectral manifest differs')
    previous = read(PREVIOUS)
    check(len(previous['source_sha256']) == 475 and len(previous['evidence_sha256']) == 5161,
          'Preserved spectral closure cardinality differs')
    for key in ('source_sha256', 'evidence_sha256', 'documents_sha256'):
        for name, digest in previous[key].items():
            check(sha(safe(name)) == digest, 'Preserved spectral bytes differ: ' + name)
    check(not (ROOT / 'artifacts/radiojepa/rt_data/confirmation').exists()
          and not (ROOT / 'artifacts/radiojepa/rt_data/confirmation.json').exists(),
          'Original SERVICE confirmation was opened')
    return {'source_files': 512, 'evidence_files': len(visible),
            'source_commit': publication['commit'], 'manifest_sha256': before_manifest,
            'previous_sources': 475, 'previous_evidence': 5161}


def verify_weights():
    weights = read(OUT / 'weights.json')
    fits = {}
    check(weights['seed'] == 0 and weights['native_DEV_selection_used'] is False,
          'Native outcomes selected a member')
    check(weights['source_receipt_sha256'] == sha(OUT / 'registration/source.json'),
          'Native weight binding source differs')
    for cell in ('joint', 'regression', 'rf_erased', 'pure_geometry'):
        for seed in range(3):
            name = cell + '-seed' + str(seed)
            fit_path = ADAPTIVE / 'fits' / (name + '.json')
            checkpoint = ADAPTIVE / 'checkpoints' / (name + '.pt')
            check(weights['all12_fit_receipts_sha256'][str(fit_path.relative_to(ROOT))] == sha(fit_path),
                  'An adaptive fit binding differs')
            check(weights['all12_checkpoints_sha256'][str(checkpoint.relative_to(ROOT))] == sha(checkpoint),
                  'An adaptive checkpoint binding differs')
            fit = read(fit_path)
            check(fit['completed'] is True and fit['cell'] == cell and type(fit['seed']) is int
                  and fit['seed'] == seed and fit['run_id'] == name
                  and fit['epochs'] == 60 and fit['updates'] == 480,
                  'An adaptive fit member/budget differs')
            check(fit['settings'] == {'cell': cell, 'width': 64, 'em_steps': 3}, 'Adaptive settings differ')
            check(fit['identity']['source_receipt_sha256'] == sha(ADAPTIVE / 'registration/source.json'),
                  'Adaptive fit source differs')
            check(fit['checkpoint_sha256'] == sha(checkpoint), 'Adaptive fit checkpoint differs')
            history = fit['history']
            check(len(history) == 60 and [h['epoch'] for h in history] == list(range(60)),
                  'Adaptive complete epoch sequence differs')
            for entry in history:
                check(len(entry['updates']) == 8 and np.isfinite(entry['validation_mean_layout_delta_rmse_db']),
                      'Adaptive optimization history incomplete')
                for update in entry['updates']:
                    check(isinstance(update, dict) and 'loss' in update and np.isfinite(list(update.values())).all(),
                          'Adaptive update is missing or nonfinite')
            score = [h['validation_mean_layout_delta_rmse_db'] for h in history]
            check(fit['selected_epoch'] == int(np.argmin(score))
                  and fit['validation_mean_layout_delta_rmse_db'] == min(score), 'Adaptive first-min selection differs')
            fits[cell, seed] = fit
    check(len(weights['all12_fit_receipts_sha256']) == len(weights['all12_checkpoints_sha256']) == 12,
          'Exact twelve-fit membership differs')
    selection = read(ADAPTIVE / 'selection.json')
    means = {}
    for cell in ('joint', 'regression'):
        values = [fits[cell, seed]['validation_mean_layout_delta_rmse_db'] for seed in range(3)]
        equal(selection['candidates'][cell]['seed_delta_rmse_db'], values, 'VAL branch member scores')
        equal(selection['candidates'][cell]['mean_delta_rmse_db'], float(np.mean(values)), 'VAL branch mean')
        means[cell] = float(np.mean(values))
    selected = min(('joint', 'regression'), key=lambda name: means[name])
    check(weights['selected_cell'] == selection['selected_cell'] == selected, 'Native branch is not VAL-selected')
    check(weights['selection_sha256'] == sha(ADAPTIVE / 'selection.json'), 'VAL selection bytes differ')
    check(weights['checkpoint_path'] == str((ADAPTIVE / 'checkpoints' / (selected + '-seed0.pt')).relative_to(ROOT)),
          'Wrong selected native checkpoint path')
    check(weights['checkpoint_sha256'] == sha(safe(weights['checkpoint_path'])), 'Wrong selected checkpoint digest')
    return weights


def boxes():
    return np.array([((a, y - .6, 0.), (b, y + .6, 3.))
                     for y in np.arange(11) * 6. + 10.
                     for a, b in ((10., 38.), (42., 78.), (82., 110.))])


def geometry_features(rx, ap, carrier, noise, tx):
    """Independent slab intersection arithmetic for the public separated boxes."""
    distance = np.linalg.norm(rx - ap, axis=1)
    count = np.zeros(len(rx)); fraction = np.zeros(len(rx))
    direction = rx - ap
    for low, high in boxes():
        entry = np.zeros(len(rx)); exit = np.ones(len(rx)); valid = np.ones(len(rx), bool)
        for axis in range(3):
            parallel = np.abs(direction[:, axis]) < 1e-14
            valid &= ~(parallel & ((ap[:, axis] < low[axis]) | (ap[:, axis] > high[axis])))
            denominator = np.where(parallel, 1., direction[:, axis])
            a = (low[axis] - ap[:, axis]) / denominator
            b = (high[axis] - ap[:, axis]) / denominator
            entry = np.maximum(entry, np.where(parallel, -np.inf, np.minimum(a, b)))
            exit = np.minimum(exit, np.where(parallel, np.inf, np.maximum(a, b)))
        valid &= exit > entry + 1e-12
        count += valid
        fraction += np.where(valid, exit - entry, 0.)
    q = np.column_stack((rx / (120., 80., 6.), ap / (120., 80., 6.), np.log10(distance) / 2,
                         np.log(carrier / 2.4e9), -np.ones(len(rx)), count / 8., np.clip(fraction, 0., 1.),
                         np.zeros((len(rx), 3)), noise / 100., tx / 30.)).astype('f4')
    fspl = tx - 20. * (math.log10(4. * math.pi / 299792458.) + np.log10(distance) + np.log10(carrier))
    return q, fspl


def context_from_history(history, length, world):
    scanned = history['scanned'][:length]; heard = history['heard'][:length]
    times, robots = np.nonzero(scanned)
    check(len(times) > 0, 'No real delivered scan evidence')
    groups = []
    for site in range(len(world['sites'])):
        for band in (0, 1):
            link = site * 3 + band
            order = sorted(range(len(times)), key=lambda i: (-int(times[i]), not bool(heard[times[i], robots[i], link]), int(robots[i])))
            groups.append([(int(times[i]), int(robots[i]), site, band) for i in order[:64]])
    selected = []
    for ordinal in range(64):
        for group in groups:
            if ordinal < len(group):
                selected.append(group[ordinal])
            if len(selected) == 64:
                break
        if len(selected) == 64:
            break
    index = np.asarray(selected, dtype='i8')
    t, robot, site, band = index.T
    check((t < length).all() and np.isin(band, (0, 1)).all(), 'Future or6.5GHz context')
    rx = np.column_stack((history['positions'][t, robot], np.ones(len(t))))
    ap = np.column_stack((world['sites'][site], np.full(len(t), 4.)))
    q, fspl = geometry_features(rx, ap, CARRIERS[band], world['noise_db'][band], world['tx_dl'][band])
    link = site * 3 + band
    h = heard[t, robot, link]; value = history['value'][t, robot, link]; bound = history['bound'][t, robot, link]
    context = np.column_stack((np.where(h, (value - fspl) / 30., 0.),
                               np.where(~h, (bound - fspl) / 30., 0.), h, ~h, q)).astype('f4')
    return index, context


def snr_from_rssi(power, world):
    denominator = np.power(10., np.asarray(world['noise_db'], dtype='f8') / 10.)
    return (power - world['tx_dl'] + 15. - 10. * np.log10(denominator)).astype('f4')


def verify_observations(history, warm, events, world):
    for name in OBSERVATIONS:
        exact_array(history[name][:60], warm[name], 'Common warm ' + name)
    length = len(history['value'])
    check(length == 71 and history['value'].shape == (71, 100, 48), 'Final strictly-past history axes differ')
    check(history['heard'].dtype == history['scanned'].dtype == bool, 'Observation masks are not Boolean')
    check(not (history['heard'] & ~history['scanned'][..., None]).any(), 'Invented unscheduled heard values')
    expected_scanned = np.zeros_like(history['scanned'])
    expected_scanned[:60] = warm['scanned']
    final = {}
    for event in events:
        measured, complete, delivered = (event[k] for k in ('measurement_slot', 'complete_slot', 'delivery_slot'))
        robot = event['robot']
        check(all(type(v) is int for v in (measured, complete, delivered, robot))
              and 0 <= measured < complete <= delivered < 1200 and 0 <= robot < 100,
              'Report measurement/completion/delivery clock order differs')
        check(event['first_usable_decision'] == delivered // 100 + 1, 'Backdated report usability')
        t = 60 + delivered // 100
        if t < length:
            final[t, robot] = event
            expected_scanned[t, robot] = True
    exact_array(history['scanned'], expected_scanned, 'Observed scan/event membership')
    for (t, robot), event in final.items():
        payload = history['value'][t, robot].tobytes() + history['heard'][t, robot].tobytes()
        check(hashlib.sha256(payload).hexdigest() == event['payload_sha256'], 'Delivered RF payload differs')
        check(int(history['heard'][t, robot].sum()) == event['heard'], 'Delivered heard count differs')
        point = world['pos'][min(event['measurement_slot'] // 20, len(world['pos']) - 1), robot]
        exact_array(history['positions'][t, robot], point, 'Measurement-time receiver position')
    for t, robot in zip(*np.nonzero(history['scanned'])):
        h, value, bound = (history[name][t, robot] for name in ('heard', 'value', 'bound'))
        check(h.sum() <= 8 and np.isfinite(value[h]).all() and (value[h] >= -90.).all(), 'Invalid top-eight report')
        check((value[~h] == 0).all(), 'Unknown RF centers are not inert')
        expected = np.float32(value[h].min() if h.sum() == 8 else -90.)
        check(np.array_equal(bound, np.full(48, expected, dtype=bound.dtype)), 'Censor bound is not the genuine top-eight/sensitivity bound')
    return len(events)


def verify_jobs(arm, config, parity):
    jobs = arm['forecast_jobs']; ready = []; previous = 0.
    check(len(jobs) == 12, 'Expected twelve native forecast jobs')
    profiles = config['forecast_profile_seconds']
    for decision, job in enumerate(jobs):
        kind = 'cold' if decision == 0 else 'warm'
        check(job['decision'] == decision and job['history_cutoff_exclusive'] == 60 + decision
              and job['kind'] == kind, 'Forecast time/cutoff/regime differs')
        equal(job['profile_seconds'], profiles[kind], 'Registered compute profile')
        equal(job['shared_extra_seconds'], config['shared_extra_seconds'], 'Registered transport delay')
        a, b = job['actual_prediction_seconds'], job['actual_issue_seconds']
        check(np.isfinite((a, b)).all() and min(a, b) >= 0, 'Invalid measured compute')
        compute = max(a, b); charged = max(profiles[kind], compute)
        service = charged + config['shared_extra_seconds']
        previous = max(float(decision), previous) + service
        equal(job['actual_compute_seconds'], compute, 'Actual compute accounting')
        equal(job['charged_compute_seconds'], charged, 'Paid compute accounting')
        equal(job['charged_seconds'], service, 'Paid FIFO service accounting')
        check(job['ready_slot'] == math.ceil(previous * 100 - 1e-10), 'Incorrect completion slot')
        check(job['runtime_overrun'] == (compute > profiles[kind])
              and job['predictor_runtime_overrun'] == (a > profiles[kind]), 'Overrun flags differ')
        check(job['completion_uses_actual_elapsed'] is True and job['offline_recorded_schedule'] is False,
              'Non-live forecast schedule')
        check(job['forecast_shape'] == [100, 9, 16, 3] and job['forecast_dtype'] == 'float32', 'Forecast output axes differ')
        ready.append(job['ready_slot'])
    last = -1; previous_slot = -1
    for record in arm['forecast_consumption']:
        slot, index = record['slot'], record['forecast_decision']
        check(slot >= previous_slot and last < index < 12 and slot >= ready[index], 'Forecast consumed before completion')
        check(index == int(np.searchsorted(ready, slot, side='right')) - 1, 'Consumer bypassed latest completed FIFO job')
        check(record['ready_slot'] == ready[index], 'Consumed completion identity differs')
        equal(record['age_seconds'], slot / 100. - index, 'Consumed forecast age')
        last, previous_slot = index, slot
    for item in arm['planner_and_admission']:
        equal(item['deadline_seconds'], .1, 'Planner deadline')
        check(np.isfinite(item['elapsed_seconds']) and item['elapsed_seconds'] >= 0
              and item['late_discarded'] == (item['elapsed_seconds'] > .1), 'Planner runtime/discard differs')
        if item['kind'] == 'admission' and item['late_discarded']:
            check(item['permitted'] == 0, 'Late admission permitted a move')
    return sum(job['runtime_overrun'] for job in jobs)


def reduce_metrics(arm):
    native = arm['native']; detail = native['packet_detail']
    check(set(detail) >= {'misses', 'sent', 'gap', 'max_gap', 'cumulative'}, 'Missing raw packet counters')
    vector = {}
    for name in ('misses', 'sent', 'gap', 'max_gap'):
        value = np.asarray(detail[name])
        check(value.shape == (100,) and value.dtype.kind in 'iu' and (value >= 0).all(), 'Invalid per-robot counter')
        vector[name] = value
    check((vector['misses'] <= vector['sent']).all() and (vector['gap'] <= vector['max_gap']).all()
          and (vector['max_gap'] <= 1200).all(), 'Invalid packet or gap bounds')
    check(detail['stop_slot'] == 1200 and detail['resume_slot'] == 0, 'Truncated/resumed native case')
    cumulative = detail['cumulative']
    check(len(cumulative) == 12, 'Missing per-second raw packet trajectory')
    previous = {k: np.zeros(100, dtype='i8') for k in ('misses', 'sent', 'max_gap')}
    curves = []
    for second, row in enumerate(cumulative, 1):
        check(row['slot_end'] == second * 100, 'Cumulative time axis differs')
        current = {k: np.asarray(row[k]) for k in ('misses', 'sent', 'gap', 'max_gap')}
        for key, value in current.items():
            check(value.shape == (100,) and value.dtype.kind in 'iu' and (value >= 0).all(), 'Malformed cumulative counters')
            if key != 'gap':
                check((value >= previous[key]).all(), 'A cumulative count decreased')
        check((current['misses'] <= current['sent']).all() and (current['gap'] <= current['max_gap']).all()
              and (current['max_gap'] <= second * 100).all(), 'Invalid cumulative event bounds')
        curves.append({'second': second, 'misses': int(current['misses'].sum()), 'sent': int(current['sent'].sum())})
        previous = current
    for key in vector:
        exact_array(vector[key], previous[key], 'Terminal/cumulative ' + key)
    fraction = vector['misses'] / np.maximum(vector['sent'], 1)
    derived = {'control_miss': float(vector['misses'].sum() / max(1, vector['sent'].sum())),
               'control_miss_worst_decile': float(np.sort(fraction)[-10:].mean()),
               'control_gap_p99_ms': float(np.percentile(vector['max_gap'], 99) * 10)}
    for key, value in derived.items():
        equal(native[key], value, 'Native ' + key)
    equal(native['control_misses'], int(vector['misses'].sum()), 'Native fleet missed total')
    equal(native['control_sent'], int(vector['sent'].sum()), 'Native fleet sent total')
    keys = ('control_miss', 'control_miss_worst_decile', 'control_gap_p99_ms', 'video_on_time',
            'bulk_mbps', 'airtime_domain_sum', 'roams', 'safety_stop_fraction', 'control_misses', 'control_sent')
    jobs, planning = arm['forecast_jobs'], arm['planner_and_admission']
    metrics = {name: native[name] for name in keys} | derived | {
        'maximum_silence_ms': int(vector['max_gap'].max() * 10),
        'paid_forecast_compute_seconds': float(sum(j['charged_compute_seconds'] for j in jobs)),
        'forecast_service_seconds': float(sum(j['charged_seconds'] for j in jobs)),
        'actual_forecast_compute_seconds': float(sum(j['actual_compute_seconds'] for j in jobs)),
        'planner_and_admission_seconds': float(sum(j['elapsed_seconds'] for j in planning)),
        'forecast_jobs': len(jobs), 'forecast_jobs_consumed': len(arm['forecast_consumption']),
        'forecast_runtime_overruns': sum(j['runtime_overrun'] for j in jobs),
        'late_planner_or_admission_discards': sum(j['late_discarded'] for j in planning)}
    return metrics, curves


def verify_world(seed):
    folder = OUT / 'worlds' / str(seed)
    receipt = read(folder / 'receipt.json')
    check(receipt['seed'] == seed and receipt['robots'] == 100 and receipt['duration_seconds'] == 12
          and receipt['warm_seconds'] == 60, 'Native world identity differs')
    check(receipt['source_receipt_sha256'] == sha(OUT / 'registration/source.json')
          and receipt['protocol_sha256'] == sha(OUT / 'protocol.json'), 'Native world source differs')
    world = arrays(reference(receipt['world'], folder / 'world.npz'))
    warm = arrays(reference(receipt['warm'], folder / 'warm.npz'))
    check(set(world) == set(WORLD_ARRAYS) and set(warm) == set(OBSERVATIONS), 'World/warm array membership differs')
    check({name: array_sha(value) for name, value in world.items()} == receipt['world_array_sha256'], 'Native physical arrays differ')
    check({name: array_sha(value) for name, value in warm.items()} == receipt['warm_array_sha256'], 'Native warm arrays differ')
    check(world['sites'].shape == (16, 2) and world['plan'].shape == (12, 100, 9, 2)
          and world['pos'].shape == (60, 100, 2), 'Native geometry/clock dimensions differ')
    check(np.array_equal(world['plan'][:, :, 0], world['pos'][::5]), 'Current plan and post-warm pose clocks differ')
    check(not warm['scanned'][np.arange(60) % 5 != 0].any(), 'Warm scan cadence changed')
    return receipt, world, warm


def verify_case(spec, parity, protocol, weights):
    folder = OUT / 'cases' / spec['case_id']
    receipt = read(folder / 'receipt.json')
    check(receipt['case'] == spec and receipt['parity'] is parity and receipt['native_simulations'] == 2,
          'Native case identity/count differs')
    check(receipt['source_receipt_sha256'] == sha(OUT / 'registration/source.json')
          and receipt['protocol_sha256'] == sha(OUT / 'protocol.json')
          and receipt['weights_sha256'] == sha(OUT / 'weights.json'), 'Native case source/weights differ')
    check(receipt['simulation_only'] is True and receipt['production_qualified'] is False
          and receipt['original_SERVICE_holdouts_opened'] is False
          and receipt['geometry_carrier_scale_transfer_unqualified'] is True, 'Native case scope differs')
    for name, filename in (('raw_result', 'native_result.json.gz'), ('candidate_inputs_forecasts', 'candidate_inputs_forecasts.npz'), ('attempt', 'attempt.json')):
        reference(receipt[name], folder / filename)
    attempt = read(folder / 'attempt.json')
    check(type(attempt['created_unix_ns']) is int and attempt['created_unix_ns'] > 0
          and attempt['case'] == spec and attempt['parity'] is parity
          and attempt['maximum_native_simulations'] == 2 and attempt['exclusive_attempt'] is True
          and attempt['unreceipted_attempt_must_not_be_reexecuted'] is True,
          'Missing or mismatched durable exclusive native attempt')
    result = json.loads(gzip.decompress((folder / 'native_result.json.gz').read_bytes()))
    config = dict(protocol['config']) | {'duration_seconds': 12, 'handover_seconds': spec['handover_seconds']}
    if parity:
        config['forecast_profile_seconds'] = protocol['parity_profile_seconds']
    equal(result['config'], config, 'Registered native config')
    equal(attempt['config'], config, 'Durable native attempt config')
    for key in ('source_receipt_sha256', 'protocol_sha256'):
        check(result[key] == attempt[key] == receipt[key], 'Raw/attempt/case identity differs')
    world_receipt, world, warm = verify_world(spec['seed'])
    check(result['world_receipt_sha256'] == receipt['world_receipt_sha256']
          == sha(OUT / 'worlds' / str(spec['seed']) / 'receipt.json'), 'Raw physical-world binding differs')
    check(result['case_id'] == spec['case_id'] and result['partition'] == 'development'
          and result['robots'] == 100 and result['native_packet_simulations'] == 2
          and result['simulation_only'] is True and result['production_qualified'] is False
          and result['original_SERVICE_holdouts_opened'] is False and result['default_controller_changed'] is False,
          'Raw native study scope differs')
    check(result['policy']['roam_cost_s'] == spec['handover_seconds']
          and result['policy']['scan_period_s'] == 5. and result['policy']['gate'] is True
          and result['policy']['cadence_scan'] is True and result['policy']['cadence_roam'] is True
          and result['policy']['online_map'] is False and result['policy']['dual_radio'] is False,
          'Raw native policy differs')
    check(set(result['arms']) == {'radiomap', 'candidate'}, 'Native arm membership differs')
    metrics, trajectories, overruns = {}, {}, {}
    for name, arm in result['arms'].items():
        native = arm['native']
        check(native['scene'] == spec['seed'] and native['duration_s'] == 12
              and native['collision_lambda'] == 0. and native['interference'] is False,
              'Native physical episode differs')
        overruns[name] = verify_jobs(arm, config, parity)
        metrics[name], trajectories[name] = reduce_metrics(arm)
        check(arm['native']['packet_detail']['delivered_report_events'] == arm['telemetry']['events'],
              'Native and observable delivery ledgers differ')
        for event in arm['telemetry']['events']:
            check(0 <= event['measurement_slot'] < event['complete_slot'] <= event['delivery_slot'] < 1200
                  and event['first_usable_decision'] == event['delivery_slot'] // 100 + 1,
                  'Native delivery ledger backdates an RF observation')
    equal(receipt['metrics'], metrics, 'Archived native metrics')
    delta = {key: metrics['candidate'][key] - metrics['radiomap'][key] for key in metrics['candidate']}
    equal(receipt['paired_delta_candidate_minus_Map'], delta, 'Archived paired deltas')
    archive = arrays(folder / 'candidate_inputs_forecasts.npz')
    history = {name: archive['last_history_' + name] for name in OBSERVATIONS}
    event_count = verify_observations(history, warm, result['arms']['candidate']['telemetry']['events'], world)
    check(len(receipt['candidate_jobs']) == 12, 'Native predictor input job membership differs')
    for i, job in enumerate(result['arms']['candidate']['forecast_jobs']):
        check(receipt['candidate_jobs'][i]['history_length'] == 60 + i, 'A predictor saw a future prefix')
        exact_array(archive[f'job{i}_plan'], world['plan'][i], 'Public post-warm plan')
        power = archive[f'job{i}_mean_dbm']
        check(power.shape == (100, 9, 16, 3) and np.isfinite(power).all(), 'Invalid raw RSSI forecast')
        snr = snr_from_rssi(power, world)
        check(hashlib.sha256(snr.tobytes()).hexdigest() == job['forecast_sha256'], 'RSSI/SNR gauge conversion differs')
        if parity:
            exact_array(snr, archive[f'job{i}_source_snr'], 'Map bitwise RSSI round trip')
            check(hashlib.sha256(archive[f'job{i}_source_std'].tobytes()).hexdigest() == job['forecast_std_sha256'], 'Map std round trip differs')
        else:
            selected, context = context_from_history(history, 60 + i, world)
            exact_array(archive[f'job{i}_selected_rows'], selected, 'Causal selected RF rows')
            np.testing.assert_allclose(archive[f'job{i}_context'], context, rtol=0., atol=2e-6)
            check(np.array_equal(archive[f'job{i}_known'], np.ones(len(selected), bool))
                  and np.array_equal(archive[f'job{i}_scheduled'], np.ones(len(selected), bool)), 'RF report masks differ')
            check(hashlib.sha256(archive[f'job{i}_std_db'].astype('f4').tobytes()).hexdigest() == job['forecast_std_sha256'], 'Native learned spread differs')
    if parity:
        a, b = result['arms']['radiomap'], result['arms']['candidate']
        check(encoded(a['native']) == encoded(b['native']), 'Map wiring changes native packet/service bodies')
        for key in ('forecast_sha256', 'forecast_std_sha256', 'ready_slot'):
            check([j[key] for j in a['forecast_jobs']] == [j[key] for j in b['forecast_jobs']], 'Map wiring changes forecast/completion identity')
        check(receipt['wiring_proof']['passes'] is True
              and all(receipt['wiring_proof']['checks'].values()), 'Original Map wiring proof failed')
    excess = [{'second': a['second'], 'cumulative_excess_misses': b['misses'] - a['misses']}
              for a, b in zip(trajectories['radiomap'], trajectories['candidate'])]
    return {'case': spec, 'parity': parity, 'metrics': metrics, 'paired_delta_candidate_minus_Map': delta,
            'forecast_overruns': overruns, 'causal_reports_reconciled': event_count,
            'descriptive_cumulative_excess_misses': excess,
            'parity_no_runtime_overruns': not any(overruns.values()) if parity else None}


def main(destination):
    closure = verify_source_and_seal()
    protocol = read(OUT / 'protocol.json')
    check(protocol['duration_seconds'] == 12 and protocol['robots'] == 100
          and len(protocol['seeds']) == 4 and len(protocol['cases']) == 8
          and protocol['maximum_native_simulations'] == 18, 'Finite native study design differs')
    check(protocol['actual_carriers_hz'] == CARRIERS.tolist()
          and protocol['config']['scan_duration_seconds'] == .09
          and protocol['config']['scan_period_seconds'] == 5., 'Carrier/scan-cost design differs')
    weights = verify_weights()
    check({p.name for p in (OUT / 'cases').iterdir() if p.is_dir()}
          == {s['case_id'] for s in [protocol['parity_case'], *protocol['cases']]}, 'Native case namespace membership differs')
    check({p.name for p in (OUT / 'worlds').iterdir() if p.is_dir()} == {str(s) for s in protocol['seeds']},
          'Native world namespace membership differs')
    summary = read(OUT / 'summary.json')
    rows = [verify_case(protocol['parity_case'], True, protocol, weights)]
    rows += [verify_case(spec, False, protocol, weights) for spec in protocol['cases']]
    expected_summary = {}
    for cost in (.03, 4.):
        selected = [row for row in rows if not row['parity'] and row['case']['handover_seconds'] == cost]
        check(len(selected) == 4 and {row['case']['seed'] for row in selected} == set(protocol['seeds']), 'Four paired episodes per cost required')
        keys = selected[0]['paired_delta_candidate_minus_Map']
        delta = {key: np.array([r['paired_delta_candidate_minus_Map'][key] for r in selected]) for key in keys}
        expected_summary[str(cost)] = {'episodes': 4, 'seeds': [r['case']['seed'] for r in selected],
            'paired_mean_candidate_minus_Map': {key: float(value.mean()) for key, value in delta.items()},
            'control_miss_harm_count': int((delta['control_miss'] > 0).sum()),
            'worst_decile_harm_count': int((delta['control_miss_worst_decile'] > 0).sum()),
            'max_silence_harm_count': int((delta['maximum_silence_ms'] > 0).sum())}
    equal(summary['separate_handover_results'], expected_summary, 'Paired native summary')
    check(summary['native_simulations'] == 18 and summary['production_qualified'] is False
          and summary['default_changed'] is False and summary['original_SERVICE_holdouts_opened'] is False,
          'Native summary scope differs')
    check(set(summary['cases']) == {r['case']['case_id'] for r in rows}, 'Native summary case membership differs')
    for row in rows:
        check(summary['cases'][row['case']['case_id']] == sha(OUT / 'cases' / row['case']['case_id'] / 'receipt.json'),
              'Native summary case digest differs')
    # Final independent consistency snapshot: nothing in the sealed native or
    # preserved evidence namespaces may have changed during this arithmetic.
    for path, expected in INPUT_HASHES.items():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                digest.update(block)
        check(digest.hexdigest() == expected, 'Input changed during independent audit: ' + str(path))
    native_manifest = read(OUT / 'verification/manifest.json')
    current_members = {str(p.relative_to(ROOT)) for p in OUT.rglob('*')
                       if p.is_file() and p != OUT / 'verification/manifest.json'
                       and not p.is_relative_to(OUT / 'registration/source')}
    check(current_members == set(native_manifest['evidence_sha256']), 'Native membership changed during audit')
    check(sha(OUT / 'verification/manifest.json') == closure['manifest_sha256'], 'Native seal changed')
    record = {'passes': True, 'scope': 'independent passive sealed native DEV arithmetic and provenance audit',
        'closure': closure, 'cases': rows, 'paired_summaries': expected_summary,
        'selected_cell': weights['selected_cell'], 'checkpoint_sha256': weights['checkpoint_sha256'],
        'source_current_and_capsules_unchanged': True, 'frozen_input_snapshots_unchanged': True,
        'model_constructors_forwards_fits_RT_or_native_simulations': 0,
        'original_SERVICE_confirmation_opened': False, 'production_qualified': False,
        'limits': ['Twelve seconds, four episodes per handover cost; no SLA, significance or long-rollout guarantee.',
                   'The90ms scan every5s retains the deterministic deadline-loss floor (about1%; finite12s boundary effects apply).',
                   '6.5GHz is native extrapolation; it is not the RT6.0GHz endpoint.',
                   'Video/bulk/airtime/stop metrics are hash-bound native counters, not independently replayed packet traces.',
                   'Planner wall time is checked against the fixed deadline; this is a synchronous simulator, not hardware qualification.'],
        'documented_parity_no_overrun_condition_satisfied': rows[0]['parity_no_runtime_overruns'],
        'audit_script_sha256': sha(Path(__file__)),
        'input_files_sha256': {str(p.relative_to(ROOT)): digest for p, digest in INPUT_HASHES.items() if p.is_relative_to(ROOT)}}
    destination = Path(destination)
    check(not destination.resolve().is_relative_to(OUT.resolve()), 'Never write a reviewer result inside sealed evidence')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded(record))
    print(json.dumps({'passes': True, 'case_count': len(rows), 'closure': closure,
                      'record': str(destination), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                      'parity_no_runtime_overruns': rows[0]['parity_no_runtime_overruns']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    main(parser.parse_args().output)
