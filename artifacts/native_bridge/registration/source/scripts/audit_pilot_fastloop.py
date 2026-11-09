"""Passive raw-evidence audit for the pilot/fast-loop study.

No World, Predictor, simulation, forward, fit, benchmark, or timing function is
called. --write creates a deterministic write-once receipt; plain invocation
recomputes it and validates an existing receipt if present.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
from threading import Event

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/pilot_fastloop'
HORIZONS = np.asarray([0., 1., 2., 3., 4., 5., 10., 20., 30.])
FAMILIES = ('radiomap', 'frozen_fleet_jepa')
VARIANTS = ('base', 'pilot', 'fast', 'pilot_fast')
ANALYSIS_WORKERS = 3  # Passive only; reserve one of four CPUs for report validation.


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def value_digest(value):
    """Canonical JSON hash without materializing an extra full trace string."""
    result = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False)
    for chunk in encoder.iterencode(value):
        result.update(chunk.encode())
    return result.hexdigest()


def equal(actual, expected, message):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and np.isfinite(actual).all()
        and np.allclose(actual, expected, rtol=0, atol=1e-10), message)


def committed(commit, name):
    return subprocess.check_output(['git', 'show', commit + ':' + name], cwd=ROOT)


def registration(c):
    frozen = c.check_sources()
    meta = read(OUT / 'registration/git.json')
    commit = meta['scientific_source_commit']
    require(isinstance(commit, str) and len(commit) == 40
        and meta['before_new_evaluations'] is True, 'Missing original pre-data Git binding')
    tree = subprocess.check_output(['git', 'rev-parse', commit + '^{tree}'], cwd=ROOT, text=True).strip()
    require(meta['scientific_source_tree'] == tree, 'Original source Git tree changed')
    require((OUT / 'registration/source.json').read_bytes() == committed(commit,
        'artifacts/pilot_fastloop/registration/source.json'), 'Registration differs from original source Git commit')
    for name, expected in frozen['source_sha256'].items():
        raw = committed(commit, name)
        require(digest(ROOT / name) == expected == hashlib.sha256(raw).hexdigest()
            and (OUT / 'registration/source' / name).read_bytes() == raw,
            'Current/capsule/original Git source differs: ' + name)
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit,
        'artifacts/pilot_fastloop'], cwd=ROOT, text=True).splitlines()
    prefixes = tuple('artifacts/pilot_fastloop/' + folder + '/' for folder in
        ('cases', 'replays', 'live_replications', 'matched'))
    require(not any(name.startswith(prefixes) for name in names), 'Original source commit already contains new evidence')
    require(frozen['new_evaluation_data_accessed'] is False and frozen['new_fits'] == 0,
        'Registration is not prospective/frozen')
    c.verify_fresh_sites()
    return frozen, meta


def pilot_check(row):
    """Decode transport bytes independently; audit positive-only source algebra."""
    reporting = row['pilot_reporting']
    n, duration, aps = row['robots'], row['duration_seconds'], row['access_points']
    wire = struct.Struct('<BBHHQQffB3f')
    retained, delivered, seen = {}, set(), set()
    previous = -1
    pilots = delivered_pilots = scans = delivered_scans = 0
    for index, event in enumerate(reporting['frame_events']):
        slot, robot, kind = event['slot'], event['robot'], event['kind']
        require(type(slot) is int and type(robot) is int and previous <= slot < duration * 100
            and 0 <= robot < n and (slot, robot) not in seen, 'Pilot frame clock/membership differs')
        previous = slot; seen.add((slot, robot))
        require(event['prepared_before_delivery_draw'] is True and type(event['delivered']) is bool
            and event['frame_bytes'] == 600 and event['reservation_bytes'] == 128,
            'Telemetry preparation or transport reservation differs')
        key = slot // 100, robot
        if kind == 'pilot':
            pilots += 1
            require(reporting['enabled'] is True, 'Disabled pilot arm sent a pilot')
            payload = bytes.fromhex(event['payload_hex'])
            require(len(payload) == wire.size == 43 and digest_bytes(payload) == event['wire_payload_sha256'],
                'Pilot wire size/hash differs')
            version, tag, rid, ap, capture, refresh, x, y, mask, *rssi = wire.unpack(payload)
            require(version == tag == 1 and rid == robot and 0 <= ap < aps and mask in range(1, 8)
                and (ap, capture, refresh) == (event['ap'], event['capture_slot'], event['refresh_slot'])
                and type(event['association_epoch_start']) is int and 0 <= event['association_epoch_start'] <= slot
                and type(event['last_primary_unavailable']) is int and -1 <= event['last_primary_unavailable'] <= slot
                and type(event['current_site_at_prepare']) is int
                and event['current_site_at_prepare'] == ap and event['primary_available_at_prepare'] is True
                and 0 <= slot - refresh < 10 and 0 <= slot - capture < 20
                and capture <= refresh <= slot and capture >= event['association_epoch_start']
                and capture > event['last_primary_unavailable'], 'Pilot association/capture/delivery provenance differs')
            source = np.asarray(event['source_measured_snr_db'], dtype=float)
            noise = np.asarray(event['source_noise_dbm'], dtype=float)
            tx = np.asarray(event['source_ap_tx_dbm'], dtype=float)
            require(all(a.shape == (3,) and np.isfinite(a).all() for a in (source, noise, tx)),
                'Pilot source operands malformed')
            physical_rssi = source + noise + tx - 15.
            heard = physical_rssi >= -90.
            require(mask == sum(int(value) << band for band, value in enumerate(heard)),
                'Pilot mask invents negative evidence or a detected link')
            expected_rssi = np.where(heard, physical_rssi.astype('f4'), 0.)
            require(np.array_equal(rssi, expected_rssi)
                and np.array_equal([x, y], np.asarray(event['position'], dtype='f4'))
                and list(rssi) == event['decoded_rssi_dbm'] and [x, y] == event['decoded_position']
                and heard.tolist() == event['decoded_heard']
                and event['telemetry_bytes'] == event['wire_payload_bytes'] == 43,
                'Pilot history differs from exact float32 transport positives')
            if event['delivered']:
                require(key not in delivered and (key not in retained or
                    reporting['frame_events'][retained[key]]['kind'] != 'full_scan'),
                    'More than one successful pilot per robot/second or pilot replaces full scan')
                delivered.add(key); delivered_pilots += 1
        else:
            require(kind == 'full_scan' and 0 <= event['capture_slot'] < event['complete_slot'] <= slot
                and event['telemetry_bytes'] == 128 and event['payload_hex'] is None,
                'Full scan provenance/priority differs')
            scans += 1; delivered_scans += int(event['delivered'])
        if event['delivered']:
            require(event['delivery_slot'] == slot and event['first_usable_decision'] == slot // 100 + 1
                and event['assimilated'] is True, 'Telemetry usable before successful delivery')
            retained[key] = index
        else:
            require(event['assimilated'] is False and event['retained_in_history'] is False,
                'Failed control exchange populated model history')
    retained_indices = set(retained.values())
    require(all(event['retained_in_history'] is (index in retained_indices)
        for index, event in enumerate(reporting['frame_events'])), 'Telemetry row replacement/retention differs')
    require(reporting['pilot_reports_attempted'] == pilots and reporting['pilot_reports_delivered'] == delivered_pilots
        and reporting['full_scan_reports_attempted'] == scans and reporting['full_scan_reports_delivered'] == delivered_scans
        and reporting['pilot_bytes_attempted'] == 43 * pilots
        and reporting['pilot_bytes_delivered'] == 43 * delivered_pilots
        and reporting['new_rf_reads'] == reporting['extra_control_frames'] == 0,
        'Pilot independent payload/accounting totals differ')
    return {'prepared_frames': len(seen), 'pilot_reports': pilots, 'delivered_pilots': delivered_pilots,
        'full_scan_reports': scans, 'delivered_full_scans': delivered_scans}


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def fast_check(row):
    """Reconstruct the continuous cached-forecast calculation without its helper."""
    execution = row['fast_local']
    enabled = row['controller_variant'] in ('fast', 'pilot_fast')
    require(execution['fast_local_enabled'] is enabled
        and execution['extra_RF_or_model_calls'] is False
        and execution['helper_work_independently_scheduled'] is False,
        'Fast-loop intervention or scheduling claim differs')
    events, catalog = execution['fast_control_updates'], execution['forecast_slices']
    if not enabled:
        require(not events and not catalog and execution['band_changes'] == 0
            and all(value == 0 for value in execution['mode_counts'].values()), 'Disabled fast arm executed local updates')
        return {'updates': 0, 'catalog_slices': 0, 'predictive': 0, 'myopic': 0, 'hold': 0}
    n, slots = row['robots'], row['duration_seconds'] * 100
    consumption = row['forecast_consumption']
    arrivals = {event['slot'] for event in consumption}
    ticks = sorted(set(range(0, slots, 10)) | arrivals)
    require(len(events) == len(ticks) * n, 'Missing/extra/double-counted fast tick or forecast-arrival updates')
    jobs = {job['decision']: job for job in row['jobs']}
    for index, item in enumerate(catalog):
        require(type(item['forecast_slice_id']) is int and item['forecast_slice_id'] == index
            and type(item['robot']) is int and type(item['current_site']) is int
            and item['horizons_seconds'] == HORIZONS.tolist()
            and 0 <= item['robot'] < n and 0 <= item['current_site'] < row['access_points'],
            'Malformed compact forecast catalog membership')
        value = np.asarray(item['original_forecast_snr_db'])
        require(value.shape == (9, 3) and np.isfinite(value).all(), 'Malformed native forecast slice')
        job = jobs.get(item['source_decision'])
        require(job is not None and item['forecast_ready_slot'] == job['ready_slot']
            and item['forecast_sha256'] == job['forecast_sha256'], 'Catalog lacks a paid original forecast binding')
    modes = {'predictive': 0, 'myopic': 0, 'hold': 0}
    used = set(); previous_band = [1] * n; cursor = 0; source = None; changes = 0
    for index, event in enumerate(events):
        slot, robot = ticks[index // n], index % n
        require(type(event['slot']) is int and type(event['robot']) is int
            and type(event['current_site']) is int and 0 <= event['current_site'] < row['access_points']
            and type(event['primary_available']) is bool
            and (event['slot'], event['robot']) == (slot, robot)
            and event['forecast_arrival'] is (slot in arrivals) and event['pilot_tick'] is (slot % 10 == 0),
            'Fast loop clock, robot, arrival or deduplication differs')
        while cursor < len(consumption) and consumption[cursor]['slot'] <= slot:
            source = consumption[cursor]['forecast_decision']; cursor += 1
        before, actual = event['control_band_before'], event['actual_control_band']
        require(type(before) is int and before == previous_band[robot] and type(actual) is int and 0 <= actual < 3
            and event['band_changed'] is (actual != before), 'Fast band state/change differs')
        previous_band[robot] = actual; changes += int(actual != before)
        measured = np.asarray(event['measured_snr_db'])
        require(measured.shape == (3,) and measured.dtype.kind in 'fi' and np.isfinite(measured).all(), 'Fast pilot operands malformed')
        measured = measured.astype(float)
        integer_names = ('pilot_site', 'pilot_capture_slot', 'pilot_refresh_slot',
            'association_epoch_start', 'last_primary_unavailable')
        valid = (all(type(event[name]) is int for name in integer_names)
            and event['pilot_site'] == event['current_site']
            and 0 <= event['pilot_capture_slot'] <= event['pilot_refresh_slot'] <= slot
            and 0 <= event['association_epoch_start'] <= slot
            and -1 <= event['last_primary_unavailable'] <= slot
            and 0 <= slot - event['pilot_refresh_slot'] < 10
            and 0 <= slot - event['pilot_capture_slot'] < 20
            and event['pilot_capture_slot'] >= event['association_epoch_start']
            and event['pilot_capture_slot'] > event['last_primary_unavailable'])
        expected_source = None
        correction = np.zeros(3); delta = np.zeros(3); worst = None
        if not valid:
            mode, reason, proposal = 'hold', 'invalid_pilot_history', before
        elif event['primary_available'] is not True:
            mode, reason, proposal = 'hold', 'primary_unavailable', before
        else:
            mode, reason, proposal = 'myopic', 'no_consumed_forecast', int(measured.argmax())
            worst = measured; expected_source = source
            if source is not None:
                age = slot / 100 - source
                capture_age = event['pilot_capture_slot'] / 100 - source
                equal([event['forecast_age_seconds'], event['forecast_capture_age_seconds']],
                    [age, capture_age], 'Fast age uses ordinal or refresh clock instead of source/capture clocks')
                if not 0 <= age <= 5. or age + 5. > 30. or not 0 <= capture_age <= age:
                    reason = 'stale_or_unsupported_forecast'
                else:
                    identity = event['forecast_slice_id']
                    require(type(identity) is int and 0 <= identity < len(catalog), 'Predictive fast event lacks compact slice')
                    item = catalog[identity]; used.add(identity)
                    require((item['source_decision'], item['robot'], item['current_site'])
                        == (source, robot, event['current_site'])
                        and event['forecast_ready_slot'] == item['forecast_ready_slot'] <= slot
                        and event['forecast_sha256'] == item['forecast_sha256'], 'Fast event consumes wrong source/AP/catalog or unpaid completion')
                    values = np.asarray(item['original_forecast_snr_db'], dtype=float)
                    queries = sorted(set((age + np.arange(6)).tolist()
                        + [float(t) for t in HORIZONS if age < t < age + 5.]))
                    interpolate = lambda times: np.stack([np.interp(times, HORIZONS, values[:, band])
                        for band in range(3)], axis=-1)
                    near = interpolate(queries); capture = interpolate(capture_age)
                    worst = np.min(measured + (near - capture), axis=0)
                    correction = near[0] - capture; delta = correction / .1
                    mode, reason, proposal = 'predictive', 'current_pilot_and_supported_forecast', int(worst.argmax())
                    equal(event['query_seconds_from_now'], np.asarray(queries) - age, 'Fast minimum omits interpolation corner/endpoints')
                    equal(event['forecast_at_capture_snr_db'], capture, 'Fast innovation anchors at the wrong capture age')
                    equal(event['forecast_at_now_snr_db'], near[0], 'Fast current forecast interpolation differs')
        require(event['source_decision'] == expected_source and event['mode'] == mode
            and event['reason'] == reason and event['proposed_control_band'] == proposal,
            'Fast guard/mode/proposed maximin band differs')
        equal(event['native_mcs_correction_db'], correction, 'Fast current-MCS innovation differs')
        equal(event['native_delta_db_per_second'], delta, 'Fast MCS normalization accumulates/uses future slope')
        require(event['native_delta_normalization_seconds'] == .1, 'Fast MCS compatibility scale differs')
        if worst is None:
            require(event['worst_future_snr_db'] is None and actual == before, 'Invalid pilot failed to hold after reflex')
        else:
            equal(event['worst_future_snr_db'], worst, 'Fast continuous five-second minimum differs')
            if slot % 10:
                require(actual == proposal, 'Off-tick fast result differs without a reflex tick')
        if mode != 'predictive':
            require(event['forecast_slice_id'] is None and event['query_seconds_from_now'] is None,
                'Fallback invents supported forecast evidence')
        modes[mode] += 1
    require(used == set(range(len(catalog))) and execution['mode_counts'] == modes
        and execution['band_changes'] == changes and row['fast_local_timing']['calls'] == len(events),
        'Fast catalog/count/helper-call totals differ')
    return {'updates': len(events), 'catalog_slices': len(catalog), **modes}


def matched_check(row):
    """Recompute RF metrics from full saved arrays without the metric helper."""
    with np.load(ROOT / row['prediction_path'], allow_pickle=False) as archive:
        require(archive.files == ['mean'], 'Unexpected matched prediction arrays')
        prediction = archive['mean']
    with np.load(ROOT / row['matched_inputs']['labels_path'], allow_pickle=False) as archive:
        require(archive.files == ['truth'], 'Unexpected matched private label arrays')
        target = archive['truth']
    require(prediction.shape == target.shape and prediction.ndim == 5
        and prediction.shape[2] == 9 and prediction.shape[-1] == 3
        and prediction.dtype == np.dtype('f4') and np.isfinite(prediction).all()
        and np.isfinite(target).all(), 'Matched raw axes/dtype/values differ')
    require(digest_bytes(prediction.tobytes()) == row['forecast_mean_sha256'], 'Matched full-mean array hash differs')
    for decision, value in enumerate(row['per_query']):
        require(value['decision'] == decision and value['history_cutoff_exclusive'] == 60 + decision
            and value['forecast_sha256'] == digest_bytes(prediction[decision].tobytes()),
            'Matched source-time prefix or actual per-query output hash differs')
    prediction, target = prediction.astype('f8'), target.astype('f8')
    errors = (prediction - prediction[:, :, :1]) - (target - target[:, :, :1])
    predicted_ap = prediction[:, :, :6].max(-1).mean(2).argmax(-1)
    actual_ap_values = target[:, :, :6].max(-1).mean(2)
    selected = np.take_along_axis(actual_ap_values, predicted_ap[..., None], -1)[..., 0]
    metrics = {'short_route_delta_rmse_db': float(np.sqrt(np.square(errors[:, :, 1:6]).mean())),
        'signal_rmse_db': float(np.sqrt(np.square(prediction - target).mean())),
        'all_route_delta_rmse_db': float(np.sqrt(np.square(errors).mean())),
        'ap_choice_regret_db': float((actual_ap_values.max(-1) - selected).mean())}
    require(row['metrics'] == metrics, 'Matched metrics do not reproduce full raw predictions/labels')
    return metrics


def native_check(row, p, profile, original=None):
    """Use the immutable independent payment audit plus new intervention checks."""
    import verify_freshness_stack as inherited
    from wirelessjev.pilot_fastloop import rollout
    require(row['simulation_only'] is True and row['hardware_qualified'] is False
        and row['production_qualified'] is False and row['new_fits'] == 0,
        'Research scope, hardware qualification or training changed')
    ledger = inherited.sparse_jobs_check(row, p, profile, original)
    inherited.telemetry_check(row, p)
    inherited.local_freshness_check(row, p)
    used_sources = [event['forecast_decision'] for event in row['forecast_consumption']]
    require([event['decision'] for event in row['decision_trace']] == used_sources,
        'Controller decision trace consumes an ordinal, skipped or unconsumed source')
    require([(event['decision'], event['hook_role']) for event in row['admission_trace']]
        == [(source, role) for source in used_sources for role in ('planned', 'reactive')],
        'Shared controller admission hooks differ from consumed source order')
    actions = row['result']['packet_detail'].get('action_trace')
    inherited_checks = {'raw_actions_saved': isinstance(actions, list),
        'action_trace_sha256_recomputed': False, 'sensing_guards_and_str_verified': False,
        'recovery_guards_and_sampled_counters_verified': False,
        'treatment_role_mapping': 'All four new variants retain the old latest_single_sense sensing/STR treatment and repair recovery treatment; only this audit view maps the role label.'}
    if isinstance(actions, list):
        require(value_digest(actions) == row['action_trace_sha256'], 'Raw action trace SHA256 differs')
        # These are read-only role views. The new experiment is genuinely a
        # factorial; mapping its shared baseline mechanics does not relabel
        # its scientific treatment or modify either frozen implementation.
        sensing = inherited.sensing_check(dict(row, controller_variant='latest_single_sense'), p)
        recovery = inherited.prior.recovery_check(dict(row, controller_variant='repair'), p)
        inherited_checks.update(action_trace_sha256_recomputed=True,
            sensing_guards_and_str_verified=True, recovery_guards_and_sampled_counters_verified=True,
            sensing_counts=sensing, recovery_counts=recovery)
    require(rollout.pilot_audit(row['pilot_reporting'], robots=row['robots'],
        duration=row['duration_seconds'], ap_count=row['access_points']) is True,
        'Complete producer-independent telemetry quota/provenance audit failed')
    return {'case': row['case'], 'accounting_mode': row['accounting_mode'],
        'live_replication': row['live_replication'], 'ledger': ledger,
        'inherited_controller': inherited_checks,
        'pilot': pilot_check(row), 'fast_local': fast_check(row)}


def _native_task(task):
    """Load one key or replication pair; never transfer raw rows between processes."""
    from wirelessjev.pilot_fastloop import common as c, rollout, study
    key, mode = task
    require(mode in ('main', 'offline', 'fresh'), 'Unknown passive archive namespace')
    p = c.protocol(); profile = c.latency()['profiles'][key[3]]
    original = rollout.load_case(key) if mode != 'main' else None
    row = rollout.load_case(key, replay=mode == 'offline', live_replication=mode == 'fresh')
    require(row is not None and (mode == 'main' or original is not None), 'Missing passive native archive/pair')
    if mode != 'offline':
        study._fresh([row], [key], live_replication=mode == 'fresh')
    checked = native_check(row, p, profile, original if mode == 'offline' else None)
    comparison = None
    if mode == 'offline':
        comparison = study._offline_comparisons([original], [row], [key])[0]
    elif mode == 'fresh':
        comparison = study._fresh_comparisons([original], [row], [key])[0]
    print('PASSIVE_NATIVE_CASE_OK ' + str(key) + ' ' + mode, flush=True)
    return {'row': compact(row), 'check': checked, 'comparison': comparison}


def native_tasks(tasks):
    """Ordered, bounded-memory analysis after all scientific timing has ended."""
    if not tasks:
        return []
    with ProcessPoolExecutor(max_workers=ANALYSIS_WORKERS) as pool:
        return list(pool.map(_native_task, tasks, chunksize=1))


def bootstrap_check(report, rows, matched, p, costs):
    """Independent paired whole-site resampling of all four primary slots."""
    require(len(report['primary']) == report['primary_multiplicity_budget'] == 4
        and report['primary_upper_quantile'] == .9875, 'Four-test budget changed')
    sites = p['confirmation_sites']; n = len(sites)
    by_case = {tuple(row['case']): row for row in rows}
    by_match = {(row['site'], row['family']): row for row in matched}
    sampled = np.random.default_rng(p['bootstrap_seed']).integers(0, n, size=(p['bootstrap_replicates'], n))
    sampled_rf = np.random.default_rng(p['matched_bootstrap_seed']).integers(0, n,
        size=(p['bootstrap_replicates'], n))
    def check(values, record, indices):
        values = np.asarray(values, dtype=float); draws = values[indices].mean(1)
        require(record['paired_site_differences'] == values.tolist()
            and record['mean_difference'] == float(values.mean())
            and record['multiplicity_upper'] == float(np.quantile(draws, .9875))
            and record['descriptive_95_interval'] == np.quantile(draws, [.025, .975]).tolist(),
            'Primary bootstrap differs from raw paired whole sites')
    for contrast in report['primary']:
        if not costs:
            require(contrast['status'] == 'not_run_development_veto' and contrast['mean_difference'] is None
                and contrast['statistical_pass'] is False, 'Unrun primary gained invented evidence')
            continue
        if contrast['kind'] == 'matched_rf':
            values = [by_match[site, FAMILIES[1]]['metrics']['short_route_delta_rmse_db']
                - by_match[site, FAMILIES[0]]['metrics']['short_route_delta_rmse_db'] for site in sites]
            check(values, contrast, sampled_rf)
            require(contrast['engineering_pass'] is None and contrast['endpoint_sensitivity'] is None
                and contrast['experimental_benefit'] is False
                and contrast['statistical_pass'] is (contrast['multiplicity_upper'] < 0),
                'Fully observed RF endpoint received a packet gate or wrong accuracy decision')
        else:
            observed, pending = [], []
            for site in sites:
                a = by_case['confirmation', site, 4., contrast['candidate_family'], contrast['candidate_variant']]['result']
                b = by_case['confirmation', site, 4., contrast['reference_family'], contrast['reference_variant']]['result']
                require(a['control_sent'] == b['control_sent'] == p['robots'] * p['duration_seconds'] * 20,
                    'Unmatched command generation')
                observed.append(a['control_misses'] / a['control_sent'] - b['control_misses'] / b['control_sent'])
                pending.append((a['control_misses'] + a['packet_detail']['pending_control_packets']) / a['control_sent']
                    - b['control_misses'] / b['control_sent'])
            check(observed, contrast, sampled); check(pending, contrast['endpoint_sensitivity'], sampled)
            require(contrast['statistical_pass'] is (contrast['multiplicity_upper'] < 0
                and contrast['endpoint_sensitivity']['multiplicity_upper'] < 0), 'Packet superiority omitted pending endpoint')
    require(len(report['secondary_component_contrasts']) == 14 and all(
        value['statistical_pass'] is None and value['statistical_superiority_claimed'] is False
        and value['experimental_benefit'] is False for value in report['secondary_component_contrasts']),
        'Descriptive factorial effects were promoted to inferential results')


def engineering_check(report, rows, p, costs):
    by_case = {tuple(row['case']): row for row in rows}
    def metrics(row, pending):
        result = row['result']; detail = result['packet_detail']; account = result['packet_accounting']
        misses = np.asarray(detail['misses'], dtype=float)
        if pending:
            misses += np.asarray(detail['pending_control_packet_age_slots']) >= 0
        rates = misses / np.asarray(detail['sent'])
        return {'worst': float(np.sort(rates)[-max(1, len(rates) // 10):].mean()),
            'silence': max(detail['max_gap']) / 100.,
            'video': account['video_delivered_bits'] / max(account['video_requested_bits'], 1e-9),
            'bulk': account['bulk_delivered_bits'] / p['duration_seconds'] / 1e6}
    if not costs:
        require(not report['engineering_gates'], 'Unrun packet gate gained invented results')
        return
    for gate in report['engineering_gates']:
        left, right = [], []
        for site in p['confirmation_sites']:
            left.append(metrics(by_case['confirmation', site, 4., gate['candidate_family'], gate['candidate_variant']], True))
            right.append(metrics(by_case['confirmation', site, 4., gate['reference_family'], gate['reference_variant']], False))
        mean = lambda rows, name: float(np.mean([row[name] for row in rows]))
        maxsilence = lambda rows: max(row['silence'] for row in rows)
        checks = {'worst_decile': mean(left, 'worst') <= mean(right, 'worst') + p['worst_decile_margin'],
            'maximum_silence': maxsilence(left) <= maxsilence(right) + p['silence_margin_seconds'],
            'original_video': mean(left, 'video') >= mean(right, 'video') - p['video_margin'],
            'bulk_delivery': mean(left, 'bulk') >= p['bulk_ratio'] * mean(right, 'bulk')}
        zero = {'worst_decile': mean(left, 'worst') <= mean(right, 'worst'),
            'maximum_silence': maxsilence(left) <= maxsilence(right),
            'original_video': mean(left, 'video') >= mean(right, 'video'),
            'bulk_delivery': mean(left, 'bulk') >= mean(right, 'bulk')}
        require(gate['checks'] == checks and gate['passes'] is all(checks.values())
            and gate['zero_margin_checks'] == zero and gate['zero_margin_passes'] is all(zero.values()),
            'Packet engineering or zero-margin diagnostics differ from native per-robot counters')


def compact(row):
    """Retain exactly analysis operands; no raw event stream survives the loop."""
    names = ('stage', 'site', 'handover_seconds', 'family', 'controller_variant', 'training_seed',
        'runtime_overrun', 'predictor_runtime_overrun', 'causality_pass', 'paid_cost_and_consumption_pass',
        'replay', 'live_replication', 'accounting_mode', 'forecast_metrics', 'jobs', 'arrivals',
        'queue_mode', 'accepted_decisions', 'forecast_consumption', 'fast_local_timing', 'case')
    value = {name: row[name] for name in names}
    pilot, fast = row['pilot_reporting'], row['fast_local']
    value['pilot_reporting'] = {name: item for name, item in pilot.items() if name not in ('frame_events', 'row_replacements')}
    value['pilot_reporting'].update(frame_events=range(len(pilot['frame_events'])), row_replacements=range(len(pilot['row_replacements'])))
    value['fast_local'] = {name: item for name, item in fast.items() if name not in ('fast_control_updates', 'forecast_slices')}
    value['fast_local'].update(fast_control_updates=range(len(fast['fast_control_updates'])), forecast_slices=range(len(fast['forecast_slices'])))
    result = row['result']; detail = result['packet_detail']; sensing = result['freshness_execution']
    value['result'] = {name: result[name] for name in ('control_misses', 'control_sent', 'control_miss', 'packet_accounting')}
    value['result']['packet_detail'] = {name: detail[name] for name in ('misses', 'sent', 'max_gap',
        'pending_control_packet_age_slots', 'pending_control_packets', 'cumulative')}
    value['result']['freshness_execution'] = {'scan_starts': range(len(sensing['scan_starts'])),
        'scan_assessments': [{'suppress_scan': event.get('suppress_scan')} for event in sensing['scan_assessments']]}
    return value


def operational_provenance(meta, gate):
    """Bind supplementary runner files without importing its cache wrapper.

    The final verifier authenticates publication, byte-map membership, hits,
    process identities and chronology. This raw audit independently reproduces
    the original development gate and report from uncached archive reads.
    """
    pause_path = OUT / 'logs/operational-pause.json'
    wrapper = ROOT / 'scripts/run_pilot_fastloop_cached.py'
    if not pause_path.exists():
        require(not wrapper.exists(), 'Operational wrapper exists without a pause/provenance record')
        return {'supplemental_runner_used': False, 'operational_files_sha256': {},
            'development_gate_recomputed_without_operational_cache': True}
    pause = read(pause_path)
    require(pause['source_commit'] == meta['scientific_source_commit']
        and pause['completed_confirmation_receipts'] == {} and pause['selection_or_tuning'] is False,
        'Operational pause did not precede every completed confirmation result')
    require(wrapper.is_file(), 'Missing supplementary operational wrapper')
    resumes = [path for path in (OUT / 'logs/operational-resume.json', OUT / 'logs/resume.json') if path.is_file()]
    require(len(resumes) == 1, 'Require one transparent operational resume receipt')
    paths = {pause_path, wrapper, *resumes}
    for pattern in ('cached-development-*.bootstrap.json', 'cached-development-*.cache-hits.jsonl',
            'cached-wrapper-tests.txt', 'cached-wrapper-preflight.json', 'cached-wrapper-fixture.py'):
        paths.update((OUT / 'logs').glob(pattern))
    bootstraps = sorted((OUT / 'logs').glob('cached-development-*.bootstrap.json'))
    require(bootstraps, 'No supplemental guard-cache bootstrap was recorded')
    for path in bootstraps:
        receipt = read(path)
        require(receipt['gate'] == gate and receipt['gate_sha256'] == value_digest(gate)
            and receipt['wrapper_sha256'] == digest(wrapper)
            and receipt['before_sha256'] == receipt['after_sha256']
            and receipt['unchanged'] is True and receipt['original_full_verified_development'] is True
            and receipt['source_guard_checked'] is True
            and receipt['registered_scientific_source_changed'] is False
            and receipt['new_worlds_models_or_training_called_by_wrapper'] is False,
            'Operational cached gate or immutable-bootstrap identity differs from independent recomputation')
    return {'supplemental_runner_used': True,
        'operational_files_sha256': {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)},
        'operational_resume_sha256': digest(resumes[0]),
        'development_gate_sha256': value_digest(gate),
        'development_gate_recomputed_without_operational_cache': True,
        'scope': 'Supplementary validation-cache/launch optimization outside the original scientific source closure; full publication/map/hit/PID/chronology authentication belongs to the final verifier. No model/controller/criterion replacement is inferred.'}


def audit(write=False):
    """Stream raw archives; retain compact operands for passive report analysis."""
    from wirelessjev.pilot_fastloop import common as c, rollout, matched, study, report as stats
    import verify_freshness_stack as inherited
    require(type(write) is bool, 'Use a boolean write flag')
    frozen, meta = registration(c); p = c.protocol()
    require(p['robots'] == 100 and p['duration_seconds'] == 60 and p['handover_seconds'] == [4]
        and p['development_sites'] == [2153000, 2153001]
        and p['confirmation_sites'] == list(range(2154000, 2154008))
        and p['bootstrap_replicates'] == 20000 and p['bootstrap_seed'] == 2156000
        and p['matched_bootstrap_seed'] == 2157000 and p['primary_upper_quantile'] == .9875
        and p['worst_decile_margin'] == 1 / 1200 and p['silence_margin_seconds'] == .05
        and p['video_margin'] == .01 and p['bulk_ratio'] == .99
        and p['development_severe_regression_absolute'] == .01, 'Registered dimensions, margins or primary budget differ')
    case_audits = []
    devkeys = study.ordered_keys('development')
    require(len(devkeys) == 16, 'Development matrix differs')
    checked = native_tasks([(key, 'main') for key in devkeys])
    dev = [entry['row'] for entry in checked]
    case_audits.extend(entry['check'] for entry in checked)
    devmatched_keys = study.ordered_matched_keys('development')
    devmatched = study._load_matched(devmatched_keys)
    require(len(devmatched) == 4, 'Development matched matrix differs')
    matched_metrics = []
    for row in devmatched:
        matched_metrics.append({'case': row['case'], 'replay': False, 'metrics': matched_check(row)})
    validation_dev = {'source_and_archive_integrity': True, 'causality_and_kernel_checks': True,
        'paid_cost_and_consumption_checks': True, **c.matched_validation(devmatched)}
    gate = stats.development_gate(dev, p, validation_dev, matched_rows=devmatched)
    sealed_dev = {'gate': gate, 'validation': validation_dev, 'source_sha256': c.source_hashes(),
        'execution_order_sha256': digest(OUT / 'execution_order.json')}
    require(read(OUT / 'development.json') == sealed_dev, 'Development seal differs from streamed raw archives')
    permission = {'development_sha256': digest(OUT / 'development.json'),
        'source_sha256': c.source_hashes(), 'passed_costs': gate['passed_costs'], 'recomputed_from_raw_archives': True}
    require(read(OUT / 'qualification.json') == permission, 'Confirmation permission differs from streamed development')
    for entry in gate['per_cost']:
        for check in entry['family_checks']:
            select = lambda variant: [row['result'] for row in dev if row['family'] == check['family'] and row['controller_variant'] == variant]
            rate = lambda rows: Fraction(sum(row['control_misses'] for row in rows), sum(row['control_sent'] for row in rows))
            difference = rate(select('pilot_fast')) - rate(select('base'))
            require(check['exact_difference_numerator'] == difference.numerator
                and check['exact_difference_denominator'] == difference.denominator
                and check['severe_regression'] is (difference > Fraction('0.01')),
                'Development veto differs from exact native counters')
    costs = gate['passed_costs']
    keys, repeats = study.ordered_keys('confirmation', costs), study.replication_keys(costs)
    require(len(keys) == (64 if costs else 0) and len(repeats) == (8 if costs else 0),
        'Conditional confirmation or repetition count differs')
    inherited.membership(OUT / 'cases', devkeys + keys, rollout)
    inherited.membership(OUT / 'replays', repeats, rollout)
    inherited.membership(OUT / 'live_replications', repeats, rollout)
    checked = native_tasks([(key, 'main') for key in keys])
    main = [entry['row'] for entry in checked]
    case_audits.extend(entry['check'] for entry in checked)
    offline, live, comparisons, sensitivities = [], [], [], []
    tasks = [(key, mode) for key in repeats for mode in ('offline', 'fresh')]
    checked = native_tasks(tasks)
    for (_, mode), entry in zip(tasks, checked):
        case_audits.append(entry['check'])
        if mode == 'offline':
            offline.append(entry['row']); comparisons.append(entry['comparison'])
        else:
            live.append(entry['row']); sensitivities.append(entry['comparison'])
    matchkeys = study.ordered_matched_keys('confirmation', costs)
    matchrepeats = study.matched_replication_keys(costs)
    streams = study._load_matched(matchkeys); replays = study._load_matched(matchrepeats, replay=True)
    require(len(streams) == (16 if costs else 0) and len(replays) == (2 if costs else 0), 'Matched conditional counts differ')
    for row in streams + replays:
        matched_metrics.append({'case': row['case'], 'replay': row['replay'], 'metrics': matched_check(row)})
    matchcomparisons = study._matched_comparisons(streams, replays, matchrepeats)
    completion = {'completed': True, 'source_sha256': c.source_hashes(),
        'qualification_sha256': digest(OUT / 'qualification.json'),
        'execution_order_sha256': digest(OUT / 'execution_order.json'), 'passed_costs': costs,
        'case_count': len(main), 'offline_replay_count': len(offline), 'fresh_live_replication_count': len(live),
        'offline_comparisons': comparisons, 'all_offline_exact': all(row['exact'] for row in comparisons),
        'fresh_live_cost_comparisons': sensitivities,
        'all_fresh_paid_cost_checks_pass': all(row['paid_cost_and_consumption_pass'] for row in sensitivities),
        'matched_sequence_count': len(streams), 'matched_offline_replay_count': len(replays),
        'matched_offline_comparisons': matchcomparisons,
        'all_matched_offline_exact': all(row['exact'] for row in matchcomparisons),
        'archive_receipt_sha256': study._receipt_hashes(keys, repeats, matchkeys, matchrepeats)}
    if not all((OUT / name).is_file() for name in ('completion.json', 'report.json', 'report.md')):
        print('PASSIVE_RAW_CHECKS_COMPLETE; waiting for the registered report to finish.', flush=True)
        # This is file availability waiting, with no inference or benchmarking.
        # The original report process remains responsible for its own outputs.
        waiting = Event()
        while not all((OUT / name).is_file() for name in ('completion.json', 'report.json', 'report.md')):
            waiting.wait(5.)
    require(completion == read(OUT / 'completion.json') and completion['all_offline_exact'] is True
        and completion['all_matched_offline_exact'] is True, 'Completion/exact native or matched replay differs')
    validation = {**validation_dev, 'exact_offline_replay': completion['all_offline_exact'],
        **c.matched_validation(streams + replays), 'exact_matched_replay': completion['all_matched_offline_exact']}
    expected = stats.aggregate(main, p, validation, passed_costs=costs, matched_rows=streams)
    first_main = [row for row in main if row['site'] == p['confirmation_sites'][0]]
    offsummary = stats.replication_summary(offline, p, 'recorded_schedule_offline', costs, first_main)
    livesummary = stats.replication_summary(live, p, 'fresh_live_cost', costs, first_main)
    sensitivity_by_case = {tuple(row['case']): row for row in sensitivities}
    for row in livesummary['differences_from_main']:
        row['packet_result_exactly_matches_original'] = sensitivity_by_case[tuple(row['case'])]['packet_result_exactly_matches_original']
    expected.update(development_gate=gate, matched_offline_replay_comparisons=matchcomparisons,
        offline_replay_comparisons=comparisons, fresh_live_cost_comparisons=sensitivities,
        offline_replay_summary=offsummary, fresh_live_cost_summary=livesummary,
        source_sha256=c.source_hashes(), completion_sha256=digest(OUT / 'completion.json'),
        development_sha256=digest(OUT / 'development.json'), qualification_sha256=digest(OUT / 'qualification.json'),
        execution_order_sha256=digest(OUT / 'execution_order.json'))
    require(expected == read(OUT / 'report.json'), 'Report differs from streamed raw-archive reproduction')
    bootstrap_check(expected, main, streams, p, costs); engineering_check(expected, main, p, costs)
    markdown = stats.markdown(expected) + ('\n## Replication scope\n\n'
        f'{len(offline)} exact-schedule OFFLINE replays; {len(live)} fresh actual-cost repetitions. '
        'The former establish only replay under recorded availability; the latter are descriptive runtime sensitivity and add no independent sites. '
        f'{len(streams)} matched offline forecasting sequences and {len(replays)} exact matched replays add no packet trials or independent sites.\n')
    require((OUT / 'report.md').read_text() == markdown, 'Markdown differs from recomputed report')
    value = {'schema': 1, 'scientific_source_commit': meta['scientific_source_commit'],
        'scientific_source_tree': meta['scientific_source_tree'], 'registered_source_files': len(frozen['source_sha256']),
        'old_source_and_evidence_preserved': c.verify_previous_evidence(),
        'passive_diagnosis_verified': c.verify_diagnosis_inputs(),
        'status': 'confirmation_complete' if costs else 'not_run_development_veto',
        'development_cases': len(dev), 'confirmation_cases': len(main), 'offline_exact_replays': len(offline),
        'fresh_live_replications': len(live), 'native_total': len(case_audits),
        'matched_development_streams': len(devmatched), 'matched_confirmation_streams': len(streams),
        'matched_exact_replays': len(replays), 'passed_development_costs': costs,
        'independent_pilot_payload_provenance_verified': True, 'independent_fast_local_arithmetic_verified': True,
        'independent_four_primary_bootstrap_verified': True, 'report_recomputed': True,
        'exact_native_and_matched_replays_verified': True, 'no_worlds_models_training_or_timing_executed': True,
        'native_case_audits': case_audits, 'matched_raw_metrics': matched_metrics,
        'consumed_source_and_admission_sequences_verified': True,
        'inherited_sensing_recovery_action_checks_complete': all(
            entry['inherited_controller']['action_trace_sha256_recomputed']
            and entry['inherited_controller']['sensing_guards_and_str_verified']
            and entry['inherited_controller']['recovery_guards_and_sampled_counters_verified']
            for entry in case_audits),
        'report_sha256': digest(OUT / 'report.json'), 'script_sha256': digest(Path(__file__)),
        'source_registration_sha256': digest(OUT / 'registration/source.json'),
        'source_git_binding_sha256': digest(OUT / 'registration/git.json'),
        'operational_provenance': operational_provenance(meta, gate),
        'independent_sites_added': 0, 'production_qualified': False, 'hardware_qualified': False,
        'passive_analysis_workers': ANALYSIS_WORKERS, 'scientific_workers_unchanged': p['workers'],
        'limits': ['Native physical RF values and every 10ms successful exchange are not resimulated; captured operands/provenance, source capsules and exact replay supply the available evidence.',
            'Native compact9x3 forecast slices bind to full-forecast hashes but cannot independently reconstruct those complete arrays.',
            'Fast guard, proposed maximin band, capture-to-now MCS correction and continuous extrema are independently recomputed. Valid-pilot100ms reflex choices lack all original threshold/occupancy operands and remain frozen-source/exact-replay backed.',
            'Matched full means and labels are independently checked; standard-deviation values are represented only by per-query hashes and exact replay.',
            'Shared sensing/STR and cost-aware recovery are checked through explicit old-final/repair audit role views when raw actions exist. Missing raw actions are reported as incomplete inherited-controller coverage; full10ms counter recurrence remains frozen-source/test backed.',
            'Unrun confirmation/replay slots after a development veto remain explicitly unmeasured. No packet benefit is inferred from the matched accuracy flag.']}
    path = OUT / 'verification/raw_audit.json'
    if write:
        c.write_once(path, value)
    elif path.exists():
        require(read(path) == value, 'Existing raw audit receipt differs from passive recomputation')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Write the immutable raw audit receipt')
    args = parser.parse_args()
    value = audit(write=args.write)
    print(json.dumps({name: value[name] for name in ('status', 'native_total', 'matched_confirmation_streams',
        'scientific_source_commit', 'report_sha256')}, indent=2))
