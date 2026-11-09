"""Immutable preregistered packet study: register/development/confirmation/report.
Registration generates no sites. Resume checks source, inputs, and paid latency.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import tempfile
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/continual_packet'
PROTOCOL = OUT / 'registration/protocol.json'
BASE_LATENCY = ROOT / 'artifacts/packet_value/latency.json'


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def protocol():
    return json.loads(PROTOCOL.read_text())


def software_versions():
    return {'python': platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ('numpy', 'scipy', 'torch')}}


def write_once(path, value, *, compressed=False):
    """Atomic creation without replacing existing evidence, even under a race."""
    path = Path(path)
    data = encoded(value)
    if compressed:
        data = gzip.compress(data, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'Immutable evidence differs: {path}')
        return
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise RuntimeError(f'Concurrent evidence differs: {path}')
    finally:
        os.unlink(temporary)


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def sources():
    from ..frontier.study import sources as inherited
    paths = list((ROOT / 'src/wirelessjev/continual_packet').glob('*.py'))
    paths += list((ROOT / 'src/wirelessjev/closed_loop').glob('*.py'))
    paths += [ROOT / 'src/wirelessjev/continual_memory.py', ROOT / 'src/wirelessjev/hardware_budget.py',
              BASE_LATENCY, PROTOCOL]
    return inherited() | {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def case_keys(stage, p=None):
    p = protocol() if p is None else p
    if stage not in ('development', 'confirmation'):
        raise ValueError('Unknown scientific stage')
    return [(stage, site, packet, float(cost), arm)
            for site in p[stage + '_sites'] for packet in p[stage + '_packet_seeds']
            for cost in p['handover_seconds'] for arm in p['arms']]


def case_name(key):
    stage, site, packet, cost, arm = key
    return f'{stage}-{site}-p{packet}-h{round(cost * 1000):04d}-{arm}'


def register():
    """Seal source capsule before any development or confirmation data access."""
    hashes = sources()
    path = OUT / 'registration/source.json'
    with locked(OUT / 'registration/.lock'):
        if path.exists():
            return check_sources()
        for name, expected in hashes.items():
            target = OUT / 'registration/source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(target) != expected:
                    raise RuntimeError('Partial source capsule differs')
            else:
                shutil.copyfile(ROOT / name, target)
        receipt = {'schema': 1, 'source_sha256': hashes, 'protocol_sha256': digest(PROTOCOL),
                   'time_unix': time.time(), 'python': platform.python_version(), 'numpy': np.__version__,
                   'software_versions': software_versions(),
                   'data_accessed': False, 'main_case_counts':
                   {s: len(case_keys(s)) for s in ('development', 'confirmation')}}
        write_once(path, receipt)
        return receipt


def check_sources():
    path = OUT / 'registration/source.json'
    if not path.exists():
        raise RuntimeError('Run register after implementation and protocol are frozen')
    receipt = json.loads(path.read_text())
    if receipt['source_sha256'] != sources():
        raise RuntimeError('Registered implementation changed; refusing mixed evidence')
    if receipt['software_versions'] != software_versions():
        raise RuntimeError('Registered numerical software versions changed')
    for name, expected in receipt['source_sha256'].items():
        if digest(OUT / 'registration/source' / name) != expected:
            raise RuntimeError('Source capsule integrity failure')
    return receipt


def public_spec(site, geometry, design, p=None):
    """Fixed public RNG draw order, independent of hidden coupling and outcomes."""
    from .probes import PublicProbeSpec
    p = protocol() if p is None else p
    rng = np.random.default_rng(np.random.SeedSequence([site, 202021]))
    positions = rng.uniform([10., 6.], [110., 74.], size=(2, 2))
    channel = int(rng.choice(sorted({int(ap.channel) for ap in geometry.aps})))
    blocks = (int(p['duration_seconds']) + 31) * 10
    rng = np.random.default_rng(np.random.SeedSequence([site, 202022]))
    active = rng.random((blocks, 2)) >= .7
    plan = np.repeat(rng.uniform(.05, .20, size=(blocks, 2)) * active, 10, axis=0)
    starts = tuple(range(0, len(plan) - 2, p['probe_period_slots']))
    for start in starts:
        plan[start:start + 3] = 0.
    return PublicProbeSpec(positions, np.full(2, 10 ** 1.5), channel, plan, starts,
        design=design, pulse_airtime_fraction=p['pulse_airtime_fraction'],
        capture_airtime_fraction=p['capture_airtime_fraction'], report_bytes=p['report_bytes_per_ap'],
        report_rate_mbps=p['report_rate_mbps'], report_delay_slots=p['report_delay_slots'],
        noise_floor_fraction=p['measurement_noise_floor_fraction'],
        missing_probability=p['measurement_missing_probability'],
        shared_receiver_blackout=p['shared_receiver_blackout'])


def input_hashes(full, geometry, warm, spec):
    def arrays(items):
        h = hashlib.sha256()
        for name, value in items:
            if value is None:
                h.update(encoded([name, None]))
            else:
                value = np.ascontiguousarray(value)
                h.update(encoded([name, value.dtype.str, value.shape]))
                h.update(memoryview(value).cast('B'))
        return h.hexdigest()
    return {'physical_sha256': arrays((n, getattr(full, n)) for n in
            ('snr', 'rssi_dl', 'pos', 'plan', 'sites', 'domain', 'avail', 'channel', 'noise_db',
             'tx_dl', 'video_mbps', 'bulk_mbps')),
        'observable_prefix_sha256': arrays((n, getattr(warm, n)) for n in
            ('positions', 'value', 'heard', 'bound', 'scanned')),
        'public_geometry_sha256': sha({'sites': geometry.sites.tolist(), 'tx': geometry.tx.tolist(),
            'noise': geometry.noise.tolist(), 'aps': [asdict(ap) for ap in geometry.aps]}),
        'public_spec_sha256': spec.digest()}


def charges(stage):
    profile = json.loads(BASE_LATENCY.read_text())['forecast_seconds']
    if set(profile) != {'cold', 'warm', 'refit'} or any(not math.isfinite(v) or v <= 0 for v in profile.values()):
        raise RuntimeError('Invalid archival forecast profile')
    head = protocol()['development_local_compute_seconds']
    result = {'forecast_seconds': profile, 'existing_extra_seconds': .5,
              'shared_head_seconds': head, 'base_latency_sha256': digest(BASE_LATENCY)}
    if stage == 'confirmation':
        seal = json.loads((OUT / 'latency.json').read_text())
        if not seal['sealed'] or seal['source_sha256'] != check_sources()['source_sha256']:
            raise RuntimeError('Confirmation requires development-only sealed latency')
        for name, expected in seal['development_receipt_sha256'].items():
            if digest(OUT / name) != expected:
                raise RuntimeError('Development timing receipts changed after latency seal')
        result.update(shared_head_seconds=seal['shared_head_seconds'], seal_sha256=digest(OUT / 'latency.json'))
    elif stage != 'development':
        raise ValueError('Unknown scientific stage')
    return result


def validate_case(row, p=None):
    """Hard accounting/causality invariants; never replace an invalid score."""
    p = protocol() if p is None else p
    result, d = row['result'], row['result']['packet_detail']
    n, duration = p['robots'], p['duration_seconds']
    misses, sent = np.asarray(d['misses']), np.asarray(d['sent'])
    if (misses.shape != (n,) or sent.shape != (n,) or np.any(misses < 0) or
            np.any(misses > sent) or np.any(sent != duration * 20) or
            np.any(misses != misses.astype('i8')) or
            int(misses.sum()) != result['control_misses'] or int(sent.sum()) != result['control_sent']):
        raise RuntimeError('Raw control counter accounting failure')
    if sum(result['miss_causes'].values()) != result['control_misses']:
        raise RuntimeError('Miss causes do not conserve integer misses')
    trace = d['cumulative']
    if len(trace) != duration or [t['slot_end'] for t in trace] != list(range(100, duration * 100 + 1, 100)):
        raise RuntimeError('Incomplete second-by-second counter trace')
    previous = np.zeros(n, dtype='i8')
    previous_gap = np.zeros(n)
    for second, entry in enumerate(trace, 1):
        m, s = np.asarray(entry['misses']), np.asarray(entry['sent'])
        if (m.shape != (n,) or s.shape != (n,) or not np.isfinite(m).all() or
                np.any(m != np.floor(m)) or np.any(m < previous) or np.any(m > s) or np.any(s != second * 20)):
            raise RuntimeError('Cumulative control accounting failure')
        if 'max_gap' in entry:
            gap = np.asarray(entry['max_gap'])
            if gap.shape != (n,) or not np.isfinite(gap).all() or np.any(gap < previous_gap):
                raise RuntimeError('Cumulative observed silence accounting failure')
            previous_gap = gap
        previous = m
    if not np.array_equal(previous, misses):
        raise RuntimeError('Final cumulative misses differ')
    for event in d['delivered_report_events']:
        if not (event['measurement_slot'] < event['complete_slot'] <= event['delivery_slot'] and
                event['first_usable_decision'] == event['delivery_slot'] // 100 + 1):
            raise RuntimeError('Noncausal delivered scan')
    previous_ready = 0.
    if len(row['forecast_timing']) != duration or len(row['predictor_records']) != duration:
        raise RuntimeError('Incomplete forecast timing ledger')
    estimator = p['arms'][row['arm']][1]
    head = 0. if estimator == 'none' else row['charged_latency']['shared_head_seconds']
    if row['extra_seconds'] != row['charged_latency']['existing_extra_seconds'] + head:
        raise RuntimeError('Incorrect common inference charge')
    local = []
    total_overruns = []
    for decision, record in enumerate(row['forecast_timing']):
        if record['decision'] != decision or record['history_cutoff_exclusive'] != 60 + decision:
            raise RuntimeError('Forecast consumed a noncausal history')
        charge = row['charged_latency']['forecast_seconds'][record['kind']] + row['extra_seconds']
        previous_ready = max(decision, previous_ready) + charge
        if record['charged_seconds'] != charge or record['ready_slot'] != math.ceil(previous_ready * 100 - 1e-10):
            raise RuntimeError('Forecast FIFO charge or completion changed')
        prediction = row['predictor_records'][decision]
        elapsed, measured = prediction['local_compute_seconds'], record['actual_prediction_seconds']
        if prediction['decision'] != decision or not math.isfinite(elapsed) or elapsed < 0 or not math.isfinite(measured) or measured < 0:
            raise RuntimeError('Invalid raw inference measurement')
        local.append(elapsed)
        if measured + (elapsed if estimator == 'none' else 0.) > charge:
            total_overruns.append(decision)
    expected_head = [i for i, value in enumerate(local) if estimator != 'none' and value > head]
    if (row['runtime']['head_overrun_decisions'] != expected_head or
            row['runtime']['total_prediction_overrun_decisions'] != total_overruns or
            row['runtime']['local_head_p95_seconds'] != float(np.quantile(local, .95))):
        raise RuntimeError('Inference overrun ledger differs from raw measurements')
    reports = row['probe_reports']
    for report in reports:
        slots = report['measured_slots']
        if (len(slots) != 3 or list(slots) != list(range(slots[0], slots[0] + 3)) or
                slots[0] // 10 != slots[-1] // 10 or
                not (slots[-1] < report['completed_slot'] <= report['delivered_slot']) or
                report['delivered_slot'] >= duration * 100 or
                report['payload_bytes'] != p['report_bytes_per_ap'] * len(report['receiver_sites'])):
            raise RuntimeError('Noncausal or malformed paid probe report')
    summary = result['managed_probe_summary']
    if summary['cycles_delivered'] != len(reports) or summary['report_bytes_transmitted'] != sum(r['bytes'] for r in summary['report_transport_slots']):
        raise RuntimeError('Probe transport accounting failure')
    design = p['arms'][row['arm']][0]
    captures = 0 if design == 'none' else 3 * len(range(0, duration * 100 - 2, p['probe_period_slots']))
    if summary['captured_slots'] != captures:
        raise RuntimeError('Unexpected probe schedule execution')
    if (summary['control_allocator_slots'] != duration * 100 or
            summary['control_allocator'] != 'per_retry_band_domain_every_managed_slot'):
        raise RuntimeError('Managed resource allocator was bypassed')
    if reports:
        unused = list(summary['report_transport_slots'])
        for report in sorted(reports, key=lambda r: r['measured_slots'][0]):
            paid = []
            last_possible = report['delivered_slot'] - p['report_delay_slots'] - 1
            for receiver in report['receiver_sites']:
                payment = next((event for event in unused if event['site'] == receiver and
                    report['completed_slot'] <= event['slot'] <= last_possible and
                    event['bytes'] == p['report_bytes_per_ap']), None)
                if payment is None:
                    raise RuntimeError('Probe report published before paid per-AP transport')
                unused.remove(payment)
                paid.append(payment['slot'])
            if not paid or max(paid) != last_possible:
                raise RuntimeError('Probe completion delay differs from paid transport')
    account = result['managed_packet_accounting']
    error = account['bulk_generated_bits'] - account['bulk_delivered_bits'] - account['final_bulk_backlog_bits'] - account['bulk_overflow_drop_bits']
    if abs(error) > max(1., 1e-12 * max(1., account['bulk_generated_bits'])) or abs(error - account['bulk_flow_conservation_error_bits']) > 1e-6:
        raise RuntimeError('Bulk queue flow conservation failure')
    ages = np.asarray(account['pending_control_packet_age_slots'])
    if (ages.shape != (n,) or np.any(ages < -1) or np.any(ages >= 5) or
            np.any(ages != np.floor(ages)) or account['pending_control_packets'] != int((ages >= 0).sum()) or
            np.any(misses + (ages >= 0) > sent)):
        raise RuntimeError('Right-censored control accounting failure')
    for name, value in account.items():
        if name.endswith('_bits') and (not math.isfinite(value) or ('error' not in name and value < -1e-6)):
            raise RuntimeError('Invalid raw service accounting')
    return True


def _receipt(row, path):
    return {'completed': True, 'case': row['case'], 'source_sha256': row['source_sha256'],
            'data_sha256': row['data_sha256'], 'charged_latency_sha256': sha(row['charged_latency']),
            'archive_sha256': digest(path), 'result_sha256': sha(row['result']), 'full_record_sha256': sha(row)}


def load_case(key, *, data=None, charged=None, replay=False):
    path = OUT / ('replays' if replay else 'cases') / (case_name(key) + '.json.gz')
    if not path.exists():
        return None
    row = json.loads(gzip.decompress(path.read_bytes()))
    if (row['case'] != list(key) or row['source_sha256'] != check_sources()['source_sha256'] or
            (data is not None and row['data_sha256'] != data) or
            (charged is not None and row['charged_latency'] != charged)):
        raise RuntimeError('Resume source/data/charged-latency mismatch')
    validate_case(row)
    # Recover only a valid atomically completed archive with an interrupted receipt write.
    write_once(path.with_suffix('').with_suffix('.receipt.json'), _receipt(row, path))
    return row


def case(task):
    key, replay = task
    stage, site, packet_seed, cost, arm = key
    import torch
    from ..fleetwm.data import episode
    from ..frontier.study import mask_scans, runtime_world, READINGS
    from ..closed_loop.telemetry import DeliveredScans
    from ..closed_loop.runtime import LiveForecasts
    from ..native.foundation import random_site
    from ..native.intent_policy import intent_policy
    from ..hardware_budget import OutageBudgetGate
    from ..gate_repair.gate import reactive_rescue
    from .control import ResponsePredictor
    from .probes import ProbeMailbox, ProbeRuntime
    from .sim import simulate
    torch.set_num_threads(1)
    frozen, p, charged = check_sources(), protocol(), charges(stage)
    path = OUT / ('replays' if replay else 'cases') / (case_name(key) + '.json.gz')
    with locked(path.with_suffix('.lock')):
        # Seed identity alone is insufficient: reconstruct and hash actual inputs on resume.
        full, _, geometry, obs = episode(site, p['robots'], p['duration_seconds'], 'main')
        warm = mask_scans(obs, 5).before(60)
        design, estimator = p['arms'][arm]
        spec = public_spec(site, geometry, design, p)
        data = input_hashes(full, geometry, warm, spec)
        old = load_case(key, data=data, charged=charged, replay=replay)
        if old is not None:
            return old
        delivered = DeliveredScans(warm, p['duration_seconds'], site)
        mailbox = ProbeMailbox()
        predictor = ResponsePredictor(geometry, spec, mailbox, estimator,
            receiver_domains=full.domain[:, 1], max_age_seconds=p['memory_max_age_seconds'])
        head = 0. if estimator == 'none' else charged['shared_head_seconds']
        extra = charged['existing_extra_seconds'] + head
        live = LiveForecasts(predictor, delivered, p['duration_seconds'], charged['forecast_seconds'], extra)
        world = runtime_world(full, p['duration_seconds'], live.mean, live.std[:, :, 0])
        runtime = ProbeRuntime.from_world(world, spec, mailbox,
            measurement_seed=site * 2 + packet_seed, site_params=random_site(site))
        reading = READINGS[p['mission']]
        policy = replace(intent_policy(reading['weights'], reading['focus'], dual_hardware='none'),
            online_map=False, gate=True, scan_period_s=5, cadence_scan=True, roam_cost_s=cost)
        if not policy.power_control:
            raise RuntimeError('Registered mission must enable canonical power-control politeness')
        started = time.perf_counter()
        result = simulate(world, policy, seed=packet_seed, interference=True, roam_gate=reactive_rescue,
            handover_admission=OutageBudgetGate(live.mean, live, cost), forecast_queue=live,
            forecast_source=live, scan_observer=delivered, scan_duration_s=.09,
            native_proposal_ttl_slots=100, managed_probes=runtime, detailed=True)
        local = [r['local_compute_seconds'] for r in predictor.records]
        head_overruns = [i for i, value in enumerate(local) if estimator != 'none' and value > head]
        total_overruns = [i for i, r in enumerate(live.records) if r['actual_prediction_seconds'] +
            (local[i] if estimator == 'none' else 0.) > r['charged_seconds']]
        row = {'schema': 1, 'case': list(key), 'stage': stage, 'site': site, 'packet_seed': packet_seed,
            'handover_seconds': cost, 'arm': arm, 'source_sha256': frozen['source_sha256'],
            'data_sha256': data, 'charged_latency': charged, 'extra_seconds': extra,
            'policy': asdict(policy), 'public_spec': asdict(spec), 'result': result,
            'forecast_timing': live.records, 'predictor_records': predictor.records,
            'forecast_mean_sha256': hashlib.sha256(live.mean.tobytes()).hexdigest(),
            'forecast_std_sha256': hashlib.sha256(live.std.tobytes()).hexdigest(),
            'scan_summary': delivered.summary(), 'probe_reports': [asdict(r) for r in runtime._records],
            'runtime': {'local_head_p95_seconds': float(np.quantile(local, .95)),
                'local_head_max_seconds': max(local), 'head_overrun_decisions': head_overruns,
                'total_prediction_overrun_decisions': total_overruns,
                'wall_seconds': time.perf_counter() - started},
            'integrity': {'causality_and_accounting_pass': True, 'truth_used_as_predictor_input': False},
            'simulation_only': True, 'hardware_qualified': False, 'replay': replay}
        try:
            validate_case(row, p)
            check_sources()
        except Exception as exc:
            row['integrity']['causality_and_accounting_pass'] = False
            row['integrity']['execution_failure'] = str(exc)
            write_once(OUT / 'failures' / (case_name(key) + ('-replay' if replay else '') + '.json.gz'),
                       row, compressed=True)
            raise
        write_once(path, row, compressed=True)
        write_once(path.with_suffix('').with_suffix('.receipt.json'), _receipt(row, path))
        print(json.dumps({'stage': stage, 'site': site, 'packet_seed': packet_seed, 'cost': cost,
            'arm': arm, 'replay': replay, 'misses': result['control_misses'],
            'runtime_overrun': bool(head_overruns or total_overruns)}), flush=True)
        return row


def seal_latency(rows):
    expected = {tuple(key) for key in case_keys('development')}
    if {tuple(row['case']) for row in rows} != expected or len(rows) != len(expected):
        raise RuntimeError('Latency seal requires every development case exactly once')
    maximum = max(r['runtime']['local_head_p95_seconds'] for r in rows
                  if protocol()['arms'][r['arm']][1] != 'none')
    head = max(protocol()['development_local_compute_seconds'], math.ceil(2 * maximum * 100) / 100)
    receipts = {str(Path('cases') / (case_name(key) + '.receipt.json')):
        digest(OUT / 'cases' / (case_name(key) + '.receipt.json')) for key in sorted(expected)}
    write_once(OUT / 'latency.json', {'sealed': True, 'shared_head_seconds': head,
        'maximum_development_local_head_p95_seconds': maximum,
        'rule': 'max(0.1,ceil(2*max(development case p95)*100)/100)', 'packet_outcomes_used': False,
        'source_sha256': check_sources()['source_sha256'], 'development_receipt_sha256': receipts})


def verification_keys():
    p = protocol()
    return [('confirmation', p['confirmation_sites'][0], p['confirmation_packet_seeds'][0], float(cost), arm)
        for cost in p['handover_seconds'] for arm in ('canonical', 'staggered_memory', 'staggered_bayesian')]


def run(stage, workers=1):
    if workers not in (1, 2):
        raise ValueError('Only one or two scientific workers are registered')
    check_sources()
    charges(stage)  # Refuse premature confirmation before opening its sites.
    tasks = [(key, False) for key in case_keys(stage)]
    if workers == 1:
        rows = [case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(case, tasks))
    replay_failed = False
    if stage == 'development':
        seal_latency(rows)
    else:
        replays = [case((key, True)) for key in verification_keys()]
        originals = {tuple(row['case']): row for row in rows}
        comparisons = [{'case': row['case'], 'exact_match': all(
            originals[tuple(row['case'])][field] == row[field] for field in
            ('result', 'data_sha256', 'forecast_mean_sha256', 'forecast_std_sha256'))} for row in replays]
        write_once(OUT / 'replay_validation.json', {'completed': True,
            'source_sha256': check_sources()['source_sha256'], 'comparisons': comparisons,
            'all_exact': all(r['exact_match'] for r in comparisons)})
        replay_failed = not all(r['exact_match'] for r in comparisons)
    write_once(OUT / (stage + '.json'), {'completed': True, 'stage': stage,
        'source_sha256': check_sources()['source_sha256'], 'main_cases': len(rows),
        'receipt_sha256': {case_name(key): digest(OUT / 'cases' / (case_name(key) + '.receipt.json'))
                          for key in case_keys(stage)}})
    if replay_failed:
        raise RuntimeError('Exact verification replay failed; evidence retained for a vetoed report')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('register', 'development', 'confirmation', 'report'))
    ap.add_argument('--workers', type=int, choices=(1, 2), default=1)
    args = ap.parse_args()
    if args.stage == 'register':
        register()
    elif args.stage == 'report':
        from .report import run_report
        run_report()
    else:
        run(args.stage, args.workers)


if __name__ == '__main__':
    main()
