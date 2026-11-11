"""Prospective slow-cost recovery identities; parent results remain frozen."""
from pathlib import Path
import json
import shutil
import time
import math
import numpy as np
from ..jepa_repair import common as parent
from ..jepa_repair import rollout as parent_rollout

ROOT = parent.ROOT
OUT = ROOT / 'artifacts/costaware_recovery'
PROTOCOL = OUT / 'protocol.json'
digest = parent.digest
write_once = parent.write_once
write_text_once = parent.write_text_once
value_digest = parent.value_digest
lock = parent.lock
versions = parent.versions


def read_json(path):
    return json.loads(Path(path).read_text())


def protocol():
    return read_json(PROTOCOL)


def latency():
    return parent.latency()


def checkpoint_hashes():
    return parent.checkpoint_hashes()


def parent_development_receipts():
    """Pin twenty completed development receipts, never ongoing confirmation."""
    expected = {parent_rollout.case_name(key) + '.receipt.json'
                for key in parent_rollout.case_keys('development')}
    paths = sorted((parent.OUT / 'cases').glob('development-*.receipt.json'))
    if len(expected) != 20 or {path.name for path in paths} != expected:
        raise RuntimeError('Require exactly twenty completed parent development receipts')
    return paths


def source_hashes():
    inherited = parent.source_hashes()
    paths = list((ROOT / 'src/wirelessjev/costaware_recovery').glob('*.py'))
    paths += list((ROOT / 'tests').glob('test_costaware_recovery*.py'))
    paths += [PROTOCOL, OUT / 'execution_order.json',
              parent.OUT / 'development.json', parent.OUT / 'qualification.json']
    paths += parent_development_receipts()
    return inherited | {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def _integers(value):
    if isinstance(value, dict):
        return set().union(*(_integers(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_integers(v) for v in value)) if value else set()
    return {value} if type(value) is int else set()


def verify_fresh_sites():
    p = protocol()
    new = p['development_sites'] + p['confirmation_sites']
    if len(set(new)) != 10 or len(p['development_sites']) != 2 or len(p['confirmation_sites']) != 8:
        raise RuntimeError('Require ten disjoint newly registered sites')
    for path in (ROOT / 'artifacts').glob('*/protocol.json'):
        if path != PROTOCOL and set(new) & _integers(read_json(path)):
            raise RuntimeError('Recovery sites overlap an existing protocol: ' + str(path))


def register():
    parent.check_sources()
    parent.verified_development()
    verify_fresh_sites()
    if any((OUT / folder).exists() for folder in ('cases', 'replays', 'live_replications')):
        raise RuntimeError('Registration must precede every new evaluation')
    with lock(OUT / 'registration.lock'):
        write_once(OUT / 'execution_order.json', execution_order())
        sources = source_hashes()
        receipt = {'schema': 1, 'source_sha256': sources,
            'new_evaluation_data_accessed': False, 'source_training_and_results_already_observed': True,
            'new_fits': 0, 'parent_development_already_observed': True,
            'selection_scope': 'Slow-cost recovery chosen after parent development veto; no training or parent-contract change.',
            'parent_development_sha256': digest(parent.OUT / 'development.json'),
            'parent_qualification_sha256': digest(parent.OUT / 'qualification.json'),
            'checkpoint_sha256': checkpoint_hashes(),
            'software_versions': versions()}
        path = OUT / 'registration/source.json'
        if path.exists():
            existing = read_json(path)
            if any(existing[key] != value for key, value in receipt.items()):
                raise RuntimeError('Existing recovery registration differs')
            return check_sources()
        for name, expected in sources.items():
            dest = OUT / 'registration/source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
            if digest(dest) != expected:
                raise RuntimeError('Source changed during registration')
        receipt['time_unix'] = time.time()
        write_once(path, receipt)
    return receipt


def execution_order():
    p = protocol()
    value = {'algorithm': 'NumPy default_rng fixed permutation within site blocks',
             'seed': p['execution_seed']}
    for offset, stage in enumerate(('development', 'confirmation')):
        keys = []
        for index, site in enumerate(p[stage + '_sites']):
            block = [(stage, site, float(cost), family, variant)
                for cost in p['handover_seconds'] for family in p['families']
                for variant in p['variants_by_cost'][str(float(cost))]]
            rng = np.random.default_rng(p['execution_seed'] + offset * 100 + index)
            keys.extend([list(block[i]) for i in rng.permutation(len(block))])
        value[stage] = keys
    return value


def development_validation(rows):
    return {'source_and_archive_integrity': True,
        'causality_and_kernel_checks': all(row['causality_pass'] for row in rows),
        'paid_cost_and_consumption_checks': all(row['paid_cost_and_consumption_pass'] for row in rows)}


def verified_development():
    from . import rollout, report
    rows = [rollout.load_case(key) for key in rollout.case_keys('development')]
    if any(row is None for row in rows):
        raise RuntimeError('Complete the registered development matrix first')
    validation = development_validation(rows)
    gate = report.development_gate(rows, protocol(), validation=validation)
    expected = {'gate': gate, 'validation': validation, 'source_sha256': source_hashes(),
                'execution_order_sha256': digest(OUT / 'execution_order.json')}
    if read_json(OUT / 'development.json') != expected:
        raise RuntimeError('Development seal differs from recomputed raw cases')
    permission = {'development_sha256': digest(OUT / 'development.json'),
        'source_sha256': source_hashes(), 'passed_costs': gate['passed_costs'],
        'recomputed_from_raw_archives': True}
    if read_json(OUT / 'qualification.json') != permission:
        raise RuntimeError('Confirmation permission differs from raw development')
    return gate


def check_confirmation_permission(cost):
    gate = verified_development()
    if float(cost) not in gate['passed_costs']:
        raise RuntimeError('This cost was vetoed before confirmation')
    return gate


def verify_paid_row(row):
    """Independently check fresh actual-cost ledgers and actual consumption."""
    p = protocol()
    profile = latency()['profiles'][row['family']]
    jobs = row['jobs']
    if len(jobs) != p['duration_seconds']:
        raise RuntimeError('Incomplete actual-cost request ledger')
    previous = 0.
    for index, job in enumerate(jobs):
        if job['decision'] != index or job['history_cutoff_exclusive'] != 60 + index:
            raise RuntimeError('Request ledger crosses the observable-history boundary')
        times = [job[name] for name in ('actual_prediction_seconds', 'actual_issue_seconds',
                                       'actual_compute_seconds', 'charged_compute_seconds', 'charged_seconds')]
        if any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times):
            raise RuntimeError('Nonfinite/negative actual-cost ledger')
        actual = max(job['actual_prediction_seconds'], job['actual_issue_seconds'])
        compute = max(profile[job['kind']], actual)
        charged = compute + p['extra_compute_seconds']
        if any(not math.isclose(job[field], expected, abs_tol=1e-12, rel_tol=1e-12)
               for field, expected in [('actual_compute_seconds', actual),
                    ('charged_compute_seconds', compute), ('charged_seconds', charged)]):
            raise RuntimeError('Forecast issue was not charged its measured actual cost')
        if job['runtime_overrun'] is not (actual > profile[job['kind']]):
            raise RuntimeError('Capacity-envelope flag differs from measured cost')
        previous = max(float(index), previous) + charged
        if job['ready_slot'] != math.ceil(previous * 100 - 1e-10):
            raise RuntimeError('Actual-cost FIFO was not preserved')
    last_slot, last_job = -1, -1
    for event in row['forecast_consumption']:
        slot, index = event['slot'], event['forecast_decision']
        if (type(slot) is not int or type(index) is not int or not last_slot < slot < p['duration_seconds'] * 100
                or not last_job < index < len(jobs) or event['ready_slot'] != jobs[index]['ready_slot']
                or slot < jobs[index]['ready_slot']
                or not math.isclose(event['age_seconds'], slot / 100 - index, abs_tol=1e-12, rel_tol=0)):
            raise RuntimeError('A forecast was consumed before its accounted completion')
        last_slot, last_job = slot, index
    if row['paid_cost_and_consumption_pass'] is not True:
        raise RuntimeError('Fresh rollout failed its actual-cost boundary')
    return True


def check_sources():
    parent.check_sources()
    path = OUT / 'registration/source.json'
    if not path.exists():
        raise RuntimeError('Register and publish recovery sources before evaluating')
    receipt = read_json(path)
    if receipt['source_sha256'] != source_hashes() or receipt['checkpoint_sha256'] != checkpoint_hashes():
        raise RuntimeError('Registered recovery source/input/weight identity changed')
    for name, expected in receipt['source_sha256'].items():
        if digest(OUT / 'registration/source' / name) != expected:
            raise RuntimeError('Recovery source capsule changed: ' + name)
    return receipt
