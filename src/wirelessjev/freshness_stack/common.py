"""Fresh stack identities; earlier registered source and evidence stay frozen."""
from pathlib import Path
import json
import shutil
import time
import numpy as np
from ..costaware_recovery import common as parent

ROOT = parent.ROOT
OUT = ROOT / 'artifacts/freshness_stack'
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


def prior_inputs():
    """Pin completed prior outcomes and the passive diagnoses used to design this study."""
    return [ROOT / name for name in (
        'artifacts/jepa_repair/report.json',
        'artifacts/costaware_recovery/report.json',
        'artifacts/jepa_repair/verification/manifest.json',
        'artifacts/costaware_recovery/verification/manifest.json',
        'scripts/diagnose_forecast_freshness.py',
        'artifacts/forecast_freshness_diagnosis/diagnosis.json',
        'artifacts/forecast_freshness_diagnosis/receipt.json',
        'scripts/diagnose_stack_sensing.py',
        'artifacts/stack_sensing_diagnosis/diagnosis.json',
        'artifacts/stack_sensing_diagnosis/receipt.json')]


def _verify_digest_map(entries, message):
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError('Missing preserved input digest entries')
    for name, expected in entries.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or digest(ROOT / path) != expected:
            raise RuntimeError(message + str(name))


def verify_previous_evidence():
    """Prior documents apply to their commit; old source/evidence stay byte-identical."""
    counts, unique = {}, {}
    for study in ('jepa_repair', 'costaware_recovery'):
        manifest = read_json(ROOT / 'artifacts' / study / 'verification/manifest.json')
        counts[study] = {}
        for field in ('sources_sha256', 'evidence_sha256'):
            entries = manifest[field]
            if not isinstance(entries, dict) or not entries:
                raise RuntimeError('Missing previous source/evidence manifest entries')
            for name, expected in entries.items():
                if name in unique and unique[name] != expected:
                    raise RuntimeError('Conflicting previous evidence identities: ' + name)
                unique[name] = expected
            counts[study][field] = len(entries)
    _verify_digest_map(unique, 'Previously completed source/evidence changed: ')
    return counts


def verify_diagnosis_inputs():
    """Receipts bind already preserved prior data without redundant nested copies."""
    counts, unique = {}, {}
    for name in ('forecast_freshness', 'stack_sensing'):
        folder = ROOT / 'artifacts' / (name + '_diagnosis')
        receipt = read_json(folder / 'receipt.json')
        script = 'scripts/diagnose_' + name + '.py'
        unique[script] = receipt['script_sha256']
        counts[name] = {}
        for field in ('inputs_sha256', 'outputs_sha256'):
            entries = receipt[field]
            if not isinstance(entries, dict) or not entries:
                raise RuntimeError('Missing passive diagnosis input/output identities')
            for path, expected in entries.items():
                if path in unique and unique[path] != expected:
                    raise RuntimeError('Conflicting diagnosis identities: ' + path)
                unique[path] = expected
            counts[name][field] = len(entries)
    _verify_digest_map(unique, 'Passive diagnosis source/input/output changed: ')
    return counts


def source_hashes():
    inherited = parent.source_hashes()
    paths = list((ROOT / 'src/wirelessjev/freshness_stack').glob('*.py'))
    paths += list((ROOT / 'tests').glob('test_freshness_stack*.py'))
    paths += [PROTOCOL, OUT / 'execution_order.json'] + prior_inputs()
    return inherited | {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def _integers(value):
    return parent._integers(value)


def verify_fresh_sites():
    p = protocol()
    new = p['development_sites'] + p['confirmation_sites']
    if len(set(new)) != 10 or len(p['development_sites']) != 2 or len(p['confirmation_sites']) != 8:
        raise RuntimeError('Require ten disjoint newly registered sites')
    for path in (ROOT / 'artifacts').glob('*/protocol.json'):
        if path != PROTOCOL and set(new) & _integers(read_json(path)):
            raise RuntimeError('Freshness sites overlap an existing protocol: ' + str(path))


def register():
    parent.check_sources()
    previous = verify_previous_evidence()
    diagnoses = verify_diagnosis_inputs()
    verify_fresh_sites()
    if any((OUT / folder).exists() for folder in ('cases', 'replays', 'live_replications')):
        raise RuntimeError('Registration must precede every new evaluation')
    with lock(OUT / 'registration.lock'):
        write_once(OUT / 'execution_order.json', execution_order())
        sources = source_hashes()
        receipt = {'schema': 1, 'source_sha256': sources,
            'new_evaluation_data_accessed': False, 'source_training_and_results_already_observed': True,
            'new_fits': 0, 'parent_results_already_observed': True,
            'selection_scope': 'Fixed nested forecast/PHY/sensing stack chosen using completed prior outcomes; no new model or parent-contract change.',
            'previous_global_document_hashes_apply_to_original_commits': True,
            'previous_source_and_evidence_preserved': previous,
            'passive_diagnosis_source_inputs_and_outputs_verified': diagnoses,
            'observed_input_sha256': {str(path.relative_to(ROOT)): digest(path) for path in prior_inputs()},
            'checkpoint_sha256': checkpoint_hashes(), 'software_versions': versions()}
        path = OUT / 'registration/source.json'
        if path.exists():
            existing = read_json(path)
            if any(existing[key] != value for key, value in receipt.items()):
                raise RuntimeError('Existing freshness registration differs')
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
    value = {'algorithm': 'NumPy default_rng fixed permutation within site blocks', 'seed': p['execution_seed']}
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


def causal_history(row):
    """Recompute all request/source identities rather than trusting a saved boolean."""
    try:
        duration = protocol()['duration_seconds']
        jobs, arrivals = row['jobs'], row['arrivals']
        mode = 'fifo' if row['controller_variant'] == 'fifo' else 'latest'
        if row['queue_mode'] != mode or len(arrivals) != duration or not jobs:
            return False
        accepted = []
        for decision, arrival in enumerate(arrivals):
            if (type(arrival['decision']) is not int or arrival['decision'] != decision
                    or type(arrival['arrival_slot']) is not int or arrival['arrival_slot'] != decision * 100
                    or type(arrival['accepted']) is not bool or arrival['mode'] != mode):
                return False
            if arrival['accepted']:
                if type(arrival['job_index']) is not int or arrival['job_index'] != len(accepted):
                    return False
                accepted.append(decision)
            elif arrival['job_index'] is not None or mode == 'fifo':
                return False
        if len(jobs) != len(accepted) or row['accepted_decisions'] != accepted:
            return False
        if any(type(job['decision']) is not int or job['decision'] != source
               or type(job['source_decision']) is not int or job['source_decision'] != source
               or type(job['job_index']) is not int or job['job_index'] != index
               or type(job['history_cutoff_exclusive']) is not int
               or job['history_cutoff_exclusive'] != 60 + source or job['mode'] != mode
               for index, (job, source) in enumerate(zip(jobs, accepted))):
            return False
        return all(event['measurement_slot'] < event['complete_slot'] <= event['delivery_slot']
            and event['first_usable_decision'] == event['delivery_slot'] // 100 + 1
            for event in row['telemetry']['events'])
    except (KeyError, TypeError, ValueError):
        return False


def development_validation(rows):
    return {'source_and_archive_integrity': True,
        'causality_and_kernel_checks': all(causal_history(row) and row['causality_pass'] for row in rows),
        'paid_cost_and_consumption_checks': all(verify_paid_row(row) for row in rows)}


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
    """Recheck sparse jobs, all arrivals and every actual consumption independently of their producer."""
    from . import rollout
    if not causal_history(row):
        raise RuntimeError('Request ledger crosses the observable-history boundary')
    p = protocol()
    audit = rollout.timing_audit(row['jobs'], row['forecast_consumption'], latency()['profiles'][row['family']],
        p['extra_compute_seconds'], p['duration_seconds'], arrivals=row['arrivals'], mode=row['queue_mode'])
    if (audit['passes'] is not True or row['paid_cost_and_consumption_pass'] is not True
            or audit['real_time_contract_checked'] is not True or audit['offline_schedule_only'] is not False):
        raise RuntimeError('Fresh rollout failed its actual-cost boundary')
    return True


def check_sources():
    parent.check_sources()
    path = OUT / 'registration/source.json'
    if not path.exists():
        raise RuntimeError('Register and publish freshness sources before evaluating')
    receipt = read_json(path)
    if verify_previous_evidence() != receipt['previous_source_and_evidence_preserved']:
        raise RuntimeError('Previous completed source/evidence membership changed')
    if verify_diagnosis_inputs() != receipt['passive_diagnosis_source_inputs_and_outputs_verified']:
        raise RuntimeError('Passive diagnosis membership changed')
    if receipt['source_sha256'] != source_hashes() or receipt['checkpoint_sha256'] != checkpoint_hashes():
        raise RuntimeError('Registered freshness source/input/weight identity changed')
    for name, expected in receipt['source_sha256'].items():
        if digest(OUT / 'registration/source' / name) != expected:
            raise RuntimeError('Freshness source capsule changed: ' + name)
    return receipt
