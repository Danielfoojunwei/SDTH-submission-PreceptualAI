"""Pilot/fast-control identities; all previously registered science remains frozen."""
from pathlib import Path
import json
import hashlib
import subprocess
import shutil
import time
import numpy as np
from ..freshness_stack import common as parent

ROOT = parent.ROOT
OUT = ROOT / 'artifacts/pilot_fastloop'
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
    """Bind the completed stack and the passive evidence motivating these controls."""
    return [ROOT / name for name in (
        'artifacts/freshness_stack/report.json',
        'artifacts/freshness_stack/verification/manifest.json',
        'scripts/audit_jepa_advantages.py',
        'artifacts/jepa_advantage_diagnosis/diagnosis.json',
        'artifacts/jepa_advantage_diagnosis/receipt.json')]


BASE_COMMIT = '9a8577d2f5334bbf1f4ea805811b40c35a11212a'


def historical_bytes(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise RuntimeError('Unsafe historical source identity')
    return subprocess.check_output(['git', 'show', BASE_COMMIT + ':' + str(path)], cwd=ROOT)


def _verify_digest_map(entries, message):
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError('Missing preserved input digest entries')
    for name, expected in entries.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or digest(ROOT / path) != expected:
            raise RuntimeError(message + str(name))


def verify_previous_evidence():
    """Anchor PR21's manifest in Git; preserve its current source/evidence bytes.

    Narrative documents are checked at their original published commit. They
    may describe the new study in the current checkout without rewriting history.
    """
    counts = parent.verify_previous_evidence()
    name = 'artifacts/freshness_stack/verification/manifest.json'
    raw = historical_bytes(name)
    if (ROOT / name).read_bytes() != raw:
        raise RuntimeError('Published freshness manifest changed')
    manifest = json.loads(raw)
    preserved = {}
    counts['freshness_stack'] = {}
    for field in ('sources_sha256', 'evidence_sha256'):
        entries = manifest[field]
        _verify_digest_map(entries, 'Previously completed source/evidence changed: ')
        counts['freshness_stack'][field] = len(entries)
        for path, expected in entries.items():
            if path in preserved and preserved[path] != expected:
                raise RuntimeError('Conflicting previous evidence identities: ' + path)
            preserved[path] = expected
    docs = manifest['documents_sha256']
    if not isinstance(docs, dict) or not docs:
        raise RuntimeError('Missing historical document identities')
    for path, expected in docs.items():
        if hashlib.sha256(historical_bytes(path)).hexdigest() != expected:
            raise RuntimeError('Historical published document changed: ' + path)
    counts['freshness_stack']['historical_documents_sha256'] = len(docs)
    return counts


def verify_diagnosis_inputs():
    """Receipt verification is passive: no worlds, labels, models or benchmarks."""
    counts = parent.verify_diagnosis_inputs()
    receipt = read_json(ROOT / 'artifacts/jepa_advantage_diagnosis/receipt.json')
    if (receipt.get('base_commit') != BASE_COMMIT
            or any(receipt.get(name) is not True for name in (
                'passive_only', 'no_worlds_forwards_fits_or_timing',
                'source_and_capsules_verified_before_and_after'))):
        raise RuntimeError('Passive advantage diagnosis changed its historical base or passive scope')
    entries = {'scripts/audit_jepa_advantages.py': receipt['script_sha256']}
    counts['jepa_advantage'] = {}
    for field in ('inputs_sha256', 'outputs_sha256'):
        current = receipt[field]
        if not isinstance(current, dict) or not current:
            raise RuntimeError('Missing passive advantage diagnosis identities')
        for path, expected in current.items():
            if path in entries and entries[path] != expected:
                raise RuntimeError('Conflicting passive advantage diagnosis identities')
            entries[path] = expected
        counts['jepa_advantage'][field] = len(current)
    _verify_digest_map(entries, 'Passive diagnosis source/input/output changed: ')
    return counts


def source_hashes():
    inherited = parent.source_hashes()
    paths = list((ROOT / 'src/wirelessjev/pilot_fastloop').glob('*.py'))
    paths += list((ROOT / 'tests').glob('test_pilot_fastloop*.py'))
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
            raise RuntimeError('Pilot/fast-control sites overlap an existing protocol: ' + str(path))


def register():
    parent.check_sources()
    previous = verify_previous_evidence()
    diagnoses = verify_diagnosis_inputs()
    verify_fresh_sites()
    if any((OUT / folder).exists() for folder in ('cases', 'replays', 'live_replications', 'matched')):
        raise RuntimeError('Registration must precede every new evaluation')
    with lock(OUT / 'registration.lock'):
        write_once(OUT / 'execution_order.json', execution_order())
        sources = source_hashes()
        receipt = {'schema': 1, 'source_sha256': sources,
            'new_evaluation_data_accessed': False, 'source_training_and_results_already_observed': True,
            'new_fits': 0, 'parent_results_already_observed': True,
            'selection_scope': 'Fixed positive pilot reporting and local forecast-control factorial; matched offline RF endpoint; no new model, tuning, intermediate selection or parent-contract change.',
            'previous_global_document_hashes_apply_to_original_commits': True,
            'previous_source_and_evidence_preserved': previous,
            'passive_diagnosis_source_inputs_and_outputs_verified': diagnoses,
            'observed_input_sha256': {str(path.relative_to(ROOT)): digest(path) for path in prior_inputs()},
            'checkpoint_sha256': checkpoint_hashes(), 'software_versions': versions()}
        path = OUT / 'registration/source.json'
        if path.exists():
            existing = read_json(path)
            if any(existing[key] != value for key, value in receipt.items()):
                raise RuntimeError('Existing pilot/fast-control registration differs')
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
        matched = []
        for index, site in enumerate(p[stage + '_sites']):
            block = [(stage, site, family) for family in p['families']]
            rng = np.random.default_rng(p['execution_seed'] + 1000 + offset * 100 + index)
            matched.extend([list(block[i]) for i in rng.permutation(len(block))])
        value['matched_' + stage] = matched
    return value


def causal_history(row):
    """Recompute all request/source identities rather than trusting a saved boolean."""
    try:
        duration = protocol()['duration_seconds']
        jobs, arrivals = row['jobs'], row['arrivals']
        mode = 'latest'
        if row['controller_variant'] not in protocol()['variants']:
            return False
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
        if not all(event['measurement_slot'] < event['complete_slot'] <= event['delivery_slot']
            and event['first_usable_decision'] == event['delivery_slot'] // 100 + 1
            for event in row['telemetry']['events']):
            return False
        from .rollout import _causal_telemetry
        p = protocol()
        return (row['pilot_enabled'] is p['variants'][row['controller_variant']]['pilot_reporting']
            and row['fast_local_enabled'] is p['variants'][row['controller_variant']]['fast_local']
            and _causal_telemetry(row['telemetry'], row['pilot_reporting'], robots=p['robots'],
                duration=p['duration_seconds'], ap_count=row['access_points']) is True)
    except (KeyError, TypeError, ValueError):
        return False


def matched_validation(rows):
    """Matched proofs come from raw cache/array validation, never saved metrics."""
    from . import matched
    for row in rows:
        if matched.validate_row(row) is not True:
            raise RuntimeError('Matched source/causality/input/prediction validation failed')
    return {'matched_source_and_archive_integrity': True,
        'matched_causality_checks': True, 'matched_metric_integrity': True}


def development_validation(rows, matched_rows):
    return {'source_and_archive_integrity': True,
        'causality_and_kernel_checks': all(causal_history(row) and row['causality_pass'] is True for row in rows),
        'paid_cost_and_consumption_checks': all(verify_paid_row(row) for row in rows),
        **matched_validation(matched_rows)}


def verified_development():
    from . import rollout, report, study
    keys = study.ordered_keys('development')
    rows = study._load(keys)
    study._fresh(rows, keys)
    matched_rows = study._load_matched(study.ordered_matched_keys('development'))
    validation = development_validation(rows, matched_rows)
    gate = report.development_gate(rows, protocol(), validation=validation, matched_rows=matched_rows)
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
        raise RuntimeError('Register and publish pilot/fast-control sources before evaluating')
    receipt = read_json(path)
    if verify_previous_evidence() != receipt['previous_source_and_evidence_preserved']:
        raise RuntimeError('Previous completed source/evidence membership changed')
    if verify_diagnosis_inputs() != receipt['passive_diagnosis_source_inputs_and_outputs_verified']:
        raise RuntimeError('Passive diagnosis membership changed')
    if receipt['source_sha256'] != source_hashes() or receipt['checkpoint_sha256'] != checkpoint_hashes():
        raise RuntimeError('Registered pilot/fast-control source/input/weight identity changed')
    for name, expected in receipt['source_sha256'].items():
        if digest(OUT / 'registration/source' / name) != expected:
            raise RuntimeError('Pilot/fast-control source capsule changed: ' + name)
    return receipt
