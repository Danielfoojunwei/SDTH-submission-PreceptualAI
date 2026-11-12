"""Archive verification only; a missing completion is never reported complete."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/service_world'
SPEC = importlib.util.spec_from_file_location('service_world_archive_audit', ROOT / 'scripts/audit_service_world.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def completed_archive():
    if not (OUT / 'completion.json').exists():
        pytest.skip('Native service-world diagnostic has not completed; no archive claim yet')


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def test_completed_native_diagnostic_preserves_contract_and_holdouts():
    completed_archive()
    result = AUDIT.audit()
    assert result['passed'] and result['original_holdouts_unused']
    assert result['trained_models'] == result['learned_policy_evaluations'] == 0
    assert not result['novelty_established'] and not result['sota_established']
    assert result['cases'] > 0 and result['factual_replays'] > 0


@pytest.fixture
def archived_copy(tmp_path):
    """Tamper a temporary evidence copy, never an executed study or holdout."""
    completed_archive()
    destination = tmp_path / 'artifacts/service_world'
    shutil.copytree(OUT, destination)
    source = AUDIT.read(OUT / 'executed_source.json')['source_sha256']
    for name in source:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    if (OUT / 'summary.json').exists():
        reporter = tmp_path / 'scripts/report_service_world.py'
        reporter.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'scripts/report_service_world.py', reporter)
    # Registration-only reads establish seed separation; no held-out case is
    # copied or opened by this test or by the auditor.
    for registration in (ROOT / 'artifacts').glob('*/protocol.json'):
        if registration.parent.name != 'service_world':
            target = tmp_path / registration.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(registration, target)
    assert AUDIT.audit(tmp_path)['passed']
    return tmp_path


@pytest.mark.parametrize('mutation', [
    'copied_source', 'current_source', 'input_digest', 'miss_delta',
    'long_hold', 'replay_count', 'trained_prototype_claim', 'case_source',
    'completion_count', 'novelty_claim', 'case_cost',
])
def test_archive_auditor_rejects_modified_evidence(archived_copy, mutation):
    root = archived_copy
    out = root / 'artifacts/service_world'
    path = next((out / 'cases').glob('*.json'))
    case = AUDIT.read(path)
    completion = AUDIT.read(out / 'completion.json')
    source_name = 'src/wirelessjev/service_world/physics.py'
    if mutation in ('copied_source', 'current_source'):
        target = (out / 'executed_source' if mutation == 'copied_source' else root) / source_name
        target.write_bytes(target.read_bytes() + b'\n# Modified after execution\n')
    elif mutation == 'input_digest':
        target = out / 'inputs' / (path.stem + '.npz')
        target.write_bytes(target.read_bytes() + b'altered evidence')
    elif mutation == 'miss_delta':
        case['records'][0]['delta_misses'] += 1
        write(path, case)
    elif mutation == 'long_hold':
        case['records'][0]['hold_slots'] = 1000
        write(path, case)
    elif mutation == 'replay_count':
        case['factual_replays'] += 1
        write(path, case)
    elif mutation == 'trained_prototype_claim':
        case['prototype']['accuracy_or_control_result'] = True
        write(path, case)
    elif mutation == 'case_source':
        case['source_sha256'][source_name] = '0' * 64
        write(path, case)
    elif mutation == 'completion_count':
        completion['cases'] += 1
        write(out / 'completion.json', completion)
    elif mutation == 'novelty_claim':
        completion['novelty_established'] = True
        write(out / 'completion.json', completion)
    elif mutation == 'case_cost':
        case['cost'] = 5.0
        write(path, case)
    with pytest.raises(AUDIT.AuditError):
        AUDIT.audit(root)


def test_cli_exits_nonzero_when_completion_totals_are_altered(archived_copy):
    completion = archived_copy / 'artifacts/service_world/completion.json'
    value = AUDIT.read(completion)
    value['native_runs'] += 1
    write(completion, value)
    process = subprocess.run([sys.executable, str(ROOT / 'scripts/audit_service_world.py'),
                              '--root', str(archived_copy)], capture_output=True, text=True)
    assert process.returncode != 0
    assert json.loads(process.stderr)['passed'] is False


def test_optional_summary_cannot_turn_diagnostics_into_novelty_or_policy_evidence():
    totals = dict(cases=8, native_runs=100, factual_replays=24, records=60, input_windows=48)
    AUDIT.check_optional_summary({'native_runs': 100, 'sota_established': False}, totals)
    for summary in ({'native_runs': 101}, {'trained_models': 1},
                    {'prototype': {'learned_policy_qualified': True}},
                    {'sota_established': True}):
        with pytest.raises(AUDIT.AuditError):
            AUDIT.check_optional_summary(summary, totals)


@pytest.mark.parametrize('observed_slot,age,valid', [(550,.08,True),(559,-.01,False),
                                                  (550,.09,False),(548,.1,False)])
def test_public_selection_timestamp_requires_fresh_causal_evidence(tmp_path,observed_slot,age,valid):
    path=tmp_path/'input.npz'
    links=np.zeros((1,2,2,3,12),dtype='f4')
    np.savez_compressed(path,slot=np.array([558]),robot=np.array([0]),links=links,
        globals=np.zeros((1,2,8),dtype='f4'),observed=np.zeros_like(links,dtype=bool),
        elapsed=np.full((1,2),.1,dtype='f4'),selection_observation_slot=np.array([observed_slot]),
        decision_age_seconds=np.array([age]))
    case={'inputs_sha256':AUDIT.digest(path),'input_windows':1}
    if valid:
        keys,aps,selection=AUDIT.public_input_keys(path,case,{'robots':100})
        assert keys=={(558,0)} and aps==2 and selection=={(558,0):550}
    else:
        with pytest.raises(AUDIT.AuditError):
            AUDIT.public_input_keys(path,case,{'robots':100})
