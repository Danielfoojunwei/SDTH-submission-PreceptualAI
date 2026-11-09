"""Verify frozen evidence without simulations, fitting or threshold selection.

Run: uv run --frozen python scripts/verify_continual_packet.py
Use --write-manifest to create the derived verification manifest.
Requires preregistration commit 3ae4ef0 in local Git history.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_COMMIT = '3ae4ef098624aa6544cfc8c77e4106fdcf448a27'
sys.path.insert(0, str(ROOT / 'src'))

from wirelessjev.continual_packet import report as reporting, study


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(*, write_manifest=False):
    frozen = study.check_sources()
    out = study.OUT
    report = json.loads((out / 'report.json').read_text())
    assert report['main_confirmation_cases'] == 192
    assert reporting.run_report() == report, 'Archived report differs from frozen derivation'
    assert report['integrity_pass'] is True
    for name, expected in frozen['source_sha256'].items():
        committed = subprocess.check_output(
            ['git', 'show', REGISTRATION_COMMIT + ':' + name], cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest() == expected, name
        assert (out / 'registration/source' / name).read_bytes() == committed, name

    counts = {}
    for stage in ['development', 'confirmation']:
        charged = study.charges(stage)
        keys = study.case_keys(stage)
        assert all(study.load_case(key, charged=charged) is not None for key in keys)
        counts[stage] = len(keys)
    replay = json.loads((out / 'replay_validation.json').read_text())
    assert replay['all_exact'] is True
    assert len(replay['comparisons']) == 6
    for key in study.verification_keys():
        original = study.load_case(key, charged=study.charges('confirmation'))
        again = study.load_case(key, charged=study.charges('confirmation'), replay=True)
        assert all(original[field] == again[field] for field in
                   ['result', 'data_sha256', 'forecast_mean_sha256', 'forecast_std_sha256'])

    # Check the archived executed tests; this does not invoke fresh tests.
    log = (out / 'logs/preflight-tests.txt').read_text()
    assert re.search(r'^697 passed, 1 warning in 110.45s', log, re.M)
    proof = out / 'verification'
    if write_manifest:
        proof.mkdir(exist_ok=True)
        (proof / 'full-tests.txt').write_text(log)
    else:
        assert (proof / 'full-tests.txt').read_text() == log
    old = json.loads((ROOT / 'artifacts/ran_research/verification.json').read_text())
    for group in ['source_sha256', 'evidence_sha256']:
        for name, expected in old[group].items():
            assert digest(ROOT / name) == expected, name

    sources = [p for folder in ['src', 'scripts', 'tests']
               for p in (ROOT / folder).rglob('*.py') if '__pycache__' not in p.parts]
    sources += [ROOT / 'pyproject.toml', ROOT / 'uv.lock']
    documents = [ROOT / name for name in [
        'README.md', 'docs/HANDOFF.md', 'docs/PAPER_PROPOSAL.md',
        'docs/CONTINUAL_PACKET_RESULTS.md', 'docs/CONTINUAL_PACKET_DESIGN.md',
        'docs/DURABLE_EXECUTION.md', 'docs/PROCESS_ISOLATION.md',
        'docs/PRODUCTION_EXECUTION_FIXES.md', 'docs/PRODUCTION_RUNTIME_FIXES.md',
        'docs/CONTINUAL_RADIO_STATUS.md']]
    files = [p for p in out.rglob('*') if p.is_file()
             and not p.name.endswith(('.lock', '.part')) and not p.name.startswith('.')
             and p != proof / 'manifest.json']
    hashes = lambda paths: {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    value = {
        'schema': 1,
        'scientific_source_commit': REGISTRATION_COMMIT,
        'scientific_source_tree': subprocess.check_output(
            ['git', 'rev-parse', REGISTRATION_COMMIT + '^{tree}'], cwd=ROOT, text=True).strip(),
        'full_tests': {'passed': 697, 'warnings': 1, 'seconds': 110.45, 'hardware_test': False},
        'main_cases': counts, 'exact_replays': 6,
        'all_registration_source_checks_passed': True,
        'all_case_hashes_and_accounting_validated': True,
        'previous_PR17_source_and_evidence_hashes_unchanged': True,
        'report_sha256': digest(out / 'report.json'),
        'experiment_service_gate_by_estimator': report['service_qualified_by_estimator'],
        'independent_information_gate_by_estimator': report['independent_information_pass_by_estimator'],
        'hardware_qualified': False, 'production_qualified': False, 'new_neural_models_trained': 0,
        'reliability_components': {
            'separate_from_scientific_source_capsule': True,
            'journal_source': 'src/wirelessjev/execution_journal.py',
            'worker_source': 'src/wirelessjev/process_worker.py',
            'activated_in_existing_services': False},
        'limits': [
            'Procedural shared-CAD warehouse; not measured 100-car or 100-robot RT evidence',
            'AP-assisted carrier sense, slot-average PHY and dedicated monitoring assumptions',
            'RF forecast correction only; estimated busy airtime is diagnostic',
            'No JEPA superiority, global SOTA or distribution-free safety claim'],
        'sources_sha256': hashes(sources),
        'documents_sha256': hashes(documents),
        'evidence_sha256': hashes(files)}
    manifest_path = proof / 'manifest.json'
    if write_manifest:
        value['verified_at_utc'] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(value, indent=2) + '\n')
    else:
        archived = json.loads(manifest_path.read_text())
        archived.pop('verified_at_utc')
        assert archived == value, 'Final manifest differs from current source/evidence'
    print(json.dumps({key: value[key] for key in [
        'main_cases', 'exact_replays', 'experiment_service_gate_by_estimator',
        'production_qualified']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true')
    args = parser.parse_args()
    verify(write_manifest=args.write_manifest)
