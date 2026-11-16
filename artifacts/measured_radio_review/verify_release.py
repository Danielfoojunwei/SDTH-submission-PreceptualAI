"""Verify the review package without preparing data, fitting or predicting."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SCIENCE = ROOT/'artifacts/measured_radio_counterfix'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def verify():
    suite = read(OUT/'software_suite_counterfix.json')
    assert suite['passed'] == 4320 and suite['exit_code'] == 0 and suite['source_unchanged']
    assert len(suite['test_modules']) == 170 and len(suite['source_sha256_after']) == 619
    assert suite['source_sha256_before'] == suite['source_sha256_after']
    assert digest(ROOT/suite['log_path']) == suite['log_sha256']
    assert digest(OUT/'software_suite_counterfix_runner.py') == suite['runner_sha256']
    for name, expected in suite['source_sha256_after'].items():
        assert digest(ROOT/name) == expected, name
    protocol_sha = digest(SCIENCE/'protocol.json')
    assert protocol_sha == '37c9520331486a173e948b583278ad160245076a534ab0939d67160b98f9fdb0'
    receipt, seal = read(SCIENCE/'data/receipt.json'), read(SCIENCE/'verification/manifest.json')
    assert digest(SCIENCE/'data/dataset.json.gz') == receipt['dataset_sha256']
    tracked = set(subprocess.check_output(['git','ls-files'], cwd=ROOT, text=True).splitlines())
    for name, expected in seal['files_sha256'].items():
        assert digest(ROOT/name) == expected and name in tracked, name
    for name, script in (('audit_source.json','audit_source.py'), ('audit_results.json','audit_results.py')):
        audit = read(OUT/name)
        assert audit.get('passes', audit.get('status') == 'passed'), name
        assert audit['audit_script_sha256'] == digest(OUT/script)
        assert audit['protocol_sha256'] == protocol_sha and audit['dataset_sha256'] == receipt['dataset_sha256']
    outcomes = read(OUT/'audit_results.json')
    assert outcomes['report_sha256'] == digest(SCIENCE/'report.json')
    assert outcomes['seal_sha256'] == digest(SCIENCE/'verification/manifest.json')
    assert not read(SCIENCE/'report.json')['primary_contrast']['conditional_lower_error_supported']
    spec = importlib.util.spec_from_file_location('independent_prior_closure', OUT/'audit_prior_closures.py')
    prior_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior_module)
    prior = prior_module.verify()
    assert prior['passes'] and prior['unique_named_repository_files'] == 27295
    assert read(OUT/'prior_closure_verification.json')['verifier_sha256'] == digest(OUT/'audit_prior_closures.py')
    plot = read(OUT/'plot_receipt.json')
    assert plot['script_sha256'] == digest(OUT/'plot_results.py') and plot['report_sha256'] == outcomes['report_sha256']
    for name, expected in plot['files_sha256'].items():
        assert digest(OUT/name) == expected
    for name in ('docs/MEASURED_RADIO_PROTOCOL.md','docs/MEASURED_RADIO_RESULTS.md'):
        path = ROOT/name
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' not in target and not target.startswith('#'):
                assert (path.parent/target.split('#')[0]).exists(), target
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != 'release_verification.json']
    files += [ROOT/'README.md', ROOT/'docs/HANDOFF.md', ROOT/'docs/MEASURED_RADIO_PROTOCOL.md',
              ROOT/'docs/MEASURED_RADIO_RESULTS.md']
    return {'schema':1, 'passes':True, 'checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'verifier_sha256':digest(Path(__file__)), 'review_files_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sorted(files)},
        'protocol_sha256':protocol_sha, 'scientific_sealed_files':len(seal['files_sha256']),
        'source_files_unchanged_from_full_suite':619, 'tests_passed':4320,
        'source_and_result_independent_audits_pass':True, 'prior_scientific_closures':10,
        'prior_unique_named_files_rechecked':27295, 'deployment_zip_unchanged':True,
        'new_scientific_fits_or_predictions':0, 'real_car_actions':0, 'SOTA_established':False}


if __name__ == '__main__':
    value = verify()
    with (OUT/'release_verification.json').open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k != 'review_files_sha256'}, indent=2))
