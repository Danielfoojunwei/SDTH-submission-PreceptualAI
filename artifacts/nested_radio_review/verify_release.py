"""Passive final release checks; no model, fitting, tracing or simulation imports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / 'artifacts/nested_radio_review'
STUDIES = ('radiojepa', 'radiojepa_spectral', 'device_transfer', 'channel_scene',
           'adaptive_radio', 'native_bridge', 'e7_calplanning', 'spatial_control',
           'hierarchical_radio', 'nested_radio')
DOCS = ('README.md', 'docs/HANDOFF.md', 'docs/PAPER_PROPOSAL.md',
        'docs/research_status/CURRENT_SYSTEM.md', 'docs/research_status/NEXT_MODEL.md',
        'docs/research_status/REMAINING.md', 'docs/research_status/EXECUTION_LEDGER.md',
        'docs/nested_radio/PLAN.md', 'docs/nested_radio/ARCHITECTURE.md',
        'docs/nested_radio/RESULTS.md', 'docs/nested_radio/REMAINING.md',
        'artifacts/nested_radio_review/viewer/README.md')


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def checked(name, expected):
    path = ROOT / name
    assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT), name
    assert sha(path) == expected, name


def verify():
    manifests, union = {}, {}
    for study in STUDIES:
        name = f'artifacts/{study}/verification/manifest.json'
        value = read(ROOT / name)
        manifests[name] = sha(ROOT / name)
        for field in ('source_sha256', 'evidence_sha256', 'documents_sha256'):
            for file, digest in value.get(field, {}).items():
                assert file not in union or union[file] == digest, file
                union[file] = digest
    for name, digest in union.items():
        checked(name, digest)

    study_root = ROOT / 'artifacts/nested_radio'
    source = read(study_root / 'registration/source.json')
    capsule = study_root / 'registration/source'
    assert {str(p.relative_to(capsule)) for p in capsule.rglob('*') if p.is_file()} == set(source['source_sha256'])
    for name, digest in source['source_sha256'].items():
        path = capsule / name
        assert path.resolve().is_relative_to(capsule) and not path.is_symlink(), name
        assert sha(path) == digest, name
    assert len(source['source_sha256']) == 578
    manifest_name = 'artifacts/nested_radio/verification/manifest.json'
    manifest = read(ROOT / manifest_name)
    assert len(manifest['source_sha256']) == 578
    assert len(manifest['evidence_sha256']) == 2770
    assert manifest['fresh_RT_cases'] == 84 and manifest['fits'] == 15
    assert manifest['native_simulations'] == 0
    assert manifest['original_SERVICE_opened'] is False and manifest['production_qualified'] is False
    assert set(manifest['prior']['manifests']) == set(STUDIES[:-1])
    for study, reference in manifest['prior']['manifests'].items():
        name = f'artifacts/{study}/verification/manifest.json'
        assert reference == {'path': name, 'sha256': manifests[name]}, study

    suite_path = REVIEW / 'software_suite_importlib.json'
    suite = read(suite_path)
    assert suite['exit_code'] == 0 and suite['passed'] == 3633 and suite['source_unchanged'] is True
    assert suite['source_sha256_before'] == suite['source_sha256_after']
    assert len(suite['test_modules']) == 153 and len(suite['source_sha256_after']) == 575
    assert suite['deferred_untracked_observation_frontier_excluded'] is True
    assert suite['no_fresh_registered_scientific_case_or_training'] is True
    for name, digest in suite['source_sha256_after'].items():
        checked(name, digest)
    checked(suite['log_path'], suite['log_sha256'])

    audit = read(REVIEW / 'independent.json')
    assert audit['passes'] is True and audit['source_evidence_unchanged'] is True
    assert audit['study_manifest_sha256'] == manifests[manifest_name]
    assert audit['SERVICE_opened'] is False and audit['production_promotion'] is False
    assert sha(REVIEW / 'audit_actual.py') == audit['audit_source_sha256']
    assert audit['scientific_calls'] == {
        'RT_solves': 0, 'fits': 0, 'forwards': 0,
        'model_constructors': 0, 'packet_simulations': 0}
    assert audit['native_simulations'] == 0 and audit['fresh_RT_cases'] == 84
    assert audit['reused_TRAIN_VAL_cases'] == 90
    assert audit['provenance']['evidence_files'] == 2770 and audit['provenance']['source_files'] == 578
    assert audit['field']['prediction_records'] == 5928
    assert len(audit['field']['primary_contrasts']) == 4
    assert len(audit['training']['members']) == 15 and audit['training']['raw_VAL_archives'] == 60
    assert audit['training']['optimizer_updates_verified'] == 28800
    assert audit['representations']['matrices_verified'] == 15
    attempt = read(REVIEW / 'actual_attempt.json')
    assert attempt['audit_source_sha256'] == audit['audit_source_sha256']
    assert attempt['study_manifest_sha256'] == audit['study_manifest_sha256']
    assert attempt['scientific_calls'] == 0
    for name in ('first_failure.json', 'failure.json'):
        assert not (REVIEW / name).exists(), name

    viewer = REVIEW / 'viewer'
    provenance = read(viewer / 'provenance.json')
    browser = read(viewer / 'browser_review.json')
    assert browser['passes'] is True and browser['scientific_input_hashes_unchanged'] is True
    assert browser['geometry_AP_band_combinations'] == 576 and browser['primary_contrasts'] == 4
    assert browser['checkpoint_scores_checked'] == 60 and browser['fit_members_checked'] == 15
    assert browser['gate_members_checked'] == 15 and browser['gated_prior_product_rows_checked'] == 240
    assert browser['fixed_epoch_curve_points_checked'] == 20
    assert browser['RF_control_means_checked'] == 9 and browser['descriptive_stress_control_rows_checked'] == 45
    assert browser['page_errors'] == browser['console_errors'] == browser['external_requests'] == []
    assert browser['scientific_calls'] == provenance['scientific_calls'] == 0
    assert browser['native_simulations'] == provenance['native_simulations'] == 0
    assert provenance['fresh_RT_cases'] == 84 and provenance['reused_TRAIN_VAL_RT_cases'] == 90
    assert provenance['production_qualified'] is False and provenance['external_assets'] == 0
    assert browser['input_files_checked'] == len(provenance['inputs_sha256']) == 217
    assert browser['scientific_seal_sha256'] == manifests[manifest_name]
    assert browser['provenance_sha256'] == sha(viewer / 'provenance.json')
    assert browser['index_sha256'] == sha(viewer / 'index.html')
    assert browser['browser_verifier_sha256'] == sha(viewer / 'verify_browser.py')
    assert provenance['generator_sha256'] == browser['generator_sha256'] == sha(viewer / 'render_results.py')
    for name, digest in provenance['inputs_sha256'].items():
        checked(name, digest)
    assert set(provenance['output_sha256']) == {'data.json', 'index.html'}
    for name, digest in provenance['output_sha256'].items():
        assert sha(viewer / name) == digest, name
    assert set(browser['screenshots_sha256']) == {
        'desktop.png', 'geometry.png', 'mobile.png', 'mobile_geometry.png', 'mobile_hero.png'}
    for name, digest in browser['screenshots_sha256'].items():
        assert sha(viewer / name) == digest, name

    documents, links = {}, []
    for name in DOCS:
        path = ROOT / name
        documents[name] = sha(path)
        for match in re.finditer(r'(?<!!)\[[^\]]+\]\(([^)\n]+)\)', path.read_text()):
            target = match[1].strip().strip('<>')
            if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', target) or target.startswith('#'):
                continue
            target = unquote(target.split('#', 1)[0])
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            assert resolved.is_relative_to(ROOT) and resolved.exists(), (name, target)
            links.append({'document': name, 'target': target})

    for name in ('artifacts/radiojepa/rt_data/confirmation', 'artifacts/radiojepa/rt_data/confirmation.json'):
        assert not (ROOT / name).exists(), name
    return {
        'schema': 1, 'passes': True, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'verifier_sha256': sha(Path(__file__)), 'scientific_calls': 0,
        'scientific_closures': len(STUDIES), 'manifest_sha256': manifests,
        'unique_named_repository_files': len(union), 'current_source_capsule_files': 578,
        'current_evidence_files': 2770, 'tracked_suite_passed': 3633,
        'tracked_suite_receipt_sha256': sha(suite_path),
        'independent_audit_sha256': sha(REVIEW / 'independent.json'),
        'first_actual_audit_attempt_sha256': sha(REVIEW / 'actual_attempt.json'),
        'viewer_provenance_sha256': sha(viewer / 'provenance.json'),
        'browser_review_sha256': sha(viewer / 'browser_review.json'),
        'documents_sha256': documents, 'local_markdown_links_checked': len(links),
        'original_SERVICE_opened': False, 'production_qualified': False,
        'native_simulations': 0, 'deferred_observation_frontier_excluded': True}


if __name__ == '__main__':
    proof = verify()
    with (REVIEW / 'release_verification.json').open('x') as handle:
        json.dump(proof, handle, sort_keys=True, indent=2)
        handle.write('\n')
    print(json.dumps({key: proof[key] for key in ('passes', 'scientific_closures',
        'unique_named_repository_files', 'tracked_suite_passed', 'local_markdown_links_checked')}))
