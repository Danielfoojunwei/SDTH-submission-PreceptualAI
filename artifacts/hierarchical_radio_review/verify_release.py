"""Passive final release checks; no model, fitting, tracing or simulation imports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / 'artifacts/hierarchical_radio_review'
STUDIES = ('radiojepa', 'radiojepa_spectral', 'device_transfer', 'channel_scene',
           'adaptive_radio', 'native_bridge', 'e7_calplanning', 'spatial_control',
           'hierarchical_radio')
DOCS = ('README.md', 'docs/HANDOFF.md', 'docs/PAPER_PROPOSAL.md',
        'docs/research_status/CURRENT_SYSTEM.md', 'docs/research_status/NEXT_MODEL.md',
        'docs/research_status/REMAINING.md', 'docs/research_status/EXECUTION_LEDGER.md',
        'docs/hierarchical_radio/PLAN.md', 'docs/hierarchical_radio/ARCHITECTURE.md',
        'docs/hierarchical_radio/RESULTS.md', 'docs/hierarchical_radio/REMAINING.md',
        'docs/hierarchical_radio/INTERPRETATION.md')


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
    assert path.is_file() and path.resolve().is_relative_to(ROOT), name
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

    study_root = ROOT / 'artifacts/hierarchical_radio'
    source = read(study_root / 'registration/source.json')
    capsule = study_root / 'registration/source'
    assert {str(p.relative_to(capsule)) for p in capsule.rglob('*') if p.is_file()} == set(source['source_sha256'])
    for name, digest in source['source_sha256'].items():
        assert sha(capsule / name) == digest, name
    assert len(source['source_sha256']) == 567
    manifest_name = 'artifacts/hierarchical_radio/verification/manifest.json'
    manifest = read(ROOT / manifest_name)
    assert len(manifest['evidence_sha256']) == 4900

    suite = read(REVIEW / 'software_suite_importlib.json')
    assert suite['exit_code'] == 0 and suite['passed'] == 3416 and suite['source_unchanged'] is True
    assert suite['source_sha256_before'] == suite['source_sha256_after']
    assert len(suite['test_modules']) == 149 and len(suite['source_sha256_after']) == 564
    for name, digest in suite['source_sha256_after'].items():
        checked(name, digest)
    checked(suite['log_path'], suite['log_sha256'])

    audit = read(REVIEW / 'independent.json')
    assert audit['passes'] is True and audit['source_evidence_unchanged'] is True
    assert audit['study_manifest_sha256'] == manifests[manifest_name]
    assert not audit['SERVICE_opened'] and not audit['production_promotion']
    assert sha(REVIEW / 'audit_actual.py') == audit['audit_source_sha256']
    assert not (REVIEW / 'failure.json').exists()

    viewer = REVIEW / 'viewer'
    provenance = read(viewer / 'provenance.json')
    browser = read(viewer / 'browser_review.json')
    assert browser['passes'] is True and browser['scientific_input_hashes_unchanged'] is True
    assert browser['geometry_AP_band_combinations'] == 576
    assert browser['primary_contrasts'] == 4 and browser['native_individual_episode_rows_checked'] == 32
    assert browser['page_errors'] == browser['console_errors'] == browser['external_requests'] == []
    assert browser['scientific_calls'] == provenance['scientific_calls'] == 0
    assert browser['index_sha256'] == sha(viewer / 'index.html')
    assert provenance['generator_sha256'] == browser['generator_sha256'] == sha(viewer / 'render_results.py')
    for name, digest in provenance['inputs_sha256'].items():
        checked(name, digest)
    for name, digest in provenance['output_sha256'].items():
        assert sha(viewer / name) == digest, name
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
    return {'schema': 1, 'passes': True, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'verifier_sha256': sha(Path(__file__)), 'scientific_calls': 0,
        'scientific_closures': len(STUDIES), 'manifest_sha256': manifests,
        'unique_named_repository_files': len(union), 'current_source_capsule_files': 567,
        'current_evidence_files': 4900, 'tracked_suite_passed': 3416,
        'tracked_suite_receipt_sha256': sha(REVIEW / 'software_suite_importlib.json'),
        'independent_audit_sha256': sha(REVIEW / 'independent.json'),
        'viewer_provenance_sha256': sha(viewer / 'provenance.json'),
        'browser_review_sha256': sha(viewer / 'browser_review.json'),
        'documents_sha256': documents, 'local_markdown_links_checked': len(links),
        'original_SERVICE_opened': False, 'production_qualified': False,
        'deferred_observation_frontier_excluded': True}


if __name__ == '__main__':
    proof = verify()
    with (REVIEW / 'release_verification.json').open('x') as handle:
        json.dump(proof, handle, sort_keys=True, indent=2)
        handle.write('\n')
    print(json.dumps({key: proof[key] for key in ('passes', 'scientific_closures',
        'unique_named_repository_files', 'tracked_suite_passed', 'local_markdown_links_checked')}))
