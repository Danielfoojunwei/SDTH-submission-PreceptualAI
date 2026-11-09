"""Actual export CLI over temporary JSON-only software fixtures.

No repository RF reports, learners, arrays, fitted models or scientific cases
are opened by these tests. Fixture numbers exercise derivation, not RF claims.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from wirelessjev.research_export import bundle as b


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding='utf-8')
    return b.file_sha(path)


def sealed_adaptive_fixture(root):
    """Labelled software-only closure, including a deliberately toxic stale row."""
    members = {}
    for cell, mean in [('regression', 1.52), ('joint', 1.53), ('rf_erased', 1.88)]:
        for seed in (0, 1, 2):
            members[f'{cell}-seed{seed}'] = {
                'cell': cell, 'seed': seed, 'stale': False,
                'layouts': [{'id': f'fixture-layout-{i:02d}', 'layout_id': f'fixture-layout-{i:02d}',
                             'delta_available': True, 'delta_rmse_db': mean} for i in range(24)],
            }
    members['bayesian_operator'] = {
        'cell': 'bayesian_operator', 'seed': None, 'stale': False,
        'layouts': [{'id': f'fixture-layout-{i:02d}', 'layout_id': f'fixture-layout-{i:02d}',
                     'delta_available': True, 'delta_rmse_db': 1.524} for i in range(24)],
    }
    members['regression-seed0-stale'] = {
        'cell': 'regression', 'seed': 0, 'stale': True,
        'layouts': [{'id': f'fixture-layout-{i:02d}', 'layout_id': f'fixture-layout-{i:02d}',
                     'delta_available': True, 'delta_rmse_db': 99999.} for i in range(24)],
    }
    evaluation_path = 'artifacts/adaptive_radio/results/test.json'
    evaluation_sha = put(root, evaluation_path, {'members': {'sparse': members}})

    def contrast(candidate, reference, metric, candidate_mean, reference_mean, interval):
        return {'candidate': candidate, 'candidate_cell': 'regression' if candidate == 'selected' else candidate,
                'reference': reference, 'metric': metric, 'regime': 'sparse', 'usable_geometry_count': 24,
                'statistics': {'candidate_mean': candidate_mean, 'reference_mean': reference_mean,
                               'adjusted_interval': interval, 'better_adjusted': interval[1] < 0,
                               'mean_paired_difference': candidate_mean-reference_mean},
                'supported': interval[1] < 0}

    contrasts = [
        contrast('selected', 'radiomap_operator', 'delta_rmse_db', 1.52, 1.65, [-.2, -.05]),
        contrast('selected', 'rf_erased', 'delta_rmse_db', 1.52, 1.88, [-.45, -.2]),
        contrast('joint', 'regression', 'delta_rmse_db', 1.53, 1.52, [-.01, .03]),
        contrast('selected', 'radiomap_operator', 'detector_brier', .12, .12, [-.01, .01]),
    ]
    report = {'schema': 1, 'selected_cell': 'regression', 'registered_test_geometries': 24,
              'usable_delta_geometries': 24, 'status': 'complete', 'contrasts': contrasts,
              'RF_adaptation_supported': True, 'matched_alignment_benefit_supported': False,
              'detector_brier_benefit_supported': False,
              'evaluation': {'path': evaluation_path, 'sha256': evaluation_sha},
              'production_qualified': False, 'original_service_confirmation_permitted': False,
              'packet_probability_or_universal_robot_environment_claim': False}
    report_path = 'artifacts/adaptive_radio/report.json'
    report_sha = put(root, report_path, report)
    source_sha = put(root, 'src/software_source.py', 'software fixture source, no model')
    put(root, 'artifacts/adaptive_radio/verification/manifest.json', {
        'source_sha256': {'src/software_source.py': source_sha},
        'evidence_sha256': {report_path: report_sha, evaluation_path: evaluation_sha},
    })
    return report, members


def isolated_cli(monkeypatch):
    """Run the real command entry point, with unrelated data tracks isolated."""
    for name in ('measured_power', 'spectral', 'phones', 'csi', 'native', 'operator_profile',
                 'context', 'vision', 'e7_calplanning'):
        monkeypatch.setattr(b, name, lambda e: b.track('Unrelated software fixture', 'software_fixture',
            'software_only', scope='No scientific data in this fixture', units={}))
    monkeypatch.setattr(b, 'r6_software', lambda e: None)
    monkeypatch.setattr(b, 'suite', lambda e: {
        'passed': 0, 'test_modules': 0, 'warnings': 0, 'scope': 'software fixture only'})
    monkeypatch.setattr(b, 'software_capabilities', lambda e: {})
    monkeypatch.setattr(b, 'original_runs', lambda e: {'status': 'unrelated software fixture'})
    original_view = b.view_fields

    def view(key, value):
        if key == 'adaptive_rf':
            return original_view(key, value)
        value['headline'] = 'Unrelated software fixture'
        value['findings'] = ['No RF result in this fixture.']

    monkeypatch.setattr(b, 'view_fields', view)
    script = Path(__file__).resolve().parents[2] / 'scripts/export_research_studio.py'
    spec = importlib.util.spec_from_file_location('software_fixture_studio_cli', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_regenerates_adaptive_findings_from_sealed_inputs_and_rejects_stale_members(tmp_path, monkeypatch, capsys):
    legacy = {'generated_at': 'legacy software fixture', 'original_negative': -7,
              'nested': {'unchanged': [1, 2, 3]}, 'default': 'legacy controller'}
    put(tmp_path, 'studio/data/bundle.json', legacy)
    report, _ = sealed_adaptive_fixture(tmp_path)
    cli = isolated_cli(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['export_research_studio.py', '--root', str(tmp_path)])
    cli.main()
    first_command = json.loads(capsys.readouterr().out)
    assert first_command['remote_publication_performed'] is False

    output = tmp_path / 'artifacts/research_studio'
    first = json.loads((output / 'bundle.json').read_text())
    evidence = first['research_evidence']; adaptive = evidence['tracks']['adaptive_rf']
    assert adaptive['status'] == 'sealed' and adaptive['metrics']['selected_cell'] == 'regression'
    presentation = adaptive['metrics']['presentation_summary']
    assert presentation['relative_gain_percent'] == pytest.approx((1-1.52/1.65)*100)
    assert presentation['candidate_delta_rmse_db'] == pytest.approx(1.52)
    assert presentation['radiomap_delta_rmse_db'] == pytest.approx(1.65)
    assert presentation['joint_delta_rmse_db'] == pytest.approx(1.53)
    assert presentation['bayesian_delta_rmse_db'] == pytest.approx(1.524)
    assert presentation['brier_adjusted_interval'] == [-.01, .01]
    assert '7.9%' in adaptive['headline'] and '99999' not in adaptive['headline']
    descriptive = adaptive['metrics']['descriptive_sparse_delta_rmse']
    for cell, expected in [('regression', 1.52), ('joint', 1.53), ('rf_erased', 1.88), ('bayesian_operator', 1.524)]:
        assert descriptive['mean_db'][cell] == pytest.approx(expected)
        if cell != 'bayesian_operator':
            assert descriptive['exact_member_ids'][cell] == [f'{cell}-seed{s}' for s in (0, 1, 2)]
    assert descriptive['stale_members_excluded'] is True
    assert descriptive['reference_member_id'] == 'bayesian_operator' and descriptive['layouts'] == 24
    assert adaptive['metrics']['contrasts'] == report['contrasts']
    assert len(adaptive['findings']) <= 2
    assert not adaptive['production_qualified'] and not adaptive['sota_established']
    assert any('packet' in warning.lower() for warning in adaptive['limits'])
    assert any('JEPA' in warning or 'regression' in warning for warning in evidence['claim_boundaries'])
    assert evidence['default_controller'] == 'RadioMap + WirelessJEV'
    assert {key: value for key, value in first.items() if key != 'research_evidence'} == legacy
    assert json.loads((tmp_path / 'studio/data/bundle.json').read_text()) == legacy

    # Simulate the prior failure mode: manual generated-output edits must not be
    # required to recover the numerical headline after the standard CLI runs.
    tampered = deepcopy(first)
    tampered['research_evidence']['tracks']['adaptive_rf']['headline'] = 'MANUAL WRONG HEADLINE 99999%'
    (output / 'bundle.json').write_text(json.dumps(tampered))
    (output / 'index.html').write_text('MANUAL WRONG HEADLINE 99999%')
    cli.main()
    assert json.loads(capsys.readouterr().out)['remote_publication_performed'] is False
    restored = json.loads((output / 'bundle.json').read_text())
    restored_adaptive = restored['research_evidence']['tracks']['adaptive_rf']
    assert restored_adaptive['headline'] == adaptive['headline']
    assert restored_adaptive['metrics']['presentation_summary'] == presentation
    assert restored_adaptive['metrics']['descriptive_sparse_delta_rmse'] == descriptive
    assert 'MANUAL WRONG HEADLINE' not in (output / 'index.html').read_text()
    assert adaptive['headline'] in (output / 'index.html').read_text()
    assert {key: value for key, value in restored.items() if key != 'research_evidence'} == legacy
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['scientific_actions'] == {'fits': 0, 'model_forwards': 0, 'simulations': 0}
    assert manifest['outputs_sha256'] == {
        name: b.file_sha(output / name) for name in ('bundle.json', 'index.html')}
