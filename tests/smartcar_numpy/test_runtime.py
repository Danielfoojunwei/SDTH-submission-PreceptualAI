"""End-to-end numerical shadow parity; API fixtures are not car observations."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from wirelessjev.smartcar_numpy.export import export_release
from wirelessjev.smartcar_numpy.runtime import prepare_numpy_inputs, predict_numpy
from wirelessjev.smartcar_runtime.field import prepare_radiojepa_inputs, predict_radiojepa
from wirelessjev.smartcar_runtime.workflow import RELEASED_CHECKPOINTS

ROOT = Path(__file__).resolve().parents[2]


def fixture(quant=0.):
    spec = importlib.util.spec_from_file_location('portable_api_software_fixture', ROOT/'tests/smartcar_runtime/test_workflow.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    site, records = module.fixture()
    if quant:
        from wirelessjev.smartcar_runtime.contracts import site_sha256
        site['receiver']['quantization_db'] = quant
        for row in records:
            row['site_sha256'] = site_sha256(site)
    return site, records


@pytest.mark.parametrize('quant', (0., 1.))
def test_numerical_preparation_preserves_every_reference_input(quant):
    site, records = fixture(quant)
    now = records[-1]['pc_received_unix_ns']+1
    expected = prepare_radiojepa_inputs(site, records, [[4., 2., .3]], now_ns=now)
    actual = prepare_numpy_inputs(site, records, [[4., 2., .3]], now_ns=now)
    for group in ('view', 'query'):
        assert set(expected[group]) == set(actual[group])
        for name in expected[group]:
            np.testing.assert_array_equal(actual[group][name], expected[group][name])
    np.testing.assert_array_equal(expected['ap_xyz'], actual['ap_xyz'])
    assert expected['selected_context'] == actual['selected_context']


@pytest.mark.parametrize('quant', (0., 1.))
def test_all_three_exported_members_match_actual_reference_shadow(tmp_path, quant):
    manifest = export_release(ROOT/'artifacts/nested_radio/checkpoints', tmp_path/'models')
    site, records = fixture(quant)
    for member in manifest['members']:
        original = predict_radiojepa(site, records, [[4., 2., .3]],
            ROOT/'artifacts/nested_radio/checkpoints'/f"{member['name']}.pt", RELEASED_CHECKPOINTS[member['name']])
        numerical = predict_numpy(site, records, [[4., 2., .3]], tmp_path/'models'/member['path'],
                                  member['sha256'], member['source_checkpoint_sha256'])
        assert len(numerical) == len(original) == 2
        for a, b in zip(numerical, original):
            assert a['bssid'] == b['bssid'] and a['status'] == b['status'] == 'prediction'
            np.testing.assert_allclose([a['mean_dbm'], a['std_db']], [b['mean_dbm'], b['std_db']],
                                       atol=1.e-4, rtol=1.e-6)
            assert not a['real_hardware_accuracy_validated'] and not a['packet_risk_qualified']


def test_whole_shadow_cli_runs_with_torch_import_forbidden(tmp_path):
    manifest = export_release(ROOT/'artifacts/nested_radio/checkpoints', tmp_path/'models')
    site, records = fixture()
    (tmp_path/'site.json').write_text(json.dumps(site))
    (tmp_path/'survey.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    code = """
import importlib.abc,sys
class Forbidden(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','sionna','mitsuba','drjit','transformers','huggingface_hub'):
   raise RuntimeError('Forbidden runtime dependency')
sys.meta_path.insert(0,Forbidden())
from wirelessjev.smartcar_numpy.workflow import main
raise SystemExit(main(sys.argv[1:]))
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-c', code, 'shadow', '--site', str(tmp_path/'site.json'),
        '--survey', str(tmp_path/'survey.jsonl'), '--xyz', '4', '2', '.3',
        '--models', str(tmp_path/'models/manifest.json'), '--out', str(tmp_path/'shadow.json')],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    output = json.loads((tmp_path/'shadow.json').read_text())
    assert output['backend'] == 'numpy_scipy' and len(output['members']) == 3
    assert output['sent_to_car'] is False and output['hardware_accuracy_measured'] is False
