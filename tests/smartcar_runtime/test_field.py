"""Field-adapter software fixtures; no capture, training, tracing or hardware trial.

The RT integration deliberately wraps archived simulated reports in disposable
API-schema records. Its clocks/BSSIDs are synthetic test identities, never a
claim of measured hardware or a new scientific evaluation.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from wirelessjev.smartcar_runtime.contracts import canonical_sha256, site_sha256
from wirelessjev.smartcar_runtime import field

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'artifacts/nested_radio/checkpoints/nested_joint-seed0.pt'
CHECKPOINT_SHA = '57c710af2db4f4bd7abfd031a03a76359c27100568d3ebe277ab3c738a69c3bf'


def site():
    return {'schema': 1, 'site_id': 'software-field-fixture', 'dimensions_m': [20., 20., 5.],
        'cad_boxes_m': [], 'aps': [
            {'bssid': '02:00:00:00:00:01', 'ssid': 'fixture', 'xyz_m': [2., 2., 4.],
             'frequency_mhz': 2412., 'tx_power_dbm': 20.},
            {'bssid': '02:00:00:00:00:02', 'ssid': 'fixture', 'xyz_m': [17., 17., 4.],
             'frequency_mhz': 5180., 'tx_power_dbm': 20.}],
        'receiver': {'noise_floor_dbm': -100., 'offset_db': 0., 'quantization_db': 0.,
                     'reference_gauge_assumed': True}}


def truth(position, ap):
    # Independent analytic inverse-square field plus a smooth spatial term.
    distance = math.dist(position, ap['xyz_m'])
    loss = 20.*math.log10(4.*math.pi*ap['frequency_mhz']*1.e6*distance/299792458.)
    return ap['tx_power_dbm'] - loss - 8. - .2*position[0] + .3*math.cos(position[1])


def record(declared, index=0, position=None, heard=(0,), *, age_seconds=3., signal=None):
    position = [3.+index, 4., 1.] if position is None else position
    now = time.time_ns()
    received = now-int(age_seconds*1.e9)
    return {'schema': 1, 'site_id': declared['site_id'], 'site_sha256': site_sha256(declared),
        'capture_id': f'software-capture-{index}', 'pose': {'xyz_m': list(position), 'source': 'operator'},
        'started_unix_ns': 10**20+index*1000, 'completed_unix_ns': 10**20+index*1000+100,
        'pc_request_started_unix_ns': received-10_000_000, 'pc_received_unix_ns': received,
        'interface': 'wlan1', 'source': 'hardware', 'fresh_scan': True,
        'software_fixture_not_hardware_measurement': True,
        'aps': [{k: ap[k] for k in ('bssid', 'ssid', 'frequency_mhz')} | {
            'signal_dbm': truth(position, ap) if signal is None else signal}
            for i, ap in enumerate(declared['aps']) if i in heard]}


def survey(declared=None):
    declared = site() if declared is None else declared
    # Delivery order is deliberately independent of unsynchronized car time.
    return [record(declared, i, [x, 4., 1.], age_seconds=10.-i)
            for i, x in enumerate((3., 5., 7., 9., 11.))]


def test_numerical_model_is_JSON_and_predicts_genuinely_held_positions():
    declared = site(); captures = survey(declared)
    model = field.fit_map(declared, captures)
    model = json.loads(json.dumps(model, allow_nan=False))
    held = [[x, 4., 1.] for x in (4., 6., 8., 10.)]
    actual = field.predict_map(model, held)
    predicted = np.array([row['mean_dbm'] for row in actual if row['bssid'] == declared['aps'][0]['bssid']])
    target = np.array([truth(p, declared['aps'][0]) for p in held])
    baseline = np.full(len(held), captures[-1]['aps'][0]['signal_dbm'])
    rmse = np.sqrt(np.mean((predicted-target)**2))
    assert rmse < .6 and rmse < np.sqrt(np.mean((baseline-target)**2))
    assert all(row['status'] == 'unknown' and row['mean_dbm'] is None and row['std_db'] is None
               for row in actual if row['bssid'] == declared['aps'][1]['bssid'])
    assert all(row['std_db'] >= 2. and not row['packet_risk_qualified']
               for row in actual if row['status'] == 'prediction')
    assert all(row['inference_elapsed_seconds'] >= 0 for row in actual)


def test_numerical_baseline_import_and_prediction_do_not_import_Torch():
    code = '''import json,sys
from wirelessjev.smartcar_runtime.field import fit_map,predict_map
site=json.loads(sys.argv[1]); records=json.loads(sys.argv[2])
model=fit_map(site,records); predict_map(model,[[4.,4.,1.]])
assert 'torch' not in sys.modules and 'sionna' not in sys.modules and 'mitsuba' not in sys.modules
print('numpy-only-map-pass')'''
    result = subprocess.run([sys.executable, '-c', code, json.dumps(site()), json.dumps(survey())],
        cwd=ROOT, env=os.environ.copy(), text=True, capture_output=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'numpy-only-map-pass' in result.stdout


def test_missing_AP_and_far_queries_do_not_acquire_fake_floor_measurements():
    declared = site(); captures = survey(declared)
    model = field.fit_map(declared, captures)
    assert model['groups'][declared['aps'][1]['bssid']]['points'] == []
    assert model['groups'][declared['aps'][1]['bssid']]['gp'] is None
    assert all(group['unreported_observations_created'] == 0 for group in model['groups'].values())
    rows = field.predict_map(model, [[19., 1., 1.]])
    assert all(row['status'] == 'unknown' and row['mean_dbm'] is None for row in rows)
    assert rows[0]['support']['reason'] == 'outside_spatial_support'
    assert rows[1]['support']['reason'] == 'unobserved_ap'


def test_repeat_pose_keeps_latest_actual_sample_without_artificial_precision():
    declared = site(); old = record(declared, 0, [4., 4., 1.], age_seconds=10., signal=-80.)
    latest = record(declared, 1, [4., 4., 1.], age_seconds=2., signal=-60.)
    one = field.fit_map(declared, [latest])
    two = field.fit_map(declared, [old, latest])
    bssid = declared['aps'][0]['bssid']
    assert len(two['groups'][bssid]['points']) == 1
    assert two['groups'][bssid]['points'][0]['signal_dbm'] == -60.
    a, b = field.predict_map(one, [[4., 4., 1.]])[0], field.predict_map(two, [[4., 4., 1.]])[0]
    assert a['mean_dbm'] == b['mean_dbm'] and a['std_db'] == b['std_db']


def test_unsynchronized_remote_clock_is_not_subtracted_from_desktop_query_clock():
    declared = site(); captures = survey(declared)
    assert captures[0]['started_unix_ns'] > time.time_ns()
    assert field.predict_map(field.fit_map(declared, captures), [[4., 4., 1.]])[0]['status'] == 'prediction'
    bad = deepcopy(captures)
    bad[-1]['pc_received_unix_ns'] = time.time_ns()+10**12
    with pytest.raises(ValueError, match='causally'):
        field.fit_map(declared, bad)


@pytest.mark.parametrize('change', [
    lambda rows: rows.append(deepcopy(rows[-1])),
    lambda rows: rows.reverse(),
    lambda rows: rows[0].update(fresh_scan=False),
    lambda rows: rows[0].update(site_sha256='0'*64),
])
def test_invalid_or_replayed_surveys_fail_closed(change):
    captures = survey(); change(captures)
    with pytest.raises(ValueError):
        field.fit_map(site(), captures)


def test_stale_samples_remain_unknown_and_refitting_does_not_reset_their_age():
    declared = site(); captures = [record(declared, age_seconds=400.)]
    rows = field.predict_map(field.fit_map(declared, captures), [[3., 4., 1.]])
    assert rows[0]['status'] == 'unknown' and rows[0]['support']['reason'] == 'stale_observations'
    prepared = field.prepare_radiojepa_inputs(declared, captures, [[3., 4., 1.]])
    assert prepared['view'] is None and not prepared['selected_context']
    assert prepared['rows'][0]['status'] == 'unknown'


def test_map_identity_and_working_assumptions_are_bound():
    model = field.fit_map(site(), survey())
    model['groups'][site()['aps'][0]['bssid']]['gp']['intercept_db'] += 10.
    with pytest.raises(ValueError, match='changed'):
        field.predict_map(model, [[4., 4., 1.]])
    model = field.fit_map(site(), survey());model['support_radius_m'] = 1000.
    model['model_sha256'] = canonical_sha256({k:v for k,v in model.items() if k != 'model_sha256'})
    with pytest.raises(ValueError, match='assumptions'):
        field.predict_map(model, [[4., 4., 1.]])


def test_neural_context_is_only_actual_heard_rows_not_scheduled_omissions():
    declared = site(); captures = survey(declared)
    prepared = field.prepare_radiojepa_inputs(declared, captures, [[4., 4., 1.]])
    view = prepared['view']
    assert len(prepared['selected_context']) == len(captures)
    assert view['known'].all() and view['scheduled'].all()
    assert (view['context'][:, 2] == 1).all() and (view['context'][:, 3] == 0).all()
    assert (view['lower'] == view['upper']).all()  # genuine declared continuous points
    assert (view['ap'] == 0).all()
    assert prepared['rows'][1]['status'] == 'unknown'
    np.testing.assert_allclose(prepared['ap_xyz'][0], np.array(declared['aps'][0]['xyz_m'])-[10.,10.,0.])


def test_declared_real_quantization_preserves_intervals_and_no_detector_cutoff_is_invented():
    declared = site(); declared['receiver']['quantization_db'] = 1.
    captures = survey(declared)
    prepared = field.prepare_radiojepa_inputs(declared, captures, [[4.,4.,1.]])
    view = prepared['view']
    np.testing.assert_allclose(view['upper']-view['lower'], 1., rtol=0, atol=1e-12)
    np.testing.assert_array_equal(view['floor'], view['lower'])
    assert not np.isneginf(view['lower']).any()  # no invented upper-censor events


def test_neural_context_cap_has_AP_coverage_and_no_RSSI_ranking():
    declared = site(); captures = []
    for i in range(120):
        position = [3.+.05*i, 4.+.01*i, 1.]
        captures.append(record(declared, i, position, heard=(0,1), age_seconds=130.-i))
    query = [[4.,4.,1.]]
    a = field.prepare_radiojepa_inputs(declared, captures, query)
    assert len(a['selected_context']) == 192
    assert {p['ap_index'] for p in a['selected_context']} == {0,1}
    altered = deepcopy(captures)
    for row in altered:
        for ap in row['aps']:ap['signal_dbm'] = -120.+int(ap['bssid'][-1])
    b = field.prepare_radiojepa_inputs(declared, altered, query)
    assert [(r['ap_index'],r['capture_id']) for r in a['selected_context']] == [
        (r['ap_index'],r['capture_id']) for r in b['selected_context']]


def test_unsupported_6GHz_is_not_a_prediction_or_a_hidden_source_for_other_APs():
    declared = site(); declared['aps'][1]['frequency_mhz'] = 6000.
    captures = [record(declared, heard=(0,1))]
    prepared = field.prepare_radiojepa_inputs(declared,captures,[[3.,4.,1.]])
    assert {r['ap_index'] for r in prepared['selected_context']} == {0}
    assert prepared['rows'][1]['status'] == 'unknown'
    assert prepared['rows'][1]['support']['reason'] == 'carrier_outside_trained_source_interval'


def test_neural_gauge_assumption_is_required_without_disabling_numerical_map():
    declared = site(); declared['receiver']['reference_gauge_assumed'] = False
    captures = survey(declared)
    assert field.predict_map(field.fit_map(declared,captures),[[4.,4.,1.]])[0]['status'] == 'prediction'
    with pytest.raises(ValueError,match='reference gauge'):
        field.prepare_radiojepa_inputs(declared,captures,[[4.,4.,1.]])


@pytest.fixture
def published_checkpoint():
    assert CHECKPOINT.is_file(), 'The actual published checkpoint is required for this integration test'
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == CHECKPOINT_SHA
    return CHECKPOINT


def test_actual_published_checkpoint_runs_strictly_as_readonly_shadow(published_checkpoint):
    import torch
    declared = site();captures = survey(declared);query = [[4.,4.,1.]]
    rng = torch.get_rng_state().clone()
    rows = field.predict_radiojepa(declared,captures,query,published_checkpoint,CHECKPOINT_SHA)
    assert torch.equal(torch.get_rng_state(),rng)
    assert rows[0]['status'] == 'prediction' and rows[0]['std_db'] > 0
    assert rows[0]['checkpoint_sha256'] == CHECKPOINT_SHA and rows[0]['cell'] == 'nested_joint'
    assert rows[0]['assumptions']['shadow_only'] and not rows[0]['real_hardware_accuracy_validated']
    assert rows[1]['status'] == 'unknown' and rows[1]['mean_dbm'] is None
    assert hashlib.sha256(published_checkpoint.read_bytes()).hexdigest() == CHECKPOINT_SHA
    with pytest.raises(ValueError,match='SHA-256'):
        field.predict_radiojepa(declared,captures,query,published_checkpoint,'0'*64)


@pytest.mark.parametrize('change', [
    lambda ck:ck['settings'].update(width=10000000),
    lambda ck:ck['settings'].update(cell='global8_reg'),
    lambda ck:ck.update(seed=True),
    lambda ck:ck.update(committed_updates=1920.),
    lambda ck:ck['state']['local_gates'].fill_(float('nan')),
    lambda ck:ck['state']['local_gates'].fill_(2.),
    lambda ck:ck['state'].update(unexpected=torch_zero()),
    lambda ck:ck['state'].update(local_gates=ck['state']['local_gates'].double()),
])
def test_tampered_checkpoint_is_refused_before_any_field_forward(tmp_path,published_checkpoint,change):
    import torch
    ck=torch.load(published_checkpoint,map_location='cpu',weights_only=True);change(ck)
    target=tmp_path/'tampered.pt';torch.save(ck,target)
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises((ValueError,RuntimeError,TypeError)):
        field.predict_radiojepa(site(),survey(),[[4.,4.,1.]],target,digest)


def torch_zero():
    import torch
    return torch.zeros(1)


def test_actual_RT_report_metadata_and_published_weights_have_manual_forward_parity(published_checkpoint):
    import torch
    from wirelessjev.nested_radio.model import NestedField
    folder=ROOT/'artifacts/hierarchical_radio/rt_data/cases/hierarchical-training-00-state0'
    source=folder/'public_views.npz';before=hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source,allow_pickle=False) as archive: a={k:archive[k].copy() for k in archive.files}
    width,depth,height=a['bbox_dims_m'];translation=np.array([width/2,depth/2,0.])
    # Read actual public interior CAD vertices, without loading private channel
    # arrays or scene materials. Exterior shell volumes extend outside the
    # declared room and are intentionally outside this runtime site's boxes.
    cad=[]
    for mesh in sorted((folder/'scene/meshes').glob('partition*.ply')):
        text=mesh.read_text().splitlines();first=text.index('end_header')+1
        vertices=np.array([[float(x) for x in line.split()] for line in text[first:first+8]])
        cad.append((np.stack((vertices.min(0),vertices.max(0)))+translation).tolist())
    assert len(cad)==2
    declared={'schema':1,'site_id':'RT-software-conversion','dimensions_m':[width,depth,height],
        'cad_boxes_m':cad, 'aps':[{'bssid':f'02:00:00:00:01:{i+1:02x}','ssid':'RT-software-fixture',
            'xyz_m':(p+translation).tolist(),'frequency_mhz':float(a['carrier_hz'][0]/1.e6),
            'tx_power_dbm':float(json.loads((folder/'receipt.json').read_text())['known_tx_power_dbm'])}
            for i,p in enumerate(a['ap_positions_m'])],
        'receiver':{'noise_floor_dbm':float(a['noise_floor_dbm'][0]),'offset_db':0.,
            'quantization_db':1.,'reference_gauge_assumed':True}}
    captures=[]
    for pose in range(8):
        heard=np.flatnonzero(a['heard'][pose,:,0,0,0])
        if not len(heard):continue
        row=record(declared,pose,(a['rx_positions_m'][pose]+translation).tolist(),tuple(heard),age_seconds=20.-pose)
        for item,i in zip(row['aps'],heard):item['signal_dbm']=float(a['rssi_dbm'][pose,i,0,0,0])
        row['software_fixture_source']='Actual archived static RT reported values; synthetic API identifiers/clocks, no measured hardware.'
        captures.append(row)
    queries=[captures[-1]['pose']['xyz_m']]
    prepared=field.prepare_radiojepa_inputs(declared,captures,queries)
    assert prepared['query'] is not None
    checkpoint=torch.load(published_checkpoint,map_location='cpu',weights_only=True)
    with torch.random.fork_rng(devices=[]): model=NestedField(**checkpoint['settings'])
    model.load_state_dict(checkpoint['state'],strict=True);model.eval()
    batched=lambda values:{k:torch.from_numpy(np.array(v,copy=True))[None] for k,v in values.items()}
    with torch.inference_mode():
        expected_mean,expected_std=model(batched(prepared['view']),batched(prepared['query']),
            torch.from_numpy(prepared['ap_xyz'].copy())[None])
    actual=field.predict_radiojepa(declared,captures,queries,published_checkpoint,CHECKPOINT_SHA)
    observed=[actual[i] for i in prepared['query_row_indices']]
    np.testing.assert_allclose([r['mean_dbm'] for r in observed],expected_mean[0].numpy(),rtol=0,atol=1e-10)
    np.testing.assert_allclose([r['std_db'] for r in observed],expected_std[0].numpy(),rtol=0,atol=1e-10)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
