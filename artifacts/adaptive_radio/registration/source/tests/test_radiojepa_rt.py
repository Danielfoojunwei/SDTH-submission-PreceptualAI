"""Analytic and known-ID real-Sionna fixtures, never registered campaign runs."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from wirelessjev.radiojepa import rt_data as rt


def known_spec():
    return rt.GeometrySpec('fixture-known-room','rack_aisles',8.,10.,4.,2.2,'metal',
        ((-2.6,-3.2,3.4),(2.6,3.2,3.4)),((0.,-2.,1.1),(0.,2.,1.1)),123)


def test_registry_is_public_specs_only_with_reserved_layouts(tmp_path,monkeypatch):
    monkeypatch.setattr(rt,'_rt',lambda: pytest.fail('Registry must not initialize a renderer'))
    value=rt.geometry_registry()
    assert [len(value['stages'][name]) for name in rt.STAGES] == [8,2,2,24]
    names=[row['scene_id'] for rows in value['stages'].values() for row in rows]
    assert len(set(names)) == 36
    physical=[rt.fingerprint({k:v for k,v in row.items() if k not in ('scene_id','seed','spec_sha256')})
              for rows in value['stages'].values() for row in rows]
    assert len(set(physical)) == 36  # No disguised split overlap via a new seed.
    assert all(len(row['rx_positions_m']) == 12 and len(row['ap_positions_m']) == 2
               for rows in value['stages'].values() for row in rows)
    assert len(value['training_device_ids']) == 3 and len(value['held_out_device_ids']) == 1
    path=tmp_path/'registry.json'
    receipt=rt.write_registry(path)
    before=path.read_bytes()
    assert rt.write_registry(path) == receipt and path.read_bytes() == before
    assert receipt['registry_sha256'] == rt.fingerprint(json.loads(before))


@pytest.mark.parametrize('field,value', [('width_m',float('nan')),('obstacle_height_m',5.),
    ('scene_id','../redirect'),('rx_positions_m',((100.,0.,1.),))])
def test_invalid_geometry_rejected(field,value):
    with pytest.raises(ValueError): replace(known_spec(),**{field:value})


def test_scene_writes_materials_and_actual_closed_meshes_without_tracing(tmp_path):
    path,files=rt.write_scene(known_spec(),tmp_path)
    assert path.name == 'scene.xml' and len(files) == 9
    xml=path.read_text()
    assert 'itu-radio-material' in xml and 'metal' in xml
    for f in files[:-1]:
        assert 'element vertex 8' in f.read_text() and 'element face 12' in f.read_text()
    hashes=[hashlib.sha256(f.read_bytes()).hexdigest() for f in files]
    rt.write_scene(known_spec(),tmp_path)
    assert hashes == [hashlib.sha256(f.read_bytes()).hexdigest() for f in files]


def test_receiver_offset_quantization_noise_and_unknown_are_distinct():
    signal=np.array([-60.3,-95.,np.nan])
    profiles=(rt.DeviceProfile('low-noise',2.,1.,0.),rt.DeviceProfile('high-noise',2.,1.,20.))
    result=rt.device_observations(signal,np.array([True,True,False]),profiles=profiles)
    assert result['rssi_dbm'][0,0] == result['rssi_dbm'][0,1] == -58.
    interval=result['quantization_interval_dbm'][0,0]
    assert interval[0] <= -58.3 <= interval[1]
    assert result['heard'][1,0] and not result['heard'][1,1]
    assert result['censored'][1,1]
    assert not result['heard'][2].any() and not result['censored'][2].any()
    assert not result['rssi_dbm'][2].any() and not result['censor_upper_dbm'][2].any()
    assert math.isclose(float(np.diff(result['noise_floor_dbm'])[0]),20.,abs_tol=1e-5)


def test_registered_fixture_bypass_and_unqualified_confirmation_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(rt,'_rt',lambda: pytest.fail('Rejected stage must not trace'))
    row=rt.geometry_registry()['stages']['training'][0]
    spec=rt.GeometrySpec(**{k:v for k,v in row.items() if k!='spec_sha256'})
    with pytest.raises(RuntimeError,match='Registered'): rt.trace_fixture(spec,tmp_path)
    with pytest.raises(RuntimeError,match='authorization'): rt.collect_stage('training',tmp_path)
    proof={'stage':'confirmation','authorized':True,'registry_sha256':rt.fingerprint(rt.geometry_registry())}
    with pytest.raises(RuntimeError,match='qualification'):
        rt.collect_stage('confirmation',tmp_path,authorize=lambda stage:proof)
    proof['confirmation_permitted']='true'
    with pytest.raises(RuntimeError,match='qualification'):
        rt.collect_stage('confirmation',tmp_path,authorize=lambda stage:proof)
    assert not list(tmp_path.iterdir())


def test_training_profile_boundary_excludes_heldout_measurements():
    training=tuple(p for p in rt.DEVICE_PROFILES if not p.held_out)
    result=rt.device_observations(np.zeros((2,2,3,2))-65.,np.ones((2,2,3,2),bool),profiles=training)
    assert result['rssi_dbm'].shape == (2,2,3,2,3)
    assert all(not p.held_out for p in training)


def test_passive_resume_rejects_empty_hash_map_without_tracing(tmp_path,monkeypatch):
    monkeypatch.setattr(rt,'_rt',lambda: pytest.fail('A passive resume must not create data'))
    spec=known_spec()
    value={'spec_sha256':spec.sha256,'spec':spec.to_dict(),'real_device_measurements':False,
           'bands_hz':list(rt.BANDS_HZ),'bandwidth_hz':rt.BANDWIDTH_HZ,
           'reference_tx_dbm':rt.REFERENCE_TX_DBM,
           'receiver_profiles':[rt.asdict(p) for p in rt.DEVICE_PROFILES if not p.held_out],
           'files_sha256':{}}
    (tmp_path/'receipt.json').write_text(json.dumps(value))
    with pytest.raises(RuntimeError,match='membership'):
        rt.validate_scene_receipt(spec,tmp_path,'training')


def test_actual_sionna_known_id_simulated_multiband_reciprocal_fixture(tmp_path):
    mi=pytest.importorskip('mitsuba')
    mi.set_variant('llvm_ad_mono_polarized')
    pytest.importorskip('sionna.rt')
    receipt=rt.trace_fixture(known_spec(),tmp_path/'known',bands_hz=(2.4e9,5.2e9),
        samples_per_src=2048,max_depth=2,max_num_paths_per_src=1024)
    assert len(receipt['solves']) == 4 and receipt['backend']=='llvm_ad_mono_polarized'
    assert receipt['real_device_measurements'] is False
    assert receipt['absolute_delays_identified'] is True
    assert receipt['unresolved_links_are_censored'] is False
    assert receipt['solves'][0]['frequency_updated_materials']['obstacle']['conductivity_s_m'] > 0
    with np.load(tmp_path/'known/public_views.npz',allow_pickle=False) as a:
        assert a['rssi_dbm'].shape == (2,2,2,2,4)
        assert a['quantization_interval_dbm'].shape == (2,2,2,2,4,2)
        assert a['physical_link_solved'].shape == (2,2,2,2)
        assert a['physical_link_solved'].all()
        assert np.isfinite(a['rssi_dbm']).all()
    with np.load(tmp_path/'known/physical_channels.npz',allow_pickle=False) as a:
        expected=20.+10*np.log10(np.mean(np.abs(a['cfr'].astype('complex128'))**2,axis=-1))
        np.testing.assert_allclose(a['signal_dbm'],expected,rtol=0,atol=1e-10)
    for name,expected in receipt['files_sha256'].items():
        assert hashlib.sha256((tmp_path/'known'/name).read_bytes()).hexdigest() == expected
