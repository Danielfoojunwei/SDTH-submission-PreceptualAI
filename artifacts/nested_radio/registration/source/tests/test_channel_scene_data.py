"""Bound-input and privacy fixtures; no measured holdout or scientific run."""
import json
from pathlib import Path

import numpy as np
import pytest

from wirelessjev.channel_scene import data as d
from wirelessjev.channel_scene.calibration import PhaseCoherenceBudget


def admission(paths, **kwargs):
    return d.ObservationAdmission('fixture-new-complex-observation-protocol','training',
        {key:d.sha256(path) for key,path in paths.items()},**kwargs)


def measured_fixture(tmp_path, *, private_object=False):
    root=tmp_path/'csi/dichasus';root.mkdir(parents=True)
    name='dichasus-cf06';paths=d.dichasus_sources(name,root)
    n,links,bins=5,3,1024
    offsets=(np.arange(bins)-512)*50.e6/bins
    h=np.ones((n,links,bins),complex)
    h+=.2*np.exp(-2j*np.pi*offsets*54.e-9)
    sto=np.array([0.,2.,-1.5]);cpo=np.array([0.,.7,-.3]);k=np.arange(bins)
    corrupt=h*np.exp(-1j*(sto[:,None]*2*np.pi*k/bins+cpo[:,None]))
    raw=np.fft.ifftshift(corrupt,axes=-1)
    raw=np.stack((raw.real,raw.imag),-1).astype('f4')
    pose=np.array(['DO NOT READ PRIVATE POSE'],dtype=object) if private_object else np.arange(n*3,dtype=float).reshape(n,3)
    np.savez_compressed(paths['measured:'+name+':npz'],csi=raw,time=np.array([4,2,0,1,3])[:,None],
                        **{'pos-tachy':pose,'material_truth':np.array(['DO NOT READ'],dtype=object)})
    receipt={'source_recording':name,'parsed_npz_sha256':d.sha256(paths['measured:'+name+':npz']),
        'all_length_and_payload_crc32c_pass':True,'acquired_records':n,'frequency_hz':1.272e9,'bandwidth_hz':50.e6}
    binding={'recording':name,'antenna_group_ids':[0,0,1],
        'array_centers_by_antenna_m':[[0,0,1],[0,0,1],[5,0,1]],'frequency_hz':1.272e9,'bandwidth_hz':50.e6}
    for key,value in [('receipt',receipt),('binding',binding),('calibration',{'sto':sto.tolist(),'cpo':cpo.tolist()})]:
        path=paths['measured:'+name+':'+key];path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    return root,paths,h


def rt_fixture(tmp_path, *, private_object=False):
    folder=tmp_path/'training/fixture-rt-training';folder.mkdir(parents=True)
    paths=d.sionna_sources(folder);base='rt:'+folder.name+':'
    shape=(3,2,2,2,64);h=np.full(shape,.01+.02j,dtype='c8');mask=np.ones(shape[:-1],bool)
    mask[0,1,0,0]=False;h[0,1,0,0]=np.nan+1j*np.nan
    np.savez_compressed(paths[base+'cfr'],cfr=h,signal_dbm=np.array(['PRIVATE POWER'],dtype=object),
                        coefficients=np.array(['PRIVATE RAYS'],dtype=object),delay_s=np.array(['PRIVATE DELAYS'],dtype=object))
    pose=np.array(['PRIVATE POSE'],dtype=object) if private_object else np.arange(9,dtype=float).reshape(3,3)
    np.savez_compressed(paths[base+'public'],ap_positions_m=np.array([[0.,0.,2.],[4.,3.,2.]]),
        carrier_hz=np.array([2.4e9,6.e9]),physical_link_solved=mask,rx_positions_m=pose,
        material_label=np.array(['PRIVATE MATERIAL'],dtype=object))
    receipt={'scene_id':folder.name,'real_device_measurements':False,
        'physical_axis_order':['rx_pose','ap','carrier','direction'],'directions':['AP_to_client','client_to_AP'],
        'files_sha256':{'physical_channels.npz':d.sha256(paths[base+'cfr']),
                        'public_views.npz':d.sha256(paths[base+'public'])},
        'bandwidth_hz':20.e6,'bands_hz':[2.4e9,6.e9],'absolute_delays_identified':True,
        'normalization_or_passivity_clipping_applied':False,
        'receiver_profile_scope':'RSSI offset/quantization/NF/sensitivity only; no CSI front-end impairments',
        'spec':{'private_material':'DO NOT PASS TO MODEL','private_delay_truth':[1,2,3]}}
    paths[base+'receipt'].write_text(json.dumps(receipt))
    return folder,paths


def test_measured_reader_applies_source_offsets_keeps_clock_and_has_no_pose_feature(tmp_path):
    root,paths,expected=measured_fixture(tmp_path)
    observed=d.load_dichasus('dichasus-cf06',admission(paths),root)
    np.testing.assert_allclose(observed.cfr,expected,atol=2.e-7)
    np.testing.assert_array_equal(observed.capture_time_s,[4,2,0,1,3])
    assert observed.gauge.sto_cpo_applied and observed.gauge.delay_reference=='unknown_sampling_origin'
    assert observed.coherence_proof is None and observed.public_mobile_xyz_m is None
    assert set(observed.encoder_features())=={'cfr','frequency_offsets_hz','carrier_hz','observed','infrastructure_xyz_m','group_ids'}
    assert 'receiver_referenced' in observed.gauge.power_reference
    with pytest.raises(ValueError,match='bound evidence'):
        d.require_coherent_view(observed)
    with pytest.raises(ValueError):observed.cfr[0,0,0]=4
    labels=d.private_labels(observed,purpose='frozen_probe')
    np.testing.assert_array_equal(labels['mobile_position_m'],np.arange(15).reshape(5,3))


def test_private_poisoned_pose_and_ray_material_values_are_not_opened_by_reader(tmp_path):
    root,paths,_=measured_fixture(tmp_path,private_object=True)
    measured=d.load_dichasus('dichasus-cf06',admission(paths),root)
    assert np.isfinite(measured.cfr).all()
    with pytest.raises(ValueError):d.private_labels(measured,purpose='frozen_probe')
    folder,paths=rt_fixture(tmp_path,private_object=True)
    simulated=d.load_sionna(folder,admission(paths))
    assert np.isfinite(simulated.cfr).all()
    assert simulated.public_mobile_xyz_m is None
    assert not any(k in simulated.encoder_features() for k in ('spec','delay_s','coefficients','rx_positions_m','material_label'))


def test_rt_cfr_is_newly_admitted_observation_and_static_clock_is_not_fabricated(tmp_path):
    folder,paths=rt_fixture(tmp_path)
    observed=d.load_sionna(folder,admission(paths))
    assert observed.cfr.shape==(12,2,64)
    assert observed.capture_time_s is None
    assert observed.gauge.delay_reference=='ideal_simulator_absolute'
    assert observed.coherence_proof is None  # Not inferred merely from solver origin.
    assert not observed.observed[0,1] and not observed.cfr[0,1].any()
    np.testing.assert_array_equal(observed.carrier_hz,[2.4e9,2.4e9,6.e9,6.e9]*3)
    assert observed.link_roles==('AP_to_client','client_to_AP')*6
    labels=d.private_labels(observed,purpose='partition')['mobile_position_m']
    np.testing.assert_array_equal(labels,np.repeat(np.arange(9).reshape(3,3),4,axis=0))
    with pytest.raises(ValueError,match='bound evidence'):d.require_coherent_view(observed)
    with pytest.raises(ValueError,match='explicit evaluation'):d.private_labels(observed,purpose='pretext_teacher')


def test_ideal_rt_phase_and_mobile_cad_pose_require_explicit_new_protocol_flags(tmp_path):
    folder,paths=rt_fixture(tmp_path)
    admitted=admission(paths,admit_ideal_rt_phase=True,public_mobile_pose=True)
    observed=d.load_sionna(folder,admitted)
    assert observed.public_mobile_xyz_m.shape==(12,3)
    assert 'public_mobile_xyz_m' not in observed.encoder_features()  # Still deliberate opt-in at model call site.
    with pytest.raises(ValueError,match='explicit protocol error budget'):
        d.require_coherent_view(observed)
    proof=d.require_coherent_view(observed,budget=PhaseCoherenceBudget('fixture-protocol',.1,1.e-10))
    assert proof.basis=='ideal_simulation' and proof.maximum_phase_error_rad==0
    assert observed.gauge.ideal_simulation_admitted


def test_missing_admission_and_changed_observation_bytes_are_rejected(tmp_path):
    folder,paths=rt_fixture(tmp_path);bound=admission(paths)
    with pytest.raises(TypeError,match='new complex'):d.load_sionna(folder,None)
    with paths['rt:'+folder.name+':cfr'].open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):d.load_sionna(folder,bound)


def test_inspection_cannot_open_unseen_recording_or_field_test(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No observation array should open')
    monkeypatch.setattr(np,'load',forbidden)
    bound=d.ObservationAdmission('fixture','training',{},purpose='inspection')
    with pytest.raises(ValueError,match='training recording'):
        d.load_dichasus('dichasus-ad00',bound,tmp_path)
    with pytest.raises(ValueError,match='training data'):
        d.load_sionna(tmp_path/'field_test/fixture-field',bound)


def test_old_role_or_empty_protocol_does_not_silently_admit_private_truth(tmp_path):
    folder,paths=rt_fixture(tmp_path)
    hashes={key:d.sha256(path) for key,path in paths.items()}
    wrong=d.ObservationAdmission('fixture','training',hashes,data_role='old_private_simulator_truth')
    with pytest.raises(ValueError,match='new explicit'):d.load_sionna(folder,wrong)
    empty=d.ObservationAdmission('','training',hashes)
    with pytest.raises(ValueError,match='new explicit'):d.load_sionna(folder,empty)
