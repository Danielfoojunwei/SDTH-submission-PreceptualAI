"""Reported-only data software fixtures; no reserved RT scenes are opened."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wirelessjev.spectral_scene import data
from wirelessjev.spectral_scene.operators import fspl_received_power_dbm


def fixture_scene(stage='training',role=None,index=0, *, query_shift=0.):
    role,bands,devices=data._role_axes(stage,role)
    positions=np.array([(x,y,1.) for y in (-3.,-1.,1.,3.) for x in (-3.,0.,3.)])
    aps=np.array([[-3.,-3.,3.],[3.,3.,3.]])
    carriers=np.array([2.4e9,5.2e9,6e9])[list(bands)]
    signal=fspl_received_power_dbm(positions[:,None,None],aps[None,:,None],carriers[None,None,:])-7.
    signal=np.broadcast_to(signal[...,None],(12,2,len(bands),2))
    offsets=np.array([0.,-3.,4.,6.])[list(devices)]
    width=np.array([1.,.5,1.,2.])[list(devices)]
    values=np.floor((signal[...,None]+offsets)/width+.5)*width
    values[data.QUERY_POSES,:,:,:,:]+=query_shift
    heard=np.ones(values.shape,bool);censored=np.zeros(values.shape,bool)
    heard[0,0,0,0,0]=False;censored[0,0,0,0,0]=True
    heard[2,1,1,1,1]=False
    bound=np.where(censored,-90.,0.)
    intervals=np.stack((values-width/2,values+width/2),-1)
    return data.PublicScene(stage,index,f'fixture-{stage}-{index}',role,aps,positions,np.array([10.,12.,4.]),
        np.array([[[-.1,-2.,0.],[.1,2.,2.]]]),bands,carriers,devices,
        tuple(f'fixture-device-{d}' for d in devices),values,heard,censored,bound,intervals,
        np.array([-97.,-94.,-92.,-90.])[list(devices)],20e6,20.,{})


def test_both_band_scene_views_have_no_query_rf_or_identity_features():
    scene=fixture_scene();cal=data.fit_frontends([scene]);example=data.make_example(scene,calibration=cal)
    assert example.context_a.shape==example.context_b.shape==(24,20)
    assert example.query.shape==(144,16) and example.query_basis.shape==(144,8)
    assert not set(example.context_rows_a.measurement_ids)&set(example.context_rows_b.measurement_ids)
    assert not set(example.context_rows_a.measurement_ids+example.context_rows_b.measurement_ids)&set(example.query_rows.measurement_ids)
    assert set(example.model_inputs())=={'context_a','context_b','query','query_basis','mask_a','mask_b','query_fspl_dbm','query_offset_db'}
    assert all(np.asarray(x).dtype.kind in 'fb' for x in example.model_inputs().values())
    assert not {'scene_id','material','seed','targets'}&set(example.model_inputs())
    assert example.context_a[0,2]==0 and example.context_a[0,3]==1 and example.context_a[0,0]==0
    assert example.mask_a[0]


def test_training_query_labels_cannot_change_context_or_query_physics_inputs():
    scene=fixture_scene();cal=data.fit_frontends([scene])
    first=data.make_example(scene,calibration=cal)
    poisoned=fixture_scene(query_shift=23.)
    second=data.make_example(poisoned,calibration=cal)
    for name,value in first.model_inputs().items():np.testing.assert_array_equal(value,second.model_inputs()[name])
    assert not np.array_equal(first.targets.center_dbm,second.targets.center_dbm)
    other_cal=data.fit_frontends([poisoned])
    assert data.calibration_record(cal)==data.calibration_record(other_cal)


def test_relative_frontend_offsets_are_estimated_from_observed_pairs_only():
    scene=fixture_scene();cal=data.fit_frontends([scene]);record=data.calibration_record(cal)
    assert abs(cal.estimate(1).offset_db+3.)<=.75 and abs(cal.estimate(2).offset_db-4.)<=1.
    assert cal.estimate(0).offset_db==0. and cal.absolute_offset_identified is False
    assert record['registry_true_offsets_used'] is False
    assert all(row['pose'] in data.CONTEXT_POSES and row['band'] in (0,1) and row['stage']=='training' for row in record['paired_reports'])
    with pytest.raises(ValueError):data.fit_frontends([fixture_scene('validation')])


def test_held_device_onboarding_uses_exact_context_pair_budgets_not_query_labels():
    cal=data.fit_frontends([fixture_scene()]);scene=fixture_scene('development','onboarding')
    no=data.onboard_device(scene,cal,n_pairs=0)
    assert no.estimate(3).offset_db==0. and no.estimate(3).known is False
    for n in (1,4,16):
        adapted=data.onboard_device(scene,cal,n_pairs=n)
        assert adapted.estimate(3).pair_count==n
        assert abs(adapted.estimate(3).offset_db-6.)<=1.5
        assert len(adapted.paired_reports)-len(cal.paired_reports)==n
        same=data.onboard_device(fixture_scene('development','onboarding',query_shift=50.),cal,n_pairs=n)
        assert adapted.estimates==same.estimates
    with pytest.raises(ValueError):data.onboard_device(scene,cal,n_pairs=2)
    with pytest.raises(ValueError):data.onboard_device(fixture_scene(),cal,n_pairs=1)


def test_held_band_and_device_are_absent_from_fitting_and_calibration():
    scene=fixture_scene();cal=data.fit_frontends([scene])
    assert scene.band_indices==(0,1) and scene.device_indices==(0,1,2)
    with pytest.raises(ValueError):data.make_example(scene,calibration=cal,query_keys=(data.ViewKey(2,0,0),))
    with pytest.raises(ValueError):data.make_example(scene,context_a=(data.ViewKey(0,0,0),),calibration=cal)
    with pytest.raises(ValueError):data.make_example(scene,context_b=data.VIEW_A,calibration=cal)
    with pytest.raises(ValueError):data._role_axes('training','held_band')
    held=fixture_scene('development','held_band')
    example=data.make_example(held,calibration=cal,query_keys=data.view_keys(held,'held_band'))
    assert example.query.shape==(72,16)
    assert np.all(example.query_rows.band_index==2)
    assert set(example.context_rows_a.band_index)=={0,1}


def test_same_link_held_band_endpoint_uses_context_poses_without_copying_rf():
    scene=fixture_scene();cal=data.fit_frontends([scene])
    with pytest.raises(ValueError):data.make_example(scene,calibration=cal,query_poses=data.CONTEXT_POSES)
    held=fixture_scene('development','held_band')
    with pytest.raises(ValueError):data.make_example(held,calibration=cal,query_poses=data.CONTEXT_POSES)
    example=data.make_example(held,calibration=cal,query_keys=data.view_keys(held,'held_band'),query_poses=data.CONTEXT_POSES)
    assert set(example.query_rows.pose_index)==set(data.CONTEXT_POSES)
    assert set(example.query_rows.band_index)=={2}
    assert not set(example.query_rows.measurement_ids)&set(example.context_rows_a.measurement_ids)
    source={(int(p),int(a)) for p,a in zip(example.context_rows_a.pose_index,example.context_rows_a.ap_index)}
    target={(int(p),int(a)) for p,a in zip(example.query_rows.pose_index,example.query_rows.ap_index)}
    assert source==target


def test_full_training_prior_rows_remain_train_only_without_held_views():
    scene=fixture_scene();cal=data.fit_frontends([scene]);rows=data.training_rows(scene,cal)
    assert len(rows.carrier_hz)==288 and set(rows.pose_index)==set(range(12))
    assert set(rows.band_index)=={0,1} and set(rows.device)=={0,1,2}
    with pytest.raises(ValueError):data.training_rows(fixture_scene('validation'),cal)
    with pytest.raises(ValueError):rows.geometry_features.setflags(write=True)


def test_public_loader_opens_only_reported_file_and_cad_geometry(tmp_path,monkeypatch):
    scene=fixture_scene('field_test')
    folder=tmp_path/'rt'/'fixture';folder.mkdir(parents=True)
    public=folder/'public_views.npz'
    arrays={name:getattr(scene,name) for name in ('rssi_dbm','heard','censored','censor_upper_dbm','quantization_interval_dbm',
        'noise_floor_dbm','ap_positions_m','rx_positions_m','carrier_hz','device_ids')}
    arrays['physical_link_solved']=np.ones((12,2,3,2),bool)
    np.savez_compressed(public,**arrays)
    receipt={'scene_id':'fixture-loader','spec_sha256':'fixture-geometry','spec':{'width_m':10.,'depth_m':12.,'ceiling_m':4.,'seed':999999,'obstacle_material':'POISON'},
        'reference_tx_dbm':20.,'bandwidth_hz':20e6,
        'receiver_profiles':[{'rssi_offset_db':99999.}],
        'files_sha256':{'public_views.npz':hashlib.sha256(public.read_bytes()).hexdigest(),
                        'physical_channels.npz':'DO_NOT_OPEN'}}
    (folder/'receipt.json').write_text(json.dumps(receipt))
    out=tmp_path/'output';(out/'registration').mkdir(parents=True);(out/'registration/source.json').write_text('{}')
    (out/'rt_data').mkdir()
    stage_manifest={'schema':1,'stage':'training','completed':True,'registry_sha256':'fixture-registry',
        'source_receipt_sha256':hashlib.sha256((out/'registration/source.json').read_bytes()).hexdigest(),
        'real_device_measurements':False,'original_service_confirmation_consumed':False,
        'scene_receipts':{'fixture-loader':{'path':str((folder/'receipt.json').relative_to(tmp_path)),
                                          'sha256':hashlib.sha256((folder/'receipt.json').read_bytes()).hexdigest()}}}
    (out/'rt_data/training.json').write_text(json.dumps(stage_manifest))
    opened=[];original=np.load
    def guarded(path,*args,**kwargs):
        opened.append(Path(path).name)
        assert Path(path).name=='public_views.npz'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',guarded)
    c=SimpleNamespace(ROOT=tmp_path,OUT=out,check_sources=lambda:None,authorize_stage=lambda stage:None,
        rt_scene_path=lambda stage,index:folder,write_once=lambda path,value:path,
        read_json=lambda path:json.loads(path.read_text()),registry_sha256=lambda:'fixture-registry',
        field_spec=lambda stage,index:SimpleNamespace(scene_id='fixture-loader',sha256='fixture-geometry'),
        protocol=lambda:{'scene_counts':{'training':1}})
    monkeypatch.setattr(data,'_common',lambda:c)
    loaded=data.load_scene('training',0)
    assert opened==['public_views.npz'] and loaded.band_indices==(0,1) and loaded.device_indices==(0,1,2)
    cal=data.fit_frontends([loaded])
    assert abs(cal.estimate(1).offset_db+3.)<1.
    assert not hasattr(loaded,'seed') and not hasattr(loaded,'material') and not hasattr(loaded,'receiver_profiles')


def test_calibration_json_round_trip_and_standard_stage_helpers(tmp_path,monkeypatch):
    root=tmp_path;out=root/'artifacts';out.mkdir()
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        text=json.dumps(value,sort_keys=True)
        if path.exists():assert path.read_text()==text
        else:path.write_text(text)
        return path
    (out/'registration').mkdir();(out/'registration/source.json').write_text('{}');(out/'rt_data').mkdir()
    for stage,count in (('training',2),('validation',1)):
        manifest={'schema':1,'stage':stage,'completed':True,'registry_sha256':'fixture-registry',
            'source_receipt_sha256':hashlib.sha256((out/'registration/source.json').read_bytes()).hexdigest(),
            'real_device_measurements':False,'original_service_confirmation_consumed':False,
            'scene_receipts':{f'fixture-{stage}-{i}':{} for i in range(count)}}
        (out/'rt_data'/f'{stage}.json').write_text(json.dumps(manifest))
    c=SimpleNamespace(ROOT=root,OUT=out,check_sources=lambda:None,protocol=lambda:{'scene_counts':{'training':2,'validation':1}},
        write_once=write,read_json=lambda path:json.loads(path.read_text()),safe_path=lambda name:root/name)
    c.registry_sha256=lambda:'fixture-registry'
    monkeypatch.setattr(data,'_common',lambda:c)
    monkeypatch.setattr(data,'load_scene',lambda stage,index,role=None:fixture_scene(stage,role,index))
    receipt=data.prepare_training()
    assert receipt['completed'] and receipt['training_scenes']==2
    loaded=data.load_calibration()
    assert loaded.estimate(0).offset_db==0.
    assert len(data.examples('validation',role='fit'))==1
    assert data.prepare_training()==receipt
