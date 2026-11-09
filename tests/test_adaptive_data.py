"""Adaptive reported-only software fixtures; no reserved study cases are read."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from wirelessjev.adaptive_radio import data,simulation as sim
from wirelessjev.spectral_scene import data as sd
from wirelessjev.spectral_scene.operators import fspl_received_power_dbm
from wirelessjev.radiojepa import rt_data as rt


def fixture_scene(stage='training',index=0,condition=0,*,held_band=False,query_shift=0.):
    internal='field_test' if stage=='test' else stage
    role='held_band' if held_band and stage=='development' else None
    role,bands,devices=sd._role_axes(internal,role)
    positions=np.array([(x,y,1.) for y in (-3.,-1.,1.,3.) for x in (-3.,0.,3.)])
    aps=np.array([[-3.,-3.,3.],[3.,3.,3.]])
    carriers=np.array(rt.BANDS_HZ)[list(bands)]
    reference=fspl_received_power_dbm(positions[:,None,None],aps[None,:,None],carriers[None,None,:],-20.)
    physical=reference-(2.,8.,16.)[condition]
    physical=np.broadcast_to(physical[...,None],(12,2,len(bands),2)).copy()
    gain=10**((physical+20.)/10.);solved=np.ones(gain.shape,bool)
    solved[2,1,1,1]=False
    profiles=tuple(rt.DEVICE_PROFILES[v] for v in devices)
    reports=sim.reports_from_rendered_gain(gain,solved,-20.,profiles=profiles)
    # A software-only query poisoning fixture checks the public input boundary.
    if query_shift:
        mask=reports['heard'][list(sd.QUERY_POSES)]
        values=reports['rssi_dbm'][list(sd.QUERY_POSES)]
        intervals=reports['quantization_interval_dbm'][list(sd.QUERY_POSES)]
        reports['rssi_dbm'][list(sd.QUERY_POSES)]=np.where(mask,values+query_shift,values)
        reports['quantization_interval_dbm'][list(sd.QUERY_POSES)]=np.where(mask[...,None],intervals+query_shift,intervals)
    reports.pop('physical_link_solved')
    layout_id=f'fixture-adaptive-{stage}-{index}'
    return data.AdaptiveScene(internal,index,layout_id+f'-rf{condition}',role,
        aps,positions,np.array([10.,12.,4.]),np.array([[[-.1,-2.,0.],[.1,2.,2.]]]),
        bands,carriers,devices,tuple(rt.DEVICE_PROFILES[v].device_id for v in devices),
        bandwidth_hz=20e6,tx_power_dbm=-20.,source_files_sha256={},
        source_stage=stage,layout_id=layout_id,case_id=layout_id+f'-rf{condition}',condition=condition,
        geometry_sha256='fixture-identical-geometry',**reports)


def calibration():return sd.fit_frontends([fixture_scene(condition=k) for k in sim.CONDITIONS])


def test_scheduled_attempts_and_actual_known_evidence_are_distinct():
    scene=fixture_scene(condition=2);cal=calibration()
    full=data.make_example(scene,cal,regime='full')
    sparse=data.make_example(scene,cal,regime='sparse')
    assert full.scheduled_mask_a.sum()==24 and sparse.scheduled_mask_a.sum()==8
    assert np.all(sparse.mask_a<=sparse.scheduled_mask_a)
    withheld=~sparse.scheduled_mask_a
    assert not sparse.context_rows_a.heard[withheld].any() and not sparse.context_rows_a.censored[withheld].any()
    assert np.all(sparse.context_a[withheld,:4]==0)
    assert sparse.targets.heard.shape==(144,) and sparse.context_a.shape==(24,20)
    assert np.array_equal(full.targets.heard,sparse.targets.heard)
    assert not full.mask_b.all()  # unresolved physical paths remain unknown
    assert set(full.model_inputs())==set(sd.SceneExample.model_inputs(full))|{'scheduled_mask_a','scheduled_mask_b'}


def test_all_conditions_share_geometry_power_and_independent_attempt_schedule():
    cal=calibration();examples=[data.make_example(fixture_scene(condition=k),cal,regime='sparse') for k in sim.CONDITIONS]
    for example in examples[1:]:
        np.testing.assert_array_equal(example.query,examples[0].query)
        np.testing.assert_array_equal(example.query_fspl_dbm,examples[0].query_fspl_dbm)
        np.testing.assert_array_equal(example.scheduled_mask_a,examples[0].scheduled_mask_a)
    assert not np.array_equal(examples[0].context_a,examples[2].context_a)
    assert not {'condition','material','case_id','layout_id'}&set(examples[0].model_inputs())


def test_stale_context_keeps_current_query_labels_and_exactly_same_public_information():
    cal=calibration();base=fixture_scene(condition=0);current=fixture_scene(condition=2)
    normal=data.make_example(current,cal,regime='full')
    stale=data.make_example(current,cal,regime='full',context_scene=base,stale=True)
    baseline=data.make_example(base,cal,regime='full')
    for name,value in baseline.model_inputs().items():np.testing.assert_array_equal(value,stale.model_inputs()[name])
    for name,value in normal.target_arrays().items():np.testing.assert_array_equal(value,stale.target_arrays()[name])
    assert stale.context_case_id==base.case_id and stale.case_id==current.case_id
    assert stale.query_rows.condition==2 and stale.context_rows_a.condition==0
    with pytest.raises(ValueError):data.make_example(replace(current,tx_power_dbm=-10.),cal,context_scene=base,stale=True)


def test_actual_detection_threshold_is_separate_from_censor_only_bound_and_not_a_feature():
    scene=fixture_scene();cal=calibration();example=data.make_example(scene,cal)
    target=example.target_arrays()
    assert 'detection_threshold_dbm' in target and 'detection_threshold_dbm' not in example.model_inputs()
    assert np.all(target['detection_threshold_dbm']<0)
    assert np.all(target['censor_upper_dbm'][target['heard']]==0)
    np.testing.assert_array_equal(target['detection_threshold_dbm'][target['censored']],target['censor_upper_dbm'][target['censored']])
    with pytest.raises(ValueError):replace(example.targets,detection_threshold_dbm=np.zeros(len(target['heard'])))
    changed=replace(scene,reported_detection_threshold_dbm=np.where(scene.heard,scene.reported_detection_threshold_dbm-1.,scene.reported_detection_threshold_dbm))
    other=data.make_example(changed,cal)
    for name,value in example.model_inputs().items():np.testing.assert_array_equal(value,other.model_inputs()[name])
    assert not np.array_equal(target['detection_threshold_dbm'],other.target_arrays()['detection_threshold_dbm'])


def test_query_values_do_not_enter_context_inputs_and_held_band_never_enters_fitting():
    cal=calibration();first=data.make_example(fixture_scene(),cal)
    second=data.make_example(fixture_scene(query_shift=20.),cal)
    for name,value in first.model_inputs().items():np.testing.assert_array_equal(value,second.model_inputs()[name])
    assert not np.array_equal(first.targets.center_dbm,second.targets.center_dbm)
    with pytest.raises(ValueError):data.make_example(fixture_scene(),cal,held_band=True)
    held=data.make_example(fixture_scene('development',held_band=True),cal,held_band=True)
    assert set(held.query_rows.band_index)=={2} and set(held.query_rows.pose_index)==set(data.CONTEXT_POSES)
    assert set(held.context_rows_a.band_index)=={0,1}


def test_zero_known_rf_is_preserved_for_prior_return_not_invented_points():
    scene=fixture_scene();cal=calibration()
    unknown=replace(scene,heard=np.zeros_like(scene.heard),censored=np.zeros_like(scene.censored))
    example=data.make_example(unknown,cal,regime='sparse')
    assert not example.mask_a.any() and example.scheduled_mask_a.sum()==8
    assert np.all(example.context_a[:,:4]==0)
    assert not example.targets.heard.any() and not example.targets.censored.any()


def test_training_rows_and_calibration_are_train_only_with_no_query_calibration():
    scene=fixture_scene();cal=calibration();rows=data.training_rows(scene,cal)
    assert len(rows.heard)==288 and set(rows.pose_index)==set(range(12))
    assert set(rows.band_index)=={0,1} and set(rows.device)=={0,1,2}
    assert all(row['pose'] in data.CONTEXT_POSES and row['band'] in (0,1) for row in cal.paired_reports)
    with pytest.raises(ValueError):data.training_rows(fixture_scene('validation'),cal)


def test_example_receipt_binds_attempts_known_masks_targets_and_context_case():
    cal=calibration();example=data.make_example(fixture_scene(),cal,regime='sparse')
    binding=data.example_binding(example)
    assert binding['scheduled_attempts_a']==8
    assert binding['sensed_known_a']==example.mask_a.sum()
    assert set(binding['target_sha256'])==set(example.target_arrays())
    assert binding['network_delivery_or_real_capture_timing_claimed'] is False
    with pytest.raises(ValueError):example.scheduled_mask_a.setflags(write=True)


def test_learned_loader_binds_stage_and_never_opens_private_gain_or_material_inputs(tmp_path,monkeypatch):
    source=fixture_scene('test');out=tmp_path/'out';folder=out/'rt_data/test'/source.case_id
    folder.mkdir(parents=True);(out/'registration').mkdir();(out/'registration/source.json').write_text('{}')
    geometry=rt.GeometrySpec(source.layout_id,'partitioned_room',10.,12.,4.,2.,'wood',
        tuple(map(tuple,source.ap_positions_m)),tuple(map(tuple,source.rx_positions_m)),1)
    spec=sim.CaseSpec('test',0,0,geometry,-20.)
    monkeypatch.setattr(sim,'case_spec',lambda *args:spec)
    names=('rssi_dbm','heard','censored','censor_upper_dbm','quantization_interval_dbm','noise_floor_dbm',
        'reported_detection_threshold_dbm','ap_positions_m','rx_positions_m','carrier_hz','device_ids')
    arrays={name:getattr(source,name) for name in names}
    arrays.update(physical_link_solved=np.zeros((12,2,3,2),bool),channel_power_gain=np.ones((12,2,3,2)))
    public=folder/'public_views.npz';np.savez_compressed(public,**arrays)
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    receipt={'schema':1,'case_id':spec.case_id,'layout_id':spec.layout_id,'public_geometry_sha256':spec.geometry_sha256,
        'known_tx_power_dbm':-20.,'source_receipt_sha256':digest(out/'registration/source.json'),
        'registry_sha256':'fixture-registry','real_device_measurements':False,'unresolved_paths_are_censored':False,
        'public_geometry':sim.public_geometry(geometry),'bandwidth_hz':20e6,'hidden_material':'POISON',
        'files_sha256':{'public_views.npz':digest(public),'reference20/physical_channels.npz':'NEVER_READ'},'mesh_sha256':{}}
    receipt_path=folder/'receipt.json';receipt_path.write_text(json.dumps(receipt))
    manifest={'schema':1,'stage':'test','completed':True,'source_receipt_sha256':digest(out/'registration/source.json'),
        'registry_sha256':'fixture-registry','case_receipts':{f'fixture-{i}':{} for i in range(3*sim.COUNTS['test']-1)}}
    manifest['case_receipts'][spec.case_id]={'path':str(receipt_path.relative_to(tmp_path)),'sha256':digest(receipt_path)}
    manifest_path=out/'rt_data/test.json';manifest_path.write_text(json.dumps(manifest))
    c=SimpleNamespace(ROOT=tmp_path,OUT=out,check_sources=lambda:None,authorize_stage=lambda stage:None,
        case_path=lambda *args:folder,digest=digest,registry_sha256=lambda:'fixture-registry',
        read_json=lambda path:json.loads(Path(path).read_text()),write_once=lambda path,value:path)
    monkeypatch.setattr(data,'_common',lambda:c)
    opened=[];original=np.load
    def guarded(path,*args,**kwargs):
        opened.append(Path(path).name)
        assert Path(path).name=='public_views.npz'
        saved=original(path,*args,**kwargs)
        class PublicOnly:
            def __enter__(self):return self
            def __exit__(self,*args):saved.close()
            def __getitem__(self,key):
                assert key not in ('physical_link_solved','channel_power_gain','cfr')
                return saved[key]
        return PublicOnly()
    monkeypatch.setattr(np,'load',guarded)
    loaded=data.load_scene('test',0,0)
    assert loaded.source_stage=='test' and loaded.stage=='field_test'
    assert not hasattr(loaded,'hidden_material') and opened==['public_views.npz']
    manifest['case_receipts'][spec.case_id]['sha256']='altered'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError,match='immutable collector'):data.load_scene('test',0,0)
    assert opened==['public_views.npz']


def test_new_calibration_and_examples_use_only_new_out_and_preserve_case_order(tmp_path,monkeypatch):
    out=tmp_path/'adaptive';(out/'registration').mkdir(parents=True);(out/'registration/source.json').write_text('{}')
    (out/'rt_data').mkdir()
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for stage in ('training','validation'):
        manifest={'schema':1,'stage':stage,'completed':True,'source_receipt_sha256':digest(out/'registration/source.json'),
            'registry_sha256':'fixture-registry','case_receipts':{f'fixture-{i}':{} for i in range(3*sim.COUNTS[stage])}}
        (out/'rt_data'/f'{stage}.json').write_text(json.dumps(manifest))
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(value,sort_keys=True)
        if path.exists():assert path.read_text()==text
        else:path.write_text(text)
        return path
    c=SimpleNamespace(ROOT=tmp_path,OUT=out,check_sources=lambda:None,digest=digest,
        registry_sha256=lambda:'fixture-registry',read_json=lambda path:json.loads(Path(path).read_text()),
        write_once=write,safe_path=lambda name:tmp_path/name)
    monkeypatch.setattr(data,'_common',lambda:c)
    monkeypatch.setattr(data,'load_scene',lambda stage,index,condition,role=None:fixture_scene(stage,index,condition))
    receipt=data.prepare_training()
    assert receipt['training_cases']==72 and receipt['calibration_sha256']==digest(out/'data/frontend-training.json')
    assert data.load_calibration().absolute_offset_identified is False
    examples=data.examples('validation',regime='sparse',stale=True)
    assert len(examples)==18
    assert [(e.layout_id,e.condition) for e in examples]==[(f'fixture-adaptive-validation-{i}',k) for i in range(6) for k in (0,1,2)]
    assert all(e.context_case_id.endswith('-rf0') for e in examples)
    assert data.prepare_training()==receipt
    assert not (tmp_path/'artifacts/radiojepa_spectral').exists()
