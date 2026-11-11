"""Reported-observation software fixtures; no RT, fitting or model execution."""
from dataclasses import replace
import io
from pathlib import Path

import numpy as np
import pytest

from wirelessjev.hierarchical_radio import data as d
from wirelessjev.hierarchical_radio import simulation as sim


def scene(stage='training',index=0,state=0,*,unknown=False):
    spec=sim.case_spec(stage,index,state);g=spec.geometry;s=len(g.ap_positions_m)
    values=np.zeros((48,s,3,2))
    for pose in range(48):
        for ap in range(s):values[pose,ap]=-59.-4*((ap+pose)%s)-state*.8
    gain=10**(values/10);solved=np.ones(gain.shape,bool)
    if unknown:solved[:,0]=False
    reports=sim.reports_from_gain(gain,solved,spec.tx_power_dbm,noise_seed=g.seed)
    boxes=np.asarray([[lo,hi] for _,lo,hi in sim.obstacle_boxes(g)])
    return d.PublicScene(stage,index,state,spec.layout_id,spec.case_id,np.asarray(g.ap_positions_m),
        np.asarray(g.rx_positions_m),np.asarray((g.width_m,g.depth_m,g.ceiling_m)),boxes,np.asarray(sim.BANDS_HZ),
        spec.tx_power_dbm,reports['noise_floor_dbm'],reports['rssi_dbm'],reports['heard'],reports['censored'],
        reports['censor_upper_dbm'],reports['quantization_interval_dbm'],reports['reported_detection_threshold_dbm'],{})


@pytest.fixture
def calibrated():
    return d.fit_calibration([scene(index=i) for i in range(3)])


@pytest.mark.parametrize('aps',[4,8])
@pytest.mark.parametrize('regime,size',[('full',192),('sparse',96)])
def test_fixed_total_context_budget_and_disjoint_grid(aps,regime,size):
    rows=d.context_schedule(aps,regime)
    assert len(rows)==size and len(set(rows))==size
    assert all(p in sim.CONTEXT_POSES and p not in sim.QUERY_POSES and 0<=a<aps and b in (0,1) for p,a,b in rows)
    assert len({sum(x[1]==a for x in rows) for a in range(aps)})==1
    if regime=='sparse':assert set(rows)<set(d.context_schedule(aps,'full'))


@pytest.mark.parametrize('stage',list(sim.COUNTS))
@pytest.mark.parametrize('regime,size',[('full',192),('sparse',96)])
def test_example_schema_axes_units_and_independent_target(stage,regime,size,calibrated):
    s=scene(stage);e=d.make_example(s,calibrated,regime=regime)
    assert set(e)=={'a','b','query','target','ap_xyz','binding'}
    for key,device,direction in (('a',0,0),('b',1,1)):
        v=e[key]
        assert v['context'].shape==(size,20)
        assert v['known'].shape==v['scheduled'].shape==v['ap'].shape==(size,)
        assert v['known'].dtype==v['scheduled'].dtype==bool and v['ap'].dtype==np.int64
        assert v['rx_xyz'].shape==(size,3)
        assert v['scheduled'].all() and np.isfinite(v['context']).all() and np.isfinite(v['floor']).all()
        heard=v['context'][:,2]>.5;censored=v['context'][:,3]>.5
        np.testing.assert_array_equal(v['known'],heard|censored)
        assert np.isneginf(v['lower'][censored]).all()
        assert np.isfinite(v['upper'][v['known']]).all()
        assert np.all(v['lower'][heard]<v['upper'][heard])
        assert e['binding']['view_ids'][key]=={'device':device,'direction':direction}
    assert set(e['target'])=={'center_dbm','lower_dbm','upper_dbm','heard','censored','censor_upper_dbm','detection_threshold_dbm'}
    assert len(e['query']['q'])==24*len(s.ap_xyz)*2
    assert e['binding']['query_heard']+e['binding']['query_censored']+e['binding']['query_unknown']==len(e['query']['q'])
    assert not set(e['binding']['context_poses'])&set(e['binding']['query_poses'])
    assert e['binding']['stage']==stage and e['binding']['case_id']==s.case_id


def test_residual_events_match_their_actual_reported_bins(calibrated):
    s=scene();e=d.make_example(s,calibrated)
    for name,dev,direction in (('a',0,0),('b',1,1)):
        v=e[name]
        for row,(p,a,b) in enumerate(d.context_schedule(4,'full')):
            ref=v['fspl'][row]+v['offset'][row]
            assert np.isclose(v['floor'][row]+ref,s.detection_threshold_dbm[p,a,b,direction,dev],atol=1e-5)
            if s.heard[p,a,b,direction,dev]:
                lo,hi=s.quantization_interval_dbm[p,a,b,direction,dev]
                assert np.isclose(v['lower'][row]+ref,max(lo,s.detection_threshold_dbm[p,a,b,direction,dev]),atol=1e-5)
                assert np.isclose(v['upper'][row]+ref,hi,atol=1e-5)
            elif s.censored[p,a,b,direction,dev]:
                assert np.isclose(v['upper'][row]+ref,s.censor_upper_dbm[p,a,b,direction,dev],atol=1e-5)


def test_unknown_links_do_not_become_zero_or_floor_observations(calibrated):
    e=d.make_example(scene(unknown=True),calibrated)
    for name in ('a','b'):
        v=e[name];unknown=~v['known']
        assert unknown.any() and np.all(v['context'][unknown,:4]==0)
        assert np.isneginf(v['lower'][unknown]).all() and np.isposinf(v['upper'][unknown]).all()
    assert e['binding']['query_unknown']>0


def test_model_arrays_are_backed_by_immutable_bytes(calibrated):
    e=d.make_example(scene(),calibrated)
    for name in ('a','b','query','target'):
        for a in e[name].values():
            assert not a.flags.writeable
            with pytest.raises(ValueError):a.setflags(write=True)
    assert not e['ap_xyz'].flags.writeable


def test_calibration_uses_only_train_context_same_direction_reports():
    s=scene();cal=d.fit_calibration([s])
    assert abs(cal['estimates']['1']['offset_db']+3)<.4
    assert cal['estimates']['0']['offset_db']==0 and not cal['absolute_hardware_offset_identified']
    assert all(p['pose'] in sim.CONTEXT_POSES and p['band'] in (0,1) and p['direction']==0 and p['stage']=='training' for p in cal['paired_reports'])
    assert not cal['query_or_held_band_or_true_offset_used']
    with pytest.raises(ValueError):d.fit_calibration([scene('validation')])
    with pytest.raises(ValueError):d.fit_calibration([s,s])


def test_query_and_held_band_changes_cannot_change_calibration():
    s=scene();values=s.rssi_dbm.copy();interval=s.quantization_interval_dbm.copy()
    values[list(sim.QUERY_POSES)]+=7.;interval[list(sim.QUERY_POSES)]+=7.
    values[:,:,2]+=9.;interval[:,:,2]+=9.
    changed=replace(s,rssi_dbm=values,quantization_interval_dbm=interval)
    assert d.fit_calibration([s])==d.fit_calibration([changed])


def test_query_values_and_topk_bounds_never_enter_query_features(calibrated):
    s=scene();base=d.make_example(s,calibrated)
    values=s.rssi_dbm.copy();interval=s.quantization_interval_dbm.copy();bounds=s.censor_upper_dbm.copy()
    values[list(sim.QUERY_POSES)]+=8.;interval[list(sim.QUERY_POSES)]+=8.;bounds[list(sim.QUERY_POSES)]+=17.
    other=d.make_example(replace(s,rssi_dbm=values,quantization_interval_dbm=interval,censor_upper_dbm=bounds),calibrated)
    for name in ('a','b','query'):
        assert all(np.array_equal(base[name][k],other[name][k]) for k in base[name])
    assert base['binding']['model_input_sha256']==other['binding']['model_input_sha256']
    assert base['binding']['target_sha256']!=other['binding']['target_sha256']


def test_condition_and_identity_metadata_do_not_enter_numeric_tensors(calibrated):
    s=scene();a=d.make_example(s,calibrated)
    b=d.make_example(replace(s,state=2,case_id='audit-only-identity',layout_id='audit-only-layout'),calibrated)
    assert a['binding']['case_id']!=b['binding']['case_id']
    assert a['binding']['model_input_sha256']==b['binding']['model_input_sha256']


@pytest.mark.parametrize('stage',['training','validation'])
def test_held_band_cannot_enter_fit_or_selection(stage,calibrated):
    with pytest.raises(ValueError):d.make_example(scene(stage),calibrated,held_band=True)


def test_held_band_queries6ghz_while_source_bands_stay_fixed(calibrated):
    s=scene('test');a=d.make_example(s,calibrated);b=d.make_example(s,calibrated,held_band=True)
    assert len(b['query']['q'])==24*4
    assert np.allclose(np.exp(b['query']['q'][:,7])*2.4e9,6e9,rtol=1e-6)
    for name in ('a','b'):assert all(np.array_equal(a[name][k],b[name][k]) for k in a[name])


def test_stale_material_context_keeps_targets_and_public_queries_unchanged(calibrated):
    current=scene('test',state=1);stale=scene('test',state=0)
    a=d.make_example(current,calibrated);b=d.make_example(current,calibrated,context_scene=stale)
    assert a['binding']['context_case_id']!=b['binding']['context_case_id']
    assert all(np.array_equal(a['target'][k],b['target'][k]) for k in a['target'])
    assert all(np.array_equal(a['query'][k],b['query'][k]) for k in a['query'])
    assert not np.array_equal(a['a']['context'],b['a']['context'])


def public_folder(folder):
    spec=sim.case_spec('training',0,0);s=scene();_,meshes=sim.write_scene(spec,folder/'scene')
    values={'rssi_dbm':s.rssi_dbm,'heard':s.heard,'censored':s.censored,'censor_upper_dbm':s.censor_upper_dbm,
        'quantization_interval_dbm':s.quantization_interval_dbm,'reported_detection_threshold_dbm':s.detection_threshold_dbm,
        'noise_floor_dbm':s.noise_floor_dbm,'ap_positions_m':s.ap_xyz.astype('f4'),'rx_positions_m':s.rx_xyz.astype('f4'),
        'bbox_dims_m':s.dimensions,'carrier_hz':s.carriers}
    stream=io.BytesIO();np.savez_compressed(stream,**values);public=folder/'public_views.npz';public.write_bytes(stream.getvalue())
    receipt={'schema':1,'case':spec.to_dict(),'case_id':spec.case_id,'layout_id':spec.layout_id,
        'public_geometry':sim.public_geometry(spec.geometry),'public_geometry_sha256':spec.geometry_sha256,
        'known_tx_power_dbm':spec.tx_power_dbm,'real_device_measurements':False,'unresolved_paths_are_censored':False,
        'source_receipt_sha256':'a'*64,'registry_sha256':'b'*64,
        'files_sha256':{str(p.relative_to(folder)):sim.digest(p) for p in meshes+[public]}}
    (folder/'receipt.json').write_bytes(sim.encoded(receipt))
    return spec


def test_loader_reads_only_public_npz_and_cad_vertices(tmp_path,monkeypatch):
    spec=public_folder(tmp_path);original=Path.open
    def guarded(path,*args,**kwargs):
        if path.name in ('scene.xml','physical_channels.npz') or path.name.endswith('-cir.npz'):
            pytest.fail('A model reader may not open private channel/material files')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    loaded=d.load_scene(tmp_path,expected_spec=spec,source_sha256='a'*64,registry_sha256='b'*64)
    assert loaded.ap_xyz.shape==(4,3) and loaded.cad_boxes.shape==(8,2,3)


@pytest.mark.parametrize('corrupt',['npz','mesh','source','registry','spec'])
def test_loader_rejects_public_hash_or_admission_changes(tmp_path,corrupt):
    spec=public_folder(tmp_path);source='a'*64;registry='b'*64
    if corrupt=='npz':(tmp_path/'public_views.npz').write_bytes(b'corrupt')
    if corrupt=='mesh':(tmp_path/'scene/meshes/floor.ply').write_text('corrupt')
    if corrupt=='source':source='0'*64
    if corrupt=='registry':registry='0'*64
    if corrupt=='spec':spec=sim.case_spec('training',1,0)
    with pytest.raises((ValueError,RuntimeError)):
        d.load_scene(tmp_path,expected_spec=spec,source_sha256=source,registry_sha256=registry)

