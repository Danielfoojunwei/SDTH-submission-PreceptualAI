"""Analytic software fixtures only; no scientific RT generation or fake reports."""
from dataclasses import replace
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from wirelessjev.hierarchical_radio import simulation as sim


def gains(aps=4,power=-60.):
    a=np.full((48,aps,3,2),10**(power/10),dtype=float)
    return a,np.ones(a.shape,dtype=bool)


def test_fixed_fresh_registry_matches186_rt_cases_and_no_prior_cad_overlap():
    registry=sim.geometry_registry()
    assert registry['case_count']==186
    assert registry['all_new_geometries_disjoint_from_prior178']
    assert sum(registry['counts'].values())==62
    fingerprints=[x['public_geometry_sha256'] for stage in registry['stages'].values() for x in stage]
    assert len(fingerprints)==len(set(fingerprints))==62


@pytest.mark.parametrize('stage',list(sim.COUNTS))
def test_localized_states_share_geometry_power_and_sampling_seed(stage):
    cases=[sim.case_spec(stage,0,i) for i in sim.STATES]
    assert len({c.geometry_sha256 for c in cases})==1
    assert len({c.geometry.seed for c in cases})==1
    assert len({c.tx_power_dbm for c in cases})==1
    assert cases[0].materials[0]!=cases[0].materials[1]
    assert cases[0].materials[0]!='metal' and cases[0].materials[1]!='metal'
    assert cases[1].materials==('metal',cases[0].materials[1])
    assert cases[2].materials==(cases[0].materials[0],'metal')
    assert len(cases[0].geometry.ap_positions_m)==(8 if stage=='ap_transfer' else 4)


@pytest.mark.parametrize('stage',list(sim.COUNTS))
def test_antenna_endpoints_avoid_obstacle_interiors(stage):
    for index in range(sim.COUNTS[stage]):
        g=sim.geometry_spec(stage,index)
        points=np.asarray(g.ap_positions_m+g.rx_positions_m)
        assert len(g.rx_positions_m)==48
        for _,lo,hi in sim.obstacle_boxes(g):
            assert not np.all((points>lo)&(points<hi),axis=-1).any()


def test_geometry_seed_or_material_name_is_not_public_geometry_fingerprint():
    g=sim.geometry_spec('training',0)
    assert sim.geometry_fingerprint(g)==sim.geometry_fingerprint(replace(g,seed=g.seed+1,obstacle_material='metal',scene_id='software-fixture'))
    assert sim.geometry_fingerprint(g)!=sim.geometry_fingerprint(replace(g,width_m=g.width_m+.01))


@pytest.mark.parametrize('stage,index,state',[('bad',0,0),('training',True,0),('training',24,0),('training',0,True),('training',0,3)])
def test_case_membership_rejects_unknown_or_boolean_indices(stage,index,state):
    with pytest.raises(ValueError):sim.case_spec(stage,index,state)


def test_cad_bytes_equal_across_states_only_material_assignment_changes(tmp_path):
    roots=[]
    for state in sim.STATES:
        folder=tmp_path/f'state{state}';xml,files=sim.write_scene(sim.case_spec('training',1,state),folder)
        roots.append((folder,ET.parse(xml).getroot()))
        assert len(files)==9
    mesh0={p.name:p.read_bytes() for p in (roots[0][0]/'meshes').glob('*.ply')}
    for folder,_ in roots[1:]:assert {p.name:p.read_bytes() for p in (folder/'meshes').glob('*.ply')}==mesh0
    def material(root,name):
        return next(n for n in root.findall('bsdf') if n.attrib['id']==name).find('string').attrib['value']
    assert material(roots[1][1],'obstacle0')=='metal'
    assert material(roots[1][1],'obstacle1')==material(roots[0][1],'obstacle1')
    assert material(roots[2][1],'obstacle1')=='metal'
    assert material(roots[2][1],'obstacle0')==material(roots[0][1],'obstacle0')


def test_cad_emitter_refuses_overwriting_existing_bytes(tmp_path):
    spec=sim.case_spec('training',0,0);sim.write_scene(spec,tmp_path)
    with pytest.raises(FileExistsError):sim.write_scene(spec,tmp_path)


def test_per_band_top_k_full_rank_uses_kth_upper_not_invented_floor():
    gain,solved=gains(power=-60.)
    result=sim.reports_from_gain(gain,solved,0.,noise_seed=7)
    assert np.all(result['heard'].sum(1)==2)
    assert np.all(result['censored'].sum(1)==2)
    assert np.all(result['censor_upper_dbm'][result['censored']]>result['reported_detection_threshold_dbm'][result['censored']]+10)
    for pose in range(48):
        for band in range(3):
            for direction in range(2):
                for dev in range(2):
                    h=result['heard'][pose,:,band,direction,dev]
                    hi=result['quantization_interval_dbm'][pose,:,band,direction,dev,1]
                    c=result['censored'][pose,:,band,direction,dev]
                    assert np.all(result['censor_upper_dbm'][pose,c,band,direction,dev]==hi[h].min())


def test_not_full_top_k_uses_detector_floor_and_unsolved_is_unknown():
    gain,solved=gains(power=-160.)
    solved[:,0]=False;gain[:,0]=np.nan
    result=sim.reports_from_gain(gain,solved,0.,noise_seed=8)
    assert not result['heard'].any()
    assert not result['censored'][:,0].any()
    assert result['censored'][:,1:].all()
    np.testing.assert_array_equal(result['censor_upper_dbm'][result['censored']],result['reported_detection_threshold_dbm'][result['censored']])


def test_held_band_ranking_cannot_change_source_band_observations():
    gain,solved=gains();base=sim.reports_from_gain(gain,solved,0.,noise_seed=9)
    changed=gain.copy();changed[:,:,2,:]*=1000
    other=sim.reports_from_gain(changed,solved,0.,noise_seed=9)
    for key in ('rssi_dbm','heard','censored','censor_upper_dbm','quantization_interval_dbm'):
        np.testing.assert_array_equal(base[key][:,:,:2],other[key][:,:,:2])


def test_receiver_noise_draw_is_paired_across_counterfactual_power_changes():
    gain,solved=gains(power=-60.)
    a=sim.reports_from_gain(gain,solved,0.,noise_seed=10)
    b=sim.reports_from_gain(gain*100,solved,0.,noise_seed=10)
    np.testing.assert_array_equal(a['heard'],b['heard'])
    np.testing.assert_allclose(b['rssi_dbm'][a['heard']]-a['rssi_dbm'][a['heard']],20.,atol=1e-6)


def test_exact_report_reconstruction_is_deterministic():
    gain,solved=gains()
    a=sim.reports_from_gain(gain,solved,-10.,noise_seed=11)
    b=sim.reports_from_gain(gain,solved,-10.,noise_seed=11)
    assert all(np.array_equal(a[k],b[k]) for k in a)


@pytest.mark.parametrize('corrupt',['zero','nan','mask','seed'])
def test_invalid_physical_arrays_are_refused(corrupt):
    gain,solved=gains();seed=1
    if corrupt=='zero':gain[0,0,0,0]=0
    if corrupt=='nan':gain[0,0,0,0]=np.nan
    if corrupt=='mask':solved=solved.astype('i4')
    if corrupt=='seed':seed=True
    with pytest.raises(ValueError):sim.reports_from_gain(gain,solved,0.,noise_seed=seed)


def test_publication_guard_precedes_any_solver_or_folder_write(tmp_path,monkeypatch):
    called=[]
    monkeypatch.setattr(sim,'_trace_case',lambda *args,**kwargs:called.append(True))
    with pytest.raises(RuntimeError):sim.trace_case(sim.case_spec('training',0,0),tmp_path/'case',authorize=lambda spec:{'authorized':False})
    assert not called and not (tmp_path/'case').exists()


def test_partial_case_refuses_retiming_after_valid_publication(tmp_path,monkeypatch):
    spec=sim.case_spec('training',0,0);folder=tmp_path/'case';folder.mkdir()
    permission={'authorized':True,'case_id':spec.case_id,'source_receipt_sha256':'a'*64,'registry_sha256':'b'*64,'case_path':str(folder)}
    monkeypatch.setattr(sim,'_trace_case',lambda *args,**kwargs:pytest.fail('No replacement scientific call permitted'))
    with pytest.raises(RuntimeError,match='Partial'):sim.trace_case(spec,folder,authorize=lambda spec:permission)


@pytest.mark.parametrize('corrupt',['alternate','relative','invalid_sha','missing_path'])
def test_authorization_binds_exact_destination_and_hex_identity_before_solver(tmp_path,monkeypatch,corrupt):
    spec=sim.case_spec('training',0,0);folder=tmp_path/'case'
    permission={'authorized':True,'case_id':spec.case_id,'source_receipt_sha256':'a'*64,
                'registry_sha256':'b'*64,'case_path':str(folder)}
    if corrupt=='alternate':permission['case_path']=str(tmp_path/'other')
    if corrupt=='relative':permission['case_path']='case'
    if corrupt=='invalid_sha':permission['source_receipt_sha256']='x'*64
    if corrupt=='missing_path':permission.pop('case_path')
    monkeypatch.setattr(sim,'_trace_case',lambda *a,**k:pytest.fail('Unauthorized RT must not run'))
    with pytest.raises(RuntimeError):sim.trace_case(spec,folder,authorize=lambda spec:permission)
    assert not folder.exists()


def receipt_fixture(folder):
    """Temporary analytical byte/likelihood fixture, never scientific evidence."""
    from dataclasses import asdict
    import io
    import time
    spec=sim.case_spec('training',0,0)
    folder.mkdir()
    for name in sim.expected_case_files(spec):
        p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'unit-fixture')
    gain,solved=gains()
    expected=sim.reports_from_gain(gain,solved,spec.tx_power_dbm,noise_seed=spec.geometry.seed)
    for name,items in (('physical_channels.npz',{'channel_power_gain':gain,'solved':solved}),('public_views.npz',expected)):
        stream=io.BytesIO();np.savez_compressed(stream,**items);(folder/name).write_bytes(stream.getvalue())
    attempt={'authorized':True,'case_id':spec.case_id,'case_path':str(folder),
        'source_receipt_sha256':'a'*64,'registry_sha256':'b'*64,'created_unix_ns':time.time_ns()}
    (folder/'attempt.json').write_bytes(sim.encoded(attempt))
    receipt={'schema':1,'case':spec.to_dict(),'case_id':spec.case_id,'public_geometry_sha256':spec.geometry_sha256,
        'unresolved_paths_are_censored':False,'source_receipt_sha256':'a'*64,'registry_sha256':'b'*64,
        'solver':{'samples_per_src':32768,'max_depth':3,'max_num_paths_per_src':4096,'seed':spec.geometry.seed},
        'runtime':{'backend':'llvm_ad_mono_polarized','drjit_threads':4,'requested_drjit_threads':4},
        'bands_hz':list(sim.BANDS_HZ),'bandwidth_hz':sim.BANDWIDTH_HZ,
        'top_k_per_band':sim.TOP_K,'receiver_measurement_sigma_db':sim.MEASUREMENT_SIGMA_DB,
        'receiver_profiles':[asdict(p) for p in sim.PROFILES],
        'files_sha256':{name:sim.digest(folder/name) for name in sim.expected_case_files(spec)}}
    (folder/'receipt.json').write_bytes(sim.encoded(receipt))
    return spec,receipt


def test_passive_scan_reducer_uses_saved_gain_without_rt(tmp_path,monkeypatch):
    spec,_=receipt_fixture(tmp_path/'case')
    monkeypatch.setattr(sim,'_trace_case',lambda *a,**k:pytest.fail('Passive audit cannot trace'))
    assert sim.validate_case_receipt(spec,tmp_path/'case',source_sha256='a'*64,registry_sha256='b'*64,reconstruct_reports=True)


@pytest.mark.parametrize('corrupt',['missing_raw','extra_raw','attempt','solver','reported_bound'])
def test_rehashed_wrong_raw_membership_configuration_or_sensor_event_refuses(tmp_path,corrupt):
    spec,receipt=receipt_fixture(tmp_path/'case');folder=tmp_path/'case'
    if corrupt=='missing_raw':
        name='band-0-direction-0-cir.npz';(folder/name).unlink();receipt['files_sha256'].pop(name)
    if corrupt=='extra_raw':
        (folder/'extra.bin').write_bytes(b'extra');receipt['files_sha256']['extra.bin']=sim.digest(folder/'extra.bin')
    if corrupt=='attempt':
        a=json.loads((folder/'attempt.json').read_text());a['case_id']='wrong';(folder/'attempt.json').write_bytes(sim.encoded(a))
    if corrupt=='solver':receipt['solver']['samples_per_src']=1
    if corrupt=='reported_bound':
        import io
        with np.load(folder/'public_views.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        a['censor_upper_dbm'][a['censored']]-=10.
        stream=io.BytesIO();np.savez_compressed(stream,**a);(folder/'public_views.npz').write_bytes(stream.getvalue())
        receipt['files_sha256']['public_views.npz']=sim.digest(folder/'public_views.npz')
    (folder/'receipt.json').write_bytes(sim.encoded(receipt))
    with pytest.raises(RuntimeError):
        sim.validate_case_receipt(spec,folder,reconstruct_reports=True)

