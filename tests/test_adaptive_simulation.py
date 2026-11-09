"""Actual-operator and known-ID RT software fixtures, not study outcomes."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from wirelessjev.adaptive_radio import simulation as sim
from wirelessjev.radiojepa import rt_data as rt


def test_registry_is_fresh_and_conditions_retain_the_exact_same_public_scene():
    registry=sim.geometry_registry()
    assert registry['case_count']==174 and registry['counts']=={'training':24,'validation':6,'development':4,'test':24}
    assert registry['all_new_public_geometries_disjoint_from_previous120'] is True
    fingerprints=[]
    for stage,count in sim.COUNTS.items():
        for i in range(count):
            cases=[sim.case_spec(stage,i,k) for k in sim.CONDITIONS]
            assert len({c.geometry_sha256 for c in cases})==1
            assert len({c.tx_power_dbm for c in cases})==1
            assert len({c.rt_spec.seed for c in cases})==1
            assert {c.hidden_material for c in cases}=={'metal','wood','brick'}
            assert len({c.case_id for c in cases})==3
            assert all(sim.public_geometry(c.rt_spec)==sim.public_geometry(cases[0].geometry) for c in cases)
            fingerprints.append(cases[0].geometry_sha256)
    assert len(fingerprints)==len(set(fingerprints))==58


def test_rendered_power_scaling_produces_actual_censors_and_retains_unresolved_unknowns():
    profiles=rt.DEVICE_PROFILES[:3]
    gain=np.array([1e-8,1e-10,0.]);solved=np.array([True,True,False])
    high=sim.reports_from_rendered_gain(gain,solved,10.,profiles=profiles)
    low=sim.reports_from_rendered_gain(gain,solved,-20.,profiles=profiles)
    assert high['heard'][0].all() and low['censored'][0].all()
    assert low['censored'][1].all()
    assert not high['heard'][2].any() and not high['censored'][2].any()
    np.testing.assert_array_equal(low['reported_detection_threshold_dbm'][low['censored']],low['censor_upper_dbm'][low['censored']])
    assert np.all(low['reported_detection_threshold_dbm']<0)
    assert np.all(high['censor_upper_dbm'][high['heard']]==0)
    assert np.all(high['reported_detection_threshold_dbm'][high['heard']]<0)
    with pytest.raises(ValueError):sim.reports_from_rendered_gain(np.array([-1.]),np.array([True]),0.,profiles=profiles)


def test_registered_cases_cannot_use_the_software_fixture_seam(tmp_path):
    with pytest.raises(RuntimeError):sim.trace_fixture(sim.geometry_spec('training',0),tmp_path)


def test_source_guard_precedes_any_official_trace(tmp_path,monkeypatch):
    called=[]
    def deny():raise RuntimeError('fixture: not published')
    monkeypatch.setattr(sim,'_common',lambda:SimpleNamespace(check_sources=deny))
    monkeypatch.setattr(sim,'_trace_case',lambda *a,**k:called.append('bad'))
    with pytest.raises(RuntimeError,match='not published'):sim.trace_case(sim.case_spec('training',0,0),tmp_path)
    assert not called


def test_actual_known_id_sionna_paths_are_thresholded_without_invented_weak_points(tmp_path,monkeypatch):
    geometry=rt.GeometrySpec('fixture-adaptive-rt','partitioned_room',8.,10.,4.,2.1,'wood',
        ((-2.,-3.,3.),),((-2.,-1.,1.),),671)
    folder=tmp_path/'case'
    receipt=sim.trace_fixture(geometry,folder,condition=0,tx_power_dbm=-70.,
        samples_per_src=128,max_depth=1,max_num_paths_per_src=128)
    assert receipt['real_device_measurements'] is False and receipt['network_delivery_simulated'] is False
    with np.load(folder/'public_views.npz',allow_pickle=False) as public:
        assert public['censored'].any() and not public['heard'].any()
        assert not np.any(public['heard']&public['censored'])
        np.testing.assert_array_equal(public['censor_upper_dbm'][public['censored']],
                                      public['reported_detection_threshold_dbm'][public['censored']])
    with np.load(folder/'reference20/physical_channels.npz',allow_pickle=False) as physical:
        assert physical['solved'].any()
        gain,solved=physical['channel_power_gain'],physical['solved']
    expected=sim.reports_from_rendered_gain(gain,solved,-70.,profiles=rt.DEVICE_PROFILES[:3])
    with np.load(folder/'public_views.npz',allow_pickle=False) as public:
        for name,value in expected.items():np.testing.assert_array_equal(public[name],value)
    out=tmp_path/'out';(out/'registration').mkdir(parents=True);(out/'registration/source.json').write_text('{}')
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    receipt.update(source_receipt_sha256=digest(out/'registration/source.json'),registry_sha256='fixture-registry')
    (folder/'receipt.json').write_text(json.dumps(receipt))
    c=SimpleNamespace(OUT=out,digest=digest,registry_sha256=lambda:'fixture-registry',
                      read_json=lambda p:json.loads(Path(p).read_text()))
    monkeypatch.setattr(sim,'_common',lambda:c)
    spec=sim.CaseSpec('training',0,0,geometry,-70.)
    assert sim.validate_case_receipt(spec,folder)['case_id']==spec.case_id
    (folder/'public_views.npz').write_bytes(b'altered')
    with pytest.raises(RuntimeError,match='raw bytes'):sim.validate_case_receipt(spec,folder)
