"""Analytic software fixtures only; these are not scientific study results."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import truncnorm

from wirelessjev.spectral_scene import baselines as b
from wirelessjev.spectral_scene.data import LinkRows
from wirelessjev.spectral_scene.operators import fspl_received_power_dbm, link_geometry_features


def rows(scene='fixture-training-0', stage='training', poses=(0,1,2,3,4,5),
         bands=(0,1), devices=(0,), shift=0.):
    indices=np.array([(p,a,f,d) for p in poses for a in range(2) for f in bands for d in devices])
    p,a,f,d=indices.T
    rx=np.column_stack((2.+p*1.4+shift,2.+p*.3,np.ones(len(p))))
    ap=np.column_stack((1.+a*10,np.ones(len(p)),np.full(len(p),2.)))
    hz=np.array((2.4e9,5.2e9,6e9))[f]
    offset=np.array((0.,-3.,4.,0.))[d]
    geometry=link_geometry_features(rx,ap,hz,np.zeros(len(p)),(24,20,5),np.empty((0,2,3)),
        calibration_pair_count=16,calibration_unknown=False,quantization_width_db=.5)
    fspl=fspl_received_power_dbm(rx,ap,hz)
    residual=2.5+.4*geometry[:,6]-3.*geometry[:,7]
    report=np.round((fspl+residual+offset)*2)/2
    return LinkRows(scene,stage,rx,ap,hz,np.zeros(len(p),int),d,f,p,a,report,report-.25,report+.25,
                    np.ones(len(p),bool),np.zeros(len(p),bool),np.full(len(p),-95.),fspl,offset,geometry,
                    tuple(f'{scene}-{i}' for i in range(len(p))))


@pytest.fixture
def prior(monkeypatch):
    common=SimpleNamespace(check_sources=lambda:None,COUNTS={'training':2},
        field_spec=lambda stage,i:SimpleNamespace(scene_id=f'fixture-training-{i}'))
    monkeypatch.setattr(b,'_common',lambda:common)
    return b.learn_prior([rows(devices=(0,1,2)),rows('fixture-training-1',devices=(0,1,2),shift=.4)])


def query_only(r):
    """A query object whose labels raise, to catch accidental target reads."""
    class Query:
        def __getattr__(self,name):
            if name in {'geometry_features','rx_xyz','ap_xyz','carrier_hz','fspl_dbm','offset_db'}:
                return getattr(r,name)
            raise AssertionError('query labels must not be read: '+name)
    return Query()


def test_interval_and_censor_moments():
    mu=np.array((.5,-1.,3.,8.));lo=np.array((-1.,-np.inf,2.5,-np.inf));hi=np.array((2.,0.,4.,-4.))
    expected,variance=b._truncated_moments(mu,lo,hi,1.2)
    tm,tv=truncnorm.stats((lo-mu)/1.2,(hi-mu)/1.2,loc=mu,scale=1.2,moments='mv')
    np.testing.assert_allclose(expected,tm,rtol=1e-9,atol=1e-9)
    np.testing.assert_allclose(variance,tv,rtol=1e-7,atol=1e-8)


def test_source_guard_precedes_training(monkeypatch):
    def blocked():raise RuntimeError('source not published')
    monkeypatch.setattr(b,'_common',lambda:SimpleNamespace(check_sources=blocked))
    with pytest.raises(RuntimeError,match='not published'):b.learn_prior([])


@pytest.mark.parametrize('violation',('validation','heldband','helddevice','missing'))
def test_training_admission(monkeypatch,violation):
    monkeypatch.setattr(b,'_common',lambda:SimpleNamespace(check_sources=lambda:None,COUNTS={'training':1},
        field_spec=lambda stage,i:SimpleNamespace(scene_id='fixture-training-0')))
    r=rows()
    if violation=='validation':r=replace(r,stage='validation')
    if violation=='heldband':r=rows(bands=(2,))
    if violation=='helddevice':r=rows(devices=(3,))
    with pytest.raises(ValueError):b.learn_prior([] if violation=='missing' else [r])


def test_actual_native_core_and_held_carrier(prior,monkeypatch):
    original=b.RadioMap._fit_one;calls=[]
    def spy(self,*args,**kwargs):calls.append(True);return original(self,*args,**kwargs)
    monkeypatch.setattr(b.RadioMap,'_fit_one',spy)
    context=rows(poses=(0,2,4));target=rows(poses=(1,3,5),bands=(2,))
    mean,std=b.FieldBaseline(prior).adapt(context).predict(query_only(target))
    assert len(calls)==4
    assert np.sqrt(np.mean((mean-target.reported_rssi_dbm)**2))<.8
    assert np.isfinite(std).all() and np.all(std>0)


@pytest.mark.parametrize('method',b.METHODS)
def test_nuisance_invariance_and_query_whitelist(prior,method):
    context=rows(poses=(0,2,4));target=rows(poses=(1,3,5),bands=(2,))
    field=b.FieldBaseline(prior,method).adapt(context)
    mean,std=field.predict(query_only(target))
    geometry=target.geometry_features.copy()
    geometry[:,8]*=-1
    geometry[:,11:15]=np.array((0.,1.,.9,-.3))
    changed=replace(target,geometry_features=geometry)
    mean2,std2=field.predict(query_only(changed))
    np.testing.assert_array_equal(mean,mean2)
    assert np.all(std2>std)
    assert prior.metadata['propagation_geometry_columns']==list(range(8))+[9,10]


def test_prior_roundtrip_immutable_and_separate_noise(prior):
    loaded=b.BaselinePrior.from_arrays(prior.as_arrays(),prior.metadata)
    for method in b.METHODS:
        m=loaded.methods[method]
        assert m.noise_db>=1 and .25<=m.measurement_noise_db<=10
        np.testing.assert_array_equal(m.coef,prior.methods[method].coef)
        with pytest.raises(ValueError):m.coef[0]=42
        assert np.all(m.x_scale>0)
    metadata=loaded.metadata;metadata['training_scenes']=99
    assert loaded.metadata['training_scenes']==2


@pytest.mark.parametrize('method',b.METHODS)
def test_censor_bound_is_not_a_fake_point(prior,method):
    context=rows(poses=(0,2,4));heard=context.heard.copy();censor=context.censored.copy()
    heard[0]=False;censor[0]=True
    cb=context.censor_upper_dbm.copy();cb[0]=context.reported_rssi_dbm[0]-7
    evidence=replace(context,heard=heard,censored=censor,censor_upper_dbm=cb)
    point=evidence.reported_rssi_dbm.copy();point[0]=100000
    poisoned=replace(evidence,reported_rssi_dbm=point)
    query=query_only(rows(poses=(1,3,5),bands=(2,)))
    result=b.FieldBaseline(prior,method).adapt(evidence).predict(query)
    other=b.FieldBaseline(prior,method).adapt(poisoned).predict(query)
    np.testing.assert_array_equal(result[0],other[0])
    np.testing.assert_array_equal(result[1],other[1])


def test_same_link_frequency_control_and_missing_mask():
    context=rows(poses=(0,2,4));target=rows(poses=(0,1,2),bands=(2,))
    example=SimpleNamespace(context_rows_a=context,query_rows=query_only(target))
    mean,std,eligible=b.direct_same_link_frequency(example)
    assert eligible.tolist()==[True,True,False,False,True,True]
    assert np.isnan(mean[~eligible]).all() and np.isnan(std[~eligible]).all()
    for i in np.flatnonzero(eligible):
        same=(context.pose_index==target.pose_index[i])&(context.ap_index==target.ap_index[i])&(context.band_index==1)
        expected=context.reported_rssi_dbm[same][0]+20*np.log10(5.2/6.)
        assert mean[i]==pytest.approx(expected)
    geometry=target.geometry_features.copy();geometry[:,12]=1
    example.query_rows=query_only(replace(target,geometry_features=geometry))
    assert not b.direct_same_link_frequency(example)[2].any()


def test_fspl_control_reads_no_query_labels():
    target=rows(bands=(2,));example=SimpleNamespace(query_rows=query_only(target))
    mean,std=b.operator_only_fspl(example)
    np.testing.assert_array_equal(mean,target.fspl_dbm+target.offset_db)
    np.testing.assert_array_equal(std,np.full(len(mean),5.))
