"""Analytic likelihood/cohort fixtures; no experimental predictions or fits."""
from copy import deepcopy
import math

import numpy as np
import pytest
from scipy.special import log_ndtr,ndtr

from wirelessjev.hierarchical_radio import evaluation as e


def target():
    return {'heard':np.array([True,False,False]),'censored':np.array([False,True,False]),
        'center_dbm':np.array([-90.,0.,0.]),'lower_dbm':np.array([-91.,0.,0.]),
        'upper_dbm':np.array([-89.,0.,0.]),'censor_upper_dbm':np.array([0.,-95.,0.]),
        'detection_threshold_dbm':np.full(3,-90.),'weights':np.ones(3)}


def test_interval_likelihood_central_one_sided_and_uninformative_events():
    lo=np.array([-1.,-np.inf,1.,-np.inf]);hi=np.array([1.,-2.,np.inf,np.inf])
    actual=e.interval_log_mass(lo,hi,0.,1.)
    expected=[math.log(ndtr(1.)-ndtr(-1.)),log_ndtr(-2.),log_ndtr(-1.),0.]
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-14)


def test_narrow_far_tail_mass_is_finite_and_reflection_symmetric():
    lo=40.;hi=40.+1e-6
    positive=e.interval_log_mass(lo,hi,0.,1.)
    negative=e.interval_log_mass(-hi,-lo,0.,1.)
    midpoint=.5*(lo+hi)
    expected=-.5*midpoint**2-.5*math.log(2*math.pi)+math.log(hi-lo)
    assert float(positive)==pytest.approx(expected,abs=2e-8)
    assert float(negative)==pytest.approx(float(positive),abs=1e-12)


@pytest.mark.parametrize('lo,hi,mean,std',[(0.,0.,0.,1.),(1.,0.,0.,1.),
    (np.nan,1.,0.,1.),(-1.,1.,np.nan,1.),(-1.,1.,0.,0.),(-1.,1.,0.,-1.)])
def test_malformed_interval_or_gaussian_rejected(lo,hi,mean,std):
    with pytest.raises(ValueError):e.interval_log_mass(lo,hi,mean,std)


def test_heard_bin_is_clipped_at_real_detector_floor_censor_stays_upper_event():
    t=target();heard,censored,known,lo,hi=e.target_events(t)
    assert lo[0]==-90. and hi[0]==-89. and np.isneginf(lo[1]) and hi[1]==-95.
    mean=np.array([-90.,-95.,0.]);std=np.ones(3)
    result=e.metrics(mean,std,t)
    expected=.5*(-math.log(ndtr(1.)-.5)-math.log(.5))
    assert result['composite_interval_nll']==pytest.approx(expected)
    assert result['rmse_db']==0 and result['heard']==1 and result['censored']==1 and result['unknown']==1


@pytest.mark.parametrize('change',[
    lambda t:t['censored'].__setitem__(0,True),
    lambda t:t.update(heard=t['heard'].astype(int)),
    lambda t:t['upper_dbm'].__setitem__(0,-91.),
    lambda t:t['center_dbm'].__setitem__(0,np.nan),
    lambda t:t.update(center_dbm=np.zeros(2)),
])
def test_metric_rejects_overlapping_masks_bad_endpoints_or_nonfinite_heard_center(change):
    t=target();change(t)
    with pytest.raises(ValueError):e.metrics(np.zeros(3),np.ones(3),t)


def test_unknown_center_is_not_a_fake_point_and_does_not_enter_error():
    t=target();t['center_dbm'][2]=np.nan
    assert e.metrics(np.array([-90.,-95.,1000.]),np.ones(3),t)['rmse_db']==0.


def test_validation_nll_uses_every_example_and_known_event_weights():
    ts={k:np.stack([v,v]) for k,v in target().items()}
    ts['weights'][0]=[3.,1.,500.];ts['weights'][1]=[1.,1.,1.]
    mean=np.tile([-90.,-95.,0.],(2,1));std=np.ones_like(mean)
    heard_loss=-math.log(ndtr(1.)-.5);censor_loss=-math.log(.5)
    expected=.5*((3*heard_loss+censor_loss)/4+(heard_loss+censor_loss)/2)
    assert e.validation_nll(mean,std,ts)==pytest.approx(expected)


@pytest.mark.parametrize('change',[
    lambda m,s,t:(m[:1],s[:1],t),
    lambda m,s,t:(m,s,{**t,'weights':np.ones((1,3))}),
    lambda m,s,t:(np.where(np.indices(m.shape)[1]==2,np.nan,m),s,t),
    lambda m,s,t:(m,np.zeros_like(s),t),
    lambda m,s,t:(m,s,{**t,'weights':np.full(m.shape,-1.)}),
])
def test_validation_cannot_score_partial_or_nonfinite_arrays(change):
    ts={k:np.stack([v,v]) for k,v in target().items()};m=np.tile([-90.,-95.,0.],(2,1));s=np.ones_like(m)
    with pytest.raises(ValueError):e.validation_nll(*change(m,s,ts))


def records():
    return [{'layout_id':layout,'state':state,'method':member,
        'metrics':{'rmse_db':float(10*i+state+offset),'heard':float(1+1000*i)}}
        for i,layout in enumerate(('L0','L1')) for state in (0,1,2)
        for member,offset in (('model-seed0',0),('model-seed1',2))]


def test_whole_layout_reduction_averages_metrics_not_predictions_or_query_counts():
    result=e.reduce_layouts(records(),methods={'model':['model-seed0','model-seed1']},layout_ids=['L0','L1'])
    assert [r['metrics']['rmse_db'] for r in result['model']]==[2.,12.]
    assert np.mean([r['metrics']['rmse_db'] for r in result['model']])==7.
    aliases=e.reduce_layouts(records(),methods={'model':['model-seed0','model-seed1'],
        'selected':['model-seed0','model-seed1']},layout_ids=['L0','L1'])
    assert aliases['model']==aliases['selected']


@pytest.mark.parametrize('change',[
    lambda r:r.pop(),lambda r:r.append(deepcopy(r[0])),
    lambda r:r.append({**deepcopy(r[0]),'layout_id':'unregistered'}),
    lambda r:r.append({**deepcopy(r[0]),'state':3}),
    lambda r:r.append({**deepcopy(r[0]),'method':'unregistered'}),
    lambda r:r[0]['metrics'].update(rmse_db=np.nan),
])
def test_reduction_refuses_missing_duplicate_extra_or_nonfinite_rows(change):
    value=records();change(value)
    with pytest.raises(ValueError):e.reduce_layouts(value,methods={'model':['model-seed0','model-seed1']},layout_ids=['L0','L1'])


@pytest.mark.parametrize('layouts,states',[(['L0','L0'],(0,1,2)),(['L0','L1'],(0,0,1,2))])
def test_declared_geometries_and_states_must_be_unique(layouts,states):
    with pytest.raises(ValueError):e.reduce_layouts(records(),methods={'model':['model-seed0','model-seed1']},layout_ids=layouts,states=states)


def test_bootstrap_is_paired_whole_geometry_and_fixed_four_family_interval():
    b=np.arange(24,dtype=float)+10.;delta=np.linspace(-2.,.5,24);a=b+delta
    actual=e.paired_interval(a,b)
    rng=np.random.default_rng(21770);indices=rng.integers(0,24,(20000,24))
    expected=np.quantile(delta[indices].mean(1),(.00625,.99375))
    np.testing.assert_allclose(actual['central_98_75_interval'],expected,rtol=0,atol=1e-14)
    assert actual['layouts']==24 and actual['bootstrap_samples']==20000 and actual['seed']==21770
    assert actual['mean_delta']==pytest.approx(delta.mean())
    reverse=e.paired_interval(b,a)
    np.testing.assert_allclose(reverse['central_98_75_interval'],-expected[::-1],rtol=0,atol=1e-14)


def test_constant_gain_and_null_have_exact_sign_flags():
    lower=e.paired_interval(np.ones(24),np.full(24,3.))
    assert lower['central_98_75_interval']==[-2.,-2.] and lower['candidate_lower_error_supported'] is True
    null=e.paired_interval(np.ones(24),np.ones(24))
    assert null['central_98_75_interval']==[0.,0.] and null['candidate_lower_error_supported'] is False


@pytest.mark.parametrize('a,b,kwargs',[(np.ones(23),np.ones(23),{}),(np.ones(24),np.ones(23),{}),
    (np.full(24,np.nan),np.ones(24),{}),(np.ones(24),np.ones(24),{'samples':True}),
    (np.ones(24),np.ones(24),{'samples':0}),(np.ones(24),np.ones(24),{'seed':True})])
def test_invalid_cohort_or_bootstrap_identity_is_rejected(a,b,kwargs):
    with pytest.raises(ValueError):e.paired_interval(a,b,**kwargs)
