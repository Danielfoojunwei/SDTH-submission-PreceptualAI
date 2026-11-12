"""Analytic and synthetic software fixtures; not scientific RF evidence."""
from copy import deepcopy
import io
import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr

from wirelessjev.hierarchical_radio import baselines as b


def example(case='software-fixture-0', *, stage='training'):
    aps=np.array([[0.,0.,4.],[12.,0.,4.]])
    rx=np.array([[1.,2.,1.],[3.,3.,1.],[7.,2.,1.],[10.,3.,1.]])
    ids=np.repeat(np.arange(2),4);points=np.tile(rx,(2,1))
    def query(points,ids):
        delta=points-aps[ids];distance=np.linalg.norm(delta,axis=1)
        q=np.zeros((len(points),16),dtype='f4')
        q[:,:3]=points/20;q[:,3:6]=aps[ids]/20;q[:,6]=np.log10(distance)/2
        q[:,7]=np.tile([0.,math.log(5.2/2.4)],len(points)//2)
        q[:,9]=.125;q[:,10]=.1;q[:,14]=-1.;q[:,15]=2/3
        return {'q':q,'ap':ids.copy(),'rx_xyz':points.copy(),
                'fspl':np.full(len(points),-50.),'offset':np.zeros(len(points))}
    c=query(points,ids);n=len(points);residual=-8.-2.*c['q'][:,6]
    context=np.column_stack((residual/30,np.zeros(n),np.ones(n),np.zeros(n),c['q'])).astype('f4')
    a={'context':context,'known':np.ones(n,bool),'scheduled':np.ones(n,bool),
       'ap':ids.copy(),'rx_xyz':points.copy(),'lower':residual-.5,'upper':residual+.5,
       'floor':np.full(n,-40.),'fspl':c['fspl'].copy(),'offset':c['offset'].copy()}
    a2=deepcopy(a);q=query(points+np.array([.3,.1,0.]),ids)
    y=-58.-2.*q['q'][:,6]
    target={'center_dbm':y,'lower_dbm':y-.5,'upper_dbm':y+.5,
            'heard':np.ones(n,bool),'censored':np.zeros(n,bool),
            'censor_upper_dbm':np.full(n,-90.),'detection_threshold_dbm':np.full(n,-90.)}
    return {'a':a,'b':a2,'query':q,'target':target,'ap_xyz':aps,
            'binding':{'stage':stage,'case_id':case,'held_band':False,'regime':'full'}}


def manual_prior(method='radiomap_operator'):
    d,k=(66,48) if method=='radiomap_operator' else (24,16)
    length=10. if method=='radiomap_operator' else 8.
    omega,phase=b._fixed_frequencies(k,length,small=method=='hierarchy_fixed')
    return b.Prior(method,length,np.zeros(d),np.ones(d),np.zeros(d),np.eye(d),
                   np.ones(d)*4,2.,omega,phase,'{"software_fixture":true}')


def test_narrow_intervals_match_dense_gaussian_posterior():
    x=np.array([[1.,0.],[1.,1.],[1.,2.]])
    y=np.array([2.,3.,4.]);weights=np.array([1.,.5,2.])
    mean=np.array([.2,-.1]);prior=np.array([[2.,.3],[.3,1.]])
    sigma=.8;precision=np.linalg.inv(prior)+(x.T*weights)@x/sigma**2
    covariance=np.linalg.inv(precision)
    expected=covariance@(np.linalg.solve(prior,mean)+x.T@(weights*y)/sigma**2)
    got,cov=b.interval_posterior(x,y-1e-5,y+1e-5,weights,sigma,mean,prior)
    np.testing.assert_allclose(got,expected,rtol=0,atol=1e-7)
    np.testing.assert_allclose(cov,covariance,rtol=0,atol=1e-7)


def test_native_continuous_point_reports_match_dense_gaussian_exactly():
    x=np.array([[1.,0.],[1.,1.],[1.,2.]])
    y=np.array([-3.,-2.,-1.]);weights=np.array([1.,.5,2.])
    mean0=np.array([.2,-.1]);cov0=np.array([[2.,.3],[.3,1.]])
    sigma=.8;precision=np.linalg.inv(cov0)+(x.T*weights)@x/sigma**2
    covariance=np.linalg.inv(precision)
    mean=covariance@(np.linalg.solve(cov0,mean0)+x.T@(weights*y)/sigma**2)
    got,cov=b.interval_posterior(x,y,y,weights,sigma,mean0,cov0)
    np.testing.assert_allclose(got,mean,rtol=0,atol=1e-12)
    np.testing.assert_allclose(cov,covariance,rtol=0,atol=1e-12)


def test_native_point_context_is_not_given_an_invented_quantization_width():
    e=example();point=(e['a']['lower']+e['a']['upper'])/2
    e['a']['lower']=point.copy();e['a']['upper']=point.copy()
    mean,std=b.predict(manual_prior(),e)
    assert np.isfinite(mean).all() and np.all(std>0)


def test_upper_censor_map_and_information_match_scalar_reference():
    def objective(beta):return .5*beta**2-log_ndtr(-beta)
    reference=minimize_scalar(objective,bounds=(-3.,1.),method='bounded',options={'xatol':1e-13})
    mean,cov=b.interval_posterior(np.ones((1,1)),[-np.inf],[0.],[1.],1.,[0.],[[1.]],max_iterations=256)
    np.testing.assert_allclose(mean,[reference.x],atol=2e-7,rtol=0)
    eps=1e-4
    hessian=(objective(mean[0]+eps)-2*objective(mean[0])+objective(mean[0]-eps))/eps**2
    np.testing.assert_allclose(cov,[[1/hessian]],atol=2e-7,rtol=0)
    assert mean[0]<0 # a zero upper bound is not a zero-dBm point


def test_lower_censor_is_upper_censor_under_sign_reversal():
    a,ca=b.interval_posterior(np.ones((1,1)),[-np.inf],[-2.],[1.],1.,[.5],[[2.]])
    z,cz=b.interval_posterior(np.ones((1,1)),[2.],[np.inf],[1.],1.,[-.5],[[2.]])
    np.testing.assert_allclose(a,-z,atol=1e-10)
    np.testing.assert_allclose(ca,cz,atol=1e-10)


def test_uninformative_and_zero_weight_reports_leave_prior_unchanged():
    mean=np.array([1.,2.]);cov=np.array([[3.,.2],[.2,2.]])
    got,c=b.interval_posterior(np.array([[1.,4.],[100.,200.]]),[-np.inf,-20.],[np.inf,-19.],
                              [1.,0.],2.,mean,cov)
    np.testing.assert_array_equal(got,mean);np.testing.assert_array_equal(c,cov)


def test_vacuous_censor_does_not_shrink_covariance_like_a_point():
    got,c=b.interval_posterior(np.ones((1,1)),[-np.inf],[100.],[1.],1.,[0.],[[2.]])
    np.testing.assert_allclose(got,[0.],atol=1e-12)
    np.testing.assert_allclose(c,[[2.]],atol=1e-12)


def test_empty_ap_uses_prior_not_a_fabricated_observation():
    prior=manual_prior();e=example();e['a']['known'][:]=False
    mean,std=b.predict(prior,e)
    public,_=b._public(e['query'],e['ap_xyz'],view=False);x=prior.design(public)
    expected=np.sqrt(np.einsum('ij,jk,ik->i',x,prior.covariance+np.diag(prior.adaptation_variance),x)+prior.noise_db**2)
    np.testing.assert_allclose(mean,e['query']['fspl'])
    np.testing.assert_allclose(std,expected)
    assert np.all(std>prior.noise_db)


@pytest.mark.parametrize('method',b.METHODS)
def test_target_values_never_enter_prediction(method):
    e=example();prior=manual_prior(method);first=b.predict(prior,e)
    class Forbidden(dict):
        def __getitem__(self,key):raise AssertionError('Target leakage')
    e['target']=Forbidden()
    second=b.predict(prior,e)
    for a,z in zip(first,second):np.testing.assert_array_equal(a,z)


@pytest.mark.parametrize('method',b.METHODS)
def test_ap_permutation_only_relabels_groups(method):
    e=example();prior=manual_prior(method);first=b.predict(prior,e)
    e['ap_xyz']=e['ap_xyz'][::-1].copy()
    for name in ('a','b','query'):e[name]['ap']=1-e[name]['ap']
    second=b.predict(prior,e)
    for a,z in zip(first,second):np.testing.assert_allclose(a,z,rtol=0,atol=1e-10)


def test_current_view_is_not_assimilated_twice():
    e=example();prior=manual_prior();mean,std=b.predict(prior,e)
    public,_=b._public(e['a'],e['ap_xyz'],view=True)
    query,_=b._public(e['query'],e['ap_xyz'],view=False)
    xc,xq=prior.design(public),prior.design(query)
    for site in range(2):
        for logf in (0.,math.log(5.2/2.4)):
            mask=(e['a']['ap']==site)&np.isclose(public['q'][:,7],logf)
            ask=(e['query']['ap']==site)&np.isclose(query['q'][:,7],logf)
            beta,cov=b.interval_posterior(xc[mask],e['a']['lower'][mask],e['a']['upper'][mask],
                np.ones(mask.sum()),prior.noise_db,prior.coefficient,prior.covariance+np.diag(prior.adaptation_variance))
            np.testing.assert_allclose(mean[ask],e['query']['fspl'][ask]+xq[ask]@beta,atol=1e-7)
            np.testing.assert_allclose(std[ask]**2,np.einsum('ij,jk,ik->i',xq[ask],cov,xq[ask])+prior.noise_db**2,atol=1e-6)


def test_source_carrier_corrections_are_adapted_separately():
    e=example();prior=manual_prior();mean,std=b.predict(prior,e)
    band0=e['a']['context'][:,11]==0.
    e['a']['lower'][band0]-=20;e['a']['upper'][band0]-=20
    e['a']['context'][band0,0]-=20/30
    # Keep the legitimately heard event above its software-fixture floor.
    e['a']['floor'][band0]-=20
    changed,_=b.predict(prior,e);band1=e['query']['q'][:,7]>0
    np.testing.assert_allclose(changed[band1],mean[band1],atol=1e-7)
    assert np.all(changed[~band1]<mean[~band1]-1.)


def test_held_carrier_transport_does_not_accept_held_context():
    e=example();e['a']['context'][:,11]=math.log(6./2.4)
    with pytest.raises(ValueError,match='source carriers'):b.predict(manual_prior(),e)


def test_known_intervals_reduce_uncertainty_relative_to_same_ap_without_reports():
    e=example();prior=manual_prior();_,observed=b.predict(prior,e)
    e['a']['known'][:]=False;_,unknown=b.predict(prior,e)
    assert np.all(observed<unknown)


def test_fixed_hierarchy_joint_block_matches_direct_dense_model():
    e=example();prior=manual_prior('hierarchy_fixed');mean,std=b.predict(prior,e)
    c,_=b._public(e['a'],e['ap_xyz'],view=True);q,sites=b._public(e['query'],e['ap_xyz'],view=False)
    xc,xq=prior.design(c),prior.design(q);dim=8+sites*16
    x=np.zeros((len(xc),dim));v=np.zeros((len(xq),dim));x[:,:8]=xc[:,:8];v[:,:8]=xq[:,:8]
    for site in range(sites):
        mc=c['ap']==site;mq=q['ap']==site
        x[mc,8+site*16:8+(site+1)*16]=xc[mc,8:]
        v[mq,8+site*16:8+(site+1)*16]=xq[mq,8:]
    m0=np.concatenate((prior.coefficient[:8],np.tile(prior.coefficient[8:],sites)))
    var=np.concatenate((prior.adaptation_variance[:8],np.tile(prior.adaptation_variance[8:],sites)))
    beta,cov=b.interval_posterior(x,e['a']['lower'],e['a']['upper'],np.ones(len(x)),prior.noise_db,m0,np.diag(var))
    np.testing.assert_allclose(mean,e['query']['fspl']+v@beta)
    np.testing.assert_allclose(std**2,np.einsum('ij,jk,ik->i',v,cov,v)+prior.noise_db**2)


def test_numeric_prior_roundtrips_without_pickle_and_cannot_be_mutated():
    prior=manual_prior();stream=io.BytesIO();np.savez_compressed(stream,**b.pack_prior(prior));stream.seek(0)
    with np.load(stream,allow_pickle=False) as archive:restored=b.load_prior(dict(archive))
    for name in ('coefficient','covariance','omega','phase'):
        np.testing.assert_array_equal(getattr(prior,name),getattr(restored,name))
    with pytest.raises(ValueError):restored.coefficient[0]=1.
    with pytest.raises(ValueError):restored.coefficient.setflags(write=True)
    with pytest.raises(ValueError):b.load_prior(b.pack_prior(prior)|{'extra':np.zeros(1)})


def test_train_prior_correlation_weight_does_not_multiply_physical_sample_size():
    e=example();prior=b.fit_prior([e],10.)
    assert prior.metadata['known_report_rows']==24
    assert prior.metadata['unique_physical_links']==16
    assert prior.metadata['effective_physical_link_mass']==16
    duplicated=b.fit_prior([e,deepcopy(e)],10.)
    np.testing.assert_allclose(duplicated.coefficient,prior.coefficient,atol=1e-10,rtol=0)
    # Rank-deficient 66-feature fixture leaves large prior directions; repeated
    # weighted dot-product summation changes their inverse at roundoff only.
    np.testing.assert_allclose(duplicated.covariance,prior.covariance,atol=1e-8,rtol=0)
    assert duplicated.metadata['effective_physical_link_mass']==16


@pytest.mark.parametrize('stage',['validation','development','test'])
def test_nontrain_reports_cannot_fit_hyperprior(stage):
    with pytest.raises(ValueError,match='TRAIN'):b.fit_prior([example(stage=stage)])


def test_held_band_cannot_fit_hyperprior_even_with_training_label():
    e=example();e['a']['context'][:,11]=math.log(6./2.4)
    with pytest.raises(ValueError,match='Held carriers'):b.fit_prior([e])


def test_explicit_held_band_role_cannot_be_mislabeled_as_training():
    e=example();e['binding']['held_band']=True
    with pytest.raises(ValueError,match='TRAIN'):b.fit_prior([e])


@pytest.mark.parametrize('length',[0.,-1.,math.nan,math.inf,True,8.,9.])
def test_only_registered_lengthscale_grid_is_admitted(length):
    with pytest.raises(ValueError):b.fit_prior([example()],length)


@pytest.mark.parametrize('change',[
    lambda e:e['a']['known'].__setitem__(0,False) or e['a']['scheduled'].__setitem__(1,False),
    lambda e:e['a']['context'].__setitem__((0,3),1.),
    lambda e:e['a']['upper'].__setitem__(0,e['a']['lower'][0]-1.),
    lambda e:e['a']['ap'].__setitem__(0,99),
])
def test_invalid_report_semantics_fail_closed(change):
    e=example();change(e)
    with pytest.raises(ValueError):b.predict(manual_prior(),e)


@pytest.mark.parametrize('lower,upper,weights,noise,prior',[
    ([1.],[0.],[1.],1.,[[1.]]),
    ([math.nan],[1.],[1.],1.,[[1.]]),
    ([-math.inf],[0.],[-1.],1.,[[1.]]),
    ([-math.inf],[0.],[1.],True,[[1.]]),
    ([-math.inf],[0.],[1.],1.,[[0.]]),
])
def test_invalid_interval_models_refused(lower,upper,weights,noise,prior):
    with pytest.raises(ValueError):b.interval_posterior([[1.]],lower,upper,weights,noise,[0.],prior)
