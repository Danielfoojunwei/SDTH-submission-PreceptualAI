"""Analytic software fixtures, never scientific RT/model result artifacts."""
from copy import deepcopy
import math

import numpy as np
import pytest
import torch
from scipy.stats import truncnorm

from wirelessjev.hierarchical_radio.model import (
    ALIGNMENT_WEIGHT, BACKTRACKING_STEPS, CELLS, GLOBAL_DIM, LOCAL_DIM,
    OBJECTIVE_TOLERANCE, HierarchicalField, _bounded_step,
    block_gaussian_posterior, interval_moments, objective, observation_nll,
)
from wirelessjev.spectral_scene.operators import query_physics_basis


def geometry(rx,aps,ids,*,dtype=torch.float64):
    chosen=torch.gather(aps,1,ids[...,None].expand(*ids.shape,3))
    q=torch.zeros((*ids.shape,16),dtype=dtype)
    q[...,:3]=rx/torch.tensor((20.,20.,5.),dtype=dtype)
    q[...,3:6]=chosen/torch.tensor((20.,20.,5.),dtype=dtype)
    q[...,6]=torch.linalg.vector_norm(rx-chosen,dim=-1).log10()/2
    q[...,7]=math.log(5.2/2.4)
    q[...,8]=-1.
    q[...,9]=.125
    q[...,10]=.06
    q[...,13]=.25
    q[...,14]=-1.
    q[...,15]=2/3
    return q


def fixture(*,batch=2,sites=4,tokens=12,queries=7,cell='hier_reg',dtype=torch.float64):
    generator=torch.Generator().manual_seed(571)
    aps=torch.rand((batch,sites,3),generator=generator,dtype=dtype)*8
    aps[...,2]=3.5
    ids=torch.arange(tokens).remainder(sites)[None].expand(batch,-1).clone()
    rx=torch.rand((batch,tokens,3),generator=generator,dtype=dtype)*6
    rx[...,2]=1.
    q=geometry(rx,aps,ids,dtype=dtype)
    residual=-12.-.3*rx[...,0]+.2*rx[...,1]
    heard=torch.ones((batch,tokens),dtype=torch.bool)
    if tokens:heard[:,-1]=False
    known=torch.ones_like(heard);scheduled=known.clone()
    lower=residual-.5;upper=residual+.5
    lower=torch.where(heard,lower,torch.full_like(lower,-torch.inf))
    context=torch.cat((torch.where(heard,residual/30,torch.zeros_like(residual))[...,None],
        torch.where(~heard,upper/30,torch.zeros_like(residual))[...,None],
        heard[...,None],(~heard)[...,None],q),-1)
    view={'context':context,'known':known,'scheduled':scheduled,'ap':ids,'rx_xyz':rx,
        'lower':lower,'upper':upper,'floor':torch.full_like(lower,-50.)}
    qids=torch.arange(queries).remainder(sites)[None].expand(batch,-1).clone()
    qrx=torch.rand((batch,queries,3),generator=generator,dtype=dtype)*6
    qrx[...,2]=1.
    query_q=geometry(qrx,aps,qids,dtype=dtype)
    query={'q':query_q,'ap':qids,'rx_xyz':qrx,'basis':query_physics_basis(query_q),
           'fspl':torch.full((batch,queries),-50.,dtype=dtype),'offset':torch.zeros((batch,queries),dtype=dtype)}
    truth=query['fspl']-12.-.3*qrx[...,0]+.2*qrx[...,1]
    target={'heard':torch.ones_like(truth,dtype=torch.bool),'censored':torch.zeros_like(truth,dtype=torch.bool),
            'lower':truth-.5,'upper':truth+.5}
    model=HierarchicalField(cell).to(dtype=dtype)
    return model,view,query,target,aps


def dense_reference(g,l,groups,w,y,gm,gp,lm,lp):
    batch,n,dg=g.shape;sites,dl=lm.shape[1:]
    full=g.new_zeros((batch,n,dg+sites*dl));full[...,:dg]=g
    for b in range(batch):
        for t in range(n):full[b,t,dg+groups[b,t]*dl:dg+(groups[b,t]+1)*dl]=l[b,t]
    precision=g.new_zeros((batch,dg+sites*dl,dg+sites*dl))
    precision[:,:dg,:dg]=gp
    for a in range(sites):precision[:,dg+a*dl:dg+(a+1)*dl,dg+a*dl:dg+(a+1)*dl]=lp[:,a]
    mean=torch.cat((gm,lm.reshape(batch,-1)),-1)
    posterior=precision+torch.einsum('bnd,bn,bne->bde',full,w,full)
    rhs=torch.einsum('bde,be->bd',precision,mean)+torch.einsum('bnd,bn,bn->bd',full,w,y)
    return torch.linalg.solve(posterior,rhs),torch.linalg.inv(posterior),full


@pytest.mark.parametrize('sites',[1,2,5])
def test_block_schur_matches_full_dense_gaussian_mean_and_covariance(sites):
    generator=torch.Generator().manual_seed(932)
    batch,n,dg,dl=2,11,3,4
    rand=lambda *shape:torch.randn(shape,generator=generator,dtype=torch.float64)
    g=rand(batch,n,dg);l=rand(batch,n,dl)
    groups=torch.arange(n).remainder(sites)[None].expand(batch,-1)
    w=rand(batch,n).square()+.1;y=rand(batch,n)
    gm=rand(batch,dg);lm=rand(batch,sites,dl)
    gp=torch.diag_embed(rand(batch,dg).square()+1)
    lp=torch.diag_embed(rand(batch,sites,dl).square()+1)
    actual=block_gaussian_posterior(g,l,groups,w,y,gm,gp,lm,lp)
    expected,cov,_=dense_reference(g,l,groups,w,y,gm,gp,lm,lp)
    torch.testing.assert_close(actual['global_mean'],expected[:,:dg],rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(actual['local_mean'].reshape(batch,-1),expected[:,dg:],rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(actual['global_covariance'],cov[:,:dg,:dg],rtol=1e-10,atol=1e-10)
    for a in range(sites):
        coupling=actual['coupling'][:,a]
        local_cov=torch.cholesky_inverse(actual['local_cholesky'][:,a])+coupling@actual['global_covariance']@coupling.transpose(-1,-2)
        torch.testing.assert_close(local_cov,cov[:,dg+a*dl:dg+(a+1)*dl,dg+a*dl:dg+(a+1)*dl],rtol=1e-10,atol=1e-10)


def test_empty_ap_retains_local_prior_and_uncertainty():
    g=torch.ones((1,3,1),dtype=torch.float64);l=g.clone();groups=torch.zeros((1,3),dtype=torch.int64)
    gm=torch.zeros((1,1),dtype=torch.float64);gp=torch.ones((1,1,1),dtype=torch.float64)
    lm=torch.tensor([[[.2],[.7]]],dtype=torch.float64);lp=torch.ones((1,2,1,1),dtype=torch.float64)*2
    out=block_gaussian_posterior(g,l,groups,torch.ones((1,3),dtype=torch.float64),torch.ones((1,3),dtype=torch.float64),gm,gp,lm,lp)
    torch.testing.assert_close(out['local_mean'][:,1],lm[:,1])
    assert torch.count_nonzero(out['coupling'][:,1])==0
    torch.testing.assert_close(torch.cholesky_inverse(out['local_cholesky'][:,1]),torch.tensor([[[.5]]],dtype=torch.float64))


@pytest.mark.parametrize('low,high',[(-math.inf,math.inf),(-math.inf,-10.),(-math.inf,-1.),(-1.,2.),(.1,.1001),(4.,5.),(8.,math.inf)])
def test_interval_moments_match_scipy_analytic_reference(low,high):
    mean=torch.tensor([.3],dtype=torch.float64,requires_grad=True)
    std=torch.tensor([1.2],dtype=torch.float64,requires_grad=True)
    m,v=interval_moments(mean,std,torch.tensor([low],dtype=torch.float64),torch.tensor([high],dtype=torch.float64))
    expected_m,expected_v=truncnorm.stats((low-.3)/1.2,(high-.3)/1.2,loc=.3,scale=1.2,moments='mv')
    assert m.item()==pytest.approx(float(expected_m),rel=3e-7,abs=2e-7)
    assert v.item()==pytest.approx(float(expected_v),rel=2e-5,abs=2e-7)
    (m.sum()+v.sum()).backward()
    assert torch.isfinite(mean.grad).all() and torch.isfinite(std.grad).all()


@pytest.mark.parametrize('low,high',[(0.,0.),(2.,1.),(float('nan'),1.)])
def test_interval_moments_reject_invalid_events(low,high):
    with pytest.raises(ValueError):interval_moments(torch.zeros(1),torch.ones(1),torch.tensor([low]),torch.tensor([high]))


@pytest.mark.parametrize('cell',CELLS)
def test_all_registered_cells_render_finite_same_shape(cell):
    model,view,query,target,aps=fixture(cell=cell)
    mean,std=model(view,query,aps)
    assert mean.shape==std.shape==target['lower'].shape
    assert torch.isfinite(mean).all() and torch.isfinite(std).all() and (std>0).all()
    belief=model.encode_view(view,aps)
    assert belief['field_covariance_qualified'] is False and belief['packet_risk_qualified'] is False


def test_hierarchy_and_global24_have_exact_same_learned_parameter_set():
    models=[HierarchicalField(c) for c in ('hier_reg','hier_joint','global_reg','hier_erased')]
    identities=[[(name,tuple(p.shape)) for name,p in m.named_parameters()] for m in models]
    assert all(x==identities[0] for x in identities)
    assert all(sum(p.numel() for p in m.parameters())==sum(p.numel() for p in models[0].parameters()) for m in models)
    assert not any('embedding' in name for name,_ in models[0].named_parameters())


@pytest.mark.parametrize('cell',CELLS)
def test_ap_permutation_equivariance(cell):
    model,view,query,_,aps=fixture(cell=cell)
    expected=model(view,query,aps)
    permutation=torch.tensor((2,0,3,1));inverse=torch.argsort(permutation)
    changed=deepcopy(view);changed['ap']=inverse[view['ap']]
    changed_query=deepcopy(query);changed_query['ap']=inverse[query['ap']]
    actual=model(changed,changed_query,aps[:,permutation])
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-8)


@pytest.mark.parametrize('sites',[1,4,8,16])
def test_model_uses_shared_parameters_for_arbitrary_ap_counts(sites):
    model,view,query,_,aps=fixture(sites=sites)
    mean,std=model(view,query,aps)
    assert mean.shape==std.shape==(2,7)
    assert model.encode_view(view,aps)['local_mean'].shape==(2,sites,LOCAL_DIM)


@pytest.mark.parametrize('cell',CELLS)
def test_no_known_context_is_a_prior_not_an_invented_report(cell):
    model,view,query,_,aps=fixture(cell=cell)
    view['known'][:]=False;view['context'][...,2:4]=0
    view['lower'][:]=-torch.inf;view['upper'][:]=torch.inf
    belief=model.encode_view(view,aps)
    assert not belief['observed_ap'].any()
    mean,std=model.render_queries(belief,query)
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()
    if model.hierarchical:
        torch.testing.assert_close(belief['global_mean'],model.global_prior_mean[None].expand(2,-1))
        torch.testing.assert_close(belief['local_mean'],model.local_prior_mean[None,None].expand(2,4,-1))


def test_empty_scheduled_view_is_valid_prior():
    model,view,query,_,aps=fixture(tokens=0)
    mean,std=model(view,query,aps)
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()


def test_inverse_duplicate_weights_preserve_posterior():
    model,view,query,_,aps=fixture()
    expected=model(view,query,aps)
    double={key:torch.repeat_interleave(value,2,dim=1) for key,value in view.items()}
    double['weights']=torch.full_like(double['lower'],.5)
    actual=model(double,query,aps)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-8)


def test_rf_erased_keeps_status_but_never_amplitude_or_topk_bound():
    model,view,query,_,aps=fixture(cell='hier_erased')
    view['floor'].fill_(-15.)  # a genuinely informative public detection event
    expected=model(view,query,aps)
    changed=deepcopy(view)
    changed['context'][...,:2]+=43.
    heard=changed['context'][...,2]>.5
    changed['lower'][heard]+=25.;changed['upper']+=25.
    actual=model(changed,query,aps)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    # Detection status itself is retained as a lower-threshold event.
    changed=deepcopy(view);changed['known'][:]=False;changed['context'][...,2:4]=0
    changed['lower'][:]=-torch.inf;changed['upper'][:]=torch.inf
    erased_without_status=model(changed,query,aps)
    assert not torch.equal(erased_without_status[0],expected[0])


def test_new_context_measurement_has_ap_specific_influence():
    model,view,query,_,aps=fixture(batch=1,sites=2)
    expected=model(view,query,aps)[0]
    changed=deepcopy(view)
    selected=changed['ap']==0
    changed['lower'][selected]+=8.;changed['upper'][selected]+=8.
    heard=changed['context'][...,2]>.5
    changed['context'][...,0]=torch.where(selected&heard,changed['context'][...,0]+8/30,changed['context'][...,0])
    changed['context'][...,1]=torch.where(selected&~heard,changed['context'][...,1]+8/30,changed['context'][...,1])
    actual=model(changed,query,aps)[0]
    own=(actual-expected)[query['ap']==0].mean()
    other=(actual-expected)[query['ap']==1].mean()
    assert own>other and own>0


@pytest.mark.parametrize('cell',('hier_reg','global_reg','global8_reg'))
def test_query_covariance_matches_render_marginals_and_nonnegative_gain_variance(cell):
    model,view,query,_,aps=fixture(cell=cell)
    belief=model.encode_view(view,aps)
    _,std=model.render_queries(belief,query)
    cov=model.predictive_covariance(belief,query['q'],query['ap'],query['rx_xyz'],query['basis'])
    discrepancy=model.discrepancy_db(query['q']).square()
    torch.testing.assert_close(cov.diagonal(dim1=-2,dim2=-1)+discrepancy,std.square(),rtol=1e-8,atol=1e-8)
    torch.testing.assert_close(cov,cov.transpose(-1,-2),rtol=1e-10,atol=1e-10)
    assert torch.linalg.eigvalsh(cov).min()>-1e-8
    gain_var=cov[:,0,0]+cov[:,1,1]-2*cov[:,0,1]
    assert (gain_var>=-1e-8).all()


@pytest.mark.parametrize('cell',CELLS)
def test_matched_training_loss_has_finite_gradients(cell):
    model,a,query,target,aps=fixture(cell=cell)
    b=deepcopy(a);b['lower']=b['lower']+.2;b['upper']=b['upper']+.2
    loss,details=objective(model,a,b,query,target,aps)
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert details['alignment_weight'].item()==(ALIGNMENT_WEIGHT if cell=='hier_joint' else 0.)
    if cell!='hier_joint':assert details['alignment']==0


def test_joint_agreement_has_no_private_target_input_and_skips_disjoint_aps():
    model,a,query,target,aps=fixture(cell='hier_joint')
    b=deepcopy(a)
    a['known'] &= a['ap']==0;b['known'] &= b['ap']==1
    for view in (a,b):view['context'][...,2:4]*=view['known'][...,None]
    _,details=objective(model,a,b,query,target,aps)
    assert details['supported_agreement_queries']==0 and details['alignment']==0
    a=fixture(cell='hier_joint')[1];b=deepcopy(a)
    _,first=objective(model,a,b,query,target,aps)
    altered={key:value.clone() for key,value in target.items()}
    altered['lower']+=5.;altered['upper']+=5.
    _,second=objective(model,a,b,query,altered,aps)
    assert first['alignment']==second['alignment']==0


def test_target_likelihood_intersects_report_bin_with_detection_floor():
    mean=torch.tensor([[-90.]],dtype=torch.float64);std=torch.ones_like(mean)
    legacy={'heard':torch.ones_like(mean,dtype=torch.bool),'censored':torch.zeros_like(mean,dtype=torch.bool),
        'lower_dbm':mean-.5,'upper_dbm':mean+.5,'detection_threshold_dbm':mean+.1,
        'censor_upper_dbm':mean+.1}
    actual=observation_nll(mean,std,legacy)
    event={'heard':legacy['heard'],'censored':legacy['censored'],'lower':mean+.1,'upper':mean+.5}
    torch.testing.assert_close(actual,observation_nll(mean,std,event))


def test_target_censor_is_not_a_point_at_the_bound():
    mean=torch.zeros((1,2),dtype=torch.float64);std=torch.ones_like(mean)
    target={'heard':torch.tensor([[True,False]]),'censored':torch.tensor([[False,True]]),
        'lower':torch.tensor([[-.5,-torch.inf]],dtype=torch.float64),'upper':torch.tensor([[.5,-1.]],dtype=torch.float64)}
    assert torch.isfinite(observation_nll(mean,std,target))
    target['lower'][0,1]=-2.
    with pytest.raises(ValueError,match='distinct'):observation_nll(mean,std,target)


@pytest.mark.parametrize('change',('bad_ap','missing_interval','heard_and_censored','fake_censor','wrong_floor','wrong_position','negative_weight','unknown_weight'))
def test_invalid_input_fails_closed(change):
    model,view,query,_,aps=fixture()
    if change=='bad_ap':view['ap'][0,0]=4
    elif change=='missing_interval':view.pop('lower')
    elif change=='heard_and_censored':view['context'][0,0,3]=1
    elif change=='fake_censor':view['lower'][0,-1]=-50
    elif change=='wrong_floor':view['floor'][0,0]=view['upper'][0,0]+1
    elif change=='wrong_position':view['rx_xyz'][0,0,0]+=10
    else:
        view['weights']=torch.ones_like(view['lower'])
        if change=='negative_weight':view['weights'][0,0]=-1
        else:view['known'][0,0]=False;view['context'][0,0,2:4]=0
    with pytest.raises((ValueError,KeyError)):
        model(view,query,aps)


def test_unknown_placeholder_amplitudes_do_not_change_prior_or_forecast():
    model,view,query,_,aps=fixture()
    view['known'][0,0]=False;view['context'][0,0,2:4]=0
    expected=model(view,query,aps)
    changed=deepcopy(view);changed['context'][0,0,:2]=torch.nan
    changed['lower'][0,0]=torch.nan;changed['upper'][0,0]=torch.nan
    # Scheduled public features must remain finite; amplitudes of an unknown
    # token are deliberately zeroed by the data contract, not silently read.
    with pytest.raises(ValueError):model(changed,query,aps)
    changed=deepcopy(view);changed['context'][0,0,:2]=1000
    changed['lower'][0,0]=torch.nan;changed['upper'][0,0]=torch.nan
    actual=model(changed,query,aps)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b)


def test_roundtrip_configuration_and_no_random_feature_rng_side_effect():
    torch.manual_seed(133)
    first=HierarchicalField('hier_reg')
    second=HierarchicalField(**first.config)
    second.load_state_dict(first.state_dict())
    assert second.architecture==first.architecture
    torch.testing.assert_close(first.rff_omega,second.rff_omega)


def test_exact_native_point_context_matches_gaussian_joint_posterior():
    model,view,query,_,aps=fixture(batch=1,sites=3)
    view['context'][...,2]=1.;view['context'][...,3]=0.
    value=-11.-.2*view['rx_xyz'][...,0]
    view['lower']=value.clone();view['upper']=value.clone()
    view['context'][...,0]=value/30.;view['context'][...,1]=0.
    view['context'][...,17]=0.  # actual continuous native reports have no bin
    belief=model.encode_view(view,aps)
    g=query_physics_basis(view['context'][...,4:])
    local=model.local_features(view['context'][...,4:],view['rx_xyz'],aps,view['ap'])
    stdg=.2+torch.nn.functional.softplus(model.raw_global_std)
    stdl=.2+torch.nn.functional.softplus(model.raw_local_std)
    gp=torch.diag_embed(stdg.reciprocal().square()+1e-8)[None]
    lp=torch.diag_embed(stdl.reciprocal().square()+1e-8)[None,None].expand(1,3,-1,-1)
    weights=100./model.discrepancy_db(view['context'][...,4:]).square()
    expected,cov,_=dense_reference(g,local,view['ap'],weights,value/10,
        model.global_prior_mean[None],gp,model.local_prior_mean[None,None].expand(1,3,-1),lp)
    torch.testing.assert_close(belief['global_mean'],expected[:,:GLOBAL_DIM],rtol=1e-9,atol=1e-9)
    torch.testing.assert_close(belief['local_mean'].reshape(1,-1),expected[:,GLOBAL_DIM:],rtol=1e-9,atol=1e-9)
    torch.testing.assert_close(belief['global_covariance'],cov[:,:GLOBAL_DIM,:GLOBAL_DIM],rtol=1e-9,atol=1e-9)


def test_explicit_point_moments_have_zero_conditional_variance_without_epsilon_bin():
    mean=torch.tensor([2.],dtype=torch.float64);std=torch.tensor([3.],dtype=torch.float64)
    point=torch.tensor([1.25],dtype=torch.float64)
    expected,variance=interval_moments(mean,std,point,point,point_mask=torch.tensor([True]))
    torch.testing.assert_close(expected,point)
    torch.testing.assert_close(variance,torch.zeros_like(point))


@pytest.mark.parametrize('cell',CELLS)
def test_fixed_newton_updates_never_increase_context_objective_beyond_roundoff(cell):
    model,view,_,_,aps=fixture(cell=cell)
    belief=model.encode_view(view,aps)
    assert model.config['inference_steps']==len(belief['inference_log'])==3
    for record in belief['inference_log']:
        before,after=record['objective_before'],record['objective_after']
        tolerance=OBJECTIVE_TOLERANCE*torch.maximum(torch.ones_like(before),before.abs())
        assert (after<=before+tolerance).all()
        assert set(record['step_size'].tolist())<=set(BACKTRACKING_STEPS+(0.,))


@pytest.mark.parametrize('cell',('hier_reg','global_reg','global8_reg'))
def test_uninformative_censors_do_not_slow_down_real_point_updates(cell):
    model,view,query,_,aps=fixture(cell=cell)
    view['context'][...,2]=1.;view['context'][...,3]=0.
    value=-11.-.2*view['rx_xyz'][...,0]
    view['lower']=value.clone();view['upper']=value.clone()
    view['context'][...,0]=value/30.;view['context'][...,1]=0.
    expected=model(view,query,aps)
    added={key:value[:,:4].repeat_interleave(20,dim=1).clone() for key,value in view.items()}
    added['context'][...,0]=0.;added['context'][...,1]=1.e6/30.
    added['context'][...,2]=0.;added['context'][...,3]=1.
    added['lower'].fill_(-torch.inf);added['upper'].fill_(1.e6)
    enlarged={key:torch.cat((value,added[key]),dim=1) for key,value in view.items()}
    actual=model(enlarged,query,aps)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1.e-9,atol=1.e-9)


def test_line_search_uses_first_allowed_step_and_has_detached_choice_gradients():
    current=torch.tensor([[1.],[0.]],dtype=torch.float64,requires_grad=True)
    proposed=torch.tensor([[-10.],[1.]],dtype=torch.float64,requires_grad=True)
    (actual,),record=_bounded_step((current,),(proposed,),lambda states:states[0].square().sum(-1))
    torch.testing.assert_close(record['step_size'],torch.tensor([.125,0.],dtype=torch.float64))
    torch.testing.assert_close(actual,torch.tensor([[-.375],[0.]],dtype=torch.float64))
    assert record['null_step'].tolist()==[False,True]
    actual.sum().backward()
    torch.testing.assert_close(current.grad,torch.tensor([[.875],[1.]],dtype=torch.float64))
    torch.testing.assert_close(proposed.grad,torch.tensor([[.125],[0.]],dtype=torch.float64))


@pytest.mark.parametrize('change',('bad_group','negative_weight','negative_prior','asymmetric_prior'))
def test_gaussian_mathematical_helper_rejects_invalid_routes_and_priors(change):
    dtype=torch.float64
    g=torch.ones((1,2,1),dtype=dtype);l=g.clone();groups=torch.zeros((1,2),dtype=torch.int64)
    weight=torch.ones((1,2),dtype=dtype);y=weight.clone();gm=torch.zeros((1,1),dtype=dtype)
    gp=torch.ones((1,1,1),dtype=dtype);lm=torch.zeros((1,2,1),dtype=dtype);lp=torch.ones((1,2,1,1),dtype=dtype)
    if change=='bad_group':groups[0,0]=2
    elif change=='negative_weight':weight[0,0]=-1
    elif change=='negative_prior':gp[0,0,0]=-1
    else:
        g=torch.ones((1,2,2),dtype=dtype);gm=torch.zeros((1,2),dtype=dtype)
        gp=torch.tensor([[[1.,.2],[0.,1.]]],dtype=dtype)
    with pytest.raises((ValueError,FloatingPointError)):
        block_gaussian_posterior(g,l,groups,weight,y,gm,gp,lm,lp)


def test_deep_censor_updates_and_line_search_remain_finite():
    model,view,query,_,aps=fixture()
    view['context'][...,0]=0.;view['context'][...,1]=-5/3
    view['context'][...,2]=0.;view['context'][...,3]=1.
    view['lower'].fill_(-torch.inf);view['upper'].fill_(-50.)
    with torch.no_grad():model.global_prior_mean[0]=10.
    belief=model.encode_view(view,aps)
    mean,std=model.render_queries(belief,query)
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()
    assert all(torch.isfinite(step['objective_after']).all() for step in belief['inference_log'])


def test_full_covariance_rejects_large_native_query_grids_before_quadratic_allocation():
    model,view,query,_,aps=fixture(queries=513)
    belief=model.encode_view(view,aps)
    with pytest.raises(ValueError,match='at most512'):
        model.predictive_covariance(belief,query['q'],query['ap'],query['rx_xyz'],query['basis'])

