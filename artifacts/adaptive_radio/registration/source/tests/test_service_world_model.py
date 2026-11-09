"""Model-mechanics checks; fixtures are not warehouse performance evidence."""
import copy
import pytest
import torch
from wirelessjev.service_world.model import ServiceWorldModel,objective,update_target


def inputs(b=4,t=6,s=5,h=3):
    g=torch.Generator().manual_seed(761)
    return {'links':torch.randn(b,t,s,3,12,generator=g),
        'globals':torch.randn(b,t,8,generator=g),
        'observed':torch.rand(b,t,s,3,12,generator=g)>.2,
        'global_observed':torch.rand(b,t,8,generator=g)>.1,
        'elapsed':torch.full((b,t),.1),'actions':torch.randn(b,h,12,generator=g),
        'ap_index':torch.randint(s,(b,h),generator=g),'physics_mean':torch.randn(b,h,2,generator=g),
        'hazard_prior':torch.rand(b,h,generator=g),
        'adjacency':torch.rand(b,3,s,s,generator=g),
        'action_elapsed':torch.tensor([.1,1.,5.]).expand(b,h).clone()}


def batch():
    x=inputs();b,h=x['actions'].shape[:2];g=torch.Generator().manual_seed(762)
    x.update(target_mean=x['physics_mean']+torch.randn(b,h,2,generator=g),
        target_hazard=torch.randint(2,(b,h),generator=g).float(),
        future_links=torch.randn(b,h,4,5,3,12,generator=g),
        future_globals=torch.randn(b,h,4,8,generator=g),
        future_observed=torch.ones(b,h,4,5,3,12,dtype=torch.bool),
        future_elapsed=torch.full((b,h,4),.1),future_valid=torch.ones(b,h,dtype=torch.bool))
    return x


def inference_kwargs(x):
    return {k:v for k,v in x.items() if not k.startswith(('future_','target_'))}


def test_zero_residual_is_exact_physics_and_missing_hazard_is_not_invented():
    torch.manual_seed(91);m=ServiceWorldModel(hidden=16);x=inputs()
    x['hazard_prior'][0,0]=0.;x['hazard_prior'][0,1]=1.
    p=m(**x)
    assert torch.equal(p['mean'],x['physics_mean']) and torch.equal(p['hazard'],x['hazard_prior'])
    assert (p['residual']==0).all() and (p['std']>0).all() and (p['latent_std']>0).all()
    assert ((p['survival']>=0)&(p['survival']<=1)).all()
    assert (p['survival'][:,1:]<=p['survival'][:,:-1]).all()
    del x['hazard_prior'];p=m(**x)
    assert p['hazard'] is None and p['survival'] is None and not p['hazard_prior_supplied']


def test_matched_inference_graph_has_no_future_input_or_target_gradients():
    torch.manual_seed(92);a=ServiceWorldModel('jepa',hidden=16)
    torch.manual_seed(92);b=ServiceWorldModel('supervised',hidden=16)
    assert a.inference_parameters==b.inference_parameters
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    x=inputs();pa=a(**x);pb=b(**x)
    for key in ('mean','std','latent','latent_std','hazard','belief','link_latent'):
        torch.testing.assert_close(pa[key],pb[key],rtol=0,atol=0)
    assert not any(p.requires_grad for p in a.target_encoder.parameters())
    with pytest.raises(TypeError):a(**x,future_links=batch()['future_links'])


def test_physical_AP_relabeling_is_equivariant_with_nonzero_trained_heads():
    torch.manual_seed(93);m=ServiceWorldModel(hidden=16)
    with torch.no_grad():
        m.service_head.weight.normal_(std=.1);m.hazard_head.weight.normal_(std=.1)
    x=inputs();original=m(**x);perm=torch.tensor([2,4,0,1,3]);inverse=perm.argsort()
    y=copy.deepcopy(x);y['links']=x['links'][:,:,perm];y['observed']=x['observed'][:,:,perm]
    y['adjacency']=x['adjacency'][:,:,perm][:,:,:,perm];y['ap_index']=inverse[x['ap_index']]
    changed=m(**y)
    for key in ('mean','std','latent','latent_std','hazard','belief'):
        torch.testing.assert_close(changed[key],original[key],rtol=2e-5,atol=2e-5)
    torch.testing.assert_close(changed['link_latent'],original['link_latent'][:,perm],rtol=2e-5,atol=2e-5)


def test_unknown_measurements_do_not_become_invented_observations():
    torch.manual_seed(94);m=ServiceWorldModel(hidden=16);x=inputs();y=copy.deepcopy(x)
    y['links'][~y['observed']]=float('nan');y['globals'][~y['global_observed']]=float('nan')
    a=m(**x);b=m(**y)
    for key in ('belief','link_latent','latent'):
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    y['observed'][:]=True
    with pytest.raises(ValueError,match='marked observed'):m(**y)


def test_new_chunk_streaming_matches_full_history_and_reuses_belief():
    torch.manual_seed(95);m=ServiceWorldModel(hidden=16);x=inputs()
    whole=m.encode(x['links'],x['globals'],x['observed'],x['elapsed'],
                   global_observed=x['global_observed'],adjacency=x['adjacency'])
    first=m.encode(x['links'][:,:3],x['globals'][:,:3],x['observed'][:,:3],x['elapsed'][:,:3],
                   global_observed=x['global_observed'][:,:3],adjacency=x['adjacency'])
    second=m.encode(x['links'][:,3:],x['globals'][:,3:],x['observed'][:,3:],x['elapsed'][:,3:],
                   global_observed=x['global_observed'][:,3:],adjacency=x['adjacency'],state=first['state'])
    torch.testing.assert_close(second['belief'],whole['belief'],rtol=2e-5,atol=2e-5)
    torch.testing.assert_close(second['links'],whole['links'],rtol=2e-5,atol=2e-5)


def test_rollout_is_action_conditioned_and_causal_across_horizons():
    torch.manual_seed(96);m=ServiceWorldModel(hidden=16);x=inputs();base=m(**x)
    changed=copy.deepcopy(x);changed['actions'][:,2]+=5
    later=m(**changed)
    torch.testing.assert_close(base['latent'][:,:2],later['latent'][:,:2],rtol=0,atol=0)
    assert not torch.allclose(base['latent'][:,2],later['latent'][:,2])
    changed=copy.deepcopy(x);changed['actions'][:,0]+=5;earlier=m(**changed)
    assert not torch.allclose(base['latent'][:,2],earlier['latent'][:,2])
    encoded=m.encode(x['links'],x['globals'],x['observed'],x['elapsed'],
                     global_observed=x['global_observed'],adjacency=x['adjacency'])
    reused=m.rollout(encoded,x['actions'],x['ap_index'],x['physics_mean'],x['hazard_prior'],action_elapsed=x['action_elapsed'])
    torch.testing.assert_close(base['latent'],reused['latent'],rtol=0,atol=0)


def test_unlabelled_pretraining_has_real_dynamics_gradients_and_no_service_labels():
    torch.manual_seed(97);m=ServiceWorldModel(hidden=16);x=batch()
    del x['target_mean'];del x['target_hazard']
    loss,metrics=objective(m,x,phase='pretrain')
    assert torch.isfinite(loss) and metrics['supervised']==0
    loss.backward()
    assert m.transition.weight_ih.grad.abs().sum()>0
    assert m.encoder.link_memory.weight_ih_l0.grad.abs().sum()>0
    assert m.service_head.weight.grad is None
    assert all(p.grad is None for p in m.target_encoder.parameters())
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    with pytest.raises(ValueError,match='no JEPA pretraining'):
        objective(ServiceWorldModel('supervised',hidden=16),x,phase='pretrain')


def test_native_finetuning_uses_no_future_observations_in_either_family():
    x=batch()
    for family in ('jepa','supervised'):
        torch.manual_seed(98);m=ServiceWorldModel(family,hidden=16)
        clean={k:v for k,v in x.items() if not k.startswith('future_')}
        expected,_=objective(m,clean,phase='finetune')
        poisoned=copy.deepcopy(x);poisoned['future_links'][:]=float('nan')
        actual,metrics=objective(m,poisoned,phase='finetune')
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert metrics['jepa']==0
        actual.backward();assert m.service_head.weight.grad.abs().sum()>0
        assert m.hazard_head.weight.grad.abs().sum()>0
        assert all(p.grad is None for p in m.target_encoder.parameters())


def test_missing_future_and_service_targets_can_be_poisoned_without_invention():
    torch.manual_seed(99);m=ServiceWorldModel(hidden=16);x=batch()
    x['future_valid'][0,1:]=False;x['future_observed'][1,:,2:]=False
    x['target_valid']=torch.ones_like(x['target_mean'],dtype=torch.bool);x['target_valid'][0]=False
    original,_=objective(m,x)
    poisoned=copy.deepcopy(x)
    poisoned['future_links'][0,1:]=float('nan');poisoned['future_globals'][0,1:]=float('nan')
    poisoned['future_elapsed'][0,1:]=float('nan')
    poisoned['future_links'][1,:,2:]=float('nan')
    poisoned['target_mean'][0]=float('nan')
    changed,_=objective(m,poisoned)
    torch.testing.assert_close(changed,original,rtol=0,atol=0)
    poisoned['future_valid'][:]=False;poisoned['future_links'][:]=float('nan')
    poisoned['future_globals'][:]=float('nan');poisoned['future_elapsed'][:]=float('nan')
    loss,metrics=objective(m,poisoned)
    assert torch.isfinite(loss) and metrics['jepa']==0


def test_explicit_stochastic_rollout_and_ema_are_reproducible():
    torch.manual_seed(100);m=ServiceWorldModel(hidden=16);x=inputs()
    p=m(**x);zero=m(**x,latent_noise=torch.zeros(4,3,16))
    torch.testing.assert_close(p['latent'],zero['latent'],rtol=0,atol=0)
    noise=torch.ones(4,3,16)*.5
    a=m(**x,latent_noise=noise);b=m(**x,latent_noise=noise)
    torch.testing.assert_close(a['latent'],b['latent'],rtol=0,atol=0)
    assert not torch.allclose(a['latent'][:,1:],p['latent'][:,1:])
    before={k:v.clone() for k,v in m.target_encoder.state_dict().items()}
    with torch.no_grad():
        for parameter in m.encoder.parameters():parameter.add_(.25)
    update_target(m,.8)
    for key,value in m.target_encoder.state_dict().items():
        torch.testing.assert_close(value,.8*before[key]+.2*m.encoder.state_dict()[key])
    clone=ServiceWorldModel(**m.config);clone.load_state_dict(m.state_dict())
    torch.testing.assert_close(m(**x)['latent'],clone(**x)['latent'],rtol=0,atol=0)


def test_bad_physics_probabilities_masks_and_times_fail_explicitly():
    m=ServiceWorldModel(hidden=16);x=inputs()
    y=copy.deepcopy(x);y['elapsed'][0,0]=-1
    with pytest.raises(ValueError,match='Elapsed'):m(**y)
    y=copy.deepcopy(x);y['hazard_prior'][0,0]=1.2
    with pytest.raises(ValueError,match='Hazard prior'):m(**y)
    y=copy.deepcopy(x);y['ap_index'][0,0]=5
    with pytest.raises(ValueError,match='AP indices'):m(**y)
    y=copy.deepcopy(x);y['physics_mean'][0,0,0]=float('nan')
    with pytest.raises(ValueError,match='numerical service prior'):m(**y)
    y=copy.deepcopy(x);y['adjacency'][0,0,0,0]=-1
    with pytest.raises(ValueError,match='Adjacency'):m(**y)
    y=batch();del y['hazard_prior']
    with pytest.raises(ValueError,match='explicit numerical hazard prior'):objective(m,y)
