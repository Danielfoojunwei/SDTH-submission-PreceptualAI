"""Structural invariants, not learned wireless quality or generalization scores."""
import pytest
import torch
from wirelessjev.visual_radio.representation import GroundedLinkEncoder,GroundedServiceHead
from wirelessjev.visual_radio.events import VisualEvent,latest_visual_event,select_dino_observation


def inputs():
    torch.manual_seed(719);b,t,s,p,d=2,4,3,5,8
    return {'radio':torch.randn(b,t,s,3,12),'radio_observed':torch.ones(b,t,s,3,12,dtype=torch.bool),
        'elapsed':torch.full((b,t),.1),'patches':torch.randn(b,t,p,d),
        'visual_age_seconds':torch.full((b,t,p),.2),
        'patch_observed':torch.ones(b,t,p,dtype=torch.bool),'patch_xy':torch.rand(b,t,p,2),
        'grounding_weights':torch.rand(b,t,s,3,p),'grounding_provenance':'manual_image_regions'}


def test_physical_bypass_and_masks_preserve_measured_state_exactly():
    x=inputs();x['radio_observed'][...,0]=False;x['radio'][...,0]=float('nan')
    net=GroundedLinkEncoder(8).eval();p=net(**x)
    assert torch.equal(p['physical_observed'],x['radio_observed'])
    assert torch.equal(p['physical_history'][x['radio_observed']],x['radio'][x['radio_observed']])
    assert (p['physical_history'][~x['radio_observed']]==0).all()
    assert p['link_latent'].shape==(2,3,3,48) and torch.isfinite(p['link_latent']).all()


def test_ap_permutation_and_candidate_remapping_preserve_service_values():
    x=inputs();net=GroundedLinkEncoder(8).eval();head=GroundedServiceHead()
    torch.nn.init.normal_(head.network[-1].weight,std=.2)
    encoded=net(**x);perm=torch.tensor([2,0,1]);inverse=perm.argsort()
    changed=x|{k:x[k][:,:,perm] for k in ('radio','radio_observed','grounding_weights')}
    reordered=net(**changed)
    torch.testing.assert_close(reordered['link_latent'],encoded['link_latent'][:,perm])
    actions=torch.randn(2,2,12);ap=torch.tensor([[0,2],[1,0]]);band=torch.tensor([[0,2],[1,2]])
    reference=torch.randn(2,2,2)
    a=head(encoded,actions,ap,band,reference)
    b=head(reordered,actions,inverse[ap],band,reference)
    torch.testing.assert_close(a['mean'],b['mean'])


def test_reordering_dense_patches_with_coordinates_and_grounding_is_equivalent():
    x=inputs();net=GroundedLinkEncoder(8).eval();a=net(**x);perm=torch.tensor([4,2,0,3,1])
    changed=x|{k:x[k][:,:,perm] for k in ('patches','patch_observed','patch_xy','visual_age_seconds')}
    changed['grounding_weights']=x['grounding_weights'][...,perm]
    b=net(**changed)
    torch.testing.assert_close(a['link_latent'],b['link_latent'])
    torch.testing.assert_close(a['attention'][...,perm],b['attention'])


def test_missing_visual_stream_does_not_invent_visual_support_or_contaminate_rf():
    x=inputs();x['patch_observed'][:]=False
    net=GroundedLinkEncoder(8).eval();a=net(**x)
    x['patches'][:]=float('nan');x['patch_xy'][:]=float('nan');b=net(**x)
    assert not a['visual_present'].any() and (a['attention']==0).all()
    torch.testing.assert_close(a['link_latent'],b['link_latent'],rtol=0,atol=0)


def test_unrelated_patch_cannot_change_a_link_outside_its_grounded_region():
    x=inputs();x['grounding_weights'][:,:,0,:,0]=0
    net=GroundedLinkEncoder(8).eval();a=net(**x)
    x['patches'][:,:,0]+=500.;b=net(**x)
    torch.testing.assert_close(a['link_latent'][:,0],b['link_latent'][:,0],rtol=0,atol=0)
    assert not torch.allclose(a['link_latent'][:,1:],b['link_latent'][:,1:])


def test_band_identity_is_not_pooled_and_anchor_remains_exact_at_construction():
    x=inputs();encoded=GroundedLinkEncoder(8)(**x);head=GroundedServiceHead()
    actions=torch.zeros(2,2,12);ap=torch.zeros(2,2,dtype=torch.long);band=torch.tensor([[0,1],[1,2]])
    reference=torch.randn(2,2,2);p=head(encoded,actions,ap,band,reference)
    assert torch.equal(p['mean'],reference) and p['calibrated_risk'] is None
    torch.nn.init.normal_(head.network[-1].weight,std=.2)
    p=head(encoded,actions,ap,band,torch.zeros_like(reference))
    assert not torch.allclose(p['mean'][:,0],p['mean'][:,1])


def test_fusion_gradients_reach_both_modalities_without_training_dino():
    x=inputs();x['radio'].requires_grad_();x['patches'].requires_grad_()
    encoded=GroundedLinkEncoder(8)(**x)
    loss=encoded['link_latent'][...,0].sum();loss.backward()
    assert torch.isfinite(x['radio'].grad).all() and x['radio'].grad.abs().sum()>0
    assert torch.isfinite(x['patches'].grad).all() and x['patches'].grad.abs().sum()>0


@pytest.mark.parametrize('field', ['grounding_weights','patch_xy'])
def test_observed_invalid_geometry_is_rejected(field):
    x=inputs();x[field].flatten()[0]=float('nan')
    with pytest.raises(ValueError):GroundedLinkEncoder(8)(**x)


def test_grounding_requires_provenance_not_implicit_dino_depth():
    x=inputs();x['grounding_provenance']='inferred_RF_material_from_RGB'
    with pytest.raises(ValueError,match='provenance'):GroundedLinkEncoder(8)(**x)


def event(key,captured,available,cal='c'):
    return VisualEvent(key,captured,available,'0'*64,cal)


def test_causal_alignment_handles_latency_out_of_order_frames_and_calibration():
    a=event('a',1.,2.);b=event('b',1.8,3.);c=event('c',1.5,2.1)
    assert latest_visual_event([b,c,a],2.2,max_age_seconds=2.)==c
    assert latest_visual_event([b,c,a],3.,max_age_seconds=2.)==b
    assert latest_visual_event([b,c,a],3.,max_age_seconds=.1) is None
    assert latest_visual_event([a],2.2,max_age_seconds=2.,calibration_id='other') is None
    with pytest.raises(ValueError):event('bad',2.,1.)


def test_dino_bridge_never_promotes_unknown_photo_capture_to_fresh_radio_evidence():
    from types import SimpleNamespace
    batch=SimpleNamespace(provenance=({'frame_id':'x','canonical_rgb_sha256':'0'*64},),
        capture_time_known=torch.tensor([False]),capture_time_seconds=torch.tensor([0.]),
        observation_time_seconds=torch.tensor([2.]),patch_tokens=torch.randn(1,2,8),
        patch_valid=torch.ones(1,2,dtype=torch.bool),
        patch_boxes_original_pixels=torch.tensor([[[0.,0.,100.,100.],[100.,0.,200.,100.]]]),
        image_geometry=({'original_size_wh':[200,100]},))
    assert select_dino_observation(batch,2.1,feature_available_seconds=[2.05],max_age_seconds=1.) is None
    batch.capture_time_known[:]=True;batch.capture_time_seconds[:]=1.8
    selected=select_dino_observation(batch,2.1,feature_available_seconds=[2.05],max_age_seconds=1.)
    torch.testing.assert_close(selected.patch_xy,torch.tensor([[.25,.5],[.75,.5]]))
    torch.testing.assert_close(selected.age_seconds,torch.full((2,),.3))
    assert select_dino_observation(batch,2.01,feature_available_seconds=[2.05],max_age_seconds=1.) is None
    with pytest.raises(ValueError):select_dino_observation(batch,2.1,feature_available_seconds=[1.99],max_age_seconds=1.)
