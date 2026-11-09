"""Analytic software fixtures only; no benchmark fits or held recordings."""
import numpy as np
import pytest
import torch

from wirelessjev.channel_scene import benchmark as b, model as m
from wirelessjev.channel_scene.baselines import linear_interpolate


def fixture():
    f=(np.arange(64)-32)*50.e6/64
    rng=np.random.default_rng(22)
    h=rng.normal(size=(12,64))+1j*rng.normal(size=(12,64))
    return f,torch.tensor(h,dtype=torch.complex64),np.arange(0,64,4),np.arange(2,64,4),np.arange(1,64,2)


def test_masks_are_disjoint_actual_samples_not_averaged_or_relabelled():
    _,_,a,bv,target=fixture()
    assert len(a)==len(bv)==16 and len(target)==32
    np.testing.assert_array_equal(np.sort(np.r_[a,bv,target]),np.arange(64))
    raw=np.arange(0,1024,16)
    assert len(raw)==64 and np.unique(np.diff(raw)).item()==16


def test_pilot_only_normalization_inverts_independent_context_gauges():
    _,h,a,bv,_=fixture()
    na,fa=m.pilot_normalize(h[:,a]);nb,fb=m.pilot_normalize(h[:,bv])
    torch.testing.assert_close(na*fa,h[:,a]);torch.testing.assert_close(nb*fb,h[:,bv])
    # Their normalizing coordinates generally differ; aligning these raw
    # canonical latents as if identical would be scientifically invalid.
    assert not torch.allclose(fa,fb)
    torch.testing.assert_close(m.common_gauge_alignment(h/fa*fa,h/fb*fb,2.),torch.tensor(0.),atol=1.e-13,rtol=0.)


def test_model_initial_anchor_matches_independent_complex_interpolator():
    f,h,a,_,target=fixture();net=m.PilotCompletionModel(f)
    y,_,_=net(h[:,a],a,1.27e9)
    expected=linear_interpolate(h[:,a].numpy(),a,f)
    np.testing.assert_allclose(y.detach(),expected,rtol=2.e-6,atol=2.e-6)
    changed=h.clone();changed[:,target]+=100.+20j
    other=net(changed[:,a],a,1.27e9)[0]
    torch.testing.assert_close(y,other)


def test_alignment_is_same_packet_receiver_gauge_and_both_branches_differentiate():
    _,h,_,_,_=fixture();a=h.clone().requires_grad_();other=(h*.8+.2j).requires_grad_()
    loss=m.common_gauge_alignment(a,other,3.);loss.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(other.grad).all()
    assert a.grad.abs().sum()>0 and other.grad.abs().sum()>0
    torch.testing.assert_close(a.grad,-other.grad)
    rotated=m.common_gauge_alignment(a*complex(0,1),other*complex(0,1),3.)
    torch.testing.assert_close(loss,rotated)
    with pytest.raises(ValueError,match='TRAIN-pilot scale'):m.common_gauge_alignment(a,other,0.)


def test_exact_matched_control_differs_only_by_alignment_and_no_ema():
    f,h,a,bv,target=fixture();joint,control=m.matched_pair(f,27)
    for x,y in zip(joint.parameters(),control.parameters()):torch.testing.assert_close(x,y)
    assert sum(v.numel() for v in joint.parameters())==sum(v.numel() for v in control.parameters())
    lj,pj=m.objective(joint,h,1.27e9,a,bv,target,joint=True,training_pilot_scale=2.)
    lr,pr=m.objective(control,h,1.27e9,a,bv,target,joint=False,training_pilot_scale=2.)
    for key in pj:torch.testing.assert_close(pj[key],pr[key])
    torch.testing.assert_close(lj-lr,.1*pj['latent'],rtol=3.e-4,atol=1.e-7)
    lj.backward();lr.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in joint.parameters())
    assert not hasattr(m,'teacher_for') and not hasattr(m,'update_teacher')


def test_actual_grid_context_changes_representation_and_keeps_physical_fourier_units():
    f,h,a,_,_=fixture();net=m.PilotCompletionModel(f)
    narrow=net.encode(h[:,a],a,1.27e9,f)[0];wide=net.encode(h[:,a],a,1.27e9,f*4)[0]
    assert not torch.allclose(narrow,wide)
    np.testing.assert_allclose(net.canonical_delay_s.numpy()[1],1/(64*(f[1]-f[0])))


def test_packet_metric_aggregates_correlated_antennas_before_cases():
    truth=np.ones((2,3,4),complex);pred=truth.copy();pred[0]+=1;pred[1]+=2
    result=b.metrics(pred,truth)
    assert result['packets']==2 and result['mean_packet_nmse']==2.5
    assert result['case_quantiles_are_not_iid_confidence_intervals']
    with pytest.raises(ValueError,match='packet'):b.metrics(pred.reshape(6,4),truth.reshape(6,4))


def test_pose_partition_does_not_depend_on_radio_values():
    positions=np.c_[np.zeros(100),np.arange(100)*.1,np.ones(100)]
    split=b._spatial(positions,.25)
    for left,right in [('training','validation'),('training','cf06_test'),('validation','cf06_test')]:
        distance=np.linalg.norm(positions[split[left],None,:2]-positions[None,split[right],:2],axis=-1)
        assert distance.min()>=.25
    assert len(set(np.concatenate(list(split.values()))))==100


def test_collapse_diagnostics_do_not_certify_a_constant_representation():
    zero=m.latent_diagnostics(np.ones((16,4)))
    assert zero['effective_rank']==0 and zero['mean_latent_std']==0
    random=m.latent_diagnostics(np.random.default_rng(22).normal(size=(128,4)))
    assert random['effective_rank']>3.5


@pytest.fixture
def fit_fixture(tmp_path,monkeypatch):
    root=tmp_path/'repo';out=root/'artifacts/channel_scene';out.mkdir(parents=True)
    monkeypatch.setattr(b,'ROOT',root);monkeypatch.setattr(b.c,'ROOT',root)
    monkeypatch.setattr(b,'OUT',out);monkeypatch.setattr(b,'PROTOCOL',out/'protocol.json')
    monkeypatch.setattr(b,'INVENTORY',out/'inventory.json')
    p={'epochs':40,'seeds':[17,43,89]}
    f=np.arange(64)*1.e6
    b.c.write_once(b.PROTOCOL,p);b.c.write_once(b.INVENTORY,{'actual_sampled_offsets_hz':f.tolist()})
    monkeypatch.setattr(b,'protocol',lambda:p);monkeypatch.setattr(b,'source_hashes',lambda:{'fixture':'a'*64})
    net=m.PilotCompletionModel(f)
    weights=b._npz(out/'fits/joint-17-weights.npz',{k:v.numpy() for k,v in net.state_dict().items()})
    row={'arm':'joint','seed':17,'epochs':40,'source_sha256':b.source_hashes(),
        'protocol_sha256':b.c.digest(b.PROTOCOL),'inventory_sha256':b.c.digest(b.INVENTORY),
        'held_recordings_opened':False,'weights':weights,'training_wall_s':1.,'selected_epoch':40,
        'history':[{'epoch':k,'loss':1.,'validation_packet_nmse':1/k} for k in range(1,41)]}
    return row


@pytest.mark.parametrize('mutation',['history','selection','protocol','checkpoint'])
def test_cached_fit_rejects_budget_selection_and_identity_drift(fit_fixture,mutation):
    row=fit_fixture;b._validate_fit(row,'joint',17)
    if mutation=='history':row['history']=row['history'][:-1]
    elif mutation=='selection':row['selected_epoch']=39
    elif mutation=='protocol':row['protocol_sha256']='b'*64
    elif mutation=='checkpoint':row['weights']['path']='artifacts/channel_scene/fits/regression-17-weights.npz'
    with pytest.raises(RuntimeError):b._validate_fit(row,'joint',17)


def test_complete_prediction_receipt_requires_every_domain_before_any_array_read(fit_fixture,monkeypatch):
    b.c.write_once(b.OUT/'fits/completed.json',{'fixture':'not scientific'})
    receipt={'source_sha256':b.source_hashes(),'fit_receipt_sha256':b.c.digest(b.OUT/'fits/completed.json'),
        'task_scores_computed_before_prediction_receipt':False,'predictions':{}}
    b.c.write_once(b.OUT/'predictions/completed.json',receipt)
    def forbidden(*args,**kwargs):raise AssertionError('No prediction array should open')
    monkeypatch.setattr(b,'_npz_read',forbidden)
    with pytest.raises(RuntimeError,match='every registered domain'):b._prediction_receipt()


def test_complete_fit_set_cannot_omit_or_duplicate_an_independent_seed(fit_fixture):
    b.c.write_once(b.OUT/'fits/completed.json',{'source_sha256':b.source_hashes(),
        'inventory_sha256':b.c.digest(b.INVENTORY),'fits':[fit_fixture]*6,
        'held_recordings_opened_before_all_fits':False})
    with pytest.raises(RuntimeError,match='All exact registered fits'):b._fits()
