"""Causality, censoring, finite-decision sensitivity and the contract bound."""
import copy
import numpy as np
import torch
from wirelessjev.contracted.data import capped_scans,examples,make_stream
from wirelessjev.contracted.model import DeviceJEPA,candidate_utility,regret_weights,update_target
from wirelessjev.contracted.evaluate import corrupt
from wirelessjev.contracted.theory import lower_bound,rank_counterexample
from wirelessjev.contracted.runtime import DeviceInputs,forecast
import pytest


def test_top_k_bound_and_absent_scan_are_different():
    truth=np.full((10,3,16,3),-60.)
    a=capped_scans(truth,np.random.default_rng(1),missing=0)
    assert np.all(a['heard'].sum(-1)==8)
    assert np.all(a['value'][~a['heard']]==0) and np.all(a['bound']>-90)
    absent=capped_scans(truth,np.random.default_rng(1),missing=1)
    assert not absent['heard'].any() and not absent['scanned'].any()
    below=capped_scans(truth-80,np.random.default_rng(1),missing=0)
    assert not below['heard'].any() and below['scanned'].all() and np.all(below['bound']==-90)


def test_future_scan_changes_do_not_change_causal_predictor_inputs():
    w,s=make_stream(250001,n=2,decisions=2);before=examples(w,s,2)
    other=copy.deepcopy(s);other['value'][62:]+=1000;other['bound'][62:]-=1000;other['heard'][62:]=~other['heard'][62:]
    after=examples(w,other,2)
    for key in ['ctx','route','prior']:np.testing.assert_array_equal(before[key],after[key])
    assert not np.array_equal(before['future_ctx'],after['future_ctx'])


def test_route_ablation_and_ema_really_change_only_the_declared_paths():
    torch.manual_seed(1);m=DeviceJEPA();ctx=torch.randn(2,48,48);route=torch.randn(2,9,48,5);prior=torch.randn(2,9,48);intent=torch.zeros(2,9)
    normal=m(ctx,route,prior,intent)['latent'];changed=route.clone();changed[:,1:]+=10
    assert not torch.equal(normal,m(ctx,changed,prior,intent)['latent'])
    m.config['no_route']=True
    torch.testing.assert_close(m(ctx,route,prior,intent)['latent'],m(ctx,changed,prior,intent)['latent'])
    old=[p.clone() for p in m.target_encoder.parameters()]
    with torch.no_grad():
        for p in m.encoder.parameters():p.add_(1)
    update_target(m)
    for a,b in zip(old,m.target_encoder.parameters()):torch.testing.assert_close(b,a+.01)


def test_native_candidate_utility_and_exact_sensitivity():
    snr=torch.tensor([[[[10.,8.,7.],[20.,2.,1.]]]*6]).reshape(1,6,6);intent=torch.zeros(1,9)
    u=candidate_utility(snr,intent,torch.tensor([0]),torch.tensor([[1.,2.]]))
    torch.testing.assert_close(u,torch.tensor([[9.85,14.7]]))
    weights=regret_weights(torch.tensor([[30.,1.]]),torch.tensor([[0.,5.]]));assert weights[0,0]>weights[0,1]


def test_uncertainty_interventions_preserve_forecasts_and_scaling_ranks():
    s=np.arange(1,101,dtype='f4').reshape(5,2,10);saved=s.copy()
    for kind in ['half','double','noise','shuffle']:assert corrupt(s,kind,1).shape==s.shape
    np.testing.assert_array_equal(s,saved)
    np.testing.assert_array_equal(np.argsort(s,-1),np.argsort(corrupt(s,'half',1),-1))
    assert not np.array_equal(s,corrupt(s,'shuffle',1))


def test_public_forecast_boundary_rejects_label_dictionaries():
    with pytest.raises(TypeError):forecast([],{'truth':np.zeros(1)},np.zeros(9))
    assert set(DeviceInputs.__dataclass_fields__)=={'context','planned_route','geometric_prior'}


def test_rank_only_claim_is_false_and_envelope_bound_is_valid():
    examples=rank_counterexample();assert examples[0]['mean_gain']<0<examples[1]['mean_gain'] and examples[2]['mean_gain']==0
    rng=np.random.default_rng(42)
    for _ in range(50):
        result=lower_bound(rng.uniform(-10,10,100),rng.uniform(-20,20,100),rng.uniform(0,5,100),B=20)
        assert result['bound_holds']
