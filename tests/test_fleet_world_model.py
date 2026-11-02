"""Causality and architectural ablations for real fleet-model training."""
from dataclasses import fields,replace
import numpy as np
import pytest
import torch
from wirelessjev.fleetwm.data import episode,inputs,fit_map,FleetInputs,scan_context
from wirelessjev.fleetwm.model import FleetJEPA,objective


@pytest.fixture(scope='module')
def example():
    torch.set_num_threads(1)
    full,scan,g,obs=episode(279001,4,4)
    rmap=fit_map(g,obs.before(60));b=inputs(g,obs.before(60),full.plan[60],rmap)
    return full,g,obs,rmap,b


def tensors(b):return {f.name:torch.tensor(getattr(b,f.name),dtype=torch.float32) for f in fields(b)}


def test_observable_boundary_and_strictly_causal_history(example):
    full,g,obs,rmap,b=example
    assert set(FleetInputs.__dataclass_fields__)=={'context','query','memory','physics','map_mean','map_std','fleet_mean'}
    changed=replace(obs,value=obs.value.copy(),heard=obs.heard.copy())
    changed.value[60:]=1e6;changed.heard[60:]=False
    later=inputs(g,changed.before(60),full.plan[60],rmap)
    for f in fields(b):np.testing.assert_array_equal(getattr(b,f.name),getattr(later,f.name))
    np.testing.assert_array_equal(b.context,scan_context(g,obs.before(60)))
    with pytest.raises(ValueError):obs.before(5)


def test_standalone_world_model_has_no_radiomap_dependency(example):
    full,g,obs,rmap,b=example;t=tensors(b);model=FleetJEPA('fleet_jepa').eval()
    # Nonzero heads ensure this is a dependency test, not a zero-head accident.
    torch.nn.init.normal_(model.head.weight,std=.1)
    original=model(t)['mean'].detach()
    t['map_mean']+=1000;t['map_std']*=100
    torch.testing.assert_close(model(t)['mean'],original)


def test_padding_has_no_effect_on_valid_predictions_or_loss(example):
    full,g,obs,rmap,b=example;t=tensors(b);model=FleetJEPA('hybrid_jepa').eval()
    torch.nn.init.normal_(model.head.weight,std=.1)
    original=model(t)['mean'].detach();L=original.shape[-1]
    padded={}
    for k,v in t.items():
        axis=1 if k in {'context','memory'} else 2
        shape=list(v.shape);shape[axis]=6
        padded[k]=torch.cat([v,torch.zeros(shape)],axis)
    padded['valid_link']=torch.cat([torch.ones((len(original),L)),torch.zeros((len(original),6))],1)
    torch.testing.assert_close(model(padded)['mean'][:,:,:L],original)
    padded['truth']=torch.randn_like(padded['map_mean']);padded['truth'][:,:,L:]=-100
    padded['future_context']=torch.zeros((len(original),9,L+6,48))
    loss,_=objective(model,padded);assert torch.isfinite(loss);loss.backward()
    assert model.encoder[0].weight.grad is not None


def test_declared_drift_changes_labels_and_observations_only_after_onset():
    a,_,_,_=episode(279002,2,40,'main');b,_,_,_=episode(279002,2,40,'drift')
    np.testing.assert_array_equal(a.snr[:60*5],b.snr[:60*5])
    assert np.max(np.abs(a.snr[-1]-b.snr[-1]))>1
