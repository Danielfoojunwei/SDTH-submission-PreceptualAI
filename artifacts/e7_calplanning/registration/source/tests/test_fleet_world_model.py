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


def test_censored_history_affects_memory_without_inventing_measurements(example):
    full,g,obs,rmap,b=example
    changed=replace(obs,value=obs.value.copy())
    # Unknown entries are not zero-dBm radio readings and cannot affect memory.
    changed.value[~changed.heard]=12345
    other=inputs(g,changed.before(60),full.plan[60],rmap)
    for name in ['context','memory','fleet_mean']:
        np.testing.assert_array_equal(getattr(b,name),getattr(other,name))


def test_ema_parameters_receive_no_gradients_and_uncertainty_is_positive(example):
    full,g,obs,rmap,b=example;t=tensors(b);model=FleetJEPA('hybrid_jepa')
    t['valid_link']=torch.ones_like(t['context'][...,0]);t['truth']=t['physics'].clone()
    t['future_context']=t['context'][:,None].expand(-1,9,-1,-1)
    cost,_=objective(model,t);cost.backward()
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert (model(t)['std']>.25).all()


def test_frozen_runtime_uses_the_same_forecasts_without_fitting_a_reference(example):
    from wirelessjev.fleet_control import FleetWorldModel
    from wirelessjev.fleetwm.study import predict
    full,g,obs,rmap,b=example
    model=FleetWorldModel(g,family='fleet_jepa')
    m,s,timing=model.forecast(obs.before(60),full.plan[60]);expected,expected_sd=predict(model.models,b)
    np.testing.assert_array_equal(m.reshape(expected.shape),expected)
    np.testing.assert_array_equal(s.reshape(expected_sd.shape),expected_sd)
    assert timing['map_refit_seconds']==0 and timing['history_cutoff_exclusive']==60
    pol,hook=model.controller();assert pol.gate and not pol.online_map
    with pytest.raises(ValueError):model.controller('protect_teleop')


def test_fleet_results_archive_is_complete_and_sealed():
    from pathlib import Path
    import importlib.util
    root=Path(__file__).resolve().parents[1];path=root/'scripts/audit_fleet_world_model.py'
    spec=importlib.util.spec_from_file_location('fleet_evidence_audit',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.audit();assert result['passed'] and result['test_rollouts']==296


def test_confirmed_profile_keeps_the_preregistered_candidate_instead_of_reranking_tests(example):
    import json
    from wirelessjev.fleet_control import FleetWorldModel
    from wirelessjev.fleetwm.study import ROOT
    full,g,obs,rmap,b=example
    evidence=json.loads((ROOT/'artifacts/fleet_confirmation/summary.json').read_text())
    # The JEPA variant has fewer confirmation misses, but was a secondary arm.
    assert evidence['arms']['hybrid_jepa']['misses']<evidence['arms']['hybrid_supervised']['misses']
    model=FleetWorldModel(g,family='confirmed')
    assert model.family==evidence['fixed_candidate']=='hybrid_supervised'
    assert model.selection_origin=='independent warehouse confirmation'
    policy,gate=model.controller();assert gate is not None and policy.gate


def test_independent_confirmation_evidence_is_complete():
    from pathlib import Path
    import importlib.util
    root=Path(__file__).resolve().parents[1];path=root/'scripts/audit_fleet_confirmation.py'
    spec=importlib.util.spec_from_file_location('independent_confirmation_audit',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.audit();assert result['passed'] and result['native_runs']==32
