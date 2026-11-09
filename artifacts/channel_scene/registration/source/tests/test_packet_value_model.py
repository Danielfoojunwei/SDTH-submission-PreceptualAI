"""Model mechanics only; these fixtures are not warehouse performance evidence."""
import copy
import pytest
import torch
from wirelessjev.packet_value.model import PacketValueModel, objective, update_target


def batch(size=5):
    generator = torch.Generator().manual_seed(408)
    return {'context': torch.randn(size,6,32,generator=generator),
            'query': torch.randn(size,24,generator=generator),
            'delta_misses': torch.arange(size,dtype=torch.float32)-2,
            'delta_gap': (torch.arange(size,dtype=torch.float32)-2)/10,
            'future_context': torch.randn(size,6,32,generator=generator),
            'future_valid': torch.ones(size,6,dtype=torch.bool)}


def test_jepa_ablation_has_identical_inference_and_no_future_in_forward():
    torch.manual_seed(11); jepa=PacketValueModel('jepa')
    torch.manual_seed(11); supervised=PacketValueModel('supervised')
    b=batch()
    a=jepa(b['context'],b['query']); s=supervised(b['context'],b['query'])
    assert jepa.inference_parameters==supervised.inference_parameters
    assert sum(p.numel() for p in jepa.parameters())==sum(p.numel() for p in supervised.parameters())
    for key in a: torch.testing.assert_close(a[key],s[key],rtol=0,atol=0)
    assert a['mean'].shape==(5,2) and (a['std']>0).all()
    assert not any(p.requires_grad for p in jepa.target_encoder.parameters())
    # Future observations/labels cannot enter the inference method.
    with pytest.raises(TypeError): jepa(b['context'],b['query'],b['future_context'])


def test_real_gradients_include_jepa_and_never_the_target_encoder():
    torch.manual_seed(12)
    for family in ('jepa','supervised'):
        model=PacketValueModel(family); loss, metrics=objective(model,batch())
        assert torch.isfinite(loss)
        assert (metrics['jepa']>0).item() == (family=='jepa')
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all(p.grad is None for p in model.target_encoder.parameters())
        assert model.head.weight.grad.abs().sum()>0
        if family=='jepa': assert model.encoder.gru.weight_ih_l0.grad.abs().sum()>0


def test_missing_future_features_are_ignored_even_when_poisoned():
    torch.manual_seed(13); model=PacketValueModel('jepa'); b=batch()
    b['future_valid']=torch.ones_like(b['future_context'],dtype=torch.bool)
    b['future_valid'][0]=False
    b['future_valid'][1,2:]=False
    b['future_valid'][2,:,4:]=False
    original,_=objective(model,b)
    poisoned=copy.deepcopy(b)
    poisoned['future_context'][~b['future_valid']]=float('nan')
    actual,_=objective(model,poisoned)
    torch.testing.assert_close(actual,original,rtol=0,atol=0)
    b['future_valid'][:]=False
    b['future_context'][:]=float('nan')
    loss,metrics=objective(model,b)
    assert torch.isfinite(loss) and metrics['jepa']==0


def test_only_jepa_objective_uses_future_observations():
    model=PacketValueModel('supervised'); b=batch()
    without_future={k:v for k,v in b.items() if not k.startswith('future_')}
    expected,_=objective(model,without_future)
    b['future_context'][:]=float('nan')
    actual,_=objective(model,b)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    with pytest.raises(KeyError): objective(PacketValueModel('jepa'),without_future)


def test_ema_is_an_explicit_interpolation_and_storage_roundtrips():
    torch.manual_seed(14); model=PacketValueModel('jepa')
    before={k:v.clone() for k,v in model.target_encoder.state_dict().items()}
    with torch.no_grad():
        for parameter in model.encoder.parameters(): parameter.add_(.5)
    update_target(model,.8)
    for key,value in model.target_encoder.state_dict().items():
        torch.testing.assert_close(value,.8*before[key]+.2*model.encoder.state_dict()[key])
    clone=PacketValueModel(**model.config);clone.load_state_dict(model.state_dict())
    b=batch(); a=model(b['context'],b['query']); c=clone(b['context'],b['query'])
    torch.testing.assert_close(a['mean'],c['mean'],rtol=0,atol=0)
    with pytest.raises(ValueError):update_target(model,1.1)


def test_invalid_inference_input_and_invalid_factual_labels_fail_explicitly():
    model=PacketValueModel('jepa'); b=batch()
    with pytest.raises(ValueError): model(b['context'],b['query'][:,:23])
    with pytest.raises(ValueError): model(b['context'][:,:,:31],b['query'])
    b['delta_misses'][0]=float('nan')
    with pytest.raises(ValueError):objective(model,b)
    b['label_valid']=torch.ones(5,2,dtype=torch.bool);b['label_valid'][0,0]=False
    loss,_=objective(model,b)
    assert torch.isfinite(loss)


def test_actions_reuse_history_without_changing_predictions_or_gradients():
    torch.manual_seed(15)
    batched=PacketValueModel('jepa')
    # A trained nonzero head is essential: a neutral zero head would hide an
    # incorrect encoder/action ordering or lost encoder gradients.
    with torch.no_grad():
        batched.head.weight.normal_(std=.1)
        batched.head.bias.normal_(std=.1)
    repeated=copy.deepcopy(batched)
    context=torch.randn(4,6,32,requires_grad=True)
    query=torch.randn(4,5,24,requires_grad=True)
    other_context=context.detach().clone().requires_grad_(True)
    other_query=query.detach().clone().requires_grad_(True)
    a=batched.forward_actions(context,query)
    b=repeated(other_context[:,None].expand(-1,5,-1,-1).reshape(20,6,32),other_query.reshape(20,24))
    for key,value in a.items():
        torch.testing.assert_close(value,b[key].reshape_as(value),rtol=2e-6,atol=2e-6)
    (a['mean'].square().mean()+a['std'].square().mean()).backward()
    (b['mean'].square().mean()+b['std'].square().mean()).backward()
    torch.testing.assert_close(context.grad,other_context.grad,rtol=5e-6,atol=5e-6)
    torch.testing.assert_close(query.grad,other_query.grad,rtol=5e-6,atol=5e-6)
    for (name,p),(_,q) in zip(batched.named_parameters(),repeated.named_parameters(),strict=True):
        if p.grad is None:
            assert q.grad is None,name
        else:
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-5,msg=name)
    with pytest.raises(ValueError):batched.forward_actions(context,query[:,:,:23])
    with pytest.raises(ValueError):batched.forward_actions(context,query[:,0])

