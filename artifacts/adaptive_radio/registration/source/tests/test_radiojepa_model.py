"""Analytic software fixtures; no public-data performance results."""
import pytest
import torch
from wirelessjev.radiojepa.model import SceneJEPA


def operands():
    torch.manual_seed(4)
    a, b = torch.randn(6, 4, 64) * 3, torch.randn(6, 4, 64) * 3
    xa, xb = torch.randn(4, 3), torch.randn(4, 3)
    origin = torch.zeros(6, 3)
    offsets = torch.linspace(-24.6e6, 24.6e6, 64)
    return a, b, xa, xb, origin, offsets


def test_physical_decoder_has_finite_gradients_and_no_power_layernorm():
    a,b,xa,xb,o,f = operands()
    model = SceneJEPA(regularizer='sigreg')
    loss, terms = model.objective(a,b,xyz_a=xa,xyz_b=xb,origin=o,carrier_hz=1.272e9,offsets_hz=f)
    loss.backward()
    assert loss.isfinite() and all(v.isfinite() for v in terms.values())
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in model.modules())
    enc = model.encode(a,xa,o,1.272e9)
    shifted = model.encode(a+10.,xa,o,1.272e9)
    torch.testing.assert_close(shifted['power_coordinate']-enc['power_coordinate'],torch.full((6,),10.))


def test_receiver_permutation_translation_and_yaw_preserve_predictions():
    a,b,xa,xb,o,f = operands()
    model = SceneJEPA(regularizer='none').eval()
    def predict(x, p, q, origin):
        return model.render(model.encode(x,p,origin,1.272e9),p,q,origin,1.272e9,f)
    base = predict(a,xa,xb,o)
    order = torch.tensor([2,0,3,1])
    torch.testing.assert_close(base,predict(a[:,order],xa[order],xb,o),atol=1e-4,rtol=1e-4)
    rotation = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    shift = torch.tensor([14.,-6.,2.])
    torch.testing.assert_close(base,predict(a,xa@rotation+shift,xb@rotation+shift,o@rotation+shift),atol=1e-4,rtol=1e-4)


def test_regression_control_removes_only_alignment_from_common_objective():
    a,b,xa,xb,o,f = operands()
    model = SceneJEPA(regularizer='none')
    joint,j = model.objective(a,b,xyz_a=xa,xyz_b=xb,origin=o,carrier_hz=1.272e9,offsets_hz=f)
    regression,r = model.objective(a,b,xyz_a=xa,xyz_b=xb,origin=o,carrier_hz=1.272e9,offsets_hz=f,alignment=False)
    torch.testing.assert_close(j['measurement_huber'],r['measurement_huber'])
    torch.testing.assert_close(joint-regression,.1*j['joint_alignment'])
    assert r['joint_alignment'] == 0


def test_unidentified_factors_and_bad_frequency_metadata_rejected():
    with pytest.raises(ValueError,match='identify'):
        SceneJEPA(regularizer='ppr')
    a,b,xa,xb,o,f = operands()
    model = SceneJEPA(regularizer='none')
    e = model.encode(a,xa,o,1.272e9)
    with pytest.raises(ValueError,match='increase'):
        model.render(e,xa,xb,o,1.272e9,-f)


def test_constant_projector_cannot_hide_from_active_regularizer():
    a,b,xa,xb,o,f = operands()
    model = SceneJEPA(regularizer='sigreg')
    with torch.no_grad():
        for p in model.encoder.projector.parameters():
            p.zero_()
    _, terms = model.objective(a,b,xyz_a=xa,xyz_b=xb,origin=o,carrier_hz=1.272e9,offsets_hz=f)
    assert terms['joint_alignment'] == 0
    assert terms['projected_prior'] > 1.


def test_query_preserves_receiver_rf_association_in_symmetric_layout():
    a,b,xa,xb,o,f = operands()
    model = SceneJEPA(regularizer='none')
    visible = torch.tensor([[-2.,0.,0.],[2.,0.,0.],[0.,-2.,0.],[0.,2.,0.]])
    queries = torch.tensor([[-1.,0.,0.],[1.,0.,0.]])
    encoded = model.encode(a,visible,o,1.272e9)
    result = model.render(encoded,visible,queries,o,1.272e9,f)
    assert not torch.allclose(result[:,0],result[:,1],atol=1e-5,rtol=1e-5)
