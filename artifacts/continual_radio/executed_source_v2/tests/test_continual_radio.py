"""Mechanics/identifiability tests; fixtures are not RF performance evidence."""
import copy
import numpy as np
import pytest
import torch
from wirelessjev.continual_radio import ResponseJEPA,response_objective,update_response_teacher,local_ridge,masked_least_squares


def sample():
    g=torch.Generator().manual_seed(909)
    return {'context':torch.randn(8,16,generator=g),'activity':torch.eye(2)[None].expand(8,2,2).clone(),
        'power':torch.rand(8,2,2,generator=g),'mask':torch.ones(8,2,2,dtype=torch.bool),
        'delta':torch.rand(8,2,generator=g)-.5,'budget':torch.full((8,),2,dtype=torch.long),
        'future_effect':torch.randn(8,2,generator=g)*.2}


def args(b):return [b[k] for k in ('context','activity','power','mask','delta','budget')]


def test_matched_graph_zero_residual_and_signed_linear_power_response():
    b=sample();prior=torch.tensor([[.7,.3],[.2,.8]])
    torch.manual_seed(5);j=ResponseJEPA(prior,family='jepa')
    torch.manual_seed(5);s=ResponseJEPA(prior,family='supervised')
    assert j.inference_parameters==s.inference_parameters
    pj=j(*args(b));ps=s(*args(b))
    assert torch.equal(pj['coupling'],prior[None].expand(8,2,2))
    torch.testing.assert_close(pj['response'],ps['response'],rtol=0,atol=0)
    x=copy.deepcopy(b);x['delta']=-b['delta']
    torch.testing.assert_close(j(*args(x))['response'],-pj['response'],rtol=0,atol=0)
    with pytest.raises(TypeError):j(*args(b),true_coupling=torch.ones(8,2,2))


def test_missing_or_unreceived_future_probe_cannot_change_predictions():
    b=sample();b['budget'][:]=1;b['mask'][:,0,1]=False
    m=ResponseJEPA(np.ones((2,2))*.5)
    x=copy.deepcopy(b);x['power'][:,0,1]=float('nan');x['power'][:,1]=float('nan');x['activity'][:,1]=123.
    torch.testing.assert_close(m(*args(x))['response'],m(*args(b))['response'],rtol=0,atol=0)
    x['mask'][:,0,1]=True
    with pytest.raises(ValueError,match='Observed'):m(*args(x))


def test_actual_jepa_auxiliary_gradients_and_frozen_teacher():
    b=sample();m=ResponseJEPA(np.ones((2,2))*.5)
    loss,metrics=response_objective(m,b);assert torch.isfinite(loss) and metrics['jepa']>0
    loss.backward()
    assert m.effect_encoder[0].weight.grad.abs().sum()>0
    assert m.predictor[0].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in m.teacher.parameters())
    before={k:v.clone() for k,v in m.teacher.state_dict().items()}
    with torch.no_grad():
        for p in m.effect_encoder.parameters():p.add_(.1)
    update_response_teacher(m,.8)
    for name,value in m.teacher.state_dict().items():torch.testing.assert_close(value,.8*before[name]+.2*m.effect_encoder.state_dict()[name])


def test_supervised_has_no_future_teacher_objective_and_checkpoint_roundtrip():
    b=sample();m=ResponseJEPA(np.ones((2,2))*.5,family='supervised')
    loss,metrics=response_objective(m,b);loss.backward()
    assert metrics['jepa']==0 and all(p.grad is None for p in m.predictor.parameters())
    clone=ResponseJEPA(m.prior,**m.config);clone.load_state_dict(m.state_dict())
    torch.testing.assert_close(m(*args(b))['response'],clone(*args(b))['response'],rtol=0,atol=0)


def test_equal_airtime_rank_is_identifiability_not_a_neural_win():
    h=np.array([[.8,.2],[.1,.7]]);sync=np.array([[1.,1.],[0.,0.]]);stagger=np.eye(2)
    assert sync.sum()==stagger.sum()==2 and np.linalg.matrix_rank(sync)==1 and np.linalg.matrix_rank(stagger)==2
    recovered,rank=masked_least_squares(stagger,stagger@h.T,np.ones((2,2),bool))
    np.testing.assert_allclose(recovered,h,rtol=1e-12,atol=1e-12);assert (rank==2).all()
    repeated=np.tile(sync,(20,1));null=np.array([1.,-1.])/np.sqrt(2)
    assert np.linalg.matrix_rank(repeated)==1 and np.allclose(repeated@null,0)
    prior=np.ones((2,2))*.5
    _,c1=local_ridge(prior,sync,sync@h.T,np.ones((2,2),bool))
    _,c2=local_ridge(prior,repeated,repeated@h.T,np.ones_like(repeated,dtype=bool))
    assert null@c1[0]@null==pytest.approx(.25) and null@c2[0]@null==pytest.approx(.25)


def test_masked_numeric_estimators_do_not_consume_unreceived_feedback():
    a=np.eye(2);y=np.array([[.8,.1],[.2,.7]]);mask=np.array([[True,False],[True,True]])
    bad=y.copy();bad[~mask]=np.nan
    prior=np.ones((2,2))*.5
    np.testing.assert_allclose(local_ridge(prior,a,bad,mask)[0],local_ridge(prior,a,y,mask)[0])
    np.testing.assert_allclose(masked_least_squares(a,bad,mask)[0],masked_least_squares(a,y,mask)[0])
