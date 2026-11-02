from dataclasses import fields,replace
import numpy as np
import pytest
import torch
from wirelessjev.frontier.fast import FastMap,FleetMemory
from wirelessjev.frontier.timing import ForecastQueue
from wirelessjev.frontier.sim import simulate as timed_simulate
from wirelessjev.native.sim import simulate,policies
from wirelessjev.native.world import build_world
from wirelessjev.fleetwm.data import episode,fit_map,inputs


def test_fast_covariance_and_rolling_statistics_match_original():
    torch.set_num_threads(1)
    w,scan,g,o=episode(299001,6,4);rmap=fit_map(g,o.before(60));mem=FleetMemory(g)
    for t in [60,61,63]:
        original=inputs(g,o.before(t),w.plan[t],rmap)
        optimized,count=mem.inputs(o.before(t),w.plan[t],rmap)
        for f in fields(original):
            np.testing.assert_allclose(getattr(original,f.name),getattr(optimized,f.name),rtol=3e-5,atol=3e-5,err_msg=f.name)
        assert count.shape==(6,48,1) and np.isfinite(count).all()
    with pytest.raises(ValueError):mem.inputs(o.before(60),w.plan[60],rmap)


def test_research_kernel_reproduces_original_without_new_options():
    w=build_world(299002,duration_s=4,n_robots=8);base={p.name:p for p in policies()}['native']
    for gate in [False,True]:
        p=replace(base,gate=gate)
        assert timed_simulate(w,p)==simulate(w,p)
        assert timed_simulate(w,p,forecast_queue=ForecastQueue(np.zeros(4)))==simulate(w,p)


def test_fifo_never_exposes_a_forecast_before_completion_and_advances_age():
    q=ForecastQueue([.2,1.4,.1]);assert q.take(19) is None and q.take(20)==0
    assert q.take(100) is None and q.take(239) is None and q.take(240)==1
    # The third job waited for the second, finishing at 2.5 rather than 2.1.
    assert q.take(249) is None and q.take(250)==2
    h=np.array([0,1,2,3,4,5,10,20,30],dtype='f4')
    f=np.broadcast_to(h[None,:,None,None],(2,9,3,3))
    q.age=.25;out=q.advance(f);np.testing.assert_allclose(out[:,:-1],f[:,:-1]+.25)
    with pytest.raises(ValueError):ForecastQueue([float('nan')])


def test_delayed_forecast_and_scan_cost_are_actual_packet_interventions():
    w=build_world(299003,duration_s=5,n_robots=8);p={p.name:p for p in policies()}['native']
    z=timed_simulate(w,p,forecast_queue=ForecastQueue([.15]*5),scan_duration_s=.09)
    assert z['control_sent']==8*5*20 and sum(z['miss_causes'].values())==z['control_misses']


def test_packet_loss_and_temporal_models_have_finite_real_gradients():
    from wirelessjev.frontier.model import FrontierModel,objective
    w,scan,g,o=episode(299004,3,2);b,c=FleetMemory(g).inputs(o.before(60),w.plan[60])
    from wirelessjev.frontier.study import tensors
    t=tensors(b,c);t['truth']=t['physics'].clone();t['teacher']=t['fleet_mean'].clone()
    for family in ['distilled','packet','temporal']:
        model=FrontierModel(family);cost=objective(model,t)
        assert torch.isfinite(cost);cost.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        risk=model.phy_risk(t['truth']);assert ((risk>=0)&(risk<=1)).all()
