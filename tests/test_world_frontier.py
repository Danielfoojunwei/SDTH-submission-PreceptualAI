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


def test_zero_cost_kernel_matches_actual_mission_controller():
    from wirelessjev.native.intent_policy import intent_policy
    from wirelessjev.contracted.data import READINGS
    from wirelessjev.gate_repair.gate import reactive_rescue
    reading=READINGS['bulk_uploads'];policy=replace(intent_policy(reading['weights'],reading['focus'],dual_hardware='none'),online_map=False)
    w=build_world(299005,duration_s=5,n_robots=8)
    expected=simulate(w,policy,roam_gate=reactive_rescue)
    assert timed_simulate(w,policy,roam_gate=reactive_rescue,forecast_queue=ForecastQueue(np.zeros(5)))==expected


def test_actual_single_model_worker_is_causal_and_does_not_return_future_results():
    from wirelessjev.frontier_control import OptimizedWorldModel,AsyncForecastService
    from wirelessjev.frontier.study import load_student,tensors
    w,scan,g,o=episode(301000,4,2);history=o.before(60);plan=w.plan[60]
    model=OptimizedWorldModel(g);b,c=FleetMemory(g).inputs(history,plan)
    with torch.no_grad():expected=load_student(model.family)(tensors(b,c))['mean'].numpy()
    service=AsyncForecastService(model)
    try:
        assert service.submit(history,plan,60.)
        # Wait only in this test; the production poll path never waits.
        service.pending[0][1].result(timeout=10)
        assert service.poll(59.9) is None
        mean,std,metadata=service.poll(60.)
        np.testing.assert_allclose(mean.reshape(expected.shape),expected,rtol=1e-6,atol=1e-6)
        assert metadata['history_cutoff_exclusive']==60 and np.isfinite(std).all()
    finally:service.close()


def test_frontier_archive_is_complete_and_all_timing_costs_are_sealed():
    from pathlib import Path
    import importlib.util
    root=Path(__file__).resolve().parents[1];path=root/'scripts/audit_world_frontier.py'
    spec=importlib.util.spec_from_file_location('frontier_archive_audit',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.audit();assert result['passed'] and result['fresh_sites']==24


def test_admission_hook_is_distinct_from_confidence_and_defaults_are_unchanged():
    from wirelessjev.admission_sim import simulate as admitted
    from wirelessjev.gate_repair.gate import reactive_rescue
    w=build_world(299010,duration_s=5,n_robots=16,site_params={'rack_scale':1.6,'pl_coef':38.,'shadow_db':7.})
    p=replace({p.name:p for p in policies()}['native'],gate=True,roam_cost_s=4.)
    expected=timed_simulate(w,p,roam_gate=reactive_rescue)
    assert admitted(w,p,roam_gate=reactive_rescue)==expected
    # Even proposals from the reactive path cannot bypass an explicit denial.
    result=admitted(w,p,roam_gate=lambda c:np.zeros(w.n,dtype=bool),handover_admission=lambda c:np.zeros(w.n,dtype=bool))
    assert result['roams']==0 and result['control_sent']==16*5*20
