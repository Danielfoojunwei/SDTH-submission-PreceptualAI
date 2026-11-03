from dataclasses import replace
import numpy as np
import pytest
import torch
from wirelessjev.closed_loop.telemetry import DeliveredScans
from wirelessjev.closed_loop.control import integral_from, OutageUtility
from wirelessjev.closed_loop.runtime import SinglePredictor, LiveForecasts
from wirelessjev.closed_loop.sim import simulate
from wirelessjev.admission_sim import simulate as previous_simulate
from wirelessjev.fleetwm.data import episode, Observations
from wirelessjev.frontier.study import mask_scans
from wirelessjev.native.sim import policies
from wirelessjev.frontier.timing import ForecastQueue


def warm(n=2, links=12):
    return Observations(np.zeros((60,n,2),dtype='f4'), np.zeros((60,n,links),dtype='f4'),
                        np.zeros((60,n,links),dtype=bool), np.full((60,n,links),-90.,dtype='f4'),
                        np.zeros((60,n),dtype=bool))


def test_scans_require_completion_delivery_and_a_strictly_past_bin():
    scans=DeliveredScans(warm(),4,999)
    scans.scan(0,np.array([0]),np.array([[3.,4.]]),np.full((1,4,3),-50.),np.array([10]))
    assert 0 in scans.pending
    scans.deliver(9,np.array([True,False]))
    assert not scans.events and not scans.before(1).scanned[-1].any()
    scans.complete(10)
    assert (scans.local[0]>-90).sum()==8 and not scans.events
    scans.deliver(11,np.array([False,True]))
    assert not scans.events
    scans.deliver(12,np.array([True,False]))
    assert not scans.before(0).scanned.any()
    past=scans.before(1)
    assert past.scanned[-1,0] and not past.scanned[-1,1]
    assert past.heard[-1,0].sum()==8
    np.testing.assert_equal(past.positions[-1,0],[3,4])
    e=scans.events[0]
    assert e['measurement_slot']<e['complete_slot']<=e['delivery_slot']<e['first_usable_decision']*100
    bound=past.bound[-1,0,0]
    assert bound==past.value[-1,0][past.heard[-1,0]].min()


def test_missing_scans_are_missing_inputs_and_not_below_threshold_readings():
    s=DeliveredScans(warm(),3,999)
    s.pose(0,np.array([[1.,2.],[3.,4.]]))
    p=s.before(1)
    assert not p.scanned[-1].any() and not p.heard[-1].any()
    np.testing.assert_equal(p.positions[-1],[[1,2],[3,4]])
    # Pose access never invents an RSSI measurement.
    assert s.summary()['reports_delivered']==0


def test_local_scan_cache_is_available_before_remote_report_and_has_no_hidden_neighbors():
    s=DeliveredScans(warm(),3,999)
    s.scan(0,np.array([0]),np.array([[0.,0.]]),np.full((1,4,3),-50.),np.array([10]))
    s.complete(10)
    r=s.neighbors(np.array([1,2]),np.full((2,3),5.),np.full(3,-95.),np.full(3,20.))
    np.testing.assert_equal(r[:,[0,1,2,3]].shape,(2,4,3))
    assert not s.events and (r[1,[0,1,3]]==-100).all()
    np.testing.assert_equal(r[0,1],np.full(3,-85.))


def test_exact_fractional_outage_integration_does_not_drop_first_second():
    probability=np.full((2,9,4),.25)
    np.testing.assert_allclose(integral_from(probability,.03,30),29.97*.25)
    np.testing.assert_allclose(integral_from(probability,4,10),6*.25)
    np.testing.assert_equal(integral_from(probability,10,10),np.zeros((2,4)))
    with pytest.raises(ValueError):integral_from(probability,5,4)
    gate=OutageUtility(np.zeros((1,2,9,4,3)),ForecastQueue([0]),4,30)
    # Equal-quality APs cannot save loss by paying an outage.
    fc=np.full((2,9,4,3),10.,dtype='f4')
    score=gate.scores(fc,np.array([0,1]))
    assert score.argmax(1).tolist()==[0,1]


def test_new_hooks_absent_preserve_previous_packet_counters():
    torch.set_num_threads(1)
    w,_,g,o=episode(299020,5,3)
    from wirelessjev.contracted.data import runtime_world
    native={p.name:p for p in policies()}['native']
    rw=runtime_world(w,3,w.forecast[60:63],w.sigma_belief[60:63])
    for gate in [False,True]:
        p=replace(native,gate=gate)
        assert simulate(rw,p)==previous_simulate(rw,p)


def test_live_forecasts_obey_fifo_and_report_cutoff_in_actual_kernel():
    torch.set_num_threads(1)
    full,_,g,o=episode(299021,5,4)
    o=mask_scans(o,5)
    history=DeliveredScans(o.before(60),4,299021)
    live=LiveForecasts(SinglePredictor(g,'distilled'),history,4,{'cold':.2,'warm':.2,'refit':.2},0.)
    from wirelessjev.contracted.data import runtime_world
    p=replace({p.name:p for p in policies()}['native'],cadence_scan=True,scan_period_s=1.,gate=True)
    from wirelessjev.gate_repair.gate import reactive_rescue
    r=simulate(runtime_world(full,4,live.mean,live.std[:,:,0]),p,scan_observer=history,
               forecast_source=live,forecast_queue=live,roam_gate=reactive_rescue)
    assert r['control_sent']==5*4*20 and len(live.records)==4
    assert [r['history_cutoff_exclusive'] for r in live.records]==[60,61,62,63]
    assert live.ready==[20,120,220,320]
    assert all(e['delivery_slot']<100*e['first_usable_decision'] for e in history.events)
    assert history.summary()['reports_delivered']<=history.summary()['scans_started']
    assert np.isfinite(live.mean).all()


def test_single_family_runtime_preserves_the_frozen_forecast_values():
    from wirelessjev.frontier.study import Predictors
    torch.set_num_threads(1)
    full,_,g,o=episode(301001,4,22)
    o=mask_scans(o,5)
    reference=Predictors(g)
    predictors={f:SinglePredictor(g,f) for f in ('radiomap','distilled')}
    for t in (60,61,80):
        expected,_,_=reference.step(o.before(t),full.plan[t])
        for family,pred in predictors.items():
            mean,sd,_,_=pred.step(o.before(t),full.plan[t])
            np.testing.assert_allclose(mean.reshape(expected[family][0].shape),expected[family][0],rtol=3e-5,atol=3e-5)
            np.testing.assert_allclose(sd.reshape(expected[family][1].shape),expected[family][1],rtol=3e-5,atol=3e-5)


def test_delivered_observation_archive_is_complete_causal_and_replayable():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/report_closed_loop.py'
    spec=importlib.util.spec_from_file_location('closed_loop_archive_audit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.audit()
    assert result['passed'] and result['native_runs']==120 and result['fresh_sites']==8
