"""Mechanics fixtures and real native-kernel checks, not study results."""
from dataclasses import asdict,replace
import numpy as np
import pytest
from wirelessjev.native.world import build_world
from wirelessjev.native.phy import LinkCurves
from wirelessjev.service_world.sim import simulate as sealed_simulate
from wirelessjev.continual_packet.sim import simulate,Policy,_chain,_managed_chain_allocation
from wirelessjev.continual_packet.probes import PublicProbeSpec,ProbeRuntime,ProbeMailbox,geometry_prior,managed_overlap


def world(seconds=2):
    return build_world(12345,duration_s=seconds,n_robots=4,sigma_map=4.)


def spec(w,*,design='staggered',starts=(4,),**kwargs):
    return PublicProbeSpec(((30.,20.),(70.,50.)),(1.,2.),int(w.channel[0,1]),
        np.zeros((round(w.duration_s*100),2)),probe_starts=starts,design=design,
        noise_floor_fraction=0.,missing_probability=0.,**kwargs)


def run_runtime(rt,stop):
    ambient=np.broadcast_to(rt._noise,(rt._s,3))
    for slot in range(stop):rt.step(slot,ambient)


def test_absent_runtime_alone_reproduces_sealed_allocator_exactly():
    w=world();p=Policy('equivalence','native')
    original=sealed_simulate(w,p,detailed=True,observation_period_slots=10,native_proposal_ttl_slots=100)
    assert simulate(w,p,detailed=True,observation_period_slots=10,native_proposal_ttl_slots=100)==original
    rt=ProbeRuntime.from_world(w,spec(w,design='none',starts=()),ProbeMailbox())
    inactive=simulate(w,p,managed_probes=rt,detailed=True,observation_period_slots=10,native_proposal_ttl_slots=100)
    summary=inactive['managed_probe_summary'];accounting=inactive['managed_packet_accounting']
    assert summary['allocated_domain_slot_airtime']==0 and summary['control_allocator_slots']==200
    assert summary['report_bytes_transmitted']==0 and abs(accounting['bulk_flow_conservation_error_bits'])<1e-5


def test_zero_duty_and_both_probe_schedules_use_real_allocator_every_slot(monkeypatch):
    import wirelessjev.continual_packet.sim as module
    original=module._managed_chain_allocation;calls=[]
    def record(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)  # Instrumentation never replaces the algorithm.
    monkeypatch.setattr(module,'_managed_chain_allocation',record)
    w=world(1);p=Policy('shared-new-study-baseline','native')
    simulate(w,p);assert not calls
    for design in ('none','synchronized','staggered'):
        calls.clear();rt=ProbeRuntime.from_world(w,spec(w,design=design),ProbeMailbox())
        result=simulate(w,p,managed_probes=rt)
        summary=result['managed_probe_summary']
        assert len(calls)==summary['control_allocator_slots']==100
        assert summary['control_allocator']=='per_retry_band_domain_every_managed_slot'


def test_public_plan_is_immutable_and_probes_cannot_mute_ordinary_data():
    plan=np.zeros((100,2));plan[20]=(.1,.3)
    s=PublicProbeSpec(((0.,0.),(1.,1.)),(1.,1.),36,plan,(4,),'staggered')
    before=s.digest();plan[20]=1.;s.activity(20)[:]=0
    np.testing.assert_array_equal(s.activity(20),[.1,.3]);assert s.digest()==before
    np.testing.assert_array_equal(s.activity(1000),[0.,0.])
    np.testing.assert_array_equal(np.stack([s.activity(k) for k in (4,5,6)]),[[0.,0.],[.02,0.],[0.,.02]])
    bad=np.zeros((100,2));bad[5,0]=1.
    with pytest.raises(ValueError,match='zero'):replace(s,activity_plan=bad)
    with pytest.raises(ValueError,match='100ms'):replace(s,probe_starts=(9,))


def test_private_native_coupling_uses_the_same_site_parameters():
    from wirelessjev.twin.warehouse import make_warehouse
    from wirelessjev.native.world import _path_losses
    params={'rack_scale':1.4,'pl_coef':38.,'shadow_db':6.}
    w=build_world(12345,duration_s=1,n_robots=2,site_params=params);s=spec(w)
    rt=ProbeRuntime.from_world(w,s,ProbeMailbox(),site_params=params)
    scene=make_warehouse(w.seed,n_robots=1,rack_scale=1.4);scene.pl_coef=38.;scene.shadow_db=6.
    tx=np.asarray(s.positions_m)
    loss=_path_losses(scene,[a for a in scene.aps if a.band=='5g'],[a for a in scene.aps if a.band=='2g'],tx[:,0],tx[:,1])[:,:,1].T
    expected=10**(-loss.astype(float)/10);expected[w.channel[:,1]!=s.channel]=0.
    np.testing.assert_array_equal(rt._g,expected)
    prior=geometry_prior(s,w.sites,w.channel[:,1])
    assert prior.shape==rt._g.shape and not np.array_equal(prior,rt._g)


def test_rank_exposure_and_subset_report_retain_occupancy_difference():
    w=world();boxes={};runtime={};receipts={}
    for design in ('synchronized','staggered'):
        # Fully sensed mechanics fixture isolates union-occupancy accounting.
        boxes[design]=ProbeMailbox();runtime[design]=ProbeRuntime(w,spec(w,design=design),boxes[design],np.full((len(w.sites),2),1e-3))
        run_runtime(runtime[design],20);receipts[design]=boxes[design].drain(19)[0]
    a=np.asarray(receipts['synchronized'].exposure_mw);b=np.asarray(receipts['staggered'].exposure_mw)
    np.testing.assert_array_equal(a.sum(0),b.sum(0))
    assert np.linalg.matrix_rank(a[1:]-a[0])==1 and np.linalg.matrix_rank(b[1:]-b[0])==2
    assert runtime['synchronized']._requested<runtime['staggered']._requested
    r=receipts['staggered'];y=np.asarray(r.received_mw,dtype=float)
    assert 0<len(r.receiver_sites)<len(w.sites)
    assert y.shape==np.shape(r.observed)==(3,len(r.receiver_sites)) and len(r.noise_std_mw)==len(r.receiver_sites)
    for column,site in enumerate(r.receiver_sites):
        estimate=np.linalg.lstsq(b[1:]-b[0],y[1:,column]-y[0,column],rcond=None)[0]
        np.testing.assert_allclose(estimate,runtime['staggered']._g[site],rtol=1e-10,atol=1e-20)
    assert 'coupling' not in asdict(r) and not hasattr(boxes['staggered'],'_g')
    assert r.payload_bytes==128*len(r.receiver_sites)
    with pytest.raises(ValueError,match='columns'):replace(r,receiver_sites=r.receiver_sites[:-1])


def test_reports_need_reserved_transport_and_causal_delivery():
    w=world();box=ProbeMailbox();rt=ProbeRuntime.from_world(w,spec(w,starts=(0,)),box)
    for slot in range(6):
        rt.step(slot,np.broadcast_to(rt._noise,(rt._s,3)));assert box.drain(slot)==[]
        with pytest.raises(ValueError,match='future'):box.drain(slot+1)
    rt.step(6,np.broadcast_to(rt._noise,(rt._s,3)));r=box.drain(6)[0]
    assert r.measured_slots==(0,1,2) and r.completed_slot==3 and r.delivered_slot==6
    summary=rt.summary();assert summary['report_bytes_transmitted']==r.payload_bytes
    assert all(v['slot']>=r.completed_slot for v in summary['report_transport_slots'])
    tiny=ProbeRuntime.from_world(w,spec(w,starts=(0,),report_rate_mbps=.001),ProbeMailbox())
    run_runtime(tiny,20)
    assert tiny.summary()['cycles_delivered']==0 and tiny.summary()['report_bytes_transmitted']==0


def test_missing_captures_remain_missing_and_still_pay_report_transport():
    w=world();s=replace(spec(w),missing_probability=1.);box=ProbeMailbox();rt=ProbeRuntime.from_world(w,s,box)
    run_runtime(rt,20);r=box.drain(19)[0]
    assert not np.asarray(r.observed).any() and all(v is None for row in r.received_mw for v in row)
    assert rt.summary()['report_bytes_transmitted']==r.payload_bytes>0


class MailboxForecasts:
    """Deterministic scheduler to check callback ordering; no learned model."""
    def __init__(self,w,box):self.base=w.forecast.copy();self.std=w.sigma_belief.copy();self.box=box;self.events=[];self.last=-1;self.age=0.
    def issue(self,decision,plan):
        self.events.append((decision,[r.delivered_slot for r in self.box.drain(decision*100)]))
        return self.base[decision].copy(),np.repeat(self.std[decision,:,None],9,axis=1)
    def take(self,slot):
        if slot%100:return None
        self.last=slot//100;return self.last
    def advance(self,forecast):return forecast


def test_mailbox_delivery_precedes_forecast_issue_at_the_same_slot():
    w=world();box=ProbeMailbox();rt=ProbeRuntime.from_world(w,spec(w,starts=(94,)),box)
    forecast=MailboxForecasts(w,box)
    simulate(w,Policy('mailbox','native'),managed_probes=rt,forecast_queue=forecast,forecast_source=forecast)
    assert forecast.events==[(0,[]),(1,[100])]


def test_actual_packet_service_pays_occupancy_before_traffic():
    w=world(21);w.bulk_mbps[:]=1.;w.video_mbps[:]=1.
    # Mechanics fixture: only the clients' 5GHz path is usable.
    w.snr[...,0]=-100.;w.snr[...,2]=-100.;w.forecast[...,0]=-100.;w.forecast[...,2]=-100.
    w.rssi_dl[...,0]=-130.;w.rssi_dl[...,2]=-130.;w.channel[:,1]=36
    p=Policy('cost','native',route_forecast=False,reflex=False,duplication=False,predictive_mcs=False,planned_roam=False,data_timing=False)
    baseline=simulate(w,p)
    s=PublicProbeSpec(((0.,0.),(1.,1.)),(1.,1.),36,np.ones((2100,2)))
    rt=ProbeRuntime(w,s,ProbeMailbox(),np.full((len(w.sites),2),1e-3))
    result=simulate(w,p,managed_probes=rt)
    assert result['control_sent']==baseline['control_sent'] and result['control_misses']>baseline['control_misses']
    assert result['bulk_mbps']==0 and result['video_delivered_vs_requested']==0
    accounting=result['managed_packet_accounting']
    assert abs(accounting['bulk_flow_conservation_error_bits'])<1e-5
    assert accounting['video_requested_bits']>=accounting['video_generated_bits']>=accounting['video_delivered_bits']
    assert accounting['bulk_generated_bits']>0 and accounting['bulk_delivered_bits']==0
    assert accounting['bulk_overflow_drop_bits']>0 and accounting['final_bulk_backlog_bits']>0


def test_shared_blackout_disables_real_phy_retry_chain():
    curves=LinkCurves();links=np.array([[1,1,1]]);mcs=np.array([[0,0,0]]);snr=np.full((1,3),100.)
    normal,_=_chain(curves,links,mcs,snr,np.array([False]))
    blocked,_=_chain(curves,links,mcs,snr,np.array([False]),link_available=np.array([[True,False,True]]))
    assert normal[0]>0 and blocked[0]==0


def test_unsupported_managed_radio_modes_fail_explicitly():
    w=world();rt=ProbeRuntime.from_world(w,spec(w),ProbeMailbox())
    for p in (Policy('nr','nr'),Policy('dual','native',dual_radio=True),Policy('layered','native',layered=True)):
        with pytest.raises(ValueError,match='profile'):simulate(w,p,managed_probes=rt)


def test_str_retry_chains_jointly_respect_every_domain_budget():
    n=20;curves=LinkCurves();mcs=np.zeros((n,3),dtype=int);domains=np.tile([0,1,2],(n,1))
    common=(mcs,np.full((n,3),100.),np.ones(n,dtype=bool),None,np.ones((n,3),dtype=bool),np.ones(n,dtype=bool),domains)
    chains=[(np.full((n,3),b,dtype=int),*common) for b in (1,2)]
    budget=np.array([.01,.012,.014]);ok,used,_,robot_air=_managed_chain_allocation(curves,chains,budget)
    assert np.all(used<=budget+1e-12) and np.all((ok>=0)&(ok<=1))
    assert robot_air.sum()==pytest.approx(used.sum())


def test_absent_runtime_preserves_100robot_predecessor_packet_results():
    w=build_world(12345,duration_s=1,n_robots=100,sigma_map=4.);p=Policy('100robot-equivalence','native')
    assert simulate(w,p)==sealed_simulate(w,p)


def test_managed_100robot_common_allocator_is_reproducible():
    w=build_world(12345,duration_s=1,n_robots=100,sigma_map=4.);p=Policy('100robot-managed','native')
    results=[]
    for _ in range(2):
        rt=ProbeRuntime.from_world(w,spec(w,design='none',starts=()),ProbeMailbox())
        results.append(simulate(w,p,managed_probes=rt))
    assert results[0]==results[1]
    assert results[0]['managed_probe_summary']['control_allocator_slots']==100


def test_carrier_sense_full_hidden_and_mixed_nested_windows():
    g=np.array([[1e-6,2e-6],[1e-12,2e-12]])
    rf,busy=managed_overlap(g,[1.,1.],[.1,.2],np.array([0,1]),np.array([36,36]),36)
    assert rf[0]==0 and busy[0]==.2 and busy[1]==0
    assert rf[1]==pytest.approx(1e-12*.1+2e-12*.2)
    mixed=np.array([[1e-6,1e-12]])
    rf,busy=managed_overlap(mixed,[1.,1.],[.1,.05],np.array([0]),np.array([36]),36)
    assert rf[0]==0 and busy[0]==.1
    rf,busy=managed_overlap(mixed,[1.,1.],[.1,.2],np.array([0]),np.array([36]),36)
    assert rf[0]==pytest.approx(1e-12*.1/.9) and busy[0]==.1
    rf,busy=managed_overlap(mixed,[1.,1.],[1.,1.],np.array([0]),np.array([36]),36)
    assert rf[0]==0 and busy[0]==1.


def test_control_retry_pays_actual_band_and_keeps_free_band_fallback():
    curves=LinkCurves();mcs=np.zeros((1,3),dtype=int);domains=np.array([[0,1,2]])
    snr=np.array([[-1000.,100.,100.]]);enabled=np.array([True]);available=np.ones((1,3),dtype=bool)
    chain=(np.array([[0,1,1]]),mcs,snr,np.array([False]),None,available,enabled,domains)
    ok,air,_,robot_air=_managed_chain_allocation(curves,[chain],np.array([1.,0.,1.]))
    assert ok[0]==0 and air[1]==0
    assert robot_air.sum()==pytest.approx(air.sum())
    # Block the primary 5GHz domain but allow its strong 2GHz fallback.
    chain=(np.array([[1,0,0]]),mcs,np.full((1,3),100.),np.array([False]),None,available,enabled,domains)
    ok,air,_,robot_air=_managed_chain_allocation(curves,[chain],np.array([1.,0.,1.]))
    assert ok[0]>0 and air[0]>0 and air[1]==0
    assert robot_air.sum()==pytest.approx(air.sum())


def test_checkpoint_preserves_pending_reports_and_packet_accounting():
    w=world();box=ProbeMailbox();rt=ProbeRuntime.from_world(w,spec(w,starts=(44,104),report_delay_slots=4),box)
    p=Policy('replay','native');checkpoints=[]
    full=simulate(w,p,managed_probes=rt,detailed=True,checkpoint_at={50},checkpoint_sink=checkpoints.append)
    replay=simulate(w,p,resume=checkpoints[0],stop_slot=200,detailed=True)
    for key in ('control_misses','control_sent','bulk_mbps','video_on_time','control_gap_p99_ms','managed_probe_summary','managed_packet_accounting'):
        assert replay[key]==full[key]
    assert replay['packet_detail']['misses']==full['packet_detail']['misses']
    assert replay['packet_detail']['sent']==full['packet_detail']['sent']
