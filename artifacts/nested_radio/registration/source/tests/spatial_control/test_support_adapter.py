"""Analytic adapter fixtures and actual sensor-RNG parity, no fresh worlds."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wirelessjev.fleetwm.data import Observations
from wirelessjev.jepa_repair.telemetry import TimedDeliveredScans
from wirelessjev.native.gate import RoamDecision
from wirelessjev.native.world import HORIZONS
from wirelessjev.spatial_control.memory import NATIVE_GAUGE, RFReport, SpatialReportMemory
from wirelessjev.spatial_control.runtime import (
    ClockedScans, SupportAdmission, SupportedAdaptivePredictor,
)


def warm(n=2, sites=2):
    shape=(60,n,sites*3)
    return Observations(np.zeros((60,n,2),dtype="f4"),np.full(shape,-75.,dtype="f4"),
        np.ones(shape,dtype=bool),np.full(shape,-90.,dtype="f4"),np.ones(shape[:2],dtype=bool))


def geometry(sites=2):
    return SimpleNamespace(sites=np.column_stack((np.arange(sites)*10.+10.,np.zeros(sites))),
        tx=np.array([20.,20.,20.]),noise=np.array([-94.,-87.,-84.]),yard=False)


def rf(identity, *, robot=0, site=1, band=1, measured=-1., value=-70.,
       kind="heard", position=(0.,0.,1.)):
    return RFReport(identity,robot,site,band,(2.4e9,5.2e9,6.5e9)[band],position,kind,
        NATIVE_GAUGE,0.,(measured,measured),value_dbm=value if kind=="heard" else None,
        upper_dbm=-89. if kind=="censored" else None)


class FixtureUtility:
    def __init__(self,allowed): self.allowed=np.asarray(allowed,dtype=bool);self.records=[]
    def __call__(self,decision): return self.allowed
    def scores(self,forecast,current): return np.ones((len(current),2))


def guard(rows, *, age=0., mask=(True,), movement=0., issue=0):
    memory=SpatialReportMemory()
    for row in rows: memory.ingest(row,at_s=float(issue))
    n=len(mask)
    plan=np.zeros((n,len(HORIZONS),2));plan[:,:,0]=np.asarray(HORIZONS)*movement
    raw=SimpleNamespace(memories={issue:memory},records=[{"plan":plan} for _ in range(issue+1)],geometry=geometry())
    queue=SimpleNamespace(last=issue,age=age)
    utility=FixtureUtility(mask)
    adapter=SupportAdmission(utility,raw,queue,SimpleNamespace(events=[]))
    decision=RoamDecision.capture(issue,np.zeros(n,dtype=int),np.ones(n,dtype=int),
        np.ones(n),np.ones(n,dtype=bool),np.ones((n,2)))
    return adapter,decision,utility


def test_recent_nearby_actual_source_heard_enables_only_already_permitted_move():
    adapter,decision,utility=guard([rf("own")])
    assert adapter(decision).tolist()==[True]
    row=adapter.records[-1]
    assert row["heard_support_bands"]==[1]
    assert not row["packet_risk_qualified"]
    assert "unchanged 64 encoder tokens" in row["support_information"]
    assert row["own_latest_source_reports"][0]["conservative_age_s"]==1.
    assert row["own_latest_source_reports"][0]["query_distance_m"]==0.
    assert utility.allowed.tolist()==[True]  # guard never mutates base permission
    blocked,decision,_=guard([rf("own")],mask=(False,))
    assert not blocked(decision).any()


@pytest.mark.parametrize("row", [rf("only6",band=2),rf("other-robot",robot=1),
    rf("other-ap",site=0),rf("old",measured=-6.01),rf("far",position=(5.01,0.,1.)),
    rf("weak",value=-80.01),rf("negative",kind="censored")])
def test_no_implicit_source_band_device_age_distance_or_censor_support(row):
    adapter,decision,_=guard([row])
    assert not adapter(decision).any()
    assert not adapter.records[-1]["packet_risk_qualified"]


def test_exact_cutoff_is_inclusive():
    adapter,decision,_=guard([rf("edge",measured=-6.,value=-80.,position=(5.,0.,1.))])
    assert adapter(decision).tolist()==[True]


def test_newest_own_censor_blocks_older_heard_even_if_censor_is_farther_away():
    adapter,decision,_=guard([rf("old-heard",measured=-2.),
        rf("new-censor",kind="censored",measured=-1.,position=(20.,0.,1.))])
    assert not adapter(decision).any()
    rows=adapter.records[-1]["own_latest_source_reports"]
    assert rows[0]["report_id"]=="new-censor" and rows[0]["kind"]=="censored"
    assert not rows[0]["within_spatial_cut"]


def test_another_actual_source_band_can_support_ap_despite_censor_on_first_band():
    adapter,decision,_=guard([rf("band0negative",band=0,kind="censored"),rf("band1heard",band=1)])
    assert adapter(decision).tolist()==[True]
    assert adapter.records[-1]["heard_support_bands"]==[1]


def test_completion_age_changes_both_observation_age_and_public_query_position():
    adapter,decision,_=guard([rf("own",measured=-1.)],age=5.,movement=1.)
    assert adapter(decision).tolist()==[True]
    row=adapter.records[-1]
    assert row["time_s"]==5. and row["position_m"]==[5.,0.,1.]
    assert row["own_latest_source_reports"][0]["conservative_age_s"]==6.
    adapter.queue.age=5.01
    assert not adapter(decision).any()


def test_guard_does_not_read_unbound_sensor_metadata_as_rf():
    adapter,decision,_=guard([])
    adapter.sensor.events.append({"would_be_new_delivery":True})
    with pytest.raises(ValueError,match="Incomplete server delivery"):
        adapter(decision)


@pytest.mark.parametrize("field,value",[("last",1),("age",-1.),("age",float("nan"))])
def test_invalid_consumed_forecast_clock_rejected(field,value):
    adapter,decision,_=guard([rf("own")]);setattr(adapter.queue,field,value)
    with pytest.raises(ValueError): adapter(decision)


def test_late_guard_denies_and_records_actual_denial(monkeypatch):
    adapter,decision,_=guard([rf("own")])
    clock=iter([0.,.101]);monkeypatch.setattr("wirelessjev.spatial_control.runtime.time.perf_counter",lambda:next(clock))
    assert not adapter(decision).any()
    assert adapter.records[-1]["late_discarded"] and not adapter.records[-1]["permitted"]
    assert adapter.utility.records[-1]["late_discarded"]


def test_guard_scores_delegate_without_changing_base_scores():
    adapter,_,_=guard([])
    assert adapter.scores(None,np.array([0])).tolist()==[[1.,1.]]


class FixtureModel:
    config={"cell":"regression"}
    def __init__(self): self.contexts=[]
    def eval(self): return self
    def encode(self,context,known,scheduled):
        self.contexts.append(context.clone())
        return {"mean":torch.zeros((1,8)),"covariance":torch.eye(8)[None]}
    def render(self,belief,query,basis,fspl,offset):
        return fspl,torch.ones_like(fspl)


def test_same_64_encoder_tokens_with_extra_support_index_and_complete_elapsed():
    past=warm();sensor=SimpleNamespace(events=[]);g=geometry()
    plan=np.zeros((2,len(HORIZONS),2));model=FixtureModel()
    predictor=SupportedAdaptivePredictor(g,model,"a"*64,sensor,support=True)
    forecast,kind,elapsed=predictor.step(past,plan)
    assert kind=="cold" and elapsed>=predictor.support_preparation[0]["elapsed_seconds"]>=0.
    assert forecast.mean_dbm.shape==(2,len(HORIZONS),2,3)
    assert len(predictor.records[0]["context"])==64
    assert model.contexts[0].shape==(1,64,20)
    assert predictor.support_preparation[0]["encoder_context_changed"] is False
    assert predictor.support_preparation[0]["snapshot_metadata"]["retention_age_s"]==6.
    assert predictor.records[0]["decision"]==0
    assert len(predictor.memories[0])==6*2*2*2
    assert predictor.last_history["value"] is not past.value
    assert np.array_equal(predictor.last_history["value"],past.value)
    baseline=SupportedAdaptivePredictor(g,FixtureModel(),"a"*64,sensor,support=False)
    other,_,_=baseline.step(past,plan)
    assert np.array_equal(forecast.mean_dbm,other.mean_dbm)
    assert np.array_equal(predictor.records[0]["context"],baseline.records[0]["context"])
    assert not baseline.memories


def test_fifo_predictor_rejects_missing_or_repeated_history_issue():
    predictor=SupportedAdaptivePredictor(geometry(),FixtureModel(),"a"*64,SimpleNamespace(events=[]),support=True)
    plan=np.zeros((2,len(HORIZONS),2));past=warm()
    predictor.step(past,plan)
    with pytest.raises(ValueError,match="FIFO"): predictor.step(past,plan)


def test_support_predictor_rejects_future_event_ledger_before_forward():
    past=warm();model=FixtureModel()
    event=dict(robot=0,measurement_slot=1,complete_slot=2,delivery_slot=3,payload_sha256="a"*64)
    predictor=SupportedAdaptivePredictor(geometry(),model,"a"*64,SimpleNamespace(events=[event]),support=True)
    with pytest.raises(ValueError,match="Future"):
        predictor.step(past,np.zeros((2,len(HORIZONS),2)))
    assert not model.contexts and not predictor.records


def assert_sensor_equal(a,b):
    for name in ("positions","value","heard","bound","scanned"):
        assert np.array_equal(getattr(a.history,name),getattr(b.history,name))
    for name in ("count","local","local_capture_slots","local_capture_heard"):
        assert np.array_equal(getattr(a,name),getattr(b,name))
    assert a.events==b.events and a.local_completion_events==b.local_completion_events
    assert a.missing==b.missing and a.replaced==b.replaced
    assert set(a.pending)==set(b.pending) and set(a.local_pending)==set(b.local_pending)
    for cache in ("pending","local_pending"):
        for key,x in getattr(a,cache).items():
            y=getattr(b,cache)[key]
            assert x[:2]==y[:2] and x[-1]==y[-1]
            for first,second in zip(x[2:5],y[2:5]): assert np.array_equal(first,second)


def test_clocked_sensor_archives_without_changing_actual_rng_or_delivery_behavior():
    original=TimedDeliveredScans(warm(sites=4),5,211)
    clocked=ClockedScans(warm(sites=4),5,211)
    ids=np.arange(2);positions=np.array([[1.,2.],[3.,4.]])
    rssi=np.broadcast_to(np.linspace(-98.,-48.,12).reshape(4,3),(2,4,3)).copy()
    for slot in range(0,400,10):
        complete=np.full(2,slot+9,dtype=int)
        for sensor in (original,clocked):
            sensor.complete(slot)
            sensor.scan(slot,ids,positions,rssi,complete)
            sensor.complete(slot+9)
            sensor.deliver(slot+9,np.ones(2,dtype=bool))
        assert_sensor_equal(original,clocked)
    assert clocked.missing>0
    assert len(clocked.local_completed_payloads)==len(clocked.local_completion_events)
    assert len(clocked.started_events)==80
    assert sum(not e["measurement_available"] for e in clocked.started_events)==clocked.missing
    for row in clocked.local_completed_payloads:
        assert row["measurement_slot"]<row["complete_slot"]<=row["processed_slot"]
        assert row["reference_id"]==NATIVE_GAUGE
        assert len(row["value_dbm"])==len(row["heard"])==12
        assert any(not x for x in row["heard"])
        assert row["upper_dbm"]>=-90.


def test_clocked_missing_draw_cannot_be_labeled_new_measurement_from_old_pending():
    sensor=ClockedScans(warm(n=1,sites=4),5,211)
    ids=np.array([0]);positions=np.zeros((1,2));rssi=np.full((1,4,3),-70.)
    # Real deterministic draws; keep old local_pending uncompleted to exercise
    # an actual missing draw while a previous measurement is still pending.
    found=False
    for slot in range(40):
        sensor.scan(slot,ids,positions,rssi,np.array([1000]))
        if sensor.missing:
            found=True
            assert not sensor.started_events[-1]["measurement_available"]
            assert 0 in sensor.local_pending
            assert sensor.local_pending[0][0]<slot
            break
    assert found


def test_invalid_clocked_scan_rejects_before_rng_or_counter_mutation():
    sensor=ClockedScans(warm(),5,211)
    with pytest.raises(ValueError):
        sensor.scan(0,np.array([0,0]),np.zeros((2,2)),np.zeros((2,2,3)),np.array([9,9]))
    assert sensor.count.sum()==0 and not sensor.started_events


def test_second_late_call_does_not_rewrite_first_call_admission_history(monkeypatch):
    adapter,decision,_=guard([rf("own")])
    assert adapter(decision).all()
    clock=iter([0.,.101]);monkeypatch.setattr("wirelessjev.spatial_control.runtime.time.perf_counter",lambda:next(clock))
    assert not adapter(decision).any()
    assert adapter.records[0]["permitted"] and not adapter.records[0]["late_discarded"]
    assert not adapter.records[1]["permitted"] and adapter.records[1]["late_discarded"]


def refresh_pair(*, measured=20, complete=29, processed=29, delivered=35,
                 kind="censored", band=1, position=(0.,0.,1.)):
    import hashlib
    values=np.zeros(6,dtype="f4");heard=np.zeros(6,dtype=bool)
    if kind=="heard":heard[3+band]=True;values[3+band]=-70.
    sha=hashlib.sha256(values.tobytes()+heard.tobytes()).hexdigest()
    payload=dict(robot=0,measurement_slot=measured,complete_slot=complete,
        processed_slot=processed,position_m=list(position),value_dbm=values.tolist(),
        heard=heard.tolist(),upper_dbm=-89.,reference_id=NATIVE_GAUGE,
        scan_id=f"scan-0-{measured}",payload_sha256=sha)
    event=dict(robot=0,measurement_slot=measured,complete_slot=complete,
        delivery_slot=delivered,first_usable_decision=delivered//100+1,
        heard=int(heard.sum()),payload_sha256=sha)
    return payload,event


def test_queued_new_server_delivered_censor_vetoes_issued_positive():
    adapter,decision,_=guard([rf("issued-positive",measured=-1.)],age=1.5)
    payload,event=refresh_pair()
    adapter.sensor.local_completed_payloads=[payload];adapter.sensor.events=[event]
    assert not adapter(decision).any()
    own=adapter.records[-1]["own_latest_source_reports"]
    assert all(row["kind"]=="censored" for row in own)
    assert own[0]["measured_interval_s"]==(.2,.2)
    batch=adapter.refresh_records[-1]
    assert batch["delivery_events_added"]==1 and batch["new_reports"]==4
    assert batch["bindings"][0]["delivery_event"]==event
    assert all(row["band_index"] in (0,1) for row in batch["bindings"][0]["reports"])
    assert not batch["locally_completed_undelivered_reports_used"]
    assert not batch["encoder_context_changed"] and not batch["packet_risk_qualified"]


def test_locally_completed_undelivered_censor_cannot_reach_server_guard():
    adapter,decision,_=guard([rf("issued-positive")],age=1.5)
    payload,_=refresh_pair()
    adapter.sensor.local_completed_payloads=[payload]
    assert adapter(decision).all()
    assert adapter.records[-1]["own_latest_source_reports"][0]["report_id"]=="issued-positive"
    assert adapter.refresh_records[-1]["delivery_events_added"]==0


def test_future_server_delivery_is_deferred_then_becomes_available():
    adapter,decision,_=guard([rf("issued-positive")],age=1.5)
    payload,event=refresh_pair(delivered=200)
    adapter.sensor.local_completed_payloads=[payload];adapter.sensor.events=[event]
    assert adapter(decision).all()
    assert adapter.refresh_records[-1]["future_delivery_events_deferred"]==1
    adapter.queue.age=2.
    assert not adapter(decision).any()
    assert adapter.refresh_records[-1]["delivery_events_added"]==1


def test_refresh_does_not_repeat_ingestion_on_second_same_issue_admission():
    adapter,decision,_=guard([],age=1.5)
    payload,event=refresh_pair(kind="heard")
    adapter.sensor.local_completed_payloads=[payload];adapter.sensor.events=[event]
    assert adapter(decision).all()
    count=len(adapter.raw.memories[0])
    assert adapter(decision).all() and len(adapter.raw.memories[0])==count
    assert adapter.refresh_records[-1]["delivery_events_added"]==0


@pytest.mark.parametrize("change", ["payload_hash","missing_payload","measurement",
    "completion_after_delivery","processing_after_delivery","reference","heard_count","position"])
def test_refresh_rejects_unbound_payload_or_inconsistent_clocks(change):
    adapter,decision,_=guard([rf("issued-positive")],age=1.5)
    payload,event=refresh_pair()
    if change=="payload_hash":payload["value_dbm"][4]=-12.
    elif change=="measurement":event["measurement_slot"]=21
    elif change=="completion_after_delivery":event["complete_slot"]=36
    elif change=="processing_after_delivery":payload["processed_slot"]=36
    elif change=="reference":payload["reference_id"]="other_device_gauge"
    elif change=="heard_count":event["heard"]=1
    elif change=="position":payload["position_m"]=[float("nan"),0.,1.]
    adapter.sensor.local_completed_payloads=[] if change=="missing_payload" else [payload]
    adapter.sensor.events=[event]
    with pytest.raises(ValueError):adapter(decision)


def test_refresh_does_not_override_newer_measurement_with_older_late_delivery():
    # A late-delivered scan measured at0 cannot be older than a valid issue0
    # timestamp. Use a direct issue1 fixture to preserve the causal inequality.
    adapter,decision,_=guard([RFReport("newer-positive",0,1,1,5.2e9,(0.,0.,1.),"heard",
        NATIVE_GAUGE,1.,(.9,.9),value_dbm=-70.)],age=.5,issue=1)
    payload,event=refresh_pair(measured=20,complete=29,delivered=120)
    adapter.sensor.local_completed_payloads=[payload];adapter.sensor.events=[event]
    assert adapter(decision).all()
    row=next(r for r in adapter.records[-1]["own_latest_source_reports"] if r["band_index"]==1)
    assert row["report_id"]=="newer-positive"


def test_undelivered_local_rf_fields_are_not_read_even_for_index_preparation():
    class UnavailableRF(dict):
        def __getitem__(self,key):
            if key in {"value_dbm","heard","position_m","upper_dbm","reference_id"}:
                raise AssertionError("Undelivered client-local RF was read by server guard")
            return super().__getitem__(key)
    adapter,decision,_=guard([rf("issued-positive")],age=1.5)
    payload,_=refresh_pair()
    adapter.sensor.local_completed_payloads=[UnavailableRF(payload)]
    assert adapter(decision).all()
    assert adapter.refresh_records[-1]["new_reports"]==0
