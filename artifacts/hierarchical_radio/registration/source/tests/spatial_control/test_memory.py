"""Software fixtures for causal support; no model, RF or hardware evidence."""
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from wirelessjev.spatial_control.memory import (
    NATIVE_CARRIERS_HZ, NATIVE_GAUGE, RFReport, SpatialReportMemory,
    inspect_native_support, native_history_memory,
)


def report(identity="one", **kwargs):
    data = dict(report_id=identity, robot_id=7, site_id=3, band_index=1,
        carrier_hz=5.2e9, position_m=(1., 2., 1.), kind="heard", gauge=NATIVE_GAUGE,
        available_at_s=1., measured_interval_s=(.5, .5), arrival_interval_s=(1., 1.), value_dbm=-70.)
    data.update(kwargs)
    return RFReport(**data)


def query(memory, **kwargs):
    data = dict(robot_id=7, site_id=3, band_index=1, carrier_hz=5.2e9,
        position_m=(1., 2., 1.), issue_time_s=2., max_age_s=6., max_distance_m=5.)
    data.update(kwargs)
    return memory.query(**data)


def history(*, length=8, n=2, sites=1):
    return SimpleNamespace(positions=np.zeros((length, n, 2), dtype="f4"),
        value=np.zeros((length, n, sites*3), dtype="f4"),
        heard=np.zeros((length, n, sites*3), dtype=bool),
        bound=np.full((length, n, sites*3), -90., dtype="f4"),
        scanned=np.zeros((length, n), dtype=bool))


def event(past, *, t=7, robot=0, measured=35, complete=44, delivered=110):
    payload = hashlib.sha256(past.value[t, robot].tobytes()+past.heard[t, robot].tobytes()).hexdigest()
    return dict(robot=robot, measurement_slot=measured, complete_slot=complete,
        delivery_slot=delivered, first_usable_decision=delivered//100+1,
        heard=int(past.heard[t, robot].sum()), payload_sha256=payload)


def test_exact_point_censor_interval_and_missing_are_distinct():
    rows = [report(), report("bound", kind="censored", value_dbm=None, upper_dbm=-88.),
        report("quantized", kind="interval", value_dbm=None, lower_dbm=-71., upper_dbm=-69.),
        report("missing", kind="missing", value_dbm=None)]
    memory=SpatialReportMemory()
    for row in rows: memory.ingest(row, at_s=2.)
    result=query(memory)
    assert result.candidate_count==4 and result.eligible_count==3 and result.missing_count==1
    assert {r.kind for r in result.reports}=={"heard","censored","interval"}
    assert not result.risk_qualified
    assert next(r for r in result.reports if r.kind=="censored").value_dbm is None


@pytest.mark.parametrize("changes", [
    {"kind":"censored"}, {"kind":"heard","upper_dbm":-80.},
    {"kind":"missing"}, {"kind":"interval","value_dbm":None,"lower_dbm":-69.,"upper_dbm":-71.},
    {"kind":"bogus"}, {"value_dbm":float("nan")}, {"carrier_hz":0.},
    {"position_m":(0.,0.)}, {"position_m":(0.,0.,float("inf"))},
    {"robot_id":True}, {"site_id":-1}, {"band_index":1.5}, {"gauge":""},
    {"position_error_m":-1.}, {"measured_interval_s":(2.,1.)},
    {"measured_interval_s":(1.1,1.1)}, {"arrival_interval_s":(.4,.7)},
    {"arrival_interval_s":(1.,1.1)}, {"source_row":(0,0,0)},
])
def test_invalid_report_semantics_or_physics_fail(changes):
    with pytest.raises(ValueError): report(**changes)


def test_duplicate_idempotence_and_conflicts():
    memory=SpatialReportMemory(maximum_reports=1)
    assert memory.ingest(report(), at_s=1.)
    assert not memory.ingest(report(), at_s=1.)
    with pytest.raises(ValueError,match="Conflicting"): memory.ingest(report(value_dbm=-71.),at_s=1.)
    with pytest.raises(ValueError,match="capacity"): memory.ingest(report("two"),at_s=1.)
    assert len(memory)==1


def test_future_ingest_and_query_visibility_are_separate():
    memory=SpatialReportMemory()
    with pytest.raises(ValueError,match="future"): memory.ingest(report(),at_s=.9)
    memory.ingest(report(),at_s=1.)
    before=query(memory,issue_time_s=.9)
    assert before.unavailable_count==1 and not before.reports
    assert query(memory,issue_time_s=1.).latest_own==report()


def test_latest_own_censor_reserved_before_other_heard_and_older_own():
    memory=SpatialReportMemory()
    rows=[report("old",measured_interval_s=(0.,0.)),
        report("neighbor",robot_id=0,measured_interval_s=(.9,.9)),
        report("own-new-negative",kind="censored",value_dbm=None,upper_dbm=-89.,measured_interval_s=(1.,1.))]
    for row in rows: memory.ingest(row,at_s=2.)
    result=query(memory,limit=1)
    assert result.latest_own.report_id=="own-new-negative"
    assert result.latest_own.kind=="censored"


def test_own_priority_does_not_admit_stale_or_distant_own_rows():
    memory=SpatialReportMemory()
    for row in (report("stale",measured_interval_s=(-10.,-10.)),
                report("distant",position_m=(30.,2.,1.)),
                report("near-fresh",robot_id=2)):
        memory.ingest(row,at_s=2.)
    result=query(memory)
    assert result.latest_own is None
    assert [r.report_id for r in result.reports]==["near-fresh"]


def test_conservative_age_and_position_uncertainty_at_boundaries():
    memory=SpatialReportMemory()
    memory.ingest(report(measured_interval_s=(-4.,-3.),arrival_interval_s=None,
        position_m=(4.,2.,1.),position_error_m=1.),at_s=2.)
    result=query(memory,max_age_s=6.,max_distance_m=5.,position_error_m=1.)
    assert result.ages_s==(6.,) and result.distances_m==(5.,)
    assert not query(memory,max_age_s=5.99).reports
    assert not query(memory,max_distance_m=4.99,position_error_m=1.).reports


def test_unknown_age_cannot_pass_even_with_recent_arrival():
    memory=SpatialReportMemory()
    memory.ingest(report(measured_interval_s=None),at_s=2.)
    result=query(memory,max_age_s=1000.)
    assert result.unknown_age_count==1 and not result.reports


def test_gauge_carrier_and_link_never_mix_implicitly():
    memory=SpatialReportMemory()
    memory.ingest(report(),at_s=2.)
    result=query(memory,gauge="other_frontend")
    assert result.gauge_mismatch_count==1 and not result.reports
    assert not query(memory,carrier_hz=6.5e9).reports
    assert not query(memory,band_index=0).reports
    assert not query(memory,site_id=4).reports


def test_retrieval_is_deterministic_across_ingestion_order_and_cache_updates():
    rows=[report(str(i),robot_id=i,position_m=(float(i),2.,1.)) for i in range(5)]
    a,b=SpatialReportMemory(),SpatialReportMemory()
    for row in rows: a.ingest(row,at_s=2.)
    for row in reversed(rows): b.ingest(row,at_s=2.)
    assert query(a)==query(b)
    a.query(**dict(robot_id=7,site_id=3,band_index=1,carrier_hz=5.2e9,
        position_m=(1.,2.,1.),issue_time_s=2.,max_age_s=6.,max_distance_m=5.))
    a.ingest(report("own"),at_s=2.)
    assert query(a).latest_own.report_id=="own"


@pytest.mark.parametrize("changes", [{"limit":0},{"limit":True},{"limit":4097},
    {"max_age_s":float("nan")},{"max_distance_m":-1.},{"position_error_m":-1.},
    {"issue_time_s":True},{"robot_id":-1},{"gauge":""}])
def test_invalid_query_contract_fails(changes):
    with pytest.raises(ValueError): query(SpatialReportMemory(),**changes)


def test_native_warm_rows_preserve_bins_and_actual_missingness():
    past=history(length=6)
    past.scanned[1,0]=True
    past.heard[1,0,1]=True;past.value[1,0,1]=-73.
    past.value[0]=np.nan;past.bound[0]=np.nan  # unmeasured, never converted into evidence
    before=[getattr(past,k).copy() for k in vars(past)]
    memory=native_history_memory(past,issue_time_s=0.,warm_seconds=6)
    assert len(memory)==2
    for band in (0,1):
        result=inspect_native_support(memory,robot_id=0,site_id=0,band_index=band,
            position_m=(0.,0.,1.),issue_time_s=0.)
        assert result.latest_own.measured_interval_s==(-5.,-4.)
        assert result.latest_own.arrival_interval_s is None
        assert result.ages_s==(5.,)
        assert "bin" in result.latest_own.clock_provenance
    assert result.latest_own.kind=="heard" and result.latest_own.value_dbm==-73.
    for name,old in zip(vars(past),before): assert np.array_equal(getattr(past,name),old,equal_nan=True)


def test_native_delivery_bin_alone_does_not_establish_measurement_age():
    past=history();past.scanned[7,0]=True
    memory=native_history_memory(past,issue_time_s=2.,warm_seconds=6)
    result=inspect_native_support(memory,robot_id=0,site_id=0,band_index=1,
        position_m=(0.,0.,1.),issue_time_s=2.)
    assert result.unknown_age_count==1 and not result.reports


def test_native_exact_events_reconcile_measurement_not_delivery_age():
    past=history();past.scanned[7,0]=True
    past.heard[7,0,1]=True;past.value[7,0,1]=-76.
    memory=native_history_memory(past,issue_time_s=2.,events=[event(past)],warm_seconds=6)
    result=inspect_native_support(memory,robot_id=0,site_id=0,band_index=1,
        position_m=(0.,0.,1.),issue_time_s=2.)
    assert result.ages_s==pytest.approx((1.65,))
    assert result.latest_own.measured_interval_s==(.35,.35)
    assert result.latest_own.arrival_interval_s==(1.1,1.1)
    assert result.latest_own.source_row==(7,0,0,1)
    assert result.latest_own.value_dbm==-76.


def test_latest_of_multiple_same_bin_deliveries_must_match_payload():
    past=history();past.scanned[7,0]=True
    actual=event(past,delivered=160)
    old=event(past,delivered=110);old["payload_sha256"]="0"*64
    memory=native_history_memory(past,issue_time_s=2.,events=[actual,old],warm_seconds=6)
    result=inspect_native_support(memory,robot_id=0,site_id=0,band_index=1,
        position_m=(0.,0.,1.),issue_time_s=2.)
    assert result.latest_own.arrival_interval_s==(1.6,1.6)
    actual["payload_sha256"]="0"*64
    with pytest.raises(ValueError,match="payload"): native_history_memory(past,issue_time_s=2.,events=[actual],warm_seconds=6)


@pytest.mark.parametrize("field,value", [("measurement_slot",110),("complete_slot",111),
    ("delivery_slot",201),("robot",2),("first_usable_decision",3),("payload_sha256","bad")])
def test_native_invalid_or_future_event_fails(field,value):
    past=history();past.scanned[7,0]=True
    row=event(past);row[field]=value
    with pytest.raises(ValueError): native_history_memory(past,issue_time_s=2.,events=[row],warm_seconds=6)


def test_future_history_bin_rejected_even_if_no_scan_in_it():
    with pytest.raises(ValueError,match="strictly before"):
        native_history_memory(history(length=9),issue_time_s=2.,warm_seconds=6)


def test_native_does_not_encode_unrequested_six_point_five_band():
    past=history(length=6);past.scanned[5,0]=True;past.heard[5,0,2]=True;past.value[5,0,2]=-60.
    memory=native_history_memory(past,issue_time_s=0.,warm_seconds=6)
    result=inspect_native_support(memory,robot_id=0,site_id=0,band_index=2,
        position_m=(0.,0.,1.),issue_time_s=0.)
    assert not result.reports
    assert all(r.carrier_hz in NATIVE_CARRIERS_HZ[:2] for rows in memory._groups.values() for r in rows)


def test_bad_native_masks_and_nonfinite_known_bounds_rejected():
    past=history(length=6);past.heard[0,0,0]=True
    with pytest.raises(ValueError): native_history_memory(past,issue_time_s=0.,warm_seconds=6)
    past.heard[:]=False;past.scanned[0,0]=True;past.bound[0,0,0]=np.nan
    with pytest.raises(ValueError): native_history_memory(past,issue_time_s=0.,warm_seconds=6)


def test_retention_keeps_identical_recent_support_and_counts_old_rows_separately():
    past=history()
    past.scanned[:,0]=True
    past.heard[:,0,1]=True
    past.value[:,0,1]=np.arange(8)-80.
    events=[event(past,t=6,measured=10,complete=20,delivered=40),
            event(past,t=7,measured=135,complete=144,delivered=160)]
    full=native_history_memory(past,issue_time_s=2.,events=events,warm_seconds=6)
    recent=native_history_memory(past,issue_time_s=2.,events=events,warm_seconds=6,retention_age_s=2.)
    kwargs=dict(robot_id=0,site_id=0,band_index=1,position_m=(0.,0.,1.),
        issue_time_s=2.,max_age_s=2.,limit=8)
    a,b=inspect_native_support(full,**kwargs),inspect_native_support(recent,**kwargs)
    assert a.reports==b.reports and a.ages_s==b.ages_s and a.distances_m==b.distances_m
    assert a.eligible_count==b.eligible_count==2
    assert recent.snapshot_metadata["observed_scan_rows"]==8
    assert recent.snapshot_metadata["retained_scan_rows"]==2
    assert recent.snapshot_metadata["retention_excluded_old_scan_rows"]==6
    assert recent.snapshot_metadata["retention_excluded_unknown_clock_scan_rows"]==0
    assert b.missing_count==0 and len(recent)==4


def test_retention_explicitly_counts_unknown_measurement_clock_exclusion():
    past=history();past.scanned[7,0]=True
    memory=native_history_memory(past,issue_time_s=2.,warm_seconds=6,retention_age_s=6.)
    assert len(memory)==0
    assert memory.snapshot_metadata["retention_excluded_unknown_clock_scan_rows"]==1
    assert memory.snapshot_metadata["retention_excluded_old_scan_rows"]==0
    assert memory.snapshot_metadata["discarded_rows_are_not_missing_observations"]


@pytest.mark.parametrize("age", [-1.,float("inf"),True])
def test_invalid_retention_rejected(age):
    with pytest.raises(ValueError):
        native_history_memory(history(length=6),issue_time_s=0.,warm_seconds=6,retention_age_s=age)


def test_delivered_event_absent_from_history_is_rejected():
    past=history()
    with pytest.raises(ValueError,match="absent"):
        native_history_memory(past,issue_time_s=2.,events=[event(past)],warm_seconds=6)


def test_retained_snapshot_rejects_wider_age_or_earlier_time_scope():
    past=history(length=6);past.scanned[5,0]=True
    memory=native_history_memory(past,issue_time_s=2.,warm_seconds=6,retention_age_s=6.)
    kwargs=dict(robot_id=0,site_id=0,band_index=1,position_m=(0.,0.,1.),issue_time_s=2.)
    with pytest.raises(ValueError,match="scope"):
        inspect_native_support(memory,**dict(kwargs,max_age_s=7.))
    with pytest.raises(ValueError,match="scope"):
        inspect_native_support(memory,**dict(kwargs,issue_time_s=1.))


def test_native_event_summary_cannot_contradict_payload_mask():
    past=history();past.scanned[7,0]=True
    row=event(past);row["heard"]=1
    with pytest.raises(ValueError,match="heard count"):
        native_history_memory(past,issue_time_s=2.,events=[row],warm_seconds=6)
