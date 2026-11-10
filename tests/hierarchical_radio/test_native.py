"""Native-adapter software fixtures, not new scientific worlds or model fits."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wirelessjev.fleetwm.data import Observations
from wirelessjev.native.gate import RoamDecision
from wirelessjev.native.world import HORIZONS
from wirelessjev.spatial_control.memory import RFReport, SpatialReportMemory, NATIVE_GAUGE
from wirelessjev.hierarchical_radio.native import (
    MAX_CONTEXT, MODEL_ORIGIN_M, NativeFieldPredictor, MatchedSupportAdmission,
    build_predictor, make_query, select_context,
)


def geometry(sites=2):
    return SimpleNamespace(sites=np.column_stack((np.arange(sites)*6.+10.,np.full(sites,10.))),
        tx=np.array([20.,20.,20.]), noise=np.array([-94.,-87.,-84.]), yard=False)


def warm(robots=4, sites=2):
    shape=(60,robots,sites*3)
    positions=np.zeros((60,robots,2),dtype='f4')
    positions[:,:,0]=np.linspace(-30.,60.,robots)
    scanned=np.zeros(shape[:2],bool);scanned[[50,55]]=True
    heard=np.broadcast_to(scanned[...,None],shape).copy()
    values=np.where(heard,-70.,0.).astype('f4')
    return Observations(positions,values,heard,np.full(shape,-90.,dtype='f4'),scanned)


def route(robots=4, point=0.):
    plan=np.zeros((robots,len(HORIZONS),2));plan[:,:,0]=point
    return plan


def postwarm():
    old=warm();n=old.value.shape[1];links=old.value.shape[2]
    def append(a):return np.concatenate((a,np.zeros((1,*a.shape[1:]),dtype=a.dtype)))
    past=Observations(*(append(getattr(old,name)) for name in ('positions','value','heard','bound','scanned')))
    values=np.full(links,-68.,dtype='f4');heard=np.ones(links,bool)
    heard[-2]=False;values[-2]=0.
    past.scanned[60,0]=True;past.positions[60,0]=[0.,0.]
    past.value[60,0]=values;past.heard[60,0]=heard;past.bound[60,0]=-75.
    digest=hashlib.sha256(values.tobytes()+heard.tobytes()).hexdigest()
    event={'robot':0,'measurement_slot':20,'complete_slot':30,'delivery_slot':50,
        'first_usable_decision':1,'heard':int(heard.sum()),'payload_sha256':digest}
    payload={'robot':0,'measurement_slot':20,'complete_slot':30,'processed_slot':30,
        'value_dbm':values.tolist(),'heard':heard.tolist(),'position_m':[0.,0.,1.],
        'upper_dbm':-75.,'reference_id':NATIVE_GAUGE,'scan_id':'fixture-scan0',
        'payload_sha256':digest}
    return past,event,payload


def test_real_rows_have_balanced_source_groups_and_explicit_warm_clock_scope():
    result=select_context(warm(),geometry(),route(),issue_time_s=0.,maximum=12)
    view=result['view'];meta=result['metadata']
    assert view['context'].shape==(12,20)
    assert meta['selected_scan_rows'].shape==(3,2)
    assert len(set(map(tuple,meta['selected_scan_rows'])))==3
    assert all(np.sum((meta['selected_rows'][:,2]==site)&(meta['selected_rows'][:,3]==band))==3
        for site in range(2) for band in (0,1))
    assert np.isnan(meta['arrival_intervals_s']).all()
    assert np.all(meta['clock_provenance']==0)
    assert np.all(view['lower']==view['upper'])  # genuine continuous heard reports
    assert np.allclose(view['lower'],view['context'][:,0]*30,atol=1.e-5)
    assert meta['selection_uses_rssi_amplitudes_or_heard_preference'] is False
    assert np.all(meta['conservative_age_s']>=5)


def test_actual_six_scan_budget_yields_192_rows_for_16_public_aps():
    result=select_context(warm(robots=8,sites=16),geometry(16),route(8),issue_time_s=0.)
    assert result['view']['context'].shape==(MAX_CONTEXT,20)
    assert result['metadata']['selected_scans']==6
    assert result['metadata']['tokens_per_ap_source_band']==6
    assert result['ap_xyz'].shape==(16,3)


def test_context_selection_is_deterministic_and_never_reads_rssi_for_ranking():
    past=warm();first=select_context(past,geometry(),route(),issue_time_s=0.,maximum=12)
    altered=Observations(past.positions.copy(),past.value.copy(),past.heard.copy(),past.bound.copy(),past.scanned.copy())
    altered.value[altered.heard]=-86.
    # Heard/censor preference must not hide a negative report at the same pose.
    altered.heard[55,0,4]=False;altered.value[55,0,4]=0.;altered.bound[55,0,4]=-82.
    second=select_context(altered,geometry(),route(),issue_time_s=0.,maximum=12)
    assert np.array_equal(first['metadata']['selected_rows'],second['metadata']['selected_rows'])
    again=select_context(past,geometry(),route(),issue_time_s=0.,maximum=12)
    assert np.array_equal(first['view']['context'],again['view']['context'])


def test_public_query_location_changes_selected_spatial_support():
    near=select_context(warm(),geometry(),route(point=-30.),issue_time_s=0.,maximum=4)
    far=select_context(warm(),geometry(),route(point=60.),issue_time_s=0.,maximum=4)
    assert near['metadata']['selected_scan_rows'][0,1]==0
    assert far['metadata']['selected_scan_rows'][0,1]==3


def test_postwarm_uses_only_payload_matched_measurement_and_delivery_clocks():
    past,event,payload=postwarm()
    result=select_context(past,geometry(),route(),issue_time_s=1.,events=[event],
        local_completed_payloads=[payload],maximum=12)
    rows=result['metadata']['selected_rows'];selected=rows[:,0]==60
    assert selected.any()
    assert np.all(result['metadata']['measurement_intervals_s'][selected]==.2)
    assert np.all(result['metadata']['arrival_intervals_s'][selected]==.5)
    assert np.all(result['metadata']['clock_provenance'][selected]==1)
    view=result['view'];censored=selected&(rows[:,2]==1)&(rows[:,3]==1)
    assert np.isneginf(view['lower'][censored]).all()
    assert np.allclose(view['upper'][censored],-75.-view['fspl'][censored])
    assert (view['context'][censored,3]==1).all()


@pytest.mark.parametrize('kind',['event_missing','payload_missing','future_delivery','hash','bound','position','value','gauge','processed'])
def test_postwarm_missing_or_inconsistent_transport_provenance_fails(kind):
    past,event,payload=postwarm();events=[event];payloads=[payload]
    if kind=='event_missing':events=[]
    elif kind=='payload_missing':payloads=[]
    elif kind=='future_delivery':event['delivery_slot']=101;event['first_usable_decision']=2
    elif kind=='hash':event['payload_sha256']='0'*64
    elif kind=='bound':past.bound[60,0]=-74.
    elif kind=='position':past.positions[60,0]=[1.,0.]
    elif kind=='value':past.value[60,0,0]=-67.
    elif kind=='gauge':payload['reference_id']='unqualified-device-gauge'
    else:payload['processed_slot']=51
    with pytest.raises(ValueError):
        select_context(past,geometry(),route(),issue_time_s=1.,events=events,
            local_completed_payloads=payloads,maximum=12)


def test_local_only_completed_payload_does_not_become_server_context():
    _,_,payload=postwarm()
    # Deliberately malformed local-only RF is never read without a delivery.
    payload['value_dbm']=['not server information'];payload['upper_dbm']='unknown'
    baseline=select_context(warm(),geometry(),route(),issue_time_s=0.,maximum=12)
    actual=select_context(warm(),geometry(),route(),issue_time_s=0.,
        local_completed_payloads=[payload],maximum=12)
    assert np.array_equal(baseline['view']['context'],actual['view']['context'])


def test_ambiguous_local_headers_fail_without_comparing_undelivered_rf():
    _,_,first=postwarm()
    class UnreadRF:
        def __eq__(self,other):raise AssertionError('Undelivered RF was compared')
    first['value_dbm']=UnreadRF();second=dict(first);second['value_dbm']=UnreadRF()
    with pytest.raises(ValueError,match='Ambiguous local scan payload identity'):
        select_context(warm(),geometry(),route(),issue_time_s=0.,
            local_completed_payloads=[first,second],maximum=12)


@pytest.mark.parametrize('setting',[0,True,193,10])
def test_invalid_context_budget_fails(setting):
    with pytest.raises(ValueError):select_context(warm(),geometry(),route(),issue_time_s=0.,maximum=setting)


@pytest.mark.parametrize('kind',['future_history','negative_clock','nan_plan','no_scans','below_floor','yard'])
def test_impossible_public_states_fail_before_prediction(kind):
    past=warm();g=geometry();plan=route();issue=0.
    if kind=='future_history':issue=1.
    elif kind=='negative_clock':issue=-1.
    elif kind=='nan_plan':plan[0,0,0]=np.nan
    elif kind=='no_scans':past.scanned[:]=False;past.heard[:]=False
    elif kind=='below_floor':past.value[past.heard]=-91.
    else:g.yard=True
    with pytest.raises(ValueError):select_context(past,g,plan,issue_time_s=issue,maximum=12)


def test_query_order_and_native_65GHz_are_explicit_public_operands():
    query,shape=make_query(geometry(),route())
    assert shape==(4,len(HORIZONS),2,3)
    assert query['q'].shape==(4*len(HORIZONS)*2*3,16)
    assert query['ap'][:6].tolist()==[0,0,0,1,1,1]
    assert np.allclose(np.exp(query['q'][:3,7])*2.4e9,[2.4e9,5.2e9,6.5e9],rtol=1e-6)
    assert np.all(query['rx_xyz'][:,2]==1.)
    assert np.all(query['offset']==0.)


def test_centered_model_frame_preserves_all_native_physical_operators():
    from wirelessjev.native_bridge.predictor import AP_HEIGHT_M, RX_HEIGHT_M, public_features
    from wirelessjev.native_bridge_ops.operators import PublicQueryGeometry, query_operands
    past=warm();g=geometry();plan=route(point=17.)
    result=select_context(past,g,plan,issue_time_s=0.,maximum=12)
    view=result['view'];meta=result['metadata'];t,robot,ap,band=meta['selected_rows'].T
    physical_rx=np.column_stack((past.positions[t,robot],np.full(len(t),RX_HEIGHT_M)))
    physical_ap=np.column_stack((g.sites,np.full(len(g.sites),AP_HEIGHT_M)))
    old_q=public_features(physical_rx,physical_ap[ap],np.array([2.4e9,5.2e9,6.5e9])[band],g.noise[band],g.tx[band])
    translated_columns=[0,1,3,4];unchanged=[i for i in range(16) if i not in translated_columns]
    assert np.array_equal(view['context'][:,4:][:,unchanged],old_q[:,unchanged])
    assert np.array_equal(view['context'][:,4:][:,translated_columns],old_q[:,translated_columns]-.5)
    assert np.allclose(view['rx_xyz'],physical_rx-MODEL_ORIGIN_M)
    assert np.allclose(result['ap_xyz'],physical_ap-MODEL_ORIGIN_M)
    assert np.array_equal(meta['physical_context_rx_xyz_m'],physical_rx.astype('f4'))
    assert np.array_equal(meta['physical_ap_xyz_m'],physical_ap.astype('f4'))
    assert np.allclose(view['rx_xyz']-result['ap_xyz'][ap],physical_rx-physical_ap[ap],atol=1e-5)
    assert meta['model_coordinate_origin_m']==(60.,40.,0.)
    query,_=make_query(g,plan);old_query=query_operands(PublicQueryGeometry(g.sites,g.tx,g.noise),plan)
    for key in ('basis','fspl','offset'):
        assert np.array_equal(query[key],old_query[key])
    assert np.array_equal(query['q'][:,unchanged],old_query['query'][:,unchanged])
    assert np.array_equal(query['q'][:,translated_columns],old_query['query'][:,translated_columns]-.5)
    assert np.all(query['rx_xyz'][:,0]==17.-60.)
    # Original public plans/reports never change when model inputs recenter.
    assert np.all(plan[:,:,0]==17.)
    assert np.array_equal(past.positions,warm().positions)


class FixtureField:
    """Known software boundary fixture, never research weights or evidence."""
    def __init__(self,cell='hier_reg'):
        self.config={'cell':cell};self.seen=[]
    def eval(self):return self
    def encode_view(self,view,ap_xyz):
        self.seen.append({key:value.clone() for key,value in view.items()})
        return {'global_mean':torch.zeros((1,8)), 'global_covariance':torch.eye(8)[None]}
    def render_queries(self,state,query):
        return query['fspl']+query['offset'],torch.ones_like(query['fspl'])*3.


def sensor():return SimpleNamespace(events=[],local_completed_payloads=[])


def test_predictor_uses_common_real_rows_point_events_and_paid_support_preparation():
    model=FixtureField();past=warm();s=sensor()
    predictor=NativeFieldPredictor('hierarchy',geometry(),s,model=model,model_sha256='a'*64)
    forecast,kind,elapsed=predictor.step(past,route())
    assert kind=='cold' and elapsed>=predictor.support_preparation[0]['elapsed_seconds']>=0.
    assert forecast.mean_dbm.shape==(4,len(HORIZONS),2,3)
    assert forecast.source_receipt_sha256=='a'*64
    assert forecast.carrier_hz.tolist()==[2.4e9,5.2e9,6.5e9]
    seen=model.seen[0]
    assert torch.equal(seen['lower'],seen['upper'])
    assert seen['ap'].dtype==torch.int64 and seen['known'].dtype==torch.bool
    record=predictor.records[0]
    assert record['context_metadata']['actual_tokens']==len(record['context'])
    assert record['selected_rows'].shape[1]==4
    assert np.allclose(record['physical_context_rx_xyz_m'],record['context_rx_xyz']+MODEL_ORIGIN_M)
    assert np.allclose(record['physical_ap_xyz_m'],record['ap_xyz']+MODEL_ORIGIN_M)
    assert np.allclose(record['physical_query_rx_xyz_m'],record['query_rx_xyz']+MODEL_ORIGIN_M)
    query,_=make_query(geometry(),route())
    assert all(np.array_equal(record['query_'+key],value) for key,value in query.items())
    assert len(predictor.memories[0])>0
    assert predictor.last_history['value'] is not past.value
    assert np.array_equal(predictor.last_history['value'],past.value)
    with pytest.raises(ValueError,match='FIFO'):
        predictor.step(past,route())


@pytest.mark.parametrize('method,cell',[('hierarchy','global_reg'),('global_reg','hier_reg'),('unknown','hier_reg')])
def test_wrong_native_branch_never_gets_a_new_method_identity(method,cell):
    with pytest.raises(ValueError):
        NativeFieldPredictor(method,geometry(),sensor(),model=FixtureField(cell),model_sha256='a'*64)


@pytest.mark.parametrize('identity',['bad','A'*64,'f'*63,None])
def test_checkpoint_identity_is_mandatory(identity):
    with pytest.raises(ValueError):
        NativeFieldPredictor('hierarchy',geometry(),sensor(),model=FixtureField(),model_sha256=identity)


def test_global_and_hierarchy_share_exact_context_selection():
    a=NativeFieldPredictor('hierarchy',geometry(),sensor(),model=FixtureField('hier_joint'),model_sha256='a'*64)
    b=NativeFieldPredictor('global_reg',geometry(),sensor(),model=FixtureField('global_reg'),model_sha256='b'*64)
    a.step(warm(),route());b.step(warm(),route())
    for key in ('context','selected_rows','context_ap','context_lower','context_upper','context_rx_xyz'):
        assert np.array_equal(a.records[0][key],b.records[0][key])


def numerical_prior():
    """Proper Gaussian software fixture; never a fitted research artifact."""
    from wirelessjev.hierarchical_radio.baselines import Prior
    return Prior('radiomap_operator',10.,np.zeros(66),np.ones(66),np.zeros(66),
        np.eye(66),np.ones(66),3.,np.zeros((48,3)),np.zeros(48),'{}')


def test_matched_numerical_adapter_shares_reports_and_never_needs_query_labels():
    neural=NativeFieldPredictor('hierarchy',geometry(),sensor(),model=FixtureField(),model_sha256='a'*64)
    numerical=NativeFieldPredictor('matched_radiomap',geometry(),sensor(),prior=numerical_prior(),prior_sha256='b'*64)
    first,_,_=neural.step(warm(),route());second,_,_=numerical.step(warm(),route())
    assert first.mean_dbm.shape==second.mean_dbm.shape
    assert np.isfinite(second.mean_dbm).all() and np.isfinite(second.std_db).all()
    for key in ('context','selected_rows','context_ap','context_lower','context_upper','context_rx_xyz'):
        assert np.array_equal(neural.records[0][key],numerical.records[0][key])
    assert second.source_receipt_sha256=='b'*64


def test_unused_artifact_identities_cannot_silently_cross_methods():
    with pytest.raises(ValueError):
        NativeFieldPredictor('hierarchy',geometry(),sensor(),model=FixtureField(),
            model_sha256='a'*64,prior_sha256='b'*64)
    with pytest.raises(ValueError):
        NativeFieldPredictor('matched_radiomap',geometry(),sensor(),prior=numerical_prior(),
            prior_sha256='b'*64,model_sha256='a'*64)


def test_full_history_reference_is_real_separate_predictor_without_matching_guard():
    bundle=build_predictor('full_radiomap',geometry(),sensor())
    from wirelessjev.closed_loop.runtime import SinglePredictor
    assert isinstance(bundle.predictor,SinglePredictor)
    assert bundle.raw is None and bundle.support_enabled is False
    with pytest.raises(ValueError):
        build_predictor('full_radiomap',geometry(),sensor(),model_sha256='a'*64)


def test_built_neural_bridge_is_non_actuating_and_explicitly_relative_gauge():
    bundle=build_predictor('hierarchy',geometry(),sensor(),model=FixtureField(),model_sha256='a'*64)
    assert bundle.raw is not None and bundle.support_enabled is True
    assert bundle.predictor.adapter.source_id==bundle.raw.source_id
    assumptions=bundle.predictor.adapter.assumptions
    assert assumptions.carrier_hz.tolist()==[2.4e9,5.2e9,6.5e9]
    assert assumptions.device0_bias_db.tolist()==[0.,0.,0.]


def test_matched_admission_keeps_the_guard_but_corrects_old_64_token_metadata():
    memory=SpatialReportMemory()
    memory.ingest(RFReport('fixture',0,1,1,5.2e9,(0.,0.,1.),'heard',NATIVE_GAUGE,
        0.,(-1.,-1.),value_dbm=-70.),at_s=0.)
    raw=SimpleNamespace(memories={0:memory},records=[{'plan':route(1),'context':np.zeros((192,20))}],geometry=geometry())
    queue=SimpleNamespace(last=0,age=0.)
    class Utility:
        def __init__(self):self.records=[]
        def __call__(self,decision):return np.ones(1,bool)
    guard=MatchedSupportAdmission(Utility(),raw,queue,sensor())
    decision=RoamDecision.capture(0,np.array([0]),np.array([1]),np.array([1.]),np.array([True]),np.ones((1,2)))
    assert guard(decision).tolist()==[True]
    row=guard.records[-1]
    assert row['forecast_context_tokens']==192 and row['forecast_context_budget']==192
    assert '64' not in row['support_information']
    assert row['packet_risk_qualified'] is False
