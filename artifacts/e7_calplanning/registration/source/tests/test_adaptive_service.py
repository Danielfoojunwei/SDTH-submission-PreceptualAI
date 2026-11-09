"""Analytic service-boundary checks; no scientific simulator case is run."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wirelessjev.adaptive_radio.service import (RadioAssumptions,RSSIForecast,RSSIOutcomeAdapter,
    CalibrationScope,HeldoutCalibration,ResidualBound,PacketRiskCalibration,PACKET_ENDPOINT,
    BridgedPredictor,TypedRoamUtility,ServiceProbeConfig,DevelopmentRegistration,
    run_development_pair,NATIVE_CARRIERS_HZ)
from wirelessjev.worldmodel.interface import CandidateAction,SceneState
from wirelessjev.worldmodel.radiomap_adapter import RadioMapAdapter
from wirelessjev.native.world import HORIZONS
from wirelessjev.closed_loop.control import OutageUtility
from wirelessjev.jev.spec import MissionSpec
from wirelessjev.jev.cost import evaluate


def assumptions(**changes):
    values=dict(carrier_hz=np.asarray(NATIVE_CARRIERS_HZ),forecast_transmit_power_dbm=np.full(3,20.),
        service_transmit_power_dbm=np.full(3,15.),receiver_noise_dbm=np.full(3,-95.),
        device0_bias_db=np.zeros(3),antenna_gain_change_db=np.zeros(3),interference_mw=np.zeros(3),
        absolute_reference='software fixture: device0 bias explicitly assumed zero',
        interference_source='software fixture: interference explicitly absent',reciprocal_path_gain=True)
    return RadioAssumptions(**(values|changes))


def forecast(n=3,s=2):
    value=np.full((n,len(HORIZONS),s,3),-70.)
    value[:,:,1,1]=-64.
    return RSSIForecast(HORIZONS,np.zeros(n,dtype=int),value,np.full_like(value,2.),NATIVE_CARRIERS_HZ,
                        'candidate','a'*64)


def scope(a,endpoint='rssi_dbm'):
    return CalibrationScope('candidate','fixture-site','fixture-device',a.sha256,endpoint,
                             'b'*64 if endpoint==PACKET_ENDPOINT else None)


def evidence(a,n=20,endpoint='rssi_dbm',**changes):
    fields=dict(partition='validation_calibration',unit_ids=tuple('cal-'+str(i) for i in range(n)),
        fit_unit_ids=('train-0',),selection_unit_ids=('selection-0',),receipt_sha256='c'*64,
        scope=scope(a,endpoint))
    return HeldoutCalibration(**(fields|changes))


def adapter(a=None,**changes):
    return RSSIOutcomeAdapter(a or assumptions(),'candidate',site_id='fixture-site',device_id='fixture-device',**changes)


def test_reference_tx_noise_and_interference_conversion():
    a=assumptions(device0_bias_db=np.full(3,2.),interference_mw=np.full(3,10**(-95/10)))
    f=forecast();actual=a.convert(f.mean_dbm)
    expected=f.mean_dbm-2.-20.+15.-(-95.+10*np.log10(2))
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    with pytest.raises(ValueError):assumptions(reciprocal_path_gain=False)
    with pytest.raises(ValueError):assumptions(interference_mw=np.array((0.,-1.,0.)))


def test_carrier_mismatch_and_readonly_forecast():
    a=adapter();f=forecast()
    with pytest.raises(ValueError,match='Carrier mismatch'):
        a.scene_state(replace(f,carrier_hz=np.array((2.4e9,5.2e9,6e9))))
    with pytest.raises(ValueError):f.mean_dbm.flags.writeable=True
    with pytest.raises(ValueError):assumptions().receiver_noise_dbm[0]=0
    with pytest.raises(ValueError,match='private World'):a.scene_state(object())
    a.scene_state(f)
    with pytest.raises(ValueError,match='receipt changed'):a.scene_state(replace(f,source_receipt_sha256='d'*64))


def test_default_outcomes_exactly_match_native_map_surrogate():
    f=forecast();model=adapter();state=model.scene_state(f)
    actions=(CandidateAction('stay',0),CandidateAction('roam',1,handover_seconds=.03),
             CandidateAction('mcs',1,'5g',0,handover_seconds=.03))
    actual=model.query(state,actions,5.)
    reference=RadioMapAdapter().query(state,actions,5.)
    for name in ('outage_probability','rate_mbps','outage_integral_seconds','switch_seconds'):
        np.testing.assert_array_equal(getattr(actual,name),getattr(reference,name))
    assert actual.outage_upper is None and actual.silence_upper_seconds is None
    assert actual.metadata['packet_deadline_risk']=='unavailable_from_RSSI_alone'
    assert actual.metadata['independent_gate_permission']=='none'
    assert actual.metadata['forecast_source_receipt_sha256']=='a'*64
    with pytest.raises(ValueError):model.query(replace(state,learned_outage_probability=np.zeros_like(state.snr_db)),actions,5.)
    with pytest.raises(ValueError):CandidateAction('power',0,mechanism='power_control')
    with pytest.raises(ValueError):model.query(state,(CandidateAction('badmcs',0,'5g',1000),),5.)


@pytest.mark.parametrize('partition',('field_test','service_test','development','training'))
def test_calibration_refuses_unadmitted_partitions(partition):
    with pytest.raises(ValueError,match='TRAIN/VAL'):evidence(assumptions(),partition=partition)


def test_calibration_unit_membership_not_correlated_robot_count():
    a=assumptions()
    with pytest.raises(ValueError,match='disjoint'):evidence(a,unit_ids=('train-0',))
    with pytest.raises(ValueError,match='distinct'):evidence(a,unit_ids=('same','same'))
    e=evidence(a,n=9);prediction=np.zeros((9,100))
    observed=-np.repeat(np.arange(9)[:,None],100,axis=1)
    residual=ResidualBound.fit(prediction,observed,evidence=e,alpha=.1)
    assert residual.lower_margin_db==8.
    assert residual.metadata['calibration_units']==9
    assert residual.metadata['qualifies_packet_risk'] is False
    np.testing.assert_array_equal(residual.lower(prediction,scope=e.scope),prediction-8)
    small=ResidualBound.fit(np.zeros((1,4)),np.zeros((1,4)),evidence=evidence(a,n=1),alpha=.01)
    with pytest.raises(ValueError,match='uninformative'):small.lower(np.zeros(4),scope=small.evidence.scope)


def test_rf_calibration_can_adjust_field_but_cannot_create_packet_risk():
    a=assumptions();e=evidence(a)
    residual=ResidualBound.fit(np.full((20,4),-60.),np.full((20,4),-63.),evidence=e)
    model=adapter(a,rssi_bound=residual);f=forecast();state=model.scene_state(f)
    np.testing.assert_allclose(state.snr_db,a.convert(f.mean_dbm-3))
    outcome=model.query(state,(CandidateAction('stay',0),),5.)
    assert outcome.outage_upper is None
    with pytest.raises(ValueError,match='match this model'):adapter(assumptions(device0_bias_db=np.ones(3)),rssi_bound=residual)


def packet_calibration(a,n=40,actions=None):
    actions=actions or (CandidateAction('stay',0,handover_seconds=.03),CandidateAction('roam',1,handover_seconds=.03))
    scores=np.full((n,len(actions),3),.15)
    failed=np.repeat((np.arange(n)%2)[:,None,None],len(actions)*3,axis=1).reshape(scores.shape)
    return PacketRiskCalibration.fit(scores,failed,evidence=evidence(a,n=n,endpoint=PACKET_ENDPOINT),
        actions=actions,edges=(0.,1.),min_units=20)


def test_packet_calibration_needs_real_endpoint_binary_labels_and_unit_support():
    a=assumptions();actions=(CandidateAction('stay',0),)
    with pytest.raises(ValueError,match='deadline labels'):
        PacketRiskCalibration.fit(np.zeros((20,1,3)),np.zeros((20,1,3)),evidence=evidence(a),actions=actions)
    with pytest.raises(ValueError,match='Boolean failures'):
        PacketRiskCalibration.fit(np.zeros((20,1,3)),np.full((20,1,3),.2),
            evidence=evidence(a,endpoint=PACKET_ENDPOINT),actions=actions)
    cal=packet_calibration(a,n=4,actions=actions)
    with pytest.raises(ValueError,match='unit support'):cal.apply(np.zeros((1,3)),scope=cal.evidence.scope,actions=actions)
    cal=packet_calibration(a)
    assert cal.unit_counts.tolist()==[[40],[40]]  # not 40×2×3 independent samples
    assert cal.mean_failure.tolist()==[[.5],[.5]]
    with pytest.raises(ValueError,match='support'):
        cal.apply(np.zeros((1,3)),scope=cal.evidence.scope,actions=(CandidateAction('unseen',1,'5g',0),))


def test_packet_action_risk_cannot_be_hidden_by_other_actions_in_the_same_bin():
    a=assumptions()
    actions=(CandidateAction('safe',0),CandidateAction('bad',1))
    scores=np.full((40,2,8),.1)
    failed=np.zeros_like(scores);failed[:,1]=1.
    cal=PacketRiskCalibration.fit(scores,failed,evidence=evidence(a,n=40,endpoint=PACKET_ENDPOINT),
        actions=actions,edges=(0.,1.),min_units=20)
    safe,upper_safe=cal.apply(np.full((3,2),.1),scope=cal.evidence.scope,actions=(actions[0],))
    bad,upper_bad=cal.apply(np.full((3,2),.1),scope=cal.evidence.scope,actions=(actions[1],))
    np.testing.assert_array_equal(safe,np.zeros((3,2)))
    np.testing.assert_array_equal(bad,np.ones((3,2)))
    np.testing.assert_array_equal(upper_bad,np.ones((3,2)))
    expected_radius=np.sqrt(np.log(2/.05)/(2*40))
    np.testing.assert_allclose(upper_safe,expected_radius)
    # Action order is explicit; reordering cannot exchange or average risk.
    both,_=cal.apply(np.full((3,2,2),.1),scope=cal.evidence.scope,actions=actions[::-1])
    np.testing.assert_array_equal(both[:,0],np.ones((3,2)))
    np.testing.assert_array_equal(both[:,1],np.zeros((3,2)))
    assert cal.metadata['actions_pooled'] is False and cal.metadata['family_cells']==2
    assert cal.metadata['independent_gate_permission'] is False


def test_packet_action_bin_support_cannot_be_borrowed_from_other_supported_cells():
    a=assumptions();actions=(CandidateAction('a',0),CandidateAction('b',1))
    scores=np.empty((40,2,8));scores[:,0]=.1;scores[:,1]=.9
    failed=np.zeros_like(scores)
    cal=PacketRiskCalibration.fit(scores,failed,evidence=evidence(a,n=40,endpoint=PACKET_ENDPOINT),
        actions=actions,edges=(0.,.5,1.),min_units=20)
    assert cal.unit_counts.tolist()==[[40,0],[0,40]]
    cal.apply(np.full((3,2),.1),scope=cal.evidence.scope,actions=(actions[0],))
    with pytest.raises(ValueError,match='unit support'):
        cal.apply(np.full((3,2),.1),scope=cal.evidence.scope,actions=(actions[1],))
    with pytest.raises(ValueError,match='forecast axes'):
        cal.apply(np.full((2,3),.1),scope=cal.evidence.scope,actions=actions)


def test_packet_action_support_uses_exact_handover_cost_not_rounded_identity():
    a=assumptions();action=CandidateAction('stay',0,handover_seconds=.03)
    cal=packet_calibration(a,actions=(action,))
    changed=replace(action,handover_seconds=np.nextafter(.03,1.))
    with pytest.raises(ValueError,match='support'):
        cal.apply(np.zeros((1,3)),scope=cal.evidence.scope,actions=(changed,))


def test_packet_head_exact_switch_boundary_and_external_qualification():
    a=assumptions();cal=packet_calibration(a);model=adapter(a,packet_calibration=cal)
    state=model.scene_state(forecast())
    actions=(CandidateAction('stay',0,handover_seconds=.03),CandidateAction('roam',1,handover_seconds=.03))
    outcome=model.query(state,actions,5.)
    np.testing.assert_allclose(outcome.outage_integral_seconds[:,0],2.5)
    np.testing.assert_allclose(outcome.outage_integral_seconds[:,1],.03+.5*(5.-.03))
    assert np.all(outcome.outage_probability[:,1,0]==1)
    assert np.all(outcome.outage_upper>=outcome.outage_probability)
    mission=MissionSpec('chance-request',max_outage_probability=1.)
    unqualified=evaluate(outcome,mission,5.)
    assert unqualified.unavailable_heads==('outage',) and not unqualified.feasible.any()
    assert evaluate(outcome,mission,5.,independently_qualified_heads=('outage',)).feasible.all()
    assert outcome.silence_upper_seconds is None


def test_unsupported_calibration_scope_and_head_refuse():
    a=assumptions()
    with pytest.raises(ValueError,match='silence/latency'):scope(a,'silence')
    with pytest.raises(ValueError,match='SHA256'):
        CalibrationScope('m','s','d',a.sha256,PACKET_ENDPOINT,None)


def test_100_robot_typed_planner_equals_existing_exact_utility():
    state=adapter().scene_state(forecast(n=100))
    queue=SimpleNamespace(age=0.)
    utility=TypedRoamUtility(RadioMapAdapter(),'candidate',queue,handover_seconds=.03,horizon=5.,
                             max_candidates=2,deadline_seconds=5.)
    actual=utility.scores(state.snr_db,state.current_ap)
    reference=OutageUtility(None,None,.03,5.).scores(state.snr_db,state.current_ap)
    np.testing.assert_array_equal(actual,reference)
    assert utility.records[0]['candidates']==2 and not utility.records[0]['late_discarded']
    late=TypedRoamUtility(RadioMapAdapter(),'candidate',queue,handover_seconds=.03,horizon=5.,
                         max_candidates=2,deadline_seconds=1e-100)
    np.testing.assert_array_equal(late.scores(state.snr_db,state.current_ap),np.zeros((100,2)))
    assert late.records[0]['late_discarded']
    with pytest.raises(ValueError,match='fixed budget'):
        TypedRoamUtility(RadioMapAdapter(),'candidate',queue,handover_seconds=.03,horizon=5.,
                         max_candidates=1,deadline_seconds=5.).scores(state.snr_db,state.current_ap)


def test_bridge_predictor_conversion_uses_declared_public_contract():
    f=forecast();a=adapter()
    class AnalyticFixture:
        geometry=SimpleNamespace(sites=np.zeros((2,2)))
        def step(self,past,plan):
            assert past=='past-only' and plan=='public-plan'
            return f,'cold',0.
    prediction=BridgedPredictor(AnalyticFixture(),a).step('past-only','public-plan')
    np.testing.assert_array_equal(prediction[0],a.scene_state(f).snr_db.astype('f4'))
    assert prediction[2]=='cold' and prediction[3]>=0


def test_probe_publication_guard_runs_before_world_or_simulator():
    config=ServiceProbeConfig(2,{'cold':.5,'warm':.5,'refit':.5})
    registration=DevelopmentRegistration('fresh-dev-0',0,'a'*64,'b'*64,'c'*64)
    class ForbiddenWorld:
        def __getattr__(self,name):raise AssertionError('world must not be read before publication')
    def blocked(receipt):
        assert receipt is registration
        raise RuntimeError('actual sources not published')
    with pytest.raises(RuntimeError,match='not published'):
        run_development_pair(ForbiddenWorld(),None,None,None,config=config,registration=registration,publication_guard=blocked)
    with pytest.raises(RuntimeError,match='refused'):
        run_development_pair(ForbiddenWorld(),None,None,None,config=config,registration=registration,publication_guard=lambda _:False)
    with pytest.raises(ValueError,match='SERVICE'):
        replace(registration,partition='service_test')


@pytest.mark.parametrize('bad',({'duration_seconds':1000},{'max_candidates':0},{'planner_deadline_seconds':0.},
                              {'handover_seconds':31.},{'forecast_profile_seconds':{'cold':1.}}))
def test_probe_budget_is_finite_and_explicit(bad):
    values={'duration_seconds':2,'forecast_profile_seconds':{'cold':.5,'warm':.5,'refit':.5}}|bad
    with pytest.raises(ValueError):ServiceProbeConfig(**values)
