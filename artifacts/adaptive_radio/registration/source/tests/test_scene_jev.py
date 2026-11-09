"""Typed planning/calibration unit fixtures, not empirical SLA results."""
from dataclasses import replace
import numpy as np
import pytest

from wirelessjev.jev.conformal import CalibrationScope, SplitConformalBound, route_heads
from wirelessjev.jev.cost import evaluate
from wirelessjev.jev.mpc import DiscreteMPC
from wirelessjev.jev.spec import MissionSpec, compile_mission
from wirelessjev.native.world import HORIZONS
from wirelessjev.worldmodel.interface import CandidateAction, SceneState
from wirelessjev.worldmodel.radiomap_adapter import RadioMapAdapter


def state():
    snr = np.full((2,9,2,3),-20.)
    snr[:,:,1] = 30.
    return SceneState(np.array(HORIZONS),np.array([0,1]),snr,'fixture')


def test_strict_numeric_mission_compiler_rejects_free_text_and_unknown_fields():
    mission = compile_mission({'mission_id':'pick','max_outage_probability':.001})
    assert mission.require_calibrated_outage
    with pytest.raises(TypeError): compile_mission('pick lane 3 and never be silent')
    with pytest.raises(ValueError): compile_mission({'mission_id':'pick','execute':'ssh...'})
    with pytest.raises(ValueError): MissionSpec('bad',max_outage_probability=2)
    with pytest.raises(ValueError): MissionSpec('bad',outage_weight=True)


def test_uncalibrated_mean_cannot_claim_a_chance_constraint_or_silence_bound():
    s = state()
    adapter = RadioMapAdapter()
    outcome = adapter.query(s,(CandidateAction('good',1),),30,'mid')
    risk = evaluate(outcome,MissionSpec('chance',max_outage_probability=.1),30)
    silence = evaluate(outcome,MissionSpec('silence',max_silence_seconds=2),30)
    assert not risk.feasible.any() and risk.unavailable_heads == ('outage',)
    assert not silence.feasible.any() and silence.unavailable_heads == ('silence',)
    # Deliberately asking for uncalibrated proxy risk is explicit and still
    # never permits physical actuation or implies a calibrated guarantee.
    proxy = evaluate(outcome,MissionSpec('proxy',max_outage_probability=1,
                                      require_calibrated_outage=False),30)
    assert proxy.feasible.all()


def test_same_planner_reproduces_legacy_outage_choice_and_requires_external_gate():
    s = state()
    actions = tuple(CandidateAction(f'ap-{i}',i) for i in range(2))
    outcome = RadioMapAdapter().query(s,actions,30,'mid')
    proposal = DiscreteMPC(deadline_seconds=10).plan(
        RadioMapAdapter(),s,actions,MissionSpec('legacy'),horizon=30)
    np.testing.assert_array_equal(proposal.action_index,outcome.outage_integral_seconds.argmin(1))
    np.testing.assert_array_equal(proposal.objective,outcome.outage_integral_seconds.min(1))
    assert proposal.requires_existing_gate and not proposal.actuation_authorized
    assert not proposal.production_qualified


def test_late_forecast_is_discarded_and_candidate_count_is_bounded():
    s = state()
    actions = tuple(CandidateAction(f'a-{i}',i%2) for i in range(6))
    ticks = iter([0.,0.,.2,.2])
    proposal = DiscreteMPC(deadline_seconds=.1,clock=lambda:next(ticks)).plan(
        RadioMapAdapter(),s,actions,MissionSpec('deadline'))
    assert proposal.deadline_overrun and proposal.evaluated_actions == 1
    assert (proposal.action_index == -1).all()
    capped = DiscreteMPC(max_candidates=2,deadline_seconds=10).plan(
        RadioMapAdapter(),s,actions,MissionSpec('budget'))
    assert capped.evaluated_actions == 2
    with pytest.raises(ValueError,match='Multi-step'): DiscreteMPC(depth=2)


def test_missing_qualified_heads_returns_no_action_and_preserves_existing_fallback():
    proposal = DiscreteMPC(deadline_seconds=10).plan(RadioMapAdapter(),state(),
        (CandidateAction('a',0),),MissionSpec('strict',max_outage_probability=.001))
    assert (proposal.action_index == -1).all()
    assert all(reason.startswith('unqualified_or_missing_head') for reason in proposal.reason)
    assert not proposal.actuation_authorized


def test_per_head_fallback_is_independent_and_default_is_all_radiomap():
    s = state()
    fallback = RadioMapAdapter().query(s,(CandidateAction('a',0),),30,'mid')
    primary = replace(fallback,source_id='candidate',rate_mbps=fallback.rate_mbps+1,
                      outage_probability=fallback.outage_probability*.5,
                      outage_integral_seconds=fallback.outage_integral_seconds*.5)
    routed = route_heads(primary,fallback)
    np.testing.assert_array_equal(routed.outage_probability,fallback.outage_probability)
    rate_only = route_heads(primary,fallback,independently_qualified_heads=('rate',))
    np.testing.assert_array_equal(rate_only.rate_mbps,primary.rate_mbps)
    np.testing.assert_array_equal(rate_only.outage_probability,fallback.outage_probability)
    assert rate_only.metadata['outage_source'] == fallback.source_id


def test_exact_one_sided_conformal_rank_scope_and_insufficient_sample_behavior():
    scope = CalibrationScope('utility_gain','site1','device1')
    fit = SplitConformalBound.fit(np.arange(9.),np.zeros(9),unit_ids=list(range(9)),
        alpha=.2,scope=scope,receipt_sha256='a'*64)
    assert fit.quantile == 7.  # ceil((9+1)*.8)=8th ordered score
    assert fit.lower(np.array([10.]),scope=scope)[0] == 3.
    with pytest.raises(ValueError,match='Unseen'):
        fit.lower([10.],scope=CalibrationScope('utility_gain','other','device1'))
    with pytest.raises(ValueError,match='distinct'):
        SplitConformalBound.fit([1.,2.],[1.,2.],unit_ids=['same','same'],
            alpha=.1,scope=scope,receipt_sha256='a'*64)
    too_few = SplitConformalBound.fit([1.],[1.],unit_ids=['one'],alpha=.01,
                                    scope=scope,receipt_sha256='a'*64)
    assert np.isneginf(too_few.lower([1.],scope=scope)).all()


def test_empirical_drift_veto_does_not_claim_probability_or_drift_guarantees():
    scope = CalibrationScope('utility_gain','site1','device1')
    fit = SplitConformalBound.fit(np.zeros(20),np.zeros(20),unit_ids=list(range(20)),
        alpha=.1,scope=scope,receipt_sha256='a'*64)
    observed = fit.monitor(np.ones(20)*10,np.zeros(20),scope=scope)
    assert observed['fallback_required']
    assert not observed['coverage_guaranteed_under_drift']
    assert not observed['probability_of_SLA_violation_bounded']


def test_horizon_mismatch_cannot_silently_reprice_shorter_forecast():
    s = state()
    outcome = RadioMapAdapter().query(s,(CandidateAction('a',0),),5,'mid')
    with pytest.raises(ValueError,match='horizon'):
        evaluate(outcome,MissionSpec('wrong-horizon'),30)


def test_wrong_action_forecast_cannot_enter_the_plan():
    adapter = RadioMapAdapter()
    class WrongIdentity:
        def query(self,s,actions,horizon,timescale):
            outcome = adapter.query(s,actions,horizon,timescale)
            return replace(outcome,action_ids=('different-action',))
    with pytest.raises(ValueError,match='requested action'):
        DiscreteMPC(deadline_seconds=10).plan(WrongIdentity(),state(),
            (CandidateAction('requested',0),),MissionSpec('mission'))


def test_no_informative_conformal_rank_means_fallback_even_with_apparent_coverage():
    scope = CalibrationScope('utility_gain','site','device')
    fit = SplitConformalBound.fit([0.],[0.],unit_ids=['one'],alpha=.01,
                                  scope=scope,receipt_sha256='a'*64)
    observed = fit.monitor([0.],[0.],scope=scope)
    assert observed['empirical_coverage'] == 1
    assert not observed['informative_bound'] and observed['fallback_required']


def test_model_cannot_hide_declared_switch_time_inside_smaller_integral():
    outcome = RadioMapAdapter().query(state(),(CandidateAction('a',0),),30,'mid')
    with pytest.raises(ValueError):
        replace(outcome,outage_integral_seconds=np.zeros_like(outcome.switch_seconds))
