"""Finite planning proofs on analytic fixtures and archived native PHY curves.

No simulator case, training, SERVICE holdout or hardware action occurs here.
"""
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from wirelessjev.adaptive_radio.planning import BoundedBeamMPC,MultiStepProposal,slice_forecast
from wirelessjev.jev.mpc import DiscreteMPC
from wirelessjev.jev.spec import MissionSpec
from wirelessjev.worldmodel.interface import CandidateAction,SceneState,OutcomeDistribution
from wirelessjev.worldmodel.radiomap_adapter import RadioMapAdapter,integrate


class AnalyticFieldOutcome:
    """Test-only exact hazard = supplied scalar field, not scientific evidence."""
    def __init__(self):self.calls=[]
    def query(self,state,actions,horizon,timescale):
        self.calls.append((state,tuple(actions),horizon,timescale))
        n=len(state.current_ap);risk=[];rates=[];integrals=[];switches=[]
        for action in actions:
            p=state.snr_db[:,:,action.ap,0]
            switching=np.where(state.current_ap==action.ap,0.,action.handover_seconds)
            risk.append(np.where(state.times_s[None]<switching[:,None],1.,p))
            rates.append(np.zeros_like(p))
            stay=integrate(p,state.times_s,0,horizon)
            move=action.handover_seconds+integrate(p,state.times_s,action.handover_seconds,horizon)
            integrals.append(np.where(switching>0,move,stay));switches.append(switching)
        return OutcomeDistribution(tuple(a.action_id for a in actions),state.times_s,
            np.stack(risk,axis=1),np.stack(rates,axis=1),np.stack(integrals,axis=1),
            np.stack(switches,axis=1),state.source_id,{'scope':'analytic_software_fixture'})


def state():
    times=np.array((0.,.5,1.,1.5,2.))
    p0=.2+.3*times
    p1=.65-.3*times
    values=np.stack((p0,p1),axis=1)[None,:,:,None]*np.ones((1,1,1,3))
    return SceneState(times,np.array((0,)),values,'analytic-fixture')


def actions():return (CandidateAction('ap0',0,handover_seconds=.25),CandidateAction('ap1',1,handover_seconds=.25))


def analytic_path_cost(path):
    """Independent closed form for each known linear hazard and switch cost."""
    current=0;cost=0.
    for stage,target in enumerate(path):
        switch=.25 if target!=current else 0.
        # p0(t)=.2+.3t, p1(t)=.65-.3t in global time.
        intercept,slope=((.2,.3),(.65,-.3))[target]
        left=stage+switch;right=stage+1
        cost+=switch+intercept*(right-left)+.5*slope*(right*right-left*left)
        current=target
    return cost


def test_two_step_exhaustive_matches_independent_closed_form():
    model=AnalyticFieldOutcome()
    result=BoundedBeamMPC(beam_width=None,max_evaluations=6,deadline_seconds=5.).plan(
        model,state(),actions(),MissionSpec('analytic'),steps=2,step_seconds=1.)
    paths=list(product(range(2),repeat=2));best=min(paths,key=lambda p:(analytic_path_cost(p),p))
    assert tuple(result.action_index[0])==best
    assert result.objective[0]==pytest.approx(analytic_path_cost(best),abs=1e-12)
    assert result.evaluated_actions==6 and result.exhaustive_search_completed
    # Prefix AP1 is used as the next association, not the original AP0.
    assert any(call[0].current_ap[0]==1 for call in model.calls[2:])
    assert result.requires_existing_gate and not result.actuation_authorized and not result.production_qualified
    assert result.dynamics=='exogenous_RF_known_association_handover_only'


def test_interpolation_preserves_source_knots_and_prediction_only_heads():
    s=state();heads={'outage':'a'*64}
    original=replace(s,learned_outage_probability=s.snr_db*.5,qualified_head=heads)
    local=slice_forecast(original,0,.25,1.25,current_ap=1)
    np.testing.assert_array_equal(local.times_s,(0.,.25,.75,1.))
    np.testing.assert_allclose(local.snr_db[0,:,0,0],.2+.3*np.array((.25,.5,1.,1.25)))
    np.testing.assert_array_equal(local.learned_outage_probability,local.snr_db*.5)
    assert local.current_ap.tolist()==[1] and dict(local.qualified_head)==heads
    np.testing.assert_array_equal(original.current_ap,[0])


def test_single_step_matches_frozen_discrete_mpc_using_real_native_curves():
    rng=np.random.default_rng(17)
    s=SceneState((0.,.5,1.),np.array((0,1)),rng.uniform(-8,22,(2,3,2,3)),'native-curve-fixture')
    a=(CandidateAction('ap0mcs',0,'5g',0,handover_seconds=.25),
       CandidateAction('ap1auto',1,handover_seconds=.25))
    mission=MissionSpec('same-cost',outage_weight=1.,rate_weight=.001,switching_weight=.1)
    model=RadioMapAdapter()
    frozen=DiscreteMPC(max_candidates=2,deadline_seconds=5.).plan(model,s,a,mission,horizon=1.)
    new=BoundedBeamMPC(beam_width=None,max_evaluations=4,deadline_seconds=5.).plan(
        model,s,a,mission,steps=1,step_seconds=1.)
    np.testing.assert_array_equal(new.action_index[:,0],frozen.action_index)
    np.testing.assert_allclose(new.objective,frozen.objective,rtol=0,atol=1e-14)


@pytest.mark.parametrize('mission,head',(
    (MissionSpec('unknown-risk',max_outage_probability=.9),'outage'),
    (MissionSpec('unknown-silence',max_silence_seconds=.5),'silence')))
def test_unknown_chance_or_silence_head_returns_fallback(mission,head):
    result=BoundedBeamMPC(deadline_seconds=5.).plan(AnalyticFieldOutcome(),state(),actions(),mission,
        steps=2,step_seconds=1.)
    assert not result.complete.any() and np.isinf(result.objective).all()
    assert head in result.reason[0] and 'independent_fallback_required' in result.reason[0]
    np.testing.assert_array_equal(result.action_index,[[-1,-1]])


def test_evaluation_cap_never_returns_a_partial_path():
    result=BoundedBeamMPC(beam_width=None,max_evaluations=2,deadline_seconds=5.).plan(
        AnalyticFieldOutcome(),state(),actions(),MissionSpec('budget'),steps=2,step_seconds=1.)
    assert result.evaluated_actions==2 and result.evaluation_budget_exhausted
    assert not result.complete.any() and 'evaluation_budget' in result.reason[0]
    np.testing.assert_array_equal(result.action_index,[[-1,-1]])


def test_cap_can_keep_best_complete_path_seen_without_an_optimality_claim():
    result=BoundedBeamMPC(beam_width=None,max_evaluations=4,deadline_seconds=5.).plan(
        AnalyticFieldOutcome(),state(),actions(),MissionSpec('budget'),steps=2,step_seconds=1.)
    assert result.complete[0] and result.evaluation_budget_exhausted and not result.search_completed
    assert tuple(result.action_index[0]) in ((0,0),(0,1))
    assert result.objective[0]==pytest.approx(min(analytic_path_cost((0,0)),analytic_path_cost((0,1))))


def test_deterministic_beam_pruning_has_no_exhaustive_claim():
    planner=BoundedBeamMPC(beam_width=1,max_evaluations=6,deadline_seconds=5.)
    first=planner.plan(AnalyticFieldOutcome(),state(),actions(),MissionSpec('beam'),steps=2,step_seconds=1.)
    second=planner.plan(AnalyticFieldOutcome(),state(),actions(),MissionSpec('beam'),steps=2,step_seconds=1.)
    np.testing.assert_array_equal(first.action_index,second.action_index)
    assert first.evaluated_actions==4 and first.beam_pruned_paths==1 and not first.exhaustive_search_completed


def test_equal_objective_ties_follow_input_action_path_order():
    s=state();s=replace(s,snr_db=np.full_like(s.snr_db,.4))
    a=(CandidateAction('first',1,handover_seconds=0.),CandidateAction('second',0,handover_seconds=0.))
    result=BoundedBeamMPC(beam_width=None,deadline_seconds=5.).plan(
        AnalyticFieldOutcome(),s,a,MissionSpec('tie'),steps=2,step_seconds=1.)
    np.testing.assert_array_equal(result.action_index,[[0,0]])


def test_late_model_result_is_discarded_even_when_complete():
    class Clock:
        now=0.
        def __call__(self):return self.now
    clock=Clock()
    class SlowAnalytic(AnalyticFieldOutcome):
        def query(self,*args,**kwargs):
            value=super().query(*args,**kwargs);clock.now+=2.;return value
    result=BoundedBeamMPC(beam_width=None,deadline_seconds=1.,clock=clock).plan(
        SlowAnalytic(),state(),actions(),MissionSpec('deadline'),steps=1,step_seconds=1.)
    assert result.evaluated_actions==1 and result.deadline_overrun and not result.complete.any()
    assert 'deadline' in result.reason[0]


@pytest.mark.parametrize('change',({'step_seconds':.1},{'steps':3},{'pending_handover_seconds':np.array((.2,))},
                                 {'timescale':'unknown'},{'independently_qualified_heads':('invented',)}))
def test_unsupported_transitions_and_scope_reject(change):
    args={'steps':2,'step_seconds':1.}|change
    with pytest.raises(ValueError):
        BoundedBeamMPC(deadline_seconds=5.).plan(AnalyticFieldOutcome(),state(),actions(),MissionSpec('bad'),**args)


def test_partial_path_or_actuation_authorization_cannot_be_forged():
    result=BoundedBeamMPC(deadline_seconds=5.).plan(AnalyticFieldOutcome(),state(),actions(),MissionSpec('a'),
        steps=2,step_seconds=1.)
    with pytest.raises(ValueError,match='incomplete'):replace(result,action_index=np.array(((0,-1),)))
    with pytest.raises(ValueError,match='authorize'):replace(result,actuation_authorized=True)
    with pytest.raises(ValueError):result.action_index.flags.writeable=True
    with pytest.raises(ValueError,match='path cap'):
        BoundedBeamMPC(max_steps=2).plan(AnalyticFieldOutcome(),state(),actions(),MissionSpec('bad'),steps=3,step_seconds=.5)
