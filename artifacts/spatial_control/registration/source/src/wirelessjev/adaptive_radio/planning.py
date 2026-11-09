"""Finite multistep association proposals over an EXOGENOUS RF forecast.

Only the association/handover state is advanced.  Actions do not alter future
propagation, routes, interference, queues or traffic in this planner.  A path
must complete within its budget before it can become a proposal; execution
still requires the existing independent gate and is never authorized here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from ..jev.cost import evaluate
from ..jev.spec import MissionSpec
from ..worldmodel.interface import (CandidateAction,OutcomeDistribution,SceneState,
                                    immutable_array)


def slice_forecast(state,robot,start_seconds,end_seconds,*,current_ap):
    """Retain every interior source knot and interpolate exact step endpoints.

    Query times are local to this step.  No new RF prediction is invented:
    all values are piecewise-linear interpolation of the supplied forecast.
    A head's provenance travels with it but never supplies gate permission.
    """
    if not isinstance(state,SceneState):raise TypeError('A predicted SceneState is required')
    if type(robot) is not int or not 0<=robot<len(state.current_ap):raise ValueError('Invalid robot index')
    if (not math.isfinite(start_seconds) or not math.isfinite(end_seconds)
            or not 0<=start_seconds<end_seconds<=state.times_s[-1]):
        raise ValueError('Step boundaries must lie within forecast support')
    if type(current_ap) is not int or not 0<=current_ap<state.snr_db.shape[2]:
        raise ValueError('Known association must name a supported AP')
    source=state.times_s
    absolute=np.r_[start_seconds,source[(source>start_seconds)&(source<end_seconds)],end_seconds]
    right=np.searchsorted(source,absolute,side='right').clip(1,len(source)-1);left=right-1
    weight=((absolute-source[left])/(source[right]-source[left]))[None,:,None,None]
    def at(value):
        return value[robot:robot+1,left]*(1-weight)+value[robot:robot+1,right]*weight
    return SceneState(absolute-start_seconds,np.array((current_ap,),dtype=np.int64),at(state.snr_db),
        state.source_id,None if state.learned_outage_probability is None else at(state.learned_outage_probability),
        state.qualified_head)


@dataclass(frozen=True)
class MultiStepProposal:
    action_index: np.ndarray
    action_ids: tuple[str,...]
    objective: np.ndarray
    reason: tuple[str,...]
    step_seconds: float
    evaluated_actions: int
    elapsed_seconds: float
    deadline_overrun: bool
    evaluation_budget_exhausted: bool
    search_completed: bool
    beam_pruned_paths: int
    requires_existing_gate: bool=True
    production_qualified: bool=False
    actuation_authorized: bool=False
    dynamics: str='exogenous_RF_known_association_handover_only'

    def __post_init__(self):
        raw=np.asarray(self.action_index)
        ids=tuple(self.action_ids)
        if (raw.ndim!=2 or not all(raw.shape) or raw.dtype.kind not in 'iu'
                or not ids or len(set(ids))!=len(ids)
                or any(not isinstance(s,str) or not s for s in ids)
                or np.any((raw< -1)|(raw>=len(ids)))):
            raise ValueError('A complete integral action path or all-minus-one fallback is required')
        valid=np.all(raw>=0,axis=1)
        if np.any(~valid&~np.all(raw==-1,axis=1)):
            raise ValueError('An incomplete prefix is not a feasible multistep proposal')
        objective=np.asarray(self.objective,dtype=float)
        if (objective.shape!=(len(raw),) or not np.isfinite(objective[valid]).all()
                or not np.all(objective[~valid]==np.inf)):
            raise ValueError('Only complete feasible paths may have a finite objective')
        if (len(self.reason)!=len(raw) or any(not isinstance(r,str) or not r for r in self.reason)
                or not math.isfinite(self.step_seconds) or self.step_seconds<=0
                or type(self.evaluated_actions) is not int or self.evaluated_actions<0
                or not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds<0
                or type(self.beam_pruned_paths) is not int or self.beam_pruned_paths<0
                or any(type(getattr(self,name)) is not bool for name in
                       ('deadline_overrun','evaluation_budget_exhausted','search_completed'))):
            raise ValueError('Invalid finite search evidence')
        if (self.requires_existing_gate is not True or self.production_qualified is not False
                or self.actuation_authorized is not False
                or self.dynamics!='exogenous_RF_known_association_handover_only'):
            raise ValueError('A search proposal cannot qualify a controller or authorize actuation')
        object.__setattr__(self,'action_index',immutable_array(raw,dtype=np.int64))
        object.__setattr__(self,'objective',np.frombuffer(objective.tobytes(),dtype=objective.dtype))
        object.__setattr__(self,'action_ids',ids)
        object.__setattr__(self,'reason',tuple(self.reason))

    @property
    def complete(self):return np.all(self.action_index>=0,axis=1)

    @property
    def exhaustive_search_completed(self):
        return self.search_completed and self.beam_pruned_paths==0


@dataclass(frozen=True)
class _Path:
    current_ap: int
    choices: tuple[int,...]
    objective: float


class BoundedBeamMPC:
    """Independent per-robot finite search, not joint fleet optimization.

    beam_width=None exhausts the finite action tree if time/evaluation budgets
    permit. Finite beams rank accumulated cost, with no admissible lookahead
    heuristic or optimality claim. Ties use the input action order/path index.
    Query evaluations are counted globally across robots and partial paths.
    """
    def __init__(self,*,beam_width=16,max_evaluations=4096,max_steps=32,deadline_seconds=.1,clock=time.monotonic):
        if beam_width is not None and (type(beam_width) is not int or beam_width<1):
            raise ValueError('Beam width must be positive or None for bounded exhaustive search')
        if type(max_evaluations) is not int or max_evaluations<1:raise ValueError('A finite positive evaluation cap is required')
        if type(max_steps) is not int or not 1<=max_steps<=1024:raise ValueError('Explicit path storage cap must be 1..1024 steps')
        if (isinstance(deadline_seconds,(bool,np.bool_)) or not math.isfinite(deadline_seconds)
                or deadline_seconds<=0 or not callable(clock)):
            raise ValueError('A positive deadline and callable clock are required')
        self.beam_width=beam_width;self.max_evaluations=max_evaluations;self.max_steps=max_steps
        self.deadline_seconds=float(deadline_seconds);self.clock=clock

    def plan(self,world_model,state,actions,mission,*,steps,step_seconds,timescale='mid',
             independently_qualified_heads=(),pending_handover_seconds=None):
        """Search only; None pending state explicitly assumes receiver ready.

        A caller with an in-progress handover must supply that observation so
        it is rejected. The later independent execution gate still checks real
        receiver availability, command freshness and action authorization.
        """
        if not isinstance(state,SceneState) or not isinstance(mission,MissionSpec):
            raise TypeError('Use predicted SceneState and typed MissionSpec; missions never modify RF forecasts')
        if type(steps) is not int or not 1<=steps<=self.max_steps:raise ValueError('A positive step count within the declared path cap is required')
        if (isinstance(step_seconds,(bool,np.bool_)) or not math.isfinite(step_seconds) or step_seconds<=0
                or steps*step_seconds>state.times_s[-1]):
            raise ValueError('Positive step duration and total horizon within RF support are required')
        if timescale not in ('fast','mid','slow'):raise ValueError('Unsupported timescale')
        heads=tuple(independently_qualified_heads)
        if not set(heads)<={'outage','rate','silence'}:raise ValueError('Unsupported independently qualified head')
        actions=tuple(actions)
        if (not actions or any(not isinstance(a,CandidateAction) for a in actions)
                or len({a.action_id for a in actions})!=len(actions)):
            raise ValueError('Distinct typed supported association candidates are required')
        if any(a.ap>=state.snr_db.shape[2] for a in actions):raise ValueError('Action AP is outside RF support')
        if any(a.handover_seconds>step_seconds for a in actions):
            raise ValueError('Each association handover must complete within one step; overlap is unsupported')
        n=len(state.current_ap)
        if pending_handover_seconds is not None:
            pending=immutable_array(pending_handover_seconds,shape=(n,),name='pending handover seconds')
            if np.any(pending!=0):raise ValueError('Pending handover overlap is unsupported; use independent fallback')
        result=np.full((n,steps),-1,dtype=np.int64);objective=np.full(n,np.inf)
        reasons=['independent_fallback_required:no_complete_feasible_plan']*n
        started=self.clock();evaluated=0;late=False;exhausted=False;all_searched=True;pruned=0
        deadline_hit=False
        for robot in range(n):
            beams=[_Path(int(state.current_ap[robot]),(),0.)];complete=[];missing=set();search_finished=True
            for depth in range(steps):
                expanded=[];stop=False
                for node in beams:
                    local=slice_forecast(state,robot,depth*step_seconds,(depth+1)*step_seconds,
                                         current_ap=node.current_ap)
                    for index,action in enumerate(actions):
                        if evaluated>=self.max_evaluations:
                            exhausted=True;stop=True;search_finished=False;break
                        if self.clock()-started>=self.deadline_seconds:
                            deadline_hit=True;stop=True;search_finished=False;break
                        outcome=world_model.query(local,(action,),float(step_seconds),timescale)
                        evaluated+=1
                        if self.clock()-started>self.deadline_seconds:
                            # A synchronous model call cannot be preempted. A
                            # late result never contributes to a complete path.
                            late=True;deadline_hit=True;stop=True;search_finished=False;break
                        if (not isinstance(outcome,OutcomeDistribution) or outcome.action_ids!=(action.action_id,)
                                or outcome.outage_probability.shape!=(1,1,len(local.times_s))
                                or not np.array_equal(outcome.times_s,local.times_s)):
                            raise ValueError('World-model outcome does not match this robot/action/step')
                        value=evaluate(outcome,mission,float(step_seconds),independently_qualified_heads=heads)
                        if self.clock()-started>self.deadline_seconds:
                            late=True;deadline_hit=True;stop=True;search_finished=False;break
                        missing.update(value.unavailable_heads)
                        if not value.feasible[0,0]:continue
                        cost=node.objective+float(value.objective[0,0])
                        if not math.isfinite(cost):raise ValueError('Cumulative feasible path cost overflowed')
                        following=_Path(action.ap,node.choices+(index,),cost)
                        if depth==steps-1:complete.append(following)
                        else:expanded.append(following)
                    if stop:break
                if stop:break
                if depth<steps-1:
                    expanded.sort(key=lambda p:(p.objective,p.choices))
                    if self.beam_width is not None and len(expanded)>self.beam_width:
                        pruned+=len(expanded)-self.beam_width
                        expanded=expanded[:self.beam_width]
                    beams=expanded
                    if not beams:break
            all_searched=all_searched and search_finished
            if complete:
                best=min(complete,key=lambda p:(p.objective,p.choices))
                result[robot]=best.choices;objective[robot]=best.objective
                reasons[robot]='complete_plan_requires_independent_existing_gate'
            elif deadline_hit:
                reasons[robot]='independent_fallback_required:deadline_exhausted'
            elif exhausted:
                reasons[robot]='independent_fallback_required:evaluation_budget_exhausted'
            elif missing:
                reasons[robot]='independent_fallback_required:unqualified_or_missing_head:'+','.join(sorted(missing))
            if deadline_hit or exhausted:
                # Remaining robots get fallback, never an incomplete prefix.
                for remaining in range(robot+1,n):reasons[remaining]=reasons[robot] if not complete else (
                    'independent_fallback_required:deadline_exhausted' if deadline_hit else
                    'independent_fallback_required:evaluation_budget_exhausted')
                all_searched=False;break
        elapsed=self.clock()-started
        return MultiStepProposal(result,tuple(a.action_id for a in actions),objective,tuple(reasons),
            float(step_seconds),evaluated,float(elapsed),bool(late),bool(exhausted),bool(all_searched),pruned)
