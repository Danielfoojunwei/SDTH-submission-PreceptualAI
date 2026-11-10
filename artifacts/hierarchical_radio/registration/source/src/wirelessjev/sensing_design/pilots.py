"""Greedy expected target-variance reduction per declared sensing cost.

This is a typed numerical design heuristic, with no RF benchmark, learned
covariance, data admission, optimality claim or hardware action side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .operators import GaussianPrior,_matrix,_readonly,_posterior_psd,expected_posterior_covariance,target_variance


@dataclass(frozen=True)
class PilotChoice:
    candidate_index: int
    candidate_id: str
    cost: float
    expected_target_variance_drop: float
    expected_drop_per_cost: float
    remaining_target_variance: float


@dataclass(frozen=True)
class PilotDesign:
    choices: tuple[PilotChoice,...]
    initial_target_variance: float
    remaining_target_variance: float
    total_cost: float
    budget: float
    cost_unit: str
    sum_choice_costs_before_roundoff: float
    cost_roundoff_adjustment: float
    cost_roundoff_tolerance: float
    prior_label: str
    posterior_covariance: np.ndarray
    stop_reason: str
    globally_optimal: bool = False


def greedy_targeted_pilots(candidate_operator,target_operator,prior:GaussianPrior,*,
                          noise_variance,costs,budget,cost_unit,candidate_ids=None,
                          existing_operator=None,existing_noise_variance=0.,maximum_choices=None):
    """Choose affordable independent-noise rows once each, recomputing gains.

    A duplicate noiseless row becomes zero-gain. Independent repeated noisy
    rows can remain useful. Costs/budget use a caller-specified common unit;
    naming that unit 'microseconds' does not measure real radio airtime.
    """
    if not isinstance(prior,GaussianPrior):raise ValueError('An explicit GaussianPrior is required')
    candidates=_matrix(candidate_operator,'Candidate operator');target=_matrix(target_operator,'Target operator')
    n,d=candidates.shape
    if d!=len(prior.covariance) or target.shape[1]!=d or not len(target):
        raise ValueError('Candidate/target/prior dimensions differ')
    cost=np.asarray(costs,dtype='f8');noise=np.asarray(noise_variance,dtype='f8')
    if cost.ndim==0:cost=np.full(n,float(cost))
    if noise.ndim==0:noise=np.full(n,float(noise))
    if (cost.shape!=(n,) or noise.shape!=(n,) or not np.isfinite(cost).all() or (cost<=0).any()
            or not np.isfinite(noise).all() or (noise<0).any()
            or isinstance(budget,bool) or not np.isfinite(budget) or budget<0
            or not isinstance(cost_unit,str) or not cost_unit.strip()):
        raise ValueError('Finite positive costs, nonnegative noise/budget and a cost unit are required')
    ids=tuple(str(k) for k in range(n)) if candidate_ids is None else tuple(candidate_ids)
    if len(ids)!=n or len(set(ids))!=n or any(not isinstance(k,str) or not k for k in ids):
        raise ValueError('Each candidate must have a distinct nonempty string ID')
    if maximum_choices is not None and (type(maximum_choices)is not int or maximum_choices<0):
        raise ValueError('Maximum choices must be a nonnegative integer')
    existing=np.zeros((0,d),complex) if existing_operator is None else existing_operator
    covariance=expected_posterior_covariance(prior,existing,existing_noise_variance)
    initial=target_variance(covariance,target);spent=0.;choices=[];selected=set();reason='no_affordable_positive_target_gain'
    cost_tolerance=float(np.finfo(float).eps*abs(budget)*8)
    raw_spent=0.
    limit=n if maximum_choices is None else min(n,maximum_choices)
    for _ in range(limit):
        best=None
        for index,row in enumerate(candidates):
            if index in selected:continue
            try:proposed=math.fsum([float(cost[k]) for k in selected]+[float(cost[index])])
            except OverflowError:continue
            if proposed-budget>cost_tolerance:continue
            with np.errstate(over='ignore',invalid='ignore'):
                v=covariance@row.conj();signal=float((row@v).real)
            if not np.isfinite(v).all() or not np.isfinite(signal):
                raise FloatingPointError('Candidate covariance arithmetic overflow')
            tolerance=np.finfo(float).eps*max(float(np.linalg.norm(covariance,2))*float(np.vdot(row,row).real),np.finfo(float).tiny)*64
            if signal < -tolerance:raise FloatingPointError('Negative candidate prior variance')
            denominator=max(signal,0.)+noise[index]
            if not np.isfinite(denominator):raise FloatingPointError('Candidate variance overflow')
            if denominator==0:
                # A PSD-zero observation has no information; actual impossible
                # observed values would require a separate consistency check.
                # Do not drop a small POSITIVE prior direction: the target
                # operator can amplify it into substantial target uncertainty.
                gain=0.
            else:
                with np.errstate(over='ignore',invalid='ignore'):
                    response=target@v;gain=float(np.vdot(response,response).real/denominator)
            with np.errstate(over='ignore',invalid='ignore'):
                ratio=gain/cost[index]
            if not np.isfinite(gain) or not np.isfinite(ratio):
                raise FloatingPointError('Target gain or gain-per-cost arithmetic overflow')
            if best is None or ratio>best[0]:best=(ratio,index,gain,v,denominator)
        if best is None or best[2]<=0:break
        ratio,index,gain,v,denominator=best
        previous=target_variance(covariance,target)
        reference_scale=max(float(np.linalg.norm(covariance,2)),np.finfo(float).tiny)
        covariance=np.asarray(covariance)-np.outer(v,v.conj())/denominator
        covariance=_posterior_psd(covariance,reference_scale,d)
        remaining=target_variance(covariance,target)
        if remaining>previous+1.e-10*max(initial,1.e-300):
            raise FloatingPointError('Target variance increased after sensing')
        selected.add(index);raw_spent=math.fsum(float(cost[k]) for k in selected)
        # Canonical reporting absorbs ONLY the explicitly bounded floating
        # cost-sum excess, e.g. 0.1+0.2 versus a declared budget of0.3.
        spent=min(raw_spent,float(budget))
        choices.append(PilotChoice(index,ids[index],float(cost[index]),gain,ratio,remaining))
    else:reason='maximum_choices' if maximum_choices is not None and limit<n else 'candidates_exhausted'
    return PilotDesign(tuple(choices),initial,target_variance(covariance,target),spent,float(budget),cost_unit,
        raw_spent,raw_spent-spent,cost_tolerance,prior.label,_readonly(covariance),reason)
