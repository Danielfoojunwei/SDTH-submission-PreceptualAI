"""Independent algebra checks between typed hierarchy levels."""
from __future__ import annotations
import math
import numpy as np
from .levels import SampledRF,LinkDistribution,AirtimeBudget,RouteRisk


def check_sample_link_consistency(samples:SampledRF, links:LinkDistribution, *, atol=1e-12):
    if samples.link_ids!=links.link_ids or set(samples.provenance.source_ids)!=set(links.provenance.source_ids):
        raise ValueError('Cross-level link identities or source provenance changed')
    mask=samples.observed;n=mask.sum(0)
    if not np.array_equal(n,links.sample_count):raise ValueError('Cross-level observed counts changed')
    if samples.log_power_dbm is not None:power=10.**(np.where(mask,samples.log_power_dbm,0.)/10.)
    elif samples.csi_power_scale_mw is not None:power=(np.abs(np.where(mask[...,None],samples.csi,0.))**2).mean(-1)*samples.csi_power_scale_mw
    else:raise ValueError('Arbitrary CSI cannot certify absolute-power coarsening')
    known=n>0
    expected=np.where(mask,power,0.).sum(0)[known]/n[known]
    if not np.allclose(expected,links.mean_power_mw[known],rtol=0,atol=atol):raise ValueError('Cross-level linear-power conservation failed')
    multiple=n>=2
    residual=np.where(mask,power-np.where(known,links.mean_power_mw,0.),0.)
    variance=(residual**2).sum(0)[multiple]/n[multiple]
    if not np.allclose(variance,links.empirical_variance_mw2[multiple],rtol=0,atol=atol):
        raise ValueError('Cross-level empirical power variation changed')
    if links.censored_count is not None:
        expected_censored=np.zeros(len(n),dtype=int) if samples.censor_upper_dbm is None else np.isfinite(samples.censor_upper_dbm).sum(0)
        if not np.array_equal(expected_censored,links.censored_count):raise ValueError('Censor bounds were lost or converted into point samples')
    return {'passes':True,'power_unit':'mW','independence_assumed':False,'qualified':False}


def check_airtime_consistency(budget:AirtimeBudget, *, atol=1e-12):
    if not np.allclose(budget.allocated_fraction.sum(-1)+budget.unallocated_fraction,1.,rtol=0,atol=atol):
        raise ValueError('Airtime budget failed domain/channel conservation')
    if budget.observed_idle_fraction is not None and np.any(budget.observed_idle_fraction>budget.unallocated_fraction+atol):
        raise ValueError('Idle airtime conflicts with allocated shares')
    return {'passes':True,'domains':len(budget.domain_channel_ids),'global_sum_constraint_applied':False,'qualified':False}


def check_route_consistency(route:RouteRisk, *, atol=1e-12):
    # For arbitrary dependence, max marginal <= P(union) <= sum marginals.
    lower=float(route.outage_probability.max());upper=min(1.,float(route.outage_probability.sum()))
    duration=float(route.times_s[-1]-route.times_s[0])
    if (not math.isclose(route.model_any_event_union_bound,upper,rel_tol=0,abs_tol=atol)
            or not 0<=route.outage_integral_seconds<=duration+atol):
        raise ValueError('Route probability/time coarsening failed')
    return {'passes':True,'any_event_frechet_lower':lower,'any_event_union_upper':upper,
            'temporal_independence_assumed':False,'continuous_time_silence_bound':None,
            'bounds_conditional_on_supplied_marginals':True,'qualified':False}
