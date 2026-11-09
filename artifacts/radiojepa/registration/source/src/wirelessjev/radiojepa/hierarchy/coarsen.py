"""Computable physical coarsening; no learned transition or actuation."""
from __future__ import annotations
import math
import numpy as np
from .levels import (SampledRF,LinkDistribution,AirtimeBudget,RouteRisk,
                     MissionSLA,MissionAssessment,Provenance)


def correlated_effective_count(correlation, observed=None):
    """n_eff=n^2/(1'R1) for equal-weight samples under a supplied covariance.

    This is a model-based variance-equivalent count, not extra independent
    experimental units or a confidence guarantee. Negative correlations can
    legitimately produce n_eff>n; a zero denominator is left unknown (NaN).
    R refers to the averaged scalar LINEAR POWER, with equal marginal variance
    assumed. Raw complex-channel/Jakes correlation is not power correlation.
    """
    r=np.asarray(correlation,dtype=float)
    if r.ndim!=2 or r.shape[0]!=r.shape[1] or not len(r) or not np.isfinite(r).all():
        raise ValueError('A finite square correlation matrix is required')
    if (not np.allclose(r,r.T,rtol=0,atol=1e-10) or not np.allclose(np.diag(r),1.,rtol=0,atol=1e-10)
            or np.linalg.eigvalsh(r).min() < -1e-9):
        raise ValueError('Correlation must be symmetric, unit-diagonal and positive semidefinite')
    mask=np.ones(len(r),bool) if observed is None else np.asarray(observed)
    if mask.shape!=(len(r),) or mask.dtype!=np.bool_: raise ValueError('Observed membership must be Boolean')
    n=int(mask.sum())
    if n==0:return float('nan')
    denominator=float(r[np.ix_(mask,mask)].sum())
    return n*n/denominator if denominator>1e-12 else float('nan')


def coarsen_samples(samples:SampledRF, *, correlation=None, threshold_mw=None,
                    max_spatial_diameter_m=None):
    """Average in linear mW, never dBm; exclude bounds/missing readings.

    Without a calibrated CSI scale, arbitrary CSI cannot become mW. Without
    correlation metadata, uncertainty in the mean remains unknown. Supplied
    power correlations give an equal-marginal-variance plug-in approximation,
    not a calibrated interval under arbitrary nonstationarity.
    """
    if not isinstance(samples,SampledRF):raise TypeError('Use typed sampled RF')
    mask=samples.observed; t,l=mask.shape
    if max_spatial_diameter_m is not None:
        if not math.isfinite(max_spatial_diameter_m) or max_spatial_diameter_m<0 or samples.positions_m is None:
            raise ValueError('A spatial average needs positions and a finite nonnegative window')
        point_positions=samples.positions_m[mask.any(-1)]
        diameter=np.linalg.norm(point_positions[:,None]-point_positions[None],axis=-1).max() if len(point_positions) else 0.
        if diameter>max_spatial_diameter_m+1e-12:raise ValueError('Samples exceed the declared local averaging window')
    if samples.log_power_dbm is not None:
        power=np.power(10.,np.where(mask,samples.log_power_dbm,0.)/10.)
    elif samples.csi_power_scale_mw is not None:
        power=np.mean(np.abs(np.where(mask[...,None],samples.csi,0.))**2,axis=-1)*samples.csi_power_scale_mw
    else:raise ValueError('Uncalibrated CSI has no identified absolute power in mW')
    if not np.isfinite(power[mask]).all():raise ValueError('Power conversion overflowed')
    n=mask.sum(0); known=n>0
    mean=np.full(l,np.nan); mean[known]=np.where(mask,power,0.).sum(0)[known]/n[known]
    variance=np.full(l,np.nan); multiple=n>=2
    centered=np.where(mask,power-np.where(known,mean,0.),0.)
    variance[multiple]=(centered**2).sum(0)[multiple]/n[multiple]
    neff=error=None
    if correlation is not None:
        r=np.asarray(correlation,dtype=float)
        if r.shape==(t,t):r=np.broadcast_to(r,(l,t,t))
        if r.shape!=(l,t,t):raise ValueError('Correlation must be T x T or link x T x T')
        neff=np.array([correlated_effective_count(r[k],mask[:,k]) for k in range(l)])
        error=np.full(l,np.nan); valid=multiple&np.isfinite(neff)
        error[valid]=np.sqrt(variance[valid]/neff[valid])
    probability=None;kind='unknown'
    if threshold_mw is not None:
        threshold=np.broadcast_to(np.asarray(threshold_mw,dtype=float),(l,))
        if not np.isfinite(threshold).all() or np.any(threshold<0):raise ValueError('RF thresholds must be nonnegative mW')
        probability=np.full(l,np.nan)
        probability[known]=((power<threshold)&mask).sum(0)[known]/n[known]
        kind='empirical_detected_threshold_fraction'
    censored=np.zeros(l,dtype=np.int64) if samples.censor_upper_dbm is None else np.isfinite(samples.censor_upper_dbm).sum(0)
    operation='linear_mW_point_average;selection_conditioned;no_IID_assumption'
    if correlation is not None:operation+=';power_correlation_equal_variance_plugin'
    provenance=samples.provenance.derived(operation)
    return LinkDistribution(samples.link_ids,mean,variance,n,provenance,neff,error,
                            probability,None,kind,censored)


def lee_average_power(samples, *, max_spatial_diameter_m, correlation=None):
    """A declared local power-average operation, not universal Lee-window lore.

    The caller chooses a physically defensible local stationarity window and
    correlation model. No wavelength-based independence is invented here.
    """
    return coarsen_samples(samples,correlation=correlation,
                           max_spatial_diameter_m=max_spatial_diameter_m)


def coarsen_airtime(link_airtime, ap_ids, contention_domains, channels, *, provenance,
                    observed_idle=None):
    """Group allocated link resources into AP/domain/channel budgets.

    Offered load must not be passed off as allocated occupancy. Overload raises
    instead of being silently normalized; unallocated remainder is not idle.
    """
    shares=np.asarray(link_airtime,dtype=float);aps=tuple(ap_ids);domains=tuple(contention_domains);bands=tuple(channels)
    if shares.ndim!=1 or not len(shares) or not (len(aps)==len(domains)==len(bands)==len(shares)) or not np.isfinite(shares).all() or np.any(shares<0):
        raise ValueError('Finite nonnegative allocated airtime and matching link metadata required')
    if any(not isinstance(x,str) or not x for x in aps+domains+bands):raise ValueError('AP/domain/channel identities are required')
    keys=tuple(sorted(set(zip(domains,bands)))); names=tuple(sorted(set(aps)))
    grouped=np.zeros((len(keys),len(names)))
    ki,ai={key:i for i,key in enumerate(keys)},{name:i for i,name in enumerate(names)}
    for share,ap,domain,band in zip(shares,aps,domains,bands):grouped[ki[domain,band],ai[ap]]+=share
    if np.any(grouped.sum(-1)>1.+1e-12):raise ValueError('Allocated airtime exceeds a contention-domain/channel budget')
    idle=None
    if observed_idle is not None:
        if set(observed_idle)!=set(keys):raise ValueError('Observed idle must cover exactly the declared domain/channel keys')
        idle=np.array([observed_idle[key] for key in keys],dtype=float)
    return AirtimeBudget(keys,names,grouped,np.maximum(0.,1.-grouped.sum(-1)),
                         provenance.derived('per_domain_channel_airtime_aggregation'),idle)


def coarsen_route(links:LinkDistribution, link_indices, times_s, *, event_ids=None,
                  positions_m=None, assume_representative=False):
    """Select link marginals along a route and apply Boole's union bound.

    Empirical detected-only fractions require an explicit representativeness
    assumption before they can be treated as model marginals. This remains
    unqualified and does not turn an RF event into a command-loss probability.
    """
    if not isinstance(links,LinkDistribution) or links.outage_probability is None:
        raise ValueError('Route risk requires supplied link event probabilities')
    if links.probability_kind=='empirical_detected_threshold_fraction' and assume_representative is not True:
        raise ValueError('Detected-only RF frequencies are not unconditional event probabilities')
    index=np.asarray(link_indices);times=np.asarray(times_s,dtype=float)
    if index.ndim!=1 or not len(index) or index.dtype.kind not in 'iu' or times.shape!=index.shape or np.any((index<0)|(index>=len(links.link_ids))):
        raise ValueError('A route requires matching event times and physical link indices')
    if not np.isfinite(times).all() or np.any(times<0) or len(np.unique(times))!=len(times):raise ValueError('Route event times must be distinct, finite and nonnegative')
    order=np.argsort(times); index=index[order];times=times[order]
    if event_ids is not None and np.asarray(event_ids,dtype=object).shape!=order.shape:
        raise ValueError('Each route event needs exactly one identity')
    ids=tuple(f'{links.link_ids[k]}@{t.hex()}' for k,t in zip(index,times)) if event_ids is None else tuple(np.asarray(event_ids,dtype=object)[order].tolist())
    p=links.outage_probability[index]
    if not np.isfinite(p).all():raise ValueError('Unknown link marginals cannot become route confidence')
    upper=None if links.outage_upper is None else links.outage_upper[index]
    position=None
    if positions_m is not None:
        position=np.asarray(positions_m,dtype=float)
        if position.shape not in ((len(order),2),(len(order),3)) or not np.isfinite(position).all():
            raise ValueError('Each route event needs a finite physical XY/XYZ position')
        position=position[order]
    integral=float(np.trapezoid(p,times)) if len(times)>1 else 0.
    return RouteRisk(ids,times,p,integral,min(1.,float(p.sum())),
        links.provenance.derived('finite_event_union_bound;piecewise_linear_outage_integral'),
        upper,None if upper is None else min(1.,float(upper.sum())),position,links.probability_kind)


def assess_mission(route:RouteRisk, mission:MissionSLA, *, silence_upper_s=None,
                    silence_externally_qualified=False):
    """Evaluate a typed model request; unknown calibration/silence stays unknown."""
    if not isinstance(route,RouteRisk) or not isinstance(mission,MissionSLA):raise TypeError('Typed route risk and mission SLA required')
    checks={};reasons=[]
    qualified=route.provenance.qualified
    if mission.any_event_probability_limit is not None:
        if mission.require_external_qualification and (not qualified or route.any_event_upper is None):
            checks['any_registered_event']=None;reasons.append('No externally qualified finite-event upper bound')
        else:
            p=route.any_event_upper if route.any_event_upper is not None else route.model_any_event_union_bound
            checks['any_registered_event']=bool(p<=mission.any_event_probability_limit)
    if mission.outage_integral_limit_s is not None:
        if mission.require_external_qualification and not qualified:
            checks['outage_integral']=None;reasons.append('Outage interpolant is not externally qualified')
        else: checks['outage_integral']=bool(route.outage_integral_seconds<=mission.outage_integral_limit_s)
    if mission.silence_limit_s is not None:
        if silence_upper_s is not None and (not math.isfinite(silence_upper_s) or silence_upper_s<0):raise ValueError('Silence bound must be nonnegative finite seconds')
        if silence_upper_s is None or mission.require_external_qualification and silence_externally_qualified is not True:
            checks['silence']=None;reasons.append('Outage marginals/integrals do not identify continuous-time silence')
        else:checks['silence']=bool(silence_upper_s<=mission.silence_limit_s)
    feasible=False if any(v is False for v in checks.values()) else (None if any(v is None for v in checks.values()) else True)
    return MissionAssessment(mission.mission_id,feasible,checks,tuple(reasons))
