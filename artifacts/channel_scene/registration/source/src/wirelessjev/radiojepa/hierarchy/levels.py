"""Typed physical summaries at five scales, with unknowns kept explicit.

These schemas do not implement a learned hierarchy or grant deployment
permission. A qualification field is external provenance, not self-proof.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

_SHA = re.compile(r'[0-9a-f]{64}\Z')


def frozen_array(value, *, dtype=None, ndim=None, shape=None, finite=True, name='array'):
    array = np.array(value, dtype=dtype, copy=True)
    if ndim is not None and array.ndim != ndim or shape is not None and array.shape != shape:
        raise ValueError(name + ': incompatible dimensions')
    if finite and not np.isfinite(array).all():
        raise ValueError(name + ': finite values required')
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def identifiers(value, name):
    if isinstance(value,(str,bytes)):raise ValueError(name + ': use a collection of identifiers')
    result = tuple(value)
    if not result or len(set(result)) != len(result) or any(not isinstance(x,str) or not x for x in result):
        raise ValueError(name + ': nonempty distinct identifiers required')
    return result


def probability(value, name):
    if isinstance(value, (bool,np.bool_)) or not isinstance(value,(int,float,np.number)):
        raise ValueError(name + ': numeric probability required')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(name + ': probability outside [0,1]')
    return value


@dataclass(frozen=True)
class Provenance:
    source_ids: tuple[str, ...]
    coordinate_frame: str | None = None
    processing: tuple[str, ...] = ()
    qualified: bool = False
    qualification_receipt: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'source_ids', identifiers(self.source_ids,'source_ids'))
        if self.coordinate_frame is not None and (not isinstance(self.coordinate_frame,str) or not self.coordinate_frame):
            raise ValueError('Coordinate frame must be identified when present')
        if type(self.qualified) is not bool:
            raise ValueError('Qualification metadata must be Boolean')
        if self.qualification_receipt is not None and (not isinstance(self.qualification_receipt,str) or not _SHA.fullmatch(self.qualification_receipt)):
            raise ValueError('Qualification provenance must be a SHA256 reference')
        if self.qualified and self.qualification_receipt is None:
            raise ValueError('Qualification cannot be asserted without external provenance')
        processing = tuple(self.processing)
        if any(not isinstance(x,str) or not x for x in processing):
            raise ValueError('Processing provenance must use explicit names')
        object.__setattr__(self,'processing',processing)

    def derived(self, operation):
        # A lower-level calibration does not automatically qualify a different
        # aggregation, prediction horizon, device or downstream capability.
        return Provenance(self.source_ids,self.coordinate_frame,self.processing+(operation,))


@dataclass(frozen=True)
class SampledRF:
    """L0: masked samples, with dBm distinct from arbitrary-unit CSI.

    CSI becomes mW only if its per-link calibrated scale is supplied. Censored
    readings remain bounds; they are never substituted into point averages.
    """
    sample_ids: tuple[str, ...]
    capture_time_s: np.ndarray
    link_ids: tuple[str, ...]
    observed: np.ndarray
    provenance: Provenance
    log_power_dbm: np.ndarray | None = None
    csi: np.ndarray | None = None
    csi_power_scale_mw: np.ndarray | None = None
    positions_m: np.ndarray | None = None
    censor_upper_dbm: np.ndarray | None = None

    def __post_init__(self):
        samples,links = identifiers(self.sample_ids,'samples'),identifiers(self.link_ids,'links')
        shape = (len(samples),len(links))
        mask = np.asarray(self.observed)
        if mask.dtype != np.bool_ or mask.shape != shape:
            raise ValueError('Observed mask must be Boolean sample x link')
        if self.log_power_dbm is None and self.csi is None:
            raise ValueError('L0 requires actual log-power or CSI samples')
        object.__setattr__(self,'sample_ids',samples); object.__setattr__(self,'link_ids',links)
        object.__setattr__(self,'capture_time_s',frozen_array(self.capture_time_s,dtype=float,shape=(shape[0],),name='capture times'))
        object.__setattr__(self,'observed',frozen_array(mask))
        if not isinstance(self.provenance,Provenance): raise ValueError('Explicit provenance required')
        if self.log_power_dbm is not None:
            power = np.asarray(self.log_power_dbm,dtype=float)
            if power.shape != shape or not np.isfinite(power[mask]).all():
                raise ValueError('Observed dBm must be finite on sample x link')
            object.__setattr__(self,'log_power_dbm',frozen_array(np.where(mask,power,np.nan),finite=False))
        if self.csi is not None:
            csi = np.asarray(self.csi)
            if csi.ndim != 3 or csi.shape[:2] != shape or csi.shape[-1] < 1 or not np.iscomplexobj(csi) or not np.isfinite(csi[mask]).all():
                raise ValueError('CSI requires finite observed complex sample x link x tone values')
            object.__setattr__(self,'csi',frozen_array(np.where(mask[...,None],csi,np.nan+1j*np.nan),finite=False))
        if self.csi_power_scale_mw is not None:
            if self.csi is None: raise ValueError('A CSI scale without CSI is meaningless')
            scale = frozen_array(self.csi_power_scale_mw,dtype=float,shape=(shape[1],),name='CSI mW scale')
            if np.any(scale <= 0): raise ValueError('Calibrated CSI power scale must be positive')
            object.__setattr__(self,'csi_power_scale_mw',scale)
        if self.positions_m is not None:
            pos = frozen_array(self.positions_m,dtype=float,ndim=2,name='positions')
            if pos.shape[0] != shape[0] or pos.shape[1] not in (2,3) or self.provenance.coordinate_frame is None:
                raise ValueError('Physical positions require an identified XY/XYZ frame')
            object.__setattr__(self,'positions_m',pos)
        if self.censor_upper_dbm is not None:
            upper = np.asarray(self.censor_upper_dbm,dtype=float)
            if upper.shape != shape or np.isinf(upper).any() or np.any(np.isfinite(upper)&mask):
                raise ValueError('Finite censor bounds apply only to non-point-observed entries')
            object.__setattr__(self,'censor_upper_dbm',frozen_array(upper,finite=False))


@dataclass(frozen=True)
class LinkDistribution:
    """L1: empirical point-observed power statistics, not an identified law.

    Unknown mean/variance entries are NaN; missing correlation means unknown
    effective count/mean error. Empirical threshold frequency is conditional
    on observation selection and is not automatically a service probability.
    """
    link_ids: tuple[str, ...]
    mean_power_mw: np.ndarray
    empirical_variance_mw2: np.ndarray
    sample_count: np.ndarray
    provenance: Provenance
    effective_sample_count: np.ndarray | None = None
    mean_se_plugin_mw: np.ndarray | None = None
    outage_probability: np.ndarray | None = None
    outage_upper: np.ndarray | None = None
    probability_kind: str = 'unknown'
    censored_count: np.ndarray | None = None

    def __post_init__(self):
        links = identifiers(self.link_ids,'links'); n=len(links)
        count = np.asarray(self.sample_count)
        if count.shape != (n,) or count.dtype.kind not in 'iu' or np.any(count < 0):
            raise ValueError('Counts must be nonnegative integral link counts')
        known = count > 0
        mean = frozen_array(self.mean_power_mw,dtype=float,shape=(n,),finite=False,name='mean mW')
        var = frozen_array(self.empirical_variance_mw2,dtype=float,shape=(n,),finite=False,name='variance mW2')
        if (not np.isfinite(mean[known]).all() or np.any(mean[known] < 0) or not np.isnan(mean[~known]).all()
                or not np.isfinite(var[count>=2]).all() or np.any(var[count>=2]<0) or not np.isnan(var[count<2]).all()):
            raise ValueError('Unobserved moments must remain unknown; known powers/variances are nonnegative')
        object.__setattr__(self,'link_ids',links); object.__setattr__(self,'sample_count',frozen_array(count,dtype=np.int64))
        object.__setattr__(self,'mean_power_mw',mean);object.__setattr__(self,'empirical_variance_mw2',var)
        for name in ('effective_sample_count','mean_se_plugin_mw','outage_probability','outage_upper'):
            value = getattr(self,name)
            if value is None: continue
            array = frozen_array(value,dtype=float,shape=(n,),finite=False,name=name)
            finite = np.isfinite(array)
            if np.isinf(array).any() or np.any(finite&~known) or np.any(array[finite]<0):
                raise ValueError('Unknown uncertainty cannot become a fabricated value')
            if name == 'effective_sample_count' and np.any(array[finite]<=0): raise ValueError('Effective counts must be positive')
            if name.startswith('outage') and (np.any(array[finite]>1) or not np.isfinite(array[known]).all()):
                raise ValueError('Known outage marginals must be probabilities')
            object.__setattr__(self,name,array)
        if self.outage_upper is not None and (self.outage_probability is None or np.any(self.outage_upper[known]<self.outage_probability[known])):
            raise ValueError('Probability upper bounds must include the supplied marginal')
        if self.mean_se_plugin_mw is not None:
            if self.effective_sample_count is None:raise ValueError('Mean-error approximation requires a supplied correlation model')
            finite=np.isfinite(self.mean_se_plugin_mw)
            if np.any(finite&((count<2)|~np.isfinite(self.effective_sample_count))):
                raise ValueError('Insufficient or singular evidence cannot receive a finite mean error')
        if self.probability_kind not in ('unknown','empirical_detected_threshold_fraction','model_marginal'):
            raise ValueError('Probability meaning must be declared')
        if self.outage_probability is None and self.probability_kind != 'unknown': raise ValueError('Probability kind requires probabilities')
        if self.outage_probability is not None and self.probability_kind == 'unknown': raise ValueError('Probability semantics cannot be unknown when values are supplied')
        if self.censored_count is not None:
            c = np.asarray(self.censored_count)
            if c.shape!=(n,) or c.dtype.kind not in 'iu' or np.any(c<0): raise ValueError('Invalid censor counts')
            object.__setattr__(self,'censored_count',frozen_array(c,dtype=np.int64))
        if not isinstance(self.provenance,Provenance): raise ValueError('Explicit provenance required')


@dataclass(frozen=True)
class AirtimeBudget:
    """L2: allocated resource fractions per contention-domain/channel.

    Unallocated capacity is not asserted to be idle. Multiple spatial-reuse
    domains or simultaneous links can legitimately sum above one globally.
    """
    domain_channel_ids: tuple[tuple[str,str], ...]
    ap_ids: tuple[str, ...]
    allocated_fraction: np.ndarray
    unallocated_fraction: np.ndarray
    provenance: Provenance
    observed_idle_fraction: np.ndarray | None = None

    def __post_init__(self):
        domains=tuple(self.domain_channel_ids); aps=identifiers(self.ap_ids,'APs')
        if not domains or any(not isinstance(x,tuple) or len(x)!=2 for x in domains) or len(set(domains))!=len(domains) or any(any(not isinstance(s,str) or not s for s in x) for x in domains):
            raise ValueError('Explicit distinct contention-domain/channel keys required')
        allocated=frozen_array(self.allocated_fraction,dtype=float,shape=(len(domains),len(aps)),name='allocated airtime')
        unallocated=frozen_array(self.unallocated_fraction,dtype=float,shape=(len(domains),),name='unallocated airtime')
        if np.any(allocated<0) or np.any(unallocated<0) or not np.allclose(allocated.sum(-1)+unallocated,1.,rtol=0,atol=1e-12):
            raise ValueError('Airtime must be conserved within each declared domain/channel')
        object.__setattr__(self,'domain_channel_ids',domains);object.__setattr__(self,'ap_ids',aps)
        object.__setattr__(self,'allocated_fraction',allocated);object.__setattr__(self,'unallocated_fraction',unallocated)
        if self.observed_idle_fraction is not None:
            idle=frozen_array(self.observed_idle_fraction,dtype=float,shape=unallocated.shape,name='observed idle')
            if np.any(idle<0) or np.any(idle>unallocated+1e-12): raise ValueError('Observed idle conflicts with the allocated resource budget')
            object.__setattr__(self,'observed_idle_fraction',idle)
        if not isinstance(self.provenance,Provenance): raise ValueError('Explicit provenance required')


@dataclass(frozen=True)
class RouteRisk:
    """L3: risk at a finite registered event grid and its declared interpolant.

    A union bound over this grid is not a continuous-time silence bound. No
    independence is assumed. Coverage/calibration must be supplied externally.
    """
    event_ids: tuple[str, ...]
    times_s: np.ndarray
    outage_probability: np.ndarray
    outage_integral_seconds: float
    model_any_event_union_bound: float
    provenance: Provenance
    event_upper: np.ndarray | None = None
    any_event_upper: float | None = None
    positions_m: np.ndarray | None = None
    probability_kind: str = 'model_marginal'

    def __post_init__(self):
        events=identifiers(self.event_ids,'route events');k=len(events)
        times=frozen_array(self.times_s,dtype=float,shape=(k,),name='route times')
        p=frozen_array(self.outage_probability,dtype=float,shape=(k,),name='route probabilities')
        if np.any(times<0) or (k>1 and not np.all(np.diff(times)>0)) or np.any((p<0)|(p>1)):
            raise ValueError('Route event times must increase; probabilities must be in [0,1]')
        expected=float(np.trapezoid(p,times)) if k>1 else 0.
        union=min(1.,float(p.sum()))
        probability(self.model_any_event_union_bound,'Route union bound')
        if (isinstance(self.outage_integral_seconds,(bool,np.bool_)) or not math.isfinite(self.outage_integral_seconds) or not math.isclose(self.outage_integral_seconds,expected,rel_tol=0,abs_tol=1e-12)
                or not math.isclose(self.model_any_event_union_bound,union,rel_tol=0,abs_tol=1e-12)):
            raise ValueError('Route summaries do not match the declared piecewise-linear marginals/union bound')
        object.__setattr__(self,'event_ids',events);object.__setattr__(self,'times_s',times);object.__setattr__(self,'outage_probability',p)
        if self.event_upper is not None:
            upper=frozen_array(self.event_upper,dtype=float,shape=(k,),name='event upper')
            bound=min(1.,float(upper.sum()))
            if self.any_event_upper is not None:probability(self.any_event_upper,'Route event upper')
            if np.any((upper<p)|(upper>1)) or self.any_event_upper is None or not math.isclose(self.any_event_upper,bound,rel_tol=0,abs_tol=1e-12):
                raise ValueError('Route upper bound must follow the supplied event upper bounds')
            object.__setattr__(self,'event_upper',upper)
        elif self.any_event_upper is not None: raise ValueError('An upper bound cannot appear without event upper bounds')
        if not isinstance(self.provenance,Provenance): raise ValueError('Explicit provenance required')
        if self.probability_kind not in ('model_marginal','empirical_detected_threshold_fraction'):
            raise ValueError('Route probability semantics must remain explicit')
        if self.positions_m is not None:
            pos=frozen_array(self.positions_m,dtype=float,ndim=2,name='route positions')
            if pos.shape[0]!=k or pos.shape[1] not in (2,3) or self.provenance.coordinate_frame is None:
                raise ValueError('Route positions need a common physical coordinate frame')
            object.__setattr__(self,'positions_m',pos)


@dataclass(frozen=True)
class MissionSLA:
    """L4: explicit any-registered-event limits, not ambiguous text/STL.

    This differs from a per-command/pointwise chance constraint. Converting
    between those semantics requires an explicit risk-allocation policy.
    """
    mission_id: str
    any_event_probability_limit: float | None = None
    outage_integral_limit_s: float | None = None
    silence_limit_s: float | None = None
    require_external_qualification: bool = True

    def __post_init__(self):
        if not isinstance(self.mission_id,str) or not self.mission_id: raise ValueError('Mission identity required')
        if self.any_event_probability_limit is not None:
            object.__setattr__(self,'any_event_probability_limit',probability(self.any_event_probability_limit,'Mission event limit'))
        for name in ('outage_integral_limit_s','silence_limit_s'):
            value=getattr(self,name)
            if value is not None and (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number)) or not math.isfinite(value) or value<0):
                raise ValueError('Mission time limits must be nonnegative finite seconds')
        if all(getattr(self,name) is None for name in ('any_event_probability_limit','outage_integral_limit_s','silence_limit_s')):
            raise ValueError('At least one explicit SLA constraint is required')
        if type(self.require_external_qualification) is not bool: raise ValueError('Qualification requirement must be Boolean')


@dataclass(frozen=True)
class MissionAssessment:
    mission_id: str
    model_feasible: bool | None
    checks: Mapping[str,bool | None]
    unknown_reasons: tuple[str, ...]
    qualified: bool = False
    safety_guaranteed: bool = False

    def __post_init__(self):
        if self.qualified is not False or self.safety_guaranteed is not False:
            raise ValueError('Hierarchy arithmetic cannot self-qualify or issue a safety guarantee')
        values=dict(self.checks)
        if not isinstance(self.mission_id,str) or not self.mission_id or not values or any(not isinstance(k,str) or not k for k in values):
            raise ValueError('Assessment requires a mission and explicit named constraints')
        if any(type(v) is not bool and v is not None for v in values.values()): raise ValueError('Checks must be Boolean or unknown')
        expected=False if any(v is False for v in values.values()) else (None if any(v is None for v in values.values()) else True)
        if self.model_feasible is not expected: raise ValueError('Assessment must preserve unknown/infeasible constraints')
        object.__setattr__(self,'checks',MappingProxyType(values))
