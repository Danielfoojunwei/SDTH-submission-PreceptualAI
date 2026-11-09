"""Opt-in RSSI/service boundary and a registered native DEV comparison.

No RF-only calibration authorizes a packet-risk head.  Default query behavior
is the existing RadioMap PHY surrogate.  Nothing here actuates a real device,
opens a SERVICE holdout, or changes a frozen simulator.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
import time

import numpy as np

from ..worldmodel.interface import (BANDS, CandidateAction, OutcomeDistribution,
    SceneState, immutable_array)
from ..worldmodel.radiomap_adapter import RadioMapAdapter, integrate

_SHA=re.compile('[0-9a-f]{64}\\Z')
NATIVE_CARRIERS_HZ=(2.4e9,5.2e9,6.5e9)
CALIBRATION_PARTITIONS=('training_calibration','validation_calibration')
PACKET_ENDPOINT='control_deadline_miss_600B_20Hz_50ms'


def _identity(value,name):
    if not isinstance(value,str) or not value.strip():raise ValueError(name+' must be explicit')
    return value


def _sha(value,name):
    if not isinstance(value,str) or not _SHA.fullmatch(value):raise ValueError(name+' must be SHA256')
    return value


def _positive(value,name):
    if isinstance(value,(bool,np.bool_)) or not math.isfinite(value) or value<=0:
        raise ValueError(name+' must be finite and positive')
    return float(value)


def _fingerprint(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RadioAssumptions:
    """Declared conversion of device0-gauge downlink RSSI to service SINR.

    device0_bias_db is an externally identified OR explicitly assumed absolute
    bias, never inferred by relative frontend calibration alone.  Interference
    is additive linear mW, distinct from the receiver's thermal/noise floor.
    All three bands must name their actual carrier; 6.0 != native 6.5 GHz.
    """
    carrier_hz: np.ndarray
    forecast_transmit_power_dbm: np.ndarray
    service_transmit_power_dbm: np.ndarray
    receiver_noise_dbm: np.ndarray
    device0_bias_db: np.ndarray
    antenna_gain_change_db: np.ndarray
    interference_mw: np.ndarray
    absolute_reference: str
    interference_source: str
    reciprocal_path_gain: bool

    def __post_init__(self):
        for name in ('carrier_hz','forecast_transmit_power_dbm','service_transmit_power_dbm',
                     'receiver_noise_dbm','device0_bias_db','antenna_gain_change_db'):
            value=immutable_array(getattr(self,name),shape=(3,),name=name)
            if name=='carrier_hz' and np.any(value<=0):raise ValueError('Actual positive carriers required')
            object.__setattr__(self,name,value)
        interference=immutable_array(self.interference_mw,name='interference mW')
        if interference.shape!=(3,) and (interference.ndim!=4 or interference.shape[-1]!=3):
            raise ValueError('Interference must be three bands or N×H×AP×3')
        if np.any(interference<0):raise ValueError('Interference mW cannot be negative')
        object.__setattr__(self,'interference_mw',interference)
        _identity(self.absolute_reference,'absolute device0 reference assumption')
        _identity(self.interference_source,'interference observation/assumption source')
        if self.reciprocal_path_gain is not True:
            raise ValueError('Downlink→uplink conversion requires an explicit path-gain reciprocity assumption')

    @property
    def sha256(self):
        return _fingerprint({name:getattr(self,name).tolist() if isinstance(getattr(self,name),np.ndarray)
                             else getattr(self,name) for name in self.__dataclass_fields__})

    def convert(self,rssi_dbm):
        value=immutable_array(rssi_dbm,ndim=4,name='predicted device0-gauge RSSI')
        if value.shape[-1]!=3:raise ValueError('RSSI requires three explicitly named bands')
        try:denominator=np.power(10.,self.receiver_noise_dbm/10)+np.broadcast_to(self.interference_mw,value.shape)
        except ValueError as exc:raise ValueError('Interference axes differ from this forecast') from exc
        if not np.isfinite(denominator).all() or np.any(denominator<=0):raise ValueError('Finite positive N+I required')
        received=(value-self.device0_bias_db-self.forecast_transmit_power_dbm
                  +self.service_transmit_power_dbm+self.antenna_gain_change_db)
        result=received-10*np.log10(denominator)
        if not np.isfinite(result).all():raise ValueError('Invalid converted SINR')
        return result


@dataclass(frozen=True)
class RSSIForecast:
    times_s: np.ndarray
    current_ap: np.ndarray
    mean_dbm: np.ndarray
    std_db: np.ndarray
    carrier_hz: np.ndarray
    source_id: str
    source_receipt_sha256: str
    gauge: str='device0_relative'
    direction: str='downlink'

    def __post_init__(self):
        # Reuse the public boundary's exact N/H/AP/band/association validation.
        shape=SceneState(self.times_s,self.current_ap,self.mean_dbm,self.source_id)
        std=immutable_array(self.std_db,shape=shape.snr_db.shape,name='predicted RSSI std')
        if np.any(std<0):raise ValueError('Predicted spread cannot be negative')
        carrier=immutable_array(self.carrier_hz,shape=(3,),name='forecast carriers')
        if np.any(carrier<=0):raise ValueError('Positive carriers required')
        if self.gauge!='device0_relative' or self.direction!='downlink':
            raise ValueError('Only explicitly device0-gauge downlink RSSI is implemented')
        _sha(self.source_receipt_sha256,'forecast source receipt')
        for name,value in (('times_s',shape.times_s),('current_ap',shape.current_ap),
                           ('mean_dbm',shape.snr_db),('std_db',std),('carrier_hz',carrier)):
            object.__setattr__(self,name,value)


@dataclass(frozen=True)
class CalibrationScope:
    model_id: str
    site_id: str
    device_id: str
    physical_sha256: str
    endpoint: str
    mac_traffic_sha256: str | None=None

    def __post_init__(self):
        for name in ('model_id','site_id','device_id','endpoint'):_identity(getattr(self,name),name)
        _sha(self.physical_sha256,'physical scope')
        if self.endpoint==PACKET_ENDPOINT:_sha(self.mac_traffic_sha256,'packet MAC/traffic scope')
        elif self.endpoint!='rssi_dbm':raise ValueError('Unsupported calibration endpoint; silence/latency unavailable')


@dataclass(frozen=True)
class HeldoutCalibration:
    """A receipt contract; its referenced bytes must be checked by the caller.

    Units are independent held-out scenes/episodes, not correlated robots.
    The caller supplies the complete backbone-fit and selection membership.
    Neither a receipt nor this fit method grants independent gate permission.
    """
    partition: str
    unit_ids: tuple[str,...]
    fit_unit_ids: tuple[str,...]
    selection_unit_ids: tuple[str,...]
    receipt_sha256: str
    scope: CalibrationScope

    def __post_init__(self):
        if self.partition not in CALIBRATION_PARTITIONS:
            raise ValueError('Only admitted TRAIN/VAL held-out calibration is allowed; never FIELD/SERVICE')
        for name in ('unit_ids','fit_unit_ids','selection_unit_ids'):
            values=tuple(getattr(self,name))
            if len(set(values))!=len(values) or any(not isinstance(v,str) or not v for v in values):
                raise ValueError('Explicit distinct whole-unit membership required')
            object.__setattr__(self,name,values)
        if not self.unit_ids or not self.fit_unit_ids:raise ValueError('Calibration and model-fit membership required')
        if set(self.unit_ids)&(set(self.fit_unit_ids)|set(self.selection_unit_ids)):
            raise ValueError('Calibration units must be disjoint from model fitting and selection')
        _sha(self.receipt_sha256,'held-out calibration receipt')
        if not isinstance(self.scope,CalibrationScope):raise TypeError('Typed calibration scope required')


def _calibration_arrays(prediction,observation,evidence):
    if not isinstance(evidence,HeldoutCalibration):raise TypeError('Admitted held-out receipt required')
    p=immutable_array(prediction,name='held-out predictions')
    y=immutable_array(observation,shape=p.shape,name='held-out observations')
    if p.ndim<2 or p.shape[0]!=len(evidence.unit_ids) or not p[0].size:
        raise ValueError('First calibration axis must index whole independent units')
    return p.reshape(len(p),-1),y.reshape(len(y),-1)


@dataclass(frozen=True)
class ResidualBound:
    evidence: HeldoutCalibration
    alpha: float
    lower_margin_db: float
    empirical_unit_coverage: float

    def __post_init__(self):
        if (not isinstance(self.evidence,HeldoutCalibration) or self.evidence.scope.endpoint!='rssi_dbm'
                or isinstance(self.alpha,bool) or not 0<self.alpha<1
                or not (math.isfinite(self.lower_margin_db) or self.lower_margin_db==math.inf)
                or self.lower_margin_db<0 or not 0<=self.empirical_unit_coverage<=1):
            raise ValueError('Invalid held-out RSSI residual bound')

    @classmethod
    def fit(cls,prediction,observation,*,evidence,alpha=.1):
        if evidence.scope.endpoint!='rssi_dbm':raise ValueError('RF residuals require the RSSI endpoint')
        if not 0<alpha<1:raise ValueError('Alpha must be in (0,1)')
        p,y=_calibration_arrays(prediction,observation,evidence)
        scores=np.maximum(0.,(p-y).max(1))
        k=math.ceil((len(scores)+1)*(1-alpha))
        quantile=float(np.sort(scores)[k-1]) if k<=len(scores) else math.inf
        return cls(evidence,float(alpha),quantile,float(np.mean(scores<=quantile)))

    def lower(self,prediction,*,scope):
        if scope!=self.evidence.scope or not math.isfinite(self.lower_margin_db):
            raise ValueError('RSSI calibration missing, out of scope, or finite-sample bound uninformative')
        value=np.asarray(prediction,dtype=float)
        if not np.isfinite(value).all():raise ValueError('Finite RF prediction required')
        return value-self.lower_margin_db

    @property
    def metadata(self):
        return {'endpoint':'rssi_dbm','calibration_units':len(self.evidence.unit_ids),
            'receipt_sha256':self.evidence.receipt_sha256,'alpha':self.alpha,
            'lower_margin_db':self.lower_margin_db,'empirical_unit_coverage':self.empirical_unit_coverage,
            'coverage_guaranteed_under_drift':False,'qualifies_packet_risk':False}


def action_signature(action):
    if not isinstance(action,CandidateAction):raise TypeError('Typed association action required')
    return f'{action.ap}|{action.band}|{action.mcs}|{float(action.handover_seconds).hex()}'


@dataclass(frozen=True)
class PacketRiskCalibration:
    """Fixed action/score bins, measured labels, whole-unit mean and UCB.

    A Hoeffding union bound concerns the mean per-unit failure rate IN each
    supported fixed action/score cell, assuming independent identically
    distributed units. Actions are never pooled: a low-risk action cannot
    provide support for a high-risk action with the same numerical score.
    This is not a per-query conditional probability or SLA theorem.
    Empty/undersupported cells refuse; no RF-only synthetic packet labels.
    """
    evidence: HeldoutCalibration
    edges: np.ndarray
    mean_failure: np.ndarray
    upper_failure: np.ndarray
    unit_counts: np.ndarray
    action_signatures: tuple[str,...]
    alpha: float
    min_units: int

    def __post_init__(self):
        if (not isinstance(self.evidence,HeldoutCalibration) or self.evidence.scope.endpoint!=PACKET_ENDPOINT
                or isinstance(self.alpha,bool) or not 0<self.alpha<1
                or type(self.min_units) is not int or self.min_units<2):
            raise ValueError('Actual scoped packet calibration and independent unit support required')
        edges=immutable_array(self.edges,ndim=1,name='fixed score bins')
        if len(edges)<2 or edges[0]!=0 or edges[-1]!=1 or np.any(np.diff(edges)<=0):
            raise ValueError('Strict score-bin edges must span [0,1]')
        if (not self.action_signatures or len(set(self.action_signatures))!=len(self.action_signatures)
                or any(not isinstance(s,str) or not s for s in self.action_signatures)):
            raise ValueError('Exact supported action membership required')
        object.__setattr__(self,'action_signatures',tuple(self.action_signatures))
        shape=(len(self.action_signatures),len(edges)-1)
        for name in ('mean_failure','upper_failure'):
            a=immutable_array(getattr(self,name),shape=shape,name=name)
            if np.any((a<0)|(a>1)):raise ValueError('Risk estimates must stay in [0,1]')
            object.__setattr__(self,name,a)
        count=np.asarray(self.unit_counts)
        if count.shape!=shape or count.dtype.kind not in 'iu' or np.any(count<0):raise ValueError('Whole-unit action/bin counts required')
        object.__setattr__(self,'unit_counts',immutable_array(count,dtype=np.int64))
        object.__setattr__(self,'edges',edges)
        if np.any(self.upper_failure<self.mean_failure):raise ValueError('Upper risk must contain empirical risk')

    @classmethod
    def fit(cls,scores,packet_failed,*,evidence,actions,edges=(0.,.01,.05,.2,.5,1.),alpha=.05,min_units=20):
        if evidence.scope.endpoint!=PACKET_ENDPOINT:raise ValueError('Actual declared control-deadline labels required')
        if not 0<alpha<1 or type(min_units) is not int or min_units<2:raise ValueError('Explicit unit support and alpha required')
        actions=tuple(actions)
        if np.ndim(scores)!=3 or np.shape(scores)[1]!=len(actions) or not actions:
            raise ValueError('Packet calibration requires whole_unit×supported_action×labeled_packet axes')
        p,y=_calibration_arrays(scores,packet_failed,evidence)
        if np.any((p<0)|(p>1)) or not np.isin(y,(0.,1.)).all():
            raise ValueError('Packet calibration needs bounded scores and observed Boolean failures')
        edge=np.asarray(edges,dtype=float)
        if edge.ndim!=1 or len(edge)<2 or edge[0]!=0 or edge[-1]!=1 or np.any(np.diff(edge)<=0):
            raise ValueError('Fixed bin edges must span [0,1]')
        p=p.reshape(len(evidence.unit_ids),len(actions),-1)
        y=y.reshape(p.shape)
        bins=np.clip(np.searchsorted(edge,p,side='right')-1,0,len(edge)-2)
        means=[];uppers=[];counts=[]
        family_cells=len(actions)*(len(edge)-1)
        for action_index in range(len(actions)):
            action_means=[];action_uppers=[];action_counts=[]
            for j in range(len(edge)-1):
                rates=[float(y[i,action_index,bins[i,action_index]==j].mean())
                       for i in range(len(y)) if np.any(bins[i,action_index]==j)]
                n=len(rates);m=float(np.mean(rates)) if n else 1.
                # Correlated packets/actions do not increase independent units.
                # Union over every declared action x fixed score bin, including
                # unsupported cells. No independence between actions is assumed.
                radius=math.sqrt(math.log(family_cells/alpha)/(2*n)) if n else math.inf
                action_means.append(m);action_uppers.append(min(1.,m+radius));action_counts.append(n)
            means.append(action_means);uppers.append(action_uppers);counts.append(action_counts)
        return cls(evidence,edge,means,uppers,np.asarray(counts,dtype=np.int64),
                   tuple(action_signature(a) for a in actions),float(alpha),min_units)

    def apply(self,scores,*,scope,actions):
        if scope!=self.evidence.scope:raise ValueError('Packet-risk scope changed; use Map fallback')
        signatures=tuple(action_signature(a) for a in actions)
        if not signatures or len(set(signatures))!=len(signatures) or any(s not in self.action_signatures for s in signatures):
            raise ValueError('Action has no held-out packet-risk support')
        p=np.asarray(scores,dtype=float)
        if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Bounded score required')
        index=np.clip(np.searchsorted(self.edges,p,side='right')-1,0,len(self.edges)-2)
        if len(signatures)==1:
            action_index=self.action_signatures.index(signatures[0])
        else:
            if p.ndim!=3 or p.shape[1]!=len(signatures):
                raise ValueError('Multiple actions require unit x action x forecast axes')
            action_index=np.asarray([self.action_signatures.index(s) for s in signatures])[None,:,None]
        if np.any(self.unit_counts[action_index,index]<self.min_units):
            raise ValueError('Packet-risk action/bin lacks independent unit support')
        return self.mean_failure[action_index,index],self.upper_failure[action_index,index]

    @property
    def metadata(self):
        return {'endpoint':PACKET_ENDPOINT,'receipt_sha256':self.evidence.receipt_sha256,
            'unit_counts':self.unit_counts.tolist(),'alpha':self.alpha,'minimum_units':self.min_units,
            'action_signatures':list(self.action_signatures),
            'bound':'fixed_action_bin_equal_unit_mean_Hoeffding_family_union_assuming_iid_whole_units',
            'family_cells':len(self.action_signatures)*(len(self.edges)-1),
            'actions_pooled':False,
            'per_query_conditional_or_drift_guarantee':False,'independent_gate_permission':False}


class RSSIOutcomeAdapter:
    """WorldModel-compatible renderer; legacy Map surrogate remains default."""
    def __init__(self,assumptions,source_id,*,site_id,device_id,rssi_bound=None,packet_calibration=None):
        if not isinstance(assumptions,RadioAssumptions):raise TypeError('Explicit physical assumptions required')
        self.assumptions=assumptions;self.source_id=_identity(source_id,'source_id')
        self.site_id=_identity(site_id,'site_id');self.device_id=_identity(device_id,'device_id')
        self.rssi_bound=rssi_bound;self.packet_calibration=packet_calibration;self.default=RadioMapAdapter()
        self._forecast_receipt=None
        for item,endpoint in ((rssi_bound,'rssi_dbm'),(packet_calibration,PACKET_ENDPOINT)):
            if item is not None:
                scope=item.evidence.scope
                if (scope.model_id,scope.site_id,scope.device_id,scope.physical_sha256,scope.endpoint)!=(
                    self.source_id,self.site_id,self.device_id,assumptions.sha256,endpoint):
                    raise ValueError('Calibration does not match this model/site/device/physics/endpoint')

    def scene_state(self,forecast):
        if not isinstance(forecast,RSSIForecast) or forecast.source_id!=self.source_id:
            raise ValueError('Use this model\'s typed predicted RSSI, not a private World')
        if not np.array_equal(forecast.carrier_hz,self.assumptions.carrier_hz):
            raise ValueError('Carrier mismatch; 6.0 GHz cannot silently become native 6.5 GHz')
        if self._forecast_receipt is not None and self._forecast_receipt!=forecast.source_receipt_sha256:
            raise ValueError('Model/source receipt changed within a service adapter runtime')
        self._forecast_receipt=forecast.source_receipt_sha256
        mean=forecast.mean_dbm
        if self.rssi_bound is not None:mean=self.rssi_bound.lower(mean,scope=self.rssi_bound.evidence.scope)
        return SceneState(forecast.times_s,forecast.current_ap,self.assumptions.convert(mean),self.source_id)

    def query(self,scene_state,candidate_actions,horizon,timescale='mid'):
        if not isinstance(scene_state,SceneState) or scene_state.source_id!=self.source_id:
            raise ValueError('Use this adapter\'s predicted SceneState')
        if scene_state.learned_outage_probability is not None:
            raise ValueError('Unsupported learned outage head; explicit held-out packet calibration required')
        actions=tuple(candidate_actions)
        result=self.default.query(scene_state,actions,horizon,timescale)
        metadata=dict(result.metadata)|{'world_model':'RSSIOutcomeAdapter',
            'physical_assumptions_sha256':self.assumptions.sha256,
            'forecast_source_receipt_sha256':self._forecast_receipt or 'no_RSSI_forecast_bound',
            'absolute_reference':self.assumptions.absolute_reference,
            'interference_source':self.assumptions.interference_source,
            'packet_deadline_risk':'unavailable_from_RSSI_alone',
            'default_controller':'RadioMap_legacy_PHY_surrogate',
            'independent_gate_permission':'none',
            'phy_payload_proxy':'1200_byte_2SS_HT20_VHT80_tables_applied_to_native_600_byte_control',
            'carrier_hz':','.join(str(x) for x in self.assumptions.carrier_hz),
            'RF_calibration':'none' if self.rssi_bound is None else json.dumps(self.rssi_bound.metadata,sort_keys=True)}
        if self.packet_calibration is None:return replace(result,metadata=metadata)
        calibration=self.packet_calibration
        risk=np.empty_like(result.outage_probability);upper=np.empty_like(risk)
        costs=np.empty_like(result.outage_integral_seconds)
        for j,action in enumerate(actions):
            # Calibrate the no-switch RF score; then add the declared exact
            # deterministic handover interval independently of learned heads.
            neutral=replace(scene_state,current_ap=np.full(len(scene_state.current_ap),action.ap,dtype=np.int64))
            raw=self.default.query(neutral,(action,),horizon,timescale)
            p,u=calibration.apply(raw.outage_probability[:,0],scope=calibration.evidence.scope,actions=(action,))
            sw=result.switch_seconds[:,j];during=scene_state.times_s[None]<sw[:,None]
            risk[:,j]=np.where(during,1.,p);upper[:,j]=np.where(during,1.,np.maximum(p,u))
            stay=integrate(p,scene_state.times_s,0.,horizon)
            move=action.handover_seconds+integrate(p,scene_state.times_s,action.handover_seconds,horizon)
            costs[:,j]=np.where(sw>0,move,stay)
        metadata.update({'packet_deadline_risk':'measured_labels_fixed_bin_unit_mean_not_pointwise_guarantee',
                         'calibration':json.dumps(calibration.metadata,sort_keys=True),
                         'integral':'expected_failure_rate_seconds_proxy_not_physical_silence'})
        return replace(result,outage_probability=risk,outage_upper=upper,
                       outage_integral_seconds=costs,metadata=metadata)


class BridgedPredictor:
    """Adapt public-only RSSI predictor.step(past,plan) to native SNR units.

    The wrapped predictor returns (RSSIForecast, timing_kind, elapsed_seconds).
    It receives no World, truth, future report, hidden blockage or packet draw.
    Queue completion charges this full conversion path too.
    """
    def __init__(self,predictor,adapter):
        if not isinstance(adapter,RSSIOutcomeAdapter):raise TypeError('Typed RSSI outcome adapter required')
        self.predictor,self.adapter=predictor,adapter;self.geometry=predictor.geometry

    def step(self,past,plan):
        started=time.perf_counter();forecast,kind,reported=self.predictor.step(past,plan)
        state=self.adapter.scene_state(forecast)
        from ..native.world import HORIZONS
        if not np.array_equal(state.times_s,HORIZONS):raise ValueError('Native queue requires its exact nine horizon knots')
        if kind not in ('cold','warm','refit') or isinstance(reported,bool) or not math.isfinite(reported) or reported<0:
            raise ValueError('Predictor must report a valid timing regime and actual elapsed time')
        return state.snr_db.astype('f4'),forecast.std_db.astype('f4'),kind,max(reported,time.perf_counter()-started)


class TypedRoamUtility:
    """Shared finite candidate planner and exact admission for both arms.

    Native simulator still applies its independent gate, stagger, TTL and
    native execution path. A synchronous over-budget computation is discarded
    and denies roaming; this is not a preemptive hard real-time guarantee.
    """
    def __init__(self,model,source_id,queue,*,handover_seconds,horizon,max_candidates,deadline_seconds):
        from ..jev.spec import MissionSpec
        self.model,self.source_id,self.queue=model,source_id,queue
        self.cost=float(handover_seconds);self.horizon=float(horizon)
        if not 0<=self.cost<=self.horizon<=30:raise ValueError('Unsupported handover/horizon')
        if type(max_candidates) is not int or max_candidates<1:raise ValueError('Finite candidate budget required')
        self.max_candidates=max_candidates;self.deadline=_positive(deadline_seconds,'planner deadline')
        self.mission=MissionSpec('same_service_surrogate',require_calibrated_outage=False)
        self.records=[]

    def _evaluate(self,forecast,current):
        from ..native.world import HORIZONS
        from ..jev.cost import evaluate
        current=np.asarray(current)
        if forecast.shape[2]>self.max_candidates:raise ValueError('All AP candidates must fit the same fixed budget')
        state=SceneState(HORIZONS,current,forecast,self.source_id)
        actions=tuple(CandidateAction('ap-'+str(i),i,handover_seconds=self.cost) for i in range(forecast.shape[2]))
        outcome=self.model.query(state,actions,self.horizon,'mid')
        return evaluate(outcome,self.mission,self.horizon).objective

    def scores(self,forecast,current):
        started=time.perf_counter();loss=self._evaluate(forecast,current);elapsed=time.perf_counter()-started
        late=elapsed>self.deadline
        self.records.append({'kind':'planner','elapsed_seconds':elapsed,'deadline_seconds':self.deadline,
                             'late_discarded':late,'candidates':int(loss.shape[1])})
        return np.zeros_like(loss) if late else -loss

    def __call__(self,decision):
        started=time.perf_counter()
        forecast=self.queue.advance(self.queue.mean[decision.decision])
        loss=self._evaluate(forecast,decision.current);ar=np.arange(len(loss))
        permit=decision.eligible&(loss[ar,decision.current]-loss[ar,decision.candidate]>1e-7)
        elapsed=time.perf_counter()-started;late=elapsed>self.deadline
        if late:permit[:]=False
        self.records.append({'kind':'admission','elapsed_seconds':elapsed,'deadline_seconds':self.deadline,
                             'late_discarded':late,'permitted':int(permit.sum())})
        return permit


@dataclass(frozen=True)
class ServiceProbeConfig:
    duration_seconds: int
    forecast_profile_seconds: dict
    shared_extra_seconds: float=.5
    handover_seconds: float=.03
    horizon_seconds: float=5.
    max_candidates: int=64
    planner_deadline_seconds: float=.1
    scan_duration_seconds: float=.09
    scan_period_seconds: float=5.
    collision_lambda: float=0.
    interference: bool=False

    def __post_init__(self):
        if type(self.duration_seconds) is not int or not 1<=self.duration_seconds<=60:
            raise ValueError('The finite DEV probe supports 1..60 seconds')
        if set(self.forecast_profile_seconds)!=set(('cold','warm','refit')):
            raise ValueError('A shared cold/warm/refit compute allowance is required')
        for name,value in self.forecast_profile_seconds.items():_positive(value,name+' forecast allowance')
        object.__setattr__(self,'forecast_profile_seconds',deepcopy(self.forecast_profile_seconds))
        for name in ('shared_extra_seconds','handover_seconds','collision_lambda'):
            value=getattr(self,name)
            if isinstance(value,bool) or not math.isfinite(value) or value<0:raise ValueError(name+' must be finite and nonnegative')
        for name in ('horizon_seconds','planner_deadline_seconds','scan_duration_seconds','scan_period_seconds'):
            _positive(getattr(self,name),name)
        if not self.handover_seconds<=self.horizon_seconds<=30:raise ValueError('Handover/horizon unsupported')
        if type(self.max_candidates) is not int or self.max_candidates<1 or type(self.interference) is not bool:
            raise ValueError('Explicit finite action budget and interference flag required')


@dataclass(frozen=True)
class DevelopmentRegistration:
    case_id: str
    seed: int
    source_receipt_sha256: str
    protocol_sha256: str
    world_receipt_sha256: str
    partition: str='development'

    def __post_init__(self):
        _identity(self.case_id,'registered case identity')
        if self.partition!='development' or type(self.seed) is not int:
            raise ValueError('Only pre-registered development cases are supported; SERVICE holdouts remain closed')
        for name in ('source_receipt_sha256','protocol_sha256','world_receipt_sha256'):_sha(getattr(self,name),name)


def run_development_pair(world,warm_history,geometry,candidate_predictor,*,config,registration,publication_guard,
                         candidate_model=None,candidate_source_id='candidate'):
    """Execute two real native packet simulations only after an external guard.

    publication_guard(registration) must verify the published source/protocol,
    admitted case+world bytes, and unchanged sealed evidence. This function
    never generates an unregistered scene or opens a fixed SERVICE case.
    Returned raw telemetry/counters require an immutable receipt by the study.
    """
    if not isinstance(config,ServiceProbeConfig) or not isinstance(registration,DevelopmentRegistration):
        raise TypeError('Typed finite budget and published DEV case required')
    if not callable(publication_guard):raise TypeError('A real publication/case verifier is mandatory')
    guarded=publication_guard(registration)  # BEFORE touching world or executing a simulator.
    if guarded is False:raise RuntimeError('Publication/case verification refused this DEV probe')
    from ..native.world import World,CLIENT_TX_DBM
    from ..fleetwm.data import Observations
    from ..closed_loop.telemetry import DeliveredScans
    from ..closed_loop.runtime import SinglePredictor
    from ..jepa_repair.runtime import CompletionSafeForecasts
    from ..service_world.sim import Policy,simulate
    from ..gate_repair.gate import reactive_rescue
    if not isinstance(world,World) or world.n!=100 or world.seed!=registration.seed:
        raise ValueError('Use the exact registered native 100-robot world')
    if not isinstance(warm_history,Observations) or warm_history.value.shape[:2]!=(60,100):
        raise ValueError('Both arms require the same actual sixty-bin observable warm history')
    if world.duration_s<config.duration_seconds or len(world.plan)<config.duration_seconds:
        raise ValueError('Registered world does not cover the DEV budget')
    if len(geometry.sites)>config.max_candidates or not np.array_equal(geometry.sites,world.sites):
        raise ValueError('Public AP geometry/candidate budget differs from registered world')
    if not np.array_equal(geometry.tx,world.tx_dl) or not np.array_equal(geometry.noise,world.noise_db):
        raise ValueError('Public TX/noise coordinates differ from simulator service coordinates')
    if not np.array_equal(candidate_predictor.geometry.sites,geometry.sites):
        raise ValueError('Candidate must use exactly the same public AP geometry')
    if isinstance(candidate_predictor,BridgedPredictor):
        a=candidate_predictor.adapter.assumptions
        if not np.array_equal(a.carrier_hz,NATIVE_CARRIERS_HZ):
            raise ValueError('Native 6.5 GHz endpoint cannot use a 6.0 GHz field without a registered transport model')
        if not np.array_equal(a.forecast_transmit_power_dbm,geometry.tx) or not np.array_equal(a.receiver_noise_dbm,geometry.noise):
            raise ValueError('Bridge reference TX/noise does not match this native geometry')
        if (not np.all(a.service_transmit_power_dbm==CLIENT_TX_DBM)
                or not np.all(a.antenna_gain_change_db==0) or not np.all(a.device0_bias_db==0)):
            raise ValueError('Native service uses 15 dBm clients, zero additional antenna change, and an explicit zero-bias RSSI gauge')
    policy=Policy('adaptive_service_development','native',roam_cost_s=config.handover_seconds,
                  scan_period_s=config.scan_period_seconds,gate=True,cadence_scan=True,
                  cadence_roam=True,online_map=False,dual_radio=False)
    results={}
    for name,predictor,model,source_id in (
        ('radiomap',SinglePredictor(geometry,'radiomap'),RadioMapAdapter(),'radiomap'),
        ('candidate',candidate_predictor,candidate_model or RadioMapAdapter(),candidate_source_id)):
        sensor=DeliveredScans(warm_history,config.duration_seconds,registration.seed)
        live=CompletionSafeForecasts(predictor,sensor,config.duration_seconds,
                                     dict(config.forecast_profile_seconds),config.shared_extra_seconds)
        physical=copy(world);physical.forecast=live.mean;physical.sigma_belief=live.std[:,:,0]
        physical.rmap=None  # no arm mutates an already fitted physical-world map
        utility=TypedRoamUtility(model,source_id,live,handover_seconds=config.handover_seconds,
            horizon=config.horizon_seconds,max_candidates=config.max_candidates,
            deadline_seconds=config.planner_deadline_seconds)
        decisions=[];executed_moves=[]
        def capture(decision,confident,site):
            decisions.append({'decision':int(decision.decision),'current':decision.current.tolist(),
                'candidate':decision.candidate.tolist(),'eligible':decision.eligible.tolist(),
                'independent_gate':confident.tolist(),'association_after_native_path':site.tolist(),
                'gain':decision.gain.tolist()})
        def executed(slot,robots,targets):
            executed_moves.append({'slot':int(slot),'robots':np.asarray(robots).tolist(),
                                   'targets':np.asarray(targets).tolist()})
        native=simulate(physical,policy,seed=0,duration_s=config.duration_seconds,
            collision_lambda=config.collision_lambda,interference=config.interference,
            roam_gate=reactive_rescue,handover_admission=utility,roam_scorer=utility.scores,
            forecast_queue=live,forecast_source=live,scan_observer=sensor,
            scan_duration_s=config.scan_duration_seconds,decision_trace=capture,actual_action_hook=executed,detailed=True,
            observation_period_slots=10,native_proposal_ttl_slots=100)
        results[name]={'native':native,'forecast_jobs':deepcopy(live.records),
            'forecast_consumption':deepcopy(live.consumption_records),'telemetry':sensor.summary(),
            'planner_and_admission':utility.records,'decisions':decisions,'executed_moves':executed_moves,
            'forecast_sha256':hashlib.sha256(live.mean.tobytes()).hexdigest(),
            'forecast_std_sha256':hashlib.sha256(live.std.tobytes()).hexdigest()}
    return {'case_id':registration.case_id,'partition':'development','robots':100,
        'native_packet_simulations':2,'simulation_only':True,'production_qualified':False,
        'original_SERVICE_holdouts_opened':False,'default_controller_changed':False,
        'source_receipt_sha256':registration.source_receipt_sha256,'protocol_sha256':registration.protocol_sha256,
        'world_receipt_sha256':registration.world_receipt_sha256,
        'equal_allowed_capture_and_compute_budgets':True,'equal_measured_runtime_claimed':False,
        'policy':policy.__dict__,'config':config.__dict__,'arms':results}
