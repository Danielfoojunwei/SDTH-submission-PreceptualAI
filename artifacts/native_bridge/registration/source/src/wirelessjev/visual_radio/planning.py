"""Mission/options hierarchy; unknown service preserves canonical recovery.

Values are absolute service predictions, not signed advantages or certified
risk bounds. This module proposes options; it never controls a physical radio.
"""
from dataclasses import dataclass,replace
from enum import Enum
import math
from ..service_world.physics import forced_deadline_misses,merge_unavailable


class OptionKind(str,Enum):
    CANONICAL='canonical'
    HOLD_AP='hold_ap'
    MOVE_AP='move_ap'
    SCAN='scan'
    SWITCH_CHANNEL='switch_channel'
    REDUNDANT_LINK='redundant_link'
    DEFER_BULK='defer_bulk'


@dataclass(frozen=True)
class ServiceIntent:
    """Slow-level objective in physical units; no text inference here."""
    horizon_slots:int=1000
    period_slots:int=5
    deadline_slots:int=5
    miss_weight:float=1.
    gap_weight_per_second:float=1.
    resource_weight:float=0.
    minimum_advantage:float=0.

    def __post_init__(self):
        for name in ('horizon_slots','period_slots','deadline_slots'):
            if type(getattr(self,name)) is not int or getattr(self,name)<=0:raise ValueError(name+' must be positive integer slots')
        for name in ('miss_weight','gap_weight_per_second','resource_weight','minimum_advantage'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<0:raise ValueError('Invalid objective weight')


@dataclass(frozen=True)
class ServiceEstimate:
    expected_misses:float
    expected_max_gap_slots:float
    provenance:str
    resource_cost:float=0.
    upper_misses:float|None=None
    upper_max_gap_slots:float|None=None
    validation:str='unvalidated'

    def __post_init__(self):
        values=(self.expected_misses,self.expected_max_gap_slots,self.resource_cost)
        if not self.provenance or any(not math.isfinite(x) or x<0 for x in values):raise ValueError('Invalid service estimate')
        for upper,mean in ((self.upper_misses,self.expected_misses),(self.upper_max_gap_slots,self.expected_max_gap_slots)):
            if upper is not None and (not math.isfinite(upper) or upper<mean):raise ValueError('Upper estimate must be finite and at least the mean')
        if self.validation not in ('unvalidated','numerical','empirical'):raise ValueError('Unknown evidence status')


@dataclass(frozen=True)
class RadioOption:
    option_id:str
    kind:OptionKind
    issued_slot:int
    valid_until_slot:int
    hold_slots:int
    estimate:ServiceEstimate|None
    # Only intervals during which ALL relevant service paths are unavailable.
    exclusive_unavailable:tuple[tuple[int,int],...]=()
    target_ap:int|None=None
    channel:int|None=None
    observation_slot:int|None=None

    def __post_init__(self):
        if not self.option_id or not isinstance(self.kind,OptionKind):raise ValueError('Typed option kind and identity required')
        for name in ('issued_slot','valid_until_slot','hold_slots'):
            if type(getattr(self,name)) is not int:raise ValueError('Options use integer slot clocks')
        if self.issued_slot<0 or not 0<self.hold_slots<=100 or not self.issued_slot<self.valid_until_slot<=self.issued_slot+100:
            raise ValueError('Proposals require a positive lease/expiry no longer than one second')
        if self.observation_slot is not None and (type(self.observation_slot) is not int or not 0<=self.observation_slot<=self.issued_slot):raise ValueError('Invalid observation timestamp')
        merge_unavailable(self.exclusive_unavailable)
        if self.kind==OptionKind.MOVE_AP and (type(self.target_ap) is not int or self.target_ap<0):raise ValueError('Move requires a candidate AP')
        if self.kind==OptionKind.SWITCH_CHANNEL and (type(self.channel) is not int or self.channel<=0):raise ValueError('Channel option requires a supported channel')


@dataclass(frozen=True)
class OptionAssessment:
    option_id:str
    admitted:bool
    reason:str
    objective:float|None=None
    known_forced_misses:int|None=None


@dataclass(frozen=True)
class HierarchyDecision:
    selected:RadioOption|None
    baseline_objective:float|None
    assessments:tuple[OptionAssessment,...]
    available_slot:int
    max_observation_age_slots:int
    fallback:str='canonical_continuation'
    certification:str='proposal_only_no_physical_or_statistical_guarantee'

    def proposal_at(self,slot):
        if self.selected is None:return None
        expiry=min(self.selected.valid_until_slot,self.selected.observation_slot+self.max_observation_age_slots+1)
        if not max(self.available_slot,self.selected.issued_slot)<=slot<expiry:return None
        return replace(self.selected,hold_slots=min(self.selected.hold_slots,expiry-slot))


def _service_cost(option,intent,now_slot,release_phase,pending_release_slots,upper):
    estimate=option.estimate
    if estimate is None:return None,None
    phases=range(intent.period_slots) if release_phase is None else (release_phase,)
    if release_phase is None and pending_release_slots:raise ValueError('Known pending releases require their actual traffic phase')
    floor=min(forced_deadline_misses(option.exclusive_unavailable,phase=p,
        period_slots=intent.period_slots,deadline_slots=intent.deadline_slots,
        start_slot=now_slot,stop_slot=now_slot+intent.horizon_slots,
        pending_release_slots=pending_release_slots) for p in phases)
    intervals=merge_unavailable(option.exclusive_unavailable)
    known_gap=max((max(0,min(b,now_slot+intent.horizon_slots)-max(a,now_slot)) for a,b in intervals),default=0)
    misses=estimate.upper_misses if upper else estimate.expected_misses
    gap=estimate.upper_max_gap_slots if upper else estimate.expected_max_gap_slots
    if misses is None or gap is None:return None,floor
    # Apply absolute physical floors, never clamp signed paired differences.
    return (intent.miss_weight*max(floor,misses)
        +intent.gap_weight_per_second*.01*max(known_gap,gap)
        +intent.resource_weight*estimate.resource_cost),floor


def choose_option(intent,baseline,options,*,now_slot,completion_slot=None,
                  supported_kinds=(OptionKind.HOLD_AP,OptionKind.MOVE_AP,OptionKind.SCAN),
                  release_phase=None,pending_release_slots=(),max_observation_age_slots=100):
    """Medium-level finite selection; fast canonical control keeps running.

    Baseline is unforced canonical continuation. Candidates need documented
    upper estimates and explicit numerical/empirical validation. Comparing a
    candidate upper cost with the baseline mean is an empirical admission rule,
    not a theorem about regret or physical safety. A provenance/status label
    cannot itself establish calibration, and no execution API is provided.
    """
    if baseline.kind!=OptionKind.CANONICAL:raise ValueError('Factual comparator must be unforced canonical continuation')
    if type(now_slot) is not int or now_slot<0:raise ValueError('Invalid decision clock')
    completion=now_slot if completion_slot is None else completion_slot
    if type(completion) is not int or completion<now_slot:raise ValueError('Completion must follow the decision')
    if release_phase is not None and (type(release_phase) is not int or not 0<=release_phase<intent.period_slots):raise ValueError('Invalid traffic phase')
    if type(max_observation_age_slots) is not int or max_observation_age_slots<0:raise ValueError('Invalid freshness policy')
    if baseline.issued_slot>now_slot:raise ValueError('Baseline cannot come from the future')
    base,_=_service_cost(baseline,intent,now_slot,release_phase,pending_release_slots,False)
    if baseline.observation_slot is None or completion-baseline.observation_slot>max_observation_age_slots:
        base=None
    rows=[];choices=[];seen={baseline.option_id}
    for option in options:
        if option.option_id in seen:raise ValueError('Duplicate option identity')
        seen.add(option.option_id);reason=None
        if option.kind not in supported_kinds:reason='unsupported_hardware_action'
        elif option.issued_slot>now_slot:reason='future_proposal'
        elif completion>=option.valid_until_slot:reason='expired_before_result'
        elif option.observation_slot is None:reason='unknown_observation_time'
        elif completion-option.observation_slot>max_observation_age_slots:reason='stale_observations'
        elif option.estimate is None:reason='unknown_service'
        elif option.estimate.validation=='unvalidated':reason='unvalidated_service'
        elif base is None or baseline.estimate.validation=='unvalidated':reason='unknown_baseline_service'
        if reason:
            rows.append(OptionAssessment(option.option_id,False,reason));continue
        cost,floor=_service_cost(option,intent,now_slot,release_phase,pending_release_slots,True)
        if cost is None:
            rows.append(OptionAssessment(option.option_id,False,'unknown_upper_service',known_forced_misses=floor));continue
        admitted=cost+intent.minimum_advantage<base
        rows.append(OptionAssessment(option.option_id,admitted,'estimated_improvement' if admitted else 'no_estimated_improvement',cost,floor))
        if admitted:choices.append((cost,option.option_id,option))
    selected=min(choices,key=lambda x:(x[0],x[1]))[2] if choices else None
    return HierarchyDecision(selected,base,tuple(rows),completion,max_observation_age_slots)
