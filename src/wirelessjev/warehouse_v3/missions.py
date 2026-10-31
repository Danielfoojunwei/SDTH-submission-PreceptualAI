"""Complete controlled-language contracts with hard floors and lexicographic goals.

Groups are resolved from application declarations and current public poses.
They remain fixed across a planning commitment. Predicted feasibility is not
a guarantee about future physical service, and candidate exhaustion is not a
proof of global infeasibility.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import time

import numpy as np


@dataclass(frozen=True)
class Group:
    kind: str = 'all'
    ids: tuple[str, ...] = ()
    axis: str | None = None
    direction: str | None = None
    boundary_m: float | None = None

    def __post_init__(self):
        if self.kind not in ['all', 'teleoperated', 'video', 'ids', 'zone']:
            raise ValueError('Unsupported robot group')
        if self.kind == 'ids' and (not self.ids or len(set(self.ids)) != len(self.ids)
                or any(not isinstance(value,str) or not value.strip() or value!=value.strip() for value in self.ids)):
            raise ValueError('An explicit robot group must have unique IDs')
        if self.kind == 'zone' and (self.axis not in ['x', 'y'] or self.direction not in ['<', '>=']
                                   or isinstance(self.boundary_m,bool) or not isinstance(self.boundary_m, (int, float)) or not np.isfinite(self.boundary_m)
                or not 0 <= self.boundary_m <= (120 if self.axis == 'x' else 80)):
            raise ValueError('Invalid spatial predicate')
        if self.kind != 'ids' and self.ids:
            raise ValueError('Explicit IDs conflict with another group predicate')
        if self.kind != 'zone' and any(v is not None for v in [self.axis,self.direction,self.boundary_m]):
            raise ValueError('Spatial fields conflict with another group predicate')


@dataclass(frozen=True)
class Floor:
    group: Group
    fraction: float

    def __post_init__(self):
        if isinstance(self.fraction, bool) or not np.isfinite(self.fraction) or not 0 <= self.fraction <= 1:
            raise ValueError('Service floors must be fractions')


@dataclass(frozen=True)
class Goal:
    kind: str
    group: Group = Group()
    fraction: float = .1

    def __post_init__(self):
        if self.kind not in ['throughput', 'min_switches', 'tail', 'group_service', 'critical']:
            raise ValueError('Unsupported priority')
        if isinstance(self.fraction,bool) or not 0 < self.fraction <= 1:
            raise ValueError('Invalid worst-served fraction')
        if self.kind!='group_service' and self.group!=Group():
            raise ValueError('This goal does not support a group predicate; use group_service')
        if self.kind!='tail' and self.fraction!=.1:
            raise ValueError('Only a tail-service goal supports a nondefault fraction')


@dataclass(frozen=True)
class Contract:
    max_moves: int
    floors: tuple[Floor, ...]
    priorities: tuple[Goal, ...]
    conditional_floor: Floor | None = None

    def __post_init__(self):
        if type(self.max_moves) is not int or not 0 <= self.max_moves <= 200 or not self.priorities:
            raise ValueError('Explicit bounded switch budget and nonempty priorities required')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, document):
        if not isinstance(document,dict) or set(document)-{'max_moves','floors','priorities','conditional_floor'} or not {'max_moves','priorities'}<=set(document):
            raise ValueError('Unknown contract fields or missing required budget/priorities')
        def group(value):
            if not isinstance(value,dict) or set(value)-{'kind','ids','axis','direction','boundary_m'} or 'kind' not in value:
                raise ValueError('Invalid robot-group fields')
            return Group(**{**value,'ids':tuple(value.get('ids',[]))})
        def floor(value):
            if set(value)!={'group','fraction'}:raise ValueError('Invalid service-floor fields')
            return Floor(group(value['group']),value['fraction'])
        goals=[]
        for value in document['priorities']:
            if not isinstance(value,dict) or set(value)-{'kind','group','fraction'} or 'kind' not in value:
                raise ValueError('Invalid ordered-priority fields')
            if value['kind']=='group_service' and 'group' not in value:raise ValueError('Group service requires an explicit group')
            goals.append(Goal(value['kind'],group(value.get('group',{'kind':'all'})),value.get('fraction',.1)))
        return cls(document['max_moves'],tuple(floor(f) for f in document.get('floors',[])),tuple(goals),
            None if document.get('conditional_floor') is None else floor(document['conditional_floor']))


def parse_group(text):
    text = re.sub(r'\brobot$', 'robots', text)
    if text in ['all robots', 'teleoperated robots', 'video robots']:
        return Group(text.split()[0] if text != 'all robots' else 'all')
    match = re.fullmatch(r'robots \[([a-zA-Z0-9_, -]+)\]', text)
    if match:
        return Group('ids', tuple(value.strip() for value in match[1].split(',')))
    match = re.fullmatch(r'robots with ([xy]) (<|>=) (\d+(?:\.\d+)?) m', text)
    if match:
        return Group('zone', axis=match[1], direction=match[2], boundary_m=float(match[3]))
    raise ValueError('Unsupported or incomplete robot group')


def compile_contract(text):
    # Decimal literals are accepted; splitting at sentence delimiters must not
    # split numbers such as 12.5%. Entire clauses are matched, never searched.
    clauses = [value.strip().lower() for value in re.split(r';|\.(?:\s+|$)', text.strip()) if value.strip()]
    if not clauses:
        raise ValueError('Empty mission')
    budget = re.fullmatch(r'switch at most (\d+) robots', clauses.pop(0))
    if budget is None:
        raise ValueError('The first clause must declare a switch budget')
    floors, priorities, condition = [], [], None
    for clause in clauses:
        floor = re.fullmatch(r'require service >= (\d+(?:\.\d+)?)% for (.+) at every offset', clause)
        conditional = re.fullmatch(r'if any (.+) has observed service < (\d+(?:\.\d+)?)%, prioritise that group', clause)
        if floor and not priorities:
            floors.append(Floor(parse_group(floor[2]), float(floor[1])/100))
        elif conditional and not priorities and condition is None:
            condition = Floor(parse_group(conditional[1]), float(conditional[2])/100)
        else:
            goal = clause.removeprefix('then ')
            if goal == 'minimise switches': priorities.append(Goal('min_switches'))
            elif goal == 'maximise throughput': priorities.append(Goal('throughput'))
            elif goal == 'minimise critical violations': priorities.append(Goal('critical'))
            elif match := re.fullmatch(r'maximise worst-served (\d+(?:\.\d+)?)% service', goal):
                priorities.append(Goal('tail', fraction=float(match[1])/100))
            elif match := re.fullmatch(r'maximise service for (.+)', goal):
                priorities.append(Goal('group_service', parse_group(match[1])))
            else:
                raise ValueError(f'Unsupported, misplaced or ambiguous clause: {clause}')
    return Contract(int(budget[1]), tuple(floors), tuple(priorities), condition)


def members(group, observation, robot_ids, roles):
    n = len(robot_ids)
    if len(set(robot_ids)) != n:
        raise ValueError('Duplicate robot IDs')
    if group.kind == 'all': mask = np.ones(n, bool)
    elif group.kind == 'video': mask = np.asarray(observation['video'], bool)
    elif group.kind == 'teleoperated':
        if 'teleoperated' not in roles or len(roles['teleoperated']) != n:
            raise ValueError('Teleoperation membership must be explicitly declared')
        mask = np.asarray(roles['teleoperated'], bool)
    elif group.kind == 'ids':
        if not set(group.ids) <= set(robot_ids): raise ValueError('An explicit robot ID is unknown')
        mask = np.isin(robot_ids, group.ids)
    else:
        position = observation['robot'][:, 0 if group.axis == 'x' else 1]*(120 if group.axis == 'x' else 80)
        mask = position < group.boundary_m if group.direction == '<' else position >= group.boundary_m
    if mask.shape != (n,) or not mask.any():
        raise ValueError('The requested robot group is empty')
    return mask


def evaluate(contract, observation, plans, service, risk, robot_ids, roles, *, floor_service=None):
    from ..warehouse_horizon.planning import Contract as SearchContract, feasible
    plans, service, risk = np.asarray(plans), np.asarray(service), np.asarray(risk)
    if service.shape != risk.shape or service.shape != (len(plans), 6, len(robot_ids)):
        raise ValueError('Expected six matched forecast offsets for every plan/robot')
    if not np.isfinite(service).all() or not np.isfinite(risk).all() or ((service < 0) | (service > 1)).any() or ((risk < 0) | (risk > 1)).any():
        raise ValueError('Invalid service/risk forecasts')
    allowed = feasible(plans, observation, SearchContract('throughput', contract.max_moves, 0.))
    floor_service = service if floor_service is None else np.asarray(floor_service)
    if floor_service.shape != service.shape or not np.isfinite(floor_service).all() or ((floor_service < 0) | (floor_service > 1)).any():
        raise ValueError('Invalid service-floor forecasts')
    shortfalls = np.zeros(len(plans))
    for floor in contract.floors:
        mask = members(floor.group, observation, robot_ids, roles)
        gap = np.maximum(floor.fraction-floor_service[:, :, mask], 0.).max((1, 2))
        shortfalls = np.maximum(shortfalls, gap)
        allowed &= gap <= 1e-8
    goals = list(contract.priorities)
    conditional_active = False
    if contract.conditional_floor is not None:
        mask = members(contract.conditional_floor.group, observation, robot_ids, roles)
        conditional_active = bool((observation['robot'][mask, 6] < contract.conditional_floor.fraction).any())
        if conditional_active: goals.insert(0, Goal('group_service', contract.conditional_floor.group))
    columns = []
    demand = observation['critical']+observation['bulk']
    for goal in goals:
        if goal.kind == 'min_switches': column = -(plans != observation['current'][None]).sum(1).astype(float)
        elif goal.kind == 'throughput': column = (service*demand[None, None]).sum(2).mean(1)/max(float(demand.sum()), 1e-9)
        elif goal.kind == 'critical': column = -risk.mean((1, 2))
        elif goal.kind == 'tail': column = np.sort(service, axis=2)[:, :, :max(1, int(np.ceil(goal.fraction*len(robot_ids))))].mean((1, 2))
        else: column = service[:, :, members(goal.group, observation, robot_ids, roles)].mean((1, 2))
        columns.append(column)
    return np.stack(columns, 1), allowed, {'conditional_priority_active': conditional_active,
        'maximum_service_shortfall_by_candidate': shortfalls.tolist(), 'group_membership_time': 'decision_start'}


def select(contract, observation, plans, service, risk, robot_ids, roles, *, value_override=None, floor_service=None):
    priorities, allowed, detail = evaluate(contract, observation, plans, service, risk, robot_ids, roles, floor_service=floor_service)
    indices = np.flatnonzero(allowed)
    if not len(indices):
        return {'choice': None, 'status': 'infeasible_within_candidates', 'globally_infeasible': False,
                'feasible_candidates': 0, **detail}
    if value_override is not None:
        # Learned scoring cannot override hard floors, switch counts, target
        # visibility or higher lexicographic priorities. It only breaks the
        # last declared priority; this is an explicitly tested research arm.
        priorities[:, -1] = np.asarray(value_override)
    # Stable deterministic tie breaking, with a documented numerical tolerance.
    choice = max(indices, key=lambda i: tuple(np.round(priorities[i], 8))+(-int(i),))
    return {'choice': int(choice), 'status': 'selected', 'feasible_candidates': int(allowed.sum()),
            'ordered_scores': priorities[choice].tolist(), 'physical_safety_guarantee': False, **detail}


class Reader:
    def __init__(self, engine=None):
        self.engine, self.cache = engine, {}

    def read(self, text):
        if text in self.cache:
            contract = self.cache[text]
            return contract, {'reader': 'cached_contract', 'model_executed': False, 'cached': True, 'latency_ms': 0.}
        started = time.perf_counter()
        contract = compile_contract(text)
        receipt = {'reader': 'complete_controlled_language', 'model_executed': False, 'cached': False}
        if self.engine is not None:
            expected = 'conditional' if contract.conditional_floor else ('floors' if contract.floors else 'ordered')
            criteria = {'conditional': 'A priority activated by an observed condition',
                        'floors': 'Unconditional service floors with additional ordered goals',
                        'ordered': 'Ordered objectives with a switch budget and no service-floor constraint or condition',
                        'unsupported': 'Unsupported, contradictory or incomplete mission'}
            choice, probabilities, _ = self.engine.choose('Operator mission (data):\n'+text,
                'Choose the contract family. Conditional priorities take precedence over the unconditional-floor label.', criteria)
            receipt.update(reader='actual_d3_family_and_complete_literal_compiler', model_executed=True,
                choice=choice, answer_probabilities=probabilities, probabilities_are_physical_risk=False)
            if choice != expected:
                raise ValueError('The semantic model abstained or disagreed with the explicit contract')
        self.cache[text] = contract
        receipt['latency_ms'] = (time.perf_counter()-started)*1000
        return contract, receipt


MISSIONS = {
 'train': [
  'Switch at most 12 robots; maximise throughput.',
  'Switch at most 12 robots; require service >= 30% for teleoperated robots at every offset; minimise switches; then maximise throughput.',
  'Switch at most 12 robots; maximise worst-served 10% service; then minimise critical violations.',
  'Switch at most 12 robots; if any teleoperated robot has observed service < 60%, prioritise that group; minimise critical violations; then maximise throughput.',
  'Switch at most 12 robots; maximise service for robots with x < 60 m; then maximise throughput.',
 ],
 'novel': [
  'Switch at most 5 robots; require service >= 12.5% for video robots at every offset; minimise switches; then maximise throughput.',
  'Switch at most 12 robots; require service >= 80% for teleoperated robots at every offset; minimise switches; then maximise throughput.',
  'Switch at most 8 robots; if any teleoperated robot has observed service < 75%, prioritise that group; maximise worst-served 20% service; then maximise throughput.',
  'Switch at most 12 robots; maximise service for robots with y >= 40 m; then minimise switches.',
  'Switch at most 12 robots; require service >= 100% for all robots at every offset; minimise switches; then maximise throughput.',
 ]}
