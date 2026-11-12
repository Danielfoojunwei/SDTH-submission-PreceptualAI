"""Explicit unit/event contracts for the pinned upstream read-only preview.

These are interchange inputs, not inferred service guarantees. Missing RF-to-
service heads must not be filled by translating RSSI, SNR or pointwise outage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping
import hashlib
import json
import math
import re
from types import MappingProxyType

UNITS = frozenset(("airtime_fraction", "mbps", "cpu_fraction", "prb_slots", "slots"))
EVIDENCE = frozenset(("SIMULATION", "CONTROLLED_TEST", "CALIBRATED_FIELD_MODEL"))


def identifier(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", value) is None:
        raise ValueError("Invalid " + name)
    return value


def number(value, name, low=0., high=float("inf")):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Invalid " + name)


def integer(value, name, low=0, high=2**63 - 1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("Invalid " + name)


def hash64(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("Explicit SHA256 required")


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def document_hash(value):
    """Same canonical hashing as upstream; does not qualify the document."""
    return hashlib.sha256(canonical(asdict(value))).hexdigest()


@dataclass(frozen=True)
class Mission:
    robot: str
    priority: int
    required_mbps: float
    deadline_ms: float
    interruption_budget_ms: float
    max_predicted_failure: float

    def __post_init__(self):
        identifier(self.robot, "robot"); integer(self.priority, "priority", 1)
        number(self.required_mbps, "required_mbps", 0.000001)
        number(self.deadline_ms, "deadline_ms", 0.000001)
        number(self.interruption_budget_ms, "interruption_budget_ms")
        number(self.max_predicted_failure, "max_predicted_failure", 0, 1)


@dataclass(frozen=True)
class Resource:
    resource_id: str
    unit: str
    capacity: float
    external_used: float
    version: int
    observed_ms: int

    def __post_init__(self):
        identifier(self.resource_id, "resource")
        if self.unit not in UNITS: raise ValueError("Explicit supported resource unit required")
        number(self.capacity, "capacity"); number(self.external_used, "external_used", 0, self.capacity)
        integer(self.version, "version", 1); integer(self.observed_ms, "observed_ms")


@dataclass(frozen=True)
class Demand:
    resource_id: str
    unit: str
    amount: float
    version: int

    def __post_init__(self):
        identifier(self.resource_id, "resource")
        if self.unit not in UNITS: raise ValueError("Explicit supported demand unit required")
        number(self.amount, "amount"); integer(self.version, "version", 1)


@dataclass(frozen=True)
class Choice:
    robot: str
    action_id: str
    mode: str
    demands: tuple[Demand, ...]
    utility: float
    predicted_failure: float | None
    predicted_interruption_ms: float
    observed_ms: int
    valid_until_ms: int
    capabilities: tuple[str, ...]
    switching: bool

    def __post_init__(self):
        for key in ("robot", "action_id", "mode"): identifier(getattr(self, key), key)
        if (type(self.demands) is not tuple or not 1 <= len(self.demands) <= 64
                or not all(isinstance(d, Demand) for d in self.demands)
                or len({d.resource_id for d in self.demands}) != len(self.demands)):
            raise ValueError("Unique typed resource demands required")
        number(self.utility, "utility", -1e12, 1e12)
        if self.predicted_failure is not None: number(self.predicted_failure, "predicted_failure", 0, 1)
        number(self.predicted_interruption_ms, "predicted_interruption_ms")
        integer(self.observed_ms, "observed_ms"); integer(self.valid_until_ms, "valid_until_ms")
        if self.valid_until_ms <= self.observed_ms: raise ValueError("Empty choice validity window")
        if (type(self.capabilities) is not tuple or not 1 <= len(self.capabilities) <= 32
                or len(set(self.capabilities)) != len(self.capabilities)):
            raise ValueError("Explicit unique capability tuple required")
        for cap in self.capabilities: identifier(cap, "capability")
        if type(self.switching) is not bool: raise ValueError("switching must be bool")


@dataclass(frozen=True)
class ServiceForecast:
    robot: str
    action_id: str
    mission_hash: str
    choice_hash: str
    goodput_floor_mbps: float
    delay_ceiling_ms: float
    failure_upper: float
    event_definition: str
    horizon_ms: int
    estimator_id: str
    evidence_level: str
    observed_ms: int
    valid_until_ms: int

    def __post_init__(self):
        for key in ("robot", "action_id", "event_definition", "estimator_id"):
            identifier(getattr(self, key), key)
        hash64(self.mission_hash); hash64(self.choice_hash)
        number(self.goodput_floor_mbps, "goodput_floor_mbps")
        number(self.delay_ceiling_ms, "delay_ceiling_ms"); number(self.failure_upper, "failure_upper", 0, 1)
        integer(self.horizon_ms, "horizon_ms", 1); integer(self.observed_ms, "observed_ms")
        integer(self.valid_until_ms, "valid_until_ms")
        if self.valid_until_ms <= self.observed_ms: raise ValueError("Empty service-forecast interval")
        if self.evidence_level not in EVIDENCE: raise ValueError("Explicit evidence level required")


@dataclass(frozen=True)
class EstimatorScope:
    event_definition: str
    horizon_ms: int
    evidence_level: str

    def __post_init__(self):
        identifier(self.event_definition, "event_definition"); integer(self.horizon_ms, "horizon_ms", 1)
        if self.evidence_level not in EVIDENCE: raise ValueError("Explicit evidence level required")


@dataclass(frozen=True)
class OwnerPolicy:
    """Caller-owned approval configuration, never synthesized from forecasts.

    An ID is provenance, not proof of independent approval. Application owners
    must control this object separately from untrusted prediction submissions.
    """
    policy_id: str
    accepted_estimators: Mapping[str, EstimatorScope]
    supported_capabilities: Mapping[str, tuple[str, ...]]
    now_ms: int
    max_age_ms: int
    max_switches: int
    beam_width: int

    def __post_init__(self):
        identifier(self.policy_id, "owner policy")
        if not isinstance(self.accepted_estimators, Mapping) or len(self.accepted_estimators) > 64:
            raise ValueError("Explicit owner estimator registry required")
        for key, value in self.accepted_estimators.items():
            identifier(key, "estimator")
            if not isinstance(value, EstimatorScope): raise ValueError("Typed estimator scope required")
        if not isinstance(self.supported_capabilities, Mapping) or len(self.supported_capabilities) > 64:
            raise ValueError("Explicit owner capability registry required")
        supported = {}
        for robot, values in self.supported_capabilities.items():
            identifier(robot, "robot")
            if type(values) is not tuple or len(values) > 32 or len(set(values)) != len(values):
                raise ValueError("Explicit capability tuple required")
            for value in values: identifier(value, "capability")
            supported[robot] = values
        integer(self.now_ms, "now_ms"); integer(self.max_age_ms, "max_age_ms", 0, 60000)
        integer(self.max_switches, "max_switches", 0, 64); integer(self.beam_width, "beam_width", 1, 128)
        object.__setattr__(self, "accepted_estimators", MappingProxyType(dict(self.accepted_estimators)))
        object.__setattr__(self, "supported_capabilities", MappingProxyType(supported))
