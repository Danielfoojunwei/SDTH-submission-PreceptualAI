"""A strict numeric mission subset; free text is not an actuation schema."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
from ..worldmodel.interface import finite_nonnegative


@dataclass(frozen=True)
class MissionSpec:
    mission_id: str
    outage_weight: float = 1.
    rate_weight: float = 0.
    switching_weight: float = 0.
    max_outage_probability: float | None = None
    max_silence_seconds: float | None = None
    require_calibrated_outage: bool = True

    def __post_init__(self):
        if not isinstance(self.mission_id, str) or not self.mission_id:
            raise ValueError('A mission identity is required')
        for name in ('outage_weight', 'rate_weight', 'switching_weight'):
            object.__setattr__(self, name, finite_nonnegative(getattr(self, name), name))
        if self.outage_weight + self.rate_weight + self.switching_weight <= 0:
            raise ValueError('At least one objective weight is required')
        if self.max_outage_probability is not None:
            p = finite_nonnegative(self.max_outage_probability, 'max_outage_probability')
            if p > 1:
                raise ValueError('An outage probability bound must lie in [0,1]')
            object.__setattr__(self, 'max_outage_probability', p)
        if self.max_silence_seconds is not None:
            object.__setattr__(self, 'max_silence_seconds',
                finite_nonnegative(self.max_silence_seconds, 'max_silence_seconds'))
        if type(self.require_calibrated_outage) is not bool:
            raise ValueError('Calibration requirement must be a boolean')


def compile_mission(value):
    """Compile only a validated mapping, with unknown keys rejected.

    max_silence_seconds is a supported *request*, not a guarantee. A planner
    rejects it when the selected world model has no qualified silence head.
    There is deliberately no arbitrary text or general STL parser here.
    """
    if not isinstance(value, Mapping):
        raise TypeError('Mission text requires an external checked parser; use a typed mapping')
    fields = set(MissionSpec.__dataclass_fields__)
    if not set(value) <= fields or 'mission_id' not in value:
        raise ValueError('Unknown or missing typed mission fields')
    return MissionSpec(**dict(value))
