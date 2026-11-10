"""Read-only declared-input proposal preview through actual pinned R6.1 code."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

from . import _worker
from .contracts import Choice, Mission, OwnerPolicy, Resource, ServiceForecast, canonical


def _freeze(value):
    if isinstance(value, dict): return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list): return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class PreviewResult:
    status: str
    reason: str
    plan: object = None
    source: object = None
    owner_policy_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "plan", _freeze(self.plan))
        object.__setattr__(self, "source", _freeze(self.source))

    def document(self):
        def thaw(value):
            if isinstance(value, MappingProxyType): return {k: thaw(v) for k, v in value.items()}
            if isinstance(value, tuple): return [thaw(v) for v in value]
            return value
        return {"status": self.status, "reason": self.reason, "plan": thaw(self.plan),
                "source": thaw(self.source), "owner_policy_id": self.owner_policy_id}


def defer_unqualified_outcomes(outcomes):
    """RF-rate/outage proxies cannot manufacture qualified service forecasts."""
    from ..worldmodel.interface import OutcomeDistribution
    if not isinstance(outcomes, OutcomeDistribution):
        raise TypeError("Expected the current typed OutcomeDistribution")
    return PreviewResult("DEFER", "QUALIFIED_SERVICE_FORECAST_REQUIRED")


def preview(missions, resources, choices, forecasts, *, owner, checkout, timeout_seconds=5.):
    """No network/backend/journal writes; defaults never invent owner approval.

    Every input is explicit. An estimator's evidence label remains a declaration
    unless independently qualified by the application owner. This routine does
    not certify that qualification or activate any resulting action.
    """
    for values, kind, maximum, minimum in ((missions, Mission, 64, 1), (resources, Resource, 256, 1),
                                           (choices, Choice, 256, 0), (forecasts, ServiceForecast, 256, 0)):
        if type(values) is not tuple or not minimum <= len(values) <= maximum or not all(isinstance(v, kind) for v in values):
            raise ValueError("Bounded explicit typed tuples required: " + kind.__name__)
    if not isinstance(owner, OwnerPolicy): raise TypeError("Explicit caller-owned policy required")
    if (type(timeout_seconds) not in (float, int) or not math.isfinite(timeout_seconds)
            or not 0.1 <= timeout_seconds <= 10):
        raise ValueError("Bounded finite timeout between 0.1 and 10 seconds required")
    request = {"missions": [asdict(v) for v in missions], "resources": [asdict(v) for v in resources],
               "choices": [asdict(v) for v in choices], "forecasts": [asdict(v) for v in forecasts],
               "owner": {"policy_id": owner.policy_id,
                         "accepted_estimators": {k: [v.event_definition, v.horizon_ms, v.evidence_level]
                                                 for k, v in owner.accepted_estimators.items()},
                         "supported_capabilities": dict(owner.supported_capabilities),
                         "now_ms": owner.now_ms, "max_age_ms": owner.max_age_ms,
                         "max_switches": owner.max_switches, "beam_width": owner.beam_width}}
    payload = canonical(request)
    if len(payload) > _worker.MAX_BYTES: raise ValueError("Input byte budget exceeded")
    # The child verifies commit and all used source bytes before loading actual
    # upstream modules. -I/-S strips parent import paths and installed packages.
    command = [sys.executable, "-I", "-S", str(Path(_worker.__file__).resolve()), str(Path(checkout).resolve())]
    try:
        process = subprocess.run(command, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=float(timeout_seconds), cwd=Path(_worker.__file__).parent,
                                 env={"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"})
    except subprocess.TimeoutExpired:
        return PreviewResult("DEFER", "UPSTREAM_PREVIEW_TIMEOUT")
    except OSError:
        return PreviewResult("DEFER", "UPSTREAM_PREVIEW_UNAVAILABLE")
    if process.returncode or not process.stdout or len(process.stdout) > _worker.MAX_BYTES:
        return PreviewResult("DEFER", "UPSTREAM_SOURCE_OR_INPUT_REJECTED")
    try:
        response = json.loads(process.stdout)
        if (set(response) != {"plan", "source", "owner_policy_id"}
                or response["source"] != {"commit": _worker.UPSTREAM_COMMIT, "files_sha256": _worker.PINS}
                or response["owner_policy_id"] != owner.policy_id
                or response["plan"]["schema"] != "preceptual.mission_plan.v1"
                or response["plan"]["actuation_performed"] is not False
                or response["plan"]["physical_service_guaranteed"] is not False
                or response["plan"]["transition_qualified"] is not False):
            raise ValueError("Unexpected upstream preview boundary")
        body = {key: value for key, value in response["plan"].items() if key != "plan_hash"}
        if hashlib.sha256(canonical(body)).hexdigest() != response["plan"]["plan_hash"]:
            raise ValueError("Upstream plan hash differs")
    except (ValueError, KeyError, TypeError):
        return PreviewResult("DEFER", "INVALID_UPSTREAM_RESPONSE")
    return PreviewResult("PREVIEW", "DECLARED_INPUTS_ONLY", **response)
