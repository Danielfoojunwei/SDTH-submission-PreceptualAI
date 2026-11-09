"""Actual pinned upstream API integration using labelled software fixtures.

No RF data, network, hardware, learned model or scientific service trial.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from wirelessjev.r6_bridge import _worker
from wirelessjev.r6_bridge.contracts import (
    Choice, Demand, EstimatorScope, Mission, OwnerPolicy, Resource, ServiceForecast,
    canonical, document_hash,
)
from wirelessjev.r6_bridge.preview import defer_unqualified_outcomes, preview
from wirelessjev.worldmodel.interface import OutcomeDistribution

UPSTREAM = Path("/workspace/didjital-r6.1")


@pytest.fixture
def declared_software_inputs():
    # Explicit software declarations are intentionally not measured forecasts.
    mission = Mission("software-car-1", 1, 1., 50., 100., .05)
    resource = Resource("software-airtime", "airtime_fraction", 1., .1, 1, 1000)
    choice = Choice(mission.robot, "software-stay", "wifi", (Demand(resource.resource_id, resource.unit, .2, 1),),
                    3., .01, 0., 1000, 2000, ("preview_wifi_association",), False)
    forecast = ServiceForecast(mission.robot, choice.action_id, document_hash(mission), document_hash(choice),
                               2., 20., .01, "software_deadline_miss", 1000,
                               "software_fixture_estimator", "SIMULATION", 1000, 2000)
    owner = OwnerPolicy("software_owner_policy", {forecast.estimator_id: EstimatorScope(
        forecast.event_definition, forecast.horizon_ms, forecast.evidence_level)},
        {mission.robot: choice.capabilities}, 1100, 2000, 2, 32)
    return mission, resource, choice, forecast, owner


def invoke(values, *, choices=None, forecasts=None, owner=None, checkout=UPSTREAM):
    m, r, c, f, o = values
    return preview((m,), (r,), (c,) if choices is None else choices,
                   (f,) if forecasts is None else forecasts, owner=o if owner is None else owner, checkout=checkout)


def upstream_direct(values):
    """Independent normal import of the actual API in a separate interpreter."""
    m, r, c, f, o = values
    request = {"mission": asdict(m), "resource": asdict(r), "choice": asdict(c), "forecast": asdict(f),
               "scope": [f.event_definition, f.horizon_ms, f.evidence_level]}
    script = """
import json,sys
sys.path.insert(0,sys.argv[1]+'/preceptual_r6')
from preceptual.contracts import Mission,Resource,Demand,Choice
from preceptual.mission_plan import ServiceForecast,plan_missions
x=json.load(sys.stdin); d=x['choice'];d['demands']=tuple(Demand(**r) for r in d['demands']);d['capabilities']=tuple(d['capabilities'])
c=Choice(**d); f=ServiceForecast(**x['forecast'])
result=plan_missions([Mission(**x['mission'])],[Resource(**x['resource'])],[c],[f],accepted_estimators={f.estimator_id:tuple(x['scope'])},now_ms=1100,supported={c.robot:set(c.capabilities)},max_age_ms=2000,max_switches=2,beam_width=32)
print(json.dumps(result,sort_keys=True))
"""
    process = subprocess.run([sys.executable, "-I", "-S", "-c", script, str(UPSTREAM)],
                             input=canonical(request), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             check=True, timeout=5)
    return json.loads(process.stdout)


def test_actual_upstream_plan_schema_hash_and_result_match_independent_api(declared_software_inputs):
    before_path, before_modules = list(sys.path), set(sys.modules)
    result = invoke(declared_software_inputs)
    assert result.status == "PREVIEW", result.reason
    assert result.document()["plan"] == upstream_direct(declared_software_inputs)
    assert result.source["commit"] == _worker.UPSTREAM_COMMIT
    assert dict(result.source["files_sha256"]) == _worker.PINS
    assert result.plan["mission_requirements_enforced"] is True
    assert result.plan["actuation_performed"] is result.plan["physical_service_guaranteed"] is False
    assert result.plan["transition_qualified"] is False
    assert list(sys.path) == before_path
    assert not any(name.startswith("preceptual") for name in set(sys.modules) - before_modules)
    with pytest.raises(TypeError): result.plan["status"] = "DEPLOYED"


def test_missing_qualified_forecast_is_not_filled(declared_software_inputs):
    result = invoke(declared_software_inputs, forecasts=())
    assert result.status == "PREVIEW" and not result.plan["selected"]
    assert result.plan["rejected"][0]["reason"] == "MISSING_SERVICE_FORECAST"


@pytest.mark.parametrize("scope_change,reason", [
    ("absent", "UNAPPROVED_ESTIMATOR_SCOPE"),
    ("different_event", "UNAPPROVED_ESTIMATOR_SCOPE"),
    ("different_horizon", "UNAPPROVED_ESTIMATOR_SCOPE"),
    ("stale", "STALE_SERVICE_FORECAST"),
    ("missing_capability", "UNSUPPORTED_CAPABILITY"),
])
def test_owner_allowlist_scope_clock_and_capability_rejections(declared_software_inputs, scope_change, reason):
    m, r, c, f, o = declared_software_inputs
    if scope_change == "absent": o = replace(o, accepted_estimators={})
    if scope_change == "different_event": o = replace(o, accepted_estimators={f.estimator_id: EstimatorScope("other_event",1000,"SIMULATION")})
    if scope_change == "different_horizon": o = replace(o, accepted_estimators={f.estimator_id: EstimatorScope(f.event_definition,2000,"SIMULATION")})
    if scope_change == "stale": o = replace(o, now_ms=2000)
    if scope_change == "missing_capability": o = replace(o, supported_capabilities={})
    result = invoke(declared_software_inputs, owner=o)
    assert result.status == "PREVIEW" and not result.plan["selected"]
    assert result.plan["rejected"][0]["reason"] == reason


def test_demand_unit_mismatch_is_not_silently_converted(declared_software_inputs):
    m, r, c, f, o = declared_software_inputs
    c = replace(c, demands=(replace(c.demands[0], unit="mbps"),))
    f = replace(f, choice_hash=document_hash(c))
    result = invoke(declared_software_inputs, choices=(c,), forecasts=(f,))
    assert result.plan["rejected"][0]["reason"] == "UNIT_MISMATCH"


def test_mission_hash_mismatch_preserves_upstream_rejection(declared_software_inputs):
    f = replace(declared_software_inputs[3], mission_hash="0" * 64)
    result = invoke(declared_software_inputs, forecasts=(f,))
    assert result.plan["rejected"][0]["reason"] == "FORECAST_CONTEXT_MISMATCH"


def test_current_rf_outcome_proxy_defers_without_service_conversion():
    times = np.array((0., 1.)); shape = (1, 1, 2)
    outcome = OutcomeDistribution(("software-action",), times, np.full(shape, .1), np.full(shape, 2.),
                                   np.full((1, 1), .1), np.zeros((1, 1)), "software-rf-fixture", {})
    result = defer_unqualified_outcomes(outcome)
    assert result.status == "DEFER" and result.reason == "QUALIFIED_SERVICE_FORECAST_REQUIRED"
    assert result.plan is result.source is None


def test_wrong_or_missing_upstream_is_a_defer_not_heuristic_fallback(declared_software_inputs, tmp_path):
    result = invoke(declared_software_inputs, checkout=tmp_path)
    assert result.status == "DEFER" and result.reason == "UPSTREAM_SOURCE_OR_INPUT_REJECTED"


def test_modified_used_upstream_file_is_rejected_without_upstream_execution(tmp_path):
    root, files = _worker.verified_source(UPSTREAM)
    assert root == UPSTREAM
    assert set(files) == set(_worker.PINS)
    clone = tmp_path / "software-negative-checkout"
    subprocess.run(["git", "clone", "--shared", "--no-checkout", "--quiet", str(UPSTREAM), str(clone)],
                   check=True, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    package = clone / "preceptual_r6/preceptual"; package.mkdir(parents=True)
    for name, value in files.items(): (package / name).write_bytes(value)
    # The genuine pinned commit exists, but its temporary working source bytes
    # are altered. The actual upstream checkout is never edited.
    with (package / "planner.py").open("ab") as stream: stream.write(b"\n# not pinned\n")
    with pytest.raises(ValueError, match="UPSTREAM_SOURCE_MISMATCH"):
        _worker.verified_source(clone)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -.1])
def test_contracts_reject_nonphysical_numbers_and_bools(value):
    with pytest.raises(ValueError): Resource("r", "mbps", value, 0., 1, 1000)


def test_owner_registry_is_immutable_and_not_enrolled_from_forecasts(declared_software_inputs):
    owner = declared_software_inputs[-1]
    with pytest.raises(TypeError): owner.accepted_estimators["another"] = EstimatorScope("e",1,"SIMULATION")
    with pytest.raises(TypeError): owner.supported_capabilities["another"] = ("capability",)


@pytest.mark.parametrize("timeout", [True, 0, float("inf"), 20])
def test_unbounded_or_invalid_timeouts_are_rejected_before_subprocess(declared_software_inputs, timeout):
    m,r,c,f,o = declared_software_inputs
    with pytest.raises(ValueError, match="timeout"):
        preview((m,), (r,), (c,), (f,), owner=o, checkout=UPSTREAM, timeout_seconds=timeout)


def test_actual_preview_subprocess_timeout_is_a_defer(declared_software_inputs, monkeypatch):
    # Explicit software boundary fixture; no service result is synthesized.
    def expired(*args, **kwargs): raise subprocess.TimeoutExpired("software-child", .1)
    monkeypatch.setattr("wirelessjev.r6_bridge.preview.subprocess.run", expired)
    result = invoke(declared_software_inputs)
    assert result.status == "DEFER" and result.reason == "UPSTREAM_PREVIEW_TIMEOUT"


def test_preview_mission_count_budget_precedes_subprocess(declared_software_inputs):
    m, r, c, f, o = declared_software_inputs
    with pytest.raises(ValueError, match="Mission"):
        preview((m,) * 65, (r,), (c,), (f,), owner=o, checkout=UPSTREAM)


def test_unavailable_child_is_a_defer(declared_software_inputs, monkeypatch):
    def unavailable(*args, **kwargs): raise OSError("software fixture unavailable child")
    monkeypatch.setattr("wirelessjev.r6_bridge.preview.subprocess.run", unavailable)
    result = invoke(declared_software_inputs)
    assert result.status == "DEFER" and result.reason == "UPSTREAM_PREVIEW_UNAVAILABLE"


def test_child_response_cannot_upgrade_transition_qualification(declared_software_inputs, monkeypatch):
    from types import SimpleNamespace
    actual = invoke(declared_software_inputs).document()
    reply = {key: actual[key] for key in ("plan", "source", "owner_policy_id")}
    reply["plan"]["transition_qualified"] = True
    body = {k: v for k, v in reply["plan"].items() if k != "plan_hash"}
    reply["plan"]["plan_hash"] = __import__("hashlib").sha256(canonical(body)).hexdigest()
    def forbidden_upgrade(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=canonical(reply), stderr=b"")
    monkeypatch.setattr("wirelessjev.r6_bridge.preview.subprocess.run", forbidden_upgrade)
    result = invoke(declared_software_inputs)
    assert result.status == "DEFER" and result.reason == "INVALID_UPSTREAM_RESPONSE"
