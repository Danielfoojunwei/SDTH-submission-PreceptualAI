"""Isolated stdlib-only execution of exact pinned upstream planning source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

UPSTREAM_COMMIT = "a458a90d34fe559addb876a2a1c69213c6ecc968"
PINS = {
    "__init__.py": "e7b2b0b6f063817623fcc37f284580a77b419694646e7f22358f5ed483fedfa0",
    "contracts.py": "ac9111e4408960ae268a412ad3531511cb224d4ce9caeb9e9f6c78a7e2e28514",
    "planner.py": "80d3ccf0901dad11d031975f450241889b1ec726742cfa762c06dfbf50e367a1",
    "mission_plan.py": "d003f59c4598fe9509ca623a101a642a0ba4ccfde7c9b8da5e02c7c81953150e",
}
MAX_BYTES = 1 << 20


def verified_source(checkout):
    root = Path(checkout).resolve(strict=True)
    process = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=3)
    if process.stdout.decode().strip() != UPSTREAM_COMMIT:
        raise ValueError("UPSTREAM_COMMIT_MISMATCH")
    source = {}
    for name, expected in PINS.items():
        path = root / "preceptual_r6/preceptual" / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("UPSTREAM_PATH_REDIRECTED")
        value = path.read_bytes()
        if hashlib.sha256(value).hexdigest() != expected:
            raise ValueError("UPSTREAM_SOURCE_MISMATCH")
        source[name] = value
    return root, source


def load_verified_modules(root, source):
    """Compile the exact verified bytes; no global install or path injection."""
    loaded = {}
    for filename in ("__init__.py", "contracts.py", "planner.py", "mission_plan.py"):
        name = "preceptual" if filename == "__init__.py" else "preceptual." + filename[:-3]
        module = types.ModuleType(name)
        module.__file__ = str(root / "preceptual_r6/preceptual" / filename)
        module.__package__ = "preceptual"
        if filename == "__init__.py": module.__path__ = []
        sys.modules[name] = module
        exec(compile(source[filename], module.__file__, "exec"), module.__dict__)
        loaded[name] = module
    return loaded


def main():
    raw = sys.stdin.buffer.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES: raise ValueError("INPUT_BYTE_BUDGET")
    request = json.loads(raw)
    if set(request) != {"missions", "resources", "choices", "forecasts", "owner"}:
        raise ValueError("INPUT_SCHEMA")
    root, source = verified_source(sys.argv[1])
    modules = load_verified_modules(root, source)
    c, m = modules["preceptual.contracts"], modules["preceptual.mission_plan"]
    missions = [c.Mission(**r) for r in request["missions"]]
    resources = [c.Resource(**r) for r in request["resources"]]
    choices = []
    for r in request["choices"]:
        body = dict(r, demands=tuple(c.Demand(**d) for d in r["demands"]), capabilities=tuple(r["capabilities"]))
        choices.append(c.Choice(**body))
    forecasts = [m.ServiceForecast(**r) for r in request["forecasts"]]
    owner = request["owner"]
    plan = m.plan_missions(missions, resources, choices, forecasts,
                          accepted_estimators={k: tuple(v) for k, v in owner["accepted_estimators"].items()},
                          supported={k: set(v) for k, v in owner["supported_capabilities"].items()},
                          now_ms=owner["now_ms"], max_age_ms=owner["max_age_ms"],
                          max_switches=owner["max_switches"], beam_width=owner["beam_width"])
    if (plan["actuation_performed"] is not False or plan["physical_service_guaranteed"] is not False
            or plan["transition_qualified"] is not False):
        raise ValueError("UPSTREAM_READ_ONLY_BOUNDARY")
    verified_source(root)
    response = {"plan": plan, "source": {"commit": UPSTREAM_COMMIT, "files_sha256": PINS},
                "owner_policy_id": owner["policy_id"]}
    output = json.dumps(response, sort_keys=True, allow_nan=False).encode()
    if len(output) > MAX_BYTES: raise ValueError("OUTPUT_BYTE_BUDGET")
    sys.stdout.buffer.write(output)


if __name__ == "__main__":
    main()
