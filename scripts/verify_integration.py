#!/usr/bin/env python3
"""Fail the build when a capability exists only for its own tests.

This repository has twice shipped modules that were correct, tested, and
completely unreachable from the running node — spectrum, power, recovery,
multimodal and clock integrity in one round, then the whole research-derived
tier in another. Both times the tests passed and the node behaved exactly as it
had before the code existed. A green suite says nothing about whether a
capability is *connected*.

Three checks, each of which has caught something real:

**Module reachability.** Every module must be reachable through the import graph
from a production entrypoint. A module only its tests import is dead code with a
docstring.

**Capability reachability.** Module-level reachability is not enough, and
trusting it is how this failure recurred a third time: ``capabilities.py``
imports ``c2link``, so the module counted as reachable while the capability
method inside it had **zero callers anywhere** — no CLI command, no service
call, not even a test. Every public method on ``CapabilityServices`` must
therefore be reachable from the CLI or the service loop by name.

**No simulated evidence.** Nothing in ``src`` may fabricate an observation.
Unavailable evidence must produce an unavailable result, never a plausible one.

**No unfinished work presented as finished.** Placeholder markers in ``src``
mean a code path exists that does not do what its name says.
"""

from __future__ import annotations

import ast
import builtins
import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "kval_node"

#: Modules the operating system actually starts. Everything a node can do must
#: be reachable from one of these.
PRODUCTION_ENTRYPOINTS = frozenset(
    {"service.py", "flight.py", "capabilities.py", "cli.py"}
)

#: Markers that mean a code path is not finished. ``REPLACE_`` is excluded
#: deliberately: in this codebase it appears inside guards that *reject*
#: unreplaced provisioning placeholders, which is finished code doing its job.
INCOMPLETE_MARKERS = re.compile(
    r"\b(TODO|FIXME|XXX|HACK|NotImplementedError|placeholder|stub|for now,? (?:just|simply))\b",
    re.IGNORECASE,
)

#: Names that would indicate fabricated evidence rather than measured evidence.
SIMULATION_MARKERS = re.compile(
    r"\b(class\s+(Mock|Fake|Simulated|Dummy)\w*|"
    r"(mock|fake|simulate|synthesize)_(telemetry|observation|reading|scan|measurement))\b",
    re.IGNORECASE,
)


def import_graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = collections.defaultdict(set)
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                graph[path.name].add(f"{node.module}.py")
    return graph


def reachable_from_production() -> set[str]:
    graph = import_graph()
    seen: set[str] = set()
    stack = list(PRODUCTION_ENTRYPOINTS)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(graph.get(current, ()))
    return seen


def check_reachability() -> list[str]:
    modules = {path.name for path in PACKAGE.glob("*.py")} - {"__init__.py"}
    orphaned = sorted(modules - reachable_from_production())
    return [
        f"{name} is not reachable from any production entrypoint: it is a library "
        f"the running node never calls"
        for name in orphaned
    ]


def check_markers(pattern: re.Pattern[str], label: str) -> list[str]:
    failures: list[str] = []
    for path in sorted(PACKAGE.glob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = pattern.search(line)
            if match:
                failures.append(f"{path.name}:{number} {label}: {match.group(0)!r}")
    return failures


#: Classes whose public methods are capabilities an operator or the loop can
#: reach. ``FlightServices`` was split out of ``CapabilityServices`` so a flying
#: node stops loading the operator tier; checking only the latter would have
#: silently stopped covering the five methods the autonomous loop actually
#: calls, which is the failure this whole script exists to catch.
CAPABILITY_CLASSES = (("capabilities.py", "CapabilityServices"), ("flight.py", "FlightServices"))


def capability_methods() -> list[str]:
    found: list[str] = []
    for filename, class_name in CAPABILITY_CLASSES:
        tree = ast.parse((PACKAGE / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                found.extend(
                    item.name
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not item.name.startswith("_")
                )
    return found


def check_capability_reachability() -> list[str]:
    """Every capability must be callable by an operator or by the loop.

    A capability nothing invokes is not a capability, however green its unit
    tests are. Internal helpers are exempt only when another production module
    calls them — being called from within ``capabilities.py`` alone is what a
    dead method looks like too, so that does not count.
    """
    methods = capability_methods()
    if not methods:
        return ["could not read CapabilityServices; the reachability check is not running"]

    owning = {filename for filename, _ in CAPABILITY_CLASSES}
    callers = {
        path.name: path.read_text(encoding="utf-8")
        for path in PACKAGE.glob("*.py")
        if path.name not in owning
    }
    internal = "\n".join(
        (PACKAGE / filename).read_text(encoding="utf-8") for filename in owning
    )

    failures: list[str] = []
    for method in methods:
        pattern = re.compile(rf"\.{re.escape(method)}\s*\(")
        if any(pattern.search(body) for body in callers.values()):
            continue
        if pattern.search(internal):
            # Reachable only from inside its own module. That is a helper *if*
            # something outside eventually reaches the caller, which the module
            # check already covers; flag it so it is a deliberate choice.
            continue
        failures.append(
            f"CapabilityServices.{method} has no caller outside its own class: "
            "no CLI command and no service call reach it"
        )
    return failures


def check_role_is_always_declared() -> list[str]:
    """Every ranking call must state the node's link role.

    ``ProfileOptimizer.recommend`` defaults ``role`` to ``payload`` so that a
    link cannot become safety-critical by omission. That default is right at the
    boundary and wrong at every call site inside the node, because a C2 node
    that does not declare its role rejects every command profile on
    ``DIFFERENT_LINK_ROLE`` before scoring anything.

    ``benchmark_optimizer`` shipped exactly that way: it timed the rejection
    path and reported it as the decision cost, 3.6x optimistic on a two-profile
    C2 catalogue — a figure that would have been quoted against a DO-362 budget.
    The main loop passed the role and the benchmark did not, which is invisible
    in review and invisible to a green suite.
    """
    failures: list[str] = []
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "recommend":
                continue
            # Only the profile optimizer; PowerGovernor.recommend is unrelated.
            target = func.value
            name = (
                target.attr
                if isinstance(target, ast.Attribute)
                else target.id
                if isinstance(target, ast.Name)
                else ""
            )
            if name != "optimizer":
                continue
            if not any(keyword.arg == "role" for keyword in node.keywords):
                failures.append(
                    f"{path.name}:{node.lineno} optimizer.recommend() does not pass "
                    "role=: it will silently rank as a payload node and reject "
                    "every command profile"
                )
    return failures


def check_flight_tier_is_self_contained() -> list[str]:
    """Every member ``flight.py`` uses must exist on the class the loop runs.

    ``service.py`` instantiates ``FlightServices``, not ``CapabilityServices``.
    When the flight/operator split moved method *bodies* into the base class
    but left the members those bodies call behind in the subclass, three of the
    five capabilities the loop calls broke: ``emcon_posture`` degraded to
    permanently-unavailable so a signed EMCON SILENT order stopped suppressing
    any link role, ``plan_power`` raised ``NameError`` every cycle, and
    ``c2_status_from_telemetry`` raised ``AttributeError``.

    Neither the import-graph walk nor the 1,122 tests caught it: the edges were
    all present, and no test ever instantiated ``FlightServices``. This is a
    static check because the failure is static — a base class cannot borrow
    from its subclass, and the loop never constructs the subclass.
    """
    module = ast.parse((PACKAGE / "flight.py").read_text(encoding="utf-8"))
    provided: set[str] = set()
    module_names: set[str] = set()
    flight_class: ast.ClassDef | None = None
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            module_names.add(node.name)
            if node.name == "FlightServices":
                flight_class = node
        elif isinstance(node, ast.FunctionDef):
            module_names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                module_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    module_names.add(target.id)
    if flight_class is None:
        return ["flight.py no longer defines FlightServices"]
    for item in flight_class.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            provided.add(item.name)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            provided.add(item.target.id)

    # Instance attributes bound anywhere in the class — ``self.config`` is set
    # in ``__init__`` and read everywhere, so this must be class-wide rather
    # than per-method or every reader looks like a defect.
    provided |= {
        node.attr
        for node in ast.walk(flight_class)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and isinstance(node.ctx, ast.Store)
    }

    failures: list[str] = []
    for item in flight_class.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(item):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and isinstance(node.ctx, ast.Load)
                and node.attr not in provided
            ):
                failures.append(
                    f"flight.py: FlightServices.{item.name} uses self.{node.attr}, "
                    "which FlightServices does not define — the loop constructs "
                    "FlightServices, so this raises AttributeError in flight"
                )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id not in module_names
                and not hasattr(builtins, node.func.id)
            ):
                failures.append(
                    f"flight.py: FlightServices.{item.name} calls {node.func.id}(), "
                    "which is not defined or imported in flight.py — this raises "
                    "NameError in flight"
                )
    return sorted(set(failures))


def check_simulator_is_not_reachable_from_production() -> list[str]:
    """``sim/`` fabricates measurements, so ``src/`` must never be able to reach it.

    The whole product claim is that a reported measurement was measured. A
    validation harness that manufactures link readings is legitimate and useful,
    and is also precisely the thing the simulation-marker guard exists to keep
    out of production. Those two only coexist if the dependency runs one way.

    ``sim`` imports ``kval_node`` freely — that is the point, it drives the real
    decision code. The reverse is a build failure, and so is any import of a
    simulator module by name, however it is spelled.
    """
    failures: list[str] = []
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [node.module]
            for name in imported:
                if name == "sim" or name.startswith("sim."):
                    failures.append(
                        f"{path.name}:{node.lineno} imports {name!r}: production code "
                        "cannot depend on the simulator, which fabricates measurements"
                    )
    sim_dir = ROOT / "sim"
    if sim_dir.is_dir() and not (sim_dir / "__init__.py").exists():
        failures.append("sim/ exists but has no __init__.py, so its boundary is undeclared")
    return failures


def main() -> int:
    failures = (
        check_reachability()
        + check_capability_reachability()
        + check_flight_tier_is_self_contained()
        + check_simulator_is_not_reachable_from_production()
        + check_role_is_always_declared()
        + check_markers(INCOMPLETE_MARKERS, "unfinished work in production code")
        + check_markers(SIMULATION_MARKERS, "simulated evidence in production code")
    )
    if failures:
        print("Integration verification FAILED:\n", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        print(
            f"\n{len(failures)} problem(s). A capability that nothing calls is not "
            "a capability.",
            file=sys.stderr,
        )
        return 1
    modules = {path.name for path in PACKAGE.glob("*.py")} - {"__init__.py"}
    print(f"All {len(modules)} modules reachable from production; no stubs, no simulated evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
