#!/usr/bin/env python3
"""Fail the build when a capability exists only for its own tests.

This repository has twice shipped modules that were correct, tested, and
completely unreachable from the running node — spectrum, power, recovery,
multimodal and clock integrity in one round, then the whole research-derived
tier in another. Both times the tests passed and the node behaved exactly as it
had before the code existed. A green suite says nothing about whether a
capability is *connected*.

Three checks, each of which has caught something real:

**Reachability.** Every module must be reachable through the import graph from
a production entrypoint. A module only its tests import is dead code with a
docstring.

**No simulated evidence.** Nothing in ``src`` may fabricate an observation.
Unavailable evidence must produce an unavailable result, never a plausible one.

**No unfinished work presented as finished.** Placeholder markers in ``src``
mean a code path exists that does not do what its name says.
"""

from __future__ import annotations

import ast
import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "kval_node"

#: Modules the operating system actually starts. Everything a node can do must
#: be reachable from one of these.
PRODUCTION_ENTRYPOINTS = frozenset(
    {"service.py", "capabilities.py", "cli.py", "sensor_cli.py"}
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


def main() -> int:
    failures = (
        check_reachability()
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
