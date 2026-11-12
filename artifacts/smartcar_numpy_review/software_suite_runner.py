"""Run every tracked test and retain before/after source hashes; no car access."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "software_suite.log"
RECEIPT = OUT / "software_suite.json"
if LOG.exists() or RECEIPT.exists():
    raise FileExistsError("Retain the existing software run; do not silently replace it")
tracked = subprocess.check_output(["git", "ls-files", "--", "src", "tests", "scripts",
    "deployment", "pyproject.toml", "uv.lock"], cwd=ROOT, text=True).splitlines()
source = sorted(name for name in tracked if name.endswith(".py") or name in ("pyproject.toml", "uv.lock"))
tests = sorted(name for name in tracked if name.startswith("tests/") and name.endswith(".py")
               and Path(name).name.startswith("test_"))


def hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source}


before = hashes()
began = time.perf_counter()
result = subprocess.run([sys.executable, "-m", "pytest", "--import-mode=importlib", "-q", *tests],
                        cwd=ROOT, text=True, capture_output=True)
elapsed = time.perf_counter()-began
after = hashes()
LOG.write_text(result.stdout+result.stderr)
passed = re.findall(r"(\d+) passed", result.stdout)
proof = {"schema": 1, "exit_code": result.returncode, "passed": int(passed[-1]) if passed else None,
    "checked_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": elapsed,
    "source_sha256_before": before, "source_sha256_after": after, "source_unchanged": before == after,
    "test_modules": tests, "log_path": str(LOG.relative_to(ROOT)),
    "log_sha256": hashlib.sha256(LOG.read_bytes()).hexdigest(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "deferred_untracked_observation_frontier_excluded": True,
    "fresh_real_car_captures_or_actuation": 0, "new_registered_scientific_experiments": 0,
    "meaning": "Software fixtures and archived-model parity, not real-car or SOTA qualification"}
RECEIPT.write_text(json.dumps(proof, indent=2, sort_keys=True)+"\n")
print(json.dumps({"exit_code": result.returncode, "passed": proof["passed"],
    "source_unchanged": before == after, "source_files": len(source), "test_modules": len(tests),
    "elapsed_seconds": elapsed}))
print("\n".join((result.stdout+result.stderr).splitlines()[-12:]))
if result.returncode or before != after:
    raise SystemExit(1)
