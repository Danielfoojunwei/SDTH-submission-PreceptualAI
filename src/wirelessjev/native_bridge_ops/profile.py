"""Finite, publication-gated profiling of sealed public native plan operands.

No model or packet calls, no new worlds, no RF/observation arrays opened. The
source and protocol must be published before the saved plan arrays are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tarfile
import time

import numpy as np

from ..native_bridge import predictor as reference
from . import operators

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/native_bridge_ops"
PROTOCOL = OUT / "protocol.json"
FROZEN = ROOT / "artifacts/native_bridge/verification/manifest.json"
FROZEN_SHA256 = "4d9f95316d9e029365ed64d4b04e43128a49239c4115ebb8a6bb49f11c12aeb9"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_once(path, value):
    path = Path(path)
    payload = encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError("Immutable profiling artifact differs: " + str(path))
        return
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def safe_path(name):
    path = ROOT / name
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError("Profiling artifact path escapes repository")
    return path


def frozen_closure():
    if digest(FROZEN) != FROZEN_SHA256:
        raise RuntimeError("The preserved native evidence manifest changed")
    value = read(FROZEN)
    if len(value["source_sha256"]) != 512 or len(value["evidence_sha256"]) != 56:
        raise RuntimeError("Preserved native source/evidence membership changed")
    for field in ("source_sha256", "evidence_sha256"):
        for name, expected in value[field].items():
            if digest(safe_path(name)) != expected:
                raise RuntimeError("Preserved native bytes changed: " + name)
    return value


def protocol_spec():
    frozen = frozen_closure()
    entries = []
    for seed in (90473001, 90473002, 90473003, 90473004):
        candidate = f"artifacts/native_bridge/cases/dev-{seed}-ho0.03/candidate_inputs_forecasts.npz"
        world = f"artifacts/native_bridge/worlds/{seed}/world.npz"
        entries.append({"case_id": f"dev-{seed}-ho0.03",
                        "candidate_archive": {"path": candidate, "sha256": frozen["evidence_sha256"][candidate]},
                        "public_geometry_archive": {"path": world, "sha256": frozen["evidence_sha256"][world]},
                        "plan_keys": [f"job{index}_plan" for index in range(12)]})
    return {"schema": 1, "protocol_id": "native-public-query-geometry-reuse-20261109-v1",
            "task": "operand-only timing on existing sealed public planned positions; no learned or packet outcomes",
            "frozen_native_manifest": {"path": str(FROZEN.relative_to(ROOT)), "sha256": FROZEN_SHA256},
            "inputs": entries, "allowed_geometry_keys": ["sites", "tx_dl", "noise_db"],
            "planned_input_shape": [100, 9, 2], "output_shape": [100, 9, 16, 3],
            "operators": ["frozen_query_operands", "band_reuse_query_operands"],
            "warmup_calls_per_operator_per_plan": 1, "timed_paired_rounds_per_plan": 3,
            "pair_order": "reference then reuse when (plan ordinal + round ordinal) is even, otherwise reuse then reference",
            "clock": "time.perf_counter_ns; operands only, excluding input I/O, parity checks and registration/byte verification",
            "primary_summary": "median reference call ns / median reuse call ns over all144 timed calls per operator",
            "parity": "identical shape and dtype and exact tobytes() for every query,basis,fspl,offset result on warmups and after every timed pair",
            "geometry_work": {"full_queries_per_call": 43200, "reference_AABB_links_per_call": 43200,
                              "reuse_AABB_links_per_call": 14400, "rack_boxes": 33},
            "thread_limits": {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            "forbidden": ["new_world", "packet_simulation", "model_forward", "training", "RF_array_access",
                          "observation_array_access", "native_runner_modification", "SLA_or_service_improvement_claim"],
            "limitations": ["One managed CPU environment and fixed existing public plans; descriptive microbenchmark only.",
                            "Operator timings exclude model rendering, context, queue, planner, controller and physical radio cost.",
                            "No native service cases rerun; previous18 simulated native outcomes remain unchanged.",
                            "Same-process warmups/repeated calls are correlated and are not independent deployment trials."]}


def write_spec():
    value = protocol_spec()
    write_once(PROTOCOL, value)
    return {"protocol_sha256": digest(PROTOCOL), "actual_plan_arrays_opened": False, "timing_calls": 0}


def sources():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list((ROOT / "tests/native_bridge_ops").glob("test_native_bridge_ops*.py"))
    paths += [PROTOCOL]
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def preflight():
    if read(PROTOCOL) != protocol_spec():
        raise RuntimeError("Write the exact profiling protocol before fixture proof")
    before = sources()
    command = ["uv", "run", "--frozen", "pytest", "-q", "tests/native_bridge_ops"]
    process = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if process.returncode != 0 or sources() != before:
        raise RuntimeError(process.stdout.decode(errors="replace"))
    log = OUT / "preflight/tests.log"
    write_once(log, {"stdout": process.stdout.decode(), "command": command})
    value = {"source_sha256_before": before, "source_sha256_after": sources(), "exit_code": 0,
             "command": command, "test_output": {"path": str(log.relative_to(ROOT)), "sha256": digest(log)},
             "frozen_native_manifest_sha256": FROZEN_SHA256,
             "actual_plan_arrays_opened": False, "timing_calls": 0}
    write_once(OUT / "preflight/tests.json", value)
    return value


def register():
    if read(PROTOCOL) != protocol_spec():
        raise RuntimeError("Profiling protocol differs from fixed public-input design")
    proof = read(OUT / "preflight/tests.json")
    if (proof["exit_code"] != 0 or proof["source_sha256_before"] != sources()
            or proof["source_sha256_after"] != sources()
            or digest(safe_path(proof["test_output"]["path"])) != proof["test_output"]["sha256"]):
        raise RuntimeError("Complete exact-source fixture proof before registration")
    if (OUT / "attempt.json").exists() or (OUT / "report.json").exists():
        raise RuntimeError("Register before any saved-plan profiling attempt")
    value = {"schema": 1, "source_sha256": sources(), "protocol_sha256": digest(PROTOCOL),
             "preflight_sha256": {"artifacts/native_bridge_ops/preflight/tests.json": digest(OUT / "preflight/tests.json"),
                                  proof["test_output"]["path"]: proof["test_output"]["sha256"]},
             "frozen_native_manifest_sha256": FROZEN_SHA256,
             "actual_plan_arrays_opened_before_registration": False, "timing_calls_before_registration": 0}
    write_once(OUT / "registration/source.json", value)
    return value


def check_sources(*, published=True):
    frozen_closure()
    receipt = read(OUT / "registration/source.json")
    if receipt["source_sha256"] != sources() or receipt["protocol_sha256"] != digest(PROTOCOL):
        raise RuntimeError("Registered profiling source or protocol changed")
    for name, expected in receipt["preflight_sha256"].items():
        if digest(safe_path(name)) != expected:
            raise RuntimeError("Profiling preflight evidence changed")
    if not published:
        return receipt
    publication = read(OUT / "registration/git.json")
    if (publication["source_receipt_sha256"] != digest(OUT / "registration/source.json")
            or publication["commit"] != publication["remote_observed_commit"]):
        raise RuntimeError("Publish and observe profiling source before opening saved plans")
    _verify_commit(publication["commit"], receipt)
    return receipt


def _verify_commit(commit, receipt):
    if not isinstance(commit, str) or not re.fullmatch("[0-9a-f]{40}", commit):
        raise RuntimeError("Exact published profiling commit required")
    expected = receipt["source_sha256"] | receipt["preflight_sha256"]
    expected["artifacts/native_bridge_ops/registration/source.json"] = digest(OUT / "registration/source.json")
    process = subprocess.Popen(["git", "archive", "--format=tar", commit, *expected], cwd=ROOT, stdout=subprocess.PIPE)
    seen = set()
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile() or member.name not in expected:
                raise RuntimeError("Unexpected published profiling source member")
            if hashlib.sha256(archive.extractfile(member).read()).hexdigest() != expected[member.name]:
                raise RuntimeError("Published profiling source differs")
            seen.add(member.name)
    if process.wait() or seen != set(expected):
        raise RuntimeError("Published profiling source is incomplete")


def publish(commit, remote, branch):
    receipt = check_sources(published=False)
    _verify_commit(commit, receipt)
    process = subprocess.run(["git", "ls-remote", "--exit-code", remote, "refs/heads/" + branch],
                             cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if process.stdout.decode().split()[0] != commit:
        raise RuntimeError("Exact profiling source commit is not the observed remote branch head")
    value = {"commit": commit, "remote": remote, "branch": branch, "remote_observed_commit": commit,
             "source_receipt_sha256": digest(OUT / "registration/source.json")}
    write_once(OUT / "registration/git.json", value)
    return value


def assert_operand_bytes(a, b):
    if set(a) != set(b) or a["shape"] != b["shape"]:
        raise RuntimeError("Operand output keys or native shape changed")
    for name in ("query", "basis", "fspl", "offset"):
        if a[name].dtype != b[name].dtype or a[name].shape != b[name].shape or a[name].tobytes() != b[name].tobytes():
            raise RuntimeError("Frozen operand byte parity failed: " + name)


def _input_reference(ref):
    path = safe_path(ref["path"])
    if digest(path) != ref["sha256"]:
        raise RuntimeError("Saved public input archive differs from protocol")
    return path


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.dtype.str, value.shape)).encode() + value.tobytes()).hexdigest()


def run():
    # All publication/source/thread guards run before any saved array is opened.
    check_sources()
    p = read(PROTOCOL)
    if any(os.environ.get(name) != value for name, value in p["thread_limits"].items()):
        raise RuntimeError("Use registered single-thread numerical environment")
    if (OUT / "attempt.json").exists() or (OUT / "report.json").exists():
        raise RuntimeError("Profiling attempt already exists; do not silently repeat")
    write_once(OUT / "attempt.json", {"source_receipt_sha256": digest(OUT / "registration/source.json"),
               "publication_sha256": digest(OUT / "registration/git.json"), "started_unix_ns": time.time_ns(),
               "finite_public_plan_inputs": 48, "timed_calls_per_operator": 144,
               "new_native_cases": 0, "model_forwards": 0})
    rows, inputs, ordinal = [], [], 0
    functions = {"reference": reference.query_operands, "reuse": operators.query_operands}
    for entry in p["inputs"]:
        world_path = _input_reference(entry["public_geometry_archive"])
        plan_path = _input_reference(entry["candidate_archive"])
        # np.load reads only these individually requested public members.
        # No World is constructed and no other archive array is interpreted.
        with np.load(world_path, allow_pickle=False) as saved:
            geometry = operators.PublicQueryGeometry(saved["sites"], saved["tx_dl"], saved["noise_db"])
        with np.load(plan_path, allow_pickle=False) as saved:
            for key in entry["plan_keys"]:
                plan = saved[key].copy()
                if list(plan.shape) != p["planned_input_shape"]:
                    raise RuntimeError("Saved planned-position shape differs")
                warm = {name: function(geometry, plan) for name, function in functions.items()}
                assert_operand_bytes(warm["reference"], warm["reuse"])
                record = {"case_id": entry["case_id"], "plan_key": key, "plan_sha256": array_sha(plan),
                          "sites_sha256": array_sha(geometry.sites), "tx_sha256": array_sha(geometry.tx),
                          "noise_sha256": array_sha(geometry.noise), "plan_shape": list(plan.shape)}
                inputs.append(record)
                for trial in range(p["timed_paired_rounds_per_plan"]):
                    order = ["reference", "reuse"] if (ordinal + trial) % 2 == 0 else ["reuse", "reference"]
                    outputs, timings = {}, {}
                    for name in order:
                        start = time.perf_counter_ns()
                        output = functions[name](geometry, plan)
                        timings[name] = time.perf_counter_ns() - start
                        outputs[name] = output
                    assert_operand_bytes(outputs["reference"], outputs["reuse"])
                    rows.append({"case_id": entry["case_id"], "plan_key": key, "round": trial,
                                 "call_order": order, "reference_ns": timings["reference"], "reuse_ns": timings["reuse"]})
                ordinal += 1
    if ordinal != 48 or len(rows) != 144:
        raise RuntimeError("Finite profiling input/round membership changed")
    check_sources()
    native_after = digest(FROZEN)
    first, second = (np.array([r[name + "_ns"] for r in rows], dtype=float) for name in ("reference", "reuse"))
    result = {"schema": 1, "passes": True, "protocol_sha256": digest(PROTOCOL),
              "source_receipt_sha256": digest(OUT / "registration/source.json"),
              "publication_sha256": digest(OUT / "registration/git.json"),
              "frozen_native_manifest_sha256_before": FROZEN_SHA256,
              "frozen_native_manifest_sha256_after": native_after,
              "exact_operand_byte_parity_every_plan_and_timed_pair": True,
              "public_inputs": inputs, "timed_pairs": rows,
              "summary": {"reference_median_ms": float(np.median(first) / 1e6),
                          "reuse_median_ms": float(np.median(second) / 1e6),
                          "ratio_of_median_call_times": float(np.median(first) / np.median(second)),
                          "median_paired_speedup": float(np.median(first / second)),
                          "reference_min_max_ms": [float(first.min() / 1e6), float(first.max() / 1e6)],
                          "reuse_min_max_ms": [float(second.min() / 1e6), float(second.max() / 1e6)],
                          "timed_calls_per_operator": 144, "warmup_calls_per_operator": 48},
              "geometry_work": p["geometry_work"], "limitations": p["limitations"],
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "numpy": np.__version__, "cpu_count": os.cpu_count(),
                              "cgroup_cpu_max": (Path("/sys/fs/cgroup/cpu.max").read_text().strip()
                                                 if Path("/sys/fs/cgroup/cpu.max").is_file() else None),
                              "thread_limits": {name: os.environ.get(name) for name in p["thread_limits"]}},
              "new_worlds": 0, "new_native_cases": 0, "model_forwards": 0, "fits": 0,
              "native_runner_changed": False, "SLA_claim": False, "service_improvement_claim": False}
    write_once(OUT / "report.json", result)
    return {"passes": True, "report_sha256": digest(OUT / "report.json"), "summary": result["summary"]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("write-spec", "preflight", "register", "publish", "run", "check"))
    parser.add_argument("--commit"); parser.add_argument("--remote", default="origin"); parser.add_argument("--branch")
    args = parser.parse_args(argv)
    if args.phase == "publish":
        result = publish(args.commit, args.remote, args.branch)
    else:
        result = {"write-spec": write_spec, "preflight": preflight, "register": register,
                  "run": run, "check": check_sources}[args.phase]()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
