"""Independently verify finite profile receipts/statistics without timed calls.

Reads only public saved-plan and geometry members; does not call either query
operator, any model, simulator or learning API. Frozen sources remain untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from wirelessjev.native_bridge_ops import profile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/native_bridge_ops"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def main():
    source = profile.check_sources()
    protocol, report = (read(OUT / name) for name in ("protocol.json", "report.json"))
    manifest_before = digest(profile.FROZEN)
    report_before = digest(OUT / "report.json")
    assert report["passes"] is True
    assert report["source_receipt_sha256"] == digest(OUT / "registration/source.json")
    assert report["publication_sha256"] == digest(OUT / "registration/git.json")
    assert report["protocol_sha256"] == digest(OUT / "protocol.json")
    assert report["frozen_native_manifest_sha256_before"] == report["frozen_native_manifest_sha256_after"] == manifest_before
    assert manifest_before == profile.FROZEN_SHA256
    assert report["geometry_work"] == protocol["geometry_work"]
    assert report["exact_operand_byte_parity_every_plan_and_timed_pair"] is True
    assert report["new_worlds"] == report["new_native_cases"] == report["model_forwards"] == report["fits"] == 0
    assert report["native_runner_changed"] is report["SLA_claim"] is report["service_improvement_claim"] is False
    actual_inputs = {(r["case_id"], r["plan_key"]): r for r in report["public_inputs"]}
    expected_inputs = [(e["case_id"], key) for e in protocol["inputs"] for key in e["plan_keys"]]
    assert len(actual_inputs) == len(report["public_inputs"]) == len(expected_inputs) == 48
    assert set(actual_inputs) == set(expected_inputs)
    for entry in protocol["inputs"]:
        refs = [entry["public_geometry_archive"], entry["candidate_archive"]]
        for ref in refs:
            assert digest(ROOT / ref["path"]) == ref["sha256"]
        with np.load(ROOT / refs[0]["path"], allow_pickle=False) as saved:
            sites, tx, noise = (np.asarray(saved[key], dtype=float) for key in ("sites", "tx_dl", "noise_db"))
        with np.load(ROOT / refs[1]["path"], allow_pickle=False) as saved:
            for key in entry["plan_keys"]:
                row, plan = actual_inputs[(entry["case_id"], key)], saved[key]
                assert row["plan_sha256"] == array_sha(plan)
                assert row["sites_sha256"] == array_sha(sites)
                assert row["tx_sha256"] == array_sha(tx)
                assert row["noise_sha256"] == array_sha(noise)
                assert row["plan_shape"] == list(plan.shape) == [100, 9, 2]
    rounds = report["timed_pairs"]
    assert len(rounds) == 144
    expected_pairs = {(case, key, trial) for case, key in expected_inputs for trial in range(3)}
    assert {(r["case_id"], r["plan_key"], r["round"]) for r in rounds} == expected_pairs
    index = {key: ordinal for ordinal, key in enumerate(expected_inputs)}
    first_counts = {"reference": 0, "reuse": 0}
    for row in rounds:
        parity = (index[(row["case_id"], row["plan_key"])] + row["round"]) % 2
        assert row["call_order"] == (["reference", "reuse"] if parity == 0 else ["reuse", "reference"])
        first_counts[row["call_order"][0]] += 1
        assert type(row["reference_ns"]) is int and row["reference_ns"] > 0
        assert type(row["reuse_ns"]) is int and row["reuse_ns"] > 0
    assert first_counts == {"reference": 72, "reuse": 72}
    a, b = (np.array([r[k] for r in rounds], dtype=float) for k in ("reference_ns", "reuse_ns"))
    recomputed = {"reference_median_ms": float(np.median(a) / 1e6), "reuse_median_ms": float(np.median(b) / 1e6),
                  "ratio_of_median_call_times": float(np.median(a) / np.median(b)),
                  "median_paired_speedup": float(np.median(a / b)),
                  "reference_min_max_ms": [float(a.min() / 1e6), float(a.max() / 1e6)],
                  "reuse_min_max_ms": [float(b.min() / 1e6), float(b.max() / 1e6)],
                  "timed_calls_per_operator": 144, "warmup_calls_per_operator": 48}
    assert recomputed == report["summary"]
    assert report["environment"]["thread_limits"] == protocol["thread_limits"]
    proof = read(OUT / "preflight/tests.json")
    assert proof["exit_code"] == 0
    assert proof["source_sha256_before"] == proof["source_sha256_after"] == source["source_sha256"]
    profile.check_sources()
    assert digest(profile.FROZEN) == manifest_before and digest(OUT / "report.json") == report_before
    review = {"schema": 1, "passes": True, "review_script_sha256": digest(__file__),
              "report_sha256_before": report_before, "report_sha256_after": digest(OUT / "report.json"),
              "frozen_native_manifest_sha256_before": manifest_before,
              "frozen_native_manifest_sha256_after": digest(profile.FROZEN),
              "registered_new_source_files": len(source["source_sha256"]),
              "preserved_native_sources": 512, "preserved_native_evidence": 56,
              "public_input_hashes_independently_reconstructed": 48,
              "timed_pairs_independently_checked": 144, "first_method_counts": first_counts,
              "summary": recomputed, "extra_operator_calls": 0, "extra_timing_calls": 0,
              "model_forwards": 0, "new_worlds": 0, "new_packet_simulations": 0,
              "source_publication": read(OUT / "registration/git.json"),
              "co_running_cpu_context": "Root reports comprehensive tracked tests (one CPU) and adaptive prediction were running concurrently. This was not an isolated CPU experiment.",
              "qualification": "Descriptive operand-call microbenchmark including allocation and physical-feature math; no full-model, service-latency or SLA claim."}
    destination = Path(__file__).with_name("review.json")
    destination.write_text(json.dumps(review, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passes": True, "path": str(destination.relative_to(ROOT)), "sha256": digest(destination),
                      "summary": recomputed}, sort_keys=True))


if __name__ == "__main__":
    main()
