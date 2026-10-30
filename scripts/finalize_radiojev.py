"""Validate and archive a complete warehouse RadioJEV run; optionally wait for it.

This never starts, stops or changes training. --publish-branch allows a normal
commit/push of the generated evidence on that existing branch; it never merges.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("e1_no_llm", "e1_radio_lora", "e1_radio_frozen", "e1_intent",
          "e2_radio_q", "e2_e1_zero", "e2_intent", "e3", "e45")
SITES = ("S1_normal", "S2_heavy_scanners", "S3_dense_racks", "S4_light_racks")
CHECKPOINTS = ("ckpt_radio_lora.pt", "ckpt_radio_frozen.pt", "ckpt_radio_q.pt", "ckpt_e3.pt")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite report metric")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_reports(state, reports):
    require(state.get("smoke") is False, "Only full runs can be finalised")
    require(state.get("status") == "completed", "Training is not complete")
    require(set(reports) == set(STAGES), "Missing or unexpected stage reports")
    for name in STAGES:
        entry = state["stages"][name]
        require(entry["status"] == "completed" and entry.get("returncode") == 0,
                f"Stage has not succeeded: {name}")
        finite(reports[name])
        if name.startswith(("e1_", "e2_")):
            r = reports[name]
            n = 200 if name.startswith("e1_") else 120
            metric = "regret" if name.startswith("e1_") else "norm_regret"
            require(r["test"]["n"] == n, f"Wrong full-test count: {name}")
            require(0 <= r["test"][metric] and r["test"][metric + "_ci95"] >= 0,
                    f"Invalid test metric: {name}")
            novel_n = 0 if name == "e1_no_llm" else (240 if name.startswith("e1_") else 120)
            require(r["novel"]["n"] == novel_n, f"Wrong new-mission count: {name}")
            if novel_n:
                require(r["novel"][metric] >= 0 and r["novel"][metric + "_ci95"] >= 0,
                        f"Invalid new-mission metric: {name}")
    for name in ("e1_radio_lora", "e1_radio_frozen", "e2_radio_q"):
        require(len(reports[name]["history"]) == 4, f"Incomplete training history: {name}")
    r3 = reports["e3"]
    for key in ("shared_before", "shared_after"):
        require(r3[key]["assoc_norm_regret"] >= 0, "Invalid E3 association regret")
        require(0 <= r3[key]["admission_acc"] <= 1, "Invalid E3 admission accuracy")
        require(all(0 <= v <= 1 for v in r3[key]["traffic_acc"].values()), "Invalid E3 traffic accuracy")
    r = reports["e45"]
    require(set(r["init_per_site"]) == set(SITES), "Incomplete E4 initial-site evaluation")
    for arm in ("local", "fedavg_all", "swarm_pers"):
        a = r[arm]
        require(set(a["per_site"]) == set(SITES), f"Incomplete E4 sites: {arm}")
        values = list(a["per_site"].values())
        require(all(0 <= v <= 1 + 1e-6 for v in values), f"Invalid E4 regret: {arm}")
        require(math.isclose(a["mean"], sum(values) / len(values), abs_tol=1e-8), f"Inconsistent E4 mean: {arm}")
        require(math.isclose(a["worst"], max(values), abs_tol=1e-8), f"Inconsistent E4 worst: {arm}")
        require(a["log_verifies"] is True, f"Swarm audit log failed verification: {arm}")
    e5 = r["e5"]
    require(e5["forgotten_site"] in SITES, "Unknown E5 forgotten site")
    for key in ("mia_auc_before", "mia_auc_exact_retrain", "mia_auc_gated_distill", "mia_auc_never_trained"):
        require(0 <= e5[key] <= 1, f"Invalid E5 AUC: {key}")
    for key in ("retained_regret_before", "retained_regret_exact", "retained_regret_distill"):
        require(0 <= e5[key] <= 1 + 1e-6, f"Invalid E5 retained regret: {key}")
    require(e5["cost_s_exact"] >= 0 and e5["cost_s_distill"] >= 0, "Invalid E5 costs")


def checkpoint_metadata(path):
    import torch
    torch.set_num_threads(1)
    state = torch.load(path, map_location="cpu", weights_only=True)

    def count(value):
        if isinstance(value, dict):
            return sum(count(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(count(v) for v in value)
        if isinstance(value, torch.Tensor):
            require(bool(torch.isfinite(value).all()), f"Non-finite checkpoint: {path.name}")
            return 1
        return 0

    n = count(state)
    require(n > 0, f"Empty checkpoint: {path.name}")
    return {"sha256": sha(path), "bytes": path.stat().st_size, "finite_tensors": n}


def preserve_report(source, destination):
    if destination.exists():
        require(source.read_bytes() == destination.read_bytes(),
                f"Refusing to replace different archived evidence: {destination.name}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())


def render_results(reports, env, audit, complete):
    lines = ["# Warehouse connectivity results", "",
             f"Snapshot: {now()} (UTC).", "",
             "Scope: WirelessJEV and RadioJEV warehouse connectivity, E1–E5 and E7. "
             "E6 is excluded. Router, E8, foundation-model, custom-PHY and space extensions are deferred.", "",
             ("All nine E1–E5 controller stages completed and passed report/checkpoint validation."
              if complete else "E1–E3 are complete. E4/E5 is still running; partial log output is not a completed result."), "",
             "These are synthetic, single-training-seed results. Case-level confidence intervals do not "
             "measure independent-training variation or scene clustering. No physical-car trial is claimed.", "",
             "## E1: mission-conditioned plan selection", "",
             "| Arm | Test regret ± 95% CI (n=200) | New-mission regret ± 95% CI (n=240) | Test latency |",
             "|---|---|---|---|"]
    for arm in ("radio_lora", "radio_frozen", "intent", "no_llm"):
        r = reports["e1_" + arm]
        metric = lambda m: f"{m['regret']:.4f} ± {m['regret_ci95']:.4f}" if m["n"] else "Not evaluated"
        lines.append(f"| {arm} | {metric(r['test'])} | {metric(r['novel'])} | {r['test']['latency_ms']:.2f} ms |")
    lines += ["", "## E2: missions beyond the fixed-weight reader interface", "",
              "| Arm | Test normalized regret ± 95% CI (n=120) | New-mission normalized regret ± 95% CI (n=120) | Test latency |",
              "|---|---|---|---|"]
    for arm in ("radio_q", "e1_zero", "intent"):
        r = reports["e2_" + arm]
        metric = lambda m: f"{m['norm_regret']:.4f} ± {m['norm_regret_ci95']:.4f}"
        lines.append(f"| {arm} | {metric(r['test'])} | {metric(r['novel'])} | {r['test']['latency_ms']:.2f} ms |")
    r = reports["e3"]
    lines += ["", "E1 and E2 use different metrics. The E2 baseline is the current weight-based reader, "
              "not a structured parser with explicit constraints.", "", "## E3: multitask", "",
              f"Association normalized regret: {r['shared_before']['assoc_norm_regret']:.4f} → {r['shared_after']['assoc_norm_regret']:.4f}.", "",
              f"DSA greedy reward: {r['dsa_radiojev']['greedy_mean_reward']:.3f} over one 500-step evaluation episode. "
              "This does not establish robustness across channel layouts.", "",
              "| Traffic task | Before | After | Rule |", "|---|---|---|---|"]
    for key in ("trend", "event", "volatility"):
        lines.append(f"| {key} | {r['shared_before']['traffic_acc'][key]:.3f} | {r['shared_after']['traffic_acc'][key]:.3f} | {r['traffic_rule_acc'][key]:.3f} |")
    if audit:
        a = audit["admission"]
        lines += ["", f"Admission labels: train {a['train']['labels']} (n={a['train']['n']}), "
                  f"test {a['test']['labels']} (n={a['test']['n']}). "
                  f"Test majority-class accuracy is {a['test']['majority_accuracy']:.3f}; "
                  "the one-class dataset does not establish admission-policy generalisation."]
    lines += ["", "## E4/E5: site adaptation and forgetting", ""]
    if "e45" not in reports:
        lines += ["Pending completion. No complete E4/E5 comparison is available yet."]
    else:
        r = reports["e45"]
        lines += ["| Arm | Mean normalized regret | Worst site | Signed log verifies |", "|---|---|---|---|"]
        for arm in ("local", "fedavg_all", "swarm_pers"):
            a = r[arm]
            lines.append(f"| {arm} | {a['mean']:.4f} | {a['worst']:.4f} | {a['log_verifies']} |")
        lines += ["", "| Site | Initial E3 | Local | FedAvg | Personalised swarm |", "|---|---|---|---|---|"]
        for site in SITES:
            values = [r["init_per_site"][site]] + [r[a]["per_site"][site] for a in ("local", "fedavg_all", "swarm_pers")]
            lines.append("| " + site + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
        e5 = r["e5"]
        lines += ["", f"Forgotten site: {e5['forgotten_site']}.", "",
                  "| State | Membership-inference AUC | Retained normalized regret |", "|---|---|---|"]
        for label, auc_key, regret_key in (
            ("Before", "mia_auc_before", "retained_regret_before"),
            ("Retrain without site", "mia_auc_exact_retrain", "retained_regret_exact"),
            ("Retained-data distillation", "mia_auc_gated_distill", "retained_regret_distill"),
            ("Initial model", "mia_auc_never_trained", None)):
            regret = f"{e5[regret_key]:.4f}" if regret_key else "—"
            lines.append(f"| {label} | {e5[auc_key]:.4f} | {regret} |")
        lines += ["", f"Elapsed costs: retraining {e5['cost_s_exact']:.1f} s; distillation {e5['cost_s_distill']:.1f} s.", "",
                  "AUC measures one loss-based membership attack, not certified data deletion. The historical "
                  "`gated_distill` field names are preserved; the implemented student distils retained-site "
                  "outputs without a separate final release gate. E5's stated scope is the shared component; "
                  "personalised retained-site states are reused for its retained-service evaluation."]
    lines += ["", "## E7: corrected environment learning", ""]
    if env:
        s = env["summary"]
        lines += [f"Completed on {s['n_envs']} environments, five missions per environment.", "",
                  "| Survey | Audible RMSE (dB) | Plan gap to truth search |", "|---|---|---|"]
        for minutes in (1, 5, 15):
            key = f"learned_{minutes}min"
            lines.append(f"| {minutes} min | {s['map_rmse_db'][key]:.3f} | {s['regret_vs_truth_oracle'][key]:.3f} |")
        lines += ["", "The severe historical censoring regression is corrected, but additional data does not "
                  "improve every environment. Truth search is a local-search reference. The optional map-only "
                  "calibration study is not promoted to the planner."]
    else:
        lines += ["Corrected E7 report is unavailable in this checkout."]
    lines += ["", "## Evidence and remaining limits", "",
              "Raw cloud reports and provenance: `artifacts/radiojev_cloud/`. Original Mac reports remain "
              "unchanged in `artifacts/radiojev/`; original E7 evidence remains in `artifacts/envlearn/`.", "",
              "Independent training seeds, a meaningful admission dataset, calibration planning validation, "
              "real-car qualification and external Claude Studio publication remain separate work. "
              "Completing the specified experiment run does not establish those claims.", ""]
    return "\n".join(lines)


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, text=True, capture_output=True).stdout.strip()


def require_clean_branch(root, branch):
    require(git(root, "branch", "--show-current") == branch, "Checkout moved to another branch")
    require(not git(root, "status", "--porcelain", "--untracked-files=all"),
            "Checkout has changes; automatic finalisation will not overwrite or commit them")


def finalise(root, run, state, publish_branch=None):
    reports = {name: json.loads((run / f"{name}.json").read_text()) for name in STAGES}
    validate_reports(state, reports)
    if publish_branch:
        require_clean_branch(root, publish_branch)
    attempts = sorted((run / "attempts").glob("e45_*/provenance.json"))
    source_manifest = json.loads((attempts[-1] if attempts else run / "manifest.json").read_text())
    required_sources = {f"src/wirelessjev/radiojev/{name}.py" for name in (
        "__init__", "candidates", "data", "dsa", "e1", "e2", "e3", "e45", "encoder", "features", "nonlinear", "trainable")}
    require(required_sources <= set(source_manifest["source_sha256"]), "Incomplete training-source manifest")
    for name, expected in source_manifest["source_sha256"].items():
        if name.startswith("src/wirelessjev/radiojev/") or name == "scripts/train_radiojev.py":
            require(sha(root / name) == expected, f"Training source changed since recorded attempt: {name}")
    checkpoints = {name: checkpoint_metadata(run / name) for name in CHECKPOINTS}
    out = root / "artifacts/radiojev_cloud"
    provenance_path = out / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {
        "run_manifest": json.loads((run / "manifest.json").read_text())}
    for name, previous in provenance.get("verified_checkpoints", {}).items():
        if name in checkpoints:
            require(checkpoints[name]["sha256"] == previous["sha256"], f"Previously verified checkpoint changed: {name}")
    # Check every old archive before writing anything into it.
    for name in STAGES:
        old = out / f"{name}.json"
        if old.exists():
            require(old.read_bytes() == (run / old.name).read_bytes(), f"Archived report differs: {old.name}")
    for name in STAGES:
        preserve_report(run / f"{name}.json", out / f"{name}.json")
    provenance.update(run_complete=True, latest_exported_at=now(), seed=state.get("seed"), smoke=False,
                      pending_stages=[], evidence_class="digital twin (synthetic)", device="CPU",
                      completed_stages={k: {a: v for a, v in state["stages"][k].items()
                                          if a in ("status", "started_at", "finished_at", "returncode")} for k in STAGES},
                      result_sha256={name + ".json": sha(run / f"{name}.json") for name in STAGES},
                      verified_checkpoints=checkpoints, finalizer_sha256=sha(Path(__file__)))
    write_json(provenance_path, provenance)
    from wirelessjev.studio_data import training_snapshot
    snapshot = training_snapshot(run)
    write_json(out / "snapshot.json", snapshot)
    env_path = root / "artifacts/envlearn_corrected/report.json"
    audit_path = out / "e3_label_audit.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else None
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    summary = render_results(reports, env, audit, complete=True)
    (root / "docs/WAREHOUSE_RESULTS.md").write_text(summary)
    (out / "README.md").write_text(
        "# Completed cloud RadioJEV run\n\nAll nine E1–E5 controller stages completed. "
        "See [warehouse results](../../docs/WAREHOUSE_RESULTS.md) for the complete comparisons and limits.\n\n"
        "Stage JSON files are unchanged raw reports. `provenance.json` records their hashes, stage completion "
        "and verified checkpoint hashes; `snapshot.json` is a static Studio export. The E4/E5 retry has separate "
        "source provenance. Original reports are preserved. This is one synthetic training seed, not field validation.\n")
    subprocess.run([sys.executable, "-m", "wirelessjev.studio_data", "--run-dir", str(run)], cwd=root, check=True)
    subprocess.run([sys.executable, "studio/build_artifact.py"], cwd=root, check=True)
    if publish_branch:
        paths = [f"artifacts/radiojev_cloud/{name}.json" for name in STAGES]
        paths += ["artifacts/radiojev_cloud/provenance.json", "artifacts/radiojev_cloud/snapshot.json",
                  "artifacts/radiojev_cloud/README.md", "docs/WAREHOUSE_RESULTS.md", "studio/data/bundle.json"]
        git(root, "add", *paths)
        if subprocess.run(["git", "-C", str(root), "diff", "--cached", "--quiet"]).returncode:
            git(root, "commit", "-m", "Archive completed warehouse E1–E5 results")
        git(root, "push", "origin", publish_branch)
    return {"status": "completed", "validated_stages": list(STAGES), "finalised_at": now(),
            "published_branch": publish_branch, "git_head": git(root, "rev-parse", "HEAD")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--publish-branch")
    args = parser.parse_args()
    require(1 <= args.poll_seconds <= 60, "Poll interval must be 1–60 seconds")
    run = args.run_dir.resolve()
    lock = (run / "finalizer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    progress = run / "finalization_status.json"
    try:
        transitions = 0
        while True:
            state = json.loads((run / "status.json").read_text())
            if state["status"] == "completed":
                write_json(progress, {"status": "validating", "pid": os.getpid(), "checked_at": now()})
                result = finalise(ROOT, run, state, args.publish_branch)
                write_json(progress, result)
                print(json.dumps(result), flush=True)
                return 0
            require(args.watch and state["status"] == "running", f"Training is {state['status']}; not finalised")
            from wirelessjev.studio_data import _training_process_alive
            active = [name for name, entry in state["stages"].items() if entry["status"] == "running"]
            alive = (_training_process_alive(state.get("pid"), run) is not False and
                     all(_training_process_alive(state["stages"][name].get("pid"), run) is not False
                         for name in active))
            if not alive:
                # A worker can exit before its parent records completion, or a
                # running-stage record can precede the child's PID assignment.
                latest = json.loads((run / "status.json").read_text())
                if latest != state:
                    transitions = 0
                    continue
                transitions += 1
                require(transitions < 5, "Training process stopped without a completed status")
                time.sleep(1)
                continue
            transitions = 0
            waiting = {"status": "waiting_for_training", "pid": os.getpid(), "checked_at": now(),
                       "completed_stages": [n for n, e in state["stages"].items() if e["status"] == "completed"],
                       "active_stages": active}
            write_json(progress, waiting)
            print(json.dumps(waiting), flush=True)
            time.sleep(args.poll_seconds)
    except Exception as error:
        write_json(progress, {"status": "failed", "checked_at": now(), "reason": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
