"""Build studio/data/bundle.json from the real run artifacts.

    uv run python -m wirelessjev.studio_data

Everything the Studio shows comes from this bundle: benchmark reports, the
closed-loop replay frames and decisions, the warehouse layout, the hardware
profile and the paper map. Missing artifacts are reported as missing, never
filled with placeholder numbers.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time

from .hardware import RASPBOT
from .missions import MISSIONS, WAREHOUSE_MISSIONS
from .policy import MODEL_REPO, MODEL_REVISION
from .twin import warehouse as W

ROOT = Path(__file__).resolve().parents[2]
RADIOJEV_STAGES = ("e1_no_llm", "e1_radio_lora", "e1_radio_frozen", "e1_intent",
                   "e2_radio_q", "e2_e1_zero", "e2_intent", "e3", "e45")


def _load(rel: str):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


def _paper_map() -> list[dict]:
    text = (ROOT / "docs" / "PAPER_MAP.md").read_text()
    rows = []
    for line in text.splitlines():
        if not line.startswith("| ") or line.startswith("| Paper") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4:
            rows.append({"paper": cells[0], "status": cells[1],
                         "mechanism": cells[2], "code": cells[3]})
    return rows


def _slim_closedloop(cl: dict | None) -> dict | None:
    if cl is None:
        return None
    runs = []
    for r in cl["runs"]:
        keep = {k: r[k] for k in ("policy", "seed", "mean_utility", "total_handovers",
                                  "transactions", "connects_beyond_one_per_txn",
                                  "final_reputation") if k in r}
        keep["mean_critical_violations"] = r.get("mean_critical_violations")
        keep["timeline"] = [{k: s[k] for k in ("t", "mission", "utility", "handovers",
                                               "critical_violations", "total_mbps",
                                               "coordinator_up")}
                            for s in r["timeline"]]
        for k in ("frames", "robots", "aps", "occupied_channels", "txn_log", "decisions"):
            if k in r:
                keep[k] = r[k]
        runs.append(keep)
    return {"summary": cl["summary"], "runs": runs, "provisional": cl.get("provisional")}


def _training_process_alive(pid, run_dir: Path) -> bool | None:
    if not Path("/proc").exists():
        return None
    if not isinstance(pid, int) or pid < 1:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return (any(script in command for script in (b"radiojev_train_all.py", b"train_radiojev.py"))
            and str(run_dir.resolve()).encode() in command)


def training_snapshot(run_dir: Path) -> dict:
    """Export only validated stage results and a point-in-time progress view."""
    state = json.loads((run_dir / "status.json").read_text())
    if state.get("smoke") is not False:
        raise ValueError("Only full training runs may be exported")
    stages, results = {}, {}
    for name in RADIOJEV_STAGES:
        entry = state.get("stages", {}).get(name, {"status": "pending"})
        status = entry["status"]
        if status == "running" and _training_process_alive(entry.get("pid"), run_dir) is False:
            status = "interrupted"
        stage = {"status": status}
        if status == "completed":
            result = json.loads((run_dir / f"{name}.json").read_text())
            if not isinstance(result, dict) or not result:
                raise ValueError(f"Invalid completed training result: {name}")
            results[name] = result
        log = run_dir / f"{name}.log"
        if log.exists():
            history = []
            pattern = r"epoch (\d+) train KL ([\d.eE+-]+) val regret ([\d.eE+-]+)"
            for epoch, loss, regret in re.findall(pattern, log.read_text(errors="replace")):
                if math.isfinite(float(loss)) and math.isfinite(float(regret)):
                    if history and int(epoch) <= history[-1]["epoch"]:
                        history = []  # A restarted arm appended a new attempt.
                    history.append({"epoch": int(epoch), "train_kl": float(loss),
                                    "val_regret": float(regret)})
            if history:
                stage["history"] = history
        stages[name] = stage
    interrupted = (any(s["status"] == "interrupted" for s in stages.values())
                   or (state["status"] == "running" and
                       _training_process_alive(state.get("pid"), run_dir) is False))
    return {"captured_at": datetime.now(timezone.utc).isoformat(),
            "status": "interrupted" if interrupted
            else state["status"], "seed": state.get("seed"), "stages": stages,
            "results": results, "snapshot": True}


def build(run_dir: Path | None = None) -> dict:
    layout = {
        "width": W.WIDTH, "height": W.HEIGHT, "rack_rows": list(W.RACK_ROWS),
        "rack_segments": [list(s) for s in W.RACK_SEGMENTS],
        "cross_aisles": list(W.CROSS_AISLES), "aisles": list(W.AISLES),
        "drop_points": [list(p) for p in W.DROP_POINTS],
        "receive_points": [list(p) for p in W.RECEIVE_POINTS],
        "chargers": [list(p) for p in W.CHARGERS],
        "ap_sites": [list(p) for p in W.AP_SITES],
        "rack_db": W.RACK_DB,
    }
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evidence_class": "digital twin (synthetic); no car or radio was touched",
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "hardware": asdict(RASPBOT),
        "layout": layout,
        "bench_10car": _load("artifacts/bench/report.json"),
        "bench_warehouse": _load("artifacts/warehouse/report.json"),
        "bench_intent": _load("artifacts/intent/report.json"),
        "envlearn": (_load("artifacts/envlearn_corrected/report.json")
                     or _load("artifacts/envlearn/report.json")),
        "envlearn_legacy": _load("artifacts/envlearn/report.json"),
        "envlearn_calibration": _load("artifacts/envlearn_calibration/report.json"),
        "radiojev": {name: _load(f"artifacts/radiojev/{name}.json") for name in (
            "e1_radio_lora", "e1_radio_frozen", "e1_intent", "e1_no_llm",
            "e2_radio_q", "e2_e1_zero", "e2_intent", "e3", "e45")},
        "radiojev_audits": {"e3": _load("artifacts/radiojev_cloud/e3_label_audit.json")},
        "training_run": (training_snapshot(run_dir) if run_dir
                         else _load("artifacts/radiojev_cloud/snapshot.json")),
        "warehouse_v2": _load("artifacts/warehouse_v2/snapshot.json"),
        "warehouse_horizon": _load("artifacts/warehouse_horizon/snapshot.json"),
        "warehouse_horizon_selector": _load("artifacts/warehouse_horizon/selector/snapshot.json"),
        "closedloop_10car": _slim_closedloop(_load("artifacts/closedloop.json")),
        "closedloop_warehouse": _slim_closedloop(_load("artifacts/closedloop_warehouse.json")),
        "papers": _paper_map(),
        "missions": {k: {"intent": m.intent, "focus": list(m.focus_cars)}
                     for k, m in {**MISSIONS, **WAREHOUSE_MISSIONS}.items()},
        "smartcarcmd_contract": {
            "invocation": 'sudo -n python3 -u "$HOME"/SmartCarCmd.py --interface wlan1 '
                          '--management-interface wlan0',
            "status": {"action": "status"},
            "connect": {"action": "connect", "ssid": "<from WiFi.json>",
                        "bssid": "aa:bb:cc:dd:ee:ff", "security": "wpa-psk",
                        "password": "<injected at send time; never logged>"},
            "outcomes": {
                "APPLIED": "ok: true and BSSID verified by status",
                "NOT_APPLIED": "AP_NOT_VISIBLE or INTERFACE_UNSAFE (activation_attempted: false)",
                "UNKNOWN": "timeout or channel loss after send; reconcile by status, never resend",
            },
        },
        "one_car_test": {"status": "not yet run",
                         "plan": "docs/ONE_CAR_TEST.md",
                         "logs": sorted(p.name for p in (ROOT / "artifacts" / "onecar").glob("*.jsonl")
                                        if "preflight" not in p.name)
                         if (ROOT / "artifacts" / "onecar").exists() else []},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="include a snapshot of a full training run")
    args = parser.parse_args()
    out = ROOT / "studio" / "data" / "bundle.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    bundle = build(args.run_dir)
    out.write_text(json.dumps(bundle, separators=(",", ":"), default=float))
    missing = [k for k in ("bench_10car", "bench_warehouse", "closedloop_10car",
                           "closedloop_warehouse") if bundle[k] is None]
    print(f"{out} ({out.stat().st_size / 1e6:.2f} MB); missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
