"""Build studio/data/bundle.json from the real run artifacts.

    uv run python -m wirelessjev.studio_data

Everything the Studio shows comes from this bundle: benchmark reports, the
closed-loop replay frames and decisions, the warehouse layout, the hardware
profile and the paper map. Missing artifacts are reported as missing, never
filled with placeholder numbers.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
import time

from .hardware import RASPBOT
from .missions import MISSIONS, WAREHOUSE_MISSIONS
from .policy import MODEL_REPO, MODEL_REVISION
from .twin import warehouse as W

ROOT = Path(__file__).resolve().parents[2]


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


def build() -> dict:
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
    out = ROOT / "studio" / "data" / "bundle.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    bundle = build()
    out.write_text(json.dumps(bundle, separators=(",", ":"), default=float))
    missing = [k for k in ("bench_10car", "bench_warehouse", "closedloop_10car",
                           "closedloop_warehouse") if bundle[k] is None]
    print(f"{out} ({out.stat().st_size / 1e6:.2f} MB); missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
