"""Intent flip inside the native stack: mission text, read by frozen d3-lite
with the distilled intent readout, changes what every layer does."""
from __future__ import annotations

import json
from pathlib import Path

from .intent_policy import intent_policy
from .run import OUT
from .sim import simulate
from .world import build_world

ROOT = Path(__file__).resolve().parents[3]
MISSIONS = {
    "protect_teleop": "Teleoperation of amr-017 is the top priority right now: keep its video "
                      "and control link rock solid, even at the cost of other robots' uploads.",
    "bulk_uploads": "End-of-shift map uploads: maximise the total data the whole fleet delivers. "
                    "Reconnections are fine.",
    # Added after seeing the bulk run: amr-067 was the worst-served robot there.
    "protect_worst": "amr-067 is carrying a safety-critical inspection right now. Protect its link "
                     "above every other robot.",
    "scanner_quiet": "Barcode scanners are in heavy use across the floor. Keep robots off their "
                     "channels and keep interference low.",
}


def read_missions():
    from safetensors.torch import load_file
    from ..intent import IntentReader
    from ..policy import WirelessJEV
    engine = WirelessJEV()
    readout = load_file(str(ROOT / "artifacts/intent/adapters/intent/readout.safetensors"))["readout"]
    reader = IntentReader(engine, readout=readout)
    out = {}
    for name, text in MISSIONS.items():
        r = reader.read(text)
        out[name] = {"text": text, "weights": r.weights, "focus": list(r.focus), "levels": r.levels,
                     "latency_ms": r.latency_ms}
    return out


def main(scene: int = 401, duration: float = 120.0):
    readings = read_missions()
    w = build_world(scene, duration_s=duration, survey_s=60)
    rows = []
    for name, rd in readings.items():
        p = intent_policy(rd["weights"], rd["focus"])
        r = simulate(w, p, collision_lambda=0.01, interference=True)
        rows.append({"mission": name, "knobs": {k: getattr(p, k) for k in (
            "roam_gain_db", "load_penalty", "gate_z", "avoid_occupied", "power_control",
            "video_share", "focus_robots", "dual_radio")}, **r})
        print(name, json.dumps(rows[-1]["knobs"], default=str), {k: rows[-1][k] for k in (
            "control_miss", "focus_control_miss", "focus_video_on_time", "video_on_time",
            "bulk_mbps", "roams", "mean_tx_reduction_db")},
              "amr-017 miss", r["per_robot_control_miss"][16], "amr-017 video", r["per_robot_video_on_time"][16],
              "amr-067 miss", r["per_robot_control_miss"][66],
              flush=True)
    (OUT / "intent_flip.json").write_text(json.dumps(
        {"scene": scene, "duration_s": duration, "evidence": "development scene, exploratory",
         "readings": readings, "rows": rows}, indent=1, default=float))


if __name__ == "__main__":
    main()
