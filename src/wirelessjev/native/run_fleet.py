"""Headroom ladder (docs/JEPA_HEADROOM.md), fleet-only capabilities
(docs/DIRECTIONS_123.md) and the live blockage detector (docs/BLOCKAGE_DETECTOR.md)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import time

from .run import OUT
from .sim import Policy, policies, simulate
from .world import build_world

CONDITIONS = {"main": (0.01, 0.0), "stress": (0.03, 10.0)}
LADDER_SCENES, FLEET_SCENES = (400, 401, 402, 403), tuple(range(550, 556))
DETECT_SCENES = tuple(range(560, 566))
DROP = ("per_robot_control_miss", "per_robot_video_on_time")


def _vert():
    pol = {p.name: p for p in policies()}
    return pol, replace(pol["native"], gate=True, online_map=True, power_control=True,
                        cadence_roam=True, adaptive_video=True)


def ladder_scene(seed: int, duration: float):
    _, vert = _vert()
    radios = {"single": vert, "two_path": replace(vert, dual_radio=True)}
    rows = []
    for cond, (lam, noise) in CONDITIONS.items():
        learned = build_world(seed, duration_s=duration, survey_s=60, noise_2g_extra_db=noise)
        perfect = build_world(seed, duration_s=duration, sigma_map=0.0, noise_2g_extra_db=noise)
        for radio, base in radios.items():
            rungs = {1: (learned, base), 2: (perfect, replace(base, gate=False, online_map=False)),
                     3: (perfect, replace(base, gate=False, online_map=False, oracle=2)),
                     4: (perfect, replace(base, gate=False, online_map=False, oracle=3))}
            for rung, (w, p) in rungs.items():
                r = simulate(w, replace(p, name=f"{radio}_r{rung}"), collision_lambda=lam, interference=True)
                rows.append({k: v for k, v in r.items() if k not in DROP}
                            | {"radio": radio, "rung": rung, "condition": cond})
    return rows


def fleet_arms() -> dict:
    pol, vert = _vert()
    single = vert
    shared = replace(vert, dual_radio=True, radio2="shared")
    return {"nr_30ms": Policy("nr_30ms", "nr", nr_ho_cost_s=0.03),
            "wifi8": Policy("wifi8", "wifi8", roam_cost_s=0.0, scan_period_s=30.0),
            "single": single, "shared": shared,
            "single+plan": replace(single, safe_plan_s=2.0),
            "single+plan+speed": replace(single, safe_plan_s=2.0, link_aware_speed=True),
            "single+relay": replace(single, relay=True), "single+ble": replace(single, ble=True),
            "single+relay+ble": replace(single, relay=True, ble=True),
            "shared+relay": replace(shared, relay=True), "shared+ble": replace(shared, ble=True),
            "shared+relay+ble": replace(shared, relay=True, ble=True)}


def fleet_scene(seed: int, duration: float):
    rows = []
    for cond, (lam, noise) in CONDITIONS.items():
        w = build_world(seed, duration_s=duration, survey_s=60, noise_2g_extra_db=noise)
        for name, p in fleet_arms().items():
            r = simulate(w, replace(p, name=name), collision_lambda=lam, interference=True)
            rows.append({k: v for k, v in r.items() if k not in DROP} | {"arm": name, "condition": cond})
    return rows


def ladder_eval(rows):
    out = {}
    for radio in ("single", "two_path"):
        for cond in CONDITIONS:
            m = {}
            for rung in (1, 2, 3, 4):
                rs = [r for r in rows if r["radio"] == radio and r["condition"] == cond and r["rung"] == rung]
                m[rung] = 1e6 * sum(r["control_misses"] for r in rs) / max(sum(r["control_sent"] for r in rs), 1)
            red = 1 - m[4] / m[1] if m[1] > 0 else 0.0
            out[f"{radio}|{cond}"] = {"misses_per_million": m, "rung4_reduction": red}
        out[f"{radio}_continue_control_learning"] = out[f"{radio}|main"]["rung4_reduction"] >= 0.25
    return out


def fleet_eval(rows):
    sc = sorted({r["scene"] for r in rows})
    v = lambda c, a, s, k="control_miss": next(r[k] for r in rows if r["condition"] == c and r["arm"] == a and r["scene"] == s)
    res = {"D1": sum(v("main", "single+plan", s, "plan_stop_fraction") <= v("main", "nr_30ms", s, "safety_stop_fraction") for s in sc),
           "D2": sum(v("main", "single+relay", s) <= 0.5 * v("main", "single", s) for s in sc),
           "D3": sum(v("main", "single+ble", s) <= 0.5 * v("main", "single", s) for s in sc),
           "D4": sum(v("stress", "shared+relay+ble", s) <= v("stress", "nr_30ms", s) for s in sc),
           "D5": sum(v("stress", "single+relay+ble", s) <= v("stress", "nr_30ms", s) for s in sc)}
    return {"scenes": sc, **{k: {"scenes": int(n), "pass": bool(n >= 5)} for k, n in res.items()}}


def detect_arms() -> dict:
    _, vert = _vert()
    shared = replace(vert, dual_radio=True, radio2="shared")
    return {"nr_30ms": Policy("nr_30ms", "nr", nr_ho_cost_s=0.03),
            "single": vert, "single+detect": replace(vert, block_detect=True),
            # Same controller, told the truth about which paths are blocked right now.
            "single+oracle": replace(vert, oracle=2),
            "shared": shared, "shared+detect": replace(shared, block_detect=True)}


def detect_scene(seed: int, duration: float):
    rows = []
    for cond, (lam, noise) in CONDITIONS.items():
        w = build_world(seed, duration_s=duration, survey_s=60, noise_2g_extra_db=noise)
        for name, p in detect_arms().items():
            r = simulate(w, replace(p, name=name), collision_lambda=lam, interference=True)
            rows.append({k: v for k, v in r.items() if k not in DROP} | {"arm": name, "condition": cond})
    return rows


def detect_eval(rows):
    sc = sorted({r["scene"] for r in rows})
    v = lambda c, a, s: next(r["control_miss"] for r in rows if r["condition"] == c and r["arm"] == a and r["scene"] == s)

    def pooled(c, a):
        rs = [r for r in rows if r["condition"] == c and r["arm"] == a]
        return 1e6 * sum(r["control_misses"] for r in rs) / max(sum(r["control_sent"] for r in rs), 1)

    m = {c: {a: pooled(c, a) for a in detect_arms()} for c in CONDITIONS}
    red = {c: 1 - m[c]["single+detect"] / m[c]["single"] if m[c]["single"] > 0 else 0.0 for c in CONDITIONS}
    wins = {c: sum(v(c, "single+detect", s) < v(c, "single", s) for s in sc) for c in CONDITIONS}
    gap = m["main"]["single"] - m["main"]["single+oracle"]
    share = (m["main"]["single"] - m["main"]["single+detect"]) / gap if gap > 0 else 0.0
    vs5g = sum(v("main", "single+detect", s) <= v("main", "nr_30ms", s) for s in sc)
    return {"scenes": sc, "misses_per_million": m,
            "B1": {"scenes": int(wins["main"]), "pooled_reduction": red["main"],
                   "pass": bool(wins["main"] >= 5 and red["main"] >= 0.30)},
            "B2": {"scenes": int(wins["stress"]), "pooled_reduction": red["stress"],
                   "pass": bool(wins["stress"] >= 5 and red["stress"] >= 0.25)},
            "B3": {"share_of_oracle_headroom": share, "pass": bool(share >= 0.40)},
            "B4": {"pass": bool(m["main"]["shared+detect"] <= m["main"]["shared"] + 5.0
                                and m["stress"]["shared+detect"] <= 1.15 * m["stress"]["shared"])},
            "B5": {"scenes": int(vs5g), "pass": bool(vs5g >= 5)}}


STUDIES = {"ladder": (ladder_scene, LADDER_SCENES, 120.0, ladder_eval, "headroom_ladder.json", "docs/JEPA_HEADROOM.md"),
           "fleet": (fleet_scene, FLEET_SCENES, 300.0, fleet_eval, "fleet_directions.json", "docs/DIRECTIONS_123.md"),
           "detect": (detect_scene, DETECT_SCENES, 300.0, detect_eval, "blockage_detector.json", "docs/BLOCKAGE_DETECTOR.md")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=tuple(STUDIES))
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    fn, scenes, dur, ev, name, proto = STUDIES[args.study]
    t = time.time()
    rows, partial = [], OUT / (name + ".partial")
    with ProcessPoolExecutor(args.workers) as pool:
        for part in pool.map(fn, scenes, [dur] * len(scenes)):
            rows += part
            # A campaign runs for hours: keep every finished scene, so a crash or a flat
            # battery loses one scene, not the run. The evaluation only ever uses complete runs.
            partial.write_text(json.dumps(rows, default=float))
    rep = {"protocol": proto, "duration_s": dur, "elapsed_s": time.time() - t, "rows": rows, "evaluation": ev(rows)}
    (OUT / name).write_text(json.dumps(rep, indent=1, default=float))
    partial.unlink(missing_ok=True)
    print(json.dumps(rep["evaluation"], indent=1, default=float))


if __name__ == "__main__":
    main()
