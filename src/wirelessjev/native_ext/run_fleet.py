"""Headroom ladder (docs/JEPA_HEADROOM.md), fleet-only capabilities
(docs/DIRECTIONS_123.md), the live blockage detector (docs/BLOCKAGE_DETECTOR.md)
the moving-blocker stress test of the headline claims (docs/MOVING_BLOCKERS.md)
foresight from obstacle sensing (docs/FORESIGHT.md) and relay realism (docs/RELAY_REALISM.md)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import time

from ..native.run import OUT
from .sim_fleet import Policy, policies, simulate
from ..native.world import build_world

CONDITIONS = {"main": (0.01, 0.0), "stress": (0.03, 10.0)}
LADDER_SCENES, FLEET_SCENES = (400, 401, 402, 403), tuple(range(550, 556))
DETECT_SCENES = tuple(range(560, 566))
MOVING_SCENES = tuple(range(570, 576))
FORESIGHT_SCENES = tuple(range(580, 586))
RELAY_SCENES = tuple(range(590, 596))
SHARED_SENSING_SCENES = tuple(range(600, 606))
TRANSFER_SCENES = tuple(range(620, 626))
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


def moving_arms() -> dict:
    _, vert = _vert()
    shared = replace(vert, dual_radio=True, radio2="shared")
    return {"nr_30ms": Policy("nr_30ms", "nr", nr_ho_cost_s=0.03),
            "wifi8": Policy("wifi8", "wifi8", roam_cost_s=0.0, scan_period_s=30.0),
            "single": vert, "single+detect": replace(vert, block_detect=True),
            "shared": shared, "shared+detect": replace(shared, block_detect=True),
            "shared+relay": replace(shared, relay=True),
            "shared+relay+ble": replace(shared, relay=True, ble=True),
            "single+relay+ble": replace(vert, relay=True, ble=True)}


def moving_scene(seed: int, duration: float):
    """Every arm under moving blockers; the two-path arm also under the default model as a paired reference."""
    rows = []
    for cond, (lam, noise) in CONDITIONS.items():
        w = build_world(seed, duration_s=duration, survey_s=60, noise_2g_extra_db=noise)
        for name, p in moving_arms().items():
            r = simulate(w, replace(p, name=name), collision_lambda=lam, interference=True, blockage_model="moving")
            rows.append({k: v for k, v in r.items() if k not in DROP} | {"arm": name, "condition": cond})
        for name in ("shared", "nr_30ms"):
            r = simulate(w, replace(moving_arms()[name], name=name), collision_lambda=lam, interference=True)
            rows.append({k: v for k, v in r.items() if k not in DROP} | {"arm": name + "@independent", "condition": cond})
    return rows


def moving_eval(rows):
    sc = sorted({r["scene"] for r in rows})
    v = lambda c, a, s: next(r["control_miss"] for r in rows if r["condition"] == c and r["arm"] == a and r["scene"] == s)

    def pooled(c, a):
        rs = [r for r in rows if r["condition"] == c and r["arm"] == a]
        return 1e6 * sum(r["control_misses"] for r in rs) / max(sum(r["control_sent"] for r in rs), 1)

    arms = list(dict.fromkeys(r["arm"] for r in rows))
    m = {c: {a: pooled(c, a) for a in arms} for c in CONDITIONS}
    n = lambda c, a, b: int(sum(v(c, a, s) <= v(c, b, s) for s in sc))
    res = {"M1": n("main", "shared", "nr_30ms"), "M2": n("main", "shared", "wifi8"),
           "M3": n("stress", "shared+relay", "nr_30ms"), "M4": n("stress", "shared+relay+ble", "nr_30ms"),
           "M5": n("stress", "single+relay+ble", "nr_30ms")}
    red = {c: 1 - m[c]["single+detect"] / m[c]["single"] if m[c]["single"] > 0 else 0.0 for c in CONDITIONS}
    return {"scenes": sc, "misses_per_million": m, **{k: {"scenes": x, "pass": bool(x >= 5)} for k, x in res.items()},
            "M6": {"pooled_reduction": red, "pass": bool(red["main"] >= 0.10 and red["stress"] >= 0.10)}}


def foresight_scene(seed: int, duration: float):
    """Moving blockers. Each warning source drives the same between-packets response."""
    from ..outage_risk import run_foresight as rf
    _, vert = _vert()
    shared = replace(vert, dual_radio=True, radio2="shared")
    model = rf.frozen_model()
    rows = []
    for cond, (lam, noise) in CONDITIONS.items():
        w = build_world(seed, duration_s=duration, survey_s=60, noise_2g_extra_db=noise)
        fs = rf.foresights_for(w, duration, model, sources=("truth", "geometric_site_known", "learned"))
        arms = [("nr_30ms", Policy("nr_30ms", "nr", nr_ho_cost_s=0.03), None)]
        for base_name, base in (("single", vert), ("shared", shared)):
            arms.append((base_name, base, None))
            act = replace(base, foresee=True, foresee_max_m=rf.MAX_MOVE_M)
            arms += [(f"{base_name}+foresee[truth]", act, fs["truth"]), (f"{base_name}+foresee[learned]", act, fs["learned"])]
        arms.append(("shared+foresee[geometric]", replace(shared, foresee=True, foresee_max_m=rf.MAX_MOVE_M), fs["geometric_site_known"]))
        for name, p, f in arms:
            r = simulate(w, replace(p, name=name), collision_lambda=lam, interference=True, blockage_model="moving", foresight=f)
            rows.append({k: v for k, v in r.items() if k not in DROP} | {"arm": name, "condition": cond})
    return rows


def foresight_eval(rows):
    sc = sorted({r["scene"] for r in rows})
    v = lambda c, a, s: next(r["control_miss"] for r in rows if r["condition"] == c and r["arm"] == a and r["scene"] == s)

    def pooled(c, a):
        rs = [r for r in rows if r["condition"] == c and r["arm"] == a]
        return 1e6 * sum(r["control_misses"] for r in rs) / max(sum(r["control_sent"] for r in rs), 1)

    arms = list(dict.fromkeys(r["arm"] for r in rows))
    m = {c: {a: pooled(c, a) for a in arms} for c in CONDITIONS}
    L, T, G = "shared+foresee[learned]", "shared+foresee[truth]", "shared+foresee[geometric]"
    wins = int(sum(v("stress", L, s) < v("stress", "shared", s) for s in sc))
    ratio = lambda c, a, b: m[c][a] / m[c][b] if m[c][b] > 0 else float("inf")
    return {"scenes": sc, "misses_per_million": m,
            "F1": {"scenes": wins, "ratio": ratio("stress", L, "shared"),
                   "pass": bool(wins >= 4 and ratio("stress", L, "shared") <= 0.90)},
            "F2": {"ratio": ratio("stress", T, "shared"), "pass": bool(ratio("stress", T, "shared") <= 0.70)},
            "F3": {"pass": bool(m["main"][L] <= m["main"]["shared"] + 5.0)},
            "F4": {"pass": bool(m["stress"][L] <= m["stress"][G])},
            "F5": {"ratio": {c: ratio(c, "single+foresee[learned]", "single") for c in CONDITIONS},
                   "pass": bool(all(m[c]["single+foresee[learned]"] <= m[c]["single"] for c in CONDITIONS))}}


def shared_sensing_scene(seed: int, duration: float):
    """Moving blockers, two AP paths: warnings from a robot's own sensing against fleet-shared sensing."""
    from ..outage_risk import run_foresight as rf
    _, vert = _vert()
    shared = replace(vert, dual_radio=True, radio2="shared")
    act = replace(shared, foresee=True, foresee_max_m=rf.MAX_MOVE_M)
    own_model, fleet_model = rf.frozen_model(), rf.frozen_fleet_model()
    rows = []
    for cond, (lam, noise) in CONDITIONS.items():
        w = build_world(seed, duration_s=duration, survey_s=60, noise_2g_extra_db=noise)
        fs = rf.foresights_for(w, duration, own_model, sources=("truth", "learned"))
        arms = [("nr_30ms", Policy("nr_30ms", "nr", nr_ho_cost_s=0.03), None), ("shared", shared, None),
                ("shared+foresee[own]", act, fs["learned"]), ("shared+foresee[fleet]", act, rf.fleet_foresight(w, duration, fleet_model)),
                ("shared+foresee[truth]", act, fs["truth"])]
        for name, p, f in arms:
            r = simulate(w, replace(p, name=name), collision_lambda=lam, interference=True, blockage_model="moving", foresight=f)
            rows.append({k: v for k, v in r.items() if k not in DROP} | {"arm": name, "condition": cond})
    return rows


def shared_sensing_eval(rows):
    sc = sorted({r["scene"] for r in rows})
    v = lambda c, a, s: next(r["control_miss"] for r in rows if r["condition"] == c and r["arm"] == a and r["scene"] == s)

    def pooled(c, a):
        rs = [r for r in rows if r["condition"] == c and r["arm"] == a]
        return 1e6 * sum(r["control_misses"] for r in rs) / max(sum(r["control_sent"] for r in rs), 1)

    arms = list(dict.fromkeys(r["arm"] for r in rows))
    m = {c: {a: pooled(c, a) for a in arms} for c in CONDITIONS}
    F, O = "shared+foresee[fleet]", "shared+foresee[own]"
    wins = int(sum(v("stress", F, s) <= v("stress", O, s) for s in sc))
    ratio = lambda c, a, b: m[c][a] / m[c][b] if m[c][b] > 0 else float("inf")
    return {"scenes": sc, "misses_per_million": m,
            "S1": {"scenes": wins, "ratio": ratio("stress", F, O), "pass": bool(wins >= 4 and ratio("stress", F, O) <= 1.0)},
            "S2": {"ratio": ratio("stress", F, "shared"), "pass": bool(ratio("stress", F, "shared") <= 0.75)},
            "S3": {"pass": bool(m["main"][F] <= m["main"]["shared"] + 5.0)}}


def transfer_scene(seed: int, duration: float):
    """A new site (APs at 10 m), moving blockers, two AP paths: which warning source copes?"""
    from ..outage_risk import run_foresight as rf, run_transfer as rt
    _, vert = _vert()
    shared = replace(vert, dual_radio=True, radio2="shared")
    act = replace(shared, foresee=True, foresee_max_m=rf.MAX_MOVE_M)
    rows = []
    for cond, (lam, noise) in CONDITIONS.items():
        w = build_world(seed, duration_s=duration, survey_s=60, noise_2g_extra_db=noise)
        arms = [("nr_30ms", Policy("nr_30ms", "nr", nr_ho_cost_s=0.03), None), ("shared", shared, None)]
        arms += [(f"shared+foresee[{k}]", act, f) for k, f in rt.warnings_for(w, duration).items()]
        for name, p, f in arms:
            r = simulate(w, replace(p, name=name), collision_lambda=lam, interference=True, blockage_model="moving",
                         foresight=f, reach_frac=rt.NEW_SITE_FRAC)
            rows.append({k: v for k, v in r.items() if k not in DROP} | {"arm": name, "condition": cond})
    return rows


def transfer_eval(rows):
    sc = sorted({r["scene"] for r in rows})
    v = lambda c, a, s: next(r["control_miss"] for r in rows if r["condition"] == c and r["arm"] == a and r["scene"] == s)

    def pooled(c, a):
        rs = [r for r in rows if r["condition"] == c and r["arm"] == a]
        return 1e6 * sum(r["control_misses"] for r in rs) / max(sum(r["control_sent"] for r in rs), 1)

    arms = list(dict.fromkeys(r["arm"] for r in rows))
    m = {c: {a: pooled(c, a) for a in arms} for c in CONDITIONS}
    A = lambda k: f"shared+foresee[{k}]"
    ratio = lambda a, b: m["stress"][a] / m["stress"][b] if m["stress"][b] > 0 else float("inf")
    wins = lambda a, b: int(sum(v("stress", a, s) <= v("stress", b, s) for s in sc))
    return {"scenes": sc, "misses_per_million": m,
            "T1": {"scenes": wins(A("new_radio"), A("rule_old_geometry")), "ratio": ratio(A("new_radio"), A("rule_old_geometry")),
                   "pass": bool(ratio(A("new_radio"), A("rule_old_geometry")) <= 1.0 and wins(A("new_radio"), A("rule_old_geometry")) >= 4)},
            "T2": {"ratio": ratio(A("new_radio"), "shared"), "pass": bool(ratio(A("new_radio"), "shared") <= 0.90)},
            "T3": {"ratio": ratio(A("new_radio"), A("rule_new_geometry")), "pass": bool(ratio(A("new_radio"), A("rule_new_geometry")) <= 1.10)},
            "T4": {"ratio": ratio(A("new_radio"), A("new_true")), "pass": bool(ratio(A("new_radio"), A("new_true")) <= 1.15)},
            "T5": {"pass": bool(m["main"][A("new_radio")] <= m["main"]["shared"] + 5.0)}}


RELAY_LEVELS = {"optimistic": dict(relay_delay_slots=0, relay_coop=1.0),
                "realistic": dict(relay_delay_slots=2, relay_coop=0.7),
                "pessimistic": dict(relay_delay_slots=3, relay_coop=0.5)}


def relay_scene(seed: int, duration: float):
    """Moving blockers. The relay with its optimistic assumptions removed step by step."""
    _, vert = _vert()
    shared = replace(vert, dual_radio=True, radio2="shared")
    arms = {"nr_30ms": Policy("nr_30ms", "nr", nr_ho_cost_s=0.03), "shared": shared}
    arms |= {f"shared+relay[{k}]": replace(shared, relay=True, **v) for k, v in RELAY_LEVELS.items()}
    rows = []
    for cond, (lam, noise) in CONDITIONS.items():
        w = build_world(seed, duration_s=duration, survey_s=60, noise_2g_extra_db=noise)
        for name, p in arms.items():
            r = simulate(w, replace(p, name=name), collision_lambda=lam, interference=True, blockage_model="moving")
            rows.append({k: v for k, v in r.items() if k not in DROP} | {"arm": name, "condition": cond})
    return rows


def relay_eval(rows):
    sc = sorted({r["scene"] for r in rows})
    v = lambda c, a, s: next(r["control_miss"] for r in rows if r["condition"] == c and r["arm"] == a and r["scene"] == s)

    def pooled(c, a):
        rs = [r for r in rows if r["condition"] == c and r["arm"] == a]
        return 1e6 * sum(r["control_misses"] for r in rs) / max(sum(r["control_sent"] for r in rs), 1)

    arms = list(dict.fromkeys(r["arm"] for r in rows))
    m = {c: {a: pooled(c, a) for a in arms} for c in CONDITIONS}
    n = lambda c, a, b: int(sum(v(c, a, s) <= v(c, b, s) for s in sc))
    res = {"R1": n("stress", "shared+relay[realistic]", "nr_30ms"), "R2": n("stress", "shared+relay[pessimistic]", "nr_30ms"),
           "R3": n("stress", "shared+relay[pessimistic]", "shared")}
    return {"scenes": sc, "misses_per_million": m, **{k: {"scenes": x, "pass": bool(x >= 5)} for k, x in res.items()}}


STUDIES = {"ladder": (ladder_scene, LADDER_SCENES, 120.0, ladder_eval, "headroom_ladder.json", "docs/JEPA_HEADROOM.md"),
           "fleet": (fleet_scene, FLEET_SCENES, 300.0, fleet_eval, "fleet_directions.json", "docs/DIRECTIONS_123.md"),
           "detect": (detect_scene, DETECT_SCENES, 300.0, detect_eval, "blockage_detector.json", "docs/BLOCKAGE_DETECTOR.md"),
           "moving": (moving_scene, MOVING_SCENES, 300.0, moving_eval, "moving_blockers.json", "docs/MOVING_BLOCKERS.md"),
           "foresight": (foresight_scene, FORESIGHT_SCENES, 300.0, foresight_eval, "foresight.json", "docs/FORESIGHT.md"),
           "relay": (relay_scene, RELAY_SCENES, 300.0, relay_eval, "relay_realism.json", "docs/RELAY_REALISM.md"),
           "transfer": (transfer_scene, TRANSFER_SCENES, 300.0, transfer_eval, "site_transfer.json", "docs/SITE_TRANSFER.md"),
           "shared-sensing": (shared_sensing_scene, SHARED_SENSING_SCENES, 300.0, shared_sensing_eval, "foresight_shared_sensing.json",
                              "docs/FORESIGHT.md")}


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
