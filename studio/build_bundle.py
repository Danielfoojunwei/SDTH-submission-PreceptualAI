"""Build studio/data/evidence.json: every canonical simulation, computed from the committed result files.

    .venv/bin/python studio/build_bundle.py

Numbers are computed from artifacts/native/*.json and the arm definitions in the campaign code.
Text (questions aside) is extracted from the protocol and results documents in docs/. Where a
number exists only in a document (Codex's studies), it is quoted with the document path. Every
entry records its source files. Simulation only: field and measured-data work is excluded.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "native"
OUT = ROOT / "studio" / "data" / "evidence.json"


# ---------------------------------------------------------------- helpers
def rel(p: Path | str) -> str:
    return str(Path(p).resolve().relative_to(ROOT)) if Path(p).is_absolute() else str(p)


def load(name: str):
    return json.loads((ART / name).read_text())


def fnum(x, nd=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), nd)


def boot_ci(values, weights=None, n=2000, seed=0):
    """95% bootstrap interval over scenes: ratio of sums when weights (sent) are given, else mean."""
    v = np.asarray(values, float)
    if len(v) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n, len(v)))
    if weights is not None:
        w = np.asarray(weights, float)
        est = v[idx].sum(1) / np.maximum(w[idx].sum(1), 1)
    else:
        est = v[idx].mean(1)
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def strip_md(s: str) -> str:
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = s.replace("**", "").replace("`", "").replace("*", "")
    return re.sub(r"\s+", " ", s).strip()


def doc_lines(doc: str) -> list[str]:
    return (ROOT / "docs" / doc).read_text(encoding="utf-8").splitlines()


def hypotheses_from(doc: str) -> dict:
    """{id: text} from '- **X1 (`main`):** text' lines, with indented sub-bullets appended."""
    out, lines = {}, doc_lines(doc)
    pat = re.compile(r"^- \*\*(?:Hypothesis )?([A-Z][0-9]+)([^*]*)\*\*:?\s*(.*)$")
    for i, line in enumerate(lines):
        m = pat.match(line)
        if not m or m.group(1) in out:
            continue
        text = (m.group(2).strip(" :") + " " + m.group(3)).strip()
        j = i + 1
        while j < len(lines) and lines[j].startswith("  ") and lines[j].strip():
            t = lines[j].strip()
            text += " " + (t[1:].strip() if t.startswith("-") else t)
            j += 1
        out[m.group(1)] = strip_md(text)
    return out


def items_from(doc: str, heading: str, marker: str | None = None, nth: int = 0, limit: int = 8) -> list[str]:
    """Top-level list items under the nth '## heading…' section (after a bold marker line if given)."""
    lines = doc_lines(doc)
    starts = [i for i, l in enumerate(lines) if l.startswith("## ") and l[3:].startswith(heading)]
    if len(starts) <= nth:
        return []
    i = starts[nth] + 1
    if marker:
        while i < len(lines) and not lines[i].startswith("## ") and marker not in lines[i]:
            i += 1
        i += 1
    items = []
    while i < len(lines) and not lines[i].startswith("## "):
        l = lines[i]
        m = re.match(r"^(?:\d+\.|-)\s+(.*)$", l)
        if m:
            items.append(strip_md(m.group(1)))
        elif items and l.startswith("   ") and not l.strip().startswith(("-", "|")) and l.strip():
            items[-1] += " " + strip_md(l)
        elif marker and items and l.strip() == "":
            pass
        i += 1
    return [x for x in items if len(x) > 30][:limit]


def policy_params(p) -> dict:
    """Fields that differ from the dataclass defaults, plus the fields that define each kind."""
    out = {"kind": p.kind}
    for f in dataclasses.fields(p):
        if f.name in ("name", "kind") or f.default is dataclasses.MISSING:
            continue
        v = getattr(p, f.name)
        always = (p.kind == "nr" and f.name == "nr_ho_cost_s") or (p.kind == "wifi8" and f.name in ("roam_cost_s", "scan_period_s"))
        if v != f.default or always:
            out[f.name] = list(v) if isinstance(v, tuple) else v
    return out


FAMILY = [("nr", "5G model"), ("wifi8", "Wi-Fi 8 model"), ("mlo", "Wi-Fi 7 baseline"), ("legacy", "Wi-Fi baseline"),
          ("ft11r", "Wi-Fi baseline")]


def family(arm: str, kind: str | None = None) -> str:
    if kind == "nr" or arm.startswith("nr"):
        return "5G model"
    if kind == "wifi8" or arm.startswith("wifi8"):
        return "Wi-Fi 8 model"
    for pre, fam in FAMILY:
        if arm.startswith(pre):
            return fam
    return "WirelessJEV"


LABELS = {
    "nr_30ms": "Private 5G, 30 ms handover", "nr_daps": "Private 5G, make-before-break", "wifi8": "Wi-Fi 8, seamless switching",
    "mlo_split": "Wi-Fi 7 fixed rule (control/bulk split)", "mlo_default": "Wi-Fi 7 default multi-link", "legacy": "Legacy roaming (4 s)",
    "legacy_1s": "Legacy roaming (1 s)", "ft11r": "802.11r fast roaming (60 ms)", "native": "Native stack (ungated)",
    "native_gated": "Native stack + trust gate", "vertical_single": "Vertical stack, one radio",
    "vertical_client": "Vertical stack + second radio", "vertical_full": "Vertical + second radio + scheduled control",
    "vertical_shared": "Vertical + second path on the same card", "vertical_onboard": "Vertical + onboard radio as second path",
    "vertical_shared-levers": "Same-card path, extra levers off", "single": "One radio", "shared": "Two AP paths (same card)",
}


def label(arm: str) -> str:
    if arm in LABELS:
        return LABELS[arm]
    s = arm.replace("_", " ")
    s = s.replace("single+", "One radio + ").replace("shared+", "Two AP paths + ").replace("+", " + ")
    return s


def summarise_rows(rows, arm_key="arm", cond_key="condition", default_cond="all") -> dict:
    """{condition: {arm: stats}} with pooled misses per million and a scene bootstrap interval."""
    out = {}
    groups = {}
    for r in rows:
        c = r.get(cond_key, default_cond) if cond_key else default_cond
        groups.setdefault(str(c), {}).setdefault(str(r[arm_key]), []).append(r)
    for c, arms in groups.items():
        out[c] = {}
        for a, rs in arms.items():
            rs = sorted(rs, key=lambda r: r["scene"])
            st = {"scenes": [r["scene"] for r in rs]}
            if all("control_misses" in r for r in rs):
                m = [r["control_misses"] for r in rs]
                s = [r["control_sent"] for r in rs]
                st["mpm"] = 1e6 * sum(m) / max(sum(s), 1)
                lo, hi = boot_ci(m, s)
                st["ci"] = [1e6 * lo, 1e6 * hi] if lo is not None else None
                st["misses"], st["sent"] = int(sum(m)), int(sum(s))
                st["per_scene"] = [fnum(1e6 * x / max(y, 1)) for x, y in zip(m, s)]
                st["pooling"] = "pooled counts"
            else:
                v = [1e6 * r["control_miss"] for r in rs]
                st["mpm"] = float(np.mean(v))
                lo, hi = boot_ci(v)
                st["ci"] = [lo, hi] if lo is not None else None
                st["per_scene"] = [fnum(x) for x in v]
                st["pooling"] = "mean of scene rates"
            for k in ("availability_1s", "safety_stop_fraction", "control_gap_p99_ms", "video_on_time", "airtime_domain_sum",
                      "plan_stop_fraction", "task_throughput"):
                vals = [r[k] for r in rs if r.get(k) is not None]
                if vals:
                    st[k] = float(np.mean(vals))
            causes = [r["miss_causes"] for r in rs if isinstance(r.get("miss_causes"), dict)]
            if causes:
                st["miss_causes"] = {k: int(sum(c.get(k, 0) for c in causes)) for k in causes[0]}
            st["roams"] = float(np.mean([r["roams"] if not isinstance(r["roams"], list) else sum(r["roams"]) for r in rs])) if "roams" in rs[0] else None
            for k in ("mpm",):
                st[k] = fnum(st[k], 2)
            if st.get("ci"):
                st["ci"] = [fnum(st["ci"][0], 2), fnum(st["ci"][1], 2)]
            out[c][a] = st
    return out


def verdicts(evaluation: dict, hyp_text: dict, order=None) -> list[dict]:
    out = []
    keys = [k for k in (order or evaluation.keys()) if k in evaluation and isinstance(evaluation[k], dict) and "pass" in evaluation[k]]
    for k in keys:
        e = evaluation[k]
        detail = {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in e.items() if kk != "pass" and not isinstance(vv, (list, dict))}
        for kk, vv in e.items():
            if isinstance(vv, dict):
                detail.update({f"{kk} {k2}": round(v2, 4) if isinstance(v2, float) else v2 for k2, v2 in vv.items() if not isinstance(v2, (list, dict))})
        out.append({"id": k, "text": hyp_text.get(k, ""), "pass": bool(e["pass"]), "detail": detail})
    return out


def scene_range(scenes) -> str:
    s = sorted({int(x) for x in scenes})
    return f"{s[0]}–{s[-1]}" if s == list(range(s[0], s[-1] + 1)) else ", ".join(map(str, s))


# ---------------------------------------------------------------- campaigns
def campaigns() -> list[dict]:
    from wirelessjev.native import run_5g, run_gate, run_learned, run_x, sim
    from wirelessjev.native_ext import blockers, run_fleet
    from wirelessjev.native_ext.sim_fleet import Policy as FPolicy
    from dataclasses import replace

    pol = {p.name: p for p in sim.policies()}
    out = []
    indep = {"model": "independent", "rate_per_robot_min": sim.BLOCKAGE_RATE_PER_MIN, "depth_db": sim.BLOCKAGE_DB,
             "mean_s": sim.BLOCKAGE_MEAN_S, "source": "src/wirelessjev/native/sim.py"}
    moving = {"model": "moving forklifts", "blockers": blockers.N_BLOCKERS, "radius_m": blockers.RADIUS_M,
              "reach_frac": round(blockers.REACH_FRAC, 3), "speed_mps": list(blockers.SPEED_MPS), "depth_db": sim.BLOCKAGE_DB,
              "source": "src/wirelessjev/native_ext/blockers.py"}
    fleet_conds = {k: {"collision_lambda": v[0], "noise_2g_extra_db": v[1], "interference": True} for k, v in run_fleet.CONDITIONS.items()}

    def add(cid, title, family_, question, file, protocol, results_doc, says, setup, arms, results, hyps, order=None, notes=None, status="fresh"):
        d = load(file)
        scenes = sorted({r["scene"] for r in d["rows"]}) if "rows" in d and d["rows"] and "scene" in d["rows"][0] else []
        setup = {"scenes": scene_range(scenes) if scenes else setup.get("scenes"), "n_scenes": len(scenes), "robots": 100,
                 "robots_source": "build_world default, src/wirelessjev/native/world.py", "duration_s": d.get("duration_s"),
                 "status": status, **setup}
        out.append({"id": cid, "title": title, "family": family_, "question": question, "status": status,
                    "sources": {"results": f"artifacts/native/{file}", "protocol": f"docs/{protocol}", "results_doc": f"docs/{results_doc}" if results_doc else None},
                    "setup": setup, "arms": arms, "results": results,
                    "hypotheses": verdicts(d.get("evaluation") or {}, hyps, order), "says": says, "notes": notes or []})

    def arm_list(defs: dict, fam_kind=True):
        return [{"id": k, "label": label(k), "family": family(k, p.kind), "params": policy_params(p)} for k, p in defs.items()]

    # 1 native stack
    d = load("campaign.json")
    arms = arm_list({k: v for k, v in pol.items()})
    arms += [{"id": "legacy_1s", "label": label("legacy_1s"), "family": "Wi-Fi baseline", "params": policy_params(replace(pol["legacy"], roam_cost_s=1.0))},
             {"id": "native_sigma8", "label": "Native, map error σ = 8 dB", "family": "WirelessJEV", "params": {**policy_params(pol["native"]), "world.sigma_map": 8.0}},
             {"id": "native_oracle", "label": "Native, perfect map", "family": "WirelessJEV", "params": {**policy_params(pol["native"]), "world.sigma_map": 0.0}},
             {"id": "native_blind", "label": "Native, no route knowledge", "family": "WirelessJEV", "params": {**policy_params(pol["native"]), "world.blind": True}}]
    for n in ("mlo_default", "mlo_split", "native"):
        arms.append({"id": f"{n}_heavy_blockage", "label": f"{label(n)}, 3× blockage (exploratory)", "family": family(n),
                     "params": {**policy_params(pol[n]), "blockage_rate_per_min": 4.5}})
    add("native_stack", "Native Wi-Fi 7 stack", "Native stack", "Does a native multi-link stack with planned roaming beat default Wi-Fi 7 multi-link?",
        "campaign.json", "NATIVE_STACK.md", "NATIVE_STACK_RESULTS.md", items_from("NATIVE_STACK_RESULTS.md", "What the ablations say"),
        {"conditions": {"all": {"collision_lambda": 0.0, "noise_2g_extra_db": 0.0, "interference": False}}, "blockage": indep,
         "map": "given map with error σ = 4 dB (variants σ = 8, perfect, blind)"},
        arms, summarise_rows(d["rows"], cond_key=None), hypotheses_from("NATIVE_STACK.md"), order=["H1", "H2", "H3", "H4", "H5"])

    # 2 trust gate
    d = load("gate_campaign.json")
    add("trust_gate", "Trust gate", "Native stack", "Does gating planned roams on the map's own uncertainty make planning safe when the map is wrong?",
        "gate_campaign.json", "NATIVE_GATE.md", "NATIVE_GATE_RESULTS.md", items_from("NATIVE_GATE_RESULTS.md", "What changed in our understanding"),
        {"conditions": {k: {"map_sigma_db": v[0], "noise_2g_extra_db": v[1], "map_error": "heterogeneous"} for k, v in run_gate.CONDITIONS.items()},
         "blockage": indep, "map": "given map, error σ = 4 or 8 dB, spatially varying"},
        arm_list(run_gate.arms()), summarise_rows(d["rows"]), hypotheses_from("NATIVE_GATE.md"))

    # 3 learned map
    d = load("learned_campaign.json")
    add("learned_map", "Learned RadioMap", "Native stack", "Can a map learned from one or five minutes of robot scans replace a given map?",
        "learned_campaign.json", "NATIVE_LEARNED_MAP.md", "NATIVE_LEARNED_MAP_RESULTS.md", items_from("NATIVE_LEARNED_MAP_RESULTS.md", "What it means"),
        {"conditions": {k: {"survey_s": v[0], "noise_2g_extra_db": v[1]} for k, v in run_learned.CONDITIONS.items()}, "blockage": indep,
         "map": "learned RadioMap from a 60 s or 300 s survey"},
        arm_list(run_learned.arms()), summarise_rows(d["rows"]), hypotheses_from("NATIVE_LEARNED_MAP.md"))

    # 4 Wi-Fi vs 5G
    d = load("wifi_vs_5g.json")
    a5 = run_5g.arms("main")
    add("wifi_vs_5g", "Wi-Fi vs private 5G", "Against 5G and Wi-Fi 8", "Does the vertical Wi-Fi stack with a second radio beat a generous private 5G model?",
        "wifi_vs_5g.json", "WIFI_VS_5G.md", "WIFI_VS_5G_RESULTS.md", items_from("WIFI_VS_5G_RESULTS.md", "What this means"),
        {"conditions": {k: {"collision_lambda": v["collision_lambda"], "noise_2g_extra_db": v["noise_2g"], "interference": v["interference"]} for k, v in run_5g.CONDITIONS.items()},
         "blockage": indep, "map": "learned RadioMap, 60 s survey"},
        arm_list(a5), summarise_rows(d["rows"]), hypotheses_from("WIFI_VS_5G.md"))

    # 5 cross-AP, no new hardware
    d = load("cross_ap.json")
    add("cross_ap", "Second path without new hardware", "Against 5G and Wi-Fi 8",
        "Does a second AP path on the existing card or onboard radio beat the 5G and Wi-Fi 8 models?",
        "cross_ap.json", "CROSS_AP.md", "CROSS_AP.md", items_from("CROSS_AP.md", "Results", marker="What it shows"),
        {"conditions": {k: {"collision_lambda": v["lam"], "noise_2g_extra_db": v["noise"], "interference": True,
                            "load": "15% of robots stream 6 Mb/s video; bulk doubled" if v["heavy"] else "standard"} for k, v in run_x.CONDITIONS.items()},
         "blockage": indep, "map": "learned RadioMap, 60 s survey"},
        arm_list(run_x.arms()), summarise_rows(d["rows"]), hypotheses_from("CROSS_AP.md"))

    # 6 headroom ladder
    d = load("headroom_ladder.json")
    _, vert = run_fleet._vert()
    rungs = {1: ("learned map", vert), 2: ("perfect map", replace(vert, gate=False, online_map=False)),
             3: ("+ blockage now (oracle)", replace(vert, gate=False, online_map=False, oracle=2)),
             4: ("+ next 1 s (oracle)", replace(vert, gate=False, online_map=False, oracle=3))}
    arms, rows = [], []
    for radio, extra in (("single", {}), ("two_path", {"dual_radio": True})):
        for r, (lab, p) in rungs.items():
            arms.append({"id": f"{radio}_r{r}", "label": f"{'One radio' if radio == 'single' else 'Two AP paths'}, rung {r}: {lab}",
                         "family": "WirelessJEV", "params": {**policy_params(replace(p, **extra)), "world": "learned map, 60 s survey" if r == 1 else "perfect map (σ = 0)"}})
    for row in d["rows"]:
        rows.append({**row, "arm": f"{row['radio']}_r{row['rung']}"})
    ev = d["evaluation"]
    kill = [{"id": f"K-{k.split('_')[0]}", "text": f"Kill criterion ({k.split('_')[0].replace('two', 'two AP paths').replace('single', 'one radio')}, realistic): learned-control work continues only if perfect knowledge of the next second (rung 4) cuts misses by at least 25% against the learned map (rung 1). Pass = continue.",
             "pass": bool(v), "detail": {"rung4_reduction_main": round(ev[f"{k.split('_c')[0]}|main"]["rung4_reduction"], 3)}}
            for k, v in ev.items() if k.endswith("continue_control_learning")]
    add("headroom_ladder", "Headroom ladder", "Learned models", "How much could any learned model gain, by kind of information: a perfect map, current blockage, or the next second?",
        "headroom_ladder.json", "JEPA_HEADROOM.md", "JEPA_HEADROOM.md", items_from("JEPA_HEADROOM.md", "Results", marker="What the rungs say"),
        {"conditions": fleet_conds, "blockage": indep, "map": "rung 1 learned (60 s survey); rungs 2–4 perfect"},
        arms, summarise_rows(rows), {}, status="development")
    out[-1]["hypotheses"] = kill

    # 7 fleet-only capabilities
    d = load("fleet_directions.json")
    add("fleet_capabilities", "Fleet-only capabilities", "Fleet capabilities", "Do a safe plan, robot relay and a Bluetooth control path close the gaps a network alone cannot?",
        "fleet_directions.json", "DIRECTIONS_123.md", "DIRECTIONS_123.md", items_from("DIRECTIONS_123.md", "What the results say"),
        {"conditions": fleet_conds, "blockage": indep, "map": "learned RadioMap, 60 s survey"},
        arm_list(run_fleet.fleet_arms()), summarise_rows(d["rows"]), hypotheses_from("DIRECTIONS_123.md"))

    # 8 blockage rule
    d = load("blockage_detector.json")
    add("blockage_rule", "Live blockage rule", "Fleet capabilities", "Does a hand-written rule that spots a blocked path from the robot's own readings cut misses?",
        "blockage_detector.json", "BLOCKAGE_DETECTOR.md", "BLOCKAGE_DETECTOR.md", items_from("BLOCKAGE_DETECTOR.md", "What the results say"),
        {"conditions": fleet_conds, "blockage": indep, "map": "learned RadioMap, 60 s survey"},
        arm_list(run_fleet.detect_arms()), summarise_rows(d["rows"]), hypotheses_from("BLOCKAGE_DETECTOR.md"))

    # 9 moving blockers
    d = load("moving_blockers.json")
    marms = arm_list(run_fleet.moving_arms())
    for k in ("shared", "nr_30ms"):
        base = next(a for a in marms if a["id"] == k)
        marms.append({**base, "id": f"{k}@independent", "label": base["label"] + " (independent blockage, paired reference)",
                      "params": {**base["params"], "blockage_model": "independent"}})
    add("moving_blockers", "Moving blockers", "Against 5G and Wi-Fi 8", "Do the headline claims survive forklifts that block several robots and both AP paths at once?",
        "moving_blockers.json", "MOVING_BLOCKERS.md", "MOVING_BLOCKERS.md", items_from("MOVING_BLOCKERS.md", "What the results say"),
        {"conditions": fleet_conds, "blockage": moving, "map": "learned RadioMap, 60 s survey"},
        marms, summarise_rows(d["rows"]), hypotheses_from("MOVING_BLOCKERS.md"), order=["M1", "M2", "M3", "M4", "M5", "M6"])

    # 10 relay realism
    d = load("relay_realism.json")
    shared = replace(vert, dual_radio=True, radio2="shared")
    rdefs = {"nr_30ms": FPolicy("nr_30ms", "nr", nr_ho_cost_s=0.03), "shared": shared}
    rdefs |= {f"shared+relay[{k}]": replace(shared, relay=True, **v) for k, v in run_fleet.RELAY_LEVELS.items()}
    add("relay_realism", "Relay realism", "Fleet capabilities", "Does the hardest-condition win survive a slower, less reliable relay?",
        "relay_realism.json", "RELAY_REALISM.md", "RELAY_REALISM.md", items_from("RELAY_REALISM.md", "What the results say"),
        {"conditions": fleet_conds, "blockage": moving, "map": "learned RadioMap, 60 s survey"},
        arm_list(rdefs), summarise_rows(d["rows"]), hypotheses_from("RELAY_REALISM.md"))

    # 11 foresight and 12 shared sensing
    from wirelessjev.outage_risk import run_foresight as rf
    act = replace(shared, foresee=True, foresee_max_m=rf.MAX_MOVE_M)
    single_act = replace(vert, foresee=True, foresee_max_m=rf.MAX_MOVE_M)
    src_note = {"truth": "perfect warning (ceiling)", "learned": f"learned predictor, own sensing, threshold {rf.STRICT_THRESHOLD}",
                "geometric": "hand-written geometric rule, own sensing", "own": f"learned predictor, own sensing, threshold {rf.STRICT_THRESHOLD}",
                "fleet": f"learned predictor, fleet-shared sensing (200 ms late), threshold {rf.STRICT_THRESHOLD}"}

    def fs_arms(names):
        out_ = [{"id": "nr_30ms", "label": label("nr_30ms"), "family": "5G model", "params": policy_params(FPolicy("nr_30ms", "nr", nr_ho_cost_s=0.03))}]
        for n in names:
            if "[" in n:
                base, src = n[:-1].split("[")
                p = single_act if base.startswith("single") else act
                out_.append({"id": n, "label": f"{'One radio' if base.startswith('single') else 'Two AP paths'} + warning: {src_note[src]}",
                             "family": "WirelessJEV", "params": {**policy_params(p), "warning_source": src_note[src]}})
            else:
                out_.append({"id": n, "label": label(n), "family": "WirelessJEV", "params": policy_params(vert if n == "single" else shared)})
        return out_

    d = load("foresight.json")
    add("foresight", "Foresight from obstacle sensing", "Learned models", "Does a one-second blockage warning from the robot's obstacle sensing cut misses?",
        "foresight.json", "FORESIGHT.md", "FORESIGHT.md", items_from("FORESIGHT.md", "What the results say"),
        {"conditions": fleet_conds, "blockage": moving, "map": "learned RadioMap, 60 s survey",
         "sensing": "10 m range, hidden by racks, 10% missed detections, 0.2 m noise (src/wirelessjev/outage_risk/foresight.py)"},
        fs_arms(["single", "single+foresee[truth]", "single+foresee[learned]", "shared", "shared+foresee[truth]", "shared+foresee[learned]", "shared+foresee[geometric]"]),
        summarise_rows(d["rows"]), hypotheses_from("FORESIGHT.md"), order=["F1", "F2", "F3", "F4", "F5"])
    d = load("foresight_shared_sensing.json")
    add("shared_sensing", "Fleet-shared sensing", "Learned models", "Does telling every robot about forklifts any robot sees improve the warning?",
        "foresight_shared_sensing.json", "FORESIGHT.md", "FORESIGHT.md", items_from("FORESIGHT.md", "What these results say"),
        {"conditions": fleet_conds, "blockage": moving, "map": "learned RadioMap, 60 s survey"},
        fs_arms(["shared", "shared+foresee[own]", "shared+foresee[fleet]", "shared+foresee[truth]"]),
        summarise_rows(d["rows"]), hypotheses_from("FORESIGHT.md"), order=["S1", "S2", "S3"])

    # 13 coverage report (map accuracy, not control)
    d = load("coverage_report.json")
    cov_rows = []
    for r in d["rows"]:
        a, rep = r["accuracy"], r["report"]
        cov_rows.append({"scene": r["scene"], "level_rmse_db": fnum(a["best_level_rmse_db"], 2), "level_bias_db": fnum(a.get("best_level_bias_db"), 2),
                         "no_backup_precision": fnum(a["no_backup_control"]["precision"], 3), "no_backup_recall": fnum(a["no_backup_control"]["recall"], 3),
                         "weak_precision": fnum(a["weak_control"]["precision"], 3), "weak_recall": fnum(a["weak_control"]["recall"], 3),
                         "clearly_weak_found": fnum(a["clearly_weak_found"]["share"], 3), "clearly_fine_flagged": fnum(a["clearly_fine_flagged"]["share"], 3),
                         "worst_ap_rank": a["worst_ap_failure"]["report_rank_of_it"],
                         "covered_control": fnum(rep["services"]["control"]["covered_share"], 3), "with_backup_control": fnum(rep["services"]["control"]["with_backup_share"], 3)})
    out.append({"id": "coverage_report", "title": "Site coverage report", "family": "Learned models", "status": "fresh",
                "question": "Is a coverage report built from one minute of robot driving accurate enough to act on?",
                "sources": {"results": "artifacts/native/coverage_report.json", "protocol": "docs/COVERAGE_REPORT.md", "results_doc": "docs/COVERAGE_REPORT.md"},
                "setup": {"scenes": scene_range([r["scene"] for r in d["rows"]]), "n_scenes": len(d["rows"]), "robots": 100, "duration_s": None,
                          "survey_s": d["survey_s"], "status": "fresh", "conditions": {"map scoring": {"grid_m": 2.0, "thresholds_dbm": "-75 basic, -67 control"}},
                          "blockage": indep, "map": f"learned RadioMap, {d['survey_s']:.0f} s survey; scored against simulator truth",
                          "robots_source": "build_world default, src/wirelessjev/native/world.py"},
                "arms": [], "results": {}, "table": cov_rows,
                "hypotheses": verdicts(d["evaluation"], hypotheses_from("COVERAGE_REPORT.md"), ["K1", "K2", "K3", "K4", "K5"]),
                "says": items_from("COVERAGE_REPORT.md", "What the results say"), "notes": []})

    # 14 cross-site prior (foundation)
    d = load("foundation.json")
    frows = {}
    for r in d["rows"]:
        frows.setdefault((r["survey_s"], r["prior"]), []).append(r)
    ftable = [{"survey_s": s, "prior": p, "rmse_db": fnum(np.mean([r["rmse_db"] for r in rs]), 2),
               "coverage_90": fnum(np.mean([r["coverage_90"] for r in rs]), 3), "sites": len(rs)} for (s, p), rs in sorted(frows.items())]
    out.append({"id": "cross_site_prior", "title": "Cross-site learned prior", "family": "Learned models", "status": "fresh",
                "question": "Does a radio prior learned on 40 other sites make the map better on a new site?",
                "sources": {"results": "artifacts/native/foundation.json", "protocol": "docs/RADIOJEV_FOUNDATION.md", "results_doc": "docs/RADIOJEV_FOUNDATION.md"},
                "setup": {"scenes": scene_range({r["seed"] for r in d["rows"]}), "n_scenes": len({r["seed"] for r in d["rows"]}), "robots": 100, "duration_s": None,
                          "status": "fresh", "conditions": {f"survey {s} s": {"survey_s": s} for s in sorted({r['survey_s'] for r in d['rows']})},
                          "blockage": indep, "map": f"learned RadioMap with a generic prior vs a prior learned on {d['train_sites']} training sites",
                          "robots_source": "build_world default, src/wirelessjev/native/world.py"},
                "arms": [], "results": {}, "table": ftable,
                "hypotheses": [{"id": "F1", "text": hypotheses_from("RADIOJEV_FOUNDATION.md").get("F1", ""), "pass": False, "detail": {"source": "docs/RADIOJEV_FOUNDATION.md"}},
                               {"id": "F2", "text": hypotheses_from("RADIOJEV_FOUNDATION.md").get("F2", ""), "pass": False, "detail": {"source": "docs/RADIOJEV_FOUNDATION.md"}}],
                "says": items_from("RADIOJEV_FOUNDATION.md", "Results", marker="Implications"), "notes": []})
    return out


# ---------------------------------------------------------------- learned-model line
def learned_line() -> dict:
    ours = []
    d = load("outage_risk_dev.json")
    for cond in ("main", "stress"):
        h = d[cond]["horizons"]["0.0"]
        ours.append({"id": f"detector_indep_{cond}", "title": f"Blockage detector, independent blockage ({cond})", "owner": "ours",
                     "model": "logistic regression and a small network (2×32) on 0.5 s of the robot's own telemetry",
                     "trained_on": "simulator telemetry, scenes 400–401; tested 402–403", "baseline": "hand-written rule (10 dB drop, 2 readings)",
                     "metric": "blocked moments caught at the rule's false-alarm rate",
                     "values": {"rule": fnum(100 * h["rule"]["caught"]), "logistic": fnum(100 * h["logistic_window"]["at_rule_false_alarm_rate"]["caught"]),
                                "small net (median of 3 seeds)": fnum(100 * float(np.median(h["small_net"]["caught_at_rule_false_alarm_rate"])))},
                     "unit": "% caught", "verdict": "rule not beaten", "outcome": "neutral", "status": "development",
                     "source": "artifacts/native/outage_risk_dev.json"})
    for cond in ("main", "stress"):
        d = load(f"outage_risk_dev_moving_{cond}.json")[cond]["horizons"]["0.0"]
        best = float(np.median(d["small_net"]["caught_at_rule_false_alarm_rate"]))
        ours.append({"id": f"detector_moving_{cond}", "title": f"Blockage detector, moving blockers ({cond})", "owner": "ours",
                     "model": "small network on own telemetry; also with neighbours' alarms", "trained_on": "simulator telemetry, scenes 400–401; tested 402–403",
                     "baseline": "hand-written rule", "metric": "blocked moments caught at the rule's false-alarm rate",
                     "values": {"rule": fnum(100 * d["rule"]["caught"]), "small net (median)": fnum(100 * best),
                                "with neighbours (median)": fnum(100 * float(np.median(d["small_net_fleet"]["caught_at_rule_false_alarm_rate"])))},
                     "unit": "% caught", "verdict": "learned beats rule narrowly" if best > d["rule"]["caught"] else "rule not beaten",
                     "outcome": "win" if best > d["rule"]["caught"] else "neutral", "status": "development",
                     "source": f"artifacts/native/outage_risk_dev_moving_{cond}.json"})
    s = load("outage_risk_dev_moving_sensing.json")
    ours.append({"id": "sensing_headroom", "title": "Seeing blockage coming: radio vs obstacle sensing", "owner": "ours",
                 "model": "geometric predictor given forklift positions within 10 m", "trained_on": "none (perfect sensing headroom)",
                 "baseline": "radio rule", "metric": "blockages flagged 1 s ahead",
                 "values": {"radio rule": fnum(100 * s["main"]["1.0"]["radio_rule"]["caught"]), "sensed blockers": fnum(100 * s["main"]["1.0"]["sensed_blockers_10m"]["caught"])},
                 "unit": "% flagged (realistic)", "verdict": "sensing carries the information", "outcome": "finding", "status": "development",
                 "source": "artifacts/native/outage_risk_dev_moving_sensing.json"})
    f = load("foresight.json")["evaluation"]
    m = f["misses_per_million"]["stress"]
    ours.append({"id": "foresight_predictor", "title": "Foresight predictor in the loop", "owner": "ours",
                 "model": "small network (2×64) from sensed forklifts to a 1 s warning", "trained_on": "simulated sensing, scenes 400–401",
                 "baseline": "two AP paths with no warning; hand-written geometric rule", "metric": "missed per million, two AP paths, stress (scenes 580–585)",
                 "values": {"no warning": fnum(m["shared"]), "learned": fnum(m["shared+foresee[learned]"]), "hand-written rule": fnum(m["shared+foresee[geometric]"]),
                            "perfect warning": fnum(m["shared+foresee[truth]"])},
                 "unit": "missed per million", "verdict": "passes F1; ties the hand-written rule", "outcome": "win", "status": "fresh",
                 "source": "artifacts/native/foresight.json"})
    rows = [json.loads(l) for l in (ART / "transfer_dev.jsonl").read_text().splitlines() if l.strip()]
    def tpool(src):
        rs = [r for r in rows if r["condition"] == "stress" and r["source"] == src]
        return fnum(1e6 * sum(r["misses"] for r in rs) / sum(r["sent"] for r in rs)) if rs else None
    ours.append({"id": "site_transfer", "title": "Site transfer (APs at 10 m)", "owner": "ours",
                 "model": "predictor retrained on the new site's own radio rule (no ground truth)", "trained_on": "5 min of new-site telemetry, scenes 400–401",
                 "baseline": "hand-written rule with the old site's geometry", "metric": "missed per million, two AP paths, stress (tuning scenes 402–403)",
                 "values": {"no warning": tpool("none"), "rule, old geometry": tpool("rule_old_geometry"), "rule, new geometry": tpool("rule_new_geometry"),
                            "old learned, not retrained": tpool("old_learned"), "learned from radio labels": tpool("new_radio_0.5")},
                 "unit": "missed per million", "verdict": "premise failed: AP height barely matters", "outcome": "loss", "status": "development",
                 "source": "artifacts/native/transfer_dev.jsonl"})
    hl = load("headroom_ladder.json")["evaluation"]
    ours.append({"id": "headroom", "title": "Headroom for any learned model", "owner": "ours", "model": "oracle rungs (no model)",
                 "trained_on": "n/a", "baseline": "RadioMap (rung 1)", "metric": "missed per million, one radio, realistic",
                 "values": {f"rung {k}": fnum(v) for k, v in hl["single|main"]["misses_per_million"].items()},
                 "unit": "missed per million", "verdict": "a perfect map is worth 9%; current blockage a further 63%", "outcome": "finding", "status": "development",
                 "source": "artifacts/native/headroom_ladder.json"})

    def codex(id_, title, doc, model, baseline, metric, values, unit, verdict, outcome, quote):
        return {"id": id_, "title": title, "owner": "Codex", "model": model, "trained_on": "simulation (see document)", "baseline": baseline,
                "metric": metric, "values": values, "unit": unit, "verdict": verdict, "outcome": outcome, "status": "fresh", "source": doc, "quote": quote}

    cx = [
        codex("spectral_scene", "Physics-structured RadioJEPA: static field", "docs/spectral_scene/RESULTS.md", "physics-structured RadioJEPA scene model",
              "RadioMap-operator control", "held-out-layout RF RMSE (24 simulated geometries)", {"RadioMap operator": 4.843, "RadioJEPA scene": 3.622}, "dB",
              "lower RF error (25.2%)", "rf-win", "lowers spatial held-out-layout RF RMSE from 4.843 to 3.622 dB"),
        codex("nested_radio", "Optional AP-local radio field", "docs/nested_radio/RESULTS.md", "nested joint (JEPA) model", "hierarchy-fixed numerical map",
              "sparse heard-RSSI RMSE", {"numerical": 2.710522, "nested joint": 2.637330}, "dB", "2.7% lower RF error; JEPA objective adds 0.002 dB",
              "rf-win", "The joint-view addition contributes only 0.002226 dB over nested regression"),
        codex("hierarchical_radio", "Hierarchical radio posterior", "docs/hierarchical_radio/RESULTS.md", "hierarchical model", "selected numerical map",
              "sparse heard-RSSI RMSE", {"numerical map": 2.951064, "global model": 2.995352, "hierarchical": 3.088402}, "dB", "higher RF error", "loss",
              "The selected hierarchical model does not improve the primary RF endpoint"),
        codex("adaptive", "Adaptive RF-condition model", "docs/research_status/ADAPTIVE_RESULTS.md", "regression branch (JEPA branch slightly worse)",
              "RadioMap operator", "condition-delta RMSE (24 TEST layouts)", {"RadioMap": 1.654205, "regression": 1.520106, "Bayesian control": 1.524141}, "dB",
              "8.1% lower; JEPA not better than regression", "rf-win", "The matched joint/JEPA branch is slightly worse than regression"),
        codex("native_diag", "Learned field in the native simulator", "docs/research_status/NATIVE_RESULTS.md", "regression-seed0 learned field", "RadioMap",
              "missed commands, 4 native episodes", {}, "", "more misses at 30 ms handover in all 4 episodes", "loss",
              "The learned field increased missed commands in all four 0.03 s handover episodes"),
        codex("fleet_world_model", "Fleet world model", "docs/FLEET_WORLD_MODEL_RESULTS.md", "Fleet JEPA (28,346 parameters per member)", "learned-prior RadioMap",
              "missed per million, 8 fresh scenes", {"RadioMap": 115.10, "Fleet JEPA": 100.00}, "missed per million", "interval includes zero", "neutral",
              "The interval does not establish a primary-condition reduction"),
        codex("fleet_confirmation", "Independent warehouse confirmation", "docs/FLEET_CONFIRMATION_RESULTS.md", "fixed supervised hybrid (not JEPA)",
              "learned-prior RadioMap", "missed per million, 8 new warehouses", {"RadioMap": 167.19, "supervised hybrid": 133.33, "hybrid JEPA": 128.12},
              "missed per million", "supervised hybrid passes; JEPA objective not the reason", "win", "The preregistered primary reduction criterion passes"),
        codex("world_frontier", "World-model optimisation frontier", "docs/WORLD_FRONTIER_RESULTS.md", "temporal + innovation model", "optimised RadioMap",
              "control misses per million, 8 fresh warehouses", {"RadioMap": 364.58, "temporal + innovation": 340.10}, "missed per million",
              "superiority claim fails", "neutral", "the superiority claim fails"),
        codex("frontier_confirmation", "Compact-model confirmation", "docs/FRONTIER_CONFIRMATION_RESULTS.md", "distilled model", "optimised RadioMap",
              "missed per million, 8 new sites", {"RadioMap": 251.04, "distilled": 228.12}, "missed per million", "primary criterion fails", "neutral",
              "The primary superiority criterion fails"),
        codex("contracted", "Contracted world model", "docs/CONTRACTED_RESULTS.md", "JEPA with uncertainty contract", "fixed rule; Bayesian (RadioMap-style) gate",
              "missed per million", {"fixed split": 622.92, "Bayesian gated": 92.36, "JEPA ungated": 114.58, "JEPA gated": 151.39}, "missed per million",
              "beats the fixed rule; not RadioMap gated", "neutral", "regret training did not establish a robust control advantage. RadioMap remains the default"),
        codex("gate_repair", "JEPA gate repair", "docs/JEPA_GATE_REPAIR_RESULTS.md", "five JEPA models with a repaired gate", "original JEPA gate",
              "missed per million, 6 fresh scenes", {"original gate": 329.17, "0.25× penalty gate": 111.11}, "missed per million", "66.2% fewer than the old gate",
              "win", "The five JEPA models and their forecasts are unchanged"),
        codex("pilot_fastloop", "Matched forecast and controller packages", "docs/PILOT_FASTLOOP_RESULTS.md", "frozen FleetJEPA + combined package",
              "RadioMap + the same package", "command loss, 8 sites, 4 s handover", {"RadioMap": 4.801354, "FleetJEPA": 4.594583}, "% loss",
              "4.31% relative reduction; neither package beats its own base", "win", "Neither combined package qualifies as an improvement over its own base"),
        codex("freshness_stack", "Freshness, radio resources and sensing", "docs/FRESHNESS_STACK_RESULTS.md", "fixed FleetJEPA with the final stack",
              "RadioMap with the same stack", "command loss", {"RadioMap": 3.6980, "FleetJEPA": 3.8100}, "% loss", "JEPA did not beat RadioMap", "loss",
              "JEPA did not beat RadioMap with the same final stack"),
        codex("jepa_repair", "Fixed-weight JEPA controller repair", "docs/JEPAREPAIR_RESULTS.md", "FleetJEPA ensemble + repaired controller", "repaired RadioMap",
              "fast-handover command misses", {}, "", "2.87% better than its own legacy; not better than RadioMap", "neutral",
              "It did not establish superiority over repaired RadioMap"),
        codex("costaware", "Cost-aware recovery", "docs/COSTAWARE_RECOVERY_RESULTS.md", "repaired JEPA + sustained recovery", "repaired RadioMap",
              "longest silences; command loss", {}, "", "no registered comparison qualified", "neutral",
              "none of the three registered comparisons qualified as experimental benefit"),
        codex("continual_packet", "Costed continual RF learning", "docs/CONTINUAL_PACKET_RESULTS.md", "response-memory and Bayesian RF adapters",
              "canonical RadioMap", "command loss, 30 ms / 4 s handover", {"RadioMap 30 ms": 1.0289, "RadioMap 4 s": 13.6070}, "% loss", "no candidate promoted",
              "loss", "Neither the response-memory nor Bayesian RF adapter passes the registered packet-service gate"),
        codex("continual_radio", "Continual radio response (Sionna RT)", "docs/CONTINUAL_RADIO_RESULTS.md", "JEPA vs matched supervised vs Bayesian identification",
              "explicit Bayesian system identification", "response identification error", {}, "", "Bayesian identification better than both learned models",
              "loss", "The six-run experiment does not establish a JEPA advantage"),
        codex("semantic_jepa", "Semantic JEPA", "docs/SEMANTIC_JEPA_RESULTS.md", "semantic JEPA", "matched supervised model; RadioMap",
              "development command loss", {"RadioMap 30 ms": 1.022292, "RadioMap 4 s": 4.406667}, "% loss", "rejected at development", "loss",
              "The new semantic-JEPA candidate fails its registered comparison against the matched controls"),
        codex("selected_jepa", "Compact context JEPA", "docs/SELECTED_JEPA_RESULTS.md", "compact context JEPA (13,970 parameters)", "RadioMap; matched supervised",
              "command loss", {}, "", "every registered comparison fails", "loss", "The new compact context-JEPA does not establish a service advantage"),
        codex("representation_repair", "Representation repair", "docs/REPRESENTATION_REPAIR_RESULTS.md", "JEPA with a training-only projection", "same JEPA without it",
              "packet-service prediction", {}, "", "no consistent service gain", "neutral", "It does not consistently improve packet-service prediction"),
        codex("spatial_control", "Causal warehouse control: support and recovery", "docs/research_status/SPATIAL_CONTROL.md", "fixed regression model + support guard + recovery",
              "RadioMap", "missed commands, fast handover", {"learned before": 1.635625, "learned after repair": 1.079375, "RadioMap": 1.044375}, "% missed",
              "repairs 34% of the loss; RadioMap still lower", "loss", "support + recovery reduces its mean misses by 34.0%"),
        codex("e7_calplanning", "E7 calibration and planner utility", "docs/research_status/E7_CALPLANNING_RESULTS.md", "geometry-and-censor calibrated map",
              "corrected v2 map", "analytical mission utility, 15 environments", {}, "", "map accuracy up, planning mixed (7–9 of 15 harmed)", "neutral",
              "Map accuracy improves at every setting. Planning gains are mixed"),
    ]
    def _norm(t):
        return re.sub(r"\s+", " ", t.replace("**", "").replace("`", ""))
    for x in cx:  # every quoted figure and quote must appear verbatim in its document
        text = _norm((ROOT / x["source"]).read_text(encoding="utf-8"))
        assert _norm(x["quote"]) in text, f"quote not found in {x['source']}: {x['quote']}"
        for k, v in x["values"].items():
            assert any(f in text for f in (f"{v}", f"{v:.2f}", f"{v:.3f}")), f"{x['source']}: {k}={v} not found"
    not_sim = [{"title": "Measured Wi-Fi transfer assay", "doc": "docs/MEASURED_RADIO_RESULTS.md"},
               {"title": "RadioJEPA measured-view pilot (factory channel)", "doc": "docs/RADIOJEPA_RESULTS.md"},
               {"title": "Real complex-CSI completion", "doc": "docs/research_status/CSI_RESULTS.md"},
               {"title": "Real phone and location transfer", "doc": "docs/device_transfer/RESULTS.md"}]
    return {"ours": ours, "codex": cx, "not_simulation": not_sim,
            "default": "RadioMap + WirelessJEV remains the application default; no learned radio model has been promoted.",
            "default_source": "README.md; docs/HANDOFF.md"}


def headline(camps: list[dict]) -> list[dict]:
    by = {c["id"]: c for c in camps}

    def g(cid, cond, arm):
        try:
            st = by[cid]["results"][cond][arm]
            return {"mpm": st["mpm"], "ci": st.get("ci")}
        except KeyError:
            return None
    stages = [
        ("Independent blockage, realistic", "Second radio", "wifi_vs_5g", "main", "vertical_client", "nr_30ms", None),
        ("Independent blockage, realistic", "Same-card second path", "cross_ap", "main", "vertical_shared", "nr_30ms", "wifi8"),
        ("Forklifts block both paths, realistic", "Two AP paths", "moving_blockers", "main", "shared", "nr_30ms", "wifi8"),
        ("Forklifts, hardest radio condition", "Two AP paths alone", "moving_blockers", "stress", "shared", "nr_30ms", "wifi8"),
        ("Forklifts, hardest radio condition", "+ relay (instant, optimistic)", "moving_blockers", "stress", "shared+relay", "nr_30ms", "wifi8"),
        ("Forklifts, hardest, realistic relay", "+ relay (20 ms, helpers free 70%)", "relay_realism", "stress", "shared+relay[realistic]", "nr_30ms", None),
        ("Forklifts, hardest, pessimistic relay", "+ relay (30 ms, helpers free 50%)", "relay_realism", "stress", "shared+relay[pessimistic]", "nr_30ms", None),
    ]
    out = []
    for stage, ours_label, cid, cond, ours, nr, w8 in stages:
        out.append({"stage": stage, "ours_label": ours_label, "campaign": cid, "condition": cond, "scenes": by[cid]["setup"]["scenes"],
                    "ours": g(cid, cond, ours), "nr": g(cid, cond, nr), "wifi8": g(cid, cond, w8) if w8 else None,
                    "source": by[cid]["sources"]["results"]})
    return out


def main():
    camps = campaigns()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = None
    bundle = {"generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "commit": commit,
              "scope": "Simulation only. Field testing (the real SmartCar, the runbook, the lap viewer) lives in the field platform, not here.",
              "metric": "Missed control packets per million (a control packet misses when it is not delivered within 50 ms).",
              "metric_source": "src/wirelessjev/native/sim.py (CONTROL_DEADLINE = 5 slots of 10 ms)",
              "intervals": "95% bootstrap over scenes (2,000 resamples, seed 0): ratio of pooled counts where counts exist, else mean of scene rates.",
              "campaigns": camps, "headline": headline(camps), "learned": learned_line()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bundle, indent=1, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o)))
    n_h = sum(len(c["hypotheses"]) for c in camps)
    print(f"wrote {rel(OUT)}: {len(camps)} campaigns, {n_h} hypotheses, {len(bundle['learned']['ours'])} + {len(bundle['learned']['codex'])} learned-model studies, "
          f"{OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
