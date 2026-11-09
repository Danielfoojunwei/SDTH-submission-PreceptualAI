"""Intent-to-objective WirelessJEV benchmark (warehouse, 100 robots).

    uv run python -m wirelessjev.bench_intent --out artifacts/intent

1. Intent reading: level accuracy and weight error, zero-shot vs a distilled
   score readout (trained on original + 2 paraphrases per known mission),
   on held-out paraphrases and on four never-seen missions.
2. Shortlist regret on the same held-out warehouse scenes and options as
   bench_warehouse (comparable to the option-wise distilled 0.113), choosing
   argmax of the exact utility under the inferred weights.
3. Plan quality: FastPlanner under inferred weights, scored with the true
   weights, against FastPlanner under true weights, the best shortlist option,
   MILP and legacy. Also for the never-seen missions.
4. Latency: intent read (once per mission), planner setup and search.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .bench import split_of
from .candidates import DEFER_ID, generate
from .distill import Example, accuracy, distill, save_adapter, dataset_digest
from .fastplan import FastPlanner
from .intent import DIMENSIONS, IntentReader, capture, true_levels
from .mission_texts import NOVEL, PARAPHRASES, TRAIN_PARAPHRASES, all_known
from .missions import WAREHOUSE_MISSIONS
from .oracle import horizon_utility
from .policy import WirelessJEV
from .teachers.assign import legacy_policy, milp_assignment
from .twin.warehouse import make_warehouse


def texts():
    """(mission_id, text, split, mission) rows."""
    rows = []
    for mid, m in all_known().items():
        rows.append((mid, m.intent, "train", m))
        for k, t in enumerate(PARAPHRASES[mid]):
            split = "train" if k < TRAIN_PARAPHRASES else ("validation" if k == TRAIN_PARAPHRASES else "test")
            rows.append((mid, t, split, m))
    for mid, (m, ts) in NOVEL.items():
        for t in ts:
            rows.append((mid, t, "novel", m))
    return rows


def level_target(level: int, k: int = 5, smooth: float = 0.05) -> torch.Tensor:
    t = torch.full((k,), smooth / (k - 1))
    t[level] = 1 - smooth
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/intent")
    ap.add_argument("--warehouse", default="artifacts/warehouse")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    engine = WirelessJEV()
    temp = float(engine.runtime.temperature)
    base = engine.base_readout.detach().float().cpu()
    report = {}

    # ---- 1. capture + distil the score readout --------------------------
    rows = texts()
    examples, zero = [], []
    for idx, (mid, text, split, m) in enumerate(rows):
        hidden, _ = capture(engine, text)
        tl = true_levels(m.weights)
        for key, _, _ in DIMENSIONS:
            examples.append(Example(f"{idx}-{key}", mid, "train" if split == "train" else
                                    ("validation" if split == "validation" else "test"),
                                    hidden[key], level_target(tl[key]), [str(i) for i in range(5)]))
        zero.append((mid, text, split, m))
    log(f"captured {len(rows)} mission texts")
    readout, info = distill(base, examples, temperature=temp, epochs=80, lr=3e-3)
    save_adapter(out / "adapters" / "intent", readout, task="intent/score_levels",
                 info=info, dataset_digest=dataset_digest(examples))
    by_split = lambda s: [e for e, r in zip(examples, [r for r in rows for _ in DIMENSIONS])
                          if r[2] == s]
    report["intent_level_accuracy"] = {
        s: {"zero_shot": accuracy(base, by_split(s), temp),
            "distilled": accuracy(readout, by_split(s), temp)}
        for s in ("train", "test", "novel")}
    log("intent distillation done")

    zs = IntentReader(engine)
    ds = IntentReader(engine, readout=readout)
    readings = {}
    lat = []
    for mid, text, split, m in rows:
        for name, reader in (("zero_shot", zs), ("distilled", ds)):
            r = reader.read(text)
            lat.append(r.latency_ms)
            readings[(name, text)] = r
    errs = {}
    for name in ("zero_shot", "distilled"):
        for split in ("test", "novel"):
            e = [np.mean([abs(readings[(name, t)].weights[k] - m.weights[k]) for k, _, _ in DIMENSIONS])
                 for mid, t, s, m in rows if s == split]
            errs[f"{name}_{split}"] = float(np.mean(e))
    report["intent_weight_mae"] = errs
    report["intent_read_latency_ms"] = float(np.mean(lat))

    # ---- 2 + 3. warehouse decisions ---------------------------------------
    cases = {c["case_id"]: c for c in json.loads((Path(args.warehouse) / "cases.json").read_text())}
    test_seeds = sorted({c["seed"] for c in cases.values() if c["split"] == "test"})
    held_out_text = {mid: PARAPHRASES[mid][-1] for mid in WAREHOUSE_MISSIONS}
    shortlist = {k: [] for k in ("old_optionwise_distilled", "true_weights",
                                 "intent_zero_shot", "intent_distilled",
                                 "intent_distilled_heldout_paraphrase")}
    plan = {k: [] for k in ("fast_true", "fast_intent_distilled", "fast_intent_zero_shot",
                            "best_shortlist", "milp", "legacy", "old_optionwise_choice")}
    timing = {"setup_ms": [], "search_ms": []}
    for seed in test_seeds:
        scene = make_warehouse(seed)
        options, _ = generate(scene, seed=seed, missions=WAREHOUSE_MISSIONS)
        snaps = options[0].snaps
        channels = {a.ap_id: a.channel for a in scene.aps}
        for mid, m in WAREHOUSE_MISSIONS.items():
            case = cases[f"s{seed}-{mid}"]
            true = {o.option_id: horizon_utility(scene, snaps, o.assoc, m, channels)
                    - (0.05 if o.is_defer else 0.0) for o in options}
            best = max(true.values())

            def pick(weights, focus):
                from .missions import Mission
                mm = Mission(mid, "", weights, focus)
                s = {o.option_id: horizon_utility(scene, snaps, o.assoc, mm, channels)
                     - (0.05 if o.is_defer else 0.0) for o in options}
                return max(s, key=s.get)

            shortlist["old_optionwise_distilled"].append(best - true[case["jev_distilled"]])
            shortlist["true_weights"].append(best - true[pick(m.weights, m.focus_cars)])
            for name, text in (("intent_zero_shot", m.intent), ("intent_distilled", m.intent),
                               ("intent_distilled_heldout_paraphrase", held_out_text[mid])):
                r = readings.get(("zero_shot" if "zero" in name else "distilled", text))
                if r is None:
                    r = (zs if "zero" in name else ds).read(text)
                shortlist[name].append(best - true[pick(r.weights, r.focus)])

            fp = FastPlanner(scene)
            timing["setup_ms"].append(fp.setup_ms)
            starts = [fp.ev.current, fp.ev.from_assoc(legacy_policy(scene))]
            score = lambda assoc: horizon_utility(scene, snaps, assoc, m, channels)
            a_true, _, st = fp.search(m.weights, m.focus_cars, starts=starts)
            timing["search_ms"].append(st["search_ms"])
            r_d, r_z = ds.read(m.intent), zs.read(m.intent)
            a_d, _, st = fp.search(r_d.weights, r_d.focus, starts=starts)
            timing["search_ms"].append(st["search_ms"])
            a_z, _, _ = fp.search(r_z.weights, r_z.focus, starts=starts)
            plan["fast_true"].append(score(fp.ev.to_assoc(a_true)))
            plan["fast_intent_distilled"].append(score(fp.ev.to_assoc(a_d)))
            plan["fast_intent_zero_shot"].append(score(fp.ev.to_assoc(a_z)))
            plan["best_shortlist"].append(best)
            plan["milp"].append(score(milp_assignment(scene, m)))
            plan["legacy"].append(score(legacy_policy(scene)))
            plan["old_optionwise_choice"].append(true[case["jev_distilled"]])
        log(f"scene {seed} done")

    report["shortlist_regret"] = {k: float(np.mean(v)) for k, v in shortlist.items()}
    ref = np.maximum(np.array(plan["fast_true"]), np.array(plan["best_shortlist"]))
    report["plan_regret_vs_best_known"] = {k: float(np.mean(ref - np.array(v))) for k, v in plan.items()}
    report["n_decisions"] = len(plan["fast_true"])

    # ---- novel missions: planning only --------------------------------------
    novel = {k: [] for k in ("fast_true", "fast_intent_distilled", "fast_intent_zero_shot", "milp", "legacy")}
    for seed in test_seeds[:5]:
        scene = make_warehouse(seed)
        fp = FastPlanner(scene)
        snaps = fp.snaps
        channels = {a.ap_id: a.channel for a in scene.aps}
        starts = [fp.ev.current, fp.ev.from_assoc(legacy_policy(scene))]
        for mid, (m, ts) in NOVEL.items():
            score = lambda assoc: horizon_utility(scene, snaps, assoc, m, channels)
            novel["fast_true"].append(score(fp.ev.to_assoc(fp.search(m.weights, m.focus_cars, starts=starts)[0])))
            for name, reader in (("fast_intent_distilled", ds), ("fast_intent_zero_shot", zs)):
                vals = []
                for t in ts:
                    r = reader.read(t)
                    vals.append(score(fp.ev.to_assoc(fp.search(r.weights, r.focus, starts=starts)[0])))
                novel[name].append(float(np.mean(vals)))
            novel["milp"].append(score(milp_assignment(scene, m)))
            novel["legacy"].append(score(legacy_policy(scene)))
    nref = np.array(novel["fast_true"])
    report["novel_mission_plan_regret"] = {k: float(np.mean(nref - np.array(v))) for k, v in novel.items()}
    report["latency_ms"] = {"planner_setup_mean": float(np.mean(timing["setup_ms"])),
                            "planner_search_mean": float(np.mean(timing["search_ms"])),
                            "planner_search_p90": float(np.percentile(timing["search_ms"], 90)),
                            "intent_read_once_per_mission": report["intent_read_latency_ms"],
                            "previous_decision_total": 15000.0}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    log(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
