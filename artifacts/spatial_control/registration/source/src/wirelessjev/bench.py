"""End-to-end benchmark for milestone 1 (simulated 10-car Raspbot fleet).

Every WirelessJEV number below comes from executing the pinned d3-lite model;
every reference number comes from the exact oracle. Run:

    uv run python -m wirelessjev.bench --scenes 100 --out artifacts/bench

Sections: fleet zero-shot vs baselines, intent-flip, contention, model-removal
ablation, DEFER rate, bias audit, oracle-teacher readout distillation,
membership-inference audit + region unlearning, swarm merge (with a poisoned
car), traffic labelling (paper ablation + robustness), admission direction.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import hmac
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .candidates import DEFER_ID, generate, oracle_scores
from .distill import (Example, accuracy, dataset_digest, distill, distill_listwise,
                      list_choices, list_membership_auc, save_adapter, soft_target)
from .missions import INTENT_FLIP_PAIRS, MISSIONS, WAREHOUSE_MISSIONS

ALL_MISSIONS = {**MISSIONS, **WAREHOUSE_MISSIONS}
from .oracle import evaluate, mission_utility
from .policy import (FLEET_RUBRIC, MODEL_REPO, MODEL_REVISION, WirelessJEV,
                     fleet_state_text, score_options)
from .swarm import Reputation, SwarmConsensus, SwarmNode
from .teachers import BASELINES, admission_oracle
from .teachers.admission import PRICE_LEVELS
from . import traffic as T
from .twin import make_scene

TAU = 0.15


def split_of(seed: int) -> str:
    r = seed % 10
    return "train" if r < 6 else "validation" if r < 8 else "test"


def _criteria(options):
    return {o.option_id: o.text for o in options}


def _norm_regret(scores, choice):
    vals = [v for k, v in scores.items()]
    best, worst = max(vals), min(vals)
    return (best - scores[choice]) / max(best - worst, 1e-9)


# --------------------------------------------------------------------------
def run_fleet(engine: WirelessJEV, n_scenes: int, log, *, scene_fn=None,
              missions=None, seed0: int = 1000) -> dict:
    scene_fn = scene_fn or make_scene
    missions = missions or MISSIONS
    cases = []
    started = time.time()
    for i in range(n_scenes):
        seed = seed0 + i
        scene = scene_fn(seed)
        options, rejected = generate(scene, seed=seed, missions=missions)
        channels = {a.ap_id: a.channel for a in scene.aps}
        for m in missions.values():
            scores = oracle_scores(scene, options, m)
            state = fleet_state_text(scene, m)
            crit = _criteria(options)
            # mode 1: one choice question over all options
            _, probs, err1 = engine.capture(state, FLEET_RUBRIC, crit)
            # mode 2: option-wise yes/no scoring (distillable, position-free)
            choice_ow, p_true, ms_ow, hidden, err2, true_row = score_options(
                engine, state, crit, capture=True)
            order = [o.option_id for o in options]
            cases.append({
                "case_id": f"s{seed}-{m.mission_id}", "seed": seed,
                "region": scene.region, "split": split_of(seed),
                "mission": m.mission_id, "order": order, "scores": scores,
                "jev_choice": max(probs, key=probs.get), "probs": probs,
                "jev": choice_ow, "p_true": p_true, "ms_optionwise": ms_ow,
                "parity_err": max(err1, err2), "hidden": hidden,
                "true_row": true_row, "scene": scene, "options": options,
                "max_load": max(o.evaluation.max_domain_load for o in options
                                if o.evaluation),
                "baselines": {
                    name: _baseline_score(scene, options, fn(scene, m), m)
                    for name, fn in BASELINES.items()},
                "n_rejected": len(rejected),
            })
        if (i + 1) % 5 == 0:
            log(f"fleet scenes {i + 1}/{n_scenes} ({time.time() - started:.0f} s)")
    return {"cases": cases}


def _baseline_score(scene, options, assoc, mission) -> float:
    """Baselines are scored with the same (horizon-aware) oracle as options."""
    from .oracle import horizon_utility
    snaps = next((getattr(o, "snaps", []) for o in options if getattr(o, "snaps", [])), [])
    if snaps:
        return horizon_utility(scene, snaps, assoc, mission)
    return mission_utility(evaluate(scene, {"assoc": assoc, "channels": {
        a.ap_id: a.channel for a in scene.aps}}), mission, len(scene.cars))


def fleet_metrics(cases, choice_key="jev") -> dict:
    def regret(c, choice):
        return max(c["scores"].values()) - c["scores"][choice]
    test = [c for c in cases if c["split"] == "test"]
    out = {"n_test_cases": len(test)}
    out["jev_regret"] = float(np.mean([regret(c, c[choice_key]) for c in test]))
    out["jev_norm_regret"] = float(np.mean([_norm_regret(c["scores"], c[choice_key]) for c in test]))
    out["jev_top1"] = float(np.mean([c["scores"][c[choice_key]] >= max(c["scores"].values()) - 1e-9
                                     for c in test]))
    out["defer_rate"] = float(np.mean([c[choice_key] == DEFER_ID for c in test]))
    rng = random.Random(0)
    out["ablation_random_option_regret"] = float(np.mean(
        [regret(c, rng.choice(c["order"])) for c in test]))
    out["ablation_first_option_regret"] = float(np.mean([regret(c, c["order"][0]) for c in test]))
    out["ablation_defer_regret"] = float(np.mean([regret(c, DEFER_ID) for c in test]))
    best = {c["case_id"]: max(c["scores"].values()) for c in test}
    for name in BASELINES:
        out[f"baseline_{name}_regret"] = float(np.mean(
            [best[c["case_id"]] - c["baselines"][name] for c in test]))
    per_mission = {}
    for m in sorted({c["mission"] for c in test}):
        sub = [c for c in test if c["mission"] == m]
        per_mission[m] = round(float(np.mean([regret(c, c[choice_key]) for c in sub])), 4)
    out["jev_regret_by_mission"] = per_mission
    # contention: split by the most loaded option in the set
    loads = np.array([c["max_load"] for c in test])
    hi = loads >= np.median(loads)
    out["regret_high_contention"] = float(np.mean([regret(c, c[choice_key]) for c, h in zip(test, hi) if h]))
    out["regret_low_contention"] = float(np.mean([regret(c, c[choice_key]) for c, h in zip(test, hi) if not h]))
    return out


def intent_flip(cases, choice_key="jev", pairs=INTENT_FLIP_PAIRS) -> dict:
    by_scene = {}
    for c in cases:
        if c["split"] == "test":
            by_scene.setdefault(c["seed"], {})[c["mission"]] = c
    differ = flips = directional = total = 0
    for scene_cases in by_scene.values():
        for a, b in pairs:
            if a not in scene_cases or b not in scene_cases:
                continue
            ca, cb = scene_cases[a], scene_cases[b]
            oa = max(ca["scores"], key=ca["scores"].get)
            ob = max(cb["scores"], key=cb["scores"].get)
            total += 1
            if oa == ob:
                continue
            differ += 1
            ja, jb = ca[choice_key], cb[choice_key]
            if ja != jb:
                flips += 1
                if (ca["scores"][ja] >= ca["scores"][jb]
                        and cb["scores"][jb] >= cb["scores"][ja]):
                    directional += 1
    return {"pairs": total, "oracle_differs": differ,
            "model_flips_when_oracle_differs": flips / max(differ, 1),
            "directionally_correct_flips": directional / max(differ, 1)}


# --------------------------------------------------------------------------
def bias_audit(engine: WirelessJEV, cases, n: int, log) -> dict:
    """Order, length and authority-wording sensitivity of the option-wise
    decision (the mode that is distilled and deployed)."""
    test = [c for c in cases if c["split"] == "test"][:n]
    order_same = length_flip = authority_flip = 0
    for c in test:
        crit = _criteria(c["options"])
        state = fleet_state_text(c["scene"], ALL_MISSIONS[c["mission"]])
        base = c["jev"]
        rev = dict(reversed(list(crit.items())))
        ch = score_options(engine, state, rev)[0]
        order_same += ch == base
        others = [k for k in crit if k not in (base, DEFER_ID)]
        target = others[0]
        padded = dict(crit)
        padded[target] = crit[target] + " " + ("Additional detail: configuration "
                                               "validated against the site survey. " * 3)
        length_flip += score_options(engine, state, padded)[0] == target
        auth = dict(crit)
        auth[target] = "Recommended by the senior site engineer. " + crit[target]
        authority_flip += score_options(engine, state, auth)[0] == target
    k = max(len(test), 1)
    log(f"bias audit on {len(test)} cases done")
    return {"n": len(test), "order_reversal_consistency": order_same / k,
            "length_padding_capture": length_flip / k,
            "authority_wording_capture": authority_flip / k,
            "chance_capture": float(np.mean([1 / max(len(c["order"]) - 1, 1) for c in test]))}


# --------------------------------------------------------------------------
def fleet_examples(cases) -> list[Example]:
    return [Example(c["case_id"], c["region"], c["split"], c["hidden"],
                    soft_target(c["scores"], c["order"], TAU), c["order"])
            for c in cases]


def run_distillation(engine: WirelessJEV, cases, out_dir: Path, log) -> dict:
    ex = fleet_examples(cases)
    true_row = cases[0]["true_row"]
    base = engine.base_readout.detach().float().cpu()
    temp = float(engine.runtime.temperature)
    readout, info = distill_listwise(base, true_row, ex, temperature=temp)
    digest = save_adapter(out_dir / "adapters" / "fleet", readout,
                          task="fleet.joint_access/option_wise", info=info,
                          dataset_digest=dataset_digest(ex))
    test = [e for e in ex if e.split == "test"]
    test_cases = [c for c in cases if c["split"] == "test"]
    for c, ch in zip(test_cases, list_choices(readout, true_row, test, temp)):
        c["jev_distilled"] = ch
    # live check: the adapter in the real runtime must reproduce the choices
    engine.load_adapter(out_dir / "adapters" / "fleet")
    sample = test_cases[:10]
    live = [engine.score(c["scene"], ALL_MISSIONS[c["mission"]], c["options"]).option_id
            for c in sample]
    engine.reset_readout()
    log("distillation done")
    members = [e for e in ex if e.split == "train"]
    nonmem = test
    region = "R0"
    unlearned, _ = distill_listwise(base, true_row, [e for e in ex if e.region != region],
                                    temperature=temp)
    r_mem = [e for e in members if e.region == region]
    r_non = [e for e in nonmem if e.region == region]
    return {
        "adapter_sha256": digest, "val_kl_before": info["val_kl_before"],
        "val_kl_after": info["val_kl_after"], "u_norm": info["u_norm"],
        "trainable_parameters": int(base.shape[1]),
        "live_adapter_agreement": float(np.mean([a == c["jev_distilled"]
                                                 for a, c in zip(live, sample)])),
        "metrics": fleet_metrics(cases, "jev_distilled"),
        "intent_flip": intent_flip(cases, "jev_distilled"),
        "mia_auc_base_readout": list_membership_auc(base, true_row, members, nonmem, temp),
        "mia_auc_distilled": list_membership_auc(readout, true_row, members, nonmem, temp),
        "mia_auc_region_before_unlearning": list_membership_auc(readout, true_row, r_mem, r_non, temp),
        "mia_auc_region_after_unlearning": list_membership_auc(unlearned, true_row, r_mem, r_non, temp),
    }, readout


# --------------------------------------------------------------------------
def run_swarm(engine: WirelessJEV, cases, log) -> dict:
    """Ten cars each distil on their own slice; consensus merges and
    validates; car-10 is a power cheater whose targets are poisoned."""
    ex = fleet_examples(cases)
    true_row = cases[0]["true_row"]
    base = engine.base_readout.detach().float().cpu()
    temp = float(engine.runtime.temperature)
    train = [e for e in ex if e.split == "train"]
    val = [e for e in ex if e.split == "validation"]
    val_cases = [c for c in cases if c["split"] == "validation"]
    cars = [f"car-{i:02d}" for i in range(1, 11)]
    key = b"site-key-demo"
    sw = SwarmConsensus(site_key=key, min_ready=5)
    rep = Reputation()
    for car in cars:
        node = SwarmNode(car, key=car.encode())
        sw.admit(node, b"round-1", hmac.new(node.key, b"round-1", hashlib.sha256).hexdigest())
    for _ in range(4):
        rep.update("car-10", rep.check("car-10", tx_power_dbm=26, cap_dbm=20,
                                       reconnects=0, used_mbps=5, declared_mbps=5))
    for car in cars[:-1]:
        rep.update(car, [])
    for car in cars:
        sw.nodes[car].reputation = rep.scores[car]

    def val_regret(w):
        chosen = list_choices(w, true_row, val, temp)
        return float(np.mean([max(c["scores"].values()) - c["scores"][ch]
                              for c, ch in zip(val_cases, chosen)]))

    base_val = val_regret(base)

    def validate(w):
        # Must beat what is live now: the current release, else the base.
        live = sw.released_readout
        ref = val_regret(live) if live is not None else base_val
        r = val_regret(w)
        return {"passed": r <= ref - 1e-6, "val_regret": r,
                "live_val_regret": ref, "base_val_regret": base_val}

    updates = {}
    for i, car in enumerate(cars):
        local = [e for j, e in enumerate(train) if j % 10 == i]
        if car == "car-10":
            local = [replace(e, target=torch.flip(e.target, [0])) for e in local]
        w, _ = distill_listwise(base, true_row, local + val, temperature=temp, epochs=40)
        updates[car] = (w, len(local))
    good = sw.merge_round(updates, cars, validate)
    poisoned = {car: (updates["car-10"][0], 10) for car in cars[:6]}
    bad = sw.merge_round(poisoned, cars, validate)
    log("swarm rounds done")
    return {"round1": {"released": good.released, "leader": good.leader,
                       "participants": good.participants,
                       "excluded_low_reputation": sorted(set(cars) - set(good.participants)),
                       "validation": good.validation},
            "round2_poisoned_updates": {"released": bad.released, "reason": bad.reason,
                                        "validation": bad.validation},
            "log_entries": len(sw.log), "log_verifies": sw.verify_log(),
            "reputation": {k: round(v, 3) for k, v in rep.scores.items()}}


# --------------------------------------------------------------------------
def run_traffic(engine: WirelessJEV, n: int, log) -> dict:
    rng = np.random.default_rng(42)
    windows = [T.synth_window(rng, f"car-{i % 10 + 1:02d}") for i in range(n)]
    res = {"n": n}
    keys = list(T.QUESTIONS)
    acc = {v: {k: 0 for k in keys} for v in ("jev_raw", "jev_digest", "rule")}
    jev_soft, lat = [], []
    for w in windows:
        raw, _ = T.jev_labels(engine, w, digest=False)
        dig, ms = T.jev_labels(engine, w, digest=True)
        lat.append(ms)
        rule = T.rule_labels(w)
        for k in keys:
            truth = getattr(w, k)
            acc["jev_raw"][k] += raw[k]["choice"] == truth
            acc["jev_digest"][k] += dig[k]["choice"] == truth
            acc["rule"][k] += rule[k]["choice"] == truth
        jev_soft.append(dig)
    res["label_accuracy"] = {v: {k: c / n for k, c in d.items()} for v, d in acc.items()}
    res["label_latency_ms_mean"] = float(np.mean(lat))
    log("traffic labels done")

    # per-question readout distillation on the digest prompt, truth as teacher
    distilled_acc = {}
    split = ["train" if i < int(0.6 * n) else "validation" if i < int(0.8 * n) else "test"
             for i in range(n)]
    temp = float(engine.runtime.temperature)
    base = engine.base_readout.detach().float().cpu()
    for k in keys:
        q = T.QUESTIONS[k]
        ex = []
        for i, w in enumerate(windows):
            h, _, _ = engine.capture(T.digest_lines(w), q["instructions"], q["criteria"])
            order = list(q["criteria"])
            target = torch.full((len(order),), 0.02 / (len(order) - 1))
            target[order.index(getattr(w, k))] = 0.98
            ex.append(Example(f"w{i}-{k}", "R0", split[i], h, target, order))
        w_k, _ = distill(base, ex, temperature=temp)
        test = [e for e in ex if e.split == "test"]
        distilled_acc[k] = {"base": accuracy(base, test, temp),
                            "distilled": accuracy(w_k, test, temp),
                            "rule": float(np.mean([T.rule_labels(windows[i])[k]["choice"]
                                                   == getattr(windows[i], k)
                                                   for i in range(n) if split[i] == "test"]))}
    res["label_distillation_test_accuracy"] = distilled_acc
    log("traffic label distillation done")

    # forecasting: frozen Chronos vs naive, plus the paper's same-size ablation
    fc = T.Forecaster("cpu")
    q, _ = fc.forecast([w.history for w in windows])
    med = q[:, :, 1]
    fut = np.stack([w.future for w in windows])
    naive = np.stack([np.repeat(w.history[-1], T.HORIZON) for w in windows])
    res["forecast_mae"] = {"chronos_bolt_small": float(np.abs(med - fut).mean()),
                           "naive_last_value": float(np.abs(naive - fut).mean())}
    resid = (fut - med).mean(1)
    sem = np.stack([T.label_vector(l) for l in jev_soft])
    cal = np.stack([T.calendar_vector(w) for w in windows])
    H = np.stack([w.history for w in windows])
    Hc = H - H.mean(0)
    _, _, Vt = np.linalg.svd(Hc[: int(0.6 * n)], full_matrices=False)
    pca = Hc @ Vt[:8].T
    rule_vec = np.stack([T.label_vector(T.rule_labels(w), soft=False) for w in windows])
    tr = slice(0, int(0.8 * n))
    te = slice(int(0.8 * n), n)
    abl = {}
    for name, X in (("none", np.zeros((n, 0))), ("semantic_jev", sem),
                    ("semantic_rule", rule_vec), ("calendar", cal), ("pca", pca)):
        W = T.ridge_fit(X[tr], resid[tr])
        pred = med[te] + T.ridge_predict(W, X[te])[:, None]
        abl[name] = float(np.sqrt(((pred - fut[te]) ** 2).mean()))
    res["residual_ablation_rmse"] = abl
    res["residual_note"] = ("closed-form ridge residual (9 parameters) fitted only "
                            "for this ablation; not deployed")

    # robustness (paper protocol): 10 dB noise, 10% missing, surge
    def perturb(kind, h, r):
        h = h.copy()
        if kind == "noise_10db":
            p = (h ** 2).mean()
            h = h + r.normal(0, math.sqrt(p / 10), h.shape)
        elif kind == "missing_10pct":
            idx = r.choice(len(h), len(h) // 10, replace=False)
            h[idx] = np.nan
            h = np.interp(np.arange(len(h)), np.flatnonzero(~np.isnan(h)), h[~np.isnan(h)])
        elif kind == "surge":
            h[-6:] *= 1.8
        return np.clip(h, 0, None)

    rob = {}
    base_rmse = float(np.sqrt(((med - fut) ** 2).mean()))
    for kind in ("noise_10db", "missing_10pct", "surge"):
        r = np.random.default_rng(7)
        qq, _ = fc.forecast([perturb(kind, w.history, r) for w in windows])
        rmse = float(np.sqrt(((qq[:, :, 1] - fut) ** 2).mean()))
        rob[kind] = {"rmse": rmse, "increase_pct": 100 * (rmse - base_rmse) / base_rmse}
    res["chronos_robustness"] = {"clean_rmse": base_rmse, **rob}
    log("traffic forecasting done")
    return res


# --------------------------------------------------------------------------
def run_admission(engine: WirelessJEV, n: int, log) -> dict:
    """Raise/hold/lower the admission level of the busiest domain. The
    numeric layer states each move's consequence; the label is the best
    one-step move (the paper's +/-1 price-step action)."""
    from .teachers.admission import admitted_flows, admitted_value
    hits = rule_hits = 0
    for i in range(n):
        scene = make_scene(5000 + i, hotspot=True)
        assoc = {c.car_id: c.current_ap for c in scene.cars}
        channels = {a.ap_id: a.channel for a in scene.aps}
        ev = evaluate(scene, {"assoc": assoc, "channels": channels})
        domains = scene.domains(channels)
        busiest = max(range(len(domains)), key=lambda d: ev.domain_load[d])
        dom = domains[busiest]
        ap = scene.ap(next(iter(dom)))
        cap = 60.0 * (1 - scene.occupancy(ap, channels))
        level = int(np.random.default_rng(i).integers(1, len(PRICE_LEVELS) - 1))
        o = admission_oracle(scene, assoc, ev.sinr_db, dom, cap, level)
        crit = {}
        for move, lv in o["neighbour_levels"].items():
            price = PRICE_LEVELS[lv]
            val = admitted_value(scene, assoc, ev.sinr_db, dom, price, cap)
            nflows = admitted_flows(scene, assoc, ev.sinr_db, dom, price)
            crit[move] = (f"{move} the admission level (to {lv}): admits {nflows} "
                          f"best-effort flows; priority-weighted delivery {val:.0f}")
        state = (f"Congested channel with about {cap:.0f} Mbps of free capacity. "
                 f"Goal: deliver the most priority-weighted best-effort traffic "
                 f"(video before telemetry before map upload).")
        q = {"type": "choice", "instructions": "Which admission move delivers "
             "the most priority-weighted traffic?", "criteria": crit}
        ans, _ = engine.ask(state, {"d": q})
        hits += ans["d"]["choice"] == o["local_direction"]
        demand = sum(f.demand_mbps for c in scene.cars if assoc[c.car_id] in dom
                     for f in c.flows if not f.critical)
        rule = "raise" if demand > cap else "lower" if demand < 0.6 * cap else "hold"
        rule_hits += rule == o["local_direction"]
    log("admission done")
    return {"n": n, "jev_local_direction_accuracy": hits / n,
            "demand_rule_accuracy": rule_hits / n, "chance": 1 / 3,
            "source": "Shen, Lam, Li (arXiv 2407.00814) price-step formulation"}

# --------------------------------------------------------------------------
class _OfflineEngine:
    """Base readout + temperature read straight from the pinned checkpoint,
    for re-running cache-based sections without loading the backbone."""

    def __init__(self):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        path = hf_hub_download(MODEL_REPO, "readout.safetensors", revision=MODEL_REVISION)
        tensors = load_file(path)
        self.base_readout = next(iter(tensors.values())).float()
        cfg = json.loads(Path(hf_hub_download(MODEL_REPO, "decision_config.json",
                                              revision=MODEL_REVISION)).read_text())

        class R:
            temperature = float(cfg.get("temperature", 1.0))
        self.runtime = R()


def rerun_swarm(out: Path) -> dict:
    cases = json.loads((out / "cases.json").read_text())
    hidden = {h["case_id"]: h["hidden"] for h in torch.load(out / "hidden_cache.pt")}
    for c in cases:
        c["hidden"] = hidden[c["case_id"]]
    report = json.loads((out / "report.json").read_text())
    report["swarm"] = run_swarm(_OfflineEngine(), cases, print)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    return report["swarm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=100)
    ap.add_argument("--traffic", type=int, default=150)
    ap.add_argument("--admission", type=int, default=60)
    ap.add_argument("--bias", type=int, default=40)
    ap.add_argument("--out", default="artifacts/bench")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "progress.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    t0 = time.time()
    engine = WirelessJEV(verify="full")
    log(f"model loaded in {engine.load_s:.1f} s on {engine.device}")
    fleet = run_fleet(engine, args.scenes, log)
    cases = fleet["cases"]
    report = {
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION,
                  "device": engine.device, "backbone_frozen": True},
        "hardware": "SmartCar.zip Raspbot: BE200 wlan1 client, brcmfmac wlan0 "
                    "management (untouched); BSSID selection only",
        "n_cases": len(cases),
        "parity_err_max": max(c["parity_err"] for c in cases),
    }
    report["fleet_zero_shot_optionwise"] = fleet_metrics(cases, "jev")
    report["intent_flip_zero_shot_optionwise"] = intent_flip(cases, "jev")
    report["fleet_zero_shot_single_choice"] = fleet_metrics(cases, "jev_choice")
    report["intent_flip_zero_shot_single_choice"] = intent_flip(cases, "jev_choice")
    report["bias_audit"] = bias_audit(engine, cases, args.bias, log)
    dist, _ = run_distillation(engine, cases, out, log)
    report["fleet_distilled"] = dist
    report["swarm"] = run_swarm(engine, cases, log)
    report["traffic"] = run_traffic(engine, args.traffic, log)
    report["admission"] = run_admission(engine, args.admission, log)
    lat = [c["ms_optionwise"] for c in cases]
    report["latency_ms"] = {"fleet_optionwise_mean": float(np.mean(lat)),
                            "fleet_optionwise_p90": float(np.percentile(lat, 90)),
                            "device": engine.device}
    report["wall_clock_s"] = time.time() - t0
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    slim = [{k: v for k, v in c.items() if k not in ("hidden", "scene", "options")}
            for c in cases]
    torch.save([{"case_id": c["case_id"], "hidden": c["hidden"]} for c in cases],
               out / "hidden_cache.pt")
    (out / "cases.json").write_text(json.dumps(slim, default=float))
    log(f"done in {report['wall_clock_s']:.0f} s -> {out / 'report.json'}")


if __name__ == "__main__":
    main()
