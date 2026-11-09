"""E2: typed value head for missions a weight vector cannot express.

    uv run python -m wirelessjev.radiojev.e2 --arm radio_q      # from the E1 checkpoint
    uv run python -m wirelessjev.radiojev.e2 --arm e1_zero      # E1 model, no E2 training
    uv run python -m wirelessjev.radiojev.e2 --arm intent       # intent weights + exact utility

Candidates include zone-, video- and budget-oriented plans; every plan is
scored by exact non-linear oracles. The value head is trained with a
listwise loss plus a z-scored value regression, and evaluated on held-out
paraphrases and on never-seen non-linear missions (new numbers, new zone).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..fastplan import FastPlanner
from ..missions import WAREHOUSE_MISSIONS
from ..teachers.assign import legacy_policy
from ..twin.warehouse import make_warehouse
from .e1 import INSTR, SMOKE, named_vec, state_text
from .encoder import RadioEncoder, widen_from
from .features import ROBOT_F_POS, plan_graph
from .nonlinear import Outcomes, nl_texts, nl_utility
from .trainable import TrainableD3, load_lora_state, lora_state

OUT = Path("artifacts/radiojev")
SEEDS = {"train": list(range(51000, 51080)), "validation": list(range(52000, 52010)),
         "test": list(range(53000, 53020))}


def outcomes(fp: FastPlanner, A: np.ndarray, scene) -> Outcomes:
    ev = fp.ev
    terms = [ev._snapshot_terms(A, snap, k == 0, 30.0) for k, snap in enumerate(ev.snaps)]
    sat = np.stack([t["sat"] for t in terms], 1)
    video = (ev.cf_dem[:, 1:] > 0).any(1) if ev.cf_dem.shape[1] > 1 else np.zeros(ev.n, bool)
    return Outcomes(sat=sat, U=np.mean([t["U"] for t in terms], 0),
                    jain=np.mean([t["jain"] for t in terms], 0),
                    viol=np.mean([t["viol"] for t in terms], 0),
                    moves=terms[0]["handovers"],
                    x=np.array([c.x for c in scene.cars]), y=np.array([c.y for c in scene.cars]),
                    video=video)


def build_scene(seed: int, k: int = 10, scene_fn=make_warehouse, linear: bool = False) -> dict:
    scene = scene_fn(seed)
    fp = FastPlanner(scene)
    ev = fp.ev
    starts = [ev.current, ev.from_assoc(legacy_policy(scene))]
    xs = np.array([c.x for c in scene.cars]); ys = np.array([c.y for c in scene.cars])
    video = [i for i in range(ev.n) if ev.cf_dem.shape[1] > 1 and ev.cf_dem[i, 1] > 0]
    plans = [ev.current.copy(), starts[1].copy()]
    for m in WAREHOUSE_MISSIONS.values():
        plans.append(fp.search(m.weights, m.focus_cars, starts=starts, max_iter=40)[0])
    base_w = dict(WAREHOUSE_MISSIONS["shift_fairness"].weights)
    focus_sets = [np.where(ys < 40)[0], np.where(ys >= 40)[0], np.where(xs < 60)[0], np.array(video)]
    for fs in focus_sets:
        w = dict(base_w, focus=4.0)
        ids = tuple(ev.car_ids[i] for i in fs)
        plans.append(fp.search(w, ids, starts=starts, max_iter=40)[0])
    for ho in (2.0, 6.0):
        w = dict(WAREHOUSE_MISSIONS["map_rollout"].weights, ho=ho)
        plans.append(fp.search(w, (), starts=starts, max_iter=40)[0])
    rng = np.random.default_rng(seed)
    pool = list(plans[2:])
    while len(plans) < 3 * k:
        a = pool[int(rng.integers(len(pool)))].copy()
        for _ in range(int(rng.integers(1, 6))):
            i = int(rng.integers(ev.n)); a[i] = int(rng.choice(ev.visible[i]))
        plans.append(a)
    uniq, seen = [], set()
    for a in plans:
        if a.tobytes() not in seen:
            seen.add(a.tobytes()); uniq.append(a)
    head, tail = uniq[:12], uniq[12:]
    rng.shuffle(tail)
    chosen = (head + tail)[:k]
    chosen = [chosen[i] for i in rng.permutation(len(chosen))]
    A = np.stack(chosen)
    o = outcomes(fp, A, scene)
    graphs = [plan_graph(ev, a, np.zeros(ev.n), positions=(xs, ys)) for a in chosen]
    util = {mid: nl_utility(mid, o) for mid in
            list(nl_texts("train")) + list(nl_texts("novel"))}
    if linear:
        from ..mission_texts import all_known
        for mid, m in all_known().items():
            focus = [ev.car_ids.index(c) for c in m.focus_cars if c in ev.car_ids]
            util[mid] = ev.utility(A, m.weights, focus).astype(np.float32)
    return {"seed": seed, "plans": A, "graphs": graphs, "util": util, "car_ids": ev.car_ids}


def build(path: Path, log) -> dict:
    if path.exists():
        return pickle.loads(path.read_bytes())
    data, t0 = {}, time.time()
    for split, ss in SEEDS.items():
        data[split] = []
        for i, s in enumerate(ss):
            data[split].append(build_scene(s))
            if (i + 1) % 20 == 0:
                log(f"{split} {i + 1}/{len(ss)} ({time.time() - t0:.0f} s)")
    path.write_bytes(pickle.dumps(data))
    return data


def cases(data, split):
    texts = nl_texts(split)
    scenes = data["test"] if split in ("test", "novel") else data[split]
    out = [(sc, mid, t) for sc in scenes for mid, ts in texts.items() for t in ts]
    return out[:3] if SMOKE and split != "train" else out


def zscore(u):
    s = u.std()
    return (u - u.mean()) / (s if s > 1e-9 else 1.0)


def evaluate(score_fn, data, log):
    res = {}
    for split in ("test", "novel"):
        nreg, top1, rho, pear, lat, by = [], [], [], [], [], {}
        for sc, mid, text in cases(data, split):
            t0 = time.perf_counter()
            s = score_fn(sc, mid, text)
            lat.append((time.perf_counter() - t0) * 1000)
            u = sc["util"][mid]
            c = int(np.argmax(s))
            span = max(float(u.max() - u.min()), 1e-9)
            r = float(u.max() - u[c]) / span
            nreg.append(r); top1.append(float(u[c] >= u.max() - 1e-9))
            by.setdefault(mid, []).append(r)
            from scipy.stats import pearsonr, spearmanr
            if np.std(s) > 0 and np.std(u) > 0:
                rho.append(spearmanr(s, u).statistic); pear.append(pearsonr(s, u).statistic)
        res[split] = {"n": len(nreg), "norm_regret": float(np.mean(nreg)),
                      "norm_regret_ci95": float(1.96 * np.std(nreg) / np.sqrt(len(nreg))),
                      "top1": float(np.mean(top1)), "spearman": float(np.nanmean(rho)),
                      "pearson": float(np.nanmean(pear)), "latency_ms": float(np.mean(lat)),
                      "by_mission": {k: float(np.mean(v)) for k, v in by.items()}}
        log(f"{split}: { {k: v for k, v in res[split].items() if k != 'by_mission'} }")
    return res


def run_radio(arm, data, log, epochs=4, per_scene=3, lr=2e-4):
    from ..policy import WirelessJEV
    engine = WirelessJEV()
    model = TrainableD3(engine, lora_layers=8)
    enc = RadioEncoder(embed_rms=model.embed_rms, robot_f=ROBOT_F_POS).to(model.device)
    ck = torch.load(OUT / "ckpt_radio_lora.pt")
    widen_from(ck["encoder"], enc)
    enc.to(model.device)
    model.readout.data.copy_(ck["readout"].to(model.device))
    load_lora_state(model.lora_layers, ck["lora"])
    params = list(enc.parameters()) + model.trainable()
    opt = torch.optim.AdamW(params, lr=lr)

    def value(sc, text, grad):
        b = RadioEncoder.batch(sc["graphs"], named_vec(sc["car_ids"], text), model.device)
        with torch.set_grad_enabled(grad):
            h = model.encode(state_text(text), [INSTR] * len(sc["graphs"]), enc(b))
            return model.noul_logit(h)

    def val():
        rs = []
        for sc, mid, text in cases(data, "validation"):
            s = value(sc, text, False).cpu().numpy(); u = sc["util"][mid]
            rs.append(float(u.max() - u[int(np.argmax(s))]) / max(float(u.max() - u.min()), 1e-9))
        return float(np.mean(rs))

    history = []
    if arm == "radio_q":
        best, best_state = val(), None
        log(f"radio_q: validation norm regret before E2 training {best:.4f}")
        train = cases(data, "train")
        rng = random.Random(1)
        t0 = time.time()
        for ep in range(epochs):
            by = {}
            for c in train:
                by.setdefault(c[0]["seed"], []).append(c)
            batch = [x for v in by.values() for x in rng.sample(v, min(per_scene, len(v)))]
            rng.shuffle(batch)
            if SMOKE:
                batch = batch[:2]
            losses = []
            for sc, mid, text in batch:
                z = torch.tensor(zscore(sc["util"][mid]), device=model.device)
                v = value(sc, text, True)
                loss = F.kl_div(F.log_softmax(v, 0), torch.softmax(z / 0.5, 0), reduction="sum") \
                    + 0.5 * F.mse_loss(v, z)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                losses.append(float(loss))
            vr = val()
            history.append({"epoch": ep + 1, "loss": float(np.mean(losses)), "val_norm_regret": vr,
                            "elapsed_s": time.time() - t0})
            log(f"radio_q: epoch {ep + 1} loss {np.mean(losses):.4f} val {vr:.4f}")
            if vr < best:
                best = vr
                best_state = {"encoder": {k: x.detach().cpu().clone() for k, x in enc.state_dict().items()},
                              "readout": model.readout.detach().cpu().clone(),
                              "lora": lora_state(model.lora_layers)}
        if best_state is None and SMOKE:
            best_state = {"encoder": {k: x.detach().cpu().clone() for k, x in enc.state_dict().items()},
                          "readout": model.readout.detach().cpu().clone(),
                          "lora": lora_state(model.lora_layers)}
        if best_state:
            enc.load_state_dict(best_state["encoder"]); enc.to(model.device)
            model.readout.data.copy_(best_state["readout"].to(model.device))
            load_lora_state(model.lora_layers, best_state["lora"])
            torch.save(best_state, OUT / "ckpt_radio_q.pt")
    res = evaluate(lambda sc, mid, t: value(sc, t, False).cpu().numpy(), data, log)
    return {"arm": arm, "history": history, **res}


def run_intent(data, log):
    from safetensors.torch import load_file
    from ..fastplan import BatchEvaluator
    from ..intent import IntentReader
    from ..oracle import horizon_snapshots
    from ..policy import WirelessJEV
    reader = IntentReader(WirelessJEV(),
                          readout=load_file("artifacts/intent/adapters/intent/readout.safetensors")["readout"])
    evs = {}

    def score(sc, mid, text):
        if sc["seed"] not in evs:
            s = make_warehouse(sc["seed"]); evs[sc["seed"]] = BatchEvaluator(s, horizon_snapshots(s))
        ev = evs[sc["seed"]]; r = reader.read(text)
        return ev.utility(sc["plans"], r.weights, [ev.car_ids.index(c) for c in r.focus if c in ev.car_ids])

    return {"arm": "intent", **evaluate(score, data, log)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["radio_q", "e1_zero", "intent"])
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    data = build(OUT / "e2_data.pkl", log)
    res = run_intent(data, log) if args.arm == "intent" else run_radio(
        args.arm, data, log, 1 if SMOKE else args.epochs)
    if SMOKE:
        log("smoke ok"); return
    (OUT / f"e2_{args.arm}.json").write_text(json.dumps(res, indent=2, default=float))
    log("saved")


if __name__ == "__main__":
    main()
