"""E1: radio as a native modality for d3-lite.

    uv run python -m wirelessjev.radiojev.e1 --arm radio_lora
    uv run python -m wirelessjev.radiojev.e1 --arm radio_frozen
    uv run python -m wirelessjev.radiojev.e1 --arm no_llm
    uv run python -m wirelessjev.radiojev.e1 --arm intent

All arms rank the same candidate plans of the same held-out warehouse scenes,
for held-out paraphrases of known missions and for never-seen missions, and
are scored by regret against the exact oracle.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import ALL_MISSIONS, MISSION_IDS, build, mission_texts
from .encoder import N_TOKENS, NoLLMScorer, RadioEncoder
from .features import ROBOT_RE
from .trainable import RADIO_TOKEN, TrainableD3

OUT = Path("artifacts/radiojev")
SMOKE = bool(os.environ.get("WJ_SMOKE"))
SEEDS = {"train": list(range(41000, 41100)), "validation": list(range(42000, 42012)),
         "test": list(range(43000, 43020))}
TAU = 0.15
INSTR = (f"Candidate Wi-Fi association plan for the robot fleet: {RADIO_TOKEN * N_TOKENS} "
         f"Is this the right plan for the mission?")


def state_text(text: str) -> str:
    return f"Operator mission for a fleet of warehouse robots on Wi-Fi:\n{text}"


def named_vec(car_ids, text):
    names = set(ROBOT_RE.findall(text))
    return np.array([c in names for c in car_ids], dtype=np.float32)


def cases(data, split):
    texts = mission_texts("novel" if split == "novel" else split)
    scenes = data["test"] if split in ("test", "novel") else data[split]
    out = []
    for sc in scenes:
        for mid, ts in texts.items():
            for t in ts:
                out.append((sc, mid, t))
    return out[:3] if SMOKE and split != "train" else out


def regret_of(util, choice):
    return float(util.max() - util[choice])


def evaluate(score_fn, data, log):
    res = {}
    for split in ("test", "novel"):
        regrets, top1, lat = [], [], []
        for sc, mid, text in cases(data, split):
            t0 = time.perf_counter()
            scores = score_fn(sc, mid, text)
            lat.append((time.perf_counter() - t0) * 1000)
            if scores is None:
                continue
            u = sc["util"][mid]
            c = int(np.argmax(scores))
            regrets.append(regret_of(u, c))
            top1.append(float(u[c] >= u.max() - 1e-9))
        res[split] = {"n": len(regrets), "regret": float(np.mean(regrets)) if regrets else None,
                      "regret_ci95": float(1.96 * np.std(regrets) / np.sqrt(len(regrets))) if regrets else None,
                      "top1": float(np.mean(top1)) if top1 else None,
                      "latency_ms": float(np.mean(lat)) if lat else None}
        log(f"{split}: {res[split]}")
    return res


# ---------------------------------------------------------------- LLM arms
def run_llm(arm, data, log, epochs=4, per_scene=2, lr=3e-4):
    from ..policy import WirelessJEV
    engine = WirelessJEV()
    model = TrainableD3(engine, lora_layers=8 if arm == "radio_lora" else 0)
    enc = RadioEncoder(embed_rms=model.embed_rms).to(model.device)
    params = list(enc.parameters()) + model.trainable()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    def logits(sc, text, grad):
        b = RadioEncoder.batch(sc["graphs"], named_vec(sc["car_ids"], text), model.device)
        with torch.set_grad_enabled(grad):
            radio = enc(b)
            h = model.encode(state_text(text), [INSTR] * len(sc["graphs"]), radio)
            return model.noul_logit(h)

    def val_regret():
        rs = []
        for sc, mid, text in cases(data, "validation"):
            s = logits(sc, text, False).detach().cpu().numpy()
            rs.append(regret_of(sc["util"][mid], int(np.argmax(s))))
        return float(np.mean(rs))

    train = cases(data, "train")
    rng = random.Random(0)
    best, best_state, history = val_regret(), None, []
    log(f"{arm}: validation regret before training {best:.4f}")
    t0 = time.time()
    for ep in range(epochs):
        by_scene = {}
        for c in train:
            by_scene.setdefault(c[0]["seed"], []).append(c)
        batch = [x for v in by_scene.values() for x in rng.sample(v, per_scene)]
        rng.shuffle(batch)
        if SMOKE:
            batch = batch[:2]
        losses = []
        for sc, mid, text in batch:
            target = torch.softmax(torch.tensor(sc["util"][mid]) / TAU, 0).to(model.device)
            lg = logits(sc, text, True)
            loss = F.kl_div(F.log_softmax(lg, 0), target, reduction="sum")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss))
        v = val_regret()
        history.append({"epoch": ep + 1, "train_kl": float(np.mean(losses)), "val_regret": v,
                        "elapsed_s": time.time() - t0})
        log(f"{arm}: epoch {ep + 1} train KL {np.mean(losses):.4f} val regret {v:.4f}")
        if v < best:
            best = v
            best_state = {"encoder": {k: x.detach().cpu().clone() for k, x in enc.state_dict().items()},
                          "readout": model.readout.detach().cpu().clone(),
                          "lora": __import__("wirelessjev.radiojev.trainable", fromlist=["x"]).lora_state(model.lora_layers)}
    if best_state is None and SMOKE:
        from .trainable import lora_state as _ls
        best_state = {"encoder": {k: x.detach().cpu().clone() for k, x in enc.state_dict().items()},
                      "readout": model.readout.detach().cpu().clone(), "lora": _ls(model.lora_layers)}
    if best_state is not None:
        enc.load_state_dict(best_state["encoder"])
        model.readout.data.copy_(best_state["readout"].to(model.device))
        if model.lora_layers:
            from .trainable import load_lora_state
            load_lora_state(model.lora_layers, best_state["lora"])
        torch.save(best_state, OUT / f"ckpt_{arm}.pt")

    def score(sc, mid, text):
        return logits(sc, text, False).detach().cpu().numpy()

    res = evaluate(score, data, log)
    return {"arm": arm, "history": history, "val_best": best, **res,
            "trainable_params": int(sum(p.numel() for p in params))}


# ---------------------------------------------------------------- no-LLM arm
def run_no_llm(data, log, epochs=8, lr=1e-3):
    known = [m for m in MISSION_IDS if m in mission_texts("train")]
    idx = {m: i for i, m in enumerate(known)}
    net = NoLLMScorer(len(known))
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    train = cases(data, "train")
    for ep in range(1 if SMOKE else epochs):
        random.Random(ep).shuffle(train)
        for sc, mid, text in (train[:5] if SMOKE else train):
            b = RadioEncoder.batch(sc["graphs"], named_vec(sc["car_ids"], text), "cpu")
            target = torch.softmax(torch.tensor(sc["util"][mid]) / TAU, 0)
            loss = F.kl_div(F.log_softmax(net(b, idx[mid]), 0), target, reduction="sum")
            opt.zero_grad()
            loss.backward()
            opt.step()

    def score(sc, mid, text):
        if mid not in idx:
            return None          # cannot read a never-seen mission
        b = RadioEncoder.batch(sc["graphs"], named_vec(sc["car_ids"], text), "cpu")
        with torch.no_grad():
            return net(b, idx[mid]).numpy()

    return {"arm": "no_llm", **evaluate(score, data, log)}


# ---------------------------------------------------------------- intent arm
def run_intent(data, log):
    from safetensors.torch import load_file
    from ..fastplan import BatchEvaluator
    from ..intent import IntentReader
    from ..oracle import horizon_snapshots
    from ..policy import WirelessJEV
    from ..twin.warehouse import make_warehouse
    engine = WirelessJEV()
    readout = load_file("artifacts/intent/adapters/intent/readout.safetensors")["readout"]
    reader = IntentReader(engine, readout=readout)
    evs = {}

    def score(sc, mid, text):
        if sc["seed"] not in evs:
            s = make_warehouse(sc["seed"])
            evs[sc["seed"]] = BatchEvaluator(s, horizon_snapshots(s))
        ev = evs[sc["seed"]]
        r = reader.read(text)
        focus = [ev.car_ids.index(c) for c in r.focus if c in ev.car_ids]
        return ev.utility(sc["plans"], r.weights, focus)

    return {"arm": "intent", **evaluate(score, data, log)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["radio_lora", "radio_frozen", "no_llm", "intent"])
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    data = build(OUT / "e1_data.pkl", SEEDS, log=log)
    if args.arm in ("radio_lora", "radio_frozen"):
        res = run_llm(args.arm, data, log, epochs=1 if SMOKE else args.epochs)
    elif args.arm == "no_llm":
        res = run_no_llm(data, log)
    else:
        res = run_intent(data, log)
    if SMOKE:
        log("smoke ok"); return
    (OUT / f"e1_{args.arm}.json").write_text(json.dumps(res, indent=2, default=float))
    log(f"saved {OUT / f'e1_{args.arm}.json'}")


if __name__ == "__main__":
    main()
