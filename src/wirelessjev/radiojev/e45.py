"""E4: personalised swarm LoRA across warehouse sites.  E5: forgetting a site.

    uv run python -m wirelessjev.radiojev.e45

Sites differ in radio conditions (scanner load, rack density). Every arm
starts from the E3 shared model. LoRA layers are split as in Zhang, Lam,
Shen, Wang & Li (hierarchical personalised FDRL, 2023): the first 6 adapted
layers and the radio encoder are the shared "basic" part, the last 2 adapted
layers and the readout rows are each site's "personal" part. Merges use the
swarm protocol of Li/Wang/Shen/Lam (leader by score, weighted merge,
validation against the live release).

Arms (same local step budget):
  local       each site trains alone
  fedavg_all  everything merged each round (no personal part)
  swarm_pers  basic merged through the gated swarm, personal kept local

E5 removes one site from swarm_pers's shared part by (a) exact retraining
without it and (b) gated distillation (Li, Shui, Lam, Shen & Wang, ICAIIC
2025): a reset student distils the current model on retained sites only.
Audit: loss-threshold membership inference AUC on the forgotten site's
training cases (members) vs its held-out cases (non-members).
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..distill import auc
from ..mission_texts import PARAPHRASES, all_known
from ..missions import WAREHOUSE_MISSIONS
from ..swarm import SwarmConsensus, SwarmNode
from ..twin.warehouse import make_warehouse
from .e1 import INSTR, SMOKE, named_vec, state_text
from .e2 import build_scene, zscore
from .e3 import MultiTask
from .nonlinear import nl_texts
from .trainable import load_lora_state, lora_state

OUT = Path("artifacts/radiojev")
SITES = {
    "S1_normal": dict(scanners="normal", rack_scale=1.0),
    "S2_heavy_scanners": dict(scanners="heavy", rack_scale=1.0),
    "S3_dense_racks": dict(scanners="light", rack_scale=1.6),
    "S4_light_racks": dict(scanners="normal", rack_scale=0.5),
}
FORGET = "S2_heavy_scanners"
N_TRAIN, N_TEST = 16, 6
ROUNDS, LOCAL_STEPS = (1, 1) if SMOKE else (3, 24)


def texts_for(split):
    out = {}
    for mid, ts in nl_texts("train" if split == "train" else "test").items():
        out[mid] = ts
    for mid in WAREHOUSE_MISSIONS:
        p = PARAPHRASES[mid]
        out[mid] = [all_known()[mid].intent] + p[:2] if split == "train" else [p[-1]]
    return out


def build_sites(log):
    path = OUT / "e45_data.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    data, t0 = {}, time.time()
    for si, (site, kw) in enumerate(SITES.items()):
        fn = lambda seed, kw=kw: make_warehouse(seed, **kw)
        base = 70000 + 1000 * si
        data[site] = {"train": [build_scene(base + i, scene_fn=fn, linear=True) for i in range(N_TRAIN)],
                      "test": [build_scene(base + 500 + i, scene_fn=fn, linear=True) for i in range(N_TEST)]}
        log(f"built {site} ({time.time() - t0:.0f} s)")
    path.write_bytes(pickle.dumps(data))
    return data


def cases(scenes, split):
    out = [(sc, mid, t) for sc in scenes for mid, ts in texts_for(split).items() for t in ts]
    return out[:2] if SMOKE else out


class Bank:
    """Holds the shared model; swaps per-site state in and out."""

    def __init__(self, engine):
        self.mt = MultiTask(engine, OUT / "ckpt_e3.pt" if (OUT / "ckpt_e3.pt").exists() else OUT / "ckpt_radio_q.pt")
        e3 = OUT / "ckpt_e3.pt"
        if e3.exists():
            ck = torch.load(e3)
            self.mt.heads.load_state_dict({k: v for k, v in ck["heads"].items()})
            self.mt.heads.to(self.mt.m.device)
            load_lora_state(self.mt.m.lora_layers, ck["lora"])
            self.mt.m.readout.data.copy_(ck["readout"].to(self.mt.m.device))
        self.init = self.snapshot()

    def snapshot(self):
        return {"lora": lora_state(self.mt.m.lora_layers),
                "enc": {k: v.detach().cpu().clone() for k, v in self.mt.heads["assoc"].state_dict().items()},
                "readout": self.mt.m.readout.detach().cpu().clone()}

    def load(self, st):
        load_lora_state(self.mt.m.lora_layers, st["lora"])
        self.mt.heads["assoc"].load_state_dict(st["enc"]); self.mt.heads.to(self.mt.m.device)
        self.mt.m.readout.data.copy_(st["readout"].to(self.mt.m.device))

    def params(self):
        return [p for l in self.mt.m.lora_layers for p in l.parameters() if p.requires_grad] + \
               list(self.mt.heads["assoc"].parameters()) + [self.mt.m.readout]

    def loss(self, sc, mid, text):
        z = torch.tensor(zscore(sc["util"][mid]), device=self.mt.m.device)
        v = self.mt.assoc(sc, text)
        return F.kl_div(F.log_softmax(v, 0), torch.softmax(z / 0.5, 0), reduction="sum")

    def train(self, case_list, steps, rng, lr=2e-4):
        opt = torch.optim.AdamW(self.params(), lr=lr)
        for _ in range(steps):
            sc, mid, text = case_list[int(rng.integers(len(case_list)))]
            loss = self.loss(sc, mid, text)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params(), 1.0); opt.step()

    @torch.no_grad()
    def regret(self, case_list):
        rs = []
        for sc, mid, text in case_list:
            s = self.mt.assoc(sc, text).cpu().numpy(); u = sc["util"][mid]
            rs.append(float(u.max() - u[int(np.argmax(s))]) / max(float(u.max() - u.min()), 1e-9))
        return float(np.mean(rs))

    @torch.no_grad()
    def case_losses(self, case_list):
        return [float(self.loss(*c)) for c in case_list]


BASIC_LAYERS = 6   # of the 8 adapted layers


def split_state(st):
    basic = {k: v for k, v in st["lora"].items() if int(k.split(".")[0]) < BASIC_LAYERS}
    pers = {k: v for k, v in st["lora"].items() if int(k.split(".")[0]) >= BASIC_LAYERS}
    return basic, pers


def merge(states, weights, part):
    out = copy.deepcopy(states[0])
    total = sum(weights)
    keys = [k for k in out["lora"] if part == "all" or (int(k.split(".")[0]) < BASIC_LAYERS)]
    for k in keys:
        out["lora"][k] = sum(s["lora"][k] * (w / total) for s, w in zip(states, weights))
    for k in out["enc"]:
        out["enc"][k] = sum(s["enc"][k] * (w / total) for s, w in zip(states, weights))
    if part == "all":
        out["readout"] = sum(s["readout"] * (w / total) for s, w in zip(states, weights))
    return out


def with_personal(shared, personal_src):
    out = copy.deepcopy(shared)
    for k, v in personal_src["lora"].items():
        if int(k.split(".")[0]) >= BASIC_LAYERS:
            out["lora"][k] = v.clone()
    out["readout"] = personal_src["readout"].clone()
    return out


def run_arm(bank, data, arm, sites, log, seed=0):
    rng = np.random.default_rng(seed)
    site_state = {s: copy.deepcopy(bank.init) for s in sites}
    live = copy.deepcopy(bank.init)
    sw = SwarmConsensus(site_key=b"e4-site-key", min_ready=2)
    for s in sites:
        node = SwarmNode(s, key=s.encode())
        sw.admit(node, b"c", hmac.new(node.key, b"c", hashlib.sha256).hexdigest())
    pooled_val = [c for s in sites for c in cases(data[s]["train"][-3:], "test")]
    log_rows = []
    for rnd in range(ROUNDS):
        finish = []
        for s in sites:
            bank.load(site_state[s])
            bank.train(cases(data[s]["train"][:-3], "train"), LOCAL_STEPS, rng)
            site_state[s] = bank.snapshot()
            finish.append(s)
        if arm == "local":
            continue
        w = [len(data[s]["train"]) for s in sites]
        if arm == "fedavg_all":
            merged = merge([site_state[s] for s in sites], w, "all")
            for s in sites:
                site_state[s] = copy.deepcopy(merged)
            live = merged
            continue
        candidate = merge([site_state[s] for s in sites], w, "basic")

        def validate(_, cand=candidate):
            scores = []
            for st in (cand, live):
                regs = []
                for s in sites:
                    bank.load(with_personal(st, site_state[s]))
                    regs.append(bank.regret(cases(data[s]["train"][-3:], "test")))
                scores.append(float(np.mean(regs)))
            return {"passed": scores[0] <= scores[1] - 1e-6, "val_regret": scores[0], "live_val_regret": scores[1]}

        res = sw.merge_round({s: (torch.zeros(1), n) for s, n in zip(sites, w)}, finish, validate)
        if res.released:
            live = candidate
            for s in sites:
                site_state[s] = with_personal(live, site_state[s])
        log_rows.append({"round": rnd + 1, "released": res.released, **res.validation})
        log(f"{arm} round {rnd + 1}: released {res.released} {res.validation}")
    per_site = {}
    for s in sites:
        bank.load(site_state[s])
        per_site[s] = bank.regret(cases(data[s]["test"], "test"))
    log(f"{arm}: per-site test regret {per_site}")
    return {"per_site": per_site, "mean": float(np.mean(list(per_site.values()))),
            "worst": float(max(per_site.values())), "rounds": log_rows,
            "log_verifies": sw.verify_log()}, site_state, live


def mia(bank, data, site, state):
    bank.load(state)
    mem = bank.case_losses(cases(data[site]["train"][:-3], "train")[:60])
    non = bank.case_losses(cases(data[site]["test"], "train")[:60])
    return auc([-x for x in mem], [-x for x in non])


def main():
    from ..policy import WirelessJEV
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    data = build_sites(log)
    bank = Bank(WirelessJEV())
    sites = list(SITES)
    res = {"init_per_site": {}}
    bank.load(bank.init)
    for s in sites:
        res["init_per_site"][s] = bank.regret(cases(data[s]["test"], "test"))
    log(f"E3 model before any site training: {res['init_per_site']}")
    for arm in ("local", "fedavg_all", "swarm_pers"):
        r, states, live = run_arm(bank, data, arm, sites, log)
        res[arm] = r
        if arm == "swarm_pers":
            pers_states, pers_live = states, live

    # ---------------- E5: forget a site ----------------
    keep = [s for s in sites if s != FORGET]
    t0 = time.time()
    exact, exact_states, exact_live = run_arm(bank, data, "swarm_pers", keep, log, seed=1)
    exact_cost = time.time() - t0
    # gated distillation: reset student, teacher = current swarm model, retained data only
    t0 = time.time()
    teacher_scores = {}
    retained = [c for s in keep for c in cases(data[s]["train"][:-3], "train")]
    bank.load(with_personal(pers_live, pers_states[keep[0]]))
    rng = np.random.default_rng(3)
    sample = [retained[int(rng.integers(len(retained)))] for _ in range(LOCAL_STEPS * len(keep))]
    with torch.no_grad():
        for i, (sc, mid, text) in enumerate(sample):
            teacher_scores[i] = torch.softmax(bank.mt.assoc(sc, text), 0).detach()
    bank.load(bank.init)
    opt = torch.optim.AdamW(bank.params(), lr=2e-4)
    for i, (sc, mid, text) in enumerate(sample):
        v = bank.mt.assoc(sc, text)
        loss = F.kl_div(F.log_softmax(v, 0), teacher_scores[i], reduction="sum")
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    distilled = bank.snapshot()
    distill_cost = time.time() - t0
    res["e5"] = {
        "forgotten_site": FORGET,
        "mia_auc_before": mia(bank, data, FORGET, with_personal(pers_live, pers_states[FORGET])),
        "mia_auc_exact_retrain": mia(bank, data, FORGET, exact_live),
        "mia_auc_gated_distill": mia(bank, data, FORGET, distilled),
        "mia_auc_never_trained": mia(bank, data, FORGET, bank.init),
        "retained_regret_before": float(np.mean([res["swarm_pers"]["per_site"][s] for s in keep])),
        "retained_regret_exact": float(np.mean([exact["per_site"][s] for s in keep])),
        "retained_regret_distill": float(np.mean([
            (bank.load(with_personal(distilled, pers_states[s])) or bank.regret(cases(data[s]["test"], "test")))
            for s in keep])),
        "cost_s_exact": exact_cost, "cost_s_distill": distill_cost,
    }
    log(f"E5: {res['e5']}")
    if SMOKE:
        log("smoke ok"); return
    (OUT / "e45.json").write_text(json.dumps(res, indent=2, default=float))
    log("saved e45.json")


if __name__ == "__main__":
    main()
