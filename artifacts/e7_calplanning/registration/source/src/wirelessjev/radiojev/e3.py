"""E3: one RadioJEV model for several radio tasks from the Li/Shen/Lam line.

    uv run python -m wirelessjev.radiojev.e3

Shared: the d3-lite backbone with one LoRA (warm-started from the E2
checkpoint) and the readout rows. Per task: a small modality encoder.
  dsa        channel access, typed Q over channels, double-DQN in DSAEnv
             (vs the papers' DDQN on an MLP, same environment steps)
  admission  raise / hold / lower the admission level (Shen, Lam & Li price
             step), supervised by the exact oracle (vs the demand rule)
  traffic    trend / event / volatility labels (Chen, Lam & Li), supervised
             by generator truth (vs fixed-threshold rules)
  assoc      non-linear association missions from E2 (keeps E2 data mixed in
             so the shared model does not forget it)
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .. import traffic as T
from ..oracle import evaluate
from ..teachers import admission_oracle
from ..teachers.admission import PRICE_LEVELS, PRIORITY
from ..twin.warehouse import make_warehouse
from .dsa import DSAEnv
from .e1 import INSTR, SMOKE, named_vec, state_text
from .e2 import cases as e2_cases, zscore
from .encoder import RadioEncoder, mlp, widen_from
from .features import ROBOT_F_POS
from .trainable import RADIO_TOKEN, TrainableD3, load_lora_state, lora_state

OUT = Path("artifacts/radiojev")
H = 128


class TokenHead(nn.Module):
    """Set/sequence of feature vectors -> tokens in the model's embedding space."""

    def __init__(self, f_in: int, n_tokens: int | None, d_model: int, rms: float, per_item=False):
        super().__init__()
        self.inp = mlp(f_in, H, H)
        self.per_item, self.n_tokens = per_item, n_tokens
        if not per_item:
            self.q = nn.Parameter(torch.randn(n_tokens, H) / H ** 0.5)
        self.glob = nn.Parameter(torch.randn(1, H) / H ** 0.5)
        self.out = nn.Linear(H, d_model)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.scale = nn.Parameter(torch.full((d_model,), rms))

    def forward(self, x):                      # x: (B, N, f_in)
        h = self.inp(x)
        g = torch.softmax(self.glob @ h.transpose(1, 2) / H ** 0.5, -1) @ h      # (B,1,H)
        if self.per_item:
            toks = torch.cat([h + g, g], 1)
        else:
            toks = torch.cat([torch.softmax(self.q @ h.transpose(1, 2) / H ** 0.5, -1) @ h, g], 1)
        return self.norm(self.out(toks)) * self.scale


class SeriesHead(nn.Module):
    def __init__(self, d_model: int, rms: float, n_tokens: int = 4):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 32, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(32, H, 5, padding=2, stride=2), nn.GELU())
        self.q = nn.Parameter(torch.randn(n_tokens, H) / H ** 0.5)
        self.out = nn.Linear(H, d_model)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.scale = nn.Parameter(torch.full((d_model,), rms))

    def forward(self, x):                      # (B, L) normalised series
        h = self.conv(x.unsqueeze(1)).transpose(1, 2)                            # (B, L', H)
        toks = torch.softmax(self.q @ h.transpose(1, 2) / H ** 0.5, -1) @ h
        return self.norm(self.out(toks)) * self.scale


# ------------------------------------------------------------------ tasks
DSA_TEXT = ("Robot radio channel selection. Each channel's recent sensing history is "
            "attached. Pick the channel most likely to be free of protected users and "
            "other robots in the next slot.")


def dsa_question(C):
    instr = DSA_TEXT + " Channels: " + RADIO_TOKEN * (C + 1)
    crit = {f"ch{c + 1}": f"channel {c + 1}" for c in range(C)}
    return instr, crit


def admission_sample(seed):
    scene = make_warehouse(seed)
    assoc = {c.car_id: c.current_ap for c in scene.cars}
    ch = {a.ap_id: a.channel for a in scene.aps}
    ev = evaluate(scene, {"assoc": assoc, "channels": ch})
    doms = scene.domains(ch)
    d = max(range(len(doms)), key=lambda k: ev.domain_load[k])
    dom = doms[d]
    ap = scene.ap(next(iter(dom)))
    cap = 60.0 * (1 - scene.occupancy(ap, ch))
    level = int(np.random.default_rng(seed).integers(1, len(PRICE_LEVELS) - 1))
    o = admission_oracle(scene, assoc, ev.sinr_db, dom, cap, level)
    flows = []
    for c in scene.cars:
        if assoc[c.car_id] in dom:
            for f in c.flows:
                if not f.critical:
                    flows.append([PRIORITY.get(f.name, 1.0) / 5, f.demand_mbps / 50,
                                  max(ev.sinr_db[c.car_id], 0) / 40,
                                  PRIORITY.get(f.name, 1.0) * max(ev.sinr_db[c.car_id], 0) / 10 / 6,
                                  cap / 60, level / 12])
    if not flows:
        return None
    return {"x": np.array(flows, np.float32), "label": ["lower", "hold", "raise"].index(o["local_direction"]),
            "cap": cap, "level": level,
            "demand": float(sum(f[1] * 50 for f in flows))}


ADM_TEXT = ("Congested Wi-Fi channel. The best-effort flows competing for it are attached "
            "(priority, demand, signal, bid, capacity, current admission level). "
            "Which admission move delivers the most priority-weighted traffic? " )
ADM_CRIT = {"lower": "lower the level: admit more flows", "hold": "keep the current level",
            "raise": "raise the level: shed low-priority flows"}
TRAFFIC_Q = {k: (q["instructions"], list(q["criteria"])) for k, q in T.QUESTIONS.items()}


def norm_series(h):
    h = np.asarray(h, np.float32)
    return (h - np.median(h)) / max(float(np.median(h)), 1e-3)


# ------------------------------------------------------------------ model
class MultiTask:
    def __init__(self, engine, init_ckpt: Path | None):
        self.m = TrainableD3(engine, lora_layers=8, readout_rows=20)
        rms, D, dev = self.m.embed_rms, 1024, self.m.device
        self.heads = nn.ModuleDict({
            "assoc": RadioEncoder(embed_rms=rms, robot_f=ROBOT_F_POS),
            "dsa": TokenHead(16, None, D, rms, per_item=True),
            "admission": TokenHead(6, 4, D, rms),
            "traffic": SeriesHead(D, rms)}).to(dev)
        if init_ckpt and init_ckpt.exists():
            ck = torch.load(init_ckpt)
            if "robot_in.0.weight" in ck["encoder"] and ck["encoder"]["robot_in.0.weight"].shape[1] == ROBOT_F_POS:
                self.heads["assoc"].load_state_dict(ck["encoder"])
            else:
                widen_from(ck["encoder"], self.heads["assoc"])
            self.heads.to(dev)
            load_lora_state(self.m.lora_layers, ck["lora"])
            self.m.readout.data[:2] = ck["readout"].to(dev)

    def params(self):
        return list(self.heads.parameters()) + self.m.trainable()

    # each returns logits over the task's options
    def assoc(self, sc, text):
        b = RadioEncoder.batch(sc["graphs"], named_vec(sc["car_ids"], text), self.m.device)
        h = self.m.encode(state_text(text), [INSTR] * len(sc["graphs"]), self.heads["assoc"](b))
        return self.m.noul_logit(h)

    def dsa(self, states: np.ndarray):          # (B, C, 2H)
        C = states.shape[1]
        instr, crit = dsa_question(C)
        toks = self.heads["dsa"](torch.as_tensor(states, device=self.m.device))
        h = self.m.encode("Spectrum access for a warehouse robot.", [instr] * len(states),
                          toks, kind="choice", criteria=crit)
        return self.m.choice_logits(h, C)

    def admission(self, sample):
        toks = self.heads["admission"](torch.as_tensor(sample["x"][None], device=self.m.device))
        h = self.m.encode("Admission control on a congested warehouse Wi-Fi channel.",
                          [ADM_TEXT + RADIO_TOKEN * 5], toks, kind="choice", criteria=ADM_CRIT)
        return self.m.choice_logits(h, 3)[0]

    def traffic(self, series, key):
        instr, opts = TRAFFIC_Q[key]
        toks = self.heads["traffic"](torch.as_tensor(norm_series(series)[None], device=self.m.device))
        h = self.m.encode("A robot's recent uplink throughput is attached.",
                          [instr + " Series: " + RADIO_TOKEN * 4], toks, kind="choice",
                          criteria={o: T.QUESTIONS[key]["criteria"][o] for o in opts})
        return self.m.choice_logits(h, len(opts))[0]


class MLPQ(nn.Module):
    """The papers' DDQN agent (MLP on the flattened sensing state)."""

    def __init__(self, C, f):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(C * f, 128), nn.ReLU(), nn.Linear(128, 128),
                                 nn.ReLU(), nn.Linear(128, C))

    def forward(self, s):
        return self.net(s.flatten(1))


def run_ddqn(qfn, params, target_fn, steps, env_seed, log, name, batch=8, gamma=0.5,
             lr=1e-3, sync=50, eps=(1.0, 0.05)):
    """Generic double-DQN loop. qfn(states)->Q (grad), target_fn(states, online)->Q (no grad)."""
    env = DSAEnv(seed=env_seed)
    opt = torch.optim.AdamW(params, lr=lr)
    buf, s = [], env.state()
    rewards, oracle_gap, t0 = [], [], time.time()
    for t in range(steps):
        e = eps[0] + (eps[1] - eps[0]) * min(1.0, t / (0.6 * steps))
        with torch.no_grad():
            q = qfn(s[None]).squeeze(0).float().cpu().numpy()
        a = int(np.random.randint(env.C)) if np.random.rand() < e else int(np.argmax(q))
        ov = env.oracle_values()
        oracle_gap.append(float(ov.max() - ov[a]))
        r, s2 = env.step(a)
        rewards.append(r)
        buf.append((s, a, r, s2))
        s = s2
        if len(buf) >= batch:
            idx = np.random.choice(len(buf), batch, replace=False)
            S = np.stack([buf[i][0] for i in idx]); A = torch.tensor([buf[i][1] for i in idx])
            R = torch.tensor([buf[i][2] for i in idx], dtype=torch.float32)
            S2 = np.stack([buf[i][3] for i in idx])
            with torch.no_grad():
                a2 = qfn(S2).argmax(1).cpu()                       # online selects
                q2 = target_fn(S2).float().cpu()                   # target evaluates
                y = R + gamma * q2.gather(1, a2[:, None]).squeeze(1)
            qs = qfn(S).float().cpu().gather(1, A[:, None]).squeeze(1)
            loss = F.smooth_l1_loss(qs, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if t % sync == 0 and hasattr(target_fn, "sync"):
            target_fn.sync()
        if (t + 1) % 250 == 0:
            log(f"{name}: step {t + 1} mean reward(last 250) {np.mean(rewards[-250:]):.3f} "
                f"oracle gap {np.mean(oracle_gap[-250:]):.3f} ({time.time() - t0:.0f} s)")
    return {"mean_reward_last500": float(np.mean(rewards[-500:])),
            "oracle_gap_last500": float(np.mean(oracle_gap[-500:])),
            "curve": [float(np.mean(rewards[i:i + 250])) for i in range(0, steps, 250)]}


def greedy_eval(qfn, env_seed, steps=None):
    steps = steps or (10 if SMOKE else 500)
    env = DSAEnv(seed=env_seed + 999)
    s, rs, gaps = env.state(), [], []
    for _ in range(steps):
        with torch.no_grad():
            a = int(qfn(s[None]).squeeze(0).float().argmax())
        ov = env.oracle_values(); gaps.append(float(ov.max() - ov[a]))
        r, s = env.step(a); rs.append(r)
    return {"greedy_mean_reward": float(np.mean(rs)), "greedy_oracle_gap": float(np.mean(gaps))}


def main():
    from ..policy import WirelessJEV
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    rng = np.random.default_rng(0)
    res = {}

    # ---- baselines first (cheap) ----
    env0 = DSAEnv(seed=7)
    C, f = env0.C, env0.state().shape[1]
    ref = {}
    for steps in ((30,) if SMOKE else (1500, 20000)):
        net = MLPQ(C, f); tgt = copy.deepcopy(net)

        class Tgt:
            def __call__(self, S): return tgt(torch.as_tensor(S))
            def sync(self): tgt.load_state_dict(net.state_dict())
        out = run_ddqn(lambda S: net(torch.as_tensor(S)), list(net.parameters()), Tgt(), steps, 7, log,
                       f"ddqn_mlp_{steps}")
        out.update(greedy_eval(lambda S: net(torch.as_tensor(S)), 7))
        ref[f"ddqn_mlp_{steps}_steps"] = out
    rnd = DSAEnv(seed=7 + 999)
    ref["random"] = {"greedy_mean_reward": float(np.mean([rnd.step(int(np.random.randint(C)))[0] for _ in range(500)]))}
    orc = DSAEnv(seed=7 + 999)
    ref["oracle"] = {"greedy_mean_reward": float(np.mean([orc.step(int(np.argmax(orc.oracle_values())))[0]
                                                          for _ in range(500)]))}
    res["dsa_baselines"] = ref
    log(f"DSA baselines: { {k: v.get('greedy_mean_reward') for k, v in ref.items()} }")

    adm = [x for x in (admission_sample(60000 + i) for i in range(12 if SMOKE else 160)) if x is not None]
    cut = len(adm) * 2 // 3
    adm_tr, adm_te = adm[:cut], adm[cut:]
    rule = lambda x: "raise" if x["demand"] > x["cap"] else ("lower" if x["demand"] < 0.6 * x["cap"] else "hold")
    res["admission_rule_acc"] = float(np.mean([["lower", "hold", "raise"].index(rule(x)) == x["label"] for x in adm_te]))
    trng = np.random.default_rng(11)
    wins = [T.synth_window(trng) for _ in range(12 if SMOKE else 260)]
    tr_tr, tr_te = wins[: len(wins) * 3 // 4], wins[len(wins) * 3 // 4:]
    res["traffic_rule_acc"] = {k: float(np.mean([T.rule_labels(w)[k]["choice"] == getattr(w, k) for w in tr_te]))
                               for k in T.QUESTIONS}
    log(f"admission rule {res['admission_rule_acc']:.3f}; traffic rules {res['traffic_rule_acc']}")

    # ---- the shared model ----
    engine = WirelessJEV()
    mt = MultiTask(engine, OUT / "ckpt_radio_q.pt")
    params = mt.params()
    opt = torch.optim.AdamW(params, lr=2e-4)
    e2 = pickle.loads((OUT / "e2_data.pkl").read_bytes())
    assoc_train = e2_cases(e2, "train")

    def assoc_eval():
        rs = []
        for sc, mid, text in e2_cases(e2, "test")[: 3 if SMOKE else 120]:
            with torch.no_grad():
                s = mt.assoc(sc, text).cpu().numpy()
            u = sc["util"][mid]
            rs.append(float(u.max() - u[int(np.argmax(s))]) / max(float(u.max() - u.min()), 1e-9))
        return float(np.mean(rs))

    def sup_eval():
        with torch.no_grad():
            a = np.mean([int(mt.admission(x).argmax()) == x["label"] for x in adm_te])
            t = {k: float(np.mean([int(mt.traffic(w.history, k).argmax()) == list(T.QUESTIONS[k]["criteria"]).index(getattr(w, k))
                                   for w in tr_te])) for k in T.QUESTIONS}
        return float(a), t

    res["shared_before"] = {"assoc_norm_regret": assoc_eval(), "admission_acc": sup_eval()[0],
                            "traffic_acc": sup_eval()[1]}
    log(f"shared model before E3 training: {res['shared_before']}")

    # supervised multitask phase: admission + traffic + association replay
    t0 = time.time()
    for step in range(6 if SMOKE else 600):
        task = ["admission", "traffic", "assoc"][step % 3]
        if task == "admission":
            x = adm_tr[int(rng.integers(len(adm_tr)))]
            loss = F.cross_entropy(mt.admission(x)[None], torch.tensor([x["label"]], device=mt.m.device))
        elif task == "traffic":
            w = tr_tr[int(rng.integers(len(tr_tr)))]
            k = list(T.QUESTIONS)[int(rng.integers(3))]
            lab = list(T.QUESTIONS[k]["criteria"]).index(getattr(w, k))
            loss = F.cross_entropy(mt.traffic(w.history, k)[None], torch.tensor([lab], device=mt.m.device))
        else:
            sc, mid, text = assoc_train[int(rng.integers(len(assoc_train)))]
            z = torch.tensor(zscore(sc["util"][mid]), device=mt.m.device)
            v = mt.assoc(sc, text)
            loss = F.kl_div(F.log_softmax(v, 0), torch.softmax(z / 0.5, 0), reduction="sum")
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if (step + 1) % 150 == 0:
            log(f"supervised step {step + 1} ({time.time() - t0:.0f} s)")

    # RL phase: typed Q over channels, double DQN, with association replay to avoid forgetting
    lora_target = lora_state(mt.m.lora_layers)
    head_target = copy.deepcopy(mt.heads["dsa"].state_dict())
    readout_target = mt.m.readout.detach().clone()

    class Target:
        def __call__(self, S):
            cur = (lora_state(mt.m.lora_layers), copy.deepcopy(mt.heads["dsa"].state_dict()),
                   mt.m.readout.detach().clone())
            load_lora_state(mt.m.lora_layers, lora_target)
            mt.heads["dsa"].load_state_dict(head_target)
            mt.m.readout.data.copy_(readout_target)
            try:
                return mt.dsa(S)
            finally:
                load_lora_state(mt.m.lora_layers, cur[0]); mt.heads["dsa"].load_state_dict(cur[1])
                mt.m.readout.data.copy_(cur[2])

        def sync(self):
            nonlocal lora_target, head_target, readout_target
            lora_target = lora_state(mt.m.lora_layers)
            head_target = copy.deepcopy(mt.heads["dsa"].state_dict())
            readout_target = mt.m.readout.detach().clone()

    res["dsa_radiojev"] = run_ddqn(mt.dsa, params, Target(), 12 if SMOKE else 1500, 7, log, "radiojev_typedQ",
                                   batch=4, lr=1e-4, sync=50)
    res["dsa_radiojev"].update(greedy_eval(mt.dsa, 7))

    res["shared_after"] = {"assoc_norm_regret": assoc_eval(), "admission_acc": sup_eval()[0],
                           "traffic_acc": sup_eval()[1]}
    log(f"shared model after E3: {res['shared_after']}")
    if SMOKE:
        log("smoke ok"); return
    torch.save({"heads": {k: v.cpu() for k, v in mt.heads.state_dict().items()},
                "lora": lora_state(mt.m.lora_layers), "readout": mt.m.readout.detach().cpu()},
               OUT / "ckpt_e3.pt")
    (OUT / "e3.json").write_text(json.dumps(res, indent=2, default=float))
    log("saved e3.json")


if __name__ == "__main__":
    main()
