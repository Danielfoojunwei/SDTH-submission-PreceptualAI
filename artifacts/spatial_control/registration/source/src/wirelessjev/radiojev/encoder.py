"""Radio encoder: plan graph -> T tokens in the decision model's embedding space.

Robots and BSSIDs are nodes; each robot is joined to the BSSID it uses in the
plan. One round of robot -> BSSID -> robot message passing (the graph-state
idea of Li, Yang, Lam, Shen & Wei's joint GNN + DQN spectrum access), then
learned-query attention pooling into a fixed set of tokens:
4 robot tokens, 2 BSSID tokens, 1 named-robot token, 1 global-terms token.
Tokens are layer-normalised and scaled to the backbone's embedding RMS so they
enter the network like ordinary token embeddings.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .features import BSS_F, GLOBAL_F, ROBOT_F, ROBOT_F_POS, PlanGraph

N_TOKENS = 8


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class RadioEncoder(nn.Module):
    def __init__(self, d_model: int = 1024, hidden: int = 128, embed_rms: float = 0.02,
                 robot_f: int = ROBOT_F):
        super().__init__()
        H = hidden
        self.robot_in = mlp(robot_f, H, H)
        self.bss_in = mlp(BSS_F, H, H)
        self.r2b = mlp(H, H, H)
        self.b2r = mlp(2 * H, H, H)
        self.glob_in = mlp(GLOBAL_F, H, H)
        self.q_robot = nn.Parameter(torch.randn(4, H) / H ** 0.5)
        self.q_bss = nn.Parameter(torch.randn(2, H) / H ** 0.5)
        self.out = nn.Linear(H, d_model)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.scale = nn.Parameter(torch.full((d_model,), float(embed_rms)))

    @staticmethod
    def batch(graphs: list[PlanGraph], named: np.ndarray, device) -> dict:
        robot = np.stack([g.robot for g in graphs]).copy()
        robot[:, :, 8] = named[None, :]
        glob = np.stack([g.glob for g in graphs]).copy()
        w = named.sum()
        glob[:, 9] = ((robot[:, :, 3] * named[None, :]).sum(1) / w) if w > 0 else 1.0
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=device)
        return {"robot": t(robot), "bss": t(np.stack([g.bss for g in graphs])),
                "assign": t(np.stack([g.assign for g in graphs]), torch.long),
                "glob": t(glob), "named": t(named)}

    def pool(self, q, x):
        att = torch.softmax(q @ x.transpose(1, 2) / x.shape[-1] ** 0.5, dim=-1)   # (K,Q,N)
        return att @ x

    def forward(self, b: dict) -> torch.Tensor:
        r = self.robot_in(b["robot"])                      # (K, n, H)
        bs = self.bss_in(b["bss"])                         # (K, m, H)
        K, n, H = r.shape
        m = bs.shape[1]
        idx = b["assign"].unsqueeze(-1).expand(-1, -1, H)
        agg = torch.zeros(K, m, H, device=r.device).scatter_add_(1, idx, r)
        cnt = torch.zeros(K, m, 1, device=r.device).scatter_add_(
            1, b["assign"].unsqueeze(-1), torch.ones(K, n, 1, device=r.device))
        bs = bs + self.r2b(agg / cnt.clamp(min=1))
        r = r + self.b2r(torch.cat([r, torch.gather(bs, 1, idx)], -1))
        named = b["named"].view(1, n, 1)
        named_tok = (r * named).sum(1, keepdim=True) / named.sum().clamp(min=1)
        toks = torch.cat([
            self.pool(self.q_robot.expand(K, -1, -1), r),
            self.pool(self.q_bss.expand(K, -1, -1), bs),
            named_tok,
            self.glob_in(b["glob"]).unsqueeze(1)], dim=1)      # (K, 8, H)
        return self.norm(self.out(toks)) * self.scale


class NoLLMScorer(nn.Module):
    """Ablation: the same encoder pooled into a score, conditioned on a
    mission id embedding instead of mission text (cannot read new missions)."""

    def __init__(self, n_missions: int, hidden: int = 128):
        super().__init__()
        self.enc = RadioEncoder(d_model=hidden, hidden=hidden, embed_rms=1.0)
        self.mission = nn.Embedding(n_missions, hidden)
        self.head = mlp(2 * hidden, hidden, 1)

    def forward(self, b: dict, mission_idx: int) -> torch.Tensor:
        toks = self.enc(b).mean(1)                          # (K, H)
        mvec = self.mission.weight[mission_idx].expand_as(toks)
        return self.head(torch.cat([toks, mvec], -1)).squeeze(-1)


def widen_from(enc_small_state: dict, enc: "RadioEncoder") -> None:
    """Load an encoder trained on ROBOT_F inputs into one with extra robot
    inputs: old columns copied, new columns start at zero."""
    state = dict(enc_small_state)
    w_old = state["robot_in.0.weight"]
    w_new = enc.robot_in[0].weight.detach().clone().zero_()
    w_new[:, : w_old.shape[1]] = w_old
    state["robot_in.0.weight"] = w_new
    enc.load_state_dict(state)
