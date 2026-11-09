"""Readout-only distillation from numerical teachers (backbone frozen).

The teacher is never another language model: for fleet decisions it is the
exact oracle (soft labels softmax(utility / tau) over the options); for
traffic labels it is the ground truth of the synthetic generator. Only the
first k rows of d3-lite's 255 x D answer-code readout receive gradients; an
L2 anchor keeps them near the released weights, and the best epoch on a
held-out split is kept.

Also here:
- membership-inference audit (loss-threshold AUC) used by Li, Shui, Lam, Shen
  & Wang (ICAIIC 2025) to measure what a model leaks about its training data;
- region unlearning: re-distil from the released readout without the
  region's cases (the exact counterpart of their gated-distillation
  unlearning; we can afford the exact version because only a head is fitted).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .policy import MODEL_REPO, MODEL_REVISION


@dataclass
class Example:
    case_id: str
    region: str
    split: str                 # train | validation | test
    hidden: torch.Tensor       # [D] final-norm state, frozen backbone
    target: torch.Tensor       # [k] teacher distribution
    option_ids: list[str] = field(default_factory=list)


def soft_target(scores: dict[str, float], order: list[str], tau: float) -> torch.Tensor:
    vals = torch.tensor([scores[k] for k in order], dtype=torch.float32)
    return torch.softmax(vals / tau, 0)


def _logits(weight: torch.Tensor, hidden: torch.Tensor, k: int,
            temperature: float) -> torch.Tensor:
    return (weight[:k] @ hidden) / temperature


def example_loss(weight, ex: Example, temperature: float) -> torch.Tensor:
    logp = F.log_softmax(_logits(weight, ex.hidden, len(ex.target), temperature), 0)
    return F.kl_div(logp, ex.target.clamp_min(1e-8), reduction="sum")


def distill(base_readout: torch.Tensor, examples: list[Example], *,
            temperature: float = 1.0, epochs: int = 60, lr: float = 2e-3,
            anchor: float = 1e-2, seed: int = 7) -> tuple[torch.Tensor, dict]:
    train = [e for e in examples if e.split == "train"]
    val = [e for e in examples if e.split == "validation"]
    if not train or not val:
        raise ValueError("TRAIN_AND_VALIDATION_REQUIRED")
    base = torch.tensor(base_readout.detach().float().cpu().numpy().copy())
    weight = torch.nn.Parameter(base.clone())
    opt = torch.optim.AdamW([weight], lr=lr, weight_decay=0.0)
    rng = random.Random(seed)

    def val_loss(w):
        with torch.no_grad():
            return sum(float(example_loss(w, e, temperature)) for e in val) / len(val)

    before = val_loss(weight)
    best_val, best = before, base.clone()
    history = []
    for epoch in range(epochs):
        rng.shuffle(train)
        for e in train:
            opt.zero_grad(set_to_none=True)
            loss = example_loss(weight, e, temperature) + anchor * (weight - base).square().sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([weight], 1.0)
            opt.step()
        v = val_loss(weight)
        history.append(round(v, 5))
        if math.isfinite(v) and v < best_val:
            best_val, best = v, weight.detach().clone()
    return best, {"val_kl_before": before, "val_kl_after": best_val,
                  "epochs": epochs, "history": history,
                  "changed": not torch.equal(best, base)}


def save_adapter(path: str | Path, readout: torch.Tensor, *, task: str,
                 info: dict, dataset_digest: str) -> str:
    from safetensors.torch import save_file
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    weights = path / "readout.safetensors"
    save_file({"readout": readout.contiguous()}, str(weights))
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest = {
        "schema": "wirelessjev.readout.v1",
        "base": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "task": task,
        "weights_sha256": digest,
        "dataset_sha256": dataset_digest,
        "backbone_updated": False,
        "teacher": "numerical oracle (no language-model teacher)",
        "val_kl_before": info["val_kl_before"],
        "val_kl_after": info["val_kl_after"],
        "deployment_approved": False,
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return digest


def dataset_digest(examples: list[Example]) -> str:
    h = hashlib.sha256()
    for e in sorted(examples, key=lambda e: e.case_id):
        h.update(e.case_id.encode())
        h.update(e.split.encode())
        h.update(e.target.numpy().tobytes())
    return h.hexdigest()


# --- privacy audit and unlearning ----------------------------------------
def auc(pos: list[float], neg: list[float]) -> float:
    """P(score_pos > score_neg), ties count half."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def membership_auc(weight: torch.Tensor, members: list[Example],
                   non_members: list[Example], temperature: float = 1.0) -> float:
    """Loss-threshold membership inference: lower loss => 'member'."""
    with torch.no_grad():
        m = [-float(example_loss(weight, e, temperature)) for e in members]
        n = [-float(example_loss(weight, e, temperature)) for e in non_members]
    return auc(m, n)


def unlearn_region(base_readout: torch.Tensor, examples: list[Example],
                   region: str, **kwargs) -> tuple[torch.Tensor, dict]:
    kept = [e for e in examples if e.region != region]
    return distill(base_readout, kept, **kwargs)


def accuracy(weight: torch.Tensor, examples: list[Example],
             temperature: float = 1.0) -> float:
    with torch.no_grad():
        hits = [int(torch.argmax(_logits(weight, e.hidden, len(e.target), temperature))
                    == torch.argmax(e.target)) for e in examples]
    return float(np.mean(hits)) if hits else float("nan")


def choices(weight: torch.Tensor, examples: list[Example],
            temperature: float = 1.0) -> list[str]:
    with torch.no_grad():
        return [e.option_ids[int(torch.argmax(_logits(weight, e.hidden,
                                                      len(e.target), temperature)))]
                for e in examples]


# --- listwise (option-wise) distillation ------------------------------------
# For option-wise scoring every option has its own hidden state h_i and the
# model's P(true) is sigmoid((r_true - r_false) . h_i / T). We learn one shared
# vector u added to that direction (r_true += u/2, r_false -= u/2) with a
# listwise KL to the oracle's soft target over the options. Position-free:
# u never sees option order.
def list_logits(readout: torch.Tensor, true_row: int, hidden: torch.Tensor,
                temperature: float, u: torch.Tensor | None = None) -> torch.Tensor:
    diff = readout[true_row] - readout[1 - true_row]
    if u is not None:
        diff = diff + u
    return (hidden @ diff) / temperature


def list_loss(readout, true_row, ex: Example, temperature, u=None):
    logp = F.log_softmax(list_logits(readout, true_row, ex.hidden, temperature, u), 0)
    return F.kl_div(logp, ex.target.clamp_min(1e-8), reduction="sum")


def distill_listwise(base_readout: torch.Tensor, true_row: int,
                     examples: list[Example], *, temperature: float = 1.0,
                     epochs: int = 80, lr: float = 3e-3, anchor: float = 1e-3,
                     seed: int = 7) -> tuple[torch.Tensor, dict]:
    train = [e for e in examples if e.split == "train"]
    val = [e for e in examples if e.split == "validation"]
    if not train or not val:
        raise ValueError("TRAIN_AND_VALIDATION_REQUIRED")
    base = torch.tensor(base_readout.detach().float().cpu().numpy().copy())
    u = torch.nn.Parameter(torch.zeros(base.shape[1]))
    opt = torch.optim.AdamW([u], lr=lr, weight_decay=0.0)
    rng = random.Random(seed)

    def val_loss(vec):
        with torch.no_grad():
            return sum(float(list_loss(base, true_row, e, temperature, vec))
                       for e in val) / len(val)

    before = val_loss(None)
    best_val, best_u = before, torch.zeros_like(u)
    history = []
    for _ in range(epochs):
        rng.shuffle(train)
        for e in train:
            opt.zero_grad(set_to_none=True)
            loss = list_loss(base, true_row, e, temperature, u) + anchor * u.square().sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([u], 1.0)
            opt.step()
        v = val_loss(u.detach())
        history.append(round(v, 5))
        if math.isfinite(v) and v < best_val:
            best_val, best_u = v, u.detach().clone()
    readout = base.clone()
    readout[true_row] += best_u / 2
    readout[1 - true_row] -= best_u / 2
    return readout, {"val_kl_before": before, "val_kl_after": best_val,
                     "epochs": epochs, "history": history,
                     "changed": bool(best_u.abs().sum() > 0),
                     "u_norm": float(best_u.norm())}


def list_choices(readout, true_row, examples, temperature=1.0) -> list[str]:
    with torch.no_grad():
        return [e.option_ids[int(torch.argmax(list_logits(readout, true_row, e.hidden,
                                                          temperature)))]
                for e in examples]


def list_membership_auc(readout, true_row, members, non_members, temperature=1.0):
    with torch.no_grad():
        m = [-float(list_loss(readout, true_row, e, temperature)) for e in members]
        n = [-float(list_loss(readout, true_row, e, temperature)) for e in non_members]
    return auc(m, n)
