"""Trainable path through d3-lite.

The released runtime only runs inference. Here we reuse its own pieces so the
pretrained decision behaviour is preserved: `runtime.prepare` renders the d3
prompt and tokenises it, `runtime.encoder` (PromptEncoder) runs the layers with
d3's non-causal attention, and the answer-code readout turns the final hidden
state into option logits. On top of that:

- radio slots: `<|image_pad|>` placeholders (the model's own vision slot) in
  the question text are replaced by learned radio-token embeddings;
- LoRA adapters on the linear layers of the last N decoder layers;
- a trainable copy of the readout rows a task uses.

With LoRA at zero and no radio tokens, outputs equal the runtime's to float
precision (checked by `parity`).
"""
from __future__ import annotations

import math

import torch
from torch import nn

RADIO_TOKEN = "<|image_pad|>"


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))

    def forward(self, x):
        y = self.base(x)
        delta = (x.float() @ self.A.T) @ self.B.T
        return y + (delta * self.scale).to(y.dtype)


def inject_lora(layers, r: int = 8, alpha: float = 16.0) -> list[nn.Parameter]:
    params = []
    for layer in layers:
        targets = [(name, mod) for name, mod in layer.named_modules()
                   if isinstance(mod, nn.Linear)]
        for name, mod in targets:
            parent = layer
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            lora = LoRALinear(mod, r, alpha)
            setattr(parent, parts[-1], lora)
            params += [lora.A, lora.B]
    return params


def lora_state(layers) -> dict[str, torch.Tensor]:
    out = {}
    for li, layer in enumerate(layers):
        for name, mod in layer.named_modules():
            if isinstance(mod, LoRALinear):
                out[f"{li}.{name}.A"] = mod.A.detach().cpu().clone()
                out[f"{li}.{name}.B"] = mod.B.detach().cpu().clone()
    return out


def load_lora_state(layers, state: dict[str, torch.Tensor]) -> None:
    for li, layer in enumerate(layers):
        for name, mod in layer.named_modules():
            if isinstance(mod, LoRALinear):
                mod.A.data.copy_(state[f"{li}.{name}.A"].to(mod.A.device))
                mod.B.data.copy_(state[f"{li}.{name}.B"].to(mod.B.device))


class TrainableD3:
    """Wraps a loaded WirelessJEV engine (its runtime and backbone)."""

    def __init__(self, engine, *, lora_layers: int = 8, r: int = 8,
                 readout_rows: int = 2):
        self.engine = engine
        self.rt = engine.runtime
        self.device = self.rt.readout.device
        self.tok = self.rt.tokenizer
        self.pad_id = self.tok.convert_tokens_to_ids(RADIO_TOKEN)
        self.embed = self.rt.backbone.get_input_embeddings()
        self.temp = float(self.rt.temperature)
        lm = self.rt.backbone.language_model
        self.lora_layers = list(lm.layers[-lora_layers:]) if lora_layers else []
        self.lora_params = inject_lora(self.lora_layers, r) if lora_layers else []
        self.readout = nn.Parameter(self.rt.readout[:readout_rows].detach().float().clone())
        self.bucket = 64
        with torch.no_grad():
            self.embed_rms = float(self.embed.weight.float().pow(2).mean().sqrt())

    # ------------------------------------------------------------------
    def encode(self, state: str, instructions: list[str],
               radio: torch.Tensor | None = None, kind: str = "noul",
               criteria=None) -> torch.Tensor:
        """Final hidden state for each question. radio: (K, T, D) tokens for
        the K questions, each question's text holding exactly T slots."""
        qs = {}
        for k, text in enumerate(instructions):
            q = {"type": kind, "instructions": text}
            if criteria is not None:
                q["criteria"] = criteria
            qs[f"q{k}"] = q
        prepared = self.rt.prepare(state, qs)
        if prepared.errors:
            raise RuntimeError(f"PROMPT_REJECTED:{prepared.errors}")
        seqs = [prepared.sequences[f"q{k}"] for k in range(len(instructions))]
        # Pad to a fixed bucket so tensor shapes repeat: Metal (MPS) compiles a
        # graph per new shape, which otherwise dominates run time.
        L = max(len(s) for s in seqs)
        L = ((L + self.bucket - 1) // self.bucket) * self.bucket
        ids = torch.full((len(seqs), L), self.tok.pad_token_id or 0, dtype=torch.long)
        present = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, L - len(s):] = torch.tensor(s)
            present[i, L - len(s):] = 1
        ids, present = ids.to(self.device), present.to(self.device)
        emb = self.embed(ids)
        if radio is not None:
            slots = (ids == self.pad_id) & present.bool()
            K, T, D = radio.shape
            if int(slots.sum()) != K * T:
                raise RuntimeError("RADIO_SLOT_COUNT_MISMATCH")
            emb = emb.clone()
            emb[slots] = radio.reshape(K * T, D).to(emb.dtype)
        return self.rt.encoder(emb, present)[:, -1].float()

    def noul_logit(self, hidden: torch.Tensor) -> torch.Tensor:
        """log-odds of 'true' (row 1) over 'false' (row 0)."""
        logits = hidden @ self.readout.T / self.temp
        return logits[:, 1] - logits[:, 0]

    def choice_logits(self, hidden: torch.Tensor, k: int) -> torch.Tensor:
        return hidden @ self.readout[:k].T / self.temp

    def trainable(self) -> list[nn.Parameter]:
        return list(self.lora_params) + [self.readout]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def parity(self, state: str, instructions: list[str]) -> float:
        """Max |P(true) ours - P(true) runtime| for text-only questions."""
        h = self.encode(state, instructions)
        ours = torch.sigmoid(self.noul_logit(h)).cpu()
        answers, _ = self.engine.ask(state, {f"q{k}": {"type": "noul", "instructions": t}
                                            for k, t in enumerate(instructions)})
        ref = torch.tensor([answers[f"q{k}"]["noul"] for k in range(len(instructions))])
        return float((ours - ref).abs().max())
