"""Matched JEPA/supervised action values learned from factual packet forks.

Positive target deltas mean that switching caused MORE missed commands or a
longer silence than staying. These are learned estimates, not safety bounds.
Only observable context and an action query enter ``forward``. Packet-fork
labels and future observable context are accepted by the training objective.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class ContextEncoder(nn.Module):
    """A finite observed sequence, with optional genuinely missing target rows."""

    def __init__(self, feature_dim: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, context: torch.Tensor, valid: torch.Tensor | None = None):
        if context.ndim != 3:
            raise ValueError('Context must have shape B x T x feature_dim')
        if valid is None:
            if not torch.isfinite(context).all():
                raise ValueError('Observable context must be finite; encode missingness explicitly')
            step_valid = torch.ones(context.shape[:2], device=context.device, dtype=torch.bool)
        else:
            valid = torch.as_tensor(valid, device=context.device)
            if valid.shape == context.shape[:1]:
                valid = valid[:, None].expand(context.shape[:2])
            if valid.shape == context.shape[:2]:
                feature_valid = valid.bool()[..., None].expand_as(context)
            elif valid.shape == context.shape:
                feature_valid = valid.bool()
            else:
                raise ValueError('Future validity must have shape B, B x T or B x T x F')
            if not torch.isfinite(torch.where(feature_valid, context, 0.)).all():
                raise ValueError('A future context feature marked observed is nonfinite')
            context = torch.where(feature_valid, context, 0.)
            step_valid = feature_valid.any(-1)
        output, _ = self.gru(context)
        # Missing trailing rows do not turn into a fictitious later observation.
        index = torch.arange(context.shape[1], device=context.device)[None]
        last = torch.where(step_valid, index, -1).max(-1).values
        latent = self.norm(output[torch.arange(len(context), device=context.device), last.clamp(min=0)])
        return torch.where((last >= 0)[:, None], latent, 0.)


class PacketValueModel(nn.Module):
    """Identical inference graph for an actual JEPA and its supervised ablation.

    ``target_scale`` normalises the two training targets; predictions retain
    native target units. The defaults expect missed-command counts and seconds
    of maximum control silence. Both families store the same frozen teacher so
    training parameter/storage comparisons also remain explicit. Only ``jepa``
    trains the action latent against the EMA encoder's future observations.
    """

    def __init__(self, family='jepa', feature_dim=32, query_dim=24, hidden=48,
                 target_scale=(20., 1.), jepa_weight=.1, variance_weight=.1,
                 ema_momentum=.99):
        super().__init__()
        if family not in {'jepa', 'supervised'}:
            raise ValueError('Choose jepa or its supervised ablation')
        if min(feature_dim, query_dim, hidden) < 1:
            raise ValueError('Model dimensions must be positive')
        scale = torch.tensor(target_scale, dtype=torch.float32)
        if scale.shape != (2,) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Use two finite positive target scales')
        if jepa_weight < 0 or variance_weight < 0 or not 0 <= ema_momentum < 1:
            raise ValueError('Invalid objective or EMA settings')
        self.config = {'family': family, 'feature_dim': feature_dim, 'query_dim': query_dim,
                       'hidden': hidden, 'target_scale': scale.tolist(),
                       'jepa_weight': jepa_weight, 'variance_weight': variance_weight,
                       'ema_momentum': ema_momentum}
        self.encoder = ContextEncoder(feature_dim, hidden)
        self.predictor = nn.Sequential(nn.Linear(hidden+query_dim, hidden*2), nn.SiLU(),
                                       nn.Linear(hidden*2, hidden), nn.LayerNorm(hidden))
        self.head = nn.Linear(hidden, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.register_buffer('target_scale', scale)

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    def _check_context(self, context):
        if (context.ndim != 3 or context.shape[-1] != self.config['feature_dim']
                or context.shape[0] < 1 or context.shape[1] < 1):
            raise ValueError('Unexpected observable context dimensions')

    def _predict(self, encoded, query):
        latent = self.predictor(torch.cat([encoded, query], -1))
        raw = self.head(latent)
        mean = raw[:, :2]*self.target_scale
        std = (F.softplus(raw[:, 2:])+.02)*self.target_scale
        return {'mean': mean, 'std': std, 'delta_misses': mean[:, 0],
                'delta_gap': mean[:, 1], 'miss_std': std[:, 0], 'gap_std': std[:, 1],
                'latent': latent, 'context_latent': encoded}

    def forward(self, context, query):
        self._check_context(context)
        if query.shape != (context.shape[0], self.config['query_dim']):
            raise ValueError('Unexpected observable action-query dimensions')
        if not torch.isfinite(query).all():
            raise ValueError('Observable action query must be finite')
        return self._predict(self.encoder(context), query)

    def forward_actions(self, context, query):
        """Score N x S action queries while encoding each robot history once.

        Every returned tensor has N x S leading axes. This is the same model
        as ``forward`` on repeated histories, with shared encoder work and
        normal floating-point roundoff from changing the GRU batch dimension.
        """
        self._check_context(context)
        if (query.ndim != 3 or query.shape[0] != context.shape[0]
                or query.shape[1] < 1 or query.shape[2] != self.config['query_dim']):
            raise ValueError('Action queries must have shape N x S x query_dim')
        if not torch.isfinite(query).all():
            raise ValueError('Observable action query must be finite')
        n, actions = query.shape[:2]
        encoded = self.encoder(context)[:, None].expand(-1, actions, -1)
        prediction = self._predict(encoded.reshape(n*actions, -1), query.reshape(n*actions, -1))
        return {key: value.reshape(n, actions, *value.shape[1:]) for key, value in prediction.items()}

    @property
    def inference_parameters(self):
        """EMA target encoder is used only during training."""
        return sum(p.numel() for module in (self.encoder, self.predictor, self.head)
                   for p in module.parameters())


def objective(model: PacketValueModel, batch: Mapping[str, torch.Tensor]):
    """Regression on native branch differences, with future-latent JEPA only.

    Required keys are ``context``, ``query``, ``delta_misses`` and ``delta_gap``.
    JEPA additionally requires ``future_context`` and ``future_valid``; the
    latter may mask an entire example, timestep or individual observed feature.
    Missing future context never becomes an invented all-zero observed target.
    ``label_valid`` may optionally mask B x 2 factual outcome labels.
    """
    prediction = model(batch['context'], batch['query'])
    target = torch.stack([batch['delta_misses'].reshape(-1), batch['delta_gap'].reshape(-1)], -1)
    if target.shape != prediction['mean'].shape:
        raise ValueError('Expected one factual branch difference per example')
    valid = torch.as_tensor(batch.get('label_valid', torch.ones_like(target)), device=target.device).bool()
    if valid.shape != target.shape:
        raise ValueError('Label validity must have shape B x 2')
    if not torch.isfinite(torch.where(valid, target, 0.)).all():
        raise ValueError('A factual label marked valid is nonfinite')
    target = torch.where(valid, target, 0.)
    error = (prediction['mean']-target)/model.target_scale
    std = prediction['std']/model.target_scale
    denominator = valid.sum().clamp(min=1)
    huber = (F.huber_loss(error, torch.zeros_like(error), delta=1., reduction='none')*valid).sum()/denominator
    nll = ((.5*(error/std).square()+std.log())*valid).sum()/denominator
    # Direction is material to admission. Zero-effect forks provide no invented
    # preference for either action and therefore do not enter the sign term.
    nonzero = valid[:, 0] & (target[:, 0] != 0)
    directional = F.softplus(-torch.sign(target[:, 0])*prediction['mean'][:, 0]/model.target_scale[0])
    sign = (directional*nonzero).sum()/nonzero.sum().clamp(min=1)
    cost = huber+.1*nll+.1*sign
    pretext = cost.new_zeros(())
    variance = cost.new_zeros(())
    if model.config['family'] == 'jepa':
        future = batch['future_context']
        future_valid = torch.as_tensor(batch['future_valid'], device=future.device).bool()
        if future.ndim != 3 or future.shape[0] != len(target) or future.shape[-1] != model.config['feature_dim']:
            raise ValueError('Unexpected future observable context dimensions')
        with torch.no_grad():
            future_latent = model.target_encoder(future, future_valid)
        example_valid = future_valid.reshape(len(target), -1).any(-1)
        latent_error = F.smooth_l1_loss(prediction['latent'], future_latent, reduction='none').mean(-1)
        pretext = (latent_error*example_valid).sum()/example_valid.sum().clamp(min=1)
        if len(target) > 1:
            spread = torch.sqrt(prediction['context_latent'].var(0, correction=0)+1e-4)
            variance = F.relu(.5-spread).mean()
        cost = cost+model.config['jepa_weight']*pretext+model.config['variance_weight']*variance
    return cost, {'huber': huber.detach(), 'nll': nll.detach(), 'sign': sign.detach(),
                  'jepa': pretext.detach(), 'variance': variance.detach(),
                  'valid_labels': valid.sum().detach()}


@torch.no_grad()
def update_target(model: PacketValueModel, momentum=None):
    """EMA updates are explicit and never receive an optimizer gradient."""
    momentum = model.config['ema_momentum'] if momentum is None else momentum
    if not 0 <= momentum <= 1:
        raise ValueError('EMA momentum must lie in [0, 1]')
    for target, online in zip(model.target_encoder.parameters(), model.encoder.parameters(), strict=True):
        target.lerp_(online, 1-momentum)
    for target, online in zip(model.target_encoder.buffers(), model.encoder.buffers(), strict=True):
        target.copy_(online)

