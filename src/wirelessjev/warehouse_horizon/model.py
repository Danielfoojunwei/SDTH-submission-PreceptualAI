"""Multi-step residual forecaster over a public numerical radio belief."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..warehouse_v2.model import mlp


def tensors(observation):
    return {k: torch.as_tensor(v) for k, v in observation.items()}


def prior_prediction(prior, observation):
    # The aggregate critical fraction is shared by the twin's priority flows;
    # the strictest public flow floor determines whether any such flow fails.
    floor = observation['robot'][:, 4]
    failure = (prior[..., 1] < floor[None, None]-1e-6) | (prior[..., 2] < -.125)
    probability = np.where(failure, .98, .02).astype('float32')
    return {'service_raw': prior[..., 0], 'failure_logits': np.log(probability/(1.-probability)),
            'signal': prior[..., 3]}


class HorizonModel(nn.Module):
    def __init__(self, *, hidden=32, graph=True):
        super().__init__()
        self.config = {'hidden': hidden, 'graph': graph}
        self.hidden, self.graph = hidden, graph
        self.history = nn.GRU(5, hidden, batch_first=True)
        self.instant = mlp(5, hidden, hidden)
        self.robot = mlp(hidden+8+6+1, hidden, hidden)
        self.ap = mlp(6+hidden+1, hidden, hidden)
        self.message = mlp(3*hidden, hidden, hidden)
        self.decoder = mlp(hidden+8+8, hidden, 3)
        # At initialisation the residual model equals the numerical baseline.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, obs, plans, prior):
        K, n = plans.shape
        H, _, m, _ = obs['history'].shape
        rows = torch.arange(n, device=plans.device)[None].expand(K, -1)
        scan = obs['history'][:, rows, plans].permute(1, 2, 0, 3)
        xy = obs['past_xy'].permute(1, 0, 2)[None].expand(K, -1, -1, -1)
        sequence = torch.cat([scan, xy], -1).reshape(K*n, H, 5)
        if self.graph:
            _, encoded = self.history(sequence)
            encoded = encoded[-1].reshape(K, n, self.hidden)
        else:
            encoded = self.instant(sequence[:, -1]).reshape(K, n, self.hidden)
        moved = (plans != obs['current'][None]).float()[..., None]
        r = self.robot(torch.cat([encoded, obs['robot'][None].expand(K, -1, -1), obs['ap'][plans], moved], -1))
        if self.graph:
            idx = plans[..., None].expand(-1, -1, self.hidden)
            total = torch.zeros(K, m, self.hidden, device=r.device).scatter_add_(1, idx, r)
            count = torch.zeros(K, m, 1, device=r.device).scatter_add_(1, plans[..., None], torch.ones(K, n, 1, device=r.device))
            ap = self.ap(torch.cat([obs['ap'][None].expand(K, -1, -1), total/count.clamp(min=1), count/100.], -1))
            same = ((obs['ap'][:, None, 2]-obs['ap'][None, :, 2]).abs() < .1) & ((obs['ap'][:, None, 3]-obs['ap'][None, :, 3]).abs() < .001)
            adjacent = torch.einsum('ij,kjh->kih', same.float(), ap)/same.sum(1).clamp(min=1)[None, :, None]
            r = r+self.message(torch.cat([r, ap.gather(1, idx), adjacent.gather(1, idx)], -1))
        S = obs['route'].shape[0]
        route = obs['route'][None].expand(K, -1, -1, -1)
        now = obs['robot'][None, None, :, :2].expand(K, S, -1, -1)
        ap_xy = obs['ap'][plans, :2][:, None].expand(-1, S, -1, -1)
        time = torch.arange(1, S+1, device=r.device).float()[None, :, None, None].expand(K, -1, n, -1)/S
        distance = ((route-now)**2).sum(-1, keepdim=True).sqrt()
        route_features = torch.cat([route, route-now, route-ap_xy, time, distance], -1)
        features = torch.cat([r[:, None].expand(-1, S, -1, -1), route_features, prior], -1)
        delta = self.decoder(features)
        failed = (prior[..., 1] < obs['robot'][None, None, :, 4]-1e-6) | (prior[..., 2] < -.125)
        base_probability = torch.where(failed, .98, .02)
        base_logits = torch.logit(base_probability)
        return {'service_raw': (prior[..., 0]+torch.tanh(delta[..., 0])).clamp(0., 1.),
                'failure_logits': base_logits+delta[..., 1],
                'signal': prior[..., 3]+delta[..., 2]}


def loss(prediction, target):
    service = F.smooth_l1_loss(prediction['service_raw'], target['service_raw'], beta=.1)
    risk = F.binary_cross_entropy_with_logits(prediction['failure_logits'], target['failure_raw'])
    signal = F.smooth_l1_loss(prediction['signal'], target['signal'], beta=.1)
    return service+risk+.2*signal
