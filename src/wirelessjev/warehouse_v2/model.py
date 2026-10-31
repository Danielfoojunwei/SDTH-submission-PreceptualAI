"""Compact temporal robot/AP graph dynamics and cached-semantic value heads.

This model is trained from scratch on the declared simulation observations.
It is not an LWM checkpoint, JEPA implementation, or foundation-model claim.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def tensor_inputs(sample):
    return {k: torch.as_tensor(v) for k, v in sample['inputs'].items()}


class TemporalRadioModel(nn.Module):
    def __init__(self, *, hidden=48, graph=True, temporal=True, semantic_dim=1024):
        super().__init__()
        self.config = dict(hidden=hidden, graph=graph, temporal=temporal, semantic_dim=semantic_dim)
        self.graph, self.temporal, self.hidden = graph, temporal, hidden
        self.history_encoder = nn.GRU(3, hidden, batch_first=True)
        self.instant_encoder = mlp(3, hidden, hidden)
        self.robot_encoder = mlp(hidden+9+6+1, hidden, hidden)
        self.ap_encoder = mlp(6+hidden+1, hidden, hidden)
        self.message = mlp(3*hidden, hidden, hidden)
        self.output = mlp(hidden, hidden, 4)
        self.text_projection = nn.Linear(semantic_dim, hidden)
        self.value = mlp(3*hidden, hidden, 1)

    def forward(self, inputs, plans, semantic=None):
        # One context can score a batch of intervention plans, at any fleet size.
        h, r, ap = inputs['history'], inputs['robot'], inputs['ap']
        K, n = plans.shape
        m = len(ap)
        rows = torch.arange(n, device=plans.device)[None].expand(K, -1)
        sequence = h[:, rows, plans].permute(1, 2, 0, 3).reshape(K*n, h.shape[0], 3)
        if self.temporal:
            _, encoded = self.history_encoder(sequence)
            history = encoded[-1].reshape(K, n, self.hidden)
        else:
            history = self.instant_encoder(sequence[:, -1]).reshape(K, n, self.hidden)
        moved = (plans != inputs['current'][None]).float()[..., None]
        robot = self.robot_encoder(torch.cat([history, r[None].expand(K, -1, -1), ap[plans], moved], -1))
        if self.graph:
            idx = plans[..., None].expand(-1, -1, self.hidden)
            total = torch.zeros(K, m, self.hidden, device=robot.device).scatter_add_(1, idx, robot)
            count = torch.zeros(K, m, 1, device=robot.device).scatter_add_(
                1, plans[..., None], torch.ones(K, n, 1, device=robot.device))
            aps = self.ap_encoder(torch.cat([ap[None].expand(K, -1, -1),
                                            total/count.clamp(min=1), count/100.], -1))
            # Co-channel contention is public channel metadata, not hidden RF.
            same = ((ap[:, None, 2]-ap[None, :, 2]).abs() < .1) & ((ap[:, None, 3]-ap[None, :, 3]).abs() < .001)
            neighbours = torch.einsum('ij,kjh->kih', same.float(), aps) / same.sum(1).clamp(min=1)[None, :, None]
            robot = robot + self.message(torch.cat([robot, aps.gather(1, idx), neighbours.gather(1, idx)], -1))
        out = self.output(robot)
        result = {'service': out[..., 0].sigmoid(), 'log_std': out[..., 1].clamp(-4., 0.),
                  'failure_logit': out[..., 2], 'signal': out[..., 3],
                  'pooled': torch.cat([robot.mean(1), robot.amax(1)], -1)}
        if semantic is not None:
            text = self.text_projection(semantic).expand(K, -1)
            result['value'] = self.value(torch.cat([result['pooled'], text], -1)).squeeze(-1)
        return result


def dynamics_loss(pred, target):
    residual = (pred['service']-target[..., 0]) / pred['log_std'].exp()
    gaussian = (.5*residual.square()+pred['log_std']).mean()
    return (gaussian + F.binary_cross_entropy_with_logits(pred['failure_logit'], target[..., 1])
            + .25*F.smooth_l1_loss(pred['signal'], target[..., 2]))


def listwise_loss(score, utility):
    target = torch.softmax((utility-utility.mean())/utility.std(unbiased=False).clamp(min=1e-4), 0)
    return -(target * F.log_softmax(score, 0)).sum()
