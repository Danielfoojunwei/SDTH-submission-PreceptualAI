# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from facebookresearch/vjepa2 app/vjepa/train.py, MIT licence.
# Exact upstream revision and original file hashes: third_party/vjepa2/SOURCE.json.
# Changes: wireless tensors, single target, optional variance regularisation;
# no video model, checkpoint or V-JEPA performance claim.
"""Power prediction loss, normalised targets and EMA from the upstream recipe."""
import torch
from torch.nn import functional as F


def prediction_loss(predicted, target, exponent=1.):
    normalised = F.layer_norm(target.detach(), (target.size(-1),))
    return torch.mean(torch.abs(predicted-normalised)**exponent)/exponent


@torch.no_grad()
def momentum_update(encoder, target_encoder, momentum=.99):
    for context, target in zip(encoder.parameters(), target_encoder.parameters(), strict=True):
        target.mul_(momentum).add_(context, alpha=1-momentum)
