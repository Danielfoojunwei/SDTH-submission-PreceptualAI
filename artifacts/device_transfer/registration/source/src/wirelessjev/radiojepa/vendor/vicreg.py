"""Exact MIT-licensed VCReg regularizer and helper bodies, same pinned source.

The common radio objective supplies the view-prediction term. This is the
variance/covariance component, not a claim of the full VICReg method.
Imports use Python logging solely for an invalid-square-matrix assertion.
"""
import logging
import torch
import torch.distributed as dist
import torch.distributed.nn.functional
import torch.nn.functional as F


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized.

    Returns:
        bool: True if distributed is available and initialized, False otherwise
    """
    return dist.is_available() and dist.is_initialized()


def all_gather(tensor, *args, **kwargs):
    """Gather tensors from all processes (autograd-aware).

    Uses ``torch.distributed.nn.functional.all_gather`` so gradients flow back
    to each rank's source tensor — the behavior SSL contrastive losses need
    when gathering negatives across GPUs.

    Note:
        The list returned **must** be used (e.g. ``torch.cat(all_gather(x), 0)``)
        — the gathered values are the return value, not an in-place mutation of
        ``tensor``.

    Args:
        tensor: The tensor to gather.
        *args: Forwarded to the underlying collective (e.g. ``group``).
        **kwargs: Forwarded to the underlying collective.

    Returns:
        list[torch.Tensor]: One tensor per rank, in rank order. A single-element
        ``[tensor]`` when distributed is not initialized.
    """
    if is_dist_avail_and_initialized():
        return list(torch.distributed.nn.functional.all_gather(tensor, *args, **kwargs))
    return [tensor]


def off_diagonal(x):
    """Return a flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m, logging.error("Input tensor must be square.")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VCRegLoss(torch.nn.Module):
    """Variance-Covariance regularization loss..

    Args:
        std_coeff (float, optional): The weight of the standard deviation loss.
            Default is 25.
        cov_coeff (float, optional): The weight of the covariance loss.
            Default is 1.
        epsilon (float, optional): Small value to avoid division by zero.
            Default is 1e-4.
    """

    def __init__(
        self,
        std_coeff: float = 25,
        cov_coeff: float = 1,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.epsilon = epsilon

    def forward(self, z_i):
        """Compute the VCReg loss.

        Args:
            z_i (torch.Tensor): Latent representation of the batch.

        Returns:
            float: The computed loss.
        """
        z_i = torch.cat(all_gather(z_i), 0)

        z_i = z_i - z_i.mean(dim=0)

        std_i = torch.sqrt(z_i.var(dim=0) + self.epsilon)
        std_loss = torch.mean(F.relu(1 - std_i)) / 2

        cov_i = (z_i.T @ z_i) / (z_i.size(0) - 1)
        cov_loss = off_diagonal(cov_i).pow_(2).sum().div(z_i.size(1))

        loss = self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss
