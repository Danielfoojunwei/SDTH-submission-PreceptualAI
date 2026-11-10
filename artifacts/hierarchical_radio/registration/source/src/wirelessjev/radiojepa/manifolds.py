"""Physical coordinate parameterizations; not a claim of identifiability."""
import math
import torch
from torch.nn import functional as F


def unit_phasor(phase_radians):
    phase = torch.as_tensor(phase_radians)
    if not phase.is_floating_point() or not torch.isfinite(phase).all():
        raise ValueError('Use finite floating-point phase angles')
    return torch.complex(torch.cos(phase), torch.sin(phase))


def positive_amplitude(parameter, *, minimum=0.):
    """Stable positive parameterization, with an explicitly declared floor."""
    x = torch.as_tensor(parameter)
    if not x.is_floating_point() or not torch.isfinite(x).all() or not math.isfinite(minimum) or minimum < 0:
        raise ValueError('Use a finite floating parameter and nonnegative floor')
    return F.softplus(x) + minimum


def airtime_simplex(logits, *, dim=-1):
    """Shares for ONE contention domain/channel; last category is idle.

    Separate spatial-reuse domains and simultaneous MLO links do not share a
    universal network-wide simplex.
    """
    x = torch.as_tensor(logits)
    if not x.is_floating_point() or not torch.isfinite(x).all() or x.shape[dim] < 2:
        raise ValueError('Use finite busy-category logits plus an idle category')
    return torch.softmax(x, dim=dim)


def busy_airtime(logits, *, dim=-1):
    shares = airtime_simplex(logits, dim=dim)
    return shares.narrow(dim, 0, shares.shape[dim] - 1)
