"""Observation-conditioned field beliefs with an explicit physical likelihood.

Eight regression coefficients describe a field, not identified materials or
scatterers. The Gaussian covariance is an approximation whose calibration must
be checked. Packet-service probabilities are outside this model's contract.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from ..radiojepa.likelihoods import gaussian_interval_log_prob
from ..spectral_scene.operators import query_physics_basis

CELLS = ('joint', 'regression', 'rf_erased', 'pure_geometry')


def propagation_features(query):
    return torch.cat((query[..., :8], query[..., 9:11]), -1)


class AdaptiveScene(nn.Module):
    """Shared geometry/status prior plus a censored measurement update.

    Both training views contain the same physical links and both source bands.
    A value-erased control retains binary heard/censored evidence. The strictly
    geometric control uses only the public attempted-observation schedule.
    """
    def __init__(self, cell='joint', *, width=64, em_steps=3):
        super().__init__()
        if cell not in CELLS or width % 4 or width < 16 or em_steps != 3:
            raise ValueError('Unregistered model configuration')
        self.config = dict(cell=cell, width=width, em_steps=em_steps)
        # Ten propagation coordinates, two status flags, noise/TX/bin metadata.
        self.token = nn.Sequential(nn.Linear(15, width), nn.SiLU(), nn.Linear(width, width))
        self.attention = nn.ModuleList([nn.MultiheadAttention(width, 4, dropout=0.,
                                                             batch_first=True) for _ in range(2)])
        self.feed = nn.ModuleList([nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(),
                                                 nn.Linear(width * 2, width)) for _ in range(2)])
        self.pool = nn.Linear(width, 1)
        self.prior_mean = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 8))
        self.log_prior_std = nn.Parameter(torch.full((8,), math.log(math.expm1(.8))))
        self.local = nn.Sequential(nn.Linear(10, width), nn.SiLU(), nn.Linear(width, 1))
        self.discrepancy = nn.Sequential(nn.Linear(10, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.normal_(self.prior_mean[-1].weight, std=.005)
        nn.init.zeros_(self.prior_mean[-1].bias)
        nn.init.normal_(self.local[-1].weight, std=.005)
        nn.init.zeros_(self.local[-1].bias)
        nn.init.zeros_(self.discrepancy[-1].weight)
        nn.init.constant_(self.discrepancy[-1].bias, math.log(3. / 11.))

    def local_db(self, query):
        return 5. * torch.tanh(self.local(propagation_features(query)).squeeze(-1))

    def discrepancy_db(self, query):
        # Unexplained FIELD variability, not receiver thermal or invented sensor noise.
        return 1. + 14. * torch.sigmoid(self.discrepancy(propagation_features(query)).squeeze(-1))

    def encode(self, context, known_mask, scheduled_mask):
        if (context.ndim != 3 or context.shape[-1] != 20 or known_mask.dtype != torch.bool
                or scheduled_mask.dtype != torch.bool or known_mask.shape != context.shape[:2]
                or scheduled_mask.shape != known_mask.shape or not scheduled_mask.any(-1).all()
                or (known_mask & ~scheduled_mask).any()):
            raise ValueError('Use explicit, nonempty schedules and disjoint unknown reports')
        if not torch.isfinite(context[scheduled_mask]).all():
            raise ValueError('Scheduled public features must be finite')
        context = torch.where(scheduled_mask[..., None], context, torch.zeros_like(context))
        query = context[..., 4:]
        heard = known_mask & (context[..., 2] > .5)
        censored = known_mask & (context[..., 3] > .5)
        if (heard & censored).any() or not torch.equal(heard | censored, known_mask):
            raise ValueError('Known rows must be genuinely heard or censored')
        pure = self.config['cell'] == 'pure_geometry'
        status = torch.zeros_like(context[..., 2:4]) if pure else context[..., 2:4]
        # Thermal floor, known TX and context-inferred quantization are public.
        features = torch.cat((propagation_features(query), status,
                              query[..., 14:16], query[..., 13:14]), -1)
        mask = scheduled_mask if pure else known_mask
        # An empty RF view returns the geometric prior, never a fabricated reading.
        mask = torch.where(mask.any(-1, keepdim=True), mask, scheduled_mask)
        tokens = self.token(features)
        for attention, feed in zip(self.attention, self.feed):
            update, _ = attention(tokens, tokens, tokens, key_padding_mask=~mask, need_weights=False)
            tokens = tokens + update
            tokens = tokens + feed(tokens)
        scores = self.pool(tokens).squeeze(-1).masked_fill(~mask, -torch.inf)
        pooled = (torch.softmax(scores, -1)[..., None] * tokens).sum(1)
        prior = self.prior_mean(pooled)
        prior_std = .2 + torch.nn.functional.softplus(self.log_prior_std)
        precision = torch.diag_embed(prior_std.reciprocal().square()).expand(len(prior), -1, -1)
        if self.config['cell'] in ('rf_erased', 'pure_geometry'):
            return {'mean': prior, 'covariance': torch.diag_embed(prior_std.square()).expand(len(prior), -1, -1)}

        basis = query_physics_basis(query)
        local = self.local_db(query) / 10.
        observed = context[..., 0] * 3. - local
        bound = context[..., 1] * 3. - local
        sigma = self.discrepancy_db(query) / 10.
        quantization_std = query[..., 13].abs() * 4. / math.sqrt(12.) / 10.
        variance = sigma.square() + torch.where(heard, quantization_std.square(), torch.zeros_like(sigma))
        weights = known_mask / variance.clamp_min(.01)
        weighted = basis.transpose(-1, -2) * weights[:, None]
        posterior_precision = precision + weighted @ basis
        factor = torch.linalg.cholesky(posterior_precision)
        mean = prior
        for _ in range(self.config['em_steps']):
            predicted = (basis * mean[:, None]).sum(-1)
            a = (torch.where(censored, bound, predicted) - predicted) / sigma
            # Conditional expectation in a true upper-censored Gaussian event.
            # It is an EM update, not a made-up reading at the detection floor.
            log_mills = -.5 * a.square() - .5 * math.log(2. * math.pi) - torch.special.log_ndtr(a)
            below = predicted - sigma * torch.exp(log_mills.clamp(max=20.))
            expected = torch.where(censored, below, observed)
            expected = torch.where(known_mask, expected, torch.zeros_like(expected))
            rhs = (precision @ prior[..., None]) + weighted @ expected[..., None]
            mean = torch.cholesky_solve(rhs, factor).squeeze(-1)
        # Observed-information covariance: an uninformative upper bound must
        # not count as a fully observed point. Complete-data EM precision above
        # is only a mean-update device, not a calibrated posterior covariance.
        predicted = (basis * mean[:, None]).sum(-1)
        a = (torch.where(censored, bound, predicted) - predicted) / sigma
        mills = torch.exp((-.5 * a.square() - .5 * math.log(2. * math.pi)
                           - torch.special.log_ndtr(a)).clamp(max=20.))
        information_fraction = (a * mills + mills.square()).clamp(0., 1.)
        information_weights = torch.where(censored, information_fraction / sigma.square(),
                                          variance.reciprocal()) * known_mask
        observed_precision = precision + (basis.transpose(-1, -2) * information_weights[:, None]) @ basis
        covariance = torch.cholesky_inverse(torch.linalg.cholesky(observed_precision))
        if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
            raise FloatingPointError('Nonfinite censored field update')
        return {'mean': mean, 'covariance': covariance}

    def render(self, belief, query, basis, fspl, offset):
        mean, covariance = belief['mean'], belief['covariance']
        if (mean.ndim != 2 or mean.shape[-1] != 8 or covariance.shape != (len(mean), 8, 8)
                or query.ndim != 3 or query.shape[-1] != 16
                or basis.shape != (*query.shape[:2], 8) or fspl.shape != query.shape[:2]
                or offset.shape != fspl.shape or query.shape[0] != len(mean)):
            raise ValueError('Belief/query dimensions disagree')
        if not all(torch.isfinite(x).all() for x in (mean, covariance, query, basis, fspl, offset)):
            raise ValueError('Nonfinite public field operands')
        power = fspl + offset + (basis * mean[:, None]).sum(-1) * 10. + self.local_db(query)
        coefficient_variance = torch.einsum('bqd,bde,bqe->bq', basis, covariance, basis) * 100.
        std = torch.sqrt(coefficient_variance.clamp_min(0.) + self.discrepancy_db(query).square())
        return power, std


def observation_nll(mean, std, target):
    heard, censored = target['heard'], target['censored']
    known = heard | censored
    if not known.any() or (heard & censored).any():
        raise ValueError('Need actual known observations')
    heard_lower = torch.maximum(target['lower_dbm'], target['detection_threshold_dbm'])
    lo = torch.where(censored, torch.full_like(mean, -torch.inf), heard_lower)
    hi = torch.where(censored, target['censor_upper_dbm'], target['upper_dbm'])
    lo = torch.where(known, lo, mean.detach() - .5)
    hi = torch.where(known, hi, mean.detach() + .5)
    value = -gaussian_interval_log_prob(lo, hi, mean, std)
    return value[known].mean()


def objective(model, a, b, query, basis, fspl, offset, target):
    ea, eb = model.encode(*a), model.encode(*b)
    pa, sa = model.render(ea, query, basis, fspl, offset)
    pb, sb = model.render(eb, query, basis, fspl, offset)
    measurement = .5 * (observation_nll(pa, sa, target) + observation_nll(pb, sb, target))
    difference = ea['mean'] - eb['mean']
    eye = torch.eye(8, device=difference.device, dtype=difference.dtype)
    # Weakly observed directions have less alignment pressure. Detaching only
    # this weighting matrix prevents covariance inflation from gaming alignment;
    # both view means remain differentiable and no stopped teacher is present.
    weight = (ea['covariance'] + eb['covariance'] + .01 * eye).detach()
    aligned = (difference * torch.linalg.solve(weight, difference[..., None]).squeeze(-1)).sum(-1).mean() / 8.
    if model.config['cell'] != 'joint':
        aligned = difference.sum() * 0.
    gm = torch.autograd.grad(measurement, (ea['mean'], eb['mean']), retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(aligned, (ea['mean'], eb['mean']), retain_graph=True, allow_unused=True)

    def norm(parts):
        entries = [g.square().mean() for g in parts if g is not None]
        return torch.stack(entries).mean().sqrt().detach() if entries else measurement.detach() * 0.

    measured_norm, alignment_norm = norm(gm), norm(ga)
    multiplier = ((.25 * measured_norm / alignment_norm.clamp_min(1.e-10)).clamp(.01, 10.)
                  if model.config['cell'] == 'joint' and alignment_norm > 0 else measured_norm * 0.)
    loss = measurement + multiplier * aligned
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite adaptive scene loss')
    return loss, {'loss': loss.detach(), 'measurement_nll': measurement.detach(),
                  'alignment': aligned.detach(), 'alignment_weight': multiplier,
                  'measurement_mean_gradient': measured_norm, 'alignment_mean_gradient': alignment_norm,
                  'mean_predictive_std_db': .5 * (sa.mean() + sb.mean()).detach()}
