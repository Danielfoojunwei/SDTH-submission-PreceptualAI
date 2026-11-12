"""Scene-only multi-view regression through a public propagation basis.

This static RSSI slice does not identify physical scatterers or an SNR posterior.
The decoder can access only an eight-dimensional scene code and public query
geometry. Receiver tokens and observed power cannot bypass that code.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from wirelessjev.radiojepa.likelihoods import gaussian_interval_log_prob
from wirelessjev.radiojepa.priors import PriorRegularizer
from .operators import query_physics_basis


class SpectralScene(nn.Module):
    def __init__(self, regularizer="sigreg", *, geometry_only=False, width=64, latent=8):
        super().__init__()
        if regularizer not in ("none", "sigreg", "vicreg"):
            raise ValueError("Unidentified material/phase coordinates are ineligible for PPR")
        if width % 4 or latent != 8:
            raise ValueError("This registered physics-basis slice has eight latent coordinates")
        self.config = dict(regularizer=regularizer, geometry_only=geometry_only,
                           width=width, latent=latent)
        self.token = nn.Sequential(nn.Linear(20, width), nn.SiLU(), nn.Linear(width, width))
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(width, 4, batch_first=True, dropout=0.) for _ in range(2)])
        self.feed = nn.ModuleList([nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(),
                                                 nn.Linear(width * 2, width)) for _ in range(2)])
        self.pool = nn.Linear(width, 1)
        self.scene = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, latent))
        self.reliability = nn.Linear(width, 1)
        self.coefficients = nn.Linear(latent, 8)
        self.local_field = nn.Sequential(nn.Linear(latent + 10, width), nn.SiLU(),
                                          nn.Linear(width, 1))
        self.prior = PriorRegularizer(regularizer, num_slices=64)
        nn.init.eye_(self.coefficients.weight)
        nn.init.zeros_(self.coefficients.bias)
        nn.init.normal_(self.scene[-1].weight, std=.005)
        nn.init.zeros_(self.scene[-1].bias)
        nn.init.zeros_(self.reliability.weight)
        nn.init.zeros_(self.reliability.bias)
        nn.init.normal_(self.local_field[-1].weight, std=.01)
        nn.init.zeros_(self.local_field[-1].bias)

    def encode(self, context, mask):
        if (context.ndim != 3 or context.shape[-1] != 20 or mask.shape != context.shape[:2]
                or mask.dtype != torch.bool or not mask.any(-1).all()
                or not torch.isfinite(context[mask]).all()):
            raise ValueError("Use finite B x context x 20 public inputs with a nonempty mask")
        context = torch.where(mask[..., None], context, torch.zeros_like(context))
        values = context
        if self.config["geometry_only"]:
            # Remove measurements and observation status; keep public geometry,
            # frequency, calibration metadata and the reported availability mask.
            # This is a value-erased control, not strictly a CAD-only predictor.
            values = torch.cat((torch.zeros_like(context[..., :4]), context[..., 4:]), -1)
        tokens = self.token(values)
        for attention, feed in zip(self.attention, self.feed):
            update, _ = attention(tokens, tokens, tokens, key_padding_mask=~mask,
                                  need_weights=False)
            tokens = tokens + update
            tokens = tokens + feed(tokens)
        scores = self.pool(tokens).squeeze(-1).masked_fill(~mask, -torch.inf)
        pooled = (torch.softmax(scores, -1)[..., None] * tokens).sum(1)
        # Invert the declared propagation basis from actual context observations.
        # This is an eight-coefficient regularized field estimate, not identified
        # scatterer/material truth. Censors are handled by the neural likelihood;
        # they are never inserted into a least-squares solve as point readings.
        basis = query_physics_basis(context[..., 4:])
        heard = mask & (values[..., 2] > .5)
        weights = (.25 + 1.5 * torch.sigmoid(self.reliability(tokens).squeeze(-1))) * heard
        weighted = basis.transpose(-1, -2) * weights[:, None]
        precision = weighted @ basis + .1 * torch.eye(8, dtype=context.dtype, device=context.device)
        rhs = weighted @ (values[..., :1] * 3.)  # residual/30 -> residual/10 dB
        inverse_state = torch.linalg.solve(precision, rhs).squeeze(-1)
        # Unbounded for both RF and geometry-only controls: a bounded correction
        # would make unit-variance prior targets infeasible for the latter.
        return inverse_state + .25 * self.scene(pooled)

    def render(self, z, query, basis, fspl, offset):
        if (z.ndim != 2 or z.shape[-1] != 8 or query.ndim != 3 or query.shape[-1] != 16
                or basis.shape != (*query.shape[:2], 8) or z.shape[0] != query.shape[0]
                or fspl.shape != query.shape[:2] or offset.shape != fspl.shape):
            raise ValueError("Scene/query axes disagree")
        if not all(torch.isfinite(x).all() for x in (z, query, basis, fspl, offset)):
            raise ValueError("Nonfinite propagation inputs")
        beta = self.coefficients(z) * 10.
        physical = (beta[:, None] * basis).sum(-1)
        # A bounded correction accommodates reflection geometry omitted by the
        # AABB direct-path features. It receives no query measurements/device ID.
        public_geometry = torch.cat((query[..., :8], query[..., 9:11]), -1)
        local = self.local_field(torch.cat((z[:, None].expand(-1, query.shape[1], -1),
                                           public_geometry), -1)).squeeze(-1)
        residual = physical + 5. * torch.tanh(local)
        return fspl + offset + residual

    def forward(self, context, mask, query, basis, fspl, offset):
        z = self.encode(context, mask)
        return self.render(z, query, basis, fspl, offset), z


def observation_nll(mean, target, *, std_db=4.):
    """Mean quantization-interval / actual censor likelihood; missing excluded.

    Four dB is a fixed unresolved-field error scale in this exploratory slice,
    not an estimated device noise floor or a calibrated safety distribution.
    """
    heard, censored = target["heard"], target["censored"]
    known = heard | censored
    if not known.any() or (heard & censored).any():
        raise ValueError("Need disjoint observed/censored events")
    lower = torch.where(censored, torch.full_like(mean, -torch.inf), target["lower_dbm"])
    upper = torch.where(censored, target["censor_upper_dbm"], target["upper_dbm"])
    # Fill unused rows with finite positive-width intervals before likelihood.
    lower = torch.where(known, lower, mean.detach() - .5)
    upper = torch.where(known, upper, mean.detach() + .5)
    result = -gaussian_interval_log_prob(lower, upper, mean,
                                         torch.full_like(mean, std_db))
    return result[known].mean()


def balanced_objective(model, a, b, query, basis, fspl, offset, target, *, alignment=True,
                       diagnose_backbone=False):
    """Equalize encoder-coordinate gradient scales using TRAIN observations.

    No EMA teacher or stopped target. Both views receive gradients. Per-update
    detached scalar weights target one quarter of the measurement gradient for
    each auxiliary term, capped to [.01, 100]; this rule is identical in matched
    regression except that its alignment term is absent.
    """
    za = model.encode(*a)
    zb = model.encode(*b)
    pa, pb = (model.render(z, query, basis, fspl, offset) for z in (za, zb))
    measurement = .5 * (observation_nll(pa, target) + observation_nll(pb, target))
    align = F.mse_loss(za, zb) if alignment else (za.sum() + zb.sum()) * 0.
    prior = .5 * (model.prior(za) + model.prior(zb))

    def norm(term):
        gradients = torch.autograd.grad(term, (za, zb), retain_graph=True, allow_unused=True)
        pieces = [g.square().mean() for g in gradients if g is not None]
        return torch.stack(pieces).mean().sqrt().detach() if pieces else term.detach() * 0.

    gm = norm(measurement)
    gp = norm(prior) if model.config["regularizer"] != "none" else gm * 0.
    ga = norm(align) if alignment else gm * 0.
    wp = ((.25 * gm / gp.clamp_min(1.e-10)).clamp(.01, 100.)
          if model.config["regularizer"] != "none" else gm * 0.)
    wa = (.25 * gm / ga.clamp_min(1.e-10)).clamp(.01, 100.) if ga > 0 else gm * 0.
    loss = measurement + wp * prior + wa * align
    diagnostics = {"measurement_nll": measurement.detach(), "alignment": align.detach(),
                   "prior": prior.detach(), "measurement_z_gradient": gm,
                   "prior_z_gradient": gp, "alignment_z_gradient": ga,
                   "prior_weight": wp, "alignment_weight": wa,
                   "prior_stationary_nonzero": ((gp == 0) & (prior.detach() > 1.e-6)).to(gm)}
    if diagnose_backbone:
        params = [p for name, p in model.named_parameters()
                  if not name.startswith(("prior.", "coefficients.", "local_field."))]

        def backbone(term):
            gradients = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
            return [torch.zeros_like(p) if g is None else g for p, g in zip(params, gradients)]

        gradients = [backbone(term) for term in (measurement, wp * prior, wa * align)]
        norms = [torch.stack([g.square().sum() for g in group]).sum().sqrt() for group in gradients]
        for label, norm_value in zip(("measurement", "weighted_prior", "weighted_alignment"), norms):
            diagnostics[label + "_backbone_gradient"] = norm_value.detach()
        for label, group, other_norm in zip(("prior", "alignment"), gradients[1:], norms[1:]):
            dot = torch.stack([(a * b).sum() for a, b in zip(gradients[0], group)]).sum()
            diagnostics[label + "_measurement_backbone_cosine"] = (
                dot / (norms[0] * other_norm).clamp_min(1.e-12)).detach()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite scene objective")
    return loss, diagnostics
