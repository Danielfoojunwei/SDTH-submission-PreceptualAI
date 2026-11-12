"""A first SCENE-JEPA slice on simultaneous receiver-array power views.

Fixed relative-delay basis tokens render channel power through a Fourier
operator. They are not identified scatterer positions or absolute path delays.
Neither simulator truth, transmitter pose, a mission nor RadioMap is an input.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .operators import render_cir
from .priors import PriorRegularizer


def _positions(positions, batch):
    if positions.ndim == 2:
        positions = positions[None].expand(batch, -1, -1)
    if positions.ndim != 3 or positions.shape[0] != batch or positions.shape[-1] != 3:
        raise ValueError('Receiver coordinates must be B x antennas x XYZ')
    if not torch.isfinite(positions).all():
        raise ValueError('Nonfinite receiver metadata')
    return positions


def receiver_features(positions, origin):
    relative = positions - origin[:, None]
    return torch.stack((relative[..., :2].norm(dim=-1) / 20., relative[..., 2] / 5.), -1)


class SceneEncoder(nn.Module):
    """Set attention preserves a separate absolute log-power coordinate."""
    def __init__(self, bins=64, width=48):
        super().__init__()
        self.bins, self.width = bins, width
        self.receiver = nn.Sequential(nn.Linear(bins + 4, width), nn.SiLU(), nn.Linear(width, width))
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True, dropout=0.)
        self.pool_score = nn.Linear(width, 1)
        self.finish = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.projector = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, power_db, receiver_xyz, origin, carrier_hz):
        if (power_db.ndim != 3 or power_db.shape[-1] != self.bins
                or not torch.isfinite(power_db).all() or power_db.shape[1] < 1):
            raise ValueError('Use finite B x visible receivers x frequency bins')
        batch, receivers, _ = power_db.shape
        xyz = _positions(receiver_xyz, batch)
        if (xyz.shape[1] != receivers or origin.shape != (batch, 3)
                or not torch.isfinite(origin).all()):
            raise ValueError('Receiver metadata and RF axes disagree')
        carrier = torch.as_tensor(carrier_hz, dtype=power_db.dtype, device=power_db.device)
        if carrier.numel() != 1 or not torch.isfinite(carrier) or carrier <= 0:
            raise ValueError('One known positive carrier per recording is required')
        level = power_db.mean(-1, keepdim=True)
        shape = (power_db - level) / 10.
        features = torch.cat((shape, level / 30., receiver_features(xyz, origin),
                              torch.ones_like(level) * carrier / 5.e9), -1)
        tokens = self.receiver(features)
        # This is receiver-geometry attention, not an invented CSMA graph.
        distance = torch.cdist(xyz, xyz) / 20.
        bias = -distance[:, None].expand(-1, 4, -1, -1).reshape(batch * 4, receivers, receivers)
        update, _ = self.attention(tokens, tokens, tokens, attn_mask=bias, need_weights=False)
        tokens = tokens + update
        weights = torch.softmax(self.pool_score(tokens), dim=1)
        z = self.finish((weights * tokens).sum(1))
        return {'z': z, 'projected': self.projector(z),
                'power_coordinate': power_db.mean((1, 2)), 'tokens': tokens}


class SceneJEPA(nn.Module):
    def __init__(self, *, bins=64, width=48, paths=16, regularizer='sigreg',
                 decoder='physical', alignment_weight=.1, prior_weight=.01):
        super().__init__()
        if (type(bins) is not int or bins < 2 or type(width) is not int or width < 4 or width % 4
                or type(paths) is not int or not 1 <= paths <= bins
                or not math.isfinite(alignment_weight) or alignment_weight < 0
                or not math.isfinite(prior_weight) or prior_weight < 0):
            raise ValueError('Invalid architecture or loss dimensions')
        if decoder not in ('physical', 'unstructured'):
            raise ValueError('Unknown decoder')
        if regularizer == 'ppr':
            raise ValueError('Power-only data do not identify physical phase/path factors for PPR')
        self.config = dict(bins=bins, width=width, paths=paths, regularizer=regularizer,
                           decoder=decoder, alignment_weight=alignment_weight, prior_weight=prior_weight)
        self.encoder = SceneEncoder(bins, width)
        self.prior = PriorRegularizer(regularizer, num_slices=64)
        outputs = paths * 2 if decoder == 'physical' else bins
        self.query_receiver_score = nn.Linear(width, 1)
        self.decoder = nn.Sequential(nn.Linear(2 * width + 6, 64), nn.SiLU(), nn.Linear(64, outputs))
        nn.init.normal_(self.decoder[-1].weight, std=.002)
        nn.init.zeros_(self.decoder[-1].bias)
        if decoder == 'physical':
            # A dominant relative-zero-delay component starts near a flat
            # spectrum, instead of an artificial equal-tap deep-null comb.
            with torch.no_grad():
                self.decoder[-1].bias[:paths].fill_(-5.)
                self.decoder[-1].bias[0] = math.log(math.expm1(math.sqrt(paths)))

    def encode(self, power_db, receiver_xyz, origin, carrier_hz):
        return self.encoder(power_db, receiver_xyz, origin, carrier_hz)

    def render(self, encoded, visible_xyz, query_xyz, origin, carrier_hz, frequency_offsets_hz):
        z, power = encoded['z'], encoded['power_coordinate']
        batch = len(z)
        if origin.shape != (batch, 3) or not torch.isfinite(origin).all():
            raise ValueError('Use finite common receiver-frame origins')
        query, visible = _positions(query_xyz, batch), _positions(visible_xyz, batch)
        distance = torch.cdist(query, visible)
        # Preserve which RF token belongs to which receiver. Radius/minimum/
        # mean distances alone lose this association in symmetric layouts.
        scores = -distance / 5. + self.query_receiver_score(encoded['tokens']).transpose(1, 2)
        query_tokens = torch.softmax(scores, -1) @ encoded['tokens']
        geometry = torch.cat((receiver_features(query, origin), distance.amin(-1, keepdim=True) / 20.,
                              distance.mean(-1, keepdim=True) / 20.), -1)
        q = query.shape[1]
        carrier = torch.as_tensor(carrier_hz, dtype=z.dtype, device=z.device)
        if carrier.numel() != 1 or not torch.isfinite(carrier) or carrier <= 0:
            raise ValueError('Use one known positive carrier')
        offsets = torch.as_tensor(frequency_offsets_hz, dtype=z.dtype, device=z.device)
        if (offsets.shape != (self.config['bins'],) or not torch.isfinite(offsets).all()
                or not (offsets[1:] > offsets[:-1]).all()):
            raise ValueError('Frequency bins must be finite and strictly increase')
        features = torch.cat((z[:, None].expand(-1, q, -1), query_tokens, geometry,
                              power[:, None, None].expand(-1, q, 1) / 30.,
                              torch.ones(batch, q, 1, device=z.device) * carrier / 5.e9), -1)
        raw = self.decoder(features)
        if self.config['decoder'] == 'unstructured':
            return raw + power[:, None, None]
        paths = self.config['paths']
        amplitude = F.softplus(raw[..., :paths]) / math.sqrt(paths)
        angle = raw[..., paths:]
        coefficients = torch.complex(amplitude * angle.cos(), amplitude * angle.sin())
        spacing = (offsets[-1] - offsets[0]) / (len(offsets) - 1)
        if spacing <= 0:
            raise ValueError('Frequency bins must increase')
        bandwidth = spacing * len(offsets)
        relative_delays = torch.arange(paths, dtype=z.dtype, device=z.device) / bandwidth
        channel = render_cir(coefficients, relative_delays, offsets)
        return 10. * channel.abs().square().clamp_min(1.e-8).log10() + power[:, None, None]

    def objective(self, a, b, *, xyz_a, xyz_b, origin, carrier_hz, offsets_hz,
                  alignment=True, temporal_b=None):
        ea = self.encode(a, xyz_a, origin, carrier_hz)
        eb = self.encode(b, xyz_b, origin, carrier_hz)
        pa = self.render(eb, xyz_b, xyz_a, origin, carrier_hz, offsets_hz)
        pb = self.render(ea, xyz_a, xyz_b, origin, carrier_hz, offsets_hz)
        measurement = .5 * (F.huber_loss(pa, a, delta=3.) + F.huber_loss(pb, b, delta=3.))
        target = eb if temporal_b is None else self.encode(temporal_b, xyz_b, origin, carrier_hz)
        joint = F.mse_loss(ea['projected'], target['projected']) if alignment else measurement * 0.
        # Both the aligned coordinates and the backbone need an explicit
        # anti-collapse ablation. A constant projector otherwise evades it.
        backbone_prior = self.prior(torch.cat((ea['z'], eb['z'])))
        projected_prior = self.prior(torch.cat((ea['projected'], eb['projected'])))
        prior = .5 * (backbone_prior + projected_prior)
        loss = measurement + self.config['alignment_weight'] * joint + self.config['prior_weight'] * prior
        return loss, {'measurement_huber': measurement.detach(), 'joint_alignment': joint.detach(),
                      'prior_penalty': prior.detach(), 'projected_prior': projected_prior.detach()}

    @property
    def inference_parameters(self):
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith('prior.'))
