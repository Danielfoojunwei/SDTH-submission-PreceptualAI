"""Direct-horizon RadioMap anchors and matched supervised/JEPA objectives.

The clean future teacher uses only the SAME SNR labels already used by every
family's supervised loss. This is privileged simulator supervision, not
label-free learning. No teacher, projector or auxiliary head runs at inference.
"""
from __future__ import annotations
import copy
import math
from collections.abc import Mapping
import torch
from torch import nn
from torch.nn import functional as F

FAMILIES = ('supervised', 'context_jepa', 'semantic_aux', 'semantic_jepa')
_HORIZON_WEIGHTS = (1., 1., 1., 1., 1., 1., .5, .3, .2)


def _finite_scalar(value, name, *, positive=False):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be finite and nonnegative')


def _clean(value, shape, valid, name, reference):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
            or not value.is_floating_point() or value.device != reference.device or value.dtype != reference.dtype):
        raise ValueError(f'{name} has incompatible shape, dtype or device')
    mask = valid
    while mask.ndim < value.ndim:
        mask = mask[..., None]
    result = torch.where(mask, value, 0.)
    if not torch.isfinite(result).all():
        raise ValueError(f'{name} has a nonfinite valid value')
    return result


def _batch(batch):
    if not isinstance(batch, Mapping):
        raise TypeError('Use a mapping of public Fleet input tensors')
    context = batch['context']
    if (not isinstance(context, torch.Tensor) or context.ndim != 3 or context.shape[-1] != 48
            or not context.is_floating_point()):
        raise ValueError('context must be floating B x AP*3 x 48')
    b, links, _ = context.shape
    if b < 1 or links < 3 or links % 3:
        raise ValueError('Preserve a nonempty AP x three-band grid')
    query = batch['query']
    if (not isinstance(query, torch.Tensor) or query.ndim != 4
            or query.shape[0] != b or query.shape[2:] != (links, 11) or query.shape[1] < 1):
        raise ValueError('query must be B x H x AP*3 x 11')
    h = query.shape[1]
    raw_valid = batch.get('valid_link', context.new_ones(b, links))
    if (not isinstance(raw_valid, torch.Tensor) or raw_valid.shape != (b, links)
            or raw_valid.device != context.device or not torch.isfinite(raw_valid).all()
            or not ((raw_valid == 0) | (raw_valid == 1)).all()):
        raise ValueError('valid_link must be a binary B x AP*3 mask')
    valid = raw_valid.bool()
    if not valid.any(1).all():
        raise ValueError('Each example needs a valid physical link')
    cleaned = {'context': _clean(context, (b, links, 48), valid, 'context', context),
               'query': _clean(query, (b, h, links, 11), valid[:, None], 'query', context),
               'memory': _clean(batch['memory'], (b, links, 13), valid, 'memory', context)}
    for name in ('map_mean', 'map_std', 'physics'):
        cleaned[name] = _clean(batch[name], (b, h, links), valid[:, None], name, context)
    if (cleaned['map_std'] < 0).any():
        raise ValueError('Map uncertainty must be nonnegative')
    return cleaned, valid


def _semantic(snr, base):
    """Fixed 5 dB margin is a representation anchor, not a calibrated PER rule."""
    return torch.stack([(snr - base) / 10., (snr - 5.) / 10.], -1)


class SemanticJEPA(nn.Module):
    """AP/band-preserving forecasts, directly decoded from one observed anchor.

    Missing six-scan readings use current map mean ONLY in the semantic path.
    Original heard/scanned masks remain in the context: imputation never marks
    an unknown measurement observed. No recursive output or latent integration.
    """
    def __init__(self, family='semantic_jepa', jepa_weight=.1, hidden=32,
                 variance_weight=.01, variance_floor=.15, ema_momentum=.99):
        super().__init__()
        if family not in FAMILIES or not isinstance(hidden, int) or isinstance(hidden, bool) or hidden < 4:
            raise ValueError('Unknown family or invalid hidden width')
        for value, name in ((jepa_weight, 'jepa_weight'), (variance_weight, 'variance_weight'),
                            (variance_floor, 'variance_floor'), (ema_momentum, 'ema_momentum')):
            _finite_scalar(value, name)
        if ema_momentum >= 1:
            raise ValueError('EMA momentum must be below one')
        self.config = dict(family=family, jepa_weight=jepa_weight, hidden=hidden,
                           variance_weight=variance_weight, variance_floor=variance_floor,
                           ema_momentum=ema_momentum)
        self.context_encoder = nn.Sequential(nn.Linear(48, hidden), nn.SiLU(),
                                             nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.radio_encoder = nn.Sequential(nn.Linear(2, 16), nn.SiLU(),
                                           nn.Linear(16, hidden), nn.LayerNorm(hidden))
        self.target_context_encoder = copy.deepcopy(self.context_encoder)
        self.target_radio_encoder = copy.deepcopy(self.radio_encoder)
        for teacher in (self.target_context_encoder, self.target_radio_encoder):
            teacher.requires_grad_(False)
            teacher.eval()
        self.link_norm = nn.LayerNorm(hidden)
        self.memory_encoder = nn.Sequential(nn.Linear(13, 16), nn.SiLU())
        self.predictor = nn.Sequential(nn.Linear(3 * hidden + 16 + 11 + 3, 64), nn.SiLU(),
                                       nn.Linear(64, hidden), nn.LayerNorm(hidden))
        self.head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[1] = 2.
        # Identical training-only module allocation in every family.
        self.projector = nn.Linear(hidden, hidden)
        self.aux_head = nn.Linear(hidden, 2)

    def train(self, mode=True):
        super().train(mode)
        self.target_context_encoder.eval()
        self.target_radio_encoder.eval()
        return self

    def forward(self, batch):
        # Explicit public-field whitelist: labels are never inspected here.
        x, valid = _batch(batch)
        context = x['context']
        b, links, _ = context.shape
        h = x['query'].shape[1]
        scans = context.reshape(b, links, 6, 8)
        heard = (scans[..., 2] > .5) & (scans[..., 3] > .5) & valid[..., None]
        index = torch.arange(6, device=context.device).expand(b, links, 6)
        last = torch.where(heard, index, -1).amax(-1)
        observed_snr = scans[..., 0].gather(-1, last.clamp(min=0)[..., None])[..., 0] * 30.
        base_now = x['map_mean'][:, 0]
        estimate = torch.where(last >= 0, observed_snr, base_now)
        semantic = torch.where(valid[..., None], _semantic(estimate, base_now), 0.)
        encoded_context = self.context_encoder(context)
        radio = self.radio_encoder(semantic)
        local = torch.where(valid[..., None], self.link_norm(encoded_context + radio), 0.)
        per_ap = local.reshape(b, links // 3, 3, -1)
        per_ap_valid = valid.reshape(b, links // 3, 3)
        across = per_ap.sum(2) / per_ap_valid.sum(2, keepdim=True).clamp(min=1)
        across = across.repeat_interleave(3, 1)
        fleet = local.sum(1, keepdim=True) / valid.sum(1, keepdim=True)[..., None].clamp(min=1)
        memory = self.memory_encoder(x['memory'])
        common = torch.cat([local, across, fleet.expand_as(local), memory], -1)
        features = torch.cat([common[:, None].expand(-1, h, -1, -1), x['query'],
                              x['map_mean'][..., None] / 30., x['map_std'][..., None] / 10.,
                              x['physics'][..., None] / 30.], -1)
        latent = self.predictor(features)
        raw = self.head(latent)
        residual = 12. * torch.tanh(raw[..., 0])
        mean = torch.where(valid[:, None], x['map_mean'] + residual, 0.)
        std = torch.where(valid[:, None], F.softplus(raw[..., 1]) + .25, .25)
        return {'mean': mean, 'std': std, 'residual': torch.where(valid[:, None], residual, 0.),
                'latent': latent, 'context': encoded_context, 'link_latent': local,
                'radio_latent': radio, 'radio_semantic': semantic,
                'semantic_observed': (last >= 0), 'valid_link': valid}

    @property
    def inference_parameters(self):
        training_only = ('target_context_encoder.', 'target_radio_encoder.', 'projector.', 'aux_head.')
        return sum(p.numel() for name, p in self.named_parameters() if not name.startswith(training_only))


def _weighted(values, weights):
    return (values * weights).sum() / weights.sum().clamp(min=1.)


def _band_variance(model, prediction):
    """Common regularizer, conditioned on physical band and nonconstant RF.

    At least four genuine heard samples are required. No variation is forced
    into constant RF groups. This is not a no-collapse theorem or rank claim.
    """
    z, semantic = prediction['radio_latent'], prediction['radio_semantic']
    observed = prediction['semantic_observed'] & prediction['valid_link']
    result = z.sum() * 0.
    groups = 0
    for band in range(3):
        keep = observed[:, band::3]
        samples, values = z[:, band::3][keep], semantic[:, band::3][keep]
        if len(samples) >= 4 and values.detach().std(0, correction=0).amax() > 1e-3:
            result = result + F.relu(model.config['variance_floor'] - samples.std(0, correction=0)).square().mean()
            groups += 1
    return result / max(groups, 1)


def objective(model, batch):
    """Common mean-field/route/AP losses plus one matched auxiliary objective.

    These are NOT native packet-loss labels. Semantic teacher/auxiliary targets
    contain only the truth SNR shared by all supervised arms and the same known
    map baseline. No actual future positions or extra future geometry enter.
    """
    p = model(batch)
    mean, std, valid = p['mean'], p['std'], p['valid_link']
    b, h, links = mean.shape
    if h > len(_HORIZON_WEIGHTS):
        raise ValueError('The registered loss supports the nine Fleet horizons')
    truth = _clean(batch['truth'], mean.shape, valid[:, None], 'truth', mean)
    hw = mean.new_tensor(_HORIZON_WEIGHTS[:h])[None, :, None]
    weights = torch.where(truth > 0, 1., .2) * hw * valid[:, None]
    error = mean - truth
    signal = _weighted(F.huber_loss(mean, truth, delta=3., reduction='none'), weights)
    nll = _weighted(.5 * (error / std).square() + std.log(), weights)
    near = min(h, 6)
    route = _weighted(F.huber_loss(mean[:, :near] - mean[:, :1],
                                  truth[:, :near] - truth[:, :1], delta=2., reduction='none'),
                      weights[:, :near])
    shaped_valid = valid.reshape(b, links // 3, 3)
    ap_valid = shaped_valid.any(-1)
    score = mean[:, :near].reshape(b, near, links // 3, 3).masked_fill(~shaped_valid[:, None], -1e4).amax(-1).mean(1)
    actual = truth[:, :near].reshape(b, near, links // 3, 3).masked_fill(~shaped_valid[:, None], -1e4).amax(-1).mean(1)
    score, actual = score.masked_fill(~ap_valid, -1e4), actual.masked_fill(~ap_valid, -1e4)
    regret = (torch.softmax(score / 3., -1) * (actual.amax(-1, keepdim=True) - actual)).sum(-1).mean()
    center = (score * ap_valid).sum(-1, keepdim=True) / ap_valid.sum(-1, keepdim=True)
    actual_center = (actual * ap_valid).sum(-1, keepdim=True) / ap_valid.sum(-1, keepdim=True)
    relative = _weighted(F.huber_loss(score - center, actual - actual_center, delta=2., reduction='none'), ap_valid)
    fleet_reference = signal + .15 * nll + .5 * regret + .1 * relative
    common = fleet_reference + .25 * route
    auxiliary = common * 0.
    family = model.config['family']
    if family == 'context_jepa':
        future = _clean(batch['future_context'], (b, h, links, 48), valid[:, None], 'future_context', mean)
        with torch.no_grad():
            target = model.target_context_encoder(future)
        auxiliary = _weighted(F.smooth_l1_loss(model.projector(p['latent']), target, reduction='none').mean(-1), weights)
    elif family in ('semantic_aux', 'semantic_jepa'):
        target_semantic = _semantic(truth, _clean(batch['map_mean'], mean.shape, valid[:, None], 'map_mean', mean))
        if family == 'semantic_aux':
            auxiliary = _weighted(F.smooth_l1_loss(model.aux_head(p['latent']), target_semantic, reduction='none').mean(-1), weights)
        else:
            with torch.no_grad():
                target = model.target_radio_encoder(target_semantic)
            auxiliary = _weighted(F.smooth_l1_loss(model.projector(p['latent']), target, reduction='none').mean(-1), weights)
    variance = _band_variance(model, p)
    total = common + model.config['jepa_weight'] * auxiliary + model.config['variance_weight'] * variance
    metrics = {'signal_huber': signal, 'nll': nll, 'route_delta': route,
               'ap_regret_soft': regret, 'ap_relative': relative,
               'fleet_reference_loss': fleet_reference, 'common_loss': common,
               'auxiliary': auxiliary, 'variance': variance}
    return total, {name: value.detach() for name, value in metrics.items()}


@torch.no_grad()
def update_target(model, momentum=None):
    momentum = model.config['ema_momentum'] if momentum is None else momentum
    _finite_scalar(momentum, 'EMA momentum')
    if momentum > 1:
        raise ValueError('EMA momentum must not exceed one')
    for online, target in ((model.context_encoder, model.target_context_encoder),
                           (model.radio_encoder, model.target_radio_encoder)):
        for source, destination in zip(online.parameters(), target.parameters(), strict=True):
            destination.lerp_(source, 1. - momentum)
        for source, destination in zip(online.buffers(), target.buffers(), strict=True):
            destination.copy_(source)


def _spectrum(values):
    if len(values) < 2:
        return {'samples': len(values), 'std_mean': 0., 'effective_rank': 0.}
    centered = values - values.mean(0)
    energy = torch.linalg.svdvals(centered).square()
    if energy.sum() <= 1e-12:
        rank = 0.
    else:
        probabilities = energy / energy.sum()
        rank = float(torch.exp(-(probabilities * probabilities.clamp(min=1e-12).log()).sum()))
    return {'samples': len(values), 'std_mean': float(values.std(0, correction=0).mean()),
            'effective_rank': rank}


@torch.no_grad()
def diagnostics(model, batch):
    """Per-band spectra and fixed-geometry RF response; never qualification."""
    p = model(batch)
    valid = p['valid_link']
    result = {'online_radio_by_band': {}, 'teacher_radio_by_band': {}}
    target = None
    if 'truth' in batch:
        truth = _clean(batch['truth'], p['mean'].shape, valid[:, None], 'truth', p['mean'])
        base = _clean(batch['map_mean'], p['mean'].shape, valid[:, None], 'map_mean', p['mean'])
        target = model.target_radio_encoder(_semantic(truth, base))
    for band in range(3):
        keep = valid[:, band::3]
        result['online_radio_by_band'][str(band)] = _spectrum(p['radio_latent'][:, band::3][keep])
        if target is not None:
            samples = target[:, :, band::3].permute(0, 2, 1, 3)[keep].reshape(-1, model.config['hidden'])
            result['teacher_radio_by_band'][str(band)] = _spectrum(samples)
    changed = dict(batch)
    context = batch['context'].clone()
    scans = context.reshape(context.shape[0], context.shape[1], 6, 8)
    heard = (scans[..., 2] > .5) & (scans[..., 3] > .5) & valid[..., None]
    scans[..., 0] += heard.to(context.dtype) / 30.
    changed['context'] = context
    q = model(changed)
    result['fixed_geometry_plus1db_radio_l2'] = float((q['radio_latent'] - p['radio_latent'])[valid].norm(dim=-1).mean())
    result['fixed_geometry_plus1db_mean_l2'] = float((q['mean'] - p['mean']).square().sum().sqrt())
    result['guarantee'] = False
    return result
