"""A genuinely zero-capable AP-local channel around the shared RF field.

This module reuses the frozen interval/Newton/Schur implementation. A gate
scales each complete local design column in both conditioning and rendering,
so it scales local mean contributions and covariance together. An all-zero
design recovers the global-eight model with the same shared parameters. No
coefficient prior is made singular and no prediction becomes a measurement.

The sixteen gates are shared across APs, positions and views, and learn only
through the fitting objective. They are not receiver-dependent attention or
an empirical risk certificate. A zero gate can become stationary because
the likelihood may depend quadratically on it. Exact fallback availability
does not guarantee optimization, reactivation or held-out improvement.
Small gates also do not by themselves identify small effective local priors:
the learned coefficient mean and SD can compensate for their scale. The
regularizer therefore penalizes their gated coefficient-prior RMS, rather
than the gate parameters alone. It is not a field-risk guarantee.
"""
from __future__ import annotations

import torch
from torch import nn

from ..hierarchical_radio.model import (
    ALIGNMENT_WEIGHT,
    LOCAL_DIM,
    HierarchicalField,
    observation_nll,
)

CELLS = ('global8_reg', 'global8_joint', 'hier_reg', 'nested_reg', 'nested_joint')
NESTED_CELLS = ('nested_reg', 'nested_joint')
JOINT_CELLS = ('global8_joint', 'nested_joint')
GATE_INITIAL_VALUE = .1
GATE_PENALTY_WEIGHT = .01
GATE_RMS_REFERENCE = .5
_BASE_CELLS = {
    'global8_reg': 'global8_reg',
    'global8_joint': 'global8_reg',
    'hier_reg': 'hier_reg',
    'nested_reg': 'hier_reg',
    'nested_joint': 'hier_joint',
}


class NestedField(HierarchicalField):
    """Shared global-eight field plus an optional gated AP-local design.

    ``config`` intentionally remains the frozen implementation's core config:
    its inference code checks the literal core-cell names. ``settings`` is
    the public, JSON-serializable constructor configuration for checkpoints.
    Call ``NestedField(**model.settings)`` to reconstruct the public variant.

    Only nested variants add sixteen trainable parameters. Global and fixed
    hierarchy controls store their constant gate vector as an inert buffer;
    no parameter-count equality with the nested variants is claimed.
    """

    def __init__(self, cell='nested_reg', *, width=32, inference_steps=3):
        if type(cell) is not str or cell not in CELLS:
            raise ValueError('Unknown registered nested-radio model cell')
        super().__init__(_BASE_CELLS[cell], width=width, inference_steps=inference_steps)
        self.variant = cell
        if cell in NESTED_CELLS:
            self.local_gates = nn.Parameter(torch.full((LOCAL_DIM,), GATE_INITIAL_VALUE))
        else:
            fixed = 1. if cell == 'hier_reg' else 0.
            self.register_buffer('local_gates', torch.full((LOCAL_DIM,), fixed))
        self.architecture = dict(self.architecture,
            public_variant=cell,
            base_cell=_BASE_CELLS[cell],
            gate_dimensions=LOCAL_DIM,
            gate_initial_value=GATE_INITIAL_VALUE if cell in NESTED_CELLS else fixed,
            gate_penalty_weight=GATE_PENALTY_WEIGHT if cell in NESTED_CELLS else 0.,
            gate_rms_reference=GATE_RMS_REFERENCE,
            gate_penalty='effective_coefficient_prior_RMS',
            gate_acts_on_context_and_query_design=True,
            singular_prior_required_for_zero_gate=False,
            zero_gate_reactivation_guaranteed=False)

    @property
    def settings(self):
        return {'cell': self.variant, 'width': self.config['width'],
                'inference_steps': self.config['inference_steps']}

    @property
    def learns_gates(self):
        return self.variant in NESTED_CELLS

    def gates(self):
        """Return feasible parameters directly, retaining endpoint gradients.

        The optimizer projects after each step. Forward evaluation rejects an
        unprojected state rather than applying a second clamp with a framework-
        dependent endpoint subgradient. There is no straight-through estimator.
        """
        if self.local_gates.shape != (LOCAL_DIM,) or not torch.isfinite(self.local_gates).all():
            raise FloatingPointError('Local design gates must be sixteen finite values')
        if self.learns_gates:
            if ((self.local_gates < 0.) | (self.local_gates > 1.)).any():
                raise ValueError('Project local gates into [0,1] before model evaluation')
            return self.local_gates
        expected = torch.ones_like(self.local_gates) if self.variant == 'hier_reg' else torch.zeros_like(self.local_gates)
        if not torch.equal(self.local_gates, expected):
            raise ValueError('Fixed-control gate buffers differ from their declared values')
        return self.local_gates

    @torch.no_grad()
    def project_gates_(self):
        """Commit finite gate parameters to [0,1] after every optimizer step.

        Nonfinite states are rejected rather than repaired. Fixed controls
        restore their declared zero/one buffers and never optimize a gate.
        """
        if self.local_gates.shape != (LOCAL_DIM,) or not torch.isfinite(self.local_gates).all():
            raise FloatingPointError('Nonfinite or malformed local gates cannot be projected')
        if self.learns_gates:
            self.local_gates.clamp_(0., 1.)
        else:
            self.local_gates.fill_(1. if self.variant == 'hier_reg' else 0.)
        return self

    def local_features(self, query, rx_xyz, ap_xyz, groups):
        scale = self.gates()
        features = super().local_features(query, rx_xyz, ap_xyz, groups)
        if self.learns_gates:
            # A single shared feature scale is used everywhere. Context and
            # query designs therefore remain one coherent conditional field.
            return features * scale.to(features)
        return features

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Reject malformed/infeasible gate checkpoints before copying weights."""
        if strict and 'local_gates' not in state_dict:
            raise RuntimeError('A strict nested-radio checkpoint must include local gates')
        if 'local_gates' in state_dict:
            incoming = state_dict['local_gates']
            if (not isinstance(incoming, torch.Tensor) or not incoming.is_floating_point()
                    or incoming.is_complex() or incoming.shape != (LOCAL_DIM,)
                    or not torch.isfinite(incoming).all()):
                raise RuntimeError('Checkpoint local gates must be sixteen finite real values')
            if ((incoming < 0.) | (incoming > 1.)).any():
                raise RuntimeError('Checkpoint gates must already be projected into [0,1]')
            if not self.learns_gates:
                expected = torch.full_like(incoming, 1. if self.variant == 'hier_reg' else 0.)
                if not torch.equal(incoming, expected):
                    raise RuntimeError('Checkpoint fixed-control gates differ from their declared values')
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.gates()
        return result

    def shrinkage_penalty(self):
        """Penalize effective coefficient-prior RMS, not gate magnitude alone.

        In normalized residual units this is .01 times the mean of
        ``alpha*sqrt(mu_local**2 + sigma_local**2)/.5``. Reciprocal changes
        to a positive gate and its local prior cannot reduce this penalty
        while retaining the same effective mean and SD. Correlated/learned
        query features still prevent a universal field-variance interpretation.
        """
        if self.learns_gates:
            return GATE_PENALTY_WEIGHT * self.effective_local_prior()['rms'].mean() / GATE_RMS_REFERENCE
        return self.global_prior_mean.new_zeros(())

    def effective_local_prior(self):
        """Gated coefficient-prior mean/SD/RMS in normalized residual units.

        These use the exact declared SD calculation in frozen inference,
        before its numerical precision jitter and before conditioning. No
        query-dependent uncertainty or physical factor identification follows.
        """
        gate = self.gates().to(torch.float64)
        mean = self.local_prior_mean.to(torch.float64)
        std = .2 + torch.nn.functional.softplus(self.raw_local_std).to(torch.float64)
        rms = gate * torch.sqrt(mean.square() + std.square())
        return {'mean': gate * mean, 'std': gate * std, 'rms': rms}

    def base_state_dict(self, *, keep_vars=False):
        """Frozen-parent-compatible state, excluding the added gate tensor."""
        return {name: value for name, value in self.state_dict(keep_vars=keep_vars).items()
                if name != 'local_gates'}

    def load_base_state_dict(self, state):
        """Copy a complete compatible parent state without altering its gates."""
        if set(state) != set(self.base_state_dict()):
            raise RuntimeError('Complete frozen-parent state keys required')
        own_gate = self.local_gates.detach().clone()
        return self.load_state_dict(dict(state, local_gates=own_gate), strict=True)

    def diagnostics(self):
        """Inert scalar/list diagnostics; no model forward or fitting."""
        active = self.gates().detach().cpu()
        # In dB before conditioning and the frozen numerical precision jitter;
        # these are coefficient-prior scales, never predictive-SD floors.
        effective = {name: 10. * value.detach().cpu()
                     for name, value in self.effective_local_prior().items()}
        return {'variant': self.variant, 'core_cell': self.config['cell'],
                'learns_gates': self.learns_gates,
                'gates': active.tolist(), 'gate_mean': float(active.mean()),
                'gate_min': float(active.min()), 'gate_max': float(active.max()),
                'gate_zero_count': int((active == 0).sum()),
                'gate_one_count': int((active == 1).sum()),
                'gate_penalty': float(self.shrinkage_penalty().detach().cpu()),
                'effective_gated_prior_mean_db': effective['mean'].tolist(),
                'effective_gated_prior_std_db': effective['std'].tolist(),
                'effective_gated_prior_rms_db': effective['rms'].tolist(),
                'effective_rms_mean': float(effective['rms'].mean() / (10. * GATE_RMS_REFERENCE)),
                'effective_prior_rms_normalized_mean': float(effective['rms'].mean() / (10. * GATE_RMS_REFERENCE)),
                'effective_prior_rms_mean_db': float(effective['rms'].mean()),
                'gate_mean_identifies_effective_prior_scale': False,
                'added_trainable_gate_parameters': LOCAL_DIM if self.learns_gates else 0,
                'allocated_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'exact_global8_fallback_at_zero_design': self.learns_gates,
                'zero_gate_reactivation_guaranteed': False,
                'physical_state_identification_claimed': False,
                'packet_risk_qualified': False}


def objective(model, a, b, query, target, ap_xyz):
    """Matched likelihood, optional agreement and gated coefficient-prior RMS.

    Both joint cells add the same fixed supported predictive-agreement term.
    Both posterior means receive gradients; the variance normalization alone
    is detached. There is no EMA teacher or classic-JEPA equivalence claim.
    """
    if not isinstance(model, NestedField):
        raise TypeError('The nested objective requires a NestedField')
    ea, eb = model.encode_view(a, ap_xyz), model.encode_view(b, ap_xyz)
    pa, sa = model.render_queries(ea, query)
    pb, sb = model.render_queries(eb, query)
    measurement = .5 * (observation_nll(pa, sa, target) + observation_nll(pb, sb, target))
    shared_ap = ea['observed_ap'] & eb['observed_ap']
    supported = torch.gather(shared_ap, 1, query['ap'].to(torch.int64))
    joint = model.variant in JOINT_CELLS
    if joint and supported.any():
        working_variance = ((sa.square() + sb.square()) / 100. + .01).detach()
        alignment = (((pa - pb) / 10.).square() / working_variance)[supported].mean()
    else:
        alignment = (pa - pb).sum() * 0.
    gate_penalty = model.shrinkage_penalty()
    alignment_weight = ALIGNMENT_WEIGHT if joint else 0.
    loss = measurement + alignment_weight * alignment + gate_penalty
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite nested-radio training objective')
    active = model.gates()
    effective_rms = model.effective_local_prior()['rms'].mean()
    steps = [step for belief in (ea, eb) for step in belief['inference_log']]
    return loss, {
        'loss': loss.detach(), 'measurement_nll': measurement.detach(),
        'alignment': alignment.detach(), 'alignment_weight': loss.new_tensor(alignment_weight),
        'gate_penalty': gate_penalty.detach(),
        'gate_penalty_weight': loss.new_tensor(GATE_PENALTY_WEIGHT if model.learns_gates else 0.),
        'gate_rms_reference': loss.new_tensor(GATE_RMS_REFERENCE),
        'effective_prior_rms_normalized_mean': (effective_rms / GATE_RMS_REFERENCE).detach(),
        'effective_prior_rms_mean_db': (10. * effective_rms).detach(),
        'gate_mean': active.mean().detach(), 'gate_zero_count': (active == 0).sum().detach(),
        'supported_agreement_queries': supported.sum().detach(),
        'mean_predictive_std_db': .5 * (sa.mean() + sb.mean()).detach(),
        'mean_inference_step_size': torch.stack([step['step_size'].mean() for step in steps]).mean().detach(),
        'inference_null_steps': torch.stack([step['null_step'].sum() for step in steps]).sum().detach(),
    }
