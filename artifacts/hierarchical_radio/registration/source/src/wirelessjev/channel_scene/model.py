"""Pilot-only complex completion model and exactly matched learning controls.

The Fourier basis is a finite-band circular relative-delay parameterization,
not identified paths, material tokens, absolute ToF or a causality certificate.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def pilot_normalize(pilots):
    """All scale/phase estimates use observed pilots only, independently per row."""
    if pilots.ndim != 2 or not pilots.is_complex() or not torch.isfinite(pilots).all():
        raise ValueError('Finite complex row-by-pilot observations are required')
    rms = pilots.abs().square().mean(-1, keepdim=True).sqrt().clamp_min(1.e-12)
    index = pilots.abs().argmax(-1, keepdim=True)
    reference = pilots.gather(-1, index)
    phase = torch.where(reference.abs() > 0, reference / reference.abs().clamp_min(1.e-12),
                        torch.ones_like(reference))
    factor = rms * phase
    return pilots / factor, factor


def torch_linear(pilots, indices, frequencies):
    """Complex linear interpolation/extrapolation from pilots, without targets."""
    i = torch.as_tensor(indices, device=pilots.device, dtype=torch.long)
    f = torch.as_tensor(frequencies, device=pilots.device, dtype=torch.float64)
    if (pilots.ndim != 2 or i.ndim != 1 or pilots.shape[-1] != len(i) or len(i) < 2
            or f.ndim != 1 or not torch.isfinite(f).all() or not (f.diff() > 0).all()
            or not (i.diff() > 0).all() or i[0] < 0 or i[-1] >= len(f)):
        raise ValueError('Ordered physical frequencies and distinct ordered pilots are required')
    lower = torch.searchsorted(f[i], f, right=True).sub(1).clamp(0, len(i)-2)
    weight = ((f-f[i[lower]])/(f[i[lower+1]]-f[i[lower]])).to(pilots.real.dtype)
    return pilots[:,lower]*(1-weight)+pilots[:,lower+1]*weight


class PilotCompletionModel(nn.Module):
    """Shared sparse pilot encoder + finite circular Fourier residual decoder."""
    def __init__(self, frequencies_hz, hidden=64, latent=24):
        super().__init__()
        f = torch.as_tensor(np.array(frequencies_hz, copy=True), dtype=torch.float64)
        if f.shape != (64,) or not torch.isfinite(f).all() or not (f.diff() > 0).all():
            raise ValueError('This registered model requires64 increasing actual sampled frequencies')
        df = f.diff()[0]
        if not torch.allclose(f.diff(), df.expand(63), atol=1.e-7, rtol=1.e-9):
            raise ValueError('Circular Fourier decoder requires the actual uniform frequency grid')
        self.register_buffer('frequencies_hz',f)
        self.register_buffer('canonical_delay_s',torch.arange(64,dtype=torch.float64)/(64*df))
        basis=torch.exp(-2j*torch.pi*self.canonical_delay_s[:,None]*(f-f[0])[None])
        self.register_buffer('fourier_basis',basis.to(torch.complex64))
        self.encoder=nn.Sequential(nn.Linear(64*3+3,hidden),nn.SiLU(),nn.Linear(hidden,latent))
        self.decoder=nn.Linear(latent,128)
        nn.init.zeros_(self.decoder.weight);nn.init.zeros_(self.decoder.bias)

    def encode(self,pilots,indices,carrier_hz,frequency_offsets_hz=None):
        p,factor=pilot_normalize(pilots)
        i=torch.as_tensor(indices,dtype=torch.long,device=p.device)
        if (p.shape[-1]!=len(i) or len(i)!=16 or len(i.unique())!=16 or i.min()<0 or i.max()>=64):
            raise ValueError('Exactly16 distinct registered pilot indices are required')
        dtype=self.encoder[0].weight.dtype
        sparse=torch.zeros((len(p),64),dtype=p.dtype,device=p.device);sparse[:,i]=p
        mask=torch.zeros((len(p),64),dtype=dtype,device=p.device);mask[:,i]=1
        carrier=torch.as_tensor(carrier_hz,dtype=torch.float64,device=p.device)
        if carrier.ndim==0:carrier=carrier.expand(len(p))
        if carrier.shape!=(len(p),) or not torch.isfinite(carrier).all() or (carrier<=0).any():
            raise ValueError('Known positive physical carrier per pilot row is required')
        f=self.frequencies_hz if frequency_offsets_hz is None else torch.as_tensor(
            frequency_offsets_hz,dtype=torch.float64,device=p.device)
        if (f.shape!=(64,) or not torch.isfinite(f).all() or not (f.diff()>0).all()
                or not torch.allclose(f.diff(),f.diff()[:1].expand(63),atol=1.e-7,rtol=1.e-9)):
            raise ValueError('Actual64-bin uniform physical query grid must remain explicit')
        context=torch.stack([(carrier/1.e9).log(),((f[1]-f[0])/1.e6).log().expand_as(carrier),
                             ((f[-1]-f[0])/1.e6).log().expand_as(carrier)],-1).to(dtype)
        features=torch.cat([sparse.real.to(dtype),sparse.imag.to(dtype),mask,context],-1)
        return self.encoder(features),factor,p

    def forward(self,pilots,indices,carrier_hz,frequency_offsets_hz=None):
        z,factor,normalized=self.encode(pilots,indices,carrier_hz,frequency_offsets_hz)
        coefficient=self.decoder(z).reshape(-1,64,2).contiguous()
        coefficient=torch.view_as_complex(coefficient)
        f=self.frequencies_hz if frequency_offsets_hz is None else torch.as_tensor(
            frequency_offsets_hz,dtype=torch.float64,device=pilots.device)
        tau=torch.arange(64,dtype=torch.float64,device=pilots.device)/(64*(f[1]-f[0]))
        basis=torch.exp(-2j*torch.pi*tau[:,None]*(f-f[0])[None]).to(coefficient.dtype)
        residual=coefficient@basis
        anchor=torch_linear(normalized,indices,f)
        return (anchor+residual)*factor,z,coefficient


def matched_pair(frequencies_hz,seed):
    """Exact architecture/initialization match; only the alignment weight differs."""
    torch.manual_seed(int(seed))
    model=PilotCompletionModel(frequencies_hz)
    return model,deepcopy(model)


def common_gauge_alignment(physical_a,physical_b,training_pilot_scale):
    """Differentiate both same-packet fields after undoing pilot-only gauges."""
    if not math.isfinite(training_pilot_scale) or training_pilot_scale<=0:
        raise ValueError('Alignment requires one frozen TRAIN-pilot scale')
    a=torch.fft.ifft(physical_a,dim=-1)/training_pilot_scale
    b=torch.fft.ifft(physical_b,dim=-1)/training_pilot_scale
    return (a-b).abs().square().mean()


def objective(model,complete,carrier_hz,pilot_a,pilot_b,target,*,joint,training_pilot_scale):
    """Held frequencies are supervision only; neither encoder receives their values."""
    a,za,ca=model(complete[:,pilot_a],pilot_a,carrier_hz)
    b,zb,cb=model(complete[:,pilot_b],pilot_b,carrier_hz)
    _,fa=pilot_normalize(complete[:,pilot_a]);_,fb=pilot_normalize(complete[:,pilot_b])
    truth=complete[:,target]
    supervised=(((a[:,target]-truth)/fa).abs().square().mean()
                +((b[:,target]-truth)/fb).abs().square().mean())/2
    # Undo BOTH pilot-dependent transforms before comparing same-packet fields.
    # The first-frequency-reference Fourier coefficients share the receiver's
    # actual raw gauge. Neither branch is detached; there is NO EMA teacher.
    # The IFFT coordinates remain circular finite-band coefficients, not ToF.
    latent=common_gauge_alignment(a,b,training_pilot_scale)
    z=torch.cat([za,zb],0)
    variance=F.relu(.5-torch.sqrt(z.var(0,unbiased=False)+1.e-4)).mean()
    center=z-z.mean(0);covariance=center.T@center/max(len(z)-1,1)
    covariance=covariance-torch.diag(covariance.diag())
    covariance_loss=covariance.square().sum()/z.shape[1]
    regularizer=(ca.abs().square().mean()+cb.abs().square().mean())/2
    # Anti-collapse/complex decoder regularization is identical in both arms.
    # The sole controlled difference is symmetric common-gauge alignment.
    total=supervised+1.e-5*regularizer+.05*variance+.001*covariance_loss
    if joint:total=total+.1*latent
    return total,{'supervised':supervised.detach(),'latent':latent.detach(),
                  'variance':variance.detach(),'covariance':covariance_loss.detach()}


def latent_diagnostics(z):
    """Collapse diagnostics, never a service quality or scientific novelty test."""
    z=np.asarray(z,dtype='f8');center=z-z.mean(0)
    singular=np.linalg.svd(center,compute_uv=False)
    energy=singular**2
    probabilities=energy/energy.sum() if energy.sum()>0 else np.zeros_like(energy)
    effective=float(np.exp(-(probabilities[probabilities>0]*np.log(probabilities[probabilities>0])).sum())) if energy.sum()>0 else 0.
    return {'mean_latent_std':float(z.std(0).mean()),'effective_rank':effective,
            'latent_dimensions':int(z.shape[1]),'rows':int(len(z))}
