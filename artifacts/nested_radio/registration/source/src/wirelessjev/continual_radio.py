"""Masked action-response identification; true RT coupling is never an input."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResponseJEPA(nn.Module):
    def __init__(self,prior,*,family='jepa',hidden=32):
        super().__init__()
        if family not in ('jepa','supervised'):raise ValueError('Unknown family')
        prior=torch.as_tensor(prior,dtype=torch.float32)
        if prior.shape!=(2,2) or not torch.isfinite(prior).all() or (prior<=0).any():raise ValueError('Positive 2x2 training prior required')
        self.config={'family':family,'hidden':hidden};self.family=family;self.register_buffer('prior',prior.clone())
        self.effect_encoder=nn.Sequential(nn.Linear(6,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.LayerNorm(hidden))
        self.geometry=nn.Sequential(nn.Linear(16,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.LayerNorm(hidden))
        self.memory=nn.GRUCell(hidden,hidden)
        self.coupling_head=nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,4))
        nn.init.zeros_(self.coupling_head[-1].weight);nn.init.zeros_(self.coupling_head[-1].bias)
        self.predictor=nn.Sequential(nn.Linear(2*hidden+2,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.LayerNorm(hidden))
        self.teacher=copy.deepcopy(self.effect_encoder);self.teacher.requires_grad_(False);self.teacher.eval()

    def train(self,mode=True):
        super().train(mode);self.teacher.eval();return self

    def forward(self,context,activity,power,mask,delta,budget=None):
        if context.ndim!=2 or context.shape[1]!=16:raise ValueError('Public context must be Bx16')
        b=len(context)
        if activity.ndim!=3 or activity.shape[0]!=b or activity.shape[2]!=2 or activity.shape[1]>2:raise ValueError('At most two BxKx2 probes')
        if power.shape!=activity.shape or mask.shape!=activity.shape or mask.dtype!=torch.bool:raise ValueError('Power/mask shape mismatch')
        if delta.shape!=(b,2) or not torch.isfinite(context).all() or not torch.isfinite(activity).all() or not torch.isfinite(delta).all():raise ValueError('Invalid geometry/actions')
        if budget is None:budget=torch.full((b,),activity.shape[1],dtype=torch.long,device=context.device)
        if budget.shape!=(b,) or (budget<0).any() or (budget>activity.shape[1]).any():raise ValueError('Invalid history budget')
        mask=mask & (torch.arange(activity.shape[1],device=context.device)[None,:,None]<budget[:,None,None])
        observed=torch.where(mask,power,0.)
        if not torch.isfinite(observed).all():raise ValueError('Observed power nonfinite')
        geom=self.geometry(context);state=torch.zeros_like(geom)
        for t in range(activity.shape[1]):
            token=torch.cat([activity[:,t],observed[:,t],mask[:,t].to(power.dtype)],-1)
            updated=self.memory(self.effect_encoder(token),state)
            state=torch.where((budget>t)[:,None],updated,state)
        coupling=self.prior[None]*torch.exp(self.coupling_head(torch.cat([geom,state],-1)).reshape(b,2,2).clamp(-12,12))
        response=torch.einsum('bij,bj->bi',coupling,delta)
        latent=self.predictor(torch.cat([geom,state,delta],-1))
        return {'response':response,'coupling':coupling,'latent':latent,'state':state}

    @property
    def inference_parameters(self):return sum(p.numel() for name,p in self.named_parameters() if not name.startswith('teacher.'))


def response_objective(model,batch):
    p=model(batch['context'],batch['activity'],batch['power'],batch['mask'],batch['delta'],batch.get('budget'))
    target=batch['future_effect'];mask=batch.get('future_mask',torch.ones_like(target,dtype=torch.bool))
    target=torch.where(mask,target,0.)
    if not torch.isfinite(target).all():raise ValueError('Observed target effect nonfinite')
    supervised=(F.huber_loss(p['response'],target,delta=.25,reduction='none')*mask).sum()/mask.sum().clamp(min=1)
    auxiliary=supervised.new_zeros(());variance=supervised.new_zeros(())
    if model.family=='jepa':
        with torch.no_grad():teacher=model.teacher(torch.cat([batch['delta'],target,mask.to(target.dtype)],-1))
        valid=mask.any(-1)
        auxiliary=(F.smooth_l1_loss(p['latent'],teacher,reduction='none').mean(-1)*valid).sum()/valid.sum().clamp(min=1)
        if len(target)>1:variance=F.relu(.5-torch.sqrt(p['latent'].var(0,correction=0)+1e-4)).mean()
    return supervised+.1*auxiliary+.1*variance,{'supervised':supervised.detach(),'jepa':auxiliary.detach(),'variance':variance.detach()}


@torch.no_grad()
def update_response_teacher(model,momentum=.99):
    if not 0<=momentum<=1:raise ValueError('EMA momentum outside [0,1]')
    for target,source in zip(model.teacher.parameters(),model.effect_encoder.parameters(),strict=True):target.lerp_(source,1-momentum)


def local_ridge(prior,activity,power,mask,*,noise_variance=.0001,prior_covariance=None):
    """Sequential Bayesian linear ID; unexcited directions retain uncertainty."""
    a=np.asarray(activity,dtype=float);y=np.asarray(power,dtype=float);m=np.asarray(mask,dtype=bool)
    mean=np.asarray(prior,dtype=float).copy()
    cov=np.broadcast_to(np.eye(2)*.25,(2,2,2)).copy() if prior_covariance is None else np.array(prior_covariance,dtype=float,copy=True)
    if mean.shape!=(2,2) or cov.shape!=(2,2,2) or a.ndim!=2 or a.shape[1]!=2 or y.shape!=a.shape or m.shape!=a.shape:raise ValueError('System-ID dimensions invalid')
    if noise_variance<=0:raise ValueError('Positive measurement noise required')
    for t,row in enumerate(a):
        for r in range(2):
            if not m[t,r]:continue
            if not np.isfinite(y[t,r]):raise ValueError('Observed power nonfinite')
            gain=cov[r]@row/(noise_variance+row@cov[r]@row)
            mean[r]+=gain*(y[t,r]-row@mean[r]);cov[r]-=np.outer(gain,row)@cov[r]
            cov[r]=(cov[r]+cov[r].T)*.5
    return np.maximum(mean,0.),cov


def masked_least_squares(activity,power,mask):
    a=np.asarray(activity,dtype=float);y=np.asarray(power,dtype=float);m=np.asarray(mask,dtype=bool)
    result=np.zeros((2,2));ranks=[]
    for r in range(2):
        x=a[m[:,r]];v=y[m[:,r],r]
        if len(x):
            result[r]=np.maximum(np.linalg.lstsq(x,v,rcond=None)[0],0.);ranks.append(int(np.linalg.matrix_rank(x)))
        else:ranks.append(0)
    return result,np.asarray(ranks)
