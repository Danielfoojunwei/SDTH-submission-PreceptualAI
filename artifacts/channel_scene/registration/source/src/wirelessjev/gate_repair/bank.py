"""Forecast-only bank: no evaluation labels can enter a roaming decision."""
from dataclasses import dataclass
import numpy as np
import torch
from ..contracted.runtime import DeviceInputs

FEATURES=('gain','paired_spread','current_snr','candidate_snr','current_route_change','candidate_route_change','current_heard','candidate_heard','current_disagreement','candidate_disagreement','current_age','candidate_age')


@dataclass
class ForecastBank:
    mean: np.ndarray
    std: np.ndarray
    member_scores: np.ndarray
    heard: np.ndarray
    observed: np.ndarray
    age: np.ndarray

    @property
    def score(self):return self.mean[:,:,:6].max(-1).mean(2)

    def features(self,ctx):
        d=ctx.decision;ar=np.arange(len(ctx.current));a=ctx.current;b=ctx.candidate
        members=self.member_scores[:,d];paired=(members[:,ar,b]-members[:,ar,a]).std(0,ddof=1)
        now=self.mean[d,:,0].max(-1);later=self.mean[d,:,5].max(-1)
        heard=self.heard[d];observed=self.observed[d];age=self.age[d]
        return np.column_stack([ctx.gain/20,paired/5,now[ar,a]/30,now[ar,b]/30,(later[ar,a]-now[ar,a])/10,(later[ar,b]-now[ar,b])/10,
            heard[ar,a],heard[ar,b],(now[ar,a]-observed[ar,a])*(heard[ar,a]>0)/10,(now[ar,b]-observed[ar,b])*(heard[ar,b]>0)/10,age[ar,a],age[ar,b]]).astype('f8')

    def uncertainty(self,ctx):
        d=ctx.decision;ar=np.arange(len(ctx.current));a=ctx.current;b=ctx.candidate
        sd=self.std[d,:,0];maxband=sd.max(-1);active=np.take_along_axis(sd,self.mean[d,:,0].argmax(-1)[...,None],-1)[...,0]
        member=self.member_scores[:,d];independent=member.std(0,ddof=1)
        return {'legacy_u':np.sqrt(maxband[ar,a]**2+maxband[ar,b]**2+1e-9),'active_u':np.sqrt(active[ar,a]**2+active[ar,b]**2+1e-9),
            'ensemble_pair_u':(member[:,ar,b]-member[:,ar,a]).std(0,ddof=1),'ensemble_independent_u':np.sqrt(independent[ar,a]**2+independent[ar,b]**2)}


@torch.no_grad()
def forecasts(models,inputs: DeviceInputs,intent,decisions,n,scale):
    if not isinstance(inputs,DeviceInputs):raise TypeError('Expected label-free DeviceInputs')
    means=[];stds=[];scores=[];S=inputs.context.shape[1]//3
    for start in range(0,len(inputs.context),128):
        sl=slice(start,start+128);ctx=torch.as_tensor(inputs.context[sl].astype('f4'));route=torch.as_tensor(inputs.planned_route[sl].astype('f4'));prior=torch.as_tensor(inputs.geometric_prior[sl])
        weights=torch.tensor(intent,dtype=torch.float32)[None].expand(len(ctx),-1)
        out=[m(ctx,route,prior,weights) for m in models];mus=torch.stack([p['mean'] for p in out]);variance=torch.stack([p['std'].square() for p in out]).mean(0)+mus.var(0,unbiased=True)
        means.append(mus.mean(0).numpy());stds.append(variance.sqrt().numpy()*scale)
        scores.append(mus[:,:,:6].reshape(len(models),len(ctx),6,S,3).max(-1).values.mean(2).numpy())
    mean=np.concatenate(means).reshape(decisions,n,9,S,3);std=np.concatenate(stds).reshape(mean.shape)
    member=np.concatenate(scores,axis=1).reshape(len(models),decisions,n,S)
    history=inputs.context.reshape(decisions,n,S,3,6,8);heard=history[...,2]>0
    fraction=heard.mean((-2,-1));observed=np.zeros((decisions,n,S),dtype='f4');age=np.ones_like(observed);found=np.zeros_like(observed,dtype=bool)
    for h in range(5,-1,-1):
        value=np.where(heard[...,h],history[...,h,0]*30,-np.inf).max(-1);take=~found&np.isfinite(value)
        observed[take]=value[take];age[take]=(5-h)/5;found|=take
    return ForecastBank(mean,std,member,fraction,observed,age)
