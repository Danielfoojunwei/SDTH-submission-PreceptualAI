"""Small action-conditioned graph with masked/JEPA pretraining ablations.

These are newly trained radio representations, not LWM or V-JEPA checkpoints.
All inputs are declared past observations; labels are isolated by the runner.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from ..warehouse_v2.model import mlp


def link_features(obs):
    n,m = len(obs['robot']),len(obs['ap'])
    history = obs['history'].permute(1,2,0,3).reshape(n,m,12)
    past = obs['past_xy'].permute(1,0,2).reshape(n,8)[:,None].expand(-1,m,-1)
    return torch.cat([history,past,obs['robot'][:,None].expand(-1,m,-1),
                      obs['ap'][None].expand(n,-1,-1)],-1)


class RadioModel(nn.Module):
    def __init__(self, *, graph=True, hidden=32):
        super().__init__()
        self.config={'graph':graph,'hidden':hidden}
        self.encoder=mlp(34,hidden,hidden)
        self.graph=graph
        self.message=mlp(hidden*2+1,hidden,hidden)
        self.output=mlp(hidden+8+5,hidden,5)
        self.pretext=nn.Linear(hidden,12)
        self.latent_predictor=mlp(hidden+3,hidden,hidden)
        self.text=nn.Linear(1024,hidden)
        self.value=mlp(hidden*2+16,hidden,1)
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)

    def forward(self,obs,plans,prior,semantic=None,contract_features=None):
        K,n=plans.shape;m=len(obs['ap']);hidden=self.config['hidden']
        embedding=self.encoder(link_features(obs))
        rows=torch.arange(n)[None].expand(K,-1)
        r=embedding[rows,plans]
        moved=(plans!=obs['current'][None]).float()[...,None]
        if self.graph:
            idx=plans[...,None].expand(-1,-1,hidden)
            total=torch.zeros(K,m,hidden).scatter_add_(1,idx,r)
            count=torch.zeros(K,m,1).scatter_add_(1,plans[...,None],torch.ones(K,n,1))
            shared=(total/count.clamp(min=1)).gather(1,idx)
            r=r+self.message(torch.cat([r,shared,moved],-1))
        route=obs['route'][None].expand(K,-1,-1,-1)
        relative=route-obs['robot'][None,None,:,:2]
        offsets=torch.arange(1,7).float()[None,:,None,None].expand(K,-1,n,-1)/6.
        delta=self.output(torch.cat([r[:,None].expand(-1,6,-1,-1),prior,route,relative,offsets],-1))
        median=(prior[...,0]+torch.tanh(delta[...,0])).clamp(0,1)
        low=(median-F.softplus(delta[...,3])).clamp(0,1)
        high=(median+F.softplus(delta[...,4])).clamp(0,1)
        bad=(prior[...,1]<obs['robot'][None,None,:,4]-1e-6)|(prior[...,2]<-.125)
        logits=torch.logit(torch.where(bad,.98,.02))+delta[...,1]
        result={'service_raw':median,'failure_logits':logits,'signal':prior[...,3]+delta[...,2],
                'quantiles':torch.stack([low,median,high],-1),'pooled':r.mean(1)}
        if semantic is not None:
            result['value']=self.value(torch.cat([result['pooled'],self.text(semantic).expand(K,-1),
                                                contract_features[None].expand(K,-1)],-1)).squeeze(-1)
        return result


def supervised_loss(prediction,target):
    q=prediction['quantiles'];error=target['service_raw'][...,None]-q
    levels=torch.tensor([.1,.5,.9])
    pinball=torch.maximum(levels*error,(levels-1)*error).mean()
    return (F.smooth_l1_loss(prediction['service_raw'],target['service_raw'],beta=.1)
            +F.binary_cross_entropy_with_logits(prediction['failure_logits'],target['failure_raw'])
            +.2*F.smooth_l1_loss(prediction['signal'],target['signal'],beta=.1)+pinball)
