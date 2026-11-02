"""Compact single-member world models with cross-band and route-change losses."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import gamma
from ..native.phy import LinkCurves


def failure_table():
    curves=LinkCurves();grid=np.linspace(-40,60,501);fades=10*np.log10(gamma.ppf((np.arange(32)+.5)/32,2,scale=.5));rows=[]
    for band in ['2g','5g','6g']:
        def p(offset):return np.mean((1-curves.success(band,np.zeros((501,32),dtype=int),grid[:,None]+fades+offset))**3,axis=1)
        # Native AP-level blockage stationary probability; an explicit surrogate.
        blocked=(1.5/60/4)/(1.5/60/4+.5)
        rows.append((1-blocked)*p(0)+blocked*p(-15))
    return np.stack(rows).astype('f4')


class FrontierModel(nn.Module):
    def __init__(self,family='packet',hidden=32):
        super().__init__()
        if family not in {'distilled','packet','temporal'}:raise ValueError('Unknown frontier model')
        self.config={'family':family,'hidden':hidden}
        if family=='temporal':self.history=nn.GRU(8,hidden,batch_first=True)
        else:self.history=nn.Sequential(nn.Linear(48,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.LayerNorm(hidden))
        self.memory=nn.Sequential(nn.Linear(14,24),nn.SiLU())
        self.predictor=nn.Sequential(nn.Linear(hidden*3+24+11+2,64),nn.SiLU(),nn.Linear(64,hidden),nn.SiLU())
        self.head=nn.Linear(hidden,2);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        with torch.no_grad():self.head.bias[1]=2
        self.register_buffer('phy_failure',torch.tensor(failure_table()))

    def forward(self,b):
        B,L,_=b['context'].shape;H=b['query'].shape[1]
        if self.config['family']=='temporal':
            _,z=self.history(b['context'].reshape(B*L,6,8));z=z[0].reshape(B,L,-1)
        else:z=self.history(b['context'])
        across_bands=z.reshape(B,L//3,3,-1).mean(2).repeat_interleave(3,1)
        pooled=z.mean(1,keepdim=True).expand_as(z)
        memory=self.memory(torch.cat([b['memory'],b['count']],-1))
        local=torch.cat([z,across_bands,pooled,memory],-1)[:,None].expand(-1,H,-1,-1)
        features=torch.cat([local,b['query'],b['fleet_mean'][...,None]/30,b['physics'][...,None]/30],-1)
        h=self.head(self.predictor(features))
        return {'mean':b['fleet_mean']+15*h[...,0],'std':F.softplus(h[...,1])+.25}

    def phy_risk(self,snr):
        shape=snr.shape;v=snr.reshape(*shape[:-1],-1,3);t=((v+40)/.2).clamp(0,500)
        lo=t.long().clamp(max=499);weight=t-lo
        band=torch.arange(3,device=v.device).expand_as(lo)
        return ((1-weight)*self.phy_failure[band,lo]+weight*self.phy_failure[band,lo+1])


def objective(model,b):
    p=model(b);truth=b['truth'];mean=p['mean'];weight=torch.where(truth>0,1.,.2)
    absolute=(F.huber_loss(mean,truth,delta=3,reduction='none')*weight).mean()
    nll=((.5*((mean-truth)/p['std']).square()+p['std'].log())*weight).mean()
    if model.config['family']=='distilled':
        teacher=F.huber_loss(mean,b['teacher'],delta=3)
        return .5*absolute+.5*teacher+.1*nll
    delta=mean[:,:6]-mean[:,:1];target=truth[:,:6]-truth[:,:1]
    route=(F.huber_loss(delta,target,delta=2,reduction='none')*weight[:,:6]).mean()
    B,H,L=truth.shape
    score=mean[:,:6].reshape(B,6,L//3,3).max(-1).values.mean(1)
    actual=truth[:,:6].reshape(B,6,L//3,3).max(-1).values.mean(1)
    rank=F.huber_loss(score-score.mean(-1,keepdim=True),actual-actual.mean(-1,keepdim=True),delta=2)
    pm=model.phy_risk(mean[:,:6]).min(-1).values.max(1).values
    with torch.no_grad():pt=model.phy_risk(truth[:,:6]).min(-1).values.max(1).values
    important=torch.where(actual>=actual.max(-1,keepdim=True).values-10,1.,.1)
    packet=((torch.log(pm+1e-4)-torch.log(pt+1e-4)).square()*important).mean()
    return .3*absolute+1.5*route+.5*rank+.2*packet+.1*nll
