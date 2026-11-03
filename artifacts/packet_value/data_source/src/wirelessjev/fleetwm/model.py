"""Geometry-conditioned fleet JEPA with distinct mean-field uncertainty."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from ..warehouse_v3.jepa import momentum_update


class FleetJEPA(nn.Module):
    def __init__(self,family='hybrid_jepa',hidden=48):
        super().__init__()
        if family not in {'hybrid_jepa','hybrid_supervised','fleet_jepa'}:
            raise ValueError('Unknown model family')
        self.config={'family':family,'hidden':hidden}
        self.encoder=nn.Sequential(nn.Linear(48,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.LayerNorm(hidden))
        self.target_encoder=copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters(): p.requires_grad=False
        self.memory_encoder=nn.Sequential(nn.Linear(13,24),nn.SiLU(),nn.Linear(24,24))
        self.predictor=nn.Sequential(nn.Linear(2*hidden+24+11+3,96),nn.SiLU(),nn.Linear(96,hidden),nn.LayerNorm(hidden))
        self.head=nn.Linear(hidden,2)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        with torch.no_grad(): self.head.bias[1]=2

    def forward(self,b):
        z=self.encoder(b['context']); H=b['query'].shape[1]
        local=z[:,None].expand(-1,H,-1,-1)
        valid=b.get('valid_link',torch.ones_like(z[...,0]))
        pool=(z*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp(min=1)
        pooled=pool[:,None,None].expand_as(local)
        memory=self.memory_encoder(b['memory'])[:,None].expand(-1,H,-1,-1)
        hybrid=self.config['family']!='fleet_jepa'
        base=b['map_mean'] if hybrid else b['fleet_mean']
        std=b['map_std'] if hybrid else torch.full_like(base,5)
        lat=self.predictor(torch.cat([local,pooled,memory,b['query'],
                    base[...,None]/30,std[...,None]/10,b['physics'][...,None]/30],-1))
        head=self.head(lat)
        return {'mean':base+15*head[...,0],'std':F.softplus(head[...,1])+.25,
                'latent':lat,'context':z}


def objective(model,b):
    p=model(b); error=p['mean']-b['truth']
    weight=torch.where(b['truth']>0,1.,.2)
    hw=torch.tensor([1,1,1,1,1,1,.5,.3,.2],device=error.device)[None,:,None]
    valid=b['valid_link'][:,None]
    w=weight*hw*valid
    signal=(F.huber_loss(p['mean'],b['truth'],delta=3,reduction='none')*w).sum()/w.sum()
    nll=((.5*(error/p['std']).square()+p['std'].log())*w).sum()/w.sum()
    B,H,L=error.shape
    score=p['mean'][:,:6].reshape(B,6,L//3,3).max(-1).values.mean(1)
    truth=b['truth'][:,:6].reshape(B,6,L//3,3).max(-1).values.mean(1)
    ap_valid=b['valid_link'][:,::3].bool()
    score=score.masked_fill(~ap_valid,-1e4)
    truth=truth.masked_fill(~ap_valid,-1e4)
    target=truth.max(-1).values[:,None]-truth
    regret=(torch.softmax(score/3,-1)*target).sum(-1).mean()
    score_center=(score*ap_valid).sum(-1,keepdim=True)/ap_valid.sum(-1,keepdim=True)
    truth_center=(truth*ap_valid).sum(-1,keepdim=True)/ap_valid.sum(-1,keepdim=True)
    relative=(F.huber_loss(score-score_center,truth-truth_center,delta=2,reduction='none')*ap_valid).sum()/ap_valid.sum()
    cost=signal+.15*nll+.5*regret+.1*relative
    if model.config['family']!='hybrid_supervised':
        with torch.no_grad(): target_z=model.target_encoder(b['future_context'])
        pretext=((p['latent']-target_z).abs().mean(-1)*valid).sum()/(valid.sum()*error.shape[1])
        variance=F.relu(.4-p['context'][b['valid_link'].bool()].std(0)).mean()
        cost=cost+.1*pretext+.1*variance
    return cost,{'signal_huber':signal.detach(),'ap_regret_soft':regret.detach(),'nll':nll.detach()}


@torch.no_grad()
def update_target(model):
    momentum_update(model.encoder,model.target_encoder,.99)
