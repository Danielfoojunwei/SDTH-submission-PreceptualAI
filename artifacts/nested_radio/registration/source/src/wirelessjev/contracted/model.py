"""A small device-side JEPA; upstream EMA recipe is reused with attribution."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..warehouse_v3.jepa import momentum_update


class DeviceJEPA(nn.Module):
    def __init__(self,hidden=32,no_route=False):
        super().__init__();self.config={'hidden':hidden,'no_route':no_route}
        self.encoder=nn.Sequential(nn.Linear(48,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.LayerNorm(hidden))
        self.target_encoder=copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():p.requires_grad=False
        self.predictor=nn.Sequential(nn.Linear(hidden*2+5+9,hidden*2),nn.SiLU(),nn.Linear(hidden*2,hidden))
        self.head=nn.Linear(hidden,2);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        with torch.no_grad():self.head.bias[1]=np.log(5.)

    def forward(self,ctx,route,prior,intent):
        z=self.encoder(ctx);H=route.shape[1]
        if self.config['no_route']:
            route=route[:,:1].expand(-1,H,-1,-1).clone();route[...,3]=torch.arange(H,device=route.device)[None,:,None]/8
            prior=prior[:,:1].expand(-1,H,-1)
        local=z[:,None].expand(-1,H,-1,-1);pooled=z.mean(1)[:,None,None].expand_as(local)
        mission=intent[:,None,None].expand(-1,H,z.shape[1],-1)/5
        latent=self.predictor(torch.cat([local,pooled,route,mission],-1));head=self.head(latent)
        return {'mean':prior+15*head[...,0],'std':head[...,1].clamp(-.5,3).exp(),'latent':latent,'context':z}


def candidate_utility(snr,intent,current,load):
    """Exact finite AP scoring from native.sim, including the intent knobs.

    It excludes future queue/stagger interactions; it is not packet-loss regret.
    """
    B,H,L=snr.shape;S=L//3;score=snr[:,:6].reshape(B,min(H,6),S,3).max(-1).values.mean(1)
    score=score-(.15+.2*intent[:,3,None])*load
    moved=torch.arange(S,device=snr.device)[None]!=current[:,None]
    return score-(5+1.5*intent[:,0,None])*moved


def regret_weights(predicted,truth):
    """Change in exact regret after correcting just one AP's predicted utility."""
    with torch.no_grad():
        B,S=truth.shape;rows=torch.arange(B,device=truth.device);idx=torch.arange(S,device=truth.device)
        regret=truth.max(-1).values[:,None]-truth
        original=regret[rows,predicted.argmax(-1)]
        hybrid=predicted[:,None].expand(-1,S,-1).clone();hybrid[:,idx,idx]=truth
        corrected=regret.gather(1,hybrid.argmax(-1));weight=1+5*(original[:,None]-corrected).abs()/truth.std(-1,keepdim=True).clamp(min=1)
        return weight.clamp(max=10)/weight.mean(-1,keepdim=True)


def loss(model,batch,intent,family):
    p=model(batch['ctx'],batch['route'],batch['prior'],intent)
    truth_u=candidate_utility(batch['truth'],intent,batch['current'].long(),batch['load'])
    pred_u=candidate_utility(p['mean'],intent,batch['current'].long(),batch['load'])
    weights=torch.ones_like(p['mean'])
    if family in {'regret','no_route'}:
        w=regret_weights(pred_u.detach(),truth_u);weights=w.repeat_interleave(3,-1)[:,None].expand_as(weights)
    z=(batch['target']-p['mean'])/p['std'];censor=(batch['bound']-p['mean'])/p['std']
    nll=torch.where(batch['heard'].bool(),.5*z.square()+p['std'].log(),-torch.special.log_ndtr(censor))
    mask=batch['valid']*weights;nll=(nll*mask).sum()/mask.sum().clamp(min=1)
    objective=nll
    if family!='scratch':
        with torch.no_grad():target=F.layer_norm(model.target_encoder(batch['future_ctx']), (model.config['hidden'],))
        # Meta V-JEPA power/L1 prediction recipe, per-link weighted for regret.
        pretext=((p['latent']-target).abs().mean(-1)*weights).mean()
        variance=F.relu(.3-p['context'].flatten(0,1).std(0)).mean()
        objective=objective+.2*pretext+.2*variance
    if family in {'regret','no_route'}:
        regrets=(truth_u.max(-1).values[:,None]-truth_u)/10
        objective=objective+.2*(pred_u.softmax(-1)*regrets).sum(-1).mean()
    return objective,nll.detach()


@torch.no_grad()
def update_target(model):momentum_update(model.encoder,model.target_encoder,.99)
