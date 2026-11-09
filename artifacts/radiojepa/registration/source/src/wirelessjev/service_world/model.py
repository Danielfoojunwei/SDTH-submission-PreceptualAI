"""Physics-anchored radio belief dynamics; established components, no novelty claim.

Shared per-link memory preserves AP/band structure. Action-conditioned latent
distributions and optional JEPA training are separately testable. Predicted
uncertainties are estimates, not certified service or control-risk bounds.
"""
from __future__ import annotations
import copy
from collections.abc import Mapping
import torch
from torch import nn
from torch.nn import functional as F


def _masked(values,validity,name):
    if validity is None:validity=torch.ones_like(values,dtype=torch.bool)
    else:
        validity=torch.as_tensor(validity,device=values.device).bool()
        if validity.shape==values.shape[:-1]:validity=validity[...,None].expand_as(values)
        elif validity.shape!=values.shape:raise ValueError(f'{name} validity has incompatible dimensions')
    clean=torch.where(validity,values,0.)
    if not torch.isfinite(clean).all():raise ValueError(f'{name} has a nonfinite feature marked observed')
    return clean,validity


class RadioBeliefEncoder(nn.Module):
    """Causal shared link memory and known-edge messages, without AP-ID embeddings.

    Streaming state is for NEW nonoverlapping chunks. Sliding windows must use
    state=None to avoid double assimilation. Elapsed times are physical seconds.
    """
    def __init__(self,link_dim=12,global_dim=8,hidden=48):
        super().__init__();self.link_dim,self.global_dim,self.hidden=link_dim,global_dim,hidden
        self.link_input=nn.Sequential(nn.Linear(2*link_dim+1,hidden),nn.SiLU())
        self.link_memory=nn.GRU(hidden,hidden,batch_first=True)
        self.message=nn.Linear(hidden,hidden,bias=False);self.link_norm=nn.LayerNorm(hidden)
        self.belief_memory=nn.GRU(2*hidden+2*global_dim+1,hidden,batch_first=True)
        self.belief_norm=nn.LayerNorm(hidden)

    def forward(self,links,globals,observed=None,elapsed=None,*,global_observed=None,adjacency=None,state=None):
        if links.ndim!=5 or links.shape[3]!=3 or links.shape[-1]!=self.link_dim:
            raise ValueError('Links must be B x T x AP x 3 x link_dim')
        b,t,s,_,_=links.shape
        if min(b,t,s)<1 or globals.shape!=(b,t,self.global_dim):raise ValueError('Invalid global history or empty radio axes')
        clean,mask=_masked(links,observed,'Radio history')
        gc,gm=_masked(globals,global_observed,'Global history')
        if elapsed is None:elapsed=links.new_ones(b,t)
        if elapsed.shape!=(b,t) or not torch.isfinite(elapsed).all() or (elapsed<0).any():
            raise ValueError('Elapsed time must be finite nonnegative B x T seconds')
        dt=torch.log1p(elapsed)
        x=torch.cat([clean,mask.to(links.dtype),dt[:,:,None,None,None].expand(b,t,s,3,1)],-1)
        x=self.link_input(x).permute(0,2,3,1,4).reshape(b*s*3,t,self.hidden)
        il=None;ig=None;seen_before=torch.zeros((b,s,3),device=links.device,dtype=torch.bool)
        if state is not None:
            if (state['link'].shape!=(b,s,3,self.hidden) or state['belief'].shape!=(b,self.hidden)
                    or state['seen'].shape!=(b,s,3)):
                raise ValueError('Streaming state has different fleet/AP dimensions')
            if not torch.isfinite(state['link']).all() or not torch.isfinite(state['belief']).all():
                raise ValueError('Streaming belief state must be finite')
            il=state['link'].reshape(b*s*3,self.hidden)[None];ig=state['belief'][None]
            seen_before=state['seen'].bool()
        sequence,final=self.link_memory(x,il)
        raw=sequence.reshape(b,s,3,t,self.hidden).permute(0,3,1,2,4)
        seen=mask.any(-1).long().cumsum(1).bool()|seen_before[:,None]
        if adjacency is None:weight=links.new_ones(b,3,s,s)
        else:
            weight=torch.as_tensor(adjacency,device=links.device,dtype=links.dtype)
            if weight.shape==(s,s):weight=weight[None,None].expand(b,3,s,s)
            elif weight.shape==(b,s,s):weight=weight[:,None].expand(b,3,s,s)
            if weight.shape!=(b,3,s,s) or not torch.isfinite(weight).all() or (weight<0).any():
                raise ValueError('Adjacency must be finite nonnegative B x 3 x AP x AP')
        weights=weight[:,None]*seen.permute(0,1,3,2)[:,:,:,None,:]
        weights=weights/weights.sum(-1,keepdim=True).clamp(min=1e-6)
        neighbor=torch.einsum('btksr,btrkd->btskd',weights,raw)
        graph=self.link_norm(raw+self.message(neighbor));active=seen[...,None]
        mean=(graph*active).sum((2,3))/active.sum((2,3)).clamp(min=1)
        maximum=graph.masked_fill(~active,-torch.inf).amax((2,3))
        maximum=torch.where(seen.any((2,3))[...,None],maximum,0.)
        pooled=torch.cat([mean,maximum,gc,gm.to(links.dtype),dt[...,None]],-1)
        belief_sequence,belief_final=self.belief_memory(pooled,ig)
        return {'belief':self.belief_norm(belief_sequence[:,-1]),'links':graph[:,-1],
            'sequence':self.belief_norm(belief_sequence),
            'valid':mask.any((1,2,3,4))|gm.any((1,2)),
            'state':{'link':final[0].reshape(b,s,3,self.hidden),'belief':belief_final[0],'seen':seen[:,-1]}}


class ServiceWorldModel(nn.Module):
    """Identical supervised/JEPA inference graphs and action-conditioned rollouts.

    physics_mean[B,H,2] carries numerical miss-count/gap values OR paired
    differences, in units declared by the caller. It is not a probability.
    hazard_prior[B,H] is an optional conditional interval-event probability;
    omitted priors return hazard=None and survival=None, not invented risk.
    Zero residuals preserve the supplied numerical means/priors exactly.
    Default rollouts use latent means; explicit noise samples stochastic
    futures reproducibly. Neither latent nor service std is a safety bound.
    """
    def __init__(self,family='jepa',link_dim=12,global_dim=8,action_dim=12,hidden=48,
                 target_scale=(20.,1.),jepa_weight=.1,variance_weight=.1,ema_momentum=.99):
        super().__init__()
        if family not in ('jepa','supervised') or min(link_dim,global_dim,action_dim,hidden)<1:
            raise ValueError('Invalid model family or dimensions')
        scale=torch.tensor(target_scale,dtype=torch.float32)
        if scale.shape!=(2,) or not torch.isfinite(scale).all() or (scale<=0).any():
            raise ValueError('Use finite positive miss-count/gap target scales')
        if min(jepa_weight,variance_weight)<0 or not 0<=ema_momentum<1:
            raise ValueError('Invalid objective or EMA settings')
        self.config=dict(family=family,link_dim=link_dim,global_dim=global_dim,action_dim=action_dim,
            hidden=hidden,target_scale=scale.tolist(),jepa_weight=jepa_weight,
            variance_weight=variance_weight,ema_momentum=ema_momentum)
        self.encoder=RadioBeliefEncoder(link_dim,global_dim,hidden)
        self.target_encoder=copy.deepcopy(self.encoder);self.target_encoder.requires_grad_(False)
        self.action_input=nn.Sequential(nn.Linear(hidden+action_dim+1,hidden),nn.SiLU())
        self.transition=nn.GRUCell(hidden,hidden);self.latent_distribution=nn.Linear(hidden,2*hidden)
        self.decoder=nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU())
        self.service_head=nn.Linear(hidden,4);self.hazard_head=nn.Linear(hidden,1)
        for head in (self.service_head,self.hazard_head):nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.register_buffer('target_scale',scale);self.target_encoder.eval()

    def train(self,mode=True):
        super().train(mode);self.target_encoder.eval();return self

    def encode(self,links,globals,observed=None,elapsed=None,**kwargs):
        return self.encoder(links,globals,observed,elapsed,**kwargs)

    def forward(self,links,globals,observed,elapsed,actions,ap_index,physics_mean,hazard_prior=None,*,
                global_observed=None,adjacency=None,action_elapsed=None,state=None,latent_noise=None):
        encoded=self.encoder(links,globals,observed,elapsed,global_observed=global_observed,adjacency=adjacency,state=state)
        return self.rollout(encoded,actions,ap_index,physics_mean,hazard_prior,
                            action_elapsed=action_elapsed,latent_noise=latent_noise)

    def rollout(self,encoded,actions,ap_index,physics_mean,hazard_prior=None,*,action_elapsed=None,latent_noise=None):
        """Reuse one encoded public history for multiple candidate sequences."""
        b=encoded['belief'].shape[0];s=encoded['links'].shape[1];d=self.config['hidden']
        if actions.ndim!=3 or actions.shape[0]!=b or actions.shape[-1]!=self.config['action_dim']:
            raise ValueError('Actions must be B x H x action_dim')
        h=actions.shape[1]
        if h<1 or physics_mean.shape!=(b,h,2) or not torch.isfinite(actions).all() or not torch.isfinite(physics_mean).all():
            raise ValueError('Invalid action horizon or numerical service prior')
        ap_index=torch.as_tensor(ap_index,device=encoded['belief'].device)
        if ap_index.shape==(b,):ap_index=ap_index[:,None].expand(b,h)
        if ap_index.shape!=(b,h) or ap_index.dtype not in (torch.int32,torch.int64) or (ap_index<0).any() or (ap_index>=s).any():
            raise ValueError('AP indices must be integer B or B x H within the public AP list')
        if action_elapsed is None:action_elapsed=actions.new_ones(b,h)
        if action_elapsed.shape!=(b,h) or not torch.isfinite(action_elapsed).all() or (action_elapsed<0).any():
            raise ValueError('Action elapsed times must be finite nonnegative B x H seconds')
        if hazard_prior is not None and (hazard_prior.shape!=(b,h) or not torch.isfinite(hazard_prior).all()
                                         or (hazard_prior<0).any() or (hazard_prior>1).any()):
            raise ValueError('Hazard prior must be probability B x H')
        if latent_noise is not None and (latent_noise.shape!=(b,h,d) or not torch.isfinite(latent_noise).all()):
            raise ValueError('Explicit rollout noise must be finite B x H x hidden')
        belief=encoded['belief'];means=[];stds=[];hazards=[];latents=[];latent_stds=[]
        ar=torch.arange(b,device=encoded['belief'].device)
        for step in range(h):
            selected=encoded['links'][ar,ap_index[:,step]].mean(1)
            action=self.action_input(torch.cat([selected,actions[:,step],torch.log1p(action_elapsed[:,step,None])],-1))
            hidden=self.transition(action,belief);mu,raw_std=self.latent_distribution(hidden).chunk(2,-1)
            latent_std=F.softplus(raw_std)+.05
            belief=mu if latent_noise is None else mu+latent_std*latent_noise[:,step]
            feature=self.decoder(torch.cat([belief,selected],-1));raw=self.service_head(feature)
            means.append(physics_mean[:,step]+raw[:,:2]*self.target_scale)
            stds.append((F.softplus(raw[:,2:])+.02)*self.target_scale)
            if hazard_prior is not None:
                prior=hazard_prior[:,step];logit=torch.logit(prior.clamp(1e-6,1-1e-6))
                change=torch.sigmoid(logit+self.hazard_head(feature)[:,0])-torch.sigmoid(logit)
                hazards.append((prior+change).clamp(0,1))
            latents.append(mu);latent_stds.append(latent_std)
        mean=torch.stack(means,1);std=torch.stack(stds,1)
        hazard=torch.stack(hazards,1) if hazard_prior is not None else None
        return {'mean':mean,'std':std,'residual':mean-physics_mean,
            'miss_value':mean[...,0],'gap_value':mean[...,1],
            'delta_misses':mean[...,0],'delta_gap':mean[...,1],'miss_std':std[...,0],'gap_std':std[...,1],
            'hazard':hazard,'survival':None if hazard is None else torch.cumprod(1-hazard,1),
            'hazard_prior_supplied':hazard_prior is not None,
            'latent':torch.stack(latents,1),'latent_std':torch.stack(latent_stds,1),
            'belief':encoded['belief'],'link_latent':encoded['links'],'state':encoded['state']}

    @property
    def inference_parameters(self):
        return sum(p.numel() for name,p in self.named_parameters() if not name.startswith('target_encoder.'))


_INPUT_KEYS=('links','globals','observed','elapsed','actions','ap_index','physics_mean',
             'hazard_prior','global_observed','adjacency','action_elapsed','state','latent_noise')


def objective(model:ServiceWorldModel,batch:Mapping[str,torch.Tensor],phase='joint'):
    """Native value learning, with a separate label-free JEPA pretrain phase.

    target_mean[B,H,2] or delta_misses/delta_gap[B,H] carry native outcomes.
    Optional binary target_hazard[B,H] requires an explicit hazard prior.
    Future observable windows are future_links[B,H,T,AP,3,F] and
    future_globals[B,H,T,G], with matching masks/elapsed times. They enter only
    the EMA target encoder during training, never the forward inference API.
    """
    if phase not in ('joint','pretrain','finetune'):raise ValueError('Unknown training phase')
    if phase=='pretrain' and model.config['family']!='jepa':
        raise ValueError('The no-JEPA ablation has no JEPA pretraining phase')
    p=model(**{k:batch[k] for k in _INPUT_KEYS if k in batch})
    zero=p['latent'].sum()*0;supervised=zero;pretext=zero;variance=zero;hazard_loss=zero
    if phase!='pretrain':
        target=batch.get('target_mean')
        if target is None:target=torch.stack([batch['delta_misses'],batch['delta_gap']],-1)
        if target.shape!=p['mean'].shape:raise ValueError('Service labels must match B x H x 2')
        target,valid=_masked(target,batch.get('target_valid'),'Service labels')
        error=(p['mean']-target)/model.target_scale;sd=p['std']/model.target_scale
        denominator=valid.sum().clamp(min=1)
        huber=(F.huber_loss(error,torch.zeros_like(error),delta=1,reduction='none')*valid).sum()/denominator
        nll=((.5*(error/sd).square()+sd.log())*valid).sum()/denominator
        supervised=huber+.1*nll
        if 'target_hazard' in batch:
            if p['hazard'] is None:raise ValueError('Hazard supervision requires an explicit numerical hazard prior')
            y=batch['target_hazard'];mask=batch.get('hazard_valid',torch.ones_like(y,dtype=torch.bool)).bool()
            if y.shape!=p['hazard'].shape or mask.shape!=y.shape:raise ValueError('Hazard labels must be B x H')
            y=torch.where(mask,y,0.)
            if not torch.isfinite(y).all() or (y<0).any() or (y>1).any():raise ValueError('Invalid event probability label')
            loss=F.binary_cross_entropy(p['hazard'].clamp(1e-6,1-1e-6),y,reduction='none')
            hazard_loss=(loss*mask).sum()/mask.sum().clamp(min=1)
            supervised=supervised+.1*hazard_loss
    if model.config['family']=='jepa' and phase!='finetune':
        future=batch['future_links'];b,h=batch['actions'].shape[:2]
        if future.ndim!=6 or future.shape[:2]!=(b,h):
            raise ValueError('Future radio windows must be B x H x T x AP x 3 x F')
        fv=batch.get('future_valid',torch.ones((b,h),device=future.device,dtype=torch.bool)).bool()
        if fv.shape!=(b,h):raise ValueError('Future validity must be B x H')
        fm=batch.get('future_observed')
        if fm is None:fm=torch.ones_like(future,dtype=torch.bool)
        elif fm.shape==future.shape[:-1]:fm=fm[...,None].expand_as(future)
        elif fm.shape!=future.shape:raise ValueError('Future radio validity has incompatible dimensions')
        fm=fm.bool() & fv[:,:,None,None,None,None]
        fg=batch['future_globals'];gm=batch.get('future_global_observed')
        if fg.ndim!=4 or fg.shape[:2]!=(b,h):raise ValueError('Future globals must be B x H x T x G')
        if gm is None:gm=torch.ones_like(fg,dtype=torch.bool)
        elif gm.shape==fg.shape[:-1]:gm=gm[...,None].expand_as(fg)
        elif gm.shape!=fg.shape:raise ValueError('Future global validity has incompatible dimensions')
        gm=gm.bool() & fv[:,:,None,None]
        dt=batch.get('future_elapsed')
        if dt is not None:
            if dt.shape!=future.shape[:3]:raise ValueError('Future elapsed times must be B x H x T')
            dt=torch.where(fv[:,:,None],dt,0.).reshape(b*h,-1)
        adj=batch.get('adjacency')
        if adj is not None and adj.ndim>=3:
            adj=adj[:,None].expand(b,h,*adj.shape[1:]).reshape(b*h,*adj.shape[1:])
        with torch.no_grad():
            target=model.target_encoder(future.reshape(b*h,*future.shape[2:]),fg.reshape(b*h,*fg.shape[2:]),
                fm.reshape(b*h,*fm.shape[2:]),dt,
                global_observed=gm.reshape(b*h,*gm.shape[2:]),adjacency=adj)
            latent=target['belief'].reshape(b,h,-1);valid=target['valid'].reshape(b,h)&fv
        latent_error=(.5*((p['latent']-latent)/p['latent_std']).square()+p['latent_std'].log()).mean(-1)
        pretext=(latent_error*valid).sum()/valid.sum().clamp(min=1)
        if b>1:variance=F.relu(.5-torch.sqrt(p['belief'].var(0,correction=0)+1e-4)).mean()
    total=supervised+model.config['jepa_weight']*pretext+model.config['variance_weight']*variance
    return total,{'supervised':supervised.detach(),'hazard':hazard_loss.detach(),
                  'jepa':pretext.detach(),'variance':variance.detach()}


@torch.no_grad()
def update_target(model:ServiceWorldModel,momentum=None):
    momentum=model.config['ema_momentum'] if momentum is None else momentum
    if not 0<=momentum<=1:raise ValueError('EMA momentum outside [0,1]')
    for target,online in zip(model.target_encoder.parameters(),model.encoder.parameters(),strict=True):
        target.lerp_(online,1-momentum)
    for target,online in zip(model.target_encoder.buffers(),model.encoder.buffers(),strict=True):target.copy_(online)
