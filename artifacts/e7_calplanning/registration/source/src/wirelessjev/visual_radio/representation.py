"""Spatial DINO/RF fusion with explicit grounding and physical-feature bypass.

This trainable adapter is untrained unless a caller fits it. It preserves the
AP/band axes and measured factors; no architecture can certify non-collapse or
derive RF attenuation, depth or hidden activity from image appearance alone.
"""
import math
import torch
from torch import nn


def _clean(values,mask,name):
    if mask.shape!=values.shape:raise ValueError(name+' feature/mask axes differ')
    if mask.dtype!=torch.bool:raise ValueError(name+' mask must be boolean')
    clean=torch.where(mask,values,0.)
    if not torch.isfinite(clean).all():raise ValueError(name+' observed values must be finite')
    return clean


class GroundedLinkEncoder(nn.Module):
    def __init__(self,visual_dim,radio_dim=12,hidden=48):
        super().__init__()
        if min(visual_dim,radio_dim,hidden)<1:raise ValueError('Positive feature dimensions required')
        self.visual_dim,self.radio_dim,self.hidden=visual_dim,radio_dim,hidden
        self.radio_input=nn.Sequential(nn.Linear(2*radio_dim+1,hidden),nn.SiLU())
        self.visual_input=nn.Linear(visual_dim+3,hidden)
        self.key=nn.Linear(hidden,hidden,bias=False)
        self.query=nn.Linear(hidden,hidden,bias=False)
        self.memory=nn.GRU(2*hidden+1,hidden,batch_first=True)
        self.norm=nn.LayerNorm(hidden)

    def forward(self,radio,radio_observed,elapsed,patches,patch_observed,patch_xy,
                grounding_weights,visual_age_seconds,*,grounding_provenance,grounding_calibration_id=None):
        """One receiver per batch item; public AP × band histories stay distinct.

        `grounding_weights[B,T,AP,3,P]` comes from an explicit calibrated
        projection or manual image ROI, never from hidden simulated blockage.
        Manual ROIs supply image locality only, not metric/Fresnel geometry.
        Patch timestamps must already be causally aligned by the data adapter.
        """
        if grounding_provenance not in ('calibrated_camera_geometry','manual_image_regions'):
            raise ValueError('Grounding needs explicit calibration/ROI provenance')
        if grounding_provenance=='calibrated_camera_geometry' and not grounding_calibration_id:
            raise ValueError('Metric grounding needs the actual camera calibration identity')
        if radio.ndim!=5 or radio.shape[3:]!=(3,self.radio_dim):raise ValueError('Radio must be B x T x AP x 3 x F')
        b,t,s,_,_=radio.shape
        if min(b,t,s)<1 or elapsed.shape!=(b,t) or not torch.isfinite(elapsed).all() or (elapsed<0).any():raise ValueError('Invalid observation clock')
        if patches.ndim!=4 or patches.shape[:2]!=(b,t) or patches.shape[-1]!=self.visual_dim:raise ValueError('Patches must be B x T x P x D')
        p=patches.shape[2]
        if p<1 or patch_observed.shape!=(b,t,p) or patch_observed.dtype!=torch.bool:raise ValueError('Invalid patch visibility')
        if patch_xy.shape!=(b,t,p,2):raise ValueError('Retain normalized image patch coordinates')
        if visual_age_seconds.shape!=(b,t,p):raise ValueError('Retain actual image capture age per patch')
        valid_xy=torch.where(patch_observed[...,None],patch_xy,0.)
        age=torch.where(patch_observed,visual_age_seconds,0.)
        if not torch.isfinite(age).all() or (age<0).any():raise ValueError('Observed capture ages must be finite and nonnegative')
        if not torch.isfinite(valid_xy).all() or (valid_xy<0).any() or (valid_xy>1).any():raise ValueError('Observed image coordinates must be in [0,1]')
        if grounding_weights.shape!=(b,t,s,3,p) or not torch.isfinite(grounding_weights).all() or (grounding_weights<0).any():raise ValueError('Invalid public patch-to-link grounding')
        if any(not x.is_floating_point() or x.dtype!=radio.dtype or x.device!=radio.device
               for x in (radio,patches,patch_xy,grounding_weights,elapsed,visual_age_seconds)):
            raise ValueError('RF, visual and geometry tensors need a common floating dtype/device')
        rf=_clean(radio,radio_observed,'Radio')
        visual=_clean(patches,patch_observed[...,None].expand_as(patches),'Visual')
        dt=torch.log1p(elapsed)[:,:,None,None,None].expand(b,t,s,3,1)
        q=self.radio_input(torch.cat([rf,radio_observed.to(rf.dtype),dt],-1))
        v=self.visual_input(torch.cat([visual,valid_xy,torch.log1p(age)[...,None]],-1))
        # Exclude unavailable patches before either attention normalization or
        # value aggregation; an empty ROI stays absent rather than NaN.
        weight=grounding_weights*patch_observed[:,:,None,None]
        visible=weight.sum(-1)>0
        scores=torch.einsum('btskd,btpd->btskp',self.query(q),self.key(v))/math.sqrt(self.hidden)
        scores=scores+weight.clamp(min=torch.finfo(weight.dtype).tiny).log()
        scores=scores.masked_fill(weight<=0,-torch.inf)
        scores=torch.where(visible[...,None],scores,0.)
        attention=torch.softmax(scores,-1)*visible[...,None]
        attended=torch.einsum('btskp,btpd->btskd',attention,v)
        tokens=torch.cat([q,attended,visible[...,None].to(q.dtype)],-1)
        sequence=tokens.permute(0,2,3,1,4).reshape(b*s*3,t,-1)
        _,last=self.memory(sequence)
        latent=self.norm(last[0].reshape(b,s,3,self.hidden))
        return {'link_latent':latent,'spatial_tokens':v,'patch_xy':valid_xy,
                'patch_observed':patch_observed,'attention':attention,'visual_present':visible,
                'visual_age_seconds':age,'grounding_calibration_id':grounding_calibration_id,
                'physical_history':rf,'physical_observed':radio_observed,
                'grounding_provenance':grounding_provenance,
                'physical_information_preserved':'Explicit measured-feature bypass; learned latents still require validation'}


class GroundedServiceHead(nn.Module):
    """Select AP AND band without pooling them; anchored, untrained residual."""
    def __init__(self,hidden=48,action_dim=12):
        super().__init__();self.action_dim=action_dim
        self.network=nn.Sequential(nn.Linear(hidden+action_dim,hidden),nn.SiLU(),nn.Linear(hidden,2))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,encoded,actions,ap_index,band_index,numerical_reference):
        links=encoded['link_latent'];b,s,k,d=links.shape
        if actions.ndim!=3 or actions.shape[0]!=b or actions.shape[-1]!=self.action_dim or not torch.isfinite(actions).all():raise ValueError('Invalid candidate action features')
        h=actions.shape[1]
        if h<1 or numerical_reference.shape!=(b,h,2) or not torch.isfinite(numerical_reference).all():raise ValueError('Invalid declared numerical reference')
        for index,limit in ((ap_index,s),(band_index,k)):
            if index.shape!=(b,h) or index.dtype not in (torch.int32,torch.int64) or (index<0).any() or (index>=limit).any():raise ValueError('Candidate AP/band outside supplied physical axes')
        selected=links[torch.arange(b,device=links.device)[:,None],ap_index,band_index]
        residual=self.network(torch.cat([selected,actions],-1))
        return {'mean':numerical_reference+residual,'residual':residual,
                'calibrated_risk':None,'trained_quality_established':False}
