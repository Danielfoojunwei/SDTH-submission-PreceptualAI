"""Shared physics and AP-local spatial field posteriors.

AP integers route blocks; they never enter a learned feature or embedding.
Gaussian point conditioning uses an exact block-Schur solve. Three bounded
Newton updates and observed-information covariance are explicitly approximate;
the working residual likelihood does not identify physical scatterers, make
repeated measurements independent, or calibrate packet risk.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..radiojepa.likelihoods import gaussian_interval_log_prob
from ..spectral_scene.operators import query_physics_basis

CELLS = ('hier_reg', 'hier_joint', 'global_reg', 'global8_reg', 'hier_erased')
GLOBAL_DIM = 8
LOCAL_DIM = 16
RFF_DIM = 8
RFF_SEED = 184910
JITTER = 1.e-8
ALIGNMENT_WEIGHT = .1
BACKTRACKING_STEPS = (1.,.5,.25,.125,.0625,.03125)
OBJECTIVE_TOLERANCE = 1.e-10
MAX_COVARIANCE_QUERIES = 512


def propagation_features(query):
    """Propagation coordinates only, excluding frontend and direction labels."""
    return torch.cat((query[..., :8], query[..., 9:11]), -1)


def _tensor(value, name, *, floating=True):
    if not isinstance(value, torch.Tensor) or value.is_complex():
        raise ValueError(name+' must be a real tensor')
    if floating and not value.is_floating_point():
        raise ValueError(name+' must be a floating tensor')
    return value


def _finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(name+' must be finite')


def _chol(matrix, name):
    matrix = .5*(matrix+matrix.transpose(-1, -2))
    factor, info = torch.linalg.cholesky_ex(matrix)
    if info.any() or not torch.isfinite(factor).all():
        raise FloatingPointError(name+' lost positive definiteness')
    return factor


def interval_moments(mean, std, lower, upper, *, point_mask=None):
    """Stable working-Gaussian truncated moments, with infinite endpoints.

    Clipping tiny negative conditional variances is a numerical safeguard,
    not an identified observation or a posterior calibration certificate.
    Inputs should be float64 when narrow bins or deep censor tails matter.
    """
    for value,name in ((mean,'mean'),(std,'std'),(lower,'lower'),(upper,'upper')):
        _tensor(value,name)
    mean,std,lower,upper = torch.broadcast_tensors(mean,std,lower,upper)
    _finite(mean,'mean'); _finite(std,'std')
    points=torch.zeros_like(mean,dtype=torch.bool) if point_mask is None else point_mask
    if points.shape!=mean.shape or points.dtype!=torch.bool or (points&(~torch.isfinite(lower)|(lower!=upper))).any():
        raise ValueError('Point observations need equal finite endpoints and an explicit point mask')
    if ((std <= 0).any() or torch.isnan(lower).any() or torch.isnan(upper).any()
            or ((lower>=upper)&~points).any()):
        raise ValueError('Use positive spread and nonempty ordered intervals')
    point_value=lower
    lower=torch.where(points,mean-std,lower);upper=torch.where(points,mean+std,upper)
    lo_finite,hi_finite = torch.isfinite(lower),torch.isfinite(upper)
    a = (torch.where(lo_finite,lower,mean)-mean)/std
    b = (torch.where(hi_finite,upper,mean)-mean)/std
    log_mass = gaussian_interval_log_prob(lower,upper,mean,std)
    ninf = torch.full_like(a,-torch.inf)
    la = torch.where(lo_finite,-.5*a.square()-.5*math.log(2*math.pi),ninf)
    lb = torch.where(hi_finite,-.5*b.square()-.5*math.log(2*math.pi),ninf)
    ra = torch.exp((la-log_mass).clamp(max=300.))
    rb = torch.exp((lb-log_mass).clamp(max=300.))
    shift = ra-rb
    expectation = torch.where(points,point_value,mean+std*shift)
    variance_fraction = (1.+a*ra-b*rb-shift.square()).clamp(0.,1.)
    variance = torch.where(points,torch.zeros_like(mean),std.square()*variance_fraction)
    if not torch.isfinite(expectation).all() or not torch.isfinite(variance).all():
        raise FloatingPointError('Nonfinite interval moments')
    return expectation,variance


def _block_factor(global_design, local_design, groups, weights,
                  global_precision, local_precision):
    """Factor one global block and independent AP blocks, exactly in float64."""
    batch,tokens,gdim = global_design.shape
    sites,ldim = local_precision.shape[1:3]
    membership = groups[:,None,:] == torch.arange(sites,device=groups.device)[None,:,None]
    routed_weight = weights[:,None,:]*membership
    ag = global_precision+torch.einsum('btg,bt,bth->bgh',global_design,weights,global_design)
    ca = local_precision+torch.einsum('btl,bst,btm->bslm',local_design,routed_weight,local_design)
    ba = torch.einsum('btg,bst,btl->bsgl',global_design,routed_weight,local_design)
    lc = _chol(ca,'AP precision')
    coupling = torch.cholesky_solve(ba.transpose(-1,-2),lc)
    schur = ag-torch.einsum('bsgl,bslh->bgh',ba,coupling)
    lg = _chol(schur,'global Schur complement')
    return {'global_cholesky':lg,'local_cholesky':lc,'coupling':coupling,
            'cross_precision':ba,'routed_weight':routed_weight}


def _block_natural_solution(global_design,local_design,groups,information_weight,natural_linear,
                            global_mean,global_precision,local_mean,local_precision):
    """Solve a block Gaussian natural system without invented pseudo values."""
    factor=_block_factor(global_design,local_design,groups,information_weight,global_precision,local_precision)
    sites=local_mean.shape[1]
    membership=groups[:,None,:]==torch.arange(sites,device=groups.device)[None,:,None]
    rg=torch.einsum('bgh,bh->bg',global_precision,global_mean)
    rg=rg+torch.einsum('btg,bt->bg',global_design,natural_linear)
    ra=torch.einsum('bslm,bsm->bsl',local_precision,local_mean)
    ra=ra+torch.einsum('btl,bst->bsl',local_design,membership*natural_linear[:,None,:])
    local_rhs=torch.cholesky_solve(ra[...,None],factor['local_cholesky']).squeeze(-1)
    rhs=rg-torch.einsum('bsgl,bsl->bg',factor['cross_precision'],local_rhs)
    posterior_global=torch.cholesky_solve(rhs[...,None],factor['global_cholesky']).squeeze(-1)
    posterior_local=local_rhs-torch.einsum('bslg,bg->bsl',factor['coupling'],posterior_global)
    return {'global_mean':posterior_global,'local_mean':posterior_local,
            'global_covariance':torch.cholesky_inverse(factor['global_cholesky']),**factor}


def block_gaussian_posterior(global_design, local_design, groups, weights, observed,
                             global_mean, global_precision, local_mean, local_precision):
    """Exact Gaussian point posterior with no sequential double conditioning.

    This public mathematical helper accepts positive definite prior precision
    matrices. It does not estimate those priors from the same observation rows.
    The learned model below calls it with fixed positive diagonal priors.
    """
    for value,name in ((global_design,'global design'),(local_design,'local design'),
            (weights,'weights'),(observed,'observations'),(global_mean,'global prior mean'),
            (global_precision,'global prior precision'),(local_mean,'local prior mean'),
            (local_precision,'local prior precision')):
        _tensor(value,name);_finite(value,name)
    _tensor(groups,'AP routes',floating=False)
    if global_design.ndim!=3 or local_design.ndim!=3 or global_design.shape[:2]!=local_design.shape[:2]:
        raise ValueError('Block Gaussian design axes disagree')
    batch,tokens,gdim=global_design.shape;ldim=local_design.shape[-1]
    if (batch<1 or gdim<1 or ldim<1 or local_mean.ndim!=3 or local_mean.shape[0]!=batch
            or local_mean.shape[-1]!=ldim or local_mean.shape[1]<1):
        raise ValueError('Positive global/local/AP dimensions required')
    sites=local_mean.shape[1]
    if (groups.shape!=(batch,tokens) or groups.dtype not in (torch.int32,torch.int64)
            or ((groups<0)|(groups>=sites)).any() or weights.shape!=(batch,tokens)
            or observed.shape!=(batch,tokens) or (weights<0).any()
            or global_mean.shape!=(batch,gdim) or global_precision.shape!=(batch,gdim,gdim)
            or local_precision.shape!=(batch,sites,ldim,ldim)):
        raise ValueError('Block Gaussian routes/weights/prior axes disagree')
    if (not torch.allclose(global_precision,global_precision.transpose(-1,-2),atol=1e-10,rtol=1e-10)
            or not torch.allclose(local_precision,local_precision.transpose(-1,-2),atol=1e-10,rtol=1e-10)):
        raise ValueError('Prior precision must be symmetric')
    _chol(global_precision,'global prior precision');_chol(local_precision,'local prior precision')
    return _block_natural_solution(global_design,local_design,groups,weights,weights*observed,
                                   global_mean,global_precision,local_mean,local_precision)


def _context_objective(predicted,sigma,lower,upper,points,weights,states,priors,precisions):
    safe_lower=torch.where(points,predicted-sigma,lower)
    safe_upper=torch.where(points,predicted+sigma,upper)
    interval=-gaussian_interval_log_prob(safe_lower,safe_upper,predicted,sigma)
    point_value=torch.where(points,lower,predicted)
    density=.5*((point_value-predicted)/sigma).square()+sigma.log()+.5*math.log(2*math.pi)
    result=(weights*torch.where(points,density,interval)).sum(-1)
    for state,prior,precision in zip(states,priors,precisions):
        difference=state-prior
        contribution=.5*torch.einsum('...d,...de,...e->...',difference,precision,difference)
        if contribution.ndim>1:contribution=contribution.sum(-1)
        result=result+contribution
    return result


def _bounded_step(current,proposed,objective):
    """Detached finite line search; gradient flows through the chosen step."""
    if any(not torch.isfinite(value).all() for value in proposed):
        raise FloatingPointError('Nonfinite Newton direction')
    with torch.no_grad():
        before=objective(tuple(value.detach() for value in current))
        chosen=torch.zeros_like(before)
        after=before.clone()
        accepted=torch.zeros_like(before,dtype=torch.bool)
        tolerance=OBJECTIVE_TOLERANCE*torch.maximum(torch.ones_like(before),before.abs())
        for fraction in BACKTRACKING_STEPS:
            candidate=tuple(a.detach()+fraction*(b.detach()-a.detach()) for a,b in zip(current,proposed))
            value=objective(candidate)
            take=(~accepted)&torch.isfinite(value)&(value<=before+tolerance)
            chosen=torch.where(take,torch.full_like(chosen,fraction),chosen)
            after=torch.where(take,value,after)
            accepted|=take
    updated=[]
    for a,b in zip(current,proposed):
        shape=(len(chosen),)+(1,)*(a.ndim-1)
        updated.append(a+chosen.reshape(shape)*(b-a))
    return tuple(updated),{'objective_before':before,'objective_after':after,'step_size':chosen,
                           'null_step':~accepted}


class HierarchicalField(nn.Module):
    """Global physical and AP-local spatial state with shared learned functions.

    ``global_reg`` pools the same 8+16 features into one 24-dimensional state;
    it has exactly the same learned parameter count as the hierarchy. The
    supplementary ``global8_reg`` uses only eight physical state coordinates,
    and is explicitly a capacity control, not literal legacy weights/code.
    """
    def __init__(self,cell='hier_reg',*,width=32,inference_steps=3):
        super().__init__()
        if cell not in CELLS or type(width) is not int or width!=32 or type(inference_steps) is not int or inference_steps!=3:
            raise ValueError('Unknown fixed hierarchical model configuration')
        self.config={'cell':cell,'width':width,'inference_steps':inference_steps}
        self.architecture={'global_dimension':GLOBAL_DIM,'local_dimension':LOCAL_DIM,
                     'rff_dimension':RFF_DIM,'rff_seed':RFF_SEED,'jitter':JITTER,
                     'inference_method':'observed_likelihood_Newton_Laplace',
                     'backtracking_steps':BACKTRACKING_STEPS,'objective_tolerance':OBJECTIVE_TOLERANCE}
        self.global_prior_mean=nn.Parameter(torch.zeros(GLOBAL_DIM))
        self.local_prior_mean=nn.Parameter(torch.zeros(LOCAL_DIM))
        self.raw_global_std=nn.Parameter(torch.full((GLOBAL_DIM,),math.log(math.expm1(.8))))
        self.raw_local_std=nn.Parameter(torch.full((LOCAL_DIM,),math.log(math.expm1(.3))))
        self.kernel=nn.Sequential(nn.Linear(13,width),nn.SiLU(),nn.Linear(width,16),nn.SiLU(),nn.Linear(16,8))
        self.discrepancy=nn.Sequential(nn.Linear(10,width),nn.SiLU(),nn.Linear(width,1))
        self.raw_discrepancy=nn.Parameter(torch.tensor(math.log(3./11.)))
        nn.init.normal_(self.kernel[-1].weight,std=.01)
        nn.init.zeros_(self.kernel[-1].bias)
        nn.init.zeros_(self.discrepancy[-1].weight)
        nn.init.zeros_(self.discrepancy[-1].bias)
        generator=torch.Generator(device='cpu').manual_seed(RFF_SEED)
        self.register_buffer('rff_omega',torch.randn(RFF_DIM,3,generator=generator)/8.)
        self.register_buffer('rff_phase',torch.rand(RFF_DIM,generator=generator)*(2*math.pi))

    @property
    def hierarchical(self):
        return self.config['cell'].startswith('hier_')

    def discrepancy_db(self,query):
        x=propagation_features(query.to(self.global_prior_mean))
        return 1.+14.*torch.sigmoid(self.raw_discrepancy+.25*torch.tanh(self.discrepancy(x).squeeze(-1)))

    def local_features(self,query,rx_xyz,ap_xyz,groups):
        """Eight fixed Fourier and eight bounded learned features, never IDs."""
        like=self.global_prior_mean
        q=query.to(like);rx=rx_xyz.to(like);aps=ap_xyz.to(like)
        chosen=torch.gather(aps,1,groups[...,None].expand(*groups.shape,3))
        relative=rx-chosen
        fixed=math.sqrt(2./RFF_DIM)*torch.cos(relative@self.rff_omega.T+self.rff_phase)
        learned=torch.tanh(self.kernel(torch.cat((propagation_features(q),relative/20.),-1)))/math.sqrt(8.)
        return torch.cat((fixed,learned),-1)

    def _geometry(self,q,groups,rx,aps,*,scheduled=None):
        _tensor(q,'public geometry');_tensor(rx,'physical RX XYZ');_tensor(aps,'physical AP XYZ')
        _tensor(groups,'AP routes',floating=False)
        if (q.ndim!=3 or q.shape[-1]!=16 or rx.shape!=(*q.shape[:2],3)
                or aps.ndim!=3 or aps.shape[0]!=q.shape[0] or aps.shape[-1]!=3
                or not 1<=aps.shape[1]<=128 or groups.shape!=q.shape[:2]
                or groups.dtype not in (torch.int32,torch.int64)
                or len({x.device for x in (q,groups,rx,aps)})!=1):
            raise ValueError('Public geometry, physical positions and integer AP axes disagree')
        _finite(aps,'physical AP positions')
        mask=torch.ones_like(groups,dtype=torch.bool) if scheduled is None else scheduled
        if ((groups[mask]<0)|(groups[mask]>=aps.shape[1])).any():
            raise ValueError('Scheduled AP route outside public AP set')
        _finite(q[mask],'scheduled geometry');_finite(rx[mask],'scheduled RX positions')
        safe_groups=torch.where(mask,groups,torch.zeros_like(groups)).to(torch.int64)
        safe_q=torch.where(mask[...,None],q,torch.zeros_like(q))
        safe_rx=torch.where(mask[...,None],rx,torch.zeros_like(rx))
        chosen=torch.gather(aps,1,safe_groups[...,None].expand(*groups.shape,3))
        distance=torch.linalg.vector_norm(safe_rx-chosen,dim=-1)
        if (distance[mask]<=0).any():
            raise ValueError('AP and receiver positions cannot coincide')
        if mask.any() and not torch.allclose(2*safe_q[...,6][mask],torch.log10(distance[mask]),atol=2.e-4,rtol=2.e-4):
            raise ValueError('Geometry distance disagrees with physical-metre positions')
        return safe_q,safe_groups,safe_rx,aps

    def encode(self,context,known,scheduled,context_ap,context_rx,ap_xyz,
               lower=None,upper=None,floor=None,weights=None):
        """Condition on residual-dB events relative to FSPL+estimated offset.

        lower/upper/floor are **residual dB**, not reported RSSI. Heard rows
        contain their actual detection-intersected bins. Censored rows use
        (-infinity, actual upper bound). Unknown rows never update a state.
        ``hier_erased`` keeps heard status as [public floor,+infinity), while
        upper-censored amplitudes become uninformative (-infinity,+infinity).
        """
        _tensor(context,'context')
        if (context.ndim!=3 or context.shape[-1]!=20 or known.shape!=context.shape[:2]
                or scheduled.shape!=known.shape or known.dtype!=torch.bool or scheduled.dtype!=torch.bool
                or (known&~scheduled).any()):
            raise ValueError('Explicit paired context20/masks required')
        _finite(context[scheduled],'scheduled context')
        context=torch.where(scheduled[...,None],context,torch.zeros_like(context))
        q,groups,rx,aps=self._geometry(context[...,4:],context_ap,context_rx,ap_xyz,scheduled=scheduled)
        heard=known&(context[...,2]>.5);censored=known&(context[...,3]>.5)
        if (heard&censored).any() or not torch.equal(heard|censored,known):
            raise ValueError('Known report semantics must be heard OR censored')
        if any(value is None for value in (lower,upper,floor)):
            raise ValueError('Explicit residual-dB interval endpoints and public floor required')
        for value,name in ((lower,'residual lower'),(upper,'residual upper'),(floor,'public residual floor')):
            _tensor(value,name)
            if value.shape!=known.shape:raise ValueError(name+' axes differ')
        if (torch.isnan(lower[known]).any() or torch.isnan(upper[known]).any()
                or not torch.isfinite(floor[scheduled]).all() or (lower[known]>upper[known]).any()
                or not torch.isfinite(lower[heard]).all() or not torch.isfinite(upper[heard]).all()
                or not torch.isneginf(lower[censored]).all() or not torch.isfinite(upper[censored]).all()
                or (lower[heard]+1.e-4<floor[heard]).any()):
            raise ValueError('Heard intervals/censor bounds/public floors are inconsistent')
        w=known.to(torch.float64) if weights is None else weights.to(dtype=torch.float64)
        if w.shape!=known.shape or not torch.isfinite(w).all() or (w<0).any() or (w>1).any() or (w[~known]!=0).any():
            raise ValueError('Explicit duplicate weights must be finite in0..1 and zero for unknown rows')
        lo=lower.to(torch.float64)/10.;hi=upper.to(torch.float64)/10.
        points=heard&(lo==hi)
        lo=torch.where(known,lo,torch.full_like(lo,-torch.inf));hi=torch.where(known,hi,torch.full_like(hi,torch.inf))
        if self.config['cell']=='hier_erased':
            lo=torch.where(heard,floor.to(torch.float64)/10.,torch.full_like(lo,-torch.inf))
            hi=torch.full_like(hi,torch.inf)
            w=w*heard  # no amplitude or invented upper event on an unheard row
            points=torch.zeros_like(points)
        g=query_physics_basis(q).to(torch.float64)
        local=self.local_features(q,rx,aps,groups).to(torch.float64)
        if self.config['cell']=='global8_reg':
            local=torch.zeros_like(local)
        sigma=self.discrepancy_db(q).to(torch.float64)/10.
        batch=context.shape[0];sites=aps.shape[1]
        gs=.2+torch.nn.functional.softplus(self.raw_global_std).to(torch.float64)
        ls=.2+torch.nn.functional.softplus(self.raw_local_std).to(torch.float64)
        gm=self.global_prior_mean.to(torch.float64)[None].expand(batch,-1)
        lm=self.local_prior_mean.to(torch.float64)[None,None].expand(batch,sites,-1)
        gp=torch.diag_embed(gs.reciprocal().square()+JITTER)[None].expand(batch,-1,-1)
        lp=torch.diag_embed(ls.reciprocal().square()+JITTER)[None,None].expand(batch,sites,-1,-1)
        inference_log=[]
        if not self.hierarchical:
            design=g if self.config['cell']=='global8_reg' else torch.cat((g,local),-1)
            prior=gm if self.config['cell']=='global8_reg' else torch.cat((gm,lm[:,0]),-1)
            precision=gp if self.config['cell']=='global8_reg' else torch.diag_embed(torch.cat((gs,ls)).reciprocal().square()+JITTER)[None].expand(batch,-1,-1)
            state=prior
            def context_objective(states):
                predicted=torch.einsum('btd,bd->bt',design,states[0])
                return _context_objective(predicted,sigma,lo,hi,points,w,states,(prior,),(precision,))
            for _ in range(self.config['inference_steps']):
                predicted=torch.einsum('btd,bd->bt',design,state)
                expected,conditional_var=interval_moments(predicted,sigma,lo,hi,point_mask=points)
                score=(expected-predicted)/sigma.square()
                curvature=(1.-conditional_var/sigma.square()).clamp(0.,1.)/sigma.square()
                info_weight=w*curvature
                natural=w*(curvature*predicted+score)
                factor=_chol(precision+torch.einsum('btd,bt,bte->bde',design,info_weight,design),'pooled Newton precision')
                rhs=torch.einsum('bde,be->bd',precision,prior)+torch.einsum('btd,bt->bd',design,natural)
                proposed=torch.cholesky_solve(rhs[...,None],factor).squeeze(-1)
                (state,),record=_bounded_step((state,),(proposed,),context_objective)
                inference_log.append(record)
            predicted=torch.einsum('btd,bd->bt',design,state)
            _,conditional_var=interval_moments(predicted,sigma,lo,hi,point_mask=points)
            info_weight=w*(1.-conditional_var/sigma.square()).clamp(0.,1.)/sigma.square()
            observed=_chol(precision+torch.einsum('btd,bt,bte->bde',design,info_weight,design),'pooled observed precision')
            belief={'global_mean':state,'global_covariance':torch.cholesky_inverse(observed),
                    'mode':'global8' if self.config['cell']=='global8_reg' else 'global24'}
        else:
            state_g,state_l=gm,lm
            def context_objective(states):
                state_global,state_local=states
                routed=torch.gather(state_local,1,groups[...,None].expand(*groups.shape,LOCAL_DIM))
                predicted=(g*state_global[:,None]).sum(-1)+(local*routed).sum(-1)
                return _context_objective(predicted,sigma,lo,hi,points,w,states,(gm,lm),(gp,lp))
            for _ in range(self.config['inference_steps']):
                routed=torch.gather(state_l,1,groups[...,None].expand(*groups.shape,LOCAL_DIM))
                predicted=(g*state_g[:,None]).sum(-1)+(local*routed).sum(-1)
                expected,conditional_var=interval_moments(predicted,sigma,lo,hi,point_mask=points)
                score=(expected-predicted)/sigma.square()
                curvature=(1.-conditional_var/sigma.square()).clamp(0.,1.)/sigma.square()
                info_weight=w*curvature
                natural=w*(curvature*predicted+score)
                posterior=_block_natural_solution(g,local,groups,info_weight,natural,gm,gp,lm,lp)
                (state_g,state_l),record=_bounded_step((state_g,state_l),
                    (posterior['global_mean'],posterior['local_mean']),context_objective)
                inference_log.append(record)
            routed=torch.gather(state_l,1,groups[...,None].expand(*groups.shape,LOCAL_DIM))
            predicted=(g*state_g[:,None]).sum(-1)+(local*routed).sum(-1)
            _,conditional_var=interval_moments(predicted,sigma,lo,hi,point_mask=points)
            info_weight=w*(1.-conditional_var/sigma.square()).clamp(0.,1.)/sigma.square()
            observed=_block_factor(g,local,groups,info_weight,gp,lp)
            belief={'global_mean':state_g,'local_mean':state_l,
                    'global_covariance':torch.cholesky_inverse(observed['global_cholesky']),
                    'local_cholesky':observed['local_cholesky'],'local_coupling':observed['coupling'],
                    'mode':'hierarchy'}
        observed_ap=torch.zeros((batch,sites),dtype=torch.bool,device=known.device)
        for site in range(sites):observed_ap[:,site]=(known&(groups==site)).any(-1)
        belief.update(ap_xyz=aps.clone(),observed_ap=observed_ap,ap_count=sites,
                      field_covariance_qualified=False,packet_risk_qualified=False,
                      inference_log=inference_log)
        if not all(torch.isfinite(v).all() for v in belief.values() if isinstance(v,torch.Tensor)):
            raise FloatingPointError('Nonfinite hierarchical belief')
        return belief

    def encode_view(self,view,ap_xyz):
        return self.encode(view['context'],view['known'],view['scheduled'],view['ap'],view['rx_xyz'],ap_xyz,
                           view['lower'],view['upper'],view['floor'],view.get('weights'))

    def _query_design(self,belief,query,query_ap,query_rx,basis):
        q,groups,rx,aps=self._geometry(query,query_ap,query_rx,belief['ap_xyz'])
        if basis.shape!=(*query.shape[:2],GLOBAL_DIM) or not torch.isfinite(basis).all():
            raise ValueError('Explicit eight-coordinate query basis required')
        expected=query_physics_basis(q)
        if not torch.allclose(basis.to(expected),expected,atol=2.e-5,rtol=2.e-5):
            raise ValueError('Query basis disagrees with public physical geometry')
        g=basis.to(torch.float64);local=self.local_features(q,rx,aps,groups).to(torch.float64)
        if belief['mode']=='global8':return g,None,groups,q
        if belief['mode']=='global24':return torch.cat((g,local),-1),None,groups,q
        shared=torch.zeros_like(g)
        for batch in range(len(local)):
            for site in range(belief['ap_count']):
                selected=torch.nonzero(groups[batch]==site,as_tuple=False).flatten()
                if len(selected):
                    shared[batch,selected]=g[batch,selected]-local[batch,selected]@belief['local_coupling'][batch,site]
        return shared,local,groups,q

    def render(self,belief,query,query_ap,query_rx,basis,fspl,offset):
        shared,local,groups,q=self._query_design(belief,query,query_ap,query_rx,basis)
        if fspl.shape!=query.shape[:2] or offset.shape!=fspl.shape or not torch.isfinite(fspl).all() or not torch.isfinite(offset).all():
            raise ValueError('Known FSPL and estimated offset must match query axes')
        if belief['mode']=='hierarchy':
            g=basis.to(torch.float64)
            routed=torch.gather(belief['local_mean'],1,groups[...,None].expand(*groups.shape,LOCAL_DIM))
            residual=(g*belief['global_mean'][:,None]).sum(-1)+(local*routed).sum(-1)
            # Solve one multi-RHS system per AP, instead of materializing a
            # 16x16 Cholesky copy for every receiver/horizon/carrier query.
            variance=torch.zeros_like(residual)
            for batch in range(len(local)):
                for site in range(belief['ap_count']):
                    selected=torch.nonzero(groups[batch]==site,as_tuple=False).flatten()
                    if not len(selected):continue
                    design=local[batch,selected]
                    solved=torch.cholesky_solve(design.T,belief['local_cholesky'][batch,site]).T
                    variance[batch,selected]=(design*solved).sum(-1)
        else:
            residual=(shared*belief['global_mean'][:,None]).sum(-1)
            variance=torch.zeros_like(residual)
        variance=variance+torch.einsum('bqd,bde,bqe->bq',shared,belief['global_covariance'],shared)
        mean=fspl.to(torch.float64)+offset.to(torch.float64)+10.*residual
        std=torch.sqrt(100.*variance.clamp_min(0.)+self.discrepancy_db(q).to(torch.float64).square())
        return mean,std

    def render_queries(self,belief,query):
        return self.render(belief,query['q'],query['ap'],query['rx_xyz'],query['basis'],query['fspl'],query['offset'])

    def predictive_covariance(self,belief,query,query_ap,query_rx,basis):
        """Coefficient-field covariance in dB², excluding discrepancy/fading.

        The additive discrepancy in marginal render spread has no identified
        cross-link correlation; this API does not invent that correlation.
        This quadratic small-probe API accepts at most 512 queries. Large
        native forecasts must use the linear-memory marginal ``render`` API.
        """
        if not isinstance(query,torch.Tensor) or query.ndim!=3 or query.shape[1]>MAX_COVARIANCE_QUERIES:
            raise ValueError('Full covariance is restricted to at most512 public probe queries')
        shared,local,groups,_=self._query_design(belief,query,query_ap,query_rx,basis)
        covariance=torch.einsum('bqd,bde,bke->bqk',shared,belief['global_covariance'],shared)
        if local is not None:
            for site in range(belief['ap_count']):
                design=local*(groups==site)[...,None]
                solved=torch.cholesky_solve(design.transpose(-1,-2),belief['local_cholesky'][:,site])
                covariance=covariance+design@solved
        return covariance*100.

    def forward(self,view,query,ap_xyz):
        return self.render_queries(self.encode_view(view,ap_xyz),query)


# A descriptive alias for callers that prefer a scene-model name.
HierarchicalScene=HierarchicalField


def observation_nll(mean,std,target):
    """Equal-example interval/censor score in the target's reported RSSI gauge."""
    heard=target['heard'];censored=target['censored'];known=heard|censored
    if 'lower' in target or 'upper' in target:
        lower,upper=target.get('lower'),target.get('upper')
    else:
        raw_lower=target.get('lower_dbm');raw_upper=target.get('upper_dbm')
        floor=target.get('detection_threshold_dbm');bound=target.get('censor_upper_dbm')
        if any(value is None for value in (raw_lower,raw_upper,floor,bound)):
            raise ValueError('Reported bins need their public detection floor and actual censor bound')
        lower=torch.where(censored,torch.full_like(mean,-torch.inf),torch.maximum(raw_lower.to(mean),floor.to(mean)))
        upper=torch.where(censored,bound.to(mean),raw_upper.to(mean))
    if (heard.dtype!=torch.bool or censored.dtype!=torch.bool or heard.shape!=mean.shape
            or censored.shape!=mean.shape or (heard&censored).any()
            or lower is None or upper is None or lower.shape!=mean.shape or upper.shape!=mean.shape
            or not known.any(-1).all()):
        raise ValueError('Known, distinct target interval/censor events required for each example')
    if (torch.isnan(lower[known]).any() or torch.isnan(upper[known]).any()
            or (lower[known]>=upper[known]).any() or not torch.isfinite(lower[heard]).all()
            or not torch.isfinite(upper[heard]).all() or not torch.isneginf(lower[censored]).all()
            or not torch.isfinite(upper[censored]).all()):
        raise ValueError('Target heard intervals and censor events must stay distinct')
    lo=torch.where(known,lower.to(mean),mean.detach()-1.)
    hi=torch.where(known,upper.to(mean),mean.detach()+1.)
    score=-gaussian_interval_log_prob(lo,hi,mean,std)
    weights=target.get('weights',known.to(mean.dtype)).to(mean)*known
    if not torch.isfinite(weights).all() or (weights<0).any() or not (weights.sum(-1)>0).all():
        raise ValueError('Positive finite target mass required')
    return ((score*weights).sum(-1)/weights.sum(-1)).mean()


def objective(model,a,b,query,target,ap_xyz):
    """Matched likelihood with a fixed, supported common-probe agreement term.

    Both branches receive the same targets. Agreement reads only two posterior
    predictions, their detached working variances and public query AP routes.
    There is no stopped teacher or claim of classic I-JEPA equivalence.
    """
    ea,eb=model.encode_view(a,ap_xyz),model.encode_view(b,ap_xyz)
    pa,sa=model.render_queries(ea,query);pb,sb=model.render_queries(eb,query)
    measurement=.5*(observation_nll(pa,sa,target)+observation_nll(pb,sb,target))
    shared_ap=ea['observed_ap']&eb['observed_ap']
    supported=torch.gather(shared_ap,1,query['ap'].to(torch.int64))
    if model.config['cell']=='hier_joint' and supported.any():
        working_variance=((sa.square()+sb.square())/100.+.01).detach()
        alignment=(((pa-pb)/10.).square()/working_variance)[supported].mean()
    else:alignment=(pa-pb).sum()*0.
    loss=measurement+ALIGNMENT_WEIGHT*alignment
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite hierarchical training loss')
    return loss,{'loss':loss.detach(),'measurement_nll':measurement.detach(),
        'alignment':alignment.detach(),'alignment_weight':loss.new_tensor(ALIGNMENT_WEIGHT if model.config['cell']=='hier_joint' else 0.),
        'supported_agreement_queries':supported.sum().detach(),
        'mean_predictive_std_db':.5*(sa.mean()+sb.mean()).detach(),
        'mean_inference_step_size':torch.stack([step['step_size'].mean() for belief in (ea,eb) for step in belief['inference_log']]).mean().detach(),
        'inference_null_steps':torch.stack([step['null_step'].sum() for belief in (ea,eb) for step in belief['inference_log']]).sum().detach()}
