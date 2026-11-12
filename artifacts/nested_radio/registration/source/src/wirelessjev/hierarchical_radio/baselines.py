"""TRAIN-only interval/censor numerical controls for AP-conditioned RF fields.

These are numerical field models, not learned scatterer identities or calibrated
packet-risk heads. ``radiomap_operator`` is a new interval-likelihood analogue
of the geometry/RFF RadioMap, not a claim of bit-identical native RadioMap code.
The caller must admit the published scientific case before any actual fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ..spectral_scene.baselines import _truncated_moments
from ..spectral_scene.operators import query_physics_basis

METHODS = ('radiomap_operator', 'hierarchy_fixed')
LENGTHSCALE_CANDIDATES_M = (5., 10., 20.)
PROPAGATION_COLUMNS = tuple(range(8)) + (9, 10)
MAX_EM = 48
SCALE_STEPS = 12
NOISE_FLOOR_DB = .5
NOISE_CEILING_DB = 30.


def _array(value, shape, name, *, finite=True, boolean=False, integer=False):
    a = np.asarray(value)
    if a.shape != shape or (boolean and a.dtype != np.bool_) or (integer and a.dtype.kind not in 'iu'):
        raise ValueError(name+' has inconsistent axes/type')
    if finite and not np.isfinite(a).all():
        raise ValueError(name+' must be finite')
    return a


def _freeze(value):
    a = np.ascontiguousarray(value, dtype='f8')
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int,float,np.integer,np.floating)):
        raise ValueError(name+' must be a finite positive number')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name+' must be a finite positive number')
    return value


def _moments(mean, lower, upper, sigma):
    """Actual point reports use a density, never an invented finite bin."""
    point=np.isfinite(lower)&(lower==upper)
    expected=np.empty_like(mean);variance=np.empty_like(mean)
    if point.any():
        expected[point]=lower[point];variance[point]=0.
    if (~point).any():
        expected[~point],variance[~point]=_truncated_moments(mean[~point],lower[~point],upper[~point],sigma)
    return expected,variance


def interval_posterior(x, lower, upper, weights, noise_db, prior_mean, prior_covariance,
                       *, max_iterations=MAX_EM):
    """Interval EM MAP and observed-information Laplace covariance.

    Complete-data precision is only an EM mean-update device. The returned
    covariance uses event curvature, so a vacuous upper bound is not counted
    as a precise point. The approximation is conditional on the working
    Gaussian field discrepancy and admitted report weights. Finite equal
    endpoints denote a genuinely continuous point report and use Gaussian
    density moments; they are not a zero-width quantization probability.
    """
    x = np.asarray(x, dtype='f8'); mean0 = np.asarray(prior_mean, dtype='f8')
    if x.ndim != 2 or mean0.ndim != 1 or x.shape[1] != len(mean0) or not len(mean0):
        raise ValueError('Finite nonempty coefficient and observation axes required')
    n, d = x.shape
    lower = _array(lower,(n,),'Lower interval',finite=False).astype('f8')
    upper = _array(upper,(n,),'Upper interval',finite=False).astype('f8')
    weights = _array(weights,(n,),'Observation weights').astype('f8')
    covariance0 = _array(prior_covariance,(d,d),'Prior covariance').astype('f8')
    sigma = _positive(noise_db,'Working discrepancy')
    if (type(max_iterations) is not int or not 1 <= max_iterations <= 512
            or not np.isfinite(x).all() or not np.isfinite(mean0).all()
            or np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper)
            or np.isposinf(lower).any() or np.isneginf(upper).any()
            or np.any(weights < 0) or not np.allclose(covariance0,covariance0.T,rtol=0,atol=1e-10)):
        raise ValueError('Invalid interval likelihood or Gaussian prior')
    try:
        prior_factor = cho_factor(covariance0, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Prior covariance must be positive definite') from exc
    precision0 = cho_solve(prior_factor,np.eye(d),check_finite=False)
    # Fully uninformative/missing intervals contribute no likelihood or EM work.
    use = (weights > 0) & ~(np.isneginf(lower)&np.isposinf(upper))
    x, lower, upper, weights = x[use], lower[use], upper[use], weights[use]
    if not len(x):
        return mean0.copy(), covariance0.copy()
    complete_precision = precision0 + (x.T*weights)@x/sigma**2
    factor = cho_factor(complete_precision,lower=True,check_finite=False)
    rhs0 = precision0@mean0
    coefficient = mean0.copy()
    for _ in range(max_iterations):
        expected, _ = _moments(x@coefficient,lower,upper,sigma)
        update = cho_solve(factor,rhs0+x.T@(weights*expected)/sigma**2,check_finite=False)
        if not np.isfinite(update).all():
            raise FloatingPointError('Nonfinite interval field update')
        delta = float(np.max(np.abs(update-coefficient)))
        coefficient = update
        if delta < 1e-7:
            break
    _, conditional_variance = _moments(x@coefficient,lower,upper,sigma)
    curvature = np.clip(1./sigma**2-conditional_variance/sigma**4,0.,1./sigma**2)
    information = precision0+(x.T*(weights*curvature))@x
    covariance = cho_solve(cho_factor(information,lower=True,check_finite=False),np.eye(d),check_finite=False)
    covariance = .5*(covariance+covariance.T)
    if not np.isfinite(covariance).all():
        raise FloatingPointError('Nonfinite observed-information covariance')
    return coefficient, covariance


def _working_scale(x, lower, upper, weights):
    d = x.shape[1]; zero = np.zeros(d)
    broad = np.diag(np.full(d,400.)); broad[0,0] = 1600.
    sigma = 6.
    for _ in range(SCALE_STEPS):
        coefficient, covariance = interval_posterior(x,lower,upper,weights,sigma,zero,broad)
        fitted = x@coefficient
        expected, variance = _moments(fitted,lower,upper,sigma)
        estimate = math.sqrt(float(np.sum(weights*(variance+(expected-fitted)**2))/weights.sum()))
        updated = float(np.clip(estimate,NOISE_FLOOR_DB,NOISE_CEILING_DB))
        if abs(updated-sigma) < 1e-5:
            sigma = updated
            break
        sigma = updated
    coefficient,covariance = interval_posterior(x,lower,upper,weights,sigma,zero,broad)
    return coefficient,covariance,sigma


def _fixed_frequencies(count, lengthscale, *, small=False):
    if small:
        # The first eight match the model's fixed local features, while the
        # other eight replace learned features with additional fixed support.
        import torch
        chunks=[];phases=[]
        for seed in (184910,184911):
            generator=torch.Generator().manual_seed(seed)
            chunks.append((torch.randn(8,3,generator=generator)/8.).numpy().astype('f8'))
            phases.append((torch.rand(8,generator=generator)*(2*math.pi)).numpy().astype('f8'))
        return np.concatenate(chunks),np.concatenate(phases)
    rng=np.random.default_rng(913107)
    return rng.normal(0.,1./lengthscale,size=(count,3)),rng.uniform(0.,2*math.pi,size=count)


def _public(rows, ap_xyz, *, view):
    aps=np.asarray(ap_xyz,dtype='f8')
    if aps.ndim!=2 or aps.shape[1]!=3 or not len(aps) or not np.isfinite(aps).all():
        raise ValueError('Finite public AP XYZ positions required')
    if view:
        context=np.asarray(rows['context']);n=len(context)
        _array(context,(n,20),'Context')
        q=context[:,4:]
    else:
        q=np.asarray(rows['q']);n=len(q)
        _array(q,(n,16),'Public query')
    ap=_array(rows['ap'],(n,),'AP grouping',integer=True).astype('i8')
    rx=_array(rows['rx_xyz'],(n,3),'Receiver XYZ').astype('f8')
    if np.any(ap<0) or np.any(ap>=len(aps)):
        raise ValueError('AP grouping must refer to current public APs')
    fspl=_array(rows['fspl'],(n,),'Free-space reference').astype('f8')
    offset=_array(rows['offset'],(n,),'Declared receiver offset').astype('f8')
    return {'q':q.astype('f8'),'ap':ap,'rx':rx,'ap_xyz':aps[ap],'fspl':fspl,'offset':offset},len(aps)


def _raw_design(method, public, omega, phase):
    q=public['q'];g=np.asarray(query_physics_basis(q),dtype='f8')
    delta=public['rx']-public['ap_xyz']
    if method=='radiomap_operator':
        rff=math.sqrt(2./48)*np.cos(delta@omega.T+phase)
        design=np.column_stack((g,q[:,PROPAGATION_COLUMNS],rff))
        if not np.isfinite(design).all():raise ValueError('Nonfinite physical map features')
        return design
    if method=='hierarchy_fixed':
        rff=math.sqrt(2./8)*np.cos(delta@omega.T+phase)
        design=np.column_stack((g,rff))
        if not np.isfinite(design).all():raise ValueError('Nonfinite physical hierarchy features')
        return design
    raise ValueError('Unknown registered numerical method')


@dataclass(frozen=True)
class Prior:
    method: str
    lengthscale_m: float
    x_mean: np.ndarray
    x_scale: np.ndarray
    coefficient: np.ndarray
    covariance: np.ndarray
    adaptation_variance: np.ndarray
    noise_db: float
    omega: np.ndarray
    phase: np.ndarray
    metadata_json: str

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError('Unknown numerical prior method')
        length=_positive(self.lengthscale_m,'Lengthscale');noise=_positive(self.noise_db,'Discrepancy')
        if self.method=='radiomap_operator' and length not in LENGTHSCALE_CANDIDATES_M:
            raise ValueError('Use the registered 5/10/20m lengthscale grid')
        if self.method=='hierarchy_fixed' and length!=8.:
            raise ValueError('The dimension-matched fixed hierarchy uses its fixed 8m features')
        d,k=(66,48) if self.method=='radiomap_operator' else (24,16)
        for name,shape in (('x_mean',(d,)),('x_scale',(d,)),('coefficient',(d,)),
                           ('covariance',(d,d)),('adaptation_variance',(d,)),('omega',(k,3)),('phase',(k,))):
            a=_array(getattr(self,name),shape,name).astype('f8')
            if name in ('x_scale','adaptation_variance') and np.any(a<=0):
                raise ValueError('Positive normalization/prior variances required')
            object.__setattr__(self,name,_freeze(a))
        if not np.allclose(self.covariance,self.covariance.T,rtol=0,atol=1e-9):
            raise ValueError('Training covariance must be symmetric')
        try:np.linalg.cholesky(self.covariance)
        except np.linalg.LinAlgError as exc:raise ValueError('Training covariance must be positive definite') from exc
        if not isinstance(self.metadata_json,str) or not isinstance(json.loads(self.metadata_json),dict):
            raise ValueError('Explicit numerical-prior provenance required')
        object.__setattr__(self,'lengthscale_m',length);object.__setattr__(self,'noise_db',noise)

    @property
    def metadata(self):
        return json.loads(self.metadata_json)

    def design(self, public):
        return (_raw_design(self.method,public,self.omega,self.phase)-self.x_mean)/self.x_scale


def _view_intervals(rows, aps):
    public,_=_public(rows,aps,view=True);n=len(public['ap'])
    known=_array(rows['known'],(n,),'Known report mask',boolean=True)
    scheduled=_array(rows['scheduled'],(n,),'Scheduled report mask',boolean=True)
    if np.any(known&~scheduled):raise ValueError('Known reports must have been attempted')
    lower=_array(rows['lower'],(n,),'Residual lower',finite=False).astype('f8')
    upper=_array(rows['upper'],(n,),'Residual upper',finite=False).astype('f8')
    if np.isnan(lower[known]).any() or np.isnan(upper[known]).any() or np.any(lower[known]>upper[known]):
        raise ValueError('Known interval and censor endpoints must be valid')
    flags=np.asarray(rows['context'])[:,2:4]
    heard=known&(flags[:,0]>.5);censored=known&(flags[:,1]>.5)
    if np.any(heard&censored) or not np.array_equal(heard|censored,known):
        raise ValueError('Known heard and censored semantics must be disjoint')
    if not np.isfinite(lower[heard]).all() or not np.isfinite(upper[known]).all() or not np.isneginf(lower[censored]).all():
        raise ValueError('A censor event is an upper bound, not a finite point')
    floor=_array(rows['floor'],(n,),'Residual event floor').astype('f8')
    if np.any(lower[heard]<floor[heard]-1e-6):raise ValueError('Heard event lower bound is below its detection floor')
    return public,known,lower,upper


def _target_intervals(example):
    public,_=_public(example['query'],example['ap_xyz'],view=False);n=len(public['ap'])
    t=example['target'];heard=_array(t['heard'],(n,),'Target heard',boolean=True)
    censored=_array(t['censored'],(n,),'Target censored',boolean=True)
    if np.any(heard&censored):raise ValueError('Target masks must be disjoint')
    known=heard|censored;reference=public['fspl']+public['offset']
    lo=_array(t['lower_dbm'],(n,),'Target lower',finite=False).astype('f8')
    hi=_array(t['upper_dbm'],(n,),'Target upper',finite=False).astype('f8')
    floor=_array(t['detection_threshold_dbm'],(n,),'Target floor').astype('f8')
    bound=_array(t['censor_upper_dbm'],(n,),'Target censor',finite=False).astype('f8')
    lower=np.where(heard,np.maximum(lo,floor)-reference,-np.inf)
    upper=np.where(censored,bound-reference,hi-reference)
    if (not np.isfinite(lower[heard]).all() or not np.isfinite(upper[known]).all()
            or np.any(lower[known]>=upper[known])):
        raise ValueError('Target report intervals and censor bounds must remain finite/ordered')
    return public,known,lower,upper


def fit_prior(train_examples, lengthscale=10., *, method='radiomap_operator'):
    """Fit a shared hyperprior only from admitted TRAIN reports and targets.

    The empirical prior is learned across TRAIN cases; new-case inference never
    fits a pooled current posterior and then re-assimilates its reports. Views
    or regimes sharing one physical link retain inverse-multiplicity mass.
    """
    examples=tuple(train_examples)
    if not examples:raise ValueError('A nonempty admitted TRAIN corpus is required')
    if method not in METHODS:raise ValueError('Unknown numerical method')
    length=_positive(lengthscale,'Lengthscale')
    if method=='radiomap_operator' and length not in LENGTHSCALE_CANDIDATES_M:raise ValueError('Unregistered lengthscale')
    if method=='hierarchy_fixed' and length!=8.:raise ValueError('Fixed hierarchy uses 8m features')
    omega,phase=_fixed_frequencies(48 if method=='radiomap_operator' else 16,length,small=method=='hierarchy_fixed')
    parts=[];keys=[];identities=[];heard_count=0
    for example in examples:
        binding=example.get('binding',{})
        if (binding.get('stage')!='training' or binding.get('held_band') is not False
                or not isinstance(binding.get('case_id'),str) or not binding['case_id']):
            raise ValueError('Numerical hyperpriors require explicit TRAIN case identities')
        case=binding['case_id'];identities.append(case)
        for public,known,lower,upper in (*(_view_intervals(example[name],example['ap_xyz']) for name in ('a','b')),
                                         _target_intervals(example)):
            # The held band has no place in any TRAIN fit, even if a caller
            # mislabeled a later evaluation example as a training case.
            hz=2.4e9*np.exp(public['q'][:,7])
            if not np.all(np.isclose(hz,2.4e9,rtol=1e-6)|np.isclose(hz,5.2e9,rtol=1e-6)):
                raise ValueError('Held carriers cannot fit numerical hyperpriors')
            use=np.flatnonzero(known)
            if not len(use):continue
            x=_raw_design(method,public,omega,phase)[use]
            parts.append((x,lower[use],upper[use]))
            heard_count+=int(np.isfinite(lower[use]).sum())
            for row in use:
                keys.append((case,int(public['ap'][row]),*public['rx'][row].tolist(),float(public['q'][row,7])))
    if not parts or heard_count==0:raise ValueError('TRAIN hyperpriors need real known and heard reports')
    counts={}
    for key in keys:counts[key]=counts.get(key,0)+1
    weights=np.array([1./counts[key] for key in keys])
    x=np.concatenate([p[0] for p in parts]);lo=np.concatenate([p[1] for p in parts]);hi=np.concatenate([p[2] for p in parts])
    center=np.average(x,axis=0,weights=weights)
    deviation=np.sqrt(np.average((x-center)**2,axis=0,weights=weights))
    scale=np.where(deviation>1e-6,deviation,1.)
    center[0]=0.;scale[0]=1.
    normalized=(x-center)/scale
    coefficient,covariance,noise=_working_scale(normalized,lo,hi,weights)
    # A TRAIN-fitted discrepancy sets local adaptation strength. Do not use
    # diag(TRAIN covariance) as an additional independent prior: duplicated
    # physical regressors have large anticorrelated coefficient uncertainty,
    # and discarding those correlations fabricates huge field variance. The
    # full TRAIN covariance is retained separately in the AP-local prior.
    local=np.full(len(coefficient),noise**2/max(1,len(coefficient)-1))
    local[0]=noise**2
    metadata={'method':method,'training_case_ids':sorted(set(identities)),
        'training_examples':len(examples),'known_report_rows':len(weights),'heard_report_rows':heard_count,
        'effective_physical_link_mass':float(weights.sum()),'unique_physical_links':len(counts),
        'correlated_view_regime_weighting':'inverse multiplicity of case/AP/RX/carrier among known reports',
        'likelihood':'quantization intervals and upper censor events; observed-information Laplace spread',
        'scale_updates':SCALE_STEPS,'maximum_interval_EM_updates':MAX_EM,
        'sigma_floor_ceiling_db':[NOISE_FLOOR_DB,NOISE_CEILING_DB],
        'lengthscale_m':length,'features':len(coefficient),'validation_or_test_labels_used':False,
        'held_band_used':False,'packet_risk_calibrated':False,'native_RadioMap_bit_identical':False,
        'prior_scope':'shared TRAIN empirical hyperprior; one current likelihood update per admitted report',
        'hierarchy_scope':'joint global8/AP-local16 fixed features' if method=='hierarchy_fixed' else 'independent AP/source-carrier66 coefficient fields',
        'held_carrier_rule':'shared TRAIN mean plus linear log-frequency interpolation/extrapolation of source-carrier residual corrections; absolute-weight SD bound, not independent-band variance' if method=='radiomap_operator' else 'shared physical log-frequency basis and AP spatial posterior; no held-carrier context'}
    return Prior(method,length,center,scale,coefficient,covariance,local,noise,omega,phase,
                 json.dumps(metadata,sort_keys=True,separators=(',',':')))


def fit_fixed_prior(train_examples):
    return fit_prior(train_examples,8.,method='hierarchy_fixed')


def _quadratic_rows(x, covariance):
    return np.sum((x@covariance)*x,axis=1)


def predict(prior, example, view='a'):
    """Adapt once from the selected reported view and render raw target RSSI.

    Target labels are never inspected. Returned standard deviation includes
    working field discrepancy, not an asserted hardware or service bound.
    """
    if not isinstance(prior,Prior) or view not in ('a','b'):
        raise ValueError('A bound numerical prior and one admitted view are required')
    context,known,lo,hi=_view_intervals(example[view],example['ap_xyz'])
    query,sites=_public(example['query'],example['ap_xyz'],view=False)
    xc,xq=prior.design(context),prior.design(query)
    ref=query['fspl']+query['offset']
    if prior.method=='radiomap_operator':
        source_logf=np.array([0.,math.log(5.2/2.4)])
        context_logf=context['q'][:,7]
        source_masks=[np.isclose(context_logf,value,rtol=0,atol=1e-6) for value in source_logf]
        if np.any(known&~(source_masks[0]|source_masks[1])):
            raise ValueError('Only the two actually observed source carriers may adapt this numerical map')
        mean=ref+xq@prior.coefficient;local_sd=np.zeros(len(ref))
        covariance0=prior.covariance+np.diag(prior.adaptation_variance)
        fraction=query['q'][:,7]/source_logf[1]
        fraction=np.where(np.isclose(query['q'][:,7],source_logf[0],rtol=0,atol=1e-6),0.,
                  np.where(np.isclose(query['q'][:,7],source_logf[1],rtol=0,atol=1e-6),1.,fraction))
        for source,mix in enumerate((1.-fraction,fraction)):
            source_query=dict(query);source_query['q']=query['q'].copy()
            source_query['q'][:,7]=source_logf[source]
            phi=prior.design(source_query)
            for site in range(sites):
                observe=known&(context['ap']==site)&source_masks[source];ask=query['ap']==site
                if not ask.any():continue
                beta,cov=interval_posterior(xc[observe],lo[observe],hi[observe],np.ones(observe.sum()),
                                           prior.noise_db,prior.coefficient,covariance0)
                mean[ask]+=mix[ask]*(phi[ask]@(beta-prior.coefficient))
                variance=np.maximum(_quadratic_rows(phi[ask],cov),0.)
                # Cross-band posterior dependence is not identified. Absolute
                # weights bound any covariance rather than declaring two
                # channels independent or averaging raw received powers.
                local_sd[ask]+=np.abs(mix[ask])*np.sqrt(variance)
        variance=local_sd**2+prior.noise_db**2
    else:
        # One joint block model, so shared global evidence and AP-local updates
        # are inferred simultaneously rather than reused as independent priors.
        dimension=8+sites*16
        def block(x,groups):
            matrix=np.zeros((len(x),dimension));matrix[:,:8]=x[:,:8]
            for site in range(sites):
                mask=groups==site;matrix[mask,8+site*16:8+(site+1)*16]=x[mask,8:]
            return matrix
        matrix=block(xc[known],context['ap'][known])
        mean0=np.concatenate((prior.coefficient[:8],np.tile(prior.coefficient[8:],sites)))
        variance0=np.concatenate((prior.adaptation_variance[:8],np.tile(prior.adaptation_variance[8:],sites)))
        beta,cov=interval_posterior(matrix,lo[known],hi[known],np.ones(known.sum()),prior.noise_db,
                                   mean0,np.diag(variance0))
        # Render sparse global/local rows by their covariance blocks. Expanding
        # 43,200 native queries into a dense (8+16S)-wide matrix would charge
        # irrelevant AP blocks and obscure the numerical control's true cost.
        g=xq[:,:8];l=xq[:,8:]
        mean=ref+g@beta[:8]
        variance=_quadratic_rows(g,cov[:8,:8])+prior.noise_db**2
        for site in range(sites):
            ask=query['ap']==site;sl=slice(8+16*site,8+16*(site+1))
            mean[ask]+=l[ask]@beta[sl]
            variance[ask]+=_quadratic_rows(l[ask],cov[sl,sl])
            variance[ask]+=2*np.sum((g[ask]@cov[:8,sl])*l[ask],axis=1)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance<=0):
        raise FloatingPointError('Numerical field predictor returned invalid moments')
    return mean,np.sqrt(variance)


def pack_prior(prior):
    if not isinstance(prior,Prior):raise TypeError('Typed numerical prior required')
    return {name:np.asarray(getattr(prior,name)).copy() for name in Prior.__dataclass_fields__}


def load_prior(arrays):
    if set(arrays)!=set(Prior.__dataclass_fields__):raise ValueError('Numerical prior archive membership differs')
    values=dict(arrays)
    for key in ('method','metadata_json'):
        a=np.asarray(values[key])
        if a.shape!=() or a.dtype.kind not in 'US':raise ValueError('Scalar prior metadata required')
        values[key]=str(a.item())
    for key in ('lengthscale_m','noise_db'):
        a=np.asarray(values[key])
        if a.shape!=():raise ValueError('Scalar prior scale required')
        values[key]=float(a)
    return Prior(**values)


pack=pack_prior
unpack=load_prior
