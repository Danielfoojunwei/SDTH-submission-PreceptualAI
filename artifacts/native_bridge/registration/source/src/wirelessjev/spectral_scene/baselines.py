"""Numerical field controls using the same reported context as the scene model.

``radiomap_operator`` reuses the actual native RadioMap Bayesian/Tobit fitter
with an explicit public-feature matrix adapter. Its old fixed warehouse rack
features are replaced, not applied to unrelated CAD. A TRAIN-learned field and
the known per-link FSPL operator transport both baselines to unseen carriers.
No physical_channels, material identities, true receiver offsets or query RF
labels are read here.
"""
from __future__ import annotations

from dataclasses import dataclass,field
from copy import deepcopy
import io
import math
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr

from ..native.radiomap import RadioMap,NOISE_DB,CENSOR_WEIGHT,N_RFF
from .operators import query_physics_basis

METHODS=('radiomap_operator','bayesian_operator')
KNOWN_HZ=(2.4e9,5.2e9)
PROPAGATION_COLUMNS=tuple(range(8))+(9,10)
_CACHE=None


def _common():
    from . import common
    return common


def _freeze(a,dtype='f8'):
    a=np.asarray(a,dtype=dtype)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


def _finite(a,shape,name):
    raw=np.asarray(a)
    if raw.dtype.kind not in 'fiu' or raw.shape!=shape or not np.isfinite(raw).all():
        raise ValueError('Invalid public baseline operand: '+name)
    return raw.astype('f8',copy=False)


def _query(rows):
    """Whitelist only metadata available before a query is measured."""
    g=np.asarray(rows.geometry_features)
    if g.ndim!=2 or g.shape[1]!=16:raise ValueError('Use the shared sixteen-column public geometry schema')
    n=len(g)
    result={'geometry':_finite(g,(n,16),'geometry'),
            'rx':_finite(rows.rx_xyz,(n,3),'RX position'),
            'ap':_finite(rows.ap_xyz,(n,3),'AP position'),
            'frequency':_finite(rows.carrier_hz,(n,),'carrier'),
            'fspl':_finite(rows.fspl_dbm,(n,),'known per-link FSPL'),
            'offset':_finite(rows.offset_db,(n,),'observed-pair offset estimate')}
    if np.any(result['frequency']<=0):raise ValueError('Carriers must be positive')
    return result


def _observations(rows):
    q=_query(rows);n=len(q['geometry'])
    heard=np.asarray(rows.heard);censored=np.asarray(rows.censored)
    if (heard.dtype!=bool or censored.dtype!=bool or heard.shape!=(n,) or censored.shape!=(n,)
            or np.any(heard&censored)):
        raise ValueError('Heard/censored observations must remain distinct Boolean evidence')
    point=np.asarray(rows.reported_rssi_dbm,dtype='f8')
    lo=np.asarray(rows.lower_dbm,dtype='f8');hi=np.asarray(rows.upper_dbm,dtype='f8')
    cb=np.asarray(rows.censor_upper_dbm,dtype='f8')
    if any(a.shape!=(n,) for a in (point,lo,hi,cb)):
        raise ValueError('Reported interval axes differ')
    if (not np.isfinite(point[heard]).all() or not np.isfinite(lo[heard]).all()
            or not np.isfinite(hi[heard]).all() or not np.isfinite(cb[censored]).all()
            or np.any(lo[heard]>=hi[heard]) or np.any(point[heard]<lo[heard])
            or np.any(point[heard]>hi[heard])):
        raise ValueError('A heard interval must contain its report; censor bounds are not point readings')
    known=heard|censored
    # Multiple simulated frontends/directions on one physical channel are
    # retained but do not masquerade as independent observation multiplicity.
    keys=np.column_stack((q['rx'],q['ap'],q['frequency']))
    _,inverse,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    weights=1./counts[inverse]
    reference=q['offset']+q['fspl']
    lower=np.where(heard,lo-reference,-np.inf)
    upper=np.where(heard,hi-reference,cb-reference)
    return q,known,heard,censored,point-reference,lower,upper,weights


def _log_interval_mass(a,b):
    # Compute a CDF difference in the better tail; expm1 preserves narrow bins.
    left=log_ndtr(b);right=log_ndtr(a)
    positive=a>0
    left=np.where(positive,log_ndtr(-a),left)
    right=np.where(positive,log_ndtr(-b),right)
    delta=np.minimum(right-left,-np.finfo('f8').eps)
    return left+np.log(-np.expm1(delta))


def _truncated_moments(mean,lower,upper,sigma):
    a=(lower-mean)/sigma;b=(upper-mean)/sigma
    logp=_log_interval_mass(a,b)
    log_phi_a=-.5*a*a-.5*math.log(2*math.pi)
    log_phi_b=-.5*b*b-.5*math.log(2*math.pi)
    ra=np.exp(np.clip(log_phi_a-logp,-745,700))
    rb=np.exp(np.clip(log_phi_b-logp,-745,700))
    ra=np.where(np.isneginf(a),0.,ra)
    rb=np.where(np.isposinf(b),0.,rb)
    correction=ra-rb
    ar=np.zeros_like(a);br=np.zeros_like(b)
    np.multiply(a,ra,out=ar,where=np.isfinite(a))
    np.multiply(b,rb,out=br,where=np.isfinite(b))
    variance=np.clip(1.+ar-br-correction*correction,0.,1.)*sigma*sigma
    expectation=mean+sigma*correction
    if not np.isfinite(expectation).all() or not np.isfinite(variance).all():
        raise FloatingPointError('Nonfinite interval/censored Gaussian moments')
    return expectation,variance


def _interval_posterior(x,lower,upper,weights,sigma,prior_variance,iterations=32):
    """Gaussian-prior MAP by interval EM; observed-information Laplace spread."""
    d=x.shape[1];precision=np.diag(1./np.asarray(prior_variance))
    if not len(x):return np.zeros(d),np.diag(prior_variance)
    a=precision+(x.T*weights)@x/(sigma*sigma)
    beta=np.zeros(d)
    for _ in range(iterations):
        expected,_=_truncated_moments(x@beta,lower,upper,sigma)
        update=np.linalg.solve(a,x.T@(weights*expected)/(sigma*sigma))
        if np.max(np.abs(update-beta))<1e-7:
            beta=update;break
        beta=update
    _,variance=_truncated_moments(x@beta,lower,upper,sigma)
    curvature=np.maximum(0.,1./(sigma*sigma)-variance/(sigma**4))
    covariance=np.linalg.inv(precision+(x.T*(weights*curvature))@x)
    return beta,covariance


@dataclass(frozen=True)
class MethodPrior:
    kind:str
    x_mean:np.ndarray
    x_scale:np.ndarray
    coef:np.ndarray
    covariance:np.ndarray
    local_variance:np.ndarray
    noise_db:float
    measurement_noise_db:float
    rff_w:np.ndarray
    rff_phase:np.ndarray

    def __post_init__(self):
        d=len(np.asarray(self.coef))
        if (self.kind not in METHODS or not math.isfinite(self.noise_db) or self.noise_db<=0
                or not math.isfinite(self.measurement_noise_db) or self.measurement_noise_db<=0):
            raise ValueError('Invalid numerical field prior')
        for name,shape in [('x_mean',(d,)),('x_scale',(d,)),('coef',(d,)),
                           ('covariance',(d,d)),('local_variance',(d,)),
                           ('rff_w',(N_RFF,2)),('rff_phase',(N_RFF,))]:
            value=_finite(getattr(self,name),shape,name)
            if name in ('x_scale','local_variance') and np.any(value<=0):raise ValueError('Positive prior scales required')
            object.__setattr__(self,name,_freeze(value))
        if not np.allclose(self.covariance,self.covariance.T,atol=1e-8):raise ValueError('Prior covariance must be symmetric')

    def raw_design(self,q):
        basis=np.asarray(query_physics_basis(q['geometry']),dtype='f8')
        if self.kind=='bayesian_operator':return basis
        # Same propagation-only CAD/GPS features as the neural decoder, plus the numerical
        # map's fixed Fourier shadowing basis in physical AP-relative XY.
        relative=q['rx'][:,:2]-q['ap'][:,:2]
        rff=math.sqrt(2./N_RFF)*np.cos(relative@self.rff_w.T+self.rff_phase)
        return np.column_stack((basis,q['geometry'][:,PROPAGATION_COLUMNS],rff))

    def design(self,q):return (self.raw_design(q)-self.x_mean)/self.x_scale

    def as_arrays(self):
        names=('x_mean','x_scale','coef','covariance','local_variance','noise_db','measurement_noise_db','rff_w','rff_phase')
        return {name:np.asarray(getattr(self,name)).copy() for name in names}

    @classmethod
    def from_arrays(cls,kind,a):
        expected={'x_mean','x_scale','coef','covariance','local_variance','noise_db','measurement_noise_db','rff_w','rff_phase'}
        if set(a)!=expected:raise ValueError('Field-prior archive membership differs')
        return cls(kind,*(a[name] for name in ('x_mean','x_scale','coef','covariance','local_variance')),
                   float(a['noise_db']),float(a['measurement_noise_db']),a['rff_w'],a['rff_phase'])


@dataclass(frozen=True)
class BaselinePrior:
    methods:dict[str,MethodPrior]
    _metadata:dict=field(default_factory=dict)

    def __post_init__(self):
        if set(self.methods)!=set(METHODS):raise ValueError('Both declared field controls are required')
        from types import MappingProxyType
        object.__setattr__(self,'methods',MappingProxyType(dict(self.methods)))
        object.__setattr__(self,'_metadata',deepcopy(self._metadata))

    @property
    def metadata(self):return deepcopy(self._metadata)

    def as_arrays(self):
        return {kind+'__'+name:value for kind,model in self.methods.items() for name,value in model.as_arrays().items()}

    @classmethod
    def from_arrays(cls,a,metadata=None):
        models={kind:MethodPrior.from_arrays(kind,{name.split('__',1)[1]:value
                    for name,value in a.items() if name.startswith(kind+'__')}) for kind in METHODS}
        if set(a)!={kind+'__'+name for kind,m in models.items() for name in m.as_arrays()}:
            raise ValueError('Unexpected global field-prior archive entries')
        return cls(models,metadata or {})


def learn_prior(training_rows):
    """Learn all global parameters from admitted TRAIN reports, never band2."""
    c=_common();c.check_sources()
    rows=tuple(training_rows)
    if not rows or any(r.stage!='training' for r in rows) or len({r.scene_id for r in rows})!=len(rows):
        raise ValueError('One admitted report block per distinct TRAIN scene is required')
    expected=[c.field_spec('training',i).scene_id for i in range(c.COUNTS['training'])]
    if [r.scene_id for r in rows]!=expected:
        raise ValueError('Numerical priors require exactly the registered TRAIN scenes in order')
    pieces=[]
    within_link_variances=[]
    for r in rows:
        q,known,heard,censored,point,lower,upper,w=_observations(r)
        if (not np.isin(np.asarray(r.device),(0,1,2)).all()
                or not np.isin(np.asarray(r.band_index),(0,1)).all()
                or not np.isin(q['frequency'],KNOWN_HZ).all()):
            raise ValueError('Held carrier/device reports cannot fit numerical priors')
        pieces.append((q,known,heard,point,lower,upper,w))
        keys=np.column_stack((q['rx'],q['ap'],q['frequency']))
        _,groups=np.unique(keys,axis=0,return_inverse=True)
        for group in np.unique(groups):
            keep=(groups==group)&heard
            if keep.sum()>1:
                quantization=np.mean(np.square(upper[keep]-lower[keep])/12.)
                within_link_variances.append(max(0.,float(np.var(point[keep],ddof=1))-quantization))
    q={key:np.concatenate([part[0][key] for part in pieces]) for key in pieces[0][0]}
    valid=np.concatenate([part[1] for part in pieces]);heard=np.concatenate([part[2] for part in pieces])
    point=np.concatenate([part[3] for part in pieces]);lower=np.concatenate([part[4] for part in pieces])
    upper=np.concatenate([part[5] for part in pieces]);weight=np.concatenate([part[6] for part in pieces])
    if not heard.any():raise ValueError('Global field priors need some actual positive TRAIN reports')
    rng=np.random.default_rng(2179321)
    omega=rng.normal(0,.1,(N_RFF,2));phase=rng.uniform(0,2*math.pi,N_RFF)
    measurement_noise=float(np.clip(np.sqrt(np.mean(within_link_variances)) if within_link_variances else .5,.25,10.))
    models={}
    for kind in METHODS:
        dimension=8+len(PROPAGATION_COLUMNS)+N_RFF if kind=='radiomap_operator' else 8
        empty=MethodPrior(kind,np.zeros(dimension),np.ones(dimension),np.zeros(dimension),
                          np.eye(dimension),np.ones(dimension),4.5,measurement_noise,omega,phase)
        x=empty.raw_design(q);mean=x[valid].mean(0);deviation=x[valid].std(0)
        scale=np.where(deviation>1e-6,deviation,1.)
        mean[0]=0.;scale[0]=1.  # retain the intercept, never normalize power away
        x=(x-mean)/scale
        noise=4.5;prior_variance=np.full(dimension,400.);prior_variance[0]=1600.
        for _ in range(2):
            coef,covariance=_interval_posterior(x[valid],lower[valid],upper[valid],weight[valid],noise,prior_variance)
            errors=point[heard]-x[heard]@coef
            effective=float(weight[heard].sum())
            denominator=max(effective-dimension,effective/2,1.)
            noise=float(np.clip(np.sqrt(np.sum(weight[heard]*errors*errors)/denominator),1.,20.))
        coef,covariance=_interval_posterior(x[valid],lower[valid],upper[valid],weight[valid],noise,prior_variance)
        local=np.full(dimension,noise*noise/max(1,dimension-1));local[0]=noise*noise
        models[kind]=MethodPrior(kind,mean,scale,coef,covariance,local,noise,measurement_noise,omega,phase)
    metadata={'training_scene_ids':[r.scene_id for r in rows], 'training_scenes':len(rows),
        'training_report_rows':len(valid),'known_report_rows':int(valid.sum()),
        'training_carriers_hz':list(KNOWN_HZ),'held_carrier_or_device_used':False,
        'frontends':'common paired-report estimates in device0 reference gauge; no true offset lookup',
        'noise':'TRAIN-only effective-physical-link weighted residual estimate, clipped1..20dB',
        'context_measurement_noise_db':measurement_noise,
        'context_noise_source':'TRAIN same-link calibrated co-report spread minus reported-bin quantization variance; working Gaussian scale, not an independence guarantee',
        'radiomap_core':'wirelessjev.native.radiomap.RadioMap._fit_one (unchanged source)',
        'native_censor_weight':CENSOR_WEIGHT,
        'quantization':'global/pooled field uses interval Gaussian likelihood; native local map uses reported interval centers and its Tobit censor likelihood',
        'spectral_transfer':'shared FSPL plus TRAIN-learned geometry/frequency prior and local log-frequency interpolation/extrapolation',
        'propagation_geometry_columns':list(PROPAGATION_COLUMNS),
        'frontend_status_noise_quantization_and_direction_are_not_mean_field_features':True,
        'TRAIN_query_measurements_used_as_permitted_prior_targets':True,
        'calibration_or_service_qualified':False}
    return BaselinePrior(models,metadata)


class _NativeDesignMap(RadioMap):
    """Matrix adapter to the native fitter; row handles are not model features."""
    def __init__(self,matrix,variance):
        zero={b:{'mean':np.zeros(3),'var':np.ones(3),'shadow_db':4.} for b in range(3)}
        super().__init__(np.zeros((1,2)),(None,),seed=0,prior=zero)
        self.matrix,self.variance=matrix,variance

    def _features(self,s,b,x,y):
        indices=np.asarray(x,dtype='i8')
        if not np.array_equal(indices,x):raise ValueError('Native matrix adapter requires integer feature-row handles')
        return self.matrix[indices]

    def _prior_precision(self,b=1):return np.diag(1./self.variance)


class FieldBaseline:
    def __init__(self,prior,method='radiomap_operator'):
        if not isinstance(prior,BaselinePrior) or method not in METHODS:raise ValueError('Unknown admitted numerical field control')
        self.prior=prior.methods[method];self.method=method;self.posterior={}

    def adapt(self,context_rows):
        q,known,heard,censored,point,lower,upper,weight=_observations(context_rows)
        if not np.isin(q['frequency'],KNOWN_HZ).all():raise ValueError('Context may contain observed carriers only')
        x=self.prior.design(q);mean=x@self.prior.coef
        if self.method=='bayesian_operator':
            self.posterior=_interval_posterior(x[known],lower[known]-mean[known],upper[known]-mean[known],
                weight[known],self.prior.measurement_noise_db,self.prior.local_variance)
        else:
            self.posterior={}
            for ap in np.unique(q['ap'],axis=0):
                selected_ap=np.all(q['ap']==ap,axis=1)
                for frequency in KNOWN_HZ:
                    selected=selected_ap&(q['frequency']==frequency)&known
                    h=selected&heard;z=selected&censored
                    matrix=np.concatenate((x[h]*np.sqrt(weight[h,None]),x[z]*np.sqrt(weight[z,None])))
                    factor=NOISE_DB/self.prior.measurement_noise_db
                    native=_NativeDesignMap(matrix,self.prior.local_variance*factor*factor)
                    nh=int(h.sum());nz=int(z.sum());ids=np.arange(nh,dtype=float);cids=np.arange(nh,nh+nz,dtype=float)
                    fit=native._fit_one(0,0,ids,np.zeros(nh),np.sqrt(weight[h])*factor*(point[h]-mean[h]),
                        cids,np.zeros(nz),np.sqrt(weight[z])*factor*(upper[z]-mean[z]))
                    self.posterior[(tuple(ap),frequency)]=(fit[0]/factor,fit[1]/(factor*factor))
        return self

    def predict(self,query_rows):
        q=_query(query_rows);x=self.prior.design(q)
        mean=x@self.prior.coef
        global_variance=np.maximum(np.sum((x@self.prior.covariance)*x,axis=1),0.)
        if self.method=='bayesian_operator':
            if not isinstance(self.posterior,tuple):raise RuntimeError('Adapt the numerical field to the same public context first')
            coefficient,covariance=self.posterior
            mean=mean+x@coefficient
            local=np.maximum(np.sum((x@covariance)*x,axis=1),0.)
        else:
            t=np.log(q['frequency']/KNOWN_HZ[0])/math.log(KNOWN_HZ[1]/KNOWN_HZ[0])
            local_std=np.zeros(len(x));correction=np.zeros(len(x))
            for frequency,mix in ((KNOWN_HZ[0],1-t),(KNOWN_HZ[1],t)):
                changed=dict(q);geometry=q['geometry'].copy();geometry[:,7]=math.log(frequency/KNOWN_HZ[0])
                changed['geometry']=geometry
                phi=self.prior.design(changed)
                for ap in np.unique(q['ap'],axis=0):
                    select=np.all(q['ap']==ap,axis=1)
                    coef,covariance=self.posterior.get((tuple(ap),frequency),
                        (np.zeros(len(self.prior.coef)),np.diag(self.prior.local_variance)))
                    correction[select]+=mix[select]*(phi[select]@coef)
                    # Absolute-weight combination is conservative to unknown
                    # cross-band posterior correlation, not independent errors.
                    variance=np.maximum(np.sum((phi[select]@covariance)*phi[select],axis=1),0.)
                    local_std[select]+=np.abs(mix[select])*np.sqrt(variance)
            mean=mean+correction;local=local_std*local_std
        # The spread includes learned model discrepancy, not just the linear
        # coefficient posterior. Unknown offset inversion remains uncertain.
        unknown=10.*q['geometry'][:,12]
        std=np.sqrt(global_variance+local+self.prior.noise_db**2+unknown**2)
        result=mean+q['fspl']+q['offset']
        if not np.isfinite(result).all() or not np.isfinite(std).all() or np.any(std<=0):
            raise FloatingPointError('Invalid numerical field forecast')
        return result,std


def _load_arrays(path):
    with np.load(path,allow_pickle=False) as saved:return {key:saved[key].copy() for key in saved.files}


def load_prior():
    global _CACHE
    c=_common();c.check_sources();receipt=c.read_json(c.OUT/'baselines/prior.json')
    if (receipt['source_receipt_sha256']!=c.digest(c.OUT/'registration/source.json')
            or receipt['protocol_sha256']!=c.digest(c.PROTOCOL)):
        raise RuntimeError('Numerical prior source identity changed')
    reference=receipt['archive'];file=c.safe_path(reference['path'])
    if c.digest(file)!=reference['sha256']:raise RuntimeError('Numerical prior bytes changed')
    prior=BaselinePrior.from_arrays(_load_arrays(file),receipt['metadata'])
    _CACHE=prior
    return prior


def prepare():
    """Fit and persist shared TRAIN calibration and numerical prior once."""
    global _CACHE
    from . import data
    c=_common();c.check_sources();target=c.OUT/'baselines/prior.json'
    if target.exists():load_prior();return c.read_json(target)
    scenes=[data.load_scene('training',i) for i in range(c.COUNTS['training'])]
    calibration=data.fit_frontends(scenes);data.save_calibration(calibration)
    prior=learn_prior([data.training_rows(scene,calibration) for scene in scenes])
    stream=io.BytesIO();np.savez_compressed(stream,**prior.as_arrays())
    file=c.OUT/'baselines/prior.npz';c.write_bytes_once(file,stream.getvalue())
    receipt={'schema':1,'source_receipt_sha256':c.digest(c.OUT/'registration/source.json'),
             'protocol_sha256':c.digest(c.PROTOCOL),
             'archive':{'path':str(file.relative_to(c.ROOT)),'sha256':c.digest(file)},
             'frontend_sha256':c.digest(c.OUT/'data/frontend-training.json'),
             'metadata':prior.metadata,'methods':list(METHODS),'no_evaluation_query_or_heldband_fit':True}
    c.write_once(target,receipt);_CACHE=prior
    return receipt


def predict(name,example,*,prior=None):
    """Measured fast path; call prepare/load_prior outside its timing scope."""
    chosen=_CACHE if prior is None else prior
    if chosen is None:raise RuntimeError('Prepare or load numerical priors before timing prediction')
    return FieldBaseline(chosen,name).adapt(example.context_rows_a).predict(example.query_rows)


def operator_only_fspl(example,*,spread_db=5.):
    """No learned material correction; spread is an explicit uncalibrated proxy."""
    if not math.isfinite(spread_db) or spread_db<=0:raise ValueError('A positive declared proxy spread is required')
    q=_query(example.query_rows)
    return q['fspl']+q['offset'],np.sqrt(spread_db**2+(10*q['geometry'][:,12])**2)


def direct_same_link_frequency(example):
    """Observed 5.2→6 GHz transport on genuinely co-located physical links.

    This supplementary conditional control assumes fixed material/antenna
    response and calibrated reciprocity when the query direction is reversed.
    Its spread accounts for quantization only, not unidentified material
    dispersion. It is not a calibrated risk head. Missing source pairs or
    unidentified frontend offsets produce NaN + eligible=False, never zeros.
    No measured query power, interval, heard mask or target is accessed.
    """
    source=example.context_rows_a;target=example.query_rows
    q=_query(target);s=_query(source)
    heard=np.asarray(source.heard)
    direction=np.asarray(source.direction)
    if heard.dtype!=bool or heard.shape!=(len(s['geometry']),) or direction.shape!=heard.shape:
        raise ValueError('Actual source heard/direction evidence is required')
    value=np.asarray(source.reported_rssi_dbm,dtype='f8')
    lower=np.asarray(source.lower_dbm,dtype='f8');upper=np.asarray(source.upper_dbm,dtype='f8')
    if any(a.shape!=heard.shape for a in (value,lower,upper)):
        raise ValueError('Source report/interval axes differ')
    mean=np.full(len(q['geometry']),np.nan);std=mean.copy();eligible=np.zeros(len(mean),bool)
    source_ok=(heard&(direction==0)&(s['frequency']==KNOWN_HZ[1])&(s['geometry'][:,12]==0))
    for row in range(len(mean)):
        if q['frequency'][row]!=6.e9 or q['geometry'][row,12]!=0:continue
        same=source_ok&np.all(s['rx']==q['rx'][row],axis=1)&np.all(s['ap']==q['ap'][row],axis=1)
        indices=np.flatnonzero(same)
        if not len(indices):continue
        index=int(indices[0])
        if (not np.isfinite(value[index]) or not np.isfinite(lower[index]) or not np.isfinite(upper[index])
                or not lower[index]<upper[index]):raise ValueError('Eligible source needs a real reported quantization interval')
        mean[row]=(value[index]-s['offset'][index]
                   +20*math.log10(s['frequency'][index]/q['frequency'][row])+q['offset'][row])
        target_width=4*q['geometry'][row,13]
        std[row]=math.sqrt(max(.25**2,((upper[index]-lower[index])**2+target_width**2)/12.))
        eligible[row]=True
    return mean,std,eligible


def verify_prior():
    """Passive byte/shape/provenance checks; no global or context fitting."""
    c=_common();prior=load_prior();receipt=c.read_json(c.OUT/'baselines/prior.json')
    expected=[c.field_spec('training',i).scene_id for i in range(c.COUNTS['training'])]
    metadata=receipt['metadata']
    if (metadata['training_scene_ids']!=expected or metadata['training_scenes']!=len(expected)
            or metadata['training_carriers_hz']!=list(KNOWN_HZ)
            or metadata['held_carrier_or_device_used'] is not False
            or receipt['no_evaluation_query_or_heldband_fit'] is not True
            or receipt['methods']!=list(METHODS)
            or metadata['frontend_status_noise_quantization_and_direction_are_not_mean_field_features'] is not True):
        raise RuntimeError('Numerical prior fit eligibility changed')
    if receipt['frontend_sha256']!=c.digest(c.OUT/'data/frontend-training.json'):
        raise RuntimeError('Shared reported-pair frontend calibration changed')
    frontend=c.read_json(c.OUT/'data/frontend-training.json')
    if (frontend['absolute_offset_identified'] is not False or frontend['registry_true_offsets_used'] is not False
            or frontend['query_reports_or_band2_used'] is not False):
        raise RuntimeError('Frontend inverse used private/unadmitted calibration data')
    return {'passes':True,'methods':list(METHODS),'training_scenes':len(expected),
            'heldband_or_helddevice_used':False,'new_fits_or_forward_passes':False,
            'native_core':metadata['radiomap_core'],'prior_archive_sha256':receipt['archive']['sha256']}
