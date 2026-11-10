"""Observed-only map fitting; declared shared-oracle planner metadata.

The fit object exposes only AP XY/band/TX and assumed-known height. Planner
side information is separately labeled oracle-conditioned; private RF arrays
are stored separately and are never passed to the map constructor or fitter.
"""
from copy import deepcopy
from dataclasses import dataclass
import importlib.util

import numpy as np

from ..envlearn import RadioMap,Survey
from ..fastplan import BatchEvaluator,Snapshot
from ..twin.radio import BANDS
from ..twin import warehouse as W
from . import common as c


@dataclass(frozen=True)
class PublicAP:
    x: float
    y: float
    band: str
    tx_power_dbm: float


@dataclass(frozen=True)
class PublicFitInputs:
    aps: tuple[PublicAP,...]
    ap_height: float


def public_fit_inputs(aps,ap_height):
    result=PublicFitInputs(tuple(PublicAP(float(a.x),float(a.y),str(a.band),float(a.tx_power_dbm)) for a in aps),float(ap_height))
    if (not result.aps or not np.isfinite(result.ap_height) or result.ap_height<=0
            or any(a.band not in BANDS or not np.isfinite([a.x,a.y,a.tx_power_dbm]).all() for a in result.aps)):
        raise ValueError('Known public AP geometry/power/height required')
    return result


def calibrated_class():
    spec=importlib.util.spec_from_file_location('e7_calplanning_fixed_calibrator',c.CALIBRATOR)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.CalibratedMap


def fit_map(kind,public,survey,seed):
    """Pure helper also exercised on explicit analytic software fixtures."""
    if kind not in c.KINDS or not isinstance(public,PublicFitInputs):raise ValueError('Fixed variant and public whitelist required')
    cls=RadioMap if kind=='corrected_v2' else calibrated_class()
    args={} if kind=='corrected_v2' else {'geometry':True,'censored':True}
    model=cls(public,public.ap_height,D=200,length_m=8.,prior_std_db=8.,noise_std_db=4.,seed=seed,**args)
    model.fit(survey);return model


def survey_arrays(survey):
    values={name:np.asarray(getattr(survey,name),dtype='i8' if name in ('ap','cap') else 'f8')
            for name in ('x','y','ap','rssi','cx','cy','cap','cbound')}
    values['link']=np.asarray(survey.link,dtype='f8').reshape(-1,4)
    keys=sorted(survey.busy)
    values['busy_keys']=np.asarray([f'{band}:{channel}' for band,channel in keys],dtype='U24')
    for i,key in enumerate(keys):values[f'busy_{i}']=np.asarray(survey.busy[key],dtype='f8')
    values['censor_weight']=np.asarray(survey.censor_weight,dtype='f8');return values


def survey_from_arrays(values):
    fields={name:values[name].tolist() for name in ('x','y','ap','rssi','cx','cy','cap','cbound')}
    fields['link']=[tuple(row) for row in values['link'].tolist()]
    fields['busy']={tuple([str(key).split(':')[0],int(str(key).split(':')[1])]):values[f'busy_{i}'].tolist()
                    for i,key in enumerate(values['busy_keys'])}
    return Survey(**fields,censor_weight=float(values['censor_weight']))


def tail_subset(survey,budget):
    """EXACT original flattened record-tail rule, never a chronological prefix."""
    if budget not in c.BUDGETS:raise ValueError('Fixed nominal tail budget required')
    n=budget*12*(len(survey.rssi)//(15*12));nc=int(len(survey.cx)*budget/15);frac=budget/15
    # Preserve original [-0:] behavior as well; these are legacy record tails.
    return Survey(survey.x[-n:],survey.y[-n:],survey.ap[-n:],survey.rssi[-n:],
        survey.cx[-nc:] if nc else [],survey.cy[-nc:] if nc else [],survey.cap[-nc:] if nc else [],survey.cbound[-nc:] if nc else [],
        survey.link[-int(len(survey.link)*frac):],
        {k:v[-max(1,int(len(v)*frac)):] for k,v in survey.busy.items()},censor_weight=survey.censor_weight)


# Explicit admission: visibility/domains/interferer/protected flags are supplied
# by the original truth infrastructure in BOTH arms. No claim of observed-only
# planning follows. Truth DL/UL, external powers and Scene pointers are excluded.
ARRAY_KEYS=('dom_of_ap','noise_mw','is5','bc_key','ext','crit','be','demand','cf_dem','cf_floor',
            'current','t5','r5','t2','r2','interferer','leak')
SCALAR_KEYS=('n','m','nd','nkeys')


def shared_oracle_arrays(truth):
    result={name:np.array(getattr(truth,name),copy=True) for name in ARRAY_KEYS}
    result.update({name:np.asarray(getattr(truth,name),dtype='i8') for name in SCALAR_KEYS})
    result['car_ids']=np.asarray(truth.car_ids,dtype='U64');result['ap_ids']=np.asarray(truth.ap_ids,dtype='U64')
    visible=np.zeros((truth.n,truth.m),bool)
    for i,indices in enumerate(truth.visible):visible[i,indices]=True
    result['visible']=visible
    return result


def evaluator(shared,snaps,*,external_noise,availability):
    ev=BatchEvaluator.__new__(BatchEvaluator)
    for name in ARRAY_KEYS:setattr(ev,name,np.array(shared[name],copy=True))
    for name in SCALAR_KEYS:setattr(ev,name,int(shared[name]))
    ev.car_ids=shared['car_ids'].tolist();ev.ap_ids=shared['ap_ids'].tolist()
    ev.visible=[np.flatnonzero(row) for row in shared['visible']]
    ev.client_itf=True;ev.ext_noise_mw=np.array(external_noise,copy=True);ev.avail=np.array(availability,copy=True)
    ev.snaps=[Snapshot(np.array(ul,copy=True),np.array(dl,copy=True)) for ul,dl in snaps]
    return ev


def make_belief(shared,positions,public,survey,mean_predict):
    """Same original E7 learned evaluator, injecting a fitted mean-map callable."""
    ap_tx=np.asarray([a.tx_power_dbm for a in public.aps]);car_tx=float(shared['client_tx_dbm'])
    snaps=[]
    for xy in positions:
        mean=mean_predict(xy[:,0],xy[:,1]);snaps.append((mean-ap_tx[None]+car_tx,mean))
    first=[int(np.flatnonzero(shared['dom_of_ap']==d)[0]) for d in range(int(shared['nd']))]
    occ={key:float(np.mean(value)) for key,value in survey.busy.items()}
    bands=shared['band_names'];channels=shared['channels']
    available=np.asarray([1.-occ.get((str(bands[j]),int(channels[j])),0.) for j in first]) if survey.busy else np.ones(len(first))
    est=np.zeros(len(public.aps));link=np.asarray(survey.link,dtype='f8').reshape(-1,4)
    if len(link):
        ids=link[:,2].astype(int);mean=mean_predict(link[:,0],link[:,1])
        implied=10**((mean[np.arange(len(ids)),ids]-ap_tx[ids]+car_tx-link[:,3])/10)-shared['noise_mw'][ids]
        for j in range(len(est)):
            values=implied[ids==j];est[j]=float(np.clip(np.median(values),0,None))*.5 if len(values) else 0.
    return evaluator(shared,snaps,external_noise=est,availability=available)


def mean_state(model,kind):
    m=len(model.aps);coef=np.zeros((m,model.D+1));present=np.zeros(m,bool)
    for j,(w,_) in model.post.items():coef[j]=w;present[j]=True
    return {'Wf':model.Wf.copy(),'bf':model.bf.copy(),'coefficient':coef,'present':present,
        'law':np.asarray([model.law[b] for b in BANDS]),
        'rack_scale':np.asarray([getattr(model,'rack_scale',{}).get(b,0.) for b in BANDS]),
        'kind':np.asarray(kind,dtype='U32')}


def replay_mean(state,public,xs,ys):
    """Mean-only retained state: no fake covariance or calibrated spread."""
    x=np.asarray(xs);y=np.asarray(ys);features=np.column_stack((np.sqrt(2/len(state['bf']))*np.cos(
        np.column_stack((x,y))@state['Wf'].T+state['bf']),np.ones(len(x))))
    result=np.empty((len(x),len(public.aps)));bands=list(BANDS)
    for j,ap in enumerate(public.aps):
        band=bands.index(ap.band);a,b=state['law'][band]
        d=np.maximum(1.,np.sqrt((ap.x-x)**2+(ap.y-y)**2+public.ap_height**2))
        prior=ap.tx_power_dbm-(a+b*np.log10(d))
        prior-=state['rack_scale'][band]*W.rack_obstruction(ap,x,y)
        result[:,j]=prior+(features@state['coefficient'][j] if state['present'][j] else 0.)
    return result
