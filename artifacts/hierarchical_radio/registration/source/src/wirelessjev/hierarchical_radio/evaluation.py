"""Passive reported-RF reductions, grouped by geometry rather than query count."""
from __future__ import annotations

import math
import numpy as np
from scipy.special import log_ndtr


def interval_log_mass(lower,upper,mean,std):
    lower,upper,mean,std=np.broadcast_arrays(np.asarray(lower,dtype='f8'),np.asarray(upper,dtype='f8'),
        np.asarray(mean,dtype='f8'),np.asarray(std,dtype='f8'))
    if (not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0)
            or np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower>=upper)):
        raise ValueError('Finite Gaussian parameters and ordered interval events required')
    a=(lower-mean)/std;b=(upper-mean)/std
    # Use a survival difference in the positive tail, avoiding subtraction of
    # two CDFs that round to one. expm1 retains narrow quantization intervals.
    positive=a>=0
    x=np.where(positive,log_ndtr(-a),log_ndtr(b))
    y=np.where(positive,log_ndtr(-b),log_ndtr(a))
    with np.errstate(divide='ignore',invalid='ignore'):
        result=x+np.log(-np.expm1(np.minimum(y-x,0.)))
    if not np.isfinite(result).all():raise FloatingPointError('Unresolved Gaussian interval probability')
    return result


def target_events(target):
    heard=np.asarray(target['heard']);censored=np.asarray(target['censored'])
    if heard.dtype!=bool or censored.dtype!=bool or heard.shape!=censored.shape or np.any(heard&censored):
        raise ValueError('Genuine reported target masks required')
    known=heard|censored
    lower=np.where(censored,-np.inf,np.maximum(target['lower_dbm'],target['detection_threshold_dbm']))
    upper=np.where(censored,target['censor_upper_dbm'],target['upper_dbm'])
    if (not known.any() or np.isnan(lower[known]).any() or np.isnan(upper[known]).any()
            or np.any(lower[known]>=upper[known])):raise ValueError('Known target intervals are malformed')
    return heard,censored,known,lower,upper


def metrics(mean,std,target):
    mean=np.asarray(mean,dtype='f8');std=np.asarray(std,dtype='f8')
    heard,censored,known,lower,upper=target_events(target)
    if (mean.shape!=known.shape or std.shape!=mean.shape or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std<=0) or not heard.any()):
        raise ValueError('Complete finite field predictions and heard targets required')
    center=np.asarray(target['center_dbm'],dtype='f8')
    if center.shape!=mean.shape or not np.isfinite(center[heard]).all():
        raise ValueError('Finite reported heard target centers required')
    error=mean[heard]-center[heard]
    nll=-interval_log_mass(lower[known],upper[known],mean[known],std[known])
    return {'rmse_db':float(np.sqrt(np.mean(error**2))), 'mae_db':float(np.mean(np.abs(error))),
        'p95_abs_error_db':float(np.percentile(np.abs(error),95)),
        'composite_interval_nll':float(np.mean(nll)),
        'heard_interval_95_coverage':float(np.mean(np.abs(error)<=1.959963984540054*std[heard])),
        'mean_std_db':float(np.mean(std)), 'heard':int(heard.sum()),
        'censored':int(censored.sum()),'unknown':int((~known).sum())}


def validation_nll(mean,std,targets):
    """Equal-example composite NLL; no center/point requirement for selection."""
    mean=np.asarray(mean);std=np.asarray(std)
    if (mean.ndim!=2 or std.shape!=mean.shape or not mean.size
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0)
            or any(np.asarray(value).shape!=mean.shape for value in targets.values())):
        raise ValueError('Complete finite example/query prediction and target axes required')
    values=[]
    for i in range(len(mean)):
        target={name:np.asarray(value)[i] for name,value in targets.items()}
        _,_,known,lo,hi=target_events(target)
        weights=np.asarray(target.get('weights',known.astype(float)),dtype=float)*known
        if not np.isfinite(weights).all() or np.any(weights<0) or not weights.sum()>0:
            raise ValueError('Positive validation target mass required')
        values.append(float(np.sum(-interval_log_mass(lo[known],hi[known],mean[i,known],std[i,known])*weights[known])/weights.sum()))
    return float(np.mean(values))


def reduce_layouts(records,*,methods,layout_ids,states=(0,1,2)):
    """Average member metrics inside state, then states inside each layout."""
    if (not methods or len(set(layout_ids))!=len(layout_ids) or not layout_ids
            or len(set(states))!=len(states) or not states
            or any(not names or len(set(names))!=len(names) for names in methods.values())):
        raise ValueError('Unique nonempty declared member/layout/state cohorts required')
    members=set().union(*map(set,methods.values()))
    if any(r['method'] not in members or r['layout_id'] not in layout_ids or r['state'] not in states for r in records):
        raise ValueError('Unregistered record cohort')
    if (not records or any(not r['metrics'] or set(r['metrics'])!=set(records[0]['metrics'])
            or not np.isfinite(list(r['metrics'].values())).all() for r in records)):
        raise ValueError('Consistent finite member metrics required')
    result={}
    for method,names in methods.items():
        result[method]=[]
        for layout in layout_ids:
            rows=[]
            for state in states:
                matching=[r for r in records if r['layout_id']==layout and r['state']==state and r['method'] in names]
                if {r['method'] for r in matching}!=set(names) or len(matching)!=len(names):
                    raise ValueError('Exact member/state/layout membership required')
                keys=matching[0]['metrics']
                rows.append({k:float(np.mean([r['metrics'][k] for r in matching])) for k in keys})
            result[method].append({'layout_id':layout,'metrics':{k:float(np.mean([row[k] for row in rows])) for k in rows[0]}})
    return result


def paired_interval(candidate,reference,*,samples=20000,seed=21770):
    if type(samples) is not int or samples<=0 or type(seed) is not int or seed<0:
        raise ValueError('Positive integer bootstrap samples and nonnegative integer seed required')
    a=np.asarray(candidate,dtype=float);b=np.asarray(reference,dtype=float)
    if a.shape!=b.shape or a.ndim!=1 or len(a)!=24 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Exactly 24 whole-layout paired metrics required')
    delta=a-b
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(delta),(samples,len(delta)))
    lo,hi=np.quantile(delta[indices].mean(1),(.00625,.99375))
    return {'candidate_mean':float(a.mean()),'reference_mean':float(b.mean()),
        'mean_delta':float(delta.mean()),'central_98_75_interval':[float(lo),float(hi)],
        'candidate_lower_error_supported':bool(hi<0),'reference_lower_error_supported':bool(lo>0),
        'layouts':24,'bootstrap_samples':samples,'seed':seed}
