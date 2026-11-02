"""Calibrate relative AP gain, then validate the bound in closed-loop control."""
from dataclasses import dataclass
import numpy as np
from scipy.stats import spearmanr


@dataclass
class GainCalibrator:
    center: np.ndarray
    spread: np.ndarray
    bias: np.ndarray
    log_scale: np.ndarray
    quantiles: dict

    def design(self,x):return np.column_stack([np.ones(len(x)),(x-self.center)/self.spread])

    def estimate(self,x):
        f=self.design(x);bias=f@self.bias
        scale=np.sqrt(np.exp(np.clip(f@self.log_scale,-1,3))**2+(x[:,1]*5)**2)
        return bias,np.maximum(scale,.25)

    def to_dict(self):return {k:getattr(self,k).tolist() for k in ['center','spread','bias','log_scale']}|{'quantiles':self.quantiles,'target':'overprediction of candidate-minus-current native AP score'}

    @classmethod
    def from_dict(cls,d):return cls(*(np.array(d[k]) for k in ['center','spread','bias','log_scale']),d['quantiles'])


def fit_model(x,y,cal_x,cal_y,quantiles=(.5,.75,.9)):
    if len(x)<30 or len(cal_x)<10:raise ValueError('Insufficient eligible decisions for calibration')
    center=x.mean(0);spread=np.maximum(x.std(0),.05);f=np.column_stack([np.ones(len(x)),(x-center)/spread])
    penalty=np.eye(f.shape[1])*10;penalty[0,0]=1e-6
    solve=lambda target:np.linalg.solve(f.T@f+penalty,f.T@target)
    bias=solve(y);log_scale=solve(np.log(np.abs(y-f@bias)+.25))
    model=GainCalibrator(center,spread,bias,log_scale,{})
    mu,sd=model.estimate(cal_x);normalized=(cal_y-mu)/sd
    model.quantiles={str(q):float(np.quantile(normalized,q,method='higher')) for q in quantiles}
    return model


class GainGate:
    def __init__(self,bank,model,quantile):self.bank=bank;self.model=model;self.quantile=str(quantile)
    def __call__(self,ctx):
        x=self.bank.features(ctx);bias,scale=self.model.estimate(x)
        bound=bias+self.model.quantiles[self.quantile]*scale
        accepted=(ctx.gain-bound>=0)&ctx.eligible
        return accepted&np.isfinite(bound)


def reactive_rescue(ctx):return ctx.eligible.copy()


class Recorder:
    """Record observable decisions. Labels are supplied only AFTER simulation."""
    def __init__(self,bank):self.bank=bank;self.parts=[]
    def __call__(self,ctx,accepted,after):
        self.parts.append({'decision':np.full(len(ctx.current),ctx.decision),'robot':np.arange(len(ctx.current)),'current':ctx.current.copy(),'candidate':ctx.candidate.copy(),
            'gain':ctx.gain.copy(),'eligible':ctx.eligible.copy(),'accepted':accepted,'after':after,'features':self.bank.features(ctx),**self.bank.uncertainty(ctx)})
    def label(self,truth):
        rows={k:np.concatenate([p[k] for p in self.parts]) for k in self.parts[0]}
        d=rows['decision'];r=rows['robot'];a=rows['current'];b=rows['candidate']
        predicted=self.bank.score;actual=truth.reshape(self.bank.mean.shape)[:,:,:6].max(-1).mean(2)
        errors=predicted-actual
        rows['error']=errors[d,r,b]-errors[d,r,a];rows['true_gain']=rows['gain']-rows['error']
        return rows


def diagnostics(rows):
    e=rows['eligible'];denied=e&~rows['accepted'];positive=rows['true_gain']>0
    def mean(v):return float(np.mean(v)) if len(v) else None
    def rho(a,b):
        if len(a)<3:return None
        r=float(spearmanr(a,b).statistic);return r if np.isfinite(r) else None
    return {'opportunities':int(e.sum()),'rejected':int(denied.sum()),'rejection_fraction':float(denied.sum()/max(e.sum(),1)),
        'beneficial_rejected':int((denied&positive).sum()),'beneficial_rejected_without_rescue':int((denied&positive&(rows['after']==rows['current'])).sum()),
        'harmful_accepted':int((e&rows['accepted']&~positive).sum()),'median_abs_gain_error_db':float(np.median(np.abs(rows['error'][e]))) if e.any() else None,
        'median_legacy_uncertainty_db':float(np.median(rows['legacy_u'][e])) if e.any() else None,
        'median_active_band_uncertainty_db':float(np.median(rows['active_u'][e])) if e.any() else None,
        'mean_ensemble_pair_spread_db':mean(rows['ensemble_pair_u'][e]),'mean_ensemble_independent_spread_db':mean(rows['ensemble_independent_u'][e]),
        'legacy_gain_error_spearman':rho(rows['legacy_u'][e],np.abs(rows['error'][e]))}
