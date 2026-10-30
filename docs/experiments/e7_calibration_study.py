"""Matched-survey calibration ablation on environment seeds not used in E7 v2."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr
from wirelessjev.envlearn import (RadioMap, Survey, drive_and_survey, random_env,
                                  truth_scene)
from wirelessjev.twin.radio import BANDS
from wirelessjev.twin import warehouse as W


class CalibratedMap(RadioMap):
    def __init__(self, *args, geometry=True, censored=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry = geometry
        self.censored_calibration = censored
        self.rack_scale = {band: 1.0 for band in BANDS}

    def features(self, x, y, ap):
        n = len(x)
        X = np.ones((n, 3 if self.geometry else 2))
        tx = np.empty(n)
        for j in np.unique(ap):
            selected = ap == j
            X[selected, 1] = np.log10(self._dist(j, x[selected], y[selected]))
            tx[selected] = self.aps[j].tx_power_dbm
            if self.geometry:
                X[selected, 2] = W.rack_obstruction(self.aps[j], x[selected], y[selected])
        return X, tx

    def prior_dl(self, j, xs, ys):
        mean = super().prior_dl(j, xs, ys)
        if self.geometry:
            mean -= self.rack_scale[self.aps[j].band] * W.rack_obstruction(self.aps[j], xs, ys)
        return mean

    def calibrate(self, survey):
        x,y,ap,v = map(np.asarray,(survey.x,survey.y,survey.ap,survey.rssi))
        X,tx = self.features(x,y,ap)
        cx,cy,ca,cb = (np.asarray(survey.cx),np.asarray(survey.cy),
                       np.asarray(survey.cap,dtype=int),np.asarray(survey.cbound))
        C,ctx = self.features(cx,cy,ca)
        bands = np.array([a.band for a in self.aps])
        for band in BANDS:
            selected = bands[ap] == band
            cens = bands[ca] == band
            if selected.sum() < 20:
                continue
            A = X[selected]
            loss = tx[selected]-v[selected]
            P = C[cens]
            offset = cb[cens]-ctx[cens]
            prior = np.array([BANDS[band][0],30.0]+([1.0] if self.geometry else []))
            scale = np.array([10.0,10.0]+([1.0] if self.geometry else []))
            sigma = np.sqrt(self.sn2)
            def objective(beta):
                residual = A @ beta-loss
                standardized = (beta-prior)/scale
                value = 0.5*(residual@residual/self.sn2+standardized@standardized)
                gradient = A.T@residual/self.sn2+(beta-prior)/scale**2
                if self.censored_calibration and len(P):
                    z=(offset+P@beta)/sigma
                    lcdf=log_ndtr(z)
                    mills=np.exp(-0.5*z*z-0.5*np.log(2*np.pi)-lcdf)
                    value-=survey.censor_weight*lcdf.sum()
                    gradient-=survey.censor_weight*P.T@mills/sigma
                return value,gradient
            bounds=[(None,None),(15.0,45.0)]+([(0.0,4.0)] if self.geometry else [])
            result=minimize(objective,prior,jac=True,method='L-BFGS-B',bounds=bounds,
                            options={'maxiter':300,'maxls':100,'ftol':1e-10,'gtol':1e-5})
            if not result.success:
                # The three calibration columns can be strongly correlated.
                # Retry the same convex objective with a second bounded solver
                # and a per-observation scaling; this does not change its optimum.
                normalizer=len(A)+survey.censor_weight*len(P)
                def normalized(beta):
                    value,gradient=objective(beta)
                    return value/normalizer,gradient/normalizer
                result=minimize(normalized,result.x,jac=True,method='SLSQP',bounds=bounds,
                                options={'maxiter':300,'ftol':1e-12})
                if not result.success:
                    raise RuntimeError(result.message)
            self.law[band]=tuple(result.x[:2])
            if self.geometry:
                self.rack_scale[band]=float(result.x[2])


def subset(survey,budget,total=15):
    n=budget*12*(len(survey.rssi)//(total*12))
    nc=int(len(survey.cx)*budget/total)
    return Survey(survey.x[-n:],survey.y[-n:],survey.ap[-n:],survey.rssi[-n:],
                  survey.cx[-nc:],survey.cy[-nc:],survey.cap[-nc:],survey.cbound[-nc:],
                  censor_weight=survey.censor_weight)


def run(seed):
    started=time.monotonic()
    spec=random_env(seed)
    scene,survey=drive_and_survey(truth_scene(30000+seed,spec),spec,15,np.random.default_rng(seed))
    truth=scene.radio().dl
    xs=np.array([c.x for c in scene.cars]);ys=np.array([c.y for c in scene.cars])
    output={'seed':seed,'budgets':{},'assumptions':'AP locations/powers/height and nominal warehouse layout known; hidden materials, walls and truth coefficients unused in fit'}
    for budget in [1,5,15]:
        sv=subset(survey,budget)
        output['budgets'][str(budget)]={}
        for method,options in [
            ('v2',None),('censored_distance',{'geometry':False,'censored':True}),
            ('geometry_detected',{'geometry':True,'censored':False}),
            ('geometry_censored',{'geometry':True,'censored':True})]:
            cls=RadioMap if options is None else CalibratedMap
            model=cls(scene,spec.ap_height,seed=seed,**(options or {}))
            model.fit(sv)
            mean,std=model.predict(xs,ys)
            assert np.isfinite(mean).all() and np.isfinite(std).all()
            metrics={key:float(np.sqrt(np.mean((truth[mask]-mean[mask])**2)))
                     for key,mask in [('audible',truth>=-90),('inaudible',truth<-90),
                                      ('all',np.ones(truth.shape,bool))] if mask.any()}
            output['budgets'][str(budget)][method]={'rmse_db':metrics,'law':model.law,
                                                   'rack_scale':getattr(model,'rack_scale',None)}
    output['elapsed_s']=time.monotonic()-started
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(15,30)))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    environments=[]
    for seed in args.seeds:
        path=args.out/f'environment_{seed}.json'
        if path.exists():
            result=json.loads(path.read_text())
        else:
            result=run(seed)
            path.write_text(json.dumps(result,indent=2)+'\n')
        environments.append(result)
        print(f'completed {len(environments)}/{len(args.seeds)}: seed {seed}, {result["elapsed_s"]:.1f}s',flush=True)
    methods=list(environments[0]['budgets']['1'])
    summary={str(b):{m:{k:float(np.mean([e['budgets'][str(b)][m]['rmse_db'][k] for e in environments]))
                       for k in ['audible','inaudible','all']} for m in methods} for b in [1,5,15]}
    report={'seeds':args.seeds,'map_only':True,'summary':summary,'environments':environments}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
