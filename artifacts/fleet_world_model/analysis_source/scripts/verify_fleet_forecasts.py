"""Recompute a complete primary case's forecasts independently of native runs."""
import json,hashlib,time
import numpy as np
import torch
from wirelessjev.fleetwm.study import ROOT,OUT,P,load,predict,forecast_metrics
from wirelessjev.fleetwm.data import episode,fit_map,inputs,map_prediction
from wirelessjev.native.world import HORIZONS
from wirelessjev.contracted.train import write


def main():
    torch.set_num_threads(1);seed=P['test_main_sites'][0];case=json.loads((OUT/'cases'/f'test-main-{seed}.json').read_text())
    began=time.time();duration=case['duration_s'];n=case['robots']
    full,scan,g,obs=episode(seed,n,duration,'main');initial=fit_map(g,obs.before(60));rmap=initial
    models={f:load(f) for f in P['families']};predictions={f:[[],[]] for f in ['static_radiomap','rolling_radiomap',*P['families']]}
    for d in range(duration):
        t=60+d
        if d>0 and d%P['map_refresh_seconds']==0:rmap=fit_map(g,obs.before(t))
        b=inputs(g,obs.before(t),full.plan[t],rmap)
        for f in predictions:
            if f in models:
                m,s=predict(models[f],b);m=m.reshape(n,9,-1,3);s=s.reshape(m.shape)
            else:m,s=map_prediction(g,initial if f=='static_radiomap' else rmap,full.plan[t])
            predictions[f][0].append(m);predictions[f][1].append(s)
    future=np.arange(60,60+duration)[:,None]+np.asarray(HORIZONS,dtype=int)
    truth=full.snr[future*5].transpose(0,2,1,3,4);checked={}
    for f,(m,s) in predictions.items():
        m=np.asarray(m,dtype='f4');s=np.asarray(s,dtype='f4');h=hashlib.sha256(m.tobytes()).hexdigest()
        row=next(r for r in case['rows'] if r['family']==f and r['mission']==P['primary_mission'])
        assert row['forecast_sha256']==h,f
        metrics=forecast_metrics(m,s,truth,g)
        for k,v in metrics.items():assert np.isclose(v,case['forecast_metrics'][f][k],rtol=1e-7,atol=1e-7),(f,k)
        checked[f]={'forecast_sha256':h,'forecast_metrics_reproduced':True}
    receipt={'passed':True,'scene':seed,'complete_fleet_decisions':duration,'families':checked,'elapsed_s':time.time()-began}
    write(OUT/'forecast_reproduction.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
