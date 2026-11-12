"""End-to-end CPU forecast timings, including causal memory and map refits."""
import time,json
import numpy as np
import torch
from wirelessjev.fleet_control import FleetWorldModel
from wirelessjev.fleetwm.study import OUT,P
from wirelessjev.fleetwm.data import episode,fit_map,map_prediction
from wirelessjev.fleet_strong_reference import fit_strong_map
from wirelessjev.contracted.train import write


def main():
    torch.set_num_threads(1)
    # A previously used validation site; this is compute measurement only.
    full,scan,g,obs=episode(P['validation_sites'][0],100,30,'main')
    results={}
    for family in P['families']:
        a=time.perf_counter();model=FleetWorldModel(g,family=family);load_seconds=time.perf_counter()-a
        rows=[]
        for t in [60,61,62,63,64,80]:
            m,s,timing=model.forecast(obs.before(t),full.plan[t]);rows.append(timing)
            assert m.shape==(100,9,16,3) and np.isfinite(m).all() and (s>0).all()
        results[family]={'checkpoint_load_seconds':load_seconds,'initial_forecast_seconds':rows[0]['total_seconds'],
            'warm_forecast_mean_seconds':float(np.mean([r['total_seconds'] for r in rows[1:5]])),
            'refresh_forecast_seconds':rows[-1]['total_seconds'],'calls':rows,'ensemble_members':3}
    for family,fit in [('rolling_radiomap',fit_map),('learned_radiomap',fit_strong_map)]:
        rows=[];rmap=None
        for t in [60,61,62,63,64,80]:
            a=time.perf_counter();refit=0
            if rmap is None or t==80:
                start=time.perf_counter();rmap=fit(g,obs.before(t));refit=time.perf_counter()-start
            map_prediction(g,rmap,full.plan[t]);rows.append({'total_seconds':time.perf_counter()-a,'map_refit_seconds':refit})
        results[family]={'initial_forecast_seconds':rows[0]['total_seconds'],
            'warm_forecast_mean_seconds':float(np.mean([r['total_seconds'] for r in rows[1:5]])),
            'refresh_forecast_seconds':rows[-1]['total_seconds'],'calls':rows}
    receipt={'cpu_only':True,'torch_threads':1,'robots':100,'query_horizons':9,'site':P['validation_sites'][0],
        'includes_input_construction_and_refits':True,'checkpoint_loading_separate':True,
        'no_hardware_inference_qualification':True,'results':results}
    write(OUT/'runtime_benchmark.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
