"""Frozen-model, fresh-scene native rollouts and uncertainty-only interventions."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import argparse,hashlib,json,time,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from ..native.sim import simulate,policies
from ..native.intent_policy import intent_policy
from ..native.world import CLIENT_TX_DBM
from .data import ROOT,PROTOCOL,READINGS,KEYS,make_stream,examples,bayesian,runtime_world
from .train import OUT,RUN,write,load_models,predict,source_hashes
from .theory import lower_bound
from .runtime import DeviceInputs,forecast


def corrupt(std,kind,seed):
    """Mean forecast is deliberately not an argument: it cannot be modified."""
    if kind=='half':return std*.5
    if kind=='double':return std*2
    rng=np.random.default_rng(seed+800000)
    if kind=='noise':return std*np.exp(rng.normal(0,1,size=std.shape[:1]+(1,std.shape[-1])))
    if kind=='shuffle':
        permutation=rng.permutation(std.shape[-1]);return std[...,permutation]
    return std.copy()


def metrics(mean,std,data,weights,z):
    error=mean-data['truth'];near=error[:,:6];sd=std[:,:6]
    score=mean[:,:6].reshape(len(mean),6,-1,3).max(-1).mean(1)
    truth=data['truth'][:,:6].reshape(len(mean),6,-1,3).max(-1).mean(1)
    uncertainty=std[:,0].reshape(len(mean),-1,3).max(-1)
    score=score-(.15+.2*weights[3])*data['load'];truth=truth-(.15+.2*weights[3])*data['load']
    ar=np.arange(len(score));current=data['current'];best=score.argmax(-1)
    gain=score[ar,best]-score[ar,current];true_gain=truth[ar,best]-truth[ar,current]
    u=np.sqrt(uncertainty[ar,best]**2+uncertainty[ar,current]**2+1e-9)
    bound=lower_bound(gain,true_gain,u,z=z,c=z,B=max(100,float(np.abs(true_gain).max()+1)),eligible=(gain>=5+1.5*weights[0])&(best!=current))
    def rho(x,y):
        r=float(spearmanr(x.ravel(),y.ravel()).statistic);return r if np.isfinite(r) else None
    # Native candidate score/threshold, not an end-to-end packet oracle.
    pu=score-(5+1.5*weights[0])*(np.arange(score.shape[1])[None]!=current[:,None])
    tu=truth-(5+1.5*weights[0])*(np.arange(score.shape[1])[None]!=current[:,None])
    heard=data['heard'][:,:6]
    return {'rmse_near_db':float(np.sqrt(np.mean(near**2))),'rmse_all_horizons_db':float(np.sqrt(np.mean(error**2))),
            'reported_link_rmse_near_db':float(np.sqrt(np.mean(near[heard]**2))),'unreported_link_rmse_near_db':float(np.sqrt(np.mean(near[~heard]**2))),
            'coverage_90':float(np.mean(np.abs(near)<=1.645*sd)),'link_error_sigma_spearman':rho(np.abs(error[:,0]),std[:,0]),
            'ap_score_error_sigma_spearman':rho(np.abs(score-truth),uncertainty),'finite_ap_choice_regret':float(np.mean(tu.max(-1)-tu[ar,pu.argmax(-1)])),
            'theory_audit':bound,'mean_forecast_sha256':hashlib.sha256(mean.tobytes()).hexdigest()}


def case(task):
    condition,seed=task;torch.set_num_threads(1);path=OUT/'cases'/f'{condition}-{seed}.json'
    if path.exists():return json.loads(path.read_text())
    began=time.time();primary=condition=='main';n=10 if condition=='small' else 100
    duration=PROTOCOL['primary_duration_s'] if primary else PROTOCOL['generalisation_duration_s']
    full,scan=make_stream(seed,n=n,decisions=duration,condition=condition);data=examples(full,scan,duration,targets=False)
    rmap=bayesian(full,scan,condition);plan=full.plan[60:60+duration]
    bm,bs=rmap.predict(plan[...,0],plan[...,1]);bm=(bm-full.tx_dl+CLIENT_TX_DBM-full.noise_db).reshape(-1,9,len(full.sites)*3);bs=bs.reshape(bm.shape)
    calibration=json.loads((OUT/'calibration.json').read_text());modelsets={f:load_models(f) for f in PROTOCOL['families']}
    predictions={};rows=[];summaries=[];device_inputs=DeviceInputs(data['ctx'],data['route'],data['prior'])
    def prediction(family,mission):
        key=(family,mission)
        if key not in predictions:
            weights=[READINGS[mission]['weights'][k] for k in KEYS]
            if family=='bayesian':predictions[key]=(bm,bs)
            else:
                m,s=forecast(modelsets[family],device_inputs,weights);predictions[key]=(m,s*calibration[family]['scale'])
        return predictions[key]
    base_mission='bulk_uploads';base=intent_policy(READINGS[base_mission]['weights'],READINGS[base_mission]['focus'],dual_hardware='none');base=replace(base,online_map=False)
    fixed=simulate(runtime_world(full,duration,bm.reshape(duration,n,9,-1,3),bs[:,0].reshape(duration,n,-1,3)),{p.name:p for p in policies()}['mlo_split'],collision_lambda=.3 if condition=='stress' else 0)
    rows.append(fixed|{'arm':'fixed','family':'fixed','gate':False,'mission':base_mission,'condition':condition})
    arms=[('bayesian',False,'base'),('bayesian',True,'base'),('jepa',False,'base'),('jepa',True,'base'),('regret',True,'base'),('no_route',True,'base'),('scratch',False,'base'),('scratch',True,'base')]
    if primary:arms += [('jepa',True,k) for k in ['half','double','noise','shuffle']]
    def rollout(family,gate,intervention,mission,member=None):
        weights=[READINGS[mission]['weights'][k] for k in KEYS]
        if member is None:mean,std=prediction(family,mission)
        else:mean,std=forecast([modelsets[family][member]],device_inputs,weights);std*=calibration[family]['scale']
        changed=corrupt(std,intervention,seed)
        pol=replace(intent_policy(READINGS[mission]['weights'],READINGS[mission]['focus'],dual_hardware='none'),online_map=False,gate=gate)
        arm=f'{family}_{"gated" if gate else "ungated"}_{intervention}'+(f'_seed{member}' if member is not None else '')
        world=runtime_world(full,duration,mean.reshape(duration,n,9,-1,3),changed[:,0].reshape(duration,n,-1,3))
        result=simulate(world,replace(pol,name=arm),collision_lambda=.3 if condition=='stress' else 0)
        rows.append(result|{'arm':arm,'family':family,'gate':gate,'mission':mission,'condition':condition,'member':member,'intervention':intervention})
        if gate or intervention!='base':summaries.append({'arm':arm,'family':family,'mission':mission,**metrics(mean,changed,data,weights,pol.gate_z)})
        print(json.dumps({'case':seed,'condition':condition,'arm':arm,'mission':mission,'control_miss':result['control_miss']}),flush=True)
    for family,gate,intervention in arms:rollout(family,gate,intervention,base_mission)
    if primary:
        for member in PROTOCOL['training_seeds']:
            for gate in [False,True]:rollout('jepa',gate,'base',base_mission,member)
        for mission in PROTOCOL['heldout_missions']:
            for family in ['bayesian','jepa','regret']:rollout(family,True,'base',mission)
    report={'completed':True,'scene':seed,'condition':condition,'robots':n,'duration_s':duration,'rows':rows,'forecast_metrics':summaries,'elapsed_s':time.time()-began,'device_history_only':True,'forecast_generation_did_not_receive_truth':True,'known_limit':'Native reactive fallback and attached-link measurement models retain their idealised access; only the world-model predictor is censored-device-side.'}
    write(path,report);return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    if not (OUT/'training.json').exists():raise RuntimeError('Training must finish before test access')
    began=time.time();hashes=source_hashes();hashes['src/wirelessjev/contracted/evaluate.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in hashes:
        dest=OUT/'evaluation_source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    write(OUT/'evaluation_source.json',{'source_sha256':hashes,'torch':torch.__version__,'test_opened_at_unix':began,'checkpoints_sha256':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/'checkpoints').glob('*.pt'))}})
    tasks=[(condition,seed) for condition,key in [('main','primary_sites'),('small','small_sites'),('stress','stress_sites'),('yard','yard_sites')] for seed in PROTOCOL[key]]
    with ProcessPoolExecutor(args.workers) as pool:
        completed=[]
        for result in pool.map(case,tasks):
            completed.append(result);write(RUN/'evaluation_status.json',{'completed_cases':len(completed),'total_cases':len(tasks),'last_scene':result['scene']})
    write(OUT/'evaluation.json',{'completed':True,'cases':[{'scene':c['scene'],'condition':c['condition'],'rows':len(c['rows'])} for c in completed],'elapsed_s':time.time()-began,'total_rollouts':sum(len(c['rows']) for c in completed)})
    print('ALL_TWENTY_FOUR_CASES_COMPLETE',flush=True)


if __name__=='__main__':main()
