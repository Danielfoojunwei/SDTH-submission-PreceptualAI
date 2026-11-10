"""Fit, seal a development choice, then open entirely fresh gate-repair tests."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import argparse,hashlib,json,shutil,time
import numpy as np
import torch
from ..contracted.data import ROOT,READINGS,KEYS,make_stream,examples,bayesian,runtime_world
from ..contracted.runtime import DeviceInputs
from ..contracted.train import load_models,write
from ..native.intent_policy import intent_policy
from ..native.sim import simulate,policies
from ..native.world import CLIENT_TX_DBM
from .bank import forecasts,FEATURES
from .gate import GainCalibrator,GainGate,Recorder,diagnostics,fit_model,reactive_rescue

OUT=ROOT/'artifacts/gate_repair';RUN=ROOT.parent/'training/gate-repair'
P=json.loads((OUT/'protocol.json').read_text());MISSION=P['mission']
WEIGHTS=[READINGS[MISSION]['weights'][k] for k in KEYS]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths=list((ROOT/'src/wirelessjev/gate_repair').glob('*.py'))+list((ROOT/'src/wirelessjev/native').glob('*.py'))+[ROOT/'src/wirelessjev/contracted/data.py',ROOT/'src/wirelessjev/contracted/model.py',ROOT/'src/wirelessjev/contracted/runtime.py',OUT/'protocol.json',ROOT/'artifacts/native/intent_flip.json',ROOT/'artifacts/warehouse_v2/studies/wifi_link_curves.json',ROOT/'artifacts/contracted/calibration.json']
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}


def checkpoints():return {str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'artifacts/contracted/checkpoints').glob('jepa-*.pt'))}


def snapshot(name):
    hashes=sources()
    for relative in hashes:
        dest=OUT/(name+'_source')/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,dest)
    write(OUT/(name+'_source.json'),{'source_sha256':hashes,'checkpoint_sha256':checkpoints(),'time_unix':time.time(),'torch':torch.__version__})


def build(seed,condition,duration):
    n=10 if condition=='small' else 100;full,scan=make_stream(seed,n=n,decisions=duration,condition=condition);data=examples(full,scan,duration,targets=False)
    inputs=DeviceInputs(data['ctx'],data['route'],data['prior']);models=load_models('jepa')
    scale=json.loads((ROOT/'artifacts/contracted/calibration.json').read_text())['jepa']['scale']
    bank=forecasts(models,inputs,WEIGHTS,duration,n,scale)
    return full,scan,data,bank


def case(task):
    stage,condition,seed=task;torch.set_num_threads(1);dest=OUT/'cases'/f'{stage}-{condition}-{seed}.json'
    if dest.exists():
        old=json.loads(dest.read_text())
        if old['source_sha256']!=sources():raise RuntimeError('Case source mismatch; do not overwrite a completed case')
        return old
    duration=P['primary_test_duration_s'] if stage=='test' and condition=='main' else P['development_duration_s'] if stage!='test' else P['generalisation_duration_s']
    began=time.time();full,scan,data,bank=build(seed,condition,duration);rows=[];trace_files=[]
    policy=replace(intent_policy(READINGS[MISSION]['weights'],READINGS[MISSION]['focus'],dual_hardware='none'),online_map=False)
    model=GainCalibrator.from_dict(json.loads((OUT/'gain_model.json').read_text())) if stage!='fit' else None
    if stage=='fit':arms=[('ungated',None),('legacy',1.)]
    elif stage=='select':arms=[('ungated',None),('rescue',None)]+[('scale',x) for x in P['scale_candidates']]+[('gain',q) for q in P['gain_quantiles']]
    else:
        selected=json.loads((OUT/'selection.json').read_text())
        arms=[('fixed',None),('bayesian',None),('ungated',None),('legacy',1.),('rescue',None),('scale',selected['scale']['parameter']),('gain',selected['gain']['parameter'])]
    for kind,parameter in arms:
        name=kind if kind not in {'gain','scale'} else f'{kind}_{parameter}';recorder=Recorder(bank);hook=None;mean=bank.mean;std=bank.std[:, :, 0]
        pol=replace(policy,name=name,gate=kind!='ungated',belief_scale=float(parameter) if kind in {'legacy','scale'} else 1.)
        if kind=='rescue':hook=reactive_rescue
        if kind=='gain':hook=GainGate(bank,model,parameter)
        if kind=='fixed':pol={p.name:p for p in policies()}['mlo_split']
        if kind=='bayesian':
            rmap=bayesian(full,scan,condition);plan=full.plan[60:60+duration];m,s=rmap.predict(plan[...,0],plan[...,1]);mean=(m-full.tx_dl+CLIENT_TX_DBM-full.noise_db).astype('f4');std=s[:,:,0].astype('f4')
        forecast_hash=hashlib.sha256(mean.tobytes()).hexdigest()
        result=simulate(runtime_world(full,duration,mean,std),pol,roam_gate=hook,decision_trace=recorder if kind not in {'fixed','bayesian'} else None,collision_lambda=.3 if condition=='stress' else 0)
        if hashlib.sha256(mean.tobytes()).hexdigest()!=forecast_hash:raise RuntimeError('Simulation mutated the forecast')
        diag=None;trace=None
        if kind not in {'fixed','bayesian'}:
            labeled=recorder.label(data['truth']);diag=diagnostics(labeled)
            eligible=labeled['eligible'];trace=f'traces/{stage}-{condition}-{seed}-{name}.npz';path=OUT/trace;path.parent.mkdir(exist_ok=True,parents=True)
            np.savez_compressed(path,**{k:v[eligible] for k,v in labeled.items()})
        rows.append({'kind':kind,'parameter':parameter,'arm':name,'forecast_sha256':forecast_hash,'gate_diagnostics':diag,'trace':trace,'result':result})
        print(json.dumps({'stage':stage,'condition':condition,'scene':seed,'arm':name,'misses':result['control_misses'],'sent':result['control_sent']}),flush=True)
    report={'completed':True,'stage':stage,'condition':condition,'scene':seed,'robots':full.n,'duration_s':duration,'rows':rows,'elapsed_s':time.time()-began,'source_sha256':sources(),'checkpoint_sha256':checkpoints(),'truth_used_by_runtime_gate':False}
    write(dest,report);return report


def execute(tasks,workers):
    with ProcessPoolExecutor(workers) as pool:
        result=[]
        for r in pool.map(case,tasks):
            result.append(r);write(RUN/'status.json',{'phase':r['stage'],'completed':len(result),'total':len(tasks),'last_scene':r['scene']})
    return result


def trace_rows(cases):
    arrays=[]
    for c in cases:
        for row in c['rows']:
            if row['trace']:
                with np.load(OUT/row['trace']) as z:arrays.append({k:z[k] for k in z.files})
    return {k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}


def fit(workers):
    if (OUT/'gain_model.json').exists():return
    snapshot('fit');parts=execute([('fit','main',s) for s in P['fit_sites']+P['calibration_sites']],workers)
    train=trace_rows([c for c in parts if c['scene'] in P['fit_sites']]);cal=trace_rows([c for c in parts if c['scene'] in P['calibration_sites']])
    model=fit_model(train['features'],train['error'],cal['features'],cal['error'],P['gain_quantiles'])
    write(OUT/'gain_model.json',model.to_dict()|{'fit_sites':P['fit_sites'],'calibration_sites':P['calibration_sites'],'fit_decisions':len(train['error']),'calibration_decisions':len(cal['error']),'features':list(FEATURES),'source_sha256':sources()})
    write(OUT/'diagnosis.json',{'fit_and_calibration':[{k:c[k] for k in ['scene','stage','condition']}|{'arms':{r['arm']:r['gate_diagnostics'] for r in c['rows']}} for c in parts],'training_rows':len(train['error']),'calibration_rows':len(cal['error'])})


def select(workers):
    if (OUT/'selection.json').exists():return
    if not (OUT/'gain_model.json').exists():raise RuntimeError('Fit the gain model first')
    snapshot('selection')
    parts=execute([('select',c,s) for c in ['main','stress','yard','small'] for s in P['selection_'+c+'_sites']],workers)
    candidates={}
    for kind,parameters in [('scale',P['scale_candidates']),('gain',P['gain_quantiles'])]:
        evaluated=[]
        for p in parameters:
            ratios={};rejections=opportunities=0
            for condition in ['main','stress','yard','small']:
                cs=[c for c in parts if c['condition']==condition]
                base=sum(next(r['result']['control_misses'] for r in c['rows'] if r['kind']=='ungated') for c in cs)
                rs=[next(r for r in c['rows'] if r['kind']==kind and r['parameter']==p) for c in cs];misses=sum(r['result']['control_misses'] for r in rs)
                ratios[condition]=(misses+1)/(base+1)
                rejections+=sum(r['gate_diagnostics']['rejected'] for r in rs);opportunities+=sum(r['gate_diagnostics']['opportunities'] for r in rs)
            rate=rejections/max(opportunities,1);evaluated.append({'parameter':p,'score':float(np.mean(list(ratios.values()))),'condition_ratios_to_ungated':ratios,'rejection_fraction':rate,'nontrivial':rate>=P['selection_rejection_rate_minimum']})
        qualified=[r for r in evaluated if r['nontrivial']]
        candidates[kind]=min(qualified or evaluated,key=lambda r:(r['score'],r['parameter']))|{'candidates':evaluated,'no_nontrivial_candidate':not bool(qualified)}
    write(OUT/'selection.json',{'sealed':True,'sealed_at_unix':time.time(),**candidates,'source_sha256':sources(),'gain_model_sha256':digest(OUT/'gain_model.json'),'checkpoint_sha256':checkpoints(),'test_accessed':False})
    print('DEVELOPMENT_SELECTION_SEALED',flush=True)


def test(workers):
    selected=json.loads((OUT/'selection.json').read_text())
    if not selected['sealed'] or selected['source_sha256']!=sources() or selected['gain_model_sha256']!=digest(OUT/'gain_model.json'):raise RuntimeError('Selection/source seal mismatch; do not open tests')
    snapshot('test')
    parts=execute([('test',c,s) for c in ['main','stress','yard','small'] for s in P['test_'+c+'_sites']],workers)
    write(OUT/'completion.json',{'completed':True,'test_scenes':len(parts),'test_native_runs':sum(len(c['rows']) for c in parts),'all_native_runs':sum(len(json.loads(p.read_text())['rows']) for p in (OUT/'cases').glob('*.json')),'completed_at_unix':time.time()})
    write(RUN/'status.json',{'phase':'complete','test_scenes':len(parts)});print('GATE_REPAIR_COMPLETE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['fit','select','test','all'],default='all');ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    OUT.mkdir(exist_ok=True,parents=True);RUN.mkdir(exist_ok=True,parents=True)
    if args.phase in ['fit','all']:fit(args.workers)
    if args.phase in ['select','all']:select(args.workers)
    if args.phase in ['test','all']:test(args.workers)


if __name__=='__main__':main()
