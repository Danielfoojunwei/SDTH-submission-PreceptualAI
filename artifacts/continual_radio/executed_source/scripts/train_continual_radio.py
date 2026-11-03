"""Registered six-fit study on actual RT labels; constructed probes are not hardware."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from wirelessjev.continual_radio import ResponseJEPA,response_objective,update_response_teacher,local_ridge,masked_least_squares

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/continual_radio'
P=json.loads((OUT/'registration/protocol.json').read_text())
PATTERNS=np.array([P['probe_patterns']['synchronized'],P['probe_patterns']['staggered']],dtype=float)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def source_hashes():
    names=['src/wirelessjev/continual_radio.py','scripts/train_continual_radio.py','scripts/collect_continual_radio.py',
           'tests/test_continual_radio.py','artifacts/continual_radio/registration/protocol.json','pyproject.toml','uv.lock']
    return {name:sha(ROOT/name) for name in names}


def register_sources():
    dest=OUT/'executed_source/manifest.json';hashes=source_hashes()
    if dest.exists():assert json.loads(dest.read_text())['source_sha256']==hashes;return
    for name in hashes:
        target=OUT/'executed_source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    write(dest,{'registered_before_data_collection':True,'registered_at_unix':time.time(),'source_sha256':hashes})


def data():
    registration=json.loads((OUT/'executed_source/manifest.json').read_text());assert registration['source_sha256']==source_hashes()
    receipt=json.loads((OUT/'data_receipt.json').read_text());path=OUT/'data/kernels.npz'
    assert receipt['actual_sionna_rt'] and receipt['simulation_only'] and receipt['data_sha256']==sha(path)
    with np.load(path) as z:a={k:z[k] for k in z.files}
    g=a['coupling'];assert g.shape==(24,2,2) and a['coupling_valid'].all()
    assert np.isfinite(g).all() and (g>0).all() and a['context'].shape==(24,16)
    assert np.array_equal(a['context_id'],np.arange(24))
    train=g[P['training_ids']];scale=train.sum(-1).mean(0);gn=train/scale[None,:,None]
    x=.3*np.log(10);eg=np.sinh(x)/x;eg2=np.sinh(2*x)/(2*x)
    prior=eg*gn.mean(0);cov=eg2*np.einsum('nri,nrj->rij',gn,gn)/len(gn)-np.einsum('ri,rj->rij',prior,prior)
    cov+=np.eye(2)[None]*1e-6
    return a,scale,prior,cov


def episodes(a,scale,split):
    rng=np.random.default_rng(P['episode_seeds'][split]);rows=[]
    for cid in P[split+'_ids']:
        for episode in range(P['episodes_per_context']):
            gain=10**(rng.uniform(-3,3,2)/10);h=a['coupling'][cid]*gain[:,None]/scale[:,None]
            delta=rng.uniform(-.5,.5,2);before=np.full(2,.5);n0,n1=rng.normal(size=(2,2))
            clean=h@delta;future=clean+.01*(n1-n0)
            assert ((h@before+.1+.01*n0)>0).all() and ((h@(before+delta)+.1+.01*n1)>0).all()
            measurement_noise=rng.normal(size=(2,2))*.01;heard=rng.random((2,2))>=.1
            for pattern,activity in enumerate(PATTERNS):
                total=activity@h.T+.1+measurement_noise
                if not (total>0).all():raise RuntimeError('Nonphysical constructed measurement; do not clip')
                power=total-.1
                for budget in P['history_budgets']:
                    available=heard & (np.arange(2)[:,None]<budget)
                    rows.append({'context':a['context'][cid],'activity':activity,'power':np.where(available,power,0),
                        'mask':available,'delta':delta,'budget':budget,'future_effect':future,'clean_effect':clean,
                        'effective_coupling':h,'context_id':cid,'episode':episode,'pattern':pattern})
    return {k:np.asarray([r[k] for r in rows]) for k in rows[0]}


def batch(a,index):
    return {k:torch.as_tensor(a[k][index],dtype=torch.bool if k=='mask' else torch.long if k=='budget' else torch.float32)
            for k in ('context','activity','power','mask','delta','budget','future_effect')}


@torch.no_grad()
def predict(model,a):
    parts=[]
    for start in range(0,len(a['context']),256):
        b=batch(a,slice(start,start+256));parts.append(model(b['context'],b['activity'],b['power'],b['mask'],b['delta'],b['budget'])['response'].numpy())
    return np.concatenate(parts)


def fit(task):
    family,seed=task;torch.set_num_threads(1);path=OUT/'checkpoints'/f'{family}-{seed}.pt';receipt=OUT/'fits'/f'{family}-{seed}.json'
    if receipt.exists():
        r=json.loads(receipt.read_text());assert r['source_sha256']==source_hashes() and r['checkpoint_sha256']==sha(path);return r
    a,scale,prior,_=data();train=episodes(a,scale,'training');val=episodes(a,scale,'validation')
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=ResponseJEPA(prior,family=family)
    optimiser=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);history=[];started=time.time()
    for epoch in range(30):
        model.train();order=rng.permutation(len(train['context']));losses=[]
        for start in range(0,len(order),64):
            b=batch(train,order[start:start+64]);loss,_=response_objective(model,b)
            assert torch.isfinite(loss);optimiser.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimiser.step()
            if family=='jepa':update_response_teacher(model)
            losses.append(float(loss.detach()))
        model.eval();validation=float(np.abs(predict(model,val)-val['future_effect']).mean())
        row={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_noisy_mae_normalized':validation};history.append(row)
        print(json.dumps({'family':family,'seed':seed,**row}),flush=True)
    path.parent.mkdir(parents=True,exist_ok=True);torch.save({'config':model.config,'prior':model.prior,'state':model.state_dict()},path)
    r={'completed':True,'family':family,'seed':seed,'history':history,'fixed_final_epoch':30,'inference_parameters':model.inference_parameters,
       'source_sha256':source_hashes(),'data_sha256':sha(OUT/'data/kernels.npz'),'checkpoint_sha256':sha(path),
       'elapsed_seconds':time.time()-started,'test_accessed':False}
    write(receipt,r);return r


def numerical(a,prior,cov,scale):
    output={k:[] for k in ('train_mean','free_space','least_squares','bayesian','oracle')};ranks=[]
    wavelength=299792458/5.8e9
    for i in range(len(a['context'])):
        k=int(a['budget'][i]);activity=a['activity'][i,:k];power=a['power'][i,:k];mask=a['mask'][i,:k];delta=a['delta'][i]
        tx=a['context'][i,3:9].reshape(2,3)*10;rx=a['context'][i,9:15].reshape(2,3)*10
        free=(wavelength/(4*np.pi*np.linalg.norm(rx[:,None]-tx[None],axis=-1)))**2/scale[:,None]
        # Scale is recorded separately and derived only from training contexts.
        ls,rank=masked_least_squares(activity,power,mask);bayes,_=local_ridge(prior,activity,power,mask,prior_covariance=cov)
        output['train_mean'].append(prior@delta);output['free_space'].append(free@delta)
        output['least_squares'].append(ls@delta);output['bayesian'].append(bayes@delta);output['oracle'].append(a['effective_coupling'][i]@delta);ranks.append(rank)
    return {k:np.asarray(v) for k,v in output.items()},np.asarray(ranks)


def crossed(delta):
    delta=np.asarray(delta);rng=np.random.default_rng(404);n,s=delta.shape
    ci=rng.integers(n,size=(20000,n));si=rng.integers(s,size=(20000,s))
    boot=delta[ci[:,:,None],si[:,None,:]].mean((1,2))
    return {'mean_difference_normalized_mae':float(delta.mean()),'crossed_context_seed_95':np.quantile(boot,[.025,.975]).tolist(),
            'contexts':n,'paired_training_seeds':s,'context_seed_differences':delta.tolist()}


def evaluate():
    a,scale,prior,cov=data();test=episodes(a,scale,'test');pred,ranks=numerical(test,prior,cov,scale)
    learned={};collapse={}
    for family in ('jepa','supervised'):
        for seed in (0,1,2):
            name=f'{family}-{seed}';state=torch.load(OUT/'checkpoints'/f'{name}.pt',weights_only=True,map_location='cpu')
            model=ResponseJEPA(state['prior'],**state['config']);model.load_state_dict(state['state']);model.eval()
            learned[name]=predict(model,test)
            with torch.no_grad():
                h=test['effective_coupling'][(test['pattern']==0)&(test['budget']==0)]
                delta=np.tile([.5,-.5],(len(h),1));effect=np.einsum('nri,ni->nr',h,delta)
                inputs=torch.tensor(np.concatenate([delta,effect,np.ones_like(effect)],-1),dtype=torch.float32)
                z=model.teacher(inputs).numpy();poison=inputs.clone();poison[:,2:4]=0
                sensitivity=float(np.linalg.norm(z-model.teacher(poison).numpy(),axis=1).mean())
                singular=np.linalg.svd(z-z.mean(0),compute_uv=False);energy=singular**2;prob=energy/max(energy.sum(),1e-30)
                collapse[name]={'target_rank':int((singular>max(singular.max()*1e-4,1e-8)).sum()),
                    'target_effective_rank':float(np.exp(-np.sum(prob*np.log(np.maximum(prob,1e-30))))),
                    'mean_target_coordinate_std':float(z.std(0).mean()),'response_sensitivity_l2':sensitivity,
                    'exact_collapse':bool(np.max(z.std(0))<1e-6),'noncollapse_is_not_usefulness':True}
    all_predictions=pred|learned;groups={};context_errors={}
    for pattern in (0,1):
        for budget in (0,1,2):
            key=f'{("synchronized","staggered")[pattern]}-K{budget}';mask=(test['pattern']==pattern)&(test['budget']==budget)
            groups[key]={'probe_slots':budget,'transmitter_on_airtime_units':float(PATTERNS[pattern,:budget].sum()),
                'observed_receiver_rank_histogram':np.bincount(ranks[mask].ravel(),minlength=3).tolist(),'methods':{}}
            context_errors[key]={}
            for name,y in all_predictions.items():
                clean_error=np.abs(y-test['clean_effect']);noisy_error=np.abs(y-test['future_effect'])
                per_context=[float(clean_error[mask&(test['context_id']==cid)].mean()) for cid in P['test_ids']]
                context_errors[key][name]=np.asarray(per_context)
                groups[key]['methods'][name]={'clean_mae_mw':float((clean_error[mask]*scale).mean()),
                    'noisy_measured_mae_mw':float((noisy_error[mask]*scale).mean()),'clean_mae_normalized':float(clean_error[mask].mean()),
                    'per_receiver_clean_mae_mw':(clean_error[mask]*scale).mean(0).tolist(),'test_context_mae_normalized':per_context}
    comparisons={}
    for k in (1,2):
        key=f'staggered-K{k}';j=np.stack([context_errors[key][f'jepa-{s}'] for s in (0,1,2)],1)
        sup=np.stack([context_errors[key][f'supervised-{s}'] for s in (0,1,2)],1)
        comparisons[key+'/supervised']=crossed(j-sup)
        for reference in ('train_mean','free_space','least_squares','bayesian'):
            comparisons[key+'/'+reference]=crossed(j-context_errors[key][reference][:,None])
    # Exact linear identifiability uses the actual RT kernels, with no noise/missingness.
    kernels=a['coupling'];identity=np.eye(2);sync=PATTERNS[0];recovered=[];sync_error=[]
    for kernel in kernels:
        recovered.append(float(np.max(np.abs(np.linalg.lstsq(identity,identity@kernel.T,rcond=None)[0].T-kernel))))
        estimate=np.linalg.lstsq(sync,sync@kernel.T,rcond=None)[0].T;sync_error.append(float(np.abs(estimate-kernel).mean()))
    _,single_cov=local_ridge(prior,sync,sync@prior.T,np.ones((2,2),bool))
    repeated=np.tile(sync,(20,1));_,many_cov=local_ridge(prior,repeated,repeated@prior.T,np.ones_like(repeated,dtype=bool))
    null=np.array([1.,-1.])/np.sqrt(2)
    rank={'same_K2_transmitter_airtime':2.,'synchronized_rank':int(np.linalg.matrix_rank(sync)),
        'staggered_rank':2,'repeated_collinear_rank':int(np.linalg.matrix_rank(repeated)),
        'actual_rt_staggered_max_recovery_error':max(recovered),'actual_rt_synchronized_mean_kernel_error':float(np.mean(sync_error)),
        'null_direction_variance_after_two':float(null@single_cov[0]@null),'null_direction_variance_after_forty':float(null@many_cov[0]@null),
        'rank_benefit_is_not_JEPA_benefit':True,'K1_designs_do_not_have_equal_transmitter_airtime':True}
    summary={'completed':True,'trained_models':6,'contexts':24,'one_warehouse_only':True,'test_context_ids':P['test_ids'],
        'scale_mw_train_only':scale.tolist(),'prior_normalized_train_only':prior.tolist(),'groups':groups,'paired_comparisons':comparisons,
        'jepa_specific_gate_passed':all(v['crossed_context_seed_95'][1]<0 for v in comparisons.values()),
        'teacher_diagnostics':collapse,'identifiability':rank,'simulation_only':True,'hardware_qualified':False,
        'backbone_online_updates':False,'local_state_assimilation_only':True,'no_sota_or_novelty_established':True,
        'source_sha256':source_hashes(),'data_sha256':sha(OUT/'data/kernels.npz')}
    write(OUT/'summary.json',summary)
    write(OUT/'completion.json',{'completed':True,'fits':6,'epochs_per_fit':30,'fixed_final_epoch':True,
        'source_sha256':source_hashes(),'summary_sha256':sha(OUT/'summary.json'),'data_sha256':sha(OUT/'data/kernels.npz')})
    report(summary)
    return summary


def report(s):
    lines=['# Continual radio response: bounded RT-backed identification study','',
        'The experiment uses 24 actual Sionna RT kernels in one warehouse and explicitly constructed probe/noise/device episodes. '
        'Five held-out contexts are new blocker poses in the same scene, not new warehouses or hardware measurements. '
        'There are no test-time backbone weight updates; local learning means conditioning or Bayesian assimilation of received feedback.','',
        'Six matched fixed-final-epoch fits completed: three JEPA effect-teacher models and three supervised models, 30 epochs each. '
        'True coupling is never an unmeasured model input. Only training contexts determine units and numerical priors.','',
        '| Probe design / feedback budget | JEPA clean MAE mW | Supervised | Bayesian | Least squares |',
        '|---|---:|---:|---:|---:|']
    for key,group in s['groups'].items():
        m=group['methods'];j=np.mean([m[f'jepa-{seed}']['clean_mae_mw'] for seed in (0,1,2)]);b=np.mean([m[f'supervised-{seed}']['clean_mae_mw'] for seed in (0,1,2)])
        lines.append(f'| {key} | {j:.6g} | {b:.6g} | {m["bayesian"]["clean_mae_mw"]:.6g} | {m["least_squares"]["clean_mae_mw"]:.6g} |')
    lines+=['',f'The preregistered JEPA-specific conjunction **{"passes" if s["jepa_specific_gate_passed"] else "does not pass"}**. '
        'It requires beating the matched supervised model and every non-oracle numerical reference at staggered K1 and K2. '
        'Intervals resample five held-out contexts and three paired training seeds; repeated probe episodes are not independent warehouses.','',
        'The equal-airtime K2 rank comparison is separate: synchronized on/off probes are rank one, whereas staggered probes are rank two and '
        'recover the actual noiseless 2x2 kernel to numerical precision. More collinear data cannot excite the missing contrast direction. '
        'This is system identification and does not establish a JEPA advantage. K1 designs have different transmitter-on airtime.','',
        'Per-receiver unknown gains, calibrated floor, Gaussian noise and missing feedback are assumptions. Powers are incoherent averages; '
        'this is not coherent waveform addition, packet-loss validation, active hardware probing, certified risk, general continual learning or SOTA. '
        'Teacher rank/variance and response-sensitivity diagnostics do not establish representation usefulness by themselves.','',
        'All metrics, per-context comparisons, normalisation and source/data hashes are in `artifacts/continual_radio/summary.json`.']
    (ROOT/'docs/CONTINUAL_RADIO_RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    if args.register:register_sources();print('CONTINUAL_RESPONSE_SOURCES_REGISTERED');return
    data();tasks=[(f,s) for f in ('jepa','supervised') for s in (0,1,2)]
    with ProcessPoolExecutor(args.workers) as pool:list(pool.map(fit,tasks))
    torch.set_num_threads(1);s=evaluate();print(json.dumps({'completed':True,'fits':6,'jepa_specific_gate_passed':s['jepa_specific_gate_passed']}))


if __name__=='__main__':main()
