"""Reproducible five-seed fits; test scenes are never constructed here."""
import argparse,hashlib,json,time,os,shutil
from pathlib import Path
import numpy as np
import torch
from .data import ROOT,PROTOCOL,READINGS,KEYS,make_stream,examples
from .model import DeviceJEPA,loss,update_target

OUT=ROOT/'artifacts/contracted';RUN=ROOT.parent/'training/contracted-world-models'


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def source_hashes():
    paths=list((ROOT/'src/wirelessjev/contracted').glob('*.py'))+[ROOT/'artifacts/contracted/protocol.json',ROOT/'src/wirelessjev/native/sim.py',ROOT/'src/wirelessjev/native/world.py',ROOT/'src/wirelessjev/native/radiomap.py',ROOT/'src/wirelessjev/native/foundation.py',ROOT/'artifacts/native/intent_flip.json',ROOT/'artifacts/warehouse_v2/studies/wifi_link_curves.json']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def data(split):
    cache=RUN/(split+'.npz');receipt=RUN/(split+'.data.json')
    identity={k:v for k,v in source_hashes().items() if k.endswith(('data.py','world.py','foundation.py','radiomap.py','protocol.json'))}
    if cache.exists() and receipt.exists():
        old=json.loads(receipt.read_text())
        if old['source_sha256']!=identity:raise RuntimeError('Dataset source changed; choose a new run directory')
        with np.load(cache) as z:return {k:z[k] for k in z.files}
    pieces=[]
    for seed in PROTOCOL[split+'_sites']:
        w,scan=make_stream(seed,n=PROTOCOL['training_robots'],decisions=PROTOCOL['training_decisions']);pieces.append(examples(w,scan,PROTOCOL['training_decisions']))
        print(json.dumps({'data':split,'site':seed}),flush=True)
    arrays={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    for k in ['ctx','future_ctx','route']:arrays[k]=arrays[k].astype('f2')
    RUN.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,**arrays)
    write(receipt,{'source_sha256':identity,'sites':PROTOCOL[split+'_sites'],'samples':len(arrays['ctx']),'npz_sha256':hashlib.sha256(cache.read_bytes()).hexdigest()})
    return arrays


def batch(arrays,indices):
    return {k:torch.as_tensor(v[indices].astype('f4') if v.dtype==np.float16 else v[indices]) for k,v in arrays.items() if k!='site'}


@torch.no_grad()
def predict(models,arrays,weights,batch_size=128):
    means=[];stds=[]
    for start in range(0,len(arrays['ctx']),batch_size):
        sl=slice(start,start+batch_size);ctx=torch.as_tensor(arrays['ctx'][sl].astype('f4'));route=torch.as_tensor(arrays['route'][sl].astype('f4'));prior=torch.as_tensor(arrays['prior'][sl]);intent=torch.tensor(weights,dtype=torch.float32)[None].expand(len(ctx),-1)
        ps=[m(ctx,route,prior,intent) for m in models];mu=torch.stack([p['mean'] for p in ps]);variance=torch.stack([p['std'].square() for p in ps]).mean(0)
        if len(models)>1:variance=variance+mu.var(0,unbiased=True)
        means.append(mu.mean(0).numpy());stds.append(variance.sqrt().numpy())
    return np.concatenate(means),np.concatenate(stds)


def load_models(family):
    models=[]
    for seed in PROTOCOL['training_seeds']:
        saved=torch.load(OUT/'checkpoints'/f'{family}-{seed}.pt',map_location='cpu',weights_only=True)
        m=DeviceJEPA(**saved['config']);m.load_state_dict(saved['state']);m.eval();models.append(m)
    return models


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch-size',type=int,default=64);args=parser.parse_args();torch.set_num_threads(2)
    started=time.time();OUT.mkdir(parents=True,exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes();capsule=OUT/'training_source'
    if not capsule.exists():
        for name in hashes:
            dest=capsule/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        write(OUT/'training_source.json',{'source_sha256':hashes,'torch':torch.__version__,'device':'cpu'})
    train=data('training');dev=data('development');missions=np.array([[READINGS[k]['weights'][key] for key in KEYS] for k in PROTOCOL['train_missions']],dtype='f4')
    reports=[]
    for family in PROTOCOL['families']:
        for seed in PROTOCOL['training_seeds']:
            dest=OUT/'checkpoints'/f'{family}-{seed}.pt';report=OUT/'fits'/f'{family}-{seed}.json'
            if report.exists() and json.loads(report.read_text()).get('completed'):
                reports.append(json.loads(report.read_text()));continue
            torch.manual_seed(seed);rng=np.random.default_rng(seed);model=DeviceJEPA(no_route=family=='no_route');optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=1e-4)
            best=float('inf');history=[];fit_start=time.time();last=RUN/f'{family}-{seed}.last.pt';first_epoch=0
            if last.exists():
                old=torch.load(last,weights_only=False,map_location='cpu');model.load_state_dict(old['state']);optim.load_state_dict(old['optimizer']);history=old['history'];best=old['best'];first_epoch=old['epoch']+1;rng.bit_generator.state=old['numpy_rng'];torch.set_rng_state(old['torch_rng'])
            for epoch in range(first_epoch,PROTOCOL['epochs']):
                model.train();order=rng.permutation(len(train['ctx']));costs=[]
                for start in range(0,len(order),args.batch_size):
                    idx=order[start:start+args.batch_size];b=batch(train,idx);intent=torch.tensor(missions[rng.integers(len(missions),size=len(idx))]);objective,nll=loss(model,b,intent,family)
                    if not torch.isfinite(objective):raise FloatingPointError(f'Non-finite objective: {family}/{seed}/{epoch}')
                    optim.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optim.step();update_target(model);costs.append(float(nll))
                model.eval();cost=[]
                with torch.no_grad():
                    for start in range(0,len(dev['ctx']),args.batch_size):
                        idx=np.arange(start,min(start+args.batch_size,len(dev['ctx'])));b=batch(dev,idx);intent=torch.tensor(missions[idx%len(missions)]);_,nll=loss(model,b,intent,'scratch');cost.append(float(nll))
                validation=float(np.mean(cost));row={'family':family,'seed':seed,'epoch':epoch+1,'train_censored_nll':float(np.mean(costs)),'development_censored_nll':validation};history.append(row);print(json.dumps(row),flush=True)
                if validation<best:
                    best=validation;torch.save({'config':model.config,'state':model.state_dict(),'epoch':epoch+1,'family':family,'seed':seed},dest)
                torch.save({'state':model.state_dict(),'optimizer':optim.state_dict(),'history':history,'best':best,'epoch':epoch,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state()},last)
                write(RUN/'status.json',row|{'phase':'training','started':started})
            result={'family':family,'seed':seed,'completed':True,'epochs':len(history),'best_development_censored_nll':best,'history':history,'elapsed_s':time.time()-fit_start,'checkpoint_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'source_receipt':'training_source.json','parameters':sum(p.numel() for p in model.parameters()),'test_accessed':False}
            write(report,result);reports.append(result)
    calibration={}
    for family in PROTOCOL['families']:
        models=load_models(family);mean,std=predict(models,dev,missions[0]);ratios=np.abs(mean[:,:6]-dev['truth'][:,:6])/np.maximum(std[:,:6],.1)
        scale=max(.1,float(np.quantile(ratios,.90)/1.645));calibration[family]={'scale':scale,'nominal_coverage':.9,'development_sites':PROTOCOL['development_sites'],'uses_test':False}
    write(OUT/'calibration.json',calibration);write(OUT/'training.json',{'completed':True,'fits':reports,'elapsed_s':time.time()-started,'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()});write(RUN/'status.json',{'phase':'training_complete','fits':len(reports)})
    print('ALL_TWENTY_FITS_COMPLETE',flush=True)


if __name__=='__main__':main()
