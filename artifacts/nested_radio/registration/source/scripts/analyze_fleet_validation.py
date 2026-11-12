"""Reproduce ensemble diagnostics on validation data, never fresh tests."""
import numpy as np
import torch
from wirelessjev.fleetwm.study import load,batch,P,OUT,RUN
from wirelessjev.contracted.train import write


def main():
    torch.set_num_threads(1)
    with np.load(RUN/'validation.npz') as z:arrays={k:z[k] for k in z.files}
    results={}
    for family in P['families']:
        models=load(family);means=[]
        with torch.no_grad():
            for start in range(0,len(arrays['context']),128):
                b=batch(arrays,slice(start,start+128))
                means.append(torch.stack([m(b)['mean'] for m in models]).mean(0).numpy())
        m=np.concatenate(means);truth=arrays['truth'].astype('f4');valid=arrays['valid_link'].astype('f4')[:,None]
        w=np.where(truth>0,1.,.2)*valid;e=m-truth
        s=m[:,:6].reshape(-1,6,16,3).max(-1).mean(1);t=truth[:,:6].reshape(-1,6,16,3).max(-1).mean(1)
        s=np.where(valid[:,0,::3].astype(bool),s,-1e4)
        regret=t.max(-1)-t[np.arange(len(t)),s.argmax(-1)]
        results[family]={'weighted_rmse':float(np.sqrt((e*e*w).sum()/w.sum())),
            'ap_choice_regret_db':float(regret.mean()),'all_link_rmse':float(np.sqrt((e*e*valid).sum()/(valid.sum()*9)))}
    write(OUT/'validation_ensemble.json',{'validation_only':True,'sites':P['validation_sites'],'families':results,'test_accessed':False})
    print(results)


if __name__=='__main__':main()
