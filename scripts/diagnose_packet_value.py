"""Post-hoc validation diagnosis; never selects a new active controller."""
import json,hashlib
from pathlib import Path
import numpy as np
from wirelessjev.native.phy import LinkCurves

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/packet_value'


def diagnose():
    pieces=[]
    for p in sorted((OUT/'data').glob('validation-*.npz')):
        with np.load(p) as z:pieces.append({k:z[k] for k in z.files})
    a={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    curves=LinkCurves();th=np.array([curves.threshold_db(b,.99)[0] for b in ('2g','5g','6g')])
    candidate=np.where(a['query'][:,17:20]>0,a['query'][:,14:17]*30-th,-np.inf).max(-1)
    current=(a['context'][:,-1,:3]*30-th).max(-1)
    feasible=(candidate>=3)&(candidate>=current-3)
    cal=json.loads((OUT/'calibration.json').read_text());groups={}
    for cost in (.03,4.):
        mask=a['cost']==cost;truth=a['delta_misses'][mask];safe=feasible[mask];rows={}
        for name,raw in cal['raw'].items():
            error=np.asarray(raw[str(cost)]['residuals']);pred=truth-error
            offset=cal['corrections'][name][str(cost)]
            proposed=(pred < -1)&safe;corrected=(pred+offset < -1)&safe
            rows[name]={'validation_mae_commands':float(np.abs(error).mean()),'correction_commands':offset,
                'raw_feasible_admissions':int(proposed.sum()),'corrected_feasible_admissions':int(corrected.sum()),
                'raw_admission_beneficial':int(((truth<0)&proposed).sum()),
                'raw_admission_harmful':int(((truth>0)&proposed).sum()),
                'raw_admission_neutral':int(((truth==0)&proposed).sum())}
        groups[str(cost)]=rows
    result={'posthoc_validation_diagnostic':True,'new_controller_selected':False,'conditions':groups,
        'calibration_sha256':hashlib.sha256((OUT/'calibration.json').read_bytes()).hexdigest(),
        'interpretation':'An unconditional residual correction can reject useful feasible moves. Raw learned values also trail the numerical estimator. Removing the correction is not a validated remedy.'}
    (OUT/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':print(json.dumps(diagnose(),indent=2))
