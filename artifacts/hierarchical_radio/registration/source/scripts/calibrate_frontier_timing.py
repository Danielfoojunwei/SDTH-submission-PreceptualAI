"""Include common tensor packing in the service-time ledger before selection.

The frozen training kernel times shared feature construction and each forward
pass separately. This independent calibration adds the shared tensor-copy cost
to each single-model deployment, then seals the complete charge before tests.
"""
from pathlib import Path
import json,time,shutil
import numpy as np
import torch
from wirelessjev.frontier.study import ROOT,OUT,P,digest,benchmark,tensors,mask_scans
from wirelessjev.frontier.fast import FleetMemory
from wirelessjev.fleetwm.data import episode
from wirelessjev.contracted.train import write


def main():
    if (OUT/'selection.json').exists():
        selected=json.loads((OUT/'selection.json').read_text())
        receipt=json.loads((OUT/'latency.json').read_text())
        assert selected['latency_sha256']==digest(OUT/'latency.json')
        assert receipt['tensor_packing_accounted']
        assert receipt['calibration_source_sha256']==digest(OUT/'timing_calibration_source/calibrate_frontier_timing.py')
        print('VERIFIED_SEALED_TIMING_WITHOUT_RECALIBRATION');return
    benchmark();path=OUT/'latency.json';base=OUT/'latency_base.json'
    if base.exists():
        current=json.loads(path.read_text());assert current.get('tensor_packing_accounted');return
    torch.set_num_threads(1);full,scan,g,original=episode(P['validation_sites'][0],100,8)
    overhead={};raw={}
    for cadence in [1,5]:
        obs=mask_scans(original,cadence);memory=FleetMemory(g);durations=[]
        for d in range(6):
            b,c=memory.inputs(obs.before(60+d),full.plan[60+d])
            for _ in range(5):
                a=time.perf_counter();tensors(b,c);durations.append(time.perf_counter()-a)
        raw[str(cadence)]=durations;overhead[str(cadence)]=float(np.quantile(durations,.95))
    shutil.copyfile(path,base);receipt=json.loads(base.read_text())
    for cadence,groups in receipt['p95_seconds'].items():
        for family in P['families']:
            for kind in groups[family]:groups[family][kind]+=overhead[cadence]
    source=Path(__file__).resolve();capsule=OUT/'timing_calibration_source'/source.name;capsule.parent.mkdir(exist_ok=True);shutil.copyfile(source,capsule)
    receipt.update({'tensor_packing_accounted':True,'tensor_packing_p95_seconds':overhead,'tensor_packing_raw_seconds':raw,
        'base_profile_sha256':digest(base),'calibration_source_sha256':digest(source),'calibration_time_unix':time.time()})
    write(path,receipt);print(json.dumps({'charged_profile_complete':True,'tensor_packing_p95_seconds':overhead},indent=2))


if __name__=='__main__':main()
