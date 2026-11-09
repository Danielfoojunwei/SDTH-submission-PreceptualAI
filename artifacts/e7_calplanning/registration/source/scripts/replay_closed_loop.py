"""Read-only exact replay of four sealed delivered-telemetry test arms."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import numpy as np
import torch
from wirelessjev.closed_loop.study import OUT, P, source_hashes, checkpoint_hashes, digest
from wirelessjev.closed_loop.study import IdealHistory
from wirelessjev.closed_loop.runtime import SinglePredictor, LiveForecasts
from wirelessjev.closed_loop.telemetry import DeliveredScans
from wirelessjev.closed_loop.control import OutageUtility
from wirelessjev.closed_loop.sim import simulate
from wirelessjev.frontier.study import mask_scans, runtime_world, READINGS
from wirelessjev.fleetwm.data import episode
from wirelessjev.native.intent_policy import intent_policy
from wirelessjev.gate_repair.gate import reactive_rescue
from wirelessjev.hardware_budget import OutageBudgetGate


def replay(path):
    torch.set_num_threads(1)
    expected=json.loads(path.read_text())
    assert expected['source_sha256']==source_hashes() and expected['checkpoint_sha256']==checkpoint_hashes()
    assert expected['latency_sha256']==digest(OUT/'latency.json')
    seed=expected['scene'];family=expected['family'];variant=expected['variant'];cost=expected['handover_seconds']
    full,_,g,o=episode(seed,100,60,'drift' if seed%2 else 'main');o=mask_scans(o,5)
    delivered=DeliveredScans(o.before(60),60,seed)
    history=IdealHistory(o) if variant=='ideal_reports' else delivered
    latency=json.loads((OUT/'latency.json').read_text())['p95_seconds'][family]
    live=LiveForecasts(SinglePredictor(g,family),history,60,latency,P['extra_compute_seconds'])
    utility=OutageUtility(live.mean,live,cost,expected['horizon_seconds']) if variant=='delivered_utility' else None
    admission=utility if utility is not None else OutageBudgetGate(live.mean,live,cost)
    reading=READINGS['bulk_uploads']
    policy=replace(intent_policy(reading['weights'],reading['focus'],dual_hardware='none'),name=family+'_'+variant,
                   online_map=False,gate=True,scan_period_s=5.,cadence_scan=True,roam_cost_s=cost)
    actual=simulate(runtime_world(full,60,live.mean,live.std[:,:,0]),policy,roam_gate=reactive_rescue,
                    handover_admission=admission,forecast_queue=live,forecast_source=live,scan_observer=delivered,
                    scan_duration_s=.09,roam_scorer=utility.scores if utility is not None else None)
    assert actual==expected['result'],path.name
    assert delivered.summary()==expected['telemetry'],path.name
    for a,b in zip(live.records,expected['jobs']):
        assert {k:v for k,v in a.items() if k!='actual_prediction_seconds'}=={k:v for k,v in b.items() if k!='actual_prediction_seconds'},path.name
    return {'case':path.name,'exact_packet_replay':True,'exact_delivery_replay':True,'exact_forecasts':len(live.records)}


if __name__=='__main__':
    paths=sorted(p for p in (OUT/'cases').glob('test-321007-4.0-*.json') if 'ideal_reports' not in p.name)
    assert len(paths)==4
    with ProcessPoolExecutor(4) as pool:results=list(pool.map(replay,paths))
    receipt={'passed':True,'replays':results,'simulation_only':True,
             'replay_code_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()}
    (OUT/'replay.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
