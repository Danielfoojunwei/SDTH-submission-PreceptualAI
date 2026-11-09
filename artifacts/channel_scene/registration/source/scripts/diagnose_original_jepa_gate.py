"""Forensic replay of the OLD six scenes, only after the new choice is sealed.

These are diagnostic labels, not training/selection or fresh test evidence.
The old command counters must reproduce exactly before interpreting the traces.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from wirelessjev.gate_repair.study import OUT,ROOT,build,MISSION,READINGS
from wirelessjev.gate_repair.gate import Recorder,diagnostics
from wirelessjev.contracted.data import runtime_world
from wirelessjev.contracted.train import write
from wirelessjev.native.intent_policy import intent_policy
from wirelessjev.native.sim import simulate


def one(seed):
    torch.set_num_threads(1);full,scan,data,bank=build(seed,'main',120)
    policy=replace(intent_policy(READINGS[MISSION]['weights'],READINGS[MISSION]['focus'],dual_hardware='none'),online_map=False)
    recorder=Recorder(bank);result=simulate(runtime_world(full,120,bank.mean,bank.std[:,:,0]),policy,decision_trace=recorder)
    original=json.loads((ROOT/f'artifacts/contracted/cases/main-{seed}.json').read_text())
    reference=next(r for r in original['rows'] if r['arm']=='jepa_gated_base' and r['mission']==MISSION)
    compared=['control_misses','control_sent','roams','video_on_time','bulk_mbps','control_gap_p99_ms']
    for key in compared:assert result[key]==reference[key],(seed,key,result[key],reference[key])
    rows=recorder.label(data['truth']);dest=OUT/'historical_diagnosis';dest.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(dest/f'{seed}.npz',**{k:v[rows['eligible']] for k,v in rows.items()})
    r={'scene':seed,'diagnostic_only':True,'used_for_fit_or_selection':False,'original_counters_match':True,'counter_fields_compared':compared,'result':result,'diagnosis':diagnostics(rows)}
    write(dest/f'{seed}.json',r);return r


def main():
    selection=OUT/'selection.json';before=hashlib.sha256(selection.read_bytes()).hexdigest();assert json.loads(selection.read_text())['sealed']
    with ProcessPoolExecutor(2) as pool:parts=list(pool.map(one,range(262000,262006)))
    assert hashlib.sha256(selection.read_bytes()).hexdigest()==before
    write(OUT/'historical_diagnosis.json',{'completed':True,'selection_sha256_before_and_after':before,'no_tuning_on_historical_replays':True,'scenes':[p['scene'] for p in parts]})
    print('ORIGINAL_SIX_SCENE_FORENSIC_REPLAY_MATCHES',flush=True)


if __name__=='__main__':main()
