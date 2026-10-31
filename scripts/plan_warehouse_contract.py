"""Produce a compositional warehouse proposal; never send a car command."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch
from wirelessjev.warehouse_horizon.belief import RadioBelief
from wirelessjev.warehouse_horizon.model import prior_prediction
from wirelessjev.warehouse_horizon.planning import Contract as SearchContract,candidate_pool,neighbours,charge
from wirelessjev.warehouse_v2.experiment import atomic_json,sha256
from wirelessjev.warehouse_v3.missions import Contract,Reader,select
from wirelessjev.warehouse_v3.clauses import ClauseReader


def make_plan(document,contract,*,temperature=1.,service_margin=0.):
    gateway_path=Path(__file__).with_name('plan_warehouse_horizon.py')
    spec=importlib.util.spec_from_file_location('original_horizon_gateway',gateway_path)
    gateway=importlib.util.module_from_spec(spec);spec.loader.exec_module(gateway)
    obs=gateway.validate_observation(document)
    roles=document.get('roles',{})
    if not isinstance(roles,dict) or set(roles)-{'teleoperated'}:
        raise ValueError('Unsupported application-role declarations')
    for values in roles.values():
        if not isinstance(values,list) or len(values)!=len(obs['current']) or any(type(v) is not bool for v in values):
            raise ValueError('Application role membership must be a matched boolean array')
    belief=RadioBelief(obs);search=SearchContract('throughput',contract.max_moves,0.)
    candidates,_=candidate_pool(obs,belief,search);count=0;best=None;action=None
    for iteration in range(3):
        predicted=prior_prediction(belief.forecast(candidates),obs)
        service,risk=charge(predicted,candidates,obs,temperature=temperature)
        result=select(contract,obs,candidates,service,risk,document['robot_ids'],roles,
            floor_service=np.clip(service-service_margin,0,1))
        count+=len(candidates)
        if result['choice'] is not None:
            if action is None or tuple(result['ordered_scores'])>tuple(best['ordered_scores']):
                best=result;action=candidates[result['choice']].copy()
            incumbent=action
        else:
            incumbent=candidates[int(np.argmin(result['maximum_service_shortfall_by_candidate']))]
            if best is None:best=result
        if iteration<2:candidates=neighbours(incumbent,obs,belief,search)
    best.update(evaluated_candidates=count,feasibility_basis='numerical_forecast_with_validation_marginal_margin',
        physical_safety_guarantee=False,globally_infeasible=False)
    if action is None:return None,best,None
    prediction=prior_prediction(belief.forecast(action[None]),obs)
    service,risk=charge(prediction,action[None],obs,temperature=temperature)
    final=select(contract,obs,action[None],service,risk,document['robot_ids'],roles,
        floor_service=np.clip(service-service_margin,0,1))
    if final['choice'] is None:raise RuntimeError('Final independent contract guard failed')
    return action,best,{'service_by_offset':service[0].tolist(),'critical_violation_risk_by_offset':risk[0].tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observation',type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--mission');group.add_argument('--contract',type=Path)
    p.add_argument('--selection',type=Path,default=Path('artifacts/warehouse_v3/core/selection.json'))
    p.add_argument('--reader',choices=['clauses','family','controlled'],default='controlled',
                   help='Canonical grammar by default; actual neural verification is an explicit option')
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2)
    document=json.loads(args.observation.read_text());start=time.perf_counter()
    if args.mission:
        from wirelessjev.policy import WirelessJEV,MODEL_REPO,MODEL_REVISION
        engine=WirelessJEV(device='cpu') if args.reader!='controlled' else None
        interpreter=Reader(engine) if args.reader=='family' else ClauseReader(engine)
        try:contract,reading=interpreter.read(args.mission)
        except ValueError as error:
            atomic_json(args.out,{'proposal_only':True,'actuation_performed':False,'status':'unsupported_or_semantic_abstention',
                'reason':str(error),'physical_safety_guarantee':False});return
        reading['model']={'repo':MODEL_REPO,'revision':MODEL_REVISION}
    else:
        contract=Contract.from_dict(json.loads(args.contract.read_text()))
        reading={'reader':'caller_provided_typed_contract','model_executed':False}
    # Evidence is required rather than silently pretending an uncalibrated
    # forecast is a calibrated service-floor check.
    selection=json.loads(args.selection.read_text())
    status=json.loads(args.selection.with_name('status.json').read_text())
    if status.get('state')!='completed' or status['reports']['selection.json']!=sha256(args.selection):
        raise ValueError('Calibration selection receipt does not match a completed campaign')
    calibration=selection['calibration']['numerical']
    action,receipt,forecast=make_plan(document,contract,temperature=calibration['temperature'],
        service_margin=calibration['one_sided_service_margin_q90'])
    assignments=None if action is None else {robot:document['ap_ids'][int(ap)] for robot,ap in zip(document['robot_ids'],action)}
    atomic_json(args.out,{'schema':'wirelessjev.compositional_plan.v1','proposal_only':True,'actuation_performed':False,
        'contract':contract.to_dict(),'reader':reading,'decision':receipt,'assignments':assignments,'forecast':forecast,
        'selection_sha256':sha256(args.selection),'observation_sha256':sha256(args.observation),
        'input_origin_authenticated':False,'input_source_self_reported':document['source'],
        'local_total_ms':(time.perf_counter()-start)*1000,'hardware_qualified':False})
    print('Proposal',receipt['status'],'; no car command sent')


if __name__=='__main__':main()
