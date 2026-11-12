"""Passive canonical case/clock validation; no models, sensors or simulations.

This checks admitted raw identities and arithmetic, not whether a handover is
safe or whether an RSSI predictor is accurate. Deeper sensor replay remains a
separate independent audit. The inputs are already hash-checked by the caller.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math

import numpy as np


def document_sha256(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError('Canonical spatial result: '+reason)


def _int(value, low=0, high=2**63-1):
    _require(type(value) is int and low <= value <= high, 'invalid integer clock/count')
    return value


def _number(value):
    _require(type(value) in (int,float) and math.isfinite(value) and value >= 0, 'invalid finite cost')
    return float(value)


def _equal(value, expected):
    return math.isclose(_number(value), float(expected), abs_tol=1e-10, rel_tol=0)


def _array(value, shape, *, kind=None, finite=True):
    _require(isinstance(value,np.ndarray) and value.shape == shape, 'array axes differ from registered case')
    if kind is not None: _require(value.dtype.kind in kind, 'array dtype differs from registered type')
    if finite: _require(bool(np.isfinite(value).all()), 'nonfinite array')
    return value


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _policy(ho):
    # Import only the local immutable policy schema, never a model or runner.
    from .kernel import Policy
    value=asdict(Policy('spatial_control_diagnostic','native',roam_cost_s=ho,
        scan_period_s=5.,gate=True,cadence_scan=True,cadence_roam=True,online_map=False,dual_radio=False))
    return json.loads(json.dumps(value))


def validate_result(receipt,result,arrays,source_receipt,world_receipt,protocol):
    """Refuse a rehashed but wrong case, cost, schedule, array or action ledger."""
    try:
        return _validate(receipt,result,arrays,source_receipt,world_receipt,protocol)
    except (KeyError,TypeError,IndexError,OverflowError) as error:
        raise ValueError('Canonical spatial result: missing/malformed required field') from None


def _validate(receipt,result,arrays,source_receipt,world_receipt,p):
    duration=_int(p['duration_seconds'],1); n=_int(p['robots'],1); slots=duration*100
    _require(duration==20 and n==100, 'unregistered episode budget')
    parity=receipt['parity_kernel']; arm=receipt['arm']; case=receipt['case']
    _require(type(case) is dict and type(arm) is str, 'case identity')
    if parity is None:
        expected=next((c for c in p['cases'] if c['id']==case.get('id')),None)
        _require(expected is not None and case==expected and arm in p['arms'], 'fresh case/arm membership')
        raw_arm=arm; profile=p['forecast']['profile_seconds']; causal=True
    else:
        expected={'id':'disabled-parity','seed':p['parity']['seed'],'handover_seconds':p['parity']['handover_seconds']}
        _require(parity in ('original','disabled') and case==expected
                 and arm=={'original':'radiomap__original','disabled':'radiomap__disabled'}[parity], 'parity case/arm membership')
        raw_arm='radiomap'; profile={key:p['parity']['profile_seconds'] for key in ('cold','warm','refit')}; causal=False
    ho=_number(case['handover_seconds']); seed=_int(case['seed'])
    _require(receipt['simulations']==1 and receipt['simulation_only'] is True
             and receipt['production_qualified'] is False and receipt['default_changed'] is False
             and receipt['original_SERVICE_opened'] is False,'receipt execution/authority scope')
    prefix='artifacts/spatial_control/cases/'+case['id']+'/'+arm+'/'
    for key,name in (('raw','raw.json.gz'),('arrays','predictions_history.npz'),('attempt','attempt.json')):
        _require(receipt[key]['path']==prefix+name, 'noncanonical '+key+' path')
    source_sha=document_sha256(source_receipt); protocol_sha=document_sha256(p)
    _require(receipt['source_sha256']==source_sha and receipt['protocol_sha256']==protocol_sha,
             'source/protocol reference mismatch')
    _require(source_receipt['protocol']=={'path':'artifacts/spatial_control/protocol.json','sha256':protocol_sha}, 'source protocol dependency')
    _require(source_receipt['checkpoint']['path']=='artifacts/adaptive_radio/checkpoints/regression-seed0.pt'
             and p['weights']['selected_cell']=='regression' and p['weights']['seed']==0 and p['weights']['new_fits']==0,
             'fixed checkpoint/no-fit scope')
    _require(receipt['world_receipt_sha256']==document_sha256(world_receipt)
             and world_receipt['source_sha256']==source_sha and world_receipt['protocol_sha256']==protocol_sha
             and world_receipt['seed']==seed, 'world/source binding')
    for key,filename in (('world','world.npz'),('warm','warm.npz')):
        _require(world_receipt[key]['path']==f'artifacts/spatial_control/worlds/{seed}/{filename}', 'world archive path')
    _require(result['arm']==raw_arm and result['seed']==seed and _equal(result['handover_seconds'],ho)
             and result['causal_central'] is causal and result['production_qualified'] is False, 'raw arm/seed/HO/authority')
    _require(document_sha256(result['policy_config'])==document_sha256(_policy(ho)), 'raw native policy configuration')
    config={'seed':0,'duration_seconds':duration,'collision_lambda':0.,'interference':False,
        'scan_duration_seconds':.09,'observation_period_slots':10,'native_proposal_ttl_slots':100,'planner':'native'}
    _require(document_sha256(result['simulation_config'])==document_sha256(config), 'raw simulator configuration')
    native=result['native']; detail=native['packet_detail']
    _require(native['scene']==seed and native['policy']=='spatial_control_diagnostic'
             and _equal(native['duration_s'],duration) and _equal(native['collision_lambda'],0)
             and native['interference'] is False, 'native case/configuration identity')
    _require(detail['stop_slot']==slots and detail['resume_slot']==0
             and detail['observation_period_slots']==10 and detail['native_proposal_ttl_slots']==100
             and detail['focal_hold_slots'] is None, 'native episode/observation support')
    sites=len(world_receipt['APs']); _require(sites>0, 'missing registered AP axis')
    shape=(duration,n,9,sites,3)
    mean=_array(arrays['forecast_mean_snr'],shape,kind='f')
    std=_array(arrays['forecast_std_db'],shape,kind='f')
    _require(mean.dtype==np.dtype('float32') and std.dtype==np.dtype('float32') and bool((std>=0).all()), 'forecast float32/std contract')
    expected_keys={'forecast_mean_snr','forecast_std_db'}
    for name,axes,kind in (('positions',(60+duration,n,2),'f'),('value',(60+duration,n,sites*3),'f'),
            ('heard',(60+duration,n,sites*3),'b'),('bound',(60+duration,n,sites*3),'f'),('scanned',(60+duration,n),'b')):
        key='history_'+name;expected_keys.add(key);_array(arrays[key],axes,kind=kind)
    heard=arrays['history_heard']; scanned=arrays['history_scanned']
    _require(not bool((heard & ~scanned[...,None]).any()), 'heard history without measured scan')
    jobs=result['forecast_jobs']; _require(type(jobs) is list and len(jobs)==duration,'exact forward/job membership')
    previous=0.; ready=[]
    learned=raw_arm.startswith('regression')
    for j,job in enumerate(jobs):
        _require(job['decision']==j and type(job['decision']) is int and job['history_cutoff_exclusive']==60+j, 'job identity/history cutoff')
        kind=job['kind'];_require(kind in profile and kind==('cold' if j==0 else 'warm'),'unregistered job timing kind')
        prediction=_number(job['actual_prediction_seconds']); issue=_number(job['actual_issue_seconds'])
        actual=max(prediction,issue); charged=max(_number(profile[kind]),actual)
        service=charged+_number(p['forecast']['shared_extra_seconds'])
        for key,value in (('actual_compute_seconds',actual),('charged_compute_seconds',charged),
                ('profile_seconds',profile[kind]),('shared_extra_seconds',p['forecast']['shared_extra_seconds']),('charged_seconds',service)):
            _require(_equal(job[key],value),'forecast cost/profile accounting')
        _require(job['runtime_overrun'] is (actual>profile[kind]) and job['predictor_runtime_overrun'] is (prediction>profile[kind])
                 and job['completion_uses_actual_elapsed'] is True and job['offline_recorded_schedule'] is False
                 and job['accounting_mode']=='actual_cost','forecast completion semantics')
        previous=max(float(j),previous)+service; slot=math.ceil(previous*100-1e-10);ready.append(slot)
        _require(job['ready_slot']==slot and type(job['ready_slot']) is int,'FIFO forecast completion')
        _require(job['forecast_shape']==list(shape[1:]) and job['forecast_dtype']=='float32'
                 and job['forecast_sha256']==_sha(mean[j]) and job['forecast_std_sha256']==_sha(std[j]),'actual forecast NPZ identity')
        if learned:
            for name in ('context','known','scheduled','selected_rows','plan','mean_dbm','std_db','belief_mean','belief_covariance'):
                key=f'job{j}_{name}';expected_keys.add(key)
                _require(key in arrays and isinstance(arrays[key],np.ndarray) and bool(np.isfinite(arrays[key]).all()),'missing/nonfinite learned forward array')
            _array(arrays[f'job{j}_plan'],(n,9,2),kind='f')
            _array(arrays[f'job{j}_mean_dbm'],shape[1:],kind='f');_array(arrays[f'job{j}_std_db'],shape[1:],kind='f')
            selected=arrays[f'job{j}_selected_rows'];_require(selected.dtype.kind in 'iu' and selected.ndim==2
                and selected.shape[1]==4 and 1<=len(selected)<=64,'encoder source-row axes')
            _require(bool((selected[:,0]<60+j).all()) and bool((selected>=0).all())
                and bool((selected[:,1]<n).all()) and bool((selected[:,2]<sites).all())
                and bool((selected[:,3]<2).all()),'encoder uses future/unsupported history')
            _require(job['heard_entries']==int(heard[59+j].sum())
                and job['scans_last_bin']==int(scanned[59+j].sum()),'forecast history-mask identity')
    _require(set(arrays)==expected_keys,'extra/missing forecast/history/forward arrays')
    last=-1; last_slot=-1
    for event in result['forecast_consumption']:
        j=_int(event['forecast_decision'],0,duration-1);slot=_int(event['slot'],0,slots-1)
        _require(j>last and slot>=last_slot and event['ready_slot']==ready[j] and ready[j]<=slot
                 and j==int(np.searchsorted(ready,slot,side='right'))-1
                 and _equal(event['age_seconds'],slot/100-j),'consumption precedes ready/newest completion')
        last,last_slot=j,slot
    for name in ('misses','sent','gap','max_gap','window_max_gap'):
        value=np.asarray(detail[name]);_require(value.shape==(n,) and value.dtype.kind in 'iu' and bool((value>=0).all()),'packet counter axes/types')
    misses=np.asarray(detail['misses']);sent=np.asarray(detail['sent']);gaps=np.asarray(detail['max_gap'])
    _require(bool((misses<=sent).all()) and native['control_misses']==int(misses.sum()) and native['control_sent']==int(sent.sum()),'packet totals')
    _require(_equal(native['control_miss'],misses.sum()/sent.sum())
             and _equal(native['control_miss_worst_decile'],np.sort(misses/np.maximum(sent,1))[-10:].mean())
             and _equal(native['control_gap_p99_ms'],np.percentile(gaps,99)*10),'packet headline arithmetic')
    if not causal:
        _require('spatial_execution' not in native and result['local_runtime'] is None,'disabled parity enabled extension')
        return {'passes':True,'jobs':duration,'causal_central':False,'scope':'passive identity/clock arithmetic only'}
    spatial=native['spatial_execution']; recovery=raw_arm in ('radiomap_recovery','regression_recovery','regression_both')
    _require(spatial['causal_central'] is True and spatial['local_recovery_enabled'] is recovery
             and (result['local_runtime'] is not None) is recovery,'local capability/arm scope')
    for key,value in (('central_creation_delay_slots',10),('central_ttl_slots',100),('scan_minimum_spacing_slots',500),
                      ('local_decision_floor_slots',1)):
        _require(spatial[key]==value,'registered action budget')
    _require(_equal(spatial['central_decision_budget_seconds'],.1) and _equal(spatial['local_decision_budget_seconds'],.005),'CPU budget changed')
    central_decisions={};busy=0
    for row in spatial['central_decisions']:
        clock=_int(row['slot'],0,slots-1);elapsed=_number(row['elapsed_seconds'])
        _require(clock>=busy and clock not in central_decisions and _equal(row['budget_seconds'],.1)
            and row['overrun'] is (elapsed>.1) and row['created_slot']==clock+10
            and row['compute_ready_slot']==clock+max(10,math.ceil(elapsed/.01-1e-12)),'central CPU occupancy/allowance')
        _require(any(e['slot']==clock and e['forecast_decision']==row['forecast_decision'] for e in result['forecast_consumption']),
            'central decision without consumed forecast')
        central_decisions[clock]=row;busy=row['compute_ready_slot']
    local_decisions={};validations={}
    for row in spatial['local_decisions']:
        clock=_int(row['slot'],0,slots-1);elapsed=_number(row['elapsed_seconds'])
        _require(clock%10==0 and clock not in local_decisions and _equal(row['budget_seconds'],.005)
            and row['overrun'] is (elapsed>.005)
            and row['ready_slot']==clock+max(1,math.ceil(elapsed/.01-1e-12)),'local CPU occupancy/allowance')
        local_decisions[clock]=row
    for row in spatial['local_validation_batches']:
        clock=_int(row['slot'],0,slots-1);elapsed=_number(row['elapsed_seconds'])
        _require(_equal(row['budget_seconds'],.005)
            and _equal(elapsed,sum(_number(c['elapsed_seconds']) for c in row['checks']))
            and row['overrun'] is (elapsed>.005 or any(c['elapsed_seconds']>.005 for c in row['checks'])),'local validation allowance')
        for check in row['checks']:
            _require((clock,check['action_id']) not in validations,'duplicate validation identity')
            validations[clock,check['action_id']]=bool(check['evidence_valid'] is True and row['overrun'] is False)
    commands={}; terminal=set(); interruption=math.ceil(ho/.01-1e-12)
    for event in spatial['central_events']:
        command=event['command_id'];kind=event['event'];clock=_int(event['slot'],0,slots)
        robot=_int(event['robot'],0,n-1);_int(event['current'],0,sites-1);_int(event['target'],0,sites-1)
        _require(event['current']!=event['target'],'central no-op')
        base={key:event[key] for key in ('command_id','robot','current','target','proposed_slot','created_slot','expires_slot','interruption_slots')}
        _require(event['created_slot']==event['proposed_slot']+10 and event['expires_slot']==event['created_slot']+100
                 and event['interruption_slots']==interruption,'central creation/expiry/HO cost')
        if kind=='queued':
            _require(command not in commands and clock==event['proposed_slot'],'central command ID reused')
            _require(clock in central_decisions and central_decisions[clock]['overrun'] is False,'central action from unpaid/overrun computation')
            commands[command]=base
        else:
            _require(command in commands and commands[command]==base and command not in terminal,'central lifecycle binding')
            _require(kind in ('cancelled','executed','right_censored'),'unknown central lifecycle')
            terminal.add(command)
            if kind=='executed':
                _require(event['created_slot']<event['delivery_slot']==event['execution_slot']==clock<event['expires_slot']
                    and event['created_slot']<=event['packet_created_slot']<=clock
                    and 0<=clock-event['packet_created_slot']<5
                    and event['outage_until_slot']==clock+1+interruption,'central command lacks paid causal delivery')
            elif kind=='right_censored':_require(clock==slots,'central censor before end')
    _require(set(commands)==terminal,'unfinished central command lacks terminal/censor record')
    initial=spatial['initial_scan_start_upper_slot'];_require(len(initial)==n,'initial scan clock axes')
    previous_scan={i:_int(value,-6000,0) for i,value in enumerate(initial)}
    for event in spatial['scan_events']:
        robot=_int(event['robot'],0,n-1);clock=_int(event['slot'],0,slots-1)
        _require(clock-previous_scan[robot]>=500 and event['measurement_slot']==clock
                 and event['interruption_start_slot']==clock+1 and event['complete_slot']==clock+10
                 and event['interruption_slots']==9,'scan clocks/cadence/90ms cost')
        _require(event['reason'] in ('autonomous_recovery','background_after_delivery'),'scan reason')
        _require((event['action_id'] is not None)==(event['reason']=='autonomous_recovery')
                 and (recovery or event['action_id'] is None),'unauthorized local scan')
        previous_scan[robot]=clock
    local_queued={};local_terminal=set()
    for event in spatial['local_events']:
        kind=event['event'];key=event['action_id'];clock=_int(event['slot'],0,slots)
        if kind=='queued':
            _require(recovery and key not in local_queued and event['request_slot']==clock and event['ready_slot']==clock+1,'local request delay/identity')
            _require(clock in local_decisions and local_decisions[clock]['overrun'] is False
                and any(r['action_id']==key and r['robot']==event['robot'] for r in local_decisions[clock]['requests']),
                'local request from unpaid/overrun computation')
            local_queued[key]=event
        elif kind=='executed':
            _require(key in local_queued and key not in local_terminal and clock>=local_queued[key]['ready_slot'],'unbound/early local execution')
            _require(validations.get((clock,key)) is True,'local action without paid valid evidence check')
            q=local_queued[key]
            _require(event['robot']==q['robot'] and event['target']==q['target'] and event['current']==q['current']
                     and event['outage_until_slot']==clock+1+interruption and event['interruption_slots']==interruption
                     and event['last_delivery_slot']<q['request_slot'],'local clock/HO cost/silence')
            local_terminal.add(key)
        elif kind=='rejected':
            if key in local_queued:local_terminal.add(key)
        elif kind=='right_censored':
            _require(key in local_queued and clock==slots,'unbound local censor')
            local_terminal.add(key)
        else:_require(False,'unknown local lifecycle')
    # A queued scan terminates in its paid scan event, not a handover event.
    local_terminal.update(e['action_id'] for e in spatial['scan_events'] if e['action_id'] is not None)
    for event in spatial['scan_events']:
        key=event['action_id']
        if key is not None:
            _require(key in local_queued and validations.get((event['slot'],key)) is True
                and event['slot']>=local_queued[key]['ready_slot'],'local scan without paid request/validation')
    _require(set(local_queued)<=local_terminal,'unaccounted queued local request')
    ages=np.asarray(spatial['pending_packet_age_end']);_require(ages.shape==(n,) and ages.dtype.kind in 'iu'
        and spatial['pending_control_packets']==int((ages>=0).sum()),'end pending packet bound')
    return {'passes':True,'jobs':duration,'causal_central':True,'central_commands':len(commands),
            'local_requests':len(local_queued),'scope':'passive identity/clock arithmetic only; no service qualification'}
