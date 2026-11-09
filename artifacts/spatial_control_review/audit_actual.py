#!/usr/bin/env python3
"""Independent passive reduction of the sealed spatial-control diagnostic.

Imports stdlib and NumPy only. No project imports, model constructors, fitting,
forwards, RT solves, simulator invocation or network requests. It refuses to
score until the complete official report and immutable closure exist.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

SEEDS=(90511001,90511002,90511003,90511004)
ARMS=('radiomap','radiomap_recovery','regression','regression_support','regression_recovery','regression_both')
CARRIERS=(2.4e9,5.2e9,6.5e9)
HORIZONS=(0.,1.,2.,3.,4.,5.,10.,20.,30.)
FIELDS=('positions','value','heard','bound','scanned')
JOB_FIELDS=('context','known','scheduled','selected_rows','plan','mean_dbm','std_db','belief_mean','belief_covariance')
PRIOR=('radiojepa','radiojepa_spectral','device_transfer','channel_scene','adaptive_radio','native_bridge','e7_calplanning')


def fail(message): raise AssertionError(message)
def require(condition,message):
    if not condition: fail(message)
def read(path): return json.loads(Path(path).read_bytes())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def array_hash(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()
def integer(value,low=0): return type(value) is int and value>=low
def close(a,b,label,tolerance=1e-10):
    if a is None or b is None:
        require(a is b,label+' unknown mismatch'); return
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=tolerance),label+f': {a} != {b}')
def numbers(actual,expected,label):
    require(set(actual)==set(expected),label+' key membership')
    for key in actual: close(actual[key],expected[key],label+'/'+key)
def same(actual,expected,label): require(actual==expected,label)
def npz(path):
    with np.load(path,allow_pickle=False) as a:
        require(all(a[n].dtype.kind!='O' for n in a.files),'inert numeric arrays required')
        return {n:a[n].copy() for n in a.files}
def contained(root,name):
    p=(root/name).resolve();require(p.is_relative_to(root.resolve()) and p.is_file(),'unsafe/missing path '+str(name));return p

def ref(root,item):
    require(isinstance(item,dict) and set(item)=={'path','sha256'},'strict artifact reference')
    p=contained(root,item['path']);same(sha(p),item['sha256'],'reference hash '+item['path']);return p


def verify_sources(root,out):
    manifest_path=out/'verification/manifest.json'
    require(manifest_path.exists() and (out/'report.json').exists(),'Complete report AND seal required before any fresh-case scoring')
    manifest=read(manifest_path);registration=read(out/'registration/source.json')
    require(manifest['source_sha256']==registration['source_sha256'],'seal source identity')
    same(set(p.relative_to(out/'registration/source').as_posix() for p in (out/'registration/source').rglob('*') if p.is_file()),set(registration['source_sha256']),'source capsule exact membership')
    for name,digest in registration['source_sha256'].items():
        same(sha(contained(root,name)),digest,'current source '+name)
        same(sha(contained(out/'registration/source',name)),digest,'source capsule '+name)
    expected={p.relative_to(root).as_posix() for p in out.rglob('*') if p.is_file()
              and not p.is_relative_to(out/'registration/source') and p!=manifest_path}
    same(set(manifest['evidence_sha256']),expected,'sealed evidence exact membership')
    for name,digest in manifest['evidence_sha256'].items(): same(sha(contained(root,name)),digest,'evidence '+name)
    for name in ('protocol','preflight','preflight_log','checkpoint','VAL_selection'): ref(root,registration[name])
    publication=read(out/'registration/git.json')
    same(publication['source_sha256'],sha(out/'registration/source.json'),'publication receipt binding')
    same(publication['commit'],publication['remote_observed_commit'],'observed published commit')
    # Read retained Git objects, not an asserted commit label; no network call.
    for name,digest in registration['source_sha256'].items():
        data=subprocess.check_output(['git','show',publication['commit']+':'+name],cwd=root)
        same(hashlib.sha256(data).hexdigest(),digest,'published source '+name)
    union={}
    require(set(registration['prior']['manifests'])==set(PRIOR),'prior study membership')
    for study in PRIOR:
        path=ref(root,registration['prior']['manifests'][study]);prior=read(path)
        for field in ('source_sha256','evidence_sha256','documents_sha256'):
            for name,digest in prior.get(field,{}).items():
                require(name not in union or union[name]==digest,'conflicting historical identity')
                union[name]=digest
    for name,digest in union.items(): same(sha(contained(root,name)),digest,'prior frozen file '+name)
    for forbidden in ('artifacts/radiojepa/rt_data/confirmation','artifacts/radiojepa/rt_data/confirmation.json'):
        require(not (root/forbidden).exists(),'original SERVICE remains closed')
    return manifest,registration,{'sources':len(registration['source_sha256']),'evidence':len(expected),'prior_unique_files':len(union),'published_commit':publication['commit']}


def reducer(raw):
    native=raw['native'];detail=native['packet_detail'];jobs=raw['forecast_jobs']
    misses=np.asarray(detail['misses']);sent=np.asarray(detail['sent']);gap=np.asarray(detail['max_gap'])
    require(misses.shape==sent.shape==gap.shape==(100,),'100 actual per-client counters')
    require(np.all(misses>=0) and np.all(misses<=sent) and np.all(sent==400) and np.all(gap>=0),'fixed40,000 command releases / valid counters')
    require(np.all(misses==np.floor(misses)) and np.all(gap==np.floor(gap)),'integer packet/gap counts')
    result={'control_miss':float(sum(misses)/sum(sent)),
        'control_miss_worst_decile':float(np.mean(sorted(misses/sent)[-10:])),
        'maximum_silence_ms':float(max(gap)*10),'control_gap_p99_ms':float(np.percentile(gap,99)*10)}
    for key in ('control_miss','control_miss_worst_decile','control_gap_p99_ms'):close(native[key],result[key],'native derived '+key)
    same(native['control_misses'],int(sum(misses)),'native total misses');same(native['control_sent'],40000,'native total releases')
    spatial=native.get('spatial_execution',{});pending=spatial.get('pending_control_packets')
    if spatial:
        ages=np.asarray(spatial['pending_packet_age_end'])
        require(ages.shape==(100,) and np.all(ages>=-1) and np.all(ages<5),'right-censored packet age ledger')
        same(pending,int(sum(ages>=0)),'pending count from actual age mask')
        require(np.all(misses+(ages>=0)<=sent),'pending upper risk cannot exceed exposure')
    result.update({key:native[key] for key in ('video_on_time','bulk_mbps','roams','safety_stop_fraction')})
    scans=spatial.get('scan_events',[])
    result.update(control_misses=int(sum(misses)),control_sent=int(sum(sent)),end_pending_commands=pending,
        observed_plus_pending_miss_fraction=float((sum(misses)+pending)/sum(sent)) if pending is not None else None,
        paid_forecast_compute_seconds=sum(j['charged_compute_seconds'] for j in jobs),
        actual_forecast_compute_seconds=sum(j['actual_compute_seconds'] for j in jobs),
        forecast_jobs_consumed=len(raw['forecast_consumption']),scans_started=raw['telemetry']['scans_started'],
        scan_interruption_seconds=sum(e['interruption_slots'] for e in scans)*.01,
        autonomous_scans=sum(e['reason']=='autonomous_recovery' for e in scans),
        central_planning_cpu_seconds=sum(e['elapsed_seconds'] for e in spatial.get('central_decisions',[])),
        central_planning_overruns=sum(e['overrun'] for e in spatial.get('central_decisions',[])),
        local_decision_cpu_seconds=sum(e['elapsed_seconds'] for e in spatial.get('local_decisions',[])),
        local_decision_overruns=sum(e['overrun'] for e in spatial.get('local_decisions',[])),
        support_preparation_cpu_seconds=sum(e['elapsed_seconds'] for e in raw.get('support_preparation',[])),
        support_vetoes=sum(not e['permitted'] for e in raw['support']),
        local_handovers=sum(e['event']=='executed' for e in spatial.get('local_events',[])),
        central_handovers=sum(e['event']=='executed' for e in spatial.get('central_events',[])))
    if 'local_validation_cpu_seconds' in raw.get('_metric_keys',()):
        result.update(local_validation_cpu_seconds=sum(e['elapsed_seconds'] for e in spatial.get('local_validation_batches',[])),
            local_validation_overruns=sum(e['overrun'] for e in spatial.get('local_validation_batches',[])),
            local_busy_tick_skips=len(spatial.get('local_compute_events',[])),
            central_busy_forecast_skips=len(spatial.get('central_compute_events',[])))
    cumulative=detail['cumulative']
    same([r['slot_end'] for r in cumulative],list(range(100,2001,100)),'complete one-second cumulative ledger')
    for key in ('misses','sent','max_gap'):
        previous=np.zeros(100)
        for row in cumulative:
            a=np.asarray(row[key]);require(a.shape==(100,) and np.all(a>=previous),'monotone actual cumulative '+key);previous=a
        require(np.array_equal(previous,np.asarray(detail[key])),'final cumulative '+key)
    return result


def check_forecasts(raw,a,world,learned,parity):
    mean=a['forecast_mean_snr'];std=a['forecast_std_db'];sites=len(world['sites'])
    same(mean.shape,(20,100,9,sites,3),'forecast axes');same(std.shape,mean.shape,'std axes')
    require(mean.dtype==std.dtype==np.dtype('f4') and np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std>=0),'finite float32 forecast ledger')
    jobs=raw['forecast_jobs'];same(len(jobs),20,'20 forecast jobs')
    previous_ready=0.;previous_slot=-1
    for d,job in enumerate(jobs):
        same(job['decision'],d,'FIFO forecast issue');same(job['history_cutoff_exclusive'],60+d,'strict past history cutoff')
        profile=4. if parity else (1. if job['kind'] in ('cold','refit') else .25)
        close(job['profile_seconds'],profile,'forecast profile');close(job['shared_extra_seconds'],.5,'shared delay')
        charged=max(profile,job['actual_prediction_seconds'],job['actual_issue_seconds'])
        close(job['actual_compute_seconds'],max(job['actual_prediction_seconds'],job['actual_issue_seconds']),'actual measured compute')
        close(job['charged_compute_seconds'],charged,'actual overrun charged');close(job['charged_seconds'],charged+.5,'transport charged')
        previous_ready=max(float(d),previous_ready)+charged+.5
        expected=math.ceil(previous_ready*100-1e-10)
        same(job['ready_slot'],expected,'measured FIFO completion slot')
        same(job['forecast_sha256'],hashlib.sha256(mean[d].tobytes()).hexdigest(),'forecast output bytes')
        same(job['forecast_std_sha256'],hashlib.sha256(std[d].tobytes()).hexdigest(),'forecast std bytes')
        require(job['actual_compute_seconds']>=0,'nonnegative actual compute')
        if learned:
            np.testing.assert_array_equal(a[f'job{d}_plan'],world['plan'][d])
            m=a[f'job{d}_mean_dbm'];s=a[f'job{d}_std_db']
            converted=(m.astype('f8')-world['tx_dl']+15-world['noise_db']).astype('f4')
            np.testing.assert_allclose(converted,mean[d],rtol=0,atol=8e-6)
            np.testing.assert_array_equal(s.astype('f4'),std[d])
            check_context(a,d,world)
    for c in raw['forecast_consumption']:
        require(integer(c['slot']) and c['slot']>previous_slot and 0<=c['forecast_decision']<20,'monotonic consumption')
        same(c['ready_slot'],jobs[c['forecast_decision']]['ready_slot'],'consumption job identity')
        require(c['slot']>=c['ready_slot'],'no forecast consumed before actual completion')
        close(c['age_seconds'],c['slot']/100-c['forecast_decision'],'actual consumption age')
        expected=max((j for j in range(20) if jobs[j]['ready_slot']<=c['slot']),default=-1)
        same(c['forecast_decision'],expected,'latest completed FIFO forecast')
        previous_slot=c['slot']


def geometry_features(rx,ap,carrier,noise,tx):
    # Independent public segment/AABB formula for the declared33 rack boxes.
    boxes=np.asarray([((x0,y-.6,0),(x1,y+.6,3)) for y in (10.+6*k for k in range(11)) for x0,x1 in ((10.,38.),(42.,78.),(82.,110.))])
    delta=(rx-ap)[:,None,:];parallel=np.abs(delta)<1e-14
    lo=boxes[None,:,0]-ap[:,None,:];hi=boxes[None,:,1]-ap[:,None,:]
    outside=np.any(parallel&((lo>0)|(hi<0)),axis=-1);divisor=np.where(parallel,1.,delta)
    left=lo/divisor;right=hi/divisor
    entry=np.maximum(np.max(np.where(parallel,-np.inf,np.minimum(left,right)),axis=-1),0.)
    exit=np.minimum(np.min(np.where(parallel,np.inf,np.maximum(left,right)),axis=-1),1.)
    valid=(exit>entry+1e-12)&~outside
    fraction=np.where(valid,exit-entry,0.).sum(axis=1)
    distance=np.linalg.norm(rx-ap,axis=-1)
    return np.column_stack((rx/np.array((120.,80.,6.)),ap/np.array((120.,80.,6.)),np.log10(distance)/2,
        np.log(carrier/2.4e9),-np.ones(len(rx)),valid.sum(1)/8,np.clip(fraction,0,1),
        np.zeros((len(rx),3)),noise/100,tx/30)).astype('f4')


def check_context(a,d,world):
    cutoff=60+d;scanned=a['history_scanned'][:cutoff];heard=a['history_heard'][:cutoff]
    t,r=np.nonzero(scanned);groups=[];s=len(world['sites'])
    for site in range(s):
        for band in (0,1):
            order=np.lexsort((r,~heard[t,r,site*3+band],-t))
            groups.append([(int(t[i]),int(r[i]),site,band) for i in order[:64]])
    rows=[]
    for ordinal in range(64):
        for group in groups:
            if ordinal<len(group): rows.append(group[ordinal])
            if len(rows)==64:break
        if len(rows)==64:break
    indices=np.asarray(rows,dtype='i8');np.testing.assert_array_equal(a[f'job{d}_selected_rows'],indices)
    t,r,site,band=indices.T;link=site*3+band
    values=a['history_value'][t,r,link];upper=a['history_bound'][t,r,link];h=heard[t,r,link]
    xy=a['history_positions'][t,r];rx=np.column_stack((xy,np.ones(len(t))))
    ap=np.column_stack((world['sites'][site],np.full(len(t),4.)))
    carrier=np.asarray(CARRIERS)[band];noise=world['noise_db'][band];tx=world['tx_dl'][band]
    distance=np.linalg.norm(rx-ap,axis=-1)
    fspl=tx-20*(math.log10(4*math.pi/299792458.)+np.log10(distance)+np.log10(carrier))
    features=geometry_features(rx,ap,carrier,noise,tx)
    expected=np.column_stack((np.where(h,(values-fspl)/30,0),np.where(~h,(upper-fspl)/30,0),h,~h,features)).astype('f4')
    np.testing.assert_allclose(a[f'job{d}_context'],expected,rtol=0,atol=3e-6)
    np.testing.assert_array_equal(a[f'job{d}_known'],np.ones(len(rows),bool))
    np.testing.assert_array_equal(a[f'job{d}_scheduled'],np.ones(len(rows),bool))
    require(all(b in (0,1) for b in band),'no6.5input token')


def check_scan_payloads(raw,a,warm,spatial):
    for name in FIELDS:
        np.testing.assert_array_equal(a['history_'+name][:60],warm[name])
    starts=raw['scan_started_events'];payloads=raw['local_completed_payloads'];by_start={(e['robot'],e['slot']):e for e in starts}
    same(len(by_start),len(starts),'unique actual scan attempts')
    require(sum(e['attempt_ordinal']>=0 for e in starts)==len(starts),'nonnegative sensor attempt ordinals')
    same(len(starts),raw['telemetry']['scans_started'],'actual sensor scan count')
    payload_index={}
    for p in payloads:
        i,m,c,processed=p['robot'],p['measurement_slot'],p['complete_slot'],p['processed_slot']
        require(integer(i) and i<100 and 0<=m<c<=processed<2000,'causal completed local payload clocks')
        event=by_start.get((i,m));require(event is not None and event['measurement_available'],'payload has actual successful measurement attempt')
        same(c,event['complete_slot'],'sensor completion clock')
        h=np.asarray(p['heard'],dtype=bool);v=np.asarray(p['value_dbm'],dtype='f4')
        require(h.shape==v.shape==(a['history_value'].shape[-1],) and np.all(v[~h]==0),'heard/censor payload semantics')
        same(hashlib.sha256(v.tobytes()+h.tobytes()).hexdigest(),p['payload_sha256'],'actual local payload bytes')
        require(p['reference_id']=='native_zero_device0_assumed','explicit simulator gauge')
        require(sum(h)<=8 and math.isfinite(p['upper_dbm']),'top8/censor boundary')
        payload_index[(i,m)]=p
    latest={}
    for e in raw['telemetry']['events']:
        i,m,c,t=e['robot'],e['measurement_slot'],e['complete_slot'],e['delivery_slot']
        require(0<=m<c<=t<2000,'actual central report clocks')
        p=payload_index.get((i,m));require(p is not None,'delivered report must actually complete locally')
        same(c,p['complete_slot'],'same sensor completion');same(e['payload_sha256'],p['payload_sha256'],'same delivered payload')
        same(e['first_usable_decision'],t//100+1,'delivery-bin causal prefix')
        latest[(60+t//100,i)]=e
    scanned=a['history_scanned'];require(scanned.shape==(80,100),'history fullperiod axes')
    same(set(zip(*np.nonzero(scanned[60:]))),{(t-60,i) for t,i in latest},'all and only delivered postwarm history rows')
    for (t,i),e in latest.items():
        p=payload_index[(i,e['measurement_slot'])]
        np.testing.assert_array_equal(a['history_value'][t,i],np.asarray(p['value_dbm'],dtype='f4'))
        np.testing.assert_array_equal(a['history_heard'][t,i],np.asarray(p['heard'],dtype=bool))
        np.testing.assert_array_equal(a['history_bound'][t,i],np.full(a['history_bound'].shape[-1],p['upper_dbm'],dtype='f4'))
        np.testing.assert_array_equal(a['history_positions'][t,i],np.asarray(p['position_m'][:2],dtype=a['history_positions'].dtype))
    if spatial:
        same(raw['native']['spatial_execution']['local_completed_payloads'],payloads,'kernel same own payload ledger')
    return payload_index


def check_support(raw,a,world):
    records=raw['support'];preparations=raw['support_preparation']
    same([r['decision'] for r in preparations],list(range(20)),'one paid support build per issue')
    events=raw['telemetry']['events'];payloads={(p['robot'],p['measurement_slot']):p for p in raw['local_completed_payloads']}
    event_index={}
    for e in events:
        key=(60+e['delivery_slot']//100,e['robot'])
        if key not in event_index or event_index[key]['delivery_slot']<e['delivery_slot']:event_index[key]=e
    refresh_by_issue=defaultdict(list);refresh_consumed=defaultdict(set)
    for batch in raw.get('support_refresh',[]):
        d=batch['forecast_issue'];now=batch['consume_time_s'];k=int(round(now*100))
        expected=[e for e in events if d*100<=e['delivery_slot']<k and
                  (e['robot'],e['measurement_slot'],e['delivery_slot']) not in refresh_consumed[d]]
        same([b['delivery_event'] for b in batch['bindings']],expected,'all and only newly arrived server deliveries refreshed')
        same(batch['delivery_events_added'],len(expected),'refresh event count')
        same(batch['new_reports'],len(expected)*len(world['sites'])*2,'refresh source-band report count')
        require(not batch['locally_completed_undelivered_reports_used'] and not batch['encoder_context_changed'],'server guard cannot use undelivered RF or re-encode')
        for binding in batch['bindings']:
            e=binding['delivery_event'];i,m,c,t=e['robot'],e['measurement_slot'],e['complete_slot'],e['delivery_slot']
            p=payloads[(i,m)];same(binding['local_scan_id'],p['scan_id'],'exact delivered scan identity')
            same(binding['payload_processed_slot'],p['processed_slot'],'payload processed before server delivery')
            require(c<=p['processed_slot']<=t<k,'refreshed server observation actually available before admission')
            expected_rows=[]
            for target in range(len(world['sites'])):
                for band in (0,1):
                    link=target*3+band;heard=p['heard'][link]
                    expected_rows.append({'report_id':f"delivered:{i}:{m}:{c}:{t}:{target}:{band}:{p['payload_sha256']}",
                        'robot_id':i,'site_id':target,'band_index':band,'carrier_hz':CARRIERS[band],
                        'position_m':p['position_m'],'kind':'heard' if heard else 'censored',
                        'gauge':'native_zero_device0_assumed','available_at_s':t/100,
                        'measured_interval_s':[m/100,m/100],'arrival_interval_s':[t/100,t/100],
                        'value_dbm':p['value_dbm'][link] if heard else None,
                        'lower_dbm':None,'upper_dbm':None if heard else p['upper_dbm'],
                        'position_error_m':0.0,'clock_provenance':'admission_refresh_exact_server_delivery_and_completed_payload',
                        'source_row':[60+t//100,i,target,band]})
            same(binding['reports'],expected_rows,'payload-exact delivered refresh rows')
            refresh_by_issue[d].extend(expected_rows)
            refresh_consumed[d].add((i,m,t))
    for preparation in preparations:
        d=preparation['decision'];require(preparation['elapsed_seconds']<=raw['forecast_jobs'][d]['actual_prediction_seconds'],'support build charged in actual prediction')
        require(not preparation['encoder_context_changed'] and not preparation['packet_risk_qualified'],'guard distinct from representation/risk qualification')
    for row in records:
        d,i,target=row['decision'],row['robot'],row['target'];now=row['time_s'];age=now-d
        require(0<=d<20 and 0<=i<100 and 0<=target<len(world['sites']) and age>=0,'actual guard query identity')
        position=np.asarray([np.interp(age,HORIZONS,world['plan'][d,i,:,axis]) for axis in range(2)]+[1.])
        np.testing.assert_allclose(row['position_m'],position,rtol=0,atol=1e-10)
        reported={r['band_index']:r for r in row['own_latest_source_reports']}
        expected_bands=[]
        for band in (0,1):
            choices=[]
            for t in np.flatnonzero(a['history_scanned'][:60+d,i]):
                if t<60:
                    measured=[float(t-60),float(t-59)];arrival=None;available=0.;provenance='warm_measurement_bin; arrival_unknown_but_available_by_zero'
                else:
                    e=event_index.get((int(t),i));require(e is not None,'guard postwarm exactclock event')
                    if e['delivery_slot']>=d*100:continue
                    measured=[e['measurement_slot']/100]*2;arrival=[e['delivery_slot']/100]*2;available=arrival[0];provenance='payload_matched_native_scan_event_exact_slots'
                if d-measured[0]>6 or not 0<=now-measured[0]<=6:continue
                link=target*3+band;h=bool(a['history_heard'][t,i,link])
                payload=hashlib.sha256(a['history_value'][t,i].tobytes()+a['history_heard'][t,i].tobytes()).hexdigest()
                scan_digest=hashlib.sha256(bytes.fromhex(payload)+a['history_bound'][t,i].tobytes()+a['history_positions'][t,i].tobytes()).hexdigest()[:20]
                report={'report_id':f'native:{t}:{i}:{target}:{band}:{scan_digest}',
                    'robot_id':i,'site_id':target,'band_index':band,'carrier_hz':CARRIERS[band],
                    'position_m':[*a['history_positions'][t,i].tolist(),1.0],
                    'kind':'heard' if h else 'censored','gauge':'native_zero_device0_assumed',
                    'available_at_s':available,'measured_interval_s':measured,'arrival_interval_s':arrival,
                    'value_dbm':float(a['history_value'][t,i,link]) if h else None,
                    'upper_dbm':None if h else float(a['history_bound'][t,i,link]),
                    'lower_dbm':None,'position_error_m':0.0,'clock_provenance':provenance,
                    'source_row':[int(t),i,target,band]}
                choices.append(report)
            choices.extend(r for r in refresh_by_issue[d] if r['robot_id']==i and r['site_id']==target
                and r['band_index']==band and r['available_at_s']<now+1e-10
                and 0<=now-r['measured_interval_s'][0]<=6)
            def order(report):
                return (now-report['measured_interval_s'][0],float(np.linalg.norm(np.asarray(report['position_m'])-position)),report['report_id'])
            choices.sort(key=order)
            if not choices:
                require(band not in reported,'no invented own support');continue
            expected=choices[0];require(band in reported,'latest own evidence cannot disappear')
            actual=dict(reported[band]);distance=float(np.linalg.norm(np.asarray(expected['position_m'])-position))
            close(actual.pop('conservative_age_s'),now-expected['measured_interval_s'][0],'guard conservative age')
            close(actual.pop('query_distance_m'),distance,'guard distance')
            same(actual.pop('within_spatial_cut'),distance<=5,'registered spatial threshold')
            same(actual,expected,'latest own source report exact payload/clock/gauge/provenance')
            if distance<=5 and expected['kind']=='heard' and expected['value_dbm']>=-80:expected_bands.append(band)
        same(row['heard_support_bands'],expected_bands,'actual source-band support')
        same(row['permitted'],bool(expected_bands) and not row.get('late_discarded',False),'guard permission')
    return len(records)


def check_dispatch(raw,payloads,ho):
    x=raw['native']['spatial_execution'];runtime=raw['local_runtime']
    central={};executed=[];seen_local={};queued_local={};scans=x['scan_events'];lastscan=list(x['initial_scan_start_upper_slot'])
    # Reconstruct association and receiver-availability history from actual
    # completed radio actions and charged scan intervals, independently of
    # the controller's declared eligibility flags.
    moves=[]
    for e in x['central_events']:
        if e['event']=='executed':moves.append((e['slot'],e['robot'],e['current'],e['target'],e['outage_until_slot']))
    for e in x['local_events']:
        if e['event']=='executed':moves.append((e['slot'],e['robot'],e['from_site'],e['target'],e['outage_until_slot']))
    initial=list(x['final_current'])
    for k,i,old,target,until in sorted(moves,reverse=True):
        same(initial[i],target,'actual association trajectory joins');initial[i]=old
    def own_state(i,slot):
        ap=initial[i];available=True
        for k,robot,old,target,until in sorted(moves):
            if robot==i and k<slot:
                ap=target
                if slot<until:available=False
        for scan in scans:
            if scan['robot']==i and scan['interruption_start_slot']<=slot<scan['complete_slot']:available=False
        return ap,available
    for e in x['central_events']:
        cid=e['command_id'];i=e['robot']
        require(0<=i<100 and e['current']!=e['target'],'actual AP transition')
        same(e['created_slot'],e['proposed_slot']+10,'100ms paid central creation')
        same(e['expires_slot'],e['created_slot']+100,'creation-relative1sTTL')
        if e['event']=='queued':
            require(cid not in central,'unique central command identity');central[cid]=e
        else:
            require(cid in central and all(e[k]==central[cid][k] for k in central[cid] if k not in ('event','slot')),'immutable central command')
        if e['event']=='executed':
            require(e['created_slot']<=e['packet_created_slot']<=e['delivery_slot']==e['execution_slot']==e['slot']<e['expires_slot'] and e['delivery_slot']>e['created_slot'],'actual subsequently created/delivered packet required')
            same(e['interruption_slots'],math.ceil(ho/.01-1e-12),'full central handover interruption')
            same(e['outage_until_slot'],e['slot']+1+e['interruption_slots'],'central charged availability')
            executed.append((e['slot'],i,e['target']))
        elif e['event']=='cancelled' and e['reason']=='expired':same(e['slot'],e['expires_slot'],'expiry clock')
    decision_by_slot={e['slot']:e for e in x['central_decisions']}
    busy_until=0
    for d in x['central_decisions']:
        require(d['slot']>=busy_until,'no central job starts while previous actual CPU unfinished')
        busy_until=d['slot']+max(10,math.ceil(d['elapsed_seconds']/.01-1e-12))
        same(d['compute_ready_slot'],busy_until,'actual central CPU availability')
        same(d['overrun'],d['elapsed_seconds']>.1,'central actual budget')
    for skip in x.get('central_compute_events',[]):require(skip['slot']<skip['ready_slot'],'central consumed forecast discarded while actual CPU busy')
    for e in x['central_events']:
        if e['event']=='queued':
            d=decision_by_slot[e['proposed_slot']];require(d['elapsed_seconds']<=.1 and not d['overrun'],'overrun cannot create command')
            if raw['support']:
                require(any(r['decision']==d['forecast_decision'] and r['robot']==e['robot'] and r['target']==e['target'] and r['permitted'] and math.isclose(r['time_s'],e['proposed_slot']/100,abs_tol=1e-10) for r in raw['support']),'guarded centralcommand has actual support permission')
    for e in x['dispatch_events']:
        if e['event']=='not_renewed':same(e['expires_slot'],central[e['command_id']]['expires_slot'],'pending target cannot renew TTL')
    tick_by_action={}
    for tick in x['local_decisions']:
        for observation in tick.get('own_observations',[]):
            i=observation['robot'];slot=tick['slot'];current,available=own_state(i,slot)
            same(observation['current'],current,'own actual issue association')
            same(observation['radio_available'],available,'own actual issue receiver availability')
            captured=observation['pilot_slot'];require(integer(captured) and captured<=slot and captured%10==0,'actual lagged100mscapture clock')
            capture_site,capture_up=own_state(i,captured)
            require(capture_up and capture_site==observation['pilot_site'],'pilot capture requires receiver up on that AP at actual historical capture')
        expected=tick['slot']+max(1,math.ceil(tick['elapsed_seconds']/.01-1e-12))
        same(tick['ready_slot'],expected,'local actual CPU completion clock')
        same(tick['overrun'],tick['elapsed_seconds']>.005,'local tick budget')
        observations={r['robot']:r for r in tick.get('own_observations',[])}
        for r in tick['requests']:
            require(r['robot'] in observations,'own actual request observation')
            require(r['action_id'] not in tick_by_action,'unique action identity');tick_by_action[r['action_id']]=(tick,r,observations[r['robot']])
    for skip in x.get('local_compute_events',[]):
        require(skip['slot']<skip['ready_slot'],'busy tick cannot execute early')
    validations={}
    for batch in x.get('local_validation_batches',[]):
        close(batch['elapsed_seconds'],sum(r['elapsed_seconds'] for r in batch['checks']),'all validation CPU counted')
        same(batch['overrun'],batch['elapsed_seconds']>.005 or any(r['elapsed_seconds']>.005 for r in batch['checks']),'batch5mscap')
        for r in batch['checks']:validations[(r['action_id'],batch['slot'])]=(batch,r)
    runtime_actions={}
    if runtime:
        for row in runtime['events']:
            if row['kind']=='local_handover_started':runtime_actions[row['action']['action_id']]=row['action']
    for event in x['local_events']:
        aid=event['action_id']
        if event['event']=='queued':queued_local[aid]=event
        if event['event']!='executed':continue
        require(aid in queued_local and aid in tick_by_action and aid in runtime_actions,'local execution has actual request/authority lifecycle')
        tick,request,observation=tick_by_action[aid];action=runtime_actions[aid];k=event['slot'];i=event['robot']
        require(k>=tick['slot']+1 and not tick['overrun'],'paid local decision cannot execute same slot/overrun')
        batch,validation=validations[(aid,k)];require(not batch['overrun'] and validation['evidence_valid'],'actual delayed own evidence validation')
        same(action['created_slot'],tick['slot'],'actual request clock');require(action['created_slot']<k<action['valid_until_slot'],'actual local expiry')
        same(action['robot_id'],i,'own client action');same(action['current_ap'],event['from_site'],'actual current AP snapshot');same(action['target_ap'],event['target'],'actual target')
        require(event['last_delivery_slot']<tick['slot'] and tick['slot']-observation['last_delivery_slot']>=20,'own silence survives delayed execution')
        own=payloads.get((i,action['measurement_slot']));require(own is not None,'local target belongs to actual own completed scan')
        require(own['complete_slot']<=tick['slot'] and 0<=k-action['measurement_slot']<=500,'5s local measurement age and actual completion')
        band=CARRIERS.index(action['carrier_hz']);link=action['target_ap']*3+band
        require(own['heard'][link] and own['value_dbm'][link]>=-75,'real heard threshold')
        close(action['rssi_dbm'],own['value_dbm'][link],'selected actual local RSSI')
        same(action['scan_id'],own['scan_id'],'actual own scan identity')
        # Reconstruct latest per-link own scan cache at request time. The most
        # recent full scan replaces every heard/censored link together.
        available=[p for (robot,_),p in payloads.items() if robot==i and p['processed_slot']<=tick['slot']]
        newest=max(available,key=lambda p:(p['measurement_slot'],p['complete_slot'],p['scan_id'])) if available else None
        require(newest is not None and newest['scan_id']==own['scan_id'],'newer own negative evidence cannot resurrect old target')
        current_values=[]
        if observation['radio_available'] and observation['pilot_site']==observation['current'] and 0<=tick['slot']-observation['pilot_slot']<=10:
            current_values.extend((np.asarray(observation['pilot_snr'])+np.asarray(raw['_world_noise'])+np.asarray(raw['_world_tx'])-15).tolist())
        if 0<=tick['slot']-newest['measurement_slot']<=500:
            for b in range(3):
                l=action['current_ap']*3+b
                if newest['heard'][l]:current_values.append(newest['value_dbm'][l])
        current=max(current_values) if current_values else None
        close(action['current_rssi_dbm'],current,'actual current-pilot/cache margin basis',tolerance=3e-5)
        require(current is None or action['rssi_dbm']-current>=8-3e-5,'registered8dBlocalmargin')
        same(event['interruption_slots'],math.ceil(ho/.01-1e-12),'full local interruption')
        same(event['outage_until_slot'],k+1+event['interruption_slots'],'paid local radio unavailability')
        executed.append((k,i,event['target']));seen_local[aid]=event
    for e in scans:
        i=e['robot'];k=e['slot'];require(k-lastscan[i]>=500,'shared autonomous/background scan start budget')
        lastscan[i]=k;same(e['complete_slot'],k+10,'90ms scan plus slot boundary');same(e['interruption_start_slot'],k+1,'actual interruption start');same(e['interruption_slots'],9,'90ms scan cost')
        if e['reason']=='background_after_delivery':same(e['last_delivery_slot'],k,'background scan follows actual delivery')
        else:
            aid=e['action_id'];tick,request,observation=tick_by_action[aid]
            require(request['kind']=='scan' and k>=tick['slot']+1 and not tick['overrun'],'paid explicit autonomous scan request')
            require(e['last_delivery_slot']<tick['slot'],'intervening delivery invalidates watchdog scan')
            batch,validation=validations[(aid,k)];require(not batch['overrun'] and validation['evidence_valid'],'local scan execution validation')
    actual=[(r['slot'],i,t) for r in raw['actual_actions'] for i,t in zip(r['robots'],r['targets'])]
    same(sorted(actual),sorted(executed),'every actual move has independent central/local execution path')
    return {'central_commands':len(central),'central_executed':len(executed)-len(seen_local),'local_executed':len(seen_local),'scan_attempts':len(scans),'local_validation_batches':len(x.get('local_validation_batches',[]))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--study',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();root=args.root.resolve();out=(args.study or root/'artifacts/spatial_control').resolve()
    output=(args.output or root/'artifacts/spatial_control_review/result.json').resolve()
    require(not output.is_relative_to(out),'review output must remain outside sealed study')
    before_script=sha(Path(__file__));started=time.perf_counter()
    manifest,registration,provenance=verify_sources(root,out)
    before={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    protocol=read(out/'protocol.json');report=read(out/'report.json')
    same(protocol['seeds'],list(SEEDS),'registered fresh seed membership');same(protocol['arms'],list(ARMS),'registered arm membership')
    same(protocol['maximum_native_simulations'],50,'fixed simulation cap')
    expected_cases={c['id']:c for c in protocol['cases']};same(len(expected_cases),8,'four seeds x twoHOcases')
    expected_dirs={'disabled-parity'}|set(expected_cases)
    same({p.name for p in (out/'cases').iterdir() if p.is_dir()},expected_dirs,'no extra attempted case')
    worlds={};warm={}
    for seed in SEEDS:
        wreceipt=read(out/'worlds'/str(seed)/'receipt.json')
        same(wreceipt['seed'],seed,'world seed');same(wreceipt['source_sha256'],sha(out/'registration/source.json'),'world source')
        worlds[seed]=npz(ref(root,wreceipt['world']));warm[seed]=npz(ref(root,wreceipt['warm']))
        for field in ('world_array_sha256','warm_array_sha256'):
            if field in wreceipt:
                data=worlds[seed] if field.startswith('world') else warm[seed]
                same({k:array_hash(v) for k,v in data.items()},wreceipt[field],'physical/warm exact identity')
    rows=[];case_reviews=[];parity_raw=[]
    for case_id in ['disabled-parity',*expected_cases]:
        case=expected_cases.get(case_id,{'id':case_id,'seed':SEEDS[0],'handover_seconds':.03})
        parity=case_id=='disabled-parity';names=('radiomap__original','radiomap__disabled') if parity else ARMS
        directory=out/'cases'/case_id
        same({p.name for p in directory.iterdir() if p.is_dir()},set(names),'exact arm membership '+case_id)
        for arm in names:
            same({p.name for p in (directory/arm).iterdir()},
                {'attempt.json','receipt.json','raw.json.gz','predictions_history.npz'},'exact once-only arm evidence membership')
            receipt_path=directory/arm/'receipt.json';receipt=read(receipt_path)
            same(receipt['case'],case,'case exact specification');same(receipt['arm'],arm,'arm identity')
            same(receipt['source_sha256'],sha(out/'registration/source.json'),'case source');same(receipt['protocol_sha256'],sha(out/'protocol.json'),'case protocol')
            same(receipt['world_receipt_sha256'],sha(out/'worlds'/str(case['seed'])/'receipt.json'),'case world identity')
            same(receipt['simulations'],1,'one finite attempt per arm')
            raw=json.loads(gzip.decompress(ref(root,receipt['raw']).read_bytes()));a=npz(ref(root,receipt['arrays']));attempt=read(ref(root,receipt['attempt']))
            require(integer(attempt.pop('created_unix_ns',None),1),'durable attempt timestamp')
            for key,value in attempt.items():same(receipt[key],value,'durable attempt exact identity '+key)
            require(not receipt['production_qualified'] and not receipt['default_changed'] and not receipt['original_SERVICE_opened'],'no promotion/SERVICE opening')
            same(raw['arm'],arm.split('__')[0],'actual arm');same(raw['seed'],case['seed'],'actual seed');close(raw['handover_seconds'],case['handover_seconds'],'actualHO')
            require(raw['causal_central']==(not parity),'new causal semantics scope')
            same(raw['simulation_config'],{'seed':0,'duration_seconds':20,'collision_lambda':0.,
                'interference':False,'scan_duration_seconds':.09,'observation_period_slots':10,
                'native_proposal_ttl_slots':100,'planner':'native'},'exact native simulation configuration')
            close(raw['native']['duration_s'],20,'actual native duration')
            require(not raw['native']['interference'] and raw['native']['collision_lambda']==0,'registered collision/interference scope')
            require(raw['policy_config']['kind']=='native' and not raw['policy_config']['power_control']
                and not raw['policy_config']['dual_radio'] and raw['policy_config']['cadence_scan']
                and raw['policy_config']['cadence_roam'],'actual native causal single-radio policy')
            close(raw['policy_config']['roam_cost_s'],case['handover_seconds'],'actual handover policy')
            close(raw['policy_config']['scan_period_s'],5.,'actual scan cadence')
            recovery=arm in ('radiomap_recovery','regression_recovery','regression_both')
            require((raw['local_runtime'] is not None)==recovery,'actual declared local-recovery arm')
            learned=arm.startswith('regression');expected_keys={'forecast_mean_snr','forecast_std_db'}|{'history_'+n for n in FIELDS}
            if learned:expected_keys|={f'job{j}_{field}' for j in range(20) for field in JOB_FIELDS}
            same(set(a),expected_keys,'raw inert array exact membership')
            check_forecasts(raw,a,worlds[case['seed']],learned,parity)
            payloads=check_scan_payloads(raw,a,warm[case['seed']],not parity)
            raw['_metric_keys']=set(receipt['metrics'])
            derived=reducer(raw);numbers(receipt['metrics'],derived,'independent per-arm metrics')
            if parity:parity_raw.append(raw);continue
            raw['_world_noise']=worlds[case['seed']]['noise_db'];raw['_world_tx']=worlds[case['seed']]['tx_dl']
            review=check_dispatch(raw,payloads,case['handover_seconds'])
            if arm in ('regression_support','regression_both'):review['support_queries']=check_support(raw,a,worlds[case['seed']])
            else:require(not raw['support'] and not raw['support_preparation'],'no undeclared support information')
            rows.append({'case':case_id,'seed':case['seed'],'handover_seconds':case['handover_seconds'],'arm':arm,'metrics':derived})
            case_reviews.append({'case':case_id,'arm':arm,**review})
    same(parity_raw[0]['native'],parity_raw[1]['native'],'disabled extension complete native outputs exact')
    same([j['forecast_sha256'] for j in parity_raw[0]['forecast_jobs']],[j['forecast_sha256'] for j in parity_raw[1]['forecast_jobs']],'disabled exact forecasts')
    same([j['ready_slot'] for j in parity_raw[0]['forecast_jobs']],[j['ready_slot'] for j in parity_raw[1]['forecast_jobs']],'disabled completion parity')
    same(rows,report['rows'],'complete official per-case metric membership')
    means=[];contrasts=[]
    for ho in (.03,4.):
        for arm in ARMS:
            cells=[r['metrics'] for r in rows if r['handover_seconds']==ho and r['arm']==arm]
            means.append({'handover_seconds':ho,'arm':arm,'episodes':4,'mean':{k:float(np.mean([c[k] for c in cells])) for k in cells[0]}})
        for candidate,baseline in protocol['analysis']['contrasts']:
            pairs=[]
            for seed in SEEDS:
                aa=next(r['metrics'] for r in rows if r['seed']==seed and r['handover_seconds']==ho and r['arm']==candidate)
                bb=next(r['metrics'] for r in rows if r['seed']==seed and r['handover_seconds']==ho and r['arm']==baseline)
                pairs.append({k:aa[k]-bb[k] for k in aa})
            contrasts.append({'handover_seconds':ho,'candidate':candidate,'baseline':baseline,
                'mean_delta':{k:float(np.mean([c[k] for c in pairs])) for k in pairs[0]},
                'episodes_with_more_misses':sum(c['control_misses']>0 for c in pairs),
                'episodes_with_larger_max_gap':sum(c['maximum_silence_ms']>0 for c in pairs)})
    same(means,report['arm_means'],'independent arm means');same(contrasts,report['contrasts'],'independent paired contrasts/harm counts')
    after={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    same(after,before,'all study bytes/membership unchanged during review');same(sha(Path(__file__)),before_script,'auditor source unchanged')
    result={'schema':1,'passes':True,'scientific_calls':{'model_constructors':0,'fits':0,'forwards':0,'RT_solves':0,'packet_simulations':0},
        'audit_source_sha256':before_script,'study_manifest_sha256':sha(out/'verification/manifest.json'),
        'provenance':provenance,'fresh_simulations':48,'disabled_parity_simulations':2,
        'total_control_releases_fresh':48*40000,'independent_raw_reductions':case_reviews,
        'arm_means':means,'contrasts':contrasts,'source_evidence_unchanged':True,
        'SERVICE_opened':False,'production_promotion':False,'seconds':time.perf_counter()-started}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:stream.write((json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps({'passes':True,'fresh_simulations':48,'parity_simulations':2,'output':str(output)},sort_keys=True))


if __name__=='__main__':main()
