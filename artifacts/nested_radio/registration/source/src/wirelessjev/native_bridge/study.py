"""Separate registered native100 DEV bridge, never a SERVICE promotion study.

CLI: write-spec, preflight, register, publish, run, audit. Publish binds the
source commit after a user-authorized git push; it does not itself push.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np
import torch

from ..adaptive_radio import common as ac
from ..adaptive_radio.service import ServiceProbeConfig,DevelopmentRegistration,run_development_pair
from . import predictor as p

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/native_bridge'
PROTOCOL=OUT/'protocol.json'
SEEDS=(90473001,90473002,90473003,90473004)
WORLD_ARRAYS=('sites','snr','rssi_dl','domain','avail','video_mbps','bulk_mbps','noise_db','channel','plan','pos','tx_dl')
OBSERVATIONS=('positions','value','heard','bound','scanned')
_PUBLICATION=set()


def digest(path):return ac.digest(path)
def encoded(value):return ac.encoded(value)
def read(path):return ac.read_json(path)
def write(path,value):return ac.write_once(path,value)


def protocol_spec():
    return {'schema':1,'purpose':'finite_unqualified_native_100robot_DEV_extrapolation',
        'seeds':list(SEEDS),'handover_seconds':[.03,4.], 'robots':100,'duration_seconds':12,
        'warm_history_seconds':60,'forecast_horizons_seconds':list(p.HORIZONS),
        'cases':[{'case_id':f'dev-{seed}-ho{cost:g}','seed':seed,'handover_seconds':cost}
                 for seed in SEEDS for cost in (.03,4.)],
        'parity_case':{'case_id':f'parity-{SEEDS[0]}-ho0.03','seed':SEEDS[0],'handover_seconds':.03},
        'candidate':{'rule':'seed0_member_of_TRAIN_VAL_selected_joint_or_regression_branch_after_all12fits',
            'native_outcomes_used_for_selection':False,'checkpoint_bound_after_fits':True,'ensemble':False},
        'capture':{'source_bands':[0,1],'maximum_tokens':64,'selection':'roundrobin_AP_band_latest_time_heard_tie_robot',
            'same_delivered_history_capacity_as_Map':True,'Map_uses_full_60bin_history':True,
            'real_continuous_native_reports_quantization_db':0.,'warm_scan_cadence_seconds':5},
        'geometry':{'hall_xyz_m':[120.,80.,6.],'receiver_height_m':1.,'AP_height_m':4.,
            'rack_height_m':3.,'rack_depth_m':1.2,'boxes':33,
            'scope':'declared_2D_native_CAD_extrusion_not_RT_geometry_truth'},
        'actual_carriers_hz':list(p.NATIVE_CARRIERS_HZ),'six_point_five_GHz':'new_extrapolative_DEV_not_the_RT6point0_endpoint',
        'config':{'forecast_profile_seconds':{'cold':1.,'warm':.25,'refit':1.},
            'shared_extra_seconds':.5,'horizon_seconds':10.,'max_candidates':64,
            'planner_deadline_seconds':.1,'scan_duration_seconds':.09,'scan_period_seconds':5.,
            'collision_lambda':0.,'interference':False},
        'parity_profile_seconds':{'cold':4.,'warm':4.,'refit':4.},
        'parity':'same_seed_cost_native_Map_SNR_to_RSSI_to_bridge; independent_of_learned_contrast',
        'kernel':'frozen_service_world.sim via adaptive_radio.service.run_development_pair',
        'controls':'same_typed_surrogate_planner_existing_gate_exact_admission_FIFO_TTL_native_packet_execution',
        'metrics':['control_miss','control_miss_worst_decile','control_gap_p99_ms','maximum_silence_ms',
                   'video_on_time','bulk_mbps','airtime_domain_sum','roams','safety_stop_fraction',
                   'paid_forecast_compute_seconds','forecast_service_seconds','planner_and_admission_seconds'],
        'analysis':'report_all4episodes_separately_and_paired_mean_harm_counts_per_handover; no_cross_cost_SLA_or_small_n_significance_claim',
        'calibration':'none; no_packet_or_silence_risk_qualification; RSSI_mean_std_only',
        'forbidden':['original24SERVICE_confirmation','native_result_tuning','production_promotion','hidden_RF_as_model_input'],
        'maximum_native_simulations':18,'production_qualified':False,'default_changed':False}


def write_spec():
    value=protocol_spec();write(PROTOCOL,value);return value


def runtime_metadata():
    cpu=platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                  if line.startswith('model name') and ':' in line),cpu)
    return {'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,
        'platform':platform.platform(),'machine':platform.machine(),'CPU_model':cpu,
        'logical_CPUs':os.cpu_count(),'torch_threads':torch.get_num_threads(),
        'torch_interop_threads':torch.get_num_interop_threads(),'model_device':'cpu',
        'timing_is_cloud_CPU_not_SmartCar_hardware':True}


def source_hashes():
    upstream=ac.check_sources(require_published=False)
    result=dict(upstream['source_sha256'])
    files=[PROTOCOL,ac.OUT/'registration/source.json',ac.OUT/'registration/git.json']
    files+=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'tests').glob('test_native_bridge*.py'))
    files+=list((ROOT/'docs/native_bridge').glob('*.md'))
    for path in files:result[str(path.relative_to(ROOT))]=digest(path)
    return dict(sorted(result.items()))


def verify_dependencies():
    ac.check_sources();proof=ac.verify_previous()
    if (read(PROTOCOL)!=protocol_spec() or (ROOT/'artifacts/radiojepa/rt_data/confirmation').exists()
            or (ROOT/'artifacts/radiojepa/rt_data/confirmation.json').exists()):
        raise RuntimeError('Native bridge protocol or original SERVICE closure changed')
    return {'adaptive_source_sha256':digest(ac.OUT/'registration/source.json'),
            'adaptive_publication_sha256':digest(ac.OUT/'registration/git.json'),'previous':proof}


def preflight():
    write_spec();before=source_hashes();prior=verify_dependencies()
    names=sorted(str(x.relative_to(ROOT)) for x in (ROOT/'tests').glob('test_native_bridge*.py'))
    result=subprocess.run([sys.executable,'-m','pytest','-q',*names],cwd=ROOT,capture_output=True,text=True)
    log=OUT/'preflight/tests.log';ac.write_bytes_once(log,(result.stdout+result.stderr).encode())
    if source_hashes()!=before:raise RuntimeError('Native bridge sources changed during tests')
    receipt={'exit_code':result.returncode,'source_sha256':before,'dependencies':prior,'runtime':runtime_metadata(),
             'stdout_path':str(log.relative_to(ROOT)),'stdout_sha256':digest(log)}
    write(OUT/'preflight/tests.json',receipt)
    if result.returncode:raise RuntimeError('Native bridge preflight tests failed')
    return receipt


def register():
    write_spec();target=OUT/'registration/source.json'
    if target.exists():return check_sources(require_published=False)
    dependencies=verify_dependencies()
    for name in ('worlds','cases','weights.json','summary.json'):
        path=OUT/name
        if path.is_file() or (path.is_dir() and any(path.rglob('*'))):raise RuntimeError('Register before any native worlds/models/cases')
    sources=source_hashes();tests=read(OUT/'preflight/tests.json')
    if (tests['exit_code']!=0 or tests['source_sha256']!=sources or tests['dependencies']!=dependencies
            or digest(ac.safe_path(tests['stdout_path']))!=tests['stdout_sha256']):
        raise RuntimeError('Passing preflight must bind exactly this source and dependencies')
    capsule=OUT/'registration/source'
    for name,expected in sources.items():
        path=ac.safe_path(name,capsule);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if digest(path)!=expected:raise RuntimeError('Partial native source capsule changed')
        else:shutil.copyfile(ac.safe_path(name),path)
    receipt={'schema':1,'source_sha256':sources,'dependencies':dependencies,
        'protocol_sha256':digest(PROTOCOL),'preflight_sha256':{
            str((OUT/'preflight/tests.json').relative_to(ROOT)):digest(OUT/'preflight/tests.json'),
            tests['stdout_path']:tests['stdout_sha256']},
        'new_worlds_fits_or_packet_cases_before_registration':False,'software_fixtures_only_before_registration':True}
    write(target,receipt);return receipt


def _archive_members(receipt):
    members=dict(receipt['source_sha256']);members.update(receipt['preflight_sha256'])
    members[str((OUT/'registration/source.json').relative_to(ROOT))]=digest(OUT/'registration/source.json')
    members.update({str((OUT/'registration/source').relative_to(ROOT))+'/'+name:h
                    for name,h in receipt['source_sha256'].items()})
    return members


def _check_commit(commit,receipt):
    names=_archive_members(receipt)
    proc=subprocess.Popen(['git','archive','--format=tar',commit,*names],cwd=ROOT,stdout=subprocess.PIPE)
    actual=set()
    try:
        with tarfile.open(fileobj=proc.stdout,mode='r|') as archive:
            for member in archive:
                if member.isdir():continue
                if not member.isfile() or member.name not in names:raise RuntimeError('Unexpected native published member')
                if hashlib.sha256(archive.extractfile(member).read()).hexdigest()!=names[member.name]:
                    raise RuntimeError('Native source commit bytes changed')
                actual.add(member.name)
        if proc.wait() or actual!=set(names):raise RuntimeError('Native source commit is incomplete')
    finally:
        proc.stdout.close()
        if proc.poll() is None:proc.terminate()
        proc.wait()


def record_publication(commit,remote,branch):
    receipt=check_sources(require_published=False)
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact published commit required')
    _check_commit(commit,receipt)
    remote_head=subprocess.check_output(['git','ls-remote',remote,'refs/heads/'+branch],cwd=ROOT,text=True).split()
    if not remote_head or remote_head[0]!=commit:raise RuntimeError('Native source commit is not the observed remote branch head')
    value={'commit':commit,'remote':remote,'branch':branch,'remote_observed_commit':commit,
           'source_receipt_sha256':digest(OUT/'registration/source.json')}
    write(OUT/'registration/git.json',value);return value


def check_sources(*,require_published=True):
    ac.check_sources();receipt=read(OUT/'registration/source.json')
    if receipt['source_sha256']!=source_hashes() or receipt['protocol_sha256']!=digest(PROTOCOL):
        raise RuntimeError('Native bridge registered source/protocol changed')
    for name,expected in receipt['preflight_sha256'].items():
        if digest(ac.safe_path(name))!=expected:raise RuntimeError('Native bridge preflight proof changed')
    capsule=OUT/'registration/source'
    if {str(x.relative_to(capsule)) for x in capsule.rglob('*') if x.is_file()}!=set(receipt['source_sha256']):
        raise RuntimeError('Native source capsule membership changed')
    for name,expected in receipt['source_sha256'].items():
        if digest(ac.safe_path(name,capsule))!=expected:raise RuntimeError('Native source capsule bytes changed')
    if require_published:
        publication=read(OUT/'registration/git.json');key=digest(OUT/'registration/git.json')
        if (publication['source_receipt_sha256']!=digest(OUT/'registration/source.json')
                or publication['commit']!=publication['remote_observed_commit']):raise RuntimeError('Native source publication identity changed')
        if key not in _PUBLICATION:_check_commit(publication['commit'],receipt);_PUBLICATION.add(key)
    return receipt


def bind_weights():
    check_sources()
    from ..adaptive_radio import training,study
    from ..adaptive_radio.model import CELLS
    identity=training.identity();fit_refs={};checkpoint_refs={}
    for cell in CELLS:
        for seed in training.SEEDS:
            path=ac.OUT/'fits'/f'{cell}-seed{seed}.json';fit=read(path)
            training.verify_completed(fit,identity,cell,seed)
            fit_refs[str(path.relative_to(ROOT))]=digest(path)
            cp=ac.OUT/'checkpoints'/f'{cell}-seed{seed}.pt';checkpoint_refs[str(cp.relative_to(ROOT))]=digest(cp)
    selected=study.verified_selection();cell=selected['selected_cell'];checkpoint=ac.OUT/'checkpoints'/f'{cell}-seed0.pt'
    receipt={'selected_cell':cell,'seed':0,'checkpoint_path':str(checkpoint.relative_to(ROOT)),
        'checkpoint_sha256':digest(checkpoint),'selection_sha256':digest(ac.OUT/'selection.json'),
        'all12_fit_receipts_sha256':fit_refs,'all12_checkpoints_sha256':checkpoint_refs,
        'native_DEV_selection_used':False,'source_receipt_sha256':digest(OUT/'registration/source.json')}
    write(OUT/'weights.json',receipt);return receipt


def _npz(path,arrays):
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);ac.write_bytes_once(path,stream.getvalue())
    return {'path':str(path.relative_to(ROOT)),'sha256':digest(path)}


def _arrays(path):
    with np.load(path,allow_pickle=False) as saved:return {name:saved[name].copy() for name in saved.files}


def _registered_case(spec,parity):
    if type(parity) is not bool or not isinstance(spec,dict):raise ValueError('Typed registered case/arm required')
    expected=protocol_spec()['parity_case'] if parity else next(
        (s for s in protocol_spec()['cases'] if s['case_id']==spec.get('case_id')),None)
    if encoded(expected)!=encoded(spec):raise ValueError('Unregistered native development case')
    return expected


def config_for(spec,parity):
    _registered_case(spec,parity)
    values=protocol_spec()['config'].copy()
    if parity:values['forecast_profile_seconds']=protocol_spec()['parity_profile_seconds']
    return ServiceProbeConfig(12,handover_seconds=spec['handover_seconds'],**values)


def _policy_for(spec,parity):
    from ..service_world.sim import Policy
    config=config_for(spec,parity)
    return Policy('adaptive_service_development','native',roam_cost_s=config.handover_seconds,
        scan_period_s=config.scan_period_seconds,gate=True,cadence_scan=True,cadence_roam=True,
        online_map=False,dual_radio=False).__dict__


def _attempt_base(spec,parity):
    _registered_case(spec,parity)
    return {'schema':1,'case':spec,'parity':parity,'maximum_native_simulations':2,
        'source_receipt_sha256':digest(OUT/'registration/source.json'),'protocol_sha256':digest(PROTOCOL),
        'config':config_for(spec,parity).__dict__,'exclusive_attempt':True,
        'unreceipted_attempt_must_not_be_reexecuted':True}


def _begin_attempt(spec,parity):
    """Exclusive durable intent BEFORE world creation, model load or execution.

    Any interrupted/partial directory is retained and refused, even if no raw
    output was written. Neither run() nor a direct case call may silently
    spend another two simulations on the same registered case.
    """
    directory=OUT/'cases'/spec['case_id']
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError('Unreceipted or partial native case exists; refuse reexecution of its finite attempt')
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'attempt.json';value=_attempt_base(spec,parity)|{'created_unix_ns':time.time_ns()}
    try:
        with path.open('xb') as stream:
            stream.write(encoded(value));stream.flush();os.fsync(stream.fileno())
    except FileExistsError as exc:raise RuntimeError('Another native attempt already owns this case') from exc
    descriptor=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    return {'path':str(path.relative_to(ROOT)),'sha256':digest(path)}


def _reference(item,path):
    expected=str(path.relative_to(ROOT))
    if not isinstance(item,dict) or set(item)!=set(('path','sha256')) or item['path']!=expected:
        raise RuntimeError('Native archive/reference must use its canonical registered case path')
    if digest(ac.safe_path(item['path']))!=item['sha256']:raise RuntimeError('Native canonical evidence bytes changed')


def _validate_receipt_metadata(receipt,spec,parity,identity):
    _registered_case(spec,parity)
    expected={'case':spec,'parity':parity,'source_receipt_sha256':identity['source_receipt_sha256'],
        'protocol_sha256':identity['protocol_sha256'],'weights_sha256':identity['weights_sha256'],
        'world_receipt_sha256':identity['world_receipt_sha256'],'native_simulations':2,
        'simulation_only':True,'production_qualified':False,'original_SERVICE_holdouts_opened':False,
        'geometry_carrier_scale_transfer_unqualified':True}
    if any(encoded(receipt.get(name))!=encoded(value) for name,value in expected.items()):
        raise RuntimeError('Native receipt differs from its exact registered case/source/config/world identity')
    directory=OUT/'cases'/spec['case_id']
    for field,filename in (('raw_result','native_result.json.gz'),('candidate_inputs_forecasts','candidate_inputs_forecasts.npz'),
                           ('attempt','attempt.json')):
        item=receipt.get(field)
        if not isinstance(item,dict) or item.get('path')!=str((directory/filename).relative_to(ROOT)):
            raise RuntimeError('Native archive/reference must use its canonical registered case path')


def _validate_raw_metadata(result,spec,parity,identity):
    config=config_for(spec,parity)
    expected={'case_id':spec['case_id'],'partition':'development','robots':100,'native_packet_simulations':2,
        'source_receipt_sha256':identity['source_receipt_sha256'],'protocol_sha256':identity['protocol_sha256'],
        'world_receipt_sha256':identity['world_receipt_sha256'],'simulation_only':True,
        'production_qualified':False,'original_SERVICE_holdouts_opened':False,'default_controller_changed':False,
        'equal_allowed_capture_and_compute_budgets':True,'equal_measured_runtime_claimed':False,
        'config':config.__dict__,'policy':_policy_for(spec,parity)}
    if any(encoded(result.get(name))!=encoded(value) for name,value in expected.items()):
        raise RuntimeError('Raw native case/config/HO/profile/source/protocol/world differs from registration')
    if set(result.get('arms',{}))!={'radiomap','candidate'}:raise RuntimeError('Native arm membership changed')
    for arm in result['arms'].values():
        native=arm['native']
        for name,value in {'scene':spec['seed'],'policy':'adaptive_service_development',
                           'collision_lambda':0.,'interference':False}.items():
            if encoded(native.get(name))!=encoded(value):raise RuntimeError('Native physical episode/traffic configuration changed')
        duration=native.get('duration_s')
        if (isinstance(duration,bool) or not isinstance(duration,(int,float))
                or not math.isfinite(duration) or float(duration)!=float(config.duration_seconds)):
            raise RuntimeError('Native physical episode duration changed')
        jobs=arm['forecast_jobs']
        if len(jobs)!=12 or [j['kind'] for j in jobs]!=['cold']+['warm']*11:
            raise RuntimeError('Native case forecast job/timing-regime membership changed')
        for decision,job in enumerate(jobs):
            if (job['decision']!=decision or job['history_cutoff_exclusive']!=60+decision
                    or job['profile_seconds']!=config.forecast_profile_seconds[job['kind']]
                    or job['shared_extra_seconds']!=config.shared_extra_seconds):
                raise RuntimeError('Native case paid forecast profile differs from registration')


def _load_verified_case(spec,parity):
    directory=OUT/'cases'/spec['case_id'];receipt=read(directory/'receipt.json')
    world_path=OUT/'worlds'/str(spec['seed'])/'receipt.json'
    identity={'source_receipt_sha256':digest(OUT/'registration/source.json'),'protocol_sha256':digest(PROTOCOL),
              'weights_sha256':digest(OUT/'weights.json'),'world_receipt_sha256':digest(world_path)}
    _validate_receipt_metadata(receipt,spec,parity,identity)
    for field,filename in (('raw_result','native_result.json.gz'),('candidate_inputs_forecasts','candidate_inputs_forecasts.npz'),
                           ('attempt','attempt.json')):_reference(receipt[field],directory/filename)
    attempt=read(directory/'attempt.json');created=attempt.pop('created_unix_ns',None)
    if type(created) is not int or created<=0 or encoded(attempt)!=encoded(_attempt_base(spec,parity)):
        raise RuntimeError('Native durable attempt identity changed')
    result=json.loads(gzip.decompress((directory/'native_result.json.gz').read_bytes()))
    _validate_raw_metadata(result,spec,parity,identity)
    metrics={name:_native_metrics(arm) for name,arm in result['arms'].items()}
    delta={name:metrics['candidate'][name]-metrics['radiomap'][name] for name in metrics['candidate']}
    if (metrics!=receipt['metrics'] or delta!=receipt['paired_delta_candidate_minus_Map']
            or len(receipt['candidate_jobs'])!=12):raise RuntimeError('Native case metrics/deltas/job membership changed')
    _audit_inputs(receipt,result)
    return receipt,result


def prepare_world(seed):
    check_sources()
    if seed not in SEEDS:raise ValueError('Unregistered native development seed')
    from ..native.world import World
    from ..fleetwm.data import Geometry,Observations,episode
    from ..contracted.data import runtime_world
    from ..twin.radio import AP
    directory=OUT/'worlds'/str(seed);receipt_path=directory/'receipt.json'
    if not receipt_path.exists():
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError('Partial unreceipted native world exists; refuse regeneration')
        full,_,g,observed=episode(seed,100,12,'main')
        warm=observed.before(60)
        allowed=(np.arange(60)%5==0)[:,None]
        h=warm.heard&allowed[...,None];scanned=warm.scanned&allowed
        warm=Observations(warm.positions,np.where(h,warm.value,0).astype('f4'),h,warm.bound,scanned)
        empty=np.zeros((12,100,len(p.HORIZONS),len(g.sites),3),dtype='f4')
        world=runtime_world(full,12,empty,empty[:,:,0])
        truth=_npz(directory/'world.npz',{name:np.asarray(getattr(world,name)) for name in WORLD_ARRAYS})
        history=_npz(directory/'warm.npz',{name:np.asarray(getattr(warm,name)) for name in OBSERVATIONS})
        receipt={'seed':seed,'robots':100,'duration_seconds':12,'warm_seconds':60,'world':truth,'warm':history,
            'protocol_sha256':digest(PROTOCOL),
            'world_array_sha256':{name:p.array_digest(getattr(world,name)) for name in WORLD_ARRAYS},
            'warm_array_sha256':{name:p.array_digest(getattr(warm,name)) for name in OBSERVATIONS},
            'public_APs':[asdict(a) for a in g.aps],'source_receipt_sha256':digest(OUT/'registration/source.json'),
            'physical_truth_archived_for_audit_not_model_input':True}
        write(receipt_path,receipt)
    receipt=read(receipt_path)
    if (receipt['seed']!=seed or receipt['source_receipt_sha256']!=digest(OUT/'registration/source.json')
            or receipt['protocol_sha256']!=digest(PROTOCOL)):
        raise RuntimeError('Native world provenance changed')
    _reference(receipt['world'],directory/'world.npz');_reference(receipt['warm'],directory/'warm.npz')
    for item in (receipt['world'],receipt['warm']):
        if digest(ac.safe_path(item['path']))!=item['sha256']:raise RuntimeError('Native world/report bytes changed')
    a=_arrays(ac.safe_path(receipt['world']['path']));o=_arrays(ac.safe_path(receipt['warm']['path']))
    if {name:p.array_digest(value) for name,value in a.items()}!=receipt['world_array_sha256']:
        raise RuntimeError('Native world array identities changed')
    if {name:p.array_digest(value) for name,value in o.items()}!=receipt['warm_array_sha256']:
        raise RuntimeError('Native warm report array identities changed')
    empty=np.zeros((12,100,len(p.HORIZONS),len(a['sites']),3),dtype='f4')
    world=World(seed,100,a['sites'],a['snr'],a['rssi_dl'],empty,a['domain'],a['avail'],a['video_mbps'],
        a['bulk_mbps'],12.,0.,False,empty[:,:,0],a['noise_db'],None,a['channel'],None,a['plan'],a['pos'],a['tx_dl'])
    aps=tuple(AP(**(item|{'allowed_channels':tuple(item['allowed_channels'])})) for item in receipt['public_APs'])
    g=Geometry(a['sites'].copy(),a['tx_dl'].copy(),a['noise_db'].copy(),aps,False)
    warm=Observations(*(o[name] for name in OBSERVATIONS))
    return world,warm,g,receipt


def _native_metrics(arm):
    native=arm['native'];detail=native['packet_detail'];jobs=arm['forecast_jobs'];planner=arm['planner_and_admission']
    keys=('control_miss','control_miss_worst_decile','control_gap_p99_ms','video_on_time','bulk_mbps',
          'airtime_domain_sum','roams','safety_stop_fraction','control_misses','control_sent')
    misses=np.asarray(detail['misses'],dtype=float);sent=np.asarray(detail['sent'],dtype=float)
    frac=misses/np.maximum(sent,1);worst=np.sort(frac)[-max(1,len(frac)//10):]
    derived={'control_miss':float(misses.sum()/max(1,sent.sum())),
             'control_miss_worst_decile':float(worst.mean()),
             'control_gap_p99_ms':float(np.percentile(detail['max_gap'],99)*10)}
    if any(not math.isclose(native[name],value,rel_tol=0,abs_tol=1e-9) for name,value in derived.items()):
        raise RuntimeError('Native summary differs from raw per-robot packet/gap counts')
    return {name:native[name] for name in keys}|derived|{'maximum_silence_ms':10*max(detail['max_gap']),
        'paid_forecast_compute_seconds':float(sum(j['charged_compute_seconds'] for j in jobs)),
        'forecast_service_seconds':float(sum(j['charged_seconds'] for j in jobs)),
        'actual_forecast_compute_seconds':float(sum(j['actual_compute_seconds'] for j in jobs)),
        'planner_and_admission_seconds':float(sum(j['elapsed_seconds'] for j in planner)),
        'forecast_jobs':len(jobs),'forecast_jobs_consumed':len(arm['forecast_consumption']),
        'forecast_runtime_overruns':sum(j['runtime_overrun'] for j in jobs),
        'late_planner_or_admission_discards':sum(j['late_discarded'] for j in planner)}


def _archive_predictor(raw,directory):
    arrays={};metadata=[]
    for i,record in enumerate(raw.records):
        metadata.append({'history_length':record['history_length']})
        for name,value in record.items():
            if isinstance(value,np.ndarray):arrays[f'job{i}_{name}']=value
    for name,value in (raw.last_history or {}).items():arrays['last_history_'+name]=value
    return _npz(directory/'candidate_inputs_forecasts.npz',arrays),metadata


def _parity(result,raw):
    a=result['arms']['radiomap'];b=result['arms']['candidate']
    # Exact bitwise native-SNR queue hashes are stronger than a plotted curve.
    same_forecasts=[j['forecast_sha256'] for j in a['forecast_jobs']]==[j['forecast_sha256'] for j in b['forecast_jobs']]
    same_std=[j['forecast_std_sha256'] for j in a['forecast_jobs']]==[j['forecast_std_sha256'] for j in b['forecast_jobs']]
    same_ready=[j['ready_slot'] for j in a['forecast_jobs']]==[j['ready_slot'] for j in b['forecast_jobs']]
    # Native bodies contain physical/service counters and packet traces, but
    # no wall-clock timings. Compare every value, not rounded miss summaries.
    same_packets=encoded(a['native'])==encoded(b['native'])
    roundtrip=all(j['forecast_sha256']==hashlib.sha256(record['source_snr'].tobytes()).hexdigest()
                  for j,record in zip(b['forecast_jobs'],raw.records))
    checks={'all_original_Map_forecasts_bitwise_equal':same_forecasts,'Map_std_bitwise_equal':same_std,
        'completion_slots_equal':same_ready,'native_packet_and_service_counters_equal':same_packets,
        'candidate_bridge_roundtrip_matches_its_actual_Map_source':roundtrip,
        'original_Map_forecasts_count_equal':len(a['forecast_jobs'])==len(raw.records)==12}
    return {'passes':all(checks.values()),'checks':checks,'learned_model_performance_result':False,
            'timing_overruns_can_invalidate_paired_wiring_equality':True}


def run_case(spec,*,parity=False):
    check_sources()  # Before model loads, world generation or packet execution.
    _registered_case(spec,parity)
    directory=OUT/'cases'/spec['case_id'];path=directory/'receipt.json'
    if path.exists():
        bind_weights()
        return _load_verified_case(spec,parity)[0]
    attempt=_begin_attempt(spec,parity)
    weights=bind_weights()
    world,warm,g,world_receipt=prepare_world(spec['seed'])
    if parity:raw=p.MapRoundTripPredictor(g,digest(OUT/'registration/source.json'))
    else:
        from ..adaptive_radio import training
        model=training.load(weights['selected_cell'],0)
        raw=p.AdaptiveNativePredictor(g,model,weights['checkpoint_sha256'])
    predictor=p.bridged(raw)
    config=config_for(spec,parity)
    world_path=OUT/'worlds'/str(spec['seed'])/'receipt.json'
    registration=DevelopmentRegistration(spec['case_id'],spec['seed'],digest(OUT/'registration/source.json'),
        digest(PROTOCOL),digest(world_path))
    def guard(item):
        check_sources()
        if item!=registration or digest(world_path)!=item.world_receipt_sha256:raise RuntimeError('Native case/world registration changed')
        if ({name:p.array_digest(getattr(world,name)) for name in WORLD_ARRAYS}!=world_receipt['world_array_sha256']
                or {name:p.array_digest(getattr(warm,name)) for name in OBSERVATIONS}!=world_receipt['warm_array_sha256']):
            raise RuntimeError('The actual native physical/report inputs differ from registered bytes')
    result=run_development_pair(world,warm,g,predictor,config=config,registration=registration,publication_guard=guard,
        candidate_model=predictor.adapter,candidate_source_id=raw.source_id if not parity else 'radiomap_roundtrip')
    inputs,jobs=_archive_predictor(raw,directory)
    raw_path=directory/'native_result.json.gz'
    ac.write_bytes_once(raw_path,gzip.compress(encoded(result),mtime=0))
    metrics={name:_native_metrics(arm) for name,arm in result['arms'].items()}
    receipt={'case':spec,'parity':parity,'simulation_only':True,'production_qualified':False,
        'runtime':runtime_metadata(),'attempt':attempt,
        'source_receipt_sha256':digest(OUT/'registration/source.json'),'protocol_sha256':digest(PROTOCOL),
        'weights_sha256':digest(OUT/'weights.json'),'world_receipt_sha256':digest(world_path),
        'raw_result':{'path':str(raw_path.relative_to(ROOT)),'sha256':digest(raw_path)},
        'candidate_inputs_forecasts':inputs,'candidate_jobs':jobs,'metrics':metrics,
        'paired_delta_candidate_minus_Map':{name:metrics['candidate'][name]-metrics['radiomap'][name]
            for name in metrics['candidate']},'original_SERVICE_holdouts_opened':False,
        'native_simulations':2,'geometry_carrier_scale_transfer_unqualified':True}
    if parity:receipt['wiring_proof']=_parity(result,raw)
    write(path,receipt)
    return _load_verified_case(spec,parity)[0]


def summarize(rows):
    result={}
    for cost in (.03,4.):
        subset=[r for r in rows if not r['parity'] and r['case']['handover_seconds']==cost]
        if len(subset)!=4:raise ValueError('Keep all four episodes for each handover cost')
        deltas={name:np.asarray([r['paired_delta_candidate_minus_Map'][name] for r in subset],dtype=float)
                for name in subset[0]['metrics']['candidate']}
        result[str(cost)]={'episodes':4,'seeds':[r['case']['seed'] for r in subset],
            'paired_mean_candidate_minus_Map':{name:float(a.mean()) for name,a in deltas.items()},
            'control_miss_harm_count':int((deltas['control_miss']>0).sum()),
            'worst_decile_harm_count':int((deltas['control_miss_worst_decile']>0).sum()),
            'max_silence_harm_count':int((deltas['maximum_silence_ms']>0).sum())}
    return result


def run():
    check_sources();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    bind_weights()  # Passive completeness check before allocating any case attempt.
    parity=run_case(protocol_spec()['parity_case'],parity=True)
    if parity['wiring_proof']['passes'] is not True:raise RuntimeError('Wiring parity failed; do not interpret a learned native comparison')
    rows=[run_case(case) for case in protocol_spec()['cases']]
    summary={'simulation_only':True,'native_simulations':18,'production_qualified':False,'default_changed':False,
        'source_receipt_sha256':digest(OUT/'registration/source.json'),'weights_sha256':digest(OUT/'weights.json'),
        'wiring_proof':parity['wiring_proof'],'separate_handover_results':summarize(rows),
        'cases':{r['case']['case_id']:digest(OUT/'cases'/r['case']['case_id']/'receipt.json') for r in [parity,*rows]},
        'no_small_sample_SLA_or_SOTA_claim':True,'original_SERVICE_holdouts_opened':False}
    write(OUT/'summary.json',summary);write(OUT/'verification/audit.json',audit());seal();return summary


def _audit_inputs(receipt,result):
    from ..fleetwm.data import Geometry,Observations
    from ..twin.radio import AP
    directory=OUT/'worlds'/str(receipt['case']['seed']);world_receipt=read(directory/'receipt.json')
    if digest(directory/'receipt.json')!=receipt['world_receipt_sha256']:raise RuntimeError('Native world receipt changed')
    expected={'seed':receipt['case']['seed'],'robots':100,'duration_seconds':12,'warm_seconds':60,
        'source_receipt_sha256':receipt['source_receipt_sha256'],'protocol_sha256':receipt['protocol_sha256'],
        'physical_truth_archived_for_audit_not_model_input':True}
    if any(encoded(world_receipt.get(k))!=encoded(v) for k,v in expected.items()):
        raise RuntimeError('Native world metadata differs from its registered episode')
    _reference(world_receipt['world'],directory/'world.npz');_reference(world_receipt['warm'],directory/'warm.npz')
    world=_arrays(ac.safe_path(world_receipt['world']['path']));warm=_arrays(ac.safe_path(world_receipt['warm']['path']))
    if (set(world)!=set(WORLD_ARRAYS) or set(warm)!=set(OBSERVATIONS)
            or {name:p.array_digest(a) for name,a in world.items()}!=world_receipt['world_array_sha256']
            or {name:p.array_digest(a) for name,a in warm.items()}!=world_receipt['warm_array_sha256']):
        raise RuntimeError('Native actual world/warm array identities changed')
    arrays=_arrays(ac.safe_path(receipt['candidate_inputs_forecasts']['path']))
    aps=tuple(AP(**(v|{'allowed_channels':tuple(v['allowed_channels'])})) for v in world_receipt['public_APs'])
    g=Geometry(world['sites'],world['tx_dl'],world['noise_db'],aps,False)
    history={name:arrays['last_history_'+name] for name in OBSERVATIONS}
    if any(not np.array_equal(history[name][:60],warm[name]) for name in OBSERVATIONS):
        raise RuntimeError('Candidate warm history differs from actual common warm reports')
    # Later deliveries are recorded in the current bin, never backdated into
    # a prefix already passed to the predictor. The last prefix recovers all
    # earlier inputs without storing twelve duplicate histories.
    last=Observations(*(history[name] for name in OBSERVATIONS));records=[]
    for i,job in enumerate(result['arms']['candidate']['forecast_jobs']):
        if receipt['candidate_jobs'][i]['history_length']!=60+i:raise RuntimeError('Candidate used a future history length')
        plan=arrays[f'job{i}_plan']
        if not np.array_equal(plan,world['plan'][i]):raise RuntimeError('Candidate plan differs from public native plan')
        if receipt['parity']:
            records.append({'source_snr':arrays[f'job{i}_source_snr']})
        else:
            expected=p.select_context(last.before(60+i),g)
            for name in ('context','known','scheduled','selected_rows'):
                if not np.array_equal(arrays[f'job{i}_{name}'],expected[name]):
                    raise RuntimeError('Candidate context differs from actual past delivered reports: '+name)
        power=arrays[f'job{i}_mean_dbm'];snr=p.radio_assumptions(g).convert(power).astype('f4')
        if hashlib.sha256(snr.tobytes()).hexdigest()!=job['forecast_sha256']:
            raise RuntimeError('Paid native candidate forecast differs from archived RSSI conversion')
        if not receipt['parity']:
            std=arrays[f'job{i}_std_db'].astype('f4')
            if hashlib.sha256(std.tobytes()).hexdigest()!=job['forecast_std_sha256']:
                raise RuntimeError('Native candidate std differs from archived model output')
    final_deliveries={}
    for event in result['arms']['candidate']['telemetry']['events']:
        if not event['measurement_slot']<event['complete_slot']<=event['delivery_slot']:
            raise RuntimeError('Delivered report time order changed')
        t=60+event['delivery_slot']//100;r=event['robot']
        if t>=len(history['value']):continue  # This event was never a forecast input.
        final_deliveries[t,r]=event
    for (t,r),event in final_deliveries.items():
        # More than one report can arrive in one bin after a delayed scan.
        # Only the last is retained by the actual coarse history memory.
        payload=history['value'][t,r].tobytes()+history['heard'][t,r].tobytes()
        if hashlib.sha256(payload).hexdigest()!=event['payload_sha256']:
            raise RuntimeError('Candidate delivered report differs from actual payload bytes')
    if receipt['parity']:
        from types import SimpleNamespace
        proof=_parity(result,SimpleNamespace(records=records))
        if proof!=receipt['wiring_proof'] or proof['passes'] is not True:raise RuntimeError('Native Map wiring proof failed')


def audit():
    check_sources();weights=bind_weights();summary=read(OUT/'summary.json');rows=[]
    specs=[(protocol_spec()['parity_case'],True)]+[(s,False) for s in protocol_spec()['cases']]
    for spec,parity in specs:
        path=OUT/'cases'/spec['case_id']/'receipt.json';receipt,result=_load_verified_case(spec,parity)
        if digest(path)!=summary['cases'][spec['case_id']] or receipt['case']!=spec or receipt['parity']!=parity:
            raise RuntimeError('Native case membership/receipt changed')
        metrics={name:_native_metrics(arm) for name,arm in result['arms'].items()}
        if metrics!=receipt['metrics']:raise RuntimeError('Native service metrics differ from raw packet counters')
        for arm in result['arms'].values():
            previous=0.
            for decision,job in enumerate(arm['forecast_jobs']):
                compute=max(job['actual_prediction_seconds'],job['actual_issue_seconds'])
                charged=max(job['profile_seconds'],compute);service=charged+job['shared_extra_seconds']
                previous=max(float(decision),previous)+service
                if (job['decision']!=decision or job['history_cutoff_exclusive']!=60+decision
                        or abs(job['actual_compute_seconds']-compute)>1e-10
                        or abs(job['charged_compute_seconds']-charged)>1e-10
                        or abs(job['charged_seconds']-service)>1e-10
                        or job['ready_slot']!=math.ceil(previous*100-1e-10)):
                    raise RuntimeError('Native completion accounting changed')
            for consumed in arm['forecast_consumption']:
                if consumed['slot']<consumed['ready_slot']:raise RuntimeError('A forecast was consumed before completion')
            detail=arm['native']['packet_detail']
            if sum(detail['misses'])!=arm['native']['control_misses'] or sum(detail['sent'])!=arm['native']['control_sent']:
                raise RuntimeError('Native fleet totals differ from per-robot counters')
        if not parity:rows.append(receipt)
    if summarize(rows)!=summary['separate_handover_results'] or summary['production_qualified'] is not False:
        raise RuntimeError('Native paired summary or qualification changed')
    return {'passes':True,'cases':9,'native_simulations':18,'new_fits_or_forwards':False,
            'production_qualified':False,'checkpoint_sha256':weights['checkpoint_sha256']}


def seal():
    check_sources()
    manifest=OUT/'verification/manifest.json'
    evidence={str(path.relative_to(ROOT)):digest(path) for path in sorted(OUT.rglob('*'))
              if path.is_file() and path!=manifest and not path.is_relative_to(OUT/'registration/source')}
    value={'schema':1,'source_sha256':source_hashes(),'evidence_sha256':evidence,
           'source_receipt_sha256':digest(OUT/'registration/source.json'),'production_qualified':False}
    write(manifest,value);return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('write-spec','preflight','register','publish','run','audit'))
    parser.add_argument('--commit');parser.add_argument('--remote',default='origin');parser.add_argument('--branch')
    args=parser.parse_args()
    if args.command=='publish':value=record_publication(args.commit,args.remote,args.branch)
    else:value={'write-spec':write_spec,'preflight':preflight,'register':register,'run':run,'audit':audit}[args.command]()
    print(json.dumps(value,sort_keys=True))


if __name__=='__main__':main()
