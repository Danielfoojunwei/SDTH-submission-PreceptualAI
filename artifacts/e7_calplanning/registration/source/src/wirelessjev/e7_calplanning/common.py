"""Registration for a conditional, already-opened E7 planning diagnostic."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile

from ..adaptive_radio.common import encoded,digest,read_json,write_once,write_bytes_once,safe_path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/e7_calplanning'
PROTOCOL=OUT/'protocol.json'
PREVIOUS=ROOT/'artifacts/native_bridge/verification/manifest.json'
PREVIOUS_SHA='4d9f95316d9e029365ed64d4b04e43128a49239c4115ebb8a6bb49f11c12aeb9'
CALIBRATOR=ROOT/'docs/experiments/e7_calibration_study.py'
OLD_REPORTS=tuple(ROOT/'artifacts'/p/'report.json' for p in ('envlearn','envlearn_corrected','envlearn_calibration'))
SEEDS=tuple(range(15,30));BUDGETS=(1,5,15);KINDS=('corrected_v2','geometry_censored')
MISSIONS=('teleop_assist','peak_picking','map_rollout','shift_fairness','scanner_coexistence')
_PUBLISHED=set()


def protocol_spec():
    return {'schema':1,'scope':'ORACLE-CONDITIONED DESCRIPTIVE; previously opened environment seeds; not deployable qualification',
        'seeds':list(SEEDS),'variants':list(KINDS),'missions':list(MISSIONS),'budgets':list(BUDGETS),
        'budget_semantics':'nominal1/5/15minute nested flattened TAIL-ROW fractions of one15minute survey, NOT chronological prefixes or elapsed-budget guarantees',
        'telemetry':'same legacy tail fractions for RSSI/censor/link-MCS-floor/busy; no record timestamps exist',
        'robots':100,'survey_steps':180,'step_seconds':5,'snapshots_seconds':[0,10,20],
        'calibrator':'CalibratedMap(geometry=True,censored=True) from exact pinned docs/experiments/e7_calibration_study.py',
        'map':{'D':200,'length_m':8.,'prior_std_db':8.,'noise_std_db':4.,'kappa':0.},
        'shared_ORACLE_metadata':['truth-derived top8 visible action masks','contention domains, interferer graph and truth-configured adjacent-channel leakage matrix',
            'protected-channel flags','legacy-policy starting plan; current associations'],
        'public_map_inputs':['AP XY/band/TX','assumed-known AP height','nominal warehouse racks','reported RSSI/censor bounds'],
        'private_scoring_only':['true current/future UL/DL','hidden walls/materials/propagation','true external noise','future forklift blockage'],
        'planner':{'max_iter':40,'screen_iters':1,'multi':True,'starts':['current','shared_truth_legacy'],
            'handover_seconds':4.,'utility_horizon_seconds':30.,'reference':'bounded truth-informed local search, not global optimum'},
        'finite_budget':{'map_fits':90,'belief_searches':450,'truth_reference_searches':75,'total_searches':525,
            'maximum_search_iterations':21000,'paired_rows':225,'environment_units':15},
        'primary':'per-environment mean over5missions of truth utility(candidate)-truth utility(v2), separately by budget',
        'all_harms_and_signed_gaps_retained':True,'iid_mission_or_robot_significance_claim':False,
        'packet_ns3_SLA_or_hardware_endpoint':False,'default_changed':False,'production_qualified':False,
        'original_SERVICE_holdouts_permitted':False,'automatic_repeat_after_interrupted_attempt':False}


def write_spec():
    value=protocol_spec();write_once(PROTOCOL,value);return value


def preserved():
    original=ROOT/'artifacts/radiojepa/rt_data'
    if (original/'confirmation').exists() or (original/'confirmation.json').exists():
        raise RuntimeError('Original SERVICE confirmation must remain unopened')
    if digest(PREVIOUS)!=PREVIOUS_SHA:raise RuntimeError('Preserved native manifest changed')
    m=read_json(PREVIOUS)
    for group in ('source_sha256','evidence_sha256'):
        for name,h in m[group].items():
            if digest(safe_path(name))!=h:raise RuntimeError('Preserved scientific bytes changed: '+name)
    return m


def sources():
    result=dict(preserved()['source_sha256'])
    paths=[PROTOCOL,PREVIOUS,CALIBRATOR,*OLD_REPORTS]
    paths+=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'tests/e7_calplanning').glob('test_*.py'))
    paths+=list((ROOT/'docs/e7_calplanning').glob('*.md'))
    for path in paths:result[str(path.relative_to(ROOT))]=digest(path)
    return dict(sorted(result.items()))


def preflight():
    write_spec();before=sources()
    result=subprocess.run([sys.executable,'-m','pytest','-q','tests/e7_calplanning'],cwd=ROOT,capture_output=True,text=True)
    log=OUT/'preflight/tests.log';write_bytes_once(log,(result.stdout+result.stderr).encode())
    if sources()!=before:raise RuntimeError('Source changed during E7 preflight')
    value={'exit_code':result.returncode,'source_sha256':before,'stdout':{'path':str(log.relative_to(ROOT)),'sha256':digest(log)}}
    write_once(OUT/'preflight/tests.json',value)
    if result.returncode:raise RuntimeError('E7 planning preflight failed')
    return value


def register():
    write_spec();path=OUT/'registration/source.json'
    if path.exists():return check_sources(published=False)
    for name in ('cases','report.json','verification'):
        q=OUT/name
        if q.is_file() or q.is_dir() and any(q.rglob('*')):raise RuntimeError('Register before scientific outputs')
    mapping=sources();test=read_json(OUT/'preflight/tests.json')
    if (test['exit_code']!=0 or test['source_sha256']!=mapping
            or digest(safe_path(test['stdout']['path']))!=test['stdout']['sha256']):raise RuntimeError('Preflight source mismatch')
    capsule=OUT/'registration/source'
    for name,h in mapping.items():
        target=safe_path(name,capsule);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if digest(target)!=h:raise RuntimeError('Partial capsule changed')
        else:shutil.copyfile(safe_path(name),target)
    value={'schema':1,'source_sha256':mapping,'protocol_sha256':digest(PROTOCOL),
        'preflight_sha256':{str((OUT/'preflight/tests.json').relative_to(ROOT)):digest(OUT/'preflight/tests.json'),
                           test['stdout']['path']:test['stdout']['sha256']},
        'old_reports_sha256':{str(p.relative_to(ROOT)):digest(p) for p in OLD_REPORTS},
        'preserved_native_manifest_sha256':PREVIOUS_SHA,'scientific_cases_before_registration':False}
    write_once(path,value);return value


def _check_commit(commit,receipt):
    members=dict(receipt['source_sha256']);members.update(receipt['preflight_sha256'])
    members[str((OUT/'registration/source.json').relative_to(ROOT))]=digest(OUT/'registration/source.json')
    members.update({str((OUT/'registration/source').relative_to(ROOT))+'/'+k:v for k,v in receipt['source_sha256'].items()})
    proc=subprocess.Popen(['git','archive','--format=tar',commit,*members],cwd=ROOT,stdout=subprocess.PIPE);seen=set()
    try:
        with tarfile.open(fileobj=proc.stdout,mode='r|') as archive:
            for item in archive:
                if item.isdir():continue
                if not item.isfile() or item.name not in members:raise RuntimeError('Unexpected published E7 member')
                if hashlib.sha256(archive.extractfile(item).read()).hexdigest()!=members[item.name]:raise RuntimeError('Published E7 bytes differ')
                seen.add(item.name)
        if proc.wait() or seen!=set(members):raise RuntimeError('Published E7 source incomplete')
    finally:
        proc.stdout.close()
        if proc.poll() is None:proc.terminate()
        proc.wait()


def check_sources(*,published=True):
    receipt=read_json(OUT/'registration/source.json')
    if (read_json(PROTOCOL)!=protocol_spec() or receipt['source_sha256']!=sources()
            or receipt['protocol_sha256']!=digest(PROTOCOL)):
        raise RuntimeError('Registered E7 bytes changed')
    capsule=OUT/'registration/source'
    if {str(x.relative_to(capsule)) for x in capsule.rglob('*') if x.is_file()}!=set(receipt['source_sha256']):raise RuntimeError('E7 capsule membership changed')
    for name,h in receipt['source_sha256'].items():
        if digest(safe_path(name,capsule))!=h:raise RuntimeError('E7 capsule changed')
    for name,h in receipt['preflight_sha256'].items():
        if digest(safe_path(name))!=h:raise RuntimeError('E7 preflight proof changed')
    if published:
        pub=read_json(OUT/'registration/git.json');key=digest(OUT/'registration/git.json')
        if (pub['source_receipt_sha256']!=digest(OUT/'registration/source.json') or pub['commit']!=pub['remote_observed_commit']):raise RuntimeError('E7 publication identity changed')
        if key not in _PUBLISHED:_check_commit(pub['commit'],receipt);_PUBLISHED.add(key)
    return receipt


def publish(commit,remote,branch):
    receipt=check_sources(published=False)
    if not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact commit required')
    _check_commit(commit,receipt)
    head=subprocess.check_output(['git','ls-remote',remote,'refs/heads/'+branch],cwd=ROOT,text=True).split()
    if not head or head[0]!=commit:raise RuntimeError('E7 commit is not observed remote branch head')
    value={'commit':commit,'remote':remote,'branch':branch,'remote_observed_commit':commit,
           'source_receipt_sha256':digest(OUT/'registration/source.json')}
    write_once(OUT/'registration/git.json',value);return value
