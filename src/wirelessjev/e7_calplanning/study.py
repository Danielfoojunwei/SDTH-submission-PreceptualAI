"""Finite original-E7 calibrated-map to analytical-planner diagnostic."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import os
import time

import numpy as np

from ..envlearn import random_env,truth_scene,drive_and_survey,truth_evaluator
from ..fastplan import FastPlanner
from ..missions import WAREHOUSE_MISSIONS
from ..teachers.assign import legacy_policy
from . import common as c,data


def save_arrays(path,arrays):
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);c.write_bytes_once(path,stream.getvalue())
    return {'path':str(path.relative_to(c.ROOT)),'sha256':c.digest(path)}


def begin_attempt(seed):
    if type(seed)is not int or seed not in c.SEEDS:raise ValueError('Unregistered E7 seed')
    directory=c.OUT/'cases'/str(seed)
    if directory.exists() and any(directory.iterdir()):raise RuntimeError('Partial E7 attempt retained; refuse regeneration/refitting/replanning')
    directory.mkdir(parents=True,exist_ok=True);path=directory/'attempt.json'
    value={'seed':seed,'source_receipt_sha256':c.digest(c.OUT/'registration/source.json'),
        'protocol_sha256':c.digest(c.PROTOCOL),'created_unix_ns':time.time_ns(),'maximum_fits':6,'maximum_searches':35}
    try:
        with path.open('xb') as stream:stream.write(c.encoded(value));stream.flush();os.fsync(stream.fileno())
    except FileExistsError as exc:raise RuntimeError('E7 attempt already exists') from exc
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return path


def generate_inputs(seed):
    c.check_sources()
    if type(seed)is not int or seed not in c.SEEDS:raise ValueError('Unregistered E7 seed')
    directory=c.OUT/'cases'/str(seed);attempt_path=directory/'attempt.json'
    if (not directory.is_dir() or {p.name for p in directory.iterdir()}!={'attempt.json'}
            or attempt_path.is_symlink()):
        raise RuntimeError('E7 generation requires an unused canonical exclusive attempt')
    attempt=c.read_json(attempt_path)
    if (attempt['seed']!=seed or attempt['source_receipt_sha256']!=c.digest(c.OUT/'registration/source.json')
            or attempt['protocol_sha256']!=c.digest(c.PROTOCOL) or attempt['maximum_fits']!=6
            or attempt['maximum_searches']!=35):raise RuntimeError('E7 generation attempt binding changed')
    generation={'seed':seed,'attempt_sha256':c.digest(attempt_path),'created_unix_ns':time.time_ns(),
        'source_receipt_sha256':attempt['source_receipt_sha256'],'protocol_sha256':attempt['protocol_sha256']}
    try:
        with (directory/'generation.json').open('xb') as stream:
            stream.write(c.encoded(generation));stream.flush();os.fsync(stream.fileno())
    except FileExistsError as exc:raise RuntimeError('E7 generation was already attempted') from exc
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    spec=random_env(seed)
    scene,survey=drive_and_survey(truth_scene(30000+seed,spec),spec,15,np.random.default_rng(seed))
    truth=truth_evaluator(scene,spec,np.random.default_rng(seed+1))
    public=data.public_fit_inputs(scene.aps,spec.ap_height)
    now=np.asarray([(car.x,car.y) for car in scene.cars])
    future=np.asarray([scene.mobility.planned_positions(car,[10.,20.]) for car in scene.cars]).transpose(1,0,2)
    positions=np.concatenate((now[None],future),axis=0)
    shared=data.shared_oracle_arrays(truth)
    shared.update(client_tx_dbm=np.asarray(scene.cars[0].tx_power_dbm),
        band_names=np.asarray([ap.band for ap in scene.aps],dtype='U8'),
        channels=np.asarray([ap.channel for ap in scene.aps],dtype='i8'))
    starts=np.stack((truth.current,truth.from_assoc(legacy_policy(scene))))
    public_arrays={'ap_xy':np.asarray([(ap.x,ap.y) for ap in public.aps]),
        'ap_band':shared['band_names'],'ap_tx':np.asarray([ap.tx_power_dbm for ap in public.aps]),
        'ap_height':np.asarray(public.ap_height),'positions':positions}
    private={'ul':np.stack([s.ul for s in truth.snaps]),'dl':np.stack([s.dl for s in truth.snaps]),
        'external_noise':truth.ext_noise_mw.copy(),'availability':truth.avail.copy()}
    return public,survey,shared,starts,public_arrays,private


def _search(evaluator,mission,starts):
    planner=FastPlanner(None,evaluator=evaluator,device=None)
    return planner.search(mission.weights,mission.focus_cars,starts=starts,max_iter=40,screen_iters=1,multi=True)


def run_environment(seed):
    c.check_sources()
    if seed not in c.SEEDS or type(seed)is not int:raise ValueError('Unregistered seed')
    directory=c.OUT/'cases'/str(seed);receipt=directory/'receipt.json'
    if receipt.exists():
        from .verification import verify_environment
        verify_environment(seed);return c.read_json(receipt)
    attempt=begin_attempt(seed);started=time.perf_counter()
    public,survey,shared,starts,public_arrays,private=generate_inputs(seed)
    refs={'attempt':{'path':str(attempt.relative_to(c.ROOT)),'sha256':c.digest(attempt)},
        'generation':{'path':str((directory/'generation.json').relative_to(c.ROOT)),'sha256':c.digest(directory/'generation.json')},
        'survey':save_arrays(directory/'survey.npz',data.survey_arrays(survey)),
        'public':save_arrays(directory/'public.npz',public_arrays),
        'shared_ORACLE':save_arrays(directory/'shared_ORACLE.npz',dict(shared,starts=starts)),
        'private_SCORING':save_arrays(directory/'private_SCORING.npz',private)}
    choices=np.empty((3,2,5,100),dtype='i8');predicted=np.empty((3,2,5));stats=[];fits=[]
    for bi,budget in enumerate(c.BUDGETS):
        sub=data.tail_subset(survey,budget)
        refs[f'survey_b{budget}']=save_arrays(directory/f'survey_b{budget}.npz',data.survey_arrays(sub))
        for ki,kind in enumerate(c.KINDS):
            then=time.perf_counter();model=data.fit_map(kind,public,sub,seed)
            fits.append({'budget':budget,'kind':kind,'elapsed_seconds':time.perf_counter()-then})
            refs[f'mean_state_{budget}_{kind}']=save_arrays(directory/f'mean_state_{budget}_{kind}.npz',data.mean_state(model,kind))
            belief=data.make_belief(shared,public_arrays['positions'],public,sub,lambda x,y:model.predict(x,y)[0])
            refs[f'belief_{budget}_{kind}']=save_arrays(directory/f'belief_{budget}_{kind}.npz',{
                'ul':np.stack([s.ul for s in belief.snaps]),'dl':np.stack([s.dl for s in belief.snaps]),
                'external_noise':belief.ext_noise_mw,'availability':belief.avail})
            for mi,name in enumerate(c.MISSIONS):
                action,utility,stat=_search(belief,WAREHOUSE_MISSIONS[name],starts)
                choices[bi,ki,mi]=action;predicted[bi,ki,mi]=utility
                stats.append({'budget':budget,'kind':kind,'mission':name,**stat})
    # All belief choices are immutable BEFORE private score/reference reads.
    refs['choices']=save_arrays(directory/'choices.npz',{'actions':choices,'predicted_utility':predicted})
    refs['choice_stats']={'path':str((directory/'choice_stats.json').relative_to(c.ROOT))}
    c.write_once(directory/'choice_stats.json',stats);refs['choice_stats']['sha256']=c.digest(directory/'choice_stats.json')
    scored=score_selected(directory,shared,starts,choices,private)
    refs['truth_references']=save_arrays(directory/'truth_references.npz',scored['references'])
    value={'seed':seed,'scope':c.protocol_spec()['scope'],'source_receipt_sha256':c.digest(c.OUT/'registration/source.json'),
        'protocol_sha256':c.digest(c.PROTOCOL),'refs':refs,'fit_records':fits,'reference_stats':scored['reference_stats'],
        'paired_rows':scored['rows'],'map_error':scored['map_error'],'fits':6,'belief_searches':30,'reference_searches':5,
        'elapsed_seconds':time.perf_counter()-started,'production_qualified':False,'default_changed':False,
        'original_SERVICE_holdouts_opened':False,'new_packet_simulations':0,'no_fresh_confirmation_claim':True}
    c.write_once(receipt,value)
    from .verification import verify_environment
    verify_environment(seed)
    print(f'E7 conditional diagnostic seed{seed} complete:30belief+5reference searches',flush=True)
    return value


def score_selected(directory,shared,starts,choices,private):
    """Private analytical scoring reader; no map/calibration selection here."""
    truth=data.evaluator(shared,list(zip(private['ul'],private['dl'])),external_noise=private['external_noise'],availability=private['availability'])
    ref_actions=[];ref_util=[];ref_stats=[];rows=[];errors={}
    for name in c.MISSIONS:
        action,utility,stat=_search(truth,WAREHOUSE_MISSIONS[name],starts)
        ref_actions.append(action);ref_util.append(utility);ref_stats.append({'mission':name,**stat})
    for bi,budget in enumerate(c.BUDGETS):
        for mi,name in enumerate(c.MISSIONS):
            mission=WAREHOUSE_MISSIONS[name];focus=[truth.car_ids.index(s) for s in mission.focus_cars if s in truth.car_ids]
            utilities=truth.utility(choices[bi,:,mi],mission.weights,focus)
            legacy=float(truth.utility(starts[1],mission.weights,focus)[0])
            rows.append({'budget':budget,'mission':name,'corrected_v2':float(utilities[0]),
                'geometry_censored':float(utilities[1]),'candidate_minus_v2':float(utilities[1]-utilities[0]),
                'truth_search_reference':float(ref_util[mi]),'legacy':legacy,
                'v2_signed_gap':float(ref_util[mi]-utilities[0]),'candidate_signed_gap':float(ref_util[mi]-utilities[1])})
    # Diagnostic subsets use truth only AFTER all actions were written.
    for budget in c.BUDGETS:
        for kind in c.KINDS:
            with np.load(directory/f'belief_{budget}_{kind}.npz',allow_pickle=False) as saved:mean=saved['dl'][0]
            target=private['dl'][0];errors[f'{budget}/{kind}']={}
            for key,mask in [('audible',target>=-90),('inaudible',target< -90),('all',np.ones(target.shape,bool))]:
                errors[f'{budget}/{kind}'][key]=float(np.sqrt(np.mean((mean[mask]-target[mask])**2))) if mask.any() else None
    return {'rows':rows,'map_error':errors,'references':{'actions':np.stack(ref_actions),'utility':np.asarray(ref_util)},'reference_stats':ref_stats}


def summarize(rows):
    if [r['seed'] for r in rows]!=list(c.SEEDS):raise ValueError('Retain all15 registered environment units')
    return {str(b):{'environment_units':15,'mission_pairs':75,
        'mean_environment_utility_delta':float(np.mean([np.mean([p['candidate_minus_v2'] for p in r['paired_rows'] if p['budget']==b]) for r in rows])),
        'environment_harms':int(sum(np.mean([p['candidate_minus_v2'] for p in r['paired_rows'] if p['budget']==b])<0 for r in rows)),
        'mission_pair_harms':int(sum(p['candidate_minus_v2']<0 for r in rows for p in r['paired_rows'] if p['budget']==b)),
        'per_environment_delta':[float(np.mean([p['candidate_minus_v2'] for p in r['paired_rows'] if p['budget']==b])) for r in rows]}
        for b in c.BUDGETS}


def run(workers=2):
    c.check_sources()
    if workers not in (1,2):raise ValueError('Fixed one/two-worker CPU budget')
    if any(os.environ.get(k)!='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')):
        raise RuntimeError('All numerical thread limits must be1 before process start')
    if workers==1:rows=[run_environment(s) for s in c.SEEDS]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(run_environment,c.SEEDS))
    value={'scope':c.protocol_spec()['scope'],'source_receipt_sha256':c.digest(c.OUT/'registration/source.json'),
        'protocol_sha256':c.digest(c.PROTOCOL),'seeds':list(c.SEEDS),'summary':summarize(rows),
        'environment_receipts':{str(s):c.digest(c.OUT/'cases'/str(s)/'receipt.json') for s in c.SEEDS},
        'map_fits':90,'belief_searches':450,'reference_searches':75,'paired_rows':225,'environment_units':15,
        'production_qualified':False,'default_changed':False,'original_SERVICE_holdouts_opened':False}
    c.write_once(c.OUT/'report.json',value)
    from .verification import audit,seal,verify_closure
    (c.OUT/'verification').mkdir(parents=True,exist_ok=True)
    c.write_once(c.OUT/'verification/audit.json',audit());seal();verify_closure();return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('write-spec','preflight','register','publish','run','audit'))
    parser.add_argument('--commit');parser.add_argument('--remote',default='origin');parser.add_argument('--branch');parser.add_argument('--workers',type=int,default=2)
    a=parser.parse_args()
    if a.command=='publish':value=c.publish(a.commit,a.remote,a.branch)
    elif a.command=='run':value=run(a.workers)
    elif a.command=='audit':
        from .verification import audit,verify_closure
        value=verify_closure() if (c.OUT/'verification/manifest.json').exists() else audit()
    else:value={'write-spec':c.write_spec,'preflight':c.preflight,'register':c.register}[a.command]()
    import json
    print(json.dumps(value,sort_keys=True))


if __name__=='__main__':main()
