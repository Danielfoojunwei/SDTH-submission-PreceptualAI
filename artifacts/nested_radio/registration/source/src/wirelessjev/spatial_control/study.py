"""Finite fresh diagnostic: support, causal command delivery and local recovery.

No training, TEST retuning, original SERVICE opening, or application promotion.
Publication and once-only attempts precede every fresh world and model forward.
"""
from __future__ import annotations

import argparse
from copy import copy
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
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'artifacts/spatial_control'
SEEDS = (90511001, 90511002, 90511003, 90511004)
DURATION = 20
ARMS = ('radiomap', 'radiomap_recovery', 'regression', 'regression_support',
        'regression_recovery', 'regression_both')
PRIOR = ('radiojepa', 'radiojepa_spectral', 'device_transfer', 'channel_scene',
         'adaptive_radio', 'native_bridge', 'e7_calplanning')
FIELDS = ('positions', 'value', 'heard', 'bound', 'scanned')
WORLD_FIELDS = ('sites','snr','rssi_dl','domain','avail','video_mbps','bulk_mbps',
                'noise_db','channel','plan','pos','tx_dl')
_PUBLISHED_COMMITS = set()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def canonical(value):
    def cast(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        raise TypeError(type(item).__name__)
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=cast)+'\n').encode()


def write_bytes(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def write(path, value): write_bytes(path, canonical(value))
def read(path): return json.loads(Path(path).read_bytes())
def reference(path): return {'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}


def npz(path, arrays):
    stream = io.BytesIO(); np.savez_compressed(stream, **arrays)
    write_bytes(path, stream.getvalue()); return reference(path)


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def array_hash(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(str((value.dtype.str,value.shape)).encode()+value.tobytes()).hexdigest()


def admit_case(case, arm, parity_kernel=None):
    """Exact finite membership, including the two distinct disabled controls."""
    if not isinstance(case,dict) or type(arm) is not str:
        raise ValueError('Exact registered case and arm required')
    if parity_kernel is None:
        expected=next((item for item in spec()['cases'] if item['id']==case.get('id')),None)
        if case!=expected or arm not in ARMS:raise ValueError('Unregistered fresh case/arm')
    else:
        expected={'id':'disabled-parity','seed':SEEDS[0],'handover_seconds':.03}
        expected_arm={'original':'radiomap__original','disabled':'radiomap__disabled'}.get(parity_kernel)
        if case!=expected or expected_arm is None or arm!=expected_arm:
            raise ValueError('Unregistered disabled parity case/arm')
    return case


def spec():
    return {'schema': 1, 'purpose': 'finite_fresh_spatial_support_and_local_recovery_diagnostic',
        'seeds': list(SEEDS), 'robots': 100, 'duration_seconds': DURATION,
        'warm_history_seconds': 60, 'handover_seconds': [.03, 4.], 'arms': list(ARMS),
        'maximum_native_simulations': 50, 'fresh_comparisons': 48, 'disabled_parity_simulations': 2,
        'cases': [{'id': f'fresh-{seed}-ho{ho:g}', 'seed': seed, 'handover_seconds': ho,
                   'order': list(ARMS[(i+j)%6:]+ARMS[:(i+j)%6])}
                  for i,seed in enumerate(SEEDS) for j,ho in enumerate((.03,4.))],
        'parity': {'seed': SEEDS[0], 'handover_seconds': .03,
                   'original_and_new_disabled': True, 'profile_seconds': 4.},
        'weights': {'selected_cell': 'regression', 'seed': 0,
            'selection': 'completed_adaptive_VAL_only; no_native_selection',
            'new_fits': 0, 'encoder_tokens': 64, 'encoder_context_selection_unchanged': True},
        'operator': 'byte_equivalent_band_shared_native_bridge_ops.query_operands_all_learned_arms',
        'forecast': {'profile_seconds': {'cold': 1., 'warm': .25, 'refit': 1.},
                     'shared_extra_seconds': .5, 'charge_actual_overrun': True},
        'support': {'kind': 'AP_neighbor_heuristic_not_packet_risk', 'source_bands': [0,1],
                    'max_age_seconds': 6., 'max_distance_m': 5., 'minimum_heard_dbm': -80.,
                    'latest_own_report_per_source_band': True,
                    'new_censor_never_resurrects_older_heard': True,
                    'additional_full_delivered_history_not_encoder_context': True,
                    'memory_build_charged_in_forecast': True,
                    'new_server_deliveries_refreshed_at_admission': True,
                    'local_only_undelivered_reports_excluded_from_server_guard': True,
                    'six_point_five_GHz_qualified': False},
        'central': {'all_handover_costs_require_delivery': True, 'creation_delay_slots': 10,
                    'creation_compute_budget_seconds': .1, 'ttl_slots_from_creation': 100,
                    'delivered_packet_created_no_earlier_than_command': True,
                    'delivery_strictly_after_creation': True,
                    'same_target_does_not_renew_pending_command': True},
        'recovery': {'explicit_simulated_preinstalled_local_authority': True,
                    'own_completed_scans_and_current_pilots_only': True,
                    'warm_scan_exact_times_unknown_so_not_used_as_recovery_payloads': True,
                    'silence_seconds': .2, 'max_scan_age_seconds': 5.,
                    'minimum_heard_dbm': -75., 'margin_db': 8.,
                    'margin_requires_fresh_current_measurement': True,
                    'unknown_current_uses_target_floor_without_invented_gain': True,
                    'dwell_seconds': 5., 'backoff_seconds': 5.,
                    'scan_seconds': .09, 'shared_scan_minimum_spacing_seconds': 5.,
                    'local_compute_budget_seconds': .005, 'decision_delay_slots': 1,
                    'delayed_validation_batch_budget_seconds': .005,
                    'next_tick_waits_for_actual_overrun_completion': True,
                    'realized_observations_equal': False, 'autonomous_scans_are_paid': True},
        'physics': {'collision_lambda': 0., 'interference': False,
                    'native_fading_blockage_contention_retained': True,
                    'lagged_measurement_requires_historical_association_and_receiver': True,
                    'last_control_estimate_held_between_valid_pilots': True,
                    'carriers_hz': [2.4e9,5.2e9,6.5e9], 'six_point_five_is_extrapolation': True,
                    'physical_device_calibration': False, 'simulation_only': True},
        'analysis': {'unit': 'four_episode_seeds_one_warehouse_family_per_handover',
                    'all_arm_per_episode_results': True, 'contrasts': [
                        ['regression','radiomap'], ['regression_support','regression'],
                        ['regression_recovery','regression'], ['regression_both','regression'],
                        ['radiomap_recovery','radiomap'], ['regression_both','radiomap_recovery']],
                    'report_mean_and_harm_counts': True, 'confirmatory_p_values': False,
                    'no_post_result_architecture_threshold_or_seed_changes': True},
        'original_SERVICE_opened': False, 'default_changed': False,
        'production_qualified': False, 'hardware_actions': 0}


def write_spec():
    path = OUT/'protocol.json'
    if path.exists():
        if read(path) != spec(): raise RuntimeError('Protocol changed')
    else: write(path, spec())


def source_hashes(names=None):
    """Register current dependencies; later verify that exact named path set.

    New unrelated phases may add tracked modules after this closure. They do
    not become dependencies retroactively. Own importable modules/tests retain
    exact membership, so a new spatial-control file cannot bypass the seal.
    """
    tops = ('src/wirelessjev/spatial_control','tests/spatial_control')
    own = {str(path.relative_to(ROOT)) for top in tops for path in (ROOT/top).rglob('*.py')}
    if names is None:
        tracked = subprocess.check_output(['git','ls-files','--','src','tests','scripts','pyproject.toml','uv.lock'],
                                         cwd=ROOT, text=True).splitlines()
        names = {name for name in tracked if name.endswith('.py') or name in ('pyproject.toml','uv.lock')}
        names |= own
        names |= {'scripts/run_spatial_control.py', 'docs/spatial_control/PLAN.md', 'artifacts/spatial_control/protocol.json'}
    else:
        names = set(names)
        registered_own = {name for name in names if isinstance(name,str) and name.endswith('.py')
                          and any(name.startswith(top+'/') for top in tops)}
        if registered_own != own:
            raise RuntimeError('Registered source changed: own module membership')
    for name in names:
        path = Path(name) if isinstance(name,str) else None
        if (path is None or path.is_absolute() or path.as_posix()!=name or '..' in path.parts
                or not (ROOT/path).is_file() or not (ROOT/path).resolve().is_relative_to(ROOT.resolve())):
            raise RuntimeError('Registered source changed: invalid or missing named file')
    return {name: digest(ROOT/name) for name in sorted(names)}


def prior_identities(*, verify_bytes=False):
    result = {}; union = {}
    for study in PRIOR:
        path = ROOT/'artifacts'/study/'verification/manifest.json'
        value = read(path); result[study] = reference(path)
        for field in ('source_sha256','evidence_sha256','documents_sha256'):
            for name, expected in value.get(field, {}).items():
                if name in union and union[name] != expected: raise RuntimeError('Prior closures disagree')
                union[name] = expected
    if (ROOT/'artifacts/radiojepa/rt_data/confirmation').exists() or (ROOT/'artifacts/radiojepa/rt_data/confirmation.json').exists():
        raise RuntimeError('Original SERVICE data must stay unopened')
    if verify_bytes:
        for name, expected in union.items():
            if digest(ROOT/name) != expected: raise RuntimeError('Prior frozen bytes changed: '+name)
    return {'manifests': result, 'named_files': len(union), 'all_named_bytes_checked': verify_bytes}


def register():
    write_spec(); path = OUT/'registration/source.json'
    if path.exists(): return check_sources(published=False)
    if (OUT/'worlds').exists() or (OUT/'cases').exists(): raise RuntimeError('Register before fresh data')
    before = source_hashes()
    tests = sorted(str(p.relative_to(ROOT)) for p in (ROOT/'tests/spatial_control').glob('test_*.py'))
    if not tests: raise RuntimeError('No preflight tests')
    test = subprocess.run([sys.executable,'-m','pytest','-q',*tests], cwd=ROOT, text=True,capture_output=True)
    log = OUT/'preflight/tests.log'; write_bytes(log,(test.stdout+test.stderr).encode())
    if test.returncode or source_hashes() != before: raise RuntimeError('Preflight failed or source changed')
    proof = {'exit_code': 0, 'source_sha256': before, 'log': reference(log),
             'scientific_fresh_worlds_or_model_forwards': 0}
    write(OUT/'preflight/tests.json', proof)
    weights = ROOT/'artifacts/adaptive_radio/checkpoints/regression-seed0.pt'
    selection = ROOT/'artifacts/adaptive_radio/selection.json'
    if read(selection)['selected_cell'] != 'regression': raise RuntimeError('Wrong VAL choice')
    receipt = {'schema': 1, 'source_sha256': before, 'protocol': reference(OUT/'protocol.json'),
        'preflight': reference(OUT/'preflight/tests.json'), 'preflight_log': reference(log),
        'prior': prior_identities(verify_bytes=True), 'checkpoint': reference(weights),
        'VAL_selection': reference(selection), 'before_fresh_worlds_or_forwards': True}
    capsule = OUT/'registration/source'
    for name in before: write_bytes(capsule/name,(ROOT/name).read_bytes())
    write(path,receipt); return receipt


def check_sources(*, published=True):
    receipt = read(OUT/'registration/source.json')
    if receipt['source_sha256'] != source_hashes(receipt['source_sha256']): raise RuntimeError('Registered source changed')
    capsule = OUT/'registration/source'
    membership = {str(path.relative_to(capsule)) for path in capsule.rglob('*') if path.is_file()}
    if membership != set(receipt['source_sha256']): raise RuntimeError('Source snapshot membership changed')
    for name,expected in receipt['source_sha256'].items():
        if digest(OUT/'registration/source'/name) != expected: raise RuntimeError('Source snapshot changed')
    for key in ('protocol','preflight','preflight_log','checkpoint','VAL_selection'):
        item=receipt[key]
        if digest(ROOT/item['path']) != item['sha256']: raise RuntimeError('Dependency changed: '+key)
    if prior_identities()['manifests'] != receipt['prior']['manifests']: raise RuntimeError('Previous seals changed')
    if published:
        publication=read(OUT/'registration/git.json')
        if publication['source_sha256'] != digest(OUT/'registration/source.json'):
            raise RuntimeError('Source publication binding changed')
        commit=publication['commit']
        if not re.fullmatch('[0-9a-f]{40}',commit) or commit != publication['remote_observed_commit']:
            raise RuntimeError('Invalid source publication')
        binding = (commit, publication['source_sha256'])
        if binding not in _PUBLISHED_COMMITS:
            for name,expected in receipt['source_sha256'].items():
                data=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
                if hashlib.sha256(data).hexdigest()!=expected: raise RuntimeError('Published source changed')
            _PUBLISHED_COMMITS.add(binding)
    return receipt


def publish(commit, branch):
    receipt=check_sources(published=False)
    if not re.fullmatch('[0-9a-f]{40}',commit): raise ValueError('Exact commit required')
    head=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()
    if not head or head[0]!=commit: raise RuntimeError('Remote did not expose source commit')
    # Validate all committed bytes, not only a remote ref string.
    for name,expected in receipt['source_sha256'].items():
        data=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
        if hashlib.sha256(data).hexdigest()!=expected: raise RuntimeError('Commit source mismatch')
    value={'commit':commit,'remote':'origin','branch':branch,'remote_observed_commit':commit,
           'source_sha256':digest(OUT/'registration/source.json')}
    write(OUT/'registration/git.json',value); return value


def prepare_world(seed):
    check_sources()  # before importing/generating any registered fresh scene
    from ..fleetwm.data import episode, Observations, Geometry
    from ..contracted.data import runtime_world
    from ..native.world import World, HORIZONS
    from ..twin.radio import AP
    if type(seed) is not int or seed not in SEEDS: raise ValueError('Unregistered seed')
    directory=OUT/'worlds'/str(seed)
    if not (directory/'receipt.json').exists():
        if directory.exists(): raise RuntimeError('Incomplete world; refuse regeneration')
        write(directory/'attempt.json',{'seed':seed,'created_unix_ns':time.time_ns(),
            'source_sha256':digest(OUT/'registration/source.json')})
        full,_,g,observed=episode(seed,100,DURATION,'main')
        warm=observed.before(60); allowed=(np.arange(60)%5==0)[:,None]
        heard=warm.heard&allowed[...,None]
        warm=Observations(warm.positions,np.where(heard,warm.value,0).astype('f4'),heard,
                          warm.bound,warm.scanned&allowed)
        empty=np.zeros((DURATION,100,len(HORIZONS),len(g.sites),3),dtype='f4')
        world=runtime_world(full,DURATION,empty,empty[:,:,0])
        truth=npz(directory/'world.npz',{name:np.asarray(getattr(world,name)) for name in WORLD_FIELDS})
        history=npz(directory/'warm.npz',{name:np.asarray(getattr(warm,name)) for name in FIELDS})
        write(directory/'receipt.json',{'seed':seed,'world':truth,'warm':history,
            'APs':[asdict(ap) for ap in g.aps], 'source_sha256':digest(OUT/'registration/source.json'),
            'world_array_sha256':{name:array_hash(getattr(world,name)) for name in WORLD_FIELDS},
            'warm_array_sha256':{name:array_hash(getattr(warm,name)) for name in FIELDS},
            'protocol_sha256':digest(OUT/'protocol.json'), 'physical_truth_not_model_input':True})
    receipt=read(directory/'receipt.json')
    for key in ('world','warm'):
        if digest(ROOT/receipt[key]['path'])!=receipt[key]['sha256']:raise RuntimeError('World bytes changed')
    if (receipt['seed']!=seed or receipt['source_sha256']!=digest(OUT/'registration/source.json')
            or receipt['protocol_sha256']!=digest(OUT/'protocol.json')):raise RuntimeError('World source/protocol changed')
    a=arrays(ROOT/receipt['world']['path']);o=arrays(ROOT/receipt['warm']['path'])
    if ({name:array_hash(a[name]) for name in WORLD_FIELDS}!=receipt['world_array_sha256']
            or {name:array_hash(o[name]) for name in FIELDS}!=receipt['warm_array_sha256']):
        raise RuntimeError('World/report arrays differ from their exact byte identities')
    empty=np.zeros((DURATION,100,len(HORIZONS),len(a['sites']),3),dtype='f4')
    world=World(seed,100,a['sites'],a['snr'],a['rssi_dl'],empty,a['domain'],a['avail'],a['video_mbps'],
        a['bulk_mbps'],float(DURATION),0.,False,empty[:,:,0],a['noise_db'],None,a['channel'],None,a['plan'],a['pos'],a['tx_dl'])
    aps=tuple(AP(**(item|{'allowed_channels':tuple(item['allowed_channels'])})) for item in receipt['APs'])
    return world,Observations(*(o[name] for name in FIELDS)),Geometry(a['sites'],a['tx_dl'],a['noise_db'],aps,False)


def _run_arm(world,warm,geometry,arm,ho,*,parity_kernel=None):
    # Only execute() calls this internal worker after exact attempt admission.
    from ..closed_loop.runtime import SinglePredictor
    from ..jepa_repair.runtime import CompletionSafeForecasts
    from ..adaptive_radio.model import AdaptiveScene
    from ..adaptive_radio.service import TypedRoamUtility
    from ..worldmodel.radiomap_adapter import RadioMapAdapter
    from ..native_bridge.predictor import bridged
    from ..gate_repair.gate import reactive_rescue
    from .runtime import (ClockedScans,SupportedAdaptivePredictor,SupportAdmission,LocalFleetRuntime)
    from . import kernel
    sensor=ClockedScans(warm,DURATION,world.seed)
    learned=arm.startswith('regression'); support=arm in ('regression_support','regression_both')
    recovery=arm in ('radiomap_recovery','regression_recovery','regression_both')
    raw=None
    if learned:
        weights=read(OUT/'registration/source.json')['checkpoint']
        checkpoint=torch.load(ROOT/weights['path'],map_location='cpu',weights_only=True)
        if checkpoint['cell']!='regression' or checkpoint['seed']!=0:raise RuntimeError('Wrong frozen checkpoint')
        model=AdaptiveScene(**checkpoint['settings']);model.load_state_dict(checkpoint['state']);model.eval()
        raw=SupportedAdaptivePredictor(geometry,model,weights['sha256'],sensor,support=support)
        predictor=bridged(raw)
    else: predictor=SinglePredictor(geometry,'radiomap')
    profile={'cold':1.,'warm':.25,'refit':1.} if parity_kernel is None else {key:4. for key in ('cold','warm','refit')}
    live=CompletionSafeForecasts(predictor,sensor,DURATION,profile,.5)
    physical=copy(world);physical.forecast=live.mean;physical.sigma_belief=live.std[:,:,0];physical.rmap=None
    utility=TypedRoamUtility(RadioMapAdapter(),arm,live,handover_seconds=ho,horizon=10.,max_candidates=64,deadline_seconds=.1)
    guard=SupportAdmission(utility,raw,live,sensor) if support else utility
    local=LocalFleetRuntime(geometry,sensor,ho) if recovery else None
    decisions=[];actions=[]
    def capture(decision,confident,site):
        decisions.append({'forecast':int(decision.decision),'current':decision.current.tolist(),
            'target':decision.candidate.tolist(),'eligible':decision.eligible.tolist(),
            'gate':confident.tolist(),'gain':decision.gain.tolist()})
    def executed(slot,robots,targets):
        actions.append({'slot':int(slot),'robots':np.asarray(robots).tolist(),'targets':np.asarray(targets).tolist()})
    module=kernel
    if parity_kernel=='original':
        from ..service_world import sim as module
    policy=module.Policy('spatial_control_diagnostic','native',roam_cost_s=ho,scan_period_s=5.,
                         gate=True,cadence_scan=True,cadence_roam=True,online_map=False,dual_radio=False)
    options={} if parity_kernel=='original' else {'causal_central':parity_kernel is None,'spatial_runtime':local}
    native=module.simulate(physical,policy,seed=0,duration_s=DURATION,
        collision_lambda=0.,interference=False,roam_gate=reactive_rescue,
        handover_admission=guard,roam_scorer=guard.scores,forecast_queue=live,forecast_source=live,
        scan_observer=sensor,scan_duration_s=.09,decision_trace=capture,actual_action_hook=executed,
        detailed=True,observation_period_slots=10,native_proposal_ttl_slots=100,**options)
    saved={'forecast_mean_snr':live.mean,'forecast_std_db':live.std}
    saved.update({'history_'+name:np.asarray(getattr(sensor.history,name)) for name in FIELDS})
    if raw is not None:
        for j,record in enumerate(raw.records):
            for key,value in record.items():
                if isinstance(value,np.ndarray):saved[f'job{j}_{key}']=value
    result={'arm':arm,'seed':world.seed,'handover_seconds':ho,'native':native,
        'forecast_jobs':live.records,'forecast_consumption':live.consumption_records,
        'telemetry':sensor.summary(),'scan_started_events':sensor.started_events,
        'local_completed_payloads':sensor.local_completed_payloads,
        'planner_and_admission':utility.records,'decisions':decisions,'actual_actions':actions,
        'policy_config':asdict(policy),
        'simulation_config':{'seed':0,'duration_seconds':DURATION,'collision_lambda':0.,
            'interference':False,'scan_duration_seconds':.09,'observation_period_slots':10,
            'native_proposal_ttl_slots':100,'planner':'native'},
        'support':guard.records if support else [],
        'support_refresh':guard.refresh_records if support else [],
        'support_preparation':raw.support_preparation if raw is not None else [],
        'local_runtime':local.summary() if local else None,
        'causal_central':parity_kernel is None,'production_qualified':False}
    return result,saved


def metrics(result):
    native=result['native'];detail=native['packet_detail'];jobs=result['forecast_jobs']
    misses=np.array(detail['misses']);sent=np.array(detail['sent']);gap=np.array(detail['max_gap'])
    derived={'control_miss':float(misses.sum()/sent.sum()),
        'control_miss_worst_decile':float(np.sort(misses/np.maximum(sent,1))[-10:].mean()),
        'maximum_silence_ms':float(gap.max()*10),
        'control_gap_p99_ms':float(np.percentile(gap,99)*10)}
    for key in ('control_miss','control_miss_worst_decile','control_gap_p99_ms'):
        if not math.isclose(derived[key],native[key],abs_tol=1e-10,rel_tol=0):raise RuntimeError('Raw counter mismatch')
    spatial=native.get('spatial_execution',{})
    pending=spatial.get('pending_control_packets',None)
    scans=spatial.get('scan_events',[])
    return derived|{key:native[key] for key in ('video_on_time','bulk_mbps','roams','safety_stop_fraction')}|{
        'control_misses':int(misses.sum()),'control_sent':int(sent.sum()),
        'end_pending_commands':pending,
        'observed_plus_pending_miss_fraction':float((misses.sum()+pending)/sent.sum()) if pending is not None else None,
        'paid_forecast_compute_seconds':sum(j['charged_compute_seconds'] for j in jobs),
        'actual_forecast_compute_seconds':sum(j['actual_compute_seconds'] for j in jobs),
        'forecast_jobs_consumed':len(result['forecast_consumption']),
        'scans_started':result['telemetry']['scans_started'],
        'scan_interruption_seconds':sum(e['interruption_slots'] for e in scans)*.01,
        'autonomous_scans':sum(e['reason']=='autonomous_recovery' for e in scans),
        'central_planning_cpu_seconds':sum(e['elapsed_seconds'] for e in spatial.get('central_decisions',[])),
        'central_planning_overruns':sum(e['overrun'] for e in spatial.get('central_decisions',[])),
        'central_busy_forecast_skips':len(spatial.get('central_compute_events',[])),
        'local_decision_cpu_seconds':sum(e['elapsed_seconds'] for e in spatial.get('local_decisions',[])),
        'local_decision_overruns':sum(e['overrun'] for e in spatial.get('local_decisions',[])),
        'local_validation_cpu_seconds':sum(e['elapsed_seconds'] for e in spatial.get('local_validation_batches',[])),
        'local_validation_overruns':sum(e['overrun'] for e in spatial.get('local_validation_batches',[])),
        'local_busy_tick_skips':len(spatial.get('local_compute_events',[])),
        'support_preparation_cpu_seconds':sum(e['elapsed_seconds'] for e in result.get('support_preparation',[])),
        'support_vetoes':sum(not x['permitted'] for x in result['support']),
        'local_handovers':sum(e['event']=='executed' for e in native.get('spatial_execution',{}).get('local_events',[])),
        'central_handovers':sum(e['event']=='executed' for e in native.get('spatial_execution',{}).get('central_events',[]))}


def execute(case, arm, world, warm, geometry, *, parity_kernel=None):
    check_sources()
    admit_case(case,arm,parity_kernel)
    if world.seed!=case['seed'] or world.n!=100 or world.duration_s!=DURATION:
        raise ValueError('World differs from the registered episode')
    world_receipt=read(OUT/'worlds'/str(world.seed)/'receipt.json')
    if ({name:array_hash(getattr(world,name)) for name in WORLD_FIELDS}!=world_receipt['world_array_sha256']
            or {name:array_hash(getattr(warm,name)) for name in FIELDS}!=world_receipt['warm_array_sha256']
            or not np.array_equal(geometry.sites,world.sites)
            or not np.array_equal(geometry.noise,world.noise_db)
            or not np.array_equal(geometry.tx,world.tx_dl)
            or geometry.yard is not False
            or canonical([asdict(ap) for ap in geometry.aps])!=canonical(world_receipt['APs'])):
        raise RuntimeError('Exact registered physical/public/history identities required')
    name=case['id']; directory=OUT/'cases'/name/arm
    identity={'case':case,'arm':arm,'parity_kernel':parity_kernel,
        'source_sha256':digest(OUT/'registration/source.json'),'protocol_sha256':digest(OUT/'protocol.json'),
        'world_receipt_sha256':digest(OUT/'worlds'/str(world.seed)/'receipt.json')}
    if (directory/'receipt.json').exists():
        previous=read(directory/'receipt.json')
        if any(previous.get(key)!=value for key,value in identity.items()):raise RuntimeError('Completed attempt identity changed')
        load_result(previous)
        return previous
    if directory.exists():raise RuntimeError('Partial attempted arm; refuse replacement: '+str(directory))
    write(directory/'attempt.json',identity|{'created_unix_ns':time.time_ns()})
    result,saved=_run_arm(world,warm,geometry,arm.split('__')[0],case['handover_seconds'],parity_kernel=parity_kernel)
    output=npz(directory/'predictions_history.npz',saved)
    raw=directory/'raw.json.gz';write_bytes(raw,gzip.compress(canonical(result),mtime=0))
    receipt=identity|{'raw':reference(raw),'arrays':output,'metrics':metrics(result),
        'attempt':reference(directory/'attempt.json'),'simulations':1,'simulation_only':True,
        'default_changed':False,'original_SERVICE_opened':False,'production_qualified':False}
    from .validation import validate_result
    receipt['canonical_validation']=validate_result(receipt,result,saved,
        read(OUT/'registration/source.json'),world_receipt,read(OUT/'protocol.json'))
    write(directory/'receipt.json',receipt); return receipt


def load_result(receipt):
    for key in ('raw','arrays','attempt'):
        if digest(ROOT/receipt[key]['path'])!=receipt[key]['sha256']:raise RuntimeError('Evidence changed')
    result=json.loads(gzip.decompress((ROOT/receipt['raw']['path']).read_bytes()))
    attempt=read(ROOT/receipt['attempt']['path'])
    created=attempt.pop('created_unix_ns',None)
    expected={key:receipt[key] for key in ('case','arm','parity_kernel','source_sha256',
                                         'protocol_sha256','world_receipt_sha256')}
    if type(created) is not int or created<=0 or attempt!=expected:
        raise RuntimeError('Durable attempt identity changed')
    from .validation import validate_result
    proof=validate_result(receipt,result,arrays(ROOT/receipt['arrays']['path']),
        read(OUT/'registration/source.json'),
        read(OUT/'worlds'/str(receipt['case']['seed'])/'receipt.json'),read(OUT/'protocol.json'))
    if proof!=receipt['canonical_validation']:raise RuntimeError('Canonical raw validation changed')
    if metrics(result)!=receipt['metrics']:raise RuntimeError('Recomputed metrics changed')
    return result


def run():
    check_sources()
    # Every case is fixed before any fresh observation or forward. Results are
    # not used for any branch/threshold/seed choice within this batch.
    protocol=spec(); print('Published source verified; running fixed 50-simulation diagnostic.',flush=True)
    world,warm,geometry=prepare_world(SEEDS[0]); parity={'id':'disabled-parity','seed':SEEDS[0],'handover_seconds':.03}
    a=execute(parity,'radiomap__original',world,warm,geometry,parity_kernel='original')
    b=execute(parity,'radiomap__disabled',world,warm,geometry,parity_kernel='disabled')
    ra,rb=load_result(a),load_result(b)
    checks={'native_outputs_exact':canonical(ra['native'])==canonical(rb['native']),
        'forecast_hashes_equal':[j['forecast_sha256'] for j in ra['forecast_jobs']]==[j['forecast_sha256'] for j in rb['forecast_jobs']],
        'ready_slots_equal':[j['ready_slot'] for j in ra['forecast_jobs']]==[j['ready_slot'] for j in rb['forecast_jobs']]}
    proof={'passes':all(checks.values()),'checks':checks,'receipts':[reference(OUT/'cases/disabled-parity'/x/'receipt.json')
        for x in ('radiomap__original','radiomap__disabled')]}
    if not (OUT/'parity.json').exists():write(OUT/'parity.json',proof)
    if not proof['passes']:raise RuntimeError('Disabled extension parity failed; no fresh comparisons permitted')
    print('Disabled extension exact packet/forecast/completion parity PASS.',flush=True)
    for case in protocol['cases']:
        world,warm,geometry=prepare_world(case['seed'])
        for arm in case['order']:
            receipt=execute(case,arm,world,warm,geometry)
            print(case['id'],arm,receipt['metrics']['control_miss'],flush=True)
    report=summarize();write(OUT/'report.json',report)
    seal();print('All 50 simulations complete; diagnostic sealed. No promotion.',flush=True)


def summarize():
    rows=[]
    for case in spec()['cases']:
        for arm in ARMS:
            receipt=read(OUT/'cases'/case['id']/arm/'receipt.json');result=load_result(receipt)
            if result['arm']!=arm or result['seed']!=case['seed']:raise RuntimeError('Case membership mismatch')
            rows.append({'case':case['id'],'seed':case['seed'],'handover_seconds':case['handover_seconds'],
                         'arm':arm,'metrics':receipt['metrics']})
    contrasts=[];means=[]
    for ho in (.03,4.):
        for arm in ARMS:
            items=[row['metrics'] for row in rows if row['handover_seconds']==ho and row['arm']==arm]
            means.append({'handover_seconds':ho,'arm':arm,'episodes':len(items),
                          'mean':{key:float(np.mean([item[key] for item in items])) for key in items[0]}})
        for candidate,baseline in spec()['analysis']['contrasts']:
            pair=[]
            for seed in SEEDS:
                a=next(r['metrics'] for r in rows if r['seed']==seed and r['handover_seconds']==ho and r['arm']==candidate)
                b=next(r['metrics'] for r in rows if r['seed']==seed and r['handover_seconds']==ho and r['arm']==baseline)
                pair.append({key:a[key]-b[key] for key in a})
            contrasts.append({'handover_seconds':ho,'candidate':candidate,'baseline':baseline,
                'mean_delta':{key:float(np.mean([x[key] for x in pair])) for key in pair[0]},
                'episodes_with_more_misses':sum(x['control_misses']>0 for x in pair),
                'episodes_with_larger_max_gap':sum(x['maximum_silence_ms']>0 for x in pair)})
    return {'schema':1,'complete':True,'fresh_simulations':48,'disabled_parity_simulations':2,
        'new_fits':0,'rows':rows,'arm_means':means,'contrasts':contrasts,
        'four_simulated_episodes_not_independent_physical_warehouses':True,
        'causal_command_kernel_differs_from_old_native_diagnostic':True,
        'no_JEPA_objective_gain_attributed':True,'default_changed':False,
        'original_SERVICE_opened':False,'production_qualified':False}


def seal():
    receipt=check_sources()
    if read(OUT/'report.json')!=summarize():raise RuntimeError('Report not from actual complete membership')
    prior=prior_identities(verify_bytes=True)
    excluded={'verification/manifest.json'}
    evidence={str(path.relative_to(ROOT)):digest(path) for path in sorted(OUT.rglob('*'))
              if path.is_file() and 'registration/source/' not in str(path.relative_to(OUT))
              and str(path.relative_to(OUT)) not in excluded}
    write(OUT/'verification/manifest.json',{'schema':1,'source_sha256':receipt['source_sha256'],
        'evidence_sha256':evidence,'prior':prior,'fresh_simulations':48,'parity_simulations':2,
        'production_qualified':False,'original_SERVICE_opened':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('write-spec','register','publish','run','summary'))
    parser.add_argument('--commit');parser.add_argument('--branch');args=parser.parse_args()
    if args.command=='write-spec':write_spec();print(OUT/'protocol.json')
    elif args.command=='register':print('registered',len(register()['source_sha256']),'source files')
    elif args.command=='publish':print(publish(args.commit,args.branch))
    elif args.command=='run':run()
    else:print(canonical(summarize()).decode())


if __name__=='__main__':main()
