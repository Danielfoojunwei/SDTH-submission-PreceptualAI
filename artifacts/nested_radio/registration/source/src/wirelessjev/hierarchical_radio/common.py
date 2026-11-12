"""Published-source authorization and immutable artifacts for one finite study."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'artifacts/hierarchical_radio'
WORK = Path('/workspace/training/hierarchical-radio')
PROTOCOL = OUT/'protocol.json'
REGISTRY = OUT/'registry.json'
SEEDS = (0, 1, 2)
NATIVE_SEEDS = (90512001, 90512002, 90512003, 90512004)
PRIOR = ('radiojepa','radiojepa_spectral','device_transfer','channel_scene',
         'adaptive_radio','native_bridge','e7_calplanning','spatial_control')
_PUBLICATION = set()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def encoded(value):
    def cast(item):
        if isinstance(item,np.ndarray): return item.tolist()
        if isinstance(item,np.generic): return item.item()
        raise TypeError(type(item).__name__)
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False,default=cast)+'\n').encode()


def read(path): return json.loads(Path(path).read_bytes())


def write_bytes(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        f.write(value);f.flush();os.fsync(f.fileno())


def write(path,value): write_bytes(path,encoded(value))
write_once = write


def atomic_write(path,value):
    """Only mutable WORK journals use atomic replacement, never evidence."""
    path=Path(path)
    if not path.resolve().is_relative_to(WORK.resolve()):raise ValueError('Only WORK journals can replace data')
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temp=tempfile.mkstemp(prefix='.atomic-',dir=path.parent)
    try:
        with os.fdopen(descriptor,'wb') as f:f.write(encoded(value));f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def reference(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():raise ValueError('Repository file reference required')
    return {'path':str(path.relative_to(ROOT)),'sha256':digest(path)}


def safe_path(name):
    path=Path(name)
    if (not isinstance(name,str) or path.is_absolute() or '..' in path.parts
            or path.as_posix()!=name or not (ROOT/path).resolve().is_relative_to(ROOT.resolve())):
        raise ValueError('Canonical repository path required')
    return ROOT/path


def npz(path,values):
    stream=io.BytesIO();np.savez_compressed(stream,**values);write_bytes(path,stream.getvalue())
    return reference(path)


def arrays(path):
    with np.load(path,allow_pickle=False) as f:return {name:f[name].copy() for name in f.files}


def array_digest(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()


def protocol_spec():
    from . import simulation as sim
    from .model import CELLS, ALIGNMENT_WEIGHT
    return {'schema':1,'purpose':'learned_shared_and_AP_local_spatial_posterior_with_matched_controls',
        'rt':{'counts':dict(sim.COUNTS),'states':list(sim.STATES),'actual_cases':186,
            'main_APs':4,'transfer_APs':8,'rx_poses':48,'context_poses':list(sim.CONTEXT_POSES),
            'query_poses':list(sim.QUERY_POSES),'full_source_tokens':192,'sparse_source_tokens':96,
            'source_carriers_hz':[2.4e9,5.2e9],'held_carrier_hz':6.0e9,
            'samples_per_source':32768,'max_depth':3,'max_paths_per_source':4096,
            'backend':'llvm_ad_mono_polarized','drjit_threads':4,
            'localized_material_intervention':True,'geometry_TX_solver_seed_shared_within_states':True,
            'static_states_not_temporal_transitions':True,'top_k_per_source_band':2,
            'unknown_paths_never_censored':True,'receiver_measurement_sigma_db':.75},
        'training':{'cells':list(CELLS),'seeds':list(SEEDS),'neural_fits':15,
            'epochs':40,'updates_per_epoch':12,'committed_updates_per_fit':480,'batch_layouts':4,
            'maximum_spawn_workers':3,'torch_threads_per_worker':1,
            'one_random_state_per_distinct_batch_layout':True,'regimes_alternate_full_sparse':True,
            'optimizer':'AdamW','learning_rate':.003,'weight_decay':1e-4,'gradient_clip_norm':5.,
            'validation_epochs':[10,20,30,40],'checkpoint_rule':'first_minimum_sparse_VAL_A_composite_interval_NLL',
            'branch_rule':'hier_reg_or_hier_joint_by_mean_three_member_sparse_VAL_NLL',
            'alignment_weight':ALIGNMENT_WEIGHT,'alignment_on_shared_supported_AP_query_probes':True,
            'teacher_EMA_claimed':False,'resume':'last_atomically_committed_optimizer_state_and_RNG; uncommitted computation may repeat; no new trial',
            'nonfinite_failure_retry':False},
        'model':{'global_dimensions':8,'AP_local_dimensions':16,'width':32,'interval_Newton_steps':3,
            'inference_method':'observed_interval_likelihood_Newton_Laplace',
            'backtracking_step_candidates':[1.,.5,.25,.125,.0625,.03125],
            'backtracking_rule':'first_nonincreasing_context_negative_log_posterior; else retain current iterate',
            'backtracking_roundoff_tolerance_relative':1e-10,
            'fixed_RFF_seed':184910,'prior_jitter':1e-8,'full_covariance_query_limit':512,
            'global_reg_dimensions':24,'global8_reg_capacity_control':8,
            'AP_IDs_only_route_blocks':True,'no_current_RF_conditioned_prior_double_count':True,
            'float64_block_Schur_conditioning':True,'covariance_approximate_for_interval_and_correlated_reports':True,
            'erased':'heard retains genuine detector lower event; ranked-unheard amplitudes become uninformative; flags retained',
            'predictions_never_reinserted_as_measurements':True,'physical_scatterer_identification_claimed':False},
        'numerical':{'radiomap_lengthscales_m':[5.,10.,20.],
            'selection':'first_minimum_mean_sparse_VAL_composite_interval_NLL',
            'prior_fits_TRAIN_only':True,'same_admitted_A_context':True,
            'fixed_hierarchy_reference':True,'native_RadioMap_bit_identity_claimed_for_RT_operator':False},
        'evaluation':{'primary_regime':'sparse','primary_metric':'heard_reported_RSSI_RMSE_per_whole_layout',
            'primary_layouts':24,'states_queries_APs_nested':True,'member_metrics_averaged_not_predictions':True,
            'contrasts':[['selected_hierarchy','global_reg'],['selected_hierarchy','radiomap_operator'],
                         ['hier_joint','hier_reg'],['hier_reg','hier_erased']],
            'bootstrap_samples':20000,'bootstrap_seed':21770,'central_interval_percent':98.75,
            'family_comparisons':4,'conditional_on_three_trained_members':True,
            'secondary':['full_context','localized_state_delta','heard_p95','empirical_heard_interval_coverage',
                'held_6GHz_sparse_selected_models','8AP_count_stress_equal_total_context','stale_state0_context'],
            'heard_topK_Brier_from_detector_only_FORBIDDEN':True,
            'DEV_gate':'technical_finite_complete_only; no efficacy screening or TEST tuning'},
        'native':{'simulation_only':True,'seeds':list(NATIVE_SEEDS),'robots':100,'duration_seconds':20,
            'handover_seconds':[.03,4.],'methods':['full_radiomap','matched_radiomap','hierarchy','global_reg'],
            'fresh_simulations':32,'parity_simulations':2,'maximum_simulations':34,
            'weights':'seed0_VAL_selected_hierarchy_and_global_reg; no native selection',
            'context_tokens':192,'source_bands':[0,1],'shared_selector':True,
            'model_coordinate_offset_xy_m':[60.,40.],
            'model_normalized_RX_AP_XY_offset':.5,
            'physical_plan_and_support_coordinates_remain_native':True,
            'support_on':['matched_radiomap','hierarchy','global_reg'],'local_recovery':False,
            'kernel':'unchanged spatial_control.kernel causal_central=True',
            'profile_seconds':{'cold':1.,'warm':.25,'refit':1.},'shared_extra_seconds':.5,
            'parity_profile_seconds':4.,'collision_lambda':0.,'interference':False,
            'actual_6point5GHz_distinct_from_RT6point0':True,
            'analysis':'all four paired procedural episodes and harms per HO; descriptive not SLA'},
        'original_SERVICE_opened':False,'production_qualified':False,'default_changed':False,
        'real_hardware_actions':0,'scope':'static conditional RF model and separate simulated warehouse transfer; not universal or temporally trained foundation model'}


def source_receipt_sha256():return digest(OUT/'registration/source.json')
def protocol_sha256():return digest(PROTOCOL)
def registry_sha256():return digest(REGISTRY)


def prior_identities(*,verify_bytes=False):
    union={};manifests={}
    for name in PRIOR:
        path=ROOT/'artifacts'/name/'verification/manifest.json';value=read(path);manifests[name]=reference(path)
        for field in ('source_sha256','evidence_sha256','documents_sha256'):
            for item,sha in value.get(field,{}).items():
                if item in union and union[item]!=sha:raise RuntimeError('Previous closures disagree')
                union[item]=sha
    for name in ('artifacts/radiojepa/rt_data/confirmation','artifacts/radiojepa/rt_data/confirmation.json'):
        if (ROOT/name).exists():raise RuntimeError('Original SERVICE holdout must stay unopened')
    if verify_bytes:
        for name,sha in union.items():
            if digest(safe_path(name))!=sha:raise RuntimeError('Prior frozen file changed: '+name)
    return {'manifests':manifests,'unique_repository_files':len(union),'all_named_bytes_checked':verify_bytes}


def source_hashes(names=None):
    tops=('src/wirelessjev/hierarchical_radio','tests/hierarchical_radio')
    own={str(p.relative_to(ROOT)) for top in tops for p in (ROOT/top).rglob('*.py')}
    if names is None:
        tracked=subprocess.check_output(['git','ls-files','--','src','tests','scripts','pyproject.toml','uv.lock'],cwd=ROOT,text=True).splitlines()
        names={n for n in tracked if n.endswith('.py') or n in ('pyproject.toml','uv.lock')}|own
        names|={'scripts/run_hierarchical_radio.py','docs/hierarchical_radio/PLAN.md',
                'artifacts/hierarchical_radio/protocol.json','artifacts/hierarchical_radio/registry.json'}
    else:
        names=set(names)
        expected={n for n in names if n.endswith('.py') and any(n.startswith(t+'/') for t in tops)}
        if own!=expected:raise RuntimeError('Own source/test module membership changed')
    return {name:digest(safe_path(name)) for name in sorted(names)}


def write_spec():
    from . import simulation as sim
    for path,value in ((REGISTRY,sim.geometry_registry()),(PROTOCOL,protocol_spec())):
        if path.exists():
            if read(path)!=value:raise RuntimeError('Registered specification changed')
        else:write(path,value)


def register():
    write_spec();target=OUT/'registration/source.json'
    if target.exists():return check_sources(published=False)
    if any((OUT/name).exists() for name in ('rt_data','fits','data','native','predictions')):
        raise RuntimeError('Register before scientific data, fits and outputs')
    source=source_hashes();prior=prior_identities(verify_bytes=True)
    tests=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'tests/hierarchical_radio').glob('test_*.py'))
    if not tests:raise RuntimeError('No software preflight tests')
    result=subprocess.run([sys.executable,'-m','pytest','-q',*tests],cwd=ROOT,capture_output=True,text=True)
    log=OUT/'preflight/tests.log';write_bytes(log,(result.stdout+result.stderr).encode())
    if result.returncode or source_hashes()!=source:raise RuntimeError('Software preflight failed or source changed')
    proof={'exit_code':0,'source_sha256':source,'log':reference(log),'fresh_scientific_cases_or_fits':0}
    write(OUT/'preflight/tests.json',proof)
    for name in source:write_bytes(OUT/'registration/source'/name,safe_path(name).read_bytes())
    receipt={'schema':1,'source_sha256':source,'protocol':reference(PROTOCOL),'registry':reference(REGISTRY),
        'preflight':reference(OUT/'preflight/tests.json'),'preflight_log':reference(log),'prior':prior,
        'before_fresh_RT_training_or_predictions':True}
    write(target,receipt);return receipt


def check_sources(*,published=True):
    receipt=read(OUT/'registration/source.json')
    names=receipt['source_sha256']
    if source_hashes(names)!=names:raise RuntimeError('Frozen source changed')
    capsule=OUT/'registration/source'
    if {str(p.relative_to(capsule)) for p in capsule.rglob('*') if p.is_file()}!=set(names):
        raise RuntimeError('Source capsule membership changed')
    for name,sha in names.items():
        if digest(capsule/name)!=sha:raise RuntimeError('Source capsule bytes changed')
    for key in ('protocol','registry','preflight','preflight_log'):
        item=receipt[key]
        if reference(safe_path(item['path']))!=item:raise RuntimeError('Bound dependency changed: '+key)
    if prior_identities()['manifests']!=receipt['prior']['manifests']:raise RuntimeError('Previous manifest changed')
    if published:
        proof=read(OUT/'registration/git.json');commit=proof['commit'];identity=(commit,source_receipt_sha256())
        if (proof['source_receipt_sha256']!=source_receipt_sha256() or commit!=proof['remote_observed_commit']
                or not re.fullmatch('[0-9a-f]{40}',commit)):
            raise RuntimeError('Publication identity changed')
        if identity not in _PUBLICATION:
            published_receipt=subprocess.check_output(['git','show',commit+':artifacts/hierarchical_radio/registration/source.json'],cwd=ROOT)
            if hashlib.sha256(published_receipt).hexdigest()!=source_receipt_sha256():
                raise RuntimeError('Published source receipt differs from the bound receipt')
            for name,sha in names.items():
                data=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
                if hashlib.sha256(data).hexdigest()!=sha:raise RuntimeError('Published source bytes differ')
            _PUBLICATION.add(identity)
    return receipt


def record_publication(commit,branch):
    receipt=check_sources(published=False)
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact commit required')
    actual=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()
    if not actual or actual[0]!=commit:raise RuntimeError('Source commit not observed on remote')
    published_receipt=subprocess.check_output(['git','show',commit+':artifacts/hierarchical_radio/registration/source.json'],cwd=ROOT)
    if hashlib.sha256(published_receipt).hexdigest()!=source_receipt_sha256():
        raise RuntimeError('Commit did not publish registered source receipt')
    for name,sha in receipt['source_sha256'].items():
        data=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
        if hashlib.sha256(data).hexdigest()!=sha:raise RuntimeError('Commit did not publish registered source')
    proof={'commit':commit,'remote':'origin','branch':branch,'remote_observed_commit':commit,
           'source_receipt_sha256':source_receipt_sha256()}
    write(OUT/'registration/git.json',proof);return proof


def case_path(spec):return OUT/'rt_data/cases'/spec.case_id


def authorize(spec,folder=None):
    """Called before every new physical solve; TEST permission is technical only."""
    from . import simulation as sim
    check_sources()
    expected=sim.case_spec(spec.stage,spec.index,spec.state)
    if spec!=expected:raise ValueError('Unregistered physical case')
    if spec.stage in ('test','ap_transfer'):
        from .study import verify_development
        verify_development()  # binds all fits/checkpoints, raw VAL scores and DEV membership
        gate=read(OUT/'development_gate.json')
        if (gate.get('technical_pass') is not True or gate['source_receipt_sha256']!=source_receipt_sha256()
                or gate['selection_sha256']!=digest(OUT/'selection.json')):
            raise RuntimeError('Technical DEV gate and immutable VAL selection required')
    path=case_path(spec)
    if folder is not None and Path(folder).resolve()!=path.resolve():raise ValueError('Alternate case path forbidden')
    return {'authorized':True,'case_id':spec.case_id,'case_path':str(path.resolve()),
            'source_receipt_sha256':source_receipt_sha256(),'registry_sha256':registry_sha256()}


def attempt(path,identity):
    write(path,dict(identity,created_unix_ns=time.time_ns()))

