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
OUT = ROOT/'artifacts/nested_radio'
WORK = Path('/workspace/training/nested-radio')
PROTOCOL = OUT/'protocol.json'
REGISTRY = OUT/'registry.json'
SEEDS = (0, 1, 2)
PARENT = ROOT/'artifacts/hierarchical_radio'
PRIOR = ('radiojepa','radiojepa_spectral','device_transfer','channel_scene',
         'adaptive_radio','native_bridge','e7_calplanning','spatial_control','hierarchical_radio')
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
    return {'schema':1,'purpose':'zero_capable_AP_residual_and_fixed_larger_budget_prospective_RF_test',
        'reused_data':{'parent':'hierarchical_radio','TRAIN_layouts':24,'VAL_layouts':6,'states_per_layout':3,
            'only_parent_TRAIN_VAL_and_TRAIN_calibration_for_fitting':True,'prior_TEST_not_confirmation':True},
        'model':{'cells':['global8_reg','global8_joint','hier_reg','nested_reg','nested_joint'],
            'global_dimensions':8,'AP_local_dimensions':16,'width':32,'inference_steps':3,
            'nested_gate':'16 shared per-feature finite parameters in0..1; forward requires bounds; projected after every update',
            'gate_initial':.1,'gate_penalty':'.01*mean(alpha*sqrt(local_mean^2+local_SD^2)/.5) on nested cells only',
            'gate_alone_not_identified_or_functional_shrinkage_certificate':True,
            'effective_prior_mean_SD_RMS_diagnostics':True,
            'local_prior_SD_floor_unchanged':True,'zero_gate_exact_global8_function_class':True,
            'one_gate_exact_legacy_hierarchy':True,'generalization_noninferiority_guaranteed':False,
            'gate_zero_can_be_absorbing_under_gradient_optimization':True,
            'joint_alignment_weight':.1,'joint_same_two_view_target_likelihood':True,'EMA_teacher':False,
            'posterior':'frozen observed interval Newton/Laplace with block Schur solves',
            'predictions_never_become_measurements':True},
        'training':{'seeds':[0,1,2],'fits':15,'epochs':160,'updates_per_epoch':12,
            'committed_updates_per_fit':1920,'total_committed_updates':28800,'batch_distinct_layouts':4,
            'one_random_state_per_layout':True,'alternating_full_sparse':True,
            'optimizer':'AdamW','learning_rate':.003,'weight_decay':1e-4,'gradient_clip':5.,
            'checkpoints':[40,80,120,160],'selection':'first minimum equal-layout/state sparse VAL A heard-RSSI RMSE',
            'NLL_secondary_not_selection':True,'all_candidate_weights_and_raw_VAL_predictions_archived':True,
            'eligible_cells':['global8_reg','global8_joint','nested_reg','nested_joint'],
            'cell_selection':'first minimum mean of all three member VAL heard RMSEs; ties in eligible-cell order',
            'max_spawn_workers':3,'threads_per_worker':1,'deterministic_torch':True,
            'resume':'same optimizer/RNG trajectory from last atomic committed step; no replacement trials',
            'nonfinite_retry':False},
        'numerical':{'prior_fit_runs':0,'reused_TRAIN_fit_priors':['radiomap-5m','radiomap-10m','radiomap-20m','hierarchy-fixed'],
            'selection':'first minimum equal-layout/state sparse VAL heard-RSSI RMSE, same as neural',
            'best_numerical_can_be_fixed_hierarchy':True,'all_four_controls_retained':True},
        'rt':{'new_layouts':{'test':24,'ap_transfer':4},'states_per_layout':3,'fresh_cases':84,
            'disjoint_from_prior_public_geometries':240,'main_APs':4,'transfer_APs':8,
            'RX_poses':48,'context_query_checkerboard':True,'full_tokens':192,'sparse_tokens':96,
            'source_carriers_hz':[2.4e9,5.2e9],'held_carrier_hz':6.e9,'top_k':2,
            'ranked_unheard_upper_bound':'kth selected quantization upper edge when full; detector floor otherwise',
            'unresolved_paths_remain_unknown':True,'receiver_noise_sigma_db':.75,
            'samples_per_source':32768,'max_depth':3,'max_paths_per_source':4096,
            'backend':'llvm_ad_mono_polarized','drjit_threads':4,'same_static_CAD_TX_and_solver_noise_streams_within_states':True},
        'evaluation':{'primary':'sparse heard reported RSSI RMSE','layouts':24,
            'contrasts':[['selected','best_numerical'],['selected','global8_reg'],['nested_reg','hier_reg'],['nested_joint','nested_reg']],
            'selected_can_equal_global8_reg_then_exact_null_not_gain':True,'member_metrics_then_states_within_layout':True,
            'bootstrap_samples':20000,'bootstrap_seed':21771,'central_interval_percent':98.75,
            'conditional_on_three_trained_members':True,'primary_model_selection_never_TEST':True,
            'technical_gate':'all15complete fits and finite complete raw VAL archives; no efficacy screen',
            'secondary':['full context','localized material deltas','held6GHz','8AP fixed total reports','stale state0 context','heard coverage and p95'],
            'VAL_representation_diagnostics':['fixed Fourier8 vs ungated learned8 vs gated local16 feature variation/effective rank',
                '40/80/120/160 learning curves and gate values; not a collapse/convergence certificate']},
        'native_simulations':0,'real_hardware_actions':0,'original_SERVICE_opened':False,
        'production_qualified':False,'default_changed':False,'scope':'static RF only; any packet, temporal or physical qualification needs a separate protocol'}


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
    tops=('src/wirelessjev/nested_radio','tests/nested_radio')
    own={str(p.relative_to(ROOT)) for top in tops for p in (ROOT/top).rglob('*.py')}
    if names is None:
        tracked=subprocess.check_output(['git','ls-files','--','src','tests','scripts','pyproject.toml','uv.lock'],cwd=ROOT,text=True).splitlines()
        names={n for n in tracked if n.endswith('.py') or n in ('pyproject.toml','uv.lock')}|own
        names|={'scripts/run_nested_radio.py','docs/nested_radio/PLAN.md',
                'artifacts/nested_radio/protocol.json','artifacts/nested_radio/registry.json'}
    else:
        names=set(names)
        expected={n for n in names if n.endswith('.py') and any(n.startswith(t+'/') for t in tops)}
        if own!=expected:raise RuntimeError('Own source/test module membership changed')
    return {name:digest(safe_path(name)) for name in sorted(names)}


def write_spec():
    from . import data as sim
    for path,value in ((REGISTRY,sim.geometry_registry()),(PROTOCOL,protocol_spec())):
        if path.exists():
            if read(path)!=value:raise RuntimeError('Registered specification changed')
        else:write(path,value)


def register():
    write_spec();target=OUT/'registration/source.json'
    if target.exists():return check_sources(published=False)
    if any((OUT/name).exists() for name in ('rt_data','fits','data','native','predictions','numerical',
            'attempts','checkpoints','epoch_checkpoints','validation_predictions','diagnostic_attempts',
            'representation_predictions','selection.json','validation_representation.json','report.json')):
        raise RuntimeError('Register before scientific data, fits and outputs')
    source=source_hashes();prior=prior_identities(verify_bytes=True)
    tests=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'tests/nested_radio').glob('test_*.py'))
    if not tests:raise RuntimeError('No software preflight tests')
    result=subprocess.run([sys.executable,'-m','pytest','--import-mode=importlib','-q',*tests],cwd=ROOT,capture_output=True,text=True)
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
            published_receipt=subprocess.check_output(['git','show',commit+':artifacts/nested_radio/registration/source.json'],cwd=ROOT)
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
    published_receipt=subprocess.check_output(['git','show',commit+':artifacts/nested_radio/registration/source.json'],cwd=ROOT)
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
    from . import data
    check_sources()
    if spec!=data.case_spec(spec.stage,spec.index,spec.state):raise ValueError('Unregistered fresh physical case')
    from .study import verify_selection
    verify_selection()  # passive fit/checkpoint/rawVAL proof, no TEST score or efficacy screen
    path=case_path(spec)
    if folder is not None and Path(folder).resolve()!=path.resolve():raise ValueError('Alternate case path forbidden')
    return {'authorized':True,'case_id':spec.case_id,'case_path':str(path.resolve()),
            'source_receipt_sha256':source_receipt_sha256(),'registry_sha256':registry_sha256()}


def attempt(path,identity):
    write(path,dict(identity,created_unix_ns=time.time_ns()))



def parent_data_references():
    """Named fitting inputs only; no parent TEST prediction/target enters fitting."""
    fixed=['data/prepared.json','data/calibration.json','rt_data/training.json','rt_data/validation.json']
    fixed += ['data/groups/'+stage+'-'+regime+suffix for stage in ('training','validation')
              for regime in ('full','sparse') for suffix in ('.json','.npz')]
    result={str((PARENT/name).relative_to(ROOT)):digest(PARENT/name) for name in fixed}
    for stage in ('training','validation'):
        manifest=read(PARENT/'rt_data'/f'{stage}.json')
        for item in manifest['case_receipts'].values():
            result[item['path']]=item['sha256']
            receipt=read(safe_path(item['path']));folder=safe_path(item['path']).parent
            for name,sha in receipt['files_sha256'].items():
                if name=='public_views.npz' or name.startswith('scene/meshes/'):
                    result[str((folder/name).relative_to(ROOT))]=sha
    return result


def verify_parent_data():
    parent_manifest=read(PARENT/'verification/manifest.json')
    union={**parent_manifest['source_sha256'],**parent_manifest['evidence_sha256']}
    refs=parent_data_references()
    for name,sha in refs.items():
        if union.get(name)!=sha or digest(safe_path(name))!=sha:
            raise RuntimeError('Frozen parent fitting input changed: '+name)
    return {'parent_manifest':reference(PARENT/'verification/manifest.json'),
            'parent_source_receipt':reference(PARENT/'registration/source.json'),
            'fitting_input_sha256':refs,'TRAIN_cases':72,'VAL_cases':18,'parent_TEST_used_for_selection':False}
