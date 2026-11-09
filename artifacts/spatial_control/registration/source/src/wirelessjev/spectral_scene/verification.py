"""Passive raw-array audit and byte closure for the spectral RF study.

This verifier never constructs a neural model, traces fields, fits a baseline,
or performs a neural forward. It checks checkpoint tensors as data and replays
public input/target transformations and the declared decision arithmetic.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import common as c,study


def require(condition,message):
    if not condition:raise RuntimeError(message)


def equivalent(actual,expected,label='value'):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),label+' membership differs')
        for key in expected:equivalent(actual[key],expected[key],label+'.'+key)
    elif isinstance(expected,(list,tuple)):
        require(isinstance(actual,(list,tuple)) and len(actual)==len(expected),label+' length differs')
        for index,(a,b) in enumerate(zip(actual,expected)):equivalent(a,b,label+f'[{index}]')
    elif isinstance(expected,(float,np.floating)):
        require(np.isfinite(actual) and np.isclose(actual,expected,rtol=1e-12,atol=1e-12),label+' number differs')
    else:require(actual==expected,label+' differs')


def _identity():
    value=study.identity()
    value.update(data_receipt_sha256=c.digest(c.OUT/'data/prepared-training.json'),
        rt_stage_receipts_sha256={stage:c.digest(c.OUT/'rt_data'/f'{stage}.json') for stage in ('training','validation')})
    # Training uses its own exact identity key schema; imported without constructing a model.
    from . import training
    return training.identity()


def verify_fits():
    from . import training
    selected=study.verified_selection()['selected_prior'];ident=_identity();receipts={}
    expected={f'{cell}-seed{seed}' for cell in (*study.INITIAL,*study.CONTROLS) for seed in study.SEEDS}
    actual={p.stem for p in (c.OUT/'fits').glob('*.json')}
    require(actual==expected,'Finite15 fit membership differs')
    require({p.stem for p in (c.OUT/'checkpoints').glob('*.pt')}==expected,'Checkpoint membership differs')
    for cell in (*study.INITIAL,*study.CONTROLS):
        for seed in study.SEEDS:
            run=f'{cell}-seed{seed}';fit=c.read_json(c.OUT/'fits'/f'{run}.json')
            option=training.settings(cell,selected if cell in study.CONTROLS else None)
            require(fit['identity']==ident and fit['cell']==cell and fit['seed']==seed and fit['run_id']==run,'Fit identity differs')
            require(fit['settings']==option and fit['completed'] is True,'Fit settings/incomplete')
            require(fit['epochs']==60 and fit['updates']==480 and fit['distinct_training_layouts']==48,'Unequal training budget')
            require(len(fit['history'])==60 and [h['epoch'] for h in fit['history']]==list(range(60)),'Epoch history differs')
            require(all(len(h['updates'])==8 for h in fit['history']),'Update budget differs')
            scores=np.array([h['validation_mean_layout_rmse_db'] for h in fit['history']],dtype=float)
            require(np.isfinite(scores).all(),'Nonfinite validation history')
            epoch=int(scores.argmin())
            require(fit['selected_epoch']==epoch and fit['validation_mean_layout_rmse_db']==scores[epoch],'Checkpoint selection differs')
            for history in fit['history']:
                for update in history['updates']:
                    require(all(np.isfinite(value) for value in update.values()),'Nonfinite optimization log')
            path=c.OUT/'checkpoints'/f'{run}.pt'
            require(c.digest(path)==fit['checkpoint_sha256'],'Checkpoint bytes differ')
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            require(checkpoint['identity']==ident and checkpoint['cell']==cell and checkpoint['seed']==seed
                and checkpoint['settings']==option and checkpoint['selected_epoch']==epoch,'Checkpoint metadata differs')
            state=checkpoint['state']
            require(isinstance(state,dict) and state,'Missing raw neural tensors')
            require(all(torch.is_tensor(v) and bool(torch.isfinite(v).all()) for v in state.values()),'Invalid neural tensors')
            require(sum(v.numel() for v in state.values())>=fit['learned_parameters'],'Parameter count inconsistent')
            require(fit['uncertainty_calibrated'] is False,'Fixedspread likelihood mislabeled calibrated')
            receipts[run]=study.reference(c.OUT/'fits'/f'{run}.json')
    return {'fits':15,'independent_seeds_per_cell':3,'epochs':60,'updates_each':480,
            'checkpoint_data_checked_without_model_constructor':True,'receipts':receipts}


def verify_public_arrays(stage):
    """Recreate only whitelisted public array transforms, never model predictions."""
    from . import data
    evaluation=study.verify_evaluation(stage);count=0
    frequency_ref=study.frequency_diagnostic(stage,study.verified_selection()['selected_cell'])
    require(evaluation['same_link_frequency']==frequency_ref,'Direct frequency reference changed')
    for regime,members in evaluation['members'].items():
        examples=study._examples(stage,regime)
        for member in members.values():
            for index,row in enumerate(member['layouts']):
                example=examples[index]
                if isinstance(example,dict):
                    require(regime.startswith('onboarding_') and row=={'scene_id':example['scene_id'],'available':False,'availability':example},'Unavailable onboarding outcome differs from actual support')
                    continue
                require(row.get('available') is True,'Genuine onboarding support was dropped')
                receipt=c.read_json(study.verify_reference(row['prediction']));example=examples[index]
                path=c.safe_path(receipt['npz_path'])
                expected=example.model_inputs()|example.target_arrays()
                expected.update(band_index=example.query_rows.band_index,device_index=example.query_rows.device,
                    pose_index=example.query_rows.pose_index,direction=example.query_rows.direction)
                with np.load(path,allow_pickle=False) as arrays:
                    for name,value in expected.items():
                        require(name in arrays and np.array_equal(arrays[name],value,equal_nan=True),'Public input/target transform differs: '+name)
                    if member['seed'] is not None:
                        require(arrays['scene_latent'].shape==(8,) and np.isfinite(arrays['scene_latent']).all(),'Bad stored latent')
                    require(arrays['context_a'].shape==(24,20) and arrays['context_b'].shape==(24,20),'Wrong scene views')
                meta=receipt['metadata']
                require(meta['query_measurement_ids']==list(example.query_rows.measurement_ids),'Query ID provenance differs')
                require(meta['context_a_measurement_ids']==list(example.context_rows_a.measurement_ids)
                    and meta['context_b_measurement_ids']==list(example.context_rows_b.measurement_ids),'Context ID provenance differs')
                count+=1
    return {'prediction_records_replayed':count,'public_input_target_replay':True,'descriptive_unavailable_support_replayed':True,'model_forwards':0}


def verify_baselines():
    """Check saved coefficient/prior bytes and TRAIN provenance, no adaptation fit."""
    from . import baselines
    admitted=baselines.verify_prior()
    path=c.OUT/'baselines/prior.json';value=c.read_json(path)
    # The complete receipt and numerical coefficient archive are closure-bound.
    arrays_path=c.OUT/'baselines/prior.npz'
    require(arrays_path.is_file(),'Missing actual classical baseline coefficients')
    with np.load(arrays_path,allow_pickle=False) as arrays:
        require(bool(arrays.files),'Empty classical prior')
        for name in arrays.files:
            a=arrays[name]
            if np.issubdtype(a.dtype,np.number):require(np.isfinite(a).all(),'Nonfinite classical coefficients')
    # Source/data hash maps are checked recursively, accommodating documented metadata.
    def walk(node):
        if isinstance(node,dict):
            if set(('path','sha256'))<=set(node):study.verify_reference(node)
            for key,val in node.items():
                if key.endswith('sha256') and isinstance(val,dict):
                    for name,sha in val.items():
                        if '/' in name and isinstance(sha,str) and len(sha)==64:
                            require(c.digest(c.safe_path(name))==sha,'Classical provenance differs')
                walk(val)
        elif isinstance(node,list):
            for val in node:walk(val)
    walk(value)
    return {'admitted':admitted,'receipt':study.reference(path),'coefficients':study.reference(arrays_path),
        'baseline_fit_or_prediction_calls':0,'posterior_calibration_claim':False}


@contextmanager
def passive_guard():
    """Tripwires prevent an audit from generating its own missing evidence."""
    from . import data,training,baselines
    from ..radiojepa import rt_data
    def forbidden(*args,**kwargs):raise RuntimeError('Passive verification cannot fit, trace, construct or infer')
    def existing(path,value):
        path=Path(path)
        require(path.is_file() and path.read_bytes()==c.encoded(value),'Passive replay needs existing identical receipt')
        return path
    with patch.object(c,'write_once',existing),patch.object(torch.nn.Module,'__init__',forbidden),\
         patch.object(training,'fit',forbidden),patch.object(training,'load',forbidden),\
         patch.object(training,'predict',forbidden),patch.object(training,'encode',forbidden),\
         patch.object(baselines,'prepare',forbidden),patch.object(baselines,'predict',forbidden),\
         patch.object(rt_data,'_trace_scene',forbidden),patch.object(data,'fit_frontends',forbidden):
        yield


def verify_campaign():
    c.check_sources();before=c.verify_previous()
    with passive_guard():
        for stage in ('training','validation','development'):study.validate_stage(stage)
        selection=study.verified_selection();fits=verify_fits();baselines=verify_baselines()
        development=study.verified_development();public={'development':verify_public_arrays('development')}
        qualification=c.read_json(c.OUT/'qualification.json')
        expected_qualification={k:development[k] for k in ('field_test_permitted','source_receipt_sha256','registry_sha256','original_service_confirmation_permitted')}
        expected_qualification.update(development_sha256=c.digest(c.OUT/'development.json'),production_qualified=False)
        equivalent(qualification,expected_qualification,'qualification')
        report=c.read_json(c.OUT/'report.json')
        if development['field_test_permitted']:
            study.validate_stage('field_test');evaluation=study.verify_evaluation('field_test')
            expected=study.field_results(evaluation,selection['selected_cell'])
            equivalent(report,expected,'field report');public['field_test']=verify_public_arrays('field_test')
        else:
            require(not (c.OUT/'rt_data/field_test').exists() and not (c.OUT/'rt_data/field_test.json').exists(),'Vetoed FIELD geometry consumed')
            require(not (c.OUT/'predictions/field_test').exists(),'Vetoed FIELD predictions exist')
            expected={'schema':1,'identity':study.identity(),'status':'development_veto',
                'development':study.reference(c.OUT/'development.json'),'field_test_opened':False,
                'rf_rmse_superiority':False,'matched_alignment_benefit_supported':False,
                'production_qualified':False,'original_service_confirmation_permitted':False}
            equivalent(report,expected,'veto report')
    require(c.verify_previous()==before,'Previous sealed evidence changed during new audit')
    return {'schema':1,'passes':True,'source_receipt_sha256':c.digest(c.OUT/'registration/source.json'),
        'previous':before,'fit_audit':fits,'baseline_audit':baselines,'public_array_audit':public,
        'selected_cell':selection['selected_cell'],'field_test_opened':development['field_test_permitted'],
        'original_service_confirmation_opened':False,'model_constructors_fits_forwards_or_rt_solves':0,
        'claims_are_field_only_not_service_or_production':True}


def _visible():
    for p in c.OUT.rglob('*'):
        require(not p.is_symlink(),'Redirected spectral evidence member')
    return {str(p.relative_to(c.ROOT)):c.digest(p) for p in sorted(c.OUT.rglob('*')) if p.is_file()
            and not p.is_symlink() and not p.name.endswith(('.lock','.part')) and not p.name.startswith('.')
            and str(p.relative_to(c.OUT))!='verification/manifest.json'}


def run_audit():
    value=verify_campaign();c.write_once(c.OUT/'verification/audit.json',value);return value


def seal(documents=()):
    c.check_sources();c.verify_previous()
    audit=c.OUT/'verification/audit.json'
    require(audit.exists(),'Run the passive raw audit before sealing')
    equivalent(c.read_json(audit),verify_campaign(),'audit')
    documents=tuple(Path(p) for p in documents)
    value={'schema':1,'source_sha256':c.source_hashes(),'evidence_sha256':_visible(),
        'documents_sha256':{str(p.relative_to(c.ROOT)):c.digest(p) for p in documents},
        'audit_sha256':c.digest(audit),'previous_manifest_sha256':c.PREVIOUS_MANIFEST_SHA256}
    c.write_once(c.OUT/'verification/manifest.json',value)
    return verify_closure()


def verify_closure():
    c.check_sources();c.verify_previous()
    value=c.read_json(c.OUT/'verification/manifest.json')
    require(value['source_sha256']==c.source_hashes(),'Sealed spectral source differs')
    require(value['evidence_sha256']==_visible(),'Sealed spectral evidence bytes/membership differ')
    for name,sha in value['documents_sha256'].items():require(c.digest(c.safe_path(name))==sha,'Sealed output document differs')
    require(value['audit_sha256']==c.digest(c.OUT/'verification/audit.json'),'Audit receipt differs')
    return {'passes':True,'source_files':len(value['source_sha256']),'evidence_files':len(value['evidence_sha256']),
        'documents':len(value['documents_sha256']),'original_service_confirmation_opened':False}
