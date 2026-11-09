"""Passive adaptive-RF raw array audit and immutable byte closure."""
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
        for i,(a,b) in enumerate(zip(actual,expected)):equivalent(a,b,label+f'[{i}]')
    elif isinstance(expected,(float,np.floating)):
        require(np.isfinite(actual) and np.isclose(actual,expected,rtol=1e-12,atol=1e-12),label+' differs')
    else:require(actual==expected,label+' differs')


def verify_fits():
    from . import training
    ident=training.identity();expected={f'{cell}-seed{s}' for cell in study.CELLS for s in study.SEEDS}
    require({p.stem for p in (c.OUT/'fits').glob('*.json')}==expected,'Exact12fit membership differs')
    require({p.stem for p in (c.OUT/'checkpoints').glob('*.pt')}==expected,'Checkpoint membership differs')
    for cell in study.CELLS:
        for seed in study.SEEDS:
            run=f'{cell}-seed{seed}';fit=c.read_json(c.OUT/'fits'/f'{run}.json');path=c.OUT/'checkpoints'/f'{run}.pt'
            require(fit['identity']==ident and fit['cell']==cell and fit['seed']==seed and fit['run_id']==run,'Fit source/cell/seed identity differs')
            require(fit['settings']=={'cell':cell,'width':64,'em_steps':3} and fit['batch_distinct_layouts']==24 and fit['training_bands']==[0,1], 'Fit model/information budget differs')
            require(fit['uncertainty_calibrated'] is False and fit['packet_or_production_qualified'] is False,'Forecast uncertainty overclaimed')
            require(fit['completed'] is True and fit['epochs']==60 and fit['updates']==480,'Fit budget incomplete')
            history=fit['history'];require(len(history)==60 and [h['epoch'] for h in history]==list(range(60)),'Epoch history differs')
            require(all(len(h['updates'])==8 for h in history),'Update budget differs')
            scores=np.array([h['validation_mean_layout_delta_rmse_db'] for h in history],dtype=float)
            require(np.isfinite(scores).all(),'Nonfinite checkpoint scores')
            require(fit['selected_epoch']==int(scores.argmin()) and fit['validation_mean_layout_delta_rmse_db']==float(scores.min()),'SparseVAL checkpoint selection differs')
            for epoch in history:
                for update in epoch['updates']:require(all(np.isfinite(v) for v in update.values()),'Nonfinite optimization log')
            require(c.digest(path)==fit['checkpoint_sha256'],'Checkpoint bytes differ')
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            require(checkpoint['identity']==ident and checkpoint['cell']==cell and checkpoint['seed']==seed and checkpoint['selected_epoch']==fit['selected_epoch'] and checkpoint['settings']==fit['settings'],'Checkpoint metadata differs')
            tensors=checkpoint['state'];require(isinstance(tensors,dict) and tensors,'Missing learned tensors')
            require(all(torch.is_tensor(t) and bool(torch.isfinite(t).all()) for t in tensors.values()),'Invalid learned tensors')
    return {'fits':12,'independent_seeds_per_cell':3,'updates_each':480,'checkpoint_data_checked_without_model_constructor':True}


def verify_public_arrays(stage):
    from . import data,baselines
    evaluation=study.verify_evaluation(stage);count=0
    for assay,members in evaluation['members'].items():
        held=assay.endswith('-held_band');regime='sparse' if held else assay
        by_stale={}
        for member in members.values():
            stale=member['stale']
            if stale not in by_stale:by_stale[stale]=study.examples(stage,regime,stale=stale,held_band=held)
            vals=by_stale[stale]
            require(len(vals)==len(member['cases']),'Raw case membership differs')
            for example,ref in zip(vals,member['cases']):
                receipt=c.read_json(study.verify_reference(ref));meta=receipt['metadata'];path=c.safe_path(receipt['npz_path'])
                require(meta['example_binding']==data.example_binding(example),'Actual reported raw input/target bindings differ')
                expected=study._input_arrays(example)|example.target_arrays()
                expected.update({name:getattr(example.query_rows,name) for name in ('pose_index','ap_index','band_index','direction','device')})
                with np.load(path,allow_pickle=False) as arrays:
                    for name,value in expected.items():
                        require(name in arrays and np.array_equal(arrays[name],value,equal_nan=True),'Cached array differs from actual reported transformation: '+name)
                    if member['cell']=='frozen_old':
                        eligibility=baselines.frozen_eligibility(example)
                        require(meta['eligibility']==eligibility and meta['available']==eligibility['eligible'],'Supplementary adapter abstention differs')
                        if not eligibility['eligible']:require(np.isnan(arrays['mean']).all() and np.isnan(arrays['std']).all(),'Abstention fabricated a forecast')
                    # Detection thresholds are scoring labels, never model inputs.
                    require('detection_threshold_dbm' not in study._input_arrays(example),'Detector target leaked into model operands')
                count+=1
    return {'prediction_records_replayed':count,'public_input_target_and_schedule_replay':True,'neural_or_RT_calls':0}


@contextmanager
def passive_guard():
    from . import training,baselines,data,simulation
    def forbidden(*args,**kwargs):raise RuntimeError('Passive verification cannot fit, construct, infer or ray trace')
    def existing(path,value):
        path=Path(path);require(path.is_file() and path.read_bytes()==c.encoded(value),'Passive replay needs an existing identical receipt');return path
    with patch.object(c,'write_once',existing),patch.object(torch.nn.Module,'__init__',forbidden),\
        patch.object(training,'fit',forbidden),patch.object(training,'load',forbidden),\
        patch.object(training,'predict',forbidden),patch.object(training,'encode',forbidden),\
        patch.object(baselines,'prepare',forbidden),patch.object(baselines,'predict',forbidden),\
        patch.object(baselines,'predict_frozen',forbidden),patch.object(simulation,'trace_case',forbidden):
        yield


def verify_campaign():
    from . import baselines
    c.check_sources();prior=c.verify_previous()
    with passive_guard():
        for stage in ('training','validation','development'):study.validate_stage(stage)
        fits=verify_fits();selection=study.verified_selection();baseline=baselines.verify_prior()
        dev=study._verified_development_uncached();public={'development':verify_public_arrays('development')}
        expected={k:dev[k] for k in ('test_permitted','source_receipt_sha256','registry_sha256','original_service_confirmation_permitted','production_qualified')}
        expected['development_sha256']=c.digest(c.OUT/'development.json')
        equivalent(c.read_json(c.OUT/'qualification.json'),expected,'qualification')
        report=c.read_json(c.OUT/'report.json')
        if dev['test_permitted']:
            require((c.OUT/'gates/development-final.json').is_file(),'Final original full gate replay must precede audit')
            study.validate_stage('test');evaluation=study.verify_evaluation('test')
            equivalent(report,study.test_results(evaluation,selection['selected_cell']),'TEST report')
            public['test']=verify_public_arrays('test')
        else:
            require(not (c.OUT/'rt_data/test').exists() and not (c.OUT/'rt_data/test.json').exists(),'Vetoed TEST geometries consumed')
            require(not (c.OUT/'predictions/test').exists(),'Vetoed TEST predictions exist')
            equivalent(report,{'schema':1,'identity':study.identity(),'status':'development_veto',
                'development':study.reference(c.OUT/'development.json'),'test_opened':False,
                'RF_adaptation_supported':False,'matched_alignment_benefit_supported':False,
                'detector_brier_benefit_supported':False,'original_service_confirmation_permitted':False,'production_qualified':False},'veto report')
    require(c.verify_previous()==prior,'Previous sealed evidence changed')
    return {'schema':1,'passes':True,'source_receipt_sha256':c.digest(c.OUT/'registration/source.json'),
        'previous':prior,'fits':fits,'baseline':baseline,'public_array_audit':public,
        'selected_cell':selection['selected_cell'],'test_opened':dev['test_permitted'],
        'model_constructors_fits_forwards_or_RT_solves':0,'original_service_confirmation_opened':False,'production_qualified':False}


def _visible():
    for p in c.OUT.rglob('*'):require(not p.is_symlink(),'Redirected adaptive evidence')
    return {str(p.relative_to(c.ROOT)):c.digest(p) for p in sorted(c.OUT.rglob('*')) if p.is_file()
        and not p.name.endswith(('.lock','.part')) and not p.name.startswith('.')
        and str(p.relative_to(c.OUT))!='verification/manifest.json'}


def run_audit():
    value=verify_campaign();c.write_once(c.OUT/'verification/audit.json',value);return value


def seal(documents=()):
    c.check_sources();c.verify_previous();path=c.OUT/'verification/audit.json'
    equivalent(c.read_json(path),verify_campaign(),'audit')
    value={'schema':1,'source_sha256':c.source_hashes(),'evidence_sha256':_visible(),
        'documents_sha256':{str(Path(p).relative_to(c.ROOT)):c.digest(p) for p in documents},
        'audit_sha256':c.digest(path),'previous_manifest_sha256':c.PREVIOUS_MANIFEST_SHA256}
    c.write_once(c.OUT/'verification/manifest.json',value);return verify_closure()


def verify_closure():
    c.check_sources();c.verify_previous();value=c.read_json(c.OUT/'verification/manifest.json')
    require(value['source_sha256']==c.source_hashes(),'Sealed adaptive source changed')
    require(value['evidence_sha256']==_visible(),'Sealed adaptive evidence bytes/membership changed')
    for name,sha in value['documents_sha256'].items():require(c.digest(c.safe_path(name))==sha,'Sealed document changed')
    require(value['audit_sha256']==c.digest(c.OUT/'verification/audit.json'),'Audit bytes changed')
    return {'passes':True,'source_files':len(value['source_sha256']),'evidence_files':len(value['evidence_sha256']),
        'documents':len(value['documents_sha256']),'original_service_confirmation_opened':False}
