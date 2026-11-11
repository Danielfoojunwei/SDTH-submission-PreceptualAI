#!/usr/bin/env python3
"""Independent passive audit of the sealed hierarchical-radio study.

No project modules, canonical calculators, model constructors, forwards,
fitting, ray tracing, packet simulation or network access. NumPy/SciPy reduce
archived observations/predictions. Torch is used only to deserialize selected
inert tensor checkpoints. Prior independent scan/support/dispatch arithmetic
is reused under an explicit immutable hash, never production verification code.

--self-test touches synthetic in-memory fixtures only. Actual auditing refuses
to begin before the final seal, and preserves its FIRST attempt and failure.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import time
import traceback

import numpy as np
from scipy.special import log_ndtr

CELLS=('hier_reg','hier_joint','global_reg','global8_reg','hier_erased')
COUNTS={'training':24,'validation':6,'development':4,'test':24,'ap_transfer':4}
SEEDS=(0,1,2)
NATIVE_SEEDS=(90512001,90512002,90512003,90512004)
NATIVE_METHODS=('full_radiomap','matched_radiomap','hierarchy','global_reg')
PRIOR=('radiojepa','radiojepa_spectral','device_transfer','channel_scene',
       'adaptive_radio','native_bridge','e7_calplanning','spatial_control')
HORIZONS=(0.,1.,2.,3.,4.,5.,10.,20.,30.)
OLD_AUDITOR='artifacts/spatial_control_review/audit_actual.py'
OLD_SHA='5dbd3b7e705f73ba3c666040e08ed9ee725c78ca64ae0c358f8db50b3b797e49'
TOL=2.e-7  # Independent SciPy versus torch float64 likelihood arithmetic.


def require(value,message):
    if not value:raise AssertionError(message)


def read(path):return json.loads(Path(path).read_bytes())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def compact(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def same(actual,expected,label):require(actual==expected,label)


def close(actual,expected,label,atol=TOL):
    require(np.shape(actual)==np.shape(expected),label+' axes')
    require(bool(np.allclose(actual,expected,rtol=1.e-9,atol=atol,equal_nan=False)),label+' numeric value')


def compare(actual,expected,label):
    if isinstance(expected,dict):
        same(set(actual),set(expected),label+' keys')
        for k in expected:compare(actual[k],expected[k],label+'.'+k)
    elif isinstance(expected,list):
        same(len(actual),len(expected),label+' length')
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,label+f'[{i}]')
    elif type(expected) is float:close(actual,expected,label)
    else:same(actual,expected,label)


def path(root,name):
    require(type(name) is str,'string repository path')
    p=Path(name)
    require(not p.is_absolute() and '..' not in p.parts and p.as_posix()==name,'canonical repository path')
    p=root/p
    require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root),'contained regular file '+name)
    return p


def ref(root,value):
    same(set(value),{'path','sha256'},'exact artifact reference')
    p=path(root,value['path']);same(sha(p),value['sha256'],'reference hash '+value['path']);return p


def arrays(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def data_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(compact({'dtype':a.dtype.str,'shape':list(a.shape)})+a.tobytes()).hexdigest()


def native_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()


def byte_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def log_mass(lo,hi,mean,std):
    lo,hi,mean,std=np.broadcast_arrays(*[np.asarray(a,dtype='f8') for a in (lo,hi,mean,std)])
    require(np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all(),'finite Gaussian')
    require(not np.isnan(lo).any() and not np.isnan(hi).any() and (lo<hi).all(),'ordered real observation events')
    a=(lo-mean)/std;b=(hi-mean)/std
    right=a>0
    large=np.where(right,log_ndtr(-a),log_ndtr(b))
    small=np.where(right,log_ndtr(-b),log_ndtr(a))
    with np.errstate(divide='ignore',invalid='ignore'):
        answer=large+np.log(-np.expm1(np.minimum(small-large,0.)))
    require(np.isfinite(answer).all(),'resolvable event mass');return answer


def event_losses(mean,std,target):
    h=target['heard'];c=target['censored'];known=h|c
    require(h.dtype==c.dtype==bool and h.shape==c.shape==np.shape(mean) and not (h&c).any(),'event mask axes/semantics')
    require(known.any(),'at least one genuine reported event')
    lo=np.where(c,-np.inf,np.maximum(target['lower_dbm'],target['detection_threshold_dbm']))
    hi=np.where(c,target['censor_upper_dbm'],target['upper_dbm'])
    loss=-log_mass(lo[known],hi[known],np.asarray(mean)[known],np.asarray(std)[known])
    weights=np.asarray(target.get('weights',np.ones(h.shape)),dtype='f8')[known]
    require(np.isfinite(weights).all() and (weights>=0).all() and weights.sum()>0,'positive genuine event mass')
    return loss,weights


def nll(mean,std,target):
    loss,w=event_losses(mean,std,target);return float(np.sum(loss*w)/w.sum())


def metrics(mean,std,target):
    mean=np.asarray(mean,dtype='f8');std=np.asarray(std,dtype='f8')
    require(mean.shape==std.shape==target['heard'].shape and np.isfinite(mean).all()
            and np.isfinite(std).all() and (std>0).all(),'complete finite prediction')
    h=target['heard'];c=target['censored'];require(h.any(),'heard endpoint available')
    error=mean[h]-target['center_dbm'][h];require(np.isfinite(error).all(),'finite heard centers')
    loss,_=event_losses(mean,std,target)
    return {'rmse_db':float(np.sqrt(np.mean(error**2))),'mae_db':float(np.mean(abs(error))),
        'p95_abs_error_db':float(np.percentile(abs(error),95)),
        'composite_interval_nll':float(np.mean(loss)),
        'heard_interval_95_coverage':float(np.mean(abs(error)<=1.959963984540054*std[h])),
        'mean_std_db':float(np.mean(std)),'heard':int(h.sum()),'censored':int(c.sum()),'unknown':int((~(h|c)).sum())}


def bootstrap(a,b):
    a=np.asarray(a,dtype='f8');b=np.asarray(b,dtype='f8');same(a.shape,b.shape,'paired geometry axes');same(a.shape,(24,),'24 whole layouts')
    delta=a-b;index=np.random.default_rng(21770).integers(0,24,(20000,24))
    lo,hi=np.quantile(delta[index].mean(1),(.00625,.99375))
    return {'candidate_mean':float(a.mean()),'reference_mean':float(b.mean()),'mean_delta':float(delta.mean()),
        'central_98_75_interval':[float(lo),float(hi)],'candidate_lower_error_supported':bool(hi<0),
        'reference_lower_error_supported':bool(lo>0),'layouts':24,'bootstrap_samples':20000,'seed':21770}


def verify_closures(root,out):
    manifest=read(out/'verification/manifest.json');registration=read(out/'registration/source.json')
    same(manifest['source_sha256'],registration['source_sha256'],'sealed source map')
    own={p.relative_to(root).as_posix() for top in ('src/wirelessjev/hierarchical_radio','tests/hierarchical_radio')
         for p in (root/top).rglob('*.py')}
    registered_own={n for n in registration['source_sha256'] if n.endswith('.py')
        and n.startswith(('src/wirelessjev/hierarchical_radio/','tests/hierarchical_radio/'))}
    same(own,registered_own,'exact current own source/test namespace')
    capsule=out/'registration/source'
    same({p.relative_to(capsule).as_posix() for p in capsule.rglob('*') if p.is_file()},set(registration['source_sha256']),'exact source capsule')
    evidence={p.relative_to(root).as_posix() for p in out.rglob('*') if p.is_file()
              and not p.is_relative_to(capsule) and p!=out/'verification/manifest.json'}
    same(evidence,set(manifest['evidence_sha256']),'exact sealed evidence membership')
    bound={}
    def merge(values):
        for name,digest in values.items():
            require(name not in bound or bound[name]==digest,'no historical hash conflict');bound[name]=digest
    merge(registration['source_sha256']);merge(manifest['evidence_sha256'])
    for name,digest in registration['source_sha256'].items():same(sha(path(capsule,name)),digest,'source capsule '+name)
    same(set(registration['prior']['manifests']),set(PRIOR),'all eight prior closures')
    for old in PRIOR:
        prior=read(ref(root,registration['prior']['manifests'][old]))
        same(manifest['prior']['manifests'][old],registration['prior']['manifests'][old],'same prior seal '+old)
        for key in ('source_sha256','evidence_sha256','documents_sha256'):merge(prior.get(key,{}))
    for name,digest in bound.items():same(sha(path(root,name)),digest,'frozen file '+name)
    for name in ('protocol','registry','preflight','preflight_log'):ref(root,registration[name])
    proof=read(out/'registration/git.json');commit=proof['commit'];source_sha=sha(out/'registration/source.json')
    same(proof['source_receipt_sha256'],source_sha,'published receipt identity');same(proof['remote_observed_commit'],commit,'remotely observed source commit')
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'Git commit syntax')
    for name,digest in dict(registration['source_sha256'],**{'artifacts/hierarchical_radio/registration/source.json':source_sha}).items():
        value=subprocess.check_output(['git','show',commit+':'+name],cwd=root)
        same(hashlib.sha256(value).hexdigest(),digest,'actual retained Git source '+name)
    for forbidden in ('artifacts/radiojepa/rt_data/confirmation','artifacts/radiojepa/rt_data/confirmation.json'):
        require(not (root/forbidden).exists(),'original SERVICE not opened')
    same(manifest['RT_cases'],186,'physical case cap');same(manifest['neural_fits'],15,'neural fit cap');same(manifest['native_simulations'],34,'native cap')
    return manifest,registration,bound,{'source_files':len(registration['source_sha256']),'evidence_files':len(evidence),
        'unique_named_files':len(bound),'source_commit':commit}


class Audit:
    def __init__(self,root,out):
        self.root,self.out=root,out;self.protocol=read(out/'protocol.json');self.registry=read(out/'registry.json')
        self.source_sha=sha(out/'registration/source.json');self.registry_sha=sha(out/'registry.json');self.groups={};self.raw={}
        self.calibration=read(out/'data/calibration.json')['calibration'];self.predictions={}

    def reports(self,case):
        if case not in self.raw:
            folder=self.out/'rt_data/cases'/case;receipt=read(folder/'receipt.json')
            same(receipt['case_id'],case,'RT case identity');same(receipt['source_receipt_sha256'],self.source_sha,'RT source')
            same(receipt['registry_sha256'],self.registry_sha,'RT registry')
            actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
            same(actual,set(receipt['files_sha256'])|{'receipt.json','attempt.json'},'exact physical case file membership')
            attempt=read(folder/'attempt.json');same(attempt['case_id'],case,'RT durable attempt case')
            require(attempt['authorized'] is True and type(attempt['created_unix_ns']) is int
                and attempt['created_unix_ns']>0,'actual source-authorized RT attempt')
            same(attempt['source_receipt_sha256'],self.source_sha,'RT attempt source');same(attempt['registry_sha256'],self.registry_sha,'RT attempt registry')
            for name,digest in receipt['files_sha256'].items():same(sha(path(folder,name)),digest,'RT case file '+case+'/'+name)
            self.raw[case]=arrays(folder/'public_views.npz')
        return self.raw[case]

    def verify_target(self,binding,target):
        ids=binding['query_measurement_ids'];require(len(ids)==len(target['heard']),'target ID count')
        raw=self.reports(binding['case_id']);indices=[]
        for value in ids:
            match=re.fullmatch(re.escape(binding['case_id'])+r':pose(\d+):ap(\d+):band(\d+):dir(\d+):dev(\d+)',value)
            require(match is not None,'explicit actual query measurement IDs');indices.append(tuple(map(int,match.groups())))
        index=tuple(np.asarray(indices,dtype='i8').T);h=raw['heard'][index];c=raw['censored'][index]
        expected={'heard':h,'censored':c,'center_dbm':np.where(h,raw['rssi_dbm'][index],0).astype('f4'),
            'lower_dbm':np.where(h,raw['quantization_interval_dbm'][index+(np.zeros(len(ids),dtype='i8'),)],0).astype('f4'),
            'upper_dbm':np.where(h,raw['quantization_interval_dbm'][index+(np.ones(len(ids),dtype='i8'),)],0).astype('f4'),
            'censor_upper_dbm':np.where(c,raw['censor_upper_dbm'][index],0).astype('f4'),
            'detection_threshold_dbm':raw['reported_detection_threshold_dbm'][index].astype('f4')}
        same(set(target),set(expected),'raw target schema')
        for key,value in expected.items():np.testing.assert_array_equal(target[key],value,err_msg='actual reported target '+key)
        require(all(row[3:]==(0,0) for row in indices),'device0 downlink query gauge')
        require(all(row[2] in ((2,) if binding['held_band'] else (0,1)) for row in indices),'held band isolated')
        context={v.split(':dir')[0] for role in ('context_measurement_ids_a','context_measurement_ids_b') for v in binding[role]}
        require(not context & {v.split(':dir')[0] for v in ids},'physical context/query separation')

    def group(self,name):
        if name in self.groups:return self.groups[name]
        receipt=read(self.out/'data/groups'/f'{name}.json');stage=receipt['stage'];same(receipt['source_receipt_sha256'],self.source_sha,'group source')
        same(receipt['registry_sha256'],self.registry_sha,'group registry');same(len(receipt['bindings']),COUNTS[stage]*3,'complete group cohort')
        same(receipt['calibration_sha256'],sha(self.out/'data/calibration.json'),'group calibration')
        same(receipt['stage_receipt_sha256'],sha(self.out/'rt_data'/f'{stage}.json'),'group actual stage')
        a=arrays(ref(self.root,receipt['arrays']));same(receipt['arrays']['path'],f'artifacts/hierarchical_radio/data/groups/{name}.npz','canonical group arrays')
        expected_keys={'ap_xyz'}|{role+'__'+k for role in ('a','b','query') for k in receipt['bindings'][0]['model_input_sha256'][role]}|{'target__'+k for k in receipt['bindings'][0]['target_sha256']}
        same(set(a),expected_keys,'exact inert group array membership')
        values=[]
        for i,binding in enumerate(receipt['bindings']):
            expected=self.registry['stages'][stage][i//3]['cases'][i%3]
            same(binding['case_id'],expected['case_id'],'registered group case order');same(binding['state'],i%3,'state order');same(binding['index'],i//3,'layout order')
            same(binding['stage'],stage,'group stage');same(binding['layout_id'],expected['layout_id'],'layout identity')
            require(all(v.shape[0]==len(receipt['bindings']) for v in a.values()),'all group example axes complete')
            roles={role:{k.split('__',1)[1]:v[i] for k,v in a.items() if k.startswith(role+'__')} for role in ('a','b','query','target')}
            for role in ('a','b','query'):same({k:data_hash(v) for k,v in roles[role].items()},binding['model_input_sha256'][role],'input hash '+role)
            same(data_hash(a['ap_xyz'][i]),binding['model_input_sha256']['ap_xyz'],'AP geometry hash')
            same({k:data_hash(v) for k,v in roles['target'].items()},binding['target_sha256'],'target array hashes')
            self.verify_target(binding,roles['target']);values.append({'binding':binding,'ap_xyz':a['ap_xyz'][i],**roles})
        self.groups[name]=values;return values

    def calibration_check(self):
        """Relative frontend calibration independently from TRAIN context only."""
        widths={0:[],1:[]};pairs=[];poses=self.registry['context_poses']
        for layout in self.registry['stages']['training']:
            for case in layout['cases']:
                case_id=case['case_id'];raw=self.reports(case_id)
                for pose in poses:
                    for ap in range(4):
                        for band in (0,1):
                            for device in (0,1):
                                if raw['heard'][pose,ap,band,0,device]:
                                    endpoints=raw['quantization_interval_dbm'][pose,ap,band,0,device]
                                    widths[device].append(float(endpoints[1]-endpoints[0]))
                            if raw['heard'][pose,ap,band,0,0] and raw['heard'][pose,ap,band,0,1]:
                                pairs.append({'case_id':case_id,'stage':'training','pose':pose,'ap':ap,'band':band,'direction':0,
                                    'reference_dbm':float(raw['rssi_dbm'][pose,ap,band,0,0]),'device_dbm':float(raw['rssi_dbm'][pose,ap,band,0,1]),
                                    'reference_interval':raw['quantization_interval_dbm'][pose,ap,band,0,0].astype(float).tolist(),
                                    'device_interval':raw['quantization_interval_dbm'][pose,ap,band,0,1].astype(float).tolist()})
        estimates={'0':{'offset_db':0.,'count':len(widths[0]),'quantization_db':float(np.median(widths[0])) if widths[0] else 0.,'unknown':False},
            '1':{'offset_db':float(np.median([p['device_dbm']-p['reference_dbm'] for p in pairs])) if pairs else 0.,
                'count':len(pairs),'quantization_db':float(np.median(widths[1])) if widths[1] else 0.,'unknown':not bool(pairs)}}
        compare(self.calibration['estimates'],estimates,'independent TRAIN-only calibration');compare(self.calibration['paired_reports'],pairs,'actual co-located TRAIN calibration pairs')
        require(self.calibration['query_or_held_band_or_true_offset_used'] is False
                and self.calibration['absolute_hardware_offset_identified'] is False,'relative-only calibration scope')
        return {'estimates':estimates,'paired_TRAIN_reports':len(pairs),'query_held_band_or_true_offsets_used':False}

    def training(self):
        import torch  # Inert weights_only deserialization, no nn construction.
        validation=self.group('validation-sparse');binding_hashes={
            'training_full':hashlib.sha256(compact([e['binding'] for e in self.group('training-full')])).hexdigest(),
            'training_sparse':hashlib.sha256(compact([e['binding'] for e in self.group('training-sparse')])).hexdigest(),
            'validation_sparse':hashlib.sha256(compact([e['binding'] for e in validation])).hexdigest()}
        expected_identity={'source_receipt_sha256':self.source_sha,'protocol_sha256':sha(self.out/'protocol.json'),
            'data_receipt_sha256':sha(self.out/'data/prepared.json')}
        members={};raw_archives=0
        expected_names={f'{cell}-seed{seed}' for cell in CELLS for seed in SEEDS}
        same({p.stem for p in (self.out/'fits').glob('*.json')},expected_names,'exact fifteen fit receipts')
        same({p.stem for p in (self.out/'checkpoints').glob('*.pt')},expected_names,'exact fifteen selected checkpoints')
        for cell in CELLS:
            for seed in SEEDS:
                name=f'{cell}-seed{seed}';fit=read(self.out/'fits'/f'{name}.json')
                same(fit['identity'],expected_identity,'fit source/data identity');same(fit['dataset_binding_sha256'],binding_hashes,'exact fitting groups')
                same((fit['cell'],fit['seed'],fit['run_id'],fit['epochs'],fit['updates'],fit['committed_updates']),(cell,seed,name,40,480,480),'fixed member/budget')
                require(fit['complete'] is True and fit['completed'] is True,'completed actual fit')
                same([h['epoch'] for h in fit['history']],list(range(1,41)),'exact40epochs')
                same(len(fit['update_trace']),480,'480 actual committed updates')
                rng=np.random.default_rng(seed)
                for k,row in enumerate(fit['update_trace']):
                    layouts=rng.choice(24,size=4,replace=False);states=rng.integers(0,3,size=4)
                    same(row['layout_indices'],layouts.tolist(),'fixed seed layout schedule');same(row['conditions'],states.tolist(),'fixed seed states')
                    same(row['step'],k+1,'committed step membership');same(row['regime'],'full' if k%2==0 else 'sparse','same alternating regime')
                scores=[]
                for epoch in (10,20,30,40):
                    row=fit['history'][epoch-1];p=ref(self.root,row['validation_predictions'])
                    same(p,self.out/'validation_predictions'/f'{name}-epoch{epoch}.npz','canonical raw VAL archive')
                    raw=arrays(p);same(set(raw),{'mean','std','metadata_utf8'},'raw VAL array membership')
                    require(raw['metadata_utf8'].dtype==np.uint8 and raw['metadata_utf8'].ndim==1,'inert UTF8 metadata')
                    metadata=json.loads(raw['metadata_utf8'].tobytes())
                    same(metadata,{'schema':1,'identity':expected_identity,'dataset_binding_sha256':binding_hashes,
                        'cell':cell,'seed':seed,'epoch':epoch,'committed_updates':epoch*12,
                        'model_state_sha256':row['validation_state_sha256'],'source_view':'a','regime':'sparse',
                        'target_binding_sha256':binding_hashes['validation_sparse'],
                        'endpoint':'equal_layout_equal_condition_composite_interval_NLL'},'actual VAL member/epoch/weight metadata')
                    same(raw['mean'].shape,raw['std'].shape,'VAL mean/std axes');same(raw['mean'].shape,(18,len(validation[0]['target']['heard'])),'all18VALexamples')
                    require(np.isfinite(raw['mean']).all() and np.isfinite(raw['std']).all() and (raw['std']>0).all(),'finite complete raw VAL')
                    per_example=[nll(raw['mean'][i],raw['std'][i],e['target']) for i,e in enumerate(validation)]
                    layout_rows=[{'layout_id':validation[i*3]['binding']['layout_id'],'conditions':[0,1,2],
                        'nll':float(np.mean(per_example[i*3:i*3+3]))} for i in range(6)]
                    score=float(np.mean([r['nll'] for r in layout_rows]));close(row['validation_nll'],score,'independent raw VAL score')
                    compare(row['validation_per_layout_nll'],layout_rows,'equal whole VAL layouts');scores.append((epoch,score));raw_archives+=1
                epoch,score=min(scores,key=lambda r:r[1]);same(fit['selected_epoch'],epoch,'independent first VAL minimum');close(fit['validation_nll'],score,'selected score')
                cp=self.out/'checkpoints'/f'{name}.pt';same(sha(cp),fit['checkpoint_sha256'],'selected checkpoint bytes')
                checkpoint=torch.load(cp,map_location='cpu',weights_only=True)
                same(checkpoint['identity'],expected_identity,'checkpoint identity');same(checkpoint['dataset_binding_sha256'],binding_hashes,'checkpoint dataset')
                same((checkpoint['cell'],checkpoint['seed'],checkpoint['selected_epoch'],checkpoint['committed_updates']),(cell,seed,epoch,480),'checkpoint selected member')
                h=hashlib.sha256()
                for key,value in sorted(checkpoint['state'].items()):
                    a=np.ascontiguousarray(value.detach().cpu().numpy());require(np.isfinite(a).all(),'finite inert weight')
                    h.update(key.encode());h.update(str((a.dtype.str,a.shape)).encode());h.update(a.tobytes())
                same(h.hexdigest(),fit['history'][epoch-1]['validation_state_sha256'],'selected weights produced that VAL archive')
                members[name]={'epoch':epoch,'validation_nll':score,'checkpoint_sha256':sha(cp)}
        same(raw_archives,60,'all sixty fixed VAL archives')
        means={cell:float(np.mean([members[f'{cell}-seed{seed}']['validation_nll'] for seed in SEEDS])) for cell in CELLS}
        selected=min(('hier_reg','hier_joint'),key=lambda cell:means[cell]);selection=read(self.out/'selection.json')
        same(selection['selected_hierarchy'],selected,'mean-three-member VAL branch');compare(selection['mean_validation_nll'],means,'all branch means')
        require(selection['selection_used_TEST_native_or_original_SERVICE'] is False,'VAL-only selection scope')
        for name in expected_names:
            ref(self.root,selection['fits'][name]);ref(self.root,selection['checkpoints'][name])
        self.selected=selected;return {'members':members,'raw_VAL_archives':raw_archives,'selected_hierarchy':selected,'branch_mean_VAL_NLL':means}

    def numerical(self):
        validation=self.group('validation-sparse');selection=read(self.out/'numerical/selection.json');scores=[]
        same(len(selection['candidates']),3,'three numerical candidates')
        for length,item in zip((5.,10.,20.),selection['candidates']):
            p=ref(self.root,item);same(p,self.out/'numerical'/f'radiomap-{length:g}m.json','canonical numerical candidate')
            row=read(p);same(row['lengthscale_m'],length,'registered lengthscale');same(row['source_receipt_sha256'],self.source_sha,'numerical source')
            ref(self.root,row['prior']);prediction=arrays(ref(self.root,row['validation_predictions']))
            same(set(prediction),{'mean','std'},'numerical VAL exact arrays');same(prediction['mean'].shape,(18,len(validation[0]['target']['heard'])),'full numerical VAL')
            score=float(np.mean([nll(prediction['mean'][i],prediction['std'][i],e['target']) for i,e in enumerate(validation)]))
            close(row['validation_nll'],score,'raw numerical VAL score');scores.append({'name':f'radiomap-{length:g}m','validation_nll':score,'prior':row['prior']})
        best=min(scores,key=lambda r:r['validation_nll']);same(selection['selected_name'],best['name'],'independent numerical first minimum')
        same(selection['selected_prior'],best['prior'],'independent selected prior');fixed=read(ref(self.root,selection['fixed_hierarchy']));ref(self.root,fixed['prior'])
        return {'candidates':scores,'selected_name':best['name']}

    def prediction_group(self,name):
        if name in self.predictions:return self.predictions[name]
        group=self.group(name);manifest=read(self.out/'predictions'/f'{name}.json')
        secondary='held6GHz' in name or 'stale' in name
        cells=(self.selected,'global_reg') if secondary else CELLS
        methods=[f'{cell}-seed{seed}' for cell in cells for seed in SEEDS]+['radiomap_operator','hierarchy_fixed']
        same(manifest['methods'],methods,'exact admitted prediction members')
        same(set(manifest['case_receipts']),{e['binding']['case_id'] for e in group},'all prediction cases')
        records=[]
        for i,e in enumerate(group):
            b=e['binding'];receipt=read(ref(self.root,manifest['case_receipts'][b['case_id']]))
            same(receipt['methods'],methods,'case members');same(receipt['example_index'],i,'actual example index');same(receipt['case_id'],b['case_id'],'case binding')
            same(receipt['source_receipt_sha256'],self.source_sha,'prediction source');same(receipt['selection_sha256'],sha(self.out/'selection.json'),'prediction selection')
            ref(self.root,receipt['attempt']);ref(self.root,receipt['example_group']);a=arrays(ref(self.root,receipt['predictions']))
            same(set(a),{'mean','std'},'inert prediction arrays');same(a['mean'].shape,(len(methods),len(e['target']['heard'])),'actual prediction axes')
            same(a['std'].shape,a['mean'].shape,'prediction spreads')
            derived=[metrics(m,sd,e['target']) for m,sd in zip(a['mean'],a['std'])]
            compare(receipt['metrics'],derived,'independent raw case metrics')
            for j,method in enumerate(methods):records.append({'method':method,'layout_id':b['layout_id'],'state':b['state'],
                'metrics':derived[j],'mean':a['mean'][j],'std':a['std'][j],'target':e['target']})
        self.predictions[name]=records;return records

    def reduce(self,records,stage,mapping):
        result={}
        for method,members in mapping.items():
            layouts=[]
            for item in self.registry['stages'][stage]:
                layout=item['layout_id'];states=[]
                for state in (0,1,2):
                    rows=[r for r in records if r['layout_id']==layout and r['state']==state and r['method'] in members]
                    same({r['method'] for r in rows},set(members),'exact state/member cells');same(len(rows),len(members),'no duplicated members')
                    states.append({k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']})
                layouts.append({'layout_id':layout,'metrics':{k:float(np.mean([r[k] for r in states])) for k in states[0]}})
            result[method]=layouts
        return result

    def field(self):
        official=read(self.out/'field_report.json');mapping={cell:[f'{cell}-seed{seed}' for seed in SEEDS] for cell in CELLS}
        mapping.update(selected_hierarchy=mapping[self.selected],radiomap_operator=['radiomap_operator'],hierarchy_fixed=['hierarchy_fixed'])
        tables={regime:self.reduce(self.prediction_group('test-'+regime),'test',mapping) for regime in ('sparse','full')}
        compare(official['layout_metrics'],tables,'independent whole-layout metrics')
        summaries={regime:{name:{k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}
            for name,rows in table.items()} for regime,table in tables.items()}
        compare(official['summary'],summaries,'whole-layout summary')
        contrasts=[]
        for candidate,reference in (('selected_hierarchy','global_reg'),('selected_hierarchy','radiomap_operator'),
                                    ('hier_joint','hier_reg'),('hier_reg','hier_erased')):
            a=[r['metrics']['rmse_db'] for r in tables['sparse'][candidate]];b=[r['metrics']['rmse_db'] for r in tables['sparse'][reference]]
            contrasts.append({'candidate':candidate,'reference':reference,'metric':'sparse_heard_RSSI_RMSE_db',**bootstrap(a,b)})
        compare(official['primary_contrasts'],contrasts,'all four independent adjusted intervals')
        raw=self.prediction_group('test-sparse');delta={}
        for method,members in mapping.items():
            layouts=[]
            for geometry in self.registry['stages']['test']:
                layout=geometry['layout_id'];errors=[];counts=[]
                for member in members:
                    states={r['state']:r for r in raw if r['layout_id']==layout and r['method']==member};changes=[]
                    for state in (1,2):
                        a,b=states[0],states[state];h=a['target']['heard']&b['target']['heard']
                        truth=b['target']['center_dbm'][h]-a['target']['center_dbm'][h]
                        changes.extend(((b['mean'][h]-a['mean'][h])-truth).tolist())
                    require(changes,'paired heard delta available');errors.append(float(np.sqrt(np.mean(np.square(changes)))));counts.append(len(changes))
                layouts.append({'layout_id':layout,'delta_rmse_db':float(np.mean(errors)),'paired_targets_per_member':counts})
            delta[method]={'mean_layout_delta_rmse_db':float(np.mean([r['delta_rmse_db'] for r in layouts])),'layouts':layouts}
        compare(official['localized_delta_descriptive'],delta,'localized paired deltas')
        secondary={}
        for name in ('test-sparse-held6GHz','test-sparse-stale','ap_transfer-sparse','ap_transfer-full'):
            records=self.prediction_group(name);stage='ap_transfer' if name.startswith('ap_transfer') else 'test'
            present={r['method'] for r in records};subset={m:members for m,members in mapping.items() if set(members)<=present}
            table=self.reduce(records,stage,subset)
            secondary[name]={m:{k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']} for m,rows in table.items()}
        compare(official['secondary_descriptive'],secondary,'all descriptive RF slices')
        # DEV is technical completeness, not a hidden efficacy screen.
        gate=read(self.out/'development_gate.json');require(gate['technical_pass'] is True and gate['efficacy_screening'] is False,'technical DEV scope')
        for regime in ('full','sparse'):self.prediction_group('development-'+regime)
        return {'primary_contrasts':contrasts,'summary':summaries,'all_prediction_records':sum(len(v) for v in self.predictions.values()),
            'all_reported_targets_rejoined_to_public_views':True}


def previous_independent_auditor(root):
    p=path(root,OLD_AUDITOR);same(sha(p),OLD_SHA,'unchanged independent clock/sensor arithmetic dependency')
    spec=importlib.util.spec_from_file_location('_previous_independent_spatial_audit',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def native_context(old,raw,a,world):
    """Independent192-row facility selection and physical public-input replay."""
    sites=len(world['sites']);same(sites,16,'native sixteenAP axis')
    payloads={(p['robot'],p['measurement_slot']):p for p in raw['local_completed_payloads']}
    for d in range(20):
        cutoff=60+d;scanned=a['history_scanned'][:cutoff];heard=a['history_heard'][:cutoff]
        events={}
        for e in raw['telemetry']['events']:
            if e['delivery_slot']<=d*100:
                key=(60+e['delivery_slot']//100,e['robot'])
                if key not in events or events[key]['delivery_slot']<e['delivery_slot']:events[key]=e
        candidates=[]
        for t,robot in zip(*np.nonzero(scanned)):
            t,robot=int(t),int(robot)
            if t<60:measured=(float(t-60),float(t-59));arrival=(np.nan,np.nan);provenance=0
            else:
                e=events[(t,robot)];p=payloads[(robot,e['measurement_slot'])]
                require(p['processed_slot']<=e['delivery_slot']<=d*100,'encoded report actually delivered')
                measured=(e['measurement_slot']/100.,)*2;arrival=(e['delivery_slot']/100.,)*2;provenance=1
            age=d-measured[0]
            if 0<=age<=60:candidates.append((t,robot,measured,arrival,provenance,age))
        candidates.sort(key=lambda row:(row[5],row[1],row[0]));require(candidates,'actual scan context available')
        plan=a[f'job{d}_plan'];np.testing.assert_array_equal(plan,world['plan'][d])
        qxy=np.unique(plan[:,np.asarray(HORIZONS)<=5].astype(float).reshape(-1,2),axis=0)
        positions=np.asarray([a['history_positions'][row[0],row[1]] for row in candidates])
        ages=np.asarray([row[5] for row in candidates]);cost=((positions[:,None]-qxy[None])**2).sum(-1)+(2*ages[:,None])**2
        chosen=[];current=None
        for _ in range(min(6,len(candidates))):
            score=cost.mean(1) if current is None else np.minimum(cost,current).mean(1)
            score[chosen]=np.inf;i=int(np.argmin(score));chosen.append(i)
            current=cost[i].copy() if current is None else np.minimum(current,cost[i])
        scans=[candidates[i] for i in chosen]
        selected=np.asarray([(row[0],row[1],ap,band) for row in scans for ap in range(sites) for band in (0,1)],dtype='i8')
        np.testing.assert_array_equal(a[f'job{d}_selected_rows'],selected)
        np.testing.assert_array_equal(a[f'job{d}_selected_scan_rows'],np.asarray([(row[0],row[1]) for row in scans],dtype='i8'))
        t,robot,ap,band=selected.T;link=ap*3+band
        rx=np.column_stack((a['history_positions'][t,robot],np.ones(len(t))))
        all_ap=np.column_stack((world['sites'],np.full(sites,4.)));tx=all_ap[ap]
        carrier=np.asarray((2.4e9,5.2e9,6.5e9))[band]
        fspl=world['tx_dl'][band]-20*(math.log10(4*math.pi/299792458.)+np.log10(np.linalg.norm(rx-tx,axis=1))+np.log10(carrier))
        features=old.geometry_features(rx,tx,carrier,world['noise_db'][band],world['tx_dl'][band]);features[:,[0,1,3,4]]-=.5
        h=heard[t,robot,link];value=a['history_value'][t,robot,link];bound=a['history_bound'][t,robot,link]
        context=np.column_stack((np.where(h,(value-fspl)/30,0),np.where(~h,(bound-fspl)/30,0),h,~h,features)).astype('f4')
        np.testing.assert_allclose(a[f'job{d}_context'],context,rtol=0,atol=4e-6)
        expected={'known':np.ones(len(t),bool),'scheduled':np.ones(len(t),bool),'context_ap':ap,
            'context_rx_xyz':(rx-[60.,40.,0.]).astype('f4'),'context_lower':np.where(h,value-fspl,-np.inf),
            'context_upper':np.where(h,value-fspl,bound-fspl),'context_floor':-90.-fspl,
            'context_fspl':fspl,'context_offset':np.zeros(len(t)),
            'physical_context_rx_xyz_m':rx.astype('f4'),'physical_ap_xyz_m':all_ap.astype('f4'),
            'ap_xyz':(all_ap-[60.,40.,0.]).astype('f4'),
            'measurement_intervals_s':np.repeat(np.asarray([row[2] for row in scans]),32,axis=0),
            'arrival_intervals_s':np.repeat(np.asarray([row[3] for row in scans]),32,axis=0),
            'clock_provenance':np.repeat(np.asarray([row[4] for row in scans]),32),
            'context_age_s':np.repeat(np.asarray([row[5] for row in scans]),32)}
        for name,expected_value in expected.items():
            np.testing.assert_allclose(a[f'job{d}_{name}'],expected_value,rtol=0,atol=4e-6,equal_nan=True,err_msg='independent native '+name)
        # Each stored query is grounded in public routes/carriers, never RF truth.
        shape=(100,9,sites,3)
        qrx=np.broadcast_to(np.concatenate((plan.astype(float),np.ones((*plan.shape[:2],1))),axis=-1)[:,:,None,None],shape+(3,)).reshape(-1,3)
        qa=np.broadcast_to(np.arange(sites)[None,None,:,None],shape).reshape(-1)
        qb=np.broadcast_to(np.arange(3)[None,None,None,:],shape).reshape(-1)
        qtx=all_ap[qa];qcarrier=np.asarray((2.4e9,5.2e9,6.5e9))[qb]
        qf=old.geometry_features(qrx,qtx,qcarrier,world['noise_db'][qb],world['tx_dl'][qb]);qf[:,[0,1,3,4]]-=.5
        qf64=qf.astype('f8');length=qf64[:,10]*10**(2*qf64[:,6])/10;frequency=qf64[:,7]
        basis=np.column_stack((np.ones(len(qrx)),qf64[:,6],qf64[:,9],length,frequency,
            frequency*qf64[:,6],frequency*qf64[:,9],frequency*length)).astype('f4')
        qfspl=world['tx_dl'][qb]-20*(math.log10(4*math.pi/299792458.)+np.log10(np.linalg.norm(qrx-qtx,axis=1))+np.log10(qcarrier))
        expected_query={'q':qf,'ap':qa,'rx_xyz':(qrx-[60.,40.,0.]).astype('f4'),
            'basis':basis,'fspl':qfspl.astype('f4'),'offset':np.zeros(len(qrx),dtype='f4')}
        for name,value in expected_query.items():np.testing.assert_allclose(a[f'job{d}_query_{name}'],value,rtol=0,atol=5e-6)
        np.testing.assert_array_equal(a[f'job{d}_physical_query_rx_xyz_m'],qrx.astype('f4'))
        m=a[f'job{d}_mean_dbm'];sd=a[f'job{d}_std_db']
        require(m.shape==sd.shape==shape and np.isfinite(m).all() and np.isfinite(sd).all() and (sd>=0).all(),'finite native marginal forecast')
        converted=(m.astype('f8')-world['tx_dl']+15.-world['noise_db']).astype('f4')
        np.testing.assert_array_equal(converted,a['forecast_mean_snr'][d]);np.testing.assert_array_equal(sd.astype('f4'),a['forecast_std_db'][d])
        metadata=raw['model_records'][d]['context_metadata']
        same(metadata['actual_tokens'],len(selected),'matched actual token count');same(metadata['maximum_tokens'],192,'shared context cap')
        same(metadata['candidate_scans'],len(candidates),'actual allowed candidate scans')
        require(metadata['selection_uses_rssi_amplitudes_or_heard_preference'] is False,'geometry/age-only retrieval')
        if raw['method'] in ('hierarchy','global_reg'):
            before=a[f'job{d}_inference_objective_before'];after=a[f'job{d}_inference_objective_after']
            require(before.shape==after.shape==(3,) and np.isfinite(before).all() and np.isfinite(after).all(),'three finite bounded inference steps')
            require((after<=before+1e-10*np.maximum(1.,abs(before))).all(),'nonincreasing accepted inference')


def native_metrics(raw,old):
    # Prior independent arithmetic also checks per-second cumulative counters.
    values=old.reducer(raw);spatial=raw['native']['spatial_execution']
    values['central_busy_forecast_skips']=len(spatial['central_compute_events'])
    keys=('control_miss','control_miss_worst_decile','control_gap_p99_ms','maximum_silence_ms',
        'video_on_time','bulk_mbps','roams','safety_stop_fraction','control_misses','control_sent',
        'end_pending_commands','observed_plus_pending_miss_fraction','paid_forecast_compute_seconds',
        'actual_forecast_compute_seconds','forecast_jobs_consumed','scans_started','scan_interruption_seconds',
        'central_planning_cpu_seconds','central_planning_overruns','central_busy_forecast_skips',
        'support_preparation_cpu_seconds','support_vetoes','central_handovers')
    return {key:values[key] for key in keys}


def audit_native(audit,old):
    root,out=audit.root,audit.out;report=read(out/'native/report.json');p=audit.protocol['native']
    same(p['seeds'],list(NATIVE_SEEDS),'registered native seed list');same(p['methods'],list(NATIVE_METHODS),'native methods')
    same(p['maximum_simulations'],34,'finite native budget');binding=report['binding']
    same(binding['source_receipt_sha256'],audit.source_sha,'native source');same(binding['selected_hierarchy'],audit.selected,'VAL-only native branch')
    for key in ('selection','field_report'):ref(root,binding[key])
    for item in binding['weights'].values():ref(root,item)
    worlds={};warms={}
    for seed in NATIVE_SEEDS:
        folder=out/'native/worlds'/str(seed);receipt=read(folder/'receipt.json')
        same(receipt['binding'],binding,'native world source/weights binding');same(receipt['seed'],seed,'native world seed')
        worlds[seed]=arrays(ref(root,receipt['world']));warms[seed]=arrays(ref(root,receipt['warm']));ref(root,receipt['attempt'])
        same({k:native_hash(v) for k,v in worlds[seed].items()},receipt['world_array_sha256'],'actual native world arrays')
        same({k:native_hash(v) for k,v in warms[seed].items()},receipt['warm_array_sha256'],'actual warm arrays')
    cases=[{'id':f'fresh-{seed}-ho{ho:g}','seed':seed,'handover_seconds':ho,
        'order':list(NATIVE_METHODS[(i+j)%4:]+NATIVE_METHODS[:(i+j)%4])}
        for i,seed in enumerate(NATIVE_SEEDS) for j,ho in enumerate((.03,4.))]
    parity_case={'id':'wiring-parity','seed':NATIVE_SEEDS[0],'handover_seconds':.03}
    expected={(case['id'],method) for case in cases for method in NATIVE_METHODS}|{('wiring-parity',m) for m in ('full_radiomap','radiomap_roundtrip')}
    same({(f.parent.parent.name,f.parent.name) for f in (out/'native/cases').rglob('receipt.json')},expected,'exact34native receipts')
    same(len(list((out/'native/cases').rglob('attempt.json'))),34,'exact34native attempts')
    rows=[];reviews=[];parity=[]
    for case in [parity_case,*cases]:
        is_parity=case==parity_case;methods=('full_radiomap','radiomap_roundtrip') if is_parity else NATIVE_METHODS
        for method in methods:
            folder=out/'native/cases'/case['id']/method;receipt=read(folder/'receipt.json')
            same(set(f.name for f in folder.iterdir()),{'receipt.json','attempt.json','started.json','raw.json.gz','predictions_history.npz'},'once-only native case directory')
            same(receipt['case'],case,'native exact case');same(receipt['method'],method,'native exact method');same(receipt['binding'],binding,'case binding')
            same(receipt['parity'],is_parity,'parity identity');same(receipt['simulations'],1,'one native simulation')
            raw=json.loads(gzip.decompress(ref(root,receipt['raw']).read_bytes()));saved=arrays(ref(root,receipt['arrays']))
            attempt=read(ref(root,receipt['attempt']));started=read(ref(root,receipt['started']))
            require(type(attempt['created_unix_ns']) is int and started['created_unix_ns']>=attempt['created_unix_ns'],'durable attempt before exclusive start')
            same(attempt['identity'],receipt['identity'],'actual attempt identity');same(started['attempt'],receipt['attempt'],'exclusive start binding')
            same(raw['method'],method,'actual method');same(raw['seed'],case['seed'],'actual scene');close(raw['handover_seconds'],case['handover_seconds'],'actualHO')
            same(raw['native']['scene'],case['seed'],'native output scene');close(raw['native']['duration_s'],20.,'20second native window')
            require(raw['causal_central'] is True and raw['local_runtime'] is None and raw['production_qualified'] is False,'no undeclared local or production authority')
            require(raw['simulation_config']['causal_central'] is True and raw['simulation_config']['local_recovery'] is False
                and raw['simulation_config']['collision_lambda']==0 and raw['simulation_config']['interference'] is False,'finite physical assumptions')
            close(raw['policy_config']['roam_cost_s'],case['handover_seconds'],'actual policyHO')
            old.check_forecasts(raw,saved,worlds[case['seed']],False,is_parity)
            same([j['kind'] for j in raw['forecast_jobs']],['cold']+['warm']*19,'actual job kind membership')
            payloads=old.check_scan_payloads(raw,saved,warms[case['seed']],True)
            matched=not is_parity and method!='full_radiomap';same(raw['support_enabled'],matched,'matched support scope')
            derived=native_metrics(raw,old);compare(receipt['metrics'],derived,'independent native per-case reductions')
            mutable=deepcopy(raw);mutable['_world_noise']=worlds[case['seed']]['noise_db'];mutable['_world_tx']=worlds[case['seed']]['tx_dl']
            review=old.check_dispatch(mutable,payloads,case['handover_seconds'])
            if matched:
                native_context(old,raw,saved,worlds[case['seed']])
                # Schema adapter only: old independent support arithmetic calls
                # the unchanged guard flag "encoder_context_changed".
                for preparation in mutable['support_preparation']:
                    same(preparation['matched_forecast_context_budget'],192,'matched support budget')
                    require(preparation['guard_modifies_issued_forecast'] is False,'no retrospective forecast edit')
                    preparation['encoder_context_changed']=preparation['guard_modifies_issued_forecast']
                review['support_queries']=old.check_support(mutable,saved,worlds[case['seed']])
            else:require(not raw['support'],'no support in full-history reference/parity')
            if is_parity:parity.append((raw,saved));continue
            rows.append({'case':case['id'],'seed':case['seed'],'handover_seconds':case['handover_seconds'],
                'method':method,'metrics':derived,'receipt':{'path':str((folder/'receipt.json').relative_to(root)),'sha256':sha(folder/'receipt.json')}})
            reviews.append({'case':case['id'],'method':method,**review})
    for key in ('forecast_mean_snr','forecast_std_db'):np.testing.assert_array_equal(parity[0][1][key],parity[1][1][key])
    same([r['ready_slot'] for r in parity[0][0]['forecast_jobs']],[r['ready_slot'] for r in parity[1][0]['forecast_jobs']],'exact numerical roundtrip completion')
    for key in parity[0][0]['native']:
        if key!='spatial_execution':same(parity[0][0]['native'][key],parity[1][0]['native'][key],'roundtrip actual native '+key)
    compare(report['rows'],rows,'complete native raw membership/metrics')
    means=[];contrasts=[]
    for ho in (.03,4.):
        for method in NATIVE_METHODS:
            items=[r['metrics'] for r in rows if r['handover_seconds']==ho and r['method']==method]
            means.append({'handover_seconds':ho,'method':method,'episodes':4,'mean':{k:float(np.mean([r[k] for r in items])) for k in items[0]}})
        for candidate,reference in (('hierarchy','matched_radiomap'),('hierarchy','global_reg'),('hierarchy','full_radiomap'),('matched_radiomap','full_radiomap')):
            pairs=[]
            for seed in NATIVE_SEEDS:
                a=next(r['metrics'] for r in rows if (r['handover_seconds'],r['seed'],r['method'])==(ho,seed,candidate))
                b=next(r['metrics'] for r in rows if (r['handover_seconds'],r['seed'],r['method'])==(ho,seed,reference))
                pairs.append({'seed':seed,'delta':{k:a[k]-b[k] for k in a}})
            contrasts.append({'handover_seconds':ho,'candidate':candidate,'reference':reference,'paired_episodes':pairs,
                'mean_delta':{k:float(np.mean([r['delta'][k] for r in pairs])) for k in pairs[0]['delta']},
                'episodes_with_more_control_misses':sum(r['delta']['control_misses']>0 for r in pairs),
                'episodes_with_larger_maximum_silence':sum(r['delta']['maximum_silence_ms']>0 for r in pairs)})
    compare(report['method_means'],means,'independent native whole-episode means');compare(report['contrasts'],contrasts,'independent native paired harms')
    return {'fresh_simulations':32,'parity_simulations':2,'actual_reductions':reviews,'method_means':means,'contrasts':contrasts}


def self_test():
    """Synthetic parser/arithmetic checks only, never experimental file reads."""
    close(log_mass(-np.inf,np.inf,0.,1.),0.,'uninformative Gaussian event')
    close(log_mass(-np.inf,-40.,0.,1.),log_ndtr(-40.),'deep censor tail')
    value=float(log_mass(40.,40.+1e-6,0.,1.));mid=40.+.5e-6
    close(value,-.5*mid**2-.5*math.log(2*math.pi)+math.log(1e-6),'narrow remote interval',atol=3e-8)
    t={'heard':np.array([True,False]),'censored':np.array([False,True]),'center_dbm':np.array([-90.,0.]),
        'lower_dbm':np.array([-91.,0.]),'upper_dbm':np.array([-89.,0.]),'censor_upper_dbm':np.array([0.,-95.]),
        'detection_threshold_dbm':np.full(2,-90.)}
    m=metrics(np.array([-90.,-95.]),np.ones(2),t);same(m['rmse_db'],0.,'heard center only')
    close(m['composite_interval_nll'],.5*(-float(log_mass(-90.,-89.,-90.,1.))-math.log(.5)),'floor clip/censor distinction')
    interval=bootstrap(np.ones(24),np.full(24,3.));same(interval['central_98_75_interval'],[-2.,-2.],'paired24geometry bootstrap')
    # Explicit five-axis public-report indexing, including interval endpoints.
    raw=np.arange(48*4*3*2*2*2).reshape(48,4,3,2,2,2)
    idx=tuple(np.asarray([(1,2,0,0,0),(3,1,1,0,0)]).T)
    np.testing.assert_array_equal(raw[idx+(np.array([0,1]),)],np.array([raw[1,2,0,0,0,0],raw[3,1,1,0,0,1]]))
    print(json.dumps({'self_test':True,'synthetic_only':True,'scientific_calls':0},sort_keys=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    if args.self_test:self_test();return
    root=args.root.resolve();out=root/'artifacts/hierarchical_radio';review=root/'artifacts/hierarchical_radio_review'
    require((out/'verification/manifest.json').is_file() and (out/'report.json').is_file(),'actual audit requires completed official seal')
    script_sha=sha(Path(__file__));manifest_sha=sha(out/'verification/manifest.json');review.mkdir(parents=True,exist_ok=True)
    attempt={'schema':1,'audit_source_sha256':script_sha,'study_manifest_sha256':manifest_sha,
        'started_unix_ns':time.time_ns(),'scientific_calls':0}
    with (review/'actual_attempt.json').open('xb') as f:f.write(encoded(attempt));f.flush()
    began=time.perf_counter()
    try:
        manifest,registration,bound,provenance=verify_closures(root,out)
        old=previous_independent_auditor(root);audit=Audit(root,out)
        same(audit.protocol['rt']['actual_cases'],186,'protocol physical budget');same(audit.protocol['training']['neural_fits'],15,'protocol fit budget')
        # The complete physical cohort is fixed independently of success.
        physical=0
        for stage,count in COUNTS.items():
            ledger=read(out/'rt_data'/f'{stage}.json');expected={case['case_id'] for g in audit.registry['stages'][stage] for case in g['cases']}
            same(set(ledger['case_receipts']),expected,'all physical cases '+stage);same(len(expected),count*3,'physical stage count')
            for case,item in ledger['case_receipts'].items():ref(root,item);audit.reports(case);physical+=1
        same(physical,186,'all actual physical receipts')
        same(len(list((out/'rt_data/cases').glob('*/receipt.json'))),186,'no extra/missing RT receipt')
        same(len(list((out/'rt_data/cases').glob('*/attempt.json'))),186,'no extra/missing RT attempt')
        calibration=audit.calibration_check();training=audit.training();numerical=audit.numerical();field=audit.field();native=audit_native(audit,old)
        for name,digest in bound.items():same(sha(path(root,name)),digest,'unchanged frozen bytes after audit '+name)
        same(sha(Path(__file__)),script_sha,'unchanged independent parser');same(sha(out/'verification/manifest.json'),manifest_sha,'unchanged study seal')
        result={'schema':1,'passes':True,'audit_source_sha256':script_sha,'study_manifest_sha256':manifest_sha,
            'provenance':provenance,'prior_independent_parser':{'path':OLD_AUDITOR,'sha256':OLD_SHA,
                'reused_scope':'scan payload, support and dispatch arithmetic only'},
            'actual_RT_cases':physical,'calibration':calibration,'training':training,'numerical_selection':numerical,'field':field,'native':native,
            'scientific_calls':{'model_constructors':0,'fits':0,'forwards':0,'RT_solves':0,'packet_simulations':0},
            'checkpoint_deserialization_only':True,'source_evidence_unchanged':True,'SERVICE_opened':False,
            'production_promotion':False,'elapsed_seconds':time.perf_counter()-began}
        with (review/'independent.json').open('xb') as f:f.write(encoded(result));f.flush()
        print(json.dumps({'passes':True,'actual_RT_cases':186,'fits_verified':15,'native_simulations':34,'output':str(review/'independent.json')},sort_keys=True))
    except BaseException as error:
        failure={**attempt,'passes':False,'error_type':type(error).__name__,'reason':str(error),
            'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-began}
        with (review/'first_failure.json').open('xb') as f:f.write(encoded(failure));f.flush()
        raise


if __name__=='__main__':main()
