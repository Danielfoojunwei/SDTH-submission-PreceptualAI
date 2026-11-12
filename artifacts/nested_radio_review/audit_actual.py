#!/usr/bin/env python3
"""Independent passive replay of the sealed nested-radio RF experiment.

No project or canonical calculator imports, model construction, prediction,
fitting, ray tracing or network access. A pinned prior independent audit supplies
reported-event arithmetic and the TRAIN-only calibration reader; every new
selection, checkpoint, cohort, interval and feature-rank reduction is checked
here. Torch only deserializes inert weights with weights_only=True.
"""
from __future__ import annotations
import argparse
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

CELLS=('global8_reg','global8_joint','hier_reg','nested_reg','nested_joint')
ELIGIBLE=('global8_reg','global8_joint','nested_reg','nested_joint')
NUMERICAL=('radiomap-5m','radiomap-10m','radiomap-20m','hierarchy-fixed')
SEEDS=(0,1,2)
EPOCHS=(40,80,120,160)
COUNTS={'test':24,'ap_transfer':4}
PRIOR=('radiojepa','radiojepa_spectral','device_transfer','channel_scene','adaptive_radio',
       'native_bridge','e7_calplanning','spatial_control','hierarchical_radio')
DEPENDENCY='artifacts/hierarchical_radio_review/audit_actual.py'
DEPENDENCY_SHA='7ecafae44a534474cdde1149b46da0aee2bb88d5a8c26c84c210ef8ee87e8d8f'
CONTRASTS=(('selected','best_numerical'),('selected','global8_reg'),('nested_reg','hier_reg'),('nested_joint','nested_reg'))


def require(ok,reason):
    if not ok:raise AssertionError(reason)


def read(p):return json.loads(Path(p).read_bytes())
def compact(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def encoded(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def same(a,b,label):require(a==b,label)
def close(a,b,label,atol=3e-6):
    require(np.shape(a)==np.shape(b) and bool(np.allclose(a,b,rtol=1e-9,atol=atol,equal_nan=False)),label)
def compare(a,b,label):
    if isinstance(b,dict):
        same(set(a),set(b),label+' keys')
        for k in b:compare(a[k],b[k],label+'.'+k)
    elif isinstance(b,list):
        same(len(a),len(b),label+' length')
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,label+f'[{i}]')
    elif type(b) is float:close(a,b,label)
    else:same(a,b,label)
def path(root,name):
    require(isinstance(name,str),'string path');p=Path(name)
    require(not p.is_absolute() and '..' not in p.parts and p.as_posix()==name,'canonical path '+name)
    p=root/p;require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root),'contained file '+name);return p
def ref(root,value):
    same(set(value),{'path','sha256'},'exact reference');p=path(root,value['path']);same(sha(p),value['sha256'],'reference bytes '+value['path']);return p
def arrays(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def data_hash(v):
    v=np.ascontiguousarray(v);return hashlib.sha256(compact({'dtype':v.dtype.str,'shape':list(v.shape)})+v.tobytes()).hexdigest()
def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        a=np.ascontiguousarray(value.detach().cpu().numpy());require(np.isfinite(a).all(),'finite inert weight')
        h.update(key.encode());h.update(str((a.dtype.str,a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()
def previous(root):
    p=path(root,DEPENDENCY);same(sha(p),DEPENDENCY_SHA,'pinned independent arithmetic dependency')
    spec=importlib.util.spec_from_file_location('_pinned_independent_hierarchy_audit',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def bootstrap(a,b):
    a=np.asarray(a,dtype='f8');b=np.asarray(b,dtype='f8');same(a.shape,(24,),'whole24geometry axes');same(b.shape,a.shape,'paired axes')
    require(np.isfinite(a).all() and np.isfinite(b).all(),'finite paired errors')
    d=a-b;idx=np.random.default_rng(21771).integers(0,24,(20000,24));lo,hi=np.quantile(d[idx].mean(1),(.00625,.99375))
    return {'candidate_mean':float(a.mean()),'reference_mean':float(b.mean()),'mean_delta':float(d.mean()),
        'central_98_75_interval':[float(lo),float(hi)],'candidate_lower_error_supported':bool(hi<0),
        'reference_lower_error_supported':bool(lo>0),'layouts':24,'bootstrap_samples':20000,'seed':21771}


def ranks(x):
    x=np.asarray(x,dtype='f8');s=np.linalg.svd(x-x.mean(0,keepdims=True),compute_uv=False);p=s*s
    return {'rows':len(x),'columns':x.shape[1],'mean_feature_variance':float(np.mean(np.var(x,axis=0))),
        'participation_rank_centered':float(p.sum()**2/(p@p)) if p.sum()>0 else 0.,'singular_values_centered':s.tolist()}


def verify_closures(root,out):
    m=read(out/'verification/manifest.json');r=read(out/'registration/source.json')
    same(m['source_sha256'],r['source_sha256'],'sealed registered source');same(set(r['prior']['manifests']),set(PRIOR),'all nine prior seals')
    own={p.relative_to(root).as_posix() for top in ('src/wirelessjev/nested_radio','tests/nested_radio') for p in (root/top).rglob('*.py')}
    same(own,{n for n in r['source_sha256'] if n.startswith(('src/wirelessjev/nested_radio/','tests/nested_radio/')) and n.endswith('.py')},'exact own source/test namespace')
    capsule=out/'registration/source';same({p.relative_to(capsule).as_posix() for p in capsule.rglob('*') if p.is_file()},set(r['source_sha256']),'exact source capsule')
    evidence={p.relative_to(root).as_posix() for p in out.rglob('*') if p.is_file() and not p.is_relative_to(capsule) and p!=out/'verification/manifest.json'}
    same(evidence,set(m['evidence_sha256']),'exact sealed evidence membership')
    union={}
    def merge(values):
        for n,d in values.items():require(n not in union or union[n]==d,'historical digest conflict');union[n]=d
    merge(r['source_sha256']);merge(m['evidence_sha256'])
    for n,d in r['source_sha256'].items():same(sha(path(capsule,n)),d,'capsule bytes '+n)
    for old in PRIOR:
        prior=read(ref(root,r['prior']['manifests'][old]));same(m['prior']['manifests'][old],r['prior']['manifests'][old],'same prior seal')
        for field in ('source_sha256','evidence_sha256','documents_sha256'):merge(prior.get(field,{}))
    for n,d in union.items():same(sha(path(root,n)),d,'unchanged frozen file '+n)
    for key in ('protocol','registry','preflight','preflight_log'):ref(root,r[key])
    p=read(out/'registration/git.json');commit=p['commit'];source_sha=sha(out/'registration/source.json')
    same(p['remote_observed_commit'],commit,'remotely observed source');same(p['source_receipt_sha256'],source_sha,'published receipt identity')
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'exact Git commit')
    for n,d in dict(r['source_sha256'],**{'artifacts/nested_radio/registration/source.json':source_sha}).items():
        raw=subprocess.check_output(['git','show',commit+':'+n],cwd=root);same(hashlib.sha256(raw).hexdigest(),d,'actual published source '+n)
    same((m['fresh_RT_cases'],m['fits'],m['native_simulations']),(84,15,0),'finite study caps')
    for n in ('artifacts/radiojepa/rt_data/confirmation','artifacts/radiojepa/rt_data/confirmation.json'):require(not (root/n).exists(),'original SERVICE stays closed')
    return m,r,union,{'source_files':len(r['source_sha256']),'evidence_files':len(evidence),'unique_named_files':len(union),'source_commit':commit}


class Audit:
    def __init__(self,root,out,base):
        self.root,self.out,self.base=root,out,base;self.registry=read(out/'registry.json');self.protocol=read(out/'protocol.json')
        self.source_sha=sha(out/'registration/source.json');self.registry_sha=sha(out/'registry.json');self.raw={};self.groups={};self.predictions={}
        self.parent=base.Audit(root,root/'artifacts/hierarchical_radio');self.calibration=self.parent.calibration
        self.selected_states={};self.weights={}
    def reports(self,case):return self.base.Audit.reports(self,case)
    def fitting_data(self):
        prepared=read(self.out/'data/prepared.json');same(prepared['source_receipt_sha256'],self.source_sha,'prepared source')
        same(prepared['protocol_sha256'],sha(self.out/'protocol.json'),'prepared protocol');same(prepared['registry_sha256'],self.registry_sha,'prepared registry')
        require(prepared['reused_parent_TEST_or_DEV_for_fitting'] is False and prepared['new_RT_or_calibration_fit_required_for_TRAIN_VAL'] is False,'only immutable parent TRAIN/VAL')
        self.validation=self.parent.group('validation-sparse');self.full=self.parent.group('training-full');self.sparse=self.parent.group('training-sparse')
        for item in prepared['parent_inputs'].values():
            if isinstance(item,dict) and set(item)=={'path','sha256'}:ref(self.root,item)
            elif isinstance(item,dict):
                for value in item.values():ref(self.root,value)
        dependency=prepared['parent_dependency_identity'];ref(self.root,dependency['parent_manifest']);ref(self.root,dependency['parent_source_receipt'])
        for name,digest in dependency['fitting_input_sha256'].items():
            require('/test-' not in name and '/test.json' not in name and '/development' not in name,'no prior evaluation fitting input')
            same(sha(path(self.root,name)),digest,'immutable TRAIN/VAL source bytes')
        self.binding={key:hashlib.sha256(compact([e['binding'] for e in examples])).hexdigest() for key,examples in
            (('training_full',self.full),('training_sparse',self.sparse),('validation_sparse',self.validation))}
        self.identity={'source_receipt_sha256':self.source_sha,'protocol_sha256':sha(self.out/'protocol.json'),'data_receipt_sha256':sha(self.out/'data/prepared.json')}
        return self.parent.calibration_check()
    def validation_scores(self,mean,std):
        require(np.shape(mean)==np.shape(std)==(18,len(self.validation[0]['target']['heard'])),'all18VAL predictions')
        values=[self.base.metrics(mean[i],std[i],e['target']) for i,e in enumerate(self.validation)]
        rows=[{'layout_id':self.validation[i*3]['binding']['layout_id'],'conditions':[0,1,2],
            'rmse_db':float(np.mean([v['rmse_db'] for v in values[i*3:i*3+3]])),
            'nll':float(np.mean([v['composite_interval_nll'] for v in values[i*3:i*3+3]]))} for i in range(6)]
        return float(np.mean([v['rmse_db'] for v in rows])),float(np.mean([v['nll'] for v in rows])),rows
    def gates(self,state,cell):
        gate=state['local_gates'].detach().cpu().numpy();same(gate.shape,(16,),'gate vector')
        require(gate.dtype.kind=='f' and np.isfinite(gate).all() and ((gate>=0)&(gate<=1)).all(),'finite feasible saved gates')
        if cell in ('global8_reg','global8_joint'):np.testing.assert_array_equal(gate,np.zeros(16))
        elif cell=='hier_reg':np.testing.assert_array_equal(gate,np.ones(16))
        return gate
    def training(self):
        import torch  # Deserialization only; no module construction or forward.
        members={f'{c}-seed{s}' for c in CELLS for s in SEEDS};epochs={f'{n}-epoch{e}' for n in members for e in EPOCHS}
        for folder,suffix,expected in (('fits','.json',members),('checkpoints','.pt',members),('epoch_checkpoints','.pt',epochs),('validation_predictions','.npz',epochs)):
            same({p.stem for p in (self.out/folder).glob('*'+suffix)},expected,'exact member/archive cohort '+folder)
        scores={c:[] for c in CELLS};details={};raw_count=0
        for cell in CELLS:
            for seed in SEEDS:
                name=f'{cell}-seed{seed}';fit=read(self.out/'fits'/f'{name}.json');settings={'cell':cell,'width':32,'inference_steps':3}
                same(fit['identity'],self.identity,'fit identity');same(fit['dataset_binding_sha256'],self.binding,'fit input/target groups')
                require(fit['complete'] is True and fit['completed'] is True,'completed fit')
                same((fit['cell'],fit['seed'],fit['run_id'],fit['epochs'],fit['updates'],fit['committed_updates']),(cell,seed,name,160,1920,1920),'member/budget')
                same(fit['settings'],settings,'public model constructor');same(len(fit['update_trace']),1920,'committed update count')
                same([r['epoch'] for r in fit['history']],list(range(1,161)),'all160epochs')
                rng=np.random.default_rng(seed)
                for k,row in enumerate(fit['update_trace']):
                    layouts=rng.choice(24,4,replace=False);states=rng.integers(0,3,4);regime='full' if k%2==0 else 'sparse'
                    same((row['step'],row['epoch'],row['update_in_epoch'],row['regime']),(k+1,k//12+1,k%12+1,regime),'committed schedule index')
                    same(row['layout_indices'],layouts.tolist(),'fixed layout RNG');same(row['conditions'],states.tolist(),'fixed state RNG')
                    es=self.full if regime=='full' else self.sparse
                    same(row['case_ids'],[es[int(i)*3+int(j)]['binding']['case_id'] for i,j in zip(layouts,states)],'admitted actual TRAIN cases')
                    close(row['alignment_weight'],.1 if cell in ('global8_joint','nested_joint') else 0.,'registered agreement')
                    close(row['gate_penalty_weight'],.01 if cell.startswith('nested_') else 0.,'registered effective-RMS penalty')
                    close(row['gate_rms_reference'],.5,'RMS normalization')
                    close(row['gate_penalty'],row['gate_penalty_weight']*row['effective_prior_rms_normalized_mean'],'effective penalty arithmetic')
                    close(row['effective_prior_rms_mean_db'],5.*row['effective_prior_rms_normalized_mean'],'coefficient prior RMS dB')
                    close(row['loss'],row['measurement_nll']+row['alignment_weight']*row['alignment']+row['gate_penalty'],'objective components')
                    require(all(type(v) in (int,float) and math.isfinite(v) for v in row.values() if not isinstance(v,(str,list))),'finite trace scalars')
                candidates=[]
                for epoch in range(1,161):
                    row=fit['history'][epoch-1];same(row['completed_updates'],epoch*12,'epoch step ledger')
                    close(row['mean_training_loss'],float(np.mean([r['loss'] for r in fit['update_trace'][(epoch-1)*12:epoch*12]])),'epoch mean')
                    if epoch not in EPOCHS:
                        require(all(row[k] is None for k in ('validation_rmse','validation_nll','validation_per_layout','validation_predictions','validation_state_sha256','epoch_checkpoint')),'no extra validation selection');continue
                    p=ref(self.root,row['validation_predictions']);same(p,self.out/'validation_predictions'/f'{name}-epoch{epoch}.npz','canonical raw VAL')
                    a=arrays(p);same(set(a),{'mean','std','metadata_utf8'},'VAL exact arrays');require(a['metadata_utf8'].dtype==np.uint8 and a['metadata_utf8'].ndim==1,'inert UTF8 VAL metadata')
                    metadata=json.loads(a['metadata_utf8'].tobytes())
                    same(metadata,{'schema':1,'identity':self.identity,'dataset_binding_sha256':self.binding,'cell':cell,'seed':seed,'epoch':epoch,
                        'committed_updates':epoch*12,'model_state_sha256':row['validation_state_sha256'],'source_view':'a','regime':'sparse',
                        'target_binding_sha256':self.binding['validation_sparse'],'selection_endpoint':'equal_layout_equal_state_heard_RSSI_RMSE',
                        'composite_NLL_for_selection':False},'VAL member/weight/data metadata')
                    score,nll,rows=self.validation_scores(a['mean'],a['std']);close(row['validation_rmse'],score,'raw VAL point score');close(row['validation_nll'],nll,'secondary VAL NLL');compare(row['validation_per_layout'],rows,'equal-state/layout scores')
                    ep=ref(self.root,row['epoch_checkpoint']);same(ep,self.out/'epoch_checkpoints'/f'{name}-epoch{epoch}.pt','epoch archive path')
                    cp=torch.load(ep,map_location='cpu',weights_only=True);same(cp['identity'],self.identity,'epoch identity');same(cp['dataset_binding_sha256'],self.binding,'epoch dataset');same(cp['settings'],settings,'epoch settings')
                    same((cp['cell'],cp['seed'],cp['epoch'],cp['committed_updates']),(cell,seed,epoch,epoch*12),'epoch member/step')
                    same(state_hash(cp['state']),row['validation_state_sha256'],'actual epoch weights produced raw VAL');self.gates(cp['state'],cell)
                    close(cp['validation_rmse'],score,'epoch point score');close(cp['validation_nll'],nll,'epoch secondary score');candidates.append((epoch,score,nll));raw_count+=1
                epoch,score,nll=min(candidates,key=lambda v:v[1]);same(fit['selected_epoch'],epoch,'FIRST raw RMSE minimum');same(fit['selected_updates'],epoch*12,'selected optimizer budget');close(fit['validation_rmse'],score,'chosen point score')
                cp_path=self.out/'checkpoints'/f'{name}.pt';same(sha(cp_path),fit['checkpoint_sha256'],'selected bytes')
                cp=torch.load(cp_path,map_location='cpu',weights_only=True);same(cp['identity'],self.identity,'selected source');same(cp['dataset_binding_sha256'],self.binding,'selected data');same(cp['settings'],settings,'selected public settings')
                same((cp['cell'],cp['seed'],cp['selected_epoch'],cp['selected_updates'],cp['committed_updates']),(cell,seed,epoch,epoch*12,1920),'selected member/step')
                same(state_hash(cp['state']),fit['history'][epoch-1]['validation_state_sha256'],'selected actual VAL weights');self.gates(cp['state'],cell)
                attempt=self.out/'attempts'/name/'start.json';same(sha(attempt),fit['initial_attempt_sha256'],'initial durable attempt');start=read(attempt)
                same(start['identity'],self.identity,'attempt source');same(start['dataset_binding_sha256'],self.binding,'attempt data');same((start['run_id'],start['cell'],start['seed'],start['maximum_committed_updates']),(name,cell,seed,1920),'one fixed fit attempt')
                self.selected_states[name]=cp['state'];scores[cell].append(score);details[name]={'selected_epoch':epoch,'VAL_RMSE':score,'VAL_NLL':nll,'checkpoint_sha256':sha(cp_path)}
        same(raw_count,60,'sixty actual raw VAL archives');selection=read(self.out/'selection.json')
        chosen=min(ELIGIBLE,key=lambda cell:float(np.mean(scores[cell])));same(selection['selected_cell'],chosen,'eligible first mean-three-member RMSE minimum')
        compare(selection['member_VAL_RMSE'],scores,'all member selection scores');compare(selection['mean_VAL_RMSE'],{k:float(np.mean(v)) for k,v in scores.items()},'all cell means')
        require(selection['technical_gate'] is True and selection['efficacy_screening'] is False and selection['TEST_or_parent_TEST_used_for_selection'] is False,'selection only locked TRAIN/VAL')
        self.selected=chosen;return {'members':details,'selected_cell':chosen,'member_VAL_RMSE':scores,'raw_VAL_archives':raw_count,'optimizer_updates_verified':28800}
    def numerical(self):
        selection=read(self.out/'numerical/selection.json');same(len(selection['candidates']),4,'all four numerical priors');scores={}
        parent_manifest=read(self.parent.out/'verification/manifest.json')
        for name,item in zip(NUMERICAL,selection['candidates']):
            p=ref(self.root,item);same(p,self.out/'numerical'/f'{name}.json','numeric candidate path');row=read(p);same(row['name'],name,'numeric identity')
            same(row['source_receipt_sha256'],self.source_sha,'numeric source');same(row['prepared_sha256'],sha(self.out/'data/prepared.json'),'numeric inputs');require(row['new_prior_fit'] is False,'zero new prior fits')
            prior=ref(self.root,row['prior']);same(prior,self.parent.out/'numerical'/f'{name}.npz','reused TRAIN prior');same(parent_manifest['evidence_sha256'][row['prior']['path']],row['prior']['sha256'],'parent seal prior')
            a=arrays(ref(self.root,row['validation_predictions']));same(set(a),{'mean','std'},'numeric raw arrays');score,nll,rows=self.validation_scores(a['mean'],a['std'])
            close(row['validation_rmse'],score,'numeric point score');close(row['validation_nll'],nll,'numeric secondary NLL');compare(row['validation_per_layout'],rows,'numeric whole VAL')
            if name!='hierarchy-fixed':
                original=ref(self.root,row['parent_predictions']);same(parent_manifest['evidence_sha256'][row['parent_predictions']['path']],row['parent_predictions']['sha256'],'old VAL sealed bytes')
                old=arrays(original)
                for k in a:np.testing.assert_array_equal(a[k],old[k])
            else:require(row['parent_predictions'] is None,'fixed prior VAL forward source distinct')
            ref(self.root,row['attempt']);scores[name]=score
        chosen=min(NUMERICAL,key=lambda n:scores[n]);same(selection['selected_name'],chosen,'best of all4 by heard RMSE');compare(selection['scores'],scores,'numeric scores')
        same(read(self.out/'selection.json')['selected_numerical'],chosen,'locked best numeric');self.numeric=chosen;return {'scores':scores,'selected_name':chosen,'new_prior_fits':0}
    def source_rows(self,e):
        b=e['binding']
        for role,id_key in (('a','context_measurement_ids_a'),('b','context_measurement_ids_b'),('query','query_measurement_ids')):
            case=b['case_id'] if role=='query' else b['context_case_id'];raw=self.reports(case);ids=b[id_key];parsed=[]
            for value in ids:
                m=re.fullmatch(re.escape(case)+r':pose(\d+):ap(\d+):band(\d+):dir(\d+):dev(\d+)',value);require(m is not None,'actual RF report ID');parsed.append(tuple(map(int,m.groups())))
            ix=tuple(np.asarray(parsed,dtype='i8').T);pose,ap,band,direction,device=ix;view=e[role]
            np.testing.assert_array_equal(view['ap'],ap);np.testing.assert_array_equal(view['rx_xyz'],raw['rx_positions_m'][pose]);np.testing.assert_array_equal(e['ap_xyz'],raw['ap_positions_m'])
            distance=np.linalg.norm(raw['rx_positions_m'][pose].astype('f8')-raw['ap_positions_m'][ap].astype('f8'),axis=-1)
            freq=raw['carrier_hz'][band];receipt=read(self.out/'rt_data/cases'/case/'receipt.json');tx=receipt['known_tx_power_dbm']
            expected_fspl=tx-20*np.log10(4*np.pi*distance*freq/299792458.)
            close(view['fspl'],expected_fspl,'physical FSPL reference',atol=2e-5)
            offset=np.array([self.calibration['estimates'][str(d)]['offset_db'] for d in device],dtype='f4');np.testing.assert_array_equal(view['offset'],offset)
            q=view['q'] if role=='query' else view['context'][:,4:];dims=raw['bbox_dims_m']
            close(q[:,:3],raw['rx_positions_m'][pose]/dims,'public RX geometry');close(q[:,3:6],raw['ap_positions_m'][ap]/dims,'public AP geometry')
            close(q[:,6],np.log10(distance)/2,'public distance');close(q[:,7],np.log(freq/2.4e9),'actual carrier');close(q[:,8],2*direction-1,'direction')
            estimate=[self.calibration['estimates'][str(d)] for d in device]
            close(q[:,11],np.minimum([x['count'] for x in estimate],16)/16,'TRAIN calibration count');close(q[:,12],[x['unknown'] for x in estimate],'calibration status')
            close(q[:,13],np.array([x['quantization_db'] for x in estimate])/4,'TRAIN quantizer');close(q[:,14],raw['noise_floor_dbm'][device]/100,'public floor not query rank cutoff');close(q[:,15],np.full(len(ids),tx/30),'known TX')
            if role=='query':continue
            h=raw['heard'][ix];c=raw['censored'][ix];np.testing.assert_array_equal(view['known'],h|c);require(view['scheduled'].all(),'declared scheduled rows')
            reference=view['fspl']+offset;detector=raw['reported_detection_threshold_dbm'][ix];center=raw['rssi_dbm'][ix];bound=raw['censor_upper_dbm'][ix]
            lo=raw['quantization_interval_dbm'][ix+(np.zeros(len(ids),dtype='i8'),)];hi=raw['quantization_interval_dbm'][ix+(np.ones(len(ids),dtype='i8'),)]
            expected=np.column_stack((np.where(h,(center-reference)/30.,0.),np.where(c,(bound-reference)/30.,0.),h,c)).astype('f4')
            np.testing.assert_array_equal(view['context'][:,:4],expected)
            np.testing.assert_array_equal(view['lower'],np.where(h,np.maximum(lo,detector)-reference,-np.inf).astype('f4'))
            np.testing.assert_array_equal(view['upper'],np.where(h,hi-reference,np.where(c,bound-reference,np.inf)).astype('f4'))
            np.testing.assert_array_equal(view['floor'],(detector-reference).astype('f4'))
    def group(self,name):
        if name in self.groups:return self.groups[name]
        p=self.out/'data/groups'/f'{name}.json';row=read(p);stage=row['stage'];same(row['source_receipt_sha256'],self.source_sha,'fresh group source');same(row['registry_sha256'],self.registry_sha,'fresh registry')
        same(row['prepared_sha256'],sha(self.out/'data/prepared.json'),'fresh fitting identity');same(row['selection_sha256'],sha(self.out/'selection.json'),'fresh selected weights')
        same(row['stage_receipt_sha256'],sha(self.out/'rt_data'/f'{stage}.json'),'fresh physical cohort');same(len(row['bindings']),COUNTS[stage]*3,'complete fresh group')
        a=arrays(ref(self.root,row['arrays']));same(row['arrays']['path'],f'artifacts/nested_radio/data/groups/{name}.npz','canonical group arrays');require(all(len(v)==len(row['bindings']) for v in a.values()),'complete cached axes')
        result=[]
        for i,b in enumerate(row['bindings']):
            spec=self.registry['stages'][stage][i//3]['cases'][i%3];same((b['case_id'],b['layout_id'],b['stage'],b['index'],b['state']),(spec['case_id'],spec['layout_id'],stage,i//3,i%3),'canonical case order')
            roles={role:{k.split('__',1)[1]:v[i] for k,v in a.items() if k.startswith(role+'__')} for role in ('a','b','query','target')}
            for role in ('a','b','query'):same({k:data_hash(v) for k,v in roles[role].items()},b['model_input_sha256'][role],'input array hash')
            same(data_hash(a['ap_xyz'][i]),b['model_input_sha256']['ap_xyz'],'AP hash');same({k:data_hash(v) for k,v in roles['target'].items()},b['target_sha256'],'target hash')
            e={'binding':b,'ap_xyz':a['ap_xyz'][i],**roles};self.base.Audit.verify_target(self,b,roles['target']);self.source_rows(e);result.append(e)
        self.groups[name]=result;return result
    def prediction_group(self,name):
        if name in self.predictions:return self.predictions[name]
        examples=self.group(name);manifest=read(self.out/'predictions'/f'{name}.json');methods=[f'{c}-seed{s}' for c in CELLS for s in SEEDS]+list(NUMERICAL)
        same(manifest['methods'],methods,'all19 predictors');same(set(manifest['case_receipts']),{e['binding']['case_id'] for e in examples},'complete predictions')
        result=[]
        for i,e in enumerate(examples):
            case=e['binding']['case_id'];p=ref(self.root,manifest['case_receipts'][case]);same(p,self.out/'predictions'/name/f'{case}.json','canonical case receipt')
            row=read(p);same(row['example_index'],i,'example index');same(row['methods'],methods,'case methods');same(row['source_receipt_sha256'],self.source_sha,'prediction source')
            same(row['selection_sha256'],sha(self.out/'selection.json'),'prediction selected state');ref(self.root,row['example_group']);ref(self.root,row['attempt'])
            a=arrays(ref(self.root,row['predictions']));same(set(a),{'mean','std'},'prediction array membership');same(a['mean'].shape,a['std'].shape,'field axes');same(a['mean'].shape,(19,len(e['target']['heard'])),'all actual means')
            derived=[self.base.metrics(m,s,e['target']) for m,s in zip(a['mean'],a['std'])];compare(row['metrics'],derived,'independent field metrics')
            for j,method in enumerate(methods):result.append({'method':method,'layout_id':e['binding']['layout_id'],'state':e['binding']['state'],'metrics':derived[j],'mean':a['mean'][j],'std':a['std'][j],'target':e['target']})
        self.predictions[name]=result;return result
    def reduce(self,rows,stage,mapping):return self.base.Audit.reduce(self,rows,stage,mapping)
    def field(self):
        official=read(self.out/'report.json');mapping={cell:[f'{cell}-seed{s}' for s in SEEDS] for cell in CELLS};mapping.update({n:[n] for n in NUMERICAL});mapping.update(selected=mapping[self.selected],best_numerical=mapping[self.numeric])
        tables={r:self.reduce(self.prediction_group('test-'+r),'test',mapping) for r in ('sparse','full')};compare(official['layout_metrics'],tables,'all whole-layout metrics')
        summaries={regime:{n:{k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']} for n,rows in table.items()} for regime,table in tables.items()};compare(official['summary'],summaries,'aggregate field')
        contrasts=[]
        for a,b in CONTRASTS:
            x=[r['metrics']['rmse_db'] for r in tables['sparse'][a]];y=[r['metrics']['rmse_db'] for r in tables['sparse'][b]];contrasts.append({'candidate':a,'reference':b,**bootstrap(x,y)})
        compare(official['primary_contrasts'],contrasts,'all four adjusted intervals')
        sparse=self.prediction_group('test-sparse');per_member=[]
        for seed in SEEDS:
            each={cell:[f'{cell}-seed{seed}'] for cell in CELLS};each.update({n:[n] for n in NUMERICAL});each.update(selected=each[self.selected],best_numerical=each[self.numeric]);members=set().union(*map(set,each.values()))
            reduced=self.reduce([r for r in sparse if r['method'] in members],'test',each)
            for a,b in CONTRASTS:
                d=np.array([r['metrics']['rmse_db'] for r in reduced[a]])-np.array([r['metrics']['rmse_db'] for r in reduced[b]])
                per_member.append({'seed':seed,'candidate':a,'reference':b,'mean_delta_db':float(d.mean()),'layouts_lower':int((d<0).sum()),'layouts_equal':int((d==0).sum()),'layouts_higher':int((d>0).sum()),'descriptive_no_extra_hypothesis_test':True})
        compare(official['per_member_contrasts_descriptive'],per_member,'per-member signs')
        secondary={}
        for n in ('test-sparse-held6GHz','test-sparse-stale','ap_transfer-sparse','ap_transfer-full'):
            stage='ap_transfer' if n.startswith('ap_transfer') else 'test';table=self.reduce(self.prediction_group(n),stage,mapping)
            secondary[n]={m:{k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']} for m,rows in table.items()}
        compare(official['secondary_descriptive'],secondary,'all secondary groups');delta={}
        for name,members in mapping.items():
            layouts=[]
            for spec in self.registry['stages']['test']:
                errors=[]
                for member in members:
                    state={r['state']:r for r in sparse if r['method']==member and r['layout_id']==spec['layout_id']};res=[]
                    for changed in (1,2):
                        a,b=state[0],state[changed];h=a['target']['heard']&b['target']['heard'];res.extend(((b['mean'][h]-a['mean'][h])-(b['target']['center_dbm'][h]-a['target']['center_dbm'][h])).tolist())
                    require(res,'paired heard delta endpoint');errors.append(float(np.sqrt(np.mean(np.square(res)))))
                layouts.append({'layout_id':spec['layout_id'],'delta_RMSE_db':float(np.mean(errors))})
            delta[name]={'mean_delta_RMSE_db':float(np.mean([r['delta_RMSE_db'] for r in layouts])),'layouts':layouts}
        compare(official['localized_delta_descriptive'],delta,'all localized changes');same((official['fits'],official['fresh_RT_cases'],official['reused_TRAIN_VAL_RT_cases'],official['native_simulations']),(15,84,90,0),'RF-only completed scope')
        require(official['production_qualified'] is False and official['default_changed'] is False and official['original_SERVICE_opened'] is False,'no promotion/SERVICE')
        same(official['selected_can_equal_global8_then_null_not_residual_gain'],self.selected=='global8_reg','exact baseline alias scope')
        return {'primary_contrasts':contrasts,'summary':summaries,'prediction_records':sum(len(v) for v in self.predictions.values()),'all_fresh_inputs_and_reported_targets_rejoined':True}
    def representations(self):
        value=read(self.out/'validation_representation.json');same(value['source_receipt_sha256'],self.source_sha,'rank source');same(value['selection_sha256'],sha(self.out/'selection.json'),'rank selected weights')
        require(value['used_for_selection'] is False and value['scientific_TEST_inputs_used'] is False,'VAL-only descriptive rank')
        names=[f'{c}-seed{s}' for c in CELLS for s in SEEDS];same([r['member'] for r in value['rows']],names,'all15 rank members');result=[]
        for row in value['rows']:
            name=row['member'];cell=name.rsplit('-seed',1)[0];a=arrays(ref(self.root,row['feature_arrays']));same(set(a),{'ungated_local_features','effective_local_features'},'rank raw arrays')
            require(all(x.shape==(1152,16) and np.isfinite(x).all() for x in a.values()),'complete VAL feature matrices')
            state=self.selected_states[name];gate=self.gates(state,cell);diag=row['gates_and_effective_priors'];np.testing.assert_array_equal(np.asarray(diag['gates'],dtype=gate.dtype),gate)
            np.testing.assert_array_equal(a['effective_local_features'],a['ungated_local_features']*gate[None])
            compare(row['fixed_Fourier8'],ranks(a['ungated_local_features'][:,:8]),'fixed rank');compare(row['learned_geometry8'],ranks(a['ungated_local_features'][:,8:]),'learned rank');compare(row['effective_local16'],ranks(a['effective_local_features']),'effective rank')
            active=cell not in ('global8_reg','global8_joint');same(row['learned_local_features_used_in_this_cell'],active,'active representation scope');same(row['inactive_global_features_untrained_not_a_representation'],not active,'inactive/random features')
            mu=state['local_prior_mean'].detach().cpu().numpy().astype('f8');raw=state['raw_local_std'].detach().cpu().numpy();sigma=.2+np.logaddexp(np.zeros_like(raw),raw).astype('f8')
            mean=10*gate*mu;sd=10*gate*sigma;rms=np.sqrt(mean*mean+sd*sd)
            close(diag['effective_gated_prior_mean_db'],mean,'effective prior mean');close(diag['effective_gated_prior_std_db'],sd,'effective prior SD');close(diag['effective_gated_prior_rms_db'],rms,'effective prior RMS')
            close(diag['gate_penalty'],.01*float(rms.mean())/5 if cell.startswith('nested_') else 0.,'effective RMS penalty')
            ref(self.root,row['checkpoint']);ref(self.root,row['attempt']);result.append({'member':name,'active':active,'learned_geometry8':row['learned_geometry8'],'effective_local16':row['effective_local16'],'gate_mean':float(gate.mean()),'effective_prior_rms_mean_db':float(rms.mean())})
        return {'members':result,'matrices_verified':15,'rows_per_matrix':1152,'no_neural_forward_repeated':True,'rank_is_not_collapse_or_identifiability_proof':True}


def physical(audit):
    count=0;expected=set()
    for stage,n in COUNTS.items():
        manifest=read(audit.out/'rt_data'/f'{stage}.json');cases={c['case_id']:c for g in audit.registry['stages'][stage] for c in g['cases']}
        same(set(manifest['case_receipts']),set(cases),'all fresh physical stage cases');same(len(cases),n*3,'fixed physical count')
        for case,spec in cases.items():
            p=ref(audit.root,manifest['case_receipts'][case]);same(p,audit.out/'rt_data/cases'/case/'receipt.json','physical receipt path');row=read(p)
            same(row['case'],spec,'exact registered physical case');audit.reports(case);expected.add(p.parent);count+=1
    same(count,84,'84new ray-traced fields');same(set((audit.out/'rt_data/cases').iterdir()),expected,'exact physical case directories')
    same(set((audit.out/'rt_data/cases').rglob('attempt.json')),{p/'attempt.json' for p in expected},'one use physical attempts')
    same(audit.registry['prior_geometry_records'],240,'all prior geometry records');require(audit.registry['all_new_geometries_disjoint_from_prior240'] is True,'registered fresh split')
    for name,digest in audit.registry['prior_registry_sha256'].items():same(sha(path(audit.root,name)),digest,'prior geometry registry')
    return count


def self_test(root):
    b=previous(root)
    target={'heard':np.array([True,False]),'censored':np.array([False,True]),'center_dbm':np.array([-70.,0.]),'lower_dbm':np.array([-70.5,0.]),'upper_dbm':np.array([-69.5,0.]),'censor_upper_dbm':np.array([0.,-75.]),'detection_threshold_dbm':np.array([-90.,-90.])}
    same(b.metrics(np.array([-69.,-75.]),np.ones(2)*3,target)['rmse_db'],1.,'analytic heard RMSE')
    zero=bootstrap(np.ones(24),np.ones(24));same(zero['central_98_75_interval'],[0.,0.],'exact null alias');require(not zero['candidate_lower_error_supported'],'null not gain')
    lower=bootstrap(np.zeros(24),np.ones(24));same(lower['central_98_75_interval'],[-1.,-1.],'paired sign')
    x=np.arange(24,dtype='f4').reshape(6,4);same(ranks(x)['participation_rank_centered'],1.,'rankone toy geometry');same(ranks(np.zeros((6,4)))['participation_rank_centered'],0.,'zero active design')
    print(json.dumps({'passes':True,'scope':'synthetic arithmetic/parser self-test only','scientific_calls':0}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);parser.add_argument('--self-test',action='store_true');args=parser.parse_args();root=args.root.resolve()
    if args.self_test:self_test(root);return
    out=root/'artifacts/nested_radio';review=root/'artifacts/nested_radio_review';require((out/'verification/manifest.json').is_file() and (out/'report.json').is_file(),'actual audit requires completed official seal')
    script_sha=sha(Path(__file__));manifest_sha=sha(out/'verification/manifest.json');attempt={'schema':1,'audit_source_sha256':script_sha,'study_manifest_sha256':manifest_sha,'started_unix_ns':time.time_ns(),'scientific_calls':0}
    with (review/'actual_attempt.json').open('xb') as f:f.write(encoded(attempt));f.flush()
    began=time.perf_counter()
    try:
        m,r,union,provenance=verify_closures(root,out);base=previous(root);audit=Audit(root,out,base)
        physical_count=physical(audit);calibration=audit.fitting_data();training=audit.training();numerical=audit.numerical();representations=audit.representations();field=audit.field()
        for name,digest in union.items():same(sha(path(root,name)),digest,'unchanged after replay '+name)
        same(sha(Path(__file__)),script_sha,'frozen auditor');same(sha(out/'verification/manifest.json'),manifest_sha,'unchanged seal')
        result={'schema':1,'passes':True,'audit_source_sha256':script_sha,'study_manifest_sha256':manifest_sha,'provenance':provenance,
            'prior_independent_parser':{'path':DEPENDENCY,'sha256':DEPENDENCY_SHA,'reused_scope':'reported target/event arithmetic and immutable parent TRAIN/VAL/calibration readers only'},
            'fresh_RT_cases':physical_count,'reused_TRAIN_VAL_cases':90,'calibration':calibration,'training':training,'numerical':numerical,'representations':representations,'field':field,
            'scientific_calls':{'model_constructors':0,'fits':0,'forwards':0,'RT_solves':0,'packet_simulations':0},'checkpoint_deserialization_only':True,
            'source_evidence_unchanged':True,'native_simulations':0,'SERVICE_opened':False,'production_promotion':False,'elapsed_seconds':time.perf_counter()-began}
        with (review/'independent.json').open('xb') as f:f.write(encoded(result));f.flush()
        print(json.dumps({'passes':True,'fresh_RT_cases':84,'fits_verified':15,'field_records':field['prediction_records'],'native_simulations':0,'output':str(review/'independent.json')}))
    except BaseException as error:
        value={**attempt,'passes':False,'error_type':type(error).__name__,'reason':str(error),'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-began}
        with (review/'first_failure.json').open('xb') as f:f.write(encoded(value));f.flush()
        raise


if __name__=='__main__':main()
