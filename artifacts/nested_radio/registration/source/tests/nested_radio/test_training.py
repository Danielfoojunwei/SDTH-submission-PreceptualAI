"""Disposable analytic trainer receipts; never registered research runs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from wirelessjev.nested_radio import training as t


def example(stage='validation',index=0,state=0,regime='sparse'):
    aps=np.array(((0.,0.,3.),(10.,0.,3.)),dtype='f4')
    routes=np.array((0,1),dtype='i8');rx=np.array(((1.,0.,1.),(11.,0.,1.)),dtype='f4')
    q=np.zeros((2,16),dtype='f4');q[:,6]=np.log10(np.linalg.norm(rx-aps,axis=-1))/2
    basis=np.zeros((2,8),dtype='f4');basis[:,0]=1.;basis[:,1]=q[:,6]
    heard=np.array((True,False));known=np.ones(2,bool)
    context=np.column_stack(([-10/30,0],[0,-12/30],heard,~heard,q)).astype('f4')
    a={'context':context,'known':known,'scheduled':known.copy(),'ap':routes,'rx_xyz':rx,
       'lower':np.array((-10.5,-np.inf),dtype='f4'),'upper':np.array((-9.5,-12.),dtype='f4'),
       'floor':np.full(2,-35.,dtype='f4'),'fspl':np.full(2,-60.,dtype='f4'),'offset':np.zeros(2,dtype='f4')}
    b={k:v.copy() for k,v in a.items()};b['context'][0,0]=7.
    query={'q':q,'ap':routes,'rx_xyz':rx,'basis':basis,'fspl':np.full(2,-60.,dtype='f4'),'offset':np.zeros(2,dtype='f4')}
    target={'center_dbm':np.array((-70.,0.),dtype='f4'),'lower_dbm':np.array((-70.5,0.),dtype='f4'),
        'upper_dbm':np.array((-69.5,0.),dtype='f4'),'heard':heard,'censored':~heard,
        'censor_upper_dbm':np.array((0.,-77.),dtype='f4'),'detection_threshold_dbm':np.full(2,-90.,dtype='f4')}
    case=f'{stage}-{index}-{state}'
    result={'a':a,'b':b,'query':query,'target':target,'ap_xyz':aps,'binding':{
        'stage':stage,'index':index,'state':state,'condition':state,'layout_id':f'{stage}-{index}',
        'case_id':case,'context_case_id':case,'regime':regime,'held_band':False,'stale_requested':False}}
    from wirelessjev.hierarchical_radio.data import array_digest
    result['binding']['model_input_sha256']={role:{key:array_digest(value) for key,value in result[role].items()} for role in ('a','b','query')}
    result['binding']['model_input_sha256']['ap_xyz']=array_digest(aps)
    result['binding']['target_sha256']={key:array_digest(value) for key,value in target.items()}
    return result


def examples(stage='validation',regime='sparse'):
    return [example(stage,i,j,regime) for i in range(24 if stage=='training' else 6) for j in range(3)]


def fixture_common(tmp_path):
    root=tmp_path/'software-repository';out=root/'artifacts/unit-nested';work=root/'WORK';out.mkdir(parents=True)
    return SimpleNamespace(ROOT=root,OUT=out,WORK=work,digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(),
        read=lambda p:json.loads(Path(p).read_bytes()))


class Probe(nn.Module):
    def __init__(self,cell='global8_reg'):
        super().__init__();self.bias=nn.Parameter(torch.tensor(0.,dtype=torch.float64))
        self.register_buffer('local_gates',torch.zeros(16,dtype=torch.float64))
        self.settings=t.settings(cell);self.config={'cell':'global8_reg'};self.calls=0;self.forbid=False
    def forward(self,view,query,ap_xyz):
        if self.forbid:raise AssertionError('Existing archive cannot trigger a replacement forward')
        self.calls+=1;assert (view['context'][:,0,0]<0).all()
        return query['fspl'].double()-10.+self.bias,torch.full_like(query['fspl'].double(),3.)


def trace_and_history(seed=0,cell='global8_reg'):
    rng=np.random.default_rng(seed);trace=[]
    for i in range(t.COMMITTED_UPDATES):
        indices,layouts,conditions,regime=t.choose_minibatch(rng,i)
        trace.append({'step':i+1,'epoch':i//12+1,'update_in_epoch':i%12+1,'regime':regime,
            'layout_indices':layouts.tolist(),'conditions':conditions.tolist(),
            'case_ids':[f'training-{a}-{b}' for a,b in zip(layouts,conditions)],
            'loss':1.,'measurement_nll':1.,'alignment':0.,'alignment_weight':0.,'gate_penalty':0.,
            'supported_agreement_queries':0.,'mean_predictive_std_db':3.,'gradient_norm_before_clip':1.,
            'mean_inference_step_size':1.,'inference_null_steps':0.,'gate_mean':0.,'gate_zero_count':16.,
            'effective_prior_rms_normalized_mean':0.,'effective_prior_rms_mean_db':0.,
            'gate_penalty_weight':0.,'gate_rms_reference':.5})
    history=[]
    for epoch in range(1,161):
        eligible=epoch in t.VALIDATION_EPOCHS
        score={40:3.,80:2.,120:2.,160:4.}.get(epoch)
        rows=[{'layout_id':f'validation-{i}','conditions':[0,1,2],'rmse_db':score,'nll':10-score} for i in range(6)] if eligible else None
        history.append({'epoch':epoch,'completed_updates':epoch*12,'mean_training_loss':1.,'validation_rmse':score,
            'validation_nll':10-score if eligible else None,'validation_per_layout':rows,
            'validation_predictions':{'path':f'unit/epoch{epoch}.npz','sha256':'a'*64} if eligible else None,
            'validation_state_sha256':'b'*64 if eligible else None,
            'epoch_checkpoint':{'path':f'unit/epoch{epoch}.pt','sha256':'c'*64} if eligible else None})
    return trace,history


def test_fixed_budget_and_matching_first480_minibatch_schedule():
    from wirelessjev.hierarchical_radio import training as old
    a,b=np.random.default_rng(2),np.random.default_rng(2)
    for i in range(480):
        actual=t.choose_minibatch(a,i);expected=old.choose_minibatch(b,i)
        for x,y in zip(actual[:3],expected[:3]):np.testing.assert_array_equal(x,y)
        assert actual[3]==expected[3]
    assert t.COMMITTED_UPDATES==1920 and t.EPOCHS==160
    assert t.choose_minibatch(a,1919)[3]=='sparse'
    for index in (1920,-1,True,1.0):
        with pytest.raises(ValueError):t.choose_minibatch(a,index)


def test_trace_replays_rng_and_real_case_membership():
    trace,_=trace_and_history();full=examples('training','full');sparse=examples('training','sparse')
    t.validate_trace(trace,cell='global8_reg',complete=True)
    t.verify_schedule(trace,0,full,sparse)
    altered=deepcopy(trace);altered[1200]['case_ids'][0]='not-a-training-case'
    with pytest.raises(RuntimeError,match='case IDs'):t.verify_schedule(altered,0,full,sparse)
    altered=deepcopy(trace);altered[0]['layout_indices'][0]=altered[0]['layout_indices'][1]
    with pytest.raises(RuntimeError):t.validate_trace(altered)


def test_checkpoint_selection_first_rmse_minimum_not_composite_nll():
    trace,history=trace_and_history();t.validate_history(history,trace,cell='global8_reg',complete=True)
    assert t.selected_validation(history)==(80,2.)
    assert min((r for r in history if r['validation_nll'] is not None),key=lambda r:r['validation_nll'])['epoch']==160
    history[40]['validation_rmse']=0.
    with pytest.raises(RuntimeError,match='Unregistered'):t.validate_history(history,trace)


@pytest.mark.parametrize('change',[
    lambda a:a.update(loss=float('nan')),
    lambda a:a.update(gate_mean=1.1),
    lambda a:a.update(gate_zero_count=17.),
    lambda a:a.update(gate_penalty=.1),
    lambda a:a.update(alignment_weight=.1),
])
def test_nonfinite_or_changed_fixed_objective_rejected(change):
    trace,_=trace_and_history();change(trace[0])
    with pytest.raises(RuntimeError):t.validate_trace(trace,cell='global8_reg')


def test_rmse_reducer_equal_states_and_layouts_not_query_pool():
    es=examples();mean=np.stack([e['target']['center_dbm'].astype('f8') for e in es]);std=np.full_like(mean,3.)
    for i,e in enumerate(es):mean[i,0]+=float(i%3+1)
    score,nll,rows=t.reduce_validation_predictions(mean,std,es)
    assert score==2. and all(row['rmse_db']==2. for row in rows) and np.isfinite(nll)
    assert score!=pytest.approx(np.sqrt((1+4+9)/3))
    with pytest.raises(RuntimeError):t.reduce_validation_predictions(mean[:9],std[:9],es)
    bad=std.copy();bad[-1,-1]=np.nan
    with pytest.raises(RuntimeError):t.reduce_validation_predictions(mean,bad,es)


def test_raw_validation_archive_uses_public_settings_and_never_reforwards(tmp_path,monkeypatch):
    c=fixture_common(tmp_path);monkeypatch.setattr(t,'_common',lambda:c)
    model=Probe('global8_joint');es=examples();binding=t.data_identity(examples('training','full'),examples('training','sparse'),es)
    ident={'source_receipt_sha256':'a'*64,'protocol_sha256':'b'*64,'data_receipt_sha256':'c'*64}
    first=t.validation_at_epoch(model,es,t.batch(es),ident,binding,'global8_joint',0,40)
    model.forbid=True;calls=model.calls
    second=t.validation_at_epoch(model,es,t.batch(es),ident,binding,'global8_joint',0,40)
    assert first==second and model.calls==calls==6
    weights=torch.load(c.OUT/'epoch_checkpoints/global8_joint-seed0-epoch40.pt',weights_only=True)
    assert weights['settings']['cell']=='global8_joint' and weights['committed_updates']==480
    with torch.no_grad():model.bias.add_(1.)
    with pytest.raises(RuntimeError,match='exact source/data/member/epoch/weight'):
        t.validation_at_epoch(model,es,t.batch(es),ident,binding,'global8_joint',0,40)


@pytest.mark.parametrize('value',[float('nan'),-1e-4,1.0001])
def test_saved_gate_values_are_rejected_not_silently_projected(value):
    state={'local_gates':torch.full((16,),.1)};state['local_gates'][0]=value
    with pytest.raises(RuntimeError):t._gates_valid(state,'nested_reg')
    assert float(state['local_gates'][0])==pytest.approx(value,nan_ok=True)


def test_control_gate_buffers_are_fixed():
    for cell,value in (('global8_reg',0.),('global8_joint',0.),('hier_reg',1.)):
        state={'local_gates':torch.full((16,),value)};t._gates_valid(state,cell)
        state['local_gates'][0]=.1
        with pytest.raises(RuntimeError,match='Fixed-control'):t._gates_valid(state,cell)


def test_spawn_context_and_bounded_workers():
    assert t.training_worker_context().get_start_method()=='spawn'
    for workers in (0,4,True,1.0):
        with pytest.raises(ValueError):t.train_all(workers)


def test_checkpoint_distinguishes_total_training_budget_from_selected_weight_epoch(tmp_path,monkeypatch):
    c=fixture_common(tmp_path);monkeypatch.setattr(t,'_common',lambda:c)
    es=examples();full=examples('training','full');sparse=examples('training','sparse')
    monkeypatch.setattr(t,'datasets',lambda:(full,sparse,es))
    binding=t.data_identity(full,sparse,es)
    ident={'source_receipt_sha256':'a'*64,'protocol_sha256':'b'*64,'data_receipt_sha256':'c'*64}
    trace,history=trace_and_history();model=Probe()
    for epoch in t.VALIDATION_EPOCHS:
        score,nll,rows,ref,weights,epoch_ref=t.validation_at_epoch(model,es,t.batch(es),ident,binding,'global8_reg',0,epoch)
        history[epoch-1].update(validation_rmse=score,validation_nll=nll,validation_per_layout=rows,
            validation_predictions=ref,validation_state_sha256=weights,epoch_checkpoint=epoch_ref)
    epoch,score=t.selected_validation(history);assert epoch==40
    checkpoint={'identity':ident,'dataset_binding_sha256':binding,'cell':'global8_reg','seed':0,'settings':t.settings('global8_reg'),
        'selected_epoch':40,'selected_updates':480,'committed_updates':1920,'validation_rmse':score,
        'validation_nll':history[39]['validation_nll'],'state':model.state_dict()}
    path=c.OUT/'checkpoints/global8_reg-seed0.pt';t._publish_torch(path,checkpoint,staging=c.WORK)
    start=c.OUT/'attempts/global8_reg-seed0/start.json'
    t._publish_json(c,start,{'identity':ident,'dataset_binding_sha256':binding,'run_id':'global8_reg-seed0','cell':'global8_reg',
        'seed':0,'maximum_committed_updates':1920,'selection_epochs':[40,80,120,160],'before_first_optimizer_step':True})
    receipt={'complete':True,'completed':True,'identity':ident,'dataset_binding_sha256':binding,'cell':'global8_reg','seed':0,
        'run_id':'global8_reg-seed0','epochs':160,'updates':1920,'committed_updates':1920,'settings':t.settings('global8_reg'),
        'optimizer_spec':json.loads(json.dumps(t.OPTIMIZER_SPEC)),'checkpoint_sha256':c.digest(path),'update_trace':trace,'history':history,
        'selected_epoch':40,'selected_updates':480,'validation_rmse':score,'validation_nll':history[39]['validation_nll'],
        'initial_attempt_sha256':c.digest(start)}
    result=t.verify_completed(receipt,ident,'global8_reg',0,validation_examples=es)
    assert result['committed_updates']==1920 and result['selected_updates']==480
    monkeypatch.setattr(t,'datasets',lambda:(_ for _ in ()).throw(AssertionError('Explicit preload must not load fitting data again')))
    preloaded=t.verify_completed(receipt,ident,'global8_reg',0,validation_examples=es,fitting_examples=(full,sparse,es))
    assert torch.equal(result['state']['bias'],preloaded['state']['bias'])
    altered=deepcopy(full);altered[0]['a']['context'][0,0]+=1.
    with pytest.raises(RuntimeError,match='array bytes'):
        t.verify_completed(receipt,ident,'global8_reg',0,fitting_examples=(altered,sparse,es))
    changed=deepcopy(receipt);changed['selected_epoch']=160
    with pytest.raises(RuntimeError,match='first sparse'):t.verify_completed(changed,ident,'global8_reg',0,fitting_examples=(full,sparse,es))


def test_resume_preserves_source_rng_optimizer_and_rejects_gate_corruption():
    es=examples();full=examples('training','full');sparse=examples('training','sparse')
    binding=t.data_identity(full,sparse,es);ident={'source':'analytic-fixture'}
    model=Probe();optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    rng=np.random.default_rng(0)
    snapshot=t._payload(model,optimizer,rng,ident,binding,'global8_reg',0,[],[],None,.2)
    assert t.validate_resume(snapshot,ident,binding,'global8_reg',0,full,sparse) is snapshot
    changed=deepcopy(snapshot);changed['numpy_rng']=np.random.default_rng(1).bit_generator.state
    with pytest.raises(RuntimeError,match='RNG'):t.validate_resume(changed,ident,binding,'global8_reg',0)
    changed=deepcopy(snapshot);changed['optimizer']['param_groups'][0]['lr']=.006
    with pytest.raises(RuntimeError,match='optimizer'):t.validate_resume(changed,ident,binding,'global8_reg',0)
    changed=deepcopy(snapshot);changed['state']['local_gates'][0]=.1
    with pytest.raises(RuntimeError,match='Fixed-control'):t.validate_resume(changed,ident,binding,'global8_reg',0)
    changed=deepcopy(snapshot);changed['identity']={'source':'different'}
    with pytest.raises(RuntimeError,match='exact source'):t.validate_resume(changed,ident,binding,'global8_reg',0)


def test_identity_requires_published_source_and_exact_prepared_parent_receipts(tmp_path,monkeypatch):
    from wirelessjev.nested_radio import data
    c=fixture_common(tmp_path);calls=[]
    c.check_sources=lambda:calls.append('published-source')
    c.verify_parent_data=lambda:calls.append('parent')
    c.source_receipt_sha256=lambda:'a'*64;c.protocol_sha256=lambda:'b'*64
    monkeypatch.setattr(t,'_common',lambda:c)
    monkeypatch.setattr(data,'verify_prepared',lambda:calls.append('prepared'))
    p=c.OUT/'data/prepared.json';p.parent.mkdir();p.write_text('{"scope":"unit"}')
    observed=t.identity()
    assert calls==['published-source','parent','prepared']
    assert observed['data_receipt_sha256']==c.digest(p)
    def reject():raise RuntimeError('Prepared fitting receipt mutated')
    monkeypatch.setattr(data,'verify_prepared',reject)
    with pytest.raises(RuntimeError,match='Prepared'):t.identity()


def production_example(stage='training',index=0,state=0,regime='full'):
    """Explicit analytic field for the real production-shaped model only."""
    from wirelessjev.spectral_scene.operators import fspl_received_power_dbm,link_geometry_features,query_physics_basis
    from wirelessjev.hierarchical_radio.data import array_digest
    aps=np.array(((-6.,-6.,4.),(6.,-6.,4.),(-6.,6.,4.),(6.,6.,4.)),dtype='f4')
    aps[:,:2]+=(.02*index,-.01*index)
    positions=np.column_stack((np.stack(np.meshgrid(np.linspace(-5.,5.,6),np.linspace(-4.,4.,4)),-1).reshape(-1,2),np.ones(24)))
    bands=np.array((2.4e9,5.2e9));dimensions=np.array((16.,16.,5.));boxes=np.empty((0,2,3))
    def links(pos,sparse=False,direction=0):
        pose,ap,band=np.indices((24,4,2)).reshape(3,-1)
        if sparse:
            keep=pose%2==0;pose,ap,band=pose[keep],ap[keep],band[keep]
        rx=pos[pose].astype('f4');tx=aps[ap]
        q=link_geometry_features(rx,tx,bands[band],direction,dimensions,boxes,
            calibration_pair_count=16,calibration_unknown=False,quantization_width_db=1.,noise_floor_dbm=-100.,tx_power_dbm=20.)
        fspl=fspl_received_power_dbm(rx,tx,bands[band],20.).astype('f4')
        residual=(-9.-.1*rx[:,0]+.03*tx[:,0]-.4*state).astype('f4')
        heard=(pose+ap+band)%4!=0
        return ap.astype('i8'),rx,q,fspl,residual,heard
    def view(direction):
        ap,rx,q,fspl,residual,heard=links(positions,regime=='sparse',direction)
        offset=np.full(len(ap),2.5*direction,dtype='f4');residual=residual+.2*direction
        return {'context':np.column_stack((np.where(heard,residual/30.,0.),np.where(~heard,(residual+.5)/30.,0.),heard,~heard,q)).astype('f4'),
            'known':np.ones(len(ap),bool),'scheduled':np.ones(len(ap),bool),'ap':ap,'rx_xyz':rx,
            'lower':np.where(heard,residual-.5,-np.inf).astype('f4'),'upper':(residual+.5).astype('f4'),
            'floor':(-90.-fspl-offset).astype('f4'),'fspl':fspl,'offset':offset}
    ap,rx,q,fspl,residual,heard=links(positions+(.13,.17,0.));reported=fspl+residual
    result=example(stage,index,state,regime)
    result.update(a=view(0),b=view(1),ap_xyz=aps,query={'q':q,'ap':ap,'rx_xyz':rx,'basis':query_physics_basis(q),'fspl':fspl,'offset':np.zeros(len(ap),dtype='f4')},
        target={'center_dbm':np.where(heard,reported,0.).astype('f4'),'lower_dbm':np.where(heard,reported-.5,0.).astype('f4'),
            'upper_dbm':np.where(heard,reported+.5,0.).astype('f4'),'heard':heard,'censored':~heard,
            'censor_upper_dbm':np.where(~heard,reported+.5,0.).astype('f4'),'detection_threshold_dbm':np.full(len(ap),-90.,dtype='f4')})
    result['binding']['model_input_sha256']={role:{k:array_digest(v) for k,v in result[role].items()} for role in ('a','b','query')}
    result['binding']['model_input_sha256']['ap_xyz']=array_digest(aps)
    result['binding']['target_sha256']={k:array_digest(v) for k,v in result['target'].items()}
    return result


@pytest.mark.parametrize('cell',['global8_joint','nested_joint'])
def test_real_model_f32_optimizer_projection_at_full_and_sparse_shapes(cell):
    from wirelessjev.nested_radio.model import NestedField,objective
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17212);model=NestedField(cell)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
        for regime,tokens in (('full',192),('sparse',96)):
            values=t.batch([production_example(index=i,state=i%3,regime=regime) for i in range(4)])
            assert values['a']['context'].shape==(4,tokens,20) and values['query']['q'].shape==(4,192,16)
            optimizer.zero_grad(set_to_none=True)
            loss,details=objective(model,values['a'],values['b'],values['query'],values['target'],values['ap_xyz'])
            assert torch.isfinite(loss) and all(v.ndim==0 and torch.isfinite(v) for v in details.values())
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();model.project_gates_();t._gates_valid(model.state_dict(),cell)
            assert t._finite_tree(model.state_dict()) and t._finite_tree(optimizer.state_dict())
            assert float(details['gate_penalty'])==pytest.approx(float(details['gate_penalty_weight']*details['effective_prior_rms_normalized_mean']))
    finally:torch.set_num_threads(previous)


def test_real_f32_validation_survives_raw_archive_and_rejects_relabelled_tensors(tmp_path):
    from wirelessjev.nested_radio.model import NestedField
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        es=[production_example('validation',i,s,'sparse') for i in range(6) for s in range(3)]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17213);model=NestedField('nested_joint')
        score,nll,rows,mean,std=t.validation_score(model,es,return_predictions=True)
        assert mean.shape==std.shape==(18,192) and np.isfinite(score)
        assert t.reduce_validation_predictions(mean,std,es)==(score,nll,rows)
        archive=tmp_path/'analytic-validation.npz';metadata={'scope':'analytic-software-only'}
        np.savez_compressed(archive,mean=mean,std=std,metadata_utf8=np.frombuffer(t._canonical(metadata),dtype=np.uint8))
        reread=t.read_validation_archive(archive,metadata,es)
        assert reread[:3]==(score,nll,rows)
        values=t.batch(es);values['a']['context'][0,0,0]+=1.
        with pytest.raises(RuntimeError,match='Batched validation'):t.validation_score(model,es,values)
    finally:torch.set_num_threads(previous)
