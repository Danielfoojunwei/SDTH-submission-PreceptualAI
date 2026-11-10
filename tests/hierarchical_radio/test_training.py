"""Trainer durability/selection software fixtures; no registered scientific fit."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from wirelessjev.hierarchical_radio import training as t


def example(stage='validation',index=0,state=0,regime='sparse'):
    ap=np.array(((0.,0.,3.),(10.,0.,3.)),dtype='f4')
    routes=np.array((0,1),dtype='i8');rx=np.array(((1.,0.,1.),(11.,0.,1.)),dtype='f4')
    q=np.zeros((2,16),dtype='f4');q[:,6]=np.log10(np.linalg.norm(rx-ap,axis=-1))/2
    basis=np.zeros((2,8),dtype='f4');basis[:,0]=1;basis[:,1]=q[:,6]
    heard=np.array((True,False));known=np.ones(2,bool)
    context=np.column_stack(([-10/30,0],[0,-12/30],heard,~heard,q)).astype('f4')
    a={'context':context,'known':known,'scheduled':known.copy(),'ap':routes,'rx_xyz':rx,
       'lower':np.array((-10.5,-np.inf),dtype='f4'),'upper':np.array((-9.5,-12.),dtype='f4'),
       'floor':np.full(2,-35.,dtype='f4'),'fspl':np.full(2,-60.,dtype='f4'),'offset':np.zeros(2,dtype='f4')}
    b={k:v.copy() for k,v in a.items()};b['context'][0,0]=7.0 # Distinct B view for A-only routing proof.
    query={'q':q,'ap':routes,'rx_xyz':rx,'basis':basis,'fspl':np.full(2,-60.,dtype='f4'),'offset':np.zeros(2,dtype='f4')}
    target={'center_dbm':np.array((-70.,0.),dtype='f4'),'lower_dbm':np.array((-70.5,0.),dtype='f4'),
        'upper_dbm':np.array((-69.5,0.),dtype='f4'),'heard':heard,'censored':~heard,
        'censor_upper_dbm':np.array((0.,-77.),dtype='f4'),'detection_threshold_dbm':np.full(2,-90.,dtype='f4')}
    case=f'{stage}-{index}-{state}'
    return {'a':a,'b':b,'query':query,'target':target,'ap_xyz':ap,
        'binding':{'stage':stage,'index':index,'state':state,'condition':state,'layout_id':f'{stage}-{index}',
        'case_id':case,'context_case_id':case,'regime':regime,'held_band':False,'stale_requested':False}}


def examples(stage='validation',regime='sparse'):
    return [example(stage,i,j,regime) for i in range(24 if stage=='training' else 6) for j in range(3)]


def fixture_common(tmp_path):
    root=tmp_path/'unit-repository';out=root/'artifacts/unit-hierarchical';work=root/'WORK';out.mkdir(parents=True)
    def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return SimpleNamespace(ROOT=root,OUT=out,WORK=work,digest=digest,read=lambda p:json.loads(Path(p).read_bytes()))


class SelectionProbe(nn.Module):
    """Analytic constant-field software fixture, not an experimental model."""
    def __init__(self):
        super().__init__();self.bias=nn.Parameter(torch.tensor(0.,dtype=torch.float64));self.calls=0;self.forbid=False
    def forward(self,view,query,ap_xyz):
        if self.forbid:raise AssertionError('An existing raw VAL archive must not trigger another forward')
        self.calls+=1
        assert torch.all(view['context'][:,0,0]<0) # A only; B has positive7.
        mean=query['fspl'].double()-10+self.bias
        return mean,torch.full_like(mean,3.)


def trace_and_history(seed=0):
    trace=[];rng=np.random.default_rng(seed)
    for i in range(480):
        _,layouts,states,regime=t.choose_minibatch(rng,i)
        trace.append({'step':i+1,'epoch':i//12+1,'update_in_epoch':i%12+1,'regime':regime,
            'layout_indices':layouts.tolist(),'conditions':states.tolist(),
            'case_ids':[f'training-{a}-{b}' for a,b in zip(layouts,states)],
            'loss':1.,'measurement_nll':1.,'alignment':0.,'alignment_weight':0.,
            'supported_agreement_queries':0.,'mean_predictive_std_db':3.,'gradient_norm_before_clip':1.})
    history=[]
    scores={10:3.,20:2.,30:2.,40:4.}
    for epoch in range(1,41):
        score=scores.get(epoch)
        rows=[{'layout_id':f'validation-{i}','conditions':[0,1,2],'nll':score} for i in range(6)] if score is not None else None
        history.append({'epoch':epoch,'completed_updates':epoch*12,'mean_training_loss':1.,
            'validation_nll':score,'validation_per_layout_nll':rows,
            'validation_predictions':{'path':f'unit/epoch{epoch}.npz','sha256':'a'*64} if score is not None else None,
            'validation_state_sha256':'b'*64 if score is not None else None})
    return trace,history


def test_batch_keeps_integer_routes_masks_and_infinite_censor_bounds():
    es=[example(),example(index=1)];v=t.batch(es)
    assert v['a']['ap'].dtype==torch.int64 and v['query']['ap'].dtype==torch.int64
    assert v['a']['known'].dtype==v['target']['heard'].dtype==torch.bool
    assert v['a']['context'].dtype==v['a']['lower'].dtype==torch.float32
    assert torch.isneginf(v['a']['lower'][:,1]).all()
    es[0]['a']['context'][0,0]=999
    assert v['a']['context'][0,0,0]<0 # Owned batch copy, never a view onto mutable data.
    assert set(v)=={'a','b','query','target','ap_xyz'}
    assert t.select(v,[1])['a']['ap'].shape==(1,2)


@pytest.mark.parametrize('role,key,value',[
    ('a','ap',np.array((0.,1.))),('a','known',np.array((1,1))),
    ('query','hidden_material',np.array((1.,1.))),('target','private_truth',np.ones(2)),
])
def test_model_batch_rejects_wrong_index_mask_or_hidden_feature_channels(role,key,value):
    e=example();e[role][key]=value
    with pytest.raises(ValueError):t.batch([e])


def test_finite_overflow_does_not_turn_a_physical_value_into_an_interval_infinity():
    with pytest.raises(ValueError,match='overflowed'):t.tensor(np.array((1e300,)),'rx_xyz')
    assert torch.isposinf(t.tensor(np.array((np.inf,)),'upper')).all()
    with pytest.raises(ValueError):t.tensor(np.array((2**63,),dtype='u8'),'ap')


def test_fixed_sampling_is_paired_across_cells_and_uses_four_distinct_layouts():
    a=np.random.default_rng(2);b=np.random.default_rng(2)
    for i in range(480):
        selected,layouts,conditions,regime=t.choose_minibatch(a,i)
        other=t.choose_minibatch(b,i)
        np.testing.assert_array_equal(selected,other[0])
        assert len(set(layouts))==4 and np.all((0<=conditions)&(conditions<3))
        np.testing.assert_array_equal(selected,3*layouts+conditions)
        assert regime==('full' if i%2==0 else 'sparse')
    with pytest.raises(ValueError):t.choose_minibatch(a,480)


def test_only_exact_layout_major_current_train_and_validation_groups_are_admitted():
    group=examples('training','full');assert len(t.validate_examples(group,'training','full'))==24
    for field,new in [('stage','test'),('held_band',True),('stale_requested',True),('condition',2),('index',20)]:
        altered=deepcopy(group);altered[0]['binding'][field]=new
        with pytest.raises(RuntimeError):t.validate_examples(altered,'training','full')
    altered=deepcopy(group);altered[1]['binding']['context_case_id']='different-physical-state'
    with pytest.raises(RuntimeError):t.validate_examples(altered,'training','full')


def test_validation_routes_only_a_and_raw_arrays_reproduce_equal_layout_selection_score():
    model=SelectionProbe();es=examples();score,rows,mean,std=t.validation_score(model,es,return_predictions=True)
    assert model.calls==6 and model.training # Restore prior train/eval mode.
    assert mean.shape==std.shape==(18,2) and len(rows)==6
    other,other_rows=t.reduce_validation_predictions(mean,std,es)
    assert score==other and rows==other_rows


def test_selection_uses_only_four_registered_epochs_and_keeps_first_exact_tie():
    trace,history=trace_and_history();t.validate_history(history,trace,complete=True)
    assert t.selected_validation(history)==(20,2.)
    changed=deepcopy(history);changed[0]['validation_nll']=-100
    with pytest.raises(RuntimeError):t.validate_history(changed,trace,complete=True)
    changed=deepcopy(history);changed[19]['validation_nll']=1.
    with pytest.raises(RuntimeError):t.validate_history(changed,trace,complete=True)


def test_committed_final_update_can_resume_before_its_epoch_metadata_is_committed():
    trace,history=trace_and_history()
    t.validate_history(history[:-1],trace)
    with pytest.raises(RuntimeError):t.validate_history(history[:-1],trace,complete=True)
    with pytest.raises(RuntimeError):t.validate_history(history[:-2],trace)
    changed=deepcopy(trace);changed[0]['layout_indices'][1]=changed[0]['layout_indices'][0]
    with pytest.raises(RuntimeError):t.validate_trace(changed)
    changed=deepcopy(trace);changed[0]['loss']=float('nan')
    with pytest.raises(RuntimeError):t.validate_trace(changed)


def test_exact_seed_schedule_verification_rejects_silent_draw_or_seed_changes():
    trace,_=trace_and_history();t.verify_schedule(trace,0)
    with pytest.raises(RuntimeError):t.verify_schedule(trace,1)
    changed=deepcopy(trace);changed[12]['conditions'][0]=(changed[12]['conditions'][0]+1)%3
    with pytest.raises(RuntimeError):t.verify_schedule(changed,0)


def test_atomic_optimizer_rng_snapshot_reproduces_the_uninterrupted_next_step(tmp_path):
    # Tiny optimization fixture checks trajectory persistence; it is not one
    # of the15 scientific models and writes only the temporary test directory.
    torch.manual_seed(231);rng=np.random.default_rng(27)
    model=nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    def step(m,o,g):
        x=torch.from_numpy(g.normal(size=(4,2)).astype('f4'))
        y=torch.randn(4,1)
        o.zero_grad(set_to_none=True);loss=(m(x)-y).square().mean();loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True);o.step()
    step(model,optimizer,rng)
    path=tmp_path/'atomic.pt';t.atomic_save(path,{'state':model.state_dict(),'optimizer':optimizer.state_dict(),
        'torch_rng':torch.get_rng_state(),'numpy_rng':rng.bit_generator.state})
    step(model,optimizer,rng);expected=deepcopy(model.state_dict())
    saved=torch.load(path,map_location='cpu',weights_only=True)
    resumed=nn.Linear(2,1);resumed.load_state_dict(saved['state'])
    ropt=torch.optim.AdamW(resumed.parameters(),lr=.003,weight_decay=1e-4);ropt.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng']);rrng=np.random.default_rng();rrng.bit_generator.state=saved['numpy_rng']
    step(resumed,ropt,rrng)
    for key in expected:assert torch.equal(expected[key],resumed.state_dict()[key])
    assert not list(tmp_path.glob('.committed-*'))


def test_exclusive_fit_lock_does_not_guess_or_reuse_a_live_owner(tmp_path):
    path=tmp_path/'member.lock'
    with t.exclusive_fit(path):
        with pytest.raises(RuntimeError,match='Another process'):
            with t.exclusive_fit(path):pass
    with t.exclusive_fit(path):pass


def test_immutable_torch_publication_is_inert_and_never_overwrites_existing_bytes(tmp_path):
    path=tmp_path/'evidence'/'best.pt';work=tmp_path/'WORK'
    t._publish_torch(path,{'state':{'weight':torch.tensor((1.,2.))}},staging=work)
    original=path.read_bytes();saved=torch.load(path,weights_only=True,map_location='cpu')
    assert torch.equal(saved['state']['weight'],torch.tensor((1.,2.)))
    with pytest.raises(FileExistsError):t._publish_torch(path,{'different':torch.tensor(3.)},staging=work)
    assert path.read_bytes()==original and not list(work.iterdir())


def test_raw_validation_archive_survives_metadata_interrupt_without_forward_repeat(tmp_path,monkeypatch):
    c=fixture_common(tmp_path);monkeypatch.setattr(t,'_common',lambda:c)
    es=examples();bindings={'validation_sparse':hashlib.sha256(t._canonical([e['binding'] for e in es])).hexdigest()}
    model=SelectionProbe();ident={'source_receipt_sha256':'a'*64,'protocol_sha256':'b'*64,'data_receipt_sha256':'c'*64}
    first=t.validation_at_epoch(model,es,t.batch(es),ident,bindings,'hier_reg',0,10)
    assert model.calls==6
    model.forbid=True
    second=t.validation_at_epoch(model,es,t.batch(es),ident,bindings,'hier_reg',0,10)
    assert first==second and model.calls==6
    assert len(list((c.OUT/'validation_predictions').glob('*.npz')))==1
    with torch.no_grad():model.bias.add_(1.)
    with pytest.raises(RuntimeError,match='exact source/data/member/epoch/weight'):
        t.validation_at_epoch(model,es,t.batch(es),ident,bindings,'hier_reg',0,10)


def test_raw_validation_identity_and_numeric_corruption_are_not_new_selection_opportunities(tmp_path,monkeypatch):
    c=fixture_common(tmp_path);monkeypatch.setattr(t,'_common',lambda:c)
    es=examples();bindings={'validation_sparse':hashlib.sha256(t._canonical([e['binding'] for e in es])).hexdigest()}
    ident={'source_receipt_sha256':'a'*64};model=SelectionProbe()
    score,rows,reference,weights=t.validation_at_epoch(model,es,t.batch(es),ident,bindings,'hier_reg',0,10)
    path=c.ROOT/reference['path'];metadata=t.validation_metadata(ident,bindings,'hier_reg',0,10,weights)
    altered=dict(metadata,seed=1)
    with pytest.raises(RuntimeError):t.read_validation_archive(path,altered,es)
    with np.load(path,allow_pickle=False) as raw:mean=raw['mean'].copy();std=raw['std'].copy();meta=raw['metadata_utf8'].copy()
    std[0,0]=np.nan
    # Deliberately corrupted disposable unit fixture; never a study artifact.
    np.savez_compressed(path,mean=mean,std=std,metadata_utf8=meta)
    with pytest.raises(RuntimeError,match='finite positive-spread'):t.read_validation_archive(path,metadata,es)


def test_scientific_fit_refuses_unpublished_sources_before_attempt_or_model(tmp_path,monkeypatch):
    c=fixture_common(tmp_path)
    def blocked():raise RuntimeError('Unit fixture: publication guard refused')
    c.check_sources=blocked;monkeypatch.setattr(t,'_common',lambda:c)
    with pytest.raises(RuntimeError,match='publication guard refused'):t.fit('hier_reg',0)
    assert not (c.OUT/'attempts').exists() and not c.WORK.exists()


@pytest.mark.parametrize('cell,seed',[('unknown',0),('hier_reg',3),('hier_reg',True),('hier_reg',.5)])
def test_unregistered_member_never_reaches_scientific_authorization(cell,seed):
    with pytest.raises(ValueError):t.fit(cell,seed)


def test_training_process_context_does_not_fork_live_torch_or_drjit_threads():
    assert t.training_worker_context().get_start_method()=='spawn'


@pytest.mark.parametrize('workers',[0,4,True,2.5,'3'])
def test_training_worker_count_is_bounded_before_any_scientific_action(workers):
    with pytest.raises(ValueError,match='one to three'):t.train_all(workers)


def test_resume_refuses_changed_identity_rng_or_optimizer_hyperparameters():
    model=nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    ident={'source':'unit-source'};data={'training':'unit-data'}
    prior={'identity':ident,'dataset_binding_sha256':data,'cell':'hier_reg','seed':0,
        'run_id':'hier_reg-seed0','settings':{'cell':'hier_reg','width':32,'inference_steps':3},
        'optimizer_spec':t.OPTIMIZER_SPEC,'committed_updates':0,'state':model.state_dict(),
        'optimizer':optimizer.state_dict(),'update_trace':[],'history':[],
        'best_epoch':None,'best_score':None,'best_state':None,'elapsed_seconds':0.,
        'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.default_rng(0).bit_generator.state}
    assert t.validate_resume(prior,ident,data,'hier_reg',0) is prior
    for key,value in [('identity',{'other':'source'}),('dataset_binding_sha256',{'other':'data'}),
                      ('seed',1),('numpy_rng',np.random.default_rng(2).bit_generator.state)]:
        altered=deepcopy(prior);altered[key]=value
        with pytest.raises(RuntimeError):t.validate_resume(altered,ident,data,'hier_reg',0)
    altered=deepcopy(prior);altered['optimizer']['param_groups'][0]['lr']=.01
    with pytest.raises(RuntimeError,match='hyperparameters'):t.validate_resume(altered,ident,data,'hier_reg',0)
    altered=deepcopy(prior);altered['state']['weight'][0,0]=torch.nan
    with pytest.raises(RuntimeError,match='Nonfinite'):t.validate_resume(altered,ident,data,'hier_reg',0)


def test_fit_identity_verifies_locked_prepared_references_before_using_data(tmp_path,monkeypatch):
    from wirelessjev.hierarchical_radio import study
    c=fixture_common(tmp_path);calls=[]
    prepared=c.OUT/'data/prepared.json';prepared.parent.mkdir(parents=True);prepared.write_text('{"unit":"prepared"}')
    c.check_sources=lambda:calls.append('published_source')
    c.source_receipt_sha256=lambda:'a'*64;c.protocol_sha256=lambda:'b'*64
    monkeypatch.setattr(t,'_common',lambda:c)
    monkeypatch.setattr(study,'verify_prepared',lambda:calls.append('locked_prepared_refs'),raising=False)
    identity=t.identity()
    assert calls==['published_source','locked_prepared_refs']
    assert identity['data_receipt_sha256']==c.digest(prepared)
    def changed():raise RuntimeError('Unit fixture: prepared group reference changed')
    monkeypatch.setattr(study,'verify_prepared',changed)
    with pytest.raises(RuntimeError,match='prepared group reference changed'):t.identity()


def test_completed_member_verification_reuses_explicit_validation_examples_without_changing_checks(tmp_path,monkeypatch):
    # Disposable analytic checkpoint/trace schema, never a registered fit.
    from wirelessjev.hierarchical_radio import study
    c=fixture_common(tmp_path);monkeypatch.setattr(t,'_common',lambda:c)
    es=examples();bindings={'validation_sparse':hashlib.sha256(t._canonical([e['binding'] for e in es])).hexdigest()}
    ident={'source_receipt_sha256':'a'*64,'protocol_sha256':'b'*64,'data_receipt_sha256':'c'*64}
    trace,history=trace_and_history();model=SelectionProbe()
    for epoch in (10,20,30,40):
        score,rows,ref,weights=t.validation_at_epoch(model,es,t.batch(es),ident,bindings,'hier_reg',0,epoch)
        history[epoch-1].update(validation_nll=score,validation_per_layout_nll=rows,
            validation_predictions=ref,validation_state_sha256=weights)
    selected,score=t.selected_validation(history)
    checkpoint={'identity':ident,'dataset_binding_sha256':bindings,'cell':'hier_reg','seed':0,
        'settings':{'cell':'hier_reg','width':32,'inference_steps':3},'state':model.state_dict(),
        'selected_epoch':selected,'validation_nll':score,'committed_updates':480}
    path=c.OUT/'checkpoints/hier_reg-seed0.pt';t._publish_torch(path,checkpoint,staging=c.WORK)
    receipt={'complete':True,'completed':True,'identity':ident,'dataset_binding_sha256':bindings,
        'cell':'hier_reg','seed':0,'run_id':'hier_reg-seed0','epochs':40,'updates':480,'committed_updates':480,
        'settings':checkpoint['settings'],'optimizer_spec':t.OPTIMIZER_SPEC,'checkpoint_sha256':c.digest(path),
        'history':history,'update_trace':trace,'selected_epoch':selected,'validation_nll':score}
    receipt=json.loads(json.dumps(receipt));loaded=[]
    def load_examples(stage,regime):
        assert stage=='validation' and regime=='sparse';loaded.append((stage,regime));return es
    monkeypatch.setattr(study,'examples',load_examples)
    calls=model.calls;model.forbid=True
    automatic=t.verify_completed(receipt,ident,'hier_reg',0)
    supplied=t.verify_completed(receipt,ident,'hier_reg',0,validation_examples=es)
    assert loaded==[('validation','sparse')] and model.calls==calls
    assert automatic['selected_epoch']==supplied['selected_epoch']==10
    assert torch.equal(automatic['state']['bias'],supplied['state']['bias'])
    # Explicitly supplied data still receives full archive binding checks.
    altered=deepcopy(es);altered[0]['binding']['case_id']='changed-unit-data';altered[0]['binding']['context_case_id']='changed-unit-data'
    with pytest.raises(RuntimeError,match='inputs/targets differ'):
        t.verify_completed(receipt,ident,'hier_reg',0,validation_examples=altered)
