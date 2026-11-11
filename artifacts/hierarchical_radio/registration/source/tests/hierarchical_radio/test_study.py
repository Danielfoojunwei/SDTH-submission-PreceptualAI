"""Passive receipt/selection software fixtures; no RT, fits or forwards."""
from copy import deepcopy

import numpy as np
import pytest

from wirelessjev.hierarchical_radio import common as c,data,simulation as sim,study as s,evaluation as ev,training as t
from wirelessjev.hierarchical_radio.model import CELLS


@pytest.fixture
def sandbox(tmp_path,monkeypatch):
    root=tmp_path/'software-repository';out=root/'artifacts/hierarchical_radio';out.mkdir(parents=True)
    monkeypatch.setattr(c,'ROOT',root);monkeypatch.setattr(c,'OUT',out)
    monkeypatch.setattr(c,'source_receipt_sha256',lambda:'a'*64)
    monkeypatch.setattr(c,'protocol_sha256',lambda:'b'*64)
    monkeypatch.setattr(c,'registry_sha256',lambda:'c'*64)
    monkeypatch.setattr(c,'check_sources',lambda **kwargs:{'software_fixture':True})
    for key in ('_GROUPS','_MODELS','_NUMERICAL'):monkeypatch.setattr(s,key,{})
    c.write(out/'data/prepared.json',{'software_fixture':'prepared TRAIN/VAL'})
    c.write(out/'data/calibration.json',{'software_fixture':'TRAIN-only calibration'})
    return out


def replace_json(path,value):
    """Deliberate corruption/rebinding of disposable SOFTWARE fixtures only."""
    path.write_bytes(c.encoded(value))


def target():
    return {'center_dbm':np.array([-60.,0.]),'lower_dbm':np.array([-60.5,0.]),
        'upper_dbm':np.array([-59.5,0.]),'heard':np.array([True,False]),'censored':np.array([False,True]),
        'censor_upper_dbm':np.array([0.,-80.]),'detection_threshold_dbm':np.full(2,-90.),'weights':np.ones(2)}


def numeric_fixture(monkeypatch,*,tie=False):
    prepared_fixture()
    group=c.read(c.OUT/'data/groups/validation-sparse.json')
    validation=s._unpack_examples(c.arrays(c.safe_path(group['arrays']['path'])),group['bindings'])
    monkeypatch.setattr(s,'examples',lambda stage,regime:validation)
    rows=[]
    for length,offset in ((5.,0.),(10.,0. if tie else 1.),(20.,0. if tie else 2.)):
        name=f'radiomap-{length:g}m'
        prior=c.npz(c.OUT/'numerical'/f'{name}.npz',{'software_fixture_prior':np.array([length])})
        mean=np.tile([-60.+offset,-80.],(18,1));std=np.ones_like(mean)
        prediction=c.npz(c.OUT/'numerical'/f'{name}-validation.npz',{'mean':mean,'std':std})
        score=ev.validation_nll(mean,std,{k:np.stack([e['target'][k] for e in validation]) for k in target()})
        row={'name':name,'lengthscale_m':length,'prior':prior,'validation_predictions':prediction,
            'validation_nll':score,'elapsed_seconds':.001,'source_receipt_sha256':c.source_receipt_sha256(),
            'prepared_sha256':c.digest(c.OUT/'data/prepared.json')}
        c.write(c.OUT/'numerical'/f'{name}.json',row);rows.append(row)
    fixed_prior=c.npz(c.OUT/'numerical/hierarchy-fixed.npz',{'software_fixture_prior':np.ones(1)})
    c.write(c.OUT/'numerical/hierarchy-fixed.json',{'prior':fixed_prior,
        'source_receipt_sha256':c.source_receipt_sha256(),'prepared_sha256':c.digest(c.OUT/'data/prepared.json')})
    selected={'schema':1,'selected_name':rows[0]['name'],'selected_prior':rows[0]['prior'],
        'candidates':[c.reference(c.OUT/'numerical'/f'{r["name"]}.json') for r in rows],
        'fixed_hierarchy':c.reference(c.OUT/'numerical/hierarchy-fixed.json'),
        'source_receipt_sha256':c.source_receipt_sha256(),'prepared_sha256':c.digest(c.OUT/'data/prepared.json'),
        'rule':'first_minimum_sparse_VAL_composite_interval_NLL'}
    c.write(c.OUT/'numerical/selection.json',selected)
    return selected,validation


def rebind_candidate(selection,index,row):
    path=c.OUT/'numerical'/f'radiomap-{(5,10,20)[index]}m.json';replace_json(path,row)
    selection['candidates'][index]=c.reference(path)
    replace_json(c.OUT/'numerical/selection.json',selection)


def test_numeric_selection_replays_all_raw_val_arrays_and_first_tie(sandbox,monkeypatch):
    selection,_=numeric_fixture(monkeypatch,tie=True)
    assert s.verify_numerical()==selection and selection['selected_name']=='radiomap-5m'
    selection['selected_name']='radiomap-10m';selection['selected_prior']=c.read(c.OUT/'numerical/radiomap-10m.json')['prior']
    replace_json(c.OUT/'numerical/selection.json',selection)
    with pytest.raises(RuntimeError,match='selection'):s.verify_numerical()


@pytest.mark.parametrize('change',[
    lambda d:d['candidates'].pop(),lambda d:d['candidates'].append(deepcopy(d['candidates'][0])),
    lambda d:d['candidates'].reverse(),lambda d:d.update(source_receipt_sha256='0'*64),
    lambda d:d.update(prepared_sha256='0'*64),lambda d:d.update(selected_prior=d['fixed_hierarchy']),
])
def test_numeric_exact_candidate_source_and_selected_prior_membership(sandbox,monkeypatch,change):
    selected,_=numeric_fixture(monkeypatch);change(selected);replace_json(c.OUT/'numerical/selection.json',selected)
    with pytest.raises(RuntimeError):s.verify_numerical()


@pytest.mark.parametrize('change',[
    lambda r:r.update(name='radiomap-20m'),lambda r:r.update(lengthscale_m=20.),
    lambda r:r.update(source_receipt_sha256='0'*64),lambda r:r.update(prepared_sha256='0'*64),
    lambda r:r.update(validation_nll=r['validation_nll']+1),
    lambda r:r.update(prior=c.read(c.OUT/'numerical/radiomap-10m.json')['prior']),
])
def test_rehashed_numeric_candidate_cannot_change_identity_or_raw_score(sandbox,monkeypatch,change):
    selected,_=numeric_fixture(monkeypatch);row=c.read(c.OUT/'numerical/radiomap-5m.json');change(row)
    rebind_candidate(selected,0,row)
    with pytest.raises(RuntimeError):s.verify_numerical()


def test_rehashed_truncated_raw_val_predictions_cannot_be_selected(sandbox,monkeypatch):
    selected,validation=numeric_fixture(monkeypatch);row=c.read(c.OUT/'numerical/radiomap-5m.json')
    path=c.safe_path(row['validation_predictions']['path']);a=c.arrays(path)
    np.savez_compressed(path,mean=a['mean'][:9],std=a['std'][:9]);row['validation_predictions']=c.reference(path)
    row['validation_nll']=ev.validation_nll(a['mean'][:9],a['std'][:9],
        {k:np.stack([e['target'][k] for e in validation[:9]]) for k in target()})
    rebind_candidate(selected,0,row)
    with pytest.raises(ValueError):s.verify_numerical()


def model_selection_fixture(monkeypatch,*,tie=False):
    calls=[]
    prepared_fixture()
    group=c.read(c.OUT/'data/groups/validation-sparse.json')
    validation=s._unpack_examples(c.arrays(c.safe_path(group['arrays']['path'])),group['bindings'])
    def val_only(stage,regime):
        assert (stage,regime)==('validation','sparse')
        return validation
    monkeypatch.setattr(s,'examples',val_only)
    monkeypatch.setattr(t,'identity',lambda:{'software_fixture_identity':True})
    def verify(fit,identity,cell,seed,*,validation_examples=None):
        assert fit['software_fixture'] is True and identity=={'software_fixture_identity':True}
        assert validation_examples is validation and len(validation_examples)==18
        calls.append((cell,seed))
    monkeypatch.setattr(t,'verify_completed',verify)
    for cell in CELLS:
        for seed in c.SEEDS:
            # A spectacular best joint seed must not replace the mean of all3.
            score=1. if cell=='hier_reg' else ([0.,3.,3.][seed] if cell=='hier_joint' and not tie else 1.)
            c.write(c.OUT/'fits'/f'{cell}-seed{seed}.json',{'software_fixture':True,'validation_nll':score})
            c.write_bytes(c.OUT/'checkpoints'/f'{cell}-seed{seed}.pt',b'explicit inert software checkpoint fixture')
    c.write(c.OUT/'numerical/selection.json',{'software_fixture':True})
    monkeypatch.setattr(s,'verify_numerical',lambda:{'software_fixture':True})
    return calls


def test_model_selection_uses_all_fifteen_verified_members_mean_not_best_seed(sandbox,monkeypatch):
    calls=model_selection_fixture(monkeypatch)
    value=s.select_models()
    assert value['selected_hierarchy']=='hier_reg'
    assert set(calls)=={(cell,seed) for cell in CELLS for seed in c.SEEDS} and len(calls)==15
    assert len(value['fits'])==len(value['checkpoints'])==15
    assert value['selection_used_TEST_native_or_original_SERVICE'] is False
    assert s.verify_selection()==value
    value['selected_hierarchy']='hier_joint';replace_json(c.OUT/'selection.json',value)
    with pytest.raises(RuntimeError,match='Locked selection'):s.verify_selection()


def test_model_selection_keeps_first_branch_on_exact_mean_tie(sandbox,monkeypatch):
    model_selection_fixture(monkeypatch,tie=True)
    assert s.select_models()['selected_hierarchy']=='hier_reg'


def software_example(stage,index,state,regime,*,held=False,stale=False):
    spec=sim.case_spec(stage,index,state);context=sim.case_spec(stage,index,0) if stale else spec
    result={'ap_xyz':np.zeros((4 if stage!='ap_transfer' else 8,3),dtype='f4'),
        'a':{'context':np.zeros((1,20),dtype='f4')},'b':{'context':np.zeros((1,20),dtype='f4')},
        'query':{'q':np.zeros((2,16),dtype='f4')},'target':target()}
    binding={'stage':stage,'index':index,'state':state,'condition':state,'layout_id':spec.layout_id,
        'case_id':spec.case_id,'context_case_id':context.case_id,'regime':regime,'held_band':held,
        'stale_requested':context.case_id!=spec.case_id}
    binding['model_input_sha256']={role:{k:data.array_digest(v) for k,v in result[role].items()} for role in ('a','b','query')}
    binding['model_input_sha256']['ap_xyz']=data.array_digest(result['ap_xyz'])
    binding['target_sha256']={k:data.array_digest(v) for k,v in result['target'].items()}
    result['binding']=binding;return result


def group_fixture(stage='development',regime='sparse',*,held=False,stale=False):
    name=s._group_name(stage,regime,held,stale)
    if not s.stage_path(stage).exists():c.write(s.stage_path(stage),{'software_fixture_stage':stage})
    values=[software_example(stage,i,state,regime,held=held,stale=stale)
            for i in range(sim.COUNTS[stage]) for state in sim.STATES]
    arrays=c.npz(c.OUT/'data/groups'/f'{name}.npz',s._pack_examples(values))
    group={'schema':1,'stage':stage,'regime':regime,'held_band':held,'stale':stale,
        'source_receipt_sha256':c.source_receipt_sha256(),'registry_sha256':c.registry_sha256(),
        'calibration_sha256':c.digest(c.OUT/'data/calibration.json'),'stage_receipt_sha256':c.digest(s.stage_path(stage)),
        'bindings':[v['binding'] for v in values],'arrays':arrays,'reported_only':True}
    path=c.OUT/'data/groups'/f'{name}.json';c.write(path,group)
    return name,values,group


def prediction_fixture(name,values,index=0):
    e=values[index];binding=e['binding'];methods=s.methods_for(secondary='held6GHz' in name or 'stale' in name)
    identity={'stage':binding['stage'],'group':name,'case_id':binding['case_id'],'layout_id':binding['layout_id'],
        'state':binding['state'],'methods':methods,'example_index':index,
        'example_group':c.reference(c.OUT/'data/groups'/f'{name}.json'),
        'source_receipt_sha256':c.source_receipt_sha256(),'selection_sha256':c.digest(c.OUT/'selection.json')}
    folder=c.OUT/'predictions'/name;case=binding['case_id'];c.attempt(folder/'attempts'/f'{case}.json',identity)
    mean=np.tile([-60.,-80.],(len(methods),1));std=np.ones_like(mean)
    output=c.npz(folder/f'{case}.npz',{'mean':mean,'std':std})
    receipt=identity|{'predictions':output,'metrics':[ev.metrics(m,sd,e['target']) for m,sd in zip(mean,std)],
        'elapsed_seconds':[.001]*len(methods),'attempt':c.reference(folder/'attempts'/f'{case}.json')}
    path=folder/f'{case}.json';c.write(path,receipt)
    return receipt,e


def initialize_selection(monkeypatch):
    c.write(c.OUT/'selection.json',{'selected_hierarchy':'hier_reg','software_fixture':True})
    monkeypatch.setattr(s,'verify_selection',lambda:c.read(c.OUT/'selection.json'))


def test_cached_prediction_passively_replays_exact_methods_arrays_and_target_metrics(sandbox,monkeypatch):
    initialize_selection(monkeypatch);name,values,_=group_fixture();receipt,e=prediction_fixture(name,values)
    assert s.verify_prediction(receipt,e)[0]==receipt
    assert s.verify_prediction(receipt)[0]==receipt


@pytest.mark.parametrize('change',[
    lambda r:r.update(stage='test'),lambda r:r.update(group='development-full'),
    lambda r:r.update(example_index=1),lambda r:r.update(example_index=True),
    lambda r:r['methods'].__setitem__(0,r['methods'][1]),lambda r:r.update(source_receipt_sha256='0'*64),
    lambda r:r['metrics'][0].update(rmse_db=99.),lambda r:r['elapsed_seconds'].__setitem__(0,-1.),
])
def test_rehashed_prediction_cannot_change_group_method_case_or_metrics(sandbox,monkeypatch,change):
    initialize_selection(monkeypatch);name,values,_=group_fixture();receipt,e=prediction_fixture(name,values)
    change(receipt)
    with pytest.raises(RuntimeError):s.verify_prediction(receipt,e)


def test_prediction_cannot_attach_another_valid_group(sandbox,monkeypatch):
    initialize_selection(monkeypatch);name,values,_=group_fixture();receipt,e=prediction_fixture(name,values)
    other,_,_=group_fixture(regime='full');receipt['example_group']=c.reference(c.OUT/'data/groups'/f'{other}.json')
    with pytest.raises(RuntimeError):s.verify_prediction(receipt,e)


@pytest.mark.parametrize('key',['predictions','attempt'])
def test_alternate_rehashed_prediction_or_attempt_path_is_not_canonical(sandbox,monkeypatch,key):
    initialize_selection(monkeypatch);name,values,_=group_fixture();receipt,e=prediction_fixture(name,values)
    old=c.safe_path(receipt[key]['path']);new=old.with_name('alternate'+old.suffix);new.write_bytes(old.read_bytes())
    receipt[key]=c.reference(new)
    with pytest.raises(RuntimeError):s.verify_prediction(receipt,e)


def test_rehashed_attempt_identity_is_checked_not_only_its_digest(sandbox,monkeypatch):
    initialize_selection(monkeypatch);name,values,_=group_fixture();receipt,e=prediction_fixture(name,values)
    path=c.safe_path(receipt['attempt']['path']);attempt=c.read(path);attempt['group']='test-sparse';replace_json(path,attempt)
    receipt['attempt']=c.reference(path)
    with pytest.raises(RuntimeError,match='attempt identity'):s.verify_prediction(receipt,e)


@pytest.mark.parametrize('change',[
    lambda g:g['bindings'].pop(),lambda g:g['bindings'].append(deepcopy(g['bindings'][0])),
    lambda g:g['bindings'].reverse(),lambda g:g['bindings'][0].update(context_case_id='unregistered'),
    lambda g:g.update(registry_sha256='0'*64),lambda g:g['bindings'][0].update(held_band=True),
])
def test_group_requires_exact_complete_ordered_geometry_state_bindings(sandbox,change):
    _,_,group=group_fixture();change(group)
    with pytest.raises(RuntimeError):s._verify_group(group)


def test_stale_state_zero_anchor_is_not_falsely_called_an_actual_stale_measurement(sandbox):
    name,_,group=group_fixture(stage='test',stale=True)
    assert group['bindings'][0]['stale_requested'] is False
    assert group['bindings'][1]['stale_requested'] is True
    assert s._verify_group(group,name)==group


def prepared_fixture():
    groups={}
    for stage in ('training','validation'):
        for regime in ('full','sparse'):
            name,_,_=group_fixture(stage=stage,regime=regime)
            groups[name]=c.reference(c.OUT/'data/groups'/f'{name}.json')
    receipt={'schema':1,'source_receipt_sha256':c.source_receipt_sha256(),
        'protocol_sha256':c.protocol_sha256(),'registry_sha256':c.registry_sha256(),
        'training_cases':72,'validation_cases':18,'target_band_indices':[0,1],
        'calibration':c.reference(c.OUT/'data/calibration.json'),'groups':groups,
        'stages':{stage:c.reference(s.stage_path(stage)) for stage in ('training','validation')}}
    replace_json(c.OUT/'data/prepared.json',receipt);return receipt


def test_prepared_authorization_binds_four_complete_groups_two_stages_and_calibration(sandbox):
    receipt=prepared_fixture()
    assert s.verify_prepared()==receipt


@pytest.mark.parametrize('change',[
    lambda r:r['groups'].pop('training-full'),lambda r:r.update(target_band_indices=[0,1,2]),
    lambda r:r.update(source_receipt_sha256='0'*64),lambda r:r['stages'].pop('validation'),
])
def test_prepared_identity_cannot_omit_group_stage_or_add_held_targets(sandbox,change):
    receipt=prepared_fixture();change(receipt);replace_json(c.OUT/'data/prepared.json',receipt)
    with pytest.raises(RuntimeError):s.verify_prepared()


def development_fixture(monkeypatch):
    initialize_selection(monkeypatch);groups={}
    for regime in ('full','sparse'):
        name,values,_=group_fixture(regime=regime);refs={}
        for index,e in enumerate(values):
            receipt,_=prediction_fixture(name,values,index)
            refs[e['binding']['case_id']]=c.reference(c.OUT/'predictions'/name/f"{receipt['case_id']}.json")
        manifest={'schema':1,'group':name,'stage':'development','case_receipts':refs,'methods':s.methods_for(),
            'source_receipt_sha256':c.source_receipt_sha256(),'selection_sha256':c.digest(c.OUT/'selection.json'),
            'example_group':c.reference(c.OUT/'data/groups'/f'{name}.json')}
        path=c.OUT/'predictions'/f'{name}.json';c.write(path,manifest);groups[regime]=c.reference(path)
    gate={'schema':1,'technical_pass':True,'efficacy_screening':False,'expected_layouts':4,'expected_states':3,
        'groups':groups,'source_receipt_sha256':c.source_receipt_sha256(),'protocol_sha256':c.protocol_sha256(),
        'registry_sha256':c.registry_sha256(),'selection_sha256':c.digest(c.OUT/'selection.json'),
        'prepared_sha256':c.digest(c.OUT/'data/prepared.json')}
    c.write(c.OUT/'development_gate.json',gate);return gate


def test_technical_dev_gate_verifies_both_regimes_all_twelve_cases_and_seventeen_methods(sandbox,monkeypatch):
    gate=development_fixture(monkeypatch)
    assert s.verify_development()==gate


@pytest.mark.parametrize('change',[
    lambda g:g.update(technical_pass=1),lambda g:g.update(technical_pass=False),
    lambda g:g.update(efficacy_screening=0),lambda g:g['groups'].pop('full'),
    lambda g:g.update(selection_sha256='0'*64),lambda g:g.update(registry_sha256='0'*64),
])
def test_dev_permission_rejects_numeric_booleans_partial_regimes_and_stale_dependencies(sandbox,monkeypatch,change):
    gate=development_fixture(monkeypatch);change(gate);replace_json(c.OUT/'development_gate.json',gate)
    with pytest.raises(RuntimeError):s.verify_development()


def test_rehashed_incomplete_dev_manifest_cannot_authorize_test(sandbox,monkeypatch):
    gate=development_fixture(monkeypatch);path=c.OUT/'predictions/development-sparse.json';manifest=c.read(path)
    manifest['case_receipts'].pop(next(iter(manifest['case_receipts'])));replace_json(path,manifest)
    gate['groups']['sparse']=c.reference(path);replace_json(c.OUT/'development_gate.json',gate)
    with pytest.raises(RuntimeError,match='schedule'):s.verify_development()


def test_dev_gate_calls_real_selection_validation_before_using_its_true_flag(sandbox,monkeypatch):
    development_fixture(monkeypatch)
    def reject():raise RuntimeError('software fixture: raw selected VAL binding invalid')
    monkeypatch.setattr(s,'verify_selection',reject)
    with pytest.raises(RuntimeError,match='raw selected VAL'):s.verify_development()
