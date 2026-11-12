"""Temporary Git/source/selection fixtures; no RT, model fits or scientific data."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import ModuleType,SimpleNamespace

import pytest

from wirelessjev.nested_radio import common as c
from wirelessjev.nested_radio import data as sim


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


@pytest.fixture
def sandbox(tmp_path,monkeypatch):
    root=tmp_path/'software-source';root.mkdir();git(root,'init','-q','-b','fixture-source')
    out=root/'artifacts/nested_radio'
    for key,value in (('ROOT',root),('OUT',out),('WORK',tmp_path/'software-work'),
                      ('PROTOCOL',out/'protocol.json'),('REGISTRY',out/'registry.json'),
                      ('PARENT',root/'artifacts/hierarchical_radio')):
        monkeypatch.setattr(c,key,value)
    monkeypatch.setattr(c,'_PUBLICATION',set())
    monkeypatch.setattr(c,'protocol_spec',lambda:{'schema':1,'software_fixture':True})
    monkeypatch.setattr(sim,'geometry_registry',lambda:{'schema':1,'software_fixture_registry':True})
    # A unit-fixture subprocess result exercises registration mechanics; it is
    # never stored as the real experiment's preflight or scientific evidence.
    real_run=subprocess.run
    def software_preflight(command,*args,**kwargs):
        if list(command[:3])==[sys.executable,'-m','pytest']:
            return SimpleNamespace(returncode=0,stdout='1 software fixture passed\n',stderr='')
        return real_run(command,*args,**kwargs)
    monkeypatch.setattr(c.subprocess,'run',software_preflight)
    # Only common.authorize's mandatory selection-helper invocation is unit-tested
    # here. The scientific helper has its own real receipt/replay tests.
    stage_helper=ModuleType('wirelessjev.nested_radio.study')
    stage_helper.calls=[]
    def verify_selection():
        stage_helper.calls.append(True)
        selection=c.read(c.OUT/'selection.json')
        if selection.get('source_receipt_sha256')!=c.source_receipt_sha256():
            raise RuntimeError('software fixture: selection source changed')
    stage_helper.verify_selection=verify_selection
    monkeypatch.setitem(sys.modules,'wirelessjev.nested_radio.study',stage_helper)
    files={
        'src/tracked_fixture.py':'# tracked software dependency\n',
        'tests/test_tracked_fixture.py':'# tracked software test\n',
        'src/wirelessjev/nested_radio/fixture.py':'# own source membership\n',
        'tests/nested_radio/test_fixture.py':'# own test membership\n',
        'scripts/run_nested_radio.py':'# software fixture entry point\n',
        'docs/nested_radio/PLAN.md':'Temporary software fixture plan\n',
        'pyproject.toml':'[project]\nname="software-fixture"\nversion="0.0.0"\n',
        'uv.lock':'# temporary software fixture lock\n',
        'src/wirelessjev/observation_frontier/deferred.py':'# unrelated untracked draft\n',
    }
    for name,value in files.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
    git(root,'add','src/tracked_fixture.py','tests/test_tracked_fixture.py','pyproject.toml','uv.lock')
    # Nine tiny independent old closures, all under this temporary repository.
    for study in c.PRIOR:
        paths={}
        for kind in ('source','evidence','documents'):
            path=root/'artifacts/software_prior'/study/(kind+'.txt')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(study+' '+kind+' fixture\n')
            paths[kind+'_sha256']={str(path.relative_to(root)):c.digest(path)}
        c.write(root/'artifacts'/study/'verification/manifest.json',paths)
    c.write_spec()
    return root


def register(sandbox):
    receipt=c.register()
    assert receipt['before_fresh_RT_training_or_predictions'] is True
    return receipt


def commit_fixture(root):
    git(root,'add','-A')
    git(root,'-c','user.name=Software Fixture','-c','user.email=fixture@example.invalid',
        'commit','-q','-m','Temporary source publication fixture')
    return git(root,'rev-parse','HEAD')


def local_remote(root,tmp_path):
    remote=tmp_path/'software-remote.git'
    git(tmp_path,'init','--bare','-q',str(remote))
    git(root,'remote','add','origin',str(remote))
    return remote


def publish_fixture(root,tmp_path):
    commit=commit_fixture(root);local_remote(root,tmp_path)
    git(root,'push','-q','origin','fixture-source')
    return c.record_publication(commit,'fixture-source')


def test_collector_binds_new_namespace_without_collecting_unrelated_drafts(sandbox):
    names=c.source_hashes()
    assert {'src/tracked_fixture.py','tests/test_tracked_fixture.py',
        'src/wirelessjev/nested_radio/fixture.py','tests/nested_radio/test_fixture.py',
        'scripts/run_nested_radio.py','docs/nested_radio/PLAN.md',
        'artifacts/nested_radio/protocol.json','artifacts/nested_radio/registry.json',
        'pyproject.toml','uv.lock'} <= set(names)
    assert not any('observation_frontier' in name for name in names)


def test_registration_binds_all_nine_prior_closures_and_current_capsule(sandbox):
    receipt=register(sandbox)
    assert set(receipt['prior']['manifests'])==set(c.PRIOR) and len(c.PRIOR)==9
    assert receipt['prior']['all_named_bytes_checked'] is True
    assert receipt['prior']['unique_repository_files']==27
    assert c.check_sources(published=False)==receipt
    assert c.register()==receipt


def test_only_registration_requests_full_prior_byte_scan_case_checks_keep_manifest_binding(sandbox,monkeypatch):
    original=c.prior_identities;requests=[]
    def checked(*,verify_bytes=False):
        requests.append(verify_bytes);return original(verify_bytes=verify_bytes)
    monkeypatch.setattr(c,'prior_identities',checked)
    register(sandbox)
    c.check_sources(published=False);c.check_sources(published=False)
    assert requests==[True,False,False]


@pytest.mark.parametrize('namespace',['rt_data','fits','data','native','predictions','numerical',
    'attempts','checkpoints','epoch_checkpoints','validation_predictions','diagnostic_attempts',
    'representation_predictions','selection.json','validation_representation.json','report.json'])
def test_registration_precedes_any_scientific_namespace(sandbox,namespace):
    (c.OUT/namespace).mkdir()
    with pytest.raises(RuntimeError,match='before scientific'):c.register()
    assert not (c.OUT/'preflight').exists()


@pytest.mark.parametrize('path',['src/tracked_fixture.py','scripts/run_nested_radio.py',
    'artifacts/nested_radio/protocol.json','artifacts/nested_radio/registry.json'])
def test_current_source_bytes_are_checked_each_authorization(sandbox,path):
    register(sandbox);target=sandbox/path;target.write_bytes(target.read_bytes()+b'changed')
    with pytest.raises(RuntimeError,match='Frozen source'):c.check_sources(published=False)


@pytest.mark.parametrize('change',['new_source','new_test','deleted_source','deleted_test'])
def test_exact_own_namespace_membership(sandbox,change):
    register(sandbox)
    if change.startswith('new'):
        top='src/wirelessjev/nested_radio' if change.endswith('source') else 'tests/nested_radio'
        (sandbox/top/'extra.py').write_text('# new importable file\n')
    else:
        name='src/wirelessjev/nested_radio/fixture.py' if change.endswith('source') else 'tests/nested_radio/test_fixture.py'
        (sandbox/name).unlink()
    with pytest.raises(RuntimeError,match='membership'):c.check_sources(published=False)


@pytest.mark.parametrize('change',['bytes','extra','deleted'])
def test_exact_capsule_membership_and_bytes(sandbox,change):
    register(sandbox);capsule=c.OUT/'registration/source'
    path=capsule/'src/tracked_fixture.py'
    if change=='bytes':path.write_text('# altered capsule\n')
    elif change=='extra':(capsule/'unregistered.py').write_text('# extra capsule\n')
    else:path.unlink()
    with pytest.raises(RuntimeError,match='capsule'):c.check_sources(published=False)


@pytest.mark.parametrize('key',['preflight','preflight_log'])
def test_preflight_artifact_bytes_remain_bound(sandbox,key):
    receipt=register(sandbox);(sandbox/receipt[key]['path']).write_text('changed software proof\n')
    with pytest.raises(RuntimeError,match='dependency'):c.check_sources(published=False)


def test_unrelated_later_tracked_phase_is_not_retroactive_dependency(sandbox):
    receipt=register(sandbox);path=sandbox/'src/later_phase.py';path.write_text('# separate future phase\n')
    git(sandbox,'add','src/later_phase.py')
    assert 'src/later_phase.py' in c.source_hashes()
    assert 'src/later_phase.py' not in receipt['source_sha256']
    assert c.check_sources(published=False)==receipt


def test_prior_manifest_change_invalidates_every_current_case(sandbox):
    register(sandbox)
    path=sandbox/'artifacts'/c.PRIOR[0]/'verification/manifest.json'
    value=c.read(path);value['software_fixture_changed']=True;path.write_bytes(c.encoded(value))
    with pytest.raises(RuntimeError,match='Previous manifest'):c.check_sources(published=False)


def test_full_prior_scan_catches_changed_source_evidence_and_document_bytes(sandbox):
    register(sandbox)
    path=sandbox/'artifacts/software_prior'/c.PRIOR[0]/'evidence.txt';path.write_text('changed old evidence\n')
    with pytest.raises(RuntimeError,match='Prior frozen file'):c.prior_identities(verify_bytes=True)


@pytest.mark.parametrize('name',['artifacts/radiojepa/rt_data/confirmation','artifacts/radiojepa/rt_data/confirmation.json'])
def test_original_service_namespace_tripwire_stays_closed(sandbox,name):
    path=sandbox/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.mkdir() if path.suffix=='' else path.write_text('{}')
    with pytest.raises(RuntimeError,match='SERVICE'):c.prior_identities()


def test_real_local_git_remote_publication_and_live_proof_checks(sandbox,tmp_path):
    register(sandbox);commit=commit_fixture(sandbox);local_remote(sandbox,tmp_path)
    with pytest.raises(RuntimeError,match='not observed'):c.record_publication(commit,'fixture-source')
    git(sandbox,'push','-q','origin','fixture-source')
    proof=c.record_publication(commit,'fixture-source')
    assert proof['commit']==proof['remote_observed_commit']==commit
    assert c.check_sources()
    proof['remote_observed_commit']='0'*40
    (c.OUT/'registration/git.json').write_bytes(c.encoded(proof))
    with pytest.raises(RuntimeError,match='Publication identity'):c.check_sources()


def test_remote_ref_alone_cannot_publish_changed_uncommitted_source(sandbox,tmp_path):
    commit=commit_fixture(sandbox)
    (sandbox/'src/tracked_fixture.py').write_text('# bytes absent from published commit\n')
    register(sandbox);local_remote(sandbox,tmp_path);git(sandbox,'push','-q','origin','fixture-source')
    with pytest.raises((RuntimeError,subprocess.CalledProcessError)):
        c.record_publication(commit,'fixture-source')


def test_cached_publication_key_binds_receipt_sha_not_only_same_commit(sandbox,tmp_path):
    receipt=register(sandbox);proof=publish_fixture(sandbox,tmp_path);assert c.check_sources()
    path=sandbox/'src/tracked_fixture.py';path.write_text('# changed post-publication\n')
    receipt['source_sha256']['src/tracked_fixture.py']=c.digest(path)
    (c.OUT/'registration/source/src/tracked_fixture.py').write_bytes(path.read_bytes())
    (c.OUT/'registration/source.json').write_bytes(c.encoded(receipt))
    proof['source_receipt_sha256']=c.source_receipt_sha256()
    (c.OUT/'registration/git.json').write_bytes(c.encoded(proof))
    with pytest.raises(RuntimeError,match='Published source'):c.check_sources()


def test_same_code_commit_cannot_rebless_rewritten_preflight_receipt_metadata(sandbox,tmp_path):
    receipt=register(sandbox);proof=publish_fixture(sandbox,tmp_path);assert c.check_sources()
    log=sandbox/receipt['preflight_log']['path'];log.write_text('different software preflight result\n')
    receipt['preflight_log']=c.reference(log)
    (c.OUT/'registration/source.json').write_bytes(c.encoded(receipt))
    proof['source_receipt_sha256']=c.source_receipt_sha256()
    (c.OUT/'registration/git.json').write_bytes(c.encoded(proof))
    with pytest.raises(RuntimeError,match='Published source receipt'):c.check_sources()


def test_authorization_requires_publication_before_case_or_attempt_write(sandbox):
    register(sandbox);spec=sim.case_spec('test',0,0)
    with pytest.raises(FileNotFoundError):c.authorize(spec)
    assert not (c.OUT/'rt_data').exists()


@pytest.mark.parametrize('stage',['test','ap_transfer'])
def test_every_fresh_case_requires_locked_selection_and_canonical_path(sandbox,tmp_path,stage):
    register(sandbox);publish_fixture(sandbox,tmp_path);spec=sim.case_spec(stage,0,0)
    with pytest.raises(FileNotFoundError):c.authorize(spec)
    c.write(c.OUT/'selection.json',{'source_receipt_sha256':c.source_receipt_sha256()})
    result=c.authorize(spec,c.case_path(spec))
    assert result['authorized'] is True and result['case_id']==spec.case_id
    assert result['case_path']==str(c.case_path(spec).resolve())
    assert not c.case_path(spec).exists()
    assert sys.modules['wirelessjev.nested_radio.study'].calls
    with pytest.raises(ValueError,match='Alternate'):c.authorize(spec,tmp_path/'other')
    with pytest.raises(ValueError,match='Unregistered'):
        c.authorize(replace(spec,tx_power_dbm=spec.tx_power_dbm+1))


def test_required_selection_replay_failure_is_never_bypassed_by_true_metadata(sandbox,tmp_path):
    register(sandbox);publish_fixture(sandbox,tmp_path)
    c.write(c.OUT/'selection.json',{'source_receipt_sha256':c.source_receipt_sha256(),'complete':True})
    def reject():raise RuntimeError('software fixture: invalid raw VAL/checkpoint proof')
    sys.modules['wirelessjev.nested_radio.study'].verify_selection=reject
    with pytest.raises(RuntimeError,match='raw VAL/checkpoint'):c.authorize(sim.case_spec('test',0,0))
    assert not (c.OUT/'rt_data').exists()


def test_write_once_attempt_and_work_only_atomic_resume_boundary(sandbox):
    path=c.OUT/'software-attempt.json';c.attempt(path,{'software_fixture':True})
    assert c.read(path)['created_unix_ns']>0
    with pytest.raises(FileExistsError):c.attempt(path,{'replacement':True})
    work=c.WORK/'resume.json';c.atomic_write(work,{'updates':1});c.atomic_write(work,{'updates':2})
    assert c.read(work)=={'updates':2}
    with pytest.raises(ValueError,match='WORK'):c.atomic_write(c.OUT/'evidence.json',{'replacement':True})


@pytest.mark.parametrize('name',['/tmp/outside.py','../outside.py','src/../outside.py','src//file.py'])
def test_noncanonical_repository_paths_are_rejected(sandbox,name):
    with pytest.raises(ValueError,match='Canonical'):c.safe_path(name)


def test_symlink_escape_cannot_enter_source_or_reference(sandbox,tmp_path):
    outside=tmp_path/'outside.py';outside.write_text('# outside software fixture\n')
    alias=sandbox/'src/wirelessjev/nested_radio/escape.py';alias.symlink_to(outside)
    with pytest.raises(ValueError,match='Canonical'):c.source_hashes()
    with pytest.raises(ValueError,match='Repository'):c.reference(outside)


# These pure selection fixtures use actual production selectors, without any
# checkpoint fitting, model construction or experimental prediction.
def score_fixture():
    from wirelessjev.nested_radio.model import CELLS
    return {cell:[3.,3.,3.] for cell in CELLS}


def test_selector_uses_all_members_and_excludes_legacy_control():
    from wirelessjev.nested_radio import study
    scores=score_fixture()
    scores.update(global8_reg=[1.,10.,10.],global8_joint=[3.,3.,3.],
                  hier_reg=[0.,0.,0.],nested_reg=[2.,4.,4.],nested_joint=[4.,4.,4.])
    assert study.choose_cell(scores)=='global8_joint'


def test_selector_ties_prefer_declared_global_baseline_then_eligible_order():
    from wirelessjev.nested_radio import study
    scores=score_fixture();scores['hier_reg']=[0.,0.,0.]
    assert study.choose_cell(scores)=='global8_reg'
    scores['global8_reg']=[4.,4.,4.]
    assert study.choose_cell(scores)=='global8_joint'
    scores['global8_joint']=[4.,4.,4.]
    assert study.choose_cell(scores)=='nested_reg'


@pytest.mark.parametrize('change',[
    lambda s:s.pop('hier_reg'), lambda s:s.update(extra=[0.,0.,0.]),
    lambda s:s.update(global8_reg=[1.,2.]),
    lambda s:s.update(global8_reg=[[1.,2.],[1.,2.],[1.,2.]]),
    lambda s:s.update(global8_reg=[True,True,True]),
    lambda s:s.update(global8_reg=[-1.,1.,1.]),
    lambda s:s.update(global8_reg=[float('nan'),1.,1.]),
    lambda s:s.update(global8_reg=[float('inf'),1.,1.]),
])
def test_selector_rejects_extra_missing_nonfinite_or_invalid_member_scores(change):
    from wirelessjev.nested_radio import study
    values=score_fixture();change(values)
    with pytest.raises(ValueError):study.choose_cell(values)


def test_numerical_selector_considers_fixed_hierarchy_and_first_ties():
    from wirelessjev.nested_radio import study
    scores={name:2. for name in study.NUMERICAL};scores['hierarchy-fixed']=1.
    assert study.choose_numeric(scores)=='hierarchy-fixed'
    scores={name:1. for name in study.NUMERICAL}
    assert study.choose_numeric(scores)=='radiomap-5m'


@pytest.mark.parametrize('change',[
    lambda s:s.pop('hierarchy-fixed'), lambda s:s.update(extra=0.),
    lambda s:s.update({'radiomap-5m':True}),lambda s:s.update({'radiomap-5m':-1.}),
    lambda s:s.update({'radiomap-5m':float('nan')}),lambda s:s.update({'radiomap-5m':float('inf')}),
])
def test_numerical_selector_rejects_incomplete_or_invalid_scores(change):
    from wirelessjev.nested_radio import study
    values={name:1. for name in study.NUMERICAL};change(values)
    with pytest.raises(ValueError):study.choose_numeric(values)


def validation_fixture():
    import numpy as np
    from wirelessjev.hierarchical_radio.data import array_digest
    values=[]
    for layout in range(6):
        for state in range(3):
            heard=np.array([True,False]);target={
                'heard':heard,'censored':~heard,'center_dbm':np.array([-70.,0.],dtype='f4'),
                'lower_dbm':np.array([-70.5,0.],dtype='f4'),'upper_dbm':np.array([-69.5,0.],dtype='f4'),
                'censor_upper_dbm':np.array([0.,-75.],dtype='f4'),'detection_threshold_dbm':np.full(2,-90.,dtype='f4')}
            case=f'unit-validation-{layout}-state{state}'
            e={'a':{'known':np.ones(2,bool)},'b':{'known':np.ones(2,bool)},
               'query':{'ap':np.zeros(2,dtype='i8')},'target':target,'ap_xyz':np.zeros((1,3),dtype='f4'),
               'binding':{'stage':'validation','index':layout,'state':state,'condition':state,
                   'layout_id':f'unit-validation-{layout}','case_id':case,'context_case_id':case,
                   'regime':'sparse','held_band':False,'stale_requested':False}}
            e['binding']['model_input_sha256']={role:{k:array_digest(v) for k,v in e[role].items()} for role in ('a','b','query')}
            e['binding']['model_input_sha256']['ap_xyz']=array_digest(e['ap_xyz'])
            e['binding']['target_sha256']={k:array_digest(v) for k,v in target.items()}
            values.append(e)
    return values


def numerical_fixture(sandbox,monkeypatch):
    """Opaque prior bytes plus analytic VAL arrays; never a fitted prior/model."""
    import numpy as np
    from wirelessjev.nested_radio import study,training
    refs={};predictions={}
    for name in study.NUMERICAL:
        refs[name]=c.npz(c.PARENT/'numerical'/f'{name}.npz',{'opaque_software_prior':np.array([1.])})
        mean=np.tile(np.array([-69.,-75.]),(18,1));std=np.full(mean.shape,2. if name=='hierarchy-fixed' else 3.)
        predictions[name]={'mean':mean,'std':std}
        if name!='hierarchy-fixed':
            refs[name+'-validation']=c.npz(c.PARENT/'numerical'/f'{name}-validation.npz',predictions[name])
    prior_path=c.PARENT/'verification/manifest.json';prior=c.read(prior_path)
    prior['evidence_sha256'].update({r['path']:r['sha256'] for r in refs.values()});prior_path.write_bytes(c.encoded(prior))
    register(sandbox)
    c.write(c.OUT/'data/prepared.json',{'scope':'temporary software validation arithmetic'})
    values=validation_fixture();rows=[];scores={}
    # The production fitting-data loader is covered by data tests; only raw
    # score/path replay is exercised here with explicit inert examples.
    monkeypatch.setattr(sim,'verify_prepared',lambda:{'scope':'software fixture'})
    for name in study.NUMERICAL:
        raw=predictions[name];rmse,nll,per_layout=training.reduce_validation_predictions(raw['mean'],raw['std'],values)
        identity={'name':name,'prior':refs[name],'source_receipt_sha256':c.source_receipt_sha256(),
                  'prepared_sha256':c.digest(c.OUT/'data/prepared.json')}
        attempt=c.OUT/'numerical/attempts'/f'{name}.json';c.attempt(attempt,identity)
        output=c.npz(c.OUT/'numerical'/f'{name}-validation.npz',raw)
        row=identity|{'validation_predictions':output,'parent_predictions':refs.get(name+'-validation'),
            'validation_rmse':rmse,'validation_nll':nll,'validation_per_layout':per_layout,
            'attempt':c.reference(attempt),'new_prior_fit':False}
        path=c.OUT/'numerical'/f'{name}.json';c.write(path,row);rows.append(c.reference(path));scores[name]=rmse
    receipt={'schema':1,'selected_name':study.choose_numeric(scores),'candidates':rows,'scores':scores,
        'source_receipt_sha256':c.source_receipt_sha256(),'prepared_sha256':c.digest(c.OUT/'data/prepared.json')}
    c.write(c.OUT/'numerical/selection.json',receipt)
    return receipt,values


def rebound_numeric(receipt,index,row):
    from wirelessjev.nested_radio import study
    path=c.OUT/'numerical'/f'{study.NUMERICAL[index]}.json';path.write_bytes(c.encoded(row))
    receipt['candidates'][index]=c.reference(path)
    (c.OUT/'numerical/selection.json').write_bytes(c.encoded(receipt))


def test_all_four_raw_numerical_scores_replay_and_rmse_tie_does_not_select_nll(sandbox,monkeypatch):
    from wirelessjev.nested_radio import study
    selection,values=numerical_fixture(sandbox,monkeypatch)
    assert study.verify_numerical(values)==selection
    assert selection['selected_name']=='radiomap-5m'
    assert c.read(c.OUT/'numerical/hierarchy-fixed.json')['validation_nll'] < c.read(c.OUT/'numerical/radiomap-5m.json')['validation_nll']


@pytest.mark.parametrize('change',[
    lambda r:r['candidates'].pop(),lambda r:r['candidates'].append(r['candidates'][0]),
    lambda r:r.update(selected_name='hierarchy-fixed'),lambda r:r['scores'].update({'radiomap-5m':0.}),
])
def test_numerical_selection_cannot_skip_add_or_relabel_a_candidate(sandbox,monkeypatch,change):
    from wirelessjev.nested_radio import study
    receipt,values=numerical_fixture(sandbox,monkeypatch);change(receipt)
    (c.OUT/'numerical/selection.json').write_bytes(c.encoded(receipt))
    with pytest.raises(RuntimeError):study.verify_numerical(values)


@pytest.mark.parametrize('field,value',[
    ('name','radiomap-10m'),('source_receipt_sha256','0'*64),('prepared_sha256','0'*64),
    ('new_prior_fit',True),('validation_rmse',0.),('validation_nll',0.),
])
def test_rehashed_candidate_still_needs_exact_identity_and_raw_scores(sandbox,monkeypatch,field,value):
    from wirelessjev.nested_radio import study
    receipt,values=numerical_fixture(sandbox,monkeypatch)
    row=c.read(c.OUT/'numerical/radiomap-5m.json');row[field]=value;rebound_numeric(receipt,0,row)
    with pytest.raises(RuntimeError):study.verify_numerical(values)


def test_rehashed_truncated_validation_predictions_do_not_reduce_a_prefix(sandbox,monkeypatch):
    import numpy as np
    from wirelessjev.nested_radio import study
    receipt,values=numerical_fixture(sandbox,monkeypatch)
    row=c.read(c.OUT/'numerical/hierarchy-fixed.json');path=c.safe_path(row['validation_predictions']['path'])
    raw=c.arrays(path);np.savez_compressed(path,mean=raw['mean'][:9],std=raw['std'][:9])
    row['validation_predictions']=c.reference(path);rebound_numeric(receipt,3,row)
    with pytest.raises(RuntimeError,match='18xQ'):study.verify_numerical(values)


def test_reused_parent_predictions_remain_bound_to_the_old_seal(sandbox,monkeypatch):
    import numpy as np
    from wirelessjev.nested_radio import study
    _,values=numerical_fixture(sandbox,monkeypatch)
    path=c.PARENT/'numerical/radiomap-5m-validation.npz';raw=c.arrays(path)
    np.savez_compressed(path,mean=raw['mean']+1.,std=raw['std'])
    with pytest.raises(RuntimeError,match='parent seal'):study.verify_numerical(values)


def prediction_fixture(sandbox):
    from copy import deepcopy
    import numpy as np
    from wirelessjev.hierarchical_radio.data import array_digest
    from wirelessjev.hierarchical_radio.evaluation import metrics
    from wirelessjev.spectral_scene.operators import query_physics_basis
    from wirelessjev.nested_radio import study
    register(sandbox)
    c.write(c.OUT/'data/prepared.json',{'scope':'canonical software prediction fixture'})
    c.write(c.OUT/'selection.json',{'scope':'canonical software prediction fixture'})
    c.write(sim.stage_path('test'),{'scope':'canonical software prediction fixture'})
    base=validation_fixture()[0];q=np.zeros((2,16),dtype='f4');q[:,6]=np.log10(np.sqrt(5.))/2
    sites=np.array([[0.,0.,3.],[10.,0.,3.]],dtype='f4');rx=np.array([[1.,0.,1.],[11.,0.,1.]],dtype='f4')
    route=np.array([0,1],dtype='i8');known=np.ones(2,bool);context=np.zeros((2,20),dtype='f4')
    context[:,4:]=q;context[0,0]=-10./30.;context[0,2]=1.;context[1,1]=-15./30.;context[1,3]=1.
    view={'context':context,'known':known,'scheduled':known.copy(),'ap':route,'rx_xyz':rx,
          'lower':np.array([-10.5,-np.inf],dtype='f4'),'upper':np.array([-9.5,-15.],dtype='f4'),
          'floor':np.full(2,-30.,dtype='f4'),'fspl':np.full(2,-60.,dtype='f4'),'offset':np.zeros(2,dtype='f4')}
    query={'q':q,'ap':route,'rx_xyz':rx,'basis':query_physics_basis(q),'fspl':view['fspl'],'offset':view['offset']}
    values=[]
    for i in range(72):
        spec=sim.case_spec('test',i//3,i%3);e=deepcopy(base)
        e.update(a=deepcopy(view),b=deepcopy(view),query=deepcopy(query),ap_xyz=sites.copy())
        e['binding'].update(stage='test',index=i//3,state=i%3,condition=i%3,layout_id=spec.layout_id,
                            case_id=spec.case_id,context_case_id=spec.case_id)
        e['binding']['model_input_sha256']={role:{k:array_digest(v) for k,v in e[role].items()} for role in ('a','b','query')}
        e['binding']['model_input_sha256']['ap_xyz']=array_digest(e['ap_xyz'])
        values.append(e)
    name='test-sparse';arrays=c.npz(c.OUT/'data/groups'/f'{name}.npz',sim.pack_examples(values))
    group={'schema':1,'stage':'test','regime':'sparse','held_band':False,'stale':False,
        'source_receipt_sha256':c.source_receipt_sha256(),'registry_sha256':c.registry_sha256(),
        'prepared_sha256':c.digest(c.OUT/'data/prepared.json'),'selection_sha256':c.digest(c.OUT/'selection.json'),
        'stage_receipt_sha256':c.digest(sim.stage_path('test')),'bindings':[e['binding'] for e in values],'arrays':arrays}
    path=c.OUT/'data/groups'/f'{name}.json';c.write(path,group);group_ref=c.reference(path)
    refs={}
    for index,e in enumerate(values):
        spec=sim.case_spec('test',index//3,index%3);folder=c.OUT/'predictions'/name
        identity={'group':name,'stage':'test','case_id':spec.case_id,'layout_id':spec.layout_id,'state':spec.state,
            'example_index':index,'example_group':group_ref,'methods':study.methods(),
            'source_receipt_sha256':c.source_receipt_sha256(),'selection_sha256':c.digest(c.OUT/'selection.json')}
        attempt=folder/'attempts'/f'{spec.case_id}.json';c.attempt(attempt,identity)
        mean=np.tile([-69.,-75.],(len(study.methods()),1));std=np.full_like(mean,3.)
        prediction=c.npz(folder/f'{spec.case_id}.npz',{'mean':mean,'std':std})
        row=identity|{'predictions':prediction,'metrics':[metrics(m,s,e['target']) for m,s in zip(mean,std)],
            'elapsed_seconds':[.01]*len(study.methods()),'attempt':c.reference(attempt)}
        output=folder/f'{spec.case_id}.json';c.write(output,row);refs[spec.case_id]=c.reference(output)
    manifest={'schema':1,'group':name,'example_group':group_ref,'case_receipts':refs,'methods':study.methods(),
        'source_receipt_sha256':c.source_receipt_sha256(),'selection_sha256':c.digest(c.OUT/'selection.json')}
    c.write(c.OUT/'predictions'/f'{name}.json',manifest)
    return values,manifest


def test_real_group_verifier_reduces_complete_canonical_prediction_cohort(sandbox):
    from wirelessjev.nested_radio import study
    _,_=prediction_fixture(sandbox)
    rows=study.records('test-sparse')
    assert len(rows)==72*19 and {r['metrics']['rmse_db'] for r in rows}=={1.}


def test_cloned_group_receipt_is_not_an_alternate_canonical_path(sandbox):
    from wirelessjev.nested_radio import study
    values,manifest=prediction_fixture(sandbox)
    path=c.OUT/'data/groups/copy.json';path.write_bytes(c.safe_path(manifest['example_group']['path']).read_bytes())
    row=c.read(c.safe_path(manifest['case_receipts'][values[0]['binding']['case_id']]['path']))
    row['example_group']=c.reference(path)
    with pytest.raises(RuntimeError,match='Canonical example group'):study.verify_prediction(row,values[0])


def test_cloned_case_receipt_cannot_replace_canonical_prediction_membership(sandbox):
    from wirelessjev.nested_radio import study
    values,manifest=prediction_fixture(sandbox);case=values[0]['binding']['case_id']
    path=c.OUT/'predictions/test-sparse/copy.json';path.write_bytes(c.safe_path(manifest['case_receipts'][case]['path']).read_bytes())
    manifest['case_receipts'][case]=c.reference(path)
    (c.OUT/'predictions/test-sparse.json').write_bytes(c.encoded(manifest))
    with pytest.raises(RuntimeError,match='Canonical case receipt'):study.records('test-sparse')


@pytest.mark.parametrize('change',[
    lambda r:r.update(methods=r['methods'][:-1]),lambda r:r.update(example_index=1),
    lambda r:r.update(group='test-full'),lambda r:r.update(state=1),
    lambda r:r.update(source_receipt_sha256='0'*64),lambda r:r.update(selection_sha256='0'*64),
    lambda r:r['metrics'][0].update(rmse_db=0.),lambda r:r['elapsed_seconds'].__setitem__(0,float('nan')),
])
def test_prediction_identity_raw_metrics_and_timing_cannot_be_relabelled(sandbox,change):
    from wirelessjev.nested_radio import study
    values,manifest=prediction_fixture(sandbox);case=values[0]['binding']['case_id']
    row=c.read(c.safe_path(manifest['case_receipts'][case]['path']));change(row)
    with pytest.raises(RuntimeError):study.verify_prediction(row,values[0])
