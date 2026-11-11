"""Operational software fixtures only: no RF, model, training or real cache run."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC=importlib.util.spec_from_file_location('scene_field_ops',Path(__file__).parents[1]/'scripts/run_scene_field_cached.py')
ops=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(ops)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(ops.encoded(value));return path


@pytest.fixture
def software(tmp_path,monkeypatch):
    root=tmp_path;out=root/'artifacts/fixture';capsule=out/'registration/source'
    sources={}
    for index in range(475):
        name=f'fixture-source/source-{index:03d}.txt';p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(str(index));sources[name]=ops.sha256(p)
        c=capsule/name;c.parent.mkdir(parents=True,exist_ok=True);c.write_bytes(p.read_bytes())
    preflight=save(out/'preflight/tests.json',{'fixture':True})
    save(out/'registration/source.json',{'source_sha256':sources,'preflight_sha256':{str(preflight.relative_to(root)):ops.sha256(preflight)}})
    for name in ops.MANDATORY:
        path=out/name
        if name!='operations/field-cache-handoff.json' and not path.exists():save(path,{'fixture':name})
    for name in ops.DIRECTORIES:save(out/name/'member.json',{'fixture':name})
    save(out/'data/frontend-training.json',{'fixture':'calibration'})
    save(out/'data/scene-development-00-development.json',{'fixture':'dev data'})
    wrapper=root/ops.WRAPPER;wrapper.parent.mkdir(parents=True,exist_ok=True);wrapper.write_text('fixture wrapper')
    test=root/ops.TEST;test.parent.mkdir(parents=True,exist_ok=True);test.write_text('fixture tests')
    amendment=save(root/'ops/amendment.json',{'fixture':'amendment'})
    publication=save(root/'ops/publication.json',{'fixture':'publication'})
    scene_receipts={}
    for index in range(24):
        name=f'fixture-field-{index:02d}';path=save(out/'rt_data/field_test'/name/'receipt.json',{'scene_id':name})
        scene_receipts[name]={'path':str(path.relative_to(root)),'sha256':ops.sha256(path)}
    save(out/'rt_data/field_test.json',{'completed':True,'stage':'field_test',
        'source_receipt_sha256':ops.sha256(out/'registration/source.json'),'registry_sha256':'fixture-registry',
        'real_device_measurements':False,'original_service_confirmation_consumed':False,'scene_receipts':scene_receipts})
    from wirelessjev.radiojepa import rt_data
    monkeypatch.setattr(rt_data,'validate_scene_receipt',lambda *a,**k:None)  # Explicit non-scientific archive fixture.
    proof={'amendment_path':'ops/amendment.json','publication_path':'ops/publication.json','amendment':{'scientific_source_receipt_sha256':ops.sha256(out/'registration/source.json')},
        'publication':{'fixture':'published'},'proof_files_sha256':ops.file_map(root,[wrapper,test,amendment,publication])}
    monkeypatch.setattr(ops,'verify_ops_publication',lambda *a:deepcopy(proof))
    calls=[]
    common=SimpleNamespace(ROOT=root,OUT=out,check_sources=lambda:calls.append('source'),
        registry_sha256=lambda:'fixture-registry',field_spec=lambda stage,index:SimpleNamespace(scene_id=f'fixture-field-{index:02d}'))
    gate={'field_test_permitted':True,'nested':{'rmse':1.},'original_service_confirmation_permitted':False,'production_qualified':False}
    def original():calls.append('full');return deepcopy(gate)
    study=SimpleNamespace(verified_development=original)
    cache=ops.DevelopmentGateCache(common,study,amendment_path=amendment,publication_path=publication)
    return SimpleNamespace(root=root,out=out,common=common,study=study,cache=cache,calls=calls,gate=gate,proof=proof)


def test_bootstrap_original_once_deepcopy_and_final_uncached_before_seal(software):
    f=software;f.cache.bootstrap();f.cache.install()
    a=f.study.verified_development();a['nested']['rmse']=999
    assert f.study.verified_development()['nested']['rmse']==1.
    assert f.calls.count('full')==1
    f.cache.final_uncached();assert f.calls.count('full')==2
    assert Path(str(f.cache.prefix)+'.final-uncached.json').is_file()
    with pytest.raises(RuntimeError):f.cache.final_uncached()
    f.cache.restore();assert f.study.verified_development is f.cache.original


@pytest.mark.parametrize('target',['source','capsule','raw_dev','checkpoint','baseline','calibration','predictions','wrapper','tests','amendment','publication','preflight'])
def test_every_dependency_class_invalidates_cached_permission(software,target):
    f=software;f.cache.bootstrap();f.cache.install()
    paths={'source':f.root/'fixture-source/source-000.txt',
        'capsule':f.out/'registration/source/fixture-source/source-000.txt',
        'raw_dev':f.out/'rt_data/development/member.json','checkpoint':f.out/'checkpoints/member.json',
        'baseline':f.out/'baselines/member.json','calibration':f.out/'data/frontend-training.json',
        'predictions':f.out/'predictions/development/member.json','wrapper':f.root/ops.WRAPPER,
        'tests':f.root/ops.TEST,'amendment':f.root/'ops/amendment.json','publication':f.root/'ops/publication.json',
        'preflight':f.out/'preflight/tests.json'}
    paths[target].write_text('changed')
    with pytest.raises(RuntimeError):f.study.verified_development()


@pytest.mark.parametrize('kind',['gate','bound','proof','bootstrap','additional_patch'])
def test_mutated_memory_cannot_replace_immutable_bootstrap_permission(software,kind):
    f=software;f.cache.bootstrap();f.cache.install()
    if kind=='gate':f.cache._gate['nested']['rmse']=999
    elif kind=='bound':f.cache._bound.clear()
    elif kind=='proof':f.cache.proof['publication']['fixture']='changed'
    elif kind=='bootstrap':f.cache.bootstrap_path.write_text('{}')
    else:f.study.verified_development=lambda:{'field_test_permitted':True}
    with pytest.raises((RuntimeError,KeyError)):f.cache()


def test_namespace_addition_deletion_and_symlink_fail_closed(software):
    f=software;f.cache.bootstrap();f.cache.install()
    extra=save(f.out/'predictions/development/extra.json',{})
    with pytest.raises(RuntimeError):f.cache()
    extra.unlink()
    member=f.out/'rt_data/development/member.json';data=member.read_bytes();member.unlink()
    with pytest.raises(RuntimeError):f.cache()
    member.write_bytes(data);member.unlink();member.symlink_to(f.out/'rt_data/training/member.json')
    with pytest.raises(RuntimeError):f.cache()


def test_new_field_outputs_are_outside_dev_gate_dependency_membership(software):
    f=software;f.cache.bootstrap();f.cache.install()
    save(f.out/'data/scene-field_test-00-field_test.json',{'new':'allowed'})
    save(f.out/'predictions/field_test/new.json',{'new':'allowed'})
    assert f.cache()['field_test_permitted'] is True


def test_failed_original_gate_never_caches_or_installs(software):
    f=software;f.study.verified_development=lambda:{'field_test_permitted':False}
    cache=ops.DevelopmentGateCache(f.common,f.study,amendment_path=f.root/'ops/amendment.json',publication_path=f.root/'ops/publication.json')
    with pytest.raises(RuntimeError,match='must permit FIELD'):cache.bootstrap()
    assert not cache.bootstrap_path.exists()
    with pytest.raises(RuntimeError):cache.install()


def test_original_verification_dependency_mutation_is_detected(software):
    f=software
    def changing():
        (f.out/'data/frontend-training.json').write_text('changed by original');return f.gate
    f.study.verified_development=changing
    cache=ops.DevelopmentGateCache(f.common,f.study,amendment_path=f.root/'ops/amendment.json',publication_path=f.root/'ops/publication.json')
    with pytest.raises(RuntimeError,match='during original'):cache.bootstrap()


def test_hit_logs_reject_tampering_and_are_parseable_jsonl(software):
    f=software;f.cache.bootstrap();f.cache.install();f.cache()
    hits=list((f.out/'operations').glob('*.hits.jsonl'));assert len(hits)==1
    value=json.loads(hits[0].read_text());assert value['all_dependency_bytes_and_membership_checked'] is True
    hits[0].write_text('{}\n')
    with pytest.raises(RuntimeError,match='hit log changed'):f.cache()


def test_unpublished_or_wrong_scope_amendment_is_rejected_before_original(tmp_path):
    amendment=save(tmp_path/'amendment.json',{'scientific_source_changed':True})
    publication=save(tmp_path/'publication.json',{})
    with pytest.raises(RuntimeError,match='permitted cache scope'):ops.verify_ops_publication(tmp_path,amendment,publication)


def test_handoff_first_entry_requires_zero_unproven_outputs_then_resumes_exactly(software):
    f=software
    save(f.out/'predictions/field_test/unproven.json',{'fixture':'unproven'})
    with pytest.raises(RuntimeError,match='Unproven existing'):f.cache.bootstrap()
    assert not (f.out/'operations/field-cache-handoff.json').exists()


def test_proven_handoff_retains_partial_outputs_without_replacing_them(software):
    f=software;first=ops.prepare_handoff(f.common,f.proof)
    output=save(f.out/'predictions/field_test/partial.json',{'fixture':'retained'})
    before=output.read_bytes()
    assert ops.prepare_handoff(f.common,f.proof)==first and output.read_bytes()==before
    f.cache.bootstrap();f.cache.install();assert f.cache()['field_test_permitted'] is True
    assert output.read_bytes()==before


@pytest.mark.parametrize('change',['source','amendment','publication','collector','receipt'])
def test_existing_handoff_cannot_rebless_changed_source_proof_or_collector(software,change):
    f=software;ops.prepare_handoff(f.common,f.proof)
    paths={'source':f.out/'registration/source.json','amendment':f.root/'ops/amendment.json',
        'publication':f.root/'ops/publication.json','collector':f.out/'rt_data/field_test.json',
        'receipt':f.out/'rt_data/field_test/fixture-field-00/receipt.json'}
    paths[change].write_text('{}')
    with pytest.raises((RuntimeError,KeyError)):ops.prepare_handoff(f.common,f.proof)


@pytest.mark.parametrize('key,value',[('original_service_confirmation_permitted',True),
    ('original_service_confirmation_permitted',0),('production_qualified',True),('production_qualified',None)])
def test_service_or_production_permission_cannot_bootstrap_cache(software,key,value):
    f=software;f.gate[key]=value
    with pytest.raises(RuntimeError,match='SERVICE/production'):f.cache.bootstrap()
    assert not f.cache.bootstrap_path.exists()


def test_original_callable_replacement_cannot_fake_final_full_replay(software):
    f=software;f.cache.bootstrap();f.cache.install()
    f.cache.original=lambda:deepcopy(f.cache._gate)
    with pytest.raises(RuntimeError,match='callable changed'):f.cache.final_uncached()
    assert not Path(str(f.cache.prefix)+'.final-uncached.json').exists()
    f.cache.restore();assert f.study.verified_development is f.cache._original_callable
