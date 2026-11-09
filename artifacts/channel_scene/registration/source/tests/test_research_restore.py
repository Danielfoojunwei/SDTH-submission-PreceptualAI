"""Provider-record fixtures and real temporary ZIPs; no live GitHub claims."""
import copy,importlib.util,json,shutil,stat,sys,zipfile
from pathlib import Path
import pytest
from wirelessjev.research_watch import canonical_json

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
SPEC=importlib.util.spec_from_file_location('research_history_restore',ROOT/'scripts/restore_research_watch.py')
RESTORE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(RESTORE)
REPO='owner/repo';META={'id':42,'full_name':REPO,'default_branch':'main'}
SEED=ROOT/'artifacts/ran_research/watch'

def run(identity=90,*,branch='main',event='schedule',workflow=7,foreign=False):
    return {'id':identity,'workflow_id':workflow,'path':RESTORE.WORKFLOW,'repository':META,
            'head_repository':{'id':43,'full_name':'fork/repo'} if foreign else META,
            'head_branch':branch,'head_sha':f'{identity:040x}','event':event,'status':'completed'}

def archive(tmp_path,source=SEED):
    path=tmp_path/'prior.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for item in source.rglob('*'):
            if item.is_file():bundle.write(item,item.relative_to(source).as_posix())
    return path

def artifact(source,identity=900):
    return {'id':identity,'name':f'ran-research-watch-{source["id"]}-1',
            'created_at':'2026-11-03T10:00:00Z','expired':False,'size_in_bytes':1000,
            'workflow_run':{'id':source['id'],'repository_id':42,
                'head_repository_id':source['head_repository']['id'],'head_branch':source['head_branch'],
                'head_sha':source['head_sha']}}

class API:
    def __init__(self,zip_path=None,prior=None,artifacts=None):
        self.current=run(100,event='workflow_dispatch');self.current['status']='in_progress'
        self.prior=[run()] if prior is None else prior
        self.artifacts=[artifact(self.prior[0])] if artifacts is None and self.prior else (artifacts or [])
        self.records={x['id']:x for x in [self.current,*self.prior]}
        self.zip_path=zip_path;self.calls=[];self.downloads=[];self.failure=None
    def json(self,endpoint):
        self.calls.append(endpoint)
        if self.failure:raise RESTORE.RecoveryError('Fixture API failure')
        path=endpoint.split('?')[0]
        if path=='repos/'+REPO:return copy.deepcopy(META)
        if path.endswith('/actions/workflows/7/runs'):
            rows=[self.current,*self.prior]
            return {'total_count':len(rows),'workflow_runs':copy.deepcopy(rows)}
        if path.endswith('/actions/artifacts'):
            return {'total_count':len(self.artifacts),'artifacts':copy.deepcopy(self.artifacts)}
        if '/actions/runs/' in path:return copy.deepcopy(self.records[int(path.rsplit('/',1)[1])])
        if '/actions/artifacts/' in path:return copy.deepcopy(next(x for x in self.artifacts if x['id']==int(path.rsplit('/',1)[1])))
        raise AssertionError(endpoint)
    def download(self,endpoint,destination):self.downloads.append(endpoint);shutil.copyfile(self.zip_path,destination)

def recover(target,api):return RESTORE.recover(target,REPO,'main',100,api=api)

def advance(directory,label):
    old=json.loads((directory/'latest.json').read_bytes());new=copy.deepcopy(old)
    new['previous_snapshot_sha256']=RESTORE.hashlib.sha256(canonical_json(old)).hexdigest()
    # Equal timestamps are valid; this tests history mechanics, not new research.
    (directory/'snapshots'/(label+'.json')).write_text(json.dumps(new))
    (directory/'latest.json').write_text(json.dumps(new));RESTORE.verify(directory)

def test_existing_cache_is_reconciled_with_latest_artifact(tmp_path):
    target=tmp_path/'watch';shutil.copytree(SEED,target);api=API(archive(tmp_path))
    before=(target/'latest.json').read_bytes();result=recover(target,api)
    assert result['source']=='verified_cache_at_or_after_latest_artifact' and api.calls and api.downloads
    assert (target/'latest.json').read_bytes()==before

def test_cache_loss_recovers_full_verified_artifact_using_official_run_records(tmp_path):
    api=API(archive(tmp_path));target=tmp_path/'watch';result=recover(target,api)
    assert result['artifact_id']==900 and result['source_run_id']==90
    assert len(api.downloads)==1 and RESTORE.verify(target)['valid_snapshot']
    assert (target/'latest.json').read_bytes()==(SEED/'latest.json').read_bytes()
    assert any('/actions/workflows/7/runs?' in p for p in api.calls)
    assert not any('/workflows/ran-research-watch.yml' in p for p in api.calls)

def test_initial_genesis_requires_no_prior_default_branch_history(tmp_path):
    api=API(prior=[],artifacts=[]);target=tmp_path/'watch';result=recover(target,api)
    assert result['source']=='initial_genesis_committed_seed' and result['no_prior_default_branch_runs']
    assert not api.downloads and RESTORE.verify(target)['valid_snapshot']

@pytest.mark.parametrize('change',[{'event':'pull_request'},{'branch':'topic'},{'foreign':True},{'workflow':9}])
def test_untrusted_or_other_workflow_artifacts_cannot_supply_history(tmp_path,change):
    trusted=run(90);foreign=run(99,**change);good=artifact(trusted,900);bad=artifact(foreign,999)
    bad['created_at']='2026-11-03T11:00:00Z'
    api=API(archive(tmp_path),prior=[foreign,trusted],artifacts=[bad,good])
    assert recover(tmp_path/'watch',api)['artifact_id']==900
    assert api.downloads==['repos/'+REPO+'/actions/artifacts/900/zip']

def test_current_run_artifact_is_excluded(tmp_path):
    api=API(archive(tmp_path));current=artifact(api.current,1000)
    current['created_at']='2026-11-03T12:00:00Z';api.artifacts.insert(0,current)
    assert recover(tmp_path/'watch',api)['source_run_id']==90

@pytest.mark.parametrize('failure',['absent','expired','origin','api'])
def test_existing_history_cannot_silently_reset_to_seed(tmp_path,failure):
    api=API(archive(tmp_path));target=tmp_path/'watch'
    if failure=='absent':api.artifacts=[]
    if failure=='expired':api.artifacts[0]['expired']=True
    if failure=='origin':api.artifacts[0]['workflow_run']['head_sha']='b'*40
    if failure=='api':api.failure=True
    with pytest.raises(RESTORE.RecoveryError):recover(target,api)
    assert not target.exists()

def test_invalid_existing_cache_is_retained_without_using_network(tmp_path):
    target=tmp_path/'watch';shutil.copytree(SEED,target)
    response=next((target/'responses').glob('*.bin'));response.write_bytes(b'corrupt');api=API()
    with pytest.raises(RESTORE.RecoveryError,match='verification'):recover(target,api)
    assert not api.calls and response.read_bytes()==b'corrupt'

@pytest.mark.parametrize('entry',['../escape','/escape','responses/../../escape','C:\\escape','payload.py'])
def test_zip_paths_are_checked_before_publication(tmp_path,entry):
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as bundle:bundle.writestr(entry,b'not executable')
    target=tmp_path/'watch';api=API(path)
    with pytest.raises(RESTORE.RecoveryError):recover(target,api)
    assert not target.exists() and not (tmp_path/'escape').exists()

def test_zip_symlinks_size_limits_and_invalid_snapshots_are_rejected(tmp_path,monkeypatch):
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as bundle:
        info=zipfile.ZipInfo('latest.json');info.create_system=3;info.external_attr=(stat.S_IFLNK|0o777)<<16
        bundle.writestr(info,'/etc/passwd')
    with pytest.raises(RESTORE.RecoveryError):recover(tmp_path/'symlink',API(path))
    with zipfile.ZipFile(path,'w') as bundle:bundle.writestr('latest.json','{}')
    with pytest.raises(RESTORE.RecoveryError,match='verification'):recover(tmp_path/'invalid',API(path))
    monkeypatch.setattr(RESTORE,'MAX_ZIP_BYTES',20)
    with pytest.raises(RESTORE.RecoveryError):recover(tmp_path/'oversize',API(path))

def test_incomplete_pagination_is_an_error_instead_of_genesis(tmp_path):
    class Incomplete(API):
        def json(self,endpoint):
            if '/workflows/7/runs?' in endpoint:return {'total_count':2,'workflow_runs':[]}
            return super().json(endpoint)
    with pytest.raises(RESTORE.RecoveryError):recover(tmp_path/'watch',Incomplete(prior=[],artifacts=[]))

def test_newer_artifact_replaces_stale_cache_without_losing_its_history(tmp_path):
    cache=tmp_path/'watch';remote=tmp_path/'remote'
    shutil.copytree(SEED,cache);shutil.copytree(SEED,remote);advance(remote,'newer-artifact')
    expected=(remote/'latest.json').read_bytes();api=API(archive(tmp_path,remote))
    result=recover(cache,api)
    assert result['source']=='authenticated_default_branch_artifact'
    assert (cache/'latest.json').read_bytes()==expected and RESTORE.verify(cache)['valid_snapshot']
    assert (cache/'snapshots'/'newer-artifact.json').exists()

def test_cache_descendant_is_retained_after_artifact_reconciliation(tmp_path):
    cache=tmp_path/'watch';shutil.copytree(SEED,cache);advance(cache,'newer-cache')
    expected=(cache/'latest.json').read_bytes();result=recover(cache,API(archive(tmp_path)))
    assert result['source']=='verified_cache_at_or_after_latest_artifact'
    assert (cache/'latest.json').read_bytes()==expected

def test_divergent_history_fails_without_replacing_cache(tmp_path):
    cache=tmp_path/'watch';remote=tmp_path/'remote';shutil.copytree(SEED,cache);shutil.copytree(SEED,remote)
    advance(cache,'cache-child');advance(remote,'remote-child')
    # Distinct valid publication metadata gives two different children.
    item=json.loads((remote/'latest.json').read_bytes());item['generated_at']=item['generated_at'].replace('Z','+00:00')
    (remote/'latest.json').write_text(json.dumps(item));(remote/'snapshots'/'remote-child.json').write_text(json.dumps(item))
    RESTORE.verify(remote);before=(cache/'latest.json').read_bytes()
    with pytest.raises(RESTORE.RecoveryError,match='diverge'):recover(cache,API(archive(tmp_path,remote)))
    assert (cache/'latest.json').read_bytes()==before

def test_valid_cache_is_not_presented_as_current_on_api_failure(tmp_path):
    cache=tmp_path/'watch';shutil.copytree(SEED,cache);before=(cache/'latest.json').read_bytes()
    api=API();api.failure=True
    with pytest.raises(RESTORE.RecoveryError):recover(cache,api)
    assert (cache/'latest.json').read_bytes()==before


@pytest.mark.parametrize('malformed',['run','artifact'])
def test_malformed_provider_records_never_become_initial_genesis(tmp_path,malformed):
    api=API(prior=[],artifacts=[])
    if malformed=='run':
        bad=run(90);del bad['path'];api.prior=[bad]
    else:api.artifacts=[{'id':900,'created_at':'2026-11-03T10:00:00Z'}]
    with pytest.raises(RESTORE.RecoveryError,match='Malformed'):recover(tmp_path/'watch',api)
    assert not (tmp_path/'watch').exists()
