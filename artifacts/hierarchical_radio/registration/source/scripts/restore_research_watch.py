"""Recover verified discovery history without resetting a durable lineage."""
import argparse,ctypes,hashlib,json,os,re,resource,shutil,stat,subprocess,tempfile,time,zipfile
from pathlib import Path,PurePosixPath
from urllib.parse import urlencode
from verify_research_watch import verify
from wirelessjev.research_watch import canonical_json

ROOT=Path(__file__).resolve().parents[1]
WORKFLOW='.github/workflows/ran-research-watch.yml'
PREFIX='ran-research-watch-'
MAX_PAGES=3
PAGE_SIZE=50
MAX_ZIP_BYTES=64*1024*1024
MAX_STORE_BYTES=256*1024*1024
MAX_FILES=10000

class RecoveryError(RuntimeError):pass

class GitHub:
    """Official REST through configured gh authentication; never emit credentials."""
    def __init__(self,seconds=120):self.deadline=time.monotonic()+seconds
    def _request(self,endpoint,destination,limit):
        remaining=self.deadline-time.monotonic()
        if remaining<=0:raise RecoveryError('Recovery time budget exhausted')
        if not endpoint.startswith('repos/') or '://' in endpoint:raise RecoveryError('Invalid REST endpoint')
        env=os.environ.copy();env.update(GH_PROMPT_DISABLED='1',GH_PAGER='',PAGER='')
        env.pop('GH_FORCE_TTY',None);env.pop('GH_DEBUG',None)
        def bound():resource.setrlimit(resource.RLIMIT_FSIZE,(limit,limit))
        try:
            with Path(destination).open('wb') as output:
                result=subprocess.run(['gh','api','--hostname','github.com',endpoint],stdout=output,
                    stderr=subprocess.DEVNULL,env=env,timeout=min(20,remaining),preexec_fn=bound,check=False)
        except (OSError,subprocess.SubprocessError) as error:
            raise RecoveryError('Official GitHub request failed or timed out') from error
        if result.returncode or Path(destination).stat().st_size>limit:
            raise RecoveryError('Official GitHub request failed or exceeded its byte limit')
    def json(self,endpoint):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'response.json';self._request(endpoint,path,2*1024*1024)
            try:return json.loads(path.read_bytes())
            except (ValueError,UnicodeError) as error:raise RecoveryError('Invalid GitHub metadata') from error
    def download(self,endpoint,destination):self._request(endpoint,destination,MAX_ZIP_BYTES)

def integer(value):
    if type(value) is not int or value<1:raise RecoveryError('Invalid GitHub record identifier')
    return value

def repository_matches(record,name,identity=None):
    return (isinstance(record,dict) and type(record.get('full_name')) is str
            and record['full_name'].casefold()==name.casefold()
            and type(record.get('id')) is int and record['id']>0
            and (identity is None or record['id']==identity))

def run_origin(run,name,identity,workflow,branch):
    if not isinstance(run,dict) or not repository_matches(run.get('repository'),name,identity):
        raise RecoveryError('Workflow run repository origin differs')
    integer(run.get('id'));integer(run.get('workflow_id'))
    head=run.get('head_repository')
    if (not isinstance(head,dict) or type(head.get('id')) is not int or type(head.get('full_name')) is not str
            or any(type(run.get(k)) is not str for k in ('path','head_branch','event','status','head_sha'))):
        raise RecoveryError('Malformed workflow run metadata cannot establish absence')
    return (run['workflow_id']==workflow and run.get('path')==WORKFLOW
            and run.get('head_branch')==branch and repository_matches(run.get('head_repository'),name,identity)
            and run.get('event') in ('push','schedule','workflow_dispatch'))

def pages(api,endpoint,key):
    for page in range(1,MAX_PAGES+1):
        response=api.json(endpoint+('&' if '?' in endpoint else '?')+urlencode({'per_page':PAGE_SIZE,'page':page}))
        if (not isinstance(response,dict) or type(response.get('total_count')) is not int
                or response['total_count']<0 or not isinstance(response.get(key),list)
                or len(response[key])>PAGE_SIZE or any(not isinstance(x,dict) for x in response[key])):
            raise RecoveryError('Malformed paginated GitHub metadata')
        extent=(page-1)*PAGE_SIZE+len(response[key])
        if response['total_count']<extent or (len(response[key])<PAGE_SIZE and response['total_count']>extent):
            raise RecoveryError('Incomplete pagination cannot establish absence')
        yield response[key]
        if len(response[key])<PAGE_SIZE:return
    raise RecoveryError('Listing exceeded its page budget; operator recovery required')

def allowed_path(name,directory=False):
    if (not isinstance(name,str) or '\\' in name or '\x00' in name or len(name)>240
            or name.startswith('/') or any(x in ('','.','..') for x in name.rstrip('/').split('/'))):
        raise RecoveryError('Unsafe recovery archive path')
    path=PurePosixPath(name.rstrip('/'))
    if directory:
        if path.as_posix() not in ('responses','snapshots'):raise RecoveryError('Unexpected archive directory')
    elif not (path.as_posix() in ('latest.json','run_receipt.json','.publish.lock')
              or re.fullmatch(r'responses/[0-9a-f]{64}\.bin',path.as_posix())
              or re.fullmatch(r'snapshots/[A-Za-z0-9_.-]{1,180}\.json',path.as_posix())):
        raise RecoveryError('Unexpected recovery store file')
    return path

def safe_extract(archive,directory):
    if archive.stat().st_size>MAX_ZIP_BYTES:raise RecoveryError('Recovery ZIP exceeds byte limit')
    total=0;seen=set()
    with zipfile.ZipFile(archive) as bundle:
        entries=bundle.infolist()
        if len(entries)>MAX_FILES:raise RecoveryError('Archive has too many entries')
        for info in entries:
            path=allowed_path(info.filename,info.is_dir());mode=stat.S_IFMT(info.external_attr>>16)
            if (info.orig_filename!=info.filename or info.flag_bits&1
                    or mode not in (0,stat.S_IFREG,stat.S_IFDIR) or path.as_posix().casefold() in seen):
                raise RecoveryError('Duplicate, linked or encrypted archive entry')
            seen.add(path.as_posix().casefold());total+=info.file_size
            if info.file_size>20*1024*1024 or total>MAX_STORE_BYTES:
                raise RecoveryError('Expanded archive exceeds byte limit')
            target=directory.joinpath(*path.parts)
            if info.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with bundle.open(info) as source,target.open('xb') as output:
                written=0
                while chunk:=source.read(65536):
                    written+=len(chunk)
                    if written>info.file_size:raise RecoveryError('Expanded archive size differs')
                    output.write(chunk)

def checked_store(directory,validator):
    total=0;count=0
    if directory.is_symlink() or not directory.is_dir():raise RecoveryError('Store must be a regular directory')
    for path in directory.rglob('*'):
        kind=path.lstat().st_mode
        if stat.S_ISLNK(kind) or not (stat.S_ISREG(kind) or stat.S_ISDIR(kind)):
            raise RecoveryError('Store contains a linked or special entry')
        allowed_path(path.relative_to(directory).as_posix(),path.is_dir())
        if path.is_file():total+=path.stat().st_size;count+=1
        if total>MAX_STORE_BYTES or count>MAX_FILES:raise RecoveryError('Store exceeds recovery limits')
    try:validator(directory)
    except Exception as error:raise RecoveryError('Retained history failed verification; refusing to reset') from error

def lineage(directory):
    history={}
    for path in (directory/'snapshots').glob('*.json'):
        item=json.loads(path.read_bytes());history[hashlib.sha256(canonical_json(item)).hexdigest()]=item['previous_snapshot_sha256']
    head=hashlib.sha256(canonical_json(json.loads((directory/'latest.json').read_bytes()))).hexdigest()
    return head,history

def ancestry(head,history):
    result=set()
    while head is not None:
        if head in result or head not in history:raise RecoveryError('Unverifiable lineage')
        result.add(head);head=history[head]
    return result

def immutable_superset(local,remote):
    for directory in ('snapshots','responses'):
        for path in (remote/directory).glob('*'):
            target=local/directory/path.name
            if not target.is_file() or target.read_bytes()!=path.read_bytes():return False
    return True

def merge_immutable(local,staged):
    for directory in ('snapshots','responses'):
        for path in (local/directory).glob('*'):
            target=staged/directory/path.name
            if target.exists() and target.read_bytes()!=path.read_bytes():raise RecoveryError('Immutable recovery paths conflict')
            if not target.exists():target.parent.mkdir(exist_ok=True);shutil.copyfile(path,target)

def atomic_exchange(staged,target):
    # Linux runner: atomically swap nonempty directories, with no missing-store
    # interval and no destructive fallback if this filesystem lacks support.
    libc=ctypes.CDLL(None,use_errno=True)
    try:exchange=libc.renameat2
    except AttributeError as error:raise RecoveryError('Atomic recovery exchange is unsupported') from error
    exchange.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    exchange.restype=ctypes.c_int
    if exchange(-100,os.fsencode(staged),-100,os.fsencode(target),2):
        raise RecoveryError('Atomic recovery exchange failed; original store retained')

def recover(output_dir,repository,default_branch,run_id,*,api=None,seed_dir=None,validator=verify):
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]{1,100}',repository)
            or repository.split('/')[1] in ('.','..') or not default_branch
            or len(default_branch)>255 or any(ord(c)<32 for c in default_branch)):
        raise RecoveryError('Invalid repository or default branch')
    run_id=integer(run_id);target=Path(output_dir)
    if target.is_symlink() or (target.exists() and not target.is_dir()):raise RecoveryError('Invalid output store')
    if target.exists() and any(target.iterdir()):
        checked_store(target,validator)
        cached=True
    else:cached=False
    api=api or GitHub();base='repos/'+repository
    repo=api.json(base)
    if not repository_matches(repo,repository) or repo.get('default_branch')!=default_branch:
        raise RecoveryError('Official repository/default-branch metadata differs')
    identity=repo['id'];current=api.json(base+'/actions/runs/'+str(run_id))
    workflow=integer(current.get('workflow_id'))
    if current.get('id')!=run_id or current.get('path')!=WORKFLOW or not repository_matches(current.get('repository'),repository,identity):
        raise RecoveryError('Current run repository, identifier or workflow differs')
    prior={}
    endpoint=base+'/actions/workflows/'+str(workflow)+'/runs?'+urlencode({'branch':default_branch})
    for batch in pages(api,endpoint,'workflow_runs'):
        for run in batch:
            if run_origin(run,repository,identity,workflow,default_branch) and run['id']!=run_id:prior[run['id']]=run
        if prior:break
    chosen=None
    # Authenticate source runs of the newest repository artifacts, including
    # reruns of old runs not present on the recent workflow-runs page.
    for batch in pages(api,base+'/actions/artifacts','artifacts'):
        for row in batch:
            integer(row.get('id'))
            if type(row.get('name')) is not str or type(row.get('created_at')) is not str:
                raise RecoveryError('Malformed artifact metadata cannot establish absence')
        batch=sorted(batch,key=lambda a:(a['created_at'],a['id']),reverse=True)
        for candidate in batch:
            if not str(candidate.get('name','')).startswith(PREFIX):continue
            source_id=integer(candidate.get('workflow_run',{}).get('id'))
            if source_id==run_id:continue
            run=api.json(base+'/actions/runs/'+str(source_id))
            if run.get('id')!=source_id:raise RecoveryError('Artifact source run identifier differs')
            if not run_origin(run,repository,identity,workflow,default_branch) or run.get('status')!='completed':continue
            artifact=api.json(base+'/actions/artifacts/'+str(integer(candidate.get('id'))))
            origin=artifact.get('workflow_run',{})
            if (artifact.get('id')!=candidate['id'] or not str(artifact.get('name','')).startswith(PREFIX)
                    or origin.get('id')!=source_id or origin.get('repository_id')!=identity
                    or origin.get('head_repository_id')!=identity or origin.get('head_branch')!=default_branch
                    or not run.get('head_sha') or origin.get('head_sha')!=run['head_sha']):
                raise RecoveryError('Artifact provenance differs from its authenticated source run')
            if artifact.get('expired') is not False:raise RecoveryError('Latest durable history artifact expired; operator recovery required')
            size=artifact.get('size_in_bytes')
            if type(size) is not int or not 0<size<=MAX_ZIP_BYTES:raise RecoveryError('Artifact size is invalid')
            chosen=(artifact,source_id);break
        if chosen:break
    if chosen is None and (prior or cached):
        raise RecoveryError('Prior default-branch history exists without a recoverable artifact; refusing to reset')
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.watch-recovery-',dir=target.parent) as temporary:
        temporary=Path(temporary);staged=temporary/'store';staged.mkdir()
        if chosen:
            archive=temporary/'history.zip';artifact,source_id=chosen
            api.download(base+'/actions/artifacts/'+str(artifact['id'])+'/zip',archive)
            safe_extract(archive,staged)
            result={'source':'authenticated_default_branch_artifact','source_run_id':source_id,'artifact_id':artifact['id'],'network_used':True}
        else:
            seed=Path(seed_dir) if seed_dir is not None else ROOT/'artifacts/ran_research/watch'
            checked_store(seed,validator);shutil.copytree(seed,staged,dirs_exist_ok=True,symlinks=True)
            result={'source':'initial_genesis_committed_seed','network_used':True,'no_prior_default_branch_runs':True}
        checked_store(staged,validator)
        if cached:
            local,remote=lineage(target),lineage(staged)
            local_chain=ancestry(*local);remote_chain=ancestry(*remote)
            if remote[0] in local_chain:
                # A verified cache may contain later, not-yet-uploaded history.
                preferred=target
                if immutable_superset(target,staged):
                    return result|{'source':'verified_cache_at_or_after_latest_artifact'}
            elif local[0] in remote_chain:preferred=staged
            else:raise RecoveryError('Cache/artifact lineages diverge; retain originals for operator recovery')
            merge_immutable(target,staged)
            if preferred==target:
                shutil.copyfile(target/'latest.json',staged/'latest.json')
                if (target/'run_receipt.json').exists():shutil.copyfile(target/'run_receipt.json',staged/'run_receipt.json')
            checked_store(staged,validator)
        if target.exists() and any(target.iterdir()):atomic_exchange(staged,target)
        else:os.replace(staged,target)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--repository',required=True);parser.add_argument('--default-branch',required=True)
    parser.add_argument('--run-id',type=int,required=True)
    args=parser.parse_args()
    try:print(json.dumps(recover(args.output_dir,args.repository,args.default_branch,args.run_id),sort_keys=True))
    except Exception as error:raise SystemExit('Research history recovery failed: '+str(error)) from None
