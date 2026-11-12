"""Source/publication guards for an independent adaptive RF-condition experiment.

The completed RadioJEPA pilot is sealed separately. Full previous closure is
checked at phase boundaries; ordinary epoch checks bind source bytes and the
unchanged previous manifest, without repeating an old numerical audit.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'artifacts/adaptive_radio'
WORK = ROOT.parent/'training/adaptive-radio'
PROTOCOL = OUT/'protocol.json'
REGISTRY = OUT/'geometry_registry.json'
PREVIOUS_MANIFEST = ROOT/'artifacts/radiojepa_spectral/verification/manifest.json'
PREVIOUS_MANIFEST_SHA256 = '53f329a2ce82d6498116fa098b1231ad31c5826be4435b4610c713701bcf598c'
COUNTS = {'training':24,'validation':6,'development':4,'test':24}
CONDITIONS=(0,1,2)
STAGES = tuple(COUNTS)
_PUBLICATION_VERIFIED = set()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def value_digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_path(name,root=None):
    root=ROOT if root is None else Path(root)
    relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise RuntimeError('Unsafe relative adaptive evidence path')
    full=root/relative
    if not full.resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Adaptive evidence escaped its root')
    for p in (full,*full.parents):
        if p==root:break
        if p.is_symlink():raise RuntimeError('Redirected adaptive evidence member')
    return full


def write_bytes_once(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open('xb') as stream:
            stream.write(data);stream.flush()
            import os
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes()!=data:raise RuntimeError('Refuse to rewrite adaptive evidence: '+str(path))
    return path


def write_once(path,value):
    return write_bytes_once(path,encoded(value))


def protocol():
    return read_json(PROTOCOL)


@contextmanager
def lock(path,*,nonblocking=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        except BlockingIOError as exc:raise RuntimeError('Adaptive study is already active') from exc
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def previous_manifest():
    if digest(PREVIOUS_MANIFEST)!=PREVIOUS_MANIFEST_SHA256:
        raise RuntimeError('The completed RadioJEPA manifest changed')
    value=read_json(PREVIOUS_MANIFEST)
    if len(value['source_sha256'])!=475 or len(value['evidence_sha256'])!=5161:
        raise RuntimeError('The completed RadioJEPA closure identity changed')
    return value


def verify_previous():
    previous_manifest()
    from ..spectral_scene.verification import verify_closure
    value=verify_closure()
    if value.get('passes') is not True:
        raise RuntimeError('Previous RadioJEPA closure failed')
    original=ROOT/'artifacts/radiojepa/rt_data'
    if (original/'confirmation').exists() or (original/'confirmation.json').exists():
        raise RuntimeError('Original SERVICE confirmation geometries must stay unopened')
    return {'manifest_sha256':PREVIOUS_MANIFEST_SHA256,'sources':475,'evidence':5161,
            'original_service_confirmation_opened':False}



def registry():
    from . import simulation as sim
    return sim.geometry_registry()


def write_registry():return write_once(REGISTRY,registry())


def registry_sha256():
    expected=registry()
    if read_json(REGISTRY)!=json.loads(encoded(expected)):raise RuntimeError('Adaptive public registry changed')
    return value_digest(expected)


def field_spec(stage,index):
    from . import simulation as sim
    if stage not in COUNTS or type(index) is not int or not 0<=index<COUNTS[stage]:raise ValueError('Unregistered adaptive geometry')
    return sim.geometry_spec(stage,index)


def case_spec(stage,index,condition):
    from . import simulation as sim
    if condition not in CONDITIONS:raise ValueError('Unregistered RF condition')
    return sim.case_spec(stage,index,condition)


def case_path(stage,index,condition):
    spec=case_spec(stage,index,condition)
    if stage=='test':authorize_stage(stage)
    return OUT/'rt_data'/stage/spec.case_id

def source_hashes():
    m=previous_manifest()
    result=dict(m['source_sha256'])
    result.update(m['documents_sha256'])
    paths=[p for p in (ROOT/'src/wirelessjev/adaptive_radio').rglob('*')
           if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.lock')]
    paths+=list((ROOT/'tests').glob('test_adaptive*.py'))
    paths+=list((ROOT/'scripts').glob('*adaptive_radio*.py'))
    paths+=[ROOT/'docs/adaptive_radio/PLAN.md',PROTOCOL,REGISTRY,OUT/'execution_order.json',PREVIOUS_MANIFEST]
    for name in ('BASELINES.md','SERVICE.md'):
        if (ROOT/'docs/adaptive_radio'/name).exists():paths.append(ROOT/'docs/adaptive_radio'/name)
    if (OUT/'data_registry.json').exists():paths.append(OUT/'data_registry.json')
    for p in sorted(set(paths)):
        name=str(p.relative_to(ROOT));result[name]=digest(safe_path(name))
    return dict(sorted(result.items()))


def register():
    with lock(OUT/'registration.lock'):
        if (OUT/'registration/source.json').exists():return check_sources(require_published=False)
        prior=verify_previous()
        for name in ('data','fits','checkpoints','predictions','probes','baselines','rt_data',
                     'results','verification','gates','report.json','development.json','qualification.json','selection.json'):
            p=OUT/name
            if p.is_file() or (p.is_dir() and any(p.rglob('*'))):
                raise RuntimeError('Register adaptive source before scientific outputs: '+name)
        registry_sha256()
        sources=source_hashes()
        tests_path=OUT/'preflight/tests.json';tests=read_json(tests_path)
        if (tests.get('exit_code')!=0 or tests.get('sources_before_sha256')!=sources
                or tests.get('sources_after_sha256')!=sources or tests.get('prior_evidence')!=prior):
            raise RuntimeError('Final tests must pass on exactly the adaptive source being registered')
        stdout=safe_path(tests['stdout_path'])
        if digest(stdout)!=tests['stdout_sha256']:raise RuntimeError('Final adaptive test log changed')
        preflight={str(tests_path.relative_to(ROOT)):digest(tests_path),str(stdout.relative_to(ROOT)):digest(stdout)}
        capsule=OUT/'registration/source'
        for name,expected in sources.items():
            destination=safe_path(name,capsule);destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                if digest(destination)!=expected:raise RuntimeError('Partial source capsule differs')
            else:shutil.copyfile(safe_path(name),destination)
        value={'schema':1,'source_sha256':sources,'previous':prior,
               'registry_sha256':registry_sha256(),'protocol_sha256':digest(PROTOCOL),
               'preflight_sha256':preflight,
               'scientific_models_fields_or_fits_before_registration':False,
               'unit_fixtures_and_public_specifications_are_not_scientific_runs':True}
        if verify_previous()!=prior:raise RuntimeError('Previous evidence changed during registration')
        write_once(OUT/'registration/source.json',value)
        return value


def check_sources(*,require_published=True):
    receipt=read_json(OUT/'registration/source.json')
    sources=source_hashes()
    for name,expected in receipt['preflight_sha256'].items():
        if digest(safe_path(name))!=expected:raise RuntimeError('Published adaptive test proof changed')
    if (receipt['source_sha256']!=sources or receipt['protocol_sha256']!=digest(PROTOCOL)
            or receipt['registry_sha256']!=registry_sha256()):
        raise RuntimeError('Registered adaptive source or specifications changed')
    capsule=OUT/'registration/source'
    actual={str(p.relative_to(capsule)) for p in capsule.rglob('*') if p.is_file()}
    if actual!=set(sources):raise RuntimeError('Adaptive source capsule membership changed')
    for name,expected in sources.items():
        if digest(safe_path(name,capsule))!=expected:raise RuntimeError('Adaptive source capsule bytes changed')
    if require_published:verify_publication(receipt)
    return receipt


def verify_publication(receipt=None):
    receipt=read_json(OUT/'registration/source.json') if receipt is None else receipt
    published=read_json(OUT/'registration/git.json')
    commit=published.get('commit')
    if (not isinstance(commit,str) or not re.fullmatch(r'[0-9a-f]{40}',commit)
            or published.get('remote_observed_commit')!=commit
            or not published.get('branch') or not published.get('remote')
            or published.get('source_receipt_sha256')!=digest(OUT/'registration/source.json')):
        raise RuntimeError('Publish and remotely verify immutable adaptive source before science')
    key=value_digest(published)
    if key in _PUBLICATION_VERIFIED:return published
    names=dict(receipt['source_sha256']);names.update(receipt['preflight_sha256'])
    names['artifacts/adaptive_radio/registration/source.json']=published['source_receipt_sha256']
    names.update({'artifacts/adaptive_radio/registration/source/'+n:h for n,h in receipt['source_sha256'].items()})
    proc=subprocess.Popen(['git','archive','--format=tar',commit,*names],cwd=ROOT,stdout=subprocess.PIPE)
    actual=set()
    try:
        with tarfile.open(fileobj=proc.stdout,mode='r|') as archive:
            for item in archive:
                if item.isdir():continue
                if not item.isfile() or item.name not in names:raise RuntimeError('Unexpected published source member')
                if hashlib.sha256(archive.extractfile(item).read()).hexdigest()!=names[item.name]:
                    raise RuntimeError('Published adaptive source differs')
                actual.add(item.name)
        if proc.wait() or actual!=set(names):raise RuntimeError('Published adaptive source is incomplete')
    finally:
        proc.stdout.close()
        if proc.poll() is None:proc.terminate()
        proc.wait()
    _PUBLICATION_VERIFIED.add(key)
    return published


@contextmanager
def phase_guard(name):
    check_sources();prior=verify_previous()
    try:yield
    finally:
        check_sources()
        if verify_previous()!=prior:raise RuntimeError('Previous closure changed during phase '+name)



def authorize_stage(stage):
    if stage not in STAGES:raise ValueError('Unregistered adaptive stage')
    check_sources()
    proof={'stage':stage,'authorized':True,'registry_sha256':registry_sha256(),
        'source_receipt_sha256':digest(OUT/'registration/source.json'),
        'original_service_confirmation_permitted':False,'production_qualified':False}
    if stage=='test':
        from . import study
        gate=study.verified_development();qualification=read_json(OUT/'qualification.json')
        if (gate.get('test_permitted') is not True or qualification.get('test_permitted') is not True
            or qualification.get('source_receipt_sha256')!=proof['source_receipt_sha256']
            or qualification.get('registry_sha256')!=proof['registry_sha256']
            or qualification.get('development_sha256')!=digest(OUT/'development.json')
            or qualification.get('original_service_confirmation_permitted') is not False
            or qualification.get('production_qualified') is not False):raise RuntimeError('Fresh TEST requires the unwaived source-bound development gate')
        proof['test_permitted']=True
    return proof


class EvidenceGateCache:
    """Source-published byte-checked permission reuse, built before any science.

    An immutable receipt binds the result of the original full raw/numerical
    development check. Every hit checks source/capsule and exact dependency
    bytes/membership, including its own receipt. It never grants SERVICE or
    production permission. Finalization repeats the full algorithm and freezes
    operational logging before the numerical audit and byte closure.
    """
    def __init__(self,original,dependencies,*,name='development'):
        import os,time
        if not re.fullmatch('[a-z_]+',name):raise ValueError('Unsafe gate cache name')
        self.original=self._original_callable=original
        self.dependencies=self._dependency_callable=dependencies;self.name=name
        self.path=OUT/'gates'/f'{name}-cache.json'
        self.cache_id=f'{os.getpid()}-{time.time_ns()}'
        self._ready=False;self._logging_frozen=False;self._pid=os.getpid()
        self._hits=0;self._log_sha=None

    @staticmethod
    def _function(function):
        import marshal
        code=getattr(function,'__code__',None)
        if code is None:raise RuntimeError('Full gate must be a source-backed function')
        from types import CodeType
        absolute=Path(code.co_filename).resolve()
        filename=str(absolute.relative_to(ROOT.resolve())) if absolute.is_relative_to(ROOT.resolve()) else str(absolute)
        def normalized(value):
            if not isinstance(value,CodeType):return value
            return value.replace(co_filename=filename,co_consts=tuple(normalized(v) for v in value.co_consts))
        return {'module':function.__module__,'qualname':function.__qualname__,
            'file':filename,'code_sha256':hashlib.sha256(marshal.dumps(normalized(code))).hexdigest()}


    @staticmethod
    def _permissions(gate):
        if (not isinstance(gate,dict) or gate.get('test_permitted') is not True
            or gate.get('original_service_confirmation_permitted') is not False
            or gate.get('production_qualified') is not False):
            raise RuntimeError('Original gate must permit TEST and explicitly reject SERVICE/production')

    def bootstrap(self):
        from copy import deepcopy
        if self._ready:raise RuntimeError('Bootstrap gate once per cache instance')
        check_sources();before=self.dependencies();fingerprint=self._function(self.original)
        dependency_function=self._function(self.dependencies)
        if self.original is not self._original_callable:raise RuntimeError('Original gate callable changed')
        gate=self.original();self._permissions(gate)
        check_sources();after=self.dependencies()
        if before!=after or fingerprint!=self._function(self.original) or self.dependencies is not self._dependency_callable:raise RuntimeError('Gate dependencies changed during full replay')
        self._gate=deepcopy(gate);self._bound=deepcopy(before);self._fingerprint=fingerprint;self._dependency_fingerprint=dependency_function
        value={'schema':1,'gate':gate,'gate_sha256':value_digest(gate),
            'dependency_files_sha256':before,'dependency_set_sha256':value_digest(before),
            'original_callable':fingerprint,'dependency_callable':dependency_function,'original_full_raw_numerical_verification':True,
            'source_receipt_sha256':digest(OUT/'registration/source.json'),
            'original_service_confirmation_permitted':False,'production_qualified':False}
        write_once(self.path,value);self._receipt_sha=digest(self.path);self._ready=True
        final_path=OUT/'gates'/f'{self.name}-final.json'
        if final_path.exists():
            if read_json(final_path)!=self._final_value():raise RuntimeError('Existing final gate replay receipt differs')
            self._final_sha=digest(final_path);self._logging_frozen=True
        return deepcopy(gate)

    def _final_value(self):
        return {'schema':1,'original_full_raw_numerical_verification':True,
            'cache_receipt_sha256':self._receipt_sha,'gate_sha256':value_digest(self._gate),
            'dependency_set_sha256':value_digest(self._bound),'logging_frozen_before_final_audit_and_seal':True,
            'original_service_confirmation_permitted':False,'production_qualified':False}

    def _check(self):
        check_sources()
        if self._logging_frozen and digest(OUT/'gates'/f'{self.name}-final.json')!=self._final_sha:raise RuntimeError('Final gate replay receipt changed')
        value=read_json(self.path);self._permissions(value.get('gate'));self._permissions(self._gate)
        if (digest(self.path)!=self._receipt_sha or value['gate']!=self._gate
            or value['gate_sha256']!=value_digest(self._gate)
            or value['dependency_files_sha256']!=self._bound
            or value['dependency_set_sha256']!=value_digest(self._bound)
            or self.dependencies is not self._dependency_callable
            or self._function(self.dependencies)!=self._dependency_fingerprint
            or value.get('dependency_callable')!=self._dependency_fingerprint
            or self.dependencies()!=self._bound or value['original_callable']!=self._fingerprint
            or self.original is not self._original_callable or self._function(self.original)!=self._fingerprint):
            raise RuntimeError('Cached permission source, dependency bytes/membership or immutable bootstrap changed')

    def _log(self,full=False):
        import os
        if self._logging_frozen:return
        pid=os.getpid()
        if pid!=self._pid:self._pid=pid;self._hits=0;self._log_sha=None
        path=OUT/'gates'/f'{self.name}-{self.cache_id}-pid{pid}.jsonl'
        if path.exists():
            if self._log_sha is None or digest(path)!=self._log_sha:raise RuntimeError('Gate hit log changed')
        elif self._log_sha is not None:raise RuntimeError('Gate hit log disappeared')
        self._hits+=1
        value={'schema':1,'pid':pid,'hit':self._hits,'cache_receipt_sha256':self._receipt_sha,
            'source_and_all_dependency_bytes_membership_checked':True,'full_numerical_replay':full,
            'original_service_confirmation_permitted':False,'production_qualified':False}
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('ab') as stream:
            stream.write(json.dumps(value,sort_keys=True,allow_nan=False).encode()+b'\n');stream.flush();os.fsync(stream.fileno())
        self._log_sha=digest(path)

    def __call__(self):
        from copy import deepcopy
        if not self._ready:return self.bootstrap()
        self._check();self._log();return deepcopy(self._gate)

    def finalize(self):
        if not self._ready:raise RuntimeError('Bootstrap before final full gate replay')
        if self._logging_frozen:raise RuntimeError('Final full replay already recorded')
        self._check();before=self.dependencies();gate=self.original();self._permissions(gate)
        self._check()
        if gate!=self._gate or self.dependencies()!=before:raise RuntimeError('Final original full gate/dependencies changed')
        self._log(full=True)
        final_path=OUT/'gates'/f'{self.name}-final.json'
        write_once(final_path,self._final_value());self._final_sha=digest(final_path)
        self._logging_frozen=True
