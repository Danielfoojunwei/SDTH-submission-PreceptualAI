"""Source/publication guards for an independent spectral-field experiment.

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
OUT = ROOT/'artifacts/radiojepa_spectral'
WORK = ROOT.parent/'training/radiojepa-spectral'
PROTOCOL = OUT/'protocol.json'
REGISTRY = OUT/'field_registry.json'
PREVIOUS_MANIFEST = ROOT/'artifacts/radiojepa/verification/manifest.json'
PREVIOUS_MANIFEST_SHA256 = 'b9114d4cf4beb65ddd4af63c87e7e154edba5c8f1bba4a56117e1e3f369acf31'
COUNTS = {'training':48,'validation':8,'development':4,'field_test':24}
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
        raise RuntimeError('Unsafe relative spectral evidence path')
    full=root/relative
    if not full.resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Spectral evidence escaped its root')
    for p in (full,*full.parents):
        if p==root:break
        if p.is_symlink():raise RuntimeError('Redirected spectral evidence member')
    return full


def write_bytes_once(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open('xb') as stream:
            stream.write(data);stream.flush()
            import os
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes()!=data:raise RuntimeError('Refuse to rewrite spectral evidence: '+str(path))
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
        except BlockingIOError as exc:raise RuntimeError('Spectral study is already active') from exc
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def previous_manifest():
    if digest(PREVIOUS_MANIFEST)!=PREVIOUS_MANIFEST_SHA256:
        raise RuntimeError('The completed RadioJEPA manifest changed')
    value=read_json(PREVIOUS_MANIFEST)
    if len(value['source_sha256'])!=452 or len(value['evidence_sha256'])!=864:
        raise RuntimeError('The completed RadioJEPA closure identity changed')
    return value


def verify_previous():
    previous_manifest()
    from ..radiojepa.verification import verify_closure
    value=verify_closure()
    if value.get('passes') is not True:
        raise RuntimeError('Previous RadioJEPA closure failed')
    original=ROOT/'artifacts/radiojepa/rt_data'
    if (original/'confirmation').exists() or (original/'confirmation.json').exists():
        raise RuntimeError('Original SERVICE confirmation geometries must stay unopened')
    return {'manifest_sha256':PREVIOUS_MANIFEST_SHA256,'sources':452,'evidence':864,
            'original_service_confirmation_opened':False}


def _spec(stage,index):
    from ..radiojepa.rt_data import GeometrySpec
    import numpy as np
    if stage not in COUNTS or type(index) is not int or not 0<=index<COUNTS[stage]:
        raise ValueError('Unregistered spectral geometry')
    stage_index=STAGES.index(stage)
    rng=np.random.default_rng(np.random.SeedSequence([2179000,stage_index,index]))
    w=float(rng.uniform(7.5,18.5));d=float(rng.uniform(9.5,23.5));h=float(rng.uniform(3.6,5.5))
    obstacle=float(min(h-.5,rng.uniform(2.0,3.2)))
    # Actual continuous geometry differences, not renamed seeds of old CAD.
    aps=((-float(rng.uniform(.27,.37))*w,-.34*d,h-.55),
         (float(rng.uniform(.27,.37))*w,.34*d,h-.55))
    poses=tuple((x*w,y*d,1.1) for y in (-.35,-.12,.13,.34) for x in (-.34,-.01,.33))
    name=f'spectral-field-{index:02d}' if stage=='field_test' else f'spectral-{stage}-{index:02d}'
    material=('metal','wood','brick')[(index+stage_index)%3]
    return GeometrySpec(name,('rack_aisles','partitioned_room')[index%2],w,d,h,obstacle,
                        material,aps,poses,2170000+stage_index*1000+index)


def field_registry():
    """Public specifications only; no solver import, field, measurement or fit."""
    from ..radiojepa import rt_data as rt
    old=rt.geometry_registry()
    def physical(row):
        return rt.fingerprint({k:v for k,v in row.items() if k not in ('scene_id','seed','spec_sha256','obstacle_material')})
    old_hashes={physical(row) for rows in old['stages'].values() for row in rows}
    stages={stage:[spec.to_dict()|{'spec_sha256':spec.sha256}
                   for spec in (_spec(stage,i) for i in range(count))] for stage,count in COUNTS.items()}
    new=[physical(row) for rows in stages.values() for row in rows]
    if len(new)!=84 or len(set(new))!=84 or set(new)&old_hashes:
        raise RuntimeError('New physical geometries overlap old or another new split')
    return {'schema':1,'role':'independent spectral RF field study, not robot SERVICE confirmation',
        'stages':stages,'counts':COUNTS,'bands_hz':list(rt.BANDS_HZ),
        'bandwidth_hz':rt.BANDWIDTH_HZ,'reference_tx_dbm':rt.REFERENCE_TX_DBM,
        'all_new_geometries_disjoint_from_old36':True,
        'original_service_confirmation_geometries_consumed':0,
        'trained_bands':[0,1],'heldout_band':2,
        'training_device_ids':[p.device_id for p in rt.DEVICE_PROFILES if not p.held_out],
        'heldout_device_ids':[p.device_id for p in rt.DEVICE_PROFILES if p.held_out]}


def write_field_registry():
    return write_once(REGISTRY,field_registry())


def field_spec(stage,index):
    return _spec(stage,index)


def registry_sha256():
    expected=field_registry()
    if read_json(REGISTRY)!=json.loads(encoded(expected)):
        raise RuntimeError('Registered spectral geometry differs from fixed specifications')
    return value_digest(expected)


def rt_scene_path(stage,index):
    spec=field_spec(stage,index)
    if stage=='field_test':authorize_stage(stage)
    return OUT/'rt_data'/stage/spec.scene_id


def source_hashes():
    m=previous_manifest()
    result=dict(m['source_sha256'])
    result.update(m['documents_sha256'])
    paths=[p for p in (ROOT/'src/wirelessjev/spectral_scene').rglob('*')
           if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.lock')]
    paths+=list((ROOT/'tests').glob('test_spectral*.py'))
    paths+=list((ROOT/'scripts').glob('*spectral_scene*.py'))
    paths+=[ROOT/'docs/spectral_scene/PLAN.md',PROTOCOL,REGISTRY,OUT/'execution_order.json',PREVIOUS_MANIFEST]
    if (ROOT/'docs/spectral_scene/BASELINES.md').exists():paths.append(ROOT/'docs/spectral_scene/BASELINES.md')
    if (OUT/'data_registry.json').exists():paths.append(OUT/'data_registry.json')
    for p in sorted(set(paths)):
        name=str(p.relative_to(ROOT));result[name]=digest(safe_path(name))
    return dict(sorted(result.items()))


def register():
    with lock(OUT/'registration.lock'):
        if (OUT/'registration/source.json').exists():return check_sources(require_published=False)
        prior=verify_previous()
        for name in ('data','fits','checkpoints','predictions','probes','baselines','rt_data',
                     'results','verification','development.json','qualification.json','selection.json'):
            p=OUT/name
            if p.is_file() or (p.is_dir() and any(p.rglob('*'))):
                raise RuntimeError('Register spectral source before scientific outputs: '+name)
        registry_sha256()
        sources=source_hashes()
        tests_path=OUT/'preflight/tests.json';tests=read_json(tests_path)
        if (tests.get('exit_code')!=0 or tests.get('sources_before_sha256')!=sources
                or tests.get('sources_after_sha256')!=sources or tests.get('prior_evidence')!=prior):
            raise RuntimeError('Final tests must pass on exactly the spectral source being registered')
        stdout=safe_path(tests['stdout_path'])
        if digest(stdout)!=tests['stdout_sha256']:raise RuntimeError('Final spectral test log changed')
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
        if digest(safe_path(name))!=expected:raise RuntimeError('Published spectral test proof changed')
    if (receipt['source_sha256']!=sources or receipt['protocol_sha256']!=digest(PROTOCOL)
            or receipt['registry_sha256']!=registry_sha256()):
        raise RuntimeError('Registered spectral source or specifications changed')
    capsule=OUT/'registration/source'
    actual={str(p.relative_to(capsule)) for p in capsule.rglob('*') if p.is_file()}
    if actual!=set(sources):raise RuntimeError('Spectral source capsule membership changed')
    for name,expected in sources.items():
        if digest(safe_path(name,capsule))!=expected:raise RuntimeError('Spectral source capsule bytes changed')
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
        raise RuntimeError('Publish and remotely verify immutable spectral source before science')
    key=value_digest(published)
    if key in _PUBLICATION_VERIFIED:return published
    names=dict(receipt['source_sha256']);names.update(receipt['preflight_sha256'])
    names['artifacts/radiojepa_spectral/registration/source.json']=published['source_receipt_sha256']
    names.update({'artifacts/radiojepa_spectral/registration/source/'+n:h for n,h in receipt['source_sha256'].items()})
    proc=subprocess.Popen(['git','archive','--format=tar',commit,*names],cwd=ROOT,stdout=subprocess.PIPE)
    actual=set()
    try:
        with tarfile.open(fileobj=proc.stdout,mode='r|') as archive:
            for item in archive:
                if item.isdir():continue
                if not item.isfile() or item.name not in names:raise RuntimeError('Unexpected published source member')
                if hashlib.sha256(archive.extractfile(item).read()).hexdigest()!=names[item.name]:
                    raise RuntimeError('Published spectral source differs')
                actual.add(item.name)
        if proc.wait() or actual!=set(names):raise RuntimeError('Published spectral source is incomplete')
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
    if stage not in STAGES:raise ValueError('Unregistered spectral stage')
    check_sources()
    proof={'stage':stage,'authorized':True,'registry_sha256':registry_sha256(),
           'source_receipt_sha256':digest(OUT/'registration/source.json'),
           'original_service_confirmation_permitted':False}
    if stage=='field_test':
        from . import study
        gate=study.verified_development()
        qualification=read_json(OUT/'qualification.json')
        if (gate.get('field_test_permitted') is not True
                or qualification.get('field_test_permitted') is not True
                or qualification.get('source_receipt_sha256')!=proof['source_receipt_sha256']
                or qualification.get('registry_sha256')!=proof['registry_sha256']
                or qualification.get('development_sha256')!=digest(OUT/'development.json')
                or qualification.get('original_service_confirmation_permitted') is not False):
            raise RuntimeError('New FIELD tests require the source-bound development gate')
        proof['field_test_permitted']=True
    return proof
