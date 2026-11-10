#!/usr/bin/env python3
"""Published operational cache for unchanged spectral FIELD gate arithmetic.

Only study.verified_development is replaced at runtime. First use and the final
check before the passive audit run its original full algorithm. Every reuse validates the
exact source/capsule/development dependency bytes and membership. No timing,
training, inference, RT, data-budget or endpoint change is authorized here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import marshal
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

SOURCE_COUNT=475
WRAPPER='scripts/run_scene_field_cached.py'
TEST='tests/test_field_ops_cache.py'
MANDATORY=('registration/source.json','registration/git.json','protocol.json','field_registry.json',
    'execution_order.json','selection.json','development.json','qualification.json',
    'rt_data/training.json','rt_data/validation.json','rt_data/development.json',
    'rt_data/field_test.json','operations/field-cache-handoff.json',
    'results/development.json','diagnostics/development-same-link-frequency.json',
    'diagnostics/development-same-link-frequency.npz')
DIRECTORIES=('fits','checkpoints','baselines','rt_data/training','rt_data/validation',
    'rt_data/development','rt_data/field_test','predictions/development')


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
    return value.hexdigest()


def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def value_sha(value):return hashlib.sha256(encoded(value)).hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def safe_file(root,name):
    root=Path(root).resolve();relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:raise RuntimeError('Unsafe operational proof path')
    path=root/relative
    if not path.is_file() or not path.resolve().is_relative_to(root):raise RuntimeError('Missing operational dependency')
    for p in (path,*path.parents):
        if p==root:break
        if p.is_symlink():raise RuntimeError('Redirected operational dependency')
    return path


def file_map(root,paths):
    root=Path(root).resolve();result={}
    for path in sorted(set(map(Path,paths))):
        relative=str(path.relative_to(root));safe_file(root,relative)
        result[relative]=sha256(path)
    return result


def directory_files(path):
    path=Path(path)
    if not path.is_dir() or path.is_symlink():raise RuntimeError('Missing or redirected gate namespace')
    result=[]
    for p in path.rglob('*'):
        if p.is_symlink():raise RuntimeError('Redirected gate namespace member')
        if p.is_file():result.append(p)
    if not result:raise RuntimeError('Empty gate dependency namespace')
    return result


def verify_ops_publication(root,amendment_path,publication_path):
    """Verify published wrapper/tests/preflight/amendment without network or secrets."""
    root=Path(root).resolve();amendment_path=safe_file(root,Path(amendment_path).relative_to(root))
    publication_path=safe_file(root,Path(publication_path).relative_to(root))
    amendment=read_json(amendment_path);published=read_json(publication_path)
    expected={'runtime_override':'wirelessjev.spectral_scene.study.verified_development',
        'scientific_source_changed':False,'new_training_or_model_fits':False,
        'new_scientific_endpoints':False,'timing_claim':False,
        'original_full_gate_before_reuse':True,'original_full_gate_at_final_audit':True,
        'source_files_count':SOURCE_COUNT}
    if any(type(amendment.get(k)) is not type(v) or amendment.get(k)!=v for k,v in expected.items()):raise RuntimeError('Operational amendment exceeds permitted cache scope')
    members=amendment.get('files_sha256',{})
    if WRAPPER not in members or TEST not in members:raise RuntimeError('Amendment must bind exact wrapper and tests')
    if members[WRAPPER]!=sha256(safe_file(root,WRAPPER)) or members[TEST]!=sha256(safe_file(root,TEST)):
        raise RuntimeError('Operational wrapper or tests changed')
    for name,expected_sha in members.items():
        if sha256(safe_file(root,name))!=expected_sha:raise RuntimeError('Amendment dependency changed')
    preflight=read_json(safe_file(root,amendment['preflight_path']))
    if (amendment['preflight_path'] not in members or preflight.get('exit_code')!=0
        or preflight.get('wrapper_sha256')!=members[WRAPPER] or preflight.get('test_sha256')!=members[TEST]
        or preflight.get('stdout_path') not in members
        or preflight.get('stdout_sha256')!=members[preflight['stdout_path']]):
        raise RuntimeError('Passing tests must bind exact operational source and stdout')
    commit=published.get('commit')
    if (not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit)
        or published.get('remote_observed_commit')!=commit or not published.get('branch') or not published.get('remote')
        or published.get('amendment_sha256')!=sha256(amendment_path)):
        raise RuntimeError('Operational amendment must be published and remotely verified before installation')
    expected_members=dict(members);expected_members[str(amendment_path.relative_to(root))]=sha256(amendment_path)
    proc=subprocess.Popen(['git','archive','--format=tar',commit,*expected_members],cwd=root,stdout=subprocess.PIPE)
    actual=set()
    try:
        with tarfile.open(fileobj=proc.stdout,mode='r|') as archive:
            for item in archive:
                if item.isdir():continue
                if not item.isfile() or item.name not in expected_members:raise RuntimeError('Unexpected operational publication member')
                if hashlib.sha256(archive.extractfile(item).read()).hexdigest()!=expected_members[item.name]:raise RuntimeError('Published operational bytes differ')
                actual.add(item.name)
        if proc.wait() or actual!=set(expected_members):raise RuntimeError('Incomplete operational publication')
    finally:
        proc.stdout.close()
        if proc.poll() is None:proc.terminate()
        proc.wait()
    return {'amendment':amendment,'publication':published,
        'amendment_path':str(amendment_path.relative_to(root)),
        'publication_path':str(publication_path.relative_to(root)),
        'proof_files_sha256':file_map(root,[safe_file(root,n) for n in members]+[amendment_path,publication_path])}



def gate_permissions(gate):
    if (not isinstance(gate,dict) or gate.get('field_test_permitted') is not True
        or gate.get('original_service_confirmation_permitted') is not False
        or gate.get('production_qualified') is not False):
        raise RuntimeError('Original gate must permit FIELD and explicitly reject SERVICE/production')


def callable_fingerprint(function):
    code=getattr(function,'__code__',None)
    if code is None:raise RuntimeError('Original full gate must be a source-backed Python function')
    return {'module':function.__module__,'qualname':function.__qualname__,
        'code_file':str(Path(code.co_filename).resolve()),
        'code_sha256':hashlib.sha256(marshal.dumps(code)).hexdigest()}


def prepare_handoff(common,proof):
    """Write-once admission proof; later resumes require its exact original bytes."""
    from wirelessjev.radiojepa import rt_data
    common.check_sources();root,out=Path(common.ROOT).resolve(),Path(common.OUT).resolve()
    for name,expected in proof['proof_files_sha256'].items():
        if sha256(safe_file(root,name))!=expected:raise RuntimeError('Published operations proof changed before handoff')
    source_sha=sha256(out/'registration/source.json')
    if source_sha!=proof['amendment']['scientific_source_receipt_sha256']:raise RuntimeError('Handoff source differs from published amendment')
    collector_path=out/'rt_data/field_test.json';collector=read_json(collector_path)
    if (collector.get('completed') is not True or collector.get('stage')!='field_test'
        or collector.get('source_receipt_sha256')!=source_sha
        or collector.get('registry_sha256')!=common.registry_sha256()
        or collector.get('real_device_measurements') is not False
        or collector.get('original_service_confirmation_consumed') is not False):raise RuntimeError('Incomplete or mismatched FIELD collector')
    receipts={}
    for index in range(24):
        spec=common.field_spec('field_test',index);folder=out/'rt_data/field_test'/spec.scene_id
        rt_data.validate_scene_receipt(spec,folder,'field_test')
        path=folder/'receipt.json'
        ref={'path':str(path.relative_to(root)),'sha256':sha256(path)}
        if collector['scene_receipts'].get(spec.scene_id)!=ref:raise RuntimeError('FIELD collector scene receipt changed')
        receipts[spec.scene_id]=ref
    if set(collector['scene_receipts'])!=set(receipts):raise RuntimeError('FIELD collector must contain exact24 complete scenes')
    value={'schema':1,'scientific_source_receipt_sha256':source_sha,'scientific_source_files':SOURCE_COUNT,
        'operations_amendment_sha256':sha256(safe_file(root,proof['amendment_path'])),
        'operations_publication_sha256':sha256(safe_file(root,proof['publication_path'])),
        'operations_proof_sha256':value_sha(proof),'field_collector_sha256':sha256(collector_path),
        'complete_scene_receipts':receipts,'field_prediction_files_at_first_entry':0,
        'field_results_at_first_entry':False,'original_resume_methods_only':True,
        'service_confirmation_permitted':False,'production_qualified':False}
    path=out/'operations/field-cache-handoff.json'
    if path.exists():
        if path.is_symlink() or read_json(path)!=value:raise RuntimeError('Existing handoff source/proof/collector identity changed')
    else:
        predictions=out/'predictions/field_test'
        if ((predictions.exists() and any(p.is_file() or p.is_symlink() for p in predictions.rglob('*')))
            or (out/'results/field_test.json').exists() or (out/'report.json').exists()):
            raise RuntimeError('Unproven existing FIELD outputs require a prior exact handoff')
        write_once(path,value)
    return value

def bound_files(common,ops_proof):
    """Exact gate dependencies; future FIELD outputs and cache logs are excluded."""
    root,out=Path(common.ROOT).resolve(),Path(common.OUT).resolve()
    receipt=read_json(out/'registration/source.json');sources=receipt['source_sha256']
    if len(sources)!=SOURCE_COUNT:raise RuntimeError('Expected exact475 scientific source files')
    if sha256(out/'registration/source.json')!=ops_proof['amendment']['scientific_source_receipt_sha256']:
        raise RuntimeError('Amendment belongs to another scientific source registration')
    paths=[out/name for name in MANDATORY]
    paths += [safe_file(root,name) for name in sources]
    capsule=out/'registration/source';capsule_files=directory_files(capsule)
    if {str(p.relative_to(capsule)) for p in capsule_files}!=set(sources):raise RuntimeError('Scientific source capsule membership changed')
    paths+=capsule_files
    for name,expected in receipt['preflight_sha256'].items():
        path=safe_file(root,name)
        if sha256(path)!=expected:raise RuntimeError('Scientific passing preflight changed')
        paths.append(path)
    for name in DIRECTORIES:paths+=directory_files(out/name)
    data=out/'data'
    for path in directory_files(data):
        # Only new FIELD public receipts may appear while the frozen phase runs.
        if not path.name.startswith('scene-field_test-'):paths.append(path)
    paths += [safe_file(root,name) for name in ops_proof['proof_files_sha256']]
    current=file_map(root,paths)
    for name,expected in ops_proof['proof_files_sha256'].items():
        if current.get(name)!=expected:raise RuntimeError('Operational proof changed after publication')
    return current


def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);data=encoded(value)
    try:
        with path.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes()!=data:raise RuntimeError('Operational receipt already differs')


class DevelopmentGateCache:
    def __init__(self,common,study,*,amendment_path,publication_path):
        self.common,self.study=common,study;self.original=study.verified_development
        self._original_callable=self.original
        self.amendment_path,self.publication_path=Path(amendment_path),Path(publication_path)
        self.parent_pid=os.getpid();self.cache_id=f'{time.time_ns()}-{self.parent_pid}'
        self.prefix=Path(common.OUT)/'operations'/('field-gate-cache-'+self.cache_id)
        self.bootstrap_path=Path(str(self.prefix)+'.bootstrap.json')
        self._ready=False;self._closed=False;self._pid=os.getpid();self._hits=0;self._last_log_sha=None

    def bootstrap(self):
        if self._ready or os.getpid()!=self.parent_pid:raise RuntimeError('Bootstrap exactly once in parent before fork')
        self.common.check_sources()
        self.proof=verify_ops_publication(self.common.ROOT,self.amendment_path,self.publication_path)
        prepare_handoff(self.common,self.proof)
        if self.original is not self._original_callable:raise RuntimeError('Original gate callable changed before bootstrap')
        self._original_fingerprint=callable_fingerprint(self.original)
        before=bound_files(self.common,self.proof);began=time.perf_counter()
        gate=self.original()
        self.common.check_sources();after=bound_files(self.common,self.proof)
        if before!=after:raise RuntimeError('Dependencies changed during original full development verification')
        gate_permissions(gate)
        self._bound,self._gate=before,deepcopy(gate)
        receipt={'schema':1,'cache_id':self.cache_id,'cache_parent_pid':self.parent_pid,
            'original_full_verified_development':True,'original_callable':self._original_fingerprint,
            'service_confirmation_permitted':False,'production_qualified':False,'elapsed_seconds':time.perf_counter()-began,
            'dependency_files_sha256':before,'dependency_set_sha256':value_sha(before),
            'gate':deepcopy(gate),'gate_sha256':value_sha(gate),'operations_proof':deepcopy(self.proof),
            'runtime_override':'wirelessjev.spectral_scene.study.verified_development',
            'scientific_source_changed':False,'new_training_or_model_fits':False,'timing_claim':False}
        write_once(self.bootstrap_path,receipt);self._bootstrap_sha=sha256(self.bootstrap_path);self._ready=True
        return deepcopy(gate)

    def _check(self):
        self.common.check_sources()
        snapshot=read_json(self.bootstrap_path)
        gate_permissions(snapshot.get('gate'));gate_permissions(self._gate)
        if (self.original is not self._original_callable or callable_fingerprint(self.original)!=self._original_fingerprint
            or snapshot.get('original_callable')!=self._original_fingerprint):raise RuntimeError('Original full gate callable changed')
        if (sha256(self.bootstrap_path)!=self._bootstrap_sha or snapshot['dependency_files_sha256']!=self._bound
            or snapshot['dependency_set_sha256']!=value_sha(self._bound) or snapshot['gate']!=self._gate
            or snapshot['gate_sha256']!=value_sha(self._gate) or snapshot['operations_proof']!=self.proof
            or snapshot['gate'].get('field_test_permitted') is not True
            or self.study.verified_development not in (self.original,self)):
            raise RuntimeError('Cached permission differs from immutable original bootstrap')
        if bound_files(self.common,self.proof)!=self._bound or sha256(self.bootstrap_path)!=self._bootstrap_sha:
            raise RuntimeError('Cached development gate invalid: source, bytes or membership changed')

    def _log(self,*,full=False,elapsed=None):
        pid=os.getpid()
        if pid!=self._pid:self._pid=pid;self._hits=0;self._last_log_sha=None
        path=Path(str(self.prefix)+f'.pid-{pid}.hits.jsonl')
        if path.exists():
            if self._last_log_sha is None or sha256(path)!=self._last_log_sha:raise RuntimeError('Operational hit log changed')
        elif self._last_log_sha is not None:raise RuntimeError('Operational hit log disappeared')
        self._hits+=1
        value={'schema':1,'cache_id':self.cache_id,'pid':pid,'parent_pid':os.getppid(),
            'process_hit_index':self._hits,'bootstrap_sha256':self._bootstrap_sha,
            'dependency_set_sha256':value_sha(self._bound),'gate_sha256':value_sha(self._gate),
            'source_guard_checked':True,'all_dependency_bytes_and_membership_checked':True,
            'original_full_gate_reexecuted':full,'full_gate_elapsed_seconds':elapsed,
            'final_uncached_before_original_passive_audit':full,
            'service_confirmation_permitted':False,'production_qualified':False,'timing_claim':False,'recorded_unix':time.time()}
        path.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(path,os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o644)
        try:
            payload=encoded(value).replace(b'\n',b' ')+b'\n'
            if os.write(fd,payload)!=len(payload):raise RuntimeError('Partial operations log append')
            os.fsync(fd)
        finally:os.close(fd)
        self._last_log_sha=sha256(path)

    def __call__(self):
        if not self._ready:raise RuntimeError('Original full parent bootstrap required before gate reuse')
        self._check();self._log()
        return deepcopy(self._gate)

    def final_uncached(self):
        if not self._ready or self._closed or os.getpid()!=self.parent_pid:raise RuntimeError('Final uncached verification must run once in parent')
        self._check();before=bound_files(self.common,self.proof);began=time.perf_counter()
        gate=self.original();elapsed=time.perf_counter()-began
        self._check();after=bound_files(self.common,self.proof)
        if before!=after or gate!=self._gate:raise RuntimeError('Final uncached original gate or dependencies differ')
        self._closed=True
        self._log(full=True,elapsed=elapsed)
        path=Path(str(self.prefix)+'.final-uncached.json')
        write_once(path,{'schema':1,'cache_id':self.cache_id,'bootstrap_sha256':self._bootstrap_sha,
            'original_full_verified_development':True,'before_sha256':before,'after_sha256':after,
            'unchanged':True,'gate_sha256':value_sha(gate),'elapsed_seconds':elapsed,
            'before_original_passive_audit_and_seal':True,'service_confirmation_permitted':False,
            'production_qualified':False,'scientific_source_changed':False})
        return deepcopy(gate)

    def install(self):
        if not self._ready or self.study.verified_development is not self.original:raise RuntimeError('Bootstrap original gate before installation')
        self.study.verified_development=self

    def restore(self):
        if self.study.verified_development is not self:raise RuntimeError('Unexpected additional runtime gate patch')
        self.study.verified_development=self._original_callable


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=2,choices=(2,))
    parser.add_argument('--amendment',type=Path)
    parser.add_argument('--publication',type=Path)
    args=parser.parse_args(argv)
    import multiprocessing
    if multiprocessing.get_start_method(allow_none=True) is None:multiprocessing.set_start_method('fork')
    if multiprocessing.get_start_method()!='fork':raise RuntimeError('Parent-verified cache requires fork inheritance')
    from wirelessjev.spectral_scene import common,study,verification,baselines
    from wirelessjev.radiojepa import rt_data
    if (common.OUT/'verification/manifest.json').exists():
        print(json.dumps(verification.verify_closure(),sort_keys=True,indent=2));return
    with common.lock(common.WORK/'supervisor.lock',nonblocking=True):
        common.check_sources()
        verification.verify_fits();baselines.verify_prior()
        ops=common.ROOT/'artifacts/radiojepa_spectral_ops'
        cache=DevelopmentGateCache(common,study,amendment_path=args.amendment or ops/'amendment.json',
            publication_path=args.publication or ops/'publication.json')
        cache.bootstrap();cache.install()
        try:
            # Original finite phase methods and endpoint definitions, unchanged.
            study.collect_stage('field_test',args.workers)
            study.evaluate_stage('field_test')
            study.write_field_results()
            cache.final_uncached()
            verification.run_audit()
            verification.seal()
            # No cached calls or operational writes occur after the seal snapshot.
            print(json.dumps(verification.verify_closure(),sort_keys=True,indent=2,allow_nan=False))
        finally:cache.restore()


if __name__=='__main__':main()
