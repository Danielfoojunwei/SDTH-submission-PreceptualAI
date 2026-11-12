import hashlib,json,os,re,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
root=Path('/workspace/SDTH-submission-PreceptualAI')
out=root/'artifacts/nested_radio_review'
out.mkdir(parents=True,exist_ok=True)
tracked=subprocess.check_output(['git','ls-files','--','src','tests','scripts','pyproject.toml','uv.lock'],cwd=root,text=True).splitlines()
source={name for name in tracked if name.endswith('.py') or name in ('pyproject.toml','uv.lock')}
source|={str(p.relative_to(root)) for top in ('src/wirelessjev/nested_radio','tests/nested_radio') for p in (root/top).rglob('*.py')}
source.add('scripts/run_nested_radio.py')
def hashes():
 return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(source)}
before=hashes()
tests=sorted({name for name in tracked if name.startswith('tests/') and Path(name).name.startswith('test_') and name.endswith('.py')}|{str(p.relative_to(root)) for p in (root/'tests/nested_radio').glob('test_*.py')})
start=time.perf_counter()
result=subprocess.run([sys.executable,'-m','pytest','--import-mode=importlib','-q',*tests],cwd=root,text=True,capture_output=True)
elapsed=time.perf_counter()-start
after=hashes()
log=out/'software_suite_importlib.log'
with log.open('x') as f:f.write(result.stdout+result.stderr)
counts=re.findall(r'(\d+) passed',result.stdout)
receipt={'schema':1,'exit_code':result.returncode,'passed':int(counts[-1]) if counts else None,
 'test_modules':tests,'source_sha256_before':before,'source_sha256_after':after,
 'source_unchanged':before==after,'subprocess_elapsed_seconds':elapsed,
 'checked_at_utc':datetime.now(timezone.utc).isoformat(),
 'log_path':str(log.relative_to(root)),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
 'deferred_untracked_observation_frontier_excluded':True,
 'no_fresh_registered_scientific_case_or_training':True}
with (out/'software_suite_importlib.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({'exit_code':result.returncode,'passed':receipt['passed'],'source_unchanged':before==after,'test_files':len(tests),'source_files':len(before),'seconds':elapsed}))
print('\n'.join((result.stdout+result.stderr).splitlines()[-12:]))
if result.returncode or before!=after:raise SystemExit(1)
