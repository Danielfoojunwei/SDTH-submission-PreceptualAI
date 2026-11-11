"""Check exact archived study bytes in Git after publication; no scientific calls."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[2]
branch='codex/hierarchical-radio-posterior'
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=root,text=True).split()[0]
assert remote==head
entries={}
for record in subprocess.check_output(['git','ls-files','-s','-z'],cwd=root).split(b'\0'):
    if not record:continue
    info,name=record.split(b'\t',1)
    mode,blob,stage=info.decode().split()
    assert stage=='0'
    entries[name.decode()]=blob
folder=root/'artifacts/hierarchical_radio'
files=[p for p in folder.rglob('*') if p.is_file()]
assert len(files)==5468
sizes=0
for p in files:
    name=str(p.relative_to(root));size=p.stat().st_size
    digest=hashlib.sha1(b'blob '+str(size).encode()+b'\0')
    with p.open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):digest.update(block)
    assert entries.get(name)==digest.hexdigest(),name
    sizes+=size
proof={'schema':1,'passes':True,'checked_at_utc':datetime.now(timezone.utc).isoformat(),
    'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'science_commit':head,'remote_observed_commit':remote,'branch':branch,
    'study_files_in_Git_matching_exact_bytes':len(files),'study_bytes':sizes,
    'scientific_calls':0,'scope':'Study archives including source capsule; final documentation commit can follow without changing these bytes.'}
with (root/'artifacts/hierarchical_radio_review/git_artifact_verification.json').open('x') as handle:
    json.dump(proof,handle,sort_keys=True,indent=2);handle.write('\n')
print(json.dumps({'passes':True,'published_study_files':len(files),'bytes':sizes,'science_commit':head}))
