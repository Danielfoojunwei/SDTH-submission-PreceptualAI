"""Resume frozen scientific phases with independently published guard caching."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/workspace/SDTH-submission-PreceptualAI')
folder = Path('/workspace/.cloud-setup')
out = root / 'artifacts/pilot_fastloop'
resume_path = out / 'logs/operational-resume.json'
resume = json.loads(resume_path.read_text())
source_commit = resume['scientific_source_commit']
operation_commit = resume['operational_source_commit']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def status(**values):
    values.update(time_unix=time.time(), supervisor_pid=os.getpid(),
                  scientific_source_commit=source_commit,
                  operational_source_commit=operation_commit,
                  operational_guard_cache=True)
    path = folder / 'pilot-fastloop-pipeline.json'
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(values, indent=2) + '\n')
    os.replace(temporary, path)
    print(json.dumps(values), flush=True)

lock = (folder / 'pilot-fastloop-resume-supervisor.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert digest(root / resume['wrapper_path']) == resume['wrapper_sha256']
assert digest(__file__) == resume['resume_supervisor_sha256']
assert resume['source_published_before_resume'] is True
launch_path = out / 'logs/resume-launch.json'
assert not launch_path.exists(), 'Resume already launched; inspect the existing status'
env = os.environ.copy()
env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           PYTHONUNBUFFERED='1', PYTHONPATH=str(root / 'src'))
launch = {'scientific_source_commit': source_commit,
          'operational_source_commit': operation_commit,
          'operational_resume_sha256': digest(resume_path),
          'wrapper_path': resume['wrapper_path'],
          'wrapper_sha256': resume['wrapper_sha256'],
          'module': 'wirelessjev.pilot_fastloop.study',
          'phase_order': ['confirmation', 'report'], 'workers': 2,
          'thread_limits': {name: env[name] for name in
              ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
          'recorded_before_first_resumed_phase_unix': time.time(),
          'resume_supervisor_sha256': digest(__file__)}
launch_path.write_text(json.dumps(launch, indent=2) + '\n')
for phase in ('confirmation', 'report'):
    command = ['uv', 'run', '--frozen', 'python', '-u',
               resume['wrapper_path'], phase, '--workers', '2']
    with (folder / f'pilot-fastloop-{phase}.log').open('a') as stream:
        child = subprocess.Popen(command, cwd=root, env=env,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        status(stage='running', phase=phase, phase_pid=child.pid, command=command)
        code = child.wait()
    if code:
        status(stage='stopped', failed_phase=phase, exit_code=code)
        raise SystemExit(code)
status(stage='complete')
