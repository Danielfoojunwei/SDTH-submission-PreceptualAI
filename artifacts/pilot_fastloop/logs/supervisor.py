"""External supervisor: execute source-frozen phases, retaining all logs."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/workspace/SDTH-submission-PreceptualAI')
folder = Path('/workspace/.cloud-setup')
status_path = folder / 'pilot-fastloop-pipeline.json'
source_commit = json.loads((root / 'artifacts/pilot_fastloop/registration/git.json').read_text())['scientific_source_commit']

def status(**values):
    values.update(time_unix=time.time(), supervisor_pid=os.getpid(), scientific_source_commit=source_commit)
    temporary = status_path.with_suffix('.part')
    temporary.write_text(json.dumps(values, indent=2) + '\n')
    os.replace(temporary, status_path)
    print(json.dumps(values), flush=True)

env = os.environ.copy()
env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           PYTHONUNBUFFERED='1', PYTHONPATH=str(root / 'src'))
launch = {'scientific_source_commit': source_commit, 'workers': 2,
    'module': 'wirelessjev.pilot_fastloop.study',
    'phase_order': ['development', 'confirmation', 'report'],
    'thread_limits': {key: env[key] for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
    'recorded_before_first_phase_unix': time.time()}
launch_path = root / 'artifacts/pilot_fastloop/logs/launch.json'
launch_path.parent.mkdir(parents=True, exist_ok=True)
if launch_path.exists():
    raise RuntimeError('Existing launch receipt; inspect status before resuming, never duplicate a running pipeline')
launch_path.write_text(json.dumps(launch, indent=2) + '\n')
(root / 'artifacts/pilot_fastloop/logs/supervisor.py').write_bytes(Path(__file__).read_bytes())
for phase in ('development', 'confirmation', 'report'):
    with (folder / f'pilot-fastloop-{phase}.log').open('a') as stream:
        command = ['uv', 'run', '--frozen', 'python', '-u', '-m',
                   'wirelessjev.pilot_fastloop.study', phase, '--workers', '2']
        child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT)
        status(stage='running', phase=phase, phase_pid=child.pid)
        code = child.wait()
    if code:
        status(stage='stopped', failed_phase=phase, exit_code=code)
        raise SystemExit(code)
status(stage='complete')
