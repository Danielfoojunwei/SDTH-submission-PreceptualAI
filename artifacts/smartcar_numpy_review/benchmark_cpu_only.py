"""Predeclared follow-up against the matching CPU-only Torch wheel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def run(python):
    import numpy as np
    path = ROOT/'artifacts/smartcar_numpy/cpu_only_protocol.json'
    protocol = json.loads(path.read_bytes())
    assert subprocess.check_output(['git', 'show', 'HEAD:artifacts/smartcar_numpy/cpu_only_protocol.json'], cwd=ROOT) == path.read_bytes()
    worker = OUT/'benchmark.py'
    assert hashlib.sha256(worker.read_bytes()).hexdigest() == protocol['worker_source_sha256']
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == protocol['runner_source_sha256']
    for name, expected in protocol['implementation_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', PYTHONPATH=str(ROOT/'src'), PYTHONNOUSERSITE='1')
    version = subprocess.run([python, '-c', 'import json,torch,numpy,scipy;print(json.dumps({"torch":torch.__version__,"numpy":numpy.__version__,"scipy":scipy.__version__}))'],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    versions = json.loads(version.stdout)
    assert versions == protocol['required_versions']
    rows = []
    for sites in (1, 4, 16):
        for seed in (0, 1, 2):
            pair = {}
            for backend in (('numpy', 'torch') if (sites+seed) % 2 else ('torch', 'numpy')):
                response = subprocess.run([python, str(worker), '--worker', backend, '--seed', str(seed), '--aps', str(sites)],
                    cwd=ROOT, env=env, capture_output=True, text=True, check=True)
                value = json.loads(response.stdout)
                assert value['torch_imported'] is (backend == 'torch')
                pair[backend] = value
                rows.append(value)
            for key in ('mean', 'std'):
                np.testing.assert_allclose(pair['numpy'][key], pair['torch'][key], atol=1.e-4, rtol=1.e-6)
    report = {'schema': 1, 'passes': True, 'versions': versions, 'rows': rows,
        'protocol_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'scope': 'Fixed same-weight software comparison against CPU-only Torch; no RF/hardware/SOTA claim.',
        'hardware_actions': 0, 'fits': 0, 'RT': 0, 'SOTA_established': False}
    with (OUT/'benchmark_cpu_only.json').open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    pairs = {(r['APs'], r['seed'], r['backend']): r for r in rows}
    for sites in (1, 4, 16):
        print(json.dumps({'APs': sites,
            'member_speed_ratios': [pairs[sites, seed, 'torch']['median_seconds']/pairs[sites, seed, 'numpy']['median_seconds'] for seed in (0, 1, 2)],
            'member_peak_memory_ratios': [pairs[sites, seed, 'torch']['process_peak_RSS_KiB']/pairs[sites, seed, 'numpy']['process_peak_RSS_KiB'] for seed in (0, 1, 2)]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True)
    run(parser.parse_args().python)
