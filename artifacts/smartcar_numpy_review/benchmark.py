"""Fixed single-thread software benchmark of two implementations of one model.

Synthetic inputs exercise computation only. No comparative RF-accuracy, packet,
hardware or SOTA claim follows. Full conditioning/render timings include all
three Newton steps; observations are never replaced by prior predictions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))


def case(sites):
    import numpy as np
    from wirelessjev.smartcar_numpy.geometry import link_geometry_features, query_physics_basis
    rng = np.random.default_rng(328701+sites)
    ap = np.column_stack((np.linspace(-15, 15, sites), np.linspace(-9, 9, sites), np.full(sites, 3.)))
    boxes = np.array([[[-4., -4., 0.], [-2., 4., 2.]], [[4., -4., 0.], [6., 4., 2.]]])
    def geometry(count):
        ids = np.arange(count, dtype='i8') % sites
        rx = np.column_stack((rng.uniform(-18, 18, count), rng.uniform(-12, 12, count), np.full(count, .3)))
        hz = np.where(ids % 2 == 0, 2.4e9, 5.2e9)
        q = link_geometry_features(rx, ap[ids], hz, 0, [40., 30., 4.], boxes,
            calibration_unknown=True, quantization_width_db=1., noise_floor_dbm=-95., tx_power_dbm=20.)
        fspl = 20.-20.*np.log10(4*np.pi*hz*np.linalg.norm(rx-ap[ids], axis=-1)/299792458.)
        return ids, rx.astype('f4'), q, fspl.astype('f4')
    ids, rx, q, fspl = geometry(192)
    y = (-8.+np.sin(rx[:, 0]/7.)+np.cos(rx[:, 1]/4.)).astype('f4')
    lo, hi = y-.5, y+.5
    points = np.arange(192) % 3 == 0
    lo[points], hi[points] = y[points], y[points]
    context = np.column_stack((y/30., np.zeros(192), np.ones(192), np.zeros(192), q)).astype('f4')
    view = {'context': context, 'known': np.ones(192, dtype=bool), 'scheduled': np.ones(192, dtype=bool),
            'ap': ids, 'rx_xyz': rx, 'lower': lo, 'upper': hi, 'floor': lo,
            'fspl': fspl, 'offset': np.zeros(192, dtype='f4')}
    ids, rx, q, fspl = geometry(192)
    query = {'q': q, 'ap': ids, 'rx_xyz': rx, 'basis': query_physics_basis(q), 'fspl': fspl,
             'offset': np.zeros(192, dtype='f4')}
    return view, query, ap.astype('f4')


def worker(backend, seed, sites):
    began = time.perf_counter()
    import resource
    import numpy as np
    view, query, aps = case(sites)
    if backend == 'numpy':
        from wirelessjev.smartcar_numpy.export import load_export
        from wirelessjev.smartcar_numpy.model import NumpyField
        manifest = json.loads((ROOT/'artifacts/smartcar_numpy/models/manifest.json').read_text())
        member = manifest['members'][seed]
        arrays, _ = load_export(ROOT/'artifacts/smartcar_numpy/models'/member['path'],
                                member['sha256'], member['source_checkpoint_sha256'])
        model = NumpyField(arrays)
        def operation():
            return model.render_queries(model.encode_view(view, aps), query)
    else:
        import torch
        from wirelessjev.smartcar_runtime.field import _checkpoint
        from wirelessjev.smartcar_runtime.workflow import RELEASED_CHECKPOINTS
        torch.set_num_threads(1)
        name = f'nested_joint-seed{seed}'
        model, _ = _checkpoint(ROOT/'artifacts/nested_radio/checkpoints'/f'{name}.pt', RELEASED_CHECKPOINTS[name])
        def tensors(values):
            return {key: torch.from_numpy(np.array(value, copy=True))[None] for key, value in values.items()}
        tv, tq, ta = tensors(view), tensors(query), torch.from_numpy(aps.copy())[None]
        @torch.inference_mode()
        def operation():
            return model.render_queries(model.encode_view(tv, ta), tq)
    first_mean, first_std = operation()
    startup = time.perf_counter()-began
    if backend == 'torch':
        first_mean, first_std = first_mean[0].numpy(), first_std[0].numpy()
    for _ in range(5):
        operation()
    samples = []
    for _ in range(25):
        t = time.perf_counter()
        operation()
        samples.append(time.perf_counter()-t)
    return {'backend': backend, 'seed': seed, 'APs': sites, 'context_tokens': 192, 'queries': 192,
        'startup_to_first_result_s': startup, 'conditioning_and_render_seconds': samples,
        'median_seconds': float(np.median(samples)), 'p95_seconds': float(np.quantile(samples, .95)),
        'process_peak_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'torch_imported': 'torch' in sys.modules, 'mean': first_mean.tolist(), 'std': first_std.tolist()}


def run():
    import numpy as np
    protocol_path = ROOT/'artifacts/smartcar_numpy/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    subprocess.run(['git', 'merge-base', '--is-ancestor', protocol['source_commit'], 'HEAD'], cwd=ROOT, check=True)
    assert subprocess.check_output(['git', 'show', 'HEAD:artifacts/smartcar_numpy/protocol.json'], cwd=ROOT) == protocol_path.read_bytes()
    assert protocol['benchmark_source_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name, digest in protocol['implementation_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', PYTHONPATH=str(ROOT/'src'), PYTHONNOUSERSITE='1')
    rows = []
    for sites in (1, 4, 16):
        for seed in (0, 1, 2):
            # Fixed alternating order prevents choosing whichever appeared faster.
            order = ('numpy', 'torch') if (sites+seed) % 2 else ('torch', 'numpy')
            paired = {}
            for backend in order:
                result = subprocess.run([sys.executable, str(Path(__file__)), '--worker', backend,
                    '--seed', str(seed), '--aps', str(sites)], cwd=ROOT, env=env,
                    text=True, capture_output=True, check=True)
                row = json.loads(result.stdout)
                assert row['torch_imported'] is (backend == 'torch')
                rows.append(row)
                paired[backend] = row
            for key in ('mean', 'std'):
                np.testing.assert_allclose(paired['numpy'][key], paired['torch'][key], atol=1.e-4, rtol=1.e-6)
    result = {'schema': 1, 'passes': True, 'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'source_commit': protocol['source_commit'], 'rows': rows,
        'scope': 'Same released weights; synthetic single-process software timing, not RF accuracy or physical deployment.',
        'hardware_actions': 0, 'fits': 0, 'RT': 0, 'SOTA_established': False}
    with (OUT/'benchmark.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    pairs = {(r['APs'], r['seed'], r['backend']): r for r in rows}
    for sites in (1, 4, 16):
        speed = [pairs[sites, seed, 'torch']['median_seconds']/pairs[sites, seed, 'numpy']['median_seconds'] for seed in (0, 1, 2)]
        memory = [pairs[sites, seed, 'torch']['process_peak_RSS_KiB']/pairs[sites, seed, 'numpy']['process_peak_RSS_KiB'] for seed in (0, 1, 2)]
        print(json.dumps({'APs': sites, 'member_speed_ratios': speed, 'member_peak_memory_ratios': memory}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('numpy', 'torch'))
    parser.add_argument('--seed', type=int, choices=(0, 1, 2))
    parser.add_argument('--aps', type=int, choices=(1, 4, 16))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.seed, args.aps), allow_nan=False))
    else:
        run()
