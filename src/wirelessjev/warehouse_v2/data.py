"""Noisy observed histories and explicit simulated action interventions.

Hidden propagation coefficients and future outcomes never enter `inputs`.
Unreported links carry a mask and censoring bound, not an invented RSSI.
Plans are constructed from the measured scan and public robot demands.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import numpy as np

from ..envlearn import (random_env, obstruction_fn, external_noise, leak_matrix)
from ..fastplan import BatchEvaluator
from ..twin.warehouse import make_warehouse

SCHEMA = 'warehouse.observed_interventions.v2'


def make_environment(seed, n=100, domain='train', environment_seed=None):
    environment_seed = seed if environment_seed is None else environment_seed
    rng = np.random.default_rng(environment_seed + 700_000)
    base = make_warehouse(seed, n_robots=n, scanners='heavy' if domain == 'shift' else 'normal')
    spec = random_env(environment_seed)
    # Public AP positions vary; hidden attenuation is never an input feature.
    spread = 8. if domain == 'shift' else 3.
    aps = [replace(a, x=float(np.clip(a.x+rng.normal(0, spread), 0, base.width_m)),
                   y=float(np.clip(a.y+rng.normal(0, spread), 0, base.height_m))) for a in base.aps]
    if domain == 'shift':
        aps = aps[4:]  # A genuinely unseen AP count, not a padded training graph.
    base = replace(base, aps=aps, obstruction=obstruction_fn(spec), pl_coef=spec.pl_coef,
                   shadow_db=spec.shadow_db, ap_height_m=spec.ap_height,
                   external_occupancy=dict(spec.occupancy), _radio=None, _radio_key=None)
    # Initial associations use only a measured scan at this simulation boundary.
    signal = base.radio().dl + rng.normal(0, 3., (n, len(aps)))
    base = base.with_associations({c.car_id: aps[int(np.argmax(signal[i]))].ap_id
                                  for i, c in enumerate(base.cars)})
    return base, spec


def measure_scan(scene, rng, *, sigma_db=3., top_k=8):
    """Simulation sensor boundary. Returns values, masks and valid bounds."""
    measured = scene.radio().dl + rng.normal(0, sigma_db, scene.radio().dl.shape)
    order = np.argsort(-measured, axis=1)[:, :top_k]
    heard = np.zeros(measured.shape, bool)
    np.put_along_axis(heard, order, np.take_along_axis(measured >= -90., order, axis=1), axis=1)
    bound = np.where(heard.sum(1) == top_k, np.min(np.where(heard, measured, np.inf), axis=1), -90.)
    # Zero in channel 0 means "absent" only when mask=0. It is not a measurement.
    return np.stack([np.where(heard, (measured+80.)/30., 0.), heard.astype(float),
                     np.broadcast_to((bound[:, None]+80.)/30., measured.shape)], axis=-1).astype('float32')


def observed_candidates(history, current, demand, rng, k=4):
    last = history[-1]
    valid = last[:, :, 1] > 0
    signal = np.where(valid, last[:, :, 0], -1e6)
    strongest = signal.argmax(1)
    no_scan = ~valid.any(1)
    strongest[no_scan] = current[no_scan]
    balanced = current.copy()
    load = np.zeros(signal.shape[1])
    for i in np.argsort(-demand):
        visible = np.flatnonzero(valid[i])
        if not len(visible):
            visible = np.array([current[i]])
        value = signal[i, visible] - .12*load[visible]
        j = visible[int(np.argmax(value))]
        balanced[i] = j
        load[j] += demand[i]
    random_plan = current.copy()
    for i in rng.choice(len(current), max(1, len(current)//10), replace=False):
        visible = np.flatnonzero(valid[i])
        if len(visible):
            random_plan[i] = rng.choice(visible)
    return np.stack([current, strongest, balanced, random_plan])[:k]


def observed_inputs(scene, spec, history, rng):
    """Public telemetry. `spec` is used only at the simulated sensor boundary."""
    ev = BatchEvaluator(scene, [])
    ev.client_itf = True
    ev.ext_noise_mw = external_noise(scene, spec)
    ev.leak = leak_matrix(scene, spec.leak_db)
    measured_service = ev._snapshot_terms(ev.current[None], ev.snaps[0], False, 5.)['sat'][0]
    measured_service = np.clip(measured_service+rng.normal(0, .02, ev.n), 0, 1)
    next_xy = [scene.mobility.planned_positions(c, [5.])[0] for c in scene.cars]
    critical = np.array([sum(f.demand_mbps for f in c.flows if f.critical) for c in scene.cars])
    best_effort = np.array([sum(f.demand_mbps for f in c.flows if not f.critical) for c in scene.cars])
    floors = np.array([sum(f.floor_mbps for f in c.flows if f.critical) for c in scene.cars])
    video = np.array([any('video' in f.name for f in c.flows) for c in scene.cars])
    robot = np.array([[c.x/scene.width_m, c.y/scene.height_m,
                       xy[0]/scene.width_m, xy[1]/scene.height_m,
                       np.log1p(critical[i])/3, np.log1p(best_effort[i])/3,
                       floors[i]/max(critical[i], 1e-6), video[i], measured_service[i]]
                      for i, (c, xy) in enumerate(zip(scene.cars, next_xy))], dtype='float32')
    ap = np.array([[a.x/scene.width_m, a.y/scene.height_m, float(a.band == '5g'),
                    a.channel/165., np.clip(scene.external_occupancy.get((a.band, a.channel), 0.) + rng.normal(0, .02), 0, 1),
                    a.tx_power_dbm/30.] for a in scene.aps], dtype='float32')
    return {'history': np.stack(history).astype('float32'), 'robot': robot, 'ap': ap,
            'current': ev.current.astype('int64')}, critical+best_effort, video


def episode(seed, *, n=100, domain='train', steps=12, environment_seed=None):
    scene, spec = make_environment(seed, n, domain, environment_seed=environment_seed)
    observation_rng = np.random.default_rng(seed + 100_000)
    intervention_rng = np.random.default_rng(seed + 200_000)
    label_rng = np.random.default_rng(seed + 300_000)
    history, samples = [], []
    for tick in range(steps):
        history.append(measure_scan(scene, observation_rng))
        history = history[-4:]
        if len(history) < 4:
            scene = scene.advance(5.)
            continue
        inputs, demand, video = observed_inputs(scene, spec, history, observation_rng)
        A = observed_candidates(inputs['history'], inputs['current'], demand, intervention_rng)
        future = scene.advance(5.)
        truth = BatchEvaluator(future, [])
        truth.client_itf = True
        truth.ext_noise_mw = external_noise(future, spec)
        truth.leak = leak_matrix(future, spec.leak_db)
        blocked = label_rng.random(truth.snaps[0].ul.shape) < spec.blockage_p
        truth.snaps[0].ul -= blocked*spec.blockage_db
        truth.snaps[0].dl -= blocked*spec.blockage_db
        terms = truth._snapshot_terms(A, truth.snaps[0], True, 5.)
        crit_got = terms['cf']*np.maximum(0., 1.-(A != truth.current[None])*.8)
        failures = ((truth.cf_dem[None]*crit_got[:, :, None] < truth.cf_floor[None]-1e-9)
                    & (truth.cf_dem[None] > 0)).any(2)
        rssi = truth.snaps[0].dl[np.arange(n)[None], A]
        targets = np.stack([terms['sat'], failures.astype(float), (rssi+80.)/30.], -1).astype('float32')
        samples.append({'inputs': inputs, 'plans': A, 'targets': targets,
                        'demand': demand.astype('float32'), 'video': video,
                        'scene': seed, 'tick': tick, 'domain': domain,
                        'robot_ids': [c.car_id for c in scene.cars]})
        # The logged behaviour policy is seeded observed-action exploration.
        # Every candidate label is a separate intervention from the same state.
        behaviour = int(intervention_rng.integers(len(A)))
        scene = future.with_associations({c.car_id: future.aps[j].ap_id for c, j in zip(future.cars, A[behaviour])})
    return samples


def build_dataset(root: Path, *, train_scenes=24, validation_scenes=6, test_scenes=12, steps=12):
    """Cache numeric arrays only; never pickle downloaded or untrusted data."""
    root.mkdir(parents=True, exist_ok=True)
    splits = {'train': [(80_000+i, [25, 50, 100][i % 3], 'train') for i in range(train_scenes)],
              'validation': [(81_000+i, 100, 'train') for i in range(validation_scenes)],
              'test': [(82_000+i, 100, 'train') for i in range(test_scenes)],
              'shift': [(83_000+i, 100, 'shift') for i in range(6)],
              'fleet200': [(84_000+i, 200, 'shift') for i in range(6)]}
    result = {}
    for split, scenes in splits.items():
        result[split] = []
        for seed, n, domain in scenes:
            path = root / f'{seed}-{n}-{domain}-{steps}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as cache:
                    count = int(cache['count'])
                    samples = []
                    for i in range(count):
                        get = lambda key: cache[f'{i}_{key}'].copy()
                        inputs = {key: get(key) for key in ('history', 'robot', 'ap', 'current')}
                        samples.append({'inputs': inputs, 'plans': get('plans'), 'targets': get('targets'),
                                        'demand': get('demand'), 'video': get('video'), 'scene': seed,
                                        'tick': i+3, 'domain': domain,
                                        'robot_ids': [f'amr-{j+1:03d}' for j in range(n)]})
            else:
                print(f'Generating simulated interventions {split} scene={seed} robots={n}', flush=True)
                samples = episode(seed, n=n, domain=domain, steps=steps)
                arrays = {'count': np.array(len(samples))}
                for i, sample in enumerate(samples):
                    arrays.update({f'{i}_{k}': v for k, v in {**sample['inputs'], **{
                        k: sample[k] for k in ('plans', 'targets', 'demand', 'video')}}.items()})
                np.savez_compressed(path, **arrays)
            result[split].extend(samples)
    manifest = {'schema': SCHEMA, 'source': 'simulation', 'history_steps': 4,
                'step_s': 5., 'top_k': 8, 'sensor_sigma_db': 3., 'episode_steps': steps,
                'scenes': splits, 'samples': {k: len(v) for k, v in result.items()},
                'policy_inputs': ['capped_noisy_scan_history_with_masks_and_bounds',
                                 'reported_current_service', 'public_demands_and_routes',
                                 'AP_metadata_and_measured_busy_fraction', 'candidate_assignment'],
                'labels_only': ['hidden_propagation', 'future_signal', 'future_service', 'future_violation'],
                'action_interventions': 'four observed-input plans from one state with paired future blockage'}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return result, manifest
