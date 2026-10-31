"""Private simulation state and public observations with independent lookahead.

The forecaster and planner do not receive World/Frame/EnvSpec objects. Scans
come from the current simulated sensor; delivery telemetry describes the
previous realised interval. Future labels are built at this boundary only.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace

import numpy as np

from ..envlearn import external_noise, leak_matrix
from ..fastplan import BatchEvaluator, Snapshot
from ..warehouse_v2.data import make_environment

OFFSETS = np.arange(5., 31., 5.)
STEP_S = 5.


def interruption_keep(moved, *, outage_s=4., offsets=OFFSETS):
    if not np.isfinite(outage_s) or outage_s < 0:
        raise ValueError('A finite nonnegative interruption is required')
    starts = np.asarray(offsets)-STEP_S
    lost = np.clip(outage_s-starts, 0., STEP_S)/STEP_S
    return 1.-np.asarray(moved)[:, None, :]*lost[None, :, None]


@dataclass
class Frame:
    scene: object
    spec: object
    blockage: np.ndarray
    _template: object = field(default=None, init=False, repr=False)

    def evaluator(self, current):
        from .belief import apply_rates
        if self._template is None:
            base = BatchEvaluator(self.scene, [])
            apply_rates(base)
            base.client_itf = True
            base.ext_noise_mw = external_noise(self.scene, self.spec)
            base.leak = leak_matrix(self.scene, self.spec.leak_db)
            raw = base.snaps[0]
            base.snaps = [Snapshot(raw.ul-self.blockage, raw.dl-self.blockage)]
            self._template = base
        # Geometry and propagation do not depend on the intervention. Reuse
        # their read-only arrays, but give each request its own association.
        ev = BatchEvaluator.__new__(BatchEvaluator)
        ev.__dict__.update(self._template.__dict__)
        ev.current = np.asarray(current).copy()
        return ev


class World:
    def __init__(self, seed, *, n=100, steps=126, domain='train', environment_seed=None, bulk_scale=1.):
        scene, spec = make_environment(seed, n, domain, environment_seed=environment_seed)
        if not np.isfinite(bulk_scale) or bulk_scale <= 0:
            raise ValueError('A positive finite traffic scale is required')
        if bulk_scale != 1.:
            scene = replace(scene, cars=[replace(car, flows=tuple(
                flow if flow.critical else replace(flow, demand_mbps=flow.demand_mbps*bulk_scale)
                for flow in car.flows)) for car in scene.cars], _radio=None, _radio_key=None)
        blockage_rng = np.random.default_rng(seed+4_000_000)
        self.seed, self.frames = seed, []
        for _ in range(steps):
            frozen = deepcopy(scene)
            hit = blockage_rng.random((n, len(scene.aps))) < spec.blockage_p
            self.frames.append(Frame(frozen, spec, hit.astype(float)*spec.blockage_db))
            # Scene.advance mutates its mobility object. Only this independent
            # working copy advances; stored frames keep their own route/RNG.
            scene = scene.advance(STEP_S)
        self.initial = self.frames[0].evaluator(
            [self.frames[0].scene.radio().ap_index[c.current_ap] for c in self.frames[0].scene.cars]).current

    def outcomes(self, tick, plans, current, *, outage_s=4.):
        plans, current = np.asarray(plans), np.asarray(current)
        K, n = plans.shape
        if tick+len(OFFSETS) >= len(self.frames):
            raise ValueError('Insufficient future frames')
        services, failures, signals = [], [], []
        critical_fraction = []
        for step in range(len(OFFSETS)):
            ev = self.frames[tick+step+1].evaluator(current)
            terms = ev._snapshot_terms(plans, ev.snaps[0], False, STEP_S)
            cf = terms['cf']
            failed = ((ev.cf_dem[None]*cf[:, :, None] < ev.cf_floor[None]-1e-9)
                      & (ev.cf_dem[None] > 0)).any(2)
            services.append(terms['sat'])
            failures.append(failed)
            signals.append(ev.snaps[0].dl[np.arange(n)[None], plans])
            critical_fraction.append(cf)
        service = np.stack(services, 1)
        raw_failure = np.stack(failures, 1)
        cf = np.stack(critical_fraction, 1)
        keep = interruption_keep(plans != current[None], outage_s=outage_s)
        # Per-flow floors, not a guessed aggregate threshold.
        ev = self.frames[tick+1].evaluator(current)
        failure = ((ev.cf_dem[None, None]*cf[..., None]*keep[..., None] < ev.cf_floor[None, None]-1e-9)
                   & (ev.cf_dem[None, None] > 0)).any(3)
        return {'service_raw': service.astype('float32'), 'failure_raw': raw_failure.astype('float32'),
                'signal': ((np.stack(signals, 1)+80.)/30.).astype('float32'),
                'service': (service*keep).astype('float32'), 'failure': failure.astype('float32')}

    def interval(self, tick, action, previous, *, outage_remaining=None):
        ev = self.frames[tick+1].evaluator(previous)
        terms = ev._snapshot_terms(np.asarray(action)[None], ev.snaps[0], False, STEP_S)
        if outage_remaining is None:
            outage_remaining = (np.asarray(action) != previous)*4.
        keep = 1.-np.clip(outage_remaining, 0., STEP_S)/STEP_S
        service = terms['sat'][0]*keep
        failed = ((ev.cf_dem*terms['cf'][0, :, None]*keep[:, None] < ev.cf_floor-1e-9)
                  & (ev.cf_dem > 0)).any(1)
        return service, failed, np.maximum(0., outage_remaining-STEP_S)


def observe(frame, current, tick, *, previous_service, since_switch, history):
    scene = frame.scene
    rng = np.random.default_rng(scene.seed+5_000_000+tick)
    signal = scene.radio().dl-frame.blockage+rng.normal(0, 3., frame.blockage.shape)
    n, m = signal.shape
    top = np.argsort(-signal, axis=1)[:, :8]
    mask = np.zeros(signal.shape, bool)
    np.put_along_axis(mask, top, np.take_along_axis(signal >= -90., top, axis=1), axis=1)
    bound = np.where(mask.sum(1) == 8, np.where(mask, signal, np.inf).min(1), -90.)
    scan = np.stack([np.where(mask, (signal+80.)/30., 0.), mask,
                     np.broadcast_to((bound[:, None]+80.)/30., signal.shape)], -1).astype('float32')
    xy = np.array([[c.x/120., c.y/80.] for c in scene.cars], dtype='float32')
    history.append((scan, xy))
    del history[:-4]
    public_routes = np.array([scene.mobility.planned_positions(c, OFFSETS.tolist()) for c in scene.cars], dtype='float32').transpose(1, 0, 2)
    public_routes /= np.array([120., 80.], dtype='float32')
    critical = np.array([sum(f.demand_mbps for f in c.flows if f.critical) for c in scene.cars])
    bulk = np.array([sum(f.demand_mbps for f in c.flows if not f.critical) for c in scene.cars])
    floor_ratio = np.array([max([f.floor_mbps/f.demand_mbps for f in c.flows if f.critical and f.demand_mbps > 0] or [0.]) for c in scene.cars])
    video = np.array([any('video' in f.name for f in c.flows) for c in scene.cars])
    # Demand/floor arrays are public application declarations, not teacher values.
    robot = np.c_[xy, np.log1p(critical)/3., np.log1p(bulk)/3., floor_ratio,
                  video, previous_service, np.minimum(since_switch/30., 1.)].astype('float32')
    ap = np.array([[a.x/120., a.y/80., float(a.band == '5g'), a.channel/165.,
                    np.clip(scene.external_occupancy.get((a.band, a.channel), 0.)+rng.normal(0, .02), 0., 1.),
                    a.tx_power_dbm/30.] for a in scene.aps], dtype='float32')
    return {'history': np.stack([s for s, _ in history]), 'past_xy': np.stack([p for _, p in history]),
            'robot': robot, 'ap': ap, 'route': public_routes, 'current': np.array(current, dtype='int64'),
            'critical': critical.astype('float32'), 'bulk': bulk.astype('float32'),
            'video': video, 'tick': np.array(tick, dtype='int64')}
