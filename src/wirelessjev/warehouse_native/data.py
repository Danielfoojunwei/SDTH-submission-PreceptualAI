"""Public map/action features and native endpoint targets.

This experiment knows the declared log-distance map. It does not supply twin
scan readings or future packet telemetry to a policy. Its AP availability is
modelled, not measured. Native packet labels are read only at the target boundary.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from ..fastplan import BatchEvaluator, Snapshot
from ..hardware import RASPBOT
from ..warehouse_horizon.belief import apply_rates, public_scene
from ..warehouse_horizon.world import World, interruption_keep

DURATION = 30
BUDGET = 12


def public_observation(seed):
    world = World(seed, steps=12)
    scene = world.frames[3].scene
    critical = np.array([sum(f.demand_mbps for f in c.flows if f.critical) for c in scene.cars])
    bulk = np.array([sum(f.demand_mbps for f in c.flows if not f.critical) for c in scene.cars])
    floor = np.array([max([f.floor_mbps/f.demand_mbps for f in c.flows if f.critical and f.demand_mbps > 0] or [0.]) for c in scene.cars])
    from ..warehouse_horizon.world import OFFSETS
    route = np.array([scene.mobility.planned_positions(c, OFFSETS.tolist()) for c in scene.cars]).transpose(1, 0, 2)
    return {'robot': np.array([[c.x/120., c.y/80.] for c in scene.cars]),
            'route': route / np.array([120., 80.]), 'current': world.initial.copy(),
            'critical': critical, 'bulk': bulk, 'floor': floor,
            'ap': np.array([[a.x/120., a.y/80., float(a.band == '5g'), a.channel/165.,
                            scene.external_occupancy.get((a.band, a.channel), 0.), a.tx_power_dbm/30.]
                           for a in scene.aps])}


def downlink(obs, xy):
    """Declared native log-distance model, not an RF sensor reading."""
    distance = np.sqrt((((xy[:, None]-obs['ap'][None, :, :2])*[120., 80.])**2).sum(-1)+4.7**2)
    loss = np.where(obs['ap'][:, 2] > .5, 46.7, 40.) + 33.*np.log10(np.maximum(distance, 1.))
    return obs['ap'][None, :, 5]*30.-loss


class MapPrior:
    def __init__(self, obs):
        self.obs = obs
        self.current_mean = downlink(obs, obs['robot'])
        self.mean = np.stack([downlink(obs, xy) for xy in obs['route']])
        # public_scene needs flow declarations; its rack propagation is replaced
        # below before any forecast and is never used as a sensor or label.
        declaration = dict(obs, robot=np.c_[obs['robot'], np.zeros((len(obs['robot']), 2)), obs['floor']])
        scene = replace(public_scene(declaration), obstruction=None, ap_height_m=4.7, _radio=None, _radio_key=None)
        self.ev = BatchEvaluator(scene, [])
        apply_rates(self.ev)
        self.ev.client_itf = True
        self.ev.leak[:] = 0.
        self.ev.noise_mw = 10.**((-174.+10.*np.log10(np.where(obs['ap'][:, 2]>.5, 80e6, 20e6))+7.)/10.)
        power = obs['ap'][None, :, 5]*30.
        self.ev.snaps = [Snapshot(dl-power+RASPBOT.client_tx_power_dbm, dl) for dl in self.mean]

    def predict(self, plans):
        rows=[]
        for snap in self.ev.snaps:
            terms=self.ev._snapshot_terms(plans, snap, False, 5.)
            rows.append(np.stack([terms['sat'], terms['cf']], -1))
        # Charge one declared application mute over the same 30-second window.
        return (np.stack(rows, 1)*interruption_keep(plans!=self.obs['current'][None])[..., None]).mean(1)


def utility(outcomes):
    """Equal weight to delivered demand and deadline-met critical demand."""
    return np.asarray(outcomes).mean(axis=(-1, -2))


def worst_decile(outcomes):
    service = np.asarray(outcomes)[..., 0]
    return np.sort(service, axis=-1)[..., :max(1, int(np.ceil(service.shape[-1]*.1)))].mean(-1)


def candidates(obs, prior):
    """Four public-map proposals, with numerical search over a larger pool."""
    current=obs['current'];n=len(current);demand=obs['critical']+obs['bulk']
    # Target eligibility is explicitly a public-map assumption, not a scan.
    visible=[]
    for i, row in enumerate(prior.current_mean):
        top=np.argsort(-row, kind='stable')[:8]
        visible.append(np.union1d(top[row[top]>=-90.], [current[i]]))
    strong=np.array([v[np.argmax(prior.current_mean[i, v])] for i,v in enumerate(visible)])
    future=prior.mean.mean(0)
    route=np.array([v[np.argmax(future[i, v])] for i,v in enumerate(visible)])
    balanced=current.copy();load=np.zeros(len(obs['ap']))
    for i in np.argsort(-demand, kind='stable'):
        v=visible[i];j=int(v[np.argmax(future[i,v]-2.*load[v]-10.*obs['ap'][v,4])])
        balanced[i]=j;load[j]+=demand[i]
    def bounded(proposal, signal):
        changed=np.flatnonzero(proposal!=current)
        gain=signal[np.arange(n),proposal]-signal[np.arange(n),current]
        order=changed[np.argsort(-gain[changed],kind='stable')]
        plan=current.copy();plan[order[:BUDGET]]=proposal[order[:BUDGET]]
        return plan
    signal=bounded(strong,prior.current_mean)
    pool=[current.copy(),signal]
    for proposal in [strong,route,balanced]:
        endpoint=bounded(proposal,future)
        changed=np.flatnonzero(endpoint!=current)
        gain=future[np.arange(n),endpoint]-future[np.arange(n),current]
        order=changed[np.argsort(-gain[changed],kind='stable')]
        for limit in [2,5,BUDGET]:
            plan=current.copy();plan[order[:limit]]=endpoint[order[:limit]];pool.append(plan)
    unique=[];seen=set()
    for plan in pool:
        if plan.tobytes() not in seen:unique.append(plan);seen.add(plan.tobytes())
    search=np.stack(unique);prediction=prior.predict(search)
    numerical=int(utility(prediction).argmax());tail=int(worst_decile(prediction).argmax())
    proposals=[current.copy(),signal,search[numerical],search[tail]]
    names=['stay','budgeted_map_signal','numerical','numerical_tail']
    # Keep aliases in the protocol; execute byte-identical native inputs once.
    return np.stack(proposals),names,prior.predict(np.stack(proposals)),{'numerical_search_candidates':len(search)}


def topology(obs, plan):
    lines=[f'WIRELESSJEV_PACKET_V2 {len(plan)} {len(obs["ap"])} {RASPBOT.client_tx_power_dbm}']
    for a in obs['ap']:
        lines.append(f'{a[0]*120:.9f} {a[1]*80:.9f} {round(a[3]*165)} {int(a[2])} {a[4]:.9f} {a[5]*30:.9f}')
    for i, target in enumerate(plan):
        xy=np.vstack([obs['robot'][i],obs['route'][:,i]])*[120.,80.]
        positions=' '.join(f'{x:.9f} {y:.9f}' for x,y in xy)
        lines.append(f'{obs["critical"][i]+obs["bulk"][i]:.9f} {obs["critical"][i]:.9f} {target} {4 if target!=obs["current"][i] else 0} {positions}')
    return '\n'.join(lines)+'\n'


def features(obs, plans, prediction):
    k,n=plans.shape;m=len(obs['ap']);rows=np.arange(n)[None]
    future=np.stack([downlink(obs, xy) for xy in obs['route']])
    now=downlink(obs,obs['robot'])
    selected=future[:,rows,plans].transpose(1,2,0)
    uplink=selected-obs['ap'][plans,5,None]*30.+RASPBOT.client_tx_power_dbm
    band=obs['ap'][:,2]>.5
    channel=np.rint(obs['ap'][:,3]*165).astype(int)
    same=(band[:,None]==band[None])&(channel[:,None]==channel[None])
    group=np.array([int(np.flatnonzero(row)[0]) for row in same])
    ap_load=np.zeros((k,m,3));values=np.c_[np.ones(n),obs['critical'],obs['bulk']]
    for j in range(k):np.add.at(ap_load[j],plans[j],values)
    channel_load=np.einsum('ab,kbc->kac',same.astype(float),ap_load)
    current_signal=now[rows,np.broadcast_to(obs['current'],plans.shape)]
    static=np.c_[obs['robot'],np.log1p(obs['critical']),np.log1p(obs['bulk']),obs['floor']]
    x=np.concatenate([np.broadcast_to(static,(k,n,5)),obs['ap'][plans],
        (plans!=obs['current'][None])[...,None],
        (selected+80.)/30.,(uplink+80.)/30.,
        np.broadcast_to(obs['route'].transpose(1,0,2).reshape(n,12),(k,n,12)),
        np.log1p(ap_load[ np.arange(k)[:,None],plans]),
        np.log1p(channel_load[np.arange(k)[:,None],plans]),
        ((now[rows,plans]-current_signal)/30.)[...,None],prediction],-1)
    if not np.isfinite(x).all():raise ValueError('Nonfinite public action features')
    return x.astype('float32'),group[plans]


def endpoint_target(report, critical_mbps):
    """Nominal demand in the full window, including demand lost to muting."""
    duration=report['duration_s'];n=report['robots']
    if duration!=DURATION or len(report['telemetry'])!=DURATION:raise ValueError('Native window must match the decision horizon')
    delivered=np.zeros(n);critical_ontime=np.zeros(n)
    for frame in report['telemetry']:
        for row in frame['robots']:
            i=row['robot'];delivered[i]+=row['delivered_bytes']
            if row['critical_late']>row['critical_rx']:raise ValueError('Invalid endpoint counts')
            critical_ontime[i]+=row['critical_rx']-row['critical_late']
    # Critical packets are exactly 240 bytes in the native source. The nominal
    # denominator is application demand, not a count of fabricated lost packets.
    critical=np.asarray(critical_mbps, dtype=float)
    if critical.shape!=(n,) or not np.isfinite(critical).all() or (critical<=0).any():
        raise ValueError('Declared critical demand must match every robot')
    total=np.array([r['offered_mbps'] for r in report['per_robot']])
    return np.stack([np.minimum(1.,delivered*8/(duration*total*1e6)),
                     np.minimum(1.,critical_ontime*240*8/(duration*critical*1e6))],-1).astype('float32')
