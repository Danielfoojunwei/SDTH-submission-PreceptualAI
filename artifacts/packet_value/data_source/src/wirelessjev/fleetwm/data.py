"""Separate observable fleet histories from simulator training labels."""
from dataclasses import dataclass, replace
import numpy as np
from ..contracted.data import ROOT, make_stream, capped_scans
from ..native.radiomap import RadioMap
from ..contracted.data import YardRadioMap
from ..native.world import CLIENT_TX_DBM, HORIZONS
from ..twin.warehouse import make_warehouse, rack_obstruction, RACK_DB


@dataclass(frozen=True)
class Geometry:
    sites: np.ndarray
    tx: np.ndarray
    noise: np.ndarray
    aps: tuple
    yard: bool = False


@dataclass(frozen=True)
class Observations:
    positions: np.ndarray
    value: np.ndarray
    heard: np.ndarray
    bound: np.ndarray
    scanned: np.ndarray

    def before(self, t):
        if t < 6 or t > len(self.value):
            raise ValueError('Need at least six strictly past scans')
        return Observations(*(np.array(v[:t], copy=True) for v in (
            self.positions, self.value, self.heard, self.bound, self.scanned)))


@dataclass(frozen=True)
class FleetInputs:
    context: np.ndarray
    query: np.ndarray
    memory: np.ndarray
    physics: np.ndarray
    map_mean: np.ndarray
    map_std: np.ndarray
    fleet_mean: np.ndarray


def episode(seed, n, duration, condition='main'):
    full, scan = make_stream(seed, n=n, decisions=duration,
                             condition='main' if condition == 'drift' else condition)
    if condition == 'drift':
        # Declared changing mean field. No model receives this hidden schedule.
        rng = np.random.default_rng(seed + 800000)
        onset = rng.uniform(65, 90, len(full.sites))
        amplitude = rng.uniform(-9, 12, len(full.sites))
        t = np.arange(len(full.snr)) * .2
        shift = amplitude[None] * np.clip((t[:, None] - onset[None]) / 25, 0, 1)
        full = replace(full, snr=full.snr - shift[:, None, :, None],
                       rssi_dl=full.rssi_dl - shift[:, None, :, None])
        scan = capped_scans(full.rssi_dl[::5], np.random.default_rng(seed + 900000))
    geom = Geometry(full.sites.copy(), full.tx_dl.copy(), full.noise_db.copy(),
                    tuple(a for a in make_warehouse(seed, n_robots=1).aps if a.band == '5g'),
                    condition == 'yard')
    obs = Observations(full.pos[::5].copy(), *(scan[k].copy() for k in
                                             ['value', 'heard', 'bound', 'scanned']))
    return full, scan, geom, obs


def geometry(g, positions):
    p = np.asarray(positions)
    delta = p[..., None, :] - g.sites
    logd = np.log10(np.maximum(1, np.linalg.norm(delta, axis=-1)))
    if g.yard:
        rows = np.zeros_like(logd)
    else:
        rows = np.stack([rack_obstruction(a, p[..., 0].ravel(), p[..., 1].ravel()).reshape(p.shape[:-1])
                         / RACK_DB['5g'] for a in g.aps], -1)
    prior = CLIENT_TX_DBM - np.array([40, 46.7, 48.64]) - 33 * logd[..., None] - rows[..., None] * np.array([2.5, 3.5, 4.5]) - g.noise
    return delta, logd, rows, prior.astype('f4')


def fit_map(g, past, window=60):
    """Same censored Bayesian fitter, using only the supplied past prefix."""
    start = max(0, len(past.value) - window)
    pos = past.positions[start:]
    obs, cens = {}, {}
    robots = np.broadcast_to(np.arange(pos.shape[1]), pos.shape[:2])
    times = np.arange(start, len(past.value))[:, None]
    for s in range(len(g.sites)):
        for b in range(3):
            j = 3*s+b
            h = past.heard[start:, :, j]
            c = past.scanned[start:] & ~h & (times % 5 == 0)
            obs[s,b] = (pos[...,0][h], pos[...,1][h], past.value[start:,:,j][h], robots[h])
            cens[s,b] = (pos[...,0][c], pos[...,1][c], past.bound[start:,:,j][c], robots[c])
    cls = YardRadioMap if g.yard else RadioMap
    return cls(g.sites, g.aps, 0).fit(obs, cens, np.arange(max(1, pos.shape[1]//5)))


def map_prediction(g, rmap, plan):
    m, sd = rmap.predict(plan[...,0], plan[...,1])
    return (m - g.tx + CLIENT_TX_DBM - g.noise).astype('f4'), sd.astype('f4')


def inputs(g, past: Observations, plan, rmap):
    """Label-free constructor: no World, truth SNR or future scan argument."""
    if not isinstance(past, Observations):
        raise TypeError('Use an explicit observable history')
    N, H = plan.shape[:2]; S = len(g.sites); L = 3*S
    positions = past.positions[-60:]
    delta, logd, racks, prior = geometry(g, positions)
    shift = np.tile(CLIENT_TX_DBM-g.tx-g.noise, S)
    heard = past.heard[-60:]
    value = past.value[-60:] + shift
    bound = past.bound[-60:] + shift
    scanned = past.scanned[-60:]
    ld = np.repeat(logd, 3, -1); rc = np.repeat(racks, 3, -1)
    dxy = np.repeat(delta, 3, -2)
    feat = np.stack([np.where(heard, value, 0)/30,
                     np.where(scanned[...,None], bound, 0)/30, heard,
                     np.broadcast_to(scanned[...,None], heard.shape),
                     dxy[...,0]/120, dxy[...,1]/80, ld/3,
                     np.broadcast_to(np.tile(np.arange(3)/2,S),heard.shape)], -1).astype('f4')
    ctx = feat[-6:].transpose(1,2,0,3).reshape(N,L,48)
    residual = value - prior.reshape(value.shape)
    # Fit a small reports-only geometric summary, NOT the per-BSS RadioMap.
    # Selection bias remains; the neural predictor must learn its correction.
    memory = np.zeros((L,13),dtype='f4'); coefs = np.zeros((L,3))
    for j in range(L):
        h = heard[:,:,j]; count = int(h.sum())
        if count:
            r = np.clip(residual[:,:,j][h],-50,50)
            X = np.stack([np.ones_like(ld[:,:,j]),ld[:,:,j]-1.5,rc[:,:,j]/3],-1)[h]
            coefs[j] = np.linalg.solve(X.T@X+np.diag([3.,10.,10.]),X.T@r)
            recent = h[-10:]; recent_r = residual[-10:,:,j][recent]
            xy = positions[h]
            memory[j,:10] = [count/h.size, np.mean(r)/20, np.std(r)/10,
                             (np.mean(recent_r)-np.mean(r))/10 if recent.any() else 0,
                             * (coefs[j]/10), np.std(xy[:,0])/60,
                             np.std(xy[:,1])/40,
                             (len(h)-1-np.nonzero(h)[0].max())/60]
        else:
            memory[j,9] = 1
        c = scanned & ~h
        memory[j,10] = float(np.mean((bound-prior.reshape(value.shape))[:,:,j][c])/20) if c.any() else 0
        band = j%3; bh = heard[:,:,band::3]
        memory[j,11] = float(np.mean(residual[:,:,band::3][bh])/20) if bh.any() else 0
        memory[j,12] = g.noise[band]/100
    qdelta, qlogd, qracks, physics = geometry(g,plan)
    qld = np.repeat(qlogd,3,-1); qrc = np.repeat(qracks,3,-1)
    qd = np.repeat(qdelta,3,-2)
    qs = np.broadcast_to(np.asarray(HORIZONS)[None,:,None]/30, (N,H,L))
    band = np.broadcast_to(np.tile(np.arange(3)/2,S),(N,H,L))
    px = np.broadcast_to(plan[...,0,None],(N,H,L)); py = np.broadcast_to(plan[...,1,None],(N,H,L))
    query = np.stack([qd[...,0]/120,qd[...,1]/80,qld/3,qrc/5,qs,band,
                      np.sin(px/15),np.cos(px/15),np.sin(py/15),np.cos(py/15),
                      qld-qld[:,:1]],-1).astype('f4')
    X = np.stack([np.ones_like(qld),qld-1.5,qrc/3],-1)
    fleet = physics.reshape(N,H,L) + np.einsum('nhli,li->nhl',X,coefs)
    mm, sd = map_prediction(g,rmap,plan)
    return FleetInputs(ctx,query,np.broadcast_to(memory,(N,L,13)).copy(),
                       physics.reshape(N,H,L),mm.reshape(N,H,L),sd.reshape(N,H,L),fleet.astype('f4'))


def scan_context(g, past):
    """Only the last six observable scans; also used for EMA training labels."""
    pos=past.positions[-6:];delta,ld,rc,_=geometry(g,pos)
    S=len(g.sites);h=past.heard[-6:];sc=past.scanned[-6:]
    shift=np.tile(CLIENT_TX_DBM-g.tx-g.noise,S)
    d=np.repeat(delta,3,-2);ld=np.repeat(ld,3,-1)
    f=np.stack([np.where(h,past.value[-6:]+shift,0)/30,
                np.where(sc[...,None],past.bound[-6:]+shift,0)/30,h,
                np.broadcast_to(sc[...,None],h.shape),d[...,0]/120,d[...,1]/80,ld/3,
                np.broadcast_to(np.tile(np.arange(3)/2,S),h.shape)],-1).astype('f4')
    return f.transpose(1,2,0,3).reshape(pos.shape[1],S*3,48)


def training_condition(index):
    return ['yard','stress','drift','main'][index%4] if index >= 20 else 'main'
