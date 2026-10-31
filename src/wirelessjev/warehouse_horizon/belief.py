"""Public-observation radio belief and numerical multi-step baseline.

Known CAD geometry is combined with actual simulated scan reports. Omission
bounds update a Gaussian belief as inequalities, never as point samples.
This module has no access to the hidden environment specification or World.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr

from ..fastplan import BatchEvaluator, Snapshot
from ..hardware import RASPBOT
from ..twin.radio import AP, BANDS, CHANNELS, Car, Flow, Scene
from ..twin.warehouse import rack_obstruction


@lru_cache(maxsize=1)
def rate_tables():
    path = Path(__file__).resolve().parents[3]/'artifacts/warehouse_v2/studies/wifi_link_curves.json'
    rows = json.loads(path.read_text())['points']
    result = {}
    for band in ['5g', '2g']:
        grouped = {}
        for row in rows:
            if row['band'] == band:
                grouped.setdefault(row['snr_db'], []).append(row['rate_mbps']*row['packet_success'])
        snr = np.array(sorted(grouped))
        result[band] = snr, np.array([max(grouped[s]) for s in snr])
    return result


def apply_rates(ev):
    ev.t5, ev.r5 = rate_tables()['5g']
    ev.t2, ev.r2 = rate_tables()['2g']


def public_scene(obs):
    aps = [AP(f'bss-{j}', float(a[0]*120), float(a[1]*80), int(round(a[3]*165)),
              '5g' if a[2] > .5 else '2g', f'site-{j}', tx_power_dbm=float(a[5]*30))
           for j, a in enumerate(obs['ap'])]
    cars = []
    for i, r in enumerate(obs['robot']):
        flows = (Flow('critical', float(obs['critical'][i]), True, float(obs['critical'][i]*r[4])),
                 Flow('bulk', float(obs['bulk'][i]), False, 0.))
        cars.append(Car(f'amr-{i+1:03d}', float(r[0]*120), float(r[1]*80), 0., 0., flows,
                        aps[int(obs['current'][i])].ap_id))
    busy = {}
    for ap, row in zip(aps, obs['ap']):
        busy.setdefault((ap.band, ap.channel), []).append(float(row[4]))
    return Scene(0, cars, aps, {k: float(np.mean(v)) for k, v in busy.items()},
                 width_m=120., height_m=80., obstruction=rack_obstruction,
                 shadow_db=0., pl_coef=33., ap_height_m=5.)


def nominal_dl(aps, xy):
    x, y = xy[:, 0]*120., xy[:, 1]*80.
    columns = []
    for ap in aps:
        d = np.maximum(1., np.sqrt((x-ap.x)**2+(y-ap.y)**2+25.))
        columns.append(ap.tx_power_dbm-BANDS[ap.band][0]-33.*np.log10(d)-rack_obstruction(ap, x, y))
    return np.stack(columns, 1)


def scan_belief(obs, aps):
    now = obs['robot'][:, :2]
    nominal = nominal_dl(aps, now)
    precision = np.full(nominal.shape, 1/64.)
    weighted = nominal/64.
    H = len(obs['history'])
    available = np.asarray(obs.get('scan_available', np.ones((H, len(now)), dtype=bool)))
    if available.shape != (H, len(now)) or not np.isin(available, [0, 1]).all():
        raise ValueError('Scan availability must identify actually collected robot scans')
    for step, scan in enumerate(obs['history']):
        age = (H-1-step)*5.
        valid = (scan[..., 1] > 0) & available[step, :, None].astype(bool)
        translated = scan[..., 0]*30.-80.+nominal-nominal_dl(aps, obs['past_xy'][step])
        weight = valid/(9.+.04*age**2)
        precision += weight
        weighted += translated*weight
    variance = 1./precision
    mean = weighted*variance
    omitted = (obs['history'][-1, :, :, 1] == 0) & available[-1, :, None].astype(bool)
    bound = obs['history'][-1, :, :, 2]*30.-80.
    noise = np.sqrt(variance+9.)
    z = (bound-mean)/noise
    mills = np.exp(-.5*z*z-.5*np.log(2*np.pi)-log_ndtr(z))
    # Gaussian conditioning on a noisy measurement below the cap/floor bound.
    corrected_mean = mean-variance/noise*mills
    corrected_var = variance-variance**2/noise**2*mills*(mills+z)
    mean = np.where(omitted, corrected_mean, mean)
    variance = np.where(omitted, np.maximum(corrected_var, .1), variance)
    future_mean, future_std = [], []
    for xy in obs['route']:
        distance = ((xy-now)*np.array([120., 80.]))**2
        future_mean.append(mean+nominal_dl(aps, xy)-nominal)
        future_std.append(np.sqrt(variance+4.+.04*distance.sum(1)[:, None]))
    return np.stack(future_mean), np.stack(future_std), mean


class RadioBelief:
    def __init__(self, observation):
        self.obs = observation
        self.scene = public_scene(observation)
        self.mean, self.std, self.current_mean = scan_belief(observation, self.scene.aps)
        self.ev = BatchEvaluator(self.scene, [])
        apply_rates(self.ev)
        ap_tx = np.array([a.tx_power_dbm for a in self.scene.aps])
        self.ev.snaps = [Snapshot(dl-ap_tx[None]+RASPBOT.client_tx_power_dbm, dl) for dl in self.mean]
        self.ev.client_itf = True
        self.ev.leak = np.zeros((self.ev.m, self.ev.m))
        for j, a in enumerate(self.scene.aps):
            for k, b in enumerate(self.scene.aps):
                if j != k and a.band == b.band and abs(CHANNELS[a.band].index(a.channel)-CHANNELS[b.band].index(b.channel)) == 1:
                    self.ev.leak[k, j] = 10.**(-28./10.)
        self.ev.visible = [np.flatnonzero(row[:, 1] > 0) for row in observation['history'][-1]]

    def forecast(self, plans):
        A = np.asarray(plans, dtype='int64')
        if A.ndim != 2 or A.shape[1] != self.ev.n or (A < 0).any() or (A >= self.ev.m).any():
            raise ValueError('Invalid candidate assignment')
        features = []
        rows = np.arange(self.ev.n)[None]
        for step, snap in enumerate(self.ev.snaps):
            t = self.ev._snapshot_terms(A, snap, False, 5.)
            activity = np.take_along_axis(t['activity_in'], A, 1)
            features.append(np.stack([t['sat'], t['cf'], t['sinr']/40.,
                                      (self.mean[step][rows, A]+80.)/30.,
                                      self.std[step][rows, A]/30., activity, t['bf'],
                                      self.obs['ap'][A, 4]], -1))
        return np.stack(features, 1).astype('float32')
