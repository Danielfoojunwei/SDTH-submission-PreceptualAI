"""Only completed scans delivered over successful control frames enter memory."""
import hashlib
import numpy as np
from ..fleetwm.data import Observations


class DeliveredScans:
    def __init__(self, warm: Observations, duration: int, seed: int):
        if len(warm.value) != 60:
            raise ValueError('Provide exactly sixty warm-start timestamps')
        self.duration = duration
        self.n, self.links = warm.value.shape[1:]
        def extend(x):
            return np.concatenate([x.copy(), np.zeros((duration, *x.shape[1:]), dtype=x.dtype)])
        self.history = Observations(*(extend(x) for x in (warm.positions, warm.value, warm.heard, warm.bound, warm.scanned)))
        self.pending = {}
        self.count = np.zeros(self.n, dtype=int)
        self.seed = seed
        self.events = []
        self.missing = self.replaced = 0
        self.local = np.full((self.n, self.links), -100., dtype='f4')
        for t in range(60):
            self.local = np.where(warm.heard[t], warm.value[t], self.local)
        self.local_pending = {}

    def pose(self, decision, positions):
        self.history.positions[60+decision] = positions

    def scan(self, slot, ids, positions, rssi, complete):
        # RSSI already includes the native fading and blockage process. Add
        # receiver measurement noise only, never a second channel draw.
        for row, i in enumerate(ids):
            i = int(i)
            ordinal = int(self.count[i]); self.count[i] += 1
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, i, ordinal, 323000]))
            if rng.random() < .05:
                self.missing += 1
                continue
            value = (rssi[row] + rng.normal(0, 1.5, rssi[row].shape)).ravel()
            order = np.argsort(-value)[:8]
            heard = np.zeros(self.links, dtype=bool)
            heard[order] = value[order] >= -90
            bound = float(value[order[-1]]) if heard.sum() == 8 else -90.
            if i in self.pending:
                self.replaced += 1
            self.pending[i] = (int(slot), int(complete[row]), positions[row].copy(),
                               np.where(heard, value, 0).astype('f4'), heard, bound)
            self.local_pending[i] = self.pending[i]

    def complete(self, slot):
        for i, report in list(self.local_pending.items()):
            if report[1] <= slot:
                self.local[i] = np.where(report[4], report[3], -100.)
                del self.local_pending[i]

    def neighbors(self, current, measured_snr, noise, tx):
        """Cached completed local scans and current-link pilots, never AP truth."""
        rssi = self.local.reshape(self.n, -1, 3).copy()
        rssi[np.arange(self.n), current] = measured_snr+noise+tx-15
        return rssi

    def deliver(self, slot, delivered):
        for i in np.flatnonzero(delivered):
            i = int(i)
            report = self.pending.get(i)
            if report is None or report[1] > slot:
                continue
            measured, complete, xy, value, heard, bound = self.pending.pop(i)
            t = 60 + slot//100
            assert measured < complete <= slot and t < len(self.history.value)
            self.history.positions[t, i] = xy
            self.history.value[t, i] = value
            self.history.heard[t, i] = heard
            self.history.bound[t, i] = bound
            self.history.scanned[t, i] = True
            self.events.append({'robot': i, 'measurement_slot': measured, 'complete_slot': complete,
                                'delivery_slot': int(slot), 'first_usable_decision': slot//100+1,
                                'heard': int(heard.sum()), 'payload_sha256': hashlib.sha256(value.tobytes()+heard.tobytes()).hexdigest()})

    def before(self, decision):
        return self.history.before(60+decision)

    def summary(self):
        age = [(e['delivery_slot']-e['measurement_slot'])/100 for e in self.events]
        return {'scans_started': int(self.count.sum()), 'receiver_missing': self.missing,
                'reports_delivered': len(self.events), 'reports_pending': len(self.pending),
                'reports_replaced': self.replaced, 'mean_delivery_age_seconds': float(np.mean(age)) if age else None,
                'p95_delivery_age_seconds': float(np.quantile(age, .95)) if age else None,
                'events': self.events, 'reserved_bytes_per_control_frame': 128}
