"""Timestamp genuine local scan captures without changing default telemetry."""
from __future__ import annotations

import hashlib
import numpy as np

from ..closed_loop.telemetry import DeliveredScans
from ..fleetwm.data import Observations

UNKNOWN_CAPTURE = int(np.iinfo(np.int64).min)
MAX_SLOT = int(np.iinfo(np.int64).max)


def _slot(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not 0 <= value <= MAX_SLOT):
        raise ValueError(f'{name} must be a nonnegative integer slot')
    return int(value)


class TimedDeliveredScans(DeliveredScans):
    """Keep all inherited default scan/delivery/cache/RNG behavior unchanged.

    Only ``recovery_neighbors`` applies capture-age and heard masks. Capture
    time belongs to the measurement, not its later completion or delivery.
    The default ``neighbors`` and ``summary`` methods retain predecessor
    semantics, allowing identical telemetry use in every comparator arm.
    """

    def __init__(self, warm: Observations, duration: int, seed: int):
        if not isinstance(warm, Observations) or warm.value.ndim != 3:
            raise ValueError('Require observable warm history with time/robot/link axes')
        t, n, links = warm.value.shape
        if (t != 60 or n < 1 or links < 3 or links % 3
                or warm.heard.shape != (t, n, links) or warm.heard.dtype != np.bool_
                or warm.scanned.shape != (t, n) or warm.scanned.dtype != np.bool_
                or warm.bound.shape != (t, n, links) or warm.positions.shape != (t, n, 2)):
            raise ValueError('Warm history must contain complete fleet/AP/band axes and boolean masks')
        super().__init__(warm, duration, seed)
        self.local_capture_slots = np.full((n, links), UNKNOWN_CAPTURE, dtype=np.int64)
        self.local_capture_heard = np.zeros((n, links), dtype=bool)
        for index in range(60):
            heard = warm.heard[index]
            self.local_capture_slots[heard] = (index - 60) * 100
            self.local_capture_heard |= heard
        # Back copies with immutable bytes so writeability cannot be restored.
        self.initial_local_capture_slots = np.frombuffer(self.local_capture_slots.tobytes(), dtype=np.int64).reshape(n, links)
        self.initial_local_capture_heard = np.frombuffer(self.local_capture_heard.tobytes(), dtype=bool).reshape(n, links)
        self.local_completion_events = []
        self._last_complete_slot = self._last_recovery_slot = -1

    def complete(self, slot):
        slot = _slot(slot, 'Completion clock')
        if slot < max(self._last_complete_slot, self._last_recovery_slot):
            raise ValueError('Completion clock must not move backwards')
        finished = []
        # The parent deletes completed local_pending entries, so inspect each
        # genuine report before invoking it. No receiver draw is repeated.
        for robot, report in list(self.local_pending.items()):
            robot = _slot(robot, 'Report robot')
            if robot >= self.n or not isinstance(report, (tuple, list)) or len(report) != 6:
                raise ValueError('Malformed completed local scan metadata')
            complete = _slot(report[1], 'Report completion')
            if complete <= slot:
                captured = _slot(report[0], 'Report capture')
                heard = np.asarray(report[4])
                values = np.asarray(report[3])
                if (captured > complete or heard.shape != (self.links,)
                        or heard.dtype != np.bool_ or values.shape != (self.links,)
                        or values.dtype.kind not in 'fi'):
                    raise ValueError('Malformed completed local scan metadata')
                event = {'robot': int(robot), 'measurement_slot': captured,
                    'complete_slot': complete, 'processed_slot': slot, 'heard_entries': int(heard.sum()),
                    'heard_links': np.flatnonzero(heard).tolist(),
                    'heard_rssi_dbm': values[heard].astype('f4', copy=False).tolist(),
                    'payload_sha256': hashlib.sha256(values.tobytes() + heard.tobytes()).hexdigest()}
                finished.append((robot, captured, heard.copy(), event))
        super().complete(slot)
        for robot, captured, heard, event in finished:
            self.local_capture_slots[robot] = np.where(heard, captured, UNKNOWN_CAPTURE)
            self.local_capture_heard[robot] = heard
            self.local_completion_events.append(event)
        self._last_complete_slot = slot

    def recovery_neighbors(self, slot, current, measured_snr, noise, tx, max_age_slots):
        """Completed, heard and recent other links; current pilots stay live.

        Unheard/stale alternative bands are ``-inf`` rather than the legacy
        cache sentinel −100 dBm. They cannot outrank a genuinely dead current
        link. Current-link RSSI always comes from the supplied measured pilots.
        """
        slot = _slot(slot, 'Recovery clock')
        max_age_slots = _slot(max_age_slots, 'Maximum capture age')
        if slot < max(self._last_complete_slot, self._last_recovery_slot):
            raise ValueError('Recovery clock must not move backwards')
        current, measured_snr, noise, tx = map(np.asarray, (current, measured_snr, noise, tx))
        aps = self.links // 3
        if (current.shape != (self.n,) or current.dtype.kind not in 'iu'
                or (current < 0).any() or (current >= aps).any()):
            raise ValueError('Current AP indices must match the public fleet/AP axes')
        for value, shape, name in ((measured_snr, (self.n, 3), 'Measured current pilots'),
                                   (noise, (3,), 'Noise'), (tx, (3,), 'Transmit power')):
            if value.shape != shape or value.dtype.kind not in 'fi' or not np.isfinite(value).all():
                raise ValueError(name + ' must have finite fleet/band axes')
        # Comparing against a cutoff avoids int64 overflow on unknown captures
        # or very large clocks. Warm captures may legitimately be negative.
        fresh = (self.local_capture_heard & (self.local_capture_slots != UNKNOWN_CAPTURE)
                 & (self.local_capture_slots >= slot - max_age_slots)
                 & (self.local_capture_slots <= slot) & np.isfinite(self.local))
        rssi = np.where(fresh, self.local, -np.inf).reshape(self.n, aps, 3)
        with np.errstate(over='ignore', invalid='ignore'):
            pilots = (measured_snr + noise + tx - 15).astype(rssi.dtype)
        if not np.isfinite(pilots).all():
            raise ValueError('Current pilots must remain finite after conversion to cached RSSI units')
        rssi[np.arange(self.n), current] = pilots
        self._last_recovery_slot = slot
        return rssi
