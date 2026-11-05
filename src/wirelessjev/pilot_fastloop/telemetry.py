"""Positive-only existing pilots in an already reserved control budget.

No RF sampler is called. Payloads are frozen before control delivery, and
only successful exchanges can populate strictly past model history.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import struct

import numpy as np

from ..jepa_repair.telemetry import TimedDeliveredScans, _slot

CONTROL_FRAME_BYTES = 600
REPORT_RESERVED_BYTES = 128
PILOT_FORMAT = '<BBHHQQffB3f'
PILOT_STRUCT = struct.Struct(PILOT_FORMAT)
PILOT_PAYLOAD_BYTES = PILOT_STRUCT.size
PILOT_VERSION = 1
PILOT_KIND = 1
PILOT_THRESHOLD_DBM = -90.
PILOT_REFRESH_SLOTS = 10


def _boolean(value, n, name):
    value = np.asarray(value)
    if value.shape != (n,) or value.dtype != np.bool_:
        raise ValueError(name + ' requires the complete boolean fleet mask')
    return value


def _integers(value, n, name, low, high):
    value = np.asarray(value)
    if (value.shape != (n,) or value.dtype.kind not in 'iu'
            or (value < low).any() or (value > high).any()):
        raise ValueError(name + ' requires integer fleet metadata within support')
    return value


def _finite(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in 'fi' or not np.isfinite(value).all():
        raise ValueError(name + ' requires finite numeric axes')
    return value


def decode_pilot_payload(payload, *, robot_count, ap_count):
    """Decode the exact 43 bytes which are allowed to enter model history."""
    if not isinstance(payload, bytes) or len(payload) != PILOT_PAYLOAD_BYTES:
        raise ValueError('Require the complete 43-byte pilot payload')
    version, kind, robot, ap, cap, refresh, x, y, mask, *rssi = PILOT_STRUCT.unpack(payload)
    if (version != PILOT_VERSION or kind != PILOT_KIND or robot >= robot_count
            or ap >= ap_count or mask == 0 or mask > 7 or cap > refresh
            or refresh > np.iinfo(np.int64).max or not np.isfinite([x, y, *rssi]).all()):
        raise ValueError('Malformed typed pilot identity, clocks, mask or values')
    heard = tuple(bool(mask & (1 << b)) for b in range(3))
    if any((h and rssi[b] < PILOT_THRESHOLD_DBM) or (not h and rssi[b] != 0.)
           for b, h in enumerate(heard)):
        raise ValueError('Pilot positives and unknown placeholders are inconsistent')
    return {'robot': robot, 'ap': ap, 'capture_slot': cap, 'refresh_slot': refresh,
            'position': (x, y), 'heard': heard, 'rssi_dbm': tuple(rssi)}


@dataclass(frozen=True)
class _ScanSnapshot:
    capture: int
    complete: int
    position: tuple
    value: bytes
    heard: bytes
    bound: float
    origin: object


@dataclass(frozen=True)
class _Prepared:
    robot: int
    kind: str
    event_index: int
    pilot_payload: bytes | None = None
    scan: _ScanSnapshot | None = None


class PilotDeliveredScans(TimedDeliveredScans):
    """Keep scans unchanged and optionally transmit current positive pilots.

    summary() retains the predecessor schema; pilot_summary() is separate.
    Legacy full scans occupy their opaque 128-byte reservation. Their decoded
    array hash is not represented as a 128-byte serialization.
    """

    def __init__(self, warm, duration, seed, *, pilot_enabled=False):
        if type(pilot_enabled) is not bool:
            raise ValueError('Pilot reporting must be an explicit boolean')
        super().__init__(warm, duration, seed)
        if pilot_enabled and (self.n > 65536 or self.links // 3 > 65536):
            raise ValueError('Pilot robot/AP identifiers exceed uint16 support')
        self.pilot_enabled = pilot_enabled
        self._pilots = [None] * self.n
        self._last_pilot_update = self._last_prepare = self._last_delivery = -1
        self._prepared_slot = None
        self._prepared = ()
        self._prepared_attempted = None
        self._success_bin = np.full(self.n, -1, dtype=np.int64)
        self._row_kind = np.full((duration, self.n), '', dtype='U9')
        self._row_last_event = np.full((duration, self.n), -1, dtype=np.int64)
        self._frame_events = []
        self._row_replacements = []
        self._selection_counts = Counter()
        self._prepared_slots = self._attempted_exchanges = 0
        self._unprepared_legacy_deliveries = 0

    def update_pilots(self, slot, *, current_sites, measured_snr, positions,
                      capture_slot, refreshed, noise, tx,
                      association_epoch_start, last_primary_unavailable):
        """Copy only existing refreshed observations; no new RF read."""
        slot = _slot(slot, 'Pilot refresh')
        if not self.pilot_enabled:
            return
        if slot <= self._last_pilot_update or slot < max(self._last_prepare, self._last_delivery):
            raise ValueError('Pilot refresh requires strict clock order')
        cap = _slot(capture_slot, 'Pilot capture')
        if cap > slot:
            raise ValueError('Pilot capture cannot be in the future')
        sites = _integers(current_sites, self.n, 'Pilot AP', 0, self.links // 3 - 1)
        refreshed = _boolean(refreshed, self.n, 'Available pilot refresh')
        measured = _finite(measured_snr, (self.n, 3), 'Existing measured SNR')
        positions = _finite(positions, (self.n, 2), 'Public capture pose')
        noise = _finite(noise, (3,), 'Public noise')
        tx = _finite(tx, (3,), 'Public AP power')
        epochs = _integers(association_epoch_start, self.n, 'Association epoch', 0, slot)
        down = _integers(last_primary_unavailable, self.n, 'Last primary unavailability', -1, slot)
        with np.errstate(over='ignore', invalid='ignore'):
            rssi = measured + noise + tx - 15.
            values = rssi.astype('f4'); xy = positions.astype('f4')
        if not np.isfinite(values).all() or not np.isfinite(xy).all():
            raise ValueError('Pilot values and poses must fit finite float32 transport')
        replacements = []
        for i in np.flatnonzero(refreshed):
            i = int(i)
            replacements.append((i, {'ap': int(sites[i]), 'capture_slot': cap,
                'refresh_slot': slot, 'position': tuple(float(v) for v in xy[i]),
                'rssi_dbm': tuple(float(v) for v in values[i]),
                'heard': tuple(bool(v) for v in (rssi[i] >= PILOT_THRESHOLD_DBM)),
                'source_measured_snr_db': tuple(float(v) for v in measured[i]),
                'source_noise_dbm': tuple(float(v) for v in noise),
                'source_ap_tx_dbm': tuple(float(v) for v in tx),
                'association_epoch_at_refresh': int(epochs[i]),
                'last_primary_unavailable_at_refresh': int(down[i])}))
        for i, record in replacements:
            self._pilots[i] = record
        self._last_pilot_update = slot

    def _snapshot_scan(self, report, slot):
        if not isinstance(report, (tuple, list)) or len(report) != 6:
            raise ValueError('Malformed ready scan report')
        cap, complete, xy, value, heard, bound = report
        cap, complete = _slot(cap, 'Scan capture'), _slot(complete, 'Scan completion')
        xy = _finite(xy, (2,), 'Scan capture pose')
        value, heard = np.asarray(value), np.asarray(heard)
        if (not cap < complete <= slot or value.shape != (self.links,)
                or value.dtype != np.dtype('f4') or not np.isfinite(value).all()
                or heard.shape != (self.links,) or heard.dtype != np.bool_
                or isinstance(bound, (bool, np.bool_))
                or not isinstance(bound, (int, float, np.integer, np.floating))
                or not np.isfinite(bound)):
            raise ValueError('Ready scan clocks or canonical payload differ')
        return _ScanSnapshot(cap, complete, tuple(float(v) for v in xy),
                             value.tobytes(), heard.tobytes(), float(bound), report)

    def _pilot_reason(self, i, slot, bin_index, history_index, primary, sites, epochs, down):
        if not self.pilot_enabled:
            return 'pilot_disabled'
        if self.history.scanned[history_index, i]:
            return 'full_scan_already_in_bin'
        if self._success_bin[i] == bin_index:
            return 'pilot_success_quota'
        if not primary[i]:
            return 'primary_unavailable'
        record = self._pilots[i]
        if record is None:
            return 'no_existing_pilot'
        if record['ap'] != sites[i]:
            return 'pilot_association_mismatch'
        cap, refresh = record['capture_slot'], record['refresh_slot']
        if not 0 <= slot - refresh < PILOT_REFRESH_SLOTS or not 0 <= slot - cap < 2 * PILOT_REFRESH_SLOTS:
            return 'stale_pilot'
        if cap <= down[i]:
            return 'pilot_capture_while_primary_unavailable'
        if cap < epochs[i]:
            return 'pilot_capture_before_association'
        if not any(record['heard']):
            return 'no_detected_pilot'
        return 'pilot_selected'

    def _pilot_frame(self, i, slot, sites, primary, epochs, down, event_index):
        record = self._pilots[i]
        mask = sum(int(h) << b for b, h in enumerate(record['heard']))
        values = tuple(record['rssi_dbm'][b] if h else 0.
                       for b, h in enumerate(record['heard']))
        payload = PILOT_STRUCT.pack(PILOT_VERSION, PILOT_KIND, i, record['ap'],
            record['capture_slot'], record['refresh_slot'], *record['position'], mask, *values)
        decoded = decode_pilot_payload(payload, robot_count=self.n, ap_count=self.links // 3)
        archival_record = {key: list(value) if isinstance(value, tuple) else value
                           for key, value in record.items()}
        event = {'slot': slot, 'robot': i, 'kind': 'pilot', 'selection': 'pilot_selected',
            **archival_record, 'current_site_at_prepare': int(sites[i]),
            'primary_available_at_prepare': bool(primary[i]),
            'association_epoch_start': int(epochs[i]), 'last_primary_unavailable': int(down[i]),
            'frame_bytes': CONTROL_FRAME_BYTES, 'reservation_bytes': REPORT_RESERVED_BYTES,
            'telemetry_bytes': len(payload), 'wire_payload_bytes': len(payload),
            'payload_hex': payload.hex(), 'wire_payload_sha256': hashlib.sha256(payload).hexdigest(),
            'wire_encoding': PILOT_FORMAT, 'decoded_position': list(decoded['position']),
            'decoded_rssi_dbm': list(decoded['rssi_dbm']), 'decoded_heard': list(decoded['heard']),
            'prepared_before_delivery_draw': True, 'delivered': None,
            'assimilated': False, 'retained_in_history': False}
        return _Prepared(i, 'pilot', event_index, pilot_payload=payload), event

    def prepare_frames(self, slot, *, attempted, current_sites, primary_available,
                       association_epoch_start, last_primary_unavailable):
        """Freeze chosen payloads before the control-delivery random draw.

        One successful pilot per robot/bin is allowed; failed exchanges may
        try again using their existing paid control frames. No report means
        unused reserved bytes, not an extra or shortened frame.
        """
        slot = _slot(slot, 'Frame preparation')
        if self._prepared_slot is not None or slot <= self._last_prepare or slot < self._last_delivery:
            raise ValueError('Prepare one exchange snapshot per slot in order')
        if slot // 100 >= self.duration:
            raise ValueError('Frame preparation is outside the episode')
        attempted = _boolean(attempted, self.n, 'Control attempts')
        primary = _boolean(primary_available, self.n, 'Primary radio availability')
        sites = _integers(current_sites, self.n, 'Current AP', 0, self.links // 3 - 1)
        epochs = _integers(association_epoch_start, self.n, 'Association epoch', 0, slot)
        down = _integers(last_primary_unavailable, self.n, 'Last primary unavailability', -1, slot)
        selected, events, reasons = [], [], Counter()
        bin_index = slot // 100; history_index = 60 + bin_index
        for i in np.flatnonzero(attempted):
            i = int(i); report = self.pending.get(i)
            index = len(self._frame_events) + len(events)
            if report is not None and report[1] <= slot:
                scan = self._snapshot_scan(report, slot)
                event = {'slot': slot, 'robot': i, 'kind': 'full_scan',
                    'selection': 'ready_full_scan_priority', 'capture_slot': scan.capture,
                    'complete_slot': scan.complete, 'position': list(scan.position),
                    'frame_bytes': CONTROL_FRAME_BYTES, 'reservation_bytes': REPORT_RESERVED_BYTES,
                    'telemetry_bytes': REPORT_RESERVED_BYTES, 'wire_payload_bytes': None,
                    'payload_hex': None, 'wire_encoding': 'legacy_full_scan_reservation',
                    'decoded_payload_sha256': hashlib.sha256(scan.value + scan.heard).hexdigest(),
                    'prepared_before_delivery_draw': True, 'delivered': None,
                    'assimilated': False, 'retained_in_history': False}
                selected.append(_Prepared(i, 'full_scan', index, scan=scan)); events.append(event)
                reasons['ready_full_scan_priority'] += 1
                continue
            reason = self._pilot_reason(i, slot, bin_index, history_index, primary, sites, epochs, down)
            if reason != 'pilot_selected':
                reasons[reason] += 1
                continue
            frame, event = self._pilot_frame(i, slot, sites, primary, epochs, down, index)
            selected.append(frame); events.append(event); reasons[reason] += 1
        self._frame_events.extend(events)
        self._selection_counts.update(reasons)
        self._prepared = tuple(selected)
        self._prepared_attempted = attempted.copy()
        self._prepared_slot = self._last_prepare = slot
        self._prepared_slots += 1; self._attempted_exchanges += int(attempted.sum())
        return tuple(deepcopy(event) for event in events)

    def _mark_row(self, frame, bin_index, slot):
        previous = int(self._row_last_event[bin_index, frame.robot])
        if previous >= 0:
            old = self._frame_events[previous]
            old['retained_in_history'] = False
            old['superseded_slot'] = slot
            old['superseded_by_event_index'] = frame.event_index
            self._row_replacements.append({'slot': slot, 'robot': frame.robot,
                'delivery_bin': bin_index, 'previous_event_index': previous,
                'previous_kind': old['kind'], 'replacement_event_index': frame.event_index,
                'replacement_kind': frame.kind})
        self._row_last_event[bin_index, frame.robot] = frame.event_index
        self._row_kind[bin_index, frame.robot] = frame.kind
        self._frame_events[frame.event_index]['assimilated'] = True
        self._frame_events[frame.event_index]['retained_in_history'] = True

    def deliver(self, slot, delivered):
        """Commit only immutable payloads from successful prepared exchanges."""
        slot = _slot(slot, 'Delivery')
        delivered = _boolean(delivered, self.n, 'Successful control deliveries')
        if slot <= self._last_delivery:
            raise ValueError('Delivery requires strict clock order')
        if self._prepared_slot is None:
            if self.pilot_enabled:
                raise RuntimeError('Prepare pilot payloads before the delivery outcome')
            super().deliver(slot, delivered)
            self._last_delivery = slot
            self._unprepared_legacy_deliveries += 1
            return
        if slot != self._prepared_slot or (delivered & ~self._prepared_attempted).any():
            raise ValueError('Delivery must match its prepared attempted exchange')
        # Decode every successful packet before changing any fleet history.
        decoded = {}
        for frame in self._prepared:
            if delivered[frame.robot] and frame.kind == 'pilot':
                decoded[frame.event_index] = decode_pilot_payload(frame.pilot_payload,
                    robot_count=self.n, ap_count=self.links // 3)
        bin_index = slot // 100; t = 60 + bin_index
        for frame in self._prepared:
            i = frame.robot; event = self._frame_events[frame.event_index]
            event['delivered'] = bool(delivered[i])
            if not delivered[i]:
                continue
            event['delivery_slot'] = slot
            event['first_usable_decision'] = bin_index + 1
            if frame.kind == 'full_scan':
                scan = frame.scan
                value = np.frombuffer(scan.value, dtype='f4')
                heard = np.frombuffer(scan.heard, dtype=bool)
                self.history.positions[t, i] = scan.position
                self.history.value[t, i] = value
                self.history.heard[t, i] = heard
                self.history.bound[t, i] = scan.bound
                self.history.scanned[t, i] = True
                if self.pending.get(i) is scan.origin:
                    del self.pending[i]
                self.events.append({'robot': i, 'measurement_slot': scan.capture,
                    'complete_slot': scan.complete, 'delivery_slot': slot,
                    'first_usable_decision': bin_index + 1, 'heard': int(heard.sum()),
                    'payload_sha256': hashlib.sha256(scan.value + scan.heard).hexdigest()})
                self._mark_row(frame, bin_index, slot)
                continue
            self._success_bin[i] = bin_index
            if self.history.scanned[t, i]:
                event['assimilation_rejection'] = 'full_scan_row_priority'
                continue
            record = decoded[frame.event_index]
            value = np.zeros(self.links, dtype='f4')
            heard = np.zeros(self.links, dtype=bool)
            for band, detected in enumerate(record['heard']):
                if detected:
                    link = record['ap'] * 3 + band
                    value[link] = record['rssi_dbm'][band]
                    heard[link] = True
            self.history.positions[t, i] = record['position']
            self.history.value[t, i] = value
            self.history.heard[t, i] = heard
            self.history.bound[t, i] = 0.
            self.history.scanned[t, i] = False
            event['decoded_payload_sha256'] = hashlib.sha256(value.tobytes() + heard.tobytes()).hexdigest()
            self._mark_row(frame, bin_index, slot)
        self._prepared_slot = None
        self._prepared = ()
        self._prepared_attempted = None
        self._last_delivery = slot

    def pilot_summary(self):
        """Return payload accounting separately from the unchanged scan summary."""
        pilots = [e for e in self._frame_events if e['kind'] == 'pilot']
        scans = [e for e in self._frame_events if e['kind'] == 'full_scan']
        paid_pilots = sum(e['delivered'] is True for e in pilots)
        paid_scans = sum(e['delivered'] is True for e in scans)
        attempted_bytes = len(pilots) * PILOT_PAYLOAD_BYTES + len(scans) * REPORT_RESERVED_BYTES
        delivered_bytes = paid_pilots * PILOT_PAYLOAD_BYTES + paid_scans * REPORT_RESERVED_BYTES
        return {'schema': 1, 'enabled': self.pilot_enabled,
            'mode': 'positive_only_existing_pilots',
            'quota': 'one_successful_pilot_per_robot_per_delivery_second',
            'control_frame_bytes': CONTROL_FRAME_BYTES,
            'reserved_bytes_per_control_frame': REPORT_RESERVED_BYTES,
            'pilot_wire_format': PILOT_FORMAT, 'pilot_payload_bytes': PILOT_PAYLOAD_BYTES,
            'full_scan_wire_bytes': None,
            'full_scan_cost_scope': 'legacy_opaque_128_byte_reservation_without_wire_serialization',
            'prepared_slots': self._prepared_slots,
            'control_exchanges_attempted': self._attempted_exchanges,
            'control_bytes_attempted': self._attempted_exchanges * CONTROL_FRAME_BYTES,
            'reserved_bytes_attempted': self._attempted_exchanges * REPORT_RESERVED_BYTES,
            'pilot_reports_attempted': len(pilots), 'pilot_reports_delivered': paid_pilots,
            'pilot_reports_failed': sum(e['delivered'] is False for e in pilots),
            'pilot_reports_assimilated': sum(e['assimilated'] for e in pilots),
            'pilot_rows_retained': sum(e['retained_in_history'] for e in pilots),
            'pilot_rows_superseded_by_full_scan': sum(e['previous_kind'] == 'pilot'
                and e['replacement_kind'] == 'full_scan' for e in self._row_replacements),
            'pilot_bytes_attempted': len(pilots) * PILOT_PAYLOAD_BYTES,
            'pilot_bytes_delivered': paid_pilots * PILOT_PAYLOAD_BYTES,
            'full_scan_reports_attempted': len(scans), 'full_scan_reports_delivered': paid_scans,
            'full_scan_reserved_bytes_attempted': len(scans) * REPORT_RESERVED_BYTES,
            'full_scan_reserved_bytes_delivered': paid_scans * REPORT_RESERVED_BYTES,
            'telemetry_budget_bytes_attempted': attempted_bytes,
            'telemetry_budget_bytes_delivered': delivered_bytes,
            'maximum_telemetry_bytes_per_exchange': max((e['telemetry_bytes']
                for e in self._frame_events), default=0),
            'selection_counts': dict(sorted(self._selection_counts.items())),
            'unprepared_legacy_deliveries': self._unprepared_legacy_deliveries,
            'new_rf_reads': 0, 'extra_control_frames': 0,
            'retransmission_scope': 'existing_600_byte_control_exchanges_with_unchanged_retry_airtime',
            'frame_events': deepcopy(self._frame_events),
            'row_replacements': deepcopy(self._row_replacements)}
