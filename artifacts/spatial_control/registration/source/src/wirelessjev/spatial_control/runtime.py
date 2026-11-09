"""Public-report support and separately authorized own-client recovery adapters.

All full-history support preparation is inside the measured forecast issue
path. The support rule is an AP-neighbor heuristic on measured 2.4/5.2 GHz,
not qualification of the forecast's 6.5 GHz band or packet risk. Client-local
controllers receive only their own actually completed scan payloads/pilots.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
import time

import numpy as np
import torch

from ..adaptive_radio.service import RSSIForecast
from ..jepa_repair.telemetry import TimedDeliveredScans
from ..native.world import HORIZONS
from ..native_bridge.predictor import (AdaptiveNativePredictor, select_context,
                                      NATIVE_CARRIERS_HZ)
from ..native_bridge_ops.operators import PublicQueryGeometry, query_operands
from .memory import RFReport, native_history_memory, inspect_native_support
from .recovery import (RecoveryCapability, RecoveryLimits, ClientLocalRecovery,
                       LocalScanReport, LocalPilot, LocalRecoveryAction, LocalScanRequest)

GAUGE = 'native_zero_device0_assumed'


class ClockedScans(TimedDeliveredScans):
    """Keep the real sensor RNG; archive all local completed report semantics."""
    def __init__(self, warm, duration, seed):
        super().__init__(warm, duration, seed)
        self.local_completed_payloads = []
        self.started_events = []

    def scan(self, slot, ids, positions, rssi, complete):
        if isinstance(slot, (bool, np.bool_)) or not isinstance(slot, (int, np.integer)) or slot < 0:
            raise ValueError('Scan capture needs a nonnegative integer slot')
        ids, positions, rssi, complete = map(np.asarray, (ids, positions, rssi, complete))
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(set(ids.tolist())) != len(ids)
                or np.any((ids < 0) | (ids >= self.n)) or positions.shape != (len(ids), 2)
                or rssi.shape not in ((len(ids), self.links//3, 3), (len(ids), self.links))
                or complete.shape != (len(ids),) or complete.dtype.kind not in 'iu'
                or np.any(complete <= slot) or not np.isfinite(positions).all()
                or not np.isfinite(rssi).all() or slot < self._last_complete_slot):
            raise ValueError('Finite own scan axes, unique robot IDs and future completion slots required')
        before = len(self.events)
        old_pending = {int(robot): self.local_pending.get(int(robot)) for robot in ids}
        ordinals = self.count[ids].copy()
        super().scan(slot, ids, positions, rssi, complete)
        assert len(self.events) == before  # scan initiation is not server delivery
        for index, robot in enumerate(ids):
            report = self.local_pending.get(int(robot))
            available = (report is not None and report is not old_pending[int(robot)]
                         and report[0] == slot)
            self.started_events.append({'robot': int(robot), 'slot': int(slot),
                'complete_slot': int(complete[index]),
                'attempt_ordinal': int(ordinals[index]),
                'measurement_available': bool(available)})

    def complete(self, slot):
        finished = []
        for robot, report in self.local_pending.items():
            measured, complete, xy, value, heard, bound = report
            if complete <= slot:
                finished.append({'robot': int(robot), 'measurement_slot': int(measured),
                    'complete_slot': int(complete), 'processed_slot': int(slot),
                    'position_m': [float(xy[0]), float(xy[1]), 1.0],
                    'value_dbm': value.tolist(), 'heard': heard.tolist(),
                    'upper_dbm': float(bound), 'reference_id': GAUGE,
                    'scan_id': f'scan-{robot}-{measured}',
                    'payload_sha256': hashlib.sha256(value.tobytes()+heard.tobytes()).hexdigest()})
        super().complete(slot)
        self.local_completed_payloads.extend(finished)


class SupportedAdaptivePredictor(AdaptiveNativePredictor):
    """Same 64-token encoder/weights with exact shared geometry operands.

Support indexing is additional permitted delivered information for a separate
admission rule, not a change to the learned representation. Its real CPU cost
is included in step's reported elapsed time and the completion-safe queue.
"""
    def __init__(self, geometry, model, checkpoint_sha256, sensor, *, support=False):
        if type(support) is not bool:
            raise ValueError('Explicit boolean support mode required')
        super().__init__(geometry, model, checkpoint_sha256)
        self.sensor = sensor
        self.support_enabled = support
        self.memories = {}
        self.support_preparation = []
        self.public_geometry = PublicQueryGeometry(geometry.sites, geometry.tx, geometry.noise)

    @torch.no_grad()
    def step(self, past, plan):
        started = time.perf_counter()
        decision = len(self.records)
        if len(past.value) != 60 + decision:
            raise ValueError('Fixed FIFO issue requires its exact strictly earlier history prefix')
        if len(plan) != np.shape(past.value)[1]:
            raise ValueError('Public route and observable history must have the same robot axis')
        self.last_history = {name: np.array(getattr(past, name), copy=True)
            for name in ('positions', 'value', 'heard', 'bound', 'scanned')}
        if self.support_enabled:
            preparing = time.perf_counter()
            self.memories[decision] = native_history_memory(past,
                issue_time_s=float(decision), events=self.sensor.events, retention_age_s=6.0)
            self.support_preparation.append({'decision': decision,
                'elapsed_seconds': time.perf_counter()-preparing,
                'reports': len(self.memories[decision]), 'retention_age_seconds': 6.0,
                'snapshot_metadata': dict(self.memories[decision].snapshot_metadata),
                'encoder_context_changed': False, 'packet_risk_qualified': False})
        context = select_context(past, self.geometry)
        q = query_operands(self.public_geometry, plan)
        def tensor(value):
            return torch.from_numpy(np.array(value, copy=True))[None]
        belief = self.model.encode(tensor(context['context']), tensor(context['known']), tensor(context['scheduled']))
        mean, std = self.model.render(belief, tensor(q['query']), tensor(q['basis']), tensor(q['fspl']), tensor(q['offset']))
        forecast = RSSIForecast(HORIZONS, np.zeros(len(plan), dtype=np.int64),
            mean[0].cpu().numpy().reshape(q['shape']), std[0].cpu().numpy().reshape(q['shape']),
            NATIVE_CARRIERS_HZ, self.source_id, self.checkpoint_sha256)
        self.records.append({'decision': decision, 'history_length': len(past.value),
            'context': context['context'].copy(), 'known': context['known'].copy(),
            'scheduled': context['scheduled'].copy(), 'selected_rows': context['selected_rows'].copy(),
            'plan': np.array(plan, copy=True), 'mean_dbm': forecast.mean_dbm.copy(),
            'std_db': forecast.std_db.copy(), 'belief_mean': belief['mean'][0].cpu().numpy().copy(),
            'belief_covariance': belief['covariance'][0].cpu().numpy().copy()})
        return forecast, 'cold' if decision == 0 else 'warm', time.perf_counter()-started


class SupportAdmission:
    """Require recent nearby own AP evidence on at least one SOURCE band.

No new RF value, service probability, or carrier transport is inferred. A new
own censor bound replaces older heard evidence for that band. An AP can pass
on another actually heard source band; this does not qualify every AP band.
"""
    def __init__(self, utility, raw_predictor, queue, sensor):
        self.utility, self.raw, self.queue, self.sensor = utility, raw_predictor, queue, sensor
        self.records = []
        self.refresh_records = []
        self._event_cursors = {}
        self._local_payload_cursor = 0
        self._local_payload_index = {}

    @staticmethod
    def _slot(value, name):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(name+' must be a nonnegative integer slot')
        return int(value)

    def _refresh_delivered(self, issue, now):
        """Join server deliveries to actual payloads, never local-only RF.

        Only header identities are indexed for local completions. RF values,
        bounds and positions are read solely after a matching server delivery
        is available at this admission clock. Refreshing does not re-encode or
        alter the already issued forecast, and its work is timed by __call__.
        """
        completed = getattr(self.sensor, 'local_completed_payloads', ())
        if len(completed) < self._local_payload_cursor:
            raise ValueError('Local completion ledger was truncated')
        for item in completed[self._local_payload_cursor:]:
            if not isinstance(item, dict):
                raise ValueError('Actual local completion payload identity required')
            try:
                key = (self._slot(item['robot'], 'Payload robot'),
                       self._slot(item['measurement_slot'], 'Payload measurement'),
                       self._slot(item['complete_slot'], 'Payload completion'), item['payload_sha256'])
            except KeyError as exc:
                raise ValueError('Incomplete local payload identity') from exc
            old = self._local_payload_index.get(key)
            if old is not None and old is not item:
                raise ValueError('Ambiguous local payload identity')
            self._local_payload_index[key] = item
        self._local_payload_cursor = len(completed)
        events = self.sensor.events
        cursor = self._event_cursors.get(issue, 0)
        if len(events) < cursor:
            raise ValueError('Server delivery ledger was truncated')
        additions = []
        memory = self.raw.memories[issue]
        sites, robots = len(self.raw.geometry.sites), len(self.raw.records[issue]['plan'])
        previous_delivery = -1
        if cursor:
            previous_delivery = self._slot(events[cursor-1]['delivery_slot'], 'Previous delivery')
        while cursor < len(events):
            event = events[cursor]
            if not isinstance(event, dict):
                raise ValueError('Actual server delivery events required')
            try:
                robot = self._slot(event['robot'], 'Delivered robot')
                measured = self._slot(event['measurement_slot'], 'Delivered measurement')
                complete = self._slot(event['complete_slot'], 'Delivered completion')
                delivered = self._slot(event['delivery_slot'], 'Server delivery')
                payload_hash = event['payload_sha256']
            except KeyError as exc:
                raise ValueError('Incomplete server delivery event') from exc
            if (robot >= robots or not measured < complete <= delivered or delivered < previous_delivery
                    or not isinstance(payload_hash, str) or len(payload_hash) != 64
                    or any(c not in '0123456789abcdef' for c in payload_hash)):
                raise ValueError('Inconsistent delivered report clocks or identity')
            if delivered/100 > now:
                break  # future ledger entries cannot enter the server guard
            if 'first_usable_decision' in event and self._slot(event['first_usable_decision'], 'First usable decision') != delivered//100+1:
                raise ValueError('Server delivery bin and first usable decision disagree')
            previous_delivery = delivered
            if delivered/100 < issue:
                cursor += 1
                continue  # represented by the issued history prefix already
            payload = self._local_payload_index.get((robot, measured, complete, payload_hash))
            if payload is None:
                raise ValueError('Server delivery has no exact completed payload match')
            try:
                processed = self._slot(payload['processed_slot'], 'Payload processing')
                values = np.asarray(payload['value_dbm'])
                heard = np.asarray(payload['heard'])
                position = np.asarray(payload['position_m'], dtype=float)
                bound = float(payload['upper_dbm'])
                reference = payload['reference_id']
            except (KeyError, TypeError) as exc:
                raise ValueError('Incomplete delivered RF payload') from exc
            if (not complete <= processed <= delivered or reference != GAUGE
                    or values.shape != (sites*3,) or values.dtype.kind not in 'fi'
                    or heard.shape != values.shape or heard.dtype != np.bool_
                    or position.shape != (3,) or not np.isfinite(position).all()
                    or not np.isfinite(values).all() or not math.isfinite(bound)):
                raise ValueError('Delivered RF payload axes, gauge or clocks disagree')
            with np.errstate(over='ignore', invalid='ignore'):
                wire_values = values.astype('f4')
            actual_hash = hashlib.sha256(wire_values.tobytes()+heard.tobytes()).hexdigest()
            if (not np.isfinite(wire_values).all() or not np.array_equal(wire_values.astype('f8'), values)
                    or actual_hash != payload_hash
                    or ('heard' in event and self._slot(event['heard'], 'Delivered heard count') != int(heard.sum()))):
                raise ValueError('Delivered RF payload hash or heard count disagrees')
            rows = []
            for site in range(sites):
                for band in (0, 1):
                    link = site*3+band
                    report = RFReport(f'delivered:{robot}:{measured}:{complete}:{delivered}:{site}:{band}:{payload_hash}',
                        robot, site, band, NATIVE_CARRIERS_HZ[band], tuple(position),
                        'heard' if heard[link] else 'censored', GAUGE, delivered/100,
                        (measured/100, measured/100), (delivered/100, delivered/100),
                        value_dbm=float(wire_values[link]) if heard[link] else None,
                        upper_dbm=None if heard[link] else bound,
                        clock_provenance='admission_refresh_exact_server_delivery_and_completed_payload',
                        source_row=(60+delivered//100, robot, site, band))
                    memory.ingest(report, at_s=now)
                    rows.append(asdict(report))
            additions.append({'delivery_event': dict(event), 'local_scan_id': payload.get('scan_id'),
                'payload_processed_slot': processed, 'reports': rows})
            cursor += 1
        self._event_cursors[issue] = cursor
        batch = {'forecast_issue': int(issue), 'consume_time_s': float(now),
            'delivery_events_added': len(additions), 'new_reports': sum(len(x['reports']) for x in additions),
            'future_delivery_events_deferred': len(events)-cursor, 'bindings': additions,
            'server_delivery_cursor': cursor, 'memory_reports_after_refresh': len(memory),
            'locally_completed_undelivered_reports_used': False,
            'encoder_context_changed': False, 'packet_risk_qualified': False}
        self.refresh_records.append(batch)
        return len(self.refresh_records)-1

    def scores(self, forecast, current):
        return self.utility.scores(forecast, current)

    def __call__(self, decision):
        started = time.perf_counter()
        issue = decision.decision
        if (isinstance(issue, (bool, np.bool_)) or not isinstance(issue, (int, np.integer))
                or not 0 <= issue < len(self.raw.records) or issue not in self.raw.memories
                or self.queue.last != issue):
            raise ValueError('Support admission requires the actual newly consumed forecast issue')
        memory = self.raw.memories[issue]
        age = float(self.queue.age)
        if not math.isfinite(age) or age < 0:
            raise ValueError('Forecast completion age must be finite and nonnegative')
        now = float(issue)+age
        allowed = np.array(self.utility(decision), copy=True)
        refresh_batch = self._refresh_delivered(issue, now)
        # Interpolate the public planned route at consume time. This is not
        # a hidden/current true-position observation or a localization bound.
        plan = self.raw.records[issue]['plan']
        targets = np.asarray(decision.candidate)
        if (allowed.dtype != np.bool_ or allowed.shape != (len(plan),)
                or np.shape(plan) != (len(plan), len(HORIZONS), 2) or not np.isfinite(plan).all()
                or targets.shape != (len(plan),) or targets.dtype.kind not in 'iu'
                or np.any((targets < 0) | (targets >= len(self.raw.geometry.sites)))):
            raise ValueError('Admission mask and target IDs must match the forecast robot axis')
        positions = np.array([[np.interp(age, HORIZONS, plan[i,:,axis])
                               for axis in range(2)] for i in range(len(plan))])
        record_start = len(self.records)
        for robot in np.flatnonzero(allowed):
            target = int(decision.candidate[robot]); supporting = []; rows = []
            for band in (0, 1):
                # Select newest own evidence BEFORE the spatial cut. A newer
                # own censor or distant report cannot resurrect an older own
                # heard value merely because that earlier position was close.
                result = inspect_native_support(memory, robot_id=int(robot), site_id=target,
                    band_index=band, position_m=(*positions[robot], 1.0), issue_time_s=now,
                    max_age_s=6.0, max_distance_m=np.finfo(float).max, limit=1)
                own = result.latest_own
                if own is not None:
                    own_index = result.reports.index(own)
                    distance, report_age = result.distances_m[own_index], result.ages_s[own_index]
                    nearby = distance <= 5.0
                    rows.append(asdict(own) | {'query_distance_m': distance,
                        'conservative_age_s': report_age, 'within_spatial_cut': nearby})
                    if nearby and own.kind == 'heard' and own.value_dbm >= -80.0:
                        supporting.append(band)
            allowed[robot] = bool(supporting)
            self.records.append({'decision': int(issue), 'time_s': now, 'robot': int(robot),
                'target': target, 'position_m': [*positions[robot].tolist(), 1.0],
                'own_latest_source_reports': rows, 'heard_support_bands': supporting,
                'permitted': bool(allowed[robot]), 'packet_risk_qualified': False,
                'late_discarded': False,
                'refresh_batch': refresh_batch,
                'position_source': 'public planned route interpolated at forecast completion age',
                'support_information': 'extra delivered history; unchanged 64 encoder tokens'})
        elapsed = time.perf_counter()-started
        # The complete shared planner/admission path is additionally timed and
        # charged by the new kernel's 100ms central-command creation allowance.
        if elapsed > .1:
            allowed[:] = False
            for record in self.records[record_start:]:
                record['permitted'] = False
                record['late_discarded'] = True
        self.utility.records.append({'kind': 'support_admission', 'elapsed_seconds': elapsed,
            'deadline_seconds': .1, 'late_discarded': elapsed > .1,
            'permitted': int(allowed.sum())})
        return allowed


class LocalFleetRuntime:
    """Separately authorized per-client recovery; no forecast/history input.

    All observations come from the attached real simulator sensor. Its local
    completed payloads do not require server delivery, and no warm observation
    is assigned an invented precise capture clock. Delayed execution must also
    pass the kernel's current association/receiver/silence/cadence checks.
    """
    supports_client_local_actions = True
    supports_autonomous_scans = True

    def __init__(self, geometry, sensor, handover_seconds):
        if (not isinstance(sensor, ClockedScans) or type(handover_seconds) not in (int, float)
                or not math.isfinite(handover_seconds) or handover_seconds <= 0):
            raise ValueError('Actual clocked sensor and positive finite handover interruption required')
        self.sensor = sensor
        self.noise = np.array(geometry.noise, dtype=float, copy=True)
        self.tx = np.array(geometry.tx, dtype=float, copy=True)
        sites = np.asarray(geometry.sites)
        if (sites.ndim != 2 or sites.shape[1:] != (2,) or not len(sites)
                or not np.isfinite(sites).all() or self.noise.shape != (3,) or self.tx.shape != (3,)
                or not np.isfinite(self.noise).all() or not np.isfinite(self.tx).all()
                or sensor.n < 1 or sensor.links != len(sites)*3):
            raise ValueError('Finite native geometry and matching actual sensor axes are required')
        self.controllers = [ClientLocalRecovery(RecoveryCapability(i, tuple(range(len(sites))),
            tuple(NATIVE_CARRIERS_HZ), math.ceil(handover_seconds/.01),
            preinstalled=True, local_authority=True, autonomous_scan=True, reference_id=GAUGE),
            RecoveryLimits()) for i in range(sensor.n)]
        self.completed_index = 0
        self.scan_started_slots = [None]*sensor.n
        self.requests = {}
        self.current = None
        self.events = []
        self._clock = -1
        self._tick_slot = None
        self._last_delivery = np.full(sensor.n, -1, dtype=np.int64)
        self._available = np.zeros(sensor.n, dtype=bool)
        # Internal single-use validation stamps, not physical authentication.
        # Full evidence checks run only in the kernel's measured validate batch.
        self._authorizations = {}
        self._state_generation = [0]*sensor.n

    @staticmethod
    def _slot(slot):
        if type(slot) is not int or slot < 0:
            raise ValueError('A nonnegative integer kernel slot is required')
        return slot

    def _advance(self, slot):
        self._slot(slot)
        if slot < self._clock:
            raise ValueError('Local runtime clock must not move backwards')
        self._clock = slot

    def _robot(self, robot):
        if type(robot) is not int or not 0 <= robot < len(self.controllers):
            raise ValueError('A valid own-client index is required')
        return robot

    def _ingest_completed(self, slot):
        """Drain actual completed scans even between the 100 ms callbacks."""
        payloads = self.sensor.local_completed_payloads
        if self.completed_index > len(payloads):
            raise ValueError('The actual completed local-scan ledger cannot shrink')
        for item in payloads[self.completed_index:]:
            robot = self._robot(item['robot'])
            measured, complete, processed = (item[name] for name in
                ('measurement_slot', 'complete_slot', 'processed_slot'))
            for value in (measured, complete, processed): self._slot(value)
            if not measured < complete <= processed <= slot or item.get('reference_id') != GAUGE:
                raise ValueError('Only post-warm actually completed scans in the declared gauge are available')
            heard = np.asarray(item['heard'])
            value = np.asarray(item['value_dbm'])
            if (heard.shape != (self.sensor.links,) or heard.dtype != np.bool_
                    or value.shape != heard.shape or value.dtype.kind not in 'fi'
                    or not np.isfinite(value[heard]).all()
                    or type(item['upper_dbm']) not in (int,float) or not math.isfinite(item['upper_dbm'])):
                raise ValueError('Actual local report values, heard masks and censor bounds are required')
            controller = self.controllers[robot]
            # In this sensor the measurement event is the actual scan-start
            # event; completion/capture clocks are never replaced by tick time.
            for link, was_heard in enumerate(heard):
                controller.observe_scan(LocalScanReport(robot, link//3, NATIVE_CARRIERS_HZ[link%3],
                    measured, complete, measured, item['scan_id'],
                    'heard' if was_heard else 'censored',
                    rssi_dbm=float(value[link]) if was_heard else None,
                    upper_dbm=None if was_heard else item['upper_dbm'], reference_id=GAUGE), slot)
            self._state_generation[robot] += 1
            self.completed_index += 1

    def tick(self, slot, current, positions, radio_available, pilot_snr,
             pilot_slot, pilot_site, last_delivery_slot, sensor):
        self._advance(slot)
        self._authorizations.clear()
        if sensor is not self.sensor:
            raise ValueError('Use this local controller\'s actual sensor')
        n = len(self.controllers)
        current, available, pilots, sample, associated, delivered = map(np.asarray,
            (current, radio_available, pilot_snr, pilot_slot, pilot_site, last_delivery_slot))
        if (current.shape != (n,) or current.dtype.kind not in 'iu' or (current < 0).any()
                or (current >= len(self.controllers[0].capability.ap_ids)).any()
                or available.shape != (n,) or available.dtype != np.bool_
                or pilots.shape != (n,3) or pilots.dtype.kind not in 'fi'
                or not np.isfinite(pilots).all()):
            raise ValueError('Actual finite per-client radio/association/pilot axes are required')
        for array in (sample, associated, delivered):
            if array.shape != (n,) or array.dtype.kind not in 'iu':
                raise ValueError('Actual integer local pilot/delivery clocks and identities required')
        if ((sample > slot).any() or (delivered > slot).any()
                or (delivered < self._last_delivery).any()
                or (associated < 0).any() or (associated >= len(self.controllers[0].capability.ap_ids)).any()):
            raise ValueError('Own local event clocks/associations must remain causal and nondecreasing')
        # Positions are deliberately unused: no remote robot or hidden field query.
        previous_current = self.current
        self.current = np.array(current, copy=True)
        self._available = np.array(available, copy=True)
        self._last_delivery = np.array(delivered, copy=True)
        self._tick_slot = slot
        self._ingest_completed(slot)
        moves = []; scans = []; scan_ids = {}
        for robot, controller in enumerate(self.controllers):
            if controller._active is not None and available[robot] and slot >= controller._active[1]:
                controller.finish(slot, int(current[robot]))
            changed = previous_current is None or previous_current[robot] != current[robot]
            # Healthy clients cannot issue a watchdog request. Avoid constructing
            # 300 pilot dataclasses every tick merely to return no action. A
            # changed association still updates the controller's dwell clock.
            if not changed and controller._pending is None and slot-int(delivered[robot]) < controller.limits.silence_slots:
                continue
            current_pilots = ()
            if available[robot] and associated[robot] == current[robot] and 0 <= slot-int(sample[robot]) <= controller.limits.max_pilot_age_slots:
                converted = pilots[robot]+self.noise+self.tx-15.
                if not np.isfinite(converted).all():
                    raise ValueError('Actual local pilot conversion overflowed RSSI units')
                current_pilots = tuple(LocalPilot(robot, int(current[robot]), NATIVE_CARRIERS_HZ[band],
                    int(sample[robot]), float(converted[band]), GAUGE) for band in range(3))
            request = controller.decide(slot, int(current[robot]), int(delivered[robot]),
                bool(available[robot]), current_pilots=current_pilots,
                last_scan_started_slot=self.scan_started_slots[robot])
            if request is None:
                continue
            self.requests[request.action_id] = request
            if isinstance(request, LocalRecoveryAction):
                moves.append((robot, request.target_ap, request.action_id))
            else:
                scans.append(robot); scan_ids[robot] = request.action_id
        return {'moves': moves, 'scan_robots': scans, 'scan_action_ids': scan_ids}

    def validate(self, robot, target, slot, action_id):
        if isinstance(action_id, str):
            self._authorizations.pop(action_id, None)
        self._advance(slot)
        robot = self._robot(robot)
        if not isinstance(action_id,str) or not action_id:
            return False
        action = self.requests.get(action_id)
        if (action is None or action.robot_id != robot or self.current is None
                or not action.created_slot < slot < action.valid_until_slot):
            return False
        controller = self.controllers[robot]
        cap = controller.capability
        if not (cap.preinstalled and cap.local_authority and slot >= cap.installed_at):
            return False
        self._ingest_completed(slot)
        if (controller._unknown or controller._active is not None
                or controller._pending != action or not self._available[robot]
                or slot-int(self._last_delivery[robot]) < controller.limits.silence_slots):
            return False
        if isinstance(action, LocalRecoveryAction):
            if type(target) is not int or target not in cap.ap_ids:
                return False
            report = controller.reports.get((action.target_ap, action.carrier_hz))
            accepted = bool(self.current[robot] == action.current_ap and target == action.target_ap
                and report is not None and report.kind == 'heard' and report.scan_id == action.scan_id
                and report.measurement_slot == action.measurement_slot and report.rssi_dbm == action.rssi_dbm
                and report.complete_slot <= slot and 0 <= slot-report.measurement_slot <= controller.limits.max_scan_age_slots)
            if accepted:
                self._authorize(robot, target, slot, action)
            return accepted
        last_start = self.scan_started_slots[robot]
        accepted = bool(target is None and cap.autonomous_scan and (last_start is None
            or slot-last_start >= controller.limits.scan_cadence_slots))
        if accepted:
            self._authorize(robot, target, slot, action)
        return accepted

    def _authorize(self, robot, target, slot, action):
        self._authorizations[action.action_id] = (robot, target, slot,
            self._state_generation[robot], self.controllers[robot].capability, action)

    def _consume_authorization(self, robot, target, slot, action_id, action_type):
        """Commit a measured check without another ingestion/validation pass."""
        if not isinstance(action_id, str) or not action_id:
            raise ValueError('Exact validated local action identity required')
        stamp = self._authorizations.pop(action_id, None)
        self._advance(slot)
        robot = self._robot(robot)
        if target is not None and type(target) is not int:
            raise ValueError('Exact validated local target required')
        if (stamp is None or stamp[:3] != (robot, target, slot)
                or stamp[3] != self._state_generation[robot]
                or stamp[4] != self.controllers[robot].capability
                or not isinstance(stamp[5], action_type)
                or self.controllers[robot]._pending != stamp[5]
                or len(self.sensor.local_completed_payloads) != self.completed_index):
            raise ValueError('Local execution requires an unchanged single-use measured validation')
        return stamp[5]

    def executed(self, robot, target, slot, action_id):
        action = self._consume_authorization(robot, target, slot, action_id, LocalRecoveryAction)
        self.controllers[robot].started(action, slot)
        self._state_generation[robot] += 1
        self.events.append({'kind': 'local_handover_started', 'slot': slot, 'action': asdict(action)})

    def scan_started(self, robot, slot, complete, action_id=None):
        action = (None if action_id is None else self._consume_authorization(
            robot, None, slot, action_id, LocalScanRequest))
        self._advance(slot)
        robot = self._robot(robot); self._slot(complete)
        controller = self.controllers[robot]
        if complete-slot < controller.limits.scan_slots:
            raise ValueError('An actual local scan must pay its full interruption')
        previous = self.scan_started_slots[robot]
        if previous is not None and slot-previous < controller.limits.scan_cadence_slots:
            raise ValueError('Background and autonomous scans share the actual scan cadence')
        if action is not None:
            controller.started(action, slot)
        self._state_generation[robot] += 1
        self.scan_started_slots[robot] = slot
        self.events.append({'kind': 'scan_started', 'robot': robot, 'slot': slot,
            'complete_slot': complete, 'action_id': action_id})

    def rejected(self, action_id, reason, slot):
        self._advance(slot)
        if not isinstance(action_id,str) or not action_id or not isinstance(reason,str) or not reason:
            raise ValueError('Explicit rejected local request and reason required')
        self._authorizations.pop(action_id, None)
        action = self.requests.get(action_id)
        if action is not None:
            self.controllers[action.robot_id].cancel_pending(action_id, slot, reason)
        self.events.append({'kind': 'rejected', 'action_id': action_id, 'reason': reason, 'slot': slot})

    def summary(self):
        return {'capability': 'simulated preinstalled local authority; not physical authentication',
            'warm_local_reports_used': False, 'events': self.events,
            'single_use_measured_validation_commits': True,
            'client_events': [dict(event, robot=i) for i,c in enumerate(self.controllers) for event in c.events],
            'charged_local_handover_slots': sum(c.charged_handover_slots for c in self.controllers),
            'charged_autonomous_scan_slots': sum(c.charged_scan_slots for c in self.controllers),
            'realized_scan_costs_equal_across_arms': False,
            'qualified_packet_risk': False}
