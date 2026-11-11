"""Explicit client-local recovery authority and event-time action lifecycle.

This controller observes only one client's completed, paid local scans and
current-association pilots. It does not receive forecasts, server history or
hidden RF. Its RSSI thresholds are engineering heuristics, not calibrated
packet-risk bounds or a guarantee that a reachable good link exists.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


def _integer(value, name, *, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f'{name} must be an integer' + (f' >= {minimum}' if minimum is not None else ''))
    return value


def _finite(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return float(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonempty text')
    return value


@dataclass(frozen=True)
class RecoveryCapability:
    """Declared installed simulator capability; not authenticated device authority."""
    robot_id: int
    ap_ids: tuple[int, ...]
    carriers_hz: tuple[float, ...]
    handover_slots: int
    preinstalled: bool = False
    local_authority: bool = False
    autonomous_scan: bool = False
    installed_at: int = 0
    reference_id: str = 'native_zero_gauge'

    def __post_init__(self):
        _integer(self.robot_id, 'Robot', minimum=0)
        _integer(self.handover_slots, 'Handover interruption', minimum=1)
        _integer(self.installed_at, 'Installation slot', minimum=0)
        if type(self.ap_ids) is not tuple or not self.ap_ids or len(set(self.ap_ids)) != len(self.ap_ids):
            raise ValueError('A nonempty unique AP allowlist is required')
        for ap in self.ap_ids: _integer(ap, 'AP', minimum=0)
        if type(self.carriers_hz) is not tuple or not self.carriers_hz or len(set(self.carriers_hz)) != len(self.carriers_hz):
            raise ValueError('A nonempty unique carrier allowlist is required')
        for carrier in self.carriers_hz: _finite(carrier, 'Carrier Hz', positive=True)
        for flag in (self.preinstalled, self.local_authority, self.autonomous_scan):
            if type(flag) is not bool: raise ValueError('Capability flags must be explicit booleans')
        _text(self.reference_id, 'Reference gauge')


@dataclass(frozen=True)
class RecoveryLimits:
    """Slots have the integrator's declared clock; defaults assume 10 ms slots."""
    silence_slots: int = 20
    request_ttl_slots: int = 20
    max_scan_age_slots: int = 500
    max_pilot_age_slots: int = 10
    scan_slots: int = 9
    scan_cadence_slots: int = 500
    dwell_slots: int = 500
    backoff_slots: int = 500
    minimum_rssi_dbm: float = -75.
    margin_db: float = 8.

    def __post_init__(self):
        for name in ('silence_slots', 'request_ttl_slots', 'max_scan_age_slots', 'max_pilot_age_slots',
                     'scan_slots', 'scan_cadence_slots', 'dwell_slots', 'backoff_slots'):
            _integer(getattr(self, name), name, minimum=1)
        _finite(self.minimum_rssi_dbm, 'Minimum RSSI')
        _finite(self.margin_db, 'Margin')
        if self.margin_db < 0 or self.scan_slots > self.scan_cadence_slots:
            raise ValueError('Require nonnegative margin and scan duration <= scan cadence')


@dataclass(frozen=True)
class LocalScanReport:
    robot_id: int
    ap_id: int
    carrier_hz: float
    measurement_slot: int
    complete_slot: int
    scan_started_slot: int
    scan_id: str
    kind: str
    rssi_dbm: float | None = None
    upper_dbm: float | None = None
    reference_id: str = 'native_zero_gauge'

    def __post_init__(self):
        _integer(self.robot_id, 'Robot', minimum=0); _integer(self.ap_id, 'AP', minimum=0)
        _finite(self.carrier_hz, 'Carrier Hz', positive=True)
        for name in ('measurement_slot', 'complete_slot', 'scan_started_slot'):
            _integer(getattr(self, name), name)
        if not self.scan_started_slot <= self.measurement_slot <= self.complete_slot or self.complete_slot <= self.scan_started_slot:
            raise ValueError('A measured scan must follow its start and complete after paid interruption')
        _text(self.scan_id, 'Scan ID'); _text(self.reference_id, 'Reference gauge')
        if self.kind == 'heard':
            _finite(self.rssi_dbm, 'Heard RSSI')
            if self.upper_dbm is not None: raise ValueError('A heard value is not a censor bound')
        elif self.kind == 'censored':
            _finite(self.upper_dbm, 'Censor bound')
            if self.rssi_dbm is not None: raise ValueError('A censor bound is not a point measurement')
        elif self.kind == 'missing':
            if self.rssi_dbm is not None or self.upper_dbm is not None:
                raise ValueError('A missing report supplies no value or bound')
        else:
            raise ValueError('Report kind must be heard, censored or missing')


@dataclass(frozen=True)
class LocalPilot:
    robot_id: int
    ap_id: int
    carrier_hz: float
    measurement_slot: int
    rssi_dbm: float
    reference_id: str = 'native_zero_gauge'

    def __post_init__(self):
        _integer(self.robot_id, 'Robot', minimum=0); _integer(self.ap_id, 'AP', minimum=0)
        _integer(self.measurement_slot, 'Pilot measurement')
        _finite(self.carrier_hz, 'Carrier Hz', positive=True)
        _finite(self.rssi_dbm, 'Pilot RSSI'); _text(self.reference_id, 'Reference gauge')


@dataclass(frozen=True)
class LocalRecoveryAction:
    action_id: str
    robot_id: int
    created_slot: int
    valid_until_slot: int
    current_ap: int
    target_ap: int
    carrier_hz: float
    measurement_slot: int
    scan_id: str
    rssi_dbm: float
    current_rssi_dbm: float | None
    interruption_slots: int


@dataclass(frozen=True)
class LocalScanRequest:
    action_id: str
    robot_id: int
    created_slot: int
    valid_until_slot: int
    interruption_slots: int


class ClientLocalRecovery:
    """Pure controller. Returned requests do nothing until explicitly started.

    The integrator supplies local association/availability and control-delivery
    status. It must cancel stale central pending commands when a local action
    actually starts and pass actual completion/status to ``finish``. Unknown
    completion blocks new actions until explicit ``reconcile``.
    """
    def __init__(self, capability: RecoveryCapability, limits: RecoveryLimits):
        if not isinstance(capability, RecoveryCapability) or not isinstance(limits, RecoveryLimits):
            raise ValueError('Explicit recovery capability and limits are required')
        self.capability, self.limits = capability, limits
        self.reports = {}
        self._scan_rows = {}
        self._scan_clocks = {}
        self._clock = -1
        self._last_delivery = -1
        self._last_scan_start = None
        self._association = None
        self._last_change = -10**30
        self._retry_after = -1
        self._pending = None
        self._active = None
        self._unknown = False
        self.events = []
        self.charged_handover_slots = self.charged_scan_slots = 0
        self._sequence = 0

    def _advance(self, now):
        _integer(now, 'Local event clock', minimum=0)
        if now < self._clock: raise ValueError('Local event clock must not move backwards')
        self._clock = now

    def _link(self, robot, ap, carrier, reference):
        cap = self.capability
        if robot != cap.robot_id or ap not in cap.ap_ids or carrier not in cap.carriers_hz or reference != cap.reference_id:
            raise ValueError('Only own allowed AP/carrier observations in the declared gauge are accepted')

    def observe_scan(self, report: LocalScanReport, now: int):
        self._advance(now)
        if not isinstance(report, LocalScanReport): raise ValueError('Typed actual local scan report required')
        self._link(report.robot_id, report.ap_id, report.carrier_hz, report.reference_id)
        if report.complete_slot > now: raise ValueError('An incomplete or future scan is unavailable')
        if report.complete_slot-report.scan_started_slot < self.limits.scan_slots:
            raise ValueError('Completed local scan lacks the declared paid interruption')
        clock_identity = (report.scan_started_slot, report.measurement_slot, report.complete_slot)
        if report.scan_id in self._scan_clocks and self._scan_clocks[report.scan_id] != clock_identity:
            raise ValueError('One scan identity cannot acquire different event clocks across links')
        self._scan_clocks[report.scan_id] = clock_identity
        identity = (report.scan_id, report.ap_id, report.carrier_hz)
        previous = self._scan_rows.get(identity)
        if previous is not None and previous != report:
            raise ValueError('An existing scan/report identity cannot change its payload')
        self._scan_rows[identity] = report
        key = (report.ap_id, report.carrier_hz)
        old = self.reports.get(key)
        if old is None or (report.measurement_slot, report.complete_slot, report.scan_id) > (old.measurement_slot, old.complete_slot, old.scan_id):
            # A later censored/missing result invalidates the older heard cache.
            self.reports[key] = report
        self._last_scan_start = max(report.scan_started_slot, self._last_scan_start if self._last_scan_start is not None else report.scan_started_slot)

    def decide(self, now: int, current_ap: int, last_delivery_slot: int,
               radio_available: bool, *, current_pilots=(), last_scan_started_slot=None):
        self._advance(now)
        cap, lim = self.capability, self.limits
        _integer(current_ap, 'Current AP', minimum=0)
        _integer(last_delivery_slot, 'Own last successful control delivery')
        if current_ap not in cap.ap_ids or type(radio_available) is not bool:
            raise ValueError('An allowed current association and explicit receiver availability are required')
        if last_delivery_slot > now or last_delivery_slot < self._last_delivery:
            raise ValueError('Own delivery clock must be causal and nondecreasing')
        self._last_delivery = last_delivery_slot
        if last_scan_started_slot is not None:
            _integer(last_scan_started_slot, 'Latest actual scan start')
            if last_scan_started_slot > now: raise ValueError('Future scan starts are unavailable')
            self._last_scan_start = max(last_scan_started_slot, self._last_scan_start if self._last_scan_start is not None else last_scan_started_slot)
        pilot_values = []
        for pilot in current_pilots:
            if not isinstance(pilot, LocalPilot): raise ValueError('Only typed own current-association pilots are accepted')
            self._link(pilot.robot_id, pilot.ap_id, pilot.carrier_hz, pilot.reference_id)
            if pilot.ap_id != current_ap or pilot.measurement_slot > now:
                raise ValueError('A prior/future association pilot cannot measure the current AP')
            if now-pilot.measurement_slot <= lim.max_pilot_age_slots:
                pilot_values.append(pilot.rssi_dbm)
        if self._association is None:
            self._association = current_ap
        elif self._association != current_ap and self._active is None:
            self._association, self._last_change = current_ap, now
            self._pending = None
        if self._pending is not None and now >= self._pending.valid_until_slot:
            self.events.append({'event':'expired','slot':now,'action_id':self._pending.action_id})
            self._pending = None
        enabled = cap.preinstalled and cap.local_authority and now >= cap.installed_at
        if (not enabled or self._unknown or self._active is not None or not radio_available
                or now-last_delivery_slot < lim.silence_slots or now < self._retry_after
                or now-self._last_change < lim.dwell_slots):
            return None
        fresh = [r for r in self.reports.values() if r.kind == 'heard' and r.complete_slot <= now
                 and 0 <= now-r.measurement_slot <= lim.max_scan_age_slots]
        current_values = pilot_values + [r.rssi_dbm for r in fresh if r.ap_id == current_ap]
        current = max(current_values) if current_values else None
        targets = [r for r in fresh if r.ap_id != current_ap and r.rssi_dbm >= lim.minimum_rssi_dbm
                   and (current is None or r.rssi_dbm-current >= lim.margin_db)]
        if self._pending is not None:
            old = self._pending
            supported = (any(r.ap_id == old.target_ap and r.carrier_hz == old.carrier_hz
                             and r.scan_id == old.scan_id and r.measurement_slot == old.measurement_slot
                             and r.rssi_dbm == old.rssi_dbm for r in targets)
                         if isinstance(old, LocalRecoveryAction) else
                         cap.autonomous_scan and (self._last_scan_start is None or now-self._last_scan_start >= lim.scan_cadence_slots))
            if supported: return old
            self.events.append({'event':'invalidated','slot':now,'action_id':old.action_id})
            self._pending = None
        self._sequence += 1
        action_id = f'local-{cap.robot_id}-{self._sequence}'
        if targets:
            best = min(targets, key=lambda r:(-r.rssi_dbm, -r.measurement_slot, r.ap_id, r.carrier_hz, r.scan_id))
            action = LocalRecoveryAction(action_id, cap.robot_id, now,
                min(now+lim.request_ttl_slots, best.measurement_slot+lim.max_scan_age_slots+1), current_ap, best.ap_id,
                best.carrier_hz, best.measurement_slot, best.scan_id, best.rssi_dbm, current,
                cap.handover_slots)
        elif cap.autonomous_scan and (self._last_scan_start is None or now-self._last_scan_start >= lim.scan_cadence_slots):
            action = LocalScanRequest(action_id, cap.robot_id, now, now+lim.request_ttl_slots, lim.scan_slots)
        else:
            return None
        self._pending = action
        self.events.append({'event':'requested','slot':now,'action_id':action.action_id,
                            'kind':'handover' if isinstance(action, LocalRecoveryAction) else 'scan'})
        return action

    def started(self, action, now: int):
        self._advance(now)
        if not isinstance(action, (LocalRecoveryAction, LocalScanRequest)) or self._unknown or self._active is not None or action != self._pending or not action.created_slot <= now < action.valid_until_slot:
            raise ValueError('Only the current unexpired local request can actually start')
        if isinstance(action, LocalRecoveryAction):
            if action.current_ap != self._association:
                raise ValueError('Association changed before local execution')
            report = self.reports.get((action.target_ap, action.carrier_hz))
            if (report is None or report.kind != 'heard' or report.scan_id != action.scan_id
                    or report.measurement_slot != action.measurement_slot or report.rssi_dbm != action.rssi_dbm
                    or not 0 <= now-report.measurement_slot <= self.limits.max_scan_age_slots):
                raise ValueError('Local target evidence changed or expired before actual execution')
        self._active = (action, now+action.interruption_slots)
        self._pending = None
        if isinstance(action, LocalRecoveryAction):
            self.charged_handover_slots += action.interruption_slots
        else:
            self._last_scan_start = now
            self.charged_scan_slots += action.interruption_slots
        self.events.append({'event':'started','slot':now,'until':self._active[1],
                            'action_id':action.action_id,'interruption_slots':action.interruption_slots})

    def cancel(self, action, now: int, reason: str):
        """A pre-execution rejection spends no physical interruption budget."""
        self._advance(now)
        _text(reason, 'Rejection reason')
        if not isinstance(action, (LocalRecoveryAction, LocalScanRequest)) or action != self._pending:
            raise ValueError('Only the current not-started request may be cancelled')
        self._pending = None
        self.events.append({'event':'cancelled','slot':now,'action_id':action.action_id,'reason':reason})

    def cancel_pending(self, action_id: str, now: int, reason: str) -> bool:
        """Clear only the matching not-started request; repeats return False.

        A rejection for an older request must not clear a later queued action.
        Active physical actions require completion/status, never cancellation.
        """
        self._advance(now)
        _text(action_id, 'Rejected action ID'); _text(reason, 'Rejection reason')
        if self._active is not None and self._active[0].action_id == action_id:
            raise ValueError('An already started physical action cannot be cancelled')
        if self._pending is None or self._pending.action_id != action_id:
            return False
        self.cancel(self._pending, now, reason)
        return True

    def finish(self, now: int, observed_ap: int | None):
        self._advance(now)
        if self._active is None or now < self._active[1]:
            raise ValueError('An active local interruption must actually finish before completion')
        action, _ = self._active
        if observed_ap is not None:
            _integer(observed_ap, 'Observed AP', minimum=0)
            if observed_ap not in self.capability.ap_ids: raise ValueError('Observed association outside capability')
        self._active = None
        self._unknown = observed_ap is None
        if observed_ap is not None:
            self._association = observed_ap
        if isinstance(action, LocalRecoveryAction):
            self._last_change = now
            self._retry_after = now+self.limits.backoff_slots
        self.events.append({'event':'unknown' if observed_ap is None else 'completed','slot':now,
                            'action_id':action.action_id,'observed_ap':observed_ap})

    def reconcile(self, now: int, observed_ap: int):
        self._advance(now)
        _integer(observed_ap, 'Observed AP', minimum=0)
        if not self._unknown or observed_ap not in self.capability.ap_ids:
            raise ValueError('Reconciliation requires UNKNOWN and an actual allowed association')
        self._association = observed_ap
        self._unknown = False
        self._last_change = now
        self._retry_after = now+self.limits.backoff_slots
        self.events.append({'event':'reconciled','slot':now,'observed_ap':observed_ap})


@dataclass(frozen=True)
class CentralCommand:
    command_id: str
    robot_id: int
    created_slot: int
    valid_until_slot: int
    current_ap: int
    target_ap: int

    def __post_init__(self):
        _text(self.command_id, 'Command ID')
        for name in ('robot_id','created_slot','current_ap','target_ap'):
            _integer(getattr(self,name),name,minimum=0)
        _integer(self.valid_until_slot,'Command expiry',minimum=self.created_slot+1)
        if self.current_ap == self.target_ap: raise ValueError('A roam command must change association')


@dataclass(frozen=True)
class CommandDelivery:
    command_id: str
    robot_id: int
    created_slot: int
    delivery_slot: int
    from_ap: int

    def __post_init__(self):
        _text(self.command_id,'Delivered command ID')
        for name in ('robot_id','created_slot','delivery_slot','from_ap'):
            _integer(getattr(self,name),name,minimum=0)
        if self.delivery_slot <= self.created_slot:
            raise ValueError('A central command needs its own delivery strictly after creation')


def central_execution_allowed(command: CentralCommand, receipt: CommandDelivery | None,
                              now: int, current_ap: int, radio_available: bool) -> bool:
    """A delivery of another/older command cannot unlock a pending central move."""
    if not isinstance(command,CentralCommand): raise ValueError('Typed central command required')
    _integer(now,'Execution clock',minimum=0); _integer(current_ap,'Current AP',minimum=0)
    if type(radio_available) is not bool: raise ValueError('Explicit radio availability required')
    if receipt is not None and not isinstance(receipt,CommandDelivery): raise ValueError('Typed command delivery required')
    return bool(receipt is not None and radio_available and command.created_slot < receipt.delivery_slot <= now < command.valid_until_slot
        and receipt.command_id == command.command_id and receipt.robot_id == command.robot_id
        and receipt.created_slot == command.created_slot and receipt.from_ap == command.current_ap == current_ap)
