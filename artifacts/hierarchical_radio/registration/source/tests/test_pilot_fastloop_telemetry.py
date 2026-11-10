"""Payload/capture contracts on fixed array fixtures, without RF worlds."""
import hashlib
import struct

import numpy as np
import pytest

from wirelessjev.fleetwm.data import Observations
from wirelessjev.jepa_repair.telemetry import TimedDeliveredScans
from wirelessjev.pilot_fastloop.telemetry import (
    PILOT_FORMAT, PILOT_PAYLOAD_BYTES, PILOT_STRUCT, PilotDeliveredScans,
    decode_pilot_payload,
)


def warm(n=1, aps=3):
    return Observations(np.zeros((60, n, 2), dtype='f4'),
        np.zeros((60, n, 3 * aps), dtype='f4'),
        np.zeros((60, n, 3 * aps), dtype=bool),
        np.full((60, n, 3 * aps), -90., dtype='f4'),
        np.zeros((60, n), dtype=bool))


def observer(n=1, *, enabled=True):
    return PilotDeliveredScans(warm(n), 4, 999, pilot_enabled=enabled)


def update(scans, slot=0, *, capture=None, sites=None, positions=None,
           rssi=None, refreshed=None, epoch=None, down=None):
    n = scans.n
    values = dict(current_sites=np.zeros(n, dtype=int) if sites is None else sites,
        measured_snr=(np.tile([-65., -95., -80.], (n, 1)) if rssi is None else rssi) + 90.,
        positions=np.tile([1.23456789, 8.76543210], (n, 1)) if positions is None else positions,
        capture_slot=max(0, slot - 10) if capture is None else capture,
        refreshed=np.ones(n, dtype=bool) if refreshed is None else refreshed,
        noise=np.full(3, -95.), tx=np.full(3, 20.),
        association_epoch_start=np.zeros(n, dtype=int) if epoch is None else epoch,
        last_primary_unavailable=np.full(n, -1, dtype=int) if down is None else down)
    scans.update_pilots(slot, **values)
    return values


def prepare(scans, slot=0, *, attempted=None, sites=None, primary=None, epoch=None, down=None):
    n = scans.n
    return scans.prepare_frames(slot,
        attempted=np.ones(n, dtype=bool) if attempted is None else attempted,
        current_sites=np.zeros(n, dtype=int) if sites is None else sites,
        primary_available=np.ones(n, dtype=bool) if primary is None else primary,
        association_epoch_start=np.zeros(n, dtype=int) if epoch is None else epoch,
        last_primary_unavailable=np.full(n, -1, dtype=int) if down is None else down)


def ready(scans, *, robot=0, cap=1, complete=9, pose=(21., 22.), link=4):
    value = np.zeros(scans.links, dtype='f4'); value[link] = -72.
    heard = np.zeros(scans.links, dtype=bool); heard[link] = True
    report = (cap, complete, np.array(pose), value, heard, -90.)
    scans.pending[robot] = report
    return report


def fields_equal(left, right):
    for name in ('positions', 'value', 'heard', 'bound', 'scanned'):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


@pytest.mark.parametrize('prepared', [False, True])
def test_disabled_preserves_seeded_full_scan_history_local_cache_and_old_summary(prepared):
    initial = warm(2)
    initial.heard[59, :, 6] = True; initial.value[59, :, 6] = -78.
    initial.scanned[59] = True
    old = TimedDeliveredScans(initial, 4, 999)
    new = PilotDeliveredScans(initial, 4, 999, pilot_enabled=False)
    for decision in range(3):
        slot = 100 * decision
        for scans in (old, new):
            scans.pose(decision, np.array([[decision, 2.], [5., decision]]))
            scans.scan(slot, np.array([0, 1]), np.array([[1., 2.], [3., 4.]]),
                       np.full((2, 3, 3), -66. - decision), np.full(2, slot + 9))
            scans.complete(slot + 9)
        if prepared:
            prepare(new, slot + 10)
        outcome = np.array([True, decision % 2 == 0])
        old.deliver(slot + 10, outcome); new.deliver(slot + 10, outcome)
        fields_equal(old.history, new.history)
        for name in ('local', 'local_capture_slots', 'local_capture_heard', 'count',
                     'initial_local_capture_slots', 'initial_local_capture_heard'):
            np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
        assert old.summary() == new.summary()
        assert old.local_completion_events == new.local_completion_events
        assert old.pending.keys() == new.pending.keys()
    assert new.pilot_summary()['pilot_reports_attempted'] == 0


def test_pilot_wire_is_43_bytes_and_only_decoded_detected_values_enter_history():
    scans = observer(); update(scans, sites=np.array([1]))
    event, = prepare(scans, sites=np.array([1]))
    payload = bytes.fromhex(event['payload_hex'])
    assert len(payload) == PILOT_PAYLOAD_BYTES == struct.calcsize(PILOT_FORMAT) == 43
    decoded = decode_pilot_payload(payload, robot_count=1, ap_count=3)
    assert decoded['ap'] == 1 and decoded['heard'] == (True, False, True)
    assert decoded['rssi_dbm'] == (-65., 0., -80.)
    scans.deliver(0, np.array([True]))
    assert not scans.history.scanned[60, 0]
    assert np.flatnonzero(scans.history.heard[60, 0]).tolist() == [3, 5]
    np.testing.assert_array_equal(scans.history.value[60, 0], [0., 0., 0., -65., 0., -80., 0., 0., 0.])
    assert not scans.history.bound[60, 0].any()
    np.testing.assert_array_equal(scans.history.positions[60, 0], decoded['position'])
    assert not (scans.history.scanned[60, :, None] & ~scans.history.heard[60]).any()
    assert scans.pilot_summary()['frame_events'][0]['decoded_payload_sha256'] == hashlib.sha256(
        scans.history.value[60, 0].tobytes() + scans.history.heard[60, 0].tobytes()).hexdigest()


@pytest.mark.parametrize('damage', ['length', 'version', 'kind', 'robot', 'ap', 'empty',
    'mask', 'clocks', 'overflow_clock', 'nan_pose', 'nan_rssi', 'undetected_nonzero', 'weak_positive'])
def test_typed_decoder_rejects_malformed_packets(damage):
    values = [1, 1, 0, 0, 0, 10, 1., 2., 5, -65., 0., -80.]
    index_values = {'version': (0, 2), 'kind': (1, 2), 'robot': (2, 1), 'ap': (3, 3),
        'empty': (8, 0), 'mask': (8, 8), 'clocks': (4, 11),
        'overflow_clock': (5, 2**63), 'nan_pose': (6, np.nan), 'nan_rssi': (9, np.nan),
        'undetected_nonzero': (10, -95.), 'weak_positive': (9, -90.01)}
    if damage != 'length':
        index, value = index_values[damage]; values[index] = value
    packet = PILOT_STRUCT.pack(*values)
    if damage == 'length':
        packet = packet[:-1]
    with pytest.raises(ValueError):
        decode_pilot_payload(packet, robot_count=1, ap_count=3)


def test_failed_controls_do_not_write_or_consume_success_quota_and_retries_are_paid():
    scans = observer(); update(scans)
    prepare(scans, 0); scans.deliver(0, np.array([False]))
    assert not scans.history.heard[60:].any()
    prepare(scans, 1); scans.deliver(1, np.array([True]))
    assert prepare(scans, 2) == ()
    scans.deliver(2, np.array([True]))
    update(scans, 100); prepare(scans, 100); scans.deliver(100, np.array([True]))
    summary = scans.pilot_summary()
    assert summary['pilot_reports_attempted'] == 3
    assert summary['pilot_reports_delivered'] == summary['pilot_reports_assimilated'] == 2
    assert summary['pilot_reports_failed'] == 1
    assert summary['pilot_bytes_attempted'] == 3 * 43 and summary['pilot_bytes_delivered'] == 2 * 43
    assert summary['control_exchanges_attempted'] == 4 and summary['control_bytes_attempted'] == 4 * 600
    assert summary['reserved_bytes_attempted'] == 4 * 128
    assert summary['selection_counts']['pilot_success_quota'] == 1
    assert summary['new_rf_reads'] == summary['extra_control_frames'] == 0


@pytest.mark.parametrize('slot,first', [(0, 1), (99, 1), (100, 2), (199, 2)])
def test_report_is_invisible_until_strictly_next_decision(slot, first):
    scans = observer(); update(scans, slot)
    prepare(scans, slot); scans.deliver(slot, np.array([True]))
    assert not scans.before(first - 1).heard[60:].any()
    assert scans.before(first).heard[60:].any()
    event, = scans.pilot_summary()['frame_events']
    assert event['first_usable_decision'] == first and event['delivery_slot'] == slot


def test_scan_frame_priority_and_scan_bin_priority():
    scans = observer(); update(scans)
    report = ready(scans)
    event, = prepare(scans, 9)
    assert event['kind'] == 'full_scan' and event['wire_payload_bytes'] is None
    assert event['payload_hex'] is None and event['telemetry_bytes'] == 128
    scans.deliver(9, np.array([True]))
    np.testing.assert_array_equal(scans.history.positions[60, 0], report[2])
    update(scans, 10)
    assert prepare(scans, 10) == ()
    scans.deliver(10, np.array([True]))
    assert scans.pilot_summary()['selection_counts']['full_scan_already_in_bin'] == 1
    assert scans.pilot_summary()['pilot_reports_attempted'] == 0


def test_later_scan_replaces_entire_pilot_row_and_audits_paid_discarded_report():
    scans = observer(); update(scans)
    prepare(scans, 0); scans.deliver(0, np.array([True]))
    report = ready(scans, pose=(55., 66.), link=7)
    prepare(scans, 9); scans.deliver(9, np.array([True]))
    for field, expected in [('positions', report[2]), ('value', report[3]), ('heard', report[4])]:
        np.testing.assert_array_equal(getattr(scans.history, field)[60, 0], expected)
    assert scans.history.scanned[60, 0] and (scans.history.bound[60, 0] == -90.).all()
    summary = scans.pilot_summary()
    assert summary['pilot_reports_delivered'] == 1 and summary['pilot_rows_retained'] == 0
    assert summary['pilot_rows_superseded_by_full_scan'] == 1
    assert summary['telemetry_budget_bytes_attempted'] == summary['telemetry_budget_bytes_delivered'] == 171
    assert summary['maximum_telemetry_bytes_per_exchange'] == 128
    assert summary['full_scan_wire_bytes'] is None
    assert summary['frame_events'][0]['retained_in_history'] is False
    assert summary['frame_events'][1]['retained_in_history'] is True
    assert summary['row_replacements'] == [{'slot': 9, 'robot': 0, 'delivery_bin': 0,
        'previous_event_index': 0, 'previous_kind': 'pilot', 'replacement_event_index': 1,
        'replacement_kind': 'full_scan'}]


def test_prepared_pilot_is_detached_from_input_cache_returned_event_and_delivery_pose():
    scans = observer(); inputs = update(scans)
    event, = prepare(scans)
    expected = bytes.fromhex(event['payload_hex'])
    inputs['measured_snr'][:] = -500.; inputs['positions'][:] = 99.
    scans._pilots[0]['rssi_dbm'] = (1., 2., 3.)
    event['decoded_position'][:] = [999., 999.]
    scans.history.positions[60, 0] = [1000., 1000.]
    scans.deliver(0, np.array([True]))
    decoded = decode_pilot_payload(expected, robot_count=1, ap_count=3)
    np.testing.assert_array_equal(scans.history.positions[60, 0], decoded['position'])
    assert scans.history.value[60, 0, 0] == -65.
    assert scans.pilot_summary()['frame_events'][0]['payload_hex'] == expected.hex()


def test_prepared_scan_snapshot_is_immutable_and_does_not_pop_a_newer_pending_scan():
    scans = observer(); old = ready(scans)
    expected_value, expected_heard, expected_xy = old[3].copy(), old[4].copy(), old[2].copy()
    prepare(scans, 9)
    old[2][:] = 999.; old[3][:] = -22.; old[4][:] = False
    newer = ready(scans, cap=9, complete=19, pose=(88., 99.), link=5)
    scans.deliver(9, np.array([True]))
    assert scans.pending[0] is newer
    np.testing.assert_array_equal(scans.history.positions[60, 0], expected_xy)
    np.testing.assert_array_equal(scans.history.value[60, 0], expected_value)
    np.testing.assert_array_equal(scans.history.heard[60, 0], expected_heard)
    prepare(scans, 19); scans.deliver(19, np.array([True]))
    assert not scans.pending
    assert scans.pilot_summary()['row_replacements'][0]['previous_kind'] == 'full_scan'


@pytest.mark.parametrize('reason,kwargs', [
    ('pilot_association_mismatch', {'sites': np.array([1])}),
    ('primary_unavailable', {'primary': np.array([False])}),
    ('pilot_capture_while_primary_unavailable', {'down': np.array([0])}),
    ('pilot_capture_before_association', {'epoch': np.array([1])}),
])
def test_public_association_and_primary_clocks_guard_capture(reason, kwargs):
    scans = observer(); update(scans)
    assert prepare(scans, 1, **kwargs) == ()
    scans.deliver(1, np.array([True]))
    assert scans.pilot_summary()['selection_counts'][reason] == 1
    assert not scans.history.heard[60:].any()


@pytest.mark.parametrize('refresh,capture,prepare_slot', [(0, 0, 10), (19, 0, 20)])
def test_stale_refresh_or_capture_is_not_reported(refresh, capture, prepare_slot):
    scans = observer(); update(scans, refresh, capture=capture)
    assert prepare(scans, prepare_slot) == ()
    scans.deliver(prepare_slot, np.array([True]))
    assert scans.pilot_summary()['selection_counts']['stale_pilot'] == 1


def test_unheard_bands_and_absent_refresh_remain_unknown_without_new_random_draws():
    scans = observer(2)
    update(scans, rssi=np.full((2, 3), -95.), refreshed=np.array([True, False]))
    assert prepare(scans) == ()
    scans.deliver(0, np.ones(2, dtype=bool))
    assert not scans.history.heard[60:].any()
    assert not scans.history.scanned[60:].any()
    assert scans.pilot_summary()['selection_counts'] == {'no_detected_pilot': 1, 'no_existing_pilot': 1}
    assert not scans.local_completion_events and not scans.count.any()


def test_pilot_only_reporting_never_changes_local_capture_cache_or_scan_rng_state():
    scans = observer()
    old_cache = scans.local.copy(); old_capture = scans.local_capture_slots.copy()
    update(scans); prepare(scans); scans.deliver(0, np.array([True]))
    np.testing.assert_array_equal(scans.local, old_cache)
    np.testing.assert_array_equal(scans.local_capture_slots, old_capture)
    assert not scans.local_capture_heard.any() and not scans.count.any()
    assert not scans.pending and not scans.local_pending and not scans.events
    assert not scans.local_completion_events and scans.summary()['reports_delivered'] == 0


def test_missing_prepare_and_invalid_delivery_order_fail_before_history_mutation():
    scans = observer(); update(scans)
    with pytest.raises(RuntimeError, match='Prepare'):
        scans.deliver(0, np.array([True]))
    prepare(scans, attempted=np.array([False]))
    with pytest.raises(ValueError, match='prepared attempted'):
        scans.deliver(0, np.array([True]))
    with pytest.raises(ValueError, match='prepared attempted'):
        scans.deliver(1, np.array([False]))
    with pytest.raises(ValueError, match='one exchange'):
        prepare(scans, 1)
    assert not scans.history.heard[60:].any()
    scans.deliver(0, np.array([False]))
    with pytest.raises(ValueError, match='strict clock'):
        scans.deliver(0, np.array([False]))


def test_summary_and_returned_event_are_copies_and_preparation_is_transactional():
    scans = observer(2); update(scans)
    good = ready(scans, robot=0)
    bad = ready(scans, robot=1)
    scans.pending[1] = (*bad[:3], bad[3][:-1], *bad[4:])
    with pytest.raises(ValueError):
        prepare(scans, 9)
    assert not scans.pilot_summary()['frame_events'] and scans._prepared_slot is None
    assert scans._last_prepare == -1
    scans.pending[1] = bad
    events = prepare(scans, 9); events[0]['position'][:] = [-999., -999.]
    scans.deliver(9, np.ones(2, dtype=bool))
    summary = scans.pilot_summary(); summary['frame_events'][0]['position'][:] = [-888., -888.]
    np.testing.assert_array_equal(scans.history.positions[60, 0], good[2])
    assert scans.pilot_summary()['frame_events'][0]['position'] == good[2].tolist()


@pytest.mark.parametrize('field,bad', [('capture_slot', 1), ('capture_slot', True),
    ('current_sites', np.array([True])), ('refreshed', np.array([1])),
    ('measured_snr', np.array([[np.inf, 1., 1.]])), ('positions', np.array([[np.nan, 0.]])),
    ('association_epoch_start', np.array([1])), ('last_primary_unavailable', np.array([-2]))])
def test_malformed_refresh_is_rejected_atomically(field, bad):
    scans = observer()
    values = dict(current_sites=np.array([0]), measured_snr=np.ones((1, 3)),
        positions=np.zeros((1, 2)), capture_slot=0, refreshed=np.array([True]),
        noise=np.full(3, -95.), tx=np.full(3, 20.),
        association_epoch_start=np.array([0]), last_primary_unavailable=np.array([-1]))
    values[field] = bad
    with pytest.raises(ValueError):
        scans.update_pilots(0, **values)
    assert scans._pilots == [None] and scans._last_pilot_update == -1


def test_refresh_after_prepare_cannot_change_payload_in_the_same_delivery_slot():
    scans = observer(); update(scans); prepare(scans)
    with pytest.raises(ValueError, match='strict clock'):
        update(scans)
    scans.deliver(0, np.array([True]))
