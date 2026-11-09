"""Genuine local scan completion/capture ages; no scientific RF worlds."""
import numpy as np
import pytest
from wirelessjev.closed_loop.telemetry import DeliveredScans
from wirelessjev.fleetwm.data import Observations
from wirelessjev.jepa_repair.telemetry import TimedDeliveredScans, UNKNOWN_CAPTURE


def warm(n=1, aps=3):
    return Observations(np.zeros((60, n, 2), dtype='f4'), np.full((60, n, aps * 3), -100., dtype='f4'),
                        np.zeros((60, n, aps * 3), dtype=bool), np.full((60, n, aps * 3), -90., dtype='f4'),
                        np.zeros((60, n), dtype=bool))


def observe(history, timestamp, link, value=-72.):
    history.value[timestamp, 0, link] = value
    history.heard[timestamp, 0, link] = True
    history.scanned[timestamp, 0] = True


def recovery(scans, slot, *, age=500, current=None, measured=None):
    current = np.zeros(scans.n, dtype=int) if current is None else current
    measured = np.full((scans.n, 3), -100.) if measured is None else measured
    return scans.recovery_neighbors(slot, current, measured, np.full(3, -95.), np.full(3, 20.), age)


def scan(scans):
    rssi = np.array([[[-80., -80., -80.], [-55., -55., -55.], [-120., -120., -120.]]])
    scans.scan(0, np.array([0]), np.array([[1., 2.]]), rssi, np.array([9]))


def test_default_methods_history_and_rng_are_exactly_the_predecessor():
    initial = warm(2, 4)
    initial.heard[::5, :, :3] = True; initial.value[::5, :, :3] = -65.; initial.scanned[::5] = True
    old, new = DeliveredScans(initial, 4, 999), TimedDeliveredScans(initial, 4, 999)
    current = np.array([0, 1]); measured = np.full((2, 3), 8.)
    noise, tx = np.full(3, -95.), np.full(3, 20.)
    for decision in range(3):
        slot = decision * 100; positions = np.array([[decision, 2.], [3., decision]])
        for instance in (old, new):
            instance.pose(decision, positions)
            instance.scan(slot, np.array([0, 1]), positions, np.full((2, 4, 3), -60. - decision), np.full(2, slot + 9))
            instance.complete(slot + 8)
        np.testing.assert_array_equal(old.neighbors(current, measured, noise, tx), new.neighbors(current, measured, noise, tx))
        for instance in (old, new):
            instance.complete(slot + 9); instance.deliver(slot + 10, np.array([True, decision % 2 == 0]))
        np.testing.assert_array_equal(old.local, new.local)
        np.testing.assert_array_equal(old.neighbors(current, measured, noise, tx), new.neighbors(current, measured, noise, tx))
        assert old.summary() == new.summary()
    for field in ('positions', 'value', 'heard', 'bound', 'scanned'):
        np.testing.assert_array_equal(getattr(old.before(3), field), getattr(new.before(3), field))
    np.testing.assert_array_equal(old.count, new.count)
    assert set(old.pending) == set(new.pending)


def test_warm_capture_uses_actual_bin_and_masks_stale_unknown_bands():
    initial = warm(); observe(initial, 59, 3, -72.); observe(initial, 54, 4, -75.)
    scans = TimedDeliveredScans(initial, 4, 999)
    assert scans.local_capture_slots[0, 3] == -100 and scans.local_capture_slots[0, 4] == -600
    assert scans.local_capture_slots[0, 5] == UNKNOWN_CAPTURE
    now = recovery(scans, 0)
    assert now[0, 1, 0] == -72. and np.isneginf(now[0, 1, 1:]).all()
    assert np.isneginf(now[0, 2]).all()
    assert recovery(scans, 400)[0, 1, 0] == -72.
    assert np.isneginf(recovery(scans, 401)[0, 1]).all()


def test_completion_uses_capture_time_and_never_invents_remote_delivery():
    scans = TimedDeliveredScans(warm(), 4, 999); scan(scans)
    assert 0 in scans.local_pending
    scans.complete(8)
    assert not scans.local_capture_heard.any() and not scans.local_completion_events
    assert np.isneginf(recovery(scans, 8)[0, 1:]).all()
    scans.complete(9)
    assert scans.local_capture_heard[0, 3:6].all() and (scans.local_capture_slots[0, 3:6] == 0).all()
    assert np.isfinite(recovery(scans, 9, age=9)[0, 1]).all()
    assert np.isneginf(recovery(scans, 10, age=9)[0, 1]).all()
    event = scans.local_completion_events[0]
    assert {field: event[field] for field in ('robot', 'measurement_slot', 'complete_slot', 'processed_slot', 'heard_entries')} == dict(
        robot=0, measurement_slot=0, complete_slot=9, processed_slot=9, heard_entries=6)
    assert not scans.events


def test_new_scan_invalidates_unheard_links_and_receiver_miss_does_not_refresh_age():
    initial = warm(); observe(initial, 59, 6, -70.)
    scans = TimedDeliveredScans(initial, 4, 999)
    assert np.isfinite(recovery(scans, 0)[0, 2, 0])
    scan(scans); scans.complete(9)
    assert not scans.local_capture_heard[0, 6:].any()
    assert (scans.local_capture_slots[0, 6:] == UNKNOWN_CAPTURE).all()
    assert np.isneginf(recovery(scans, 9)[0, 2]).all()
    # Select an actual first-draw receiver miss, without patching the RNG.
    seed = next(s for s in range(1000) if np.random.default_rng(np.random.SeedSequence([s, 0, 0, 323000])).random() < .05)
    missed = TimedDeliveredScans(initial, 4, seed); scan(missed); missed.complete(9)
    assert missed.missing == 1 and not missed.local_pending and not missed.local_completion_events
    assert missed.local_capture_slots[0, 6] == -100
    assert np.isneginf(recovery(missed, 401)[0, 2]).all()


def test_unknown_cannot_outrank_dead_current_pilots():
    scans = TimedDeliveredScans(warm(), 4, 999); measured = np.array([[-100., -105., -110.]])
    observed = recovery(scans, 0, current=np.array([2]), measured=measured)
    np.testing.assert_allclose(observed[0, 2], measured[0] - 95 + 20 - 15)
    assert np.isneginf(observed[0, :2]).all() and observed.max(2).argmax(1).tolist() == [2]
    assert np.isfinite(scans.neighbors(np.array([2]), measured, np.full(3, -95.), np.full(3, 20.))).all()


@pytest.mark.parametrize('value', [-1, True, .5, np.nan, np.inf, 10 ** 40])
def test_invalid_clock_or_age_does_not_refresh_metadata(value):
    scans = TimedDeliveredScans(warm(), 4, 999)
    for call in (lambda: scans.complete(value), lambda: recovery(scans, value), lambda: recovery(scans, 0, age=value)):
        with pytest.raises(ValueError, match='nonnegative integer'):
            call()
    assert scans._last_complete_slot == scans._last_recovery_slot == -1 and not scans.local_completion_events


def test_backwards_clock_cannot_expose_future_completed_cache():
    scans = TimedDeliveredScans(warm(), 4, 999); scan(scans); scans.complete(9)
    with pytest.raises(ValueError, match='backwards'):
        recovery(scans, 8)
    recovery(scans, 10)
    with pytest.raises(ValueError, match='backwards'):
        scans.complete(9)


@pytest.mark.parametrize('current', [np.array([0.]), np.array([True]), np.array([3]), np.array([0, 1])])
def test_invalid_current_ap_axes_are_rejected(current):
    with pytest.raises(ValueError, match='AP indices'):
        recovery(TimedDeliveredScans(warm(), 4, 999), 0, current=current)


@pytest.mark.parametrize('field,value', [('measured', np.zeros((1, 2))), ('measured', np.full((1, 3), np.nan)),
                                       ('noise', np.zeros(2)), ('noise', np.full(3, np.inf)),
                                       ('tx', np.zeros(2)), ('tx', np.full(3, np.nan))])
def test_pilot_inputs_require_finite_complete_band_axes(field, value):
    scans = TimedDeliveredScans(warm(), 4, 999)
    values = dict(measured=np.zeros((1, 3)), noise=np.full(3, -95.), tx=np.full(3, 20.)); values[field] = value
    with pytest.raises(ValueError, match='finite fleet/band'):
        scans.recovery_neighbors(0, np.array([0]), values['measured'], values['noise'], values['tx'], 500)
    assert scans._last_recovery_slot == -1


def test_overflowing_current_pilot_conversion_is_rejected_without_clock_mutation():
    scans = TimedDeliveredScans(warm(), 4, 999)
    with pytest.raises(ValueError, match='remain finite'):
        scans.recovery_neighbors(0, np.array([0]), np.full((1, 3), 1e308), np.full(3, 1e308), np.full(3, 1e308), 500)
    assert scans._last_recovery_slot == -1


def test_large_valid_age_clock_does_not_overflow_unknown_capture_sentinel():
    scans = TimedDeliveredScans(warm(), 4, 999)
    result = recovery(scans, np.iinfo(np.int64).max, age=np.iinfo(np.int64).max)
    assert np.isfinite(result[0, 0]).all() and np.isneginf(result[0, 1:]).all()


@pytest.mark.parametrize('damage', ['robot_type', 'robot_range', 'value_shape'])
def test_malformed_pending_metadata_is_checked_before_any_cache_publication(damage):
    scans = TimedDeliveredScans(warm(), 4, 999); scan(scans)
    report = scans.local_pending[0]
    if damage == 'robot_type':
        scans.local_pending[True] = report
    elif damage == 'robot_range':
        scans.local_pending[2] = report
    else:
        scans.local_pending[0] = (*report[:3], report[3][:-1], *report[4:])
    before = scans.local.copy()
    with pytest.raises(ValueError):
        scans.complete(9)
    np.testing.assert_array_equal(scans.local, before)
    assert not scans.local_capture_heard.any() and not scans.local_completion_events
    assert scans._last_complete_slot == -1


def test_local_timeline_reconstructs_actual_completed_payload_and_central_hash():
    import hashlib
    scans = TimedDeliveredScans(warm(), 4, 999); scan(scans)
    original_value = scans.local_pending[0][3].copy()
    original_heard = scans.local_pending[0][4].copy()
    assert not scans.local_completion_events
    scans.complete(8)
    assert not scans.local_completion_events
    scans.complete(9)
    event = scans.local_completion_events[0]
    assert event['heard_links'] == np.flatnonzero(original_heard).tolist()
    rebuilt_value = np.zeros(scans.links, dtype='f4')
    rebuilt_heard = np.zeros(scans.links, dtype=bool)
    rebuilt_heard[event['heard_links']] = True
    rebuilt_value[event['heard_links']] = np.asarray(event['heard_rssi_dbm'], dtype='f4')
    np.testing.assert_array_equal(rebuilt_value, original_value)
    np.testing.assert_array_equal(rebuilt_heard, original_heard)
    assert event['payload_sha256'] == hashlib.sha256(rebuilt_value.tobytes() + rebuilt_heard.tobytes()).hexdigest()
    scans.deliver(12, np.array([True]))
    assert scans.events[0]['payload_sha256'] == event['payload_sha256']


def test_initial_capture_snapshots_are_read_only_and_detached_from_live_updates():
    initial = warm(); observe(initial, 59, 6)
    scans = TimedDeliveredScans(initial, 4, 999)
    slots, heard = scans.local_capture_slots.copy(), scans.local_capture_heard.copy()
    np.testing.assert_array_equal(scans.initial_local_capture_slots, slots)
    np.testing.assert_array_equal(scans.initial_local_capture_heard, heard)
    with pytest.raises(ValueError):
        scans.initial_local_capture_slots[0, 6] = 0
    with pytest.raises(ValueError):
        scans.initial_local_capture_heard[0, 6] = False
    with pytest.raises(ValueError):
        scans.initial_local_capture_slots.flags.writeable = True
    with pytest.raises(ValueError):
        scans.initial_local_capture_heard.flags.writeable = True
    scan(scans); scans.complete(9)
    assert not scans.local_capture_heard[0, 6] and scans.initial_local_capture_heard[0, 6]
    np.testing.assert_array_equal(scans.initial_local_capture_slots, slots)
    np.testing.assert_array_equal(scans.initial_local_capture_heard, heard)
