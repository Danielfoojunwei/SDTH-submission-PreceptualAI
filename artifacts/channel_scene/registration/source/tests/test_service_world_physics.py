"""Exact slot arithmetic and explicit units, independent of learned accuracy."""
import numpy as np
import pytest

from wirelessjev.service_world.physics import (
    AbsoluteServicePrediction,
    db_to_power_ratio,
    dbm_to_mw,
    forced_deadline_misses,
    forced_deadline_release_slots,
    merge_unavailable,
    mw_to_dbm,
    paired_service_advantage,
    power_ratio_to_db,
    project_absolute_service,
    sinr_db,
    sinr_linear,
    thermal_noise_dbm,
)


def test_union_merges_overlap_adjacency_and_duplicates_once():
    intervals = [(20, 21), (4, 9), (0, 4), (2, 8), (0, 4), (12, 12)]
    assert merge_unavailable(intervals) == ((0, 9), (20, 21))
    assert merge_unavailable([]) == ()
    assert forced_deadline_misses([(0, 10), (5, 15)]) == 3
    assert forced_deadline_misses([(0, 4), (4, 9)]) == 1
    np.testing.assert_array_equal(
        forced_deadline_release_slots([(0, 10), (5, 15)]), [0, 5, 10])


def test_forced_scan_and_outage_counts_respect_every_control_phase():
    assert [forced_deadline_misses([(0, 9)], phase=p) for p in range(5)] == [1] * 5
    assert [forced_deadline_misses([(0, 400)], phase=p) for p in range(5)] == [80, 79, 79, 79, 79]
    assert [forced_deadline_misses([(0, 3)], phase=p) for p in range(5)] == [0] * 5
    assert forced_deadline_misses([(0, 5)]) == 1
    assert forced_deadline_misses([(1, 5)]) == 0
    assert forced_deadline_misses([(0, 4)]) == 0


def test_phase_is_periodic_and_negative_slot_boundaries_are_supported():
    expected = [-9, -4, 1, 6]
    np.testing.assert_array_equal(
        forced_deadline_release_slots([(-9, 11)], phase=1), expected)
    np.testing.assert_array_equal(
        forced_deadline_release_slots([(-9, 11)], phase=6), expected)


def test_window_counts_only_expiring_new_or_explicitly_pending_packets():
    # The command released at 0 is already undelivered at slot 3. Only slots
    # 3 and 4 remain usable; their unavailability therefore forces its miss.
    options = dict(start_slot=3, stop_slot=8)
    assert forced_deadline_misses([(3, 5)], **options) == 0
    assert forced_deadline_misses([(3, 5)], pending_release_slots=[0, 0], **options) == 1
    assert forced_deadline_misses([(3, 4)], pending_release_slots=[0], **options) == 0
    assert forced_deadline_misses([(3, 8)], pending_release_slots=[0], **options) == 1
    # The new command at 5 expires at 10, beyond this scoring window.
    assert forced_deadline_misses([(3, 10)], **options) == 0
    assert forced_deadline_misses([(3, 10)], start_slot=3, stop_slot=10,
                                  pending_release_slots=[0]) == 2
    # The boundary at 5 includes a deadline at 5 but no later expiration.
    np.testing.assert_array_equal(
        forced_deadline_release_slots([(3, 10)], start_slot=3, stop_slot=5,
                                      pending_release_slots=[0]), [0])
    assert forced_deadline_misses([(3, 10)], start_slot=3, stop_slot=4,
                                  pending_release_slots=[0]) == 0


def test_union_count_matches_independent_slot_enumeration():
    rng = np.random.default_rng(9326)
    for _ in range(150):
        raw = rng.integers(-10, 35, size=(8, 2))
        intervals = [tuple(sorted(pair)) for pair in raw]
        phase = int(rng.integers(-10, 10))
        period = int(rng.integers(1, 8))
        deadline = int(rng.integers(1, 9))
        start, stop = -3, 28
        unavailable = {t for a, b in intervals for t in range(a, b)}
        # This oracle tests every attempt slot; it shares no merge or ceil
        # division implementation with the production count formula.
        expected = [r for r in range(start, stop) if (r - phase) % period == 0
                    and r + deadline <= stop
                    and all(t in unavailable for t in range(r, r + deadline))]
        options = dict(phase=phase, period_slots=period, deadline_slots=deadline,
                       start_slot=start, stop_slot=stop)
        assert forced_deadline_misses(intervals, **options) == len(expected)
        np.testing.assert_array_equal(forced_deadline_release_slots(intervals, **options), expected)


@pytest.mark.parametrize("intervals, options, error", [
    ([(5, 4)], {}, ValueError),
    ([(0, 1.5)], {}, TypeError),
    ([(False, 5)], {}, TypeError),
    ([(0, 1, 2)], {}, ValueError),
    ([(0, 10)], dict(period_slots=0), ValueError),
    ([(0, 10)], dict(deadline_slots=0), ValueError),
    ([(0, 10)], dict(start_slot=9, stop_slot=8), ValueError),
    ([(0, 10)], dict(pending_release_slots=[0]), ValueError),
    ([(0, 10)], dict(start_slot=5, pending_release_slots=[0]), ValueError),
    ([(0, 10)], dict(start_slot=3, pending_release_slots=[5]), ValueError),
    ([(0, 10)], dict(start_slot=3, pending_release_slots=[1]), ValueError),
])
def test_invalid_slot_units_or_pending_contract_is_rejected(intervals, options, error):
    with pytest.raises(error):
        forced_deadline_misses(intervals, **options)


def test_power_units_round_trip_and_zero_are_explicit():
    assert dbm_to_mw(0) == 1
    assert dbm_to_mw(10) == 10
    assert dbm_to_mw(-np.inf) == 0
    assert mw_to_dbm(0) == -np.inf
    powers = np.array([0., 1e-10, 1e-4, 1., 100.])
    np.testing.assert_allclose(dbm_to_mw(mw_to_dbm(powers)), powers)
    assert db_to_power_ratio(10) == 10
    assert power_ratio_to_db(100) == 20


@pytest.mark.parametrize("call", [
    lambda: mw_to_dbm(-1), lambda: mw_to_dbm(np.nan),
    lambda: dbm_to_mw(np.inf), lambda: dbm_to_mw(1e10),
    lambda: dbm_to_mw(True), lambda: dbm_to_mw(1 + 2j),
    lambda: mw_to_dbm("1"),
])
def test_power_units_reject_invalid_numeric_values(call):
    with pytest.raises((TypeError, ValueError)):
        call()


def test_sinr_uses_linear_milliwatts_with_explicit_interference():
    assert sinr_linear(1., .1, .1) == 5.
    assert sinr_db(1., .1, .1) == pytest.approx(6.98970004336)
    np.testing.assert_allclose(sinr_linear([0., 1., 2.], .1, .1), [0., 5., 10.])
    assert sinr_db(0., 1.) == -np.inf
    # Adding interference and noise powers in dBm would fail this identity.
    assert sinr_db(dbm_to_mw(-60), dbm_to_mw(-90), dbm_to_mw(-90)) == pytest.approx(26.9897000434)


@pytest.mark.parametrize("values", [(-1., 1., 0.), (1., -1., 2.),
    (1., 0., 0.), (1., 1., -1.), (np.inf, 1., 0.)])
def test_sinr_rejects_invalid_power_or_denominator(values):
    with pytest.raises(ValueError):
        sinr_linear(*values)


def test_thermal_noise_uses_hz_and_rounded_room_temperature_density():
    base = thermal_noise_dbm(20e6, noise_figure_db=7)
    assert base == pytest.approx(-93.9897000434)
    step = 10 * np.log10(2)
    assert thermal_noise_dbm(40e6, 7) - base == pytest.approx(step)
    assert thermal_noise_dbm(20e6, 7, 580) - base == pytest.approx(step)
    np.testing.assert_allclose(thermal_noise_dbm([20e6, 40e6], 7), [base, base + step])


@pytest.mark.parametrize("options", [dict(bandwidth_hz=0), dict(bandwidth_hz=-1),
    dict(bandwidth_hz=20e6, noise_figure_db=-1),
    dict(bandwidth_hz=20e6, temperature_k=0)])
def test_thermal_noise_rejects_invalid_units(options):
    with pytest.raises(ValueError):
        thermal_noise_dbm(**options)


def project(misses, gap, **kwargs):
    options = dict(forced_miss_floor=2, released_plus_pending=10,
                   gap_floor_slots=9, gap_upper_slots=30)
    options.update(kwargs)
    return project_absolute_service(misses, gap, **options)


def test_absolute_projection_enforces_floor_capacity_and_known_certainty():
    result = project([-3., 6., 20.], [0., 12., 50.],
                     predicted_hazard=[-.1, .2, 1.2],
                     certain_failure=[False, True, False])
    np.testing.assert_allclose(result.expected_misses, [2., 6., 10.])
    np.testing.assert_allclose(result.maximum_gap_slots, [9., 12., 30.])
    np.testing.assert_allclose(result.hazard, [0., 1., 1.])
    # A gap already in progress before a ten-slot prediction window may
    # legitimately reach 110 slots; window length is not the absolute cap.
    result = project(4., 105., gap_floor_slots=100, gap_upper_slots=110)
    assert result.maximum_gap_slots == 105


@pytest.mark.parametrize("options", [
    dict(forced_miss_floor=11), dict(forced_miss_floor=-1),
    dict(forced_miss_floor=1.5), dict(released_plus_pending=-1),
    dict(released_plus_pending=10.5), dict(gap_floor_slots=31),
    dict(gap_upper_slots=-1), dict(gap_floor_slots=2.5),
    dict(certain_failure=[True]), dict(prediction_kind="advantage"),
])
def test_projection_rejects_inconsistent_bounds_or_wrong_prediction_semantics(options):
    with pytest.raises(ValueError):
        project(4., 10., **options)


def test_probability_one_event_mask_requires_boolean_units():
    with pytest.raises(TypeError):
        project(4., 10., predicted_hazard=.2, certain_failure=1.)


def test_signed_paired_advantage_retains_improvements_after_absolute_projection():
    intervention = project(2., 10., predicted_hazard=.1)
    baseline = project(7., 20., predicted_hazard=.9)
    delta = paired_service_advantage(intervention, baseline)
    assert delta.delta_misses == -5.
    assert delta.delta_maximum_gap_slots == -10.
    assert delta.delta_hazard == pytest.approx(-.8)
    with pytest.raises(TypeError):
        paired_service_advantage(delta, AbsoluteServicePrediction(1., 10.))
