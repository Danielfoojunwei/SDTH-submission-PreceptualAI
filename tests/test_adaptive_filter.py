"""Analytic software fixtures; no measured or simulated scientific episodes."""
from dataclasses import replace
import math

import numpy as np
import pytest

from wirelessjev.adaptive_radio.filter import Belief, FilterConfig, ReportEvent, EventTimeFilter


def initial():
    return Belief(0., np.zeros(8), np.eye(8), np.zeros(3))


def event(identity='unit-rx-event', **kwargs):
    return ReportEvent(identity, 0., np.eye(8)[0], -60., **kwargs)


def test_heard_bin_intersects_real_detector_event():
    raw=event(heard=True,lower_dbm=-60.5,upper_dbm=-59.5,detection_threshold_dbm=-59.9)
    truncated=event(heard=True,lower_dbm=-59.9,upper_dbm=-59.5)
    assert raw.interval==(-59.9,-59.5)
    actual,_=EventTimeFilter(initial()).observe(raw)
    expected,_=EventTimeFilter(initial()).observe(truncated)
    np.testing.assert_allclose(actual.mean,expected.mean)
    np.testing.assert_allclose(actual.covariance,expected.covariance)
    with pytest.raises(ValueError):
        event(heard=True,lower_dbm=-60.5,upper_dbm=-59.5,detection_threshold_dbm=-59.4)


def test_narrow_heard_interval_matches_manual_gaussian_posterior():
    f = EventTimeFilter(initial())
    h = 10.*np.eye(8)[0]
    sigma = 2.
    received = -55.
    precision = np.eye(8)+np.outer(h, h)/sigma**2
    covariance = np.linalg.inv(precision)
    mean = covariance@(h*(received+60.)/sigma**2)
    posterior, diagnostic = f.observe(event(heard=True, lower_dbm=received-1e-4,
                                          upper_dbm=received+1e-4, discrepancy_sigma_db=sigma))
    np.testing.assert_allclose(posterior.mean, mean, atol=2e-8)
    np.testing.assert_allclose(posterior.covariance, covariance, atol=2e-8)
    assert diagnostic['measurement_information_added'] is True
    assert f.metadata['uncertainty_calibrated'] is False
    with pytest.raises(ValueError):
        posterior.mean[0] = 1.


def test_time_ou_and_distance_process_inflation_are_explicit():
    belief = replace(initial(), mean=np.ones(8), covariance=2.*np.eye(8))
    config = FilterConfig(mean_reversion_seconds=5., time_variance_per_second=.1,
                          distance_variance_per_metre=.2)
    f = EventTimeFilter(belief, config=config, reference_mean=np.zeros(8),
                        reference_covariance=3.*np.eye(8))
    posterior = f.advance(2., np.array((3., 4., 0.)))
    rho = math.exp(-2./5.)
    np.testing.assert_allclose(posterior.mean, rho*np.ones(8))
    np.testing.assert_allclose(posterior.covariance,
                               (rho*rho*2+(1-rho*rho)*3+.1*2+.2*5)*np.eye(8))
    # Returning to an old position does not falsely undo process uncertainty.
    returned = f.advance(2., np.zeros(3))
    np.testing.assert_allclose(returned.covariance, posterior.covariance+np.eye(8))
    assert 'not a spatial OU' in f.metadata['distance_forgetting']


def test_unknown_adds_no_measurement_and_ignored_slots_are_not_fake_points():
    f = EventTimeFilter(initial())
    before = f.belief
    after, diagnostic = f.observe(event(lower_dbm=9999., upper_dbm=-9999., censor_upper_dbm=math.nan))
    np.testing.assert_array_equal(after.mean, before.mean)
    np.testing.assert_array_equal(after.covariance, before.covariance)
    assert diagnostic['measurement_information_added'] is False
    assert diagnostic['normalized_innovation'] is None


def test_missing_later_event_only_propagates_declared_prior():
    config = FilterConfig(time_variance_per_second=.1)
    f = EventTimeFilter(initial(), config=config)
    expected = EventTimeFilter(initial(), config=config).advance(3.)
    after, diagnostic = f.observe(replace(event(), timestamp=3.))
    np.testing.assert_array_equal(after.mean, expected.mean)
    np.testing.assert_array_equal(after.covariance, expected.covariance)
    assert diagnostic['measurement_information_added'] is False


def test_weak_censor_has_no_hard_floor_and_no_spurious_complete_data_precision():
    f = EventTimeFilter(initial())
    after, diagnostic = f.observe(event(censored=True, censor_upper_dbm=1000.))
    np.testing.assert_allclose(after.mean, initial().mean, atol=1e-12)
    np.testing.assert_allclose(after.covariance, initial().covariance, atol=1e-12)
    assert diagnostic['observed_information_db_inverse_squared'] == pytest.approx(0.)
    assert not diagnostic['change_flag']


def test_informative_censor_updates_smoothly_with_finite_pd_covariance():
    f = EventTimeFilter(initial())
    after, diagnostic = f.observe(event(censored=True, censor_upper_dbm=-66.,
                                       discrepancy_sigma_db=2.))
    assert after.mean[0] < 0.
    assert after.covariance[0, 0] < 1.
    assert diagnostic['observed_information_db_inverse_squared'] > 0.
    np.linalg.cholesky(after.covariance)
    # The event is an inequality, never an invented exact observation at -66.
    predicted, _ = f.predict_power(np.eye(8)[:1], -60., discrepancy_sigma_db=2.)
    assert predicted[0] != pytest.approx(-66., abs=1e-3)


def test_clock_and_duplicate_rejection_leave_state_unchanged():
    f = EventTimeFilter(initial())
    f.observe(event(heard=True, lower_dbm=-59., upper_dbm=-58.))
    before = f.belief
    with pytest.raises(ValueError, match='Duplicate'):
        f.observe(event())
    assert f.belief is before
    with pytest.raises(ValueError, match='Out-of-order'):
        f.observe(replace(event('older'), timestamp=-1.))
    assert f.belief is before


def test_forecasts_never_feed_back_or_reduce_uncertainty():
    f = EventTimeFilter(initial())
    before = f.belief
    one = f.predict_power(np.eye(8), -60.)
    for _ in range(100):
        other = f.predict_power(np.eye(8), -60.)
    np.testing.assert_array_equal(one, other)
    assert f.belief is before and len(f._seen_ids) == 0


def test_change_flags_and_optional_reset_are_predeclared():
    extreme = event(heard=True, lower_dbm=-1., upper_dbm=0.)
    ordinary = EventTimeFilter(initial())
    _, diagnostic = ordinary.observe(extreme)
    assert diagnostic['change_flag'] and not diagnostic['reset_applied']
    reset = EventTimeFilter(initial(), config=FilterConfig(reset_on_change=True,
                                                          reset_covariance_multiplier=5.))
    after, diagnostic = reset.observe(extreme)
    assert diagnostic['change_flag'] and diagnostic['reset_applied']
    assert after.covariance[1, 1] == pytest.approx(5.)
    assert reset.metadata['reset_enabled'] is True


@pytest.mark.parametrize('violation', ('covariance', 'mask', 'bin', 'time_scale', 'basis'))
def test_invalid_inputs_rejected(violation):
    with pytest.raises((ValueError, TypeError)):
        if violation == 'covariance':
            Belief(0., np.zeros(8), -np.eye(8))
        elif violation == 'mask':
            event(heard=True, censored=True)
        elif violation == 'bin':
            event(heard=True, lower_dbm=-60., upper_dbm=-61.)
        elif violation == 'time_scale':
            FilterConfig(mean_reversion_seconds=-1.)
        else:
            replace(event(), basis=np.zeros(7))
