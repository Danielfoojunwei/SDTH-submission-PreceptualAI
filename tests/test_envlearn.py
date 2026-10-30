"""Regression checks for scan censoring and the radio-map likelihood."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr

from wirelessjev.envlearn import RadioMap, Survey, drive_and_survey, map_error, random_env, truth_scene
from wirelessjev.twin.radio import AP


class ScanNoise:
    def normal(self, loc, scale, size=None):
        if size is None:
            return 0.0
        noise = np.zeros(size)
        noise[8] = 20.0  # Ninth in truth becomes the strongest measured AP.
        return noise

    def random(self, size):
        return np.ones(size)


@pytest.mark.parametrize("visible", [9, 4])
def test_scan_bounds_follow_measured_top_eight_and_detection_floor(visible):
    spec = replace(random_env(9), blockage_p=0.0)
    scene = truth_scene(30009, spec)
    scene.radio().dl[:] = -120.0
    scene.radio().dl[:, :visible] = -40.0 - np.arange(visible)
    _, survey = drive_and_survey(scene, spec, 1 / 12, ScanNoise())
    if visible == 9:
        assert set(survey.ap) == {*range(7), 8}
        assert set(survey.cbound) == {-46.0}
        assert 7 in survey.cap and 8 not in survey.cap
    else:
        assert set(survey.ap) == set(range(4))
        assert set(survey.cbound) == {-90.0}


def constant_map(monkeypatch):
    scene = SimpleNamespace(aps=[AP("ap", 0.0, 0.0, 36, "5g")])
    rm = RadioMap(scene, ap_height=4, D=8)
    monkeypatch.setattr(rm, "calibrate", lambda survey: None)
    return rm


def test_censored_fit_matches_independent_scalar_likelihood(monkeypatch):
    rm = constant_map(monkeypatch)
    x = np.array([10.0]); y = np.array([0.0])
    prior = rm.prior_dl(0, x, y)[0]
    survey = Survey(x=[10.0] * 3, y=[0.0] * 3, ap=[0] * 3,
                    rssi=[prior + 4] * 3, cx=[10.0] * 20,
                    cy=[0.0] * 20, cap=[0] * 20, cbound=[prior] * 20)
    variance = rm.sw2 * (rm.feats(x, y)[0] ** 2).sum()

    def objective(offset):
        return (offset ** 2 / (2 * variance) + 3 * (offset - 4) ** 2 / (2 * rm.sn2)
                - 20 * log_ndtr(-offset / np.sqrt(rm.sn2)))

    expected = minimize_scalar(objective).x
    rm.fit(survey)
    mean, std = rm.predict(x, y)
    assert mean[0, 0] - prior == pytest.approx(expected, abs=1e-4)
    assert np.isfinite(std).all() and (std > 0).all()


@pytest.mark.parametrize("thinning", [1, 3])
def test_repeated_censored_samples_recover_known_latent_signal(monkeypatch, thinning):
    rm = constant_map(monkeypatch)
    rng = np.random.default_rng(42)
    x = np.array([10.0]); y = np.array([0.0])
    truth = rm.prior_dl(0, x, y)[0]
    measured = truth + rng.normal(0, 4, size=4000)
    heard = measured >= truth
    n, nc = int(heard.sum()), len(measured[~heard][::thinning])
    survey = Survey(x=[10.0] * n, y=[0.0] * n, ap=[0] * n,
                    rssi=measured[heard].tolist(), cx=[10.0] * nc,
                    cy=[0.0] * nc, cap=[0] * nc, cbound=[truth] * nc,
                    censor_weight=float(thinning))
    rm.fit(survey)
    mean, _ = rm.predict(x, y)
    assert mean[0, 0] == pytest.approx(truth, abs=0.3)


def test_extreme_censor_bound_stays_finite(monkeypatch):
    rm = constant_map(monkeypatch)
    survey = Survey(x=[10.0] * 3, y=[0.0] * 3, ap=[0] * 3, rssi=[-50.0] * 3,
                    cx=[10.0], cy=[0.0], cap=[0], cbound=[-1000.0])
    rm.fit(survey)
    mean, std = rm.predict(np.array([10.0]), np.array([0.0]))
    assert np.isfinite(mean).all() and np.isfinite(std).all()


def test_error_partitions_expose_inaudible_paths():
    truth = SimpleNamespace(snaps=[SimpleNamespace(dl=np.array([[-80.0, -100.0]]))])
    belief = SimpleNamespace(snaps=[SimpleNamespace(dl=np.array([[-83.0, -112.0]]))])
    assert map_error(truth, belief) == 3.0
    assert map_error(truth, belief, subset="unheard") == 12.0
    assert map_error(truth, belief, subset="all") == pytest.approx(np.sqrt((9 + 144) / 2))
