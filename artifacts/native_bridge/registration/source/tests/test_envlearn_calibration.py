"""Synthetic recovery check for the optional E7 calibration study."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wirelessjev.envlearn import Survey
from wirelessjev.twin.radio import AP


def test_censored_geometry_calibration_recovers_latent_loss_law():
    path = Path(__file__).resolve().parents[1] / "docs/experiments/e7_calibration_study.py"
    spec = importlib.util.spec_from_file_location("e7_calibration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scene = SimpleNamespace(aps=[AP("site", 15.0, 12.0, 36, "5g")])
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 120, size=12000)
    y = rng.uniform(0, 80, size=12000)
    ap = np.zeros(len(x), dtype=int)
    model = module.CalibratedMap(scene, 6.0)
    X, tx = model.features(x, y, ap)
    coefficients = np.array([43.0, 28.0, 0.8])
    latent = tx - X @ coefficients
    measured = latent + rng.normal(0, 4, size=len(x))
    heard = measured >= -90
    absent = np.flatnonzero(~heard)[::3]
    survey = Survey(x=x[heard].tolist(), y=y[heard].tolist(), ap=ap[heard].tolist(),
        rssi=measured[heard].tolist(), cx=x[absent].tolist(), cy=y[absent].tolist(),
        cap=ap[absent].tolist(), cbound=[-90.0] * len(absent), censor_weight=3.0)
    model.calibrate(survey)
    prediction = model.prior_dl(0, x, y)
    observed = module.CalibratedMap(scene, 6.0, censored=False)
    observed.calibrate(survey)
    assert np.sqrt(np.mean((prediction - latent) ** 2)) < 0.5
    assert np.sqrt(np.mean((prediction - latent) ** 2)) < np.sqrt(
        np.mean((observed.prior_dl(0, x, y) - latent) ** 2))
    assert abs(model.rack_scale["5g"] - coefficients[2]) < 0.1
