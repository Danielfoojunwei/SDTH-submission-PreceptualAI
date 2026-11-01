"""Native cross-layer simulator: PHY helpers, retry chains and fairness."""
from dataclasses import replace

import numpy as np
import pytest

from wirelessjev.native.phy import LinkCurves, hierarchical_points
from wirelessjev.native.sim import _chain, _mrr_chain, policies, simulate
from wirelessjev.native.world import build_world


@pytest.fixture(scope="module")
def curves():
    return LinkCurves()


@pytest.fixture(scope="module")
def world():
    return build_world(400, duration_s=10.0, n_robots=20)


def test_hierarchical_constellation_unit_energy_and_gray_case():
    for alpha in (2.0, 3.0, 4.0):
        pts = hierarchical_points(alpha)
        assert np.isclose(np.mean(np.abs(pts) ** 2), 1.0)
        assert len(set(np.round(pts, 9))) == 16
    # alpha = 2 is uniform 16-QAM: per-dimension levels {-3, -1, 1, 3}.
    levels = np.unique(np.round(hierarchical_points(2.0).real * np.sqrt(10), 6))
    assert np.allclose(levels, [-3, -1, 1, 3])
    # Base bits pick the quadrant.
    pts = hierarchical_points(3.0)
    for idx in range(16):
        assert np.sign(pts[idx].real) == (1 - 2 * ((idx >> 3) & 1))
        assert np.sign(pts[idx].imag) == (1 - 2 * ((idx >> 2) & 1))


def test_link_curves_monotone_and_6g_doubles_rate(curves):
    for band in ("2g", "5g"):
        th = curves.threshold_db(band)
        assert np.all(np.diff(th) > 0)
        s = curves.success(band, np.zeros(3, dtype=int), np.array([-5.0, 5.0, 30.0]))
        assert s[0] <= s[1] <= s[2] and s[2] > 0.99
    assert np.allclose(curves.rates("6g"), 2 * curves.rates("5g"))


def test_retry_chain_matches_manual(curves):
    n = 4
    link = np.array([1, 1, 0, 2])
    mcs = {b: np.full(n, 5) for b in range(3)}
    snr = np.array([[30.0, 12.0, 5.0]] * n)
    ar = np.arange(n)
    links, mcss = _mrr_chain(link, mcs, ar)
    ok, air = _chain(curves, links, mcss, snr, np.zeros(n, dtype=bool))
    for i in range(n):
        b = link[i]
        p = [curves.success(["2g", "5g", "6g"][b], np.array([m]), np.array([snr[i, b]]))[0]
             for m in (5, 3, 0)]
        assert np.isclose(ok[i], 1 - np.prod([1 - x for x in p]))
    assert np.all(air > 0)


def test_policies_share_hidden_draws_and_static_split(world):
    pol = {p.name: p for p in policies()}
    a = simulate(world, pol["mlo_default"])
    b = simulate(world, pol["mlo_default"])
    assert a == b                                          # deterministic
    split = simulate(world, pol["mlo_split"])
    assert split["roams"] == a["roams"]                    # same roam process, only TTLM differs
    for r in (a, split, simulate(world, pol["native"])):
        assert 0 <= r["control_miss"] <= 1
        assert 0 <= r["video_on_time"] <= 1 or r["video_on_time"] == 0
        assert r["airtime_domain_sum"] >= 0


def test_legacy_roam_cost_creates_outage(world):
    pol = {p.name: p for p in policies()}
    slow = simulate(world, replace(pol["legacy"], roam_cost_s=4.0))
    fast = simulate(world, replace(pol["legacy"], roam_cost_s=0.06))
    assert slow["roams"] == fast["roams"] or slow["roams"] <= fast["roams"]
    assert slow["ctrl_pending_slots_down"] >= fast["ctrl_pending_slots_down"]


def test_learned_map_survey_and_calibration_shapes():
    w = build_world(400, duration_s=5.0, n_robots=10, survey_s=10)
    c = w.map_calibration
    assert c["links"] > 0 and 0 <= c["coverage_90"] <= 1 and c["rmse_db"] > 0
    assert w.sigma_belief.shape == (w.forecast.shape[0], 10, 16, 3)
    assert np.all(w.sigma_belief > 0)


def test_studio_native_section_reads_campaigns():
    from wirelessjev.studio_data import _native_stack
    n = _native_stack()
    assert n is None or all("arms" in v for k, v in n.items() if k != "phy")
