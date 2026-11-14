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


def test_fleet_capabilities_only_help_and_default_is_unchanged(world):
    pol = {p.name: p for p in policies()}
    nat = replace(pol["native"], gate=False)
    base = simulate(world, nat, collision_lambda=0.01, interference=True)
    again = simulate(world, replace(nat, safe_plan_s=0.0, relay=False, ble=False, oracle=0),
                     collision_lambda=0.01, interference=True)
    assert base["control_misses"] == again["control_misses"]
    assert set(base["stop_fraction_at_s"]) == {"0.2", "1.0", "2.0"}
    s = base["stop_fraction_at_s"]
    assert s["0.2"] >= s["1.0"] >= s["2.0"] >= 0
    plan = simulate(world, replace(nat, safe_plan_s=2.0), collision_lambda=0.01, interference=True)
    assert plan["control_misses"] == base["control_misses"]          # the plan changes stops, not packets
    assert plan["plan_stop_fraction"] <= base["plan_stop_fraction"]
    for extra in (dict(relay=True), dict(ble=True), dict(oracle=3)):
        r = simulate(world, replace(nat, **extra), collision_lambda=0.01, interference=True)
        assert 0 <= r["control_miss"] <= 1 and r["task_throughput"] <= 1


def test_blockage_detector_is_off_by_default_and_counts_detections(world):
    pol = {p.name: p for p in policies()}
    nat = replace(pol["native"], gate=True, cadence_roam=True)
    base = simulate(world, nat, collision_lambda=0.01, interference=True)
    off = simulate(world, replace(nat, block_detect=False, oracle_urgent=False), collision_lambda=0.01, interference=True)
    assert base["control_misses"] == off["control_misses"] and base["blockage_detections"] == 0
    on = simulate(world, replace(nat, block_detect=True), collision_lambda=0.01, interference=True)
    assert on["blockage_detections"] >= 0 and 0 <= on["control_miss"] <= 1
    # A threshold no drop can reach must leave the run untouched.
    never = simulate(world, replace(nat, block_detect=True, block_drop_db=500.0), collision_lambda=0.01, interference=True)
    assert never["blockage_detections"] == 0 and never["control_misses"] == base["control_misses"]
    two = simulate(world, replace(nat, block_detect=True, dual_radio=True, radio2="shared"),
                   collision_lambda=0.01, interference=True)
    assert 0 <= two["control_miss"] <= 1


def test_detector_evaluation_uses_pooled_counts():
    from wirelessjev.native.run_fleet import DETECT_SCENES, detect_arms, detect_eval
    miss = {"nr_30ms": 50, "single": 300, "single+detect": 150, "single+oracle": 100, "shared": 4, "shared+detect": 4}
    rows = [{"scene": s, "arm": a, "condition": c, "control_misses": miss[a], "control_sent": 1_000_000,
             "control_miss": miss[a] / 1e6} for s in DETECT_SCENES for a in detect_arms() for c in ("main", "stress")]
    e = detect_eval(rows)
    assert e["B1"]["pass"] and e["B2"]["pass"] and e["B4"]["pass"] and not e["B5"]["pass"]
    assert abs(e["B1"]["pooled_reduction"] - 0.5) < 1e-9 and abs(e["B3"]["share_of_oracle_headroom"] - 0.75) < 1e-9


def test_telemetry_export_changes_nothing_and_feeds_the_detector_dataset(world):
    from wirelessjev.outage_risk import dataset as ds
    from wirelessjev.outage_risk.models import Logistic, at_false_alarm_rate, auc, brier, yes_no
    pol = {p.name: p for p in policies()}
    nat = replace(pol["native"], gate=True, cadence_roam=True)
    base = simulate(world, nat, collision_lambda=0.01, interference=True)
    tel = []
    again = simulate(world, nat, collision_lambda=0.01, interference=True, telemetry=tel)
    assert again["control_misses"] == base["control_misses"] and again["roams"] == base["roams"]
    tr = ds.collect(world, nat, collision_lambda=0.01, interference=True)
    t_n = len(tr["slot"])
    assert t_n == len(tel) and tr["level"].shape == (t_n, world.n, 3) and tr["blocked"].shape == (t_n, world.n, 3)
    x, y, rule = ds.examples(tr)
    rows = (t_n - ds.WINDOW + 1) * world.n
    assert x.shape == (rows, ds.WINDOW * ds.PER_TICK) and y.shape == (rows, 3) and rule.shape == (rows,)
    assert np.isfinite(x).all() and rule.dtype == bool
    # Scoring helpers on a case with a known answer.
    rng = np.random.default_rng(0)
    xs = rng.normal(size=(4000, 3))
    ys = (xs[:, 0] + 0.3 * rng.normal(size=4000) > 1.0)
    m = Logistic().fit(xs, ys)
    p = m.predict(xs)
    assert auc(p, ys) > 0.95 and brier(p, ys) < brier(np.full(4000, ys.mean()), ys)
    assert auc(np.array([0.1, 0.9, 0.5, 0.5]), np.array([0, 1, 0, 1])) == 0.875
    op = at_false_alarm_rate(p, ys, 0.01)
    assert op["false_alarm_rate"] <= 0.011 and 0 < op["caught"] <= 1
    assert yes_no(np.array([True, False, True]), np.array([True, True, False]))["caught"] == 0.5

