"""Native cross-layer simulator: PHY helpers, retry chains and fairness."""
from dataclasses import replace

import numpy as np
import pytest

from wirelessjev.native.phy import LinkCurves, hierarchical_points
from wirelessjev.native.sim import Policy, _chain, _mrr_chain, policies, simulate
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


def test_small_net_is_repeatable_and_rule_state_fires_on_a_sustained_drop():
    from wirelessjev.outage_risk import dataset as ds
    from wirelessjev.outage_risk.models import SmallNet, auc
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 4))
    y = (x[:, 0] > 0.5) & (x[:, 1] > 0.5)                    # an AND: not linearly separable
    a, b = SmallNet(seed=0, epochs=30).fit(x, y).predict(x), SmallNet(seed=0, epochs=30).fit(x, y).predict(x)
    assert np.allclose(a, b) and auc(a, y) > 0.9
    # One robot: steady at 30 dB, a one-reading dip (fade), then a 15 dB drop held for 6 readings.
    lvl = np.full((30, 1, 3), 30.0)
    lvl[8] -= 15.0
    lvl[15:21] -= 15.0
    tr = {"level": lvl, "expected": np.full((30, 1, 3), 30.0), "site": np.zeros((30, 1), dtype=int)}
    fired = ds.rule_state(tr)[:, 0]
    assert not fired[:16].any() and fired[16:21].all() and not fired[21:].any()
    # Arriving at an AP that is already blocked: the map's expectation is the baseline.
    lvl2 = np.full((12, 1, 3), 30.0)
    lvl2[5:] = 14.0
    site2 = np.zeros((12, 1), dtype=int)
    site2[5:] = 1
    tr2 = {"level": lvl2, "expected": np.full((12, 1, 3), 29.0), "site": site2}
    assert ds.rule_state(tr2)[6:, 0].all()


def test_moving_blockers_are_shared_deterministic_and_leave_the_default_alone(world):
    from wirelessjev.native import blockers as bl
    from wirelessjev.native.sim import hidden_draws
    tr = bl.blocker_tracks(7, 300, 5)
    assert tr.shape == (300, 5, 2) and np.array_equal(tr, bl.blocker_tracks(7, 300, 5))
    step = np.linalg.norm(np.diff(tr, axis=0), axis=2)
    assert step.max() <= bl.SPEED_MPS[1] * bl.BLOCK_S + 1e-4 and (step > 0).any()      # they move, never faster than allowed
    # A robot at the origin, AP 20 m east: a forklift 2 m ahead blocks, one behind or beyond the reach does not.
    start, end = np.array([[0.0, 0.0]]), np.array([[20.0, 0.0]])
    hit = lambda x, y: bool(bl.lines_blocked(start, end, np.array([[x, y]]))[0])
    assert hit(2.0, 0.3) and not hit(-2.0, 0.0) and not hit(2.0, 1.5) and not hit(12.0, 0.0)
    assert bool(bl.lines_blocked(start, end, np.array([[12.0, 0.0]]), frac=1.0)[0])      # robot-to-robot: whole line
    # One forklift, two robots in line behind it: both blocked towards the same AP at once.
    two = bl.lines_blocked(np.array([[0.0, 0.0], [1.0, 0.0]])[:, None, :], np.array([[20.0, 0.0]])[None, :, :],
                           np.array([[3.0, 0.2]]))
    assert two.shape == (2, 1) and two.all()
    n_slots = int(world.duration_s / 0.01)
    f0, b0, _ = hidden_draws(world, n_slots)
    f1, b1, _ = hidden_draws(world, n_slots, blockage_model="moving")
    assert np.array_equal(f0, f1) and b1.shape == b0.shape and b1.dtype == bool and not np.array_equal(b0, b1)
    pol = {p.name: p for p in policies()}
    nat = replace(pol["native"], gate=True, cadence_roam=True)
    base = simulate(world, nat, collision_lambda=0.01, interference=True)
    same = simulate(world, nat, collision_lambda=0.01, interference=True, blockage_model="independent")
    assert base["control_misses"] == same["control_misses"] and base["blockage_model"] == "independent"
    for p in (replace(nat, dual_radio=True, radio2="shared", relay=True, ble=True), pol["wifi8"] if "wifi8" in pol else nat,
              Policy("nr_30ms", "nr", nr_ho_cost_s=0.03)):
        r = simulate(world, p, collision_lambda=0.01, interference=True, blockage_model="moving")
        assert 0 <= r["control_miss"] <= 1 and r["blockage_model"] == "moving"
    with pytest.raises(ValueError):
        simulate(world, nat, blockage_model="forklift")


def test_fleet_features_count_other_robots_alarms_only():
    from wirelessjev.outage_risk import dataset as ds
    t_n, n = 20, 3
    lvl = np.full((t_n, n, 3), 30.0)
    lvl[8:14, 0] -= 15.0                                     # robot 0 is blocked from tick 8
    pos = np.zeros((t_n, n, 2), dtype=np.float32)
    pos[:, 1] = (4.0, 0.0)                                   # robot 1 is 4 m from robot 0
    pos[:, 2] = (50.0, 0.0)                                  # robot 2 is far away, on the same AP as robot 0
    site = np.zeros((t_n, n), dtype=int)
    site[:, 1] = 1
    tr = {"level": lvl, "expected": np.full((t_n, n, 3), 30.0), "site": site, "pos": pos,
          "sent": np.zeros((t_n, n)), "missed": np.zeros((t_n, n)), "roaming": np.zeros((t_n, n), dtype=bool),
          "blocked": np.zeros((t_n, n, 3), dtype=bool)}
    alarm = ds.rule_state(tr)
    assert alarm[9:14, 0].all() and not alarm[:, 1:].any()
    f = ds.fleet_features(tr)
    assert f.shape == (t_n, n, ds.FLEET_PER_TICK)
    assert (f[:, 0] == 0).all()                              # robot 0 never counts its own alarm
    assert f[9, 1, 0] == 0 and f[10, 1, 0] == 1 and f[10, 1, 1] == 1    # heard one tick late
    assert f[10, 1, 2] == 0 and f[10, 2, 2] == 1 and f[10, 2, 0] == 0   # same AP but far; near but other AP
    assert abs(f[10, 1, 3] - 1 / 5.0) < 1e-6 and f[19, 1, 1] == 0       # nearest alarmed robot 4 m away; forgotten after 0.5 s
    x, y, rule = ds.examples(tr, fleet=True)
    assert x.shape == ((t_n - ds.WINDOW + 1) * n, ds.WINDOW * (ds.PER_TICK + ds.FLEET_PER_TICK))


def test_sensed_blocker_alarm_predicts_a_crossing_and_respects_racks():
    from wirelessjev.outage_risk import dataset as ds
    t_n = 20
    # One robot parked in the aisle at y = 13, AP 20 m east in the same aisle; a forklift in that aisle drives at it.
    pos = np.tile(np.array([[[20.0, 13.0]]], dtype=np.float32), (t_n, 1, 1))
    tr = {"pos": pos, "site": np.zeros((t_n, 1), dtype=int)}
    near_ap = np.array([[40.0, 13.0]], dtype=np.float32)
    tracks = np.zeros((t_n, 1, 2), dtype=np.float32)
    tracks[:, 0, 0] = 31.0 - 0.2 * np.arange(t_n)             # westwards at 2 m/s; inside the vulnerable 7.7 m at tick 17
    tracks[:, 0, 1] = 13.0
    now = ds.sensed_blocker_alarm(tr, tracks, near_ap, 0, window=1)
    soon = ds.sensed_blocker_alarm(tr, tracks, near_ap, 5, window=1)
    seen = ds.sensed_blocker_alarm(tr, tracks, near_ap, 5, window=1, line_of_sight=True)
    assert now.any() and int(np.argmax(soon)) == int(np.argmax(now)) - 5      # half a second of warning
    assert np.array_equal(soon, seen)                                          # same aisle: nothing hides it
    # A far AP: the line crosses the next aisle (y = 19), where a forklift is hidden behind the rack row at y = 16.
    far_ap = np.array([[26.0, 32.0]], dtype=np.float32)
    tracks[:, 0, 0] = 24.0 - 0.2 * np.arange(t_n)             # passes x = 21.9, where the line crosses y = 19
    tracks[:, 0, 1] = 19.0
    assert ds.sensed_blocker_alarm(tr, tracks, far_ap, 0, window=1).any()
    assert not ds.sensed_blocker_alarm(tr, tracks, far_ap, 0, window=1, line_of_sight=True).any()
    assert ds._rack_between(np.array([[20.0, 13.0]]), np.array([[20.0, 19.0], [25.0, 13.0], [60.0, 19.0]])).tolist() == [[True, False, False]]


def test_foresight_is_inert_unless_asked_for_and_runs_with_warnings(world):
    from wirelessjev.native.blockers import truth_foresight
    from wirelessjev.native.sim import hidden_draws
    b = np.zeros((6, 1, 1), dtype=bool)
    b[4] = True
    assert truth_foresight(b, ahead=2)[:, 0, 0].tolist() == [False, False, True, True, False, False]
    pol = {p.name: p for p in policies()}
    nat = replace(pol["native"], gate=True, cadence_roam=True)
    kw = dict(collision_lambda=0.01, interference=True, blockage_model="moving")
    _, blocked, _ = hidden_draws(world, int(world.duration_s / 0.01), blockage_model="moving")
    fs = truth_foresight(blocked)
    base = simulate(world, nat, **kw)
    ignored = simulate(world, nat, foresight=fs, **kw)                               # warnings given, policy does not read them
    silent = simulate(world, replace(nat, foresee=True), foresight=np.zeros_like(fs), **kw)   # reads them, none raised
    assert base["control_misses"] == ignored["control_misses"] == silent["control_misses"]
    assert base["foresight_moves"] == 0 and silent["foresight_moves"] == 0
    for p in (replace(nat, foresee=True), replace(nat, foresee=True, dual_radio=True, radio2="shared")):
        r = simulate(world, p, foresight=fs, **kw)
        assert 0 <= r["control_miss"] <= 1 and r["foresight_moves"] >= 0


def test_sensing_and_foresight_predictors(tmp_path):
    from wirelessjev.outage_risk import foresight as fo
    t_n = 30
    # Robot parked at (20, 13); AP 20 m east in its aisle; a forklift drives at it from x = 31 at 2 m/s.
    pos = np.tile(np.array([[[20.0, 13.0]]], dtype=np.float32), (t_n, 1, 1))
    sites = np.array([[40.0, 13.0], [20.0, 33.0]], dtype=np.float32)
    tracks = np.zeros((t_n, 2, 2), dtype=np.float32)
    tracks[:, 0, 0], tracks[:, 0, 1] = 31.0 - 0.2 * np.arange(t_n), 13.0
    tracks[:, 1] = (22.0, 19.0)                                    # a second forklift parked behind the rack row
    perfect = fo.sense(pos, tracks, 1, p_detect=1.0, sigma_m=0.0)
    assert perfect.seen[:, 0, 0].tolist() == [bool(31.0 - 0.2 * k - 20.0 < fo.RANGE_M) for k in range(t_n)]
    assert not perfect.seen[:, 0, 1].any()                         # hidden by the rack
    assert np.allclose(perfect.vel[11:, 0, 0], [-2.0, 0.0], atol=1e-3)       # first seen at tick 6, so known from 11
    noisy = fo.sense(pos, tracks, 1)
    again = fo.sense(pos, tracks, 1)
    assert np.array_equal(noisy.seen, again.seen) and np.array_equal(noisy.pos, again.pos)
    truth = fo.truth_blocked(pos, sites, tracks)
    assert truth.shape == (t_n, 1, 2) and truth[:, 0, 0].any() and not truth[:, 0, 1].any()
    first_blocked = int(np.argmax(truth[:, 0, 0]))
    warn = fo.geometric_foresight(pos, sites, perfect)
    first_warn = int(np.argmax(warn[:, 0, 0]))
    assert warn[:, 0, 0].any() and 5 <= first_blocked - first_warn <= 10 and not warn[:, 0, 1].any()
    # Not knowing the AP height (whole line counts) warns earlier.
    assert int(np.argmax(fo.geometric_foresight(pos, sites, perfect, frac=1.0)[:, 0, 0])) < first_warn
    f, has = fo.pair_features(pos, sites, perfect, 12)
    assert f.shape == (1, 2, fo.N_FEATURES) and np.isfinite(has).tolist() == [True]      # distance to the nearest known obstacle
    assert abs(f[0, 0, 0] - 20.0) < 1e-4 and abs(f[0, 0, 3] - (31.0 - 2.4 - 20.0)) < 1e-3 and abs(f[0, 0, 5] + 2.0) < 1e-3
    assert abs(f[0, 1, 4] + (31.0 - 2.4 - 20.0)) < 1e-3              # same forklift, seen across the line to the other AP
    x, y = fo.training_rows(pos, sites, perfect, truth, every=1)
    assert x.shape[1] == fo.N_FEATURES and len(x) == len(y) and y.any()
    rng = np.random.default_rng(0)
    xs = rng.normal(size=(2000, fo.N_FEATURES)).astype(np.float32)
    ys = xs[:, 3] > 0.3
    m = fo.LearnedForesight(epochs=15).fit(xs, ys)
    m.save(tmp_path / "m.pt")
    m2 = fo.LearnedForesight.load(tmp_path / "m.pt")
    assert np.allclose(m.net.predict(xs), m2.net.predict(xs), atol=1e-6)
    s = fo.score(np.array([True, True, False, False]), np.array([True, False, True, False]))
    assert s["caught"] == 0.5 and s["false_alarm_rate"] == 0.5 and s["precision"] == 0.5


def test_foresight_move_limit_and_campaign_evaluation(world):
    from wirelessjev.native.blockers import truth_foresight
    from wirelessjev.native.run_fleet import FORESIGHT_SCENES, foresight_eval
    from wirelessjev.native.sim import hidden_draws
    pol = {p.name: p for p in policies()}
    nat = replace(pol["native"], gate=True, cadence_roam=True, foresee=True)
    kw = dict(collision_lambda=0.01, interference=True, blockage_model="moving")
    _, blocked, _ = hidden_draws(world, int(world.duration_s / 0.01), blockage_model="moving")
    fs = truth_foresight(blocked)
    free = simulate(world, nat, foresight=fs, **kw)
    wide = simulate(world, replace(nat, foresee_max_m=1e6), foresight=fs, **kw)       # a limit nothing exceeds changes nothing
    tight = simulate(world, replace(nat, foresee_max_m=0.5), foresight=fs, **kw)      # no AP is that close: never moves
    assert free["control_misses"] == wide["control_misses"] and free["foresight_moves"] == wide["foresight_moves"]
    assert tight["foresight_moves"] == 0
    miss = {"nr_30ms": 170, "single": 3000, "single+foresee[truth]": 1700, "single+foresee[learned]": 2800, "shared": 200,
            "shared+foresee[truth]": 110, "shared+foresee[learned]": 150, "shared+foresee[geometric]": 160}
    rows = [{"scene": s, "arm": a, "condition": c, "control_misses": m, "control_sent": 1_000_000, "control_miss": m / 1e6}
            for s in FORESIGHT_SCENES for a, m in miss.items() for c in ("main", "stress")]
    e = foresight_eval(rows)
    assert e["F1"]["pass"] and e["F2"]["pass"] and e["F3"]["pass"] and e["F4"]["pass"] and e["F5"]["pass"]
    assert abs(e["F1"]["ratio"] - 0.75) < 1e-9 and e["F1"]["scenes"] == 6


def test_relay_realism_knobs_only_weaken_the_relay(world):
    from wirelessjev.native.run_fleet import RELAY_LEVELS, RELAY_SCENES, relay_eval
    pol = {p.name: p for p in policies()}
    nat = replace(pol["native"], gate=False, relay=True)
    kw = dict(collision_lambda=0.03, interference=True, blockage_model="moving")
    base = simulate(world, nat, **kw)
    same = simulate(world, replace(nat, relay_delay_slots=0, relay_coop=1.0), **kw)
    assert base["control_misses"] == same["control_misses"] and base["relay_rescues"] == same["relay_rescues"]
    never = simulate(world, replace(nat, relay_delay_slots=5), **kw)          # always too late: no rescue ever counts
    busy = simulate(world, replace(nat, relay_coop=0.0), **kw)                # no helper is ever free
    assert never["relay_rescues"] == 0 and busy["relay_rescues"] == 0
    none = simulate(world, replace(nat, relay=False), **kw)
    assert never["control_misses"] == none["control_misses"]
    real = simulate(world, replace(nat, **RELAY_LEVELS["realistic"]), **kw)
    assert 0 < real["relay_rescues"] <= base["relay_rescues"] or base["relay_rescues"] == 0
    miss = {"nr_30ms": 150, "shared": 250, "shared+relay[optimistic]": 10, "shared+relay[realistic]": 60, "shared+relay[pessimistic]": 180}
    rows = [{"scene": s, "arm": a, "condition": c, "control_misses": m, "control_sent": 1_000_000, "control_miss": m / 1e6}
            for s in RELAY_SCENES for a, m in miss.items() for c in ("main", "stress")]
    e = relay_eval(rows)
    assert e["R1"]["pass"] and not e["R2"]["pass"] and e["R3"]["pass"]


def test_coverage_report_on_a_known_floor():
    from wirelessjev.native import coverage as cv
    gx, gy = cv.grid(2.0, width=20.0, height=10.0)
    assert gx.shape == (5, 10)
    # Two APs: one at the west end, one at the east end; level falls 3 dB per 2 m.
    sites = np.array([[0.0, 5.0], [20.0, 5.0]])
    lv = np.stack([-50.0 - 1.5 * np.abs(gx - sites[k, 0]) for k in range(2)], axis=-1)
    best, backup, server = cv.best_and_backup(lv)
    assert best.max() <= -50 and (server[:, 0] == 0).all() and (server[:, -1] == 1).all()
    assert (cv.best_and_backup(lv, without=0)[0] <= best).all()
    area = np.ones_like(gx, dtype=bool)
    rep = cv.report(lv, np.full_like(lv, 2.0), gx, gy, area, sites)
    ctl = rep["services"]["control"]
    assert ctl["covered_share"] == 1.0 and ctl["with_backup_share"] < 1.0 and rep["not_enough_data_share"] == 0.0
    assert rep["if_an_ap_fails"][0]["new_weak_m2"] > 0 and len(rep["if_an_ap_fails"]) == 2
    weak = np.zeros_like(area)
    weak[1:3, 2:4] = True
    weak[4, 9] = True                                           # a single cell is too small to list
    h = cv.holes(weak, gx, gy, best)
    assert len(h) == 1 and h[0]["area_m2"] == 16.0
    exact = cv.score(lv, lv, area)
    assert exact["best_level_rmse_db"] == 0.0 and exact["worst_ap_failure"]["in_report_top3"]
    low = cv.score(lv - 4.0, lv, area)                           # a map that reads 4 dB low over-flags
    assert abs(low["best_level_bias_db"] + 4.0) < 1e-6 and low["no_backup_control"]["recall"] == 1.0
    assert low["no_backup_control"]["precision"] < 1.0
    pts = np.array([[[1.0, 5.0]], [[19.0, 5.0]]])
    m = cv.route_mask(gx, gy, pts, near_m=2.5)
    assert m[2, 0] and m[2, 9] and not m[2, 5]


def test_site_geometry_setting_and_radio_labels(world):
    from wirelessjev.native import blockers as bl
    from wirelessjev.native.sim import hidden_draws
    from wirelessjev.outage_risk import dataset as ds, foresight as fo
    assert abs(bl.reach_frac_for(6.0) - bl.REACH_FRAC) < 1e-9 and bl.reach_frac_for(10.0) < bl.REACH_FRAC
    n_slots = int(world.duration_s / 0.01)
    _, old, _ = hidden_draws(world, n_slots, blockage_model="moving")
    _, same, _ = hidden_draws(world, n_slots, blockage_model="moving", reach_frac=bl.REACH_FRAC)
    _, high, _ = hidden_draws(world, n_slots, blockage_model="moving", reach_frac=bl.reach_frac_for(10.0))
    assert np.array_equal(old, same) and high.sum() < old.sum() and not (high & ~old).any()   # a higher AP is blocked less, never more
    pol = {p.name: p for p in policies()}
    nat = replace(pol["native"], gate=True, cadence_roam=True)
    tr = ds.collect(world, nat, collision_lambda=0.01, interference=True, blockage_model="moving", reach_frac=bl.reach_frac_for(10.0))
    blocks = tr["slot"] // 10
    tracks = bl.blocker_tracks(world.seed, int(blocks.max()) + 1)[blocks]
    sensed = fo.sense(tr["pos"], tracks, world.seed)
    x, y = fo.radio_label_rows(tr, world.sites.astype(np.float32), sensed)
    assert x.shape[1] == fo.N_FEATURES and len(x) == len(y) and y.dtype == bool
    # Rows are one per robot that senses something, for the AP it is using.
    has = sensed.seen[: len(tr["slot"]) - 10].any(axis=2)
    assert len(x) == int(has.sum())

