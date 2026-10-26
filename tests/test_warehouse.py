"""Warehouse twin contracts (no model)."""
import math

import numpy as np
import pytest

from wirelessjev.hardware import RASPBOT
from wirelessjev.missions import WAREHOUSE_MISSIONS
from wirelessjev.oracle import evaluate, horizon_snapshots, horizon_utility, mission_utility
from wirelessjev.twin.warehouse import (HEIGHT, RACK_CAP_DB, RACK_DB, WIDTH,
                                        make_warehouse, rack_obstruction)
from wirelessjev.twin.radio import AP


@pytest.fixture(scope="module")
def wh():
    return make_warehouse(11)


def test_hundred_robots_on_smartcar_hardware(wh):
    assert len(wh.cars) == 100 and len(wh.aps) == 32
    assert all(c.tx_power_dbm == RASPBOT.client_tx_power_dbm for c in wh.cars)
    for car in wh.cars:
        assert len(wh.visible_aps(car)) <= RASPBOT.max_visible_aps


def test_racks_attenuate_monotonically():
    ap = AP("x", 60.0, 2.0, 36, "5g")
    ys = np.array([5.0, 20.0, 40.0, 70.0])
    loss = rack_obstruction(ap, np.full(4, 60.0), ys)
    assert loss[0] == 0 and np.all(np.diff(loss) >= 0) and loss[-1] <= RACK_CAP_DB
    # a path straight down the cross-aisle at x = 40 crosses no racks
    on_aisle = AP("y", 40.0, 2.0, 36, "5g")
    assert rack_obstruction(on_aisle, np.array([40.0]), np.array([70.0]))[0] == 0
    assert rack_obstruction(ap, np.array([60.0]), np.array([20.0]))[0] == pytest.approx(
        2 * RACK_DB["5g"])


def test_robots_stay_in_hall_and_move(wh):
    s = wh
    start = [(c.x, c.y) for c in s.cars]
    for _ in range(24):
        s = s.advance(5.0)
    for c in s.cars:
        assert -1e-6 <= c.x <= WIDTH + 1e-6 and -1e-6 <= c.y <= HEIGHT + 1e-6
    moved = np.mean([math.hypot(c.x - x, c.y - y) for c, (x, y) in zip(s.cars, start)])
    assert moved > 5.0


def test_planned_positions_do_not_mutate_plans():
    s = make_warehouse(12)
    plan_before = {k: (list(v.waypoints), v.dwell_s) for k, v in s.mobility.plans.items()}
    snaps = horizon_snapshots(s)
    assert len(snaps) == 2
    assert {k: (list(v.waypoints), v.dwell_s) for k, v in s.mobility.plans.items()} == plan_before


def test_horizon_charges_handover_once():
    s = make_warehouse(13)
    m = WAREHOUSE_MISSIONS["peak_picking"]
    snaps = horizon_snapshots(s)
    ch = {a.ap_id: a.channel for a in s.aps}
    hold = {c.car_id: c.current_ap for c in s.cars}
    car = s.cars[0]
    other = next(ap.ap_id for ap, _ in s.visible_aps(car) if ap.ap_id != car.current_ap)
    moved = dict(hold, **{car.car_id: other})
    for snap in snaps:
        later = snap.with_associations(moved)
        assert evaluate(later, {"assoc": moved, "channels": ch}).handovers == 0
    now = evaluate(s, {"assoc": moved, "channels": ch})
    assert now.handovers == 1
    assert horizon_utility(s, snaps, hold, m, ch) != horizon_utility(s, snaps, moved, m, ch)


def test_batched_evaluator_equals_oracle():
    from wirelessjev.fastplan import BatchEvaluator
    s = make_warehouse(31)
    snaps = horizon_snapshots(s)
    ev = BatchEvaluator(s, snaps)
    rng = np.random.default_rng(1)
    A = np.stack([np.array([rng.choice(v) for v in ev.visible]) for _ in range(6)])
    A[0] = ev.current
    for m in WAREHOUSE_MISSIONS.values():
        fi = [ev.car_ids.index(c) for c in m.focus_cars]
        fast = ev.utility(A, m.weights, fi)
        ref = [horizon_utility(s, snaps, ev.to_assoc(a), m) for a in A]
        assert np.allclose(fast, ref, atol=1e-12)


def test_fast_planner_is_exact_and_monotone():
    from wirelessjev.fastplan import FastPlanner
    s = make_warehouse(32)
    m = WAREHOUSE_MISSIONS["map_rollout"]
    fp = FastPlanner(s, pinned=frozenset({"amr-005"}))
    start_u = horizon_utility(s, fp.snaps, fp.ev.to_assoc(fp.ev.current), m)
    a, u, stats = fp.search(m.weights, m.focus_cars)
    assert u == pytest.approx(horizon_utility(s, fp.snaps, fp.ev.to_assoc(a), m), abs=1e-12)
    assert u >= start_u - 1e-12
    assert a[fp.ev.car_ids.index("amr-005")] == fp.ev.current[fp.ev.car_ids.index("amr-005")]
    assert stats["plans_evaluated"] > 0
