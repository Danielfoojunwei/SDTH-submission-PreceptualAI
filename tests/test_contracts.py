"""Software-contract tests. No model is loaded and no radio is touched."""
import hashlib
import hmac

import numpy as np
import pytest
import torch

from wirelessjev.candidates import DEFER_ID, generate, hard_filter, oracle_scores
from wirelessjev.execution import (DryRunTransport, Ledger, Transaction,
                                   build_connect, execute, reconcile, redact)
from wirelessjev.hardware import RASPBOT
from wirelessjev.missions import MISSIONS
from wirelessjev.oracle import evaluate, mission_utility
from wirelessjev.swarm import ChannelPriors, Reputation, SwarmConsensus, SwarmNode
from wirelessjev.swarm.share import bilevel_kmeans
from wirelessjev.teachers import BASELINES, local_search
from wirelessjev.traffic import EVENT, TREND, VOLATILITY, rule_labels, synth_window
from wirelessjev.twin import MobilityPredictor, interference_state, make_scene


def _cfg(scene, assoc=None):
    return {"assoc": assoc or {c.car_id: c.current_ap for c in scene.cars},
            "channels": {a.ap_id: a.channel for a in scene.aps}}


def test_scene_is_deterministic_and_hardware_faithful():
    a, b = make_scene(5), make_scene(5)
    assert [(c.x, c.current_ap) for c in a.cars] == [(c.x, c.current_ap) for c in b.cars]
    assert len(a.cars) == 10
    assert len(a.aps) <= RASPBOT.max_visible_aps
    assert {ap.band for ap in a.aps} <= set(RASPBOT.bands)
    for car in a.cars:
        assert len(a.visible_aps(car)) <= RASPBOT.max_visible_aps


def test_oracle_charges_measured_handover():
    s = make_scene(1)
    base = evaluate(s, _cfg(s))
    car = s.cars[0]
    other = next(ap.ap_id for ap, _ in s.visible_aps(car) if ap.ap_id != car.current_ap)
    moved = dict(_cfg(s)["assoc"], **{car.car_id: other})
    ev = evaluate(s, _cfg(s, moved))
    assert ev.handovers == base.handovers + 1
    assert ev.critical_violations >= 1      # control floor broken by ~4 s outage


def test_interference_codes():
    s = make_scene(2)
    st = interference_state(s, _cfg(s)["assoc"], _cfg(s)["channels"])
    for car in s.cars:
        assert st[car.car_id][car.current_ap] in (-1, 1, 2)
        assert all(v == 0 for k, v in st[car.car_id].items() if k != car.current_ap)


def test_local_search_never_worse_than_its_start():
    s = make_scene(3)
    for m in MISSIONS.values():
        start = BASELINES["milp"](s, m)
        best = local_search(s, m, start)
        u0 = mission_utility(evaluate(s, _cfg(s, start)), m, 10)
        u1 = mission_utility(evaluate(s, _cfg(s, best)), m, 10)
        assert u1 >= u0 - 1e-9


def test_candidates_include_defer_and_hide_provenance():
    s = make_scene(4)
    opts, _ = generate(s, seed=4)
    ids = [o.option_id for o in opts]
    assert ids[-1] == DEFER_ID and len(set(ids)) == len(ids)
    for o in opts:
        for word in ("milp", "ls:", "hungarian", "coloring", "protect_video",
                     "bulk_upload", "fair_share"):
            assert word not in o.text
    scores = oracle_scores(s, opts, MISSIONS["bulk_upload"])
    assert set(scores) == set(ids)


def test_hard_filter_rejects_invisible_bss_and_power_violation():
    s = make_scene(6)
    assoc = _cfg(s)["assoc"]
    far = min(s.aps, key=lambda ap: -((ap.x - s.cars[0].x) ** 2 + (ap.y - s.cars[0].y) ** 2))
    if far.ap_id not in {ap.ap_id for ap, _ in s.visible_aps(s.cars[0])}:
        bad = dict(assoc, **{s.cars[0].car_id: far.ap_id})
        assert hard_filter(s, bad)
    from dataclasses import replace
    hot = s._copy(cars=[replace(s.cars[0], tx_power_dbm=30.0)] + s.cars[1:])
    assert "power" in hard_filter(hot, assoc)


def test_consensus_rejects_failed_validation_and_detects_tamper():
    key = b"site"
    sw = SwarmConsensus(site_key=key, min_ready=2)
    nodes = [SwarmNode(f"car-{i}", key=f"k{i}".encode()) for i in range(3)]
    for n in nodes:
        resp = hmac.new(n.key, b"c", hashlib.sha256).hexdigest()
        assert sw.admit(n, b"c", resp)
    rogue = SwarmNode("rogue", key=b"x", algorithm="other")
    assert not sw.admit(rogue, b"c", hmac.new(b"x", b"c", hashlib.sha256).hexdigest())
    ups = {n.node_id: (torch.ones(4, 2) * (i + 1), 10) for i, n in enumerate(nodes)}
    bad = sw.merge_round(ups, [n.node_id for n in nodes], lambda r: {"passed": False})
    assert not bad.released and sw.released_readout is None
    good = sw.merge_round(ups, [n.node_id for n in nodes], lambda r: {"passed": True})
    assert good.released and torch.allclose(good.readout, torch.full((4, 2), 2.0))
    assert sw.verify_log()
    sw.log[1]["body"] = sw.log[1]["body"].replace("true", "false")
    assert not sw.verify_log()


def test_reputation_excludes_power_cheater():
    rep = Reputation()
    for _ in range(4):
        v = rep.check("car-09", tx_power_dbm=24, cap_dbm=20, reconnects=0,
                      used_mbps=5, declared_mbps=5)
        rep.update("car-09", v)
    assert not rep.trusted("car-09")
    rep.update("car-01", [])
    assert rep.trusted("car-01")


def test_unknown_outcome_is_never_resent_and_reconciles_from_status():
    t = DryRunTransport(state={"car-01": "ap1-5g"}, unknown_rate=1.0, seed=1)
    ledger = Ledger(budgets={"d0": 1.0})
    txn = Transaction("t1", "car-01", "ap1-5g", "ap2-5g",
                      build_connect("fleet_5g", "ap2-5g"))
    assert ledger.prepare(txn, "d0", 0.3)
    execute(txn, ledger, t)
    assert txn.state == "UNKNOWN" and ledger.reserved["d0"] == pytest.approx(0.3)
    connects = [r for _, r in t.sent if r["action"] == "connect"]
    reconcile(txn, ledger, t)
    assert [r for _, r in t.sent if r["action"] == "connect"] == connects
    assert txn.state in ("COMMITTED", "ABORTED")


def test_clean_failure_is_not_applied_and_releases():
    t = DryRunTransport(state={"car-01": "ap1-5g"}, failure_rate=1.0)
    ledger = Ledger(budgets={"d0": 1.0})
    txn = Transaction("t2", "car-01", "ap1-5g", "ap2-5g",
                      build_connect("fleet_5g", "ap2-5g"))
    ledger.prepare(txn, "d0", 0.4)
    execute(txn, ledger, t)
    assert txn.outcome == "NOT_APPLIED" and ledger.reserved["d0"] == 0.0


def test_requests_carry_no_secrets():
    req = build_connect("fleet_5g", "aa:bb:cc:dd:ee:ff")
    assert set(req) == {"action", "ssid", "bssid", "security"}
    redact(req)
    with pytest.raises(ValueError):
        redact({"password": "x"})


def test_mobility_and_sharing_and_clusters():
    mp = MobilityPredictor()
    for x in range(10):
        mp.observe("car-01", 1.0 + x, 1.0)
    top = mp.predict("car-01")
    assert top and abs(sum(p for _, p in mp.predict("car-01", 50)) - 1) < 1e-9
    cp = ChannelPriors()
    cp.observe("car-01", "ap1-5g", 30.0)
    cp.observe("car-02", "ap1-5g", 20.0)
    for _ in range(10):
        cp.step({"c0": ["car-01", "car-02"]})
    assert cp.estimate("car-01", "c0", "ap1-5g") == pytest.approx(0.8 * 25 + 0.2 * 30)
    cl = bilevel_kmeans({f"car-{i}": [(i * 5.0, 0.0)] for i in range(6)}, 3)
    assert sum(len(v) for v in cl.values()) == 6 and len(cl) == 3


def test_rule_labels_are_valid():
    rng = np.random.default_rng(0)
    for _ in range(20):
        lab = rule_labels(synth_window(rng))
        assert lab["trend"]["choice"] in TREND
        assert lab["event"]["choice"] in EVENT
        assert lab["volatility"]["choice"] in VOLATILITY


def test_consensus_gate_compares_against_live_release():
    key = b"site"
    sw = SwarmConsensus(site_key=key, min_ready=1)
    node = SwarmNode("car-01", key=b"k")
    sw.admit(node, b"c", hmac.new(b"k", b"c", hashlib.sha256).hexdigest())
    quality = {1.0: 0.3, 2.0: 0.5}           # readout value -> val regret

    def validate(w):
        live = sw.released_readout
        ref = quality[float(live[0])] if live is not None else 1.5
        return {"passed": quality[float(w[0])] <= ref - 1e-6}

    assert sw.merge_round({"car-01": (torch.tensor([1.0]), 1)}, ["car-01"], validate).released
    worse = sw.merge_round({"car-01": (torch.tensor([2.0]), 1)}, ["car-01"], validate)
    assert not worse.released and float(sw.released_readout[0]) == 1.0


def test_vectorised_evaluate_matches_reference():
    from wirelessjev.oracle import evaluate_reference
    for seed in range(8):
        s = make_scene(seed)
        rng = np.random.default_rng(seed)
        for _ in range(3):
            assoc = {c.car_id: str(rng.choice([ap.ap_id for ap, _ in s.visible_aps(c)]))
                     for c in s.cars}
            cfg = _cfg(s, assoc)
            fast, ref = evaluate(s, cfg), evaluate_reference(s, cfg)
            for key in ("total_mbps", "U", "V", "W", "B", "jain", "piecewise",
                        "qos_margin", "pm1", "max_domain_load", "access_rate",
                        "worst_critical_margin"):
                assert getattr(fast, key) == pytest.approx(getattr(ref, key), rel=1e-9, abs=1e-9), key
            for key in ("handovers", "critical_violations", "protected_hits", "retunes"):
                assert getattr(fast, key) == getattr(ref, key), key
