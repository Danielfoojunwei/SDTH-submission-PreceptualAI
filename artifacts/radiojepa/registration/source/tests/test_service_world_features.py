"""Public radio contracts, physical clocks, and real native-input integration.

The small seeded scenes here exercise interfaces; they are not benchmark or
trained-model evidence. Only receipt locations and scene size are overridden
in the end-to-end study check; packet, scan, map, and replay code remain real.
"""
import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from wirelessjev.fleetwm.data import Geometry
from wirelessjev.native.sim import policies
from wirelessjev.native.world import build_world
from wirelessjev.service_world.features import (
    GLOBAL_FEATURES,
    LINK_FEATURES,
    RadioHistory,
    public_features,
)
from wirelessjev.service_world.model import ServiceWorldModel
from wirelessjev.service_world.sim import simulate


class FeatureRecorder(RadioHistory):
    def __init__(self, geometry, length=20):
        super().__init__(geometry, length)
        self.public_rows = []

    def observe(self, public):
        super().observe(public)
        self.public_rows.append(copy.deepcopy(public))


@pytest.fixture(scope="module")
def native_case():
    torch.set_num_threads(1)
    world = build_world(299100, duration_s=3, n_robots=4)
    geometry = Geometry(world.sites, world.tx_dl, world.noise_db, ())
    policy = replace(
        {p.name: p for p in policies()}["native"],
        cadence_scan=True,
        scan_period_s=100.,
    )
    recorder = FeatureRecorder(geometry)
    saved = []
    result = simulate(
        world, policy, observation_hook=recorder.observe,
        observation_period_slots=10, checkpoint_at={65},
        checkpoint_sink=saved.append, detailed=True,
    )
    return world, geometry, policy, recorder, saved[0], result


def test_real_public_history_encodes_with_masked_missing_values(native_case):
    world, _, _, recorder, _, _ = native_case
    window = recorder.window()
    assert window["links"].shape == (world.n, 20, len(world.sites), 3, 12)
    assert window["globals"].shape == (world.n, 20, 8)
    assert not window["observed"][..., LINK_FEATURES.index("cached_snr_over30")].any()
    assert np.isfinite(window["links"][window["observed"]]).all()
    assert np.isfinite(window["globals"]).all()

    torch.manual_seed(299101)
    model = ServiceWorldModel(hidden=16).eval()
    tensors = {k: torch.as_tensor(v) for k, v in window.items()}
    with torch.no_grad():
        clean = model.encode(**tensors)
        poisoned = tensors["links"].clone()
        poisoned[~tensors["observed"]] = float("nan")
        masked = model.encode(**(tensors | {"links": poisoned}))
    assert torch.isfinite(clean["belief"]).all()
    assert torch.isfinite(clean["links"]).all()
    torch.testing.assert_close(masked["belief"], clean["belief"], rtol=0, atol=0)
    torch.testing.assert_close(masked["links"], clean["links"], rtol=0, atol=0)


def test_neighbor_cache_does_not_invent_pilot_age_or_configured_mcs(native_case):
    _, geometry, _, recorder, _, _ = native_case
    public = copy.deepcopy(recorder.public_rows[0])
    n, sites = len(public["current"]), len(geometry.sites)
    # An explicit completed-cache fixture: heard and unheard neighbors both
    # exist, but the public scanner has supplied no per-neighbor sample age.
    public["local_rssi"] = np.full((n, sites, 3), -100., dtype="f4")
    public["local_heard"] = np.zeros((n, sites, 3), dtype=bool)
    neighbor = (int(public["current"][0]) + 1) % sites
    public["local_rssi"][0, neighbor, 1] = -70.
    public["local_heard"][0, neighbor, 1] = True
    links, _, observed = public_features(geometry, public)

    is_neighbor = np.arange(sites)[None, :] != public["current"][:, None]
    for name in ("current_pilot_age_over30s", "current_pilot_snr_over30", "configured_mcs_over11"):
        assert not observed[..., LINK_FEATURES.index(name)][is_neighbor].any()
    age = LINK_FEATURES.index("current_pilot_age_over30s")
    assert not observed[0, neighbor, :, age].any()
    cached = LINK_FEATURES.index("cached_snr_over30")
    assert observed[0, neighbor, 1, cached]
    assert not observed[0, neighbor, 0, cached]
    expected = (-70. - geometry.tx[1] + 15. - geometry.noise[1]) / 30.
    assert links[0, neighbor, 1, cached] == pytest.approx(expected)

    changed = copy.deepcopy(public)
    changed["local_rssi"][~changed["local_heard"]] = np.nan
    again, again_global, again_mask = public_features(geometry, changed)
    np.testing.assert_array_equal(again_mask, observed)
    np.testing.assert_array_equal(again, links)
    assert np.isfinite(again_global).all()


def test_sensitivity_threshold_and_unavailable_pilot_preserve_unknowns(native_case):
    _, geometry, _, recorder, _, _ = native_case
    public = copy.deepcopy(recorder.public_rows[0])
    public["pilot_valid"][:] = True
    public["pilot_snr"][0] = -91. - geometry.noise - geometry.tx + 15.
    public["pilot_valid"][1] = False
    public["pilot_site"][1] = (public["current"][1] + 1) % len(geometry.sites)
    public["forecast"] = None
    links, _, observed = public_features(geometry, public)
    snr = LINK_FEATURES.index("current_pilot_snr_over30")
    age = LINK_FEATURES.index("current_pilot_age_over30s")
    valid = LINK_FEATURES.index("current_pilot_valid")
    for robot in (0, 1):
        current = public["current"][robot]
        assert not observed[robot, current, :, snr].any()
        assert not links[robot, current, :, valid].any()
        assert observed[robot, current, :, valid].all()
    assert not observed[1, public["current"][1], :, age].any()
    for name in ("completed_map_snr_over30", "completed_map_std_over10", "forecast_age_over30s"):
        assert not observed[..., LINK_FEATURES.index(name)].any()


def test_public_pilot_queue_and_packet_changes_affect_their_typed_features(native_case):
    _, geometry, _, recorder, _, _ = native_case
    public = copy.deepcopy(recorder.public_rows[0])
    public["pilot_valid"][:] = True
    public["pilot_snr"][:] = 20.
    baseline, glob, mask = public_features(geometry, public)
    changed = copy.deepcopy(public)
    changed["pilot_snr"][0, 1] += 3.
    changed["video_queue_bits"][0] = 999.
    changed["bulk_queue_bits"][0] = 9999.
    changed["outcome_history"][:, 0] = 0
    changed["outcome_history"][-4:-1, 0, 0] = 1
    changed["outcome_history"][-1, 0, 1] = 1
    features, other_glob, other_mask = public_features(geometry, changed)
    expected = np.zeros_like(baseline)
    expected[0, public["current"][0], 1, LINK_FEATURES.index("current_pilot_snr_over30")] = .1
    np.testing.assert_allclose(features - baseline, expected, atol=1e-7)
    np.testing.assert_array_equal(mask, other_mask)
    assert other_glob[0, GLOBAL_FEATURES.index("recent_miss_fraction")] == pytest.approx(.25)
    assert other_glob[0, GLOBAL_FEATURES.index("recent_delivery_fraction")] == pytest.approx(.75)
    assert other_glob[0, GLOBAL_FEATURES.index("log_video_queue_bits_over20")] == pytest.approx(np.log1p(999.) / 20)
    assert other_glob[0, GLOBAL_FEATURES.index("log_bulk_queue_bits_over20")] == pytest.approx(np.log1p(9999.) / 20)
    np.testing.assert_array_equal(other_glob[1:], glob[1:])


def test_history_uses_actual_irregular_clock_without_startup_duplication(native_case):
    _, geometry, _, recorder, _, _ = native_case
    history = RadioHistory(geometry, length=2)
    with pytest.raises(ValueError, match="No causal observations"):
        history.window()
    public = copy.deepcopy(recorder.public_rows[0])
    for i, slot in enumerate((20, 27, 80)):
        public["slot"] = slot
        history.observe(public)
        window = history.window()
        assert window["links"].shape[1] == min(i + 1, 2)
        if i == 0:
            np.testing.assert_array_equal(window["elapsed"], np.zeros((4, 1)))
    np.testing.assert_allclose(window["elapsed"], np.broadcast_to([.07, .53], (4, 2)), atol=1e-7)
    assert history.event_count == 3
    # Evicting a row does not turn the first retained delta into a fabricated
    # zero or padded observation. Returned windows also cannot rewrite memory.
    untouched = history.window()
    window["links"][:] = 12345.
    public["pilot_snr"][:] = 12345.
    np.testing.assert_array_equal(history.window()["links"], untouched["links"])
    for slot in (80, 79):
        public["slot"] = slot
        with pytest.raises(ValueError, match="Event time must advance"):
            history.observe(public)
    assert history.event_count == 3


def test_forked_history_is_past_only_and_replay_keeps_feature_provenance(native_case):
    world, _, policy, recorder, checkpoint, factual = native_case
    copied = [
        hook.__self__ for hook in checkpoint.bundle["hooks"]
        if hasattr(hook, "__self__") and isinstance(hook.__self__, FeatureRecorder)
    ]
    assert len(copied) == 1
    prefix = copied[0]
    assert prefix.last_slot == 60 and prefix.event_count == 7
    assert [row["slot"] for row in prefix.public_rows] == list(range(0, 65, 10))
    assert prefix.window()["links"].shape[1] == 7
    assert recorder.last_slot == 290 and recorder.event_count == 30
    np.testing.assert_allclose(prefix.window()["elapsed"][0], [0.] + [.1] * 6)
    for public in prefix.public_rows:
        assert (public["outcome_slot"][public["outcome_valid"]] < public["slot"]).all()
        assert not {"snr", "fade_db", "blocked", "true_snr"} & public.keys()
    before = prefix.window()
    replay = simulate(world, policy, resume=checkpoint, detailed=True)
    replay["packet_detail"].pop("resume_slot")
    replay["packet_detail"].pop("window_max_gap")
    # Only documented suffix metadata differs, including the same feature rows
    # captured by a deep-copied, bound public observer.
    expected = copy.deepcopy(factual)
    expected["packet_detail"].pop("resume_slot")
    expected["packet_detail"].pop("window_max_gap")
    assert replay == expected
    assert prefix.last_slot == 60 and prefix.event_count == 7
    np.testing.assert_array_equal(prefix.window()["links"], before["links"])


def test_small_real_label_case_exercises_maps_interventions_and_receipts(monkeypatch, tmp_path):
    from wirelessjev.service_world import study

    protocol = study.P | {
        "robots": 6,
        "duration_seconds": 20,
        "checkpoint_forecast_issues_seconds": [5],
    }
    monkeypatch.setattr(study, "P", protocol)
    monkeypatch.setattr(study, "OUT", tmp_path)
    # Scope this unit receipt to interface validation. Real source sealing is
    # separately owned by the experiment runner, not changed by this test.
    monkeypatch.setattr(study, "sources", lambda: {"unit": "shape-only"})
    receipt = study.label_case((299198, .03))
    assert receipt["completed"] and receipt["simulation_only"]
    assert receipt["no_learned_policy_evaluated"] and receipt["no_heldout_access"]
    assert receipt["factual_replays"] == 1
    assert 1 <= receipt["input_windows"] <= 3
    assert receipt["records"] and receipt["forecast_timing"]
    assert receipt["native_runs"] == 3 + len(receipt["records"])
    probe = receipt["prototype"]
    assert probe["untrained"] and not probe["accuracy_or_control_result"]
    assert probe["zero_residual_exact"] and probe["unknown_hazard_is_none"]
    assert probe["clients"] == 6
    path = tmp_path / "inputs" / "299198-0.03.npz"
    with np.load(path) as inputs:
        count = receipt["input_windows"]
        assert inputs["links"].shape[:2] == (count, 20)
        assert inputs["links"].shape[-2:] == (3, len(LINK_FEATURES))
        assert inputs["globals"].shape == (count, 20, len(GLOBAL_FEATURES))
        assert inputs["observed"].dtype == np.bool_
        assert np.isfinite(inputs["links"][inputs["observed"]]).all()
        assert np.isfinite(inputs["globals"]).all()
        np.testing.assert_allclose(inputs["elapsed"], .1)
        assert np.all(inputs["slot"] > 500)
    persisted = json.loads((tmp_path / "cases" / "299198-0.03.json").read_text())
    assert persisted["source_sha256"] == {"unit": "shape-only"}
    assert persisted["inputs_sha256"] == study.digest(path)
    assert study.label_case((299198, .03)) == persisted
