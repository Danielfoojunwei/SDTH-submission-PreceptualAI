import numpy as np
import pytest
import torch
import json
from pathlib import Path

from wirelessjev.radiojev.nonlinear import Outcomes, NL_MISSIONS, NOVEL_NL, nl_utility
from wirelessjev.warehouse_v2.missions import (MissionSpec, UnsupportedMission,
    parse_mission, evaluate_mission, choose_candidate, StructuredMissionReader, route_mission)
from wirelessjev.warehouse_v2.data import episode, measure_scan, make_environment
from wirelessjev.warehouse_v2.model import TemporalRadioModel, tensor_inputs, dynamics_loss
from wirelessjev.warehouse_v2.metrics import robot_metrics, risk_metrics, cluster_interval


def outcomes():
    rng = np.random.default_rng(191)
    sat = rng.uniform(.6, 1., (8, 4, 7))
    sat[0] = .95
    return Outcomes(sat, rng.uniform(0, 1, 8), rng.uniform(0, 1, 8),
                    np.arange(8), np.arange(8), np.arange(7)*15., np.arange(7)*12.,
                    np.array([True, False, False, True, False, False, False]))


@pytest.mark.parametrize('mission', list(NL_MISSIONS)+list(NOVEL_NL))
def test_historical_parser_and_evaluator_match_every_paraphrase(mission):
    o = outcomes()
    expected = nl_utility(mission, o)
    for text in {**NL_MISSIONS, **NOVEL_NL}[mission][1]:
        spec = parse_mission(text, semantics='historical_e2')
        score, feasible = evaluate_mission(spec, o)
        np.testing.assert_allclose(score, expected, atol=1e-6)
        assert feasible.all()


def test_hard_constraints_are_not_historical_soft_penalties():
    o = outcomes()
    spec = MissionSpec('move_budget', max_moves=0)
    result = choose_candidate(spec, o)
    assert result['choice'] == 0
    o.moves[:] = 1
    result = choose_candidate(spec, o)
    assert result['choice'] is None
    assert result['status'] == 'infeasible_within_candidates'
    assert result['globally_infeasible'] is False


def test_hard_video_floor_checks_every_snapshot():
    o = outcomes()
    o.sat[0] = 1.
    o.sat[0, 0, 0] = .7  # Horizon average passes .9, instantaneous minimum fails.
    hard = evaluate_mission(MissionSpec('video_floor', floor=.9), o)[1]
    old = evaluate_mission(MissionSpec('video_floor', floor=.9, semantics='historical_e2'), o)[0]
    assert not hard[0]
    assert old[0] == o.jain[0]


def test_unsupported_and_ambiguous_missions_defer():
    with pytest.raises(UnsupportedMission):
        parse_mission('Optimise everything somehow')
    assert route_mission('Minimise reconnections and maximise video quality', StructuredMissionReader())['route'] == 'defer'
    with pytest.raises(UnsupportedMission):
        parse_mission('Video needs 80% or maybe 90%')
    with pytest.raises(ValueError):
        MissionSpec('video_floor', floor=1.2)


def test_real_simulation_scan_masks_and_censoring():
    scene, _ = make_environment(411, n=12)
    scan = measure_scan(scene, np.random.default_rng(5))
    assert np.all(scan[:, :, 1].sum(1) <= 8)
    assert np.all(scan[:, :, 0][scan[:, :, 1] == 0] == 0)
    for row in scan:
        heard = row[:, 1] > 0
        if heard.sum() == 8:
            np.testing.assert_allclose(row[:, 2], row[heard, 0].min())
        else:
            np.testing.assert_allclose(row[:, 2], -1/3)


def test_temporal_graph_interventions_permutation_and_learning():
    torch.set_num_threads(2)
    sample = episode(412, n=12, steps=4)[0]
    assert set(sample['inputs']) == {'history', 'robot', 'ap', 'current'}
    assert np.isfinite(sample['targets']).all()
    inputs = tensor_inputs(sample)
    plans = torch.tensor(sample['plans'])
    model = TemporalRadioModel(hidden=16)
    out = model(inputs, plans)
    loss = dynamics_loss(out, torch.tensor(sample['targets']))
    loss.backward()
    assert model.message[0].weight.grad.abs().sum() > 0
    permutation = torch.randperm(12)
    reordered = dict(inputs, robot=inputs['robot'][permutation],
                     history=inputs['history'][:, permutation], current=inputs['current'][permutation])
    with torch.no_grad():
        changed = model(reordered, plans[:, permutation])
    torch.testing.assert_close(changed['service'], out['service'][:, permutation], atol=1e-6, rtol=1e-5)


def test_per_robot_metrics_do_not_hide_worst_robot():
    sat = np.array([[1., .1], [1., .2], [1., .3]])
    violations = sat < .5
    metrics = robot_metrics(sat, violations, np.zeros_like(sat), robot_ids=['a', 'b'])
    assert metrics['worst_decile_mean_service'] == pytest.approx(.2)
    assert metrics['per_robot']['b']['longest_critical_violation_s'] == 15.
    assert metrics['per_robot']['a']['longest_critical_violation_s'] == 0.


def test_calibration_and_scene_uncertainty():
    good = risk_metrics([.1, .9], [0, 1])
    bad = risk_metrics([.9, .1], [0, 1])
    assert good['brier'] < bad['brier']
    assert cluster_interval([1., 2.], [0, 0])['ci95'] is None
    assert cluster_interval([1., 2., 3., 4.], [0, 0, 1, 1])['clusters'] == 2


def test_frozen_radio_features_feed_actual_trainable_multitask_head():
    from wirelessjev.warehouse_v2.multitask import SharedTasks
    sample = episode(414, n=12, steps=4)[0]
    backbone = TemporalRadioModel()
    with torch.no_grad():
        features = backbone(tensor_inputs(sample), torch.tensor(sample['plans']))['pooled'].detach()
    head = SharedTasks(shared=True)
    loss = head.radio(features).square().mean()
    loss.backward()
    assert head.radio_input[0].weight.grad is not None


def test_known_handover_cost_is_factored_from_learned_service():
    from wirelessjev.warehouse_v2.factorized import FactorizedRadioModel, raw_target
    sample = episode(415, n=12, steps=4)[0]
    inputs = tensor_inputs(sample)
    plans = torch.tensor(sample['plans'])
    model = FactorizedRadioModel(horizon_s=30., hidden=16)
    with torch.no_grad():
        p = model(inputs, plans)
    moved = plans != inputs['current'][None]
    torch.testing.assert_close(p['service'], p['uninterrupted_service']*(1-moved.float()*4/30))
    raw = raw_target(sample)
    restored = raw[..., 0]*(1-moved.float()*.8)
    torch.testing.assert_close(restored, torch.tensor(sample['targets'][..., 0]), atol=1e-6, rtol=1e-6)


def test_typed_family_does_not_drop_a_literal_hard_constraint():
    with pytest.raises(UnsupportedMission):
        parse_mission('Video needs 90%; reconnect at most 2 robots', kind='video_floor')
    with pytest.raises(UnsupportedMission):
        parse_mission('Video needs 90%', kind='worst')
    with pytest.raises(UnsupportedMission):
        parse_mission('Prioritise the front half, x below 40 m')
    with pytest.raises(UnsupportedMission):
        parse_mission('Video needs -10% of its rate')
    with pytest.raises(UnsupportedMission):
        parse_mission('Robots with x above 50 m matter most')
    spec = parse_mission('Robots with x below 37 m matter most')
    assert (spec.axis, spec.side, spec.boundary_m) == ('x', 'lt', 37.)


def test_model_schema_rejects_non_numeric_thresholds_and_nonfinite_outcomes():
    with pytest.raises(ValueError):
        MissionSpec('video_floor', floor='0.9')
    with pytest.raises(ValueError):
        MissionSpec('zone', axis='x', side='lt', boundary_m=float('nan'))
    o = outcomes()
    o.moves[0] = 2
    o.jain[0] = float('nan')
    with pytest.raises(ValueError):
        choose_candidate(MissionSpec('move_budget', max_moves=5), o)


def test_executed_wifi_curve_integration_has_physical_units_and_margin():
    from wirelessjev.fastplan import BatchEvaluator
    from wirelessjev.warehouse_v2.link_curves import apply_curves
    archive = Path(__file__).resolve().parents[1]/'artifacts/warehouse_v2/studies/wifi_link_curves.json'
    curves = json.loads(archive.read_text())
    assert len(curves['points']) == 3618
    assert all(0 <= row['packet_success'] <= 1 and row['rate_mbps'] > 0 for row in curves['points'])
    scene, _ = make_environment(416, n=12)
    zero, margin = BatchEvaluator(scene, []), BatchEvaluator(scene, [])
    apply_curves(zero, curves)
    apply_curves(margin, curves, 3.)
    for threshold, rate in [('t5', 'r5'), ('t2', 'r2')]:
        np.testing.assert_allclose(getattr(margin, threshold)-getattr(zero, threshold), 3.)
        assert np.all(np.diff(getattr(zero, rate)) >= -1e-9)


def test_archived_removal_has_nonzero_contribution_and_exact_reference():
    archive = Path(__file__).resolve().parents[1]/'artifacts/warehouse_v2/studies/federation.json'
    report = json.loads(archive.read_text())
    assert {r['seed'] for r in report['records']} == {0, 1, 2}
    for row in report['records']:
        removal = row['removal']
        assert removal['site_contribution_changed_weights']
        assert removal['removed_state_sha256'] == removal['clean_reference_state_sha256']
