from dataclasses import asdict
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wirelessjev.warehouse_horizon.world import World, observe, interruption_keep
from wirelessjev.warehouse_horizon.belief import RadioBelief
from wirelessjev.warehouse_horizon.model import HorizonModel, tensors, prior_prediction, loss
from wirelessjev.warehouse_horizon.planning import (Contract, compile_contract, MISSION_TEXTS,
    candidate_pool, feasible, neighbours, charge, budgeted_signal, choose)
from wirelessjev.warehouse_horizon.data import save_examples, load_examples


def problem(n=12):
    world = World(145999, n=n, steps=12)
    history = []
    for tick in range(4):
        observation = observe(world.frames[tick], world.initial, tick,
                              previous_service=np.ones(n), since_switch=np.full(n, 30.), history=history)
    belief = RadioBelief(observation)
    plans, _ = candidate_pool(observation, belief, Contract('throughput', 3, 1.))
    return world, observation, belief, plans


def test_interruption_is_charged_once_and_can_cross_intervals():
    moved = np.array([[True, False]])
    for seconds in [0., 1., 4., 8., 15., 30.]:
        keep = interruption_keep(moved, outage_s=seconds)
        assert (5.*(1-keep[0, :, 0])).sum() == pytest.approx(seconds)
        assert (keep[0, :, 1] == 1.).all()
    keep = interruption_keep(moved, outage_s=8.)
    np.testing.assert_allclose(keep[0, :, 0], [0., .4, 1., 1., 1., 1.])
    with pytest.raises(ValueError):
        interruption_keep(moved, outage_s=-1.)


def test_counterfactuals_preserve_route_rng_and_radio_arrays():
    world, obs, belief, plans = problem()
    frame = world.frames[3]
    state = json.dumps(frame.scene.mobility.rng.bit_generator.state, sort_keys=True)
    routes = json.dumps({k: asdict(v) for k, v in frame.scene.mobility.plans.items()}, sort_keys=True)
    ul = world.frames[4].scene.radio().ul.copy()
    first = world.outcomes(3, plans, obs['current'])
    second = world.outcomes(3, plans[::-1], obs['current'])
    for key in first:
        np.testing.assert_array_equal(first[key][::-1], second[key])
    assert state == json.dumps(frame.scene.mobility.rng.bit_generator.state, sort_keys=True)
    assert routes == json.dumps({k: asdict(v) for k, v in frame.scene.mobility.plans.items()}, sort_keys=True)
    np.testing.assert_array_equal(ul, world.frames[4].scene.radio().ul)


def test_multistep_targets_distinguish_switch_cost_from_radio_forecast():
    world, obs, belief, plans = problem()
    target = world.outcomes(3, plans, obs['current'])
    moved = plans != obs['current'][None]
    np.testing.assert_allclose(target['service'], target['service_raw']*interruption_keep(moved), atol=1e-6)
    np.testing.assert_array_equal(target['failure'][:, 1:], target['failure_raw'][:, 1:])
    assert target['failure'][:, 0][moved].all()
    assert target['service'].shape[1] == 6


def test_partial_moves_and_neighbour_swaps_respect_budget_and_scan():
    world, obs, belief, plans = problem()
    contract = Contract('throughput', 3, 1.)
    assert feasible(plans, obs, contract).all()
    for plan in plans:
        assert feasible(neighbours(plan, obs, belief, contract), obs, contract).all()
    no_moves = Contract('throughput', 0, 1.)
    zero, _ = candidate_pool(obs, belief, no_moves)
    assert (zero == obs['current']).all()
    invalid = plans[:1].copy(); invalid[0, 0] = len(obs['ap'])
    with pytest.raises(ValueError):
        feasible(invalid, obs, contract)
    assert (budgeted_signal(obs, 3) != obs['current']).sum() <= 3


def test_actual_curve_belief_and_zero_residual_agree_at_every_offset():
    torch.set_num_threads(2)
    world, obs, belief, plans = problem()
    prior = belief.forecast(plans)
    model = HorizonModel()
    output = model(tensors(obs), torch.tensor(plans), torch.tensor(prior))
    expected = prior_prediction(prior, obs)
    for key in expected:
        np.testing.assert_allclose(output[key].detach().numpy(), expected[key], atol=1e-5)
    target = {k: torch.tensor(v) for k, v in world.outcomes(3, plans, obs['current']).items()}
    objective = loss(output, target)
    objective.backward()
    assert torch.isfinite(objective)
    assert model.decoder[-1].weight.grad.abs().sum() > 0


def test_forecaster_robot_permutation_equivariance_with_nonzero_residual():
    torch.manual_seed(11)
    world, obs, belief, plans = problem()
    prior = torch.tensor(belief.forecast(plans))
    model = HorizonModel()
    torch.nn.init.normal_(model.decoder[-1].weight, std=.01)
    original = tensors(obs)
    permutation = torch.randperm(12)
    changed = dict(original)
    for name in ['robot', 'current', 'critical', 'bulk', 'video']:
        changed[name] = original[name][permutation]
    for name in ['history', 'past_xy', 'route']:
        changed[name] = original[name][:, permutation]
    with torch.no_grad():
        before = model(original, torch.tensor(plans), prior)
        after = model(changed, torch.tensor(plans)[:, permutation], prior[:, :, permutation])
    for key in before:
        torch.testing.assert_close(after[key], before[key][:, :, permutation], atol=1e-5, rtol=1e-5)


def test_public_cache_has_no_future_truth_in_observation(tmp_path):
    world, obs, belief, plans = problem()
    sample = {'observation': obs, 'plans': plans, 'prior': belief.forecast(plans),
              'target': world.outcomes(3, plans, obs['current']), 'scene': world.seed, 'tick': 3}
    path = tmp_path/'episode.npz'
    save_examples(path, [sample], 'verified-source')
    reloaded = load_examples(path, 'verified-source', world.seed)[0]
    assert set(reloaded['observation']) == set(obs)
    assert not {'target', 'spec', 'truth', 'future_actual_xy'} & set(obs)
    np.testing.assert_array_equal(reloaded['observation']['route'], obs['route'])
    with pytest.raises(RuntimeError):
        load_examples(path, 'changed-source', world.seed)


def test_composite_contract_and_numerical_planner_execute():
    world, obs, belief, plans = problem()
    contract = compile_contract(MISSION_TEXTS['tail'])
    assert contract.objective == 'tail' and contract.max_moves == 12
    assert contract.tail_fraction == .1 and contract.critical_penalty == 1.
    with pytest.raises(ValueError):
        compile_contract(MISSION_TEXTS['tail'], 'throughput')
    def forecast(A):
        return prior_prediction(belief.forecast(A), obs)
    action, receipt = choose(obs, Contract('tail', 3, 1.), belief, forecast)
    assert feasible(action[None], obs, Contract('tail', 3, 1.))[0]
    assert receipt['candidates'] > 1
    assert receipt['physical_safety_guarantee'] is False


def gateway():
    path = Path(__file__).resolve().parents[1]/'scripts/plan_warehouse_horizon.py'
    spec = importlib.util.spec_from_file_location('horizon_proposal_gateway', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observation_document():
    world, obs, belief, plans = problem()
    return {'schema': 'wirelessjev.horizon_observation.v1', 'source': 'simulation',
            'warehouse_size_m': [120, 80], 'offsets_s': [5, 10, 15, 20, 25, 30],
            'robot_ids': [f'amr-{i:03d}' for i in range(len(obs['current']))],
            'ap_ids': [f'sim-bss-{i}' for i in range(len(obs['ap']))],
            'observation': {k: v.tolist() for k, v in obs.items()}}


def test_proposal_gateway_checks_observed_units_masks_and_whole_contract():
    module = gateway()
    document = observation_document()
    result = module.validate_observation(document)
    assert result['history'].shape[0] == 4
    assert module.validate_controlled_text(MISSION_TEXTS['throughput']).max_moves == 12
    with pytest.raises(ValueError):
        module.validate_controlled_text(MISSION_TEXTS['throughput']+' Also switch every robot.')
    with pytest.raises(ValueError):
        module.validate_controlled_text(MISSION_TEXTS['throughput'].replace('maximise total data delivered',
                                           'maximise total data delivered and worst-served 10%'))
    broken = deepcopy(document)
    broken['observation']['current'][0] = 1.5
    with pytest.raises(ValueError):
        module.validate_observation(broken)
    broken = deepcopy(document)
    broken['observation']['robot'][0][2] = 999.
    with pytest.raises(ValueError):
        module.validate_observation(broken)
    broken = deepcopy(document)
    broken['observation']['history'][0][0][0][2] = 123.
    with pytest.raises(ValueError):
        module.validate_observation(broken)


def test_hardware_label_needs_real_bssid_format_and_fresh_timestamp():
    module = gateway()
    document = observation_document()
    document['source'] = 'hardware'
    document['observed_at'] = '2026-01-01T00:00:00+00:00'
    with pytest.raises(ValueError):
        module.validate_observation(document)
    document['ap_ids'] = [f'02:00:00:00:00:{i:02x}' for i in range(len(document['ap_ids']))]
    with pytest.raises(ValueError, match='stale'):
        module.validate_observation(document, now=datetime(2026, 10, 31, tzinfo=timezone.utc))


def script_module(name):
    path = Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selector_does_not_mix_independently_accepted_mission_phases():
    module = script_module('validate_warehouse_horizon_gate')
    baseline = {'selected_strength': 0., 'temperature': 2.}
    candidate = {'selected_strength': .5, 'temperature': 1.}
    for throughput, tail in [(True, False), (False, True), (False, False), (True, True)]:
        recipe = module.whole_schedule_recipe(
            {'throughput': {'accepted': throughput}, 'tail': {'accepted': tail}}, baseline, candidate)
        expected = candidate if throughput and tail else baseline
        assert recipe['whole_schedule_accepted'] == (throughput and tail)
        assert not recipe['independent_phase_mixing']
        for branch in recipe['by_mission'].values():
            assert branch['selected_strength'] == expected['selected_strength']
            assert branch['temperature'] == expected['temperature']


def test_archived_selector_falls_back_and_rejects_tampered_policy(tmp_path):
    import shutil
    module = gateway()
    archive = Path(__file__).resolve().parents[1]/'artifacts/warehouse_horizon'
    world, obs, belief, plans = problem()
    contract = compile_contract(MISSION_TEXTS['throughput'])
    recipe, receipt = module.guarded_recipe(archive, '0-False', contract, obs)
    assert receipt['mode'] == 'numerical_fallback'
    assert receipt['reason'] == 'closed_loop_validation_rejected'
    assert not receipt['model_executed'] and recipe['selected_strength'] == 0.
    with pytest.raises(ValueError, match='simulation input only'):
        module.guarded_recipe(archive, '0-False', contract, obs, research=True, source='hardware')
    shutil.copyfile(archive/'selection.json', tmp_path/'selection.json')
    recipe, receipt = module.guarded_recipe(tmp_path, '0-False', contract, obs)
    assert recipe['selected_strength'] == 0. and receipt['reason'].endswith('missing')
    shutil.copytree(archive/'selector', tmp_path/'selector')
    policy = tmp_path/'selector/gate_policy.json'
    broken = json.loads(policy.read_text())
    broken['models']['0-False']['whole_schedule_accepted'] = True
    policy.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match='receipt mismatch'):
        module.guarded_recipe(tmp_path, '0-False', contract, obs)


def test_prospective_audit_proves_rejected_fallback_is_numerical():
    import gzip
    module = script_module('archive_warehouse_horizon_gate')
    archive = Path(__file__).resolve().parents[1]/'artifacts/warehouse_horizon/selector'
    with gzip.open(archive/'gate_audit.json.gz', 'rt') as stream:
        report = json.load(stream)
    policy = json.loads((archive/'gate_policy.json').read_text())
    summary = module.summarize(report, policy)
    assert summary['audit_environments'] == [143000, 143001, 143002]
    assert summary['accepted_learned_models'] == 0 and summary['verified_identical_fallbacks'] == 18
    assert summary['validation_rollouts'] == 42 and summary['audit_rollouts'] == 24
    candidate = next(r for r in report['records'] if r['method'] == 'guarded_residual')
    candidate['metrics']['mean_service'] += .01
    with pytest.raises(RuntimeError, match='did not reproduce'):
        module.summarize(report, policy)
