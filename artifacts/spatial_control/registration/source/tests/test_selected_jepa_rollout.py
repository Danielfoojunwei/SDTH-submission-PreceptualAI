"""Real frozen runtime/kernel fixtures and immutable selected-case archives."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch

from wirelessjev.fleetwm.data import episode
from wirelessjev.frontier.study import mask_scans
from wirelessjev.semantic_jepa.common import OUT as FROZEN_OUT
from wirelessjev.semantic_jepa.runtime import Predictor
from wirelessjev.selected_jepa.rollout import (
    _evaluate, _read_record, _write_record, arms, case_keys, case_name,
    run_case, selected_family, validate_case_key,
)


@pytest.fixture
def registration():
    # Known software fixtures, not any of the fresh scientific IDs.
    return {'candidate_family': 'context_jepa', 'confirmation_sites': [745, 746],
            'handover_seconds': [.03, 4.], 'training_seeds': [0, 1, 2],
            'robots': 2, 'duration_seconds': 3, 'mission': 'bulk_uploads',
            'scan_period_seconds': 5, 'scan_outage_seconds': .09,
            'extra_compute_seconds': .5}


@pytest.fixture
def selection():
    return {'selected_family': 'context_jepa', 'sealed': True}


def test_registered_matrix_preserves_actual_method_and_seed_identities(registration, selection):
    assert selected_family(registration, selection) == 'context_jepa'
    matrix = arms(registration, selection)
    assert len(matrix) == 8 and len(set(matrix)) == 8
    assert matrix[:2] == [('radiomap', None), ('frozen_fleet_jepa', None)]
    keys = case_keys(registration, selection)
    assert len(keys) == 32 and len({case_name(key) for key in keys}) == 32
    for key in keys:
        validate_case_key(key, registration, selection)
    assert case_name((745, .03, 'context_jepa', 2)) == 'confirmation-745-h0030-context_jepa-s2'


@pytest.mark.parametrize('key', [
    (2114000, .03, 'context_jepa', 0),  # Reserved scientific site cannot be a unit fixture.
    (745, .06, 'context_jepa', 0), (745, float('nan'), 'context_jepa', 0),
    (745, .03, 'semantic_jepa', 0), (745, .03, 'context_jepa', False),
    (745, .03, 'radiomap', 0), (745, .03, 'context_jepa', None),
    (True, .03, 'context_jepa', 0), (745, .03, '../escape', 0),
])
def test_unregistered_keys_are_rejected_without_opening_a_world(registration, selection, key):
    with pytest.raises(ValueError, match='Unregistered'):
        validate_case_key(key, registration, selection)


def test_selection_mismatch_and_non_jepa_selection_are_rejected(registration, selection):
    with pytest.raises(ValueError, match='differ'):
        selected_family(registration, {'selected_family': 'semantic_jepa'})
    with pytest.raises(ValueError, match='JEPA family'):
        selected_family(registration, {'selected_family': 'supervised'})
    with pytest.raises(ValueError, match='sealed'):
        selected_family(registration, {'selected_family': 'context_jepa', 'sealed': False})
    bad = dict(registration, training_seeds=[0, 0])
    with pytest.raises(ValueError, match='distinct'):
        arms(bad, selection)


def test_public_entry_rejects_unregistered_fixture_before_registration_or_world():
    with pytest.raises(ValueError, match='Unregistered'):
        run_case((745, .03, 'context_jepa', 0))


@pytest.fixture(scope='module')
def actual_numerical_row():
    torch.set_num_threads(1)
    p = {'robots': 2, 'duration_seconds': 3, 'mission': 'bulk_uploads',
         'scan_period_seconds': 5, 'scan_outage_seconds': .09,
         'extra_compute_seconds': .5}
    profile = json.loads((FROZEN_OUT / 'latency.json').read_text())['profiles']['radiomap']
    value = _evaluate((745, .03, 'radiomap', None), p, profile)
    value['replay'] = False
    return value, p, profile


def test_actual_numerical_rollout_keeps_exact_counts_and_paid_deliveries(actual_numerical_row):
    value, _, _ = actual_numerical_row
    assert value['stage'] == 'confirmation'
    assert value['case'] == [745, .03, 'radiomap', None]
    assert value['causality_pass'] is True
    assert value['hardware_qualified'] is False and value['simulation_only'] is True
    assert len(value['jobs']) == 3
    result = value['result']; detail = result['packet_detail']
    assert sum(detail['sent']) == result['control_sent'] == 120
    assert sum(detail['misses']) == result['control_misses']
    assert detail['pending_control_right_censored'] is True
    assert len(detail['action_trace']) == 30
    assert all(job['history_cutoff_exclusive'] == 60 + i for i, job in enumerate(value['jobs']))
    for event in value['telemetry']['events']:
        assert event['measurement_slot'] < event['complete_slot'] <= event['delivery_slot']
        assert event['first_usable_decision'] == event['delivery_slot'] // 100 + 1
    assert 'Issued forecasts' in value['forecast_metric_scope']
    assert all(np.isfinite(metric) for metric in value['forecast_metrics'].values())


def test_real_numerical_rollout_replays_results_forecasts_and_actions_exactly(actual_numerical_row):
    value, p, profile = actual_numerical_row
    again = _evaluate((745, .03, 'radiomap', None), p, profile)
    for name in ('result', 'forecast_mean_sha256', 'forecast_std_sha256',
                 'telemetry', 'decision_trace', 'action_trace_sha256'):
        assert again[name] == value[name]


def test_real_frozen_trained_member_first_forecast_uses_only_past_public_inputs(registration):
    torch.set_num_threads(1)
    key = (745, .03, 'context_jepa', 0)
    profile = json.loads((FROZEN_OUT / 'latency.json').read_text())['profiles']['context_jepa-s0']
    value = _evaluate(key, registration, profile)
    full, _, geometry, observed = episode(745, 2, 3, 'drift')
    past = mask_scans(observed, 5).before(60)
    mean, _, _, _ = Predictor(geometry, 'context_jepa', 0).step(past, full.plan[60])
    assert value['jobs'][0]['forecast_sha256'] == hashlib.sha256(mean.tobytes()).hexdigest()
    assert value['family'] == 'context_jepa' and value['training_seed'] == 0
    assert value['causality_pass'] is True
    assert value['result']['control_sent'] == 120


def test_actual_slow_rollout_uses_existing_utility_without_rewriting_the_policy(registration):
    torch.set_num_threads(1)
    profile = json.loads((FROZEN_OUT / 'latency.json').read_text())['profiles']['radiomap']
    value = _evaluate((746, 4., 'radiomap', None), registration, profile)
    assert value['handover_seconds'] == 4.
    assert value['causality_pass'] is True and len(value['jobs']) == 3
    assert value['result']['duration_s'] == 3
    assert value['result']['control_sent'] == 120


def test_actual_rollout_archive_hash_tampering_and_partial_write_fail_closed(tmp_path, actual_numerical_row):
    value, _, _ = actual_numerical_row
    key = tuple(value['case'])
    _write_record(tmp_path, key, value)
    assert _read_record(tmp_path, key) == value
    _write_record(tmp_path, key, value)  # Identical repeat is idempotent.
    changed = deepcopy(value); changed['result']['control_misses'] += 1
    with pytest.raises(RuntimeError, match='differs'):
        _write_record(tmp_path, key, changed)
    receipt_path = tmp_path / (case_name(key) + '.receipt.json')
    receipt = json.loads(receipt_path.read_text()); receipt['result_sha256'] = '0' * 64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match='Result receipt'):
        _read_record(tmp_path, key)
    receipt_path.unlink()
    with pytest.raises(RuntimeError, match='Partial'):
        _read_record(tmp_path, key)
    assert _read_record(tmp_path / 'absent', key) is None


def test_archive_detects_compressed_body_mutation_before_decoding(tmp_path, actual_numerical_row):
    value, _, _ = actual_numerical_row
    key = tuple(value['case'])
    _write_record(tmp_path, key, value)
    path = tmp_path / (case_name(key) + '.json.gz')
    data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    with pytest.raises(RuntimeError, match='hash'):
        _read_record(tmp_path, key)
