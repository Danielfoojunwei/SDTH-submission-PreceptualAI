"""Genuine native-delivery collection fixtures; no registered scientific sites."""
import json

import numpy as np
import pytest

from wirelessjev.frontier.study import mask_scans
from wirelessjev.fleetwm.data import episode
from wirelessjev.semantic_jepa.collection import (
    COLLECTION_PROFILE_SECONDS, collect,
)


@pytest.fixture(scope='module')
def collected():
    import torch
    torch.set_num_threads(1)
    return collect(12347, 2, 2, 'main', 0)


def test_all_postwarm_rf_values_require_actual_delivered_report_bins(collected):
    full, geometry, observed, receipt = collected
    assert observed.value.shape == (92, 2, len(geometry.sites) * 3)
    assert len(full.plan) > len(observed.value)
    actual = np.argwhere(observed.scanned[60:]).tolist()
    assert actual == receipt['delivered_history_bins']
    assert len(actual) <= receipt['telemetry']['reports_delivered']
    assert receipt['telemetry']['reports_delivered'] > 0
    assert receipt['telemetry']['reports_delivered'] <= receipt['telemetry']['scans_started']
    events = receipt['telemetry_events']
    assert len(events) == receipt['telemetry']['reports_delivered']
    assert receipt['delivery_bin_collision_events'] == len(events) - len(actual)
    assert all(event['measurement_slot'] < event['complete_slot'] <= event['delivery_slot']
               < event['first_usable_decision'] * 100 for event in events)
    missing = ~observed.scanned[60:]
    assert not observed.heard[60:][missing].any()
    assert (observed.value[60:][missing] == 0).all()
    assert (observed.bound[60:][missing] == 0).all()
    # Public pose still exists in missing bins, without inventing RSSI.
    assert np.any(observed.positions[60:][missing] != 0)


def test_warm_start_is_separate_from_postwarm_packet_transport(collected):
    _, _, observed, receipt = collected
    _, _, _, original = episode(12347, 2, 32, 'main')
    expected = mask_scans(original, 5).before(60)
    for name in ('positions', 'value', 'heard', 'bound', 'scanned'):
        np.testing.assert_array_equal(getattr(observed, name)[:60], getattr(expected, name))
    assert receipt['warm_start_seconds'] == 60
    assert receipt['warm_start_is_conditional_simulated_survey'] is True
    assert receipt['postwarm_observations_use_actual_deliveries_only'] is True
    # The independent full scan stream is not silently used after warm start.
    assert observed.scanned[60:].sum() < original.scanned[60:92].sum()


def test_future_context_extension_and_actual_causal_forecast_cutoffs(collected):
    _, _, observed, receipt = collected
    assert receipt['query_duration_seconds'] == 2
    assert receipt['collection_duration_seconds'] == 32
    jobs = receipt['forecast_jobs']
    assert [row['history_cutoff_exclusive'] for row in jobs] == list(range(60, 92))
    assert [row['decision'] for row in jobs] == list(range(32))
    assert jobs[0]['ready_slot'] == 150  # Declared 1 s cold + .5 s FIFO charge.
    assert jobs[0]['charged_seconds'] == 1.5
    assert jobs[1]['ready_slot'] >= jobs[0]['ready_slot']
    assert receipt['future_truth_is_predictor_input'] is False
    assert receipt['latest_observation_strictly_before_decision'] is True
    for row in jobs:
        last = row['history_cutoff_exclusive'] - 1
        assert row['heard_entries'] == int(observed.heard[last].sum())
        assert row['scans_last_bin'] == int(observed.scanned[last].sum())
        assert 'actual_prediction_seconds' not in row and 'runtime_overrun' not in row
    # Every queried future horizon, including30s, has an actual history prefix.
    for query in (60, 61):
        assert len(observed.before(query + 30).value) == query + 30


def test_shared_policy_receipt_is_deterministic_and_contains_no_measured_budget_claim(collected):
    _, _, observed, receipt = collected
    _, _, repeated, second = collect(12347, 2, 2, 'main', 0)
    assert second == receipt
    for name in ('positions', 'value', 'heard', 'bound', 'scanned'):
        np.testing.assert_array_equal(getattr(repeated, name), getattr(observed, name))
    assert receipt['collector_policy']['profile_seconds'] == COLLECTION_PROFILE_SECONDS
    assert receipt['collector_policy']['extra_compute_seconds'] == .5
    assert receipt['collector_policy']['scan_outage_seconds'] == .09
    assert receipt['collection_profile_is_training_assumption'] is True
    assert receipt['collection_profile_is_measured_qualification'] is False
    assert receipt['actual_wall_clock_timings_omitted_from_stable_receipt'] is True
    assert receipt['training_collection_shared_by_all_families'] is True
    assert receipt['simulation_only'] is True
    for name in ('telemetry_events_sha256', 'mean_forecast_sha256',
                 'std_forecast_sha256', 'action_trace_sha256', 'packet_result_sha256'):
        assert len(receipt[name]) == 64
    json.dumps(receipt, allow_nan=False)


def test_slow_collection_uses_same_declared_utility_policy_with_real_delivery():
    import torch
    torch.set_num_threads(1)
    _, _, observed, receipt = collect(12348, 2, 2, 'drift', 4)
    assert receipt['collection_handover_seconds'] == 4.
    assert receipt['collector_policy']['controller'] == 'outage_utility_30'
    assert receipt['condition'] == 'drift'
    assert receipt['index'] == 4
    assert receipt['delivered_history_bins'] == np.argwhere(observed.scanned[60:]).tolist()
    assert receipt['collector_policy']['reactive_rescue'] is True


@pytest.mark.parametrize('args', [
    (-1, 2, 2, 'main', 0), (12347, 0, 2, 'main', 0),
    (12347, 2, 0, 'main', 0), (12347, 2, 2, 'unknown', 0),
    (12347, 2, 2, 'main', -1), (True, 2, 2, 'main', 0),
])
def test_invalid_collection_requests_fail_before_generation(args):
    with pytest.raises(ValueError):
        collect(*args)
