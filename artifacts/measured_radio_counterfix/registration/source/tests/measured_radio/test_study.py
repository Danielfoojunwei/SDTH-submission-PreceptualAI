"""Analytic reduction/bootstrap checks, never measured-data performance runs."""
from copy import deepcopy

import numpy as np
import pytest

from wirelessjev.measured_radio import study


def _query(scenario, location, phone, *, session='01'):
    return {'scenario': scenario, 'location_id': location, 'phone': phone, 'session': session}


def _predictions(queries, means, *, methods=('reference',), target=None, supported=None):
    n = len(queries)
    mu = np.asarray(means, dtype=np.float64).reshape(len(methods), n)
    known = np.ones(mu.shape, dtype=bool) if supported is None else np.asarray(supported, dtype=bool)
    y = np.zeros(n) if target is None else np.asarray(target, dtype=np.float64)
    return {'queries': queries, 'methods': list(methods)}, {'mean': mu, 'target': y, 'supported': known}


def _metadata(coordinates):
    return {'scenarios': {scenario: {'locations': [
        {'id': loc, 'xyz_provider': [float(x), float(y), .8]} for loc, (x, y) in points.items()]}
        for scenario, points in coordinates.items()}}


def test_location_reduction_uses_ap_rmse_then_equal_phones():
    queries = [_query('A', 'L1', 'P')] * 2 + [_query('A', 'L1', 'Q')]
    evidence, arrays = _predictions(queries, [[3., 4., 10.]])
    actual = study.location_metrics(evidence, arrays, 'reference')
    assert actual == {('A', 'L1'): pytest.approx((np.sqrt(12.5) + 10.) / 2.)}
    # Three AP errors pooled or three absolute errors averaged are different endpoints.
    assert actual[('A', 'L1')] != pytest.approx(np.sqrt(125. / 3.))
    assert actual[('A', 'L1')] != pytest.approx(17. / 3.)


def test_scenarios_locations_and_phones_have_declared_equal_mass(monkeypatch):
    monkeypatch.setattr(study.c, 'SCENARIOS', ('A', 'B'))
    queries = ([_query('A', 'L1', 'P')] * 2 + [_query('A', 'L1', 'Q')]
               + [_query('A', 'L2', 'P')] * 3 + [_query('B', 'L3', 'P')])
    evidence, arrays = _predictions(queries, [[0., 0., 6., 4., 4., 4., 10.]])
    assert study.location_metrics(evidence, arrays, 'reference') == {
        ('A', 'L1'): 3., ('A', 'L2'): 4., ('B', 'L3'): 10.}
    # A gets 1/2 total mass despite having two locations and six AP rows.
    assert study.metric(evidence, arrays, 'reference') == pytest.approx(((3. + 4.) / 2. + 10.) / 2.)


def test_member_errors_are_averaged_after_each_member_rmse_not_before(monkeypatch):
    monkeypatch.setattr(study.c, 'SCENARIOS', ('A',))
    monkeypatch.setattr(study.c, 'SEEDS', (0, 1, 2))
    evidence, arrays = _predictions([_query('A', 'L1', 'P')], [[1.], [-1.], [4.]],
        methods=('radiojepa-seed0', 'radiojepa-seed1', 'radiojepa-seed2'))
    assert study.metric(evidence, arrays, 'radiojepa') == pytest.approx(2.)
    assert abs(float(arrays['mean'].mean())) == pytest.approx(4. / 3.)
    assert study.metric(evidence, arrays, 'radiojepa-seed1') == 1.


def test_mean_member_rmse_order_is_preserved_with_multiple_aps(monkeypatch):
    monkeypatch.setattr(study.c, 'SEEDS', (0, 1, 2))
    evidence, arrays = _predictions([_query('A', 'L1', 'P')] * 2,
        [[3., 4.], [-3., -4.], [0., 0.]],
        methods=('radiojepa-seed0', 'radiojepa-seed1', 'radiojepa-seed2'))
    assert study.location_metrics(evidence, arrays, 'radiojepa') == {
        ('A', 'L1'): pytest.approx(2. * np.sqrt(12.5) / 3.)}
    # Averaging predictions would cancel every error in this constructed example.
    np.testing.assert_array_equal(arrays['mean'].mean(axis=0), 0.)


def test_known_mask_and_session_partition_exclude_unscored_values(monkeypatch):
    monkeypatch.setattr(study.c, 'SCENARIOS', ('A',))
    queries = [_query('A', 'L1', 'P'), _query('A', 'L1', 'P', session='02'),
               _query('A', 'L2', 'P')]
    evidence, arrays = _predictions(queries, [[8., 103., 1.e30]], target=[5., 3., -1.e30],
                                   supported=[[True, True, False]])
    assert study.location_metrics(evidence, arrays, 'reference') == {('A', 'L1'): 3.}
    assert study.metric(evidence, arrays, 'reference', session='02') == 100.
    assert study.location_metrics(evidence, arrays, 'reference', session='03') == {}


def test_no_cross_session_pooling_changes_primary_score(monkeypatch):
    monkeypatch.setattr(study.c, 'SCENARIOS', ('A',))
    evidence, arrays = _predictions([_query('A', 'L1', 'P'), _query('A', 'L1', 'P', session='02')],
                                   [[2., 200.]])
    assert study.metric(evidence, arrays, 'reference') == 2.
    arrays['mean'][0, 1] = -1.e15
    assert study.metric(evidence, arrays, 'reference') == 2.


def test_metric_rejects_a_missing_declared_scenario(monkeypatch):
    monkeypatch.setattr(study.c, 'SCENARIOS', ('A', 'B'))
    evidence, arrays = _predictions([_query('A', 'L1', 'P')], [[2.]])
    with pytest.raises(RuntimeError, match='Every declared scenario'):
        study.metric(evidence, arrays, 'reference')


@pytest.mark.parametrize('partition,sessions', [('validation', ['01']), ('test', ['01', '02'])])
def test_episode_membership_keeps_validation_session01_and_both_test_days(monkeypatch, partition, sessions):
    monkeypatch.setattr(study.c, 'SCENARIOS', ('A',))
    monkeypatch.setattr(study.c, 'PHONES', ('P',))
    monkeypatch.setattr(study.c, 'BUDGETS', (4,))
    records = []
    for stage, loc in [('train', 'T1'), ('validation', 'V1'), ('test', 'Q1')]:
        for session in ('01', '02'):
            records.append({'id': stage + '-' + session, 'scenario': 'A', 'phone': 'P',
                'partition': stage, 'session': session, 'location_id': loc, 'ap_id': 'AP1', 'rssi_dbm': -42.})
    value = {'records': records, 'metadata': {'support_location_ids': {'A': {'4': ['T1']}}}}
    calls = []

    def public_only(row):
        return {key: item for key, item in row.items() if key != 'rssi_dbm'}

    def stub_prepare(support, queries, metadata):
        assert [r['id'] for r in support] == ['train-01']
        assert metadata is value['metadata']
        assert all('rssi_dbm' not in query for query in queries)
        calls.append([query['session'] for query in queries])
        return {'support_ids': [r['id'] for r in support]}

    monkeypatch.setattr(study, 'query_metadata', public_only)
    monkeypatch.setattr(study, 'prepare_episode', stub_prepare)
    episode, = study.episodes(value, partition, 4)
    assert calls == [sessions]
    assert [q['session'] for q in episode['public_queries']] == sessions
    assert episode['private_targets'] == [-42.] * len(sessions)


def test_paired_intervals_require_identical_whole_location_cohorts(monkeypatch):
    monkeypatch.setattr(study.c, 'SCENARIOS', ('A',))
    with pytest.raises(ValueError, match='cohorts must match'):
        study.paired_intervals({('A', 'L1'): 1.}, {('A', 'L2'): 1.}, {})


def test_constant_location_differences_give_exact_stratified_macro_interval(monkeypatch):
    monkeypatch.setattr(study.c, 'SCENARIOS', ('A', 'B'))
    positions = {'A': {f'A{i}': (i * 6., 0.) for i in range(4)},
                 'B': {f'B{i}': (i * 6., 0.) for i in range(8)}}
    candidate = {(s, loc): (4. if s == 'A' else 2.) for s, locs in positions.items() for loc in locs}
    reference = {key: 5. for key in candidate}
    result = study.paired_intervals(candidate, reference, _metadata(positions))
    assert result['candidate_minus_reference_db'] == -2.
    assert result['location_resampling_95_interval'] == [-2., -2.]
    assert result['spatial_5m_block_95_sensitivity'] == [-2., -2.]
    assert result['locations'] == 12
    assert result['block_counts'] == {'A': 4, 'B': 8}
    assert result['conditional_lower_error_supported'] is True
    assert result['bootstrap_draws'] == 20000


def test_block_resampling_keeps_locations_together_and_reweights_block_sizes(monkeypatch):
    monkeypatch.setattr(study.c, 'SCENARIOS', ('A',))
    # Three singleton blocks at zero, and one four-location block at ten.
    # The point statistic is location weighted (40/7), not block weighted (2.5).
    positions = {'A': {'Z0': (0., 0.), 'Z1': (6., 0.), 'Z2': (12., 0.),
                       'T0': (18., 0.), 'T1': (18.1, 0.), 'T2': (18.2, 0.), 'T3': (18.3, 0.)}}
    candidate = {('A', loc): (10. if loc.startswith('T') else 0.) for loc in positions['A']}
    reference = {key: 0. for key in candidate}
    result = study.paired_intervals(candidate, reference, _metadata(positions))
    assert result['candidate_minus_reference_db'] == pytest.approx(40. / 7.)
    assert result['block_counts'] == {'A': 4}
    # Enumerating the binomial number of times that fourth block is selected
    # gives 2.5% = 0 and 97.5% = three copies: 120/(3*4+1).
    assert result['spatial_5m_block_95_sensitivity'] == pytest.approx([0., 120. / 13.])
    assert result['location_resampling_95_interval'][0] > 0.
    assert result['conditional_lower_error_supported'] is False


def test_fewer_than_four_blocks_prevents_superiority_statement(monkeypatch):
    monkeypatch.setattr(study.c, 'SCENARIOS', ('A', 'B'))
    positions = {'A': {f'A{i}': (i * 6., 0.) for i in range(4)},
                 'B': {f'B{i}': (i * 6., 0.) for i in range(3)}}
    candidate = {(s, loc): 1. for s, locs in positions.items() for loc in locs}
    reference = {key: 2. for key in candidate}
    result = study.paired_intervals(candidate, reference, _metadata(positions))
    assert result['location_resampling_95_interval'] == [-1., -1.]
    assert result['spatial_5m_block_95_sensitivity'] is None
    assert result['block_counts'] == {'A': 4, 'B': 3}
    assert result['conditional_lower_error_supported'] is False


def test_paired_bootstrap_is_reproducible_and_invariant_to_shared_offset_translation(monkeypatch):
    monkeypatch.setattr(study.c, 'SCENARIOS', ('A',))
    positions = {'A': {f'L{i}': (i * 6., -i * 2.) for i in range(5)}}
    candidate = {('A', f'L{i}'): float(i + 1) for i in range(5)}
    reference = {key: 2. for key in candidate}
    metadata = _metadata(positions)
    before = deepcopy(metadata)
    expected = study.paired_intervals(candidate, reference, metadata)
    assert expected == study.paired_intervals(candidate, reference, metadata)
    translated = deepcopy(metadata)
    for row in translated['scenarios']['A']['locations']:
        row['xyz_provider'] = (np.asarray(row['xyz_provider']) + [-20., 15., -3.]).tolist()
    shifted_candidate = {key: value + 13. for key, value in candidate.items()}
    shifted_reference = {key: value + 13. for key, value in reference.items()}
    assert expected == study.paired_intervals(shifted_candidate, shifted_reference, translated)
    assert metadata == before
