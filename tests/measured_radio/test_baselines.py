"""Analytic software checks; fixtures are not measured RF performance data."""
from copy import deepcopy
import math

import numpy as np
import pytest

from wirelessjev.measured_radio.baselines import configuration_grid, predict_baseline


def _config(method, **criteria):
    return next(c for c in configuration_grid() if c['method'] == method
                and all(c.get(k) == value for k, value in criteria.items()))


def _episode(xyz, rssi, asks, *, groups=None, ask_groups=None, ap_xyz=None):
    """Construct the common point-report schema without fitting a field model."""
    x = np.asarray(xyz, dtype='f8').reshape(-1, 3)
    q = np.asarray(asks, dtype='f8').reshape(-1, 3)
    y = np.asarray(rssi, dtype='f8')
    ap_xyz = np.asarray([[0., 0., 0.]] if ap_xyz is None else ap_xyz, dtype='f8')
    ap = np.zeros(len(x), dtype='i8') if groups is None else np.asarray(groups, dtype='i8')
    qa = np.zeros(len(q), dtype='i8') if ask_groups is None else np.asarray(ask_groups, dtype='i8')
    sf = -20. * np.log10(np.linalg.norm(x - ap_xyz[ap], axis=1))
    qf = -20. * np.log10(np.linalg.norm(q - ap_xyz[qa], axis=1))
    refs = {int(a): float((y[ap == a] - sf[ap == a]).mean()) for a in np.unique(ap)}
    so = np.asarray([refs[int(a)] for a in ap])
    qo = np.asarray([refs[int(a)] for a in qa])
    residual = y - sf - so
    return {'view': {'ap': ap, 'rx_xyz': x, 'fspl': sf, 'offset': so,
                     'lower': residual, 'upper': residual.copy(),
                     'known': np.ones(len(x), dtype=bool), 'scheduled': np.ones(len(x), dtype=bool)},
            'query': {'ap': qa, 'rx_xyz': q, 'fspl': qf, 'offset': qo},
            'ap_xyz': ap_xyz, 'query_indices': list(range(len(q))),
            'statuses': [{'query_id': f'q{i}', 'status': 'prediction', 'reason': None} for i in range(len(q))],
            'support_ids': [f's{i}' for i in range(len(x))]}


def test_finite_grid_order_size_and_fresh_dicts():
    grid = configuration_grid()
    assert len(grid) == 172 == len({c['id'] for c in grid})
    assert [c['id'] for c in grid[:3]] == ['reference', 'logdistance', 'gp_rbf_l1_a3_n1']
    assert grid[55]['id'] == 'gp_rbf_l40_a12_n4'
    assert grid[56]['id'] == 'gp_matern12_l1_a3_n1'
    assert grid[110]['id'] == 'gp_matern32_l1_a3_n1'
    assert [c['id'] for c in grid[-8:]] == ['idw_k1_p1', 'idw_k1_p2', 'idw_k3_p1', 'idw_k3_p2',
                                         'idw_k5_p1', 'idw_k5_p2', 'idw_k9_p1', 'idw_k9_p2']
    grid[0]['id'] = 'changed'
    assert configuration_grid()[0]['id'] == 'reference'


def test_reference_is_support_only_unknown_intercept_plus_free_space_decay():
    e = _episode([[1., 0., 0.], [10., 0., 0.]], [-40., -70.], [[math.sqrt(10), 0., 0.]])
    row, = predict_baseline(e, _config('reference'))
    assert row == {'query_id': 'q0', 'status': 'prediction', 'reason': None,
                   'mean_dbm': pytest.approx(-55.), 'std_db': None}


def test_fitted_logdistance_recovers_non_free_space_slope():
    e = _episode([[1., 0., 0.], [10., 0., 0.]], [-10., -40.], [[math.sqrt(10), 0., 0.], [100., 0., 0.]])
    rows = predict_baseline(e, _config('logdistance'))
    assert [r['mean_dbm'] for r in rows] == pytest.approx([-25., -70.])
    assert all(r['std_db'] is None for r in rows)


@pytest.mark.parametrize('powers,expected', [([-50., -20.], [-35., -35.]), ([-10., -110.], [-30., -90.])])
def test_logdistance_slope_clipping_refits_intercept(powers, expected):
    e = _episode([[1., 0., 0.], [10., 0., 0.]], powers, [[1., 0., 0.], [10., 0., 0.]])
    assert [r['mean_dbm'] for r in predict_baseline(e, _config('logdistance'))] == pytest.approx(expected)


@pytest.mark.parametrize('positions,powers', [([[1., 0., 0.]], [-50.]),
                                          ([[1., 0., 0.], [-1., 0., 0.]], [-10., -20.])])
def test_logdistance_falls_back_to_common_reference_without_distinct_distances(positions, powers):
    e = _episode(positions, powers, [[2., 0., 0.]])
    assert predict_baseline(e, _config('logdistance')) == predict_baseline(e, _config('reference'))


def test_idw_interpolates_residual_not_raw_mobile_rssi():
    e = _episode([[1., 0., 0.], [4., 0., 0.]], [-30., -50.], [[2., 0., 0.]])
    c = _config('idw', neighbors=3, power=2)
    row, = predict_baseline(e, c)
    # Distances one/two give weights 1 and 1/4, independently of the AP distance.
    residual = e['view']['lower']
    expected = e['query']['fspl'][0] + e['query']['offset'][0] + (residual[0] + residual[1] / 4.) / 1.25
    assert row['mean_dbm'] == pytest.approx(expected)
    assert row['std_db'] is None


def test_idw_exact_duplicate_coordinates_average_all_actual_reports_even_for_k1():
    e = _episode([[1., 0., 0.], [1., 0., 0.], [3., 0., 0.]], [-30., -40., -90.], [[1., 0., 0.]])
    row, = predict_baseline(e, _config('idw', neighbors=1, power=1))
    assert row['mean_dbm'] == pytest.approx(-35.)
    assert e['support_ids'] == ['s0', 's1', 's2']


def test_idw_equal_distance_tie_uses_common_source_order():
    e = _episode([[1., 0., 0.], [3., 0., 0.]], [-30., -90.], [[2., 0., 0.]])
    row, = predict_baseline(e, _config('idw', neighbors=1, power=1))
    assert row['mean_dbm'] == pytest.approx(e['query']['fspl'][0] + e['query']['offset'][0] + e['view']['lower'][0])


@pytest.mark.parametrize('kernel', ['rbf', 'matern12', 'matern32'])
def test_single_support_ordinary_kriging_closed_form(kernel):
    c = _config('gp', kernel=kernel, lengthscale_m=2., amplitude_db=3., noise_std_db=1.)
    e = _episode([[1., 0., 0.]], [-40.], [[1., 0., 0.], [3., 0., 0.]])
    rows = predict_baseline(e, c)
    correlation = {'rbf': math.exp(-.5), 'matern12': math.exp(-1.),
                   'matern32': (1. + math.sqrt(3.)) * math.exp(-math.sqrt(3.))}[kernel]
    assert [r['mean_dbm'] for r in rows] == pytest.approx([-40., -40. - 20. * math.log10(3.)])
    # An unknown constant from one noisy report contributes its own uncertainty.
    assert rows[0]['std_db'] ** 2 == pytest.approx(2. + 1.e-8)
    assert rows[1]['std_db'] ** 2 == pytest.approx(18. * (1. - correlation) + 2. + 1.e-8)


@pytest.mark.parametrize('kernel', ['rbf', 'matern12', 'matern32'])
def test_two_support_kriging_matches_symmetric_analytic_solution(kernel):
    c = _config('gp', kernel=kernel, lengthscale_m=2., amplitude_db=3., noise_std_db=1.)
    e = _episode([[1., 0., 0.], [-1., 0., 0.]], [-45., -35.], [[1., 0., 0.]])
    correlation = {'rbf': math.exp(-.5), 'matern12': math.exp(-1.),
                   'matern32': (1. + math.sqrt(3.)) * math.exp(-math.sqrt(3.))}[kernel]
    b = 9. * correlation
    expected = -40. - 5. * (9. - b) / (10. + 1.e-8 - b)
    row, = predict_baseline(e, c)
    assert row['mean_dbm'] == pytest.approx(expected)
    assert row['std_db'] > 1.


@pytest.mark.parametrize('kernel', ['rbf', 'matern12', 'matern32'])
def test_gp_repeated_actual_coordinates_remain_finite(kernel):
    e = _episode([[1., 0., 0.], [1., 0., 0.]], [-30., -40.], [[1., 0., 0.]])
    row, = predict_baseline(e, _config('gp', kernel=kernel))
    assert row['mean_dbm'] == pytest.approx(-35.)
    assert math.isfinite(row['std_db']) and row['std_db'] > 0.


@pytest.mark.parametrize('config', [_config('reference'), _config('logdistance'), _config('idw'),
                                  _config('gp', kernel='rbf'), _config('gp', kernel='matern12'),
                                  _config('gp', kernel='matern32')])
def test_computational_gauge_and_rigid_translation_do_not_change_predictions(config):
    e = _episode([[1., 0., 0.], [3., 2., 0.]], [-30., -50.], [[2., 1., 0.]])
    expected = predict_baseline(e, config)
    changed = deepcopy(e)
    for key in ('view', 'query'):
        changed[key]['fspl'] += 200.
        changed[key]['offset'] -= 200.
        changed[key]['rx_xyz'] += [-4., 9., -12.]
    changed['ap_xyz'] += [-4., 9., -12.]
    row, = predict_baseline(changed, config)
    assert row['mean_dbm'] == pytest.approx(expected[0]['mean_dbm'], abs=1.e-11)
    if row['std_db'] is not None:
        assert row['std_db'] == pytest.approx(expected[0]['std_db'], abs=1.e-11)


def test_per_ap_context_never_leaks_other_ap_power():
    e = _episode([[1., 0., 0.], [11., 0., 0.]], [-30., -90.], [[2., 0., 0.], [12., 0., 0.]],
                 groups=[0, 1], ask_groups=[0, 1], ap_xyz=[[0., 0., 0.], [10., 0., 0.]])
    for c in configuration_grid():
        rows = predict_baseline(e, c)
        assert [r['mean_dbm'] for r in rows] == pytest.approx([-30. - 20. * math.log10(2.), -90. - 20. * math.log10(2.)])


def test_preserves_common_unsupported_statuses_original_order_and_input_arrays():
    e = _episode([[1., 0., 0.]], [-40.], [[2., 0., 0.]])
    e['statuses'].insert(0, {'query_id': 'outside', 'status': 'unsupported', 'reason': 'unobserved_ap'})
    e['query_indices'] = [1]
    before = deepcopy(e)
    rows = predict_baseline(e, _config('gp'))
    assert rows[0] == dict(e['statuses'][0], mean_dbm=None, std_db=None)
    assert rows[1]['query_id'] == 'q0'
    assert 'mean_dbm' not in e['statuses'][1]
    for name in ('view', 'query'):
        for key in e[name]:
            np.testing.assert_array_equal(e[name][key], before[name][key])


def test_empty_supported_query_axis_does_not_create_predictions():
    e = _episode([], [], [])
    e['statuses'] = [{'query_id': 'none', 'status': 'unsupported', 'reason': 'unobserved_ap'}]
    for c in configuration_grid():
        assert predict_baseline(e, c) == [dict(e['statuses'][0], mean_dbm=None, std_db=None)]


@pytest.mark.parametrize('with_support', [True, False])
def test_real_adapter_schema_integration_with_only_analytic_coordinates(with_support):
    from wirelessjev.measured_radio.model import prepare_episode

    def location(identifier, xyz, partition=None):
        result = {'id': identifier, 'xyz_model': xyz, 'xyz_provider': list(xyz)}
        if partition is not None:
            result['partition'] = partition
        return result

    xyz = {'T1': [1., 0., .5], 'T2': [3., 0., .5], 'Q1': [2., 1., .5]}
    metadata = {'scenarios': {'S01': {
        'aps': [location('AP1', [0., 0., .5]), location('AP2', [5., 0., .5])],
        'locations': [location(p, point, 'test' if p == 'Q1' else 'train') for p, point in xyz.items()],
        'frame': {'origin_provider_xyz': [0., 0., 0.], 'dimensions_m': [10., 10., 3.]}}}}

    def record(point, ap, value=None):
        result = {'id': point + '-' + ap, 'scenario': 'S01', 'phone': 'analytic-phone',
                  'session': '01', 'location_id': point, 'ap_id': ap,
                  'partition': 'test' if point == 'Q1' else 'train',
                  'xyz_model': xyz[point], 'carrier_hz': 5.18e9}
        if value is not None:
            result['rssi_dbm'] = value
        return result

    support = [record('T1', 'AP1', -40.), record('T2', 'AP1', -55.)] if with_support else []
    prepared = prepare_episode(support, [record('Q1', 'AP1'), record('Q1', 'AP2')], metadata)
    for config in configuration_grid():
        rows = predict_baseline(prepared, config)
        assert len(rows) == 2
        assert rows[1] == dict(prepared['statuses'][1], mean_dbm=None, std_db=None)
        assert rows[0]['status'] == ('prediction' if with_support else 'unsupported')
        assert rows[0]['mean_dbm'] is not None if with_support else rows[0]['mean_dbm'] is None


@pytest.mark.parametrize('edit', [lambda e: e['support_ids'].__setitem__(1, 's0'),
                                 lambda e: e['view']['known'].__setitem__(0, False),
                                 lambda e: e['view']['upper'].__setitem__(0, 1.),
                                 lambda e: e['view']['rx_xyz'].__setitem__((0, 0), np.nan),
                                 lambda e: e['query']['ap'].__setitem__(0, 1),
                                 lambda e: e.__setitem__('query_indices', []),
                                 lambda e: e['statuses'][0].__setitem__('status', 'unsupported')])
def test_rejects_mutated_common_membership_or_measurement_schema(edit):
    e = _episode([[1., 0., 0.], [3., 0., 0.]], [-30., -50.], [[2., 0., 0.]])
    edit(e)
    with pytest.raises(ValueError):
        predict_baseline(e, _config('reference'))


@pytest.mark.parametrize('config', [{'id': 'invented', 'method': 'reference'},
                                  {'id': 'reference', 'method': 'reference', 'extra': 1},
                                  dict(_config('gp'), noise_std_db=0.),
                                  dict(_config('idw'), neighbors=True)])
def test_rejects_unregistered_or_boolean_configuration(config):
    e = _episode([[1., 0., 0.]], [-30.], [[2., 0., 0.]])
    with pytest.raises(ValueError):
        predict_baseline(e, config)
