"""Analytic software contracts only; no measured-data fitting or scoring.

The three authentic frozen NPZ members are exercised on constructed geometry
and arbitrary RSSI numbers to verify invariances, never hardware accuracy.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from wirelessjev.measured_radio.model import CARRIER_HZ, prepare_episode, predict
from wirelessjev.smartcar_numpy.export import load_export, state_sha256
from wirelessjev.smartcar_numpy.model import NumpyField, STATE_SHAPES

ROOT = Path(__file__).resolve().parents[2]


def analytic_episode():
    origin = np.array((-25., 41., -3.))
    aps = [('AP1', (0., 0., .5)), ('AP2', (7., 1., 1.2)), ('AP3', (-2., 4., .8))]
    locations = [('T1', (-1., 1., .1), 'train'), ('T2', (2., 3., .2), 'train'),
                 ('T3', (5., -1., .2), 'train'), ('V1', (3., 2., .15), 'validation'),
                 ('Q1', (4., 4., .2), 'test'), ('Q2', (1., -2., .1), 'test')]

    def public(identifier, xyz):
        return {'id': identifier, 'xyz_model': list(xyz),
                'xyz_provider': (np.asarray(xyz) + origin).tolist()}

    metadata = {'scenarios': {'S01': {
        'aps': [public(identifier, xyz) for identifier, xyz in aps],
        'locations': [{**public(identifier, xyz), 'partition': partition}
                      for identifier, xyz, partition in locations],
        'frame': {'origin_provider_xyz': origin.tolist(), 'dimensions_m': [12., 10., 3.],
                  'semantics': 'computational catalogue frame; not a measured floor'}}}}
    points = {identifier: (xyz, partition) for identifier, xyz, partition in locations}

    def row(location, ap, *, session='01', value=None):
        xyz, partition = points[location]
        result = {'id': f'S01-PHONE-{session}-{location}-{ap}', 'scenario': 'S01',
                  'phone': 'PHONE', 'session': session, 'location_id': location,
                  'partition': partition, 'ap_id': ap, 'xyz_model': list(xyz),
                  'carrier_hz': CARRIER_HZ}
        if value is not None:
            result['rssi_dbm'] = value
        return result

    support = [row('T1', 'AP1', value=-54.), row('T2', 'AP1', value=-62.),
               row('T3', 'AP1', value=-68.), row('T1', 'AP2', value=-72.),
               row('T2', 'AP2', value=-65.), row('T3', 'AP2', value=-58.)]
    queries = [row('Q1', 'AP1'), row('Q2', 'AP2'), row('Q1', 'AP3'),
               row('Q1', 'AP1', session='02')]
    return support, queries, metadata


@pytest.fixture(scope='module', params=(0, 1, 2), ids=lambda seed: f'actual-frozen-seed{seed}')
def frozen(request):
    directory = ROOT / 'artifacts/smartcar_numpy/models'
    manifest = json.loads((directory / 'manifest.json').read_text())
    member = manifest['members'][request.param]
    arrays, _ = load_export(directory / member['path'], member['sha256'], member['source_checkpoint_sha256'])
    return NumpyField(arrays)


def test_canonical_reference_matches_independent_gauge_algebra():
    support, queries, metadata = analytic_episode()
    episode = prepare_episode(support, queries, metadata)
    view, query = episode['view'], episode['query']
    assert episode['support_ids'] == [row['id'] for row in support]
    assert episode['query_indices'] == [0, 1, 3]
    assert episode['statuses'][2]['status'] == 'unsupported'
    assert set(episode['reference_by_ap']) == {'AP1', 'AP2'}
    for index, ap in enumerate(episode['ap_ids'][:2]):
        selected = view['ap'] == index
        y = np.array([row['rssi_dbm'] for row in support])[selected]
        loss = -view['fspl'][selected]
        c = np.mean(y + loss)
        assert episode['reference_by_ap'][ap] == pytest.approx(c, abs=1.e-13)
        assert np.mean(view['lower'][selected]) == pytest.approx(0., abs=1.e-13)
        qselected = query['ap'] == index
        for pref in (-73., 0., 20., 150.):
            b = np.mean(y - (pref - loss))
            direct = pref + query['fspl'][qselected] + b
            np.testing.assert_allclose(direct,
                query['fspl'][qselected] + query['offset'][qselected], atol=5.e-14, rtol=0.)
    assert view['fspl'].dtype == query['fspl'].dtype == np.float64
    assert view['lower'].dtype == query['offset'].dtype == np.float64
    assert view['context'].dtype == episode['ap_xyz'].dtype == np.float32
    np.testing.assert_array_equal(view['lower'], view['upper'])
    np.testing.assert_array_equal(view['lower'], view['floor'])
    np.testing.assert_array_equal(view['context'][:, 0], (view['lower'] / 30.).astype('f4'))


def test_unknown_features_are_explicit_and_physical_basis_has_rank_at_most_two():
    episode = prepare_episode(*analytic_episode())
    known = episode['feature_known_mask']
    np.testing.assert_array_equal(np.flatnonzero(~known), [9, 10, 13, 14, 15])
    np.testing.assert_array_equal(episode['view']['context'][:, [4+9, 4+10]], 0.)
    np.testing.assert_array_equal(episode['query']['q'][:, 9:11], 0.)
    basis = episode['query']['basis'].astype('f8')
    assert np.linalg.matrix_rank(basis, tol=1.e-6) <= 2
    assert episode['assumptions']['unknown_cad_active_features_zero_imputed'] is True
    assert episode['assumptions']['reference_identifies_tx_or_device_calibration'] is False
    assert episode['assumptions']['spread_includes_nuisance_reference_uncertainty'] is False
    assert episode['assumptions']['full_prediction_translation_invariant'] is False


def test_actual_frozen_predictions_shift_by_independent_ap_gauge_constants(frozen):
    support, queries, metadata = analytic_episode()
    initial = prepare_episode(support, queries, metadata)
    shifted_support = deepcopy(support)
    shifts = {'AP1': 13., 'AP2': -7.}
    for row in shifted_support:
        row['rssi_dbm'] += shifts[row['ap_id']]
    shifted = prepare_episode(shifted_support, queries, metadata)
    np.testing.assert_allclose(initial['view']['lower'], shifted['view']['lower'], atol=3.e-14, rtol=0.)
    before = state_sha256(frozen.arrays)
    predicted, changed = predict(frozen, initial), predict(frozen, shifted)
    for row, other, query in zip(predicted, changed, queries):
        assert row['query_id'] == query['id']
        if row['status'] == 'prediction':
            assert other['mean_dbm'] - row['mean_dbm'] == pytest.approx(shifts[query['ap_id']], abs=1.e-11)
            assert other['std_db'] == pytest.approx(row['std_db'], abs=1.e-12)
        else:
            assert row['mean_dbm'] is other['mean_dbm'] is None
    assert state_sha256(frozen.arrays) == before


def test_actual_frozen_core_is_invariant_to_inactive_columns(frozen):
    prepared = prepare_episode(*analytic_episode())
    changed = deepcopy(prepared)
    inactive = np.array([8, 11, 12, 13, 14, 15])
    changed['view']['context'][:, inactive + 4] = np.arange(6, dtype='f4') + 21.
    changed['query']['q'][:, inactive] = np.arange(6, dtype='f4') - 31.
    initial, modified = predict(frozen, prepared), predict(frozen, changed)
    assert initial == modified


def test_actual_frozen_predictions_are_support_order_invariant(frozen):
    support, queries, metadata = analytic_episode()
    expected = predict(frozen, prepare_episode(support, queries, metadata))
    actual = predict(frozen, prepare_episode(support[::-1], queries, metadata))
    for first, second in zip(expected, actual):
        assert first['status'] == second['status']
        if first['status'] == 'prediction':
            assert first['mean_dbm'] == pytest.approx(second['mean_dbm'], abs=1.e-11)
            assert first['std_db'] == pytest.approx(second['std_db'], abs=1.e-12)


def test_empty_support_returns_explicit_unsupported_without_fabricating_reference(frozen):
    _, queries, metadata = analytic_episode()
    prepared = prepare_episode([], queries, metadata)
    assert prepared['view'] is prepared['query'] is None
    assert prepared['reference_by_ap'] == {}
    assert prepared['query_indices'] == []
    assert all(row['status'] == 'unsupported' and row['mean_dbm'] is None
               for row in predict(frozen, prepared))


@pytest.mark.parametrize('key', ['rssi_dbm', 'RSSI', 'target', 'label', 'received_power_dbm'])
def test_public_queries_reject_private_target_fields(key):
    support, queries, metadata = analytic_episode()
    queries[0][key] = -61.
    with pytest.raises(ValueError, match='Query RSSI/target'):
        prepare_episode(support, queries, metadata)


def test_nested_private_query_label_is_also_rejected():
    support, queries, metadata = analytic_episode()
    queries[0]['extras'] = {'private': {'rssi_dbm': -61.}}
    with pytest.raises(ValueError, match='Query RSSI/target'):
        prepare_episode(support, queries, metadata)


@pytest.mark.parametrize('case', ['scenario', 'phone', 'support_session02', 'query_session03',
    'partition', 'coordinate', 'carrier', 'missing_carrier', 'unknown_ap', 'sentinel',
    'nonfinite_rssi', 'duplicate_id', 'duplicate_statistic', 'budget', 'provider_translation',
    'dimension', 'unknown_location', 'query_train', 'duplicate_query'])
def test_invalid_admission_fails_before_model_execution(case):
    support, queries, metadata = analytic_episode()
    if case in {'scenario', 'phone'}:
        support[0][case] = 'other'
    elif case == 'support_session02':
        support[0]['session'] = '02'
    elif case == 'query_session03':
        queries[0]['session'] = '03'
    elif case == 'partition':
        support[0]['partition'] = 'test'
    elif case == 'coordinate':
        queries[0]['xyz_model'][2] += .1
    elif case == 'carrier':
        support[0]['carrier_hz'] = 2.4e9
    elif case == 'missing_carrier':
        queries[0].pop('carrier_hz')
    elif case == 'unknown_ap':
        queries[0]['ap_id'] = 'unverified'
    elif case == 'sentinel':
        support[0]['rssi_dbm'] = -200.
    elif case == 'nonfinite_rssi':
        support[0]['rssi_dbm'] = np.nan
    elif case == 'duplicate_id':
        support[1]['id'] = support[0]['id']
    elif case == 'duplicate_statistic':
        extra = deepcopy(support[0]); extra['id'] += '-duplicate'
        support.append(extra)
    elif case == 'budget':
        support = support * 33
    elif case == 'provider_translation':
        metadata['scenarios']['S01']['aps'][0]['xyz_provider'][0] += 1.
    elif case == 'dimension':
        metadata['scenarios']['S01']['frame']['dimensions_m'][0] = 0.
    elif case == 'unknown_location':
        queries[0]['location_id'] = 'unverified'
    elif case == 'query_train':
        queries[0].update(location_id='T1', xyz_model=[-1., 1., .1], partition='train')
    elif case == 'duplicate_query':
        extra = deepcopy(queries[0]); extra['id'] += '-duplicate'
        queries.append(extra)
    with pytest.raises(ValueError):
        prepare_episode(support, queries, metadata)


@pytest.mark.parametrize('field', ['context', 'lower', 'floor', 'offset', 'weights', 'support_rssi_dbm'])
def test_redundant_fields_are_rechecked_before_frozen_execution(frozen, field):
    prepared = prepare_episode(*analytic_episode())
    if field == 'support_rssi_dbm':
        prepared[field][0] += .1
    elif field == 'context':
        prepared['view'][field][0, 0] += .1
    else:
        prepared['view'][field][0] += .1
    with pytest.raises(ValueError, match='Redundant'):
        predict(frozen, prepared)


def test_frame_translation_retains_provider_coordinates_without_claiming_floor():
    support, queries, metadata = analytic_episode()
    before = deepcopy(metadata)
    episode = prepare_episode(support, queries, metadata)
    assert metadata == before
    provider = np.array(before['scenarios']['S01']['aps'][0]['xyz_provider'])
    assert provider[2] < 0.
    np.testing.assert_allclose(provider - np.array(episode['assumptions']['frame']['origin_provider_xyz']),
                               episode['ap_xyz'][0], atol=1.e-7)
    assert episode['assumptions']['frame_is_computational_not_a_measured_floor'] is True


def test_validation_queries_and_empty_query_batch_are_explicit():
    support, queries, metadata = analytic_episode()
    query = deepcopy(queries[0])
    query.update(id='validation', location_id='V1', partition='validation', xyz_model=[3., 2., .15])
    assert prepare_episode(support, [query], metadata)['statuses'][0]['status'] == 'prediction'
    episode = prepare_episode(support, [], metadata)
    assert episode['view'] is not None and episode['query'] is None
    assert episode['statuses'] == []


def test_structurally_valid_but_unreleased_arrays_cannot_claim_frozen_inference():
    arrays = {name: np.zeros(shape, dtype='f4') for name, shape in STATE_SHAPES.items()}
    arrays['local_gates'][:5] = .1
    with pytest.raises(ValueError, match='exact released'):
        predict(NumpyField(arrays), prepare_episode(*analytic_episode()))


def test_receiver_android_group_cannot_change_between_support_and_queries():
    support, queries, metadata = analytic_episode()
    for row in support + queries:
        row['android_version'] = 16
    prepared = prepare_episode(support, queries, metadata)
    assert prepared['assumptions']['android_version'] == 16
    queries[-1]['android_version'] = 15
    with pytest.raises(ValueError, match='Android calibration group'):
        prepare_episode(support, queries, metadata)
    queries[-1].pop('android_version')
    with pytest.raises(ValueError, match='Android calibration group'):
        prepare_episode(support, queries, metadata)
