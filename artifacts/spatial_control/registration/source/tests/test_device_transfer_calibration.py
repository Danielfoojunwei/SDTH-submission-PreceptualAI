"""Analytic known fixtures; no downloaded benchmark samples or scores."""
import numpy as np
import pytest

from wirelessjev.device_transfer import data as d, calibration as m, benchmark as b


def training_fixture():
    power = np.array([[-60,-70,100,-80],[-80,-50,-70,100],[-50,-60,100,-70],[-70,-40,-60,100.]])
    features = d._fingerprints(power, [1,1,2,2], links=4)
    return d.TrainingData(features, d.seal([[0,0,0],[1,0,0],[0,0,0],[1,0,0]], 'f8'),
                          d.seal([0]*4, 'i8'), d.seal([0]*4, 'i8'))


def test_genuine_colocation_offset_recovers_known_shift_and_preserves_unknowns():
    value = training_fixture()
    offset = m.fit_offsets(value, 2, [2], shrinkage=4.)
    assert offset.global_offset_db == 10.
    np.testing.assert_array_equal(offset.ap_offset_db, np.full(4, 10.))
    np.testing.assert_array_equal(offset.per_ap_pair_counts, [1,1,0,1])
    fixed = m.corrected(value.fingerprints.subset([2]), offset.ap_offset_db)
    np.testing.assert_array_equal(fixed.rssi_dbm[0], value.fingerprints.rssi_dbm[0])
    np.testing.assert_array_equal(fixed.heard, value.fingerprints.heard[[2]])
    zero = m.fit_offsets(value, 2, np.array([], 'i8'))
    assert zero.global_offset_db == 0 and zero.paired_ap_observations == 0
    np.testing.assert_array_equal(zero.ap_offset_db, np.zeros(4))


@pytest.mark.parametrize('rows', [[0], [2,2], [2.5], [-1], [100]])
def test_invalid_or_source_phone_support_rejected(rows):
    with pytest.raises(ValueError):
        m.fit_offsets(training_fixture(), 2, rows)


def test_knn_common_power_reference_and_explicit_no_coverage_fallback():
    value = training_fixture()
    source = value.fingerprints.subset([0,1])
    predictor = m.MaskedKNN(source, value.xyz_m[:2], neighbors=1, minimum_shared=3)
    exact = predictor.predict(source)
    np.testing.assert_allclose(exact['xyz_m'], value.xyz_m[:2], atol=0)
    unknown = d._fingerprints(np.full((1, 4), 100.), [2], links=4)
    fallback = predictor.predict(unknown)
    np.testing.assert_array_equal(fallback['xyz_m'], [[.5,0,0]])
    assert fallback['has_shared_ap_support'].tolist() == [False]
    assert fallback['neighbor_source_rows'].tolist() == [[-1]]


def test_hidden_unknown_power_is_inert_and_ridge_never_refits_on_query():
    value = training_fixture()
    source = value.fingerprints.subset([0,1])
    predictor = m.MaskedKNN(source, value.xyz_m[:2], neighbors=1)
    poisoned = d.Fingerprints(d.seal(np.where(source.heard, source.rssi_dbm, 1.e20)), source.heard,
                             source.device_ids, source.row0)
    np.testing.assert_array_equal(predictor.predict(poisoned)['xyz_m'], predictor.predict(source)['xyz_m'])
    ridge = m.RidgeLocalizer(source, value.xyz_m[:2])
    before = ridge.mean.copy(), ridge.scale.copy(), ridge.coefficient.copy()
    ridge.predict(poisoned)
    for expected, actual in zip(before, (ridge.mean, ridge.scale, ridge.coefficient)):
        np.testing.assert_array_equal(expected, actual)


def test_closed_form_normal_equation_audit_rejects_corrupt_coefficient():
    value = training_fixture()
    p = {'ridge_alpha':10.,'support_seeds':[0], 'support_sizes':[0,1], 'ap_shrinkage_pseudocount':4.}
    inventory = {'support_orders':{'2':{'0':[2]}}}
    state = b._fold_fit(value, 2, p, inventory)
    assert b.audit_fit(value, 2, state, p, inventory)['passes'] is True
    state['ridge_coefficient'] = state['ridge_coefficient'] + 1.
    with pytest.raises(RuntimeError, match='normal-equation'):
        b.audit_fit(value, 2, state, p, inventory)


def test_metrics_cluster_repeated_positions_and_keep_tail_failures():
    truth = np.array([[0,0,0],[0,0,0],[2,0,0.]])
    predicted = np.array([[1,0,0],[3,0,0],[2,0,0.]])
    result = b.metric(predicted, truth, [True,False,True])
    assert result['mean_2d_error_m'] == pytest.approx(4/3)
    assert result['location_balanced_mean_2d_error_m'] == 1.
    assert result['shared_ap_fallback_fraction'] == pytest.approx(1/3)
    assert result['p95_2d_error_m'] > 2.7
