"""Diagnostic invariants distinguish geometry, missingness and prediction."""
import numpy as np
import pytest

from wirelessjev.representation_audit import (
    action_sensitivity,
    crossfit_ridge,
    observed_feature_stats,
    prediction_metrics,
    representation_stats,
    site_folds,
)


def test_literal_constant_and_zero_vectors_have_zero_rank_not_one():
    for x in (np.zeros((24, 6)), np.full((24, 6), .1)):
        stats = representation_stats(x)
        assert stats['numerically_constant_on_sample']
        assert stats['entropy_effective_rank'] == 0.
        assert stats['participation_rank'] == 0.
        assert stats['numerical_rank'] == 0
        assert stats['covariance_trace'] == 0.
        assert stats['centered_cosine']['pairs'] == 0
    assert representation_stats(np.ones((24, 6)))['raw_cosine']['mean'] == pytest.approx(1.)


def test_large_common_direction_high_cosine_does_not_imply_collapse():
    rng = np.random.default_rng(299180)
    signal = rng.normal(size=(300, 6))
    stats = representation_stats(signal + 1000.)
    assert stats['raw_cosine']['mean'] > .999
    assert not stats['numerically_constant_on_sample']
    assert stats['entropy_effective_rank'] > 5.
    assert stats['centered_energy_fraction'] < 1e-5
    assert abs(stats['centered_cosine']['mean']) < .05


def test_spectrum_is_rotation_translation_and_scale_invariant_but_axis_std_is_not():
    rng = np.random.default_rng(299181)
    x = rng.normal(size=(90, 5)) * np.array([4., 2., 1., .3, .01])
    q = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    original = representation_stats(x)
    rotated = representation_stats(x @ q + 17.)
    scaled = representation_stats(x * .001)
    for key in ('entropy_effective_rank', 'participation_rank', 'top_pc_variance_fraction'):
        assert rotated[key] == pytest.approx(original[key], rel=1e-10)
        assert scaled[key] == pytest.approx(original[key], rel=1e-10)
    np.testing.assert_allclose(rotated['covariance_eigenvalues'], original['covariance_eigenvalues'])
    assert not np.allclose(rotated['axis_std'], original['axis_std'])
    assert scaled['covariance_trace'] == pytest.approx(original['covariance_trace'] * 1e-6)


def test_collinear_varying_features_are_not_literal_constant_collapse():
    x = np.arange(30.)[:, None] @ np.ones((1, 8))
    stats = representation_stats(x)
    assert not stats['numerically_constant_on_sample']
    assert stats['entropy_effective_rank'] == pytest.approx(1.)
    assert stats['participation_rank'] == pytest.approx(1.)
    assert stats['top_pc_variance_fraction'] == pytest.approx(1.)
    assert min(stats['axis_std']) > .5  # coordinate variance penalties can pass


def test_float32_layernorm_constraint_is_not_counted_as_an_extra_dimension():
    rng = np.random.default_rng(299183)
    x = rng.normal(size=(300, 8)).astype('f4')
    x = (x-x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    stats = representation_stats(x)
    assert stats['numerical_rank'] == 7
    assert stats['input_numeric_precision'] == 'float32'
    assert not stats['numerically_constant_on_sample']


def test_masked_unknown_nan_values_never_become_constant_observations():
    x = np.array([[1., np.nan, 0.], [3., np.nan, 9.]])
    mask = np.array([[True, False, True], [True, False, False]])
    stats = observed_feature_stats(x, mask, ['heard', 'unknown_age', 'partial'])
    assert stats['heard']['std'] == 1.
    assert stats['unknown_age']['observations'] == 0
    assert stats['unknown_age']['std'] is None
    assert stats['partial']['std'] == 0.
    assert stats['partial']['observed_fraction'] == .5
    mask[0, 1] = True
    with pytest.raises(ValueError, match='observed feature'):
        observed_feature_stats(x, mask, ['heard', 'unknown_age', 'partial'])


def test_site_probe_folds_do_not_split_scenes_and_linear_information_is_accessible():
    rng = np.random.default_rng(299182)
    scene = np.repeat(np.arange(16), 20)
    x = rng.normal(size=(len(scene), 3))
    y = x @ np.array([[3., 1.], [-2., .5], [.5, -2.]]) + [9., 2.]
    covered = np.zeros(len(scene), int)
    for train, test in site_folds(scene):
        assert not set(scene[train]) & set(scene[test])
        covered[test] += 1
    assert (covered == 1).all()
    prediction = crossfit_ridge(x, y, scene, alpha=.001)
    constant = crossfit_ridge(np.zeros((len(x), 3)), y, scene, alpha=.001)
    assert np.mean((prediction-y)**2) < 1e-8
    assert np.mean((constant-y)**2) > 1.
    result = prediction_metrics(prediction, y, np.tile([.03, 4.], len(y)//2))
    assert result['all']['missed_commands']['r2'] > .99999
    assert set(result['by_cost']) == {'0.03', '4.0'}


def test_action_response_uses_only_distinct_queries_of_same_recorded_history():
    query = np.array([[0.], [2.], [100.], [1000.]])
    latent = np.array([[1., 1.], [1., 3.], [9., 0.], [-9., 0.]])
    mean = np.array([[0., 0.], [6., .5], [99., 9.], [-99., 9.]])
    result = action_sensitivity(query, latent, mean, ['same', 'same', 'other1', 'other2'])
    assert result['pairs'] == 1
    assert result['query_rms_change'] == 2.
    assert result['latent_rms_change'] == pytest.approx(np.sqrt(2.))
    assert result['mean_absolute_miss_prediction_change'] == 6.
    assert result['mean_absolute_gap_prediction_change_seconds'] == .5
    assert action_sensitivity(query, latent, mean, ['a', 'b', 'c', 'd']) == {'pairs': 0}


@pytest.mark.parametrize('x', [np.array([1., 2.]), np.empty((0, 3)), np.array([[np.nan]])])
def test_invalid_representation_inputs_fail_closed(x):
    with pytest.raises(ValueError, match='finite nonempty'):
        representation_stats(x)
