"""Adversarial acceptance fixtures, without any registered RF/model experiment."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from wirelessjev.adaptive_radio import study


def report(mean, heard, censored, *, bound=-80.0):
    """Explicit report fixture: unknown/censored values are never point targets."""
    heard = np.asarray(heard, dtype=bool)
    censored = np.asarray(censored, dtype=bool)
    center = np.where(heard, np.asarray(mean, dtype=float), 0.0)
    return {
        'center_dbm': center,
        'lower_dbm': np.where(heard, center - 0.5, 0.0),
        'upper_dbm': np.where(heard, center + 0.5, 0.0),
        'heard': heard,
        'censored': censored,
        'censor_upper_dbm': np.where(censored, bound, 0.0),
        'detection_threshold_dbm': np.where(heard | censored, bound, 0.0),
    }


def example(condition, target, *, layout='software-fixture', poses=None):
    n = len(target['heard'])
    query = SimpleNamespace(
        pose_index=np.arange(n) if poses is None else np.asarray(poses),
        ap_index=np.zeros(n, dtype=int),
        band_index=np.zeros(n, dtype=int),
        direction=np.zeros(n, dtype=int),
        device=np.zeros(n, dtype=int),
    )
    return SimpleNamespace(
        layout_id=layout,
        condition=condition,
        case_id=f'{layout}-rf{condition}',
        query_rows=query,
        target_arrays=lambda: deepcopy(target),
    )


def states(*, layout='software-fixture'):
    target = [
        report([-60, -70, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0]),
        report([-57, 0, -65, 0], [1, 0, 1, 0], [0, 1, 0, 0]),
        report([-55, -67, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0]),
    ]
    prediction = [
        (np.array([-60, -70, -80, -90]), np.full(4, 4.0)),
        (np.array([-55, -30, -40, -90]), np.full(4, 4.0)),
        (np.array([-53, -68, -80, -90]), np.full(4, 4.0)),
    ]
    return prediction, [example(k, t, layout=layout) for k, t in enumerate(target)]


def test_delta_uses_only_identical_both_heard_links_and_retains_other_events():
    prediction, examples = states()
    result, = study.condition_metrics(prediction, examples)
    # Delta residuals are +2 for pair0->1 and +2,-1 for pair0->2.
    # The other physical links remain detector/NLL observations, not fake points.
    assert result['delta_rmse_db'] == pytest.approx(np.sqrt(3.0))
    assert result['delta_both_heard_count'] == 3
    assert result['heard_count'] == 6
    assert result['censored_count'] == 3
    assert result['missing_count'] == 3
    assert result['pair_detection'] == [
        {'pair': [0, 1], 'both_heard_count': 1,
         'anchor_heard_only_count': 1, 'changed_heard_only_count': 1,
         'both_censored_count': 0, 'missing_pair_count': 1},
        {'pair': [0, 2], 'both_heard_count': 2,
         'anchor_heard_only_count': 0, 'changed_heard_only_count': 0,
         'both_censored_count': 1, 'missing_pair_count': 1},
    ]


@pytest.mark.parametrize('conditions', ((0, 1), (0, 1, 1), (0, 1, 3)))
def test_condition_groups_cannot_drop_repeat_or_invent_a_state(conditions):
    prediction, examples = states()
    examples = [deepcopy(examples[min(k, 2)]) for k in conditions]
    for item, condition in zip(examples, conditions):
        item.condition = condition
    with pytest.raises(ValueError, match='condition'):
        study.condition_metrics(prediction[:len(examples)], examples)


def test_corresponding_indices_cannot_hide_a_changed_physical_query():
    prediction, examples = states()
    examples[2].query_rows.pose_index = np.array([1, 0, 2, 3])
    with pytest.raises(ValueError, match='physical links'):
        study.condition_metrics(prediction, examples)


def test_unavailable_pair_is_retained_and_never_substituted_by_the_other_pair():
    prediction, examples = states()
    unavailable = report([0, 0, -65, 0], [0, 0, 1, 0], [1, 1, 0, 0])
    examples[1] = example(1, unavailable)
    result, = study.condition_metrics(prediction, examples)
    assert result['delta_available'] is False
    assert result['delta_rmse_db'] is None
    assert result['delta_both_heard_count'] == 2
    assert result['pair_detection'][0]['both_heard_count'] == 0
    assert result['pair_detection'][1]['both_heard_count'] == 2
    assert result['delta_unavailable_reason']
    with pytest.raises(RuntimeError, match='no usable'):
        study.validation_score(prediction, examples)


def test_detector_brier_uses_actual_detection_bound_and_omits_unknown_reports():
    known = report([-70, 0, 0], [1, 0, 0], [0, 1, 0])
    actual = study.observation_metrics([-70, -90, 1e9], [2, 2, 2], known)
    # Symmetric predictions lie five standard deviations inside each event.
    # An unheard value invented as -90, or a heard threshold erased to zero,
    # would produce a different endpoint.
    from scipy.special import ndtr
    assert actual['detector_brier'] == pytest.approx(float(ndtr(-5.0) ** 2), abs=1e-20)
    assert (actual['heard_count'], actual['censored_count'], actual['missing_count']) == (1, 1, 1)
    changed_unknown = study.observation_metrics([-70, -90, -1e9], [2, 2, 2], known)
    assert actual == changed_unknown


def test_all_censored_is_known_detector_evidence_without_a_point_rmse():
    target = report([0, 0], [0, 0], [1, 1])
    actual = study.observation_metrics([-90, -91], [2, 2], target)
    assert actual['rmse_db'] is None and actual['p95_abs_db'] is None
    assert actual['observation_nll'] >= 0
    assert actual['detector_brier'] < 1e-10
    assert actual['heard_count'] == 0 and actual['censored_count'] == 2


def test_metric_nll_intersects_a_heard_quantization_bin_with_detection_event():
    from scipy.special import ndtr
    target = report([-80.], [1], [0], bound=-80.)
    actual = study.observation_metrics([-80.], [1.], target)
    assert actual['observation_nll'] == pytest.approx(-np.log(ndtr(.5) - .5), abs=1e-12)
    assert actual['detector_brier'] == .25


@pytest.mark.parametrize('mutation', ('overlap', 'nonfinite', 'zero_std', 'bad_interval'))
def test_corrupt_forecasts_or_report_events_cannot_enter_acceptance(mutation):
    target = report([-70, 0], [1, 0], [0, 1])
    mean, std = np.array([-70.0, -90.0]), np.array([2.0, 2.0])
    if mutation == 'overlap':
        target['censored'][0] = True
    elif mutation == 'nonfinite':
        mean[0] = np.nan
    elif mutation == 'zero_std':
        std[0] = 0
    else:
        target['upper_dbm'][0] = target['lower_dbm'][0]
    with pytest.raises((ValueError, FloatingPointError)):
        study.observation_metrics(mean, std, target)


def test_validation_weights_whole_geometries_not_report_or_condition_counts():
    prediction, examples = [], []
    for layout, n, error in (('small-fixture', 1, 9.0), ('large-fixture', 20, 1.0)):
        for condition in (0, 1, 2):
            value = np.full(n, -60.0 + condition)
            target = report(value, np.ones(n, bool), np.zeros(n, bool))
            examples.append(example(condition, target, layout=layout))
            prediction.append((value + (error if condition else 0), np.full(n, 4.0)))
    assert study.validation_score(prediction, examples) == pytest.approx(5.0)


def test_bootstrap_uses_paired_layouts_and_cannot_accept_nonfinite_or_unpaired_rows():
    actual = study.paired_bootstrap([1.0, 7.0, 11.0], [2.0, 8.0, 12.0], samples=100)
    assert actual['layouts'] == 3
    assert actual['adjusted_interval'] == [-1.0, -1.0]
    assert actual['better_adjusted'] is True
    for candidate, reference in (([1.0], [2.0]), ([1.0, np.nan], [2.0, 3.0]),
                                 ([1.0, 2.0], [3.0, 4.0, 5.0])):
        with pytest.raises(ValueError):
            study.paired_bootstrap(candidate, reference, samples=100)
