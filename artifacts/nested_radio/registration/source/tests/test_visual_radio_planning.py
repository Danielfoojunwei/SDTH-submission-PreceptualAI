"""Physical floors, causal leases and canonical fallback are independent of a net."""
from dataclasses import replace
import pytest
from wirelessjev.visual_radio.planning import ServiceIntent,ServiceEstimate,RadioOption,OptionKind,choose_option


def estimate(misses,gap=0):
    return ServiceEstimate(misses,gap,'explicit unit fixture',upper_misses=misses,
                           upper_max_gap_slots=gap,validation='numerical')


def option(key='a',kind=OptionKind.MOVE_AP,misses=0,**kwargs):
    return RadioOption(key,kind,500,600,100,estimate(misses),target_ap=1,observation_slot=490,**kwargs)


def baseline(misses=10):return option('base',OptionKind.CANONICAL,misses)


def test_unknown_rejected_advice_preserves_unforced_baseline_not_a_stay():
    unknown=replace(option(),estimate=None)
    decision=choose_option(ServiceIntent(),baseline(),[unknown],now_slot=500)
    assert decision.selected is None and decision.proposal_at(501) is None
    assert decision.fallback=='canonical_continuation'
    assert decision.assessments[0].reason=='unknown_service'


def test_known_four_second_service_outage_overrides_optimistic_neural_mean():
    move=option(exclusive_unavailable=((500,900),))
    d=choose_option(ServiceIntent(),baseline(),[move],now_slot=500,release_phase=0)
    assert d.selected is None and d.assessments[0].known_forced_misses==80
    assert d.assessments[0].objective>=84
    unknown_phase=choose_option(ServiceIntent(),baseline(),[move],now_slot=500)
    assert unknown_phase.assessments[0].known_forced_misses==79


def test_scan_floor_count_does_not_double_count_overlapping_unavailability():
    scan=option(kind=OptionKind.SCAN,exclusive_unavailable=((500,509),(501,508)))
    d=choose_option(ServiceIntent(),baseline(),[scan],now_slot=500)
    assert d.assessments[0].known_forced_misses==1


def test_hierarchy_selects_finite_supported_option_and_fast_lease_expires():
    a=option('a',misses=3);b=option('b',misses=2)
    d=choose_option(ServiceIntent(),baseline(),[a,b],now_slot=500)
    assert d.selected==b and d.proposal_at(590).option_id==b.option_id and d.proposal_at(590).hold_slots==1
    assert d.proposal_at(591) is None
    assert d.proposal_at(600) is None
    assert d.proposal_at(499) is None


@pytest.mark.parametrize('change,reason',[
    ({'completion_slot':600},'expired_before_result'),
    ({'supported_kinds':()},'unsupported_hardware_action'),
    ({'max_observation_age_slots':5},'stale_observations')])
def test_compute_hardware_and_freshness_constraints(change,reason):
    d=choose_option(ServiceIntent(),baseline(),[option()],now_slot=500,**change)
    assert d.selected is None and d.assessments[0].reason==reason


def test_missing_upper_estimates_are_unknown_rather_than_zero_risk():
    mean_only=replace(option(),estimate=ServiceEstimate(0,0,'uncalibrated mean',validation='numerical'))
    d=choose_option(ServiceIntent(),baseline(),[mean_only],now_slot=500)
    assert d.selected is None and d.assessments[0].reason=='unknown_upper_service'


def test_delayed_inference_cannot_publish_early_or_extend_the_remaining_lease():
    d=choose_option(ServiceIntent(),baseline(),[replace(option(),observation_slot=500)],now_slot=500,completion_slot=550)
    assert d.selected is not None and d.proposal_at(549) is None
    assert d.proposal_at(550).hold_slots==50
    assert d.proposal_at(599).hold_slots==1 and d.proposal_at(600) is None


def test_result_that_finishes_after_observation_freshness_returns_canonical():
    d=choose_option(ServiceIntent(),baseline(),[option()],now_slot=500,completion_slot=599)
    assert d.selected is None and d.assessments[0].reason=='stale_observations'


def test_ties_duplicates_and_invalid_semantics():
    d=choose_option(ServiceIntent(),baseline(0),[option()],now_slot=500)
    assert d.selected is None
    with pytest.raises(ValueError,match='Duplicate'):choose_option(ServiceIntent(),baseline(),[option(),option()],now_slot=500)
    with pytest.raises(ValueError):choose_option(ServiceIntent(),option(),[],now_slot=500)
    with pytest.raises(ValueError):replace(option(),valid_until_slot=601)
    with pytest.raises(ValueError):replace(option(),observation_slot=501)
