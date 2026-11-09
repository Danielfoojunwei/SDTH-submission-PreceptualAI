"""Typed hierarchy arithmetic, never learned-system or hardware evidence."""
from dataclasses import replace
import numpy as np
import pytest

from wirelessjev.radiojepa.hierarchy.levels import (
    Provenance,SampledRF,LinkDistribution,AirtimeBudget,RouteRisk,MissionSLA,MissionAssessment)
from wirelessjev.radiojepa.hierarchy.coarsen import (
    coarsen_samples,correlated_effective_count,lee_average_power,
    coarsen_airtime,coarsen_route,assess_mission)
from wirelessjev.radiojepa.hierarchy.consistency import (
    check_sample_link_consistency,check_airtime_consistency,check_route_consistency)
from wirelessjev.radiojepa.dynamics import spatial_shadow_covariance


def samples():
    return SampledRF(('s0','s1','s2'),[0.,.1,.2],('ap0','ap1'),
        np.array([[True,False],[True,False],[False,False]]),Provenance(('capture',),'warehouse_xy'),
        log_power_dbm=np.array([[0.,100.],[-10.,np.nan],[999.,999.]]),
        positions_m=np.array([[0.,0.],[.1,0.],[.2,0.]]),
        censor_upper_dbm=np.array([[np.nan,-90.],[np.nan,-90.],[-90.,np.nan]]))


def modeled_links():
    return LinkDistribution(('ap0','ap1'),[.1,.2],[.01,.02],np.array([10,10]),Provenance(('model',),'world_xy'),
        outage_probability=[.1,.2],outage_upper=[.15,.3],probability_kind='model_marginal')


def test_linear_power_average_preserves_missingness_and_censoring():
    raw=samples();result=coarsen_samples(raw,threshold_mw=.2)
    assert result.mean_power_mw[0]==pytest.approx(.55)
    assert 10*np.log10(result.mean_power_mw[0])!=pytest.approx(-5.)
    assert result.sample_count.tolist()==[2,0]
    assert result.censored_count.tolist()==[1,2]
    assert np.isnan(result.mean_power_mw[1]) and np.isnan(result.empirical_variance_mw2[1])
    assert result.effective_sample_count is None and result.mean_se_plugin_mw is None
    assert result.outage_probability[0]==.5 and np.isnan(result.outage_probability[1])
    assert result.provenance.qualified is False
    assert check_sample_link_consistency(raw,result)['power_unit']=='mW'
    with pytest.raises(ValueError):check_sample_link_consistency(raw,replace(result,mean_power_mw=np.array([.6,np.nan])))


def test_csi_requires_absolute_power_calibration():
    raw=SampledRF(('s0','s1'),[0.,1.],('link',),np.ones((2,1),bool),Provenance(('csi_capture',)),
                  csi=np.array([[[1+1j,1-1j]],[[2+2j,2-2j]]]))
    with pytest.raises(ValueError,match='Uncalibrated'):coarsen_samples(raw)
    result=coarsen_samples(replace(raw,csi_power_scale_mw=np.array([.5])))
    assert result.mean_power_mw[0]==pytest.approx(2.5)  # mean(2,8)*.5 mW
    assert check_sample_link_consistency(replace(raw,csi_power_scale_mw=np.array([.5])),result)['passes']


def test_effective_count_is_conditional_on_supplied_correlation():
    assert correlated_effective_count(np.eye(4))==4.
    assert correlated_effective_count(np.ones((4,4)))==1.
    assert correlated_effective_count(np.array([[1.,-.5],[-.5,1.]]))==4.
    assert np.isnan(correlated_effective_count(np.array([[1.,-1.],[-1.,1.]])))
    assert np.isnan(correlated_effective_count(np.eye(2),np.zeros(2,bool)))
    result=coarsen_samples(samples(),correlation=np.ones((3,3)))
    assert result.effective_sample_count[0]==1.
    assert result.mean_se_plugin_mw[0]==pytest.approx(np.sqrt(result.empirical_variance_mw2[0]))
    assert np.isnan(result.effective_sample_count[1])


@pytest.mark.parametrize('matrix',[
    np.array([[1.,2.],[2.,1.]]),np.array([[1.,.5],[.3,1.]]),np.eye(2)*2,
    np.array([[1.,np.nan],[np.nan,1.]]),np.zeros((2,3))])
def test_invalid_correlation_never_creates_confidence(matrix):
    with pytest.raises(ValueError):correlated_effective_count(matrix)


def test_sample_permutation_and_rigid_coordinate_transforms_preserve_coarsening():
    raw=samples();r=spatial_shadow_covariance(raw.positions_m,.1,1.)
    result=lee_average_power(raw,max_spatial_diameter_m=.3,correlation=r)
    order=np.array([2,0,1]);rotation=np.array([[0.,-1.],[1.,0.]])
    moved=replace(raw,sample_ids=tuple(raw.sample_ids[k] for k in order),capture_time_s=raw.capture_time_s[order],
        observed=raw.observed[order],log_power_dbm=raw.log_power_dbm[order],
        censor_upper_dbm=raw.censor_upper_dbm[order],positions_m=raw.positions_m[order]@rotation+np.array([20.,-6.]))
    changed=lee_average_power(moved,max_spatial_diameter_m=.3,
                             correlation=spatial_shadow_covariance(moved.positions_m,.1,1.))
    np.testing.assert_allclose(result.mean_power_mw,changed.mean_power_mw,equal_nan=True)
    np.testing.assert_allclose(result.effective_sample_count,changed.effective_sample_count,equal_nan=True,atol=1e-12)
    with pytest.raises(ValueError,match='window'):lee_average_power(raw,max_spatial_diameter_m=.05)


def test_allocated_airtime_is_per_domain_channel_and_remainder_is_not_idle():
    p=Provenance(('allocation',))
    result=coarsen_airtime([.6,.2,.7],['a','b','a'],['room0','room0','room1'],['5g36','5g36','5g36'],provenance=p)
    assert result.allocated_fraction.sum()==pytest.approx(1.5)  # legitimate spatial reuse
    np.testing.assert_allclose(result.unallocated_fraction,[.2,.3])
    assert result.observed_idle_fraction is None
    assert check_airtime_consistency(result)['global_sum_constraint_applied'] is False
    with pytest.raises(ValueError):coarsen_airtime([.8,.4],['a','b'],['d','d'],['c','c'],provenance=p)
    with pytest.raises(ValueError):coarsen_airtime([.8],['a'],['d'],['c'],provenance=p,observed_idle={('d','c'):.3})


def test_airtime_grouping_is_permutation_invariant_and_unknown_idle_is_preserved():
    shares=np.array([.2,.1,.4,.3]);aps=np.array(['b','a','a','a']);domains=np.array(['d1','d0','d1','d0']);channels=np.array(['c1','c0','c1','c0'])
    first=coarsen_airtime(shares,aps,domains,channels,provenance=Provenance(('capture',)))
    order=[3,1,0,2]
    second=coarsen_airtime(shares[order],aps[order],domains[order],channels[order],provenance=Provenance(('capture',)))
    assert first.domain_channel_ids==second.domain_channel_ids and first.ap_ids==second.ap_ids
    np.testing.assert_allclose(first.allocated_fraction,second.allocated_fraction)
    with pytest.raises(ValueError):replace(first,unallocated_fraction=np.ones(2))


def test_route_union_bound_never_invents_temporal_independence():
    route=coarsen_route(modeled_links(),np.array([0,1,0]),[0.,1.,2.])
    assert route.model_any_event_union_bound==pytest.approx(.4)
    assert route.model_any_event_union_bound!=pytest.approx(1-.9*.8*.9)
    assert route.any_event_upper==pytest.approx(.6)
    assert route.outage_integral_seconds==pytest.approx(.3)
    proof=check_route_consistency(route)
    assert proof['any_event_frechet_lower']==.2
    assert proof['continuous_time_silence_bound'] is None and proof['temporal_independence_assumed'] is False
    assert route.provenance.qualified is False


def test_route_ordering_and_rigid_geometry_do_not_change_risk():
    first=coarsen_route(modeled_links(),np.array([0,1,0]),[0.,1.,2.],positions_m=[[0.,0.],[1.,0.],[1.,1.]])
    second=coarsen_route(modeled_links(),np.array([0,0,1]),[2.,0.,1.],positions_m=[[11.,-1.],[10.,0.],[10.,-1.]])
    np.testing.assert_allclose(first.outage_probability,second.outage_probability)
    assert first.event_ids==second.event_ids
    assert first.outage_integral_seconds==second.outage_integral_seconds
    with pytest.raises(ValueError):replace(first,outage_integral_seconds=.4)


def test_empirical_detection_conditioning_cannot_silently_become_route_probability():
    empirical=coarsen_samples(samples(),threshold_mw=.2)
    with pytest.raises(ValueError,match='Detected-only'):coarsen_route(empirical,np.array([0,0]),[0.,1.])
    route=coarsen_route(empirical,np.array([0,0]),[0.,1.],assume_representative=True)
    assert route.probability_kind=='empirical_detected_threshold_fraction'
    with pytest.raises(ValueError,match='Unknown'):coarsen_route(empirical,np.array([1,1]),[0.,1.],assume_representative=True)


def test_sla_is_model_advice_and_unknown_silence_or_calibration_fails_closed():
    route=coarsen_route(modeled_links(),np.array([0,1,0]),[0.,1.,2.])
    mission=MissionSLA('m',any_event_probability_limit=.7)
    assessment=assess_mission(route,mission)
    assert assessment.model_feasible is None and assessment.qualified is False and assessment.safety_guaranteed is False
    advisory=assess_mission(route,replace(mission,require_external_qualification=False))
    assert advisory.model_feasible is True and advisory.safety_guaranteed is False
    unknown=assess_mission(route,MissionSLA('m',silence_limit_s=2.,require_external_qualification=False))
    assert unknown.model_feasible is None
    fail=assess_mission(route,MissionSLA('m',any_event_probability_limit=.3,silence_limit_s=2.,require_external_qualification=False))
    assert fail.model_feasible is False and fail.checks['silence'] is None
    with pytest.raises(ValueError):replace(advisory,safety_guaranteed=True)


def test_lower_level_qualification_does_not_promote_a_new_skill():
    raw=replace(samples(),provenance=Provenance(('capture',),'xy',qualified=True,qualification_receipt='a'*64))
    assert coarsen_samples(raw).provenance.qualified is False
    with pytest.raises(ValueError):Provenance(('capture',),qualified=True)
    with pytest.raises(ValueError):MissionSLA('m')
    with pytest.raises(ValueError):MissionSLA('m',any_event_probability_limit='0.01')


def test_typed_arrays_are_immutable_and_unknown_uncertainty_cannot_be_fabricated():
    raw=samples();distribution=coarsen_samples(raw)
    with pytest.raises(ValueError):raw.observed.setflags(write=True)
    with pytest.raises(ValueError):distribution.mean_power_mw.setflags(write=True)
    with pytest.raises(ValueError):replace(distribution,mean_se_plugin_mw=np.array([0.,np.nan]))
    with pytest.raises(ValueError):replace(distribution,mean_power_mw=np.array([.55,0.]))
    with pytest.raises(ValueError):replace(raw,censor_upper_dbm=np.zeros((3,2)))
    with pytest.raises(ValueError):coarsen_route(modeled_links(),np.array([0,1]),[0.,1.],event_ids=['one'])
    with pytest.raises(ValueError):coarsen_route(modeled_links(),np.array([0,1]),[0.,1.],positions_m=[[0.,0.]])
