"""Analytic linear-algebra software cases, not new radio experiments."""
import numpy as np
import pytest

from wirelessjev.sensing_design.operators import (GaussianPrior,certify_targets,
    expected_posterior_covariance,target_variance)
from wirelessjev.sensing_design.pilots import greedy_targeted_pilots


def test_nullspace_is_target_specific_not_a_ban_on_underdetermined_models():
    a=np.array([[1.,0.,0.]])
    missing=certify_targets(a,[[0.,1.,0.]])
    measured=certify_targets(a,[[1.,0.,0.]])
    assert missing.ambient.rank==1 and missing.ambient.nullity==2
    assert not missing.ambient.target_identifiable
    assert missing.ambient.target_nullspace_frobenius_squared==1
    assert missing.ambient.full_column_condition_number is None
    assert missing.ambient.observed_subspace_condition_number==1
    assert measured.ambient.target_identifiable and measured.ambient.nullity==2


def test_low_rank_prior_support_is_explicit_and_does_not_rewrite_ambient_claim():
    a=[[1.,0.]];t=[[0.,1.]]
    prior=GaussianPrior('explicit perfectly correlated toy support',np.ones((2,2)))
    certificate=certify_targets(a,t,prior=prior)
    assert not certificate.ambient.target_identifiable
    assert certificate.prior_supported.target_identifiable
    assert certificate.prior_supported.unknown_dimensions==1
    covariance=expected_posterior_covariance(prior,a,0.)
    assert target_variance(covariance,t)<1.e-13
    noisy=expected_posterior_covariance(prior,a,1.)
    np.testing.assert_allclose(target_variance(noisy,t),.5,atol=1.e-14)
    broad=certify_targets(a,t,prior=GaussianPrior('full support',np.eye(2)))
    assert not broad.prior_supported.target_identifiable


def test_actual_regular_16_of_64_mask_and_two_view_union_have_target_nullspace():
    k=np.arange(64);basis=np.exp(-2j*np.pi*k[:,None]*k[None]/64)
    first=np.arange(0,64,4);second=np.arange(2,64,4);targets=np.arange(1,64,2)
    a=certify_targets(basis[first],basis[targets]);b=certify_targets(basis[second],basis[targets])
    assert a.ambient.rank==b.ambient.rank==16
    assert a.ambient.nullity==b.ambient.nullity==48
    union=certify_targets(basis[np.r_[first,second]],basis[targets])
    assert union.ambient.rank==32 and union.ambient.nullity==32
    assert union.ambient.target_ambiguous_rank==32
    # Both views miss e_m-e_(m+32), which every odd target responds to.
    delta=np.zeros(64,complex);delta[3]=1;delta[35]=-1
    np.testing.assert_allclose(basis[np.r_[first,second]]@delta,0,atol=1.e-12)
    np.testing.assert_allclose(abs(basis[targets]@delta),2,atol=1.e-12)
    assert not union.ambient.target_identifiable


def test_complex_conjugation_and_noise_give_correct_target_variance():
    prior=GaussianPrior('proper complex toy',np.array([[1,.5j],[-.5j,1]]))
    covariance=expected_posterior_covariance(prior,[[1,1j]],1.)
    np.testing.assert_allclose(target_variance(covariance,[[0,1]]),.4375,atol=1.e-14)
    design=greedy_targeted_pilots([[1,1j]],[[0,1]],prior,noise_variance=1.,costs=1.,budget=1.,cost_unit='declared units')
    np.testing.assert_allclose(design.choices[0].expected_target_variance_drop,.5625,atol=1.e-14)
    np.testing.assert_allclose(design.posterior_covariance,covariance,atol=1.e-14)


def test_target_gain_cost_budget_and_irrelevant_sensor():
    prior=GaussianPrior('toy diagonal',np.diag([4.,9.]))
    design=greedy_targeted_pilots([[1.,0.],[0.,1.]],[[0.,1.]],prior,
        noise_variance=[0.,3.],costs=[.01,2.],budget=2.,cost_unit='microseconds',candidate_ids=['irrelevant','target'])
    assert len(design.choices)==1 and design.choices[0].candidate_id=='target'
    np.testing.assert_allclose(design.choices[0].expected_target_variance_drop,6.75,atol=1.e-13)
    np.testing.assert_allclose(design.remaining_target_variance,2.25,atol=1.e-13)
    assert design.total_cost==2. and not design.globally_optimal
    empty=greedy_targeted_pilots([[0.,1.]],[[0.,1.]],prior,noise_variance=0.,costs=3.,budget=2.,cost_unit='units')
    assert not empty.choices and empty.total_cost==0


def test_small_positive_prior_direction_can_be_large_in_target_units():
    prior=GaussianPrior('small physically supported direction',np.diag([1.,1.e-20]))
    design=greedy_targeted_pilots([[0.,1.]],[[0.,1.e10]],
        prior,
        noise_variance=0.,costs=1.,budget=1.,cost_unit='units')
    assert len(design.choices)==1
    np.testing.assert_allclose(design.initial_target_variance,1.,atol=1.e-14)
    np.testing.assert_allclose(design.choices[0].expected_target_variance_drop,1.,atol=1.e-14)
    assert design.remaining_target_variance<1.e-14
    certificate=certify_targets([[1.,0.]],[[0.,1.e10]],prior=prior)
    assert not certificate.prior_supported.target_identifiable
    assert certificate.prior_supported.unknown_dimensions==2
    assert not certificate.prior_support_truncated and certificate.omitted_prior_target_variance==0.
    approximate=certify_targets([[1.,0.]],[[0.,1.e10]],prior=prior,prior_support_rtol=1.e-12)
    assert approximate.prior_support_truncated
    np.testing.assert_allclose(approximate.omitted_prior_target_variance,1.)
    observed=expected_posterior_covariance(prior,np.eye(2),0.)
    assert target_variance(observed,[[0.,1.e10]])<1.e-14


def test_sequential_independent_conditioning_matches_joint_complex_schur_complement():
    rng=np.random.default_rng(91)
    loading=rng.normal(size=(5,5))+1j*rng.normal(size=(5,5))
    covariance=loading@loading.conj().T+.1*np.eye(5)
    rows=rng.normal(size=(3,5))+1j*rng.normal(size=(3,5));noise=np.array([.2,.3,.4])
    expected=covariance-covariance@rows.conj().T@np.linalg.inv(rows@covariance@rows.conj().T+np.diag(noise))@rows@covariance
    actual=expected_posterior_covariance(GaussianPrior('proper complex SPD',covariance),rows,noise)
    np.testing.assert_allclose(actual,expected,rtol=1.e-12,atol=1.e-12)


def test_greedy_recomputes_redundancy_and_noisy_repeats():
    prior=GaussianPrior('two directions',np.eye(2))
    rows=[[1.,0.],[1.,0.],[0.,1.]]
    noiseless=greedy_targeted_pilots(rows,np.eye(2),prior,noise_variance=0.,costs=1.,budget=3.,cost_unit='units')
    assert [v.candidate_index for v in noiseless.choices]==[0,2]
    assert noiseless.remaining_target_variance<1.e-14
    noisy=greedy_targeted_pilots([[1.],[1.]],[[1.]],GaussianPrior('one direction',[[1.]]),
        noise_variance=1.,costs=1.,budget=2.,cost_unit='units')
    assert len(noisy.choices)==2
    np.testing.assert_allclose(noisy.remaining_target_variance,1/3,atol=1.e-14)


def test_fractional_cost_budget_has_explicit_bounded_roundoff_policy():
    design=greedy_targeted_pilots(np.eye(2),np.eye(2),GaussianPrior('unit',np.eye(2)),
        noise_variance=0.,costs=[.1,.2],budget=.3,cost_unit='units')
    assert len(design.choices)==2 and design.total_cost==.3
    assert 0<=design.cost_roundoff_adjustment<=design.cost_roundoff_tolerance
    np.testing.assert_allclose(design.sum_choice_costs_before_roundoff,.1+.2,rtol=0,atol=0)
    zero=greedy_targeted_pilots([[1.]],[[1.]],GaussianPrior('unit',[[1.]]),
        noise_variance=0.,costs=np.nextafter(0.,1.),budget=0.,cost_unit='units')
    assert not zero.choices and zero.cost_roundoff_tolerance==0.
    tiny=np.nextafter(0.,1.)
    subnormal=greedy_targeted_pilots([[1.]],[[np.sqrt(tiny)]],GaussianPrior('unit',[[1.]]),
        noise_variance=0.,costs=5*tiny,budget=tiny,cost_unit='units')
    assert not subnormal.choices and subnormal.cost_roundoff_tolerance==0.


def test_small_positive_target_gain_is_not_lost_to_unrelated_large_variance():
    design=greedy_targeted_pilots([[0.,1.]],np.eye(2),GaussianPrior('mixed scales',np.diag([1.e20,1.])),
        noise_variance=0.,costs=1.,budget=1.,cost_unit='units')
    assert len(design.choices)==1 and design.choices[0].expected_target_variance_drop==1.


def test_finite_operands_that_overflow_results_are_refused():
    with pytest.raises(FloatingPointError,match='overflow'):
        target_variance(np.eye(2),1.e200*np.eye(2))
    with pytest.raises(FloatingPointError,match='overflow'):
        certify_targets(np.zeros((0,2)),1.e200*np.eye(2))
    with pytest.raises(FloatingPointError,match='overflow'):
        greedy_targeted_pilots([[1.]],[[1.]],GaussianPrior('one',[[1.]]),
            noise_variance=0.,costs=1.e-320,budget=1.,cost_unit='units')


def test_targeted_design_escapes_even_bin_alias_with_additional_odd_pilots():
    k=np.arange(64);basis=np.exp(-2j*np.pi*k[:,None]*k[None]/64)/8
    design=greedy_targeted_pilots(basis,basis[1::2],GaussianPrior('explicit isotropic coefficients',np.eye(64)),
        noise_variance=0.,costs=1.,budget=3.,cost_unit='sounding actions',existing_operator=basis[::2])
    assert len(design.choices)==3 and all(v.candidate_index%2 for v in design.choices)
    np.testing.assert_allclose(design.initial_target_variance,32.,atol=1.e-12)
    np.testing.assert_allclose(design.remaining_target_variance,29.,atol=1.e-11)


def test_empty_observations_zero_prior_and_zero_budget_are_finite():
    prior=GaussianPrior('known deterministic zero',np.zeros((2,2)))
    certificate=certify_targets(np.zeros((0,2)),[[0.,1.]],prior=prior)
    assert not certificate.ambient.target_identifiable
    assert certificate.prior_supported.target_identifiable and certificate.prior_supported.unknown_dimensions==0
    result=greedy_targeted_pilots(np.eye(2),np.eye(2),prior,noise_variance=0.,costs=1.,budget=3.,cost_unit='units')
    assert not result.choices and result.remaining_target_variance==0.
    zero=greedy_targeted_pilots(np.eye(2),np.eye(2),GaussianPrior('unit',np.eye(2)),
        noise_variance=0.,costs=1.,budget=0.,cost_unit='units')
    assert not zero.choices and np.isfinite(zero.posterior_covariance).all()


@pytest.mark.parametrize('covariance',[[[1,1j],[1j,1]],[[1,0],[0,-.1]]])
def test_invalid_covariance_is_not_silently_a_prior(covariance):
    with pytest.raises(ValueError):GaussianPrior('invalid',covariance)


def test_prior_and_result_arrays_cannot_be_mutated():
    original=np.eye(2);prior=GaussianPrior('fixed',original);original[:]=99
    np.testing.assert_array_equal(prior.covariance,np.eye(2))
    with pytest.raises(ValueError):prior.covariance.setflags(write=True)
    with pytest.raises(ValueError):greedy_targeted_pilots([[1,0]],[[1,0]],prior,
        noise_variance=0.,costs=0.,budget=1.,cost_unit='units')
