"""Analytic/negative software tests; not a physical-radio experiment."""
from dataclasses import replace
import math
import numpy as np
import pytest
from scipy import special, stats
import torch

from wirelessjev.radiojepa import operators as op, likelihoods as lk, dynamics as dyn
from wirelessjev.radiojepa.manifolds import unit_phasor, positive_amplitude, airtime_simplex, busy_airtime
from wirelessjev.radiojepa.masking import SimultaneousViews, validate_disjoint_views, coherence_mask


def test_power_units_and_noise_analytic():
    dbm = np.array([-90., -30., 0., 20.])
    np.testing.assert_allclose(op.mw_to_dbm(op.dbm_to_mw(dbm)), dbm, atol=1e-13)
    assert op.dbm_to_mw(0.) == 1.
    assert op.mw_to_dbm(0.) == -np.inf
    assert op.mw_to_dbm(0., floor_mw=1e-10) == -100.
    expected = 10 * math.log10(1.380649e-23 * 290 * 20e6) + 30 + 7
    assert op.thermal_noise_dbm(20e6, noise_figure_db=7.) == pytest.approx(expected)
    assert op.bandwidth_noise_shift_db(20e6, 80e6) == pytest.approx(6.020599913279624)
    assert op.fspl_frequency_shift_db(3e9, 6e9) == pytest.approx(-6.020599913279624)


@pytest.mark.parametrize('call', [
    lambda: op.mw_to_dbm(-1.), lambda: op.dbm_to_mw(np.nan),
    lambda: op.thermal_noise_dbm(0.), lambda: op.thermal_noise_dbm(1., temperature_k=-1.),
    lambda: op.bandwidth_noise_shift_db(-1., 1.), lambda: op.fspl_frequency_shift_db(1., 0.),
    lambda: op.snr_after_bandwidth_change(10., 20., 80., signal_assumption='unknown'),
    lambda: op.dbm_to_mw(torch.tensor(1+1j)),
])
def test_invalid_units_and_assumptions_rejected(call):
    with pytest.raises(ValueError): call()


def test_bandwidth_signal_convention_is_not_invented():
    fixed = op.snr_after_bandwidth_change(10., 20., 80., signal_assumption='fixed_total_received_power')
    psd = op.snr_after_bandwidth_change(10., 20., 80., signal_assumption='flat_received_power_spectral_density')
    assert fixed == pytest.approx(10. - 10 * np.log10(4))
    assert psd == 10.


def test_reciprocity_transposes_without_conjugating_and_requires_calibration():
    h = np.array([[1+2j, 3-4j], [5+6j, 7-8j]])
    kw = dict(frequency_hz=5e9, reverse_frequency_hz=5e9, delta_time_s=.001, coherence_time_s=.01, calibrated=True)
    np.testing.assert_array_equal(op.reciprocal_channel(h, **kw), h.T)
    assert not np.array_equal(op.reciprocal_channel(h, **kw), h.conj().T)
    for changes in ({'calibrated':False}, {'reverse_frequency_hz':6e9}, {'delta_time_s':.02}):
        with pytest.raises(ValueError): op.reciprocal_channel(h, **(kw | changes))


def test_cir_renderer_analytic_phase_and_differentiability():
    coefficients = torch.tensor([[1+0j, .5j]], dtype=torch.complex128, requires_grad=True)
    delay = torch.tensor([0., 1e-7], dtype=torch.float64, requires_grad=True)
    frequencies = torch.tensor([0., 2.5e6, 5e6], dtype=torch.float64)
    rendered = op.render_cir(coefficients, delay, frequencies)
    expected = np.array([[1+.5j, 1.5+0j, 1-.5j]])
    np.testing.assert_allclose(rendered.detach(), expected, atol=2e-15)
    rendered.abs().square().sum().backward()
    assert torch.isfinite(coefficients.grad).all() and torch.isfinite(delay.grad).all()
    np.testing.assert_allclose(op.render_cir(coefficients.detach().numpy(), delay.detach().numpy(), frequencies.numpy()), expected, atol=2e-15)


def test_coherent_paths_cancel_but_incoherent_powers_add():
    assert op.coherent_power_sum(np.array([1+0j, -1+0j])) == 0.
    assert op.incoherent_power_sum([1., 1.], incoherent=True) == 2.
    with pytest.raises(ValueError): op.incoherent_power_sum([1., 1.])


def test_causality_and_passivity_require_identified_units():
    assert op.validate_absolute_delays([1e-7], [3.], absolute_toa_identified=True)
    with pytest.raises(ValueError): op.validate_absolute_delays([1e-7], [3.])
    with pytest.raises(ValueError): op.validate_absolute_delays([0.], [3.], absolute_toa_identified=True)
    assert op.validate_normalized_port_power(np.eye(2)*.7, normalized_ports=True)
    with pytest.raises(ValueError): op.validate_normalized_port_power(np.eye(2)*.7)
    with pytest.raises(ValueError): op.validate_normalized_port_power(np.eye(2)*1.01, normalized_ports=True)
    # Checking each coefficient separately would miss this coherent port gain.
    with pytest.raises(ValueError): op.validate_normalized_port_power(np.full((2,2), .7), normalized_ports=True)


def test_product_coordinates_keep_phase_power_and_idle_semantics():
    phase = torch.tensor([0., math.pi, 2*math.pi], dtype=torch.float64, requires_grad=True)
    phasor = unit_phasor(phase)
    torch.testing.assert_close(phasor.abs(), torch.ones(3, dtype=torch.float64))
    amplitude = positive_amplitude(torch.tensor([-100., 0., 100.]))
    assert (amplitude > 0).all() and torch.isfinite(amplitude).all()
    shares = airtime_simplex(torch.tensor([[1.,2.,3.],[0.,0.,0.]], requires_grad=True))
    torch.testing.assert_close(shares.sum(-1), torch.ones(2))
    assert (busy_airtime(torch.tensor([[1.,2.,3.]])).sum(-1) < 1).all()
    assert lk.circular_loss(0., 2*math.pi) == pytest.approx(0., abs=1e-15)
    (phasor.imag.sum() + shares[0,0]).backward()
    assert torch.isfinite(phase.grad).all()


def test_tobit_matches_density_and_actual_censor_likelihood():
    value = np.array([-62., np.nan, -74.])
    mean = np.array([-60., -85., -70.])
    std = np.array([2.,3.,4.])
    bound = np.array([np.nan,-90.,np.nan])
    got = lk.gaussian_tobit_log_prob(value, mean, std, censored=np.array([False,True,False]), upper_bound=bound)
    expected = [stats.norm.logpdf(-62.,-60.,2.), stats.norm.logcdf(-90.,-85.,3.), stats.norm.logpdf(-74.,-70.,4.)]
    np.testing.assert_allclose(got, expected, atol=2e-14)
    with pytest.raises(ValueError): lk.gaussian_tobit_log_prob(0., 0., 1., censored=1, upper_bound=-1.)


@pytest.mark.parametrize('lower,upper', [(-np.inf,-10.),(-11.,-10.),(-1.,1.),(10.,10.01),(40.,40.01),(10.,np.inf),(-np.inf,np.inf)])
def test_interval_likelihood_is_stable_in_both_extreme_tails(lower,upper):
    got = lk.gaussian_interval_log_prob(lower,upper,0.,1.)
    if lower > 0:
        a,b = stats.norm.logsf(lower), stats.norm.logsf(upper)
    else:
        a,b = stats.norm.logcdf(upper), stats.norm.logcdf(lower)
    expected = a + np.log(-np.expm1(b-a))
    assert got == pytest.approx(expected, abs=1e-10)
    mean = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    std = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    value = lk.gaussian_interval_log_prob(lower,upper,mean,std)
    value.backward()
    assert torch.isfinite(mean.grad) and torch.isfinite(std.grad)


def test_interval_order_and_quantization_probability():
    assert lk.gaussian_interval_log_prob(1.,1.,0.,1.) == -np.inf
    with pytest.raises(ValueError): lk.gaussian_interval_log_prob(2.,1.,0.,1.)
    bins = np.arange(-12., 12.5, .5)
    probabilities = np.exp(lk.gaussian_interval_log_prob(bins[:-1],bins[1:],0.,2.))
    assert probabilities.sum() == pytest.approx(stats.norm.cdf(6)-stats.norm.cdf(-6), abs=2e-14)


def test_fading_likelihoods_match_scipy_and_have_finite_interior_gradients():
    x = np.array([.02,.3,1.,3.,10.])
    for nu,sigma in ((0.,1.),(2.,.5)):
        np.testing.assert_allclose(lk.rician_log_prob(x,nu,sigma), stats.rice.logpdf(x,nu/sigma,scale=sigma), atol=2e-9, rtol=1e-12)
    near = np.array([49.9,50.,50.1])
    np.testing.assert_allclose(lk.rician_log_prob(near,50.,.1), stats.rice.logpdf(near,500.,scale=.1), atol=2e-11)
    # scipy.stats first evaluates pdf and underflows in this far high-K tail;
    # the scaled-Bessel log implementation retains the finite log density.
    assert np.isfinite(lk.rician_log_prob(x,50.,.1)).all()
    for m,omega in ((.5,2.),(1.,1.),(3.,.2)):
        np.testing.assert_allclose(lk.nakagami_log_prob(x,m,omega), stats.nakagami.logpdf(x,m,scale=np.sqrt(omega)), atol=1e-11)
    np.testing.assert_allclose(lk.lognormal_log_prob(x,.3,.7), stats.lognorm.logpdf(x,.7,scale=np.exp(.3)), atol=1e-12)
    for call in (lambda x: lk.rician_log_prob(x,.7,.2), lambda x: lk.nakagami_log_prob(x,1.5,.8), lambda x: lk.lognormal_log_prob(x,.1,.8)):
        value = torch.tensor([.2, .7, 2.], dtype=torch.float64, requires_grad=True)
        call(value).sum().backward()
        assert torch.isfinite(value.grad).all()


def test_burst_filter_exact_first_emission_missing_event_and_gradients():
    transition = torch.tensor([[.9,.1],[.2,.8]], dtype=torch.float64, requires_grad=True)
    drop = torch.tensor([.01,.8], dtype=torch.float64, requires_grad=True)
    result = lk.gilbert_elliott_filter([1,0,1], transition, drop, [.75,.25], observed=[True,False,True])
    prior = np.array([.75,.25]); p=np.array([.01,.8]); t=transition.detach().numpy()
    likelihood = prior@p
    first = prior*p/likelihood
    second = first@t
    third_prior = second@t
    third_likelihood = third_prior@p
    np.testing.assert_allclose(result['filtered'].detach(), [first,second,third_prior*p/third_likelihood], atol=1e-14)
    assert float(result['log_likelihood'].detach()) == pytest.approx(np.log(likelihood*third_likelihood))
    result['log_likelihood'].backward()
    assert torch.isfinite(drop.grad).all() and torch.isfinite(transition.grad).all()
    with pytest.raises(ValueError): lk.gilbert_elliott_filter([1], [[1.,0.],[0.,1.]], [0.,0.], [.5,.5])


def test_ou_exact_moments_and_irregular_semigroup():
    whole = dyn.ou_transition(2.,3.,.7,2.,4.,equilibrium=-1.)
    part = dyn.ou_transition(2.,3.,.2,2.,4.,equilibrium=-1.)
    repeated = dyn.ou_transition(*part,.5,2.,4.,equilibrium=-1.)
    np.testing.assert_allclose(whole,repeated,atol=1e-14)
    assert dyn.ou_transition(2.,3.,0.,2.,4.) == (2.,3.)
    np.testing.assert_allclose(dyn.ou_transition(2.,3.,1e6,2.,4.,equilibrium=-1.),[-1.,4.])
    with pytest.raises(ValueError): dyn.ou_transition(0.,1.,-.1,1.,1.)


def test_spatial_covariance_loop_closure_and_rigid_motion():
    positions = np.array([[0.,0.],[1.,0.],[1.,2.],[0.,0.]])
    covariance = dyn.spatial_shadow_covariance(positions,2.,4.)
    np.testing.assert_array_equal(covariance[0], covariance[-1])
    rotation = np.array([[0.,-1.],[1.,0.]])
    np.testing.assert_allclose(dyn.spatial_shadow_covariance(positions@rotation+np.array([7.,-3.]),2.,4.),covariance,atol=1e-14)
    assert np.linalg.eigvalsh(covariance).min() > -1e-12


def test_jakes_correlation_and_first_derivative_match_reference():
    dt = torch.tensor([0.,.001,.01,.1], dtype=torch.float64, requires_grad=True)
    actual = dyn.jakes_correlation(12.,dt)
    np.testing.assert_allclose(actual.detach(),special.j0(2*np.pi*12*dt.detach().numpy()),atol=2e-8)
    actual.sum().backward()
    np.testing.assert_allclose(dt.grad,-2*np.pi*12*special.j1(2*np.pi*12*dt.detach().numpy()),atol=2e-6)
    mean,var = dyn.jakes_ar_transition(1.,2.,12.,0.,3.)
    assert mean == 1. and var == 2.
    mean,var = dyn.jakes_ar_transition(1+2j,2.,12.,0.,3.)
    assert mean == 1+2j and var == 2.
    complex_mean = torch.tensor(1+2j, dtype=torch.complex128, requires_grad=True)
    value,_ = dyn.jakes_ar_transition(complex_mean,1.,12.,.01,2.)
    value.abs().square().backward()
    assert torch.isfinite(complex_mean.grad)


def test_simultaneous_metadata_and_disjoint_group_masks():
    views = SimultaneousViews('packet7',1.,1.,(0,1),(4,5))
    left,right = validate_disjoint_views(views,6)
    assert not (left&right).any() and not left.flags.writeable
    for change in ({'target_indices':(1,2)}, {'target_capture_s':1.1}, {'provenance':'next_step'}, {'scene_id':''}):
        with pytest.raises(ValueError): validate_disjoint_views(replace(views,**change),6)
    group = np.array([0,0,1,1,2,2])
    coords = np.array([0.,.1,2.,2.1,5.,5.1])
    a,b = coherence_mask(group,[0],[2],coordinates=coords,minimum_gap=4.)
    np.testing.assert_array_equal(a,left)
    np.testing.assert_array_equal(b,right)
    with pytest.raises(ValueError): coherence_mask(group,[0],[1],coordinates=coords,minimum_gap=3.)
    with pytest.raises(ValueError): coherence_mask(group,[0],[0])
