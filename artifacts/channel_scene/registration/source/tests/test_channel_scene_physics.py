"""Analytic software fixtures, never new scientific runs or measured scores."""
import numpy as np
import pytest
import torch

from wirelessjev.channel_scene import calibration as c
from wirelessjev.channel_scene.encoding import ComplexCSIEncoder, CausalFourierOperator, spectral_curvature


def channels(batch=3, links=4, bins=64):
    f = (np.arange(bins)-bins/2) * 50.e6/bins
    coeff = np.ones((batch,links,3),complex)
    coeff[...,1] = .4+.3j;coeff[...,2] = -.2+.1j
    delay = np.broadcast_to([0.,35.e-9,92.e-9],coeff.shape)
    return c.render_cfr(coeff,delay,f), f


def test_actual_sto_cpo_convention_and_subselected_bin_identity():
    h,f = channels(bins=32)
    sto=np.array([0.,1.4,-2.,.5]);cpo=np.array([.3,-.2,1.1,-2.])
    k=np.arange(32);phase=sto[:,None]*(2*np.pi*k/32)+cpo[:,None]
    corrupted=h*np.exp(-1j*phase)
    restored=c.sto_cpo_correct(corrupted,sto,cpo,fft_size=32)
    np.testing.assert_allclose(restored,h,atol=2.e-15)
    chosen=np.array([1,4,9,22])
    selected=c.sto_cpo_correct(corrupted[...,chosen],sto,cpo,fft_size=32,shifted_bin_indices=chosen)
    np.testing.assert_allclose(selected,h[...,chosen],atol=2.e-15)
    with pytest.raises(ValueError,match='Subselected'):
        c.sto_cpo_correct(corrupted[...,chosen],sto,cpo,fft_size=32)
    np.testing.assert_allclose(abs(restored),abs(corrupted),atol=2.e-15)


def test_phase_curvature_cancels_independent_phase_and_sampling_origin():
    h,f=channels();rng=np.random.default_rng(10)
    phi=rng.uniform(-np.pi,np.pi,h.shape[:-1]);clock=rng.uniform(-.8e-6,.8e-6,h.shape[:-1])
    changed=h*np.exp(1j*phi[...,None]-2j*np.pi*clock[...,None]*f)
    a,mask=c.phase_curvature(h,f);b,other=c.phase_curvature(changed,f)
    for x,y,m,n in zip(a,b,mask,other):
        np.testing.assert_allclose(x,y,atol=5.e-14);np.testing.assert_array_equal(m,n)
    torch_values,torch_masks=spectral_curvature(torch.tensor(h),f)
    for x,y,m,n in zip(a,torch_values,mask,torch_masks):
        np.testing.assert_allclose(x,y.numpy(),atol=2.e-15);np.testing.assert_array_equal(m,n.numpy())


def test_complex_curvature_retains_information_absent_from_power():
    f=np.arange(32,dtype=float);h=np.ones((1,1,32),complex)
    other=np.exp(1j*.025*f**2)[None,None]
    np.testing.assert_allclose(abs(h),abs(other),atol=2.e-16)
    a,_=c.phase_curvature(h,f);b,_=c.phase_curvature(other,f)
    assert np.max(abs(a[0]-b[0]))>.04


def test_null_and_unobserved_phase_triplets_are_not_fabricated():
    h,f=channels();mask=np.ones(h.shape,bool);mask[...,3]=False
    h[...,3]=np.nan+1j*np.nan;h[...,12]=0
    values,valid=c.phase_curvature(h,f,observed=mask)
    assert not valid[0][...,1:4].any()
    assert np.isfinite(values[0]).all() and (values[0][~valid[0]]==0).all()
    broken=f.copy();broken[5]+=.1
    with pytest.raises(ValueError,match='uniform'):
        c.phase_curvature(np.nan_to_num(h),broken)


def test_cir_fft_roundtrip_and_offgrid_sidelobes_do_not_prove_noncausality():
    n=64;df=312500.;f=(np.arange(n)-31.5)*df;dt=1/(n*df)
    h=c.render_cfr(np.array([2.+1j]),np.array([4*dt]),f)
    result=c.spectrum_to_cir(h,f,absolute_reference_admitted=True)
    np.testing.assert_allclose(np.fft.fft(result.samples),h,atol=1.e-15)
    assert np.argmax(abs(result.samples))==4
    assert result.resolution_s==dt and result.ambiguity_period_s==1/df
    offgrid=c.spectrum_to_cir(c.render_cfr(np.array([1.+0j]),np.array([.5*dt]),f),f)
    assert abs(offgrid.samples[-1])>.1  # A causal off-grid path has boundary sidelobes.
    assert offgrid.causality_certified_from_finite_band is False
    tau=np.array([3.4*dt]);a=np.array([.5+.2j]);period=1/df
    # Finite uniform frequencies cannot distinguish this delay/coefficient gauge.
    shifted_a=a*np.exp(2j*np.pi*f[0]*period)
    np.testing.assert_allclose(c.render_cfr(a,tau,f),c.render_cfr(shifted_a,tau+period,f),atol=3.e-14)


def test_analytic_and_torch_causal_renderer_match_and_differentiate():
    f=(np.arange(64)-31.5)*312500.
    a=np.array([[2.+.2j,.3-.4j],[.5-.1j,.2+.8j]])
    tau=np.array([[25.e-9,100.e-9],[50.e-9,160.e-9]])
    actual=CausalFourierOperator()(torch.tensor(a,requires_grad=True),
        torch.tensor(tau,requires_grad=True),f)
    np.testing.assert_allclose(actual.detach(),c.render_cfr(a,tau,f),atol=3.e-15)
    # |H|>1 is allowed for referenced coefficients; no false passivity clipping.
    assert float(actual.detach().abs().max())>1
    aa=torch.tensor(a,requires_grad=True);tt=torch.tensor(tau,requires_grad=True)
    CausalFourierOperator()(aa,tt,f).abs().square().mean().backward()
    assert torch.isfinite(aa.grad).all() and torch.isfinite(tt.grad).all()
    with pytest.raises(ValueError,match='nonnegative'):
        CausalFourierOperator()(aa,-tt,f)
    with pytest.raises(ValueError,match='unknown relative'):
        CausalFourierOperator()(aa,tt,f,minimum_delay_s=10.e-9)
    with pytest.raises(ValueError,match='violates'):
        c.render_cfr(a,tau,f,minimum_delay_s=80.e-9)


def test_coherence_proof_binds_source_groups_and_ideal_assumption():
    proof=c.PhaseCoherenceProof('fixture-proof','global','ideal_simulation',{'source':'a'*64},(0,1),0.,0.)
    budget=c.PhaseCoherenceBudget('fixture-protocol',.1,1.e-10)
    with pytest.raises(ValueError,match='explicitly admitted'):
        proof.require({'source':'a'*64},[0,1],budget=budget)
    proof.require({'source':'a'*64},[0,1],ideal_simulation_admitted=True,budget=budget)
    with pytest.raises(ValueError,match='different source'):
        proof.require({'source':'b'*64},[0,1],ideal_simulation_admitted=True,budget=budget)
    with pytest.raises(ValueError,match='every requested group'):
        proof.require({'source':'a'*64},[2],ideal_simulation_admitted=True,budget=budget)
    with pytest.raises(ValueError,match='explicit protocol error budget'):
        proof.require({'source':'a'*64},[0,1],ideal_simulation_admitted=True)
    for phase,timing in [(100.,0.),(0.,1.e-6)]:
        poor=c.PhaseCoherenceProof('poor-proof','global','instrument_measurement',{'source':'a'*64},(0,1),phase,timing)
        with pytest.raises(ValueError,match='exceeds'):
            poor.require({'source':'a'*64},[0,1],budget=budget)


def encoder_case():
    h,f=channels();groups=torch.tensor([0,0,1,1]);xyz=torch.tensor([[0.,0.,1.],[0.,0.,1.],[5.,3.,2.],[5.,3.,2.]],dtype=torch.float64)
    return torch.tensor(h),f,dict(group_ids=groups,carrier_hz=3.44e9,infrastructure_xyz_m=xyz)


def test_encoder_is_gauge_invariant_and_preserves_absolute_power_bypass():
    torch.manual_seed(20);model=ComplexCSIEncoder(hidden=16,heads=4).eval();h,f,kw=encoder_case()
    a=model(h,f,**kw)
    phi=torch.arange(h.shape[0]*h.shape[1],dtype=torch.float64).reshape(h.shape[:-1])*.4
    clock=torch.sin(phi)*.4e-6
    changed=h*torch.exp(1j*phi[...,None]-2j*torch.pi*clock[...,None]*torch.tensor(f))
    b=model(changed,f,**kw)
    torch.testing.assert_close(a.scene,b.scene,rtol=1.e-6,atol=1.e-6)
    louder=model(h*10,f,**kw)
    torch.testing.assert_close(louder.mean_power_db-a.mean_power_db,torch.full_like(a.mean_power_db,20.))
    assert not any(isinstance(x,torch.nn.LayerNorm) for x in model.modules())


def test_encoder_power_summaries_average_linear_power_with_missing_bin_masks():
    model=ComplexCSIEncoder(hidden=16,heads=4)
    h=torch.ones((1,2,16),dtype=torch.complex128)
    h[:,0,8:]=10.;h[:,1,:]=2.
    mask=torch.ones(h.shape,dtype=torch.bool);mask[:,1,4:]=False
    output=model(h,np.arange(16)*1.e6,group_ids=[0,1],carrier_hz=3.4e9,observed=mask)
    expected_link=10*torch.log10(torch.tensor([[50.5,4.]],dtype=torch.float64))
    expected_scene=10*torch.log10(torch.tensor([(16*50.5+4*4)/20],dtype=torch.float64))
    torch.testing.assert_close(output.link_power_db,expected_link)
    torch.testing.assert_close(output.mean_power_db,expected_scene)
    assert float(output.link_power_db[0,0])>17.  # The mean of bin dB values would be only 10 dB.


def test_encoder_retains_actual_frequency_scale_for_delay_generalization():
    torch.manual_seed(24);model=ComplexCSIEncoder(hidden=16,heads=4).eval();h,f,kw=encoder_case()
    narrow=model(h,f/2.5,**kw);wide=model(h,f*1.6,**kw)
    # Same discrete samples at 20 and80MHz have different physical delay scales.
    assert float((narrow.scene-wide.scene).detach().abs().max())>1.e-5


def test_encoder_set_permutation_group_relabel_and_se3_geometry_invariance():
    torch.manual_seed(21);model=ComplexCSIEncoder(hidden=16,heads=4).eval();h,f,kw=encoder_case()
    a=model(h,f,**kw);order=torch.tensor([3,1,0,2]);other=dict(kw)
    other['group_ids']=kw['group_ids'][order]*17+9;other['infrastructure_xyz_m']=kw['infrastructure_xyz_m'][order]
    b=model(h[:,order],f,**other)
    torch.testing.assert_close(a.scene,b.scene,rtol=1.e-6,atol=1.e-6)
    torch.testing.assert_close(a.link_tokens[:,order],b.link_tokens,rtol=1.e-6,atol=1.e-6)
    rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=torch.float64)
    other=dict(kw);other['infrastructure_xyz_m']=kw['infrastructure_xyz_m']@rotation.T+1.e6
    d=model(h,f,**other)
    torch.testing.assert_close(a.scene,d.scene,rtol=1.e-6,atol=1.e-6)


def test_unobserved_nan_padding_is_inert_and_empty_context_requests_fallback():
    torch.manual_seed(22);model=ComplexCSIEncoder(hidden=16,heads=4).eval();h,f,kw=encoder_case()
    a=model(h,f,**kw)
    padded=torch.cat([h,torch.full_like(h[:,:1],complex(float('nan'),float('nan')))],1)
    extra=dict(kw);extra['group_ids']=torch.cat([kw['group_ids'],torch.tensor([-1])])
    extra['infrastructure_xyz_m']=torch.cat([kw['infrastructure_xyz_m'],torch.zeros(1,3)])
    mask=torch.ones(padded.shape[:-1],dtype=torch.bool);mask[:,-1]=False
    b=model(padded,f,observed=mask,**extra)
    torch.testing.assert_close(a.scene,b.scene,rtol=1.e-6,atol=1.e-6)
    with pytest.raises(ValueError,match='fallback'):
        model(h,f,observed=torch.zeros(h.shape[:-1],dtype=torch.bool),**kw)


def test_complex_encoder_uses_phase_information_and_has_finite_gradients():
    torch.manual_seed(23);model=ComplexCSIEncoder(hidden=16,heads=4);h,f,kw=encoder_case()
    h=torch.ones_like(h).requires_grad_(True)
    a=model(h,f,**kw)
    normalized=torch.linspace(-1,1,h.shape[-1],dtype=torch.float64)
    dispersive=h*torch.exp(12j*normalized.square())
    b=model(dispersive,f,**kw)
    assert float((a.scene-b.scene).detach().abs().max())>1.e-7
    b.scene.square().sum().backward()
    assert torch.isfinite(h.grad).all() and float(h.grad.abs().sum())>0


def test_coherent_and_absolute_encoding_fail_closed_without_actual_evidence():
    h,f,kw=encoder_case();source={'source':'a'*64}
    relative=ComplexCSIEncoder(hidden=16,heads=4,phase_mode='relative_coherent')
    with pytest.raises(ValueError,match='bound phase evidence'):
        relative(h,f,**kw)
    proof=c.PhaseCoherenceProof('fixture','global','ideal_simulation',source,(0,1),0.,0.)
    budget=c.PhaseCoherenceBudget('fixture-protocol',.1,1.e-10)
    with pytest.raises(ValueError,match='explicit protocol error budget'):
        relative(h,f,coherence_proof=proof,source_sha256=source,ideal_simulation_admitted=True,**kw)
    relative(h,f,coherence_proof=proof,coherence_budget=budget,source_sha256=source,ideal_simulation_admitted=True,**kw)
    absolute=ComplexCSIEncoder(hidden=16,heads=4,phase_mode='absolute_coherent')
    with pytest.raises(ValueError,match='absolute clock'):
        absolute(h,f,coherence_proof=proof,coherence_budget=budget,source_sha256=source,ideal_simulation_admitted=True,**kw)
    absolute(h,f,coherence_proof=proof,coherence_budget=budget,source_sha256=source,ideal_simulation_admitted=True,
             absolute_reference_admitted=True,**kw)
