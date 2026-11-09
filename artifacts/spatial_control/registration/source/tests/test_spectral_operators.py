"""Analytic spectral operator fixtures, not new experiment outcomes."""
import math
import numpy as np
import pytest
import torch

from wirelessjev.spectral_scene import operators as op


def test_link_specific_fspl_units_and_frequency_are_not_cross_ap_mean():
    rx=np.array([[1.,0.,0.],[10.,0.,0.]])
    ap=np.zeros((2,3));f=np.array([2.4e9,5.2e9])
    received=op.fspl_received_power_dbm(rx,ap,f,20.)
    expected=20.-20*np.log10(4*np.pi*np.array([1.,10.])*f/299792458.)
    np.testing.assert_allclose(received,expected,atol=1e-12)
    assert received[0]!=received[1]
    assert op.fspl_loss_db([0.,0.,0.],[0.,0.,1.],2.4e9)==pytest.approx(40.0520080561155)
    with pytest.raises(ValueError):op.fspl_loss_db([0.,0.,0.],[0.,0.,0.],2.4e9)


def test_fspl_tensor_endpoint_frequency_and_power_gradients():
    rx=torch.tensor([[2.,0.,0.]],dtype=torch.float64,requires_grad=True)
    frequency=torch.tensor([5.2e9],dtype=torch.float64,requires_grad=True)
    power=torch.tensor([20.],dtype=torch.float64,requires_grad=True)
    result=op.fspl_received_power_dbm(rx,np.zeros((1,3)),frequency,power)
    result.sum().backward()
    assert torch.isfinite(rx.grad).all() and torch.isfinite(frequency.grad).all()
    torch.testing.assert_close(power.grad,torch.ones(1,dtype=torch.float64))


def test_noise_and_frequency_transport_state_identifiability_conditions():
    expected=10*math.log10(1.380649e-23*290.*20e6*1000.)+7.
    assert op.thermal_noise_dbm(20e6,noise_figure_db=7.)==pytest.approx(expected)
    with pytest.raises(ValueError):op.receiver_snr_db(-70.,-95.)
    assert op.receiver_snr_db(-70.,-95.,reference_gauge_assumed=True)==25.
    with pytest.raises(ValueError):op.frequency_transport_db(-60.,2.4e9,5.2e9)
    value=op.frequency_transport_db(-60.,2.4e9,5.2e9,same_physical_link=True,fixed_material_antenna_response=True)
    assert value==pytest.approx(-60.+20*np.log10(2.4/5.2))


def test_public_box_intersections_use_union_length_not_double_counting():
    rx=np.array([[10.,0.,1.],[10.,3.,1.]])
    ap=np.array([[0.,0.,1.],[0.,3.,1.]])
    boxes=np.array([[[2.,-1.,0.],[4.,1.,2.]],[[3.,-1.,0.],[6.,1.,2.]]])
    count,fraction=op.segment_box_features(rx,ap,boxes)
    np.testing.assert_array_equal(count,[2.,0.])
    np.testing.assert_allclose(fraction,[.4,0.])
    assert np.all((fraction>=0)&(fraction<=1))


def test_public_geometry_schema_and_eight_dimensional_physics_basis():
    rx=np.array([[10.,0.,1.],[2.,0.,1.]])
    ap=np.array([[0.,0.,1.],[0.,0.,1.]])
    boxes=np.array([[[2.,-1.,0.],[6.,1.,2.]]])
    geometry=op.link_geometry_features(rx,ap,[2.4e9,5.2e9],[0,1],[20.,10.,4.],boxes,
        calibration_pair_count=[100,0],calibration_unknown=[False,True],quantization_width_db=[1.,0.],noise_floor_dbm=[-97.,-90.])
    assert geometry.shape==(2,16)
    np.testing.assert_allclose(geometry[:,11],[1.,0.]);np.testing.assert_allclose(geometry[:,12],[0.,1.])
    basis=op.query_physics_basis(geometry)
    assert basis.shape==(2,8)
    assert basis[0,0]==1. and basis[0,3]==pytest.approx(.4)
    np.testing.assert_allclose(basis[:,5],geometry[:,7]*geometry[:,6])
    tensor=torch.tensor(geometry,dtype=torch.float64,requires_grad=True)
    rendered=op.query_physics_basis(tensor)
    np.testing.assert_allclose(rendered.detach(),basis,atol=1e-7)
    rendered.sum().backward();assert torch.isfinite(tensor.grad).all()


def test_rigid_translation_preserves_fspl_and_box_features():
    rx=np.array([[10.,0.,1.]]);ap=np.array([[0.,0.,1.]])
    boxes=np.array([[[2.,-1.,0.],[6.,1.,2.]]]);translation=np.array([7.,-3.,4.])
    np.testing.assert_allclose(op.fspl_received_power_dbm(rx,ap,5.2e9),op.fspl_received_power_dbm(rx+translation,ap+translation,5.2e9))
    first=op.segment_box_features(rx,ap,boxes);second=op.segment_box_features(rx+translation,ap+translation,boxes+translation)
    np.testing.assert_allclose(first,second)


@pytest.mark.parametrize('changes',[{'direction':[2]},{'carrier_hz':[-1.]},{'calibration_pair_count':[-1]}, {'quantization_width_db':[-1.]}])
def test_invalid_public_physical_metadata_rejected(changes):
    kwargs=dict(rx_xyz=np.array([[1.,0.,1.]]),ap_xyz=np.array([[0.,0.,2.]]),carrier_hz=[2.4e9],direction=[0],
                bbox_dims_m=[10.,10.,4.],cad_boxes_m=[])
    kwargs.update(changes)
    with pytest.raises(ValueError):op.link_geometry_features(**kwargs)
