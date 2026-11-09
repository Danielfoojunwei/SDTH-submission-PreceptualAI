"""Bounded arithmetic/invariance tests, not RF or learned-model performance."""
from dataclasses import replace
import math

import numpy as np
import pytest

from wirelessjev.visual_radio.physics import (
    Channel, ReceiverState, TransmitterState, RFCoupling, VisualSilhouette,
    channel_overlap_fraction, coupling_from_isolated_power_dbm,
    first_fresnel_radius_m, free_space_coupling, fresnel_silhouette_clearance,
    receiver_link_budget, SPEED_OF_LIGHT_M_S,
)


C = Channel(5.2e9, 20e6)


def public_rf():
    receiver = ReceiverState('robot', (0., 0., 1.), C, 1e-6)
    desired = TransmitterState('ap/radio0', (10., 0., 2.), C, 10.)
    interferers = [TransmitterState('other/radio0', (5., 3., 2.), C, 10., .5),
                  TransmitterState('other/radio1', (4., -2., 2.), C, 0., 1.)]
    couplings = [RFCoupling('robot', desired.transmitter_id, C.center_hz, 1e-4),
                 RFCoupling('robot', interferers[0].transmitter_id, C.center_hz, 1e-5),
                 RFCoupling('robot', interferers[1].transmitter_id, C.center_hz, 2e-5)]
    return receiver, desired, interferers, couplings


def budget(receiver, desired, interferers=(), couplings=()):
    return receiver_link_budget(receiver, desired, interferers, couplings, inventory_complete=True)


def test_receiver_centric_power_adds_in_linear_units_and_conditions_desired_packet():
    receiver, desired, interferers, couplings = public_rf()
    result = budget(receiver, desired, interferers, couplings)
    assert result.desired.received_mw == pytest.approx(1e-3)
    assert [r.received_mw for r in result.interferers] == pytest.approx([5e-5, 2e-5])
    assert result.known_interference_mw == pytest.approx(7e-5)
    assert result.sinr_linear == pytest.approx(1e-3 / (1e-6 + 7e-5))
    assert result.sinr_db == pytest.approx(10 * math.log10(result.sinr_linear))
    # Duty cycle describes population occupancy; it cannot attenuate a desired
    # packet conditioned on its actual/hypothetical transmission.
    conditioned = budget(receiver, replace(desired, activity_fraction=.01), interferers, couplings)
    assert conditioned == result


def test_rectangular_overlap_uses_interferer_bandwidth_and_exact_endpoints():
    wide = Channel(C.center_hz, 40e6)
    assert channel_overlap_fraction(C, wide) == .5
    assert channel_overlap_fraction(wide, C) == 1.
    assert channel_overlap_fraction(C, Channel(C.center_hz + 10e6, 20e6)) == .5
    assert channel_overlap_fraction(C, Channel(C.center_hz + 20e6, 20e6)) == 0.
    assert channel_overlap_fraction(C, Channel(C.center_hz + 21e6, 20e6)) == 0.


def test_filter_integrated_measurement_is_not_filtered_twice():
    receiver, desired, _, _ = public_rf()
    desired = replace(desired, channel=Channel(C.center_hz, 40e6))
    before = RFCoupling('robot', desired.transmitter_id, C.center_hz, 1e-4)
    integrated = RFCoupling('robot', desired.transmitter_id, C.center_hz, 5e-5,
                           includes_receiver_filter=True, receiver_channel=C, transmitter_channel=desired.channel)
    a = budget(receiver, desired, couplings=[before])
    b = budget(receiver, desired, couplings=[integrated])
    assert a.desired.received_mw == b.desired.received_mw == pytest.approx(5e-4)
    assert b.desired.channel_overlap_fraction == .5
    assert b.desired.applied_spectral_fraction == 1.
    # Integrated gain is tied to BOTH measured channel widths.
    changed = budget(receiver, replace(desired, channel=Channel(C.center_hz, 80e6)), couplings=[integrated])
    assert changed.desired.received_mw is None and changed.sinr_linear is None


def test_measured_filter_gain_can_represent_adjacent_leakage_without_inventing_it():
    receiver, desired, _, _ = public_rf()
    desired = replace(desired, channel=Channel(C.center_hz + 100e6, 20e6))
    gain = RFCoupling('robot', desired.transmitter_id, desired.channel.center_hz, 1e-8,
                      includes_receiver_filter=True, receiver_channel=C, transmitter_channel=desired.channel)
    measured = budget(receiver, desired, couplings=[gain])
    assert measured.desired.channel_overlap_fraction == 0.
    assert measured.desired.received_mw == pytest.approx(1e-7)


def test_isolated_power_adapter_preserves_unknown_and_filter_provenance():
    gain = coupling_from_isolated_power_dbm('r', 't', C.center_hz, -60., 20.,
             receiver_channel=C, transmitter_channel=C)
    assert gain.gain_linear == pytest.approx(1e-8)
    assert gain.origin == 'measured' and gain.includes_receiver_filter
    missing = coupling_from_isolated_power_dbm('r', 't', C.center_hz, None, 20.,
             receiver_channel=C, transmitter_channel=C)
    assert missing.gain_linear is None
    masked = coupling_from_isolated_power_dbm('r', 't', np.nan, np.nan, np.nan, valid=False)
    assert not masked.valid and masked.gain_linear is None


def test_missing_or_hidden_interference_is_unknown_rather_than_zero():
    receiver, desired, interferers, couplings = public_rf()
    hidden = TransmitterState('unobserved/radio')
    result = budget(receiver, desired, [*interferers, hidden], couplings)
    assert result.known_interference_mw == pytest.approx(7e-5)
    assert result.total_interference_mw is None and result.sinr_linear is None
    assert result.unknown_interferers == ('unobserved/radio',)
    assert result.sinr_upper_bound_linear == pytest.approx(1e-3 / 71e-6)
    unlisted = receiver_link_budget(receiver, desired, (), couplings)
    assert unlisted.total_interference_mw is None and unlisted.sinr_linear is None
    declared_empty = budget(receiver, desired, (), couplings)
    assert declared_empty.total_interference_mw == 0. and declared_empty.sinr_linear == pytest.approx(1000.)


def test_known_zero_activity_differs_from_unknown_activity():
    receiver, desired, interferers, couplings = public_rf()
    quiet = replace(interferers[0], activity_fraction=0., tx_power_dbm=None)
    known = budget(receiver, desired, [quiet], couplings[:1])
    assert known.total_interference_mw == 0.
    unknown = budget(receiver, desired, [replace(quiet, activity_fraction=None)], couplings[:1])
    assert unknown.total_interference_mw is None and unknown.unknown_interferers == (quiet.transmitter_id,)


def test_masked_nan_placeholders_never_enter_rf_arithmetic():
    receiver, desired, _, couplings = public_rf()
    masked = TransmitterState('hidden', (np.nan, np.nan, np.nan), C, np.nan, np.nan, observed=False)
    edge = RFCoupling('robot', 'hidden', np.nan, np.nan, valid=False)
    result = budget(receiver, desired, [masked], [*couplings, edge])
    assert result.total_interference_mw is None and result.known_interference_mw == 0.
    unavailable = ReceiverState('robot', (np.nan, np.nan, np.nan), C, np.nan, observed=False)
    assert budget(unavailable, desired, couplings=couplings).sinr_linear is None


def test_missing_geometry_does_not_erase_measured_coupling_or_invent_new_coupling():
    receiver, desired, _, couplings = public_rf()
    receiver = replace(receiver, position_m=None)
    assert budget(receiver, desired, couplings=couplings).desired.received_mw == pytest.approx(1e-3)
    assert budget(receiver, desired).desired.received_mw is None


def test_radio_permutation_and_duplicate_edges_cannot_change_power():
    receiver, desired, interferers, couplings = public_rf()
    expected = budget(receiver, desired, interferers, couplings)
    rng = np.random.default_rng(486)
    for _ in range(12):
        tx = [interferers[i] for i in rng.permutation(len(interferers))]
        edges = [couplings[i] for i in rng.permutation(len(couplings))]
        assert budget(receiver, desired, [*tx, desired, tx[0]], [*edges, *edges]) == expected
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        budget(receiver, desired, [interferers[0], replace(interferers[0], tx_power_dbm=11.)], couplings)
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        budget(receiver, desired, interferers, [*couplings, replace(couplings[0], gain_linear=1e-3)])


def test_ambiguous_near_frequency_measurements_are_rejected_not_order_selected():
    receiver, desired, _, couplings = public_rf()
    ambiguous = replace(couplings[0], frequency_hz=C.center_hz + .5, gain_linear=1e-2)
    for edges in ([couplings[0], ambiguous], [ambiguous, couplings[0]]):
        with pytest.raises(ValueError, match='Ambiguous'):
            budget(receiver, desired, couplings=edges)


def transform(position):
    angle=.73; c,s=np.cos(angle),np.sin(angle)
    x,y,z=position
    return (c*x-s*y+31.,s*x+c*y-12.,z)


def test_joint_se2_transform_preserves_rf_geometry_and_calibrated_fresnel_clearance():
    receiver, desired, interferers, _ = public_rf()
    nodes = [desired,*interferers]
    edges = [free_space_coupling(receiver,t,assume_isotropic_far_field=True) for t in nodes]
    before=budget(receiver,desired,interferers,edges)
    moved_receiver=replace(receiver,position_m=transform(receiver.position_m))
    moved_nodes=[replace(t,position_m=transform(t.position_m)) for t in nodes]
    moved_edges=[free_space_coupling(moved_receiver,t,assume_isotropic_far_field=True) for t in moved_nodes]
    after=budget(moved_receiver,moved_nodes[0],moved_nodes[1:],moved_edges)
    assert before.sinr_linear == pytest.approx(after.sinr_linear,rel=1e-12)
    assert before.desired.distance_m == pytest.approx(after.desired.distance_m)
    silhouette=VisualSilhouette('crate',(5.,.4,1.5),.1,calibrated=True)
    a=fresnel_silhouette_clearance(desired.position_m,receiver.position_m,silhouette,C.center_hz)
    b=fresnel_silhouette_clearance(transform(desired.position_m),transform(receiver.position_m),
                                  replace(silhouette,center_m=transform(silhouette.center_m)),C.center_hz)
    assert a.clearance_m == pytest.approx(b.clearance_m,abs=1e-12)
    assert a.silhouette_intersects_fresnel == b.silhouette_intersects_fresnel


def test_known_signal_noise_activity_and_interference_have_monotonic_sinr_effects():
    receiver, desired, interferers, couplings = public_rf()
    reference=budget(receiver,desired,interferers,couplings).sinr_linear
    assert budget(receiver,replace(desired,tx_power_dbm=20.),interferers,couplings).sinr_linear > reference
    assert budget(replace(receiver,noise_mw=1e-3),desired,interferers,couplings).sinr_linear < reference
    assert budget(receiver,desired,[replace(interferers[0],tx_power_dbm=20.),interferers[1]],couplings).sinr_linear < reference
    assert budget(receiver,desired,[replace(interferers[0],activity_fraction=0.),interferers[1]],couplings).sinr_linear > reference
    stronger=[couplings[0],replace(couplings[1],gain_linear=1e-4),couplings[2]]
    assert budget(receiver,desired,interferers,stronger).sinr_linear < reference


def test_free_space_is_explicit_assumption_and_scales_with_distance_and_frequency():
    receiver, desired, _, _=public_rf()
    with pytest.raises(ValueError,match='explicit'):
        free_space_coupling(receiver,desired)
    a=free_space_coupling(receiver,desired,assume_isotropic_far_field=True)
    twice_distance=replace(desired,position_m=tuple(2*np.asarray(desired.position_m)-np.asarray(receiver.position_m)))
    b=free_space_coupling(receiver,twice_distance,assume_isotropic_far_field=True)
    c=free_space_coupling(receiver,replace(desired,channel=Channel(2*C.center_hz,C.bandwidth_hz)),assume_isotropic_far_field=True)
    assert a.origin=='declared_free_space'
    assert b.gain_linear==pytest.approx(a.gain_linear/4)
    assert c.gain_linear==pytest.approx(a.gain_linear/4)


def test_first_fresnel_formula_symmetry_frequency_and_endpoints():
    expected=math.sqrt((SPEED_OF_LIGHT_M_S/C.center_hz)*5*5/10)
    assert first_fresnel_radius_m(C.center_hz,5.,5.)==pytest.approx(expected)
    assert first_fresnel_radius_m(C.center_hz,3.,7.)==first_fresnel_radius_m(C.center_hz,7.,3.)
    assert first_fresnel_radius_m(2*C.center_hz,5.,5.)==pytest.approx(expected/math.sqrt(2))
    assert first_fresnel_radius_m(C.center_hz,0.,10.)==0.


def test_calibrated_silhouette_intersection_is_geometry_not_rf_attenuation():
    silhouette=VisualSilhouette('crate',(5.,.4,0.),.1,calibrated=True)
    a=fresnel_silhouette_clearance((0.,0.,0.),(10.,0.,0.),silhouette,C.center_hz)
    assert a.known and not a.silhouette_intersects_los and a.silhouette_intersects_fresnel
    b=fresnel_silhouette_clearance((0.,0.,0.),(10.,0.,0.),silhouette,C.center_hz,radius_fraction=.6)
    assert b.clearance_m>0 and not b.silhouette_intersects_fresnel
    hit=replace(silhouette,center_m=(5.,.05,0.))
    assert fresnel_silhouette_clearance((0.,0.,0.),(10.,0.,0.),hit,C.center_hz).silhouette_intersects_los
    outside=replace(silhouette,center_m=(-1.,0.,0.))
    assert not fresnel_silhouette_clearance((0.,0.,0.),(10.,0.,0.),outside,C.center_hz).silhouette_intersects_fresnel


def test_unobserved_uncalibrated_and_missing_geometry_remain_unknown():
    silhouette=VisualSilhouette('pixel-mask',(5.,.4,0.),.1)
    assert not fresnel_silhouette_clearance((0.,0.,0.),(10.,0.,0.),silhouette,C.center_hz).known
    calibrated=replace(silhouette,calibrated=True)
    assert not fresnel_silhouette_clearance(None,(10.,0.,0.),calibrated,C.center_hz).known
    masked=VisualSilhouette('masked',(np.nan,np.nan,np.nan),np.nan,observed=False)
    assert not fresnel_silhouette_clearance((0.,0.,0.),(10.,0.,0.),masked,C.center_hz).known


@pytest.mark.parametrize('call', [
    lambda: Channel(0.,20e6), lambda: Channel(5e9,-1.),
    lambda: ReceiverState('r',noise_mw=-1.),
    lambda: TransmitterState('t',activity_fraction=1.1),
    lambda: TransmitterState('t',tx_power_dbm=np.nan),
    lambda: RFCoupling('r','t',5e9,-1.),
    lambda: RFCoupling('r','t',5e9,1.,includes_receiver_filter=True),
    lambda: first_fresnel_radius_m(0.,5.,5.),
    lambda: first_fresnel_radius_m(5e9,-1.,5.),
    lambda: first_fresnel_radius_m(5e9,0.,0.),
    lambda: fresnel_silhouette_clearance((0.,0.,0.),(10.,0.,0.),
                VisualSilhouette('x',(-1.,0.,0.),.1,calibrated=True),-1.),
])
def test_invalid_physical_units_or_provenance_are_rejected(call):
    with pytest.raises((TypeError,ValueError)):
        call()
