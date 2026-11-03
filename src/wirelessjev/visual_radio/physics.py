"""Public receiver-centric RF factors and separately calibrated geometry.

These are established link-budget relationships and explicit approximations,
not learned performance or a novelty claim. Missing RF evidence remains None.
Visual silhouettes never supply RF material, coupling or transmitter activity.
"""
from dataclasses import dataclass
import math

import numpy as np

from ..service_world.physics import dbm_to_mw, power_ratio_to_db, sinr_linear as _sinr

SPEED_OF_LIGHT_M_S = 299_792_458.


def _number(value, name, *, minimum=None, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f'{name} must be a real scalar in the declared units')
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum) or (positive and value <= 0):
        raise ValueError(f'Invalid {name}')
    return value


def _position(value):
    if value is None:
        return None
    if len(value) != 3:
        raise ValueError('Metric world positions require three coordinates')
    return tuple(_number(v, 'position metres') for v in value)


def _identity(value):
    if not isinstance(value, str) or not value:
        raise ValueError('A nonempty physical radio/receiver identity is required')


def _mask(value):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError('Availability masks must be boolean')


@dataclass(frozen=True)
class Channel:
    center_hz: float
    bandwidth_hz: float

    def __post_init__(self):
        _number(self.center_hz, 'centre frequency Hz', positive=True)
        _number(self.bandwidth_hz, 'bandwidth Hz', positive=True)
        if self.bandwidth_hz >= 2 * self.center_hz:
            raise ValueError('A passband cannot extend to nonpositive frequency')


@dataclass(frozen=True)
class ReceiverState:
    receiver_id: str
    position_m: tuple | None = None
    channel: Channel | None = None
    noise_mw: float | None = None
    observed: bool = True

    def __post_init__(self):
        _identity(self.receiver_id); _mask(self.observed)
        if not self.observed:
            for key in ('position_m', 'channel', 'noise_mw'):
                object.__setattr__(self, key, None)
            return
        object.__setattr__(self, 'position_m', _position(self.position_m))
        if self.channel is not None and not isinstance(self.channel, Channel):
            raise TypeError('Receiver channel must be a Channel or None')
        if self.noise_mw is not None:
            _number(self.noise_mw, 'receiver noise mW', positive=True)


@dataclass(frozen=True)
class TransmitterState:
    """One physical radio, rather than an entire multi-radio AP.

    Activity is a measured/configured fraction in [0,1] over the input window.
    None means unknown. Binary fractions describe instantaneous availability;
    fractional activity gives a mean-power approximation.
    """
    transmitter_id: str
    position_m: tuple | None = None
    channel: Channel | None = None
    tx_power_dbm: float | None = None
    activity_fraction: float | None = None
    observed: bool = True

    def __post_init__(self):
        _identity(self.transmitter_id); _mask(self.observed)
        if not self.observed:
            for key in ('position_m', 'channel', 'tx_power_dbm', 'activity_fraction'):
                object.__setattr__(self, key, None)
            return
        object.__setattr__(self, 'position_m', _position(self.position_m))
        if self.channel is not None and not isinstance(self.channel, Channel):
            raise TypeError('Transmitter channel must be a Channel or None')
        if self.tx_power_dbm is not None:
            _number(self.tx_power_dbm, 'transmit power dBm')
        if self.activity_fraction is not None:
            activity = _number(self.activity_fraction, 'activity fraction', minimum=0)
            if activity > 1:
                raise ValueError('Activity fraction cannot exceed one')


@dataclass(frozen=True)
class RFCoupling:
    """Linear received/transmitted POWER ratio, at a declared RF frequency.

    Default gain precedes receiver filtering. A measured filter-integrated gain
    sets includes_receiver_filter=True and identifies both measured channels;
    overlap must not be applied twice. A free-space reference has a distinct
    origin and must never be described as a measurement.
    """
    receiver_id: str
    transmitter_id: str
    frequency_hz: float | None
    gain_linear: float | None
    valid: bool = True
    origin: str = 'measured'
    includes_receiver_filter: bool = False
    receiver_channel: Channel | None = None
    transmitter_channel: Channel | None = None

    def __post_init__(self):
        _identity(self.receiver_id); _identity(self.transmitter_id); _mask(self.valid)
        _mask(self.includes_receiver_filter)
        if self.origin not in ('measured', 'declared_free_space'):
            raise ValueError('Coupling origin must distinguish measurement and free-space assumption')
        if not self.valid:
            object.__setattr__(self, 'frequency_hz', None)
            object.__setattr__(self, 'gain_linear', None)
            object.__setattr__(self, 'receiver_channel', None)
            object.__setattr__(self, 'transmitter_channel', None)
            return
        if self.frequency_hz is not None:
            _number(self.frequency_hz, 'coupling frequency Hz', positive=True)
        if self.gain_linear is not None:
            _number(self.gain_linear, 'linear RF power coupling', minimum=0)
        if any(channel is not None and not isinstance(channel, Channel)
               for channel in (self.receiver_channel, self.transmitter_channel)):
            raise TypeError('Coupling channel provenance must be typed Channels or None')
        if self.includes_receiver_filter and (self.receiver_channel is None or self.transmitter_channel is None):
            raise ValueError('Filter-integrated coupling requires both measured transmitter and receiver channels')


@dataclass(frozen=True)
class PowerTerm:
    transmitter_id: str
    received_mw: float | None
    channel_overlap_fraction: float | None
    applied_spectral_fraction: float | None
    distance_m: float | None
    coupling_origin: str | None
    unknown_fields: tuple = ()


@dataclass(frozen=True)
class LinkBudget:
    desired: PowerTerm
    interferers: tuple
    noise_mw: float | None
    known_interference_mw: float
    total_interference_mw: float | None
    sinr_linear: float | None
    sinr_db: float | None
    sinr_upper_bound_linear: float | None
    unknown_interferers: tuple
    inventory_complete: bool
    power_model: str = 'desired transmission conditioned; mean interferer power; rectangular filters unless measured'


def channel_overlap_fraction(receiver_channel, transmitter_channel):
    """Flat transmitter PSD fraction admitted by an ideal receiver filter.

    Denominator is TRANSMITTER bandwidth. Actual spectral masks and receiver
    leakage are excluded unless supplied through filter-integrated measurements.
    """
    if not isinstance(receiver_channel, Channel) or not isinstance(transmitter_channel, Channel):
        raise TypeError('Channel overlap needs explicit receiver and transmitter channels')
    rx, tx = receiver_channel, transmitter_channel
    overlap = max(0., min(rx.center_hz + rx.bandwidth_hz / 2, tx.center_hz + tx.bandwidth_hz / 2)
                  - max(rx.center_hz - rx.bandwidth_hz / 2, tx.center_hz - tx.bandwidth_hz / 2))
    return min(1., overlap / tx.bandwidth_hz)


def coupling_from_isolated_power_dbm(receiver_id, transmitter_id, frequency_hz,
                                     received_dbm, tx_power_dbm, *, receiver_channel=None,
                                     transmitter_channel=None, includes_receiver_filter=True, valid=True):
    """Adapt an isolated calibrated RF power measurement, never visual appearance.

    Aggregate channel energy is not an isolated link measurement. The caller
    aligns transmitter power, filters and measurement time. Missing power stays
    unknown; masked placeholders never enter a physical coupling calculation.
    """
    _mask(valid)
    gain = None
    if valid and received_dbm is not None and tx_power_dbm is not None:
        rx = _number(received_dbm, 'isolated received power dBm')
        tx = _number(tx_power_dbm, 'matching transmit power dBm')
        gain = dbm_to_mw(rx - tx)
    return RFCoupling(receiver_id, transmitter_id, frequency_hz, gain, valid,
                      includes_receiver_filter=includes_receiver_filter, receiver_channel=receiver_channel,
                      transmitter_channel=transmitter_channel)


def _unique(rows, key):
    result = {}
    for row in rows:
        identity = key(row)
        if identity in result and result[identity] != row:
            raise ValueError(f'Conflicting duplicate public evidence: {identity}')
        result[identity] = row
    return result


def _power_term(receiver, transmitter, couplings, *, desired=False):
    distance = None
    if receiver.position_m is not None and transmitter.position_m is not None:
        distance = _number(math.dist(receiver.position_m, transmitter.position_m), 'link distance metres', minimum=0)
    overlap = None if receiver.channel is None or transmitter.channel is None else channel_overlap_fraction(receiver.channel, transmitter.channel)
    # Desired signal is conditional on a packet, so its population duty cycle
    # cannot attenuate that packet's power.
    activity = 1. if desired else transmitter.activity_fraction
    if activity == 0:
        return PowerTerm(transmitter.transmitter_id, 0., overlap, None, distance, None)
    matches = [row for row in couplings if row.valid and row.receiver_id == receiver.receiver_id
               and row.transmitter_id == transmitter.transmitter_id and row.frequency_hz is not None
               and transmitter.channel is not None
               and math.isclose(row.frequency_hz, transmitter.channel.center_hz, rel_tol=1e-9, abs_tol=1.)]
    filtered = [row for row in matches if row.includes_receiver_filter and row.receiver_channel == receiver.channel
                and row.transmitter_channel == transmitter.channel]
    unfiltered = [row for row in matches if not row.includes_receiver_filter]
    if len(filtered) > 1 or (not filtered and len(unfiltered) > 1):
        raise ValueError('Ambiguous matching RF coupling measurements; resolve their time/frequency provenance')
    coupling = (filtered or unfiltered or [None])[0]
    spectral = 1. if filtered else overlap
    if not filtered and overlap == 0:
        return PowerTerm(transmitter.transmitter_id, 0., overlap, 0., distance, 'ideal_rectangular_filter')
    missing = []
    if transmitter.tx_power_dbm is None: missing.append('transmit_power')
    if activity is None: missing.append('activity')
    if spectral is None: missing.append('channel_overlap')
    if coupling is None or coupling.gain_linear is None: missing.append('rf_coupling')
    if missing:
        return PowerTerm(transmitter.transmitter_id, None, overlap, spectral, distance,
                         None if coupling is None else coupling.origin, tuple(missing))
    power = dbm_to_mw(transmitter.tx_power_dbm) * coupling.gain_linear * activity * spectral
    _number(power, 'received power mW', minimum=0)
    return PowerTerm(transmitter.transmitter_id, power, overlap, spectral, distance, coupling.origin)


def receiver_link_budget(receiver, desired, interferers=(), couplings=(), *, inventory_complete=False):
    """S/(N+sum I_j), preserving incompleteness of public RF evidence.

    Desired signal is conditional on an actual/hypothetical desired packet.
    Fractional activity yields S/(N+E[I]), not E[SINR] or delivery probability.
    inventory_complete=False keeps unlisted interferers unknown. The SINR
    upper bound assumes known desired power/noise are correct and omitted
    interference is nonnegative under the declared power model.
    """
    if not isinstance(receiver, ReceiverState) or not isinstance(desired, TransmitterState):
        raise TypeError('Link budget needs typed public receiver and transmitter states')
    _mask(inventory_complete)
    rows = list(interferers)
    if not all(isinstance(row, TransmitterState) for row in rows):
        raise TypeError('Interferers must be typed physical radio states')
    radios = _unique([desired, *rows], lambda row: row.transmitter_id)
    edges = list(couplings)
    if not all(isinstance(row, RFCoupling) for row in edges):
        raise TypeError('Couplings must carry RF units and provenance')
    edges = list(_unique(edges, lambda row: (row.receiver_id, row.transmitter_id, row.frequency_hz,
                       row.includes_receiver_filter, row.receiver_channel, row.transmitter_channel)).values())
    signal = _power_term(receiver, desired, edges, desired=True)
    terms = tuple(_power_term(receiver, row, edges) for key, row in sorted(radios.items()) if key != desired.transmitter_id)
    unknown = tuple(term.transmitter_id for term in terms if term.received_mw is None)
    known = math.fsum(term.received_mw for term in terms if term.received_mw is not None)
    total = known if inventory_complete and not unknown else None
    upper = None
    if signal.received_mw is not None and receiver.noise_mw is not None:
        upper = _sinr(signal.received_mw, receiver.noise_mw, known)
    point = upper if total is not None else None
    return LinkBudget(signal, terms, receiver.noise_mw, known, total, point,
                      None if point is None else power_ratio_to_db(point), upper, unknown, bool(inventory_complete))


def free_space_coupling(receiver, transmitter, *, assume_isotropic_far_field=False):
    """Explicit Friis reference; never silently replace missing RF evidence.

    Caller declares unobstructed isotropic far-field propagation. Antenna
    aperture/far-field validity, multipath and materials are not inferred here.
    """
    if assume_isotropic_far_field is not True:
        raise ValueError('A free-space reference requires an explicit isotropic far-field assumption')
    if receiver.position_m is None or transmitter.position_m is None or transmitter.channel is None:
        raise ValueError('Free-space reference needs calibrated metric endpoints and RF frequency')
    distance = _number(math.dist(receiver.position_m, transmitter.position_m), 'link distance metres', positive=True)
    wavelength = SPEED_OF_LIGHT_M_S / transmitter.channel.center_hz
    gain = (wavelength / (4 * math.pi * distance)) ** 2
    if gain > 1:
        raise ValueError('Declared distance is incompatible with the free-space far-field reference')
    return RFCoupling(receiver.receiver_id, transmitter.transmitter_id, transmitter.channel.center_hz,
                      gain, origin='declared_free_space')


@dataclass(frozen=True)
class VisualSilhouette:
    """Metric circular silhouette cross-section, not appearance-derived RF data."""
    object_id: str
    center_m: tuple | None = None
    radius_m: float | None = None
    calibrated: bool = False
    observed: bool = True

    def __post_init__(self):
        _identity(self.object_id); _mask(self.observed); _mask(self.calibrated)
        if not self.observed:
            object.__setattr__(self, 'center_m', None); object.__setattr__(self, 'radius_m', None)
            return
        object.__setattr__(self, 'center_m', _position(self.center_m))
        if self.radius_m is not None:
            _number(self.radius_m, 'silhouette radius metres', minimum=0)


@dataclass(frozen=True)
class FresnelClearance:
    known: bool
    silhouette_intersects_los: bool | None = None
    silhouette_intersects_fresnel: bool | None = None
    clearance_m: float | None = None
    first_fresnel_radius_m: float | None = None
    along_fraction: float | None = None
    approximation: str = 'circular silhouette in the plane normal to LOS at its projected centre'


def first_fresnel_radius_m(frequency_hz, d1_m, d2_m):
    frequency = _number(frequency_hz, 'frequency Hz', positive=True)
    d1 = _number(d1_m, 'first axial distance metres', minimum=0)
    d2 = _number(d2_m, 'second axial distance metres', minimum=0)
    if d1 + d2 <= 0:
        raise ValueError('Fresnel endpoints must be distinct')
    return math.sqrt((SPEED_OF_LIGHT_M_S / frequency) * d1 * d2 / (d1 + d2))


def fresnel_silhouette_clearance(tx_position_m, rx_position_m, silhouette, frequency_hz, *, radius_fraction=1.):
    """Screen approximation only; no attenuation or material prediction.

    Pixel masks without metric calibration produce an unknown result. The
    circular cross-section is compared at its projected centre, rather than an
    exact intersection of a 3-D obstacle volume with a Fresnel ellipsoid.
    """
    if not isinstance(silhouette, VisualSilhouette):
        raise TypeError('A typed calibrated silhouette is required')
    fraction = _number(radius_fraction, 'Fresnel radius fraction', minimum=0)
    if fraction > 1:
        raise ValueError('Use a Fresnel radius fraction in [0,1]')
    tx, rx = _position(tx_position_m), _position(rx_position_m)
    if (tx is None or rx is None or not silhouette.observed or not silhouette.calibrated
            or silhouette.center_m is None or silhouette.radius_m is None or frequency_hz is None):
        return FresnelClearance(False)
    _number(frequency_hz, 'frequency Hz', positive=True)
    direction = np.asarray(rx) - tx
    length = _number(np.linalg.norm(direction), 'LOS length metres', positive=True)
    along = float(np.dot(np.asarray(silhouette.center_m) - tx, direction) / length ** 2)
    if not 0 <= along <= 1:
        return FresnelClearance(True, False, False, None, None, along)
    radial = float(np.linalg.norm(np.asarray(silhouette.center_m) - (np.asarray(tx) + along * direction)))
    radius = first_fresnel_radius_m(frequency_hz, along * length, (1 - along) * length)
    clearance = radial - silhouette.radius_m - fraction * radius
    return FresnelClearance(True, radial <= silhouette.radius_m, clearance <= 0, clearance, radius, along)
