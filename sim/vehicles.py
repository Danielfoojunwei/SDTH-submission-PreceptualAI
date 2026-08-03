"""UxS types, their bearers, and the conditions that degrade them.

Four platform classes, because the interesting differences between them are not
cosmetic — they change which bearers exist, how much headroom the flow table can
ask for, and which conditions bite. A multirotor carries a short-range mesh and
dies to a rain cell on its Ka backhaul; a maritime surface vessel has satcom and
does not care about terrain masking; a ground vehicle loses line of sight behind
a ridge that an aircraft flies over.

Frequency is carried per bearer purely so the weather model can attenuate the
right ones. It is **not** spectrum sensing: nothing here observes a band, and the
node never sees a frequency — it sees the degraded throughput and loss that the
condition produced, which is exactly what a real vehicle experiences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

#: Coarse, textbook attenuation used only to make conditions differ *between*
#: bearers in a defensible direction. Rain hurts high frequencies far more than
#: low ones; that ordering is real and is all this needs to be right about.
#: The absolute values are not calibrated and no result here is evidence about
#: real fade depth.
_RAIN_REFERENCE_GHZ = 20.0


@dataclass(frozen=True)
class Bearer:
    """One radio a platform carries."""

    bearer_id: str
    interface: str
    link_role: str
    centre_ghz: float
    #: Clear-air capacity, megabits per second, before any condition applies.
    clear_air_mbps: float
    base_rtt_ms: float
    #: How much of this bearer's path is line-of-sight-dependent, in [0, 1].
    #: A satcom bearer is near zero — a ridge does not block a satellite at
    #: high elevation the way it blocks a mesh peer across a valley.
    terrain_sensitivity: float = 1.0

    def rain_attenuation_db(self, rain_mm_per_hour: float) -> float:
        """Attenuation for this bearer under a given rain rate.

        Deliberately crude: a power law in frequency, scaled by rain rate. It
        reproduces the one property the simulation needs — that a 20 GHz link
        fades far harder than a 1 GHz link in the same cell — and nothing else.
        Not ITU-R P.838, not calibrated, not evidence.
        """
        if rain_mm_per_hour <= 0.0:
            return 0.0
        frequency_factor = (self.centre_ghz / _RAIN_REFERENCE_GHZ) ** 2.0
        return 12.0 * frequency_factor * (rain_mm_per_hour / 25.0) ** 0.6


@dataclass(frozen=True)
class Condition:
    """What the environment is doing to a platform this cycle."""

    name: str
    rain_mm_per_hour: float = 0.0
    #: Fraction of line-of-sight blocked, in [0, 1].
    terrain_blockage: float = 0.0
    #: Jammer power expressed as an effective noise rise, dB. Applied to every
    #: bearer inside the jammed span.
    jammer_db: float = 0.0
    jammer_low_ghz: float = 0.0
    jammer_high_ghz: float = 0.0

    def jams(self, bearer: Bearer) -> bool:
        return (
            self.jammer_db > 0.0
            and self.jammer_low_ghz <= bearer.centre_ghz <= self.jammer_high_ghz
        )

    def degradation_db(self, bearer: Bearer) -> float:
        """Total link margin lost by this bearer under this condition."""
        loss = bearer.rain_attenuation_db(self.rain_mm_per_hour)
        loss += 30.0 * self.terrain_blockage * bearer.terrain_sensitivity
        if self.jams(bearer):
            loss += self.jammer_db
        return loss

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rain_mm_per_hour": self.rain_mm_per_hour,
            "terrain_blockage": self.terrain_blockage,
            "jammer_db": self.jammer_db,
            "jammer_span_ghz": [self.jammer_low_ghz, self.jammer_high_ghz],
        }


@dataclass(frozen=True)
class VehicleType:
    """A platform class: what it carries and what it must carry."""

    type_id: str
    description: str
    bearers: tuple[Bearer, ...]
    #: Flows this class must carry, as (flow_id, role, kbps, minimum_slices).
    demands: tuple[tuple[str, str, float, int], ...] = field(default=())

    def bearer_ids(self) -> tuple[str, ...]:
        return tuple(bearer.bearer_id for bearer in self.bearers)


def _c2(bearer_id: str, iface: str, ghz: float, mbps: float, rtt: float, terrain: float = 1.0):
    return Bearer(bearer_id, iface, "c2_cnpc", ghz, mbps, rtt, terrain)


def _payload(bearer_id: str, iface: str, ghz: float, mbps: float, rtt: float, terrain: float = 1.0):
    return Bearer(bearer_id, iface, "payload", ghz, mbps, rtt, terrain)


#: Four platform classes. The differences that matter are bearer count, the
#: frequency spread (which decides what weather does to them), and terrain
#: sensitivity (which decides whether masking matters at all).
VEHICLE_TYPES: tuple[VehicleType, ...] = (
    VehicleType(
        type_id="multirotor",
        description="Short-endurance rotary wing: dense low-frequency mesh, one Ka backhaul.",
        bearers=(
            _c2("mr-c2-uhf", "wlan0", 0.9, 1.2, 45.0),
            _c2("mr-c2-mesh", "wlan1", 2.4, 4.0, 60.0),
            _c2("mr-c2-lte", "wwan0", 1.8, 8.0, 70.0),
            _payload("mr-pay-ka", "wwan1", 20.0, 40.0, 120.0),
            _payload("mr-pay-mesh", "wlan2", 5.8, 25.0, 55.0),
        ),
        demands=(("c2-primary", "c2_cnpc", 96.0, 3), ("video", "payload", 4000.0, 2)),
    ),
    VehicleType(
        type_id="fixed_wing",
        description="Long-endurance fixed wing: satcom BLOS plus terrestrial CNPC.",
        bearers=(
            _c2("fw-c2-cnpc", "wwan0", 5.06, 3.0, 55.0),
            _c2("fw-c2-l", "wwan1", 1.6, 1.0, 90.0, terrain=0.2),
            _c2("fw-c2-uhf", "wlan0", 0.4, 0.5, 65.0),
            _payload("fw-pay-ku", "wwan2", 14.0, 60.0, 180.0, terrain=0.1),
            _payload("fw-pay-ka", "wwan3", 20.0, 90.0, 175.0, terrain=0.1),
        ),
        demands=(("c2-primary", "c2_cnpc", 128.0, 3), ("isr-downlink", "payload", 12000.0, 2)),
    ),
    VehicleType(
        type_id="ground",
        description="Ground vehicle: terrain-masked, low frequencies, no satcom.",
        bearers=(
            _c2("gv-c2-vhf", "wlan0", 0.15, 0.3, 80.0),
            _c2("gv-c2-uhf", "wlan1", 0.45, 0.8, 70.0),
            _c2("gv-c2-lte", "wwan0", 1.8, 6.0, 85.0),
            _payload("gv-pay-mesh", "wlan2", 5.8, 20.0, 60.0),
            _payload("gv-pay-lte", "wwan1", 2.6, 15.0, 75.0),
        ),
        demands=(("c2-primary", "c2_cnpc", 64.0, 3), ("sensor-feed", "payload", 2000.0, 2)),
    ),
    VehicleType(
        type_id="surface",
        description="Maritime surface vessel: satcom-heavy, terrain irrelevant, rain matters.",
        bearers=(
            _c2("sv-c2-l", "wwan0", 1.6, 1.0, 110.0, terrain=0.05),
            _c2("sv-c2-uhf", "wlan0", 0.3, 0.4, 75.0, terrain=0.3),
            _c2("sv-c2-hf", "wlan1", 0.01, 0.05, 400.0, terrain=0.05),
            _payload("sv-pay-ka", "wwan1", 20.0, 70.0, 190.0, terrain=0.05),
            _payload("sv-pay-ku", "wwan2", 12.0, 45.0, 185.0, terrain=0.05),
        ),
        demands=(("c2-primary", "c2_cnpc", 64.0, 3), ("ais-and-video", "payload", 6000.0, 2)),
    ),
)


#: Named environments. Each is chosen to make a different platform class fail,
#: so a run exercises more than one decision path.
CONDITIONS: tuple[Condition, ...] = (
    Condition("clear_air"),
    Condition("light_rain", rain_mm_per_hour=4.0),
    Condition("heavy_rain", rain_mm_per_hour=45.0),
    Condition("terrain_masked", terrain_blockage=0.7),
    Condition(
        "ism_jammed",
        jammer_db=28.0,
        jammer_low_ghz=2.0,
        jammer_high_ghz=6.0,
    ),
    Condition(
        "wideband_jammed",
        jammer_db=35.0,
        jammer_low_ghz=0.1,
        jammer_high_ghz=25.0,
    ),
    Condition(
        "storm_and_jamming",
        rain_mm_per_hour=30.0,
        terrain_blockage=0.3,
        jammer_db=22.0,
        jammer_low_ghz=1.0,
        jammer_high_ghz=8.0,
    ),
)


def degraded_link(bearer: Bearer, condition: Condition) -> dict[str, float | None]:
    """Measurements a platform would report for this bearer under this condition.

    Margin lost becomes throughput lost, latency gained and loss gained. Past a
    total loss the link is reported down rather than merely poor, because a link
    that cannot pass a probe is not a slow link — and the node must be able to
    tell those apart.
    """
    lost_db = condition.degradation_db(bearer)
    # Shannon-ish: every 3 dB of margin roughly halves what the link carries.
    scale = 0.5 ** (lost_db / 3.0)
    throughput = bearer.clear_air_mbps * scale
    if lost_db >= 24.0 or throughput < bearer.clear_air_mbps * 0.01:
        return {"up": 0.0, "rtt_ms": None, "loss_percent": None, "jitter_ms": None,
                "rx_mbps": None, "tx_mbps": None, "signal_percent": 0.0}
    loss_percent = min(95.0, 100.0 * (1.0 - scale) ** 2)
    rtt = bearer.base_rtt_ms * (1.0 + lost_db / 6.0)
    return {
        "up": 1.0,
        "rtt_ms": round(rtt, 3),
        "loss_percent": round(loss_percent, 3),
        "jitter_ms": round(rtt * 0.12, 3),
        "rx_mbps": round(throughput * 0.6, 4),
        "tx_mbps": round(throughput * 0.4, 4),
        "signal_percent": round(max(0.0, 100.0 * scale), 3),
    }


def link_budget_margin_db(bearer: Bearer, condition: Condition) -> float:
    """How much margin this bearer has left. Negative means the link is gone."""
    return 24.0 - condition.degradation_db(bearer)


def describe(value: float) -> str:
    return "unusable" if not math.isfinite(value) or value <= 0.0 else f"{value:.1f} dB"
