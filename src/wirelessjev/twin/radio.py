"""Digital twin of a 10-car Raspbot fleet on customer Wi-Fi.

Hardware is the SmartCar hardware (see wirelessjev.hardware): each car's only
fleet-controllable radio is its BE200 on wlan1, and its only network action is
selecting a BSSID. Infrastructure APs belong to the customer, so their
channels are fixed inputs (channel planning is modelled but disabled unless
an infrastructure adapter grants it).

Radio model (assumptions are explicit; measured anchors are in hardware.py):
- Each physical AP site exposes a 5 GHz BSS (VHT 80 MHz, 2SS) and a 2.4 GHz
  BSS (HT 20 MHz, 2SS) under the configured SSIDs.
- Path loss: log-distance with FSPL at 1 m (46.7 dB @ 5.2 GHz, 40.0 dB @
  2.4 GHz), exponent 3.3 indoor, plus a smooth deterministic shadowing field
  (std ~4 dB) per BSS.
- Fleet traffic is mostly uplink (video, maps, telemetry), so the rate uses
  the uplink SNR: AP-received power from the car's BE200.
- Airtime: BSSs on the same channel that hear each other share one contention
  domain. External occupancy (neighbouring networks) on a channel plays the
  role of the primary users in Li/Shen/Lam's DSA papers.
- 802.11e-style priority inside a domain: critical flows first, best-effort
  shares the rest proportionally.
- Handover: a BSSID change costs the measured ~4 s connect latency inside a
  5 s decision interval.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math

import numpy as np

from ..hardware import RASPBOT

CS_THRESHOLD_DBM = -82.0
MAC_EFFICIENCY = 0.6
HANDOVER_OUTAGE_S = RASPBOT.connect_latency_s
DECISION_INTERVAL_S = RASPBOT.policy_interval_s
# A BSSID choice is held for several policy intervals; benefits and the
# one-off handover outage are both scored over this commitment horizon.
DECISION_HORIZON_S = 6 * DECISION_INTERVAL_S

BANDS = {
    # band: (FSPL at 1 m, channel width Hz, rate table [(min SNR dB, Mbps)])
    "5g": (46.7, 80e6, (
        (5.0, 58.5), (8.0, 117.0), (11.0, 175.5), (14.0, 234.0),
        (18.0, 351.0), (22.0, 468.0), (24.0, 526.5), (26.0, 585.0),
        (30.0, 702.0), (33.0, 780.0))),
    "2g": (40.0, 20e6, (
        (2.0, 13.0), (5.0, 26.0), (9.0, 39.0), (11.0, 52.0),
        (15.0, 78.0), (18.0, 104.0), (20.0, 117.0), (25.0, 130.0))),
}
CHANNELS = {"5g": (36, 52, 100, 116, 132, 149), "2g": (1, 6, 11)}

# Neighbour-network occupancy presets, mirroring the PU-held/total channel
# splits used in the papers' setups (20/14 congested, 10/5 light).
CHANNEL_PRESETS = {
    "light_10_5": dict(busy_fraction=0.5, occupancy=(0.2, 0.6)),
    "congested_20_14": dict(busy_fraction=0.7, occupancy=(0.4, 0.85)),
    "open": dict(busy_fraction=0.25, occupancy=(0.1, 0.4)),
}


def noise_dbm(band: str) -> float:
    return -174 + 10 * math.log10(BANDS[band][1]) + 7.0


def rate_table(band: str):
    return BANDS[band][2]


def phy_rate_mbps(snr_db: float, band: str = "5g") -> float:
    rate = 0.0
    for threshold, mbps in rate_table(band):
        if snr_db >= threshold:
            rate = mbps
    return rate


@dataclass(frozen=True)
class Flow:
    name: str
    demand_mbps: float
    critical: bool
    floor_mbps: float  # contractual minimum when critical


@dataclass(frozen=True)
class Car:
    car_id: str
    x: float
    y: float
    vx: float
    vy: float
    flows: tuple[Flow, ...]
    current_ap: str | None
    tx_power_dbm: float = RASPBOT.client_tx_power_dbm
    tx_power_cap_dbm: float = 20.0

    @property
    def demand_mbps(self) -> float:
        return sum(f.demand_mbps for f in self.flows)


@dataclass(frozen=True)
class AP:
    """One BSS (a BSSID). Physical sites expose one BSS per band."""
    ap_id: str
    x: float
    y: float
    channel: int
    band: str = "5g"
    site: str = ""
    ssid: str = "fleet"
    priority: int = 100
    tx_power_dbm: float = 20.0
    allowed_channels: tuple[int, ...] = ()


@dataclass
class Scene:
    seed: int
    cars: list[Car]
    aps: list[AP]
    external_occupancy: dict[tuple[str, int], float]  # (band, ch) -> airtime
    width_m: float = 40.0
    height_m: float = 25.0
    region: str = "R0"
    _shadow: dict = field(default_factory=dict, repr=False)

    # --- propagation -----------------------------------------------------
    def _shadowing_db(self, ap: AP, x: float, y: float) -> float:
        key = ap.ap_id
        if key not in self._shadow:
            digest = hashlib.sha256(f"{self.seed}:{key}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            self._shadow[key] = (rng.uniform(0.08, 0.35, size=(4, 2)),
                                 rng.uniform(0, 2 * math.pi, size=4))
        freqs, phases = self._shadow[key]
        value = sum(math.sin(fx * x + fy * y + p)
                    for (fx, fy), p in zip(freqs, phases))
        return 4.0 * value / math.sqrt(2.0)

    def _path_loss(self, ap: AP, x: float, y: float) -> float:
        d = max(1.0, math.hypot(ap.x - x, ap.y - y))
        return BANDS[ap.band][0] + 33.0 * math.log10(d) + self._shadowing_db(ap, x, y)

    def rssi_dbm(self, ap: AP, x: float, y: float) -> float:
        """Downlink RSSI at the car (what `iw scan` reports)."""
        return ap.tx_power_dbm - self._path_loss(ap, x, y)

    def uplink_rssi_dbm(self, car: Car, ap: AP) -> float:
        return car.tx_power_dbm - self._path_loss(ap, car.x, car.y)

    def ap(self, ap_id: str) -> AP:
        for ap in self.aps:
            if ap.ap_id == ap_id:
                return ap
        raise KeyError(ap_id)

    def car(self, car_id: str) -> Car:
        for car in self.cars:
            if car.car_id == car_id:
                return car
        raise KeyError(car_id)

    def occupancy(self, ap: AP, channels: dict[str, int]) -> float:
        return self.external_occupancy.get((ap.band, channels[ap.ap_id]), 0.0)

    # --- contention domains ------------------------------------------------
    def domains(self, channels: dict[str, int]) -> list[set[str]]:
        ids = [ap.ap_id for ap in self.aps]
        parent = {i: i for i in ids}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a in self.aps:
            for b in self.aps:
                if (a.ap_id < b.ap_id and a.band == b.band
                        and channels[a.ap_id] == channels[b.ap_id]
                        and self.rssi_dbm(a, b.x, b.y) >= CS_THRESHOLD_DBM):
                    parent[find(a.ap_id)] = find(b.ap_id)
        groups: dict[str, set[str]] = {}
        for i in ids:
            groups.setdefault(find(i), set()).add(i)
        return list(groups.values())

    def sinr_db(self, car: Car, ap_id: str, channels: dict[str, int],
                activity: dict[str, float],
                domains: list[set[str]] | None = None) -> float:
        """Uplink SINR at the AP. Co-channel BSSs outside the AP's domain
        interfere in proportion to their airtime activity."""
        ap = self.ap(ap_id)
        signal_mw = 10 ** (self.uplink_rssi_dbm(car, ap) / 10)
        noise_mw = 10 ** (noise_dbm(ap.band) / 10)
        domains = domains or self.domains(channels)
        same = next(d for d in domains if ap_id in d)
        interference_mw = 0.0
        for other in self.aps:
            if (other.ap_id in same or other.band != ap.band
                    or channels[other.ap_id] != channels[ap_id]):
                continue
            interference_mw += activity.get(other.ap_id, 0.0) * 10 ** (
                self.rssi_dbm(other, ap.x, ap.y) / 10)
        return 10 * math.log10(signal_mw / (noise_mw + interference_mw))

    def visible_aps(self, car: Car) -> list[tuple[AP, float]]:
        """What SmartCatScan would report: at most 8 BSSs above -90 dBm,
        strongest first."""
        seen = [(ap, self.rssi_dbm(ap, car.x, car.y)) for ap in self.aps]
        seen = [s for s in seen if s[1] >= -90.0]
        seen.sort(key=lambda s: -s[1])
        return seen[:RASPBOT.max_visible_aps]

    # --- evolution -----------------------------------------------------------
    def _copy(self, cars=None, aps=None) -> "Scene":
        return Scene(self.seed, list(cars if cars is not None else self.cars),
                     list(aps if aps is not None else self.aps),
                     dict(self.external_occupancy), self.width_m,
                     self.height_m, self.region, self._shadow)

    def advance(self, dt: float) -> "Scene":
        moved = []
        for c in self.cars:
            x, y, vx, vy = c.x + c.vx * dt, c.y + c.vy * dt, c.vx, c.vy
            if not 0 <= x <= self.width_m:
                vx, x = -vx, min(max(x, 0.0), self.width_m)
            if not 0 <= y <= self.height_m:
                vy, y = -vy, min(max(y, 0.0), self.height_m)
            moved.append(replace(c, x=x, y=y, vx=vx, vy=vy))
        return self._copy(cars=moved)

    def with_associations(self, assoc: dict[str, str]) -> "Scene":
        return self._copy(cars=[replace(c, current_ap=assoc.get(c.car_id, c.current_ap))
                                for c in self.cars])

    def with_channels(self, channels: dict[str, int]) -> "Scene":
        return self._copy(aps=[replace(a, channel=channels.get(a.ap_id, a.channel))
                               for a in self.aps])

    def with_flows(self, car_id: str, flows: tuple[Flow, ...]) -> "Scene":
        return self._copy(cars=[replace(c, flows=flows) if c.car_id == car_id else c
                                for c in self.cars])


def _flows(rng: np.random.Generator, index: int) -> tuple[Flow, ...]:
    """Raspbot workloads: control, telemetry, camera video, map/log upload."""
    flows = [Flow("control", 0.2, True, 0.2),
             Flow("telemetry", round(float(rng.uniform(0.3, 1.5)), 2), False, 0.0)]
    if index in (2, 6) or rng.random() < 0.35:
        demand = round(float(rng.uniform(4.0, 12.0)), 1)
        critical = index in (2, 6)
        flows.append(Flow("video_uplink", demand, critical,
                          round(0.9 * demand, 1) if critical else 0.0))
    if rng.random() < 0.6:
        flows.append(Flow("map_upload", round(float(rng.uniform(10, 60)), 1),
                          False, 0.0))
    return tuple(flows)


def make_scene(seed: int, *, n_cars: int = 10, n_sites: int = 3,
               preset: str = "light_10_5", hotspot: bool | None = None,
               region: str | None = None) -> Scene:
    """Reproducible scene: n_sites dual-band customer AP sites (<= 8 BSSs
    visible, matching MaxAP), 10 Raspbots, neighbour-network occupancy."""
    rng = np.random.default_rng(seed)
    spec = CHANNEL_PRESETS[preset]
    occupancy: dict[tuple[str, int], float] = {}
    for band, chans in CHANNELS.items():
        for ch in chans:
            busy = rng.random() < spec["busy_fraction"]
            occupancy[(band, ch)] = float(rng.uniform(*spec["occupancy"])) if busy else 0.0

    width, height = 40.0, 25.0
    sites = [(6.0, 5.0), (34.0, 6.0), (20.0, 21.0), (5.0, 21.0)][:n_sites]
    aps = []
    for i, (x, y) in enumerate(sites):
        for band, prio in (("5g", 100), ("2g", 50)):
            channel = int(rng.choice(CHANNELS[band]))
            aps.append(AP(f"ap{i + 1}-{band}", x, y, channel, band,
                          site=f"ap{i + 1}", ssid=f"fleet_{band}", priority=prio,
                          tx_power_dbm=20.0 if band == "5g" else 17.0,
                          allowed_channels=CHANNELS[band]))

    if hotspot is None:
        hotspot = bool(rng.random() < 0.4)
    cars = []
    for i in range(n_cars):
        if hotspot and i < 7:
            hx, hy = sites[0]
            x = float(np.clip(hx + rng.normal(0, 4), 0, width))
            y = float(np.clip(hy + rng.normal(0, 4), 0, height))
        else:
            x, y = float(rng.uniform(0, width)), float(rng.uniform(0, height))
        speed = float(rng.uniform(0.1, 0.5))          # Raspbot-class speeds
        heading = float(rng.uniform(0, 2 * math.pi))
        cars.append(Car(f"car-{i + 1:02d}", x, y, speed * math.cos(heading),
                        speed * math.sin(heading), _flows(rng, i), None))

    scene = Scene(seed, cars, aps, occupancy, width, height,
                  region or f"R{seed % 4}")
    # Initial association = legacy rule (SSID priority, then strongest
    # signal) applied where each car was 30-150 s ago. The legacy policy only
    # reconnects below -80 dBm, so associations go stale as cars drive.
    assoc = {}
    for c in cars:
        ago = float(rng.uniform(30, 150))
        px = float(np.clip(c.x - c.vx * ago, 0, width))
        py = float(np.clip(c.y - c.vy * ago, 0, height))
        visible = scene.visible_aps(replace(c, x=px, y=py))
        best = max(visible, key=lambda s: (s[0].priority, s[1]))[0]
        assoc[c.car_id] = best.ap_id
    return scene.with_associations(assoc)
