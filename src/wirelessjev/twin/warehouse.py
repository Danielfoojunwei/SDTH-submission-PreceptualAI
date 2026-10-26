"""AGV/AMR warehouse twin: 100 robots on SmartCar hardware in a racked hall.

Robots are the same hardware as the SmartCar fleet (wirelessjev.hardware:
BE200 client radio on wlan1, BSSID selection is the only network action).
Only the environment changes:

- Hall 120 m x 80 m. Twelve rack rows run along x (1.2 m deep, 6 m pitch),
  split by cross-aisles at x = 40 and x = 80; staging/shipping strip at the
  bottom, receiving at the top, chargers on the left wall.
- Racks attenuate: each rack row a radio path crosses adds RACK_DB[band]
  (assumption: loaded steel racks, ceiling-mounted APs, so partial blockage),
  capped at RACK_CAP_DB.
- 16 ceiling AP sites on a 4 x 4 grid, dual-band (32 BSSIDs). The customer's
  channel plan is fixed: 6-channel 5 GHz reuse and 1/6/11 at 2.4 GHz.
  Handheld barcode scanners and other site systems occupy part of some
  channels' airtime (the "primary users" of Li/Shen/Lam's DSA papers).
- Robots run pick -> drop cycles along aisles with dwell times and go to a
  charger when low; their planned route is known to the fleet manager, so
  positions over the next 30 s are available to the planner without
  prediction (MobilityPredictor remains for robots without a plan).
- Traffic per robot: safety heartbeat (critical), pose/telemetry, WMS task
  messages, periodic map-sync bursts; a few robots stream teleoperation video
  (critical floor).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np

from .radio import AP, Car, Flow, Scene

WIDTH, HEIGHT = 120.0, 80.0
RACK_ROWS = tuple(10.0 + 6.0 * k for k in range(11))          # y of rack rows
RACK_SEGMENTS = ((10.0, 38.0), (42.0, 78.0), (82.0, 110.0))     # x extents
AISLES = tuple(13.0 + 6.0 * k for k in range(10))                # aisle centres
CROSS_AISLES = (6.0, 40.0, 80.0, 114.0)
DROP_POINTS = tuple((x, 4.0) for x in (20.0, 45.0, 70.0, 95.0))
RECEIVE_POINTS = tuple((x, 76.0) for x in (25.0, 60.0, 95.0))
CHARGERS = tuple((2.0, 15.0 + 8.0 * k) for k in range(7))
RACK_DB = {"5g": 3.5, "2g": 2.5}
RACK_CAP_DB = 21.0

AP_SITES = tuple((x, y) for y in (12.0, 32.0, 52.0, 72.0) for x in (15.0, 45.0, 75.0, 105.0))
CH5 = (36, 52, 100, 116, 132, 149)
CH2 = (1, 6, 11)


def rack_obstruction(ap: AP, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """dB of rack loss on the straight path between the AP and each point."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    crossings = np.zeros_like(xs)
    lo = np.minimum(ys, ap.y)
    hi = np.maximum(ys, ap.y)
    dy = ys - ap.y
    for yr in RACK_ROWS:
        spans = (lo < yr) & (hi > yr)
        if not spans.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(dy != 0, (yr - ap.y) / dy, 0.0)
        xc = ap.x + t * (xs - ap.x)
        inside = np.zeros_like(xs, dtype=bool)
        for x0, x1 in RACK_SEGMENTS:
            inside |= (xc >= x0) & (xc <= x1)
        crossings += spans & inside
    return np.minimum(RACK_CAP_DB, crossings * RACK_DB[ap.band])


def _route(start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
    """Manhattan route on the aisle network: leave the current aisle to the
    nearest cross-aisle, travel along it, enter the goal aisle."""
    sx, sy = start
    gx, gy = goal
    if abs(sy - gy) < 1e-6:
        return [goal]
    cross = min(CROSS_AISLES, key=lambda c: abs(c - sx) + abs(c - gx))
    return [(cross, sy), (cross, gy), goal]


@dataclass
class RobotPlan:
    waypoints: list[tuple[float, float]]
    dwell_s: float = 0.0
    speed: float = 0.8
    phase: str = "to_pick"
    battery: float = 1.0


@dataclass
class WarehouseMobility:
    """Task-driven aisle mobility; deterministic given its seed."""
    seed: int
    plans: dict[str, RobotPlan] = field(default_factory=dict)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def _pick_point(self):
        aisle = float(self.rng.choice(AISLES))
        seg = RACK_SEGMENTS[int(self.rng.integers(len(RACK_SEGMENTS)))]
        return float(self.rng.uniform(seg[0] + 1, seg[1] - 1)), aisle

    def _next_goal(self, plan: RobotPlan):
        if plan.battery < 0.2:
            plan.phase = "to_charge"
            return CHARGERS[int(self.rng.integers(len(CHARGERS)))]
        if plan.phase in ("to_pick", "charging"):
            plan.phase = "to_drop"
            pts = DROP_POINTS if self.rng.random() < 0.75 else RECEIVE_POINTS
            return pts[int(self.rng.integers(len(pts)))]
        plan.phase = "to_pick"
        return self._pick_point()

    def assign(self, car: Car, plan: RobotPlan) -> None:
        self.plans[car.car_id] = plan

    @staticmethod
    def _step(x, y, plan: RobotPlan, dt: float):
        remaining = dt
        while remaining > 1e-9:
            if plan.dwell_s > 0:
                use = min(plan.dwell_s, remaining)
                plan.dwell_s -= use
                remaining -= use
                continue
            if not plan.waypoints:
                return x, y, True
            tx, ty = plan.waypoints[0]
            d = math.hypot(tx - x, ty - y)
            travel = plan.speed * remaining
            if travel >= d:
                x, y = tx, ty
                remaining -= d / plan.speed
                plan.waypoints.pop(0)
                plan.battery = max(0.0, plan.battery - 0.002 * d)
                if not plan.waypoints:
                    return x, y, True
            else:
                x += (tx - x) / d * travel
                y += (ty - y) / d * travel
                plan.battery = max(0.0, plan.battery - 0.002 * travel)
                remaining = 0.0
        return x, y, False

    def advance(self, cars: list[Car], dt: float) -> list[Car]:
        out = []
        for car in cars:
            plan = self.plans[car.car_id]
            x, y, arrived = self._step(car.x, car.y, plan, dt)
            if arrived and plan.dwell_s <= 0:
                if plan.phase == "to_charge":
                    plan.phase, plan.battery, plan.dwell_s = "charging", 1.0, 60.0
                else:
                    plan.dwell_s = float(self.rng.uniform(8, 35))
                goal = self._next_goal(plan)
                plan.waypoints = _route((x, y), goal)
            out.append(replace(car, x=x, y=y))
        return out

    def planned_positions(self, car: Car, times: list[float]) -> list[tuple[float, float]]:
        """Positions along the robot's committed plan (no new goals drawn):
        what the fleet manager knows ahead of time."""
        plan = self.plans[car.car_id]
        sim = RobotPlan(list(plan.waypoints), plan.dwell_s, plan.speed, plan.phase,
                        plan.battery)
        x, y, t, out = car.x, car.y, 0.0, []
        for target in times:
            x, y, _ = self._step(x, y, sim, target - t)
            t = target
            out.append((x, y))
        return out


def _flows(rng: np.random.Generator, index: int) -> tuple[Flow, ...]:
    flows = [Flow("safety_heartbeat", 0.1, True, 0.1),
             Flow("pose_telemetry", round(float(rng.uniform(0.2, 0.8)), 2), False, 0.0),
             Flow("wms_tasks", 0.1, False, 0.0)]
    if index in (16, 41, 77) or rng.random() < 0.04:
        demand = round(float(rng.uniform(3.0, 8.0)), 1)
        flows.append(Flow("video_uplink", demand, True, round(0.9 * demand, 1)))
    if rng.random() < 0.45:
        flows.append(Flow("map_upload", round(float(rng.uniform(3, 20)), 1), False, 0.0))
    return tuple(flows)


def make_warehouse(seed: int, *, n_robots: int = 100, scanners: str = "normal",
                   region: str | None = None) -> Scene:
    rng = np.random.default_rng(seed)
    occupancy = {}
    busy = {"light": 0.15, "normal": 0.3, "heavy": 0.5}[scanners]
    for band, chans in (("5g", CH5), ("2g", CH2)):
        for ch in chans:
            hit = rng.random() < (busy if band == "5g" else min(1.0, busy * 2.5))
            occupancy[(band, ch)] = float(rng.uniform(0.05, 0.35)) if hit else 0.0

    aps = []
    for k, (x, y) in enumerate(AP_SITES):
        row, col = divmod(k, 4)
        ch5 = CH5[(col + 2 * row) % len(CH5)]
        ch2 = CH2[(col + row) % len(CH2)]
        aps.append(AP(f"ap{k + 1:02d}-5g", x, y, ch5, "5g", f"ap{k + 1:02d}",
                      "wh_5g", 100, 18.0, CH5))
        aps.append(AP(f"ap{k + 1:02d}-2g", x, y, ch2, "2g", f"ap{k + 1:02d}",
                      "wh_2g", 50, 14.0, CH2))

    mobility = WarehouseMobility(seed)
    cars = []
    for i in range(n_robots):
        start_kind = rng.random()
        if start_kind < 0.7:
            x, y = mobility._pick_point()
        elif start_kind < 0.9:
            x, y = DROP_POINTS[int(rng.integers(len(DROP_POINTS)))]
        else:
            x, y = CHARGERS[int(rng.integers(len(CHARGERS)))]
        car = Car(f"amr-{i + 1:03d}", float(x), float(y), 0.0, 0.0, _flows(rng, i), None)
        plan = RobotPlan([], dwell_s=float(rng.uniform(0, 20)),
                         speed=float(rng.uniform(0.6, 1.2)),
                         phase=str(rng.choice(["to_pick", "to_drop"])),
                         battery=float(rng.uniform(0.25, 1.0)))
        plan.waypoints = _route((x, y), mobility._next_goal(plan))
        mobility.assign(car, plan)
        cars.append(car)

    scene = Scene(seed, cars, aps, occupancy, WIDTH, HEIGHT,
                  region or f"W{seed % 4}", obstruction=rack_obstruction,
                  mobility=mobility)
    # Stale legacy associations: chosen where each robot was some time ago,
    # kept unless the signal drops below -80 dBm.
    assoc = {}
    for car in cars:
        visible = scene.visible_aps(car)
        # pick by priority then strongest at a jittered earlier position
        jx = float(np.clip(car.x + rng.normal(0, 12), 0, WIDTH))
        jy = float(np.clip(car.y + rng.normal(0, 8), 0, HEIGHT))
        old = Scene(seed, [replace(car, x=jx, y=jy)], aps, occupancy, WIDTH, HEIGHT,
                    obstruction=rack_obstruction)
        old_vis = old.visible_aps(old.cars[0])
        choice = max(old_vis, key=lambda s: (s[0].priority, s[1]))[0].ap_id
        if choice not in {ap.ap_id for ap, _ in visible}:
            choice = max(visible, key=lambda s: (s[0].priority, s[1]))[0].ap_id
        assoc[car.car_id] = choice
    return scene.with_associations(assoc)
