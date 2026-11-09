"""Per-car reputation under a contract.

From Shen, Lam, Yang, Liu & Li, "LEO Satellite-Enabled Networks: A
Privacy-Preserving Framework for Spectrum Pricing and Power Control
Optimization" (MSN 2024): reputation rises by p*l*#compliant when no user
misbehaves and falls by p*l*#malicious otherwise; misbehaviour means exceeding
the allowed transmit power, overusing access, or overrunning the lease.

Here each car is scored individually per period. Its contract: transmit power
<= cap, at most `max_reconnects` BSSID changes per window, and staying within
its declared demand. Cars below the threshold are excluded from swarm merges
and from high-priority options.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Reputation:
    p_hat: float = 0.05
    level: float = 1.0           # l: power-control level weight
    floor: float = 0.0
    ceiling: float = 2.0
    threshold: float = 0.5
    max_reconnects: int = 3
    demand_slack: float = 1.2
    scores: dict[str, float] = field(default_factory=dict)
    violations: dict[str, list[str]] = field(default_factory=dict)

    def check(self, car_id: str, *, tx_power_dbm: float, cap_dbm: float,
              reconnects: int, used_mbps: float, declared_mbps: float) -> list[str]:
        found = []
        if tx_power_dbm > cap_dbm:
            found.append("POWER_OVER_CAP")
        if reconnects > self.max_reconnects:
            found.append("ACCESS_OVERUSE")
        if used_mbps > self.demand_slack * declared_mbps:
            found.append("LEASE_OVERRUN")
        return found

    def update(self, car_id: str, violations: list[str]) -> float:
        s = self.scores.get(car_id, 1.0)
        if violations:
            s -= self.p_hat * self.level * len(violations) * 4
            self.violations.setdefault(car_id, []).extend(violations)
        else:
            s += self.p_hat * self.level
        s = min(self.ceiling, max(self.floor, s))
        self.scores[car_id] = s
        return s

    def trusted(self, car_id: str) -> bool:
        return self.scores.get(car_id, 1.0) >= self.threshold
