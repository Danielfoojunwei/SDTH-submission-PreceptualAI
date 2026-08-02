"""Power, thermal, and duty-cycle governance.

The governor is a pure function from a measured power/thermal reading and a
signed power policy to a *bounded recommendation*. It never actuates directly:
changing a Jetson power mode is a leased action executed by a root-gated path.
The governor can only select from an allowlist of approved ``nvpmodel`` IDs and
tighten duty cycles within policy — it cannot invent a power state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .models import parse_timestamp, utc_now
from .validation import bounded_number, non_empty_str, positive_int


class PowerReason(str, Enum):
    NOMINAL = "NOMINAL"
    THERMAL_LIMIT = "THERMAL_LIMIT"
    THERMAL_RECOVERING = "THERMAL_RECOVERING"
    BATTERY_FLOOR = "BATTERY_FLOOR"
    POWER_BUDGET = "POWER_BUDGET"
    THROTTLED_BY_HARDWARE = "THROTTLED_BY_HARDWARE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class PowerPolicy:
    policy_id: str
    created_at: datetime
    expires_at: datetime
    thermal_ceiling_c: float
    thermal_resume_c: float
    battery_floor_percent: float
    max_sustained_watts: float
    allowed_nvpmodel_ids: tuple[int, ...]
    economy_nvpmodel_id: int
    performance_nvpmodel_id: int
    min_telemetry_interval_seconds: float
    min_scan_interval_seconds: float
    max_inference_duty: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, now: datetime | None = None) -> "PowerPolicy":
        current = now or utc_now()
        policy_id = non_empty_str(raw.get("policy_id"), "policy_id")
        created_at = parse_timestamp(str(raw["created_at"]))
        expires_at = parse_timestamp(str(raw["expires_at"]))
        if expires_at <= created_at:
            raise ValueError("power policy expiry must be after creation")
        if expires_at <= current:
            raise ValueError("power policy has expired")
        ceiling = bounded_number(raw["thermal_ceiling_c"], "thermal_ceiling_c", 20.0, 120.0)
        resume = bounded_number(raw["thermal_resume_c"], "thermal_resume_c", 0.0, 120.0)
        if resume >= ceiling:
            raise ValueError("thermal_resume_c must be below thermal_ceiling_c (hysteresis)")
        battery_floor = bounded_number(raw.get("battery_floor_percent", 15.0), "battery_floor_percent", 0.0, 100.0)
        max_watts = bounded_number(raw["max_sustained_watts"], "max_sustained_watts", 1.0, 200.0)
        allowed = tuple(sorted({positive_int(item, "nvpmodel_id", minimum=0) for item in raw["allowed_nvpmodel_ids"]}))
        if not allowed:
            raise ValueError("allowed_nvpmodel_ids cannot be empty")
        economy = positive_int(raw["economy_nvpmodel_id"], "economy_nvpmodel_id", minimum=0)
        performance = positive_int(raw["performance_nvpmodel_id"], "performance_nvpmodel_id", minimum=0)
        if economy not in allowed or performance not in allowed:
            raise ValueError("economy and performance modes must be in allowed_nvpmodel_ids")
        min_telemetry = bounded_number(
            raw.get("min_telemetry_interval_seconds", 1.0), "min_telemetry_interval_seconds", 0.1, 3600.0
        )
        min_scan = bounded_number(
            raw.get("min_scan_interval_seconds", 1.0), "min_scan_interval_seconds", 0.1, 3600.0
        )
        max_duty = bounded_number(raw.get("max_inference_duty", 1.0), "max_inference_duty", 0.0, 1.0)
        return cls(
            policy_id=policy_id,
            created_at=created_at,
            expires_at=expires_at,
            thermal_ceiling_c=ceiling,
            thermal_resume_c=resume,
            battery_floor_percent=battery_floor,
            max_sustained_watts=max_watts,
            allowed_nvpmodel_ids=allowed,
            economy_nvpmodel_id=economy,
            performance_nvpmodel_id=performance,
            min_telemetry_interval_seconds=min_telemetry,
            min_scan_interval_seconds=min_scan,
            max_inference_duty=max_duty,
        )


@dataclass(frozen=True)
class PowerReading:
    observed_at: datetime
    source: str
    cpu_temp_c: float | None
    gpu_temp_c: float | None
    module_watts: float | None
    battery_percent: float | None
    hardware_throttled: bool
    nvpmodel_id: int | None
    junction_temp_c: float | None = None

    def hottest(self) -> float | None:
        temps = [
            value
            for value in (self.cpu_temp_c, self.gpu_temp_c, self.junction_temp_c)
            if value is not None
        ]
        return max(temps) if temps else None

    def has_thermal_evidence(self) -> bool:
        return self.hottest() is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_at": self.observed_at.isoformat(),
            "source": self.source,
            "cpu_temp_c": self.cpu_temp_c,
            "gpu_temp_c": self.gpu_temp_c,
            "junction_temp_c": self.junction_temp_c,
            "module_watts": self.module_watts,
            "battery_percent": self.battery_percent,
            "hardware_throttled": self.hardware_throttled,
            "nvpmodel_id": self.nvpmodel_id,
        }


_TEMP_FIELD = re.compile(r"(?P<name>cpu|gpu|tj|soc\d?|thermal)@(?P<value>-?[0-9]+(?:\.[0-9]+)?)C", re.IGNORECASE)
#: Module-input power rails, most preferred first. `tegrastats` prints several
#: rails per line and their order varies by carrier board, so the rail is
#: selected by name rather than by position — matching the first rail that
#: appears would report an SoC-only rail as total module power.
_POWER_RAILS = ("VDD_IN", "POM_5V_IN", "VDD_SYS_GPU", "VDD_SYS_SOC")
_POWER_FIELD = re.compile(
    r"\b(?P<rail>[A-Z0-9_]+)\s+(?P<instant>[0-9]+)mW/(?P<average>[0-9]+)mW", re.IGNORECASE
)
#: `tegrastats` does not print the word "throttled"; hardware throttling shows
#: up as a non-zero thermal-throttling counter or an explicit flag when the
#: platform exposes one.
_THROTTLE_FIELD = re.compile(r"(?:throttl\w*)[@ :]*(?P<value>[0-9]+)?", re.IGNORECASE)


def parse_tegrastats(line: str, *, observed_at: datetime | None = None) -> PowerReading:
    """Parse a single ``tegrastats`` line into a thermal/power reading.

    ``tegrastats`` output varies by carrier and L4T release, so unmatched fields
    stay ``None`` rather than being guessed. A line with no thermal field yields
    a reading with no thermal evidence, which the governor treats as
    insufficient rather than safe.

    Junction temperature (``tj@``) is retained as its own signal rather than
    discarded when a ``cpu@`` field is also present, because on Orin the
    junction sensor is often the hottest one and it is what
    :meth:`PowerReading.hottest` must see.
    """
    when = observed_at or utc_now()
    cpu_temp: float | None = None
    gpu_temp: float | None = None
    junction_temp: float | None = None
    for match in _TEMP_FIELD.finditer(line):
        value = float(match.group("value"))
        name = match.group("name").lower()
        if name == "cpu":
            cpu_temp = value
        elif name == "gpu":
            gpu_temp = value
        elif name.startswith("tj") or name == "thermal" or name.startswith("soc"):
            junction_temp = value if junction_temp is None else max(junction_temp, value)

    module_watts: float | None = None
    rails = {
        match.group("rail").upper(): float(match.group("instant"))
        for match in _POWER_FIELD.finditer(line)
    }
    for rail in _POWER_RAILS:
        if rail in rails:
            module_watts = round(rails[rail] / 1000.0, 3)
            break

    throttled = False
    throttle_match = _THROTTLE_FIELD.search(line)
    if throttle_match:
        counter = throttle_match.group("value")
        throttled = counter is None or int(counter) > 0

    return PowerReading(
        observed_at=when,
        source="tegrastats",
        cpu_temp_c=cpu_temp,
        gpu_temp_c=gpu_temp,
        junction_temp_c=junction_temp,
        module_watts=module_watts,
        battery_percent=None,
        hardware_throttled=throttled,
        nvpmodel_id=None,
    )


@dataclass(frozen=True)
class PowerRecommendation:
    reason: PowerReason
    target_nvpmodel_id: int
    max_inference_duty: float
    min_scan_interval_seconds: float
    min_telemetry_interval_seconds: float
    detail: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "target_nvpmodel_id": self.target_nvpmodel_id,
            "max_inference_duty": self.max_inference_duty,
            "min_scan_interval_seconds": self.min_scan_interval_seconds,
            "min_telemetry_interval_seconds": self.min_telemetry_interval_seconds,
            "detail": list(self.detail),
        }

    def to_lease_bounds(self) -> dict[str, float]:
        """Bounds an operator lease must cover for this recommendation to execute."""
        return {
            "nvpmodel_id": float(self.target_nvpmodel_id),
            "max_inference_duty": self.max_inference_duty,
            "min_scan_interval_seconds": self.min_scan_interval_seconds,
        }


class PowerGovernor:
    """Deterministic power/thermal governor over a signed policy.

    Constraints stack: thermal ceiling wins over battery floor wins over the
    sustained-watt budget. Recovery to the performance mode requires the hottest
    sensor to fall to the policy resume temperature (hysteresis), so the node
    does not oscillate around the ceiling.
    """

    def recommend(
        self,
        policy: PowerPolicy,
        reading: PowerReading,
        *,
        current_nvpmodel_id: int | None = None,
    ) -> PowerRecommendation:
        detail: list[str] = []
        economy = policy.economy_nvpmodel_id
        performance = policy.performance_nvpmodel_id
        conservative_scan = max(policy.min_scan_interval_seconds, policy.min_scan_interval_seconds * 4)
        conservative_telemetry = max(policy.min_telemetry_interval_seconds, policy.min_telemetry_interval_seconds * 4)
        reduced_duty = min(policy.max_inference_duty, 0.25)

        if reading.hardware_throttled:
            detail.append("hardware reported thermal throttling")
            return PowerRecommendation(
                PowerReason.THROTTLED_BY_HARDWARE,
                economy,
                min(reduced_duty, 0.1),
                conservative_scan,
                conservative_telemetry,
                tuple(detail),
            )

        hottest = reading.hottest()
        # Thermal evidence is mandatory: without it the governor cannot know
        # whether raising the power mode is safe, so it must not return NOMINAL.
        if hottest is None:
            detail.append("no thermal evidence; holding economy mode")
            return PowerRecommendation(
                PowerReason.INSUFFICIENT_EVIDENCE,
                economy,
                reduced_duty,
                conservative_scan,
                conservative_telemetry,
                tuple(detail),
            )

        if hottest >= policy.thermal_ceiling_c:
            detail.append(f"hottest sensor {hottest}C at/above ceiling {policy.thermal_ceiling_c}C")
            return PowerRecommendation(
                PowerReason.THERMAL_LIMIT,
                economy,
                reduced_duty,
                conservative_scan,
                conservative_telemetry,
                tuple(detail),
            )

        # Hysteresis: if we are in economy for thermal reasons, stay there until
        # the resume temperature is reached.
        if (
            current_nvpmodel_id == economy
            and economy != performance
            and hottest > policy.thermal_resume_c
        ):
            detail.append(f"holding economy until sensor cools to {policy.thermal_resume_c}C")
            return PowerRecommendation(
                PowerReason.THERMAL_RECOVERING,
                economy,
                reduced_duty,
                conservative_scan,
                conservative_telemetry,
                tuple(detail),
            )

        if reading.battery_percent is not None and reading.battery_percent <= policy.battery_floor_percent:
            detail.append(f"battery {reading.battery_percent}% at/below floor {policy.battery_floor_percent}%")
            return PowerRecommendation(
                PowerReason.BATTERY_FLOOR,
                economy,
                reduced_duty,
                conservative_scan,
                conservative_telemetry,
                tuple(detail),
            )

        if reading.module_watts is not None and reading.module_watts > policy.max_sustained_watts:
            detail.append(f"module {reading.module_watts}W above budget {policy.max_sustained_watts}W")
            return PowerRecommendation(
                PowerReason.POWER_BUDGET,
                economy,
                min(policy.max_inference_duty, 0.5),
                max(policy.min_scan_interval_seconds, policy.min_scan_interval_seconds * 2),
                policy.min_telemetry_interval_seconds,
                tuple(detail),
            )

        detail.append("within thermal, battery, and power budgets")
        return PowerRecommendation(
            PowerReason.NOMINAL,
            performance,
            policy.max_inference_duty,
            policy.min_scan_interval_seconds,
            policy.min_telemetry_interval_seconds,
            tuple(detail),
        )
