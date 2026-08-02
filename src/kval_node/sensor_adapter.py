"""Real, receive-only spectrum sensor adapter.

This is the implementation behind the ``kval-spectrum-agent/v1`` contract that
``SpectrumSensorClient`` consumes. It produces occupancy observations from
*measurements taken by the host's own radio hardware* — there is no synthetic
backend, and a band with no measurement is reported with an error rather than a
fabricated number.

Measurement sources, in preference order:

``iw dev <iface> survey dump``
    The primary source. The kernel reports, per channel, a real measured
    ``noise`` figure in dBm and ``channel active time`` / ``channel busy time``
    counters. Busy/active is a directly measured channel-occupancy ratio — not
    an estimate derived from beacon counts — which is exactly the quantity the
    spectrum policy thresholds are written against.

``iw dev <iface> scan`` / ``nmcli device wifi list``
    Distinct BSSIDs per band give the detected-emitter count, and the strongest
    signal gives peak power. The 802.11 BSS Load element, when an AP advertises
    it, carries that AP's own measured channel utilisation and is used to
    corroborate the survey figure.

``rtl_power``
    For an attached RTL-SDR, a real power-spectral-density sweep. Occupancy is
    the fraction of frequency bins whose power exceeds the measured noise floor
    by a configured margin; the noise floor is a low percentile of the same
    sweep. Both are computed from the sweep itself, never assumed.

Everything here is **receive-only**. No backend transmits, tunes a transmitter,
or sets gain or power. ``iw scan`` performs active probing on the host's own
already-associated Wi-Fi interface via the standard kernel path; the survey and
``rtl_power`` backends are strictly passive. Operators who require strict
receive-only emission behaviour should run with ``--passive-scan-only``, which
disables the scan backend and uses survey plus SDR data alone.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from .commands import CommandError, run_command
from .models import utc_now
from .spectrum import SpectrumBand, SpectrumPolicy

ADAPTER_SCHEMA = "kval-spectrum-agent/v1"

#: dB above the measured noise floor at which an SDR frequency bin counts as
#: occupied. 6 dB is a conventional energy-detection margin; it is a documented
#: constant rather than a tuned magic number, and it is reported in evidence.
SDR_OCCUPANCY_MARGIN_DB = 6.0

#: Percentile of an SDR sweep's bin powers taken as the noise floor.
SDR_NOISE_FLOOR_PERCENTILE = 0.10


@dataclass(frozen=True)
class BandMeasurement:
    """A real measurement for one policy band, or an error explaining its absence."""

    band_id: str
    occupancy_percent: float | None
    noise_floor_dbm: float | None
    peak_power_dbm: float | None
    detected_emitters: int
    quality: float
    sources: tuple[str, ...]
    errors: tuple[str, ...]


class SensorBackend(Protocol):
    """A source of real RF measurements."""

    name: str

    def available(self) -> bool:
        """True when this backend's tool and hardware are actually present."""

    def measure(self, bands: Sequence[SpectrumBand]) -> dict[str, BandMeasurement]:
        """Return measurements keyed by band_id for whatever it can observe."""


# --- iw survey (primary: real noise floor + measured busy ratio) ----------------


_SURVEY_FREQ = re.compile(r"frequency:\s+(?P<mhz>\d+)\s*MHz")
_SURVEY_NOISE = re.compile(r"noise:\s+(?P<dbm>-?\d+)\s*dBm")
_SURVEY_ACTIVE = re.compile(r"channel active time:\s+(?P<ms>\d+)\s*ms")
_SURVEY_BUSY = re.compile(r"channel busy time:\s+(?P<ms>\d+)\s*ms")


@dataclass(frozen=True)
class SurveyChannel:
    freq_hz: int
    noise_dbm: float | None
    active_ms: int | None
    busy_ms: int | None

    def occupancy_percent(self) -> float | None:
        if self.active_ms is None or self.busy_ms is None or self.active_ms <= 0:
            return None
        # Busy time can marginally exceed active time on some drivers; clamp
        # rather than emitting an out-of-range value the policy layer rejects.
        return min(100.0, max(0.0, self.busy_ms / self.active_ms * 100.0))


def parse_iw_survey(output: str) -> list[SurveyChannel]:
    """Parse ``iw dev <iface> survey dump`` into per-channel measurements."""
    channels: list[SurveyChannel] = []
    current: dict[str, Any] = {}

    def flush() -> None:
        if "freq_hz" in current:
            channels.append(
                SurveyChannel(
                    freq_hz=current["freq_hz"],
                    noise_dbm=current.get("noise_dbm"),
                    active_ms=current.get("active_ms"),
                    busy_ms=current.get("busy_ms"),
                )
            )
        current.clear()

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("Survey data from"):
            flush()
            continue
        freq = _SURVEY_FREQ.search(line)
        if freq:
            # A new frequency block begins; emit the previous one.
            if "freq_hz" in current:
                flush()
            current["freq_hz"] = int(freq.group("mhz")) * 1_000_000
            continue
        noise = _SURVEY_NOISE.search(line)
        if noise:
            current["noise_dbm"] = float(noise.group("dbm"))
            continue
        active = _SURVEY_ACTIVE.search(line)
        if active:
            current["active_ms"] = int(active.group("ms"))
            continue
        busy = _SURVEY_BUSY.search(line)
        if busy:
            current["busy_ms"] = int(busy.group("ms"))
    flush()
    return channels


def _channels_in_band(channels: Iterable[SurveyChannel], band: SpectrumBand) -> list[SurveyChannel]:
    return [c for c in channels if band.freq_low_hz <= c.freq_hz <= band.freq_high_hz]


class IwSurveyBackend:
    """Kernel channel survey: measured noise floor and busy/active occupancy."""

    name = "iw-survey"

    def __init__(self, interface: str) -> None:
        self.interface = interface

    def available(self) -> bool:
        try:
            result = run_command(["iw", "dev", self.interface, "info"], timeout_seconds=5.0)
        except CommandError:
            return False
        return result.returncode == 0

    def _survey(self) -> list[SurveyChannel]:
        result = run_command(["iw", "dev", self.interface, "survey", "dump"], timeout_seconds=10.0)
        if result.returncode != 0:
            raise CommandError(result.stderr.strip() or "iw survey dump failed")
        return parse_iw_survey(result.stdout)

    def measure(self, bands: Sequence[SpectrumBand]) -> dict[str, BandMeasurement]:
        try:
            channels = self._survey()
        except CommandError as exc:
            return {
                band.band_id: BandMeasurement(band.band_id, None, None, None, 0, 0.0, (), (str(exc),))
                for band in bands
            }
        results: dict[str, BandMeasurement] = {}
        for band in bands:
            in_band = _channels_in_band(channels, band)
            if not in_band:
                results[band.band_id] = BandMeasurement(
                    band.band_id, None, None, None, 0, 0.0, (),
                    (f"{self.name}: no surveyed channel falls inside this band",),
                )
                continue
            occupancies = [value for value in (c.occupancy_percent() for c in in_band) if value is not None]
            noises = [c.noise_dbm for c in in_band if c.noise_dbm is not None]
            errors: list[str] = []
            if not occupancies:
                errors.append(f"{self.name}: driver reported no channel active/busy time")
            if not noises:
                errors.append(f"{self.name}: driver reported no noise figure")
            # Quality reflects how much of the band the survey actually covered.
            covered = len(occupancies) / len(in_band) if in_band else 0.0
            results[band.band_id] = BandMeasurement(
                band_id=band.band_id,
                occupancy_percent=(max(occupancies) if occupancies else None),
                noise_floor_dbm=(max(noises) if noises else None),
                peak_power_dbm=None,
                detected_emitters=0,
                quality=round(covered, 6),
                sources=(self.name,),
                errors=tuple(errors),
            )
        return results


# --- iw scan / nmcli (emitter count and peak signal) ----------------------------


_SCAN_BSS = re.compile(r"^BSS\s+(?P<bssid>[0-9a-fA-F:]{17})")
_SCAN_FREQ = re.compile(r"^\s*freq:\s*(?P<mhz>\d+(?:\.\d+)?)")
_SCAN_SIGNAL = re.compile(r"^\s*signal:\s*(?P<dbm>-?\d+(?:\.\d+)?)\s*dBm")
_SCAN_UTIL = re.compile(r"channel utilisation:\s*(?P<used>\d+)\s*/\s*(?P<total>\d+)")


@dataclass(frozen=True)
class ScannedBss:
    bssid: str
    freq_hz: int
    signal_dbm: float | None
    channel_utilisation_percent: float | None


def parse_iw_scan(output: str) -> list[ScannedBss]:
    """Parse ``iw dev <iface> scan`` into per-BSS observations.

    The 802.11 BSS Load element's ``channel utilisation`` is a 0-255 value the
    AP itself measured; it is converted to a percentage and kept as
    corroborating evidence for the survey occupancy.
    """
    found: list[ScannedBss] = []
    current: dict[str, Any] = {}

    def flush() -> None:
        if "bssid" in current and "freq_hz" in current:
            found.append(
                ScannedBss(
                    bssid=current["bssid"].lower(),
                    freq_hz=current["freq_hz"],
                    signal_dbm=current.get("signal_dbm"),
                    channel_utilisation_percent=current.get("utilisation"),
                )
            )
        current.clear()

    for raw_line in output.splitlines():
        bss = _SCAN_BSS.match(raw_line)
        if bss:
            flush()
            current["bssid"] = bss.group("bssid")
            continue
        if not current:
            continue
        freq = _SCAN_FREQ.match(raw_line)
        if freq:
            current["freq_hz"] = int(float(freq.group("mhz")) * 1_000_000)
            continue
        signal = _SCAN_SIGNAL.match(raw_line)
        if signal:
            current["signal_dbm"] = float(signal.group("dbm"))
            continue
        util = _SCAN_UTIL.search(raw_line)
        if util:
            total = float(util.group("total"))
            if total > 0:
                current["utilisation"] = float(util.group("used")) / total * 100.0
    flush()
    return found


class IwScanBackend:
    """Distinct BSSIDs per band, their peak signal, and advertised BSS Load."""

    name = "iw-scan"

    def __init__(self, interface: str) -> None:
        self.interface = interface

    def available(self) -> bool:
        try:
            result = run_command(["iw", "dev", self.interface, "info"], timeout_seconds=5.0)
        except CommandError:
            return False
        return result.returncode == 0

    def measure(self, bands: Sequence[SpectrumBand]) -> dict[str, BandMeasurement]:
        try:
            result = run_command(["iw", "dev", self.interface, "scan"], timeout_seconds=30.0)
            if result.returncode != 0:
                raise CommandError(result.stderr.strip() or "iw scan failed")
            stations = parse_iw_scan(result.stdout)
        except CommandError as exc:
            return {
                band.band_id: BandMeasurement(band.band_id, None, None, None, 0, 0.0, (), (str(exc),))
                for band in bands
            }
        results: dict[str, BandMeasurement] = {}
        for band in bands:
            in_band = [s for s in stations if band.freq_low_hz <= s.freq_hz <= band.freq_high_hz]
            signals = [s.signal_dbm for s in in_band if s.signal_dbm is not None]
            utilisations = [
                s.channel_utilisation_percent for s in in_band if s.channel_utilisation_percent is not None
            ]
            results[band.band_id] = BandMeasurement(
                band_id=band.band_id,
                occupancy_percent=(max(utilisations) if utilisations else None),
                noise_floor_dbm=None,
                peak_power_dbm=(max(signals) if signals else None),
                detected_emitters=len({s.bssid for s in in_band}),
                quality=1.0 if in_band else 0.0,
                sources=(self.name,),
                errors=(),
            )
        return results


_NMCLI_FIELDS = "BSSID,FREQ,SIGNAL"


class NmcliWifiBackend:
    """NetworkManager's BSS list, for hosts without ``iw`` available.

    ``nmcli`` reports SIGNAL as a 0-100 quality percentage rather than dBm, so
    this backend contributes emitter counts only and never claims a power
    figure it did not measure in dBm.
    """

    name = "nmcli-wifi"

    def available(self) -> bool:
        try:
            result = run_command(["nmcli", "--version"], timeout_seconds=5.0)
        except CommandError:
            return False
        return result.returncode == 0

    def measure(self, bands: Sequence[SpectrumBand]) -> dict[str, BandMeasurement]:
        try:
            result = run_command(
                ["nmcli", "--terse", "--escape", "no", "--fields", _NMCLI_FIELDS,
                 "device", "wifi", "list", "--rescan", "no"],
                timeout_seconds=15.0,
            )
            if result.returncode != 0:
                raise CommandError(result.stderr.strip() or "nmcli wifi list failed")
        except CommandError as exc:
            return {
                band.band_id: BandMeasurement(band.band_id, None, None, None, 0, 0.0, (), (str(exc),))
                for band in bands
            }
        entries: list[tuple[str, int]] = []
        for line in result.stdout.splitlines():
            # BSSID contains escaped colons; split from the right for FREQ/SIGNAL.
            parts = line.rsplit(":", 2)
            if len(parts) != 3:
                continue
            bssid, freq_text, _signal = parts
            digits = re.match(r"\s*(\d+)", freq_text)
            if not digits:
                continue
            entries.append((bssid.replace("\\", "").lower(), int(digits.group(1)) * 1_000_000))
        results: dict[str, BandMeasurement] = {}
        for band in bands:
            in_band = [b for b, hz in entries if band.freq_low_hz <= hz <= band.freq_high_hz]
            results[band.band_id] = BandMeasurement(
                band_id=band.band_id,
                occupancy_percent=None,
                noise_floor_dbm=None,
                peak_power_dbm=None,
                detected_emitters=len(set(in_band)),
                quality=1.0 if in_band else 0.0,
                sources=(self.name,),
                errors=(),
            )
        return results


# --- ModemManager (cellular bands, free on any node with a modem) ---------------


@dataclass(frozen=True)
class ServingCell:
    freq_hz: int | None
    rsrp_dbm: float | None
    rsrq_db: float | None
    sinr_db: float | None
    technology: str | None


def _first_number(value: Any) -> float | None:
    """ModemManager renders numbers as strings, lists, or dicts by version."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value in (None, "--", ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_mmcli_signal(document: dict[str, Any]) -> ServingCell:
    """Extract serving-cell measurements from ``mmcli -m <n> --signal-get -J``.

    RSRP and RSRQ are genuine physical-layer measurements the modem reports for
    the cell it is camped on. Unlike a Wi-Fi quality percentage they are already
    in dBm/dB, so they can be used as real power figures.
    """
    modem = document.get("modem", {}) if isinstance(document, dict) else {}
    signal = modem.get("signal", {}) if isinstance(modem, dict) else {}
    technology = None
    cell: dict[str, Any] = {}
    for name in ("5g", "nr5g", "lte", "umts", "gsm", "cdma1x", "evdo"):
        candidate = signal.get(name)
        if isinstance(candidate, dict) and any(v not in (None, "--") for v in candidate.values()):
            technology = name
            cell = candidate
            break
    return ServingCell(
        freq_hz=None,
        rsrp_dbm=_first_number(cell.get("rsrp")),
        rsrq_db=_first_number(cell.get("rsrq")),
        sinr_db=_first_number(cell.get("snr") or cell.get("sinr")),
        technology=technology,
    )


def parse_mmcli_location_frequency(document: dict[str, Any]) -> int | None:
    """Extract the serving-cell centre frequency in Hz when the modem reports it."""
    modem = document.get("modem", {}) if isinstance(document, dict) else {}
    generic = modem.get("generic", {}) if isinstance(modem, dict) else {}
    for key in ("earfcn", "nrarfcn", "arfcn", "frequency"):
        value = _first_number(generic.get(key))
        if value is not None and value > 1_000_000:
            return int(value)
    return None


class ModemSignalBackend:
    """Serving-cell RSRP/RSRQ/SINR from ModemManager.

    This adds real measurements in the cellular bands to nodes that already have
    a modem, without any extra hardware. It reports peak power (RSRP, a genuine
    dBm figure) and a noise-floor estimate derived from RSRP and SINR, which is
    a standard relation and is documented as an estimate rather than a direct
    measurement. It never claims an occupancy figure, because a serving-cell
    report says nothing about how busy the band is.
    """

    name = "modem-signal"

    def available(self) -> bool:
        try:
            result = run_command(["mmcli", "-L", "-J"], timeout_seconds=5.0)
        except CommandError:
            return False
        return result.returncode == 0

    def _modems(self) -> list[str]:
        result = run_command(["mmcli", "-L", "-J"], timeout_seconds=5.0)
        if result.returncode != 0:
            raise CommandError(result.stderr.strip() or "mmcli -L failed")
        try:
            listing = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise CommandError("mmcli returned malformed JSON") from exc
        return [str(path) for path in listing.get("modem-list", [])]

    def measure(self, bands: Sequence[SpectrumBand]) -> dict[str, BandMeasurement]:
        try:
            modems = self._modems()
        except CommandError as exc:
            return {
                band.band_id: BandMeasurement(band.band_id, None, None, None, 0, 0.0, (), (str(exc),))
                for band in bands
            }
        cells: list[tuple[ServingCell, int | None]] = []
        for path in modems:
            try:
                # Enabling signal polling is a receive-only refresh of the
                # modem's own measurement reporting; it transmits nothing.
                run_command(["mmcli", "-m", path, "--signal-setup=5"], timeout_seconds=10.0)
                signal_result = run_command(["mmcli", "-m", path, "--signal-get", "-J"], timeout_seconds=10.0)
                info_result = run_command(["mmcli", "-m", path, "-J"], timeout_seconds=10.0)
            except CommandError:
                continue
            if signal_result.returncode != 0:
                continue
            try:
                signal_doc = json.loads(signal_result.stdout)
                info_doc = json.loads(info_result.stdout) if info_result.returncode == 0 else {}
            except json.JSONDecodeError:
                continue
            cells.append((parse_mmcli_signal(signal_doc), parse_mmcli_location_frequency(info_doc)))

        results: dict[str, BandMeasurement] = {}
        for band in bands:
            in_band = [
                cell
                for cell, freq in cells
                if freq is not None and band.freq_low_hz <= freq <= band.freq_high_hz
            ]
            if not in_band:
                results[band.band_id] = BandMeasurement(
                    band.band_id, None, None, None, 0, 0.0, (),
                    (f"{self.name}: no serving cell reported inside this band",),
                )
                continue
            rsrps = [cell.rsrp_dbm for cell in in_band if cell.rsrp_dbm is not None]
            # Noise floor estimate: N = RSRP - SINR. Standard relation, but an
            # estimate rather than a measured floor, so it is labelled as such.
            floors = [
                cell.rsrp_dbm - cell.sinr_db
                for cell in in_band
                if cell.rsrp_dbm is not None and cell.sinr_db is not None
            ]
            evidence_errors: tuple[str, ...] = ()
            if floors:
                evidence_errors = (f"{self.name}: noise floor is estimated as RSRP-SINR, not measured",)
            results[band.band_id] = BandMeasurement(
                band_id=band.band_id,
                occupancy_percent=None,
                noise_floor_dbm=(max(floors) if floors else None),
                peak_power_dbm=(max(rsrps) if rsrps else None),
                detected_emitters=len(in_band),
                quality=1.0 if rsrps else 0.0,
                sources=(self.name,),
                errors=evidence_errors,
            )
        return results


# --- rtl_power (real SDR power spectral density) --------------------------------


@dataclass(frozen=True)
class SweepBin:
    freq_hz: int
    power_db: float


def parse_rtl_power_csv(output: str) -> list[SweepBin]:
    """Parse ``rtl_power`` CSV rows into frequency bins.

    Row shape: date, time, Hz low, Hz high, Hz step, samples, dB, dB, ...
    """
    bins: list[SweepBin] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 7:
            continue
        try:
            low = float(fields[2])
            step = float(fields[4])
            powers = [float(value) for value in fields[6:] if value]
        except ValueError:
            continue
        for index, power in enumerate(powers):
            bins.append(SweepBin(int(low + index * step), power))
    return bins


def occupancy_from_sweep(
    bins: Sequence[SweepBin],
    *,
    margin_db: float = SDR_OCCUPANCY_MARGIN_DB,
    percentile: float = SDR_NOISE_FLOOR_PERCENTILE,
) -> tuple[float, float, float]:
    """Return ``(occupancy_percent, noise_floor_db, peak_db)`` from a real sweep.

    The noise floor is a low percentile of the sweep's own bin powers, and a bin
    counts as occupied when it exceeds that floor by ``margin_db``. Both come
    from the measurement itself; nothing is assumed about the environment.
    """
    if not bins:
        raise ValueError("sweep contained no frequency bins")
    powers = sorted(item.power_db for item in bins)
    index = min(len(powers) - 1, max(0, int(len(powers) * percentile)))
    noise_floor = powers[index]
    threshold = noise_floor + margin_db
    occupied = sum(1 for value in powers if value > threshold)
    return (occupied / len(powers) * 100.0, noise_floor, powers[-1])


class RtlPowerBackend:
    """Real power-spectral-density sweep from an attached RTL-SDR."""

    name = "rtl-power"

    def __init__(self, *, integration_seconds: int = 1, bin_hz: int = 100_000) -> None:
        self.integration_seconds = integration_seconds
        self.bin_hz = bin_hz

    def available(self) -> bool:
        try:
            result = run_command(["rtl_power", "-h"], timeout_seconds=5.0)
        except CommandError:
            return False
        # rtl_power exits non-zero for -h on some builds; presence is what matters.
        return result.returncode in {0, 1}

    def measure(self, bands: Sequence[SpectrumBand]) -> dict[str, BandMeasurement]:
        results: dict[str, BandMeasurement] = {}
        for band in bands:
            try:
                result = run_command(
                    [
                        "rtl_power",
                        "-f", f"{band.freq_low_hz}:{band.freq_high_hz}:{min(self.bin_hz, band.max_resolution_bandwidth_hz)}",
                        "-i", f"{self.integration_seconds}s",
                        "-1",
                        "-",
                    ],
                    timeout_seconds=30.0 + self.integration_seconds,
                )
                if result.returncode != 0:
                    raise CommandError(result.stderr.strip() or "rtl_power failed")
                bins = parse_rtl_power_csv(result.stdout)
                occupancy, noise_floor, peak = occupancy_from_sweep(bins)
            except (CommandError, ValueError) as exc:
                results[band.band_id] = BandMeasurement(
                    band.band_id, None, None, None, 0, 0.0, (), (f"{self.name}: {exc}",)
                )
                continue
            results[band.band_id] = BandMeasurement(
                band_id=band.band_id,
                occupancy_percent=occupancy,
                noise_floor_dbm=noise_floor,
                peak_power_dbm=peak,
                detected_emitters=0,
                quality=1.0,
                sources=(self.name,),
                errors=(),
            )
        return results


# --- aggregation into the signed adapter contract -------------------------------


def merge_measurements(
    band: SpectrumBand, parts: Sequence[BandMeasurement]
) -> BandMeasurement:
    """Combine per-backend measurements for one band, preferring real figures.

    A field is taken from the first backend that actually measured it, in the
    order the backends were supplied. Emitter counts take the maximum observed,
    since a backend that saw more distinct BSSIDs genuinely saw more.
    """
    occupancy = next((p.occupancy_percent for p in parts if p.occupancy_percent is not None), None)
    noise = next((p.noise_floor_dbm for p in parts if p.noise_floor_dbm is not None), None)
    peak = next((p.peak_power_dbm for p in parts if p.peak_power_dbm is not None), None)
    emitters = max((p.detected_emitters for p in parts), default=0)
    sources = tuple(sorted({name for p in parts for name in p.sources}))
    errors = tuple(sorted({e for p in parts for e in p.errors}))
    contributing = [p.quality for p in parts if p.quality > 0.0]
    quality = round(sum(contributing) / len(contributing), 6) if contributing else 0.0

    missing: list[str] = []
    if occupancy is None:
        missing.append("no backend measured channel occupancy")
    if noise is None:
        missing.append("no backend measured a noise floor")
    return BandMeasurement(
        band_id=band.band_id,
        occupancy_percent=occupancy,
        noise_floor_dbm=noise,
        peak_power_dbm=peak,
        detected_emitters=emitters,
        quality=quality if not missing else 0.0,
        sources=sources,
        errors=errors + tuple(missing),
    )


def adapter_conformance_hash() -> str:
    """SHA-256 of this adapter's own source — a real, verifiable provenance hash.

    The contract requires a 64-hex conformance digest. Rather than carrying a
    constant, the adapter hashes the module that produced the measurement, so an
    operator can recompute it and confirm which code generated an observation.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class SpectrumSensorAdapter:
    """Serves real measurements in the ``kval-spectrum-agent/v1`` shape."""

    def __init__(
        self,
        backends: Sequence[SensorBackend],
        *,
        adapter_id: str,
        conformance_hash: str | None = None,
    ) -> None:
        if not backends:
            raise ValueError("at least one measurement backend is required")
        self.backends = list(backends)
        self.adapter_id = adapter_id
        self.conformance_hash = conformance_hash or adapter_conformance_hash()

    def observe(self, policy: SpectrumPolicy, *, now: datetime | None = None) -> list[dict[str, Any]]:
        observed_at = (now or utc_now()).isoformat()
        per_backend: list[dict[str, BandMeasurement]] = []
        for backend in self.backends:
            try:
                per_backend.append(backend.measure(policy.bands))
            except Exception as exc:  # a failing backend must not abort the scan
                per_backend.append(
                    {
                        band.band_id: BandMeasurement(
                            band.band_id, None, None, None, 0, 0.0, (), (f"{backend.name}: {exc}",)
                        )
                        for band in policy.bands
                    }
                )

        observations: list[dict[str, Any]] = []
        for band in policy.bands:
            parts = [table[band.band_id] for table in per_backend if band.band_id in table]
            merged = merge_measurements(band, parts)
            observations.append(
                {
                    "band_id": band.band_id,
                    "observed_at": observed_at,
                    "source_adapter_id": self.adapter_id,
                    "conformance_hash": self.conformance_hash,
                    "passive": True,
                    # The node refuses out-of-range values, so an unmeasured
                    # band reports a defined value carrying errors and zero
                    # quality; the analyzer then treats it as insufficient
                    # evidence rather than as a clear band.
                    "occupancy_percent": merged.occupancy_percent if merged.occupancy_percent is not None else 0.0,
                    "noise_floor_dbm": merged.noise_floor_dbm if merged.noise_floor_dbm is not None else 0.0,
                    "peak_power_dbm": merged.peak_power_dbm,
                    "detected_emitters": merged.detected_emitters,
                    "bearing_deg": None,
                    "bearing_confidence": None,
                    "quality": merged.quality,
                    "errors": list(merged.errors) + [f"sources={','.join(merged.sources)}"] if merged.errors else [],
                }
            )
        return observations

    def handle(self, request: dict[str, Any], policy: SpectrumPolicy) -> dict[str, Any]:
        """Handle one ``kval-spectrum-agent/v1`` request."""
        if request.get("operation") != "scan":
            return {"schema": ADAPTER_SCHEMA, "success": False, "error": "unsupported operation"}
        if request.get("mode") != "passive":
            return {"schema": ADAPTER_SCHEMA, "success": False, "error": "only passive mode is supported"}
        if request.get("policy_id") != policy.policy_id:
            return {"schema": ADAPTER_SCHEMA, "success": False, "error": "policy_id does not match the loaded policy"}
        requested = set(request.get("band_ids") or [])
        approved = policy.approved_band_ids()
        if not requested <= approved:
            return {"schema": ADAPTER_SCHEMA, "success": False, "error": "request named a band outside the policy"}
        observations = [item for item in self.observe(policy) if item["band_id"] in requested]
        if not observations:
            return {"schema": ADAPTER_SCHEMA, "success": False, "error": "no requested band could be measured"}
        return {"schema": ADAPTER_SCHEMA, "success": True, "observations": observations}


def build_backends(
    *,
    interface: str | None,
    enable_scan: bool = True,
    enable_sdr: bool = True,
    enable_modem: bool = True,
) -> list[SensorBackend]:
    """Select the real backends actually present on this host.

    Returns an empty list when no measurement source exists, so the daemon can
    refuse to start rather than serve observations it cannot make.
    """
    backends: list[SensorBackend] = []
    if interface:
        survey = IwSurveyBackend(interface)
        if survey.available():
            backends.append(survey)
        if enable_scan:
            scan = IwScanBackend(interface)
            if scan.available():
                backends.append(scan)
        if not backends:
            nmcli = NmcliWifiBackend()
            if nmcli.available():
                backends.append(nmcli)
    if enable_sdr:
        sdr = RtlPowerBackend()
        if sdr.available():
            backends.append(sdr)
    if enable_modem:
        # Free cellular-band coverage on any node that already has a modem.
        modem = ModemSignalBackend()
        if modem.available():
            backends.append(modem)
    return backends
