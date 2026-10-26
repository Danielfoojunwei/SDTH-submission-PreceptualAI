"""Hardware profile of the fleet car, taken from SmartCar.zip.

Every car in the fleet is the same hardware as the existing SmartCar:
- Yahboom Raspbot on a Raspberry Pi (hostname 'yahboom').
- Management radio: onboard Broadcom brcmfmac (MMC/SDIO bus), role HOTPOT,
  runs the Raspbot hotspot with fixed management address 192.168.1.11.
  WirelessJEV never selects, scans or modifies this radio (SmartCarGuard
  owns its recovery).
- Customer radio: Intel BE200 (PCI ID 8086:272b, iwlwifi) on PCIe, role WIFI,
  managed mode, normally wlan1. This is the only radio WirelessJEV acts on.
- Control surface on the car: SmartCarCmd.py, one JSON request per call on
  stdin: {"action": "status"} or {"action": "connect", "ssid", "bssid",
  "security", ...}. The car resolves credentials; NetworkManager (nmcli)
  activates a BSSID-pinned profile and the script verifies SSID/BSSID after.
- Scanning: SmartCatScan.py via NetworkManager rescan (~5 s), cached results,
  APs older than 15 s dropped, at most 8 APs reported.
- No motor, camera or navigation interface ships in SmartCar.zip, so motion
  exists only in the digital twin.

Measured values from the SmartCar notes and logs (one car, one site):
- Link: 5 GHz (5200 MHz), VHT 80 MHz, 2 spatial streams; rx VHT-MCS9
  780 Mbit/s, tx VHT-MCS3 234 Mbit/s at -38 dBm.
- RTT to the gateway: ~5.5 ms average, 0% loss over 20 pings.
- BSSID connect request -> verified success: ~4 s (17:25:29 -> 17:25:33).
- Legacy policy: reconnect when connected signal < -80 dBm; choose by SSID
  priority, then strongest signal; algorithm interval 5 s.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CarHardware:
    platform: str = "Yahboom Raspbot (Raspberry Pi)"
    management_radio: str = "brcmfmac SDIO onboard (HOTPOT, wlan0)"
    management_address: str = "192.168.1.11"
    customer_radio: str = "Intel BE200 PCIe 8086:272b (WIFI, wlan1)"
    customer_interface: str = "wlan1"
    spatial_streams: int = 2
    bands: tuple[str, ...] = ("2g", "5g")      # SSIDs in WiFi.json are 2.4/5 GHz
    max_visible_aps: int = 8
    scan_period_s: float = 5.0
    ap_max_age_s: float = 15.0
    status_period_s: float = 0.5
    connect_latency_s: float = 4.0             # measured, request -> verified
    weak_signal_dbm: float = -80.0             # legacy reconnect threshold
    policy_interval_s: float = 5.0             # legacy AlgorithmInterval
    gateway_rtt_ms: float = 5.5
    client_tx_power_dbm: float = 15.0          # assumption: not in the zip
    supported_actions: tuple[str, ...] = ("status", "connect")
    infra_channel_control: bool = False        # customer APs are not ours


RASPBOT = CarHardware()
