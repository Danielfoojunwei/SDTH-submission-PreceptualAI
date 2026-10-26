# WirelessJEV — milestone 1

Mission-conditioned Wi-Fi coordination for a 10-car Raspbot fleet. A frozen open System One model chooses the fleet plan. Exact numerical oracles teach and score it. A swarm protocol shares what the cars learn.

Milestone 1 runs entirely in a digital twin. Nothing here touches a car, a radio or a network.

## Hardware it targets (from SmartCar.zip)
Every car is the same hardware as the existing SmartCar (`src/wirelessjev/hardware.py`):
- **Platform:** a Yahboom Raspbot on a Raspberry Pi.
- **Management radio:** the onboard brcmfmac SDIO radio (`wlan0`) runs the management hotspot at 192.168.1.11. WirelessJEV never scans, selects or modifies it; SmartCarGuard owns its recovery.
- **Customer radio:** an Intel BE200 on PCIe (`8086:272b`, `wlan1`). It is the only radio acted on, and the only action is choosing a BSSID.
- **Wire format:** exactly the existing `SmartCarCmd.py` JSON: `status`, and `connect` with ssid/bssid/security plus the Wi-Fi password, which SmartCarCmd requires, just as the PC app sends it. Plans, receipts and logs never contain the password; the real transport (`carlink.py`) injects it from the operator's WiFi.json at send time.
- **Measured anchors:**
  - about 4 s per BSSID switch;
  - 5 s policy interval;
  - -80 dBm legacy reconnect threshold;
  - VHT 80 MHz with 2 spatial streams;
  - at most 8 visible APs.
- Customer APs are not ours, so AP channel planning is modelled but disabled.

## Architecture
1. **Numeric layer**: digital twin (`twin/`), exact evaluator and paper reward families (`oracle.py`), teachers (`teachers/`), and the candidate generator with hard filters (`candidates.py`). It produces up to 7 complete 10-car plans plus DEFER, each described by its consequences.
2. **System One layer**: frozen `vllm-sr/d3-lite@03560c2` (Apache-2.0) in `policy.py`. It picks a plan from the mission text. In the default mode, each option is scored separately by a yes/no question.
3. **Distillation**: one 1024-parameter readout direction (backbone frozen), distilled from the exact oracle (`distill.py`). No language-model teacher is used.
4. **Swarm**: validated merge with a signed log, plus reputation/quarantine and two-tier priors (`swarm/`).
5. **Traffic**: d3-lite typed labels plus frozen Chronos-Bolt forecasts (`traffic.py`).

How each Bowen Shen / Feng Li / Kwok-Yan Lam paper is used is in [docs/PAPER_MAP.md](docs/PAPER_MAP.md).

## 100-robot AGV/AMR warehouse
`twin/warehouse.py`:
- 120 × 80 m racked hall, with rack loss per row crossed;
- 16 dual-band ceiling AP sites on a fixed customer channel plan;
- scanner-occupied channels;
- 100 route-planned AMRs on the same SmartCar hardware.

Plans are scored over the robots' planned positions for the next 20 s.

```bash
~/.local/bin/uv run python -m wirelessjev.bench_warehouse --scenes 50 --out artifacts/warehouse
```
```bash
~/.local/bin/uv run python -m wirelessjev.closedloop --warehouse --total 600 --adapter artifacts/warehouse/adapters/warehouse --out artifacts/closedloop_warehouse.json
```

## Intent-to-objective WirelessJEV (fast loop)
d3-lite reads the mission text into objective weights through typed score questions (`intent.py`). This happens once per mission change and is cached; the score readout is distilled.

An exact batched planner (`fastplan.py`) then optimises those weights every step. It scores every robot × BSSID move in one batch (numpy or torch, CPU or GPU) and accepts only exactly verified moves.

Results on the 100-robot warehouse:
- **Plan gap to the best known plan:** 0.016, versus 0.179 for the previous option-wise choice.
- **Held-out paraphrase shortlist regret:** 0.027.
- **Decision latency:** about 134 ms, versus about 15 s before.
```bash
~/.local/bin/uv run python -m wirelessjev.bench_intent --out artifacts/intent
```
```bash
~/.local/bin/uv run python -m wirelessjev.closedloop --warehouse --policies wirelessjev_fast --intent-adapter artifacts/intent/adapters/intent --out artifacts/closedloop_warehouse_fast.json
```

## Studio (front end)
A static app in `studio/` that shows only the exported artifacts. Its sections are overview, fleet map replay, decisions, evidence, swarm & trust, execution, research map, and hardware & data.
```bash
~/.local/bin/uv run python -m wirelessjev.studio_data
```
```bash
python3 -m http.server 8766 --bind 127.0.0.1 --directory studio
```
Then open http://localhost:8766.

## Run
```bash
~/.local/bin/uv run pytest -q tests
```
```bash
~/.local/bin/uv run python -m wirelessjev.bench --scenes 80 --out artifacts/bench
```
```bash
~/.local/bin/uv run python -m wirelessjev.closedloop --adapter artifacts/bench/adapters/fleet
```
```bash
~/.local/bin/uv run python -m wirelessjev.report
```

## Boundaries
- No motor, camera or navigation control: SmartCar.zip ships no such interface.
- No OCUDU, 5G NR or custom-PHY execution. Those need Linux plus SDR hardware.
- d3-lite takes about 1–3 s per decision on the M5 GPU. That suits the 5 s policy interval. Anything sub-10 ms stays conventional.
- Real-car tooling: `python -m wirelessjev.onecar` (see [docs/ONE_CAR_TEST.md](docs/ONE_CAR_TEST.md)) does read-only status/scan and one operator-confirmed BSSID switch, with the host key pinned. The fleet `SSHTransport` in `execution.py` still refuses to send.
