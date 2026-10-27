# Handoff: what is not done (2026-10-27)

All finished work is on `main`. This lists what is stopped, open, stale, waiting on hardware, or only proposed.

## 1. Stopped mid-way (restart needed)
**RadioJEV E1–E5 full run** (`artifacts/radiojev/run_all.sh`). It was stopped on request.

- **Done:** E1 `radio_lora`, saved in `artifacts/radiojev/e1_radio_lora.json`. Test regret is 0.093 ± 0.022 and new-mission regret 0.074 ± 0.020, at ~3.4–4 s per decision. That's weaker than the mission reader + fast planner (0.027 at ~134 ms).
- **Lost:** E1 `radio_frozen` was about halfway through, with no checkpoint, so it must restart from scratch.
- **Not started:**
  - E1 `intent`
  - E2 (`radio_q`, `e1_zero`, `intent`)
  - E3 multitask
  - E4/E5 swarm + unlearning
- **To resume:** `run_all.sh` reruns the smoke tests and `radio_lora` first, so it needs editing to skip both.
  - Estimated time: 1–2 days on the M5, plus idle-waits for other jobs.
  - Afterwards: write up the results, update the Studio and paper, and commit.

## 2. Done, with open issues
**E7 environment learning** (`src/wirelessjev/envlearn.py`, `artifacts/envlearn/report.json`)

- **Map error rises with survey time:** 5.0 dB after 1 min, 7.9 dB after 15 min, while plan regret improves. That's likely a calibration or Tobit-refinement bug. Fix it before claiming more data helps.
- **Distant interference paths are still about 10 dB off,** because about 77% are unmeasured.

## 3. Out of date
| Item | Missing |
|---|---|
| Studio (https://claude.ai/artifact/NiohoyLYJsy5vEKE5BBswC) | E7 and E1 results. Regenerate the bundle with `studio_data.py` and republish to the same URL. |
| `README.md` | E7, the capability probe, RadioJEV status |
| `docs/PAPER_PROPOSAL.md` | E7 results. Also weaken the RadioJEV claim to match the E1 result. |

## 4. Waiting on real hardware (operator runs these)
- **One-car test** (`docs/ONE_CAR_TEST.md`): preflight, read-only status, the capability probe (step 1b), scan, proposal, one supervised BSSID switch, then switching back. All of it is tested against a simulated car; none of it has run on the real one.
- **Change the car's SSH password.** It is the vendor factory default.
- **Proposed, not built:** timestamp each stage of the switch (connect, authentication, DHCP, first packet) to find where the 4 s handover cost goes.

## 5. Proposed, not started
**Recommended next:**
1. **Stage 0 PHY realism:** Sionna link curves replace the ideal signal-to-rate table, plus a rate-margin experiment. Optionally, Sionna RT ray-traced ground truth for E7.
2. **Route-aware link adaptation:** choose rate, width, band and link ahead of time from planned positions plus the radio map.
3. **E8, sensed-input RadioJEV:** busy time, FTM ranging, Bluetooth relay, stock levels, scanning at stops, and map uncertainty as model inputs.

**Further:**
- **Wi-Fi 7 twin mode:** multi-link operation (MLO), 320 MHz, puncturing; benchmark against default MLO behaviour.
- **Joint motion and radio planning:** waiting, choosing routes, uploading at stops.
- **Application, transport and middleware co-planning:** video bitrate, message priorities, pre-warning the transport before switches.
- **Connectivity-aware safety limits and fleet task assignment.**
- **Protocol-agnostic link graph,** with Bluetooth as the second protocol on the SmartCar.
- **Busy-time waterfall as an image vs as numeric tokens.**
- **Embodied AI:**
  - a SmartCar stand-in experiment (camera to an edge policy, with an on-board fallback);
  - a humanoid class in the twin (body blockage, crowds, compute-tier decision).
- **Hardware-gated:**
  - a CSI check on the BE200;
  - an SDR spectrum sentinel;
  - an openwifi SDR lab for learned PHY;
  - a srsRAN/OAI 5G xApp.

## 6. From the original plan, never done
- **Clone the didjital R6.1 branch.** No access, so the project was built standalone.
- **E6.** Excluded by request.
- **Multi-car trial.** It needs the other nine cars.

## 7. Notes
- **Kept out of git on purpose:** `artifacts/radiojev/*.pkl` data caches and run logs, which are rebuilt automatically. Also `artifacts/onecar/` and any `WiFi.json`.
- **Security rules for the car stay in force:**
  - never commit SmartCar passwords;
  - never touch `wlan0` or 192.168.1.11;
  - never resend a connect whose outcome is unknown.
