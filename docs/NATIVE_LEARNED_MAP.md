# RadioMap v0 drives the trust gate: pre-registered protocol

Follows [NATIVE_GATE_RESULTS.md](NATIVE_GATE_RESULTS.md), where the gate was
*given* the true map error. Here the map and its uncertainty are learned.

## The learned map (`src/wirelessjev/native/radiomap.py`)
- **Survey:** robots drive for 60 s or 300 s before evaluation. Each second, every robot reports a capped scan: the 8 strongest BSSs, measured with fading and occasional blockage, at −90 dBm or above.
- **Model:** each BSS (site × band, 48 in total) gets a Bayesian linear model.
  - **Known geometry from the CAD:** log distance and rack rows crossed.
  - **Unknown shadowing:** 48 random Fourier features.
- **Fitting:**
  - Non-reports are censored bounds, fitted by Tobit EM.
  - A model-discrepancy variance, estimated on held-out robots (20%), is added to the posterior variance.
- **Use:** the planner uses the learned mean as its forecast and the learned std as σ_belief.

## Development (scene 400 only)
| Version | Calibration |
|---|---|
| Uncensored, no discrepancy term | Badly overconfident: ~1 dB predicted std against ~4.5 dB error; 90% intervals covered 13–27% |
| After both fixes | RMSE 3.5–3.9 dB, 90% coverage 0.72–0.83, bias −1.7 to −1.9 dB (the map learns *experienced* signal, which includes the fading mean) |

## Test (fresh scenes 520–525, 300 s evaluation)
- **Conditions:**
  - `survey60`
  - `survey300`
  - `survey300_2g_noisy` (+10 dB 2.4 GHz noise rise)
- **Arms:**
  - `mlo_split` (fixed rule)
  - `native_learned` (ungated)
  - `gated_learned` (z = 2)
  - `gated_learned_x1.5` (σ inflated by 1.5, a declared conservative variant)

**Hypotheses:**
- **L1:** `gated_learned` beats `mlo_split` on control misses in at least 5 of 6 scenes at `survey300`.
- **L2:** the same at `survey60`.
- **L3:** at `survey300_2g_noisy`, `gated_learned` beats ungated `native_learned` in at least 5 of 6 scenes.
- **L4:** mean 90%-interval coverage of the learned map on test scenes is at least 0.70.

**Reported, not tested:** RMSE, bias, worst-decile misses, p99 gap, airtime, and the ×1.5 variant.
