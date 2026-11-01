# Trust-gated planning and layered safety broadcast: pre-registered protocol

Follows [NATIVE_STACK_RESULTS.md](NATIVE_STACK_RESULTS.md). That campaign found
two things:
- At σ_map = 8 dB, planned roaming is worse than a fixed rule.
- Layered modulation helps little where retries exist.

This protocol tests the two follow-ups. Development uses scenes 400–401.
Tests use **new scenes 510–515**, so no scene whose results were already
seen is reused.

## A. Trust gate (WirelessJEV knows when not to trust RadioJEV)
- **Map error varies in space.** The error std is σ_map × m(x, y) per site-band, where m is a smooth log-normal field with RMS 1. Some areas are well known; others are not.
- **The planner sees its own uncertainty.** Its posterior std at each position is σ_belief = scale × σ_map × m(x, y). The default scale is 1 (calibrated).
- **The gate:**
  - **Confident:** a planned roam from the current site to the best-scoring site goes ahead only if gain ≥ 5 dB *and* gain / √(σ_cur² + σ_best²) ≥ 2.
  - **Not confident:** the robot falls back to the reactive roam rule (−75 dBm, 8 dB better, 802.11r cost), as in `mlo_split`.
- **Arms:**
  - `mlo_split`
  - `native` (ungated)
  - `native_gated`
  - `native_gated_overconfident` (scale 0.5, so it believes the map is twice as good as it is)

**Hypotheses:**
- **G1:** at mean σ_map = 8 dB, `native_gated` has fewer control misses than `mlo_split` in at least 5 of 6 scenes. (Ungated `native` lost 6/6 at 8 dB.)
- **G2:** at mean σ_map = 4 dB, `native_gated` both:
  - stays within 1.2× of ungated `native`'s misses in at least 5 of 6 scenes;
  - beats `mlo_split` in at least 5 of 6.
- **G3 (exploratory):** report how much the overconfident gate degrades.

## B. Fleet safety broadcast (layered modulation where there are no retries)
- **The event:** each AP sends a fleet-wide safety message every second. A robot must decode it within 50 ms on any link of its AP.
- **Channel:** downlink SNR from the same truth, fading and blockage as the slot simulator. Robots are attached to their strongest site at each moment, so the comparison does not depend on any roam policy.
- **Schemes:**
  - `multicast_mcs0`: one group-addressed frame at MCS 0 on each of the 3 links, with no acknowledgement or retry (the standard behaviour).
  - `multicast_x3`: three repeats per link.
  - `layered_f`: the message rides the base layer of every downlink frame the AP sends, using the Sionna α = 3 critical-bit curve (AWGN, applied at the instantaneous faded SNR). A frame is present on a link in a given slot with probability f ∈ {0.2, 0.5, 1.0}. No extra airtime.
- **Hypothesis B1:** `layered_0.5` misses no more often than `multicast_mcs0`, at zero added airtime, in at least 5 of 6 scenes.
- **Also reported:** each scheme's added airtime per AP.

## C. Pre-registered sensitivity, added before any test run
The base twin models 2.4 GHz scanner occupancy as lost airtime only, with no
rise in interference plus noise. The fixed rule's robustness depends on a
clean 2.4 GHz link. All part A arms are rerun at mean σ_map = 4 dB with a
+10 dB 2.4 GHz noise rise (`2g_noisy`). Part B is rerun with the same rise
applied to downlink SNR.

**Exploratory claim to check:** native's advantage over `mlo_split` grows when
2.4 GHz is noisy.

## D. Arm added after development, before any test run
On development scene 400, z = 2 was too cautious at σ = 4 dB, and the
overconfident variant did better everywhere. `native_gated_z1` (z = 1,
calibrated) is added as a development-informed arm. G1 and G2 are still
judged on the pre-registered z = 2 arm; z = 1 is reported separately and
labelled as tuned on development scenes.
