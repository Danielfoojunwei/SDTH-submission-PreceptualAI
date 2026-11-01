# Presenter script: current RadioMap + WirelessJEV system

Canonical evidence base: PR #7, commit `998fe3736a28d30cf04931e9ef182bda8cd4caf8`. This deck explains the current architecture and design rationale; it does not present a development timeline or version comparisons.

Files: [PowerPoint](slides/generated/wirelessjev_system_architecture.pptx), [PDF](slides/generated/wirelessjev_system_architecture.pdf), [browser deck](slides/generated/wirelessjev_system_architecture.html). All headline measurements are simulation evidence, with explicitly separate software-tool and field-validation boundaries.

## 1. RadioMap + WirelessJEV: coordinated warehouse connectivity

**Core insight:** A comprehensive feedback architecture connects radio state, intent, decisions and outcomes.

Open with the system, not the benchmark. We are building a warehouse connectivity controller that learns a site radio map, interprets the operator mission and chooses how robots use their links. The most important measured lever is another path through a different access point. All numerical comparisons in this deck are simulations. No physical robot fleet, Wi-Fi 8 product or private-5G installation has been qualified.

**Evidence:** SIMULATION + SOFTWARE PROTOTYPE
**Sources:** `docs/VERTICAL_STACK.md`, `docs/WIFI_VS_5G_RESULTS.md`

## 2. A robot needs commands on time—not just a fast link

**Core insight:** Reliability is a path-continuity problem before it is a peak-rate problem.

The slot simulator generates 600-byte control messages at 20 Hz and gives them a 50 ms deadline. Video and bulk traffic compete for airtime, but missing a short control deadline is a different failure from delivering fewer megabytes. Uptime here means robot-seconds without a missed command; safety-stop time is a model based on command silence, not a certified safety function. These definitions shape the architecture.

**Evidence:** MODELLED APPLICATION REQUIREMENTS
**Sources:** `src/wirelessjev/native/sim.py`, `docs/WIFI_VS_5G.md`

## 3. Each system requirement has a corresponding control mechanism

**Core insight:** Comprehensiveness means covering distinct failure modes—not adding unrelated features.

Explain the construction from requirements. Unknown coverage requires a map learned from scans. Different operator goals require mission interpretation and explicit knobs. Imperfect estimates require a confidence gate and reactive fallback. A transiently dead AP path requires recovery through another attachment. Uncertain physical outcomes require clocked measurement and an operator-controlled adapter. These are complementary mechanisms in the current architecture; none is presented as a historical version or an automatic guarantee.

**Evidence:** CURRENT IMPLEMENTATION DESIGN
**Sources:** `docs/VERTICAL_STACK.md`, `src/wirelessjev/native/sim.py`, `src/wirelessjev/native/radiomap.py`, `src/wirelessjev/handover.py`

## 4. RadioMap estimates; WirelessJEV decides; the executor recovers

**Core insight:** Clear interfaces keep uncertainty, decisions and physical outcomes from being confused.

RadioMap estimates the mean signal and uncertainty of each AP link from survey measurements and known geometry. WirelessJEV interprets the mission, compiles policy parameters and selects numerical radio actions under a confidence gate. The simulated executor handles retries, queues and delivery accounting. Supervised car tooling exposes a separate physical boundary. The connection between these components is the system: a map alone cannot recover a packet, and a retry rule alone cannot express an operator mission.

**Evidence:** CURRENT CANONICAL SYSTEM
**Sources:** `docs/VERTICAL_STACK.md`, `src/wirelessjev/native/radiomap.py`, `src/wirelessjev/native/intent_policy.py`, `src/wirelessjev/native/sim.py`

## 5. The system learns, plans, gates and closes the loop

**Core insight:** The map’s uncertainty controls where planning is allowed to act.

Follow the solid arrows: scans and site geometry feed RadioMap; the mission reader compiles weights into native policy knobs; the planner combines these with routes and lagged measurements; confidence gating chooses planned or reactive action; the simulated link executor retries and updates the map. Dashed boundaries identify the separate compositional proposal interface and the supervised car adapter. The native policy uses explicit numerical heuristics; it is not a formally verified global optimizer.

**Evidence:** EXECUTABLE SIMULATOR + SEPARATE TOOLS
**Sources:** `src/wirelessjev/native/world.py`, `src/wirelessjev/native/radiomap.py`, `src/wirelessjev/native/sim.py`, `src/wirelessjev/native/intent_policy.py`

## 6. Separate fast packet recovery from slower mission planning

**Core insight:** Language interpretation belongs outside the per-packet path.

The 10 ms slot handles retry and delivery accounting. Link measurements and link adaptation update on 100 ms blocks. Native planning runs once per second and online map updates every ten seconds. Mission interpretation runs when the mission changes. A robot-edge deployment split is an implementation target, not a measured placement result. These multiple cadences let us test the whole controller without putting a language model in every packet decision.

**Evidence:** EXECUTED SIMULATOR CADENCES
**Sources:** `src/wirelessjev/native/sim.py`, `src/wirelessjev/native/intent_flip.py`

## 7. Every policy sees the same hidden warehouse conditions

**Core insight:** Shared scenes isolate policy differences within the model—not reality error.

The physical layout is a 120 by 80 metre racked warehouse. Each site has modelled 2.4, 5 and 6 GHz links. Routes, mean propagation, fading and blockage are shared across compared policies, while hidden instantaneous radio state is not supplied to their planners. The main comparison uses six fresh scenes of 300 seconds. Pairing supports a controlled policy comparison, but it does not validate the building, device or interference model against hardware.

**Evidence:** CONTROLLED SIMULATION
**Sources:** `src/wirelessjev/native/world.py`, `src/wirelessjev/native/sim.py`, `docs/WIFI_VS_5G.md`

## 8. Standard link curves anchor a simplified slot model

**Core insight:** This cross-layer Python simulator is not full ns-3 Wi-Fi or a 3GPP NR stack.

The link lookup tables are actual archived ns-3 NIST packet-error curves for VHT80 and HT20, two streams and 1200-byte payloads. Six GHz is extrapolated by doubling the 5 GHz rate at a 160 MHz width, not by simulating a full EHT implementation. The slot model also uses those curves for smaller control packets and for its stylised cellular baseline. These approximations need calibration. A separate Sionna chain evaluates coded modulation; it is not a custom waveform running on stock Wi-Fi hardware.

**Evidence:** MODEL ASSUMPTIONS SHOWN
**Sources:** `src/wirelessjev/native/phy.py`, `src/wirelessjev/native/sim.py`, `artifacts/warehouse_v2/studies/wifi_link_curves.json`

## 9. A capped scan is evidence—not a list of invented measurements

**Core insight:** Correct censoring is essential before trusting any learned radio map.

The map uses per-BSS geometry coefficients and random Fourier features for local structure. It fits heard RSSI readings and treats omitted APs as censored evidence. A capped scan does not imply every unheard AP is below the receiver floor: the list may already be filled with stronger APs. The prototype also estimates a discrepancy term using held-out robot measurements. This is how the system converts incomplete sensing into a usable spatial estimate. No hardware survey has yet calibrated the pipeline.

**Evidence:** IMPLEMENTED BAYESIAN MAP
**Sources:** `src/wirelessjev/native/radiomap.py`, `docs/NATIVE_LEARNED_MAP.md`

## 10. One minute learns useful structure—but uncertainty undercovers

**Core insight:** More data improves mean error while structural uncertainty can become overconfident.

The map figure is an illustrative simulator rendering, not a field RF survey. The adjacent plots use the frozen learned-map campaign: a 60-second survey has mean error about 3.8 dB, falling to about 3.5 after 300 seconds. Nominal 90% interval coverage is only about 67% and 57%, respectively. The map is useful for ranking alternatives, but its uncertainty cannot be described as a calibrated failure probability.

**Evidence:** FROZEN HELD-OUT SIMULATION
**Sources:** `docs/NATIVE_LEARNED_MAP_RESULTS.md`, `artifacts/native/learned_campaign.json`

## 11. Mission text becomes explicit policy knobs

**Core insight:** The native mission adapter is a weight compiler; the richer contract tool is separate.

The heatmap uses actual cached d3-lite readings from four author-written missions. Native intent_policy maps those weights into roaming thresholds, load penalties, gate confidence, protected-channel penalties, video share and scarce second-radio allocation. The named robot can receive priority without retraining a classifier for that mission. This result is exploratory on one development scene. The separate compositional contract compiler supports explicit groups and floors, but its exact checks are not automatically proof of every native simulator action.

**Evidence:** ACTUAL MODEL INFERENCE / EXPLORATORY
**Sources:** `src/wirelessjev/native/intent_policy.py`, `artifacts/native/intent_flip.json`, `scripts/plan_warehouse_contract.py`

## 12. One planning step chooses AP, control link and data behavior

**Core insight:** Different traffic classes need different decisions over the same physical links.

Trace the native planner in code. For AP choice it scores the near-term predicted best-link signal and subtracts an AP load penalty. A gain threshold controls planned roaming, and the trust gate may defer to a reactive rule. For control, the policy chooses a link using a conservative minimum over the near-term forecast. For bulk it chooses predicted goodput; application timing can queue bulk for a better upcoming link. These are explicit numerical heuristics with distinct objectives, not a global exact-optimization claim.

**Evidence:** EXECUTABLE NATIVE POLICY
**Sources:** `src/wirelessjev/native/sim.py`, `src/wirelessjev/native/intent_policy.py`

## 13. A confidence gate prevents uncertain maps from steering freely

**Core insight:** The gate is a decision safeguard, not a certificate of calibrated safety.

The decision rule forms a gain-to-uncertainty ratio using the current and candidate AP map standard deviations. A high-enough ratio permits a planned roam. Otherwise a reactive rule can decide using lagged observed signal. The learned-map campaign shows fewer control misses than the fixed split baseline, including noisy 2.4 GHz. Uncertainty still undercovers, so the right interpretation is that its ranking is useful in this model, not that the gate proves a physical risk bound.

**Evidence:** FROZEN HELD-OUT SIMULATION
**Sources:** `src/wirelessjev/native/sim.py`, `docs/NATIVE_LEARNED_MAP_RESULTS.md`, `artifacts/native/learned_campaign.json`

## 14. A teardown and a within-connection roam are different paths

**Core insight:** Measure the interruption before claiming a fast-roam result on hardware.

SmartCarCmd activates a per-BSSID NetworkManager profile, which can tear the connection down and request a new DHCP lease. The new tool can inspect the active profile, collect journal stages and ping timing, and issue an operator-confirmed wpa_supplicant D-Bus roam within the existing connection. That path is tested against a fake car only. The chart compares assumed 4-second and 60-millisecond roam costs in simulation; it is not a measured SmartCar handover distribution. Management wlan0 remains outside the automatic control path.

**Evidence:** SOFTWARE TOOL / NO HARDWARE TRIAL
**Sources:** `src/wirelessjev/handover.py`, `src/wirelessjev/onecar.py`, `docs/ONE_CAR_TEST.md`, `docs/NATIVE_STACK_RESULTS.md`

## 15. MLO protects bands; a second AP path protects attachment

**Core insight:** Cross-AP diversity is the decisive new lever in the model.

Ordinary multi-link operation is modelled as several bands attached to one AP site. Blockage is shared across a robot-to-site path, so those links can fail together. A second client association to a different site provides another blockage path and allows make-before-break. The main successful comparison assumes a second client radio. Sharing one BE200 or using the onboard chip are unverified alternatives with separate protocols; standard Wi-Fi 7 support alone does not prove two simultaneous AP attachments.

**Evidence:** SIMULATED CAPABILITY / HARDWARE CONDITIONAL
**Sources:** `src/wirelessjev/native/sim.py`, `docs/WIFI_VS_5G_RESULTS.md`, `docs/CROSS_AP.md`

## 16. When one AP path fades, a second path can remain usable

**Core insight:** The reference line is a 99% single-attempt operating point—not certain packet loss.

This diagnostic is generated from the same test-scene model but is not a new pre-registered outcome. We select an illustrative robot window with many recovery opportunities and disclose that selection. A point below the SNR reference has lower than the selected single-attempt reliability target; retries can still deliver it. The alternative-path availability calculation is a channel-block proxy, not a count of rescued end-to-end control packets. The actual packet result comes from the frozen campaign.

**Evidence:** POST-HOC SIMULATOR ILLUSTRATION
**Sources:** `docs/figures/make_deck_figures.py`, `src/wirelessjev/native/sim.py`, `src/wirelessjev/native/phy.py`

## 17. Single-AP outages dominate the observed failure modes

**Core insight:** Do not generalise zero contention-attributed misses to every workload.

The stacked chart reproduces the documented 120-second, 100-robot diagnostic-scene breakdown. Retry chains absorb collisions at this particular load, while deep path fades and blockages remain. A second AP reduces the dominant failure mode. This is why adding scheduling or power controls moved the result only slightly. Attribution is generated by the simulator rules and the category precedence; it is not independent RF capture or proof that congestion cannot cause a miss elsewhere.

**Evidence:** EXPLORATORY SINGLE-SCENE DIAGNOSTIC
**Sources:** `docs/CROSS_AP.md`, `src/wirelessjev/native/sim.py`

## 18. The second path matters more than the extra knobs

**Core insight:** Show every lever’s measured contribution—including no measurable effect.

The full single-radio stack has about 294 misses per million, while the dual-radio version has about 3.1 in the main condition. Removing online learning or power politeness changes the dual-radio mean only to about 3.6. Removing cadence timing or adaptive video changes almost nothing, as does scheduled control. These findings distinguish an implemented capability from a capability responsible for the headline gain. They prevent the architecture diagram from becoming a list of equally important features.

**Evidence:** FROZEN HELD-OUT SIMULATION
**Sources:** `artifacts/native/wifi_vs_5g.json`, `docs/WIFI_VS_5G_RESULTS.md`

## 19. The command path makes uncertain outcomes explicit

**Core insight:** UNKNOWN means observe and reconcile—not automatically resend a radio-changing command.

The physical adapter is intentionally separate from the simulator. Fixed read-only probes inspect the customer interface, active profile and available methods. A supervised change uses typed confirmation and one request. Outcomes distinguish applied, not applied and unknown; an unknown result is reconciled by reading status, without blindly repeating the request. Management wlan0 is excluded. These controls are implemented and tested against a fake car. They do not substitute for the first real supervised hardware trial.

**Evidence:** SOFTWARE TOOL / FAKE-CAR TESTED
**Sources:** `src/wirelessjev/handover.py`, `src/wirelessjev/carlink.py`, `src/wirelessjev/onecar.py`, `docs/ONE_CAR_TEST.md`

## 20. Fleet state, decisions and evidence are inspectable

**Core insight:** Observability is a system capability—not a substitute for a live hardware trial.

This is an actual browser capture of the implemented Studio, not a fabricated product mockup. The fleet-map view plays recorded warehouse-twin data and lets an operator inspect robots, mission changes, links and utility. Other views expose decisions, execution outcomes and native research evidence. The screenshot is not live car telemetry or a replay of the new dual-radio packet campaign. It shows the operator-facing software boundary that accompanies the control modules.

**Evidence:** ACTUAL STUDIO SCREEN / RECORDED SIMULATION
**Sources:** `studio/app.js`, `docs/slides/assets/studio_fleet_map.png`, `scripts/check_research_deck_browser.mjs`

## 21. The cellular comparator is a model—not a private-5G network

**Core insight:** A simulated advantage does not establish commercial-5G superiority.

The model uses n78, 100 MHz, 23 dBm UEs, clean licensed spectrum, no inter-cell interference and ideal scheduled control access on the same sites. It includes 30-millisecond or zero-interruption handovers and retry-based delivery. It reuses the archived Wi-Fi link curves with rate and SNR scaling; it is not a calibrated 3GPP NR PHY, RAN stack or measured private network. Favorable assumptions are not a mathematical guarantee that every model bias favors 5G. A real comparison remains a validation step.

**Evidence:** STYLISED CELLULAR BASELINE
**Sources:** `docs/WIFI_VS_5G.md`, `src/wirelessjev/native/sim.py`

## 22. Compare the same control task under the same hidden conditions

**Core insight:** 3.6 million commands per arm still come from only six independent scene realisations.

The current primary comparison uses scenes 530 to 535, with 100 robots for 300 seconds and 20 control commands per second. That is 3.6 million commands per arm and condition. Every arm uses the same routes, fading and blockage for its scene. The declared conditions are clean radio, main interference and heavier collision plus 2.4 GHz noise. Control miss rate, robot-second availability, worst-decile robots and command gaps describe different aspects of continuity. Large packet counts do not remove scene correlation; the plots expose the actual paired scene results.

**Evidence:** FROZEN PAIRED SIMULATION
**Sources:** `docs/WIFI_VS_5G.md`, `src/wirelessjev/native/run_5g.py`, `artifacts/native/wifi_vs_5g.json`

## 23. Two AP paths reduce modelled misses by about 17×

**Core insight:** The headline depends on cross-AP diversity, not existing single-radio hardware.

Every dot is one of the six paired held-out scenes; the bars show their means. The dual-radio controller has about 3.1 misses per million compared with 52.8 for the 30-millisecond cellular model and 45 for its zero-interruption version. The existing single-radio version is about 294, which does not beat either cellular comparator. This chart uses one consistent campaign and interference condition. It demonstrates the specific contribution of an independent AP attachment within the declared model, not universal superiority over commercial cellular networks.

**Evidence:** FROZEN SIX-SCENE SIMULATION
**Sources:** `artifacts/native/wifi_vs_5g.json`, `docs/WIFI_VS_5G_RESULTS.md`

## 24. The same experiment measures the tail and interruption

**Core insight:** These are operational proxies—not a field uptime SLA or safety certification.

Read the four panels as complementary model outcomes. One-second availability is the share of robot-seconds with no missed command. The p99 gap is the 99th percentile of robots’ longest command gaps, not per-packet latency. Worst-decile performance prevents a fleet mean from hiding poorly served robots. Safety-stop time comes from the declared command-silence and restart model. The dual-radio results are favorable in the main condition, but none of these quantities is a physical fleet guarantee.

**Evidence:** FROZEN SIX-SCENE SIMULATION
**Sources:** `artifacts/native/wifi_vs_5g.json`, `docs/WIFI_VS_5G_RESULTS.md`

## 25. The cellular model wins in the hardest interference regime

**Core insight:** The supported claim has an operating envelope.

Under the declared stress condition, the dual-radio Wi-Fi controller has about 91 misses per million versus about 53 for the licensed-spectrum cellular model. In the clean and main conditions it does better. This boundary is part of the result, not an appendix caveat. The model does not support claiming that Wi-Fi replaces private 5G in arbitrary warehouses or under arbitrary contention.

**Evidence:** FROZEN SIX-SCENE SIMULATION
**Sources:** `artifacts/native/wifi_vs_5g.json`, `docs/WIFI_VS_5G_RESULTS.md`

## 26. Three hardware routes can expose a second AP path

**Core insight:** Probe capability first; do not assume Wi-Fi 7 means two usable associations.

Route one asks whether the existing BE200 can hold another station association and retune fast enough; support is unverified. Route two uses the onboard Wi-Fi chip, which would require moving management off the protected hotspot and accepting a weaker link. Route three adds a second client adapter, which is the assumed capability in the completed dual-radio campaign. The capability probe decides which paths exist on the actual car. Driver support, IP handling, power and AP configuration still need measurement.

**Evidence:** HARDWARE OPTIONS / NOT QUALIFIED
**Sources:** `docs/CROSS_AP.md`, `docs/BOM.md`, `docs/ONE_CAR_TEST.md`

## 27. Per-packet path recovery differs from 100 ms steering

**Core insight:** This comparison tests policy granularity—not a fundamental Wi-Fi 8 standard limit.

The diagnostic scene compares about 304 misses per million for a stylised seamless Wi-Fi 8 policy with 12.5 for a two-radio path-diverse policy. The distinguishing assumption is 100-millisecond AP selection versus packet-level recovery. This does not establish a general limit of the Wi-Fi 8 standard or a vendor implementation. The visual explains the mechanism; a fresh-scene protocol exists, but its result files are absent from the pinned snapshot, so the figure remains explicitly exploratory.

**Evidence:** EXPLORATORY MODEL COMPARISON
**Sources:** `docs/CROSS_AP.md`, `src/wirelessjev/native/sim.py`

## 28. Control, video and bulk share a radio—but not a policy

**Core insight:** Cross-layer control connects application deadlines to link and queue decisions.

Control is served first and can use robust retries or another AP path. Video is subject to an explicit deadline and can be limited to a chosen share of predicted goodput. Bulk can be deferred when the route forecast predicts a materially better link; it stays buffered rather than being discarded. The simulator also models airtime sharing and power politeness. These mechanisms make the system comprehensive, while the ablation slide honestly separates their implementation from the amount each contributes to the reliability headline.

**Evidence:** EXECUTED SIMULATOR CAPABILITY
**Sources:** `src/wirelessjev/native/sim.py`, `src/wirelessjev/native/intent_policy.py`, `docs/WIFI_VS_5G.md`

## 29. The system covers estimation, decisions, recovery and operations

**Core insight:** The connected simulation is comprehensive; the field-execution boundary remains measurable.

Use the matrix to locate each current capability. Site estimation, mission compilation, AP planning, link mapping, retries, traffic timing and telemetry feedback execute in the simulation. The richer contract proposal interface and supervised handover diagnostics exist as separate software tools. No row is presented as field-qualified, and the native controller is not described as a formally verified global optimizer. This is a coverage map of the current system, not a roadmap or comparison between historical versions.

**Evidence:** CURRENT IMPLEMENTATION TRACEABILITY
**Sources:** `docs/VERTICAL_STACK.md`, `src/wirelessjev/native`, `scripts/plan_warehouse_contract.py`, `src/wirelessjev/handover.py`

## 30. Four evidence gates turn the model into a field claim

**Core insight:** Real hardware is the next evidence source—not another favorable simulation.

Start with read-only probes on the customer radio and preserve management access. Next measure supervised association, authentication, IP readiness, first packet, scan interruption and power. Then calibrate the simulator with those measurements and rerun frozen comparisons. Finally run a small fleet before the 100-robot deployment claim, and compare with an actual private-5G test network on the same route. Missing stage times remain missing; they are not inferred from request completion.

**Evidence:** VALIDATION PLAN / NOT EXECUTED
**Sources:** `docs/ONE_CAR_TEST.md`, `docs/BOM.md`, `docs/WIFI_VS_5G_RESULTS.md`

## 31. A bounded pilot needs equipment—not invented prices

**Core insight:** The second-radio bill depends on the first capability probe.

The source bill requests four Wi-Fi 7 APs and nine additional matching SmartCars to bring the milestone fleet to ten. Twelve client adapters include two spares, but their purchase is conditional on probing the existing card and onboard option. It also lists ten out-of-band management paths, two monitor adapters and a laptop, two power meters, and access to a private-5G test network with modems. We make no dollar savings claim before obtaining vendor quotes.

**Evidence:** BILL OF MATERIALS / QUOTES PENDING
**Sources:** `docs/BOM.md`

## 32. Validate the paths, calibrate the model, then scale

**Core insight:** Current system: RadioMap + WirelessJEV. Next milestone: measured warehouse continuity.

Close on the architectural lesson: site learning helps planning, uncertainty limits autonomy, and independent AP paths address the dominant modelled outage. The completed comparisons support a bounded simulation claim, including a stress case where the cellular model wins. They do not support a universal foundation model, a real-5G win or a physical 100-car guarantee. The ask is hardware access, independent measurement and a partner pilot that can falsify the assumptions.

**Evidence:** BOUNDED CLAIM + NEXT MILESTONE
**Sources:** `docs/WIFI_VS_5G_RESULTS.md`, `docs/BOM.md`, `docs/VERTICAL_STACK.md`
