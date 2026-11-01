# RadioMap + WirelessJEV: ten-slide presenter notes

Schematic 3D models; all reported performance comes from simulation. No physical fleet is qualified.

## 1. Connect the whole warehouse.

The current system combines a learned site radio map, mission interpretation, numerical planning, confidence gating and cross-AP packet recovery. The scene is a schematic 3D model based on the warehouse layout, not a photograph or measured field deployment. All reported performance comes from simulation. The existing hardware and supervised tools have not qualified a physical fleet.

Sources: `docs/VERTICAL_STACK.md`, `src/wirelessjev/native/world.py`, `docs/WIFI_VS_5G_RESULTS.md`

## 2. One system. Three loops.

Read the exploded model from the warehouse up through radio state to decision and execution. RadioMap estimates mean signal and uncertainty from scans and geometry. WirelessJEV interprets the mission and selects explicit numerical actions; its confidence gate can defer to a reactive rule. The executor models retry chains, queues and delivery, with link measurements feeding map updates. A separate supervised vehicle adapter exposes the physical boundary. This is not a formally verified global optimizer or a live field deployment.

Sources: `src/wirelessjev/native/radiomap.py`, `src/wirelessjev/native/intent_policy.py`, `src/wirelessjev/native/sim.py`

## 3. Make radio state visible.

The elevated surface encodes an actual predicted RSSI grid from the RadioMap model for an illustrative scene. Height and color are visual encodings of signal, not a physical RF surface. The 3.8 dB number is the separate six-scene learned-map campaign mean after a 60-second survey. Nominal 90% interval coverage was about 67%, so the uncertainty is useful for ranking but overconfident. Omitted scan entries are treated as censored evidence, not invented floor measurements.

Sources: `src/wirelessjev/native/radiomap.py`, `artifacts/native/learned_campaign.json`, `docs/NATIVE_LEARNED_MAP_RESULTS.md`

## 4. The mission sets the policy.

The input example is the actual cached mission to protect amr-017. d3-lite readings are compiled into explicit native policy parameters: roam gain, load penalty, gate confidence, channel preference, video share and focus-based allocation. The native adapter is a weight compiler. Its evidence is exploratory on one development scene; the richer compositional contract interface remains a separate tool. The model is a conceptual view of these implemented interfaces, not a depiction of language inference occurring on every packet.

Sources: `artifacts/native/intent_flip.json`, `src/wirelessjev/native/intent_policy.py`

## 5. Confidence controls autonomy.

The confidence gate compares predicted AP gain with uncertainty at the current and candidate attachments. It permits planned action only above its threshold, otherwise using the reactive rule. In the learned-map 60-second-survey campaign, gated control misses averaged about 0.013% versus 0.068% for the fixed split rule. This is simulation evidence; the posterior still undercovers. The gate does not certify calibrated physical failure probabilities or safety.

Sources: `src/wirelessjev/native/sim.py`, `artifacts/native/learned_campaign.json`

## 6. Recover through a different AP.

The model shows a blockage on one AP path and a second attachment through a different AP. Ordinary multi-link operation at one AP site does not provide the same geometric diversity because blockage is shared across bands for that robot-to-site path. The completed successful comparison assumes a second client radio. Two associations on one BE200 and reuse of the onboard chip are unverified hardware options, not capabilities established by this image. The scene is illustrative, not a recorded RF outage.

Sources: `docs/WIFI_VS_5G_RESULTS.md`, `src/wirelessjev/native/sim.py`, `docs/CROSS_AP.md`

## 7. Coordinate every layer.

The exploded stack represents application demand, queues, traffic-to-link mapping, retry behavior and AP attachment. Control is prioritized; video may be limited to a share of predicted goodput; bulk can remain buffered while waiting for a better link. Numerical AP planning and map updates run more slowly than packet recovery. The stack is a model of implemented simulator behavior and separate tools; it does not imply a deployed QUIC controller, custom stock-Wi-Fi waveform or formally verified global planner.

Sources: `src/wirelessjev/native/sim.py`, `src/wirelessjev/native/intent_policy.py`, `docs/VERTICAL_STACK.md`

## 8. 17× fewer missed commands.

The numbers are calculated from the immutable six-scene main-condition comparison: 3.0556 and 52.7778 misses per million, rounded to 3.1 and 52.8. The ratio is about 17.3. Each arm contributes 3.6 million control commands, but they come from only six paired scene realisations with correlated fading and blockage. The cellular comparator uses scaled link curves and idealized access, not a full 3GPP stack or a real private network. This result does not qualify existing single-radio hardware or commercial-5G superiority.

Sources: `artifacts/native/wifi_vs_5g.json`, `docs/WIFI_VS_5G.md`, `docs/WIFI_VS_5G_RESULTS.md`

## 9. The boundary is visible.

These three dioramas represent the three conditions actually tested. They are not an interpolated operating surface. The dual-radio Wi-Fi arm has approximately 0.6, 3.1 and 91.4 misses per million in clean, main and stress cases. The cellular baseline remains approximately 52.8 in all three because its licensed channel is not affected by the declared Wi-Fi contention and 2.4 GHz noise. The stress condition reverses the ranking. Real radio, interference and second-radio behavior still need calibration.

Sources: `artifacts/native/wifi_vs_5g.json`, `docs/WIFI_VS_5G_RESULTS.md`

## 10. Measure it. Then scale it.

The final model is a schematic validation aisle with measurement equipment and staged tests. Start with read-only capability probes, including whether the existing card can support a second association. Measure supervised handover stages, first-packet interruption, scans and power. Calibrate the simulator and replay frozen comparisons. Then qualify a small fleet before a 100-robot field claim and perform an actual private-5G comparison on the same route. No physical trial, equipment quote or safety guarantee is implied by this model.

Sources: `docs/ONE_CAR_TEST.md`, `docs/BOM.md`, `src/wirelessjev/handover.py`
