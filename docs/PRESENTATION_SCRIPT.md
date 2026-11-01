# Presenter script: WirelessJEV vs Private 5G

Speaker notes from the deck, in slide order. Deck: https://claude.ai/artifact/NNiER9MJbK5MARW2F9iH53

Current system: RadioMap + WirelessJEV. All results: 100-robot warehouse simulation; every system in a comparison ran the same scenes, routes, fading and blockage.

# Part: The problem fleets pay private 5G to solve, and what actually breaks robot links

## 1. Private-5G reliability for robot fleets, on the Wi-Fi you already have

The image behind the title is real output from our simulator: the 5 GHz coverage of the 100-robot test warehouse, with the steel rack rows, the sixteen AP sites in orange and the robots as white dots. Open with the promise in buyer terms: the reliability people buy private 5G for, delivered on the Wi-Fi they already have. Then say immediately that this is a simulation result, pre-registered and like-for-like, and that the deck ends with the hardware plan that turns it into a field result.

## 2. Fleets pay for a second network just to get four guarantees

**Why you should care:** If Wi-Fi delivers these four guarantees, you skip the second network entirely.

Frame the buyer’s reasoning fairly. Private 5G is bought for predictability, not speed: on-time commands, no stops while moving, clean air and an uptime SLA. The cost is a whole second network. Our proposition is simple: deliver the same four guarantees on the Wi-Fi the site already runs.

## 3. Robot links break when the only AP path dies, not from congestion

**Why you should care:** More spectrum or airtime won’t fix it. A second path through another AP does, and that is what we built.

We tagged every missed control packet in the simulator with what it was up against. Contention caused none of them: Wi-Fi retries absorb collisions. What kills packets is a moment when the robot’s only AP path is dead, from a deep fade, a person or forklift in the way, or a noisy band. Wi-Fi 8 and our one-radio version suffer that; private 5G less so; a second path through a different AP almost removes it. Source: development scene 400, 120 s, 100 robots, realistic collisions and interference.

# Part: Our answer: RadioMap and WirelessJEV, seen working, with every capability and its measured outcome

## 4. One system senses, decides and acts across the whole Wi-Fi stack

**Why you should care:** Runs on the Wi-Fi 7 network you already need: no spectrum, no core network, no AP changes.

Walk the numbers in order. One: robots drive for a minute and report ordinary capped scans. Two: RadioMap turns those into a map of every AP link, including what robots failed to hear, and attaches an error estimate to every spot. Three: WirelessJEV’s d3-lite JEV reads the operator’s mission in plain language. Four: it plans every robot’s AP, links and timing once a second. Five: the trust gate only lets it act where the map is confident, otherwise a safe fixed rule decides. Six: every action is verified exactly. Seven: on each robot the plan becomes actions at every layer, the key one being that a failed packet is retried through a second AP within ten milliseconds. Eight: the robots’ live measurements flow back and the map keeps learning.

## 5. When one AP path fades, the second path saves the packet

**Why you should care:** Your robots keep getting safety and motion commands through fades and blockages, instead of stopping.

This is one robot, robot 14, for twenty seconds of test scene 530, with the same hidden fading and blockage every system faced. The blue line is its best AP path, the orange its second path through a different AP. Below the dashed line a packet on that path is lost even at the most robust rate. Orange circles are moments the best path was down but the second was fine: those packets are rescued. Black crosses are the rare moments both were down. Across all 100 robots over 300 seconds, 84% of best-path drop-outs had a working second path.

## 6. One minute of ordinary driving maps your site’s radio

**Why you should care:** No survey project before go-live, and the map keeps updating as racks and stock move.

Left is the true signal map of one access point in the test warehouse, which no system ever sees. Right is RadioMap’s map after robots drove normally for sixty seconds, reporting only ordinary capped scans. The racks’ shadows and the coverage shape come through. Measured map error is about 3.5 to 3.8 dB; its uncertainty estimate is still somewhat overconfident, which we report on the limits slide.

## 7. Only the models that measured best made it into the product

**Why you should care:** You deploy what was proven best. What did not pay off stays in the lab.

Orange rows are WirelessJEV: the d3-lite JEV that reads the mission, the exact planner and the trust gate. Blue is RadioMap, the Bayesian radio map learned from robot scans that the product uses today; it is not RadioJEV. Grey rows did not earn a place: a cross-site prior added nothing once 100 robots survey, and RadioJEV as tested in E1–E5, a d3-lite JEV with radio tokens, lost to intent plus planning (0.107 vs 0.014 regret). The proposed RadioJEV, a JEPA encoder feeding radio tokens into a JEV, is not built; it would sit where RadioMap sits and only ships if it beats RadioMap under the same trust contract.

## 8. Each capability fixes one failure, with a measured result

**Why you should care:** Every claim maps to a number you can audit, not to a feature list.

Read across each row: capability, the technology behind it, and the number we measured in pre-registered simulation. The first row carries the headline. The others make it safe and steerable. The intent row is exploratory, from a single scene; the rest come from pre-registered campaigns on fresh scenes.

## 9. Fleet trust and learning, built on peer-reviewed research

**Why you should care:** One rogue robot or bad update can’t degrade your fleet, and a site’s data can be removed.

This is the research lineage: Bowen Shen, Feng Li and Kwok-Yan Lam’s published work on swarm spectrum access, reputation, unlearning, traffic prediction and spectrum pricing, reused as exact rules. Venues: swarm consensus WCNC 2025 and IEEE IoT-J 2025; reputation MSN 2024; unlearning ICAIIC 2025; trend-aware traffic IEEE IoT-J 2025; pricing IEEE TWC 2025. Unlearning is approximate, not certified. Admission is barely above chance and the efficiency levers barely move reliability; we show both.

# Part: What is new, and the like-for-like 100-robot test

## 10. Known building blocks, combined in a way no one ships today

**Why you should care:** You get 5G-class reliability without waiting for a new standard or new radios.

Be precise with a technical audience. We do not claim to have invented path diversity or radio maps. The new part is the combination, running on today’s Wi-Fi 7 clients, orchestrated across a fleet from plain-language intent, with a learned map whose own error decides when the AI is allowed to act, plus the measured comparison against 5G and Wi-Fi 8. This is a literature position, not a patent search.

## 11. Every system faced the same warehouse, routes, fades and blockages

**Why you should care:** A strict like-for-like test. Where it bends the rules, it bends them in 5G’s favour.

The image is the real simulated warehouse: coverage heat, steel rack rows, the sixteen AP sites and the robots. Each system replays identical routes and exactly the same random fading and blockage, generated from the same seeds and never visible to any system. The three conditions are applied identically to every Wi-Fi system. Private 5G is modelled generously: licensed spectrum untouched by Wi-Fi collisions or 2.4 GHz noise, perfect scheduling, 23 dBm devices and no inter-cell interference.

# Part: Results against private 5G and Wi-Fi 8, including where 5G still wins

## 12. 17× fewer missed commands than private 5G in realistic conditions

**Why you should care:** Fewer missed commands means fewer robot stops, on Wi-Fi alone, with no second network.

Each bar is measured over 3.6 million control packets on six fresh test scenes, with exact 95% intervals. In realistic conditions WirelessJEV with two AP paths missed 3.1 per million against 52.8 for the 30 ms-handover 5G model and 45 with make-before-break; the intervals do not overlap. In clean radio conditions the gap is larger. In the hardest condition 5G wins, which the next slides address directly. Note the light orange bar: without the second path we do not beat 5G.

## 13. Ahead on every number an SLA is written in

**Why you should care:** These are the terms you would put in a 5G contract, met on Wi-Fi.

Translate the result into contract language. Uptime: the share of robot-seconds with no missed command. Safety stops: time robots spend stopped because commands went silent for more than 200 ms. Longest gap at the 99th percentile. And the worst-served tenth of the fleet, because SLAs fail on the tail, not the average. All four favour our system in this test, and the pre-registered verdicts passed on all six scenes.

## 14. Per-packet diversity beats Wi-Fi 8’s seamless switching by 24×

**Why you should care:** No need to wait for Wi-Fi 8 or replace your APs: this works on Wi-Fi 7 now.

Wi-Fi 8’s multi-AP features are the strongest future competitor, so we modelled them generously: switching to any AP costs nothing. It still misses about 24 times more control packets than our two-path system on the same scene, because it picks an AP every 100 ms rather than per packet. These figures are from a development scene; the pre-registered fresh-scene test was still running when the deck was built.

## 15. In the most crowded, noisiest air, 5G’s licensed band still wins

**Why you should care:** We tell you where 5G is still the better buy: very dense, very noisy sites. Elsewhere, Wi-Fi wins.

Lead with the limit rather than letting a skeptic find it. In the hardest condition we tested, licensed spectrum wins because it never sees Wi-Fi contention or 2.4 GHz noise. We already have a lead on closing the gap, and the pre-registered test is running. For buyers, this is a clear rule of thumb about where each technology fits.

# Part: Why the numbers can be trusted: grounding, robustness and the full experiment trail

## 16. Built on standard simulators, with every assumption stress-tested

**Why you should care:** You can audit every input. Calibration on real robots is the next step, and it is planned.

Answer “is this simulation real?” directly. Left: the parts that come from standard external tools or from our actual hardware and code, plus the rigour: pass marks committed before each test run, and very large samples on scenes never used for tuning. Right: what we had to assume, and the range we swept for each. Nothing has yet been calibrated on the real car; the probes to do that are built and in the validation plan.

## 17. Change the assumptions and the ranking holds, with one exception

**Why you should care:** The advantage doesn’t hang on one lucky assumption, and we name the one case where it flips.

A skeptic will say the assumptions decide the answer, so we varied them. The ranking holds across map error, a learned map, a noisy 2.4 GHz band and no collisions. It flips in one place, heavy contention plus noise, shown in blue. Each row compares systems on the same scenes, routes, fading and blockage. Percentages are missed control packets; “/ M” is per million.

## 18. Six pre-registered tests passed and failed in public; the claims that remain survived

**Why you should care:** We publish our failures, so you can trust the results we keep.

We did not run one flattering experiment. Each test had its pass marks committed to git before its fresh test scenes ran, and every failure is kept. The failures shaped the product: route forecasting was not the lever, our map’s uncertainty was overconfident, a cross-site prior added nothing at fleet scale, and a single radio is not enough to beat 5G. The cross-AP test of the no-new-hardware routes was still running when the deck was built.

# Part: Hard questions, cost and what is not yet proven

## 19. The four questions a skeptic asks first, answered with evidence

**Why you should care:** The evidence was built to survive scrutiny, not to impress.

Take these one at a time and point back to the evidence slides. On the 5G objection, every simplification in our 5G model helps 5G. On tuning, mention the concrete case: a threshold tuned on a development scene looked better and then lost on fresh scenes, and we kept the pre-registered one anyway. On Wi-Fi 8, the comparison is from a development scene; the fresh-scene test is running.

## 20. And the four that come next

**Why you should care:** You get straight answers, including where our own ideas didn’t pay off.

The second-radio question is the one buyers press hardest: answer with the three routes and say which need hardware verification. On the AI question, the learning is in RadioMap and its uncertainty, which is what makes autonomous planning safe, while the d3-lite JEV turns mission text into priorities. We do not claim a learned controller beats exact planning.

## 21. The fewest new things to buy, and the best measured reliability

**Why you should care:** Lowest new spend: no spectrum, no core, no new radio sites, maybe no new robot hardware.

The cost argument is structural and needs no invented prices. Private 5G adds spectrum, a core network, new radio sites and a modem per robot. Wi-Fi 8 adds new APs and new clients that are not shipping yet. We add software to the Wi-Fi 7 network the site already needs, plus at most a cheap second client radio per robot, possibly none if the car probes confirm a second association on the existing card. Fill the price row with real vendor quotes before presenting externally.

## 22. Proven in simulation; the field proof is the next step

**Why you should care:** You know exactly what is proven, and exactly what the next phase will prove.

Draw the line yourself before someone else draws it. The left column is what the pre-registered campaigns support. Every item on the right maps to a step in the validation plan on the next slide.

# Part: Validation plan, bill of materials and the ask

## 23. Four steps from simulation to a field-proven claim

**Why you should care:** Steps 1 and 2 need only the robot we already have; the tooling is built and tested.

Each step de-risks the next. Step one is read-only and decides whether we need any new client radios. Step two measures the handover we have only assumed so far. Step three feeds real measurements back into the simulator and reruns every pre-registered test. Step four is the fleet trial, then a real head-to-head on a private 5G test site.

## 24. A small, mostly off-the-shelf list to prove it in the field

**Why you should care:** A field proof costs a few access points and robots, not a network build.

Required items make the 10-robot trial possible. Conditional items depend on step one of the plan: if the car probe shows the existing card can hold two client connections, we skip the adapters and the management rework. The private 5G site turns “beats a 5G model” into “beats 5G”. Prices need quotes; the full list is in docs/BOM.md.

## 25. Help us turn a simulated win into a field-proven one

Close on the ask. The results are strong and carefully earned; what remains is measurement on hardware. We need the equipment on the bill of materials, access to a private 5G test site for a real comparison, and one pilot floor.
