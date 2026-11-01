# Presenter script: WirelessJEV vs Private 5G

Speaker notes from the deck, in slide order. Deck: https://claude.ai/artifact/NNiER9MJbK5MARW2F9iH53

All results: 100-robot warehouse simulation; every system in a table ran the same scenes, routes, fading and blockage.

# Part: The problem fleets pay private 5G to solve, and what actually breaks their links

## 1. Wi-Fi that out-performs private 5G for indoor robot fleets

Open with the claim, then immediately qualify it: everything you will see is simulation, every campaign was pre-registered before its test scenes ran, and the comparison systems are modelled generously. The point of this deck is to show why the result is credible and exactly what hardware test turns it into a measured fact.

## 2. Robot fleets buy private 5G for four guarantees Wi-Fi has not offered

Frame the buyer’s reasoning fairly. Private 5G is not bought for speed; it is bought for predictability. Everything that follows tests Wi-Fi against exactly these four guarantees, using the metrics a 5G SLA would be written in: on-time control delivery, uptime, worst-robot tails and safety stops.

## 3. What actually breaks a robot’s link: a dead AP path, never contention

This slide explains why the system looks the way it does. We instrumented the simulator so every missed control packet records what it was up against. Contention caused zero misses: Wi-Fi’s retries absorb collisions. The misses come from moments when the single AP path is dead. Wi-Fi 8 and our one-radio system both suffer those; 5G suffers them less thanks to clean spectrum and higher power; a second path through another AP almost eliminates them. This is development-scene evidence that led to the pre-registered campaigns.

# Part: Our answer: the system, the models inside it and every capability with its measured outcome

## 4. Our answer: one system that senses, decides and acts at every layer

Walk the numbers in order. One: robots drive for a minute and report ordinary capped scans. Two: RadioJEV turns those into a map of every AP link, including what robots failed to hear, and attaches an error estimate to every spot. Three: WirelessJEV reads the operator’s mission in plain language. Four: it plans every robot’s AP, links and timing once a second. Five: the trust gate only lets it act where the map is confident, otherwise a safe fixed rule decides. Six: every action is verified exactly. Seven: on each robot the plan becomes actions at every layer, the key one being that a failed packet is retried through a second AP within ten milliseconds. Eight: the robots’ live measurements flow back and the map keeps learning.

## 5. The models inside, how we built them, and what they measured

This is what is actually inside. The WirelessJEV side is a frozen open decision model reading the mission, an exact planner fast enough to replan every second, and the trust gate with an exact verifier. The RadioJEV side is the learned radio map that the product uses today. We also show two RadioJEV models that did not earn a place: a cross-site prior that added nothing at fleet scale, and an end-to-end radio-token model that lost to intent plus planning. Keeping them on the slide is part of why the rest is credible.

## 6. Capabilities: reliability core

Read across each row: what the system can do, what technology makes it possible, and the number we measured. The first row carries the headline. The rest are what make it safe and steerable: separating control from bulk traffic, roaming between packets rather than mid-packet, refusing to act where the map is unsure, learning the map from ordinary driving, and letting the operator’s words decide which robot gets protected.

## 7. Capabilities: fleet intelligence and trust, built on published research

This slide is the research lineage. The trust and fleet-learning mechanisms come straight from Bowen Shen, Feng Li and Kwok-Yan Lam’s published work on swarm spectrum access, reputation, unlearning, traffic prediction and spectrum pricing. We reuse their formulations as exact rules and measure what they buy us. Note the honest weak spots: admission is barely above chance, and the efficiency levers do little for reliability at this load.

# Part: What is new, and how we tested it fairly in one shared environment

## 8. What exists already, and what we believe is new

Be precise here, because a technical audience will know 5G duplication and Wi-Fi 8. We do not claim to have invented path diversity. We claim a system that delivers it on today’s Wi-Fi clients, orchestrated across a fleet from plain-language intent, with a learned map whose uncertainty decides when the system acts, and we claim the measured comparison. The footer is deliberate: this is a literature position, not a freedom-to-operate opinion.

## 9. Every system faces the same warehouse, the same fades, the same blockages

The fairness argument: all systems face exactly the same site, routes, fading draws and blockage events, generated from the same random streams, and none of them can see those hidden processes. The rigour argument: hypotheses and pass marks were committed to git before the test scenes ran, the tuning happened only on separate development scenes, and the sample sizes are large: 3.6 million control packets per system per condition at 100 robots.

## 10. The test environment: one warehouse, identical for every system

Before any result, show the arena. This is the 100-robot warehouse every system was tested in: eleven rows of steel racks, sixteen AP sites each with 2.4, 5 and 6 GHz links, and a hundred robots on real pick-and-drop routes. The key point for fairness: each system replays the same routes and exactly the same random fading and blockage, generated from the same seeds and hidden from every system. The three conditions on the right are applied identically to every Wi-Fi system; private 5G uses licensed spectrum, so Wi-Fi collisions and 2.4 GHz noise do not touch it, which is generous to 5G.

## 11. What the simulator rests on, and what we stress because it is assumed

This slide answers “is your simulation real?” before anyone asks. Left: the parts that come from established external models or from our actual hardware and code. Right: the parts we had to assume, and for each one the range we deliberately swept in pre-registered runs. The footer is the honest limit: no part has yet been calibrated against measurements on the real car. That is exactly what the validation plan and the bill of materials fix.

# Part: Results against private 5G and Wi-Fi 8 in the same 100-robot warehouse

## 12. 100 robots: 17× fewer missed control packets than private 5G

This is the headline. On six fresh scenes with realistic collisions and interference, WirelessJEV with two AP paths missed 3.1 control packets per million; the generous private 5G model missed 52.8, or 45 with make-before-break handover. The confidence intervals do not overlap. Uptime, safety stops, the worst gaps and the worst-served robots all favour our system, and the verdicts passed on every scene. Note the one-radio bar: without the second path we do not beat 5G, which is why the second path is the core of the system.

## 13. Across radio conditions: ahead in two, behind in the hardest

Be upfront here. In clean and realistic conditions we lead by one to two orders of magnitude. In the hardest condition, heavy contention plus a noisy 2.4 GHz band, the generous 5G model wins, because licensed spectrum does not see Wi-Fi contention at all. We are testing a lead from a development scene that some of our own levers hurt under stress; that campaign is running and its result will be added.

## 14. Against Wi-Fi 8: per-packet diversity beats seamless switching

Wi-Fi 8’s multi-AP features are the strongest future competitor, so we modelled them generously: switching to any AP costs nothing. It still misses about 25 times more control packets than our two-path system on the same scene, because it chooses an AP every 100 ms rather than per packet. Be clear that these numbers are from a development scene; the pre-registered fresh-scene version was still running when the deck was built.

# Part: Does it hold up: robustness and the full experiment trail, failures included

## 15. Change the assumptions, and the ranking mostly holds

A skeptic will say our assumptions decide the answer. So we varied them. The ranking holds across map error, a learned map, a noisy 2.4 GHz band and different collision levels. It flips in one place, which we show in blue: heavy unlicensed contention plus a noisy 2.4 GHz band. That is private 5G’s home ground, and we say so. The last row is a reminder that the single biggest real-world lever is the roam time, which is why we built the car tooling to measure it.

## 16. Seven pre-registered campaigns, failures kept in the record

This is the slide that should build trust. We did not run one flattering experiment. We ran seven, each with pass marks frozen in advance, and we kept every failure. The failures shaped the system: H4 told us route forecasting was not the lever; L4 told us our uncertainty was overconfident; V4 and V5 told us a single radio is not enough and heavy contention is 5G’s home ground. The highlighted rows are the canonical comparison on the next slides.

# Part: Hard questions, cost and what is not yet proven

## 17. The questions a skeptic will ask, answered with evidence

Take these one at a time and point back to the slides with the evidence. The key move on the 5G objection is that every simplification in our 5G model helps 5G: no inter-cell interference, perfect scheduling, clean spectrum. On the tuning objection, mention the concrete case where a development-tuned threshold looked better and lost on fresh scenes, and we kept the pre-registered one anyway.

## 18. More hard questions

The second-radio objection is the one investors and buyers will press hardest. Answer it with the three routes and be clear which ones need hardware verification. On the AI objection, the honest answer is that the learning is in the map and its uncertainty, and the measurable effect is that it makes autonomous planning safe; the language model decides priorities. We do not claim a learned controller beats exact planning.

## 19. Fewest new parts to buy, best measured reliability

The cost argument is structural and you can make it without a single price: private 5G adds spectrum, a core network, new radio sites and a modem per robot; Wi-Fi 8 adds new access points and new clients that are not shipping yet; our approach adds software to the Wi-Fi 7 network the site already needs, plus at most a cheap second client radio per robot, and possibly nothing if the car probes confirm a second association on the existing card. Fill the price row with real quotes before presenting externally; do not quote numbers from memory.

## 20. What is proven, and what is not yet

End the evidence section by drawing the line yourself, before someone else draws it for you. The left column is what the pre-registered campaigns support. The right column is what none of them can support yet, and every item on it maps to a step in the validation plan on the next slide.

# Part: Validation plan, bill of materials and the ask

## 21. From simulation to a measured claim in four steps

The plan is ordered so each step de-risks the next. Step one is read-only and tells us whether a second association on the existing card is possible, which decides whether we buy any client radios at all. Step two measures the handover we have so far only assumed. Step three feeds real measurements back into the simulator and reruns the pre-registered campaigns. Step four is the fleet trial and, finally, a real head-to-head on a private 5G test site.

## 22. What we need to procure

Read the Need column. Required items make the 10-robot validation possible. Conditional items depend on step one of the plan: if the car probe shows the BE200 can hold two client connections, we skip the adapters and the management rework. The private 5G site is what turns “beats a 5G model” into “beats 5G”.

## 23. Fund the hardware validation that turns these results into a measured, field-proven claim

Close on the ask, which follows directly from the honest limits: the results are strong and carefully earned, and the remaining step is measurement on hardware. We need the equipment on the bill of materials, access to a private 5G test site for a real comparison, and one pilot floor.
