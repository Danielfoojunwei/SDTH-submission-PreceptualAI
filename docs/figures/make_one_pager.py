"""Build the one-page status PDF with PyMuPDF's HTML layout (no extra packages)."""
import sys
import pymupdf

OUT = sys.argv[1]
PREVIEW = sys.argv[2] if len(sys.argv) > 2 else None
BASE = float(sys.argv[3]) if len(sys.argv) > 3 else 7.4

CSS = f"""
* {{ font-family: sans-serif; }}
body {{ font-size: {BASE}pt; color: #13212C; line-height: 1.22; }}
h1 {{ font-size: {BASE + 7.5}pt; margin: 0 0 1pt 0; color: #10202B; }}
.sub {{ font-size: {BASE + 0.4}pt; color: #4A5A66; margin: 0; }}
.flag {{ font-size: {BASE + 0.2}pt; color: #B4561E; font-weight: bold; margin: 2pt 0 0 0; }}
h2 {{ font-size: {BASE + 2.2}pt; color: #B4561E; margin: 5pt 0 2pt 0; padding: 0; }}
p {{ margin: 0 0 2.5pt 0; }}
ul {{ margin: 0 0 2pt 0; padding-left: 10pt; }}
ol {{ margin: 0 0 2pt 0; padding-left: 12pt; }}
li {{ margin: 0 0 1.4pt 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 1pt 0 3pt 0; }}
th {{ background-color: #E8EDF0; text-align: left; font-weight: bold; padding: 1.4pt 2.5pt; border: 0.4pt solid #C9D2D8; }}
td {{ padding: 1.4pt 2.5pt; border: 0.4pt solid #C9D2D8; vertical-align: top; }}
.num {{ text-align: right; }}
.k {{ font-weight: bold; }}
.muted {{ color: #5A6A75; }}
.hdr {{ font-weight: bold; color: #10202B; margin: 3pt 0 1pt 0; }}
"""

HEADER = """
<h1>WirelessJEV + RadioMap: status and field plan</h1>
<p class="sub">One-page summary, 17 Nov 2026 · branch <i>wirelessjev/real-site-fleet</i> · details in the repo's <i>docs/</i></p>
<p class="flag">Every result below is simulation (100 robots, pre-registered tests on fresh scenes). Nothing has run on the real SmartCar yet; today's site visit is the first hardware test.</p>
"""

BODY = """
<h2>1. The system, and why it is capable</h2>
<ul>
<li><span class="k">RadioMap</span>: a Bayesian radio map learned from about one minute of robot driving. Every estimate carries its own uncertainty, and it updates as robots move.</li>
<li><span class="k">WirelessJEV</span>: turns mission text into priorities; an exact planner picks AP, band and roam timing per robot; a <span class="k">trust gate</span> lets it act only when the map is sure enough, otherwise a fixed rule runs.</li>
<li><span class="k">Native Wi-Fi 7 stack</span>: a second AP path ready for every control packet (diversity across APs, not only bands); control and bulk split across links; roams timed between packets with 802.11r keys.</li>
<li><span class="k">Fleet-only capabilities</span> a network vendor cannot add: robot relay when both AP paths fail; a 2 s safe plan so one missed command does not stop a robot; a one-second blockage warning from the robot's obstacle sensing.</li>
<li><span class="k">Site coverage report</span> from the same measurements: weak areas, backup coverage, and the effect of any AP failing.</li>
</ul>
<p><span class="k">Why it is novel:</span> known pieces combined in a way no one ships. Per-packet cross-AP diversity on today's Wi-Fi 7 clients, fleet-wide; a learned map whose own error decides when the AI may act; fleet capabilities layered on the network; every claim pre-registered, with failures kept.</p>

<h2>2. What simulation shows</h2>
<p class="muted">Missed control packets per million (lower is better); 100 robots; six fresh scenes per test.</p>
<table>
<tr><th>Test (scenes)</th><th class="num">Ours</th><th class="num">5G</th><th class="num">Wi-Fi 8</th></tr>
<tr><td>Realistic, two AP paths (530–535)</td><td class="num">3.1</td><td class="num">52.8</td><td class="num">–</td></tr>
<tr><td>Second path on the existing card (540–545)</td><td class="num">5.0</td><td class="num">58.9</td><td class="num">244.2</td></tr>
<tr><td>Forklifts block both paths at once (570–575)</td><td class="num">14.4</td><td class="num">171.4</td><td class="num">439.2</td></tr>
<tr><td>Hardest radio condition, APs + relay at 20 ms, helpers free 70% (590–595)</td><td class="num">21.4</td><td class="num">136.4</td><td class="num">–</td></tr>
<tr><td>Uptime, realistic (530–535)</td><td class="num">99.995%</td><td class="num">99.933%</td><td class="num">–</td></tr>
</table>
<p class="muted">5G and Wi-Fi 8 are generous models of each, run in the same scenes. "–": not in that test.</p>
<p>Also: the safe plan cuts stopped time by 94% in the hardest condition; obstacle-sensing foresight cuts two-path misses by 24–44% there (two tests); the coverage report has 3.6 dB level error and its "no backup AP" map is 88% right and finds 95%.</p>
<p><span class="k">Kept failures:</span> one radio alone loses to 5G (about 200 vs 55); APs alone lose the hardest condition (234 vs 136); the live blockage rule hurt under short blockages and is off; fleet-shared sensing added nothing; learned radio models (RadioJEPA) predicted signal better but did not control better; link-aware speed cost throughput.</p>

<h2>3. What we must test, and how</h2>
<table>
<tr><th>Question</th><th>How (runbook block)</th></tr>
<tr><td>Does the car's real output match our tools?</td><td>10 s format check, PASS/WARN/FAIL per item (1b)</td></tr>
<tr><td>Can the existing card hold two AP connections?</td><td>Capabilities probe: <i>two_clients_two_channels</i> (1)</td></tr>
<tr><td>Real handover outage: today's switch vs fast roam</td><td>Gateway ping gaps, 5 or more of each (3)</td></tr>
<tr><td>How long do real blockages last?</td><td>Person or cart blocking at marked spots (6)</td></tr>
<tr><td>Does real signal fit RadioMap's assumptions?</td><td>3 survey + 10 route laps; held-out comparison (2, 4)</td></tr>
<tr><td>10-robot result on measured signals</td><td>Replay of the 10 route laps as a fleet (7)</td></tr>
<tr><td>Coverage of the site</td><td>Coverage report from all laps (7)</td></tr>
</table>
<p class="muted">Not testable today: two live AP paths, relay, Bluetooth path, obstacle sensing, a real 5G comparison.</p>

<h2>4. Hardware</h2>
<p class="hdr">Today</p>
<ul>
<li>SmartCar + charger; the vendor remote or driving app (to drive only; never its Wi-Fi buttons or automatic mode).</li>
<li>2–4 Wi-Fi APs: one network name, fixed 5 GHz channels (e.g. 36, 52, 100), 802.11r on, one DHCP server <span class="k">not</span> on 192.168.1.x.</li>
<li>Router or switch, Ethernet cables, power strips; this laptop + charger.</li>
<li>USB-C Ethernet adapter + any phone on the test Wi-Fi (ghost load); floor tape, tape measure, USB stick, a cart; 2–3 people.</li>
</ul>
<p class="hdr">Next phase (prices need quotes)</p>
<ul>
<li>4 Wi-Fi 7 enterprise APs; 9 more SmartCars for a 10-robot trial.</li>
<li>USB Wi-Fi 6E/7 adapters with in-kernel Linux drivers, <i>only if</i> today's probe says the card cannot hold two connections.</li>
<li>Out-of-band management (frees the onboard radio); 2 capture adapters + a capture laptop; power meters.</li>
<li>Access to a private 5G test site for a real head-to-head.</li>
</ul>

<h2>5. Step by step</h2>
<p class="hdr">Before leaving</p>
<ol>
<li>Copy the car's host key and Wi-Fi list to the laptop; add the test network; run <i>preflight</i>.</li>
<li>Configure the APs at home; write down each AP's BSSID and channel.</li>
<li>Charge the car, laptop, power bank, remote and phone.</li>
</ol>
<p class="hdr">On site</p>
<ol start="4">
<li>Place the APs; measure the room and each AP; tape and measure waypoints; write <i>site.json</i> and <i>waypoints.json</i>.</li>
<li>Join the laptop to the car's hotspot; start the live viewer; preflight, status, capabilities, handover probe.</li>
<li>Join the car to the test network; run the format check. Stop only on no samples, no gateway or no pings.</li>
<li>3 survey laps (one person drives slowly, one types waypoint marks).</li>
<li>Handover timing: timed switch, then fast roam, 5 or more each.</li>
<li>10 route laps, r01–r10.</li>
<li>Blockage trials; then laps under ghost load if time allows.</li>
<li>Run the analysis, 10-robot replay and coverage report; back up the laps to the USB stick.</li>
</ol>
<p class="hdr">After the visit</p>
<ol start="12">
<li>Calibrate the simulator with the real handover time, blockage length and fading; rerun the tests.</li>
<li>Build second-path logging on the car; buy adapters only if the probe requires them.</li>
<li>10-robot trial, then a real private 5G head-to-head.</li>
</ol>
"""

W, H = pymupdf.paper_size("a4")
M, GAP = 26, 14
header_rect = pymupdf.Rect(M, M, W - M, M + 64 * BASE / 7.4)
col_w = (W - 2 * M - GAP) / 2
top = header_rect.y1 + 4
cols = [pymupdf.Rect(M, top, M + col_w, H - M - 10), pymupdf.Rect(M + col_w + GAP, top, W - M, H - M - 10)]

writer = pymupdf.DocumentWriter(OUT)
dev = writer.begin_page(pymupdf.Rect(0, 0, W, H))
head = pymupdf.Story(html=HEADER, user_css=CSS)
more, _ = head.place(header_rect)
head.draw(dev)
assert not more, "header does not fit"
body = pymupdf.Story(html=BODY, user_css=CSS)
for i, r in enumerate(cols):
    more, filled = body.place(r)
    body.draw(dev)
    if not more:
        break
writer.end_page()
writer.close()

doc = pymupdf.open(OUT)
page = doc[0]
page.draw_line((M, header_rect.y1 + 1), (W - M, header_rect.y1 + 1), color=(0.85, 0.4, 0.12), width=1.2)
foot = "Sources: docs/TUESDAY_RUNBOOK.md, CROSS_AP.md, DIRECTIONS_123.md, MOVING_BLOCKERS.md, RELAY_REALISM.md, FORESIGHT.md, COVERAGE_REPORT.md, BOM.md, TEST_SUITE.md"
page.insert_text((M, H - M + 6), foot, fontsize=6, color=(0.35, 0.42, 0.46))
doc.saveIncr()
print("fits on one page" if not more else "OVERFLOW: content left after two columns", "| last column filled to y =", round(pymupdf.Rect(filled).y1, 1), "of", round(cols[-1].y1, 1))
if PREVIEW:
    page.get_pixmap(dpi=110).save(PREVIEW)
