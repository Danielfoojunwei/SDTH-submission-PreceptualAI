"""Two-page status brief with diagrams and charts: docs/STATUS_BRIEF.pdf.

    .venv/bin/python docs/figures/make_brief.py

Figures are drawn with matplotlib into docs/figures/brief/; the PDF is laid
out with PyMuPDF's HTML engine (both already installed; no new dependencies).
Every number is copied from the committed result files named in each caption.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs/figures/brief"
OUT = ROOT / "docs/STATUS_BRIEF.pdf"

# Reference chart palette, light mode: the first three categorical slots (validated all-pairs).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
OURS_FILL, OURS_EDGE = "#fde8dc", "#d9661f"
INPUT_FILL, INPUT_EDGE = "#eef2f4", "#9aa8b1"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "bold", "axes.titlesize": 8,
                     "axes.titlecolor": INK, "svg.fonttype": "none"})


def _box(ax, x, y, w, h, title, body, ours=True, tsize=7.2, bsize=5.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=OURS_FILL if ours else INPUT_FILL, ec=OURS_EDGE if ours else INPUT_EDGE, lw=1.0))
    ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top", fontsize=tsize, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 1.5 - tsize * 0.62, body, ha="center", va="top", fontsize=bsize, color=INK2, linespacing=1.28)


def _arrow(ax, a, b, color=INK2, style="-|>", ls="-"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle=style, color=color, lw=1.1, ls=ls, shrinkA=0, shrinkB=0))


def architecture(path):
    fig, ax = plt.subplots(figsize=(7.5, 2.75))
    ax.set_xlim(0, 100), ax.set_ylim(0, 40), ax.axis("off")
    W, top, bot, H = 18.4, 21.5, 1.0, 17.5
    xs = [0.5, 20.9, 41.3, 61.7, 81.1]
    _box(ax, xs[0], top, W, H, "Robot scans", "signal to every AP,\ncollected during\nnormal driving", ours=False)
    _box(ax, xs[1], top, W, H, "RadioMap", "Bayesian radio map;\neach estimate has\nits own uncertainty;\nupdated online")
    _box(ax, xs[2], top, W, H, "WirelessJEV", "intent to priorities;\nexact planner per\nrobot; trust gate:\nacts only when sure")
    _box(ax, xs[3], top, W, H, "Wi-Fi 7 stack", "second AP path for\nevery packet;\ncontrol/bulk split;\nroams between packets")
    _box(ax, xs[4], top, W, H, "100 robots", "control commands\ndelivered on time\n(scored as missed\nper million)", ours=False)
    _box(ax, xs[1], bot, W, H, "Coverage report", "weak areas, backup\ncoverage, effect of\nan AP failing; from\nthe same data")
    _box(ax, xs[2], bot, W, H, "Mission text", "operator priorities\nin plain language", ours=False)
    _box(ax, xs[3], bot, W, H, "Fleet layer", "robot relay when\nboth paths fail;\n2 s safe plan;\n1 s blockage warning")
    _box(ax, xs[4], bot, W, H, "Obstacle sensing", "the robot's lidar\nor camera\n(simulated so far)", ours=False)
    mid = top + H / 2
    for i in range(4):
        _arrow(ax, (xs[i] + W, mid), (xs[i + 1], mid))
    _arrow(ax, (xs[1] + W / 2, top), (xs[1] + W / 2, bot + H))                    # map -> coverage
    _arrow(ax, (xs[2] + W / 2, bot + H), (xs[2] + W / 2, top))                    # mission text -> WirelessJEV
    _arrow(ax, (xs[4], bot + H / 2), (xs[3] + W, bot + H / 2))                    # sensing -> fleet layer
    _arrow(ax, (xs[3] + W - 2, bot + H), (xs[4] + 2, top))                        # fleet layer -> robots
    ax.text(1, 9.5, "Orange: our components\nGrey: inputs and outputs", fontsize=6, color=MUTED, va="center")
    fig.savefig(path, dpi=230, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def site_setup(path):
    fig, ax = plt.subplots(figsize=(7.5, 2.55))
    ax.set_xlim(0, 100), ax.set_ylim(0, 44), ax.axis("off")

    def node(x, y, w, h, title, body, ours=False):
        _box(ax, x - w / 2, y - h / 2, w, h, title, body, ours, tsize=7, bsize=5.7)

    def line(a, b, color, ls, lw=1.2):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, ls=ls, lw=lw, solid_capstyle="round", zorder=1)

    def label(x, y, s, ha="center"):
        ax.text(x, y, s, fontsize=5.8, color=INK2, ha=ha, va="center",
                bbox=dict(fc="white", ec="none", pad=0.6), zorder=3)

    # links first, so boxes sit on top
    line((16.5, 21), (53, 13), BLUE, (0, (4, 2)))                                 # laptop -> car hotspot
    line((16.5, 28), (23, 34), INK2, "-")                                         # laptop -> switch (Ethernet)
    line((37.5, 38.5), (84, 38.5), INK2, "-", lw=0.9)                             # switch -> APs
    for x in (54, 72, 90):
        line((62, 18), (x, 33), ORANGE, (0, (1.5, 1.5)), lw=1.4)                  # car test radio -> APs
    line((90, 17.5), (90, 33), AQUA, (0, (1.5, 1.5)), lw=1.4)                     # phone -> AP 3
    line((43, 5.5), (53, 9), MUTED, (0, (2, 2)), lw=1.0)                          # remote -> car
    node(9, 24, 15, 11, "Laptop", "commands and\nlive viewer")
    node(30, 38.5, 15, 11, "Router / switch", "DHCP, not\n192.168.1.x")
    for i, x in enumerate((54, 72, 90)):
        node(x, 38.5, 13, 11, f"AP {i + 1}", "5 GHz, fixed\nchannel")
    node(62, 12, 19, 12, "SmartCar", "wlan0: hotspot (mgmt)\nwlan1: test radio", ours=True)
    node(90, 12, 15, 11, "Phone", "on test Wi-Fi\n(ghost-load target)")
    node(35.5, 5, 16, 9, "Vendor remote", "drive only")
    label(32, 14.5, "Wi-Fi: car hotspot (management)")
    label(21, 29.5, "Ethernet", ha="left")
    label(47, 27, "test radio roams\nbetween APs")
    label(67.5, 25.5, "blocker: person\nor cart, in the path")
    fig.text(0.01, -0.02, "Blue dashes: management link (car hotspot)    Grey: cables    Orange dots: the car's test radio    "
             "Aqua dots: phone on the test Wi-Fi", fontsize=6, color=MUTED)
    fig.savefig(path, dpi=230, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def results(path):
    groups = ["Realistic conditions,\ntwo AP paths\n(scenes 530–535)", "Second path on the\nexisting card\n(scenes 540–545)",
              "Forklifts block both\npaths at once\n(scenes 570–575)"]
    series = [("Private 5G model", BLUE, [52.8, 58.9, 171.4]), ("WirelessJEV", ORANGE, [3.1, 5.0, 14.4]),
              ("Wi-Fi 8 model", AQUA, [None, 244.2, 439.2])]
    fig, ax = plt.subplots(figsize=(7.5, 2.35))
    h, gap = 0.24, 0.02
    for g in range(len(groups)):
        for k, (name, color, vals) in enumerate(series):
            v = vals[g]
            y = g + (k - 1) * (h + gap)
            if v is None:
                ax.text(4, y, "not in this test", va="center", fontsize=6.3, color=MUTED, style="italic")
                continue
            ax.barh(y, v, height=h, color=color, label=name if g == 1 else None)
            ax.text(v + 5, y, f"{v:g}", va="center", fontsize=7, color=INK, weight="bold" if name == "WirelessJEV" else None)
    ax.set_yticks(range(len(groups)), groups)
    ax.invert_yaxis()
    ax.set_xlim(0, 480)
    ax.set_xlabel("Missed control packets per million (lower is better)")
    _clean(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=7, ncols=3, bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Same scenes, same interference: 12–49× fewer missed commands", loc="left", pad=16)
    fig.savefig(path, dpi=230, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def hardest(path):
    labels = ["Two AP paths alone", "+ relay, 30 ms, helpers\nfree 50% (pessimistic)", "+ relay, 20 ms, helpers\nfree 70% (realistic)",
              "+ relay, instant, always\nfree (optimistic)"]
    vals = [234.2, 56.7, 21.4, 1.1]
    fig, ax = plt.subplots(figsize=(3.65, 2.1))
    ax.barh(range(4), vals, height=0.55, color=ORANGE)
    for i, v in enumerate(vals):
        ax.text(v + 4, i, f"{v:g}", va="center", fontsize=7, color=INK)
    ax.axvline(136.4, color=BLUE, ls=(0, (4, 2)), lw=1.3)
    ax.text(139, -0.62, "Private 5G model 136.4", fontsize=6.5, color=INK2, va="center")
    ax.set_yticks(range(4), labels, fontsize=6.6)
    ax.set_ylim(3.5, -0.95)
    ax.set_xlim(0, 270)
    ax.set_xlabel("Missed per million (scenes 590–595)")
    _clean(ax)
    ax.set_title("Hardest radio condition: robot relay wins", loc="left", fontsize=7.6)
    fig.savefig(path, dpi=230, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def foresight(path):
    tests = ["Test 1\n(scenes 580–585)", "Test 2\n(scenes 600–605)"]
    series = [("No warning", "#f6c1a3", [229.7, 320.0]), ("Warning from obstacle sensing", ORANGE, [175.0, 179.7]),
              ("Perfect warning (ceiling)", "#9c3d12", [136.7, 131.1])]
    fig, ax = plt.subplots(figsize=(3.65, 2.1))
    h, gap = 0.24, 0.02
    for g in range(2):
        for k, (name, color, vals) in enumerate(series):
            y = g + (k - 1) * (h + gap)
            ax.barh(y, vals[g], height=h, color=color, label=name if g == 0 else None)
            ax.text(vals[g] + 4, y, f"{vals[g]:g}", va="center", fontsize=6.6, color=INK)
    ax.set_yticks(range(2), tests, fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 560)
    ax.set_xlabel("Missed per million, two AP paths")
    _clean(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=6.2, bbox_to_anchor=(1.02, -0.02))
    ax.set_title("Foresight: 24–44% fewer misses", loc="left", fontsize=7.6)
    fig.savefig(path, dpi=230, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def timeline(path):
    blocks = [("Setup\n(45 min)", 50, False), ("Connect +\nformat check", 58, True), ("Survey laps +\nhandover timing", 130, True),
              ("10 route laps", 95, True), ("Blockage trials +\nlaps under load", 130, True), ("Analyse,\nback up", 50, False)]
    fig, ax = plt.subplots(figsize=(7.5, 0.62))
    x = 0
    for label, w, car in blocks:
        ax.barh(0, w - 1.5, left=x, height=0.8, color=OURS_FILL if car else INPUT_FILL, edgecolor=OURS_EDGE if car else INPUT_EDGE, lw=0.8)
        ax.text(x + (w - 1.5) / 2, 0, label, ha="center", va="center", fontsize=6.1, color=INK)
        x += w
    ax.set_xlim(0, x), ax.set_ylim(-0.5, 0.5), ax.axis("off")
    fig.savefig(path, dpi=230, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


CSS = """
* { font-family: sans-serif; }
body { font-size: 8.1pt; color: #13212C; line-height: 1.24; }
h1 { font-size: 15.5pt; margin: 0 0 1pt 0; color: #10202B; }
.sub { font-size: 8.4pt; color: #4A5A66; margin: 0; }
.flag { font-size: 8.2pt; color: #B4561E; font-weight: bold; margin: 2pt 0 3pt 0; }
h2 { font-size: 10.2pt; color: #B4561E; margin: 5pt 0 2pt 0; }
p { margin: 0 0 2.5pt 0; }
ul, ol { margin: 0 0 2pt 0; padding-left: 11pt; }
li { margin: 0 0 1.3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 1pt 0 3pt 0; }
th { background-color: #E8EDF0; text-align: left; padding: 1.5pt 3pt; border: 0.4pt solid #C9D2D8; }
td { padding: 1.5pt 3pt; border: 0.4pt solid #C9D2D8; vertical-align: top; }
td.img, table.img td { border: none; padding: 0 2pt 0 0; }
.k { font-weight: bold; }
.muted { color: #5A6A75; }
.cap { font-size: 7pt; color: #5A6A75; margin: 0 0 3pt 0; }
.hdr { font-weight: bold; margin: 2pt 0 1pt 0; }
table.small { font-size: 7.5pt; }
"""

PAGE1 = """
<h1>WirelessJEV + RadioMap: status and field plan</h1>
<p class="sub">17 Nov 2026 · branch <i>wirelessjev/real-site-fleet</i> · details in the repo's <i>docs/</i></p>
<p class="flag">Every result here is simulation: 100 robots, pre-registered tests on fresh scenes. Nothing has run on the real SmartCar yet; today's site visit is the first hardware test.</p>

<h2>1. The system, and why it is capable</h2>
<img src="architecture.png" width="543"/>
<p class="cap">How the parts connect. RadioMap learns the site from robot scans; WirelessJEV plans each robot's radio and only acts when the map is sure; the Wi-Fi 7 stack keeps a second AP path ready for every packet; the fleet layer adds what a network alone cannot.</p>
<p><span class="k">Why it is novel:</span> known pieces combined in a way no one ships. Per-packet diversity across APs on today's Wi-Fi 7 clients, fleet-wide; a learned map whose own error decides when the AI may act; fleet capabilities layered on the network (relay, safe plan, sensing-based warning); every claim pre-registered, with failures kept.</p>

<h2>2. What simulation shows</h2>
<img src="results.png" width="543"/>
<table class="img"><tr><td class="img"><img src="hardest.png" width="268"/></td><td class="img"><img src="foresight.png" width="268"/></td></tr></table>
<p class="cap">Missed control packets per million (lower is better). 5G and Wi-Fi 8 are generous models of each, run in the same scenes. Left: in the noisiest, most crowded air, APs alone lose to 5G; robots relaying for each other win even when the relay is slow and unreliable. Right: a one-second warning from the robot's obstacle sensing, replicated in two tests.</p>
<p>Also: uptime 99.995% against 99.933% for 5G (realistic); a 2 s safe plan cuts stopped time by 94% in the hardest condition; the coverage report's level error is 3.6 dB and its "no backup AP" map is 88% right, finding 95%.</p>
<p><span class="k">Kept failures:</span> one radio alone loses to 5G (about 200 vs 55); the live blockage rule hurt under short blockages and is off; fleet-shared sensing added nothing; learned radio models (RadioJEPA) predicted signal better but did not control better; link-aware speed cost throughput.</p>
"""

PAGE2 = """
<h2>3. What we must test today, and how</h2>
<table class="small">
<tr><th>Question</th><th>How (runbook block)</th><th>Why it matters</th></tr>
<tr><td>Does the car's real output match our tools?</td><td>10 s format check, PASS/WARN/FAIL (1b)</td><td>Protects every lap after it</td></tr>
<tr><td>Can the existing card hold two AP connections?</td><td>Capabilities probe: <i>two_clients_two_channels</i> (1)</td><td>Decides whether extra radios must be bought</td></tr>
<tr><td>Real handover outage: today's switch vs fast roam</td><td>Gateway ping gaps, 5+ of each (3)</td><td>Simulator assumes 4 s and 30 ms</td></tr>
<tr><td>How long do real blockages last?</td><td>Person or cart blocking at marked spots (6)</td><td>Decides whether the blockage rule helps</td></tr>
<tr><td>Does real signal fit RadioMap's assumptions?</td><td>3 survey + 10 route laps, held-out comparison (2, 4)</td><td>The only way a learned model earns a place</td></tr>
<tr><td>10-robot result and coverage on measured signals</td><td>Lap replay and coverage report (7)</td><td>First fleet estimate from real data</td></tr>
</table>
<p class="muted">Not testable today: two live AP paths, relay, Bluetooth path, obstacle sensing, a real 5G comparison.</p>
<img src="site_setup.png" width="520"/>
<p class="cap">Today's setup. The laptop's Wi-Fi stays on the car's hotspot to control it; the car's test radio roams between your APs; the laptop reaches the test network by cable to send ghost load to a phone.</p>

<h2>4. Hardware</h2>
<table class="small">
<tr><th>Bring today</th><th>Next phase (prices need quotes)</th></tr>
<tr><td><ul>
<li>SmartCar + charger; vendor remote or app (drive only)</li>
<li>2–4 Wi-Fi APs: one network name, fixed 5 GHz channels, 802.11r, DHCP not on 192.168.1.x</li>
<li>Router/switch, Ethernet cables, power strips</li>
<li>Laptop + charger; USB-C Ethernet adapter + a phone</li>
<li>Floor tape, tape measure, USB stick, a cart; 2–3 people</li>
</ul></td><td><ul>
<li>4 Wi-Fi 7 enterprise APs; 9 more SmartCars (10-robot trial)</li>
<li>USB Wi-Fi 6E/7 adapters with in-kernel Linux drivers, only if today's probe says the card cannot hold two connections</li>
<li>Out-of-band management (frees the onboard radio)</li>
<li>2 capture adapters + capture laptop; power meters</li>
<li>Access to a private 5G test site</li>
</ul></td></tr>
</table>

<h2>5. Step by step</h2>
<img src="timeline.png" width="543"/>
<p class="cap">Order of the day (not to scale). Orange blocks use the car.</p>
<table class="small">
<tr><th>Before leaving</th><th>On site</th><th>After the visit</th></tr>
<tr><td><ol>
<li>Copy the car's host key and Wi-Fi list; add the test network; run <i>preflight</i></li>
<li>Configure the APs; note each BSSID and channel</li>
<li>Charge car, laptop, power bank, remote, phone</li>
</ol></td><td><ol start="4">
<li>Place and measure APs; tape waypoints; write <i>site.json</i>, <i>waypoints.json</i></li>
<li>Join the car's hotspot; start the live viewer; probes</li>
<li>Join the car to the test network; format check</li>
<li>3 survey laps; handover timing</li>
<li>10 route laps; blockage trials; load laps if time</li>
<li>Analysis, replay, coverage; back up to USB</li>
</ol></td><td><ol start="10">
<li>Calibrate the simulator with real handover, blockage and fading; rerun tests</li>
<li>Build second-path logging; buy adapters only if needed</li>
<li>10-robot trial, then a real 5G head-to-head</li>
</ol></td></tr>
</table>
<table class="img"><tr><td class="img"><img src="coverage.png" width="330"/></td><td class="img"><img src="viewer.jpg" width="160"/></td></tr></table>
<p class="cap">Left: coverage report from robot measurements (simulated site; orange line −67 dBm). Right: the read-only live lap viewer used on site (demo lap). Sources: docs/TUESDAY_RUNBOOK.md, CROSS_AP.md, DIRECTIONS_123.md, MOVING_BLOCKERS.md, RELAY_REALISM.md, FORESIGHT.md, COVERAGE_REPORT.md, BOM.md.</p>
"""


def build_pdf():
    W, H = pymupdf.paper_size("a4")
    M = 26
    rect = pymupdf.Rect(M, M, W - M, H - M)
    archive = pymupdf.Archive(str(FIG))
    writer = pymupdf.DocumentWriter(str(OUT))
    report = []
    for n, html in enumerate((PAGE1, PAGE2), 1):
        dev = writer.begin_page(pymupdf.Rect(0, 0, W, H))
        story = pymupdf.Story(html=html, user_css=CSS, archive=archive)
        more, filled = story.place(rect)
        story.draw(dev)
        writer.end_page()
        report.append((n, bool(more), round(pymupdf.Rect(filled).y1, 1)))
    writer.close()
    doc = pymupdf.open(str(OUT))
    for i, page in enumerate(doc):
        page.insert_text((W - M - 40, H - 12), f"page {i + 1} of {len(doc)}", fontsize=6.5, color=(0.45, 0.5, 0.53))
    doc.saveIncr()
    return report


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    architecture(FIG / "architecture.png")
    site_setup(FIG / "site_setup.png")
    results(FIG / "results.png")
    hardest(FIG / "hardest.png")
    foresight(FIG / "foresight.png")
    timeline(FIG / "timeline.png")
    (FIG / "coverage.png").write_bytes((ROOT / "docs/figures/coverage_scene610.png").read_bytes())
    (FIG / "viewer.jpg").write_bytes((ROOT / "docs/figures/lap_viewer_demo.jpg").read_bytes())
    report = build_pdf()
    for n, overflow, y in report:
        print(f"page {n}: {'OVERFLOW' if overflow else 'fits'}; content ends at y = {y} of {pymupdf.paper_size('a4')[1] - 26}")
    if len(sys.argv) > 1:
        doc = pymupdf.open(str(OUT))
        for i, page in enumerate(doc):
            page.get_pixmap(dpi=110).save(f"{sys.argv[1]}_p{i + 1}.png")


if __name__ == "__main__":
    main()
