"""NeurIPS-style overview figure for WirelessJEV.

    ~/.local/bin/uv run python docs/make_figure.py

Writes docs/wirelessjev_figure.png (300 dpi) and .pdf. Result panels use the
measured numbers from artifacts/bench/report.json and artifacts/closedloop.json.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Polygon, Rectangle, Wedge)
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9})

C = dict(ink="#1f2430", grey="#8a8f99", light="#eef1f5", blue="#3b6fb6", lblue="#dbe7f6",
         green="#3c8d5a", lgreen="#dcefe2", orange="#d9822b", lorange="#fbe8d3",
         red="#c0392b", purple="#7a5ba6", lpurple="#ece4f6", ice="#cfe3f3", gold="#e2b13c")

fig = plt.figure(figsize=(14, 7.2), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 140)
ax.set_ylim(0, 72)
ax.axis("off")


def box(x, y, w, h, fc, ec=None, r=1.2, lw=0.9, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec or C["ink"], lw=lw, zorder=z))


def arrow(p, q, color=None, ls="-", lw=1.4, rad=0.0, z=5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=lw,
                                 color=color or C["ink"], linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z))


def label(x, y, s, size=9, w="normal", c=None, ha="center", va="center", **k):
    ax.text(x, y, s, fontsize=size, fontweight=w, color=c or C["ink"], ha=ha, va=va, **k)


def robot(x, y, s=1.0, color=C["blue"], z=6):
    ax.add_patch(FancyBboxPatch((x - 1.1 * s, y - 0.55 * s), 2.2 * s, 1.1 * s,
                                boxstyle="round,pad=0,rounding_size=0.3", fc=color, ec="white",
                                lw=0.4, zorder=z))
    for dx in (-0.65, 0.65):
        ax.add_patch(Circle((x + dx * s, y - 0.6 * s), 0.32 * s, fc=C["ink"], zorder=z + 1))
    ax.add_patch(Rectangle((x - 0.15 * s, y + 0.55 * s), 0.3 * s, 0.5 * s, fc=C["ink"], zorder=z))
    ax.add_patch(Circle((x, y + 1.1 * s), 0.22 * s, fc=C["gold"], ec=C["ink"], lw=0.3, zorder=z + 1))


def ap_icon(x, y, s=1.0, z=7):
    ax.add_patch(Polygon([[x - 0.6 * s, y - 0.7 * s], [x + 0.6 * s, y - 0.7 * s], [x, y + 0.5 * s]],
                         fc=C["ink"], zorder=z))
    for r in (0.9, 1.5):
        ax.add_patch(Wedge((x, y + 0.5 * s), r * s, 45, 135, width=0.18 * s, fc=C["orange"], zorder=z))


# ------------------------------------------------------------------ title
label(70, 69.3, r"$\bf{WirelessJEV}$: mission-conditioned fleet Wi-Fi decisions with a frozen System-One model",
      size=13)

# ------------------------------------------------------------------ (a) environment
box(1.5, 22, 38, 44, "white", C["grey"], lw=0.8)
label(3, 64, "(a) Environment: 100-AMR warehouse twin", size=10, w="bold", ha="left")
ex0, ey0, ew, eh = 3.5, 24.5, 34, 36
map_rect = Rectangle((ex0, ey0), ew, eh, fc="#f7f7f4", ec=C["grey"], lw=0.6, zorder=1)
ax.add_patch(map_rect)
sx, sy = ew / 120, eh / 80
for k in range(11):                                  # rack rows
    yr = ey0 + (10 + 6 * k) * sy
    for x0, x1 in ((10, 38), (42, 78), (82, 110)):
        ax.add_patch(Rectangle((ex0 + x0 * sx, yr - 0.3), (x1 - x0) * sx, 0.6,
                               fc="#9aa3ad", ec="none", zorder=2))
rng = np.random.default_rng(3)
aps = [(x, y) for y in (12, 32, 52, 72) for x in (15, 45, 75, 105)]
for (x, y) in aps:
    px, py = ex0 + x * sx, ey0 + y * sy
    cov = Circle((px, py), 5.2, fc=C["orange"], alpha=0.07, ec="none", zorder=2)
    ax.add_patch(cov)
    cov.set_clip_path(map_rect)
    ap_icon(px, py, 0.55)
for _ in range(26):                                  # robots + routes
    aisle = 13 + 6 * rng.integers(0, 10)
    x = rng.uniform(12, 108)
    px, py = ex0 + x * sx, ey0 + aisle * sy
    tx = px + rng.choice([-1, 1]) * rng.uniform(2, 6)
    ax.plot([px, tx], [py, py], color=C["blue"], lw=0.7, ls=(0, (2, 1.5)), zorder=3, alpha=0.8)
    robot(px, py, 0.38, C["blue"] if rng.random() > 0.12 else C["red"])
label(ex0, 23.0, "racks attenuate · 16 sites / 32 BSSIDs · planned routes · 4 s BSSID switch",
      size=7.5, c=C["grey"], ha="left")
# hardware inset
box(26.5, 26.5, 10.5, 8.2, "white", C["ink"], r=0.6, lw=0.6, z=8)
robot(29.3, 31.6, 0.9, C["blue"], z=9)
label(33.9, 33.0, "Raspbot", size=7, w="bold", zorder=10)
label(33.9, 31.2, "wlan1 BE200", size=6.5, c=C["blue"], zorder=10)
label(33.9, 29.6, "wlan0 mgmt", size=6.5, c=C["grey"], zorder=10)
label(31.8, 27.6, "act = choose BSSID", size=6.5, c=C["ink"], zorder=10)

# ------------------------------------------------------------------ (b) method
box(41, 22, 97.5, 44, "white", C["grey"], lw=0.8)
label(42.5, 64, "(b) Method", size=10, w="bold", ha="left")

# mission prompt
box(43, 45, 17, 13, C["light"])
label(51.5, 56, "Mission intent", size=9, w="bold")
for i, line in enumerate(["“amr-017 is under", "teleoperation: keep its", "video at full rate…”"]):
    label(51.5, 53 - 2.2 * i, line, size=7.5, style="italic")
ax.add_patch(Polygon([[46, 45], [48.5, 45], [45.5, 42.8]], fc=C["light"], ec=C["ink"], lw=0.9))

# twin + oracle + teachers
box(43, 25, 17, 15, C["lgreen"])
label(51.5, 37.6, "Numeric layer", size=9, w="bold")
label(51.5, 35.0, "digital twin · exact oracle", size=7.3)
label(51.5, 32.8, "MILP · local search", size=7.3)
label(51.5, 30.6, "Hungarian · colouring", size=7.3)
label(51.5, 28.0, r"$U_m(\pi)=B+\alpha U-\gamma W+\beta V$", size=7.6, c=C["green"])

# candidate plan cards
label(70, 61.5, "K complete plans + DEFER", size=9, w="bold")
for i in range(4):
    y = 49 - i * 6.2
    box(63, y, 14, 5.2, "white", C["ink"], r=0.5, lw=0.7)
    label(64, y + 4.0, f"P{i + 1}", size=7.5, w="bold", ha="left")
    vals = rng.uniform(0.3, 1.0, 4)
    for j, v in enumerate(vals):
        ax.add_patch(Rectangle((67 + j * 2.4, y + 0.8), 1.7, 2.6 * v,
                               fc=[C["blue"], C["green"], C["orange"], C["purple"]][j], ec="none"))
    label(76.3, y + 4.0, "…", size=7, c=C["grey"], ha="right")
box(63, 23.6, 14, 3.4, C["light"], C["ink"], r=0.5, lw=0.7)
label(70, 25.3, "DEFER", size=7.5, w="bold")
label(70, 58.6, "described as consequences:", size=7, c=C["grey"])
label(70, 56.8, "reconnects · floors · Mbps · fairness", size=7, c=C["grey"])

# frozen d3-lite
box(81, 31, 20, 26, C["ice"], C["blue"], lw=1.1)
label(91, 54.6, "d3-lite (0.85B)", size=9.5, w="bold", c=C["blue"])
label(91, 52.5, "frozen backbone", size=7.5, c=C["blue"])
for i in range(5):
    y = 37.5 + i * 2.6
    box(84, y, 14, 2.0, "white", C["blue"], r=0.4, lw=0.6)
    label(91, y + 1.0, "attn / MLP", size=6.5, c=C["blue"])
# snowflake
for a in range(0, 180, 30):
    t = np.deg2rad(a)
    ax.plot([98.6 - 1.1 * np.cos(t), 98.6 + 1.1 * np.cos(t)],
            [54.3 - 1.1 * np.sin(t), 54.3 + 1.1 * np.sin(t)], color=C["blue"], lw=0.8)
# adapter
box(84, 32.2, 14, 4.2, C["lorange"], C["orange"], lw=1.1)
label(91, 35.0, r"readout $+\,\mathbf{u}$", size=8.5, w="bold", c=C["orange"])
label(91, 33.2, "1,024 trainable params", size=6.5, c=C["orange"])

# P(yes) bars
label(111, 61.5, "option-wise  P(yes | plan, mission)", size=9, w="bold")
probs = [0.18, 0.71, 0.33, 0.12, 0.06]
names = ["P1", "P2", "P3", "P4", "DEF."]
for i, (n, p) in enumerate(zip(names, probs)):
    y = 55 - i * 4.3
    label(104.8, y + 1.2, n, size=7.5, ha="right")
    ax.add_patch(Rectangle((106.3, y), 13, 2.4, fc=C["light"], ec="none"))
    ax.add_patch(Rectangle((106.3, y), 13 * p, 2.4, fc=C["green"] if p == max(probs) else C["grey"],
                           ec="none"))
label(120.2, 55 - 1 * 4.3 + 1.2, "✓", size=11, c=C["green"], ha="left", w="bold", fontfamily="DejaVu Sans")

# execution
box(103, 23.5, 33, 11.5, C["lpurple"], C["purple"])
label(119.5, 33.2, "Verified execution", size=9, w="bold", c=C["purple"])
label(119.5, 30.9, "PREPARED → ISSUED → COMMITTED", size=7.2)
label(119.5, 28.9, "UNKNOWN never resent · pinned SSH", size=7.2)
label(119.5, 26.9, r"SmartCarCmd: {connect ssid, bssid}", size=7.2)
robot(132.5, 25.6, 0.75, C["blue"])

# swarm + oracle teaching
box(122, 40, 14.5, 9.5, "white", C["purple"], lw=0.8)
label(129.2, 47.5, "Swarm layer", size=8.5, w="bold", c=C["purple"])
label(129.2, 45.3, "validate-vs-live merge", size=6.8)
label(129.2, 43.5, "reputation · quarantine", size=6.8)
label(129.2, 41.7, "HMAC audit log", size=6.8)

# arrows
arrow((60, 51), (63, 51))
arrow((60, 32), (63, 36))
arrow((77, 44), (81, 44))
arrow((51.5, 45), (51.5, 40.2))
arrow((101, 44), (104, 50))
# oracle -> adapter (distillation, dashed)
arrow((60, 27), (84, 33.5), color=C["orange"], ls="--", rad=-0.15)
label(71, 29.0, "distil: KL to softmax(U_m/τ)", size=7.2, c=C["orange"])
arrow((122, 44.7), (98, 35.5), color=C["purple"], ls="--", rad=0.25)
# feedback
arrow((114, 37.6), (114, 35.2))
label(115, 36.4, "chosen plan", size=7, c=C["green"], ha="left")
arrow((103, 25.5), (79, 25.5), color=C["grey"], ls=":")
label(91, 22.6, "scan / status feedback", size=7, c=C["grey"])

# ------------------------------------------------------------------ results
def panel(rect, title):
    a = fig.add_axes(rect)
    a.set_title(title, fontsize=9.5, fontweight="bold", loc="left", pad=4)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.tick_params(labelsize=7.5)
    return a


a1 = panel([0.045, 0.06, 0.27, 0.19], "(c) Fleet decision regret ↓ (10-car, 80 test)")
names = ["SmartCar\nlegacy", "zero-shot\nd3-lite", "MILP\nplanner", "WirelessJEV\n(distilled)"]
vals = [1.30, 0.80, 0.215, 0.08]
cols = [C["grey"], C["blue"], C["green"], C["orange"]]
b = a1.bar(names, vals, color=cols, width=0.62)
for r, v in zip(b, vals):
    a1.text(r.get_x() + r.get_width() / 2, v + 0.03, f"{v:g}", ha="center", fontsize=7.5)
a1.set_ylim(0, 1.5)

a2 = panel([0.37, 0.06, 0.27, 0.19], "(d) Mission sensitivity & accuracy ↑")
x = np.arange(2)
a2.bar(x - 0.18, [0.04, 0.28], 0.34, color=C["blue"], label="zero-shot")
a2.bar(x + 0.18, [0.75, 0.70], 0.34, color=C["orange"], label="distilled")
a2.set_xticks(x, ["intent flip\n(correct)", "top-1\nchoice"])
for xi, (z, d) in zip(x, [(0.04, 0.75), (0.28, 0.70)]):
    a2.text(xi - 0.18, z + 0.02, f"{z:.0%}", ha="center", fontsize=7.5)
    a2.text(xi + 0.18, d + 0.02, f"{d:.0%}", ha="center", fontsize=7.5)
a2.set_ylim(0, 1)
a2.legend(fontsize=7, frameon=False, loc="upper right")

a3 = panel([0.69, 0.06, 0.28, 0.19], "(e) Closed loop: mean mission utility ↑")
pol = ["legacy", "MILP", "WirelessJEV"]
util = [1.46, 1.51, 2.43]
ho = [14, 15, 29]
b = a3.barh(pol, util, color=[C["grey"], C["green"], C["orange"]], height=0.55)
for r, v, h in zip(b, util, ho):
    a3.text(v + 0.04, r.get_y() + r.get_height() / 2, f"{v:.2f}  ({h} handovers)",
            va="center", fontsize=7.5)
a3.set_xlim(0, 3.4)

out = Path(__file__).with_name("wirelessjev_figure")
fig.savefig(out.with_suffix(".png"), dpi=300)
fig.savefig(out.with_suffix(".pdf"))
print(out.with_suffix(".png"))
