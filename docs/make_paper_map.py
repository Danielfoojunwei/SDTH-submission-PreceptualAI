"""Render the Shen / Li / Lam paper map as a PNG table.

    ~/.local/bin/uv run python docs/make_paper_map.py
"""
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})

SECTIONS = [
    ("Spectrum access (IoT / vehicles)", "#3b6fb6", [
        ("Hierarchical federated DRL DSA — Ad Hoc Netw. 2023 (FT)",
         "Interference codes {2,1,0,−1}; piecewise reward SNR/SINR/0/−10; two-tier 80/20 sharing (edge ~5, global ~10)",
         "twin/observe.py · oracle.py · swarm/share.py"),
        ("Federated DRL DSA — IEEE TVT 2022 (FT)",
         "Reward = SINR − QoS threshold μ; cluster non-interfering users",
         "oracle.py (qos_margin) · clustering"),
        ("Joint GNN + DQN DSA — Ad Hoc Netw. 2024 (Abs)",
         "Graph-derived spectrum state → action selection",
         "contention graph → option features; d3-lite replaces DQN"),
        ("Improved-FL DSA — JNCA 2025 (Abs)",
         "GNN + DQN + federated aggregation",
         "covered by swarm protocol"),
        ("Swarm DSA — WCNC 2025 (FT)",
         "+1 / −0.5 / −1 reward; leader election; weighted merge",
         "oracle.py (pm1) · swarm/consensus.py"),
        ("Secure swarm-learning DSA — IEEE IoT-J 2025 (FT)",
         "Admission check; performance-score leader; FedAvg; validate before release; auditable log",
         "swarm/consensus.py (HMAC hash-chained log)"),
        ("Machine-unlearning DSA — ICAIIC 2025 (FT)",
         "R = B + αU − γW + βV; location-tagged forgetting; membership-inference audit",
         "mission_utility · distill.py (MIA, region unlearning)"),
        ("IoV spectrum + mobility prediction — WPC 2024 (FT)",
         "P ~ visits/d², top-2, dwell; collaborative max-min colouring argmax b/(D+1)",
         "twin/mobility.py (no LSTM) · coloring teacher"),
    ]),
    ("LEO satellite: pricing, power, trust", "#d9822b", [
        ("Federated DSA for LEO IoT — ICCT 2024 (FT)",
         "max Σ SINR·idle s.t. channel exists/idle, power bounds, ≤1 channel/user; bi-level k-means",
         "hungarian teacher · hard_filter · bilevel_kmeans"),
        ("Privacy-aware pricing & power — arXiv 2407.00814 / TWC 2025 (FT)",
         "Revenue = price·#{budget ≥ price}, budget ~ SINR; ±1 price step, +1/0/−1",
         "teachers/admission.py · admission task"),
        ("Privacy-preserving pricing framework — MSN 2024 (FT)",
         "Reputation ± p̂·l·count; threats: power over cap, access overuse, lease overrun",
         "swarm/reputation.py · quarantine"),
        ("Certified LEO smart management — ICISS 2024 (Abs)",
         "Certified aggregation",
         "covered by consensus log"),
    ]),
    ("LLMs for networking", "#7a5ba6", [
        ("Trend-aware LLM traffic prediction — IEEE IoT-J 2025 (FT)",
         "Labels trend/event/volatility; per-step prompt; same-size calendar/PCA ablation; noise/missing/surge tests",
         "traffic.py: d3-lite typed questions replace LLaMA-LoRA; Chronos replaces CNN-LSTM"),
        ("LLM-as-judge bias in comms — arXiv 2510.12462 (Lam co-author)",
         "Explicit rubrics make judges stricter; audit for bias",
         "rubrics in prompts · bias audit in bench.py"),
    ]),
]

COLS = [("Paper (venue, FT = full text / Abs = abstract)", 0.012, 34),
        ("Mechanism we took", 0.335, 52),
        ("Where it lives in WirelessJEV", 0.70, 40)]

rows = []
for title, color, items in SECTIONS:
    rows.append(("section", title, color))
    for item in items:
        wrapped = [textwrap.wrap(t, w) for t, (_, _, w) in zip(item, COLS)]
        rows.append(("row", wrapped, color))

line_h = 0.0165
heights = [0.032 if r[0] == "section" else line_h * max(len(c) for c in r[1]) + 0.012
           for r in rows]
total = sum(heights) + 0.25
fig = plt.figure(figsize=(14, 14 * total * 0.62), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, total)

y = total - 0.035
ax.text(0.5, y, "What WirelessJEV uses from Bowen Shen, Feng Li and Kwok-Yan Lam",
        ha="center", va="center", fontsize=17, fontweight="bold")
y -= 0.035
ax.text(0.5, y, "Formulations and protocols taken; every learned model (DQN/PPO/GNN/LSTM/LLaMA) "
        "replaced by an exact oracle, a frozen open model, or counting statistics",
        ha="center", va="center", fontsize=10.5, color="#555", style="italic")
y -= 0.03
for name, x, _ in COLS:
    ax.text(x, y, name, fontsize=10.5, fontweight="bold", va="center")
y -= 0.012
ax.plot([0.01, 0.99], [y, y], color="#1f2430", lw=1.2)

stripe = False
for r, h in zip(rows, heights):
    if r[0] == "section":
        y -= h
        ax.add_patch(FancyBboxPatch((0.01, y + 0.004), 0.98, h - 0.008,
                                    boxstyle="round,pad=0,rounding_size=0.004",
                                    fc=r[2], ec="none"))
        ax.text(0.02, y + h / 2, r[1], color="white", fontsize=11.5, fontweight="bold",
                va="center")
        stripe = False
        continue
    y -= h
    if stripe:
        ax.add_patch(plt.Rectangle((0.01, y), 0.98, h, fc="#f3f5f8", ec="none"))
    stripe = not stripe
    ax.add_patch(plt.Rectangle((0.01, y), 0.004, h, fc=r[2], ec="none"))
    for col, (lines, (_, x, _)) in enumerate(zip(r[1], COLS)):
        for k, line in enumerate(lines):
            ax.text(x + (0.008 if col == 0 else 0), y + h - 0.009 - k * line_h, line,
                    fontsize=9.6, va="top",
                    fontweight="bold" if col == 0 else "normal",
                    color="#1f2430" if col != 2 else "#2f6f5e",
                    )

y -= 0.02
ax.plot([0.01, 0.99], [y, y], color="#1f2430", lw=0.8)
y -= 0.025
summary = [
    "Their rewards → oracle mission utility    ·    their constraints → hard filters on plans    ·    "
    "their swarm/federated protocols → consensus merge + two-tier sharing",
    "Reputation / unlearning / MIA audit → quarantine + region forgetting    ·    Hungarian & colouring → teachers    ·    "
    "trend labels → typed d3-lite questions",
    "Not used: their trained weights or reported performance numbers. No code was released by any of these papers.",
]
for k, s in enumerate(summary):
    ax.text(0.5, y - k * 0.022, s, ha="center", va="top", fontsize=9.5,
            color="#c0392b" if k == 2 else "#1f2430")

out = Path(__file__).with_name("paper_map.png")
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.15)
print(out)
