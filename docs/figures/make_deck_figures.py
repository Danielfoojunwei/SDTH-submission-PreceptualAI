"""Render the pitch-deck figures from the simulator and campaign results.

    uv run python docs/figures/make_deck_figures.py

Every image is computed: the coverage map and traces from the 100-robot
warehouse twin (test scene 530), the charts from artifacts/native/*.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import chi2

from wirelessjev.native.phy import LinkCurves
from wirelessjev.native.sim import BLOCKAGE_DB, BLOCK_SLOTS, hidden_draws
from wirelessjev.native.world import NOISE_DBM, _path_losses, build_world
from wirelessjev.twin.warehouse import AP_SITES, HEIGHT, RACK_ROWS, RACK_SEGMENTS, WIDTH, make_warehouse

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"
NAT = ROOT / "artifacts" / "native"
DARK, LIGHT, ORANGE, BLUE, TEAL, GREY, MUTED = "#10202B", "#F3F5F1", "#E2702F", "#2D6CB5", "#3E7C72", "#8A96A0", "#B9C7D0"
plt.rcParams.update({"font.size": 22, "font.family": "DejaVu Sans", "axes.edgecolor": GREY,
                     "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK})
SCENE = 530


def floor(ax, color="#F3F5F1", ap=True):
    for y in RACK_ROWS:
        for x0, x1 in RACK_SEGMENTS:
            ax.add_patch(Rectangle((x0, y - 0.6), x1 - x0, 1.2, color=color, alpha=0.55, lw=0))
    if ap:
        xs, ys = zip(*AP_SITES)
        ax.scatter(xs, ys, s=220, c=ORANGE, edgecolors=LIGHT, linewidths=2.5, zorder=5)
    ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT); ax.set_aspect("equal"); ax.axis("off")


def coverage_hero():
    scene = make_warehouse(SCENE, n_robots=100)
    aps5 = [a for a in scene.aps if a.band == "5g"]
    aps2 = [a for a in scene.aps if a.band == "2g"]
    gx, gy = np.meshgrid(np.linspace(0.5, WIDTH - 0.5, 480), np.linspace(0.5, HEIGHT - 0.5, 320))
    pl = _path_losses(scene, aps5, aps2, gx, gy)                      # (H,W,S,3)
    snr = (15.0 - pl[..., 1] - NOISE_DBM["5g"]).max(axis=2)           # best 5 GHz uplink SNR
    cmap = LinearSegmentedColormap.from_list("ink", ["#0B1720", "#16384F", BLUE, "#7FB2E5", "#F6E7C8", ORANGE])
    fig = plt.figure(figsize=(19.2, 12.8), dpi=100, facecolor=DARK)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(DARK)
    ax.imshow(snr, origin="lower", extent=(0, WIDTH, 0, HEIGHT), cmap=cmap, vmin=-5, vmax=40, interpolation="bilinear")
    floor(ax, color="#0B1720AA")
    xs = [c.x for c in scene.cars]; ys = [c.y for c in scene.cars]
    ax.scatter(xs, ys, s=40, c=LIGHT, alpha=0.9, zorder=6)
    fig.savefig(OUT / "hero_coverage.png", facecolor=DARK); plt.close(fig)


def radiomap_triptych():
    w = build_world(SCENE, duration_s=300.0, survey_s=60)
    rm = w.rmap
    gx, gy = np.meshgrid(np.linspace(1, WIDTH - 1, 240), np.linspace(1, HEIGHT - 1, 160))
    scene = make_warehouse(SCENE, n_robots=1)
    aps5 = [a for a in scene.aps if a.band == "5g"]
    aps2 = [a for a in scene.aps if a.band == "2g"]
    truth = (aps5[0].tx_power_dbm - _path_losses(scene, aps5, aps2, gx, gy))[..., 5, 1]
    mean, std = rm.predict(gx, gy)
    mean, std = mean[..., 5, 1], std[..., 5, 1]
    fig, axs = plt.subplots(1, 2, figsize=(22, 7.2), dpi=100, facecolor=LIGHT)
    for ax, img, title, cm, lo, hi in ((axs[0], truth, "Truth (hidden from every system)", "viridis", -95, -35),
                                       (axs[1], mean, "RadioMap, after 60 s of driving", "viridis", -95, -35)):
        im = ax.imshow(img, origin="lower", extent=(0, WIDTH, 0, HEIGHT), cmap=cm, vmin=lo, vmax=hi)
        floor(ax, color="#FFFFFF", ap=False)
        ax.scatter([AP_SITES[5][0]], [AP_SITES[5][1]], s=300, c=ORANGE, edgecolors="white", linewidths=3, zorder=6)
        ax.set_title(title, fontsize=26, color=DARK, pad=12)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.tick_params(labelsize=18)
        cb.set_label("dBm" if "error" not in title else "dB", fontsize=18)
    fig.tight_layout(); fig.savefig(OUT / "radiomap_triptych.png", facecolor=LIGHT); plt.close(fig)
    return w


def link_trace(w):
    """A 20 s window for one robot: SNR on its best AP path and on its second
    AP path (re-ranked every 100 ms as it moves), with the hidden fading and
    blockage every system faces. Below the dashed line a packet on that path
    fails even at the most robust rate."""
    n_blocks = 3000
    fade, blocked, _ = hidden_draws(w, n_blocks * BLOCK_SLOTS)
    th = LinkCurves().threshold_db("5g", 0.99)[0]
    kp = np.minimum(np.arange(n_blocks) * BLOCK_SLOTS // 20, w.snr.shape[0] - 1)
    mean = w.snr[kp][..., 1]                                           # (T,N,S) large-scale
    inst = mean + fade[:n_blocks][..., 1] - BLOCKAGE_DB * blocked[:n_blocks]
    order = np.argsort(-mean, axis=2)
    T = np.arange(n_blocks)[:, None]; R = np.arange(w.n)[None, :]
    a = inst[T, R, order[..., 0]]; b = inst[T, R, order[..., 1]]       # (T,N)
    rescue = (a < th) & (b >= th); lost = (a < th) & (b < th)
    win = 200
    best, bi, bt = -1, 0, 0
    for r in range(w.n):
        c = np.convolve(rescue[:, r].astype(float), np.ones(win), "valid")
        t0 = int(c.argmax())
        if c[t0] > best:
            best, bi, bt = c[t0], r, t0
    sl = slice(bt, bt + win)
    t = np.arange(win) * 0.1
    fig, ax = plt.subplots(figsize=(24, 7), dpi=100, facecolor=LIGHT)
    ax.set_facecolor(LIGHT)
    ax.plot(t, a[sl, bi], color=BLUE, lw=4, label="Best AP path")
    ax.plot(t, b[sl, bi], color=ORANGE, lw=4, label="Second AP path")
    ax.axhline(th, color=DARK, lw=2, ls="--")
    ax.text(0.2, th + 0.8, "packet lost below this line", ha="left", fontsize=20, color=DARK)
    rr, ll = rescue[sl, bi], lost[sl, bi]
    ax.scatter(t[rr], a[sl, bi][rr], s=260, marker="o", facecolors="none", edgecolors=ORANGE, linewidths=4,
               zorder=6, label="Rescued by the second path")
    ax.scatter(t[ll], a[sl, bi][ll], s=260, marker="x", c=DARK, linewidths=4, zorder=6, label="Both paths down")
    ax.set_xlabel("seconds"); ax.set_ylabel("uplink SNR (dB)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=20, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(); fig.savefig(OUT / "link_trace.png", facecolor=LIGHT); plt.close(fig)
    return {"robot": int(bi), "rescued_blocks": int(rr.sum()), "both_down_blocks": int(ll.sum()),
            "fleet_rescued_share": float(rescue.sum() / max(rescue.sum() + lost.sum(), 1))}


def ci(k, n):
    lo = chi2.ppf(0.025, 2 * k) / 2 / n * 1e6 if k > 0 else 0.0
    return lo, chi2.ppf(0.975, 2 * k + 2) / 2 / n * 1e6


def results_bars():
    rows = json.loads((NAT / "wifi_vs_5g.json").read_text())["rows"]
    arms = [("vertical_client", "WirelessJEV, two AP paths", ORANGE), ("nr_daps", "Private 5G, make-before-break", BLUE),
            ("nr_30ms", "Private 5G, 30 ms handover", BLUE), ("vertical_single", "WirelessJEV, one radio", "#E9A87E"),
            ("mlo_split", "Wi-Fi 7, fixed rule", GREY)]
    conds = [("lenient", "Clean radio"), ("main", "Realistic"), ("stress", "Hardest")]
    fig, axs = plt.subplots(1, 3, figsize=(24, 7.5), dpi=100, facecolor=LIGHT, sharey=True)
    for ax, (c, cname) in zip(axs, conds):
        vals, los, his = [], [], []
        for a, _, _ in arms:
            rs = [r for r in rows if r["condition"] == c and r["arm"] == a]
            k = sum(r["control_misses"] for r in rs); n = sum(r["control_sent"] for r in rs)
            v = 1e6 * k / n; lo, hi = ci(k, n)
            vals.append(max(v, 0.3)); los.append(max(v, 0.3) - max(lo, 0.3)); his.append(hi - max(v, 0.3))
        y = np.arange(len(arms))[::-1]
        ax.barh(y, vals, xerr=[los, his], color=[a[2] for a in arms], capsize=6, error_kw={"lw": 2})
        for yy, v, h in zip(y, vals, his):
            ax.text((v + h) * 1.35, yy, f"{v:,.1f}" if v < 100 else f"{v:,.0f}", va="center", fontsize=20,
                    color=DARK, fontweight="bold" if yy == y[0] else "normal")
        ax.set_xscale("log"); ax.set_xlim(0.3, 60000); ax.set_title(cname, fontsize=26, color=DARK)
        ax.set_facecolor(LIGHT)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("missed control packets per million (log)", fontsize=18)
    axs[0].set_yticks(np.arange(len(arms))[::-1]); axs[0].set_yticklabels([a[1] for a in arms], fontsize=20)
    fig.tight_layout(); fig.savefig(OUT / "results_bars.png", facecolor=LIGHT); plt.close(fig)


def causes_chart():
    data = [("Private 5G", 46, [0, 5, 6, 0]), ("Wi-Fi 8, seamless", 304, [0, 7, 66, 0]),
            ("WirelessJEV, one radio", 454, [34, 20, 55, 0]), ("WirelessJEV, two AP paths", 12.5, [1, 0, 2, 0])]
    labels = ["No radio up", "All AP paths blocked", "Weak link", "Contention"]
    cols = [GREY, DARK, BLUE, "#C9D3CF"]
    fig, ax = plt.subplots(figsize=(22, 6.5), dpi=100, facecolor=DARK)
    ax.set_facecolor(DARK)
    y = np.arange(len(data))[::-1]
    for yy, (name, rate, counts) in zip(y, data):
        tot = sum(counts); left = 0
        for c, col, lab in zip(counts, ["#8A96A0", "#F3F5F1", "#7FB2E5", "#E2702F"], labels):
            share = rate * c / tot if tot else 0
            ax.barh(yy, share, left=left, color=col, label=lab if yy == y[0] else None)
            left += share
        ax.text(left + 6, yy, f"{rate:g} per million", va="center", fontsize=22, color=LIGHT)
    ax.set_yticks(y); ax.set_yticklabels([d[0] for d in data], fontsize=22, color=LIGHT)
    ax.tick_params(colors=LIGHT); ax.set_xlim(0, 560)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    leg = ax.legend(loc="lower right", frameon=False, fontsize=20)
    for tx in leg.get_texts():
        tx.set_color(LIGHT)
    fig.tight_layout(); fig.savefig(OUT / "miss_causes.png", facecolor=DARK); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coverage_hero()
    w = radiomap_triptych()
    print("trace", link_trace(w))
    results_bars()
    causes_chart()
    print(sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
