# Admission research kernel derived from frontier.sim; absent-hook equivalence is tested.
# Versioned research kernel derived from native.sim; default equivalence is tested.
"""Cross-layer slot simulator (10 ms) with legacy, 802.11r, default multi-link
and the native WirelessJEV + RadioMap stack.

Layers modelled per slot:
- L1/L2: per-MCS packet success from ns-3 curves; STR dual-link halves each
  link's streams (1SS, -3 dB); 6 GHz uses double rate at 160 MHz.
- L3: airtime shared within contention domains after external occupancy,
  priority control > video > bulk.
- L4: multi-link device per site: EMLSR (one active link) or STR (two).
- L5: roams with explicit outage (legacy 4 s, 802.11r 60 ms, pre-keyed 30 ms);
  link switches inside one multi-link AP cost nothing.
- L6-L8: IP retention is part of the roam cost; data timing defers bulk when
  the route forecast says the link is about to improve.
Hidden from every policy: fading (Nakagami m=2 per 100 ms block per link)
and blockage (-15 dB on all links of one robot-site path).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np

from .native.phy import LinkCurves
from .native.world import BANDS, CLIENT_TX_DBM, HORIZONS, NOISE_DBM, POS_DT, World

SLOT = 0.01
CONTROL_BITS = 600 * 8
CONTROL_PERIOD = 5           # slots (20 Hz)
CONTROL_DEADLINE = 5         # slots (50 ms)
VIDEO_DEADLINE = 15          # slots (150 ms)
MAC_EFF = 0.6
FRAME_OVERHEAD_S = 100e-6
ATTEMPTS_PER_SLOT = 3
BLOCK_SLOTS = 10             # fading block 100 ms
BLOCKAGE_DB = 15.0
BLOCKAGE_RATE_PER_MIN = 1.5  # per robot, spread over its sites
BLOCKAGE_MEAN_S = 2.0
FADE_M = 2
CAUSES = ("no_radio_up", "all_paths_blocked", "weak_link", "contention", "other")
SHARED_SWITCH_S = 0.002      # retune to the second AP's channel and back
ONBOARD_SNR_DB = -5.0        # 1x1 (no combining gain) and a weaker PCB antenna
WIFI8_HYST_DB = 3.0          # seamless multi-AP switching hysteresis
SAFETY_GAP_SLOTS = 20       # robot stops when control is silent for > 200 ms...
RESTART_S = 1.0             # ...and needs 1 s to resume after control returns
BULK_BUFFER_S = 20.0       # map uploads queue on the robot; deferral sends later


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str                     # legacy | mlo_default | native
    roam_cost_s: float = 4.0
    scan_period_s: float = 5.0    # background scans (outage SCAN_S each)
    # native components
    route_forecast: bool = True
    duplication: bool = True
    predictive_mcs: bool = True
    planned_roam: bool = True
    stagger: bool = True
    data_timing: bool = True
    reflex: bool = True
    layered: bool = False
    layered_gain_db: float = 4.0
    risk_threshold: float = 0.05
    static_split: bool = False    # mlo_split: control pinned to 2.4 GHz
    gate: bool = False            # trust gate on planned roams (NATIVE_GATE.md)
    gate_z: float = 2.0
    belief_scale: float = 1.0     # <1: overconfident about the map
    dual_radio: bool = False      # second client radio on another AP (make-before-break)
    sched_control: bool = False   # AP schedules control slots (trigger-based / restricted TWT)
    nr_ho_cost_s: float = 0.03    # kind="nr": handover interruption (0 = DAPS)
    power_control: bool = False   # fleet transmit-power politeness (needs interference=True)
    cadence_roam: bool = False    # fire planned roams right after a control delivery
    adaptive_video: bool = False  # video bitrate follows the predicted link
    online_map: bool = False      # RadioMap keeps learning from robots' measurements
    # Mission-compiled knobs (see intent_policy); defaults reproduce earlier runs.
    roam_gain_db: float = 5.0
    load_penalty: float = 0.15
    focus_robots: tuple = ()      # robot indices the mission names
    dual_focus_only: bool = False # second radio only on focus robots
    avoid_occupied: float = 0.0   # penalty for links on channels with protected equipment
    video_share: float = 0.6      # adaptive video encodes at this share of predicted goodput
    radio2: str = "be200"         # second association: "be200" (second card), "onboard" (Pi
                                  # CYW43455: 1x1 Wi-Fi 5, no 6 GHz, PCB antenna), or "shared"
                                  # (time-shared on the same BE200 / Wi-Fi 8 multi-AP: used only
                                  # after the primary chain fails, plus a channel-switch cost)
    cadence_scan: bool = False    # background scans start right after a control delivery


SCAN_S = 0.03

# Private 5G baseline (kind="nr"), deliberately generous to 5G: n78 100 MHz
# TDD on the same 16 sites, licensed and clean, scheduled (no contention, no
# inter-cell interference modelled), UE at 23 dBm vs the BE200's 15 dBm, 3.5 vs
# 5.2 GHz path loss, CQI-driven link adaptation, A3 handover with ~200 ms TTT.
NR_SNR_OFFSET_DB = 20 * math.log10(5.2 / 3.5) + (23.0 - 15.0) - 10 * math.log10(100 / 80)
NR_RATE_SCALE = 100 / 80
NR_UL_EFF = 0.85 * 0.35       # scheduler efficiency x uplink share of a DDDSU frame
NR_A3_DB = 3.0


def policies(roam_cost_legacy=4.0):
    native = Policy("native", "native", roam_cost_s=0.03, scan_period_s=30.0)
    out = [Policy("legacy", "legacy", roam_cost_s=roam_cost_legacy),
           Policy("ft11r", "legacy", roam_cost_s=0.06),
           Policy("mlo_default", "mlo_default", roam_cost_s=0.06),
           Policy("mlo_split", "mlo_default", roam_cost_s=0.06, static_split=True),
           native]
    for comp in ("route_forecast", "duplication", "predictive_mcs", "planned_roam",
                 "stagger", "data_timing", "reflex"):
        p = replace(native, name=f"native-no_{comp}", **{comp: False})
        if comp == "planned_roam":
            p = replace(p, roam_cost_s=0.06, scan_period_s=5.0)
        out.append(p)
    out.append(replace(native, name="native_layered", layered=True))
    return out


def _mcs_for(curves, band, snr, target=None):
    """Reactive/predictive MCS: highest MCS whose 90%-success threshold is
    below `snr`; with target, the highest whose curve meets target at snr."""
    th = curves.threshold_db(band, 0.9 if target is None else target)
    idx = np.searchsorted(th, snr, side="right") - 1
    return np.clip(idx, 0, len(th) - 1)


def _tally(store, flags, mask):
    """Attribute each missed packet to the most serious condition it met."""
    for i in np.nonzero(mask)[0]:
        f = flags[i]
        for bit, name in enumerate(CAUSES[:-1]):
            if f & (1 << bit):
                store[name] += 1
                break
        else:
            store["other"] += 1


def _chain(curves, links, mcss, snr_eff, str_mode, q=None):
    """Retry chain within one slot: attempt a uses link links[:, a] at MCS
    mcss[:, a]; q (N,) scales every attempt (e.g. 1 - collision probability).
    Returns (success in slot, expected airtime fraction)."""
    n, k = links.shape
    p = np.zeros((n, k))
    t = np.zeros((n, k))
    half = np.where(str_mode, 0.5, 1.0)
    for a in range(k):
        for b in range(3):
            sel = links[:, a] == b
            if not sel.any():
                continue
            mm = mcss[sel, a]
            p[sel, a] = curves.success(BANDS[b], mm, snr_eff[sel, b])
            t[sel, a] = (CONTROL_BITS / (curves.rates(BANDS[b])[mm] * half[sel] * 1e6)
                         + FRAME_OVERHEAD_S) / SLOT
    if q is not None:
        p = p * q[:, None]
    reach = np.cumprod(np.concatenate([np.ones((n, 1)), 1 - p[:, :-1]], axis=1), axis=1)
    return 1 - np.prod(1 - p, axis=1), (reach * t).sum(1)


def _mrr_chain(link, mcs, ar):
    """Minstrel-style multi-rate retry on one link: MCS, MCS-2, MCS0."""
    m = np.stack([mcs[b] for b in range(3)], 1)[ar, link]
    links = np.stack([link, link, link], 1)
    return links, np.stack([m, np.maximum(m - 2, 0), np.zeros_like(m)], 1)


def hidden_draws(world: World, n_slots: int, blockage_rate: float = BLOCKAGE_RATE_PER_MIN):
    """Fading (dB) and blockage per 100 ms block, never seen by any policy."""
    n, s = world.n, world.sites.shape[0]
    hid = np.random.default_rng(77 + world.seed)
    n_blocks = n_slots // BLOCK_SLOTS + 1
    fade_db = 10 * np.log10(hid.gamma(FADE_M, 1 / FADE_M, size=(n_blocks, n, s, 3))).astype(np.float32)
    p_on = blockage_rate / 60 * (BLOCK_SLOTS * SLOT) / 4   # ~4 relevant sites
    p_off = (BLOCK_SLOTS * SLOT) / BLOCKAGE_MEAN_S
    blocked = np.zeros((n_blocks, n, s), dtype=bool)
    state = np.zeros((n, s), dtype=bool)
    for k in range(n_blocks):
        u = hid.random((n, s))
        state = np.where(state, u > p_off, u < p_on)
        blocked[k] = state
    return fade_db, blocked, hid


def simulate(world: World, policy: Policy, *, seed: int = 0, curves: LinkCurves | None = None,
             duration_s: float | None = None, blockage_rate: float = BLOCKAGE_RATE_PER_MIN,
             collision_lambda: float = 0.0, interference: bool = False,
             roam_gate=None, decision_trace=None, forecast_queue=None, planner="native", scan_duration_s=None, handover_admission=None) -> dict:
    if policy.kind == "nr":
        return simulate_nr(world, policy, seed=seed, curves=curves, duration_s=duration_s,
                           blockage_rate=blockage_rate)
    curves = curves or LinkCurves()
    rng = np.random.default_rng(1_000_003 * (world.seed + 1) + seed)   # same draws for all policies
    n, s = world.n, world.sites.shape[0]
    duration = duration_s or world.duration_s
    n_slots = int(round(duration / SLOT))
    slots_per_pos = int(round(POS_DT / SLOT))
    slots_per_dec = 100
    rates = {b: curves.rates(b) for b in BANDS}
    th90 = {b: curves.threshold_db(b, 0.9) for b in BANDS}
    th99 = {b: curves.threshold_db(b, 0.99) for b in BANDS}
    th99_0 = np.array([th99[b][0] for b in BANDS])
    n_dom = len(world.avail)
    ar = np.arange(n)

    # Hidden processes, identical across policies (own RNG streams).
    fade_db, blocked, hid = hidden_draws(world, n_slots, blockage_rate)
    scan_phase = hid.integers(0, 10_000, size=n)

    noise_mw = 10 ** (world.noise_db / 10)
    ipen_site = np.zeros((s, 3))          # interference rise over noise per site/band, mW
    tx_adj = np.zeros(n)                  # client transmit-power change, dB
    act_acc = np.zeros(n)

    def true_snr(k_slot, sites_idx=None):
        kp = min(k_slot // slots_per_pos, world.snr.shape[0] - 1)
        kb = k_slot // BLOCK_SLOTS
        base = world.snr[kp] + fade_db[kb] - BLOCKAGE_DB * blocked[kb][..., None]
        if sites_idx is None:
            return base
        x = base[ar, sites_idx] + tx_adj[:, None]
        if interference:
            x = x - 10 * np.log10(1 + ipen_site[sites_idx] / noise_mw[None, :])
        return x                                                     # (N,3)

    def rssi_dl(k_slot):
        kp = min(k_slot // slots_per_pos, world.snr.shape[0] - 1)
        kb = k_slot // BLOCK_SLOTS
        return world.rssi_dl[kp] + fade_db[kb] - BLOCKAGE_DB * blocked[kb][..., None]

    # --- per-robot state ----------------------------------------------------
    k0 = 0
    init = rssi_dl(0)
    site = np.argmax(init[:, :, 1], axis=1)                 # start on best 5 GHz site
    band_single = np.full(n, 1)                              # legacy BSS band
    outage_until = np.zeros(n, dtype=int)
    ctrl_link = np.full(n, 1)
    second_link = np.full(n, -1)                             # STR partner (-1 = EMLSR)
    bulk_link = np.full(n, 1)
    mcs = {b: np.zeros(n, dtype=int) for b in range(3)}
    meas = true_snr(0, site)                                 # last measured SNR per link
    bulk_cap = np.ones(n)
    backlog = np.zeros(n)
    roams = np.zeros(n, dtype=int)
    last_scan = None

    # traffic state
    ctrl_age = np.full(n, -1)                                # -1 = no pending packet
    ctrl_sent = ctrl_missed = 0
    ctrl_miss_r = np.zeros(n)
    ctrl_total_r = np.zeros(n)
    gap_cur = np.zeros(n)
    gap_max = np.zeros(n)
    vq = np.zeros((n, VIDEO_DEADLINE))                       # video bits by age
    video_gen = video_ok = 0.0
    video_req_bits = 0.0
    video_gen_r = np.zeros(n)
    video_ok_r = np.zeros(n)
    bulk_bits_r = np.zeros(n)
    airtime_used = 0.0
    dup_slots = 0
    has_video = world.video_mbps > 0

    roam_log = []
    ctrl_down = ctrl_fail = ctrl_congested = 0
    win_miss = np.zeros(n, dtype=bool)
    windows = clean_windows = 0
    due = np.zeros(n, dtype=bool)                # cadence_scan: a scan is waiting for a quiet moment
    scan_until = np.zeros(n, dtype=int)
    cause_flags = np.zeros(n, dtype=int)         # bit per cause seen while a packet waits
    miss_causes = {c: 0 for c in CAUSES}
    stopped_slots = 0
    stop_events = 0
    prev_gap = np.zeros(n)
    video_rate = world.video_mbps.copy()
    dual = policy.kind == "native" and policy.dual_radio
    focus = np.zeros(n, dtype=bool)
    focus[list(policy.focus_robots)] = True
    r2_off = ONBOARD_SNR_DB if policy.radio2 == "onboard" else 0.0
    dual_mask = (focus if policy.dual_focus_only else np.ones(n, dtype=bool)) if dual else np.zeros(n, dtype=bool)
    occ_pen = policy.avoid_occupied * (1 - world.avail[world.domain])      # (S,3)
    site2 = np.full(n, -1)
    outage2_until = np.zeros(n, dtype=int)
    mcs2 = {b: np.zeros(n, dtype=int) for b in range(3)}
    link2 = np.full(n, 1)
    swaps = 0

    native_delta = np.zeros((n, 3))  # reactive control until the first forecast is ready
    pending_roam = np.full(n, -1)          # cadence_roam: target waiting for a quiet moment

    def move_primary(i, target, cost_s):
        nonlocal swaps
        dual = bool(dual_mask[i])
        if policy.cadence_roam and cost_s < CONTROL_PERIOD * SLOT and not (dual and site2[i] == target):
            pending_roam[i] = target       # executed right after the next control delivery
            return
        if dual and site2[i] == target:
            # Make-before-break: the second radio is already on the target AP.
            site[i], site2[i] = site2[i], site[i]
            outage_until[i], outage2_until[i] = outage2_until[i], outage_until[i]
            swaps += 1
        else:
            site[i] = target
            outage_until[i] = k + int(round(cost_s / SLOT))
        roams[i] += 1
        roam_log.append(k)

    cochan = [(world.channel[:, b][:, None] == world.channel[:, b][None, :])
              & (world.domain[:, b][:, None] != world.domain[:, b][None, :]) for b in range(3)]

    online = policy.kind == "native" and policy.online_map and world.rmap is not None
    if online:
        import copy
        world = copy.copy(world)
        world.rmap = copy.deepcopy(world.rmap)       # each run learns from its own operation
    obs_buf = {}
    map_updates = 0

    for k in range(n_slots):
        if online and k % (5 * BLOCK_SLOTS) == 0 and k > 0:
            # Beacon RSSI each robot measures on its AP's three links.
            kp = min((k - BLOCK_SLOTS) // slots_per_pos, world.pos.shape[0] - 1)
            r = rssi_dl(k - BLOCK_SLOTS)[ar, site]                     # (N,3)
            xy = world.pos[kp]
            for b in range(3):
                for i in range(n):
                    if r[i, b] >= -90.0:
                        obs_buf.setdefault((int(site[i]), b), []).append((xy[i, 0], xy[i, 1], r[i, b]))
        if online and k % 1000 == 999:
            for (ss, bb), rows in obs_buf.items():
                a = np.array(rows)
                world.rmap.update(ss, bb, a[:, 0], a[:, 1], a[:, 2])
            obs_buf = {}
            map_updates += 1
        if interference and k % BLOCK_SLOTS == 0:
            # Co-channel stations in *other* contention domains do not defer to
            # each other: their uplink airtime is interference at our AP.
            kp = min(k // slots_per_pos, world.snr.shape[0] - 1)
            act = act_acc / BLOCK_SLOTS
            act_acc[:] = 0.0
            for b in range(3):
                prx = 10 ** ((world.snr[kp][:, :, b] + world.noise_db[b] + tx_adj[:, None]) / 10)  # (N,S)
                w = prx * (act * (bulk_link == b))[:, None]
                ipen_site[:, b] = (cochan[b][:, site] * w.T).sum(1)
        # ---------------- measurements (lagged by one fading block) --------
        if k % BLOCK_SLOTS == 0:
            meas = true_snr(max(0, k - BLOCK_SLOTS), site)
            for b in range(3):
                band = BANDS[b]
                m = meas[:, b] - (3.0 if policy.kind != "legacy" else 0.0) * (second_link >= 0)
                mcs[b] = _mcs_for(curves, band, m - 2.0)          # Minstrel-like backoff

        # ---------------- decisions -------------------------------------------
        if policy.kind in ("legacy", "mlo_default") and k % slots_per_dec == 0:
            r = rssi_dl(max(0, k - BLOCK_SLOTS))                  # (N,S,3)
            if policy.kind == "legacy":
                cur = r[ar, site, band_single]
                cand = np.maximum(r[:, :, 1] + 5.0, r[:, :, 0])   # band steering prefers 5 GHz
                best_site = np.argmax(cand, axis=1)
                best_band = np.where(r[ar, best_site, 1] + 5.0 >= r[ar, best_site, 0], 1, 0)
                gain = r[ar, best_site, best_band] - cur
                go = (cur < -75.0) & (gain >= 8.0) & (outage_until <= k)
                band_single = np.where(go, best_band, band_single)
                ctrl_link = band_single.copy()
                bulk_link = band_single.copy()
            else:
                cur = r[ar, site].max(axis=1)
                best_site = np.argmax(r.max(axis=2), axis=1)
                gain = r[ar, best_site].max(axis=1) - cur
                go = (cur < -75.0) & (gain >= 8.0) & (outage_until <= k)
            for i in np.nonzero(go & (best_site != site))[0]:
                site[i] = best_site[i]
                outage_until[i] = k + int(round(policy.roam_cost_s / SLOT))
                roams[i] += 1
                roam_log.append(k)

        if policy.kind == "wifi8" and k % BLOCK_SLOTS == 0:
            # Wi-Fi 8 seamless mobility (generous): one radio, any AP of the
            # domain, switched every 100 ms on lagged measurements, no outage.
            lag = true_snr(max(0, k - BLOCK_SLOTS))                      # (N,S,3)
            best_site = np.argmax(lag.max(axis=2), axis=1)
            cur = lag[ar, site].max(axis=1)
            switch = lag[ar, best_site].max(axis=1) > cur + WIFI8_HYST_DB
            roams += switch
            site = np.where(switch, best_site, site)
            meas = true_snr(max(0, k - BLOCK_SLOTS), site)
            for b in range(3):
                mcs[b] = _mcs_for(curves, BANDS[b], meas[:, b] - 2.0)
            tput = np.stack([curves.rates(BANDS[b])[mcs[b]] *
                             curves.success(BANDS[b], mcs[b], meas[:, b]) for b in range(3)], 1)
            bulk_link = np.argmax(tput, axis=1)
            ctrl_link = np.argmax(meas - th99_0[None, :], axis=1)

        if policy.kind == "mlo_default" and k % BLOCK_SLOTS == 0:
            tput = np.stack([curves.rates(BANDS[b])[mcs[b]] *
                             curves.success(BANDS[b], mcs[b], meas[:, b]) for b in range(3)], 1)
            ctrl_link = np.argmax(tput, axis=1)                    # EMLSR on best lagged throughput
            bulk_link = ctrl_link.copy()
            if policy.static_split:
                ctrl_link = np.zeros(n, dtype=int)

        arrived = None if forecast_queue is None else forecast_queue.take(k)
        do_native = (k % slots_per_dec == 0) if forecast_queue is None else arrived is not None
        if policy.kind == "native" and do_native:
            d = min(k // slots_per_dec, world.forecast.shape[0] - 1) if forecast_queue is None else arrived
            fc = world.forecast[d]                                 # (N,H,S,3)
            sigma_now = world.sigma_belief[d]
            if forecast_queue is not None:
                fc = forecast_queue.advance(fc)
            if online:
                pl = world.plan[d]
                mean, std = world.rmap.predict(pl[..., 0], pl[..., 1])
                fc = (mean - world.tx_dl + CLIENT_TX_DBM - world.noise_db).astype(np.float32)
                sigma_now = std[:, 0]
            if not policy.route_forecast:
                fc = np.repeat(fc[:, :1], len(HORIZONS), axis=1)
            sig = 5.0                                              # unforecast fading/blockage spread
            # Roam planning over 30 s: mean best-link SNR, load-aware.
            if policy.planned_roam:
                # Lookahead scales with switching cost: with ~30 ms pre-keyed
                # roams only the next few seconds matter.
                w = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=float)
                score = (fc.max(axis=3) * w[None, :, None]).sum(1) / w.sum()   # (N,S)
                if planner == "reliability":
                    from .frontier.timing import reliability_scores
                    score = reliability_scores(fc)
                elif planner != "native":
                    raise ValueError("Unknown native planner")
                load = np.bincount(site, minlength=s).astype(float)
                score = score - policy.load_penalty * load[None, :]
                best = np.argmax(score, axis=1)
                gain = score[ar, best] - score[ar, site]
                go = (gain >= policy.roam_gain_db) & (best != site) & (outage_until <= k)
                decision = None
                if roam_gate is not None or decision_trace is not None or handover_admission is not None:
                    from .native.gate import RoamDecision
                    decision = RoamDecision.capture(d,site,best,gain,go,score)
                confident = np.ones(n,dtype=bool)
                if policy.gate:
                    sb = sigma_now * policy.belief_scale                     # (N,S,3)
                    sbest = sb.max(axis=2)
                    z = gain / np.sqrt(sbest[ar, site] ** 2 + sbest[ar, best] ** 2 + 1e-9)
                    confident = z >= policy.gate_z
                    if roam_gate is not None:
                        from .native.gate import checked_mask
                        confident = checked_mask(roam_gate(decision),n)
                    go = go & confident
                    if handover_admission is not None:
                        from .native.gate import checked_mask
                        go = go & checked_mask(handover_admission(decision), n)
                    # Not confident: the reactive rule decides for this robot.
                    r = rssi_dl(max(0, k - BLOCK_SLOTS))
                    cur = r[ar, site].max(axis=1)
                    rbest = np.argmax(r.max(axis=2), axis=1)
                    rgain = r[ar, rbest].max(axis=1) - cur
                    rgo = (~confident) & (cur < -75.0) & (rgain >= 8.0) & (rbest != site) & (outage_until <= k)
                    if handover_admission is not None:
                        from .native.gate import RoamDecision, checked_mask
                        reactive_decision = RoamDecision.capture(d, site, rbest, rgain, rgo, score)
                        rgo = rgo & checked_mask(handover_admission(reactive_decision), n)
                    for i in np.nonzero(rgo)[0]:
                        # Still our client: pre-keyed and cadence-timed like a planned roam.
                        move_primary(i, rbest[i], policy.roam_cost_s)
                budget = {}
                for i in np.nonzero(go)[0]:
                    if policy.stagger and budget.get(best[i], 0) >= 2:
                        continue
                    budget[best[i]] = budget.get(best[i], 0) + 1
                    move_primary(i, best[i], policy.roam_cost_s)
                if decision_trace is not None:
                    decision_trace(decision,confident.copy(),site.copy())
            else:
                r = rssi_dl(max(0, k - BLOCK_SLOTS))
                cur = r[ar, site].max(axis=1)
                best_site = np.argmax(r.max(axis=2), axis=1)
                gain = r[ar, best_site].max(axis=1) - cur
                go = (cur < -75.0) & (gain >= 8.0) & (outage_until <= k)
                for i in np.nonzero(go & (best_site != site))[0]:
                    site[i] = best_site[i]
                    outage_until[i] = k + int(round(policy.roam_cost_s / SLOT))
                    roams[i] += 1
                    roam_log.append(k)
            # Control link: highest predicted 5 s minimum on the current site.
            near = fc[:, :6][ar, :, site]                          # (N,6,3)
            if policy.route_forecast:   # map bias cancels in the change along the route
                near = meas[:, None, :] + (near - near[:, :1])
            else:
                near = np.repeat(meas[:, None, :], 6, axis=1)
            pmin = near.min(axis=1)                                # (N,3)
            ctrl_link = np.argmax(pmin, axis=1)
            th_ctrl = np.array([th99[b][0] for b in BANDS])        # control at MCS0
            margin = pmin - th_ctrl[None, :]
            risk = 0.5 * (1 - np.vectorize(math.erf)(margin / (sig * math.sqrt(2))))
            primary_risk = risk[ar, ctrl_link]
            if policy.duplication:
                order = np.argsort(-margin, axis=1)
                second_link = np.where(primary_risk > policy.risk_threshold, order[:, 1], -1)
            else:
                second_link = np.full(n, -1)
            # Data timing: defer bulk if the link improves by > 6 dB within 5 s.
            if policy.data_timing:
                now_best = near[:, 0].max(axis=1)
                later_best = near[:, 1:].max(axis=(1, 2))
                bulk_cap = np.where(later_best - now_best > 6.0, 0.0, 1.0)
            if dual:
                # Second radio: best-scoring *other* site (an independent path
                # for blockage), moved only when it is not carrying the robot.
                sc2 = (fc[:, :6].max(axis=3)).mean(1)              # (N,S)
                sc2[ar, site] = -np.inf
                cand = np.argmax(sc2, axis=1)
                move2 = (site2 < 0) | ((cand != site2) & (sc2[ar, cand] - sc2[ar, np.maximum(site2, 0)] >= 5.0))
                move2 &= outage2_until <= k
                for i in np.nonzero(move2)[0]:
                    site2[i] = cand[i]
                    outage2_until[i] = k + int(round(policy.roam_cost_s / SLOT))
            fc_now = fc[:, 0][ar, site]                            # (N,3)
            fc_next = fc[:, 1][ar, site]
            native_delta = (fc_next - fc_now)                      # dB per second along route

        if policy.kind == "native" and k % BLOCK_SLOTS == 0:
            # Route-as-pilot: lagged measurement + route-predicted change.
            est = meas.copy()
            if policy.route_forecast and policy.predictive_mcs:
                est = est + native_delta * (BLOCK_SLOTS * SLOT)
            for b in range(3):
                band = BANDS[b]
                e = est[:, b] - 3.0 * (second_link >= 0)
                if policy.predictive_mcs:
                    mcs[b] = _mcs_for(curves, band, e - 2.0)
                else:
                    mcs[b] = _mcs_for(curves, band, meas[:, b] - 2.0 - 3.0 * (second_link >= 0))
            tput = np.stack([curves.rates(BANDS[b])[mcs[b]] *
                             curves.success(BANDS[b], mcs[b], est[:, b]) for b in range(3)], 1)
            tput = tput * (1 - occ_pen[site] / max(policy.avoid_occupied, 1e-9) * min(policy.avoid_occupied, 1.0))
            bulk_link = np.argmax(tput, axis=1)                    # TTLM: bulk on the fast link
            if policy.adaptive_video:
                # Encode for 60% of the predicted goodput on the bulk link, never above demand.
                g = np.array([curves.rates(BANDS[b])[mcs[b][i]] * curves.success(BANDS[b], mcs[b][i:i + 1], est[i:i + 1, b])[0]
                              for i, b in enumerate(bulk_link)]) * MAC_EFF
                share = np.where(focus, 1.0, policy.video_share)
                video_rate = np.where(world.video_mbps > 0,
                                      np.minimum(world.video_mbps, np.maximum(0.5, share * g)), 0.0)
            if policy.power_control:
                # Politeness: a robot with a very strong link lowers its power so
                # co-channel APs elsewhere hear less of it; never below the MCS ceiling.
                top = np.array([th90[b][-1] for b in BANDS])
                head = est[ar, bulk_link] - tx_adj - top[bulk_link]
                tx_adj = -np.clip(np.floor((head - 3.0) / 5.0) * 5.0, 0.0, 10.0)
                tx_adj[focus] = 0.0                                    # never quieten a named robot
            if policy.reflex:
                margin_now = est - np.array([th99[b][0] for b in BANDS])[None, :] - occ_pen[site]
                ctrl_link = np.where(margin_now[ar, ctrl_link] < 3.0,
                                     np.argmax(margin_now, axis=1), ctrl_link)

        if dual and k % BLOCK_SLOTS == 0:
            meas2 = true_snr(max(0, k - BLOCK_SLOTS), np.maximum(site2, 0)) + r2_off
            for b in range(3):
                mcs2[b] = _mcs_for(curves, BANDS[b], meas2[:, b] - 2.0)
            m2 = meas2 - np.array([th99[b][0] for b in BANDS])[None, :]
            if policy.radio2 == "onboard":
                m2[:, 2] = -np.inf                                  # no 6 GHz on the onboard chip
            link2 = np.argmax(m2, axis=1)

        # ---------------- scans ------------------------------------------------
        if policy.cadence_scan:
            due |= ((k + scan_phase) % int(round(policy.scan_period_s / SLOT))) == 0
            scanning = scan_until > k
        else:
            scanning = ((k + scan_phase) % int(round(policy.scan_period_s / SLOT))) < int(round((SCAN_S if scan_duration_s is None else scan_duration_s) / SLOT))
        up = (outage_until <= k) & ~scanning

        # ---------------- traffic arrivals -------------------------------------
        new_ctrl = ((k + scan_phase) % CONTROL_PERIOD) == 0
        if new_ctrl.any():
            stale = new_ctrl & (ctrl_age >= 0)                    # previous not delivered
            _tally(miss_causes, cause_flags, stale)
            cause_flags = np.where(new_ctrl, 0, cause_flags)
            ctrl_missed += int(stale.sum())
            ctrl_miss_r += stale
            win_miss |= stale
            ctrl_age = np.where(new_ctrl, 0, ctrl_age)
            ctrl_sent += int(new_ctrl.sum())
            ctrl_total_r += new_ctrl
        vq = np.roll(vq, 1, axis=1)
        expired = vq[:, 0].copy()
        vq[:, 0] = video_rate * 1e6 * SLOT
        video_gen += float(vq[:, 0].sum())
        video_gen_r += vq[:, 0]
        video_req_bits += float(world.video_mbps.sum() * 1e6 * SLOT)

        # ---------------- PHY per robot-link -----------------------------------
        snr_now = true_snr(k, site)                               # (N,3)
        str_mode = second_link >= 0
        snr_eff = snr_now - 3.0 * str_mode[:, None]
        p_link = np.zeros((n, 3))
        rate = np.zeros((n, 3))
        for b in range(3):
            band = BANDS[b]
            p_link[:, b] = curves.success(band, mcs[b], snr_eff[:, b])
            rate[:, b] = rates[band][mcs[b]] * np.where(str_mode, 0.5, 1.0)
        doms = world.domain[site]                                  # (N,3)

        # CSMA collisions: per-attempt loss grows with stations in the domain.
        if collision_lambda > 0:
            act = np.bincount(doms[ar, bulk_link], minlength=n_dom)
            pc_dom = 1 - np.exp(-collision_lambda * np.maximum(act - 1, 0))
        else:
            pc_dom = np.zeros(n_dom)
        q_ctrl = None if policy.sched_control else 1 - pc_dom[doms[ar, ctrl_link]]
        up2 = dual_mask & (site2 >= 0) & (outage2_until <= k)
        if policy.radio2 == "shared":
            up2 &= ~scanning                       # same radio: a scan blocks both associations
        if not dual:
            up2 = np.zeros(n, dtype=bool)

        # control: attempts on ctrl link (+ second link in STR, + second radio)
        pend = (ctrl_age >= 0) & (up | up2)
        ctrl_down += int(((ctrl_age >= 0) & ~(up | up2)).sum())
        links, mcss = _mrr_chain(ctrl_link, mcs, ar)
        if policy.kind == "native" and policy.reflex:
            # Cross-link retry: second attempt on the other (bulk) link, so
            # one band's fade does not take out the retry chain.
            other = np.where(bulk_link != ctrl_link, bulk_link, (ctrl_link + 1) % 3)
            links[:, 1] = other
            mcss[:, 1] = 0
        ok1, air1 = _chain(curves, links, mcss, snr_eff, str_mode, q_ctrl)
        ok1 = np.where(up, ok1, 0.0)
        air1 = np.where(up, air1, 0.0)
        if policy.layered:
            # First attempt rides the base layer of a bulk frame (no own
            # access). Sionna's gain is relative to 16-QAM r=1/2 time
            # sharing, i.e. Wi-Fi MCS 3; failures fall back to normal retries.
            lay = snr_eff[ar, bulk_link] + policy.layered_gain_db
            pl = np.zeros(n)
            for b in range(3):
                sel = bulk_link == b
                if sel.any():
                    pl[sel] = curves.success(BANDS[b], np.full(sel.sum(), 3), lay[sel])
            ok1 = 1 - (1 - pl) * (1 - ok1)
            air1 = (1 - pl) * air1
        sl = np.maximum(second_link, 0)
        l2, m2 = _mrr_chain(sl, mcs, ar)
        ok2, air2 = _chain(curves, l2, m2, snr_eff, str_mode)
        ok2 = np.where(str_mode, ok2, 0.0)
        ok2 = np.where(up, ok2, 0.0)
        ok = 1 - (1 - ok1) * (1 - ok2)
        air_ctrl = np.zeros(n_dom)
        np.add.at(air_ctrl, doms[ar, ctrl_link], pend * air1)
        np.add.at(air_ctrl, doms[ar, sl], pend * str_mode * air2)
        if dual:
            s2 = np.maximum(site2, 0)
            snr2 = true_snr(k, s2) + r2_off
            d2 = world.domain[s2][ar, link2]
            l3, m3 = _mrr_chain(link2, mcs2, ar)
            q2 = None if policy.sched_control else 1 - pc_dom[d2]
            ok3, air3 = _chain(curves, l3, m3, snr2, np.full(n, policy.radio2 == "onboard"), q2)
            if policy.radio2 == "shared":
                # One radio: the second AP is tried only when the primary chain
                # failed, after retuning; the primary must also be up to retune from.
                air3 = (1 - ok) * (air3 + SHARED_SWITCH_S / SLOT)
            ok3 = np.where(up2, ok3, 0.0)
            ok = 1 - (1 - ok) * (1 - ok3)
            np.add.at(air_ctrl, d2, pend * up2 * air3)
        dup_slots += int((pend & str_mode).sum())
        scale_c = np.minimum(1.0, world.avail / np.maximum(air_ctrl, 1e-12))
        grant_c = np.ones(n) if policy.sched_control else scale_c[doms[ar, ctrl_link]]
        delivered = pend & (rng.random(n) < ok * grant_c)
        # Forensics: what this waiting packet is up against right now.
        waiting = ctrl_age >= 0
        kb_now = k // BLOCK_SLOTS
        blk_p = blocked[kb_now][ar, site]
        blk_2 = blocked[kb_now][ar, np.maximum(site2, 0)] | ~up2
        weak_p = snr_eff[ar, ctrl_link] < th99_0[ctrl_link]
        cont = (pc_dom[doms[ar, ctrl_link]] > 0.05) | (grant_c < 1)
        cause_flags |= waiting * (((~(up | up2)) * 1) | ((blk_p & blk_2) * 2) | (weak_p * 4) | (cont * 8))
        ctrl_fail += int((pend & ~delivered).sum())
        ctrl_congested += int((pend & (grant_c < 1)).sum())
        ctrl_age = np.where(delivered, -1, np.where(ctrl_age >= 0, ctrl_age + 1, -1))
        late = ctrl_age >= CONTROL_DEADLINE
        _tally(miss_causes, cause_flags, late)
        cause_flags = np.where(late | delivered, 0, cause_flags)
        ctrl_missed += int(late.sum())
        ctrl_miss_r += late
        win_miss |= late
        ctrl_age = np.where(late, -1, ctrl_age)
        if k % 100 == 99:
            windows += n
            clean_windows += int((~win_miss).sum())
            win_miss[:] = False
        gap_cur = np.where(delivered, 0, gap_cur + 1)
        gap_max = np.maximum(gap_max, gap_cur)
        stopped_slots += ((gap_cur > SAFETY_GAP_SLOTS) & (gap_cur > 0)).sum()
        resumed = delivered & (prev_gap > SAFETY_GAP_SLOTS)
        stopped_slots += resumed.sum() * int(RESTART_S / SLOT)
        stop_events += int(resumed.sum())
        prev_gap = gap_cur.copy()
        if policy.cadence_scan:
            go_scan = delivered & due
            scan_until = np.where(go_scan, k + 1 + int(round((SCAN_S if scan_duration_s is None else scan_duration_s) / SLOT)), scan_until)
            due &= ~go_scan
        if policy.cadence_roam:
            fire = delivered & (pending_roam >= 0) & (outage_until <= k)
            for i in np.nonzero(fire)[0]:
                site[i] = pending_roam[i]
                outage_until[i] = k + 1 + int(round(policy.roam_cost_s / SLOT))
                roams[i] += 1
                pending_roam[i] = -1
        left = world.avail - np.minimum(air_ctrl, world.avail)

        # video then bulk on the bulk link (the better link when in STR)
        bl = np.where(str_mode & (p_link[ar, np.maximum(second_link, 0)] * rate[ar, np.maximum(second_link, 0)]
                                  > p_link[ar, bulk_link] * rate[ar, bulk_link]),
                      np.maximum(second_link, 0), bulk_link)
        dv = doms[ar, bl]
        good = rate[ar, bl] * 1e6 * MAC_EFF * p_link[ar, bl] * SLOT * (1 - pc_dom[dv])
        vneed = np.where(up & (good > 0), vq.sum(1) / np.maximum(good, 1e-9), 0.0)
        need_v = np.bincount(dv, weights=vneed, minlength=n_dom)
        sv = np.minimum(1.0, left / np.maximum(need_v, 1e-12))
        vair = vneed * sv[dv]
        vbits = vair * good
        # drain video oldest first
        remaining = vbits.copy()
        for age in range(VIDEO_DEADLINE - 1, -1, -1):
            take = np.minimum(vq[:, age], remaining)
            vq[:, age] -= take
            remaining -= take
        video_ok += float(vbits.sum() - remaining.sum())
        video_ok_r += vbits - remaining
        used_v = np.bincount(dv, weights=vair, minlength=n_dom)
        left2 = np.maximum(0.0, left - used_v)
        backlog = np.minimum(backlog + world.bulk_mbps * 1e6 * SLOT, world.bulk_mbps * 1e6 * BULK_BUFFER_S)
        send = backlog * np.where(bulk_cap < 1.0, 0.0, 1.0)
        bneed = np.where(up & (good > 0), send / np.maximum(good, 1e-9), 0.0)
        need_b = np.bincount(dv, weights=bneed, minlength=n_dom)
        sb = np.minimum(1.0, left2 / np.maximum(need_b, 1e-12))
        bair = bneed * sb[dv]
        sent = np.minimum(bair * good, backlog)
        backlog -= sent
        bulk_bits_r += sent
        airtime_used += float(air_ctrl.sum() + used_v.sum() + bair.sum())
        act_acc += pend * air1 + vair + bair

    ctrl_frac_r = ctrl_miss_r / np.maximum(ctrl_total_r, 1)
    order = np.sort(ctrl_frac_r)
    worst = order[-max(1, n // 10):]
    vr = has_video
    return {
        "policy": policy.name, "scene": world.seed, "sigma_map": world.sigma_map,
        "blind_world": world.blind, "blockage_rate_per_min": blockage_rate,
        "duration_s": duration,
        "control_miss": ctrl_missed / max(ctrl_sent, 1),
        "control_miss_worst_decile": float(worst.mean()),
        "control_gap_p99_ms": float(np.percentile(gap_max, 99) * SLOT * 1000),
        "video_on_time": video_ok / max(video_gen, 1e-9),
        "video_on_time_worst_robot": float((video_ok_r[vr] / np.maximum(video_gen_r[vr], 1e-9)).min())
        if vr.any() else None,
        "bulk_mbps": float(bulk_bits_r.sum() / duration / 1e6),
        "airtime_domain_sum": airtime_used / n_slots,
        "ctrl_pending_slots_down": ctrl_down, "ctrl_failed_attempt_slots": ctrl_fail,
        "ctrl_congested_slots": ctrl_congested,
        "dup_fraction": dup_slots / max(n_slots * n, 1),
        "roams": int(roams.sum()),
        "control_misses": int(ctrl_missed), "control_sent": int(ctrl_sent),
        "availability_1s": clean_windows / max(windows, 1),
        "make_before_break_swaps": swaps, "collision_lambda": collision_lambda,
        "interference": interference, "mean_tx_reduction_db": float(-tx_adj.mean()),
        "safety_stop_fraction": stopped_slots / (n_slots * n), "safety_stops": stop_events,
        "video_delivered_vs_requested": video_ok / max(video_req_bits, 1e-9),
        "map_updates": map_updates, "miss_causes": miss_causes,
        "per_robot_control_miss": (ctrl_miss_r / np.maximum(ctrl_total_r, 1)).round(6).tolist(),
        "per_robot_video_on_time": (video_ok_r / np.maximum(video_gen_r, 1e-9)).round(6).tolist(),
        "focus_control_miss": float(ctrl_miss_r[focus].sum() / max(ctrl_total_r[focus].sum(), 1)) if focus.any() else None,
        "focus_video_on_time": float(video_ok_r[focus & has_video].sum() / max(video_gen_r[focus & has_video].sum(), 1e-9))
        if (focus & has_video).any() else None,
    }


def simulate_nr(world: World, policy: Policy, *, seed: int = 0, curves: LinkCurves | None = None,
                duration_s: float | None = None,
                blockage_rate: float = BLOCKAGE_RATE_PER_MIN) -> dict:
    """Private 5G baseline on the same truth, fading and blockage draws.

    One n78 carrier per site, uplink scheduled by the gNB: control gets its
    resources first and never contends; HARQ retries within the slot; CQI
    link adaptation on the current fading block (1 dB backoff); network A3
    handover (neighbour 3 dB better for two consecutive 100 ms blocks) with
    `policy.nr_ho_cost_s` interruption (0 = DAPS make-before-break)."""
    curves = curves or LinkCurves()
    rng = np.random.default_rng(1_000_003 * (world.seed + 1) + seed)
    n, s = world.n, world.sites.shape[0]
    duration = duration_s or world.duration_s
    n_slots = int(round(duration / SLOT))
    slots_per_pos = int(round(POS_DT / SLOT))
    ar = np.arange(n)
    fade_db, blocked, hid = hidden_draws(world, n_slots, blockage_rate)
    phase = hid.integers(0, 10_000, size=n)
    rates = curves.rates("5g") * NR_RATE_SCALE

    def snr_all(k_slot):
        kp = min(k_slot // slots_per_pos, world.snr.shape[0] - 1)
        kb = k_slot // BLOCK_SLOTS
        return (world.snr[kp][..., 1] + fade_db[kb][..., 1] - BLOCKAGE_DB * blocked[kb]
                + NR_SNR_OFFSET_DB)                                   # (N,S)

    serv = np.argmax(snr_all(0), axis=1)
    a3_count = np.zeros(n, dtype=int)
    outage_until = np.zeros(n, dtype=int)
    mcs = np.zeros(n, dtype=int)
    ho = 0
    ctrl_age = np.full(n, -1)
    ctrl_sent = ctrl_missed = 0
    ctrl_miss_r = np.zeros(n)
    ctrl_total_r = np.zeros(n)
    gap_cur = np.zeros(n)
    gap_max = np.zeros(n)
    win_miss = np.zeros(n, dtype=bool)
    windows = clean = 0
    cause_flags = np.zeros(n, dtype=int)
    miss_causes = {c: 0 for c in CAUSES}
    stopped = stop_events = 0
    prev_gap = np.zeros(n)
    vq = np.zeros((n, VIDEO_DEADLINE))
    video_gen = video_ok = 0.0
    backlog = np.zeros(n)
    bulk_bits = 0.0
    airtime = 0.0
    th = curves.threshold_db("5g", 0.9)
    for k in range(n_slots):
        if k % BLOCK_SLOTS == 0:
            cur = snr_all(k)
            mcs = np.clip(np.searchsorted(th, cur[ar, serv] - 1.0, side="right") - 1, 0, len(th) - 1)
            lag = snr_all(max(0, k - BLOCK_SLOTS))
            best = np.argmax(lag, axis=1)
            better = (best != serv) & (lag[ar, best] > lag[ar, serv] + NR_A3_DB)
            a3_count = np.where(better, a3_count + 1, 0)
            go = (a3_count >= 2) & (outage_until <= k)
            serv = np.where(go, best, serv)
            outage_until = np.where(go, k + int(round(policy.nr_ho_cost_s / SLOT)), outage_until)
            a3_count = np.where(go, 0, a3_count)
            ho += int(go.sum())
        up = outage_until <= k
        new = ((k + phase) % CONTROL_PERIOD) == 0
        if new.any():
            stale = new & (ctrl_age >= 0)
            _tally(miss_causes, cause_flags, stale)
            cause_flags = np.where(new, 0, cause_flags)
            ctrl_missed += int(stale.sum())
            ctrl_miss_r += stale
            win_miss |= stale
            ctrl_age = np.where(new, 0, ctrl_age)
            ctrl_sent += int(new.sum())
            ctrl_total_r += new
        vq = np.roll(vq, 1, axis=1)
        vq[:, 0] = world.video_mbps * 1e6 * SLOT
        video_gen += float(vq[:, 0].sum())

        x = snr_all(k)[ar, serv]
        p = [curves.success("5g", m, x) for m in (mcs, np.maximum(mcs - 2, 0), np.zeros_like(mcs))]
        ok = 1 - (1 - p[0]) * (1 - p[1]) * (1 - p[2])
        pend = (ctrl_age >= 0) & up
        delivered = pend & (rng.random(n) < ok)
        kb_now = k // BLOCK_SLOTS
        cause_flags |= (ctrl_age >= 0) * ((~up) * 1 | blocked[kb_now][ar, serv] * 2
                                          | (x < th[0]) * 4)
        ctrl_age = np.where(delivered, -1, np.where(ctrl_age >= 0, ctrl_age + 1, -1))
        late = ctrl_age >= CONTROL_DEADLINE
        _tally(miss_causes, cause_flags, late)
        cause_flags = np.where(late | delivered, 0, cause_flags)
        ctrl_missed += int(late.sum())
        ctrl_miss_r += late
        win_miss |= late
        ctrl_age = np.where(late, -1, ctrl_age)
        gap_cur = np.where(delivered, 0, gap_cur + 1)
        gap_max = np.maximum(gap_max, gap_cur)
        stopped += ((gap_cur > SAFETY_GAP_SLOTS)).sum()
        resumed = delivered & (prev_gap > SAFETY_GAP_SLOTS)
        stopped += resumed.sum() * int(RESTART_S / SLOT)
        stop_events += int(resumed.sum())
        prev_gap = gap_cur.copy()
        if k % 100 == 99:
            windows += n
            clean += int((~win_miss).sum())
            win_miss[:] = False
        reach = np.stack([np.ones(n), 1 - p[0], (1 - p[0]) * (1 - p[1])], 1)
        t = np.stack([(CONTROL_BITS / (rates[m] * 1e6) + 20e-6) / SLOT
                      for m in (mcs, np.maximum(mcs - 2, 0), np.zeros_like(mcs))], 1)
        air_c = np.bincount(serv, weights=pend * (reach * t).sum(1), minlength=s)
        left = np.maximum(0.0, 1.0 - air_c)
        good = rates[mcs] * 1e6 * NR_UL_EFF * p[0] * SLOT
        vneed = np.where(up & (good > 0), vq.sum(1) / np.maximum(good, 1e-9), 0.0)
        sv = np.minimum(1.0, left / np.maximum(np.bincount(serv, weights=vneed, minlength=s), 1e-12))
        vbits = vneed * sv[serv] * good
        remaining = vbits.copy()
        for age in range(VIDEO_DEADLINE - 1, -1, -1):
            take = np.minimum(vq[:, age], remaining)
            vq[:, age] -= take
            remaining -= take
        video_ok += float(vbits.sum() - remaining.sum())
        left2 = np.maximum(0.0, left - np.bincount(serv, weights=vneed * sv[serv], minlength=s))
        backlog = np.minimum(backlog + world.bulk_mbps * 1e6 * SLOT, world.bulk_mbps * 1e6 * BULK_BUFFER_S)
        bneed = np.where(up & (good > 0), backlog / np.maximum(good, 1e-9), 0.0)
        sb = np.minimum(1.0, left2 / np.maximum(np.bincount(serv, weights=bneed, minlength=s), 1e-12))
        sent = np.minimum(bneed * sb[serv] * good, backlog)
        backlog -= sent
        bulk_bits += float(sent.sum())
        airtime += float(air_c.sum() + (vneed * sv[serv]).sum() + (bneed * sb[serv]).sum())

    frac = ctrl_miss_r / np.maximum(ctrl_total_r, 1)
    worst = np.sort(frac)[-max(1, n // 10):]
    return {
        "policy": policy.name, "scene": world.seed, "sigma_map": world.sigma_map,
        "blind_world": world.blind, "blockage_rate_per_min": blockage_rate, "duration_s": duration,
        "control_miss": ctrl_missed / max(ctrl_sent, 1),
        "control_miss_worst_decile": float(worst.mean()),
        "control_gap_p99_ms": float(np.percentile(gap_max, 99) * SLOT * 1000),
        "video_on_time": video_ok / max(video_gen, 1e-9), "video_on_time_worst_robot": None,
        "bulk_mbps": bulk_bits / duration / 1e6, "airtime_domain_sum": airtime / n_slots,
        "ctrl_pending_slots_down": 0, "ctrl_failed_attempt_slots": 0, "ctrl_congested_slots": 0,
        "dup_fraction": 0.0, "roams": ho, "control_misses": int(ctrl_missed),
        "control_sent": int(ctrl_sent), "availability_1s": clean / max(windows, 1),
        "make_before_break_swaps": 0, "collision_lambda": 0.0,
        "interference": False, "mean_tx_reduction_db": 0.0,
        "safety_stop_fraction": stopped / (n_slots * n), "safety_stops": stop_events,
        "video_delivered_vs_requested": video_ok / max(video_gen, 1e-9),
        "miss_causes": miss_causes,
    }
