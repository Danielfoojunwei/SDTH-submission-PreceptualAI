# Managed-probe research kernel; service-world and prior evidence stay sealed.
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
import copy
import hashlib

import numpy as np

from ..native.phy import LinkCurves
from ..native.world import BANDS, CLIENT_TX_DBM, HORIZONS, NOISE_DBM, POS_DT, World

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


def _chain(curves, links, mcss, snr_eff, str_mode, q=None, link_available=None):
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
            if link_available is not None:p[sel,a] *= link_available[sel,b]
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


def _managed_chain_allocation(curves,chains,available):
    """Retry rounds pay their actual band/domain before earning PHY success.

    The predecessor collapses fallback airtime onto the primary domain. That
    shortcut cannot account for new band-specific probe reservations. Here
    each round shares only the remaining resource of its actual domain.
    STR copies compete in the same round; they retain independent PHY trials.
    """
    n=len(chains[0][0]);parts=[];remaining=np.array(available,dtype=float,copy=True)
    used=np.zeros_like(remaining);minimum_grant=np.ones(n);robot_airtime=np.zeros(n)
    for links,mcss,snr,str_mode,q,link_available,enabled,domains in chains:
        p=np.zeros_like(links,dtype=float);t=np.zeros_like(p);half=np.where(str_mode,.5,1.)
        for a in range(links.shape[1]):
            for b in range(3):
                sel=links[:,a]==b
                if not sel.any():continue
                mm=mcss[sel,a];p[sel,a]=curves.success(BANDS[b],mm,snr[sel,b])
                if link_available is not None:p[sel,a]*=link_available[sel,b]
                t[sel,a]=(CONTROL_BITS/(curves.rates(BANDS[b])[mm]*half[sel]*1e6)+FRAME_OVERHEAD_S)/SLOT
        if q is not None:p*=q[:,None]
        p*=enabled[:,None];t*=enabled[:,None]
        parts.append((p,t,np.take_along_axis(domains,links,axis=1),np.ones(n)))
    for a in range(3):
        requested=np.zeros_like(remaining);requests=[]
        for p,t,domains,failed in parts:
            demand=failed*t[:,a];requests.append(demand)
            np.add.at(requested,domains[:,a],demand)
        scale=np.minimum(1.,remaining/np.maximum(requested,1e-12));served=np.zeros_like(remaining)
        for (p,t,domains,failed),demand in zip(parts,requests):
            grant=scale[domains[:,a]]
            np.add.at(served,domains[:,a],demand*grant)
            robot_airtime+=demand*grant
            failed*=1.-p[:,a]*grant
            minimum_grant=np.minimum(minimum_grant,np.where(demand>0,grant,1.))
        used+=served;remaining=np.maximum(0.,remaining-served)
    return 1.-np.prod(np.stack([p[3] for p in parts]),axis=0),used,minimum_grant,robot_airtime


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


# Persistent slot state, explicitly restored rather than mutating CPython locals.
_STATE_NAMES = """ipen_site tx_adj act_acc site band_single outage_until ctrl_link
second_link bulk_link mcs meas bulk_cap backlog roams last_scan ctrl_age ctrl_sent
ctrl_missed ctrl_miss_r ctrl_total_r gap_cur gap_max vq video_gen video_ok
video_req_bits video_gen_r video_ok_r bulk_bits_r airtime_used dup_slots roam_log
ctrl_down ctrl_fail ctrl_congested win_miss windows clean_windows due scan_until
cause_flags miss_causes stopped_slots stop_events prev_gap video_rate site2
outage2_until mcs2 link2 swaps native_delta pending_roam pending_roam_expiry obs_buf map_updates
outcome_history cumulative_trace observation_history arrived
visible_pilot pilot_sample_slot pilot_site""".split()
_STATIC_WORLD_ARRAYS = """sites snr rssi_dl domain avail video_mbps bulk_mbps
noise_db channel plan pos tx_dl""".split()


def _copy_bundle(bundle):
    # Truth/CAD/routes are immutable inputs; mutable forecasts and callback
    # objects are copied together to preserve queue/gate/bank aliases.
    world = bundle['world']
    memo = {id(getattr(world, name)): getattr(world, name)
            for name in _STATIC_WORLD_ARRAYS if getattr(world, name, None) is not None}
    return copy.deepcopy(bundle, memo)


@dataclass(frozen=True)
class KernelCheckpoint:
    """Exact label-generation state. Predictors may receive only observable()."""
    slot: int
    config: dict
    bundle: dict
    public: dict
    observation_history: tuple

    def observable(self):
        return copy.deepcopy(self.public)


def _public_state(world, k, values, scan_observer, forecast_queue, managed_probes=None):
    n = world.n
    pilot = values['visible_pilot']
    associated = values['pilot_site'] == values['site']
    pilot_age = np.maximum(k-values['pilot_sample_slot'], 0)
    receiver_available = ((values['outage_until'] <= k)
                          & (values['scan_until'] <= k))
    if managed_probes is not None:
        receiver_available &= ~managed_probes.effect(k).blackout[values['site']].any(-1)
    pilot_valid = receiver_available & associated & (pilot_age <= BLOCK_SLOTS)
    local = (None if scan_observer is None else
             np.array(scan_observer.local, copy=True).reshape(n, -1, 3))
    if local is not None:
        # A prior AP's last pilot must not become a measurement on a newly
        # associated AP while its actual receiver remains unavailable.
        ids = np.flatnonzero(associated)
        local[ids, values['site'][ids]] = pilot[ids]+world.noise_db+world.tx_dl-CLIENT_TX_DBM
    index = (min(k//100, len(world.forecast)-1) if forecast_queue is None else forecast_queue.last)
    forecast = None
    if index >= 0:
        from ..frontier.timing import ForecastQueue
        age = ForecastQueue([0.]); age.age = k/100-index
        forecast = age.advance(world.forecast[index])
    return {'slot': int(k), 'current': values['site'].copy(),
            'forecast': forecast, 'forecast_source': index,
            **({'forecast_std': world.sigma_belief[index].copy()}
               if forecast is not None and world.sigma_belief is not None else {}),
            'forecast_age_seconds': None if index < 0 else k/100-index,
            'pilot_snr': pilot.copy(),
            'pilot_valid': np.broadcast_to(pilot_valid[:, None], (n, 3)).copy(),
            'pilot_age_slots': pilot_age.copy(), 'pilot_site': values['pilot_site'].copy(),
            'local_rssi': local,
            'local_heard': None if local is None else local >= -90.,
            'positions': world.pos[min(k//int(round(POS_DT/SLOT)), len(world.pos)-1)].copy(),
            'plan': world.plan[min(k//100, len(world.plan)-1)].copy(),
            'load': np.bincount(values['site'], minlength=len(world.sites)),
            'pending_packet_age': values['ctrl_age'].copy(),
            'gap': values['gap_cur'].copy(), 'max_gap': values['gap_max'].copy(),
            'misses': values['ctrl_miss_r'].copy(), 'sent': values['ctrl_total_r'].copy(),
            'roams': values['roams'].copy(),
            'pending_target': values['pending_roam'].copy(),
            'pending_valid_until': values['pending_roam_expiry'].copy(),
            'outage_remaining': np.maximum(values['outage_until']-k, 0),
            'scan_remaining': np.maximum(values['scan_until']-k, 0),
            'control_link': values['ctrl_link'].copy(),
            'bulk_link': values['bulk_link'].copy(),
            'second_link': values['second_link'].copy(),
            'mcs': np.stack([values['mcs'][b] for b in range(3)], -1),
            'tx_power_adjust_db': values['tx_adj'].copy(),
            'video_queue_bits': values['vq'].sum(1).copy(),
            'bulk_queue_bits': values['backlog'].copy(),
            'outcome_history': values['outcome_history'][(np.arange(100)+k) % 100].copy(),
            'outcome_slot': np.where(np.arange(k-100, k) >= 0,
                                     np.arange(k-100, k), -1),
            'outcome_valid': np.arange(k-100, k) >= 0,
            'observed_history_slots': min(k, 100)}


def simulate(world: World, policy: Policy, *, seed: int = 0, curves: LinkCurves | None = None,
             duration_s: float | None = None, blockage_rate: float = BLOCKAGE_RATE_PER_MIN,
             collision_lambda: float = 0.0, interference: bool = False,
             roam_gate=None, decision_trace=None, forecast_queue=None, planner="native", scan_duration_s=None, handover_admission=None,
             scan_observer=None, forecast_source=None, roam_scorer=None,
             checkpoint_at=None, checkpoint_sink=None, resume=None, stop_slot=None,
             force_action=None, observation_hook=None, actual_action_hook=None,
             external_primary=None, detailed=False, focal_hold_slots=None,
             observation_period_slots=None, native_proposal_ttl_slots=None,
             managed_probes=None) -> dict:
    """Execute native packets, optionally branching one bounded intervention.

    ``focal_hold_slots=100`` suppresses further primary moves for one second
    after a forced decision, then restores the same canonical recovery policy.
    A no-force suffix is the factual baseline, not a forced ten-second stay.
    ``None`` retains the predecessor's until-stop focal lock for reproduction.

    Observation period defaults to 100 slots (one second), or the checkpoint's
    captured period on resume. At period 10 the hook receives 100-ms snapshots
    of legitimate local measurements and past packet outcomes. Forecast issues
    and ``external_primary.observe`` keep their original one-second cadence.
    Future forecasts remain unavailable until their completion queue permits
    them. Explicit receiver masks/ages distinguish missing from fresh pilots.

    ``native_proposal_ttl_slots=100`` gives every native primary proposal a
    one-second lifetime, including the internal canonical planner. An explicit
    earlier proposal deadline still applies. ``None`` preserves predecessor
    behavior. This command-freshness baseline is not an architectural novelty.

    Any managed runtime selects one common per-retry-domain control allocator
    for EVERY slot, including canonical and zero-duty controls. Only an absent
    runtime reproduces the predecessor's collapsed-domain allocator exactly.
    """
    if resume is not None:
        if world.seed != resume.config['scene'] or world.n != resume.config['robots']:
            raise ValueError("Checkpoint scene or fleet differs")
        # A checkpoint owns its physical scenario. Use it before deriving noise,
        # contention geometry or array sizes; a caller's similarly named world
        # must never silently alter the resumed trajectory.
        world = resume.bundle['world']
        saved_probes=resume.bundle.get('managed_probes')
        if managed_probes is not None and (saved_probes is None or managed_probes.spec.digest()!=saved_probes.spec.digest()):
            raise ValueError('Checkpoint managed probe configuration differs')
        managed_probes=saved_probes
        if scan_duration_s is None:
            scan_duration_s = resume.config['scan_duration_s']
        if observation_period_slots is None:
            observation_period_slots = resume.config['observation_period_slots']
        if focal_hold_slots is None:
            focal_hold_slots = resume.config['focal_hold_slots']
        if native_proposal_ttl_slots is None:
            native_proposal_ttl_slots = resume.config['native_proposal_ttl_slots']
    if observation_period_slots is None:
        observation_period_slots = 100
    if (isinstance(observation_period_slots, (bool, np.bool_))
        or not isinstance(observation_period_slots, (int, np.integer))
        or observation_period_slots < 1):
        raise ValueError("Use a positive integer observation period in slots")
    observation_period_slots = int(observation_period_slots)
    if focal_hold_slots is not None:
        if (isinstance(focal_hold_slots, (bool, np.bool_))
            or not isinstance(focal_hold_slots, (int, np.integer)) or focal_hold_slots < 1):
            raise ValueError("Use a positive integer focal hold or None")
        focal_hold_slots = int(focal_hold_slots)
    if native_proposal_ttl_slots is not None:
        if (isinstance(native_proposal_ttl_slots, (bool, np.bool_))
            or not isinstance(native_proposal_ttl_slots, (int, np.integer))
            or native_proposal_ttl_slots < 1):
            raise ValueError("Use a positive integer native proposal TTL or None")
        native_proposal_ttl_slots = int(native_proposal_ttl_slots)
    if (resume is not None or force_action is not None or checkpoint_at is not None) and policy.kind != "native":
        raise ValueError("State forks support native warehouse control only")
    if force_action is not None and resume is None:
        raise ValueError("A forced action requires a factual checkpoint")
    if managed_probes is not None and (policy.kind!='native' or policy.dual_radio or policy.layered):
        raise ValueError('Managed probes support native primary-device MLO only; dual-radio/layered modes are outside this profile')
    if policy.kind == "nr":
        return simulate_nr(world, policy, seed=seed, curves=curves, duration_s=duration_s,
                           blockage_rate=blockage_rate)
    if scan_observer is not None and not policy.cadence_scan:
        raise ValueError("Delivered telemetry requires cadence-triggered scans")
    if forecast_source is not None and forecast_queue is not forecast_source:
        raise ValueError("Use the live forecast source as its completion queue")
    curves = curves or LinkCurves()
    rng = np.random.default_rng(1_000_003 * (world.seed + 1) + seed)   # same draws for all policies
    n, s = world.n, world.sites.shape[0]
    duration = duration_s or world.duration_s
    n_slots = int(round(duration / SLOT))
    final_slot = n_slots if stop_slot is None else int(stop_slot)
    if final_slot < 0 or final_slot > n_slots:
        raise ValueError("stop_slot outside original full scenario")
    checkpoints = set() if checkpoint_at is None else ({int(checkpoint_at)} if np.isscalar(checkpoint_at) else set(map(int, checkpoint_at)))
    if checkpoints and checkpoint_sink is None:
        raise ValueError("A checkpoint_sink is required")
    fork_config = {'seed': seed, 'scene': world.seed, 'robots': world.n,
                   'duration_slots': n_slots, 'blockage_rate': blockage_rate,
                   'collision_lambda': collision_lambda, 'interference': interference,
                   'scan_duration_s': scan_duration_s, 'planner': planner, 'policy': policy,
                   'observation_period_slots': observation_period_slots,
                   'focal_hold_slots': focal_hold_slots,
                   'native_proposal_ttl_slots': native_proposal_ttl_slots,
                   'curves_sha256': hashlib.sha256(b''.join(np.asarray(a).tobytes()
                        for band in BANDS for a in curves.tables[band])).hexdigest()}
    if managed_probes is not None:
        if managed_probes._world_seed!=world.seed:raise ValueError('Managed runtime belongs to a different native world')
        fork_config['managed_probe_spec_sha256']=managed_probes.spec.digest()
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
        if interference or managed_probes is not None:
            total=ipen_site[sites_idx] if interference else np.zeros_like(x)
            if managed_probes is not None:total=total+managed_probes.interference(k_slot)[sites_idx]
            x = x - 10 * np.log10(1 + total / noise_mw[None, :])
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
    visible_pilot = meas.copy()
    pilot_sample_slot = np.zeros(n, dtype=int)
    pilot_site = site.copy()
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
    pending_roam_expiry = np.full(n, n_slots+1, dtype=int)
    locked_robot = None if force_action is None else int(force_action[0])
    if locked_robot is not None and not 0 <= locked_robot < n:
        raise ValueError("Invalid focal robot")
    locked_until = n_slots+1
    forcing = False

    def move_primary(i, target, cost_s, valid_until=None):
        nonlocal swaps
        if locked_robot == i and not forcing:
            return
        if policy.kind == "native" and native_proposal_ttl_slots is not None:
            ttl_expiry = k + native_proposal_ttl_slots
            valid_until = ttl_expiry if valid_until is None else min(ttl_expiry, int(valid_until))
        dual = bool(dual_mask[i])
        if policy.cadence_roam and cost_s < CONTROL_PERIOD * SLOT and not (dual and site2[i] == target):
            pending_roam[i] = target       # executed right after the next control delivery
            pending_roam_expiry[i] = n_slots+1 if valid_until is None else int(valid_until)
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
        if actual_action_hook is not None:
            actual_action_hook(k, np.array([i]), np.array([target]))
        if external_primary is not None and actual_action_hook != external_primary.executed:
            external_primary.executed(k, np.array([i]), np.array([target]))

    cochan = [(world.channel[:, b][:, None] == world.channel[:, b][None, :])
              & (world.domain[:, b][:, None] != world.domain[:, b][None, :]) for b in range(3)]

    online = policy.kind == "native" and policy.online_map and world.rmap is not None
    if online:
        world = copy.copy(world)
        world.rmap = copy.deepcopy(world.rmap)       # each run learns from its own operation
    obs_buf = {}
    map_updates = 0
    outcome_history = np.zeros((100, n, 2), dtype=bool)
    cumulative_trace = []
    observation_history = []
    arrived = None
    if online and (resume is not None or checkpoints):
        raise ValueError("Online rmap fitting is outside state-fork profile")
    if resume is not None:
        expected_config = resume.config
        if force_action is not None:
            # This parameter describes the new intervention, not different
            # physical dynamics or a changed factual observation prefix.
            expected_config = dict(expected_config, focal_hold_slots=focal_hold_slots)
        if expected_config != fork_config:
            raise ValueError("Checkpoint physics, policy or scenario differs")
        if not 0 <= resume.slot < final_slot:
            raise ValueError("Checkpoint must precede stop_slot")
        restored = _copy_bundle(resume.bundle)
        world = restored['world']
        managed_probes=restored.get('managed_probes')
        rng.bit_generator.state = restored['rng']
        (roam_gate, forecast_queue, scan_observer, forecast_source,
         roam_scorer, handover_admission, decision_trace, observation_hook, actual_action_hook, external_primary) = restored['hooks']
        (ipen_site, tx_adj, act_acc, site, band_single, outage_until,
         ctrl_link, second_link, bulk_link, mcs, meas, bulk_cap,
         backlog, roams, last_scan, ctrl_age, ctrl_sent, ctrl_missed,
         ctrl_miss_r, ctrl_total_r, gap_cur, gap_max, vq, video_gen,
         video_ok, video_req_bits, video_gen_r, video_ok_r, bulk_bits_r, airtime_used,
         dup_slots, roam_log, ctrl_down, ctrl_fail, ctrl_congested, win_miss,
         windows, clean_windows, due, scan_until, cause_flags, miss_causes,
         stopped_slots, stop_events, prev_gap, video_rate, site2, outage2_until,
         mcs2, link2, swaps, native_delta, pending_roam, pending_roam_expiry, obs_buf,
         map_updates, outcome_history, cumulative_trace, observation_history, arrived,
         visible_pilot, pilot_sample_slot, pilot_site) = (restored['state'][name] for name in _STATE_NAMES)
        k0 = resume.slot
        if force_action is None:
            locked_robot = restored.get("locked_robot")
            locked_until = restored.get("locked_until", n_slots+1)
        else:
            locked_until = n_slots+1 if focal_hold_slots is None else k0+focal_hold_slots

    window_max_gap = gap_cur.copy()
    for k in range(k0, final_slot):
        if locked_robot is not None and k >= locked_until:
            locked_robot = None
        slot_missed = np.zeros(n, dtype=bool)
        if resume is None or k != k0:
            # Advance report transport and publish only completed receipts BEFORE
            # the live predictor issues its current forecast. Capture below uses
            # the newly updated native fleet-interference background.
            if managed_probes is not None:managed_probes.advance(k)
            expired = (pending_roam >= 0) & (pending_roam_expiry <= k)
            pending_roam[expired] = -1
            pending_roam_expiry[expired] = n_slots+1
            if scan_observer is not None:
                scan_observer.complete(k)
            if scan_observer is not None and k % slots_per_dec == 0:
                scan_observer.pose(k // slots_per_dec, world.pos[min(k // slots_per_pos, len(world.pos)-1)])
            if forecast_source is not None and k % slots_per_dec == 0:
                d = k // slots_per_dec
                mean, sd = forecast_source.issue(d, world.plan[d])
                world.forecast[d] = mean
                world.sigma_belief[d] = sd[:, 0]
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
            if managed_probes is not None:
                ambient=np.broadcast_to(noise_mw,(s,3))+(ipen_site if interference else 0.)
                managed_probes.capture(k,ambient)
            # ---------------- measurements (lagged by one fading block) --------
            if k % BLOCK_SLOTS == 0:
                observed = true_snr(max(0, k - BLOCK_SLOTS), site)
                receiver_available = (outage_until <= k) & (scan_until <= k)
                if managed_probes is not None:receiver_available &= ~managed_probes.effect(k).blackout[site].any(-1)
                visible_pilot = np.where(receiver_available[:, None], observed, visible_pilot)
                pilot_sample_slot = np.where(receiver_available, max(0, k-BLOCK_SLOTS), pilot_sample_slot)
                pilot_site = np.where(receiver_available, site, pilot_site)
                if scan_observer is None and observation_period_slots == 100:
                    meas = observed
                else:
                    available = (outage_until <= k) & (scan_until <= k)
                    if managed_probes is not None:available &= ~managed_probes.effect(k).blackout[site].any(-1)
                    meas = np.where(available[:, None], observed, meas)
                for b in range(3):
                    band = BANDS[b]
                    m = meas[:, b] - (3.0 if policy.kind != "legacy" else 0.0) * (second_link >= 0)
                    mcs[b] = _mcs_for(curves, band, m - 2.0)          # Minstrel-like backoff

        if resume is None or k != k0:
            arrived = None if forecast_queue is None else forecast_queue.take(k)
        sample_observation = k % observation_period_slots == 0
        external_observation = k % 100 == 0 and external_primary is not None
        if (k in checkpoints or external_observation
            or (sample_observation and (observation_hook is not None or detailed or checkpoints))):
            scope = locals()
            values = {name: scope[name] for name in _STATE_NAMES}
            public = _public_state(world, k, values, scan_observer, forecast_queue,managed_probes)
            fresh_observation = sample_observation and (not observation_history or observation_history[-1]['slot'] != k)
            if fresh_observation:
                observation_history.append(copy.deepcopy(public))
                del observation_history[:-max(6, math.ceil(600/observation_period_slots))]
                if observation_hook is not None:
                    observation_hook(public)
            if external_observation and (resume is None or k != k0):
                external_primary.observe(public)
            if k in checkpoints:
                # Observer callbacks may have updated their own linked state;
                # capture after every callback belonging to this slot.
                scope = locals()
                values = {name: scope[name] for name in _STATE_NAMES}
                raw = {'world': world, 'state': values, 'rng': rng.bit_generator.state,
                       'managed_probes':managed_probes,
                       'locked_robot': locked_robot, 'locked_until': locked_until,
                       'hooks': (roam_gate, forecast_queue, scan_observer, forecast_source,
                                 roam_scorer, handover_admission, decision_trace,
                                 observation_hook, actual_action_hook, external_primary)}
                checkpoint_sink(KernelCheckpoint(k, copy.deepcopy(fork_config), _copy_bundle(raw),
                                                 public, tuple(copy.deepcopy(observation_history))))
        if force_action is not None and k == k0:
            target = int(force_action[1])
            if not 0 <= target < s:
                raise ValueError("Invalid target AP")
            pending_roam[locked_robot] = -1
            pending_roam_expiry[locked_robot] = n_slots+1
            if target != site[locked_robot]:
                if outage_until[locked_robot] > k or scan_until[locked_robot] > k:
                    raise ValueError("A new handover requires an available focal radio")
                forcing = True
                move_primary(locked_robot, target, policy.roam_cost_s, valid_until=k0+100)
                forcing = False

        if external_primary is not None:
            proposal = external_primary.take(k)
            if proposal is not None:
                old = np.asarray(proposal['current'])
                target = np.asarray(proposal['target'])
                eligible = np.asarray(proposal['eligible'])
                valid_until = np.asarray(proposal.get('valid_until', np.full(n, n_slots+1, dtype=int)))
                if valid_until.ndim == 0:
                    valid_until = np.broadcast_to(valid_until, (n,))
                if (old.shape != (n,) or target.shape != (n,) or eligible.shape != (n,)
                    or eligible.dtype != np.bool_ or not np.issubdtype(target.dtype, np.integer)
                    or valid_until.shape != (n,) or not np.issubdtype(valid_until.dtype, np.integer)
                    or (target < 0).any() or (target >= s).any()):
                    raise ValueError("Invalid external primary proposal")
                go_external = eligible & (old == site) & (target != site) & (outage_until <= k) & (scan_until <= k) & (k < valid_until)
                budget_external = {}
                for i in np.flatnonzero(go_external):
                    if i == locked_robot:
                        continue
                    ap = int(target[i])
                    if policy.stagger and budget_external.get(ap, 0) >= 2:
                        continue
                    budget_external[ap] = budget_external.get(ap, 0)+1
                    move_primary(int(i), ap, policy.roam_cost_s, valid_until=int(valid_until[i]))

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
            if policy.planned_roam and external_primary is None:
                # Lookahead scales with switching cost: with ~30 ms pre-keyed
                # roams only the next few seconds matter.
                w = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=float)
                score = (fc.max(axis=3) * w[None, :, None]).sum(1) / w.sum()   # (N,S)
                if planner == "reliability":
                    from ..frontier.timing import reliability_scores
                    score = reliability_scores(fc)
                elif planner != "native":
                    raise ValueError("Unknown native planner")
                load = np.bincount(site, minlength=s).astype(float)
                score = score - policy.load_penalty * load[None, :]
                minimum_gain = policy.roam_gain_db
                if roam_scorer is not None:
                    score = np.asarray(roam_scorer(fc, site), dtype=float)
                    if score.shape != (n, s) or not np.isfinite(score).all():
                        raise ValueError("Invalid observable roam scores")
                    minimum_gain = 1e-7  # cost-aware scores are seconds of predicted loss
                best = np.argmax(score, axis=1)
                gain = score[ar, best] - score[ar, site]
                go = (gain >= minimum_gain) & (best != site) & (outage_until <= k)
                decision = None
                if roam_gate is not None or decision_trace is not None or handover_admission is not None:
                    from ..native.gate import RoamDecision
                    decision = RoamDecision.capture(d,site,best,gain,go,score)
                confident = np.ones(n,dtype=bool)
                if policy.gate:
                    sb = sigma_now * policy.belief_scale                     # (N,S,3)
                    sbest = sb.max(axis=2)
                    z = gain / np.sqrt(sbest[ar, site] ** 2 + sbest[ar, best] ** 2 + 1e-9)
                    confident = z >= policy.gate_z
                    if roam_gate is not None:
                        from ..native.gate import checked_mask
                        confident = checked_mask(roam_gate(decision),n)
                    go = go & confident
                    if handover_admission is not None:
                        from ..native.gate import checked_mask
                        go = go & checked_mask(handover_admission(decision), n)
                    # Not confident: the reactive rule decides for this robot.
                    r = (rssi_dl(max(0, k - BLOCK_SLOTS)) if scan_observer is None else
                         scan_observer.neighbors(site, meas, world.noise_db, world.tx_dl))
                    cur = r[ar, site].max(axis=1)
                    rbest = np.argmax(r.max(axis=2), axis=1)
                    rgain = r[ar, rbest].max(axis=1) - cur
                    rgo = (~confident) & (cur < -75.0) & (rgain >= 8.0) & (rbest != site) & (outage_until <= k)
                    if handover_admission is not None:
                        from ..native.gate import RoamDecision, checked_mask
                        reactive_decision = RoamDecision.capture(d, site, rbest, rgain, rgo, score)
                        rgo = rgo & checked_mask(handover_admission(reactive_decision), n)
                    for i in np.nonzero(rgo)[0]:
                        # Still our client: pre-keyed and cadence-timed like a planned roam.
                        move_primary(i, rbest[i], policy.roam_cost_s)
                budget = {}
                for i in np.nonzero(go)[0]:
                    if i == locked_robot:
                        continue
                    if policy.stagger and budget.get(best[i], 0) >= 2:
                        continue
                    budget[best[i]] = budget.get(best[i], 0) + 1
                    move_primary(i, best[i], policy.roam_cost_s)
                if decision_trace is not None:
                    decision_trace(decision,confident.copy(),site.copy())
            elif external_primary is None:
                r = rssi_dl(max(0, k - BLOCK_SLOTS))
                cur = r[ar, site].max(axis=1)
                best_site = np.argmax(r.max(axis=2), axis=1)
                gain = r[ar, best_site].max(axis=1) - cur
                go = (cur < -75.0) & (gain >= 8.0) & (outage_until <= k)
                for i in np.nonzero(go & (best_site != site))[0]:
                    if i == locked_robot:
                        continue
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
            slot_missed |= stale
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
        managed_effect=None if managed_probes is None else managed_probes.effect(k)
        link_available=None if managed_effect is None else ~managed_effect.blackout[site]
        slot_avail=world.avail if managed_effect is None else np.maximum(0.,world.avail-managed_effect.reserved_airtime)
        p_link = np.zeros((n, 3))
        rate = np.zeros((n, 3))
        for b in range(3):
            band = BANDS[b]
            p_link[:, b] = curves.success(band, mcs[b], snr_eff[:, b])
            if link_available is not None:p_link[:,b] *= link_available[:,b]
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
        ok1, air1 = _chain(curves, links, mcss, snr_eff, str_mode, q_ctrl,link_available)
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
                    if link_available is not None:pl[sel] *= link_available[sel,b]
            ok1 = 1 - (1 - pl) * (1 - ok1)
            air1 = (1 - pl) * air1
        sl = np.maximum(second_link, 0)
        l2, m2 = _mrr_chain(sl, mcs, ar)
        ok2, air2 = _chain(curves, l2, m2, snr_eff, str_mode,link_available=link_available)
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
            secondary_available=None if managed_effect is None else ~managed_effect.blackout[s2]
            ok3, air3 = _chain(curves, l3, m3, snr2, np.full(n, policy.radio2 == "onboard"), q2,secondary_available)
            if policy.radio2 == "shared":
                # One radio: the second AP is tried only when the primary chain
                # failed, after retuning; the primary must also be up to retune from.
                air3 = (1 - ok) * (air3 + SHARED_SWITCH_S / SLOT)
            ok3 = np.where(up2, ok3, 0.0)
            ok = 1 - (1 - ok) * (1 - ok3)
            np.add.at(air_ctrl, d2, pend * up2 * air3)
        dup_slots += int((pend & str_mode).sum())
        managed_control=managed_probes is not None
        if managed_control:
            chains=[(links,mcss,snr_eff,str_mode,q_ctrl,link_available,pend&up,doms)]
            if str_mode.any():chains.append((l2,m2,snr_eff,str_mode,None,link_available,pend&up&str_mode,doms))
            ok,air_ctrl,grant_c,managed_control_airtime=_managed_chain_allocation(curves,chains,slot_avail)
            managed_probes._control_allocator_slots+=1
            delivered=pend & (rng.random(n)<ok)
        else:
            scale_c = np.minimum(1.0, slot_avail / np.maximum(air_ctrl, 1e-12))
            grant_c = np.ones(n) if policy.sched_control else scale_c[doms[ar, ctrl_link]]
            delivered = pend & (rng.random(n) < ok * grant_c)
        if scan_observer is not None:
            scan_observer.deliver(k, delivered)
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
        slot_missed |= late
        outcome_history[k % 100, :, 0] = delivered
        outcome_history[k % 100, :, 1] = slot_missed
        win_miss |= late
        ctrl_age = np.where(late, -1, ctrl_age)
        if k % 100 == 99:
            windows += n
            clean_windows += int((~win_miss).sum())
            win_miss[:] = False
        gap_cur = np.where(delivered, 0, gap_cur + 1)
        gap_max = np.maximum(gap_max, gap_cur)
        window_max_gap = np.maximum(window_max_gap, gap_cur)
        stopped_slots += ((gap_cur > SAFETY_GAP_SLOTS) & (gap_cur > 0)).sum()
        resumed = delivered & (prev_gap > SAFETY_GAP_SLOTS)
        stopped_slots += resumed.sum() * int(RESTART_S / SLOT)
        stop_events += int(resumed.sum())
        prev_gap = gap_cur.copy()
        if policy.cadence_scan:
            go_scan = delivered & due
            scan_until = np.where(go_scan, k + 1 + int(round((SCAN_S if scan_duration_s is None else scan_duration_s) / SLOT)), scan_until)
            if scan_observer is not None and go_scan.any():
                ids = np.flatnonzero(go_scan)
                kp = min(k // slots_per_pos, len(world.pos)-1)
                scan_observer.scan(k, ids, world.pos[kp, ids], rssi_dl(k)[ids], scan_until[ids])
            due &= ~go_scan
        if policy.cadence_roam:
            fire = delivered & (pending_roam >= 0) & (outage_until <= k) & (k < pending_roam_expiry)
            for i in np.nonzero(fire)[0]:
                site[i] = pending_roam[i]
                outage_until[i] = k + 1 + int(round(policy.roam_cost_s / SLOT))
                roams[i] += 1
                pending_roam[i] = -1
                pending_roam_expiry[i] = n_slots+1
                if actual_action_hook is not None:
                    actual_action_hook(k, np.array([i]), np.array([int(site[i])]))
                if external_primary is not None and actual_action_hook != external_primary.executed:
                    external_primary.executed(k, np.array([i]), np.array([int(site[i])]))
        left = slot_avail - np.minimum(air_ctrl, slot_avail)

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
        if managed_probes is not None:
            incoming=world.bulk_mbps*1e6*SLOT
            managed_probes._bulk_generated+=float(incoming.sum())
            managed_probes._bulk_overflow+=float(np.maximum(backlog+incoming-world.bulk_mbps*1e6*BULK_BUFFER_S,0.).sum())
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
        if managed_effect is not None:airtime_used+=float(managed_effect.reserved_airtime.sum())
        if managed_control:act_acc+=managed_control_airtime+vair+bair
        else:act_acc += pend * air1 + vair + bair
        if detailed and k % 100 == 99:
            cumulative_trace.append({'slot_end': k+1, 'misses': ctrl_miss_r.astype(int).tolist(),
                                     'sent': ctrl_total_r.astype(int).tolist(),
                                     'gap': gap_cur.astype(int).tolist(),
                                     'max_gap': gap_max.astype(int).tolist()})

    ctrl_frac_r = ctrl_miss_r / np.maximum(ctrl_total_r, 1)
    order = np.sort(ctrl_frac_r)
    worst = order[-max(1, n // 10):]
    vr = has_video
    result_duration = duration if stop_slot is None else final_slot*SLOT
    result_slots = n_slots if stop_slot is None else final_slot
    result = {
        "policy": policy.name, "scene": world.seed, "sigma_map": world.sigma_map,
        "blind_world": world.blind, "blockage_rate_per_min": blockage_rate,
        "duration_s": result_duration,
        "control_miss": ctrl_missed / max(ctrl_sent, 1),
        "control_miss_worst_decile": float(worst.mean()),
        "control_gap_p99_ms": float(np.percentile(gap_max, 99) * SLOT * 1000),
        "video_on_time": video_ok / max(video_gen, 1e-9),
        "video_on_time_worst_robot": float((video_ok_r[vr] / np.maximum(video_gen_r[vr], 1e-9)).min())
        if vr.any() else None,
        "bulk_mbps": float(bulk_bits_r.sum() / max(result_duration, 1e-9) / 1e6),
        "airtime_domain_sum": airtime_used / max(result_slots, 1),
        "ctrl_pending_slots_down": ctrl_down, "ctrl_failed_attempt_slots": ctrl_fail,
        "ctrl_congested_slots": ctrl_congested,
        "dup_fraction": dup_slots / max(result_slots * n, 1),
        "roams": int(roams.sum()),
        "control_misses": int(ctrl_missed), "control_sent": int(ctrl_sent),
        "availability_1s": clean_windows / max(windows, 1),
        "make_before_break_swaps": swaps, "collision_lambda": collision_lambda,
        "interference": interference, "mean_tx_reduction_db": float(-tx_adj.mean()),
        "safety_stop_fraction": stopped_slots / max(result_slots * n, 1), "safety_stops": stop_events,
        "video_delivered_vs_requested": video_ok / max(video_req_bits, 1e-9),
        "map_updates": map_updates, "miss_causes": miss_causes,
        "per_robot_control_miss": (ctrl_miss_r / np.maximum(ctrl_total_r, 1)).round(6).tolist(),
        "per_robot_video_on_time": (video_ok_r / np.maximum(video_gen_r, 1e-9)).round(6).tolist(),
        "focus_control_miss": float(ctrl_miss_r[focus].sum() / max(ctrl_total_r[focus].sum(), 1)) if focus.any() else None,
        "focus_video_on_time": float(video_ok_r[focus & has_video].sum() / max(video_gen_r[focus & has_video].sum(), 1e-9))
        if (focus & has_video).any() else None,
    }
    if detailed:
        observer = getattr(observation_hook, '__self__', observation_hook)
        observer_rows = list(getattr(observer, 'observations', []))
        result['packet_detail'] = {'stop_slot': final_slot, 'resume_slot': k0,
            'misses': ctrl_miss_r.astype(int).tolist(), 'sent': ctrl_total_r.astype(int).tolist(),
            'gap': gap_cur.astype(int).tolist(), 'max_gap': gap_max.astype(int).tolist(),
            'window_max_gap': window_max_gap.astype(int).tolist(),
            'controller_frames': ([{'slot': int(row['slot']), 'frame': np.asarray(row['frame']).tolist()}
                                   for row in external_primary.observations[-6:]]
                                  if external_primary is not None and hasattr(external_primary, 'observations') else []),
            'controller_summary': (external_primary.summary()
                                   if external_primary is not None and hasattr(external_primary, 'summary') else None),
            'cumulative': cumulative_trace,
            'delivered_report_events': (None if scan_observer is None else copy.deepcopy(scan_observer.events))}
        if observation_period_slots != 100 or focal_hold_slots is not None or native_proposal_ttl_slots is not None:
            result['packet_detail'].update(
                observation_frames=[{'slot': int(row['slot']), 'frame': np.asarray(row['frame']).tolist()}
                                    for row in observer_rows[-max(6, math.ceil(600/observation_period_slots)):]
                                    if isinstance(row, dict) and 'slot' in row and 'frame' in row],
                observation_period_slots=observation_period_slots,
                focal_hold_slots=focal_hold_slots,
                native_proposal_ttl_slots=native_proposal_ttl_slots)
    if managed_probes is not None:
        result['managed_probe_summary']=managed_probes.summary()
        result['managed_packet_accounting']={
            'video_requested_bits':float(video_req_bits),'video_generated_bits':float(video_gen),
            'video_delivered_bits':float(video_ok),'bulk_generated_bits':managed_probes._bulk_generated,
            'bulk_delivered_bits':float(bulk_bits_r.sum()),'final_bulk_backlog_bits':float(backlog.sum()),
            'bulk_overflow_drop_bits':managed_probes._bulk_overflow,
            'pending_control_packets':int((ctrl_age>=0).sum()),
            'pending_control_packet_age_slots':ctrl_age.astype(int).tolist(),
            'bulk_flow_conservation_error_bits':float(managed_probes._bulk_generated-bulk_bits_r.sum()-backlog.sum()-managed_probes._bulk_overflow)}
    return result


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
