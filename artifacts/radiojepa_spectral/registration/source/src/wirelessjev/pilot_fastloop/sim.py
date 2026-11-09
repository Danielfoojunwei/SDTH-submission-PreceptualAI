# Pilot/fast-loop kernel; predecessor mechanics retained when both features are off.
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
import time

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


def scan_suppression_evidence(queue, slot, robot, current_site, measured_snr, noise, tx,
                              control_gap_slots, primary_available, *, pilot_site=None,
                              pilot_capture_slot=None, pilot_refresh_slot=None):
    """Use only consumed forecasts and current public pilots to postpone a scan.

    The original forecast is interpolated at the actual age plus0..5s.
    Its route innovation is added to the lagged local pilot. Missing, stale
    or unsupported evidence permits a full scan. No propagation truth or
    newly manufactured observation enters the decision.
    """
    evidence = {'slot': int(slot), 'robot': int(robot), 'current_site': int(current_site),
        'control_gap_slots': float(control_gap_slots), 'primary_available': bool(primary_available),
        'source_decision': None, 'forecast_age_seconds': None,
        'forecast_ready_slot': None, 'original_forecast_snr_db': None,
        'pilot_site': pilot_site, 'pilot_capture_slot': pilot_capture_slot,
        'pilot_refresh_slot': pilot_refresh_slot,
        'current_best_rssi_dbm': None, 'predicted_best_worst_rssi_dbm': None,
        'measured_snr_db': None, 'noise_dbm': None, 'ap_tx_dbm': None,
        'current_rssi_dbm': None, 'forecast_at_age_plus_samples_snr_db': None,
        'route_innovation_db': None,
        'prediction_sample_seconds': list(range(6)),
        'suppress_scan': False, 'reason': 'missing_forecast'}
    try:
        measured_snr, noise, tx = map(np.asarray, (measured_snr, noise, tx))
    except (ValueError, TypeError):
        evidence['reason'] = 'invalid_pilot'; return evidence
    if any(v.shape != (3,) or v.dtype.kind not in 'fi' or not np.isfinite(v).all()
           for v in (measured_snr, noise, tx)):
        evidence['reason'] = 'invalid_pilot'; return evidence
    evidence['measured_snr_db'] = measured_snr.astype(float).tolist()
    evidence['noise_dbm'] = noise.astype(float).tolist()
    evidence['ap_tx_dbm'] = tx.astype(float).tolist()
    rssi = measured_snr + noise + tx - CLIENT_TX_DBM
    evidence['current_rssi_dbm'] = rssi.astype(float).tolist()
    evidence['current_best_rssi_dbm'] = float(rssi.max())
    if (pilot_site != current_site or type(pilot_capture_slot) is not int
            or type(pilot_refresh_slot) is not int
            or not 0 <= slot - pilot_refresh_slot < BLOCK_SLOTS
            or not 0 <= slot - pilot_capture_slot < 2 * BLOCK_SLOTS):
        evidence['reason'] = 'stale_or_wrong_association_pilot'; return evidence
    if not primary_available or control_gap_slots >= SAFETY_GAP_SLOTS:
        evidence['reason'] = 'control_recovery_required'; return evidence
    if rssi.max() < -67.:
        evidence['reason'] = 'weak_current_pilot'; return evidence
    source = getattr(queue, 'last', -1) if queue is not None else -1
    if isinstance(source, (bool, np.bool_)) or not isinstance(source, (int, np.integer)) or source < 0:
        return evidence
    source = int(source); age = slot / 100 - source
    evidence['source_decision'], evidence['forecast_age_seconds'] = source, age
    if not math.isfinite(age) or age < 0 or age > 5.:
        evidence['reason'] = 'stale_or_future_forecast'; return evidence
    if age + 5. > HORIZONS[-1]:
        evidence['reason'] = 'unsupported_forecast_horizon'; return evidence
    forecasts = getattr(queue, 'mean', None)
    if forecasts is None:
        return evidence
    try:
        values = np.asarray(forecasts[source, robot, :, current_site])
    except (IndexError, TypeError, ValueError, KeyError):
        evidence['reason'] = 'invalid_forecast_axes'; return evidence
    if values.shape != (len(HORIZONS), 3) or values.dtype.kind not in 'fi' or not np.isfinite(values).all():
        evidence['reason'] = 'invalid_forecast_values'; return evidence
    evidence['original_forecast_snr_db'] = values.astype(float).tolist()
    job = next((job for job in getattr(queue, 'records', ()) if job['decision'] == source), None)
    if job is None or type(job.get('ready_slot')) is not int or job['ready_slot'] > slot:
        evidence['reason'] = 'unverified_forecast_completion'; return evidence
    evidence['forecast_ready_slot'] = job['ready_slot']
    evidence['forecast_sha256'] = job['forecast_sha256']
    times = age + np.arange(6, dtype=float)
    shifted = np.stack([np.interp(times, HORIZONS, values[:, b]) for b in range(3)], -1)
    innovation = shifted - shifted[:1]
    evidence['forecast_at_age_plus_samples_snr_db'] = shifted.tolist()
    evidence['route_innovation_db'] = innovation.tolist()
    future_rssi = rssi[None] + innovation
    predicted = float(future_rssi.min(0).max())
    evidence['predicted_best_worst_rssi_dbm'] = predicted
    evidence['suppress_scan'] = predicted >= -67.
    evidence['reason'] = 'strong_current_and_predicted_link' if evidence['suppress_scan'] else 'predicted_link_risk'
    return evidence


def fast_control_evidence(queue, slot, robot, current_site, measured_snr, *,
                          control_band, primary_available, pilot_site,
                          pilot_capture_slot, pilot_refresh_slot,
                          association_epoch_start, last_primary_unavailable,
                          route_forecast=True):
    """Local cached-forecast control, without a new model or RF observation.

    Only a matching, available pilot can change the band. A missing or stale
    consumed forecast falls back to that pilot, whereas an invalid pilot
    holds the prior band and removes the old route innovation. The pilot is
    anchored at its genuine capture clock, rather than being treated as a
    current measurement. Forecast minima include original interpolation
    corners inside the five-second window and native integer query points.
    """
    evidence = {'slot': int(slot), 'robot': int(robot), 'current_site': int(current_site),
        'control_band_before': int(control_band), 'proposed_control_band': int(control_band),
        'primary_available': bool(primary_available), 'pilot_site': pilot_site,
        'pilot_capture_slot': pilot_capture_slot, 'pilot_refresh_slot': pilot_refresh_slot,
        'association_epoch_start': association_epoch_start,
        'last_primary_unavailable': last_primary_unavailable,
        'mode': 'hold', 'reason': 'invalid_pilot', 'source_decision': None,
        'forecast_ready_slot': None, 'forecast_age_seconds': None,
        'forecast_capture_age_seconds': None,
        'forecast_sha256': None, 'original_forecast_snr_db': None,
        'measured_snr_db': None, 'query_seconds_from_now': None,
        'forecast_at_queries_snr_db': None, 'route_innovation_db': None,
        'forecast_at_capture_snr_db': None, 'forecast_at_now_snr_db': None,
        'native_mcs_correction_db': [0., 0., 0.],
        'native_delta_normalization_seconds': float(BLOCK_SLOTS * SLOT),
        'worst_future_snr_db': None, 'native_delta_db_per_second': [0., 0., 0.]}
    try:
        measured = np.asarray(measured_snr)
    except (TypeError, ValueError):
        return evidence
    if (measured.shape != (3,) or measured.dtype.kind not in 'fi'
            or not np.isfinite(measured).all()):
        return evidence
    evidence['measured_snr_db'] = measured.astype(float).tolist()
    integer_fields = (pilot_site, pilot_capture_slot, pilot_refresh_slot,
                      association_epoch_start, last_primary_unavailable)
    if (any(type(value) is not int for value in integer_fields)
            or pilot_site != current_site
            or not 0 <= pilot_capture_slot <= pilot_refresh_slot <= slot
            or not 0 <= association_epoch_start <= slot
            or not -1 <= last_primary_unavailable <= slot
            or not 0 <= slot - pilot_refresh_slot < BLOCK_SLOTS
            or not 0 <= slot - pilot_capture_slot < 2 * BLOCK_SLOTS
            or pilot_capture_slot < association_epoch_start
            or pilot_capture_slot <= last_primary_unavailable):
        evidence['reason'] = 'invalid_pilot_history'
        return evidence
    if not primary_available:
        evidence['reason'] = 'primary_unavailable'
        return evidence

    # Current valid pilots remain usable before inference or after expiry.
    evidence.update(mode='myopic', reason='no_consumed_forecast',
        proposed_control_band=int(np.argmax(measured)),
        worst_future_snr_db=measured.astype(float).tolist())
    if not route_forecast:
        evidence['reason'] = 'route_forecast_disabled'
        return evidence
    source = getattr(queue, 'last', -1) if queue is not None else -1
    if (isinstance(source, (bool, np.bool_))
            or not isinstance(source, (int, np.integer)) or source < 0):
        return evidence
    source = int(source)
    age = slot / 100 - source
    capture_age = pilot_capture_slot / 100 - source
    evidence.update(source_decision=source, forecast_age_seconds=age,
                    forecast_capture_age_seconds=capture_age)
    if (not math.isfinite(age) or age < 0 or age > 5. or age + 5. > HORIZONS[-1]
            or not math.isfinite(capture_age) or not 0 <= capture_age <= age):
        evidence['reason'] = 'stale_or_unsupported_forecast'
        return evidence
    try:
        values = np.asarray(queue.mean[source, robot, :, current_site])
    except (AttributeError, IndexError, TypeError, ValueError, KeyError):
        evidence['reason'] = 'invalid_forecast_axes'
        return evidence
    if (values.shape != (len(HORIZONS), 3) or values.dtype.kind not in 'fi'
            or not np.isfinite(values).all()):
        evidence['reason'] = 'invalid_forecast_values'
        return evidence
    job = next((row for row in getattr(queue, 'records', ())
                if hasattr(row, 'get') and type(row.get('decision')) is int
                and row['decision'] == source), None)
    digest = job.get('forecast_sha256') if job is not None else None
    if (job is None or type(job.get('ready_slot')) is not int
            or not 0 <= job['ready_slot'] <= slot
            or not isinstance(digest, str) or len(digest) != 64
            or any(character not in '0123456789abcdef' for character in digest)):
        evidence['reason'] = 'unverified_forecast_completion'
        return evidence
    evidence.update(forecast_ready_slot=job['ready_slot'],
        forecast_sha256=job.get('forecast_sha256'),
        original_forecast_snr_db=values.astype(float).tolist())
    knots = np.asarray(HORIZONS, dtype=float)
    # Exact extrema of each piecewise-linear band occur at endpoints/corners.
    absolute_queries = np.unique(np.r_[age + np.arange(6, dtype=float),
        knots[(knots > age) & (knots < age + 5.)]])
    shifted = np.stack([np.interp(absolute_queries, knots, values[:, band])
                        for band in range(3)], axis=-1)
    at_capture = np.array([np.interp(capture_age, knots, values[:, band])
                           for band in range(3)])
    innovation = shifted - at_capture[None]
    worst = (measured[None] + innovation).min(axis=0)
    correction = shifted[0] - at_capture
    # The retained MCS block multiplies native_delta by its 100 ms scale.
    # Correct the actual pilot-to-now interval, including off-tick arrivals;
    # this quantity is a compatibility normalization, not the instantaneous
    # slope when the capture interval differs from exactly 100 ms.
    delta = correction / (BLOCK_SLOTS * SLOT)
    evidence.update(mode='predictive', reason='current_pilot_and_supported_forecast',
        proposed_control_band=int(np.argmax(worst)),
        query_seconds_from_now=(absolute_queries - age).tolist(),
        forecast_at_queries_snr_db=shifted.tolist(), route_innovation_db=innovation.tolist(),
        forecast_at_capture_snr_db=at_capture.tolist(),
        forecast_at_now_snr_db=shifted[0].tolist(),
        native_mcs_correction_db=correction.tolist(),
        worst_future_snr_db=worst.tolist(), native_delta_db_per_second=delta.tolist())
    return evidence


def simulate(world: World, policy: Policy, *, seed: int = 0, curves: LinkCurves | None = None,
             duration_s: float | None = None, blockage_rate: float = BLOCKAGE_RATE_PER_MIN,
             collision_lambda: float = 0.0, interference: bool = False,
             roam_gate=None, decision_trace=None, forecast_queue=None, planner="native", scan_duration_s=None, handover_admission=None,
             scan_observer=None, forecast_source=None, roam_scorer=None, emergency_recovery=False,
             risk_gated_scans=False, fast_local=False) -> dict:
    if type(fast_local) is not bool or (fast_local and (
            policy.kind != 'native' or policy.duplication or policy.dual_radio)):
        raise ValueError('Fast local control requires native control without STR or a second radio')
    if type(risk_gated_scans) is not bool or (risk_gated_scans and (policy.kind != 'native' or not policy.cadence_scan)):
        raise ValueError('Risk-gated scans require explicit native cadence scans')
    if type(emergency_recovery) is not bool or (emergency_recovery and policy.kind != 'native'):
        raise ValueError('Emergency recovery is an explicit native-only option')
    if emergency_recovery:
        from ..jepa_repair.telemetry import TimedDeliveredScans
        if (not isinstance(scan_observer, TimedDeliveredScans)
                or not callable(getattr(scan_observer, 'recovery_neighbors', None))):
            raise TypeError('Emergency recovery requires observable local scans via TimedDeliveredScans')
        if not math.isfinite(policy.roam_cost_s) or policy.roam_cost_s < 0:
            raise ValueError('Recovery requires a finite nonnegative handover outage')
    # Use the configured interruption as a persistence threshold, not a fitted
    # coefficient. Scanning can still occur at the existing stop threshold.
    handover_threshold_slots = (SAFETY_GAP_SLOTS + math.ceil(policy.roam_cost_s / SLOT)
                                if emergency_recovery else None)
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

    def observed_neighbors(slot, *, watchdog=False):
        # Keep the ordinary legacy fallback cache/admission path unchanged.
        # Only the independent cost-thresholded watchdog requires freshness.
        if emergency_recovery and watchdog:
            return scan_observer.recovery_neighbors(slot, site, meas, world.noise_db,
                world.tx_dl, max_age_slots=int(round(policy.scan_period_s / SLOT)))
        if scan_observer is None:
            return rssi_dl(max(0, slot - BLOCK_SLOTS))
        return scan_observer.neighbors(site, meas, world.noise_db, world.tx_dl)

    def neighbor_observation(robot, target, band, slot):
        """Bind an alternate's observed RSSI to its genuine capture metadata.

        The selected band explains the AP-recovery trigger; it does not command
        the transmit band, which remains recorded in the native action trace.
        """
        if not emergency_recovery or target == int(site[robot]):
            raise RuntimeError('Recovery capture metadata requires an observed alternate AP')
        link = 3 * target + band
        heard = bool(scan_observer.local_capture_heard[robot, link])
        captured = int(scan_observer.local_capture_slots[robot, link])
        age = int(slot) - captured
        maximum = int(round(policy.scan_period_s / SLOT))
        if not heard or age < 0 or age > maximum:
            raise RuntimeError('Recovery neighbor lacks a fresh heard local capture')
        return {'selected_band': int(band),
                'selected_band_scope': 'observed neighbor RSSI, not a transmit-band command',
                'neighbor_capture_slot': captured, 'neighbor_capture_age_slots': age,
                'neighbor_heard': heard, 'neighbor_max_age_slots': maximum}

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
    pilot_site = site.copy()
    pilot_capture_slot = np.zeros(n, dtype=int)
    pilot_refresh_slot = np.zeros(n, dtype=int)
    association_epoch_site = site.copy()
    association_epoch_start = np.zeros(n, dtype=np.int64)
    last_primary_unavailable = np.full(n, -1, dtype=np.int64)
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
    # Failed control while the PRIMARY radio was available. Handover outage
    # is not evidence that another move is needed; scan slots pause the clock.
    available_silence = np.zeros(n, dtype=np.int64)
    gap_max = np.zeros(n)
    vq = np.zeros((n, VIDEO_DEADLINE))                       # video bits by age
    video_gen = video_ok = 0.0
    video_req_bits = 0.0
    video_gen_r = np.zeros(n)
    video_ok_r = np.zeros(n)
    bulk_bits_r = np.zeros(n)
    # Audit state only. The offer uses the exact existing per-slot expression,
    # including any input-dtype rounding before its addition to float64 queues.
    audit_bulk_offer = np.asarray(world.bulk_mbps * 1e6 * SLOT, dtype=np.float64)
    packet_trace = []
    action_trace = []
    audit_video_expired = 0.0
    airtime_used = 0.0
    dup_slots = 0
    has_video = world.video_mbps > 0

    roam_log = []
    ctrl_down = ctrl_fail = ctrl_congested = 0
    win_miss = np.zeros(n, dtype=bool)
    windows = clean_windows = 0
    due = np.zeros(n, dtype=bool)                # cadence_scan: a scan is waiting for a quiet moment
    scan_until = np.zeros(n, dtype=int)
    scan_reassess_after = np.zeros(n, dtype=int)
    scan_assessments, scan_starts, held_forecast_age = [], [], []
    fast_updates = []
    fast_forecast_slices = []
    fast_forecast_slice_ids = {}
    fast_mode_counts = {'predictive': 0, 'myopic': 0, 'hold': 0}
    fast_band_changes = 0
    fast_timing = {'calls': 0, 'total_seconds': 0., 'max_seconds': 0.}
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

    emergency_actions = []
    emergency_scans = []
    recovery_checks = []

    def sync_association_epoch(slot):
        changed = site != association_epoch_site
        association_epoch_start[changed] = slot
        association_epoch_site[:] = site

    def move_primary(i, target, cost_s, emergency=False):
        nonlocal swaps
        dual = bool(dual_mask[i])
        if policy.cadence_roam and not emergency and cost_s < CONTROL_PERIOD * SLOT and not (dual and site2[i] == target):
            pending_roam[i] = target       # executed right after the next control delivery
            return
        available_silence[i] = 0  # new association resets old-link evidence
        if emergency:
            pending_roam[i] = -1
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
        sync_association_epoch(k)
        last_primary_unavailable[(outage_until > k) | (scan_until > k)] = k
        fast_step_events = []
        if scan_observer is not None:
            scan_observer.complete(k)
        if scan_observer is not None and k % slots_per_dec == 0:
            scan_observer.pose(k // slots_per_dec, world.pos[min(k // slots_per_pos, len(world.pos)-1)])
        if forecast_source is not None and k % slots_per_dec == 0:
            d = k // slots_per_dec
            issued = forecast_source.issue(d, world.plan[d])
            if issued is not None:
                mean, sd = issued
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
        # ---------------- measurements (lagged by one fading block) --------
        if k % BLOCK_SLOTS == 0:
            observed = true_snr(max(0, k - BLOCK_SLOTS), site)
            if scan_observer is None:
                meas = observed
                refreshed = np.ones(n, dtype=bool)
            else:
                available = (outage_until <= k) & (scan_until <= k)
                meas = np.where(available[:, None], observed, meas)
                refreshed = available
            pilot_site = np.where(refreshed, site, pilot_site)
            pilot_capture_slot = np.where(refreshed, max(0, k - BLOCK_SLOTS), pilot_capture_slot)
            pilot_refresh_slot = np.where(refreshed, k, pilot_refresh_slot)
            if scan_observer is not None and callable(getattr(scan_observer, 'update_pilots', None)):
                captured = max(0, k - BLOCK_SLOTS)
                kp_capture = min(captured // slots_per_pos, len(world.pos) - 1)
                scan_observer.update_pilots(k, current_sites=site, measured_snr=meas,
                    positions=world.pos[kp_capture], capture_slot=captured, refreshed=refreshed,
                    noise=world.noise_db, tx=world.tx_dl,
                    association_epoch_start=association_epoch_start,
                    last_primary_unavailable=last_primary_unavailable)
            for b in range(3):
                band = BANDS[b]
                m = meas[:, b] - (3.0 if policy.kind != "legacy" else 0.0) * (second_link >= 0)
                mcs[b] = _mcs_for(curves, band, m - 2.0)          # Minstrel-like backoff

        # Robot-local watchdog: no forecast arrival or central report is
        # required. Only completed local neighbor scans and current pilots
        # enter this decision. The existing 200 ms stop threshold permits observation; a move
        # additionally requires PRIMARY-radio-available silence spanning its
        # configured outage. Our own handovers reset the evidence clock.
        if emergency_recovery and k % BLOCK_SLOTS == 0:
            radio_ready = (outage_until <= k) & (scan_until <= k)
            silent = gap_cur >= SAFETY_GAP_SLOTS
            if (silent & radio_ready).any():
                r = observed_neighbors(k, watchdog=True)
                cur = r[ar, site].max(axis=1)
                target = np.argmax(r.max(axis=2), axis=1)
                target_band = np.argmax(r[ar, target], axis=1)
                gain = r[ar, target].max(axis=1) - cur
                neighbor_rssi = r[ar, target].max(axis=1)
                sustained = available_silence >= handover_threshold_slots
                rescue = (sustained & radio_ready & (cur < -75.) & (gain >= 8.)
                          & (neighbor_rssi >= -90.) & (target != site))
                other = r.max(axis=2).copy()
                other[ar, site] = -np.inf
                best_other = np.argmax(other, axis=1)
                best_other_band = np.argmax(r[ar, best_other], axis=1)
                for i in np.flatnonzero(silent & radio_ready):
                    neighbor_value = other[i, best_other[i]]
                    metadata = (neighbor_observation(i, int(best_other[i]), int(best_other_band[i]), k)
                        if np.isfinite(neighbor_value) else {
                            'selected_band': None,
                            'selected_band_scope': 'observed neighbor RSSI, not a transmit-band command',
                            'neighbor_capture_slot': None, 'neighbor_capture_age_slots': None,
                            'neighbor_heard': False})
                    recovery_checks.append({'slot': int(k), 'robot': int(i),
                        'current_site': int(site[i]), 'current_rssi_dbm': float(cur[i]),
                        'best_other_site': int(best_other[i]) if np.isfinite(neighbor_value) else None,
                        'best_other_rssi_dbm': float(neighbor_value) if np.isfinite(neighbor_value) else None,
                        'silence_slots': int(gap_cur[i]), 'recovery_triggered': bool(rescue[i]),
                        'available_silence_slots': int(available_silence[i]),
                        'handover_threshold_slots': handover_threshold_slots,
                        'periodic_scan_due_before_traffic_update': bool(due[i]),
                        'pending_cadence_target': int(pending_roam[i]),
                        'neighbor_max_age_slots': int(round(policy.scan_period_s / SLOT)),
                        **metadata})
                for i in np.flatnonzero(rescue):
                    previous_site = int(site[i])
                    pending_before = int(pending_roam[i])
                    metadata = neighbor_observation(i, int(target[i]), int(target_band[i]), k)
                    available_before = int(available_silence[i])
                    move_primary(i, target[i], policy.roam_cost_s, emergency=True)
                    emergency_actions.append({'slot': int(k), 'robot': int(i),
                        'from_site': previous_site, 'to_site': int(target[i]),
                        'silence_slots': int(gap_cur[i]), 'current_rssi_dbm': float(cur[i]),
                        'neighbor_gain_db': float(gain[i]), 'neighbor_rssi_dbm': float(neighbor_rssi[i]),
                        'outage_until_slot': int(outage_until[i]),
                        'handover_threshold_slots': handover_threshold_slots,
                        'available_silence_slots': available_before,
                        'available_silence_slots_after': int(available_silence[i]),
                        'pending_cadence_target_before': pending_before,
                        'pending_cadence_target_after': int(pending_roam[i]), **metadata})

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
                    r = observed_neighbors(k)
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
            if not fast_local:
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
            if not fast_local:
                fc_now = fc[:, 0][ar, site]                        # (N,3)
                fc_next = fc[:, 1][ar, site]
                native_delta = (fc_next - fc_now)                  # dB per second along route

        # Cached-prediction feedback is separate from AP planning/inference.
        # It uses the latest consumed source, never a merely issued forecast.
        sync_association_epoch(k)
        last_primary_unavailable[(outage_until > k) | (scan_until > k)] = k
        if fast_local and (do_native or k % BLOCK_SLOTS == 0):
            fast_band_before = ctrl_link.copy()
            fast_pilot_valid = np.zeros(n, dtype=bool)
            for i in range(n):
                started_helper = time.perf_counter()
                evidence = fast_control_evidence(forecast_queue, k, i, int(site[i]), meas[i],
                    control_band=int(ctrl_link[i]),
                    primary_available=bool(outage_until[i] <= k and scan_until[i] <= k),
                    pilot_site=int(pilot_site[i]),
                    pilot_capture_slot=int(pilot_capture_slot[i]),
                    pilot_refresh_slot=int(pilot_refresh_slot[i]),
                    association_epoch_start=int(association_epoch_start[i]),
                    last_primary_unavailable=int(last_primary_unavailable[i]),
                    route_forecast=policy.route_forecast)
                elapsed_helper = time.perf_counter() - started_helper
                if not math.isfinite(elapsed_helper) or elapsed_helper < 0:
                    raise RuntimeError('Invalid measured local-control work')
                fast_timing['calls'] += 1
                fast_timing['total_seconds'] += elapsed_helper
                fast_timing['max_seconds'] = max(fast_timing['max_seconds'], elapsed_helper)
                evidence.update(forecast_arrival=bool(arrived is not None),
                                pilot_tick=bool(k % BLOCK_SLOTS == 0))
                original = evidence.pop('original_forecast_snr_db')
                # Piecewise-linear query values/innovations are reproducible
                # from the source catalog and clocks. Avoid repeating these
                # large arrays for every robot on every 100 ms tick.
                evidence.pop('forecast_at_queries_snr_db')
                evidence.pop('route_innovation_db')
                evidence['forecast_slice_id'] = None
                if original is not None:
                    slice_key = (evidence['source_decision'], i, int(site[i]),
                                 evidence['forecast_sha256'])
                    if slice_key not in fast_forecast_slice_ids:
                        slice_id = len(fast_forecast_slices)
                        fast_forecast_slice_ids[slice_key] = slice_id
                        fast_forecast_slices.append({'forecast_slice_id': slice_id,
                            'source_decision': evidence['source_decision'],
                            'robot': i, 'current_site': int(site[i]),
                            'forecast_ready_slot': evidence['forecast_ready_slot'],
                            'forecast_sha256': evidence['forecast_sha256'],
                            'horizons_seconds': [float(h) for h in HORIZONS],
                            'original_forecast_snr_db': original})
                    evidence['forecast_slice_id'] = fast_forecast_slice_ids[slice_key]
                ctrl_link[i] = evidence['proposed_control_band']
                native_delta[i] = evidence['native_delta_db_per_second']
                fast_pilot_valid[i] = evidence['mode'] != 'hold'
                fast_mode_counts[evidence['mode']] += 1
                fast_step_events.append(evidence)

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
            if fast_local:
                # Preserve the existing reflex on usable pilots. An old-AP
                # or unavailable capture cannot undo the new hold decision.
                ctrl_link = np.where(fast_pilot_valid, ctrl_link, fast_band_before)

        for evidence in fast_step_events:
            i = evidence['robot']
            evidence['actual_control_band'] = int(ctrl_link[i])
            evidence['band_changed'] = bool(ctrl_link[i] != evidence['control_band_before'])
            fast_band_changes += int(evidence['band_changed'])
            fast_updates.append(evidence)

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
        audit_video_expired += float(expired.sum())
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
        if scan_observer is not None and callable(getattr(scan_observer, 'prepare_frames', None)):
            sync_association_epoch(k)
            last_primary_unavailable[~up] = k
            scan_observer.prepare_frames(k, attempted=pend, current_sites=site,
                primary_available=up, association_epoch_start=association_epoch_start,
                last_primary_unavailable=last_primary_unavailable)
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
        win_miss |= late
        ctrl_age = np.where(late, -1, ctrl_age)
        if k % 100 == 99:
            windows += n
            clean_windows += int((~win_miss).sum())
            win_miss[:] = False
        if emergency_recovery:
            # A successful control resets the evidence. Known primary outage
            # also resets it; scan time only pauses it. Quiet inter-packet
            # slots count as available silence until the next20Hz delivery.
            available_silence = np.where(delivered | (outage_until > k), 0,
                np.where(scanning, available_silence, available_silence + 1))
        gap_cur = np.where(delivered, 0, gap_cur + 1)
        gap_max = np.maximum(gap_max, gap_cur)
        stopped_slots += ((gap_cur > SAFETY_GAP_SLOTS) & (gap_cur > 0)).sum()
        resumed = delivered & (prev_gap > SAFETY_GAP_SLOTS)
        stopped_slots += resumed.sum() * int(RESTART_S / SLOT)
        stop_events += int(resumed.sum())
        prev_gap = gap_cur.copy()
        if policy.cadence_scan:
            watchdog_scan = (emergency_recovery & (gap_cur >= SAFETY_GAP_SLOTS)
                             & (outage_until <= k) & (scan_until <= k))
            go_scan = (delivered | watchdog_scan) & due
            if risk_gated_scans:
                # Postponement changes the next ordinary assessment only.
                # The genuinely due scan remains outstanding so persistent
                # control silence bypasses this delay.
                ordinary = go_scan & ~watchdog_scan
                check = ordinary & (scan_reassess_after <= k)
                go_scan &= watchdog_scan | (scan_reassess_after <= k)
                for i in np.flatnonzero(check):
                    evidence = scan_suppression_evidence(forecast_queue, k, int(i),
                        int(site[i]), meas[i], world.noise_db, world.tx_dl,
                        float(gap_cur[i]), bool(outage_until[i] <= k),
                        pilot_site=int(pilot_site[i]), pilot_capture_slot=int(pilot_capture_slot[i]),
                        pilot_refresh_slot=int(pilot_refresh_slot[i]))
                    scan_assessments.append(evidence)
                    if evidence['suppress_scan']:
                        go_scan[i] = False
                        scan_reassess_after[i] = k + int(round(policy.scan_period_s / SLOT))
                for i in np.flatnonzero(go_scan & watchdog_scan):
                    scan_assessments.append({'slot': int(k), 'robot': int(i),
                        'current_site': int(site[i]), 'control_gap_slots': float(gap_cur[i]),
                        'primary_available': bool(outage_until[i] <= k),
                        'suppress_scan': False, 'reason': 'emergency_silence_bypass'})
            for i in np.flatnonzero(go_scan):
                scan_starts.append({'slot': int(k), 'robot': int(i),
                    'current_site': int(site[i]), 'control_gap_slots': float(gap_cur[i]),
                    'emergency_silence': bool(watchdog_scan[i]),
                    'complete_slot': k + 1 + int(round((SCAN_S if scan_duration_s is None else scan_duration_s) / SLOT))})
            scan_reassess_after[go_scan] = 0
            for i in np.flatnonzero(go_scan & ~delivered):
                emergency_scans.append({'slot': int(k), 'robot': int(i),
                    'silence_slots': int(gap_cur[i]),
                    'scan_complete_slot': k + 1 + int(round((SCAN_S if scan_duration_s is None else scan_duration_s) / SLOT))})
            scan_until = np.where(go_scan, k + 1 + int(round((SCAN_S if scan_duration_s is None else scan_duration_s) / SLOT)), scan_until)
            if scan_observer is not None and go_scan.any():
                ids = np.flatnonzero(go_scan)
                kp = min(k // slots_per_pos, len(world.pos)-1)
                scan_observer.scan(k, ids, world.pos[kp, ids], rssi_dl(k)[ids], scan_until[ids])
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

        # These snapshots are taken after all native actions and traffic in the
        # slot. They neither consume randomness nor feed any control decision.
        if (k + 1) % 100 == 0 or k + 1 == n_slots:
            source = getattr(forecast_queue, 'last', -1) if forecast_queue is not None else -1
            source = int(source) if source is not None else -1
            held_forecast_age.append({'slot_end': int(k + 1),
                'source_decision': source if source >= 0 else None,
                'age_seconds': (k + 1) / 100 - source if source >= 0 else None,
                'phase': 'right_boundary_before_next_slot_consumption'})
            packet_trace.append({
                "slot_end": k + 1,
                "misses": ctrl_miss_r.astype(np.int64).tolist(),
                "sent": ctrl_total_r.astype(np.int64).tolist(),
                "max_gap": gap_max.astype(np.int64).tolist(),
                "current_gap": gap_cur.astype(np.int64).tolist(),
                "pending_control_packet_age_slots": ctrl_age.astype(np.int64).tolist(),
                **({'available_silence_slots': available_silence.tolist()}
                   if emergency_recovery else {}),
            })
        if (k + 1) % BLOCK_SLOTS == 0 or k + 1 == n_slots:
            action_trace.append({
                "slot_end": k + 1,
                "primary_site": site.astype(np.int64).tolist(),
                "control_band": ctrl_link.astype(np.int64).tolist(),
                "bulk_band": bulk_link.astype(np.int64).tolist(),
                "secondary_band": second_link.astype(np.int64).tolist(),
                "primary_unavailable_until_slot": outage_until.astype(np.int64).tolist(),
                "mcs": np.stack([mcs[b] for b in range(3)], axis=-1).astype(np.int64).tolist(),
            })

    audit_bulk_generated = audit_bulk_offer * n_slots
    audit_bulk_residual = audit_bulk_generated - bulk_bits_r - backlog
    audit_bulk_overflow = np.maximum(audit_bulk_residual, 0.)
    audit_video_pending = float(vq.sum())
    audit_video_residual = float(video_gen - video_ok - audit_video_pending)
    ctrl_frac_r = ctrl_miss_r / np.maximum(ctrl_total_r, 1)
    order = np.sort(ctrl_frac_r)
    worst = order[-max(1, n // 10):]
    vr = has_video
    return {
        "fast_local_execution": {
            'fast_local_enabled': fast_local, 'fast_control_updates': fast_updates,
            'forecast_slices': fast_forecast_slices,
            'mode_counts': fast_mode_counts, 'band_changes': fast_band_changes,
            'existing_block_slots': BLOCK_SLOTS,
            'zero_intra_ap_switch_cost_assumption': True,
            'extra_RF_or_model_calls': False,
            'helper_work_independently_scheduled': False,
            'shared_other_work_allowance_remains_assumed': True,
            'AP_planning_still_requires_forecast_arrival': True},
        "fast_local_timing": fast_timing,
        "freshness_execution": {'risk_gated_scans_enabled': risk_gated_scans,
            'scan_guard_rssi_dbm': -67., 'maximum_forecast_age_seconds': 5.,
            'lookahead_seconds': 5., 'reassessment_seconds': float(policy.scan_period_s),
            'emergency_due_scan_bypasses_postponement': True,
            'postponement_preserves_due_and_capture_timestamps': True,
            'scan_assessments': scan_assessments, 'scan_starts': scan_starts,
            'held_forecast_age': held_forecast_age},
        "repair_execution": {"emergency_recovery_enabled": emergency_recovery,
                             "watchdog_threshold_slots": SAFETY_GAP_SLOTS,
                             **({'handover_threshold_slots': handover_threshold_slots,
                                 'handover_evidence': 'primary-radio-available control silence; handover resets, scans pause',
                                 'available_silence_slots_final': available_silence.tolist()}
                                if emergency_recovery else {}),
                             "watchdog_checks": recovery_checks,
                             "emergency_handovers": emergency_actions,
                             "emergency_scans": emergency_scans,
                             "fresh_central_reports_still_require_control_delivery": True},
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
        "packet_detail": {
            "misses": ctrl_miss_r.astype(np.int64).tolist(),
            "sent": ctrl_total_r.astype(np.int64).tolist(),
            "max_gap": gap_max.astype(np.int64).tolist(),
            "pending_control_packet_age_slots": ctrl_age.astype(np.int64).tolist(),
            "pending_control_packets": int((ctrl_age >= 0).sum()),
            "pending_control_right_censored": True,
            "per_robot_final_bulk_backlog_bits": backlog.tolist(),
            "cumulative": packet_trace,
            "cumulative_period_slots": 100,
            "action_trace": action_trace,
            "action_trace_period_slots": BLOCK_SLOTS,
            "action_trace_phase": "end_of_slot_state_not_complete_event_log",
        },
        "packet_accounting": {
            "audited_slots": n_slots,
            "slot_seconds": SLOT,
            "video_requested_bits": float(video_req_bits),
            "video_generated_bits": float(video_gen),
            "video_delivered_bits": float(video_ok),
            "video_pending_bits": audit_video_pending,
            "video_expired_bits": audit_video_expired,
            "video_flow_conservation_error_bits": audit_video_residual - audit_video_expired,
            "video_expiry_measured_from_queue": True,
            "bulk_generated_bits": float(audit_bulk_generated.sum()),
            "bulk_delivered_bits": float(bulk_bits_r.sum()),
            "final_bulk_backlog_bits": float(backlog.sum()),
            "bulk_overflow_drop_bits": float(audit_bulk_overflow.sum()),
            "per_robot_bulk_generated_bits": audit_bulk_generated.tolist(),
            "per_robot_bulk_delivered_bits": bulk_bits_r.tolist(),
            "per_robot_final_bulk_backlog_bits": backlog.tolist(),
            "per_robot_bulk_overflow_drop_bits": audit_bulk_overflow.tolist(),
            "per_robot_bulk_raw_unaccounted_bits": audit_bulk_residual.tolist(),
            "bulk_flow_conservation_error_bits": float((audit_bulk_residual - audit_bulk_overflow).sum()),
            "bulk_overflow_inferred_from_mass_balance": True,
            "pending_control_packet_age_slots": ctrl_age.astype(np.int64).tolist(),
            "pending_control_packets": int((ctrl_age >= 0).sum()),
            "pending_control_right_censored": True,
        },
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
