"""Small actual native/RadioMap pipeline; not a scientific cohort or score."""
from dataclasses import replace
import numpy as np
import torch

from wirelessjev.fleetwm.data import episode
from wirelessjev.frontier.study import mask_scans, runtime_world, READINGS
from wirelessjev.closed_loop.telemetry import DeliveredScans
from wirelessjev.closed_loop.runtime import LiveForecasts
from wirelessjev.native.foundation import random_site
from wirelessjev.native.intent_policy import intent_policy
from wirelessjev.gate_repair.gate import reactive_rescue
from wirelessjev.hardware_budget import OutageBudgetGate
from wirelessjev.continual_packet.control import ResponsePredictor
from wirelessjev.continual_packet.probes import ProbeMailbox, ProbeRuntime
from wirelessjev.continual_packet.study import public_spec, protocol
from wirelessjev.continual_packet.sim import simulate


def test_real_map_forecasts_assimilate_only_paid_delivered_native_reports():
    torch.set_num_threads(1)
    seed, duration, robots = 1999000, 3, 4
    full, _, geometry, observations = episode(seed, robots, duration, 'main')
    p = protocol() | {'duration_seconds': duration, 'robots': robots,
                      'measurement_missing_probability': 0.}
    spec = public_spec(seed, geometry, 'staggered', p)
    mailbox = ProbeMailbox()
    predictor = ResponsePredictor(geometry, spec, mailbox, 'memory',
                                  receiver_domains=full.domain[:, 1])
    history = DeliveredScans(mask_scans(observations, 5).before(60), duration, seed)
    live = LiveForecasts(predictor, history, duration,
                         {'cold': .01, 'warm': .01, 'refit': .01}, .6)
    world = runtime_world(full, duration, live.mean, live.std[:, :, 0])
    private = ProbeRuntime.from_world(world, spec, mailbox, site_params=random_site(seed))
    reading = READINGS['scanner_quiet']
    policy = replace(intent_policy(reading['weights'], reading['focus'], dual_hardware='none'),
                     online_map=False, cadence_scan=True, scan_period_s=5., roam_cost_s=.03)
    result = simulate(world, policy, interference=True, managed_probes=private,
        forecast_queue=live, forecast_source=live, scan_observer=history,
        scan_duration_s=.09, roam_gate=reactive_rescue,
        handover_admission=OutageBudgetGate(live.mean, live, .03), detailed=True)
    assert result['control_sent'] == robots * duration * 20
    assert len(predictor.records) == duration
    assert predictor.records[0]['used_observations'] == 0
    assert sum(r['used_observations'] for r in predictor.records[1:]) > 0
    assert private.summary()['cycles_delivered'] == 1
    assert private.summary()['report_bytes_transmitted'] > 0
    assert np.isfinite(live.mean).all()
    for record in live.records:
        assert record['history_cutoff_exclusive'] == 60 + record['decision']
        assert record['ready_slot'] > record['decision'] * 100
