"""Use real ns-3 payload-error curves in the existing batched evaluator."""
from __future__ import annotations

import json
import subprocess

import numpy as np

from ..fastplan import BatchEvaluator
from ..oracle import horizon_snapshots
from ..missions import WAREHOUSE_MISSIONS
from .data import make_environment
from .experiment import atomic_json, sha256
from .studies import ns3_binary


def apply_curves(evaluator, curves, margin_db=0.):
    for band, threshold_name, rate_name in [('5g', 't5', 'r5'), ('2g', 't2', 'r2')]:
        rows = [row for row in curves['points'] if row['band'] == band]
        thresholds = np.array(sorted({r['snr_db'] for r in rows}), dtype=float)
        rates = []
        for snr in thresholds:
            rates.append(max(r['rate_mbps']*r['packet_success'] for r in rows if r['snr_db'] == snr))
        if not np.isfinite(rates).all() or len(thresholds) < 2:
            raise ValueError('Invalid upstream Wi-Fi payload-error curves')
        setattr(evaluator, threshold_name, thresholds+margin_db)
        setattr(evaluator, rate_name, np.array(rates))


def study(root, data):
    curves_path = root/'wifi_link_curves.json'
    binary = ns3_binary('warehouse-link-curves')
    subprocess.run([str(binary), '--output='+str(curves_path.resolve())], check=True)
    curves = json.loads(curves_path.read_text())
    rows = []
    # Shared known-physics sensitivity experiment: every method gets the same
    # propagation. This isolates the rate abstraction, not map generalisation.
    for seed in sorted({s['scene'] for s in data['test']}):
        scene, _ = make_environment(seed)
        evs = {}
        for name, margin in [('ideal_table', None), ('nist_margin0', 0.), ('nist_margin3', 3.), ('nist_margin6', 6.)]:
            ev = BatchEvaluator(scene, horizon_snapshots(scene))
            if margin is not None:
                apply_curves(ev, curves, margin)
            evs[name] = ev
        sample = next(s for s in data['test'] if s['scene'] == seed)
        plans = sample['plans']
        for mission, definition in WAREHOUSE_MISSIONS.items():
            reference = evs['nist_margin0'].utility(plans, definition.weights)
            for name, ev in evs.items():
                choice = int(ev.utility(plans, definition.weights).argmax())
                rows.append({'scene': seed, 'mission': mission, 'method': name, 'choice': choice,
                             'reference_gap': float(reference.max()-reference[choice]),
                             'reference_utility': float(reference[choice])})
    return {'source': 'ns3_error_model_integrated_with_warehouse_evaluator', 'curve_sha256': sha256(curves_path),
            'records': rows,
            'limits': ['NIST payload-error model, not measured hardware curves or a complete Wi-Fi receiver',
                       'Maximum expected payload rate over MCS values; not a model of deployed Minstrel adaptation',
                       'Existing MAC efficiency factor and airtime model are retained',
                       'Nominal two spatial streams; no correlated fading, channel-estimation or beamforming model',
                       'Gap to best of the fixed candidates under the zero-margin NIST reference']}
