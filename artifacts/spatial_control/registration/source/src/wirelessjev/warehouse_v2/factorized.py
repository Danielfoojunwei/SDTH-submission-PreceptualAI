"""Separate known switch interruption from learned radio service.

The first v2 controller learnt a five-second, interruption-dominated target and
then held decisions for thirty seconds. This prospective correction learns
uninterrupted service and charges the declared switch cost analytically.
It does not turn a five-second signal predictor into a thirty-second world model.
"""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
import torch

from .model import TemporalRadioModel, tensor_inputs, dynamics_loss
from .experiment import atomic_json, sha256
from .closedloop import rollout
from .studies import load_model


class FactorizedRadioModel(TemporalRadioModel):
    def __init__(self, *, horizon_s=30., **kwargs):
        super().__init__(**kwargs)
        if horizon_s <= 4.:
            raise ValueError('The declared commitment horizon must exceed the four-second switch cost')
        self.horizon_s = float(horizon_s)
        self.config['horizon_s'] = self.horizon_s

    def raw_prediction(self, inputs, plans):
        return super().forward(inputs, plans)

    def forward(self, inputs, plans, semantic=None):
        p = super().forward(inputs, plans, semantic)
        p['uninterrupted_service'] = p['service']
        keep = 1.-(plans != inputs['current'][None]).float()*4./self.horizon_s
        p['service'] = p['uninterrupted_service']*keep
        return p


def raw_target(sample):
    target = torch.tensor(sample['targets']).clone()
    moved = sample['plans'] != sample['inputs']['current'][None]
    keep = torch.tensor(1.-moved*.8, dtype=torch.float32)
    target[..., 0] = (target[..., 0]/keep).clamp(0, 1)
    return target


def fit(samples, validation, root, *, seed, graph):
    path = root/'checkpoints'/f'factorized-{seed}-{graph}.pt'
    meta_path = root/f'factorized-train-{seed}-{graph}.json'
    if path.exists() and meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())
        if sha256(path) != meta['checkpoint_sha256']:
            raise RuntimeError('Factorized checkpoint changed')
        checkpoint = torch.load(path, weights_only=True)
        model = FactorizedRadioModel(**checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return model, meta
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FactorizedRadioModel(graph=graph, temporal=graph)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, history = float('inf'), None, []
    started = time.perf_counter()
    for epoch in range(20):
        model.train()
        for i in rng.permutation(len(samples)):
            sample = samples[i]
            prediction = model.raw_prediction(tensor_inputs(sample), torch.tensor(sample['plans']))
            loss = dynamics_loss(prediction, raw_target(sample))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(np.mean([float(dynamics_loss(model.raw_prediction(tensor_inputs(s), torch.tensor(s['plans'])), raw_target(s))) for s in validation]))
        history.append({'epoch': epoch+1, 'validation_loss': validation_loss})
        if validation_loss < best:
            best, best_state = validation_loss, deepcopy(model.state_dict())
        print(f'Factorized graph={graph} seed={seed} epoch={epoch+1}/20 validation={validation_loss:.5f}', flush=True)
    model.load_state_dict(best_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'config': model.config, 'model': model.state_dict()}, path)
    metadata = {'seed': seed, 'graph': graph, 'epochs': 20, 'history': history,
                'checkpoint_sha256': sha256(path), 'training_elapsed_s': time.perf_counter()-started}
    atomic_json(meta_path, metadata)
    return model, metadata


def study(root, data, seeds):
    rows = []
    for seed in seeds:
        models = {}
        for graph in [False, True]:
            model, meta = fit(data['train'], data['validation'], root, seed=seed, graph=graph)
            models['factorized_graph' if graph else 'factorized_mlp'] = model
        # New rollout seeds were not used in the initial failure analysis.
        for policy in ['stay', 'legacy_hysteresis', 'observed_load_balance', 'instant_mlp', 'temporal_graph', 'factorized_mlp', 'factorized_graph']:
            if policy in models:
                model = models[policy]
            elif policy in ['instant_mlp', 'temporal_graph']:
                model = load_model(root, seed, policy)[0]
            else:
                model = None
            report = rollout(131000+seed, policy, model)
            report['training_seed'] = seed
            rows.append(report)
            print(f'Prospective horizon correction seed={seed} policy={policy} service={report["metrics"]["mean_service"]:.4f}', flush=True)
            atomic_json(root/'factorized-progress.json', {'records': rows})
    return {'source': 'prospective_simulated_handover_factorization', 'records': rows,
            'limits': ['Designed after observing the first v2 rollout failure; evaluated on new rollout seeds',
                       'Five-second observed-intervention training labels; thirty-second analytic interruption adjustment',
                       'Does not predict all intermediate route states or supply a calibrated thirty-second risk probability',
                       'Assumes a four-second interruption, pending actual-car calibration']}
