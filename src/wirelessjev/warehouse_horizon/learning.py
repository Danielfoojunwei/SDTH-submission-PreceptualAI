"""Independent compact training seeds and validation-only residual selection."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
import torch

from ..warehouse_v2.experiment import atomic_json, sha256
from ..warehouse_v2.metrics import risk_metrics, cluster_interval
from .data import CONFIG
from .model import HorizonModel, tensors, prior_prediction, loss
from .planning import Contract, charge, values


def model_prediction(model, sample):
    if model is None:
        return prior_prediction(sample['prior'], sample['observation'])
    with torch.inference_mode():
        output = model(tensors(sample['observation']), torch.as_tensor(sample['plans']), torch.as_tensor(sample['prior']))
    return {k: v.numpy() for k, v in output.items()}


def blend(base, learned, strength):
    probability = (1-strength)*expit(base['failure_logits'])+strength*expit(learned['failure_logits'])
    return {'service_raw': (1-strength)*base['service_raw']+strength*learned['service_raw'],
            'failure_logits': logit(np.clip(probability, 1e-6, 1-1e-6)),
            'signal': (1-strength)*base['signal']+strength*learned['signal']}


def fit(samples, validation, root, *, seed, graph, fingerprint):
    path = root/'checkpoints'/f'horizon-{seed}-{graph}.pt'
    meta_path = root/f'train-{seed}-{graph}.json'
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['fingerprint'] != fingerprint or meta['checkpoint_sha256'] != sha256(path):
            raise RuntimeError('Training checkpoint provenance/configuration mismatch')
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        model = HorizonModel(**ckpt['config'])
        model.load_state_dict(ckpt['model'])
        model.eval()
        return model, meta
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = HorizonModel(graph=graph)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best, selected, best_state, history = float('inf'), 0, None, []
    started = time.perf_counter()
    for epoch in range(CONFIG['epochs']+1):
        if epoch:
            model.train()
            losses = []
            for i in rng.permutation(len(samples)):
                sample = samples[i]
                prediction = model(tensors(sample['observation']), torch.as_tensor(sample['plans']), torch.as_tensor(sample['prior']))
                target = {k: torch.as_tensor(v) for k, v in sample['target'].items()}
                objective = loss(prediction, target)
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                optimizer.step()
                losses.append(float(objective.detach()))
        model.eval()
        validation_losses = []
        with torch.inference_mode():
            for sample in validation:
                prediction = model(tensors(sample['observation']), torch.as_tensor(sample['plans']), torch.as_tensor(sample['prior']))
                validation_losses.append(float(loss(prediction, {k: torch.as_tensor(v) for k, v in sample['target'].items()})))
        score = float(np.mean(validation_losses))
        history.append({'epoch': epoch, 'validation_loss': score,
                        'train_loss': float(np.mean(losses)) if epoch else None})
        if score < best:
            best, selected, best_state = score, epoch, deepcopy(model.state_dict())
        print(f'Horizon graph={graph} seed={seed} epoch={epoch}/{CONFIG["epochs"]} validation={score:.5f}', flush=True)
        atomic_json(root/'progress.json', {'stage': 'training', 'seed': seed, 'graph': graph, 'epoch': epoch})
    model.load_state_dict(best_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'config': model.config, 'model': model.state_dict(), 'seed': seed}, path)
    meta = {'seed': seed, 'graph': graph, 'fingerprint': fingerprint, 'checkpoint_sha256': sha256(path),
            'history': history, 'selected_epoch': selected, 'epochs_trained': CONFIG['epochs'],
            'parameters': sum(p.numel() for p in model.parameters()), 'elapsed_s': time.perf_counter()-started}
    atomic_json(meta_path, meta)
    return model, meta


def temperature(logits, labels):
    def objective(log_t):
        value = logits/np.exp(log_t)
        return np.mean(np.logaddexp(0, value)-labels*value)
    result = minimize_scalar(objective, bounds=(-3., 3.), method='bounded')
    if not result.success:
        raise RuntimeError('Validation risk calibration failed')
    return float(np.exp(result.x))


def assess(samples, predictions, *, risk_temperature=1.):
    squared, raw_squared, signal_squared = [], [], []
    probabilities, events, regrets, scenes = [], [], [], []
    for sample, p in zip(samples, predictions):
        o = sample['observation']
        service, failure = charge(p, sample['plans'], o, temperature=risk_temperature)
        squared.append(((service-sample['target']['service'])**2).mean((0, 2)))
        raw_squared.append(((p['service_raw']-sample['target']['service_raw'])**2).mean((0, 2)))
        signal_squared.append((((p['signal']-sample['target']['signal'])*30)**2).mean((0, 2)))
        probabilities.extend(failure.ravel().tolist()); events.extend(sample['target']['failure'].ravel().tolist())
        for kind in ['throughput', 'tail']:
            contract = Contract(kind, CONFIG['max_moves'], 1.)
            score = values(contract, o, service, failure)
            gold = values(contract, o, sample['target']['service'], sample['target']['failure'])
            regret = float((gold.max()-gold[int(score.argmax())])/max(float(np.ptp(gold)), 1e-6))
            regrets.append(regret); scenes.append(sample['scene'])
    risk = risk_metrics(probabilities, events)
    return {'service_rmse_by_offset': np.sqrt(np.mean(squared, 0)).tolist(),
            'raw_service_rmse_by_offset': np.sqrt(np.mean(raw_squared, 0)).tolist(),
            'signal_rmse_db_by_offset': np.sqrt(np.mean(signal_squared, 0)).tolist(),
            'risk': risk, 'candidate_regret': cluster_interval(regrets, scenes),
            'samples': len(samples), 'environment_count': len(set(scenes))}


def select_residual(model, validation):
    bases = [model_prediction(None, sample) for sample in validation]
    learned = [model_prediction(model, sample) for sample in validation] if model else bases
    choices = []
    for strength in ([0., .25, .5, 1.] if model else [0.]):
        predictions = [blend(base, p, strength) for base, p in zip(bases, learned)]
        logits = np.concatenate([p['failure_logits'].ravel() for p in predictions])
        labels = np.concatenate([sample['target']['failure_raw'].ravel() for sample in validation])
        t = temperature(logits, labels)
        report = assess(validation, predictions, risk_temperature=t)
        # Predeclared selection score; all terms come from validation scenes.
        score = report['candidate_regret']['mean']+.05*np.mean(report['service_rmse_by_offset'])+.05*report['risk']['brier']
        choices.append({'strength': strength, 'temperature': t, 'selection_score': float(score), 'validation': report})
    selected = min(choices, key=lambda x: x['selection_score'])
    return {'selected_strength': selected['strength'], 'temperature': selected['temperature'],
            'selection': choices, 'fit_on': 'validation_environments_only',
            'physical_safety_guarantee': False}
