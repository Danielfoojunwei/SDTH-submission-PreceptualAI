"""Real training and held-out evaluation for the warehouse v2 campaign."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
import torch

from ..policy import MODEL_REPO, MODEL_REVISION
from ..radiojev.nonlinear import Outcomes, NL_MISSIONS, NOVEL_NL
from .missions import parse_mission, evaluate_mission, choose_candidate, StructuredMissionReader
from .metrics import cluster_interval, risk_metrics
from .model import TemporalRadioModel, tensor_inputs, dynamics_loss, listwise_loss


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mission_texts(split):
    if split == 'novel':
        return [text for _, texts in NOVEL_NL.values() for text in texts]
    index = {'train': 0, 'validation': 1, 'test': 2}[split]
    return [texts[index] for _, texts in NL_MISSIONS.values()]


def semantic_contexts(root: Path):
    """Run pinned d3-lite; cache actual hidden states and actual typed answers."""
    cache = root/'cache'/'mission_embeddings.npz'
    receipt_path = root/'semantic_reader.json'
    texts = list(dict.fromkeys(sum([mission_texts(s) for s in ['train', 'validation', 'test', 'novel']], [])))
    if cache.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['model'] != {'repo': MODEL_REPO, 'revision': MODEL_REVISION} or receipt['cache_sha256'] != sha256(cache):
            raise RuntimeError('Semantic cache provenance mismatch')
        with np.load(cache, allow_pickle=False) as loaded:
            if loaded['texts'].tolist() != texts:
                raise RuntimeError('Semantic cache text mismatch')
            return {text: torch.tensor(row) for text, row in zip(texts, loaded['embeddings'])}
    from ..policy import WirelessJEV
    from ..radiojev.trainable import TrainableD3
    engine = WirelessJEV(device='cpu')
    encoder = TrainableD3(engine, lora_layers=0)
    reader = StructuredMissionReader(engine, semantics='historical_e2')
    embeddings, readings = [], []
    for text in texts:
        start = time.perf_counter()
        with torch.no_grad():
            hidden = encoder.encode('Warehouse operator mission:\n'+text,
                                    ['Does this mission describe a warehouse connectivity objective?'])
        embeddings.append(hidden[0].cpu().numpy())
        item = {'text': text, 'embedding_ms': (time.perf_counter()-start)*1000}
        try:
            spec, receipt = reader.read(text)
            item.update(spec=spec.to_dict(), receipt=receipt,
                        matches_controlled_parser=spec == parse_mission(text, semantics='historical_e2'))
        except (ValueError, RuntimeError) as error:
            item.update(error=str(error), matches_controlled_parser=False)
        readings.append(item)
        print(f'd3 semantic context {len(readings)}/{len(texts)} match={item["matches_controlled_parser"]}', flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, texts=np.array(texts), embeddings=np.stack(embeddings))
    atomic_json(receipt_path, {'model': {'repo': MODEL_REPO, 'revision': MODEL_REVISION},
                              'model_executed': True, 'device': 'cpu', 'source': 'real_pretrained_inference',
                              'load_s': engine.load_s, 'readings': readings,
                              'cache_sha256': sha256(cache),
                              'family_accuracy': np.mean([r['matches_controlled_parser'] for r in readings]).item(),
                              'limits': ['Family labels compare against the declared controlled-language parser',
                                         'Answer probabilities are not calibrated physical-risk probabilities']})
    return {text: torch.tensor(row) for text, row in zip(texts, embeddings)}


def make_outcomes(sample, service=None, failures=None):
    target = sample['targets']
    sat = target[..., 0] if service is None else np.clip(service, 0, 1)
    fail = target[..., 1] if failures is None else np.asarray(failures)
    demand = sample['demand']
    U = (sat*demand[None]).sum(1)/max(demand.sum(), 1e-9)
    jain = sat.sum(1)**2 / (sat.shape[1]*np.maximum((sat**2).sum(1), 1e-12))
    return Outcomes(sat[:, None, :], U, jain, fail.sum(1),
                    (sample['plans'] != sample['inputs']['current'][None]).sum(1),
                    sample['inputs']['robot'][:, 0]*120., sample['inputs']['robot'][:, 1]*80., sample['video'])


def calibrate(model, samples):
    residuals, logits, events = [], [], []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            pred = model(tensor_inputs(sample), torch.tensor(sample['plans']))
            residuals.extend(((pred['service']-torch.tensor(sample['targets'][..., 0])).abs()/pred['log_std'].exp()).flatten().tolist())
            logits.extend(pred['failure_logit'].flatten().tolist())
            events.extend(sample['targets'][..., 1].flatten().tolist())
    logits, events = np.array(logits), np.array(events)
    def objective(log_temperature):
        l = logits/np.exp(log_temperature)
        return np.mean(np.logaddexp(0, l)-events*l)
    fitted = minimize_scalar(objective, bounds=(-3., 3.), method='bounded')
    if not fitted.success:
        raise RuntimeError('Risk temperature fit failed')
    return {'temperature': float(np.exp(fitted.x)),
            'service_abs_standardized_q90': float(np.quantile(residuals, .9)),
            'fitted_on': 'validation_environments_only', 'coverage_guarantee': False}


def fit_model(samples, validation, embeddings, root, *, seed, arm, epochs=20):
    path = root/'checkpoints'/f'{arm}-{seed}.pt'
    meta_path = root/f'train-{arm}-{seed}.json'
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['epochs_requested'] != epochs or meta['checkpoint_sha256'] != sha256(path):
            raise RuntimeError('Checkpoint configuration/provenance mismatch')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = TemporalRadioModel(**checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        return model, meta
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TemporalRadioModel(graph=arm != 'instant_mlp', temporal=arm != 'instant_mlp')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history, best, best_state = [], float('inf'), None
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        losses = []
        for i in rng.permutation(len(samples)):
            sample = samples[i]
            text = mission_texts('train')[int(rng.integers(6))]
            inputs = tensor_inputs(sample)
            pred = model(inputs, torch.tensor(sample['plans']), embeddings[text] if arm == 'semantic_graph' else None)
            loss = dynamics_loss(pred, torch.tensor(sample['targets']))
            if arm == 'semantic_graph':
                utility, _ = evaluate_mission(parse_mission(text, semantics='historical_e2'), make_outcomes(sample))
                loss = loss + .2*listwise_loss(pred['value'], torch.tensor(utility, dtype=torch.float32))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for sample in validation:
                pred = model(tensor_inputs(sample), torch.tensor(sample['plans']))
                validation_losses.append(float(dynamics_loss(pred, torch.tensor(sample['targets']))))
        score = float(np.mean(validation_losses))
        history.append({'epoch': epoch+1, 'train_loss': float(np.mean(losses)), 'validation_loss': score})
        if score < best:
            best = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f'{arm} seed={seed} epoch={epoch+1}/{epochs} validation={score:.5f}', flush=True)
        atomic_json(root/'progress.json', {'stage': 'training', 'arm': arm, 'seed': seed,
                                          'epoch': epoch+1, 'epochs': epochs, 'elapsed_s': time.perf_counter()-started})
    model.load_state_dict(best_state)
    calibration = calibrate(model, validation)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'config': model.config, 'model': model.state_dict(), 'seed': seed,
                'calibration': calibration}, path)
    meta = {'source': 'simulation_training', 'arm': arm, 'seed': seed, 'epochs_requested': epochs,
            'history': history, 'train_samples': len(samples), 'validation_samples': len(validation),
            'parameters': sum(p.numel() for p in model.parameters()),
            'elapsed_s': time.perf_counter()-started, 'checkpoint_sha256': sha256(path),
            'calibration': calibration}
    atomic_json(meta_path, meta)
    return model, meta


def evaluate_model(model, samples, embeddings, *, split, arm, calibration):
    model.eval()
    errors, signal_errors, probabilities, events = [], [], [], []
    records, forward_times = [], []
    semantic = arm == 'semantic_graph'
    texts = mission_texts('test')+mission_texts('novel')
    with torch.inference_mode():
        for sample in samples:
            started = time.perf_counter()
            inputs = tensor_inputs(sample)
            pred = model(inputs, torch.tensor(sample['plans']))
            forward_times.append((time.perf_counter()-started)*1000)
            sat = pred['service'].numpy()
            failure_p = expit(pred['failure_logit'].numpy()/calibration['temperature'])
            errors.extend((sat-sample['targets'][..., 0]).flatten().tolist())
            signal_errors.extend(((pred['signal'].numpy()-sample['targets'][..., 2])*30.).flatten().tolist())
            probabilities.extend(failure_p.flatten().tolist())
            events.extend(sample['targets'][..., 1].flatten().tolist())
            predicted = make_outcomes(sample, sat, failure_p)
            actual = make_outcomes(sample)
            for text in texts:
                t0 = time.perf_counter()
                spec = parse_mission(text, semantics='historical_e2')
                gold, _ = evaluate_mission(spec, actual)
                if semantic:
                    value = model.value(torch.cat([pred['pooled'], model.text_projection(embeddings[text]).expand(len(sat), -1)], -1)).squeeze(-1).numpy()
                else:
                    value, _ = evaluate_mission(spec, predicted)
                selected = int(value.argmax())
                span = max(float(np.ptp(gold)), 1e-8)
                regret = float((gold.max()-gold[selected])/span)
                records.append({'scene': sample['scene'], 'tick': sample['tick'], 'text': text,
                                'family': spec.kind, 'new_mission': text in mission_texts('novel'),
                                'choice': selected, 'regret': regret,
                                'service': actual.sat[selected, 0].tolist(),
                                'critical_failure': sample['targets'][selected, :, 1].tolist(),
                                'moved': (sample['plans'][selected] != sample['inputs']['current']).astype(int).tolist(),
                                'selection_ms': (time.perf_counter()-t0)*1000})
    return {'source': 'offline_simulated_counterfactuals', 'split': split, 'arm': arm,
            'service_rmse': float(np.sqrt(np.mean(np.square(errors)))),
            'signal_rmse_db': float(np.sqrt(np.mean(np.square(signal_errors)))),
            'risk': risk_metrics(probabilities, events),
            'regret': cluster_interval([r['regret'] for r in records], [r['scene'] for r in records]),
            'by_family': {family: cluster_interval([r['regret'] for r in records if r['family'] == family],
                                                  [r['scene'] for r in records if r['family'] == family])
                          for family in sorted({r['family'] for r in records})},
            'forward_ms_mean': float(np.mean(forward_times)), 'forward_ms_p95': float(np.quantile(forward_times, .95)),
            'timing_includes': ['tensor_conversion', 'observation_graph_forward'],
            'timing_excludes': ['sensor_collection', 'candidate_generation', 'cached_mission_embedding_first_use', 'actuation'],
            'records': records,
            'limits': ['Counterfactual choices evaluated on logged behaviour states, not a closed-loop policy rollout',
                       'Intervals cluster scenes within one trained model, not independent training seeds',
                       'Candidate-set regret is not global optimisation regret']}
