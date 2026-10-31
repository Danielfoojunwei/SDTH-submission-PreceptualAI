"""Cross-site learning curves and one-round adapter federation/removal.

Removal here has a narrow, checkable target: recompute a weighted average of
independent site adapters from the immutable pre-site model. It is not a
claim about arbitrary iterative FL or physical erasure of all data copies.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import time

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
import torch

from .data import episode
from .experiment import atomic_json, make_outcomes, mission_texts
from .missions import parse_mission, evaluate_mission
from .model import tensor_inputs, dynamics_loss
from .studies import load_model


def site_data(site, *, examples=100, validation=27, test=54):
    def gather(offset, count):
        collected = []
        for i in range((count+8)//9):
            collected.extend(episode(960_000+site*1000+offset+i, n=100, domain='shift',
                                     environment_seed=95000+site))
        return collected[:count]
    return {'train': gather(0, examples), 'validation': gather(200, validation), 'test': gather(400, test)}


def score(model, samples, *, details=False):
    errors, regrets, losses = [], [], []
    risk_errors, scene_rows = [], []
    with torch.inference_mode():
        for sample in samples:
            pred = model(tensor_inputs(sample), torch.tensor(sample['plans']))
            sat = pred['service'].numpy()
            residual = sat-sample['targets'][..., 0]
            errors.extend(residual.ravel().tolist())
            p = pred['failure_logit'].sigmoid().numpy()
            risk_errors.extend(((p-sample['targets'][..., 1])**2).ravel().tolist())
            losses.append(float(dynamics_loss(pred, torch.tensor(sample['targets']))))
            guessed = make_outcomes(sample, sat, p)
            truth = make_outcomes(sample)
            case_regrets = []
            for text in mission_texts('test')+mission_texts('novel'):
                spec = parse_mission(text, semantics='historical_e2')
                choices, _ = evaluate_mission(spec, guessed)
                gold, _ = evaluate_mission(spec, truth)
                case_regrets.append(float((gold.max()-gold[int(choices.argmax())])/max(float(np.ptp(gold)), 1e-8)))
            regrets.extend(case_regrets)
            scene_rows.append({'scene': sample['scene'], 'tick': sample['tick'], 'mean_regret': float(np.mean(case_regrets))})
    result = {'service_rmse': float(np.sqrt(np.mean(np.square(errors)))),
              'brier': float(np.mean(risk_errors)), 'mean_regret': float(np.mean(regrets)),
              'dynamics_loss': float(np.mean(losses)), 'samples': len(samples)}
    if details:
        result['per_case'] = scene_rows
    return result


def adapt(base, train, validation, *, seed, epochs=10):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = deepcopy(base)
    # Independent, removable output/message adapters. The shared observation
    # representation remains fixed; no site information is written into it.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(('output.', 'message.')))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    best = float('inf')
    chosen = None
    history = []
    for epoch in range(epochs):
        model.train()
        for i in rng.permutation(len(train)):
            sample = train[i]
            prediction = model(tensor_inputs(sample), torch.tensor(sample['plans']))
            loss = dynamics_loss(prediction, torch.tensor(sample['targets']))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            losses = [float(dynamics_loss(model(tensor_inputs(s), torch.tensor(s['plans'])), torch.tensor(s['targets'])))
                      for s in validation]
        value = float(np.mean(losses))
        history.append({'epoch': epoch+1, 'validation_loss': value})
        if value < best:
            best, chosen = value, deepcopy(model.state_dict())
    model.load_state_dict(chosen)
    model.eval()
    return model, history


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def merge_adapters(base, models, weights=None):
    if not models:
        raise ValueError('At least one retained adapter is required')
    weights = np.ones(len(models)) if weights is None else np.asarray(weights, float)
    if len(weights) != len(models) or (weights <= 0).any() or not np.isfinite(weights).all():
        raise ValueError('Positive finite, matched weights are required')
    weights = weights/weights.sum()
    merged = deepcopy(base)
    state = deepcopy(base.state_dict())
    for name in state:
        if name.startswith(('output.', 'message.')):
            state[name] = sum(float(w)*m.state_dict()[name] for w, m in zip(weights, models))
    merged.load_state_dict(state)
    merged.eval()
    return merged


def auc(member, nonmember):
    member, nonmember = np.asarray(member), np.asarray(nonmember)
    ranks = rankdata(np.r_[member, nonmember])
    return float((ranks[:len(member)].sum()-len(member)*(len(member)+1)/2)/(len(member)*len(nonmember)))


def attack_signals(model, samples):
    signals = {'negative_service_mse': [], 'failure_confidence': [], 'negative_signal_mse': []}
    with torch.inference_mode():
        for sample in samples:
            p = model(tensor_inputs(sample), torch.tensor(sample['plans']))
            y = torch.tensor(sample['targets'])
            signals['negative_service_mse'].append(-float((p['service']-y[..., 0]).square().mean()))
            signals['failure_confidence'].append(float((p['failure_logit'].sigmoid()-.5).abs().mean()))
            signals['negative_signal_mse'].append(-float((p['signal']-y[..., 2]).square().mean()))
    return signals


def transfer_study(root, embeddings, seeds):
    site = site_data(0)
    rows = []
    for seed in seeds:
        base, checkpoint = load_model(root, seed)
        for count in [0, 10, 100]:
            start = time.perf_counter()
            if count:
                adapted, history = adapt(base, site['train'][:count], site['validation'], seed=seed+300, epochs=10)
            else:
                adapted, history = base, []
            rows.append({'seed': seed, 'adaptation_examples': count, 'training_elapsed_s': time.perf_counter()-start,
                         'metrics': score(adapted, site['test'], details=True), 'history': history,
                         'model_state_sha256': state_digest(adapted)})
            path = root/'checkpoints'/f'transfer-{seed}-{count}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'config': adapted.config, 'model': adapted.state_dict()}, path)
            print(f'Transfer seed={seed} examples={count} regret={rows[-1]["metrics"]["mean_regret"]:.4f}', flush=True)
    return {'source': 'simulated_new_site_adaptation', 'site_environment_seed': 95000,
            'split': 'different robot/trajectory seeds within one held-out environment', 'records': rows,
            'limits': ['Three initial model seeds, one target site', 'Adaptation examples are correlated within episodes',
                       'No new-site hardware data or foundation-model generality claim']}


def federation_study(root, embeddings, seeds):
    sites = [site_data(i) for i in range(4)]
    rows = []
    for seed in seeds:
        base, _ = load_model(root, seed)
        local, histories = [], []
        for i, site in enumerate(sites):
            adapted, history = adapt(base, site['train'], site['validation'], seed=seed*1000+i, epochs=5)
            local.append(adapted); histories.append(history)
        fedavg = merge_adapters(base, local)
        validation_base = [score(base, site['validation']) for site in sites]
        validation_candidate = [score(fedavg, site['validation']) for site in sites]
        mean_improves = np.mean([r['mean_regret'] for r in validation_candidate]) < np.mean([r['mean_regret'] for r in validation_base])
        no_site_harm = all(c['mean_regret'] <= b['mean_regret'] for b, c in zip(validation_base, validation_candidate))
        accepted = bool(mean_improves and no_site_harm)
        gated = fedavg if accepted else base
        evaluations = {'initial': [score(base, s['test']) for s in sites],
                       'local': [score(m, s['test']) for m, s in zip(local, sites)],
                       'fedavg': [score(fedavg, s['test']) for s in sites],
                       'gated_shared': [score(gated, s['test']) for s in sites]}
        forgotten = 1
        retained = [i for i in range(4) if i != forgotten]
        start = time.perf_counter()
        removed = merge_adapters(base, [local[i] for i in retained])
        removal_s = time.perf_counter()-start
        start = time.perf_counter()
        rebuilt = [adapt(base, sites[i]['train'], sites[i]['validation'], seed=seed*1000+i, epochs=5)[0] for i in retained]
        clean_reference = merge_adapters(base, rebuilt)
        retrain_s = time.perf_counter()-start
        exact = state_digest(removed) == state_digest(clean_reference)
        if not exact:
            raise RuntimeError('Adapter removal differs from the clean counterfactual retrain')
        changed = state_digest(fedavg) != state_digest(removed)
        attacks = {}
        for name, model in [('before_fedavg', fedavg), ('removed', removed), ('clean_retrain', clean_reference), ('initial', base)]:
            members = attack_signals(model, sites[forgotten]['train'])
            nonmembers = attack_signals(model, sites[forgotten]['test'])
            attacks[name] = {key: auc(members[key], nonmembers[key]) for key in members}
        record = {'seed': seed, 'site_environment_seeds': [95000+i for i in range(4)],
                  'independent_adapter_histories': histories, 'validation_base': validation_base,
                  'validation_candidate': validation_candidate, 'shared_gate_accepted': accepted,
                  'test': evaluations, 'forget_site': forgotten,
                  'removal': {'scope': 'one_round_unconditional_fedavg_of_independent_adapters',
                              'accepted_site_weight_before': .25, 'site_contribution_changed_weights': changed,
                              'equals_clean_retrain_weights': exact, 'removed_state_sha256': state_digest(removed),
                              'clean_reference_state_sha256': state_digest(clean_reference),
                              'removal_s': removal_s, 'clean_retrain_s': retrain_s, 'attacks': attacks,
                              'retained_test_before': [score(fedavg, sites[i]['test']) for i in retained],
                              'retained_test_after': [score(removed, sites[i]['test']) for i in retained]}}
        rows.append(record)
        for name, model in [('fedavg', fedavg), ('gated', gated), ('removed', removed), ('clean_retrain', clean_reference)]:
            path = root/'checkpoints'/f'federation-{seed}-{name}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'config': model.config, 'model': model.state_dict()}, path)
        atomic_json(root/'federation-progress.json', {'completed_seeds': [r['seed'] for r in rows], 'records': rows})
        print(f'Federation seed={seed} gate={accepted} exact_counterfactual_removal={exact}', flush=True)
    return {'source': 'simulated_site_adapter_training', 'rounds': 1, 'records': rows,
            'limits': ['Unconditional FedAvg is a research baseline; rejected updates are not released by the gated arm',
                       'The removal equivalence applies only to independent one-round adapters and the shared model weights',
                       'Attack AUCs are diagnostics, not certified deletion; personal states/data/checkpoint copies are retained for this audit',
                       'Initial backbone saw none of the four adaptation sites; no arbitrary iterative federated-unlearning claim']}
