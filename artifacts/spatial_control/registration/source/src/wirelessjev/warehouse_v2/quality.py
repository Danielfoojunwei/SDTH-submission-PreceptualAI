"""Validation-only routing, simple baselines, risk guards and real CPU int8."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import time

import numpy as np
import torch

from .data import observed_candidates
from .experiment import make_outcomes, mission_texts, atomic_json
from .missions import parse_mission, evaluate_mission, choose_candidate
from .model import tensor_inputs
from .metrics import cluster_interval
from .studies import load_model


ARMS = ['instant_mlp', 'temporal_graph', 'semantic_graph']


def predictions(models, sample, embeddings, texts):
    result = {}
    actual = make_outcomes(sample)
    with torch.inference_mode():
        for arm, model in models.items():
            output = model(tensor_inputs(sample), torch.tensor(sample['plans']))
            physical = make_outcomes(sample, output['service'].numpy(), output['failure_logit'].sigmoid().numpy())
            result[arm] = {}
            for text in texts:
                spec = parse_mission(text, semantics='historical_e2')
                gold, _ = evaluate_mission(spec, actual)
                if arm == 'semantic_graph':
                    text_feature = model.text_projection(embeddings[text]).expand(len(sample['plans']), -1)
                    value = model.value(torch.cat([output['pooled'], text_feature], -1)).squeeze(-1).numpy()
                else:
                    value, _ = evaluate_mission(spec, physical)
                choice = int(value.argmax())
                result[arm][text] = {'choice': choice, 'regret': float((gold.max()-gold[choice])/max(float(np.ptp(gold)), 1e-8))}
    return result


def pipeline_latency(model, samples):
    """Measure full local computation on already-available sensor histories."""
    times, statuses = [], defaultdict(int)
    with torch.inference_mode():
        for sample in samples:
            for text in mission_texts('test'):
                start = time.perf_counter()
                spec = parse_mission(text, semantics='hard_v2')
                plans = observed_candidates(sample['inputs']['history'], sample['inputs']['current'],
                                             sample['demand'], np.random.default_rng(sample['scene']+sample['tick']))
                prediction = model(tensor_inputs(sample), torch.tensor(plans))
                proposed = dict(sample, plans=plans)
                # Risk threshold is declared, not fitted to test outcomes.
                o = make_outcomes(proposed, prediction['service'].numpy(),
                                  (prediction['failure_logit'].sigmoid().numpy() > .1).astype(float))
                selection = choose_candidate(spec, o)
                if selection['choice'] is not None:
                    A = plans[selection['choice']]
                    moved = A != sample['inputs']['current']
                    heard = sample['inputs']['history'][-1, np.arange(len(A)), A, 1] > 0
                    if np.any(moved & ~heard):
                        selection['status'] = 'deferred_unobserved_target'
                statuses[selection['status']] += 1
                times.append((time.perf_counter()-start)*1000)
    return {'mean_ms': float(np.mean(times)), 'p95_ms': float(np.quantile(times, .95)),
            'p99_ms': float(np.quantile(times, .99)), 'decisions': len(times), 'statuses': dict(statuses),
            'includes': ['literal_mission_parse', 'candidate_generation', 'tensor_conversion',
                         'radio_prediction', 'hard_constraint_candidate_selection', 'target_visibility_guard'],
            'excludes': ['physical_sensor_collection', 'first_use_d3_mission_interpretation', 'network_transport', 'actuation'],
            'risk_threshold': .1, 'risk_threshold_is_safety_guarantee': False}


def study(root, data, embeddings, seeds):
    rows = []
    for seed in seeds:
        models = {arm: load_model(root, seed, arm)[0] for arm in ARMS}
        validation = defaultdict(lambda: defaultdict(list))
        for sample in data['validation']:
            guesses = predictions(models, sample, embeddings, mission_texts('validation'))
            for arm in ARMS:
                for text, item in guesses[arm].items():
                    validation[parse_mission(text).kind][arm].append(item['regret'])
        routes = {kind: min(ARMS, key=lambda arm: np.mean(scores[arm])) for kind, scores in validation.items()}
        records = []
        for split in ['test', 'shift', 'fleet200']:
            for sample in data[split]:
                texts = mission_texts('test')+mission_texts('novel')
                guessed = predictions(models, sample, embeddings, texts)
                actual = make_outcomes(sample)
                for text in texts:
                    spec = parse_mission(text, semantics='historical_e2')
                    gold, _ = evaluate_mission(spec, actual)
                    values = {arm: guessed[arm][text] for arm in ARMS}
                    for name, choice in [('stay', 0), ('strongest_scan', 1), ('observed_load_balance', 2), ('random_sparse_switch', 3)]:
                        values[name] = {'choice': choice, 'regret': float((gold.max()-gold[choice])/max(float(np.ptp(gold)), 1e-8))}
                    values['validation_family_router'] = dict(values[routes[spec.kind]])
                    # Evaluation ceiling only; it is never used by the router.
                    values['perfect_branch_ceiling'] = min([values[a] for a in ARMS], key=lambda v: v['regret'])
                    for method, value in values.items():
                        choice = value['choice']
                        records.append({'seed': seed, 'scene': sample['scene'], 'tick': sample['tick'],
                                        'split': split, 'family': spec.kind, 'text': text, 'method': method,
                                        'regret': value['regret'], 'mean_service': float(actual.sat[choice].mean()),
                                        'worst_robot_service': float(actual.sat[choice].min()),
                                        'worst_decile_service': float(np.sort(actual.sat[choice, 0])[:max(1, len(sample['demand'])//10)].mean()),
                                        'critical_failure_count': float(sample['targets'][choice, :, 1].sum()),
                                        'moves': int(actual.moves[choice])})
        latency = {arm: pipeline_latency(model, data['test']) for arm, model in models.items() if arm != 'semantic_graph'}
        # Actual quantized operators, not a simulated bit-width annotation.
        quantized = torch.ao.quantization.quantize_dynamic(deepcopy(models['temporal_graph']), {torch.nn.Linear, torch.nn.GRU}, dtype=torch.qint8)
        quantized.eval()
        quantized_latency = pipeline_latency(quantized, data['test'])
        with torch.inference_mode():
            quantized_error = []
            for sample in data['test']:
                output = quantized(tensor_inputs(sample), torch.tensor(sample['plans']))
                quantized_error.extend((output['service'].numpy()-sample['targets'][..., 0]).ravel().tolist())
        record_path = root/'cache'/f'quality-{seed}.json'
        atomic_json(record_path, records)
        summaries = {}
        for split in ['test', 'shift', 'fleet200']:
            summaries[split] = {}
            for method in sorted({r['method'] for r in records}):
                subset = [r for r in records if r['split'] == split and r['method'] == method]
                summaries[split][method] = {'regret': cluster_interval([r['regret'] for r in subset], [r['scene'] for r in subset]),
                                           'mean_service': float(np.mean([r['mean_service'] for r in subset])),
                                           'worst_decile_service': float(np.mean([r['worst_decile_service'] for r in subset])),
                                           'critical_failure_count': float(np.mean([r['critical_failure_count'] for r in subset]))}
        rows.append({'seed': seed, 'routes_fitted_on_validation_only': routes, 'summary': summaries,
                     'pipeline_latency': latency, 'cpu_int8': {'operators_executed': True, 'latency': quantized_latency,
                          'service_rmse': float(np.sqrt(np.mean(np.square(quantized_error))))}, 'per_case_records': str(record_path)})
        print(f'Quality/routing/int8 seed={seed} completed', flush=True)
    return {'source': 'simulation_and_actual_cpu_quantized_inference', 'records': rows,
            'limits': ['Validation routing uses known mission families, with new thresholds/zones only at test time',
                       'Perfect branch choice is an unattainable evaluation ceiling',
                       'Int8 quality is measured; reduced bit width is not assumed to make this CPU faster',
                       'Hard constraints operate on predictions and can defer; they are not physical safety guarantees']}
