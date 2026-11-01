"""Repaired admission coverage and genuine shared-vs-separate task training."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .. import traffic as T
from ..teachers.admission import PRICE_LEVELS, PRIORITY, admitted_value
from ..twin.radio import Flow
from ..twin.warehouse import make_warehouse
from ..fastplan import BatchEvaluator
from .experiment import atomic_json, make_outcomes
from .model import mlp, tensor_inputs, listwise_loss
from .studies import core_root, load_model
from .data import build_dataset


def admission_dataset(seeds):
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        scene = make_warehouse(seed, n_robots=12)
        ap = scene.aps[0]
        cars = []
        for c in scene.cars:
            kind = str(rng.choice(['telemetry', 'map_upload', 'video_uplink']))
            demand = float(rng.uniform(.2, 15))
            cars.append(replace(c, x=float(np.clip(ap.x+rng.normal(0, 12), 0, 120)),
                                y=float(np.clip(ap.y+rng.normal(0, 12), 0, 80)),
                                flows=(Flow(kind, demand, False, 0.),), current_ap=ap.ap_id))
        scene = replace(scene, cars=cars, aps=[ap], mobility=None, _radio=None, _radio_key=None)
        ev = BatchEvaluator(scene, [])
        terms = ev._snapshot_terms(ev.current[None], ev.snaps[0], False, 5.)
        sinr = {c.car_id: float(terms['sinr'][0, i]) for i, c in enumerate(cars)}
        association = {c.car_id: ap.ap_id for c in cars}
        capacity = float(rng.uniform(1, 100))
        values = np.array([admitted_value(scene, association, sinr, {ap.ap_id}, p, capacity)-.05*p for p in PRICE_LEVELS])
        flows = np.array([[PRIORITY[c.flows[0].name]/5, np.log1p(c.demand_mbps)/3,
                           max(sinr[c.car_id], 0)/40,
                           PRIORITY[c.flows[0].name]*max(sinr[c.car_id], 0)/60] for c in cars])
        summary = np.r_[flows.mean(0), flows.std(0), flows.min(0), flows.max(0)]
        for level, price in enumerate(PRICE_LEVELS):
            # HOLD must be an actual candidate. Old E3 omitted it from the
            # local candidate comparison and had nearly homogeneous priorities.
            candidates = [(1, level), (0, level-1), (2, level+1)]
            candidates = [(label, j) for label, j in candidates if 0 <= j < len(values)]
            label, best = max(candidates, key=lambda pair: values[pair[1]])
            x = np.r_[summary, capacity/100., price/6.].astype('float32')
            rows.append({'x': x, 'label': label, 'scene': seed,
                         'oracle_best_neighbour': best, 'current_level': level})
    return rows


class SharedTasks(nn.Module):
    def __init__(self, *, shared=True):
        super().__init__()
        self.shared = shared
        self.admission_input = mlp(18, 64, 64)
        self.series = nn.GRU(1, 64, batch_first=True)
        self.series_input = nn.Linear(66, 64)
        self.radio_input = mlp(96, 64, 64)
        self.cores = nn.ModuleDict({key: mlp(64, 64, 64) for key in (['shared'] if shared else ['admission', 'traffic', 'radio'])})
        self.admission_head = nn.Linear(64, 3)
        self.traffic_head = nn.ModuleDict({k: nn.Linear(64, n) for k, n in [('trend', 3), ('event', 4), ('volatility', 3)]})
        self.forecast_head = nn.Linear(64, T.HORIZON)
        self.radio_head = nn.Linear(64, 1)

    def core(self, x, task):
        return self.cores['shared' if self.shared else task](x)

    def admission(self, x):
        return self.admission_head(self.core(self.admission_input(x), 'admission'))

    def traffic(self, history):
        mean = history.mean(1, keepdim=True)
        std = history.std(1, keepdim=True).clamp(min=1.)
        _, hidden = self.series(((history-mean)/std)[..., None])
        encoded = self.core(self.series_input(torch.cat([hidden[-1], mean/30., std/30.], -1)), 'traffic')
        return {k: head(encoded) for k, head in self.traffic_head.items()}, self.forecast_head(encoded)*std+mean

    def radio(self, pooled):
        return self.radio_head(self.core(self.radio_input(pooled), 'radio')).squeeze(-1)


def balanced_accuracy(y, pred):
    present = sorted(set(y.tolist()))
    return float(np.mean([np.mean(pred[y == c] == c) for c in present]))


def evaluate(model, admission, traffic, radio):
    with torch.inference_mode():
        y = np.array([s['label'] for s in admission])
        p = model.admission(torch.tensor(np.stack([s['x'] for s in admission]))).argmax(1).numpy()
        history = torch.tensor(np.stack([w.history for w in traffic]), dtype=torch.float32)
        labels, forecast = model.traffic(history)
        traffic_metrics = {key: float(np.mean(logits.argmax(1).numpy() == [list(T.QUESTIONS[key]['criteria']).index(getattr(w, key)) for w in traffic]))
                           for key, logits in labels.items()}
        future = torch.tensor(np.stack([w.future for w in traffic]), dtype=torch.float32)
        regret = []
        for pooled, target in radio:
            score = model.radio(pooled).numpy()
            target = target.numpy()
            regret.append(float((target.max()-target[int(score.argmax())])/max(float(np.ptp(target)), 1e-8)))
        return {'admission_accuracy': float(np.mean(p == y)), 'admission_balanced_accuracy': balanced_accuracy(y, p),
                'admission_per_label_recall': {str(c): float(np.mean(p[y == c] == c)) for c in sorted(set(y.tolist()))},
                'traffic_accuracy': traffic_metrics, 'forecast_mae_mbps': float((forecast-future).abs().mean()),
                'association_throughput_norm_regret': float(np.mean(regret))}


def study(root, seeds):
    adm = {split: admission_dataset(range(start, start+count)) for split, start, count in
           [('train', 110000, 120), ('validation', 111000, 30), ('test', 112000, 60)]}
    distributions = {split: dict(Counter(['lower', 'hold', 'raise'][r['label']] for r in rows)) for split, rows in adm.items()}
    if any(len(counts) != 3 for counts in distributions.values()):
        raise RuntimeError(f'Admission protocol still lacks label coverage: {distributions}')
    windows = {split: [T.synth_window(np.random.default_rng(start+i)) for i in range(count)] for split, start, count in
               [('train', 120000, 300), ('validation', 121000, 60), ('test', 122000, 120)]}
    ds, _ = build_dataset(core_root(root)/'cache'/'observations')
    test_windows = windows['test']
    rule = {key: float(np.mean([T.rule_labels(w)[key]['choice'] == getattr(w, key) for w in test_windows])) for key in T.QUESTIONS}
    majority = Counter(r['label'] for r in adm['train']).most_common(1)[0][0]
    baseline = {'admission_majority_accuracy': float(np.mean([r['label'] == majority for r in adm['test']])),
                'admission_exact_numeric_oracle_accuracy': 1., 'traffic_rule_accuracy': rule,
                'forecast_last_value_mae_mbps': float(np.mean([np.abs(w.future-w.history[-1]).mean() for w in test_windows]))}
    # Actual pinned Chronos weights and actual forward calls; failures propagate.
    print('Loading and running actual Chronos-Bolt small baseline', flush=True)
    start = time.perf_counter()
    forecaster = T.Forecaster(device='cpu')
    quantiles, forecast = forecaster.forecast([w.history for w in test_windows])
    baseline['chronos'] = {'repository': T.CHRONOS_REPO, 'revision': T.CHRONOS_REVISION,
                          'model_executed': True, 'elapsed_s': time.perf_counter()-start,
                          'forecast_mae_mbps': float(np.abs(forecast-np.stack([w.future for w in test_windows])).mean()),
                          'quantile_80_empirical_coverage': float(np.mean((np.stack([w.future for w in test_windows]) >= quantiles[..., 0]) &
                                                                        (np.stack([w.future for w in test_windows]) <= quantiles[..., 2])))}
    rows = []
    for seed in seeds:
        graph, _ = load_model(root, seed)
        radio = {}
        # These frozen features feed a trainable head. no_grad creates normal
        # tensors; inference_mode tensors cannot be saved by autograd later.
        with torch.no_grad():
            for split in ['train', 'validation', 'test']:
                radio[split] = [(graph(tensor_inputs(s), torch.tensor(s['plans']))['pooled'].detach(),
                                 torch.tensor(make_outcomes(s).U, dtype=torch.float32)) for s in ds[split]]
        for shared in [False, True]:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = SharedTasks(shared=shared)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            best, chosen = float('inf'), None
            history = []
            for epoch in range(30):
                model.train()
                for _ in range(30):
                    aa = [adm['train'][i] for i in rng.integers(len(adm['train']), size=64)]
                    ww = [windows['train'][i] for i in rng.integers(len(windows['train']), size=32)]
                    loss = F.cross_entropy(model.admission(torch.tensor(np.stack([a['x'] for a in aa]))), torch.tensor([a['label'] for a in aa]))
                    labels, predicted = model.traffic(torch.tensor(np.stack([w.history for w in ww]), dtype=torch.float32))
                    for key, logits in labels.items():
                        target = torch.tensor([list(T.QUESTIONS[key]['criteria']).index(getattr(w, key)) for w in ww])
                        loss = loss+F.cross_entropy(logits, target)
                    loss = loss+.01*F.mse_loss(predicted, torch.tensor(np.stack([w.future for w in ww]), dtype=torch.float32))
                    pooled, target = radio['train'][int(rng.integers(len(radio['train'])))]
                    loss = loss+listwise_loss(model.radio(pooled), target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                    optimizer.step()
                model.eval()
                validation = evaluate(model, adm['validation'], windows['validation'], radio['validation'])
                objective = (1-validation['admission_balanced_accuracy'] +
                             sum(1-x for x in validation['traffic_accuracy'].values())+
                             .01*validation['forecast_mae_mbps']+validation['association_throughput_norm_regret'])
                history.append({'epoch': epoch+1, 'validation': validation})
                if objective < best:
                    best, chosen = objective, {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(chosen)
            path = root/'checkpoints'/f'multitask-{seed}-{shared}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'shared': shared, 'state': model.state_dict(), 'seed': seed}, path)
            record = {'seed': seed, 'shared_core': shared,
                      'parameters': sum(p.numel() for p in model.parameters()), 'history': history,
                      'test': evaluate(model, adm['test'], windows['test'], radio['test'])}
            rows.append(record)
            print(f'Multitask seed={seed} shared={shared}: {record["test"]}', flush=True)
            atomic_json(root/'multitask-progress.json', {'records': rows, 'admission_label_counts': distributions})
    return {'source': 'simulated_multitask_training_and_real_pretrained_forecasting',
            'admission_label_counts': distributions, 'baselines': baseline, 'records': rows,
            'admission_protocol': 'heterogeneous best-effort priorities; all 13 initial levels; hold included; scene-disjoint splits',
            'limits': ['Counterfactual price-level coverage is not deployment prevalence',
                       'Traffic and admission inputs remain synthetic; Chronos inference uses real pinned weights',
                       'Shared/separate cores differ in parameter count, which is explicitly reported',
                       'Association head selects fleet throughput among four candidates, not arbitrary language missions']}
