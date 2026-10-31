"""Per-robot evidence, scene-cluster intervals, and calibrated risk metrics."""
from __future__ import annotations

import numpy as np


def robot_metrics(service, violations, moves, *, interval_s=5., robot_ids=None):
    service, violations, moves = map(np.asarray, (service, violations, moves))
    if service.ndim != 2 or service.shape != violations.shape or service.shape != moves.shape:
        raise ValueError('Expected matching [time, robot] arrays')
    if not np.isfinite(service).all() or np.any((service < 0) | (service > 1)):
        raise ValueError('Service is a finite fraction of demand')
    n = service.shape[1]
    means = service.mean(0)
    longest = np.zeros(n, dtype=int)
    run = np.zeros(n, dtype=int)
    for row in violations.astype(bool):
        run = np.where(row, run + 1, 0)
        longest = np.maximum(longest, run)
    ids = robot_ids if robot_ids is not None else [str(i) for i in range(n)]
    if len(ids) != n or len(set(ids)) != n:
        raise ValueError('Robot IDs must be unique and match the data')
    return {'mean_service': float(means.mean()),
            'worst_decile_mean_service': float(np.sort(means)[:max(1, int(np.ceil(.1*n)))].mean()),
            'worst_robot_mean_service': float(means.min()),
            'critical_violation_robot_steps': int(violations.sum()),
            'reconnections': int(moves.sum()),
            'per_robot': {str(cid): {'mean_service': float(means[i]),
                         'critical_violation_steps': int(violations[:, i].sum()),
                         'longest_critical_violation_s': float(longest[i]*interval_s),
                         'reconnections': int(moves[:, i].sum())} for i, cid in enumerate(ids)}}


def cluster_interval(values, clusters, *, seed=0, draws=2000):
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    if values.shape != clusters.shape or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite paired values and cluster IDs are required')
    groups = [values[clusters == c] for c in np.unique(clusters)]
    if len(groups) < 2:
        return {'mean': float(values.mean()), 'ci95': None, 'clusters': len(groups)}
    rng = np.random.default_rng(seed)
    means = [np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]).mean()
             for _ in range(draws)]
    return {'mean': float(values.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist(),
            'clusters': len(groups), 'method': 'scene_cluster_percentile_bootstrap'}


def risk_metrics(probability, actual, bins=10):
    p, y = np.asarray(probability, float).ravel(), np.asarray(actual, float).ravel()
    if p.shape != y.shape or not len(p) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Finite matched probabilities and binary outcomes are required')
    if np.any((p < 0) | (p > 1)) or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid probability or event label')
    assignments = np.minimum((p*bins).astype(int), bins-1)
    reliability = [{'n': int((assignments == b).sum()),
                    'predicted': float(p[assignments == b].mean()),
                    'observed': float(y[assignments == b].mean())}
                   for b in range(bins) if (assignments == b).any()]
    return {'brier': float(np.mean((p-y)**2)),
            'log_loss': float(-np.mean(y*np.log(np.clip(p, 1e-7, 1-1e-7)) +
                                       (1-y)*np.log(np.clip(1-p, 1e-7, 1-1e-7)))),
            'ece': float(sum(r['n']*abs(r['predicted']-r['observed']) for r in reliability)/len(p)),
            'reliability': reliability}
