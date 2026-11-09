"""Pure measured-view metrics and train/validation-only linear controls.

Power values are reference-relative dB, not RSSI dBm, SINR, or packet risk.
Representation alignment follows shared cosine-neighbor fractions, with
tie-neutral boundary weights and undefined scores for collapsed coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import math
import numpy as np

POWER_UNIT = 'dB of referenced linear channel power; not dBm'
CELLS = ('joint_none','joint_sigreg','joint_vicreg','temporal_sigreg',
         'regression_sigreg','unstructured_sigreg')


def _array(value, name, *, ndim=None, minimum_rows=1):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu' or np.iscomplexobj(raw):
        raise ValueError(name + ': real numeric values are required')
    result = np.asarray(raw, dtype='f8')
    if ((ndim is not None and result.ndim != ndim) or result.ndim < 1
            or len(result) < minimum_rows or not np.isfinite(result).all()
            or any(size < 1 for size in result.shape[1:])):
        raise ValueError(name + ': finite, nonempty, compatible axes are required')
    return result


def _readonly(value, dtype='f8'):
    value = np.asarray(value, dtype=dtype)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def metrics(prediction, truth):
    prediction, truth = _array(prediction, 'prediction'), _array(truth, 'truth')
    if prediction.shape != truth.shape:
        raise ValueError('Prediction and measured target axes differ')
    difference = prediction - truth
    if not np.isfinite(difference).all():
        raise ValueError('Overflow in prediction errors')
    try:
        with np.errstate(over='raise',invalid='raise'):
            squared = np.square(difference)
    except FloatingPointError as exc:
        raise ValueError('Overflow in squared prediction errors') from exc
    per_row = np.sqrt(squared.reshape(len(difference), -1).mean(1))
    count = max(1, math.ceil(.1 * len(per_row)))
    return {'rmse_db': float(np.sqrt(squared.mean())),
            'mae_db': float(np.abs(difference).mean()),
            'worst_decile_rmse_db': float(np.sort(per_row)[-count:].mean()),
            'worst_decile_definition': 'mean of the largest ceil(0.1*N) per-source-row RMSEs',
            'row_rmse_db': per_row.tolist(), 'samples': len(per_row),
            'target_elements': int(difference.size), 'power_unit': POWER_UNIT}


def _representation_summary(z):
    coordinate_std = z.std(0, ddof=0)
    constant = bool(np.array_equal(z, np.broadcast_to(z[0], z.shape)))
    if constant:
        coordinate_std = np.zeros(z.shape[1])
        rank, nonzero = 0., 0
    else:
        singular = np.linalg.svd(z-z.mean(0), compute_uv=False)
        # Removing numerical null singular values cannot create rank from a
        # constant embedding. Mean coordinate standard deviation stays visible.
        tolerance = singular[0] * max(z.shape) * np.finfo('f8').eps
        energy = np.square(singular[singular > tolerance])
        nonzero = len(energy)
        if not len(energy) or energy.sum() == 0:
            rank = 0.
        else:
            p = energy / energy.sum()
            rank = float(np.exp(-(p * np.log(p)).sum()))
    return {'effective_rank': rank, 'numerical_rank': nonzero,
            'coordinate_std': coordinate_std.tolist(),
            'mean_coordinate_std': float(coordinate_std.mean()), 'constant': constant}


def _neighbor_weights(z, k):
    scale = np.abs(z).max(1)
    if np.any(scale == 0):
        return None
    normalized = z/scale[:,None]
    normalized = normalized/np.linalg.norm(normalized,axis=1)[:,None]
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    boundary = np.partition(similarity, -k, axis=1)[:, -k]
    greater = similarity > boundary[:, None]
    tied = similarity == boundary[:, None]
    missing = k - greater.sum(1)
    weights = greater.astype('f8') + tied * (missing / tied.sum(1))[:, None]
    if not np.allclose(weights.sum(1), k, rtol=0, atol=1e-10):
        raise ArithmeticError('Neighbor tie weights do not conserve k')
    return weights


def representation_metrics(z_a, z_b, k=5):
    a, b = _array(z_a, 'view-A coordinates', ndim=2), _array(z_b, 'view-B coordinates', ndim=2)
    if len(a) != len(b) or type(k) is not int or k < 1:
        raise ValueError('Paired source rows and a positive integral k are required')
    sa, sb = _representation_summary(a), _representation_summary(b)
    effective_k = min(k, len(a)-1)
    reason, alignment = None, None
    if len(a) < 2:
        reason = 'fewer_than_two_source_rows'
    elif sa['constant'] or sb['constant']:
        reason = 'constant_coordinates'
    else:
        wa, wb = _neighbor_weights(a,effective_k), _neighbor_weights(b,effective_k)
        if wa is None or wb is None:
            reason = 'undefined_cosine_for_zero_norm_row'
        else:
            alignment = float((wa * wb).sum(1).mean() / effective_k)
    random = effective_k / (len(a)-1) if len(a) > 1 else None
    return {'samples':len(a), 'k_requested':k, 'k':effective_k,
        'effective_rank_a':sa['effective_rank'], 'effective_rank_b':sb['effective_rank'],
        'mean_coordinate_std_a':sa['mean_coordinate_std'], 'mean_coordinate_std_b':sb['mean_coordinate_std'],
        'view_a':sa, 'view_b':sb, 'mutual_knn_alignment':alignment,
        'alignment_defined':alignment is not None, 'alignment_undefined_reason':reason,
        'random_alignment_expected':random,
        'alignment_minus_random':None if alignment is None else alignment-random,
        'rank_definition':'entropy of centered nonzero singular-value energies',
        'alignment_definition':'shared cosine k-neighbor fraction with neutral fractional boundary ties',
        'collapsed_coordinates_score':None,
        'rank_alone_does_not_prove_useful_RF_information':True}


@dataclass(frozen=True)
class RidgeModel:
    coef: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    alpha: float
    output_shape: tuple[int, ...]
    _metadata: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        coef = _array(self.coef,'ridge coefficients',ndim=2)
        mean = _array(self.x_mean,'training feature mean',ndim=1)
        scale = _array(self.x_scale,'training feature scale',ndim=1)
        target_mean = _array(self.y_mean,'training target mean',ndim=1)
        shape = tuple(self.output_shape)
        if (coef.shape != (len(mean),len(target_mean)) or scale.shape != mean.shape
                or np.any(scale <= 0) or any(type(x) is not int or x < 1 for x in shape)
                or math.prod(shape) != len(target_mean) or isinstance(self.alpha,bool)
                or not math.isfinite(self.alpha) or self.alpha <= 0):
            raise ValueError('Incompatible ridge parameter axes or penalty')
        for name,value in [('coef',coef),('x_mean',mean),('x_scale',scale),('y_mean',target_mean)]:
            object.__setattr__(self,name,_readonly(value))
        object.__setattr__(self,'alpha',float(self.alpha))
        object.__setattr__(self,'output_shape',shape)
        object.__setattr__(self,'_metadata',deepcopy(dict(self._metadata)))

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def predict(self, x):
        x = _array(x,'ridge query',ndim=2)
        if x.shape[1] != len(self.x_mean):
            raise ValueError('Ridge query feature count differs from training')
        predicted = ((x-self.x_mean)/self.x_scale) @ self.coef + self.y_mean
        if not np.isfinite(predicted).all():
            raise ArithmeticError('Nonfinite ridge prediction')
        return predicted.reshape((len(x),)+self.output_shape)

    def as_arrays(self):
        return {name:np.array(getattr(self,name),dtype='i8' if name == 'output_shape' else 'f8',copy=True)
                for name in ('coef','x_mean','x_scale','y_mean','alpha','output_shape')}

    @classmethod
    def from_arrays(cls, arrays, *, metadata=None):
        names = {'coef','x_mean','x_scale','y_mean','alpha','output_shape'}
        if set(arrays) != names:
            raise ValueError('Ridge archive membership differs')
        raw_shape = np.asarray(arrays['output_shape'])
        alpha = np.asarray(arrays['alpha'])
        if raw_shape.ndim != 1 or raw_shape.dtype.kind not in 'iu' or alpha.shape != ():
            raise ValueError('Invalid stored ridge shape/alpha')
        return cls(*(arrays[name] for name in ('coef','x_mean','x_scale','y_mean')),
                   float(alpha),tuple(int(x) for x in raw_shape),metadata or {})


def ridge_fit(train_x, train_y, validation_x, validation_y, alphas=(.1,1.,10.,100.)):
    """Fit on train only; select alpha on validation only, without test access.

    Penalty convention is sum squared errors + alpha*||coef||². Intercept is
    unpenalized. Dual solves keep high-dimensional power-view controls bounded.
    """
    x, vx = _array(train_x,'training features',ndim=2,minimum_rows=2), _array(validation_x,'validation features',ndim=2)
    y, vy = _array(train_y,'training targets'), _array(validation_y,'validation targets')
    if len(x) != len(y) or len(vx) != len(vy) or vx.shape[1] != x.shape[1] or y.shape[1:] != vy.shape[1:]:
        raise ValueError('Ridge train/validation rows, features or target axes differ')
    penalties = tuple(alphas)
    if (not penalties or len(set(penalties)) != len(penalties)
            or any(isinstance(a,(bool,np.bool_)) or not isinstance(a,(int,float,np.number))
                   or not math.isfinite(a) or a <= 0 for a in penalties)):
        raise ValueError('Use distinct finite positive ridge penalties')
    mean = x.mean(0)
    scale = np.maximum(x.std(0,ddof=0),1e-8)
    tx, validation = (x-mean)/scale, (vx-mean)/scale
    target_shape = y.shape[1:]
    flat_y, flat_vy = y.reshape(len(y),-1), vy.reshape(len(vy),-1)
    y_mean = flat_y.mean(0)
    centered = flat_y-y_mean
    primal = x.shape[1] <= len(x)
    gram = tx.T@tx if primal else tx@tx.T
    rhs = tx.T@centered if primal else centered
    eye = np.eye(len(gram))
    candidates, coefficients = [], []
    for alpha in penalties:
        solved = np.linalg.solve(gram+float(alpha)*eye,rhs)
        coef = solved if primal else tx.T@solved
        errors = validation@coef+y_mean-flat_vy
        mse = float(np.square(errors).mean())
        if not np.isfinite(coef).all() or not math.isfinite(mse):
            raise ArithmeticError('Nonfinite ridge coefficients or validation error')
        candidates.append({'alpha':float(alpha),'validation_mse':mse,'validation_rmse':math.sqrt(mse)})
        coefficients.append(coef)
    selected = min(range(len(candidates)),key=lambda i:(candidates[i]['validation_mse'],i))
    metadata = {'selected_alpha':float(penalties[selected]), 'solver':'primal' if primal else 'dual',
        'training_samples':len(x),'validation_samples':len(vx),'features':x.shape[1],
        'output_shape':list(target_shape),'alpha_grid':[float(a) for a in penalties],
        'validation_grid':candidates, 'feature_standardization':'train_only_mean_and_population_std_floor1e-8',
        'target_centering':'train_only_mean','penalty':'sum_squared_error_plus_alpha_l2_squared',
        'test_accessed':False,'target_unit':'caller_defined;position probes are meters, power controls referenced dB'}
    return RidgeModel(coefficients[selected],mean,scale,y_mean,float(penalties[selected]),target_shape,metadata)


def mean_template(train_targets, samples):
    targets = _array(train_targets,'training template targets')
    if type(samples) is not int or samples < 1:
        raise ValueError('A positive query count is required')
    return np.broadcast_to(targets.mean(0),(samples,)+targets.shape[1:]).copy()


def power_level_transfer(train_input, train_targets, query_input):
    """Train template plus each query's measured input-level shift, no OOD labels."""
    x,y,q = (_array(train_input,'training input'),_array(train_targets,'training targets'),
             _array(query_input,'query input'))
    if len(x) != len(y):
        raise ValueError('Training input/target source rows differ')
    train_level = float(x.mean())
    query_level = q.reshape(len(q),-1).mean(1)-train_level
    return mean_template(y,len(q)) + query_level.reshape((len(q),)+(1,)*(y.ndim-1))


def _run_summary(values, *, undefined=False):
    available = [float(v) for v in values if v is not None]
    if (not undefined and len(available) != len(values)) or any(not math.isfinite(v) for v in available):
        raise ValueError('Invalid or undeclared undefined run metric')
    return {'mean':float(np.mean(available)) if available else None,
            'training_seed_sd':float(np.std(available,ddof=1)) if len(available)>1 else None,
            'individual':list(values),'defined_runs':len(available)}


def aggregate_fit_reports(fit_reports, baselines):
    """Descriptive member means/SDs; no ensembling or extra independent sites."""
    reports = list(fit_reports)
    expected = {(cell,seed) for cell in CELLS for seed in (0,1,2)}
    index = {}
    for item in reports:
        key = item['cell'],item['seed']
        if key not in expected or key in index or item.get('completed') is not True:
            raise ValueError('Incomplete, duplicated or unregistered fit identity')
        index[key] = item
    if set(index) != expected:
        raise ValueError('All eighteen independently trained members are required')
    domains = set(index[CELLS[0],0]['domains'])
    if (not domains or any(set(item['domains']) != domains for item in index.values())
            or set(baselines['domains']) != domains):
        raise ValueError('Every member and baseline must cover the same measured domains')
    result = {}
    representation_fields = ('effective_rank_a','effective_rank_b','mean_coordinate_std_a',
                             'mean_coordinate_std_b','mutual_knn_alignment')
    for cell in CELLS:
        domain_values = {}
        for domain in sorted(domains):
            rows = [index[cell,seed]['domains'][domain] for seed in (0,1,2)]
            sizes = [row['metrics']['samples'] for row in rows]
            if len(set(sizes)) != 1:
                raise ValueError('Training members were evaluated on different source rows')
            value = {'samples_per_run':sizes[0],
                     'prediction_metrics':{name:_run_summary([row['metrics'][name] for row in rows])
                         for name in ('rmse_db','mae_db','worst_decile_rmse_db')}}
            for name in ('representation_z','representation_projected'):
                value[name] = {field:_run_summary([row[name][field] for row in rows],
                                                 undefined=field=='mutual_knn_alignment')
                               for field in representation_fields}
            interventions = set(rows[0]['interventions'])
            if any(set(row['interventions']) != interventions for row in rows):
                raise ValueError('Intervention membership differs across independent runs')
            value['interventions'] = {kind:{'rmse_db':_run_summary([
                row['interventions'][kind]['rmse_db'] for row in rows])} for kind in sorted(interventions)}
            if all('ab_metrics' in row for row in rows):
                value['unperturbed_ab_rmse_db'] = _run_summary([row['ab_metrics']['rmse_db'] for row in rows])
                for kind in interventions:
                    value['interventions'][kind]['delta_vs_unperturbed_ab_rmse_db'] = _run_summary([
                        row['interventions'][kind]['rmse_db']-row['ab_metrics']['rmse_db'] for row in rows])
            if any('frozen_position_probe' in row for row in rows):
                if not all('frozen_position_probe' in row for row in rows):
                    raise ValueError('Frozen-probe membership differs across members')
                value['frozen_position_probe'] = {field:_run_summary([
                    row['frozen_position_probe'][field] for row in rows])
                    for field in ('mean_error_m','median_error_m')}
            value['baseline_rmse_differences'] = {name:value['prediction_metrics']['rmse_db']['mean']-metric['rmse_db']
                for name,metric in baselines['domains'][domain]['metrics'].items()}
            value['differences_are_descriptive_not_significance_tests'] = True
            domain_values[domain] = value
        result[cell] = {'training_seeds':[0,1,2], 'domains':domain_values,
                        'inference_parameters':[index[cell,seed]['parameters'] for seed in (0,1,2)]}
    return {'schema':1,'completed_fits':18,'training_seeds':[0,1,2],
        'cells':result,'baseline_metrics':{name:deepcopy(value['metrics']) for name,value in baselines['domains'].items()},
        'power_unit':POWER_UNIT,'prediction_ensembling':False,
        'training_seeds_are_not_independent_sites':True,
        'intervals_or_adjusted_superiority_tests_performed':False,
        'scope':'one constructed factory split and two held-out recordings, not official full-dataset benchmark protocols',
        'robot_service_qualified':False,'global_sota_claimed':False,'hardware_qualified':False}
