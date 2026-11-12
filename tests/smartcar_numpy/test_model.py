"""Portable-backend software parity, never new RF cases or model training.

The three released checkpoints are pinned by SHA256. Inputs below are
deterministic analytic software fixtures; no private RF labels, fitting,
ray tracing or packet simulation enters these tests.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from wirelessjev.smartcar_numpy.model import NumpyField, STATE_SHAPES, validate_arrays

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SHA256 = (
    '57c710af2db4f4bd7abfd031a03a76359c27100568d3ebe277ab3c738a69c3bf',
    '2de1c28d4ea41298cac58ca64cabe6d86a8c76688646a3dd4f479ab6686a9b52',
    '16571ca0adb781349e1988fae74a648caf06c76c13c1cf5f605b0e5a4fee974c',
)
ATOL_DB = 1e-4
ATOL_COV_DB2 = 1e-4
RTOL = 1e-6


def analytic_state():
    """Explicit toy parameters for structural preflight, not a fitted model."""
    arrays = {name: np.zeros(shape, dtype=np.float32) for name, shape in STATE_SHAPES.items()}
    arrays['local_gates'][:5] = np.float32(.1)
    return arrays


def basis(q):
    # This is the archived query path: calculate from float32 public geometry
    # in float64, then return an explicit float32 query basis.
    x = q.astype(np.float64)
    length = x[:, 10] * np.power(10., 2*x[:, 6]) / 10.
    f = x[:, 7]
    return np.stack((np.ones(len(x)), x[:, 6], x[:, 9], length,
        f, f*x[:, 6], f*x[:, 9], f*length), axis=-1).astype(np.float32)


def fixture(*, sites=4, tokens=192, queries=192, event='mixed'):
    rng = np.random.default_rng(481913)
    aps = rng.uniform((-45., -30., 3.5), (45., 30., 4.5), (sites, 3)).astype(np.float32)

    def geometry(count):
        route = np.arange(count, dtype=np.int64) % sites
        rx = rng.uniform((-48., -32., 1.), (48., 32., 1.2), (count, 3)).astype(np.float32)
        chosen = aps[route]
        q = np.zeros((count, 16), dtype=np.float32)
        q[:, :3] = rx / np.array((120., 80., 6.), dtype=np.float32)
        q[:, 3:6] = chosen / np.array((120., 80., 6.), dtype=np.float32)
        q[:, 6] = np.log10(np.linalg.norm(rx.astype('f8')-chosen, axis=-1))/2.
        q[:, 7] = np.log(np.where(np.arange(count) % 2, 5.2, 2.4)/2.4)
        q[:, 8] = -1.
        q[:, 9] = (route % 3)/8.
        q[:, 10] = .08 + .01*(np.arange(count) % 5)
        q[:, 11] = 1.
        q[:, 13] = .25
        q[:, 14] = -1.
        q[:, 15] = 2/3
        return q, route, rx

    q, routes, rx = geometry(tokens)
    value = (-12. - .02*rx[:, 0] + .03*rx[:, 1]).astype(np.float64)
    known = np.ones(tokens, dtype=bool)
    heard = known.copy()
    point = np.zeros(tokens, dtype=bool)
    half_width = .5
    if event == 'point':
        point[:] = True
    elif event == 'interval':
        pass
    elif event == 'narrow':
        half_width = 1e-4
    elif event == 'censor':
        heard[:] = False
    elif event == 'tail':
        heard[:] = False
        value[:] = -160.
    elif event == 'unknown':
        known[:] = False
        heard[:] = False
    elif event == 'mixed':
        point[::4] = True
        heard[2::4] = False
        known[3::4] = False
        heard &= known
    else:
        raise ValueError('Unknown analytic event fixture')
    censored = known & ~heard
    lower = np.where(heard, value-half_width, -np.inf)
    upper = np.where(known, value+half_width, np.inf)
    lower[point] = value[point]
    upper[point] = value[point]
    context = np.column_stack((np.where(heard, value/30., 0.),
        np.where(censored, upper/30., 0.), heard, censored, q)).astype(np.float32)
    view = {'context': context, 'known': known, 'scheduled': np.ones(tokens, dtype=bool),
        'ap': routes, 'rx_xyz': rx, 'lower': lower.astype(np.float64),
        'upper': upper.astype(np.float64), 'floor': np.full(tokens, -250., dtype=np.float64)}
    qq, qr, qrx = geometry(queries)
    query = {'q': qq, 'ap': qr, 'rx_xyz': qrx, 'basis': basis(qq),
        'fspl': np.full(queries, -50., dtype=np.float32),
        'offset': np.zeros(queries, dtype=np.float32)}
    return view, query, aps


@pytest.fixture(scope='module', params=(0, 1, 2), ids=lambda seed: f'pinned-seed{seed}')
def released(request):
    import torch
    from wirelessjev.nested_radio.model import NestedField
    seed = request.param
    path = ROOT / f'artifacts/nested_radio/checkpoints/nested_joint-seed{seed}.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CHECKPOINT_SHA256[seed]
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['settings'] == {'cell': 'nested_joint', 'width': 32, 'inference_steps': 3}
    arrays = {name: value.detach().cpu().numpy().copy() for name, value in checkpoint['state'].items()}
    with torch.random.fork_rng(devices=[]):
        reference = NestedField(**checkpoint['settings'])
    reference.load_state_dict(checkpoint['state'], strict=True)
    reference.eval()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield NumpyField(arrays), reference, arrays
    finally:
        torch.set_num_threads(previous_threads)


def tensor_mapping(values):
    import torch
    return {name: torch.from_numpy(value.copy())[None] for name, value in values.items()}


def reference_prediction(reference, view, query, aps):
    import torch
    with torch.no_grad():
        state = reference.encode_view(tensor_mapping(view), torch.from_numpy(aps.copy())[None])
        mean, std = reference.render_queries(state, tensor_mapping(query))
    return state, mean[0].numpy(), std[0].numpy()


def assert_log_parity(actual, expected):
    assert len(actual) == len(expected) == 3
    for row, wanted in zip(actual, expected):
        assert row['step_size'] == float(wanted['step_size'][0])
        assert row['null_step'] is bool(wanted['null_step'][0])
        np.testing.assert_allclose(row['objective_before'], float(wanted['objective_before'][0]),
            rtol=RTOL, atol=ATOL_DB)
        np.testing.assert_allclose(row['objective_after'], float(wanted['objective_after'][0]),
            rtol=RTOL, atol=ATOL_DB)


@pytest.mark.parametrize('event', ['point', 'interval', 'narrow', 'censor', 'tail', 'unknown', 'mixed'])
def test_pinned_float32_features_float64_inference_and_three_step_parity(released, event):
    backend, reference, _ = released
    view, query, aps = fixture(event=event)
    expected_state, expected_mean, expected_std = reference_prediction(reference, view, query, aps)
    state = backend.encode_view(view, aps)
    mean, std = backend.render_queries(state, query)
    assert mean.dtype == std.dtype == np.float64
    assert state['global_mean'].dtype == state['global_covariance'].dtype == np.float64
    assert not state['field_covariance_qualified'] and not state['packet_risk_qualified']
    np.testing.assert_allclose(mean, expected_mean, rtol=RTOL, atol=ATOL_DB)
    np.testing.assert_allclose(std, expected_std, rtol=RTOL, atol=ATOL_DB)
    np.testing.assert_allclose(state['global_mean'], expected_state['global_mean'][0].numpy(),
        rtol=RTOL, atol=ATOL_DB/10.)
    np.testing.assert_allclose(state['local_mean'], expected_state['local_mean'][0].numpy(),
        rtol=RTOL, atol=ATOL_DB/10.)
    np.testing.assert_array_equal(state['observed_ap'], expected_state['observed_ap'][0].numpy())
    assert_log_parity(state['inference_log'], expected_state['inference_log'])


@pytest.mark.parametrize('sites', [1, 4, 16])
def test_released_field_covariance_matches_torch_and_render_variance(released, sites):
    import torch
    backend, reference, _ = released
    view, query, aps = fixture(sites=sites, queries=21)
    source_state, _, _ = reference_prediction(reference, view, query, aps)
    state = backend.encode_view(view, aps)
    actual = backend.predictive_covariance(state, query)
    tq = tensor_mapping(query)
    with torch.no_grad():
        expected = reference.predictive_covariance(source_state, tq['q'], tq['ap'],
            tq['rx_xyz'], tq['basis'])[0].numpy()
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL_COV_DB2)
    np.testing.assert_allclose(actual, actual.T, rtol=RTOL, atol=ATOL_COV_DB2)
    _, std = backend.render_queries(state, query)
    np.testing.assert_allclose(np.diag(actual)+backend.discrepancy_db(query['q']).astype('f8')**2,
        std**2, rtol=RTOL, atol=ATOL_COV_DB2)
    pair = actual[0, 0]+actual[1, 1]-2*actual[0, 1]
    assert pair >= -ATOL_COV_DB2


def test_active_fourier_priors_match_independent_dense_gaussian_point_conditioning(released):
    import torch
    from wirelessjev.spectral_scene.operators import query_physics_basis
    backend, reference, arrays = released
    view, query, aps = fixture(sites=4, tokens=192, queries=17, event='point')
    tv = tensor_mapping(view)
    ta = torch.from_numpy(aps.copy())[None]
    with torch.no_grad():
        g = query_physics_basis(tv['context'][..., 4:])[0].double().numpy()
        local = reference.local_features(tv['context'][..., 4:], tv['rx_xyz'], ta, tv['ap'])[0].double().numpy()
        sigma = reference.discrepancy_db(tv['context'][..., 4:])[0].double().numpy()/10.
        gs = .2+torch.nn.functional.softplus(reference.raw_global_std).double().numpy()
        ls = .2+torch.nn.functional.softplus(reference.raw_local_std).double().numpy()
    active = np.flatnonzero(arrays['local_gates'])
    assert len(active) in (5, 6) and (active < 8).all()
    dimension = 8+len(aps)*len(active)
    design = np.zeros((len(g), dimension), dtype=np.float64)
    design[:, :8] = g
    for site in range(len(aps)):
        selected = view['ap'] == site
        design[selected, 8+site*len(active):8+(site+1)*len(active)] = local[selected][:, active]
    prior_mean = np.concatenate((arrays['global_prior_mean'],
        np.tile(arrays['local_prior_mean'][active], len(aps)))).astype(np.float64)
    precision = np.diag(np.concatenate((gs**-2+1e-8,
        np.tile(ls[active]**-2+1e-8, len(aps)))))
    information = 1./sigma**2
    observed_precision = precision+design.T@(information[:, None]*design)
    rhs = precision@prior_mean+design.T@(information*(view['lower']/10.))
    expected_mean = np.linalg.solve(observed_precision, rhs)
    expected_cov = np.linalg.inv(observed_precision)
    state = backend.encode_view(view, aps)
    np.testing.assert_allclose(state['global_mean'], expected_mean[:8], rtol=RTOL, atol=ATOL_DB/10.)
    np.testing.assert_allclose(state['local_mean'][:, active].reshape(-1), expected_mean[8:],
        rtol=RTOL, atol=ATOL_DB/10.)
    np.testing.assert_allclose(state['global_covariance'], expected_cov[:8, :8],
        rtol=RTOL, atol=ATOL_COV_DB2/100.)


def test_ap_renaming_and_inverse_duplicate_weights_preserve_released_forecasts(released):
    backend, _, _ = released
    view, query, aps = fixture(sites=16, tokens=96, event='mixed')
    original = backend.render_queries(backend.encode_view(view, aps), query)
    perm = np.roll(np.arange(16), 5)
    inverse = np.argsort(perm)
    renamed_view, renamed_query = deepcopy(view), deepcopy(query)
    renamed_view['ap'] = inverse[view['ap']]
    renamed_query['ap'] = inverse[query['ap']]
    renamed = backend.render_queries(backend.encode_view(renamed_view, aps[perm]), renamed_query)
    for observed, expected in zip(renamed, original):
        np.testing.assert_allclose(observed, expected, rtol=RTOL, atol=ATOL_DB)
    duplicated = {name: np.repeat(value, 2, axis=0) for name, value in view.items()}
    duplicated['weights'] = np.repeat(view['known'].astype(np.float64)/2., 2)
    repeated = backend.render_queries(backend.encode_view(duplicated, aps), query)
    for observed, expected in zip(repeated, original):
        np.testing.assert_allclose(observed, expected, rtol=RTOL, atol=ATOL_DB)


def test_large_native_grid_renders_marginals_and_full_covariance_fails_before_allocation(released):
    backend, reference, _ = released
    view, query, aps = fixture(sites=16, queries=43200, event='point')
    state = backend.encode_view(view, aps)
    mean, std = backend.render_queries(state, query)
    _, expected_mean, expected_std = reference_prediction(reference, view, query, aps)
    assert mean.shape == std.shape == (43200,)
    np.testing.assert_allclose(mean, expected_mean, rtol=RTOL, atol=ATOL_DB)
    np.testing.assert_allclose(std, expected_std, rtol=RTOL, atol=ATOL_DB)
    with pytest.raises(ValueError, match='512'):
        backend.predictive_covariance(state, query)


def test_structural_arrays_are_owned_and_read_only_without_loading_a_checkpoint():
    arrays = analytic_state()
    validated = validate_arrays(arrays)
    assert set(validated) == set(STATE_SHAPES)
    for name in arrays:
        assert not np.shares_memory(arrays[name], validated[name])
        assert validated[name].dtype == np.float32 and not validated[name].flags.writeable


@pytest.mark.parametrize('change', ['missing', 'extra', 'wrong-shape', 'float64', 'nan',
    'negative-gate', 'large-gate', 'tiny-learned-gate', 'four-active'])
def test_invalid_or_unreleased_weight_structures_fail_closed(change):
    arrays = analytic_state()
    if change == 'missing':
        del arrays['kernel.0.bias']
    elif change == 'extra':
        arrays['unexpected'] = np.zeros(1, dtype=np.float32)
    elif change == 'wrong-shape':
        arrays['local_gates'] = np.zeros(15, dtype=np.float32)
    elif change == 'float64':
        arrays['raw_discrepancy'] = arrays['raw_discrepancy'].astype(np.float64)
    elif change == 'nan':
        arrays['kernel.0.weight'][0, 0] = np.nan
    elif change == 'negative-gate':
        arrays['local_gates'][0] = -.1
    elif change == 'large-gate':
        arrays['local_gates'][0] = 1.1
    elif change == 'tiny-learned-gate':
        arrays['local_gates'][8] = np.nextafter(np.float32(0.), np.float32(1.))
    else:
        arrays['local_gates'][4] = 0.
    with pytest.raises(ValueError):
        NumpyField(arrays)


@pytest.mark.parametrize('change', ['context-f64', 'q-f64', 'rx-f64', 'aps-f64',
    'known-int', 'bad-ap', 'wrong-distance', 'known-not-scheduled', 'context-too-large'])
def test_input_precision_routes_and_causal_masks_are_checked_on_analytic_weights(change):
    backend = NumpyField(analytic_state())
    view, query, aps = fixture()
    if change == 'context-f64':
        view['context'] = view['context'].astype(np.float64)
    elif change == 'q-f64':
        query['q'] = query['q'].astype(np.float64)
    elif change == 'rx-f64':
        view['rx_xyz'] = view['rx_xyz'].astype(np.float64)
    elif change == 'aps-f64':
        aps = aps.astype(np.float64)
    elif change == 'known-int':
        view['known'] = view['known'].astype(np.int64)
    elif change == 'bad-ap':
        view['ap'][0] = len(aps)
    elif change == 'wrong-distance':
        view['rx_xyz'][0, 0] += 3.
    elif change == 'known-not-scheduled':
        view['scheduled'][0] = False
    else:
        view = {name: np.concatenate((value, value[:1])) for name, value in view.items()}
    with pytest.raises((ValueError, FloatingPointError)):
        state = backend.encode_view(view, aps)
        backend.render_queries(state, query)
