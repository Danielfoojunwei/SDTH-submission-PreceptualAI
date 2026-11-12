"""Software parity for the portable public-geometry operators.

The frozen reference is used only for feature arithmetic. Public fixture inputs
come from the existing nested registry and ASCII CAD vertices; no RT, fitting,
model inference, hardware capture, or new scientific outcome runs here.
"""
from copy import deepcopy
import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wirelessjev.smartcar_numpy import geometry as portable
from wirelessjev.spectral_scene import operators as frozen


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'src/wirelessjev/smartcar_numpy/geometry.py'


def _public_box(path):
    """Read only public ASCII vertex positions, never material/solver labels."""
    lines = path.read_text().splitlines()
    assert lines[:2] == ['ply', 'format ascii 1.0']
    count = next(int(line.split()[-1]) for line in lines if line.startswith('element vertex '))
    start = lines.index('end_header') + 1
    vertices = np.array([[float(x) for x in line.split()[:3]] for line in lines[start:start + count]])
    assert vertices.shape == (count, 3) and np.isfinite(vertices).all()
    return np.stack([vertices.min(axis=0), vertices.max(axis=0)])


@pytest.fixture(scope='module', params=[('test', 0), ('test', 7), ('test', 15), ('ap_transfer', 0)])
def public_links(request):
    registry = json.loads((ROOT / 'artifacts/nested_radio/registry.json').read_text())
    stage, index = request.param
    layout = registry['stages'][stage][index]
    public = layout['public_geometry']
    ap = np.array(public['ap_positions_m'])
    rx = np.array(public['rx_positions_m'])
    case_id = layout['cases'][0]['case_id']
    meshes = ROOT / 'artifacts/nested_radio/rt_data/cases' / case_id / 'scene/meshes'
    paths = sorted(meshes.glob('*.ply'))
    assert paths, 'The immutable public CAD fixture must be available.'
    boxes = np.stack([_public_box(path) for path in paths])
    pose, access_point, band, direction = np.meshgrid(
        np.arange(len(rx)), np.arange(len(ap)), np.arange(3), np.arange(2), indexing='ij')
    pose, access_point, band, direction = [value.ravel() for value in (pose, access_point, band, direction)]
    return dict(rx_xyz=rx[pose], ap_xyz=ap[access_point],
        carrier_hz=np.array(registry['bands_hz'])[band], direction=direction,
        bbox_dims_m=np.array([public['width_m'], public['depth_m'], public['ceiling_m']]),
        cad_boxes_m=boxes, calibration_pair_count=(pose + access_point) % 23,
        calibration_unknown=access_point % 2 == 0,
        quantization_width_db=(pose % 4) * .5,
        noise_floor_dbm=-100. + access_point, tx_power_dbm=17. + band)


def _simple_links():
    return dict(rx_xyz=np.array([[10., 0., 1.], [2., 3., 1.]]),
        ap_xyz=np.array([[0., 0., 2.], [0., 0., 2.]]),
        carrier_hz=[2.4e9, 5.2e9], direction=[0, 1], bbox_dims_m=[20., 10., 4.],
        cad_boxes_m=np.array([[[2., -1., 0.], [6., 1., 3.]]]))


def test_portable_geometry_has_no_torch_or_scientific_pipeline_dependency():
    tree = ast.parse(SOURCE.read_text(), feature_version=(3, 9))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or '')
    assert not any(name.split('.')[0] in ('torch', 'scipy', 'sionna', 'mitsuba', 'drjit') for name in imports)
    assert not any('spectral_scene' in name or 'nested_radio' in name for name in imports)


def test_public_feature_names_and_reference_units_keep_the_frozen_contract():
    assert portable.GEOMETRY_FEATURE_NAMES == frozen.GEOMETRY_FEATURE_NAMES
    assert portable.CONTEXT_FEATURE_NAMES == frozen.CONTEXT_FEATURE_NAMES
    assert portable.PHYSICS_BASIS_NAMES == frozen.PHYSICS_BASIS_NAMES
    assert portable.REFERENCE_CARRIER_HZ == frozen.REFERENCE_CARRIER_HZ == 2.4e9


def test_real_public_cad_segments_match_frozen_float64_union_arithmetic(public_links):
    args = [public_links[key] for key in ('rx_xyz', 'ap_xyz', 'cad_boxes_m')]
    expected = frozen.segment_box_features(*args)
    actual = portable.segment_box_features(*args)
    assert len(actual) == 2
    for result, reference in zip(actual, expected):
        assert result.dtype == np.float64 and result.shape == (len(args[0]),)
        np.testing.assert_array_equal(result, reference)
    assert np.all(actual[0] >= 0) and np.all((actual[1] >= 0) & (actual[1] <= 1))


def test_real_public_cad_geometry_matches_all_sixteen_frozen_features(public_links):
    before = deepcopy(public_links)
    expected = frozen.link_geometry_features(**public_links)
    actual = portable.link_geometry_features(**public_links)
    assert actual.dtype == np.float32 and actual.shape == (len(public_links['rx_xyz']), 16)
    np.testing.assert_array_equal(actual, expected)
    for key, value in public_links.items():
        np.testing.assert_array_equal(value, before[key])
    np.testing.assert_array_equal(actual[:, 8], 2 * np.asarray(public_links['direction']) - 1)
    assert np.max(actual[:, 11]) == 1.  # Calibration counts above 16 are clipped.


def test_public_query_basis_matches_frozen_numpy_double_then_single_path(public_links):
    features = frozen.link_geometry_features(**public_links)
    expected = frozen.query_physics_basis(features)
    actual = portable.query_physics_basis(features, context=False)
    assert actual.dtype == np.float32 and actual.shape == (len(features), 8)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(portable.query_physics_basis(features), expected)


def test_public_context_basis_matches_frozen_torch_single_precision_path(public_links):
    features = frozen.link_geometry_features(**public_links)
    expected = frozen.query_physics_basis(torch.from_numpy(features.copy())).numpy()
    actual = portable.query_physics_basis(features, context=True)
    assert actual.dtype == np.float32 and actual.shape == (len(features), 8)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_overlapping_boxes_count_each_solid_but_material_length_uses_the_union():
    boxes = np.array([[[2., -1., 0.], [4., 1., 2.]], [[3., -1., 0.], [6., 1., 2.]]])
    ap = np.array([[0., 0., 1.], [0., 3., 1.], [0., 0., 1.], [3., 0., 1.], [0., 1., 1.]])
    rx = np.array([[10., 0., 1.], [10., 3., 1.], [2., 0., 1.], [10., 0., 1.], [10., 1., 1.]])
    counts, fractions = portable.segment_box_features(rx, ap, boxes)
    np.testing.assert_array_equal(counts, [2., 0., 0., 2., 2.])
    np.testing.assert_allclose(fractions, [.4, 0., 0., 3. / 7., .4], rtol=0, atol=1e-15)
    reference = frozen.segment_box_features(rx, ap, boxes)
    np.testing.assert_array_equal(counts, reference[0])
    np.testing.assert_array_equal(fractions, reference[1])
    # Reversing a physical segment preserves its solids and occupied length.
    reverse = portable.segment_box_features(ap, rx, boxes)
    np.testing.assert_array_equal(reverse[0], counts)
    np.testing.assert_allclose(reverse[1], fractions, rtol=0, atol=1e-15)


def test_touching_boxes_join_without_double_counting_and_degenerate_face_crossing_has_no_length():
    boxes = np.array([[[2., -1., 0.], [4., 1., 2.]], [[4., -1., 0.], [6., 1., 2.]]])
    counts, fractions = portable.segment_box_features([[10., 0., 1.]], [[0., 0., 1.]], boxes)
    np.testing.assert_array_equal(counts, [2.])
    np.testing.assert_allclose(fractions, [.4], rtol=0, atol=1e-15)
    plane = np.array([[[2., -1., 1.], [6., 1., 1.]]])
    np.testing.assert_array_equal(portable.segment_box_features([[4., 0., 2.]], [[4., 0., 0.]], plane), [[0.], [0.]])


def test_segment_translation_and_box_permutation_preserve_public_intersections():
    args = _simple_links()
    rx, ap, boxes = [np.array(args[key]) for key in ('rx_xyz', 'ap_xyz', 'cad_boxes_m')]
    shift = np.array([7., -3., 4.])
    first = portable.segment_box_features(rx, ap, boxes)
    second = portable.segment_box_features(rx + shift, ap + shift, boxes[::-1] + shift)
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-15)


def test_empty_public_geometry_batches_and_no_boxes_keep_their_schema():
    counts, fractions = portable.segment_box_features(np.empty((0, 3)), np.empty((0, 3)), [])
    assert counts.dtype == fractions.dtype == np.float64 and counts.shape == fractions.shape == (0,)
    values = _simple_links(); values['cad_boxes_m'] = []
    result = portable.link_geometry_features(**values)
    np.testing.assert_array_equal(result[:, 9:11], np.zeros((2, 2), dtype='f4'))
    empty = result[:0]
    for context in (False, True):
        basis = portable.query_physics_basis(empty, context=context)
        assert basis.dtype == np.float32 and basis.shape == (0, 8)


def test_basis_path_is_explicit_and_preserves_batched_dimensions():
    features = frozen.link_geometry_features(**_simple_links())
    batched = np.stack([features, features])
    for context in (False, True):
        actual = portable.query_physics_basis(batched, context=context)
        expected = frozen.query_physics_basis(torch.from_numpy(batched)) if context else frozen.query_physics_basis(batched)
        expected = expected.numpy() if context else expected
        assert actual.dtype == np.float32 and actual.shape == (2, 2, 8)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    # Query callers may already hold f4 geometry, but still require f8 arithmetic
    # before the final f4 cast. The context path must not silently replace it.
    rng = np.random.default_rng(8341)
    q = np.zeros((64, 16), dtype='f4')
    q[:, 6] = rng.uniform(-.4, 1.7, len(q)); q[:, 7] = rng.uniform(-.1, 1., len(q))
    q[:, 9] = rng.uniform(0., 1., len(q)); q[:, 10] = rng.uniform(0., 1., len(q))
    query = portable.query_physics_basis(q, context=False)
    context = portable.query_physics_basis(q, context=True)
    np.testing.assert_array_equal(query, frozen.query_physics_basis(q))
    np.testing.assert_allclose(context, frozen.query_physics_basis(torch.from_numpy(q)).numpy(), rtol=1e-6, atol=1e-6)
    assert np.any(query != context), 'The two arithmetic contracts need a discriminating fixture.'


@pytest.mark.parametrize('features', [np.zeros((2, 16), dtype='f8'), np.zeros((2, 16), dtype='i4'), [[0.] * 16]])
def test_context_basis_refuses_an_implicit_float32_conversion(features):
    with pytest.raises(ValueError):
        portable.query_physics_basis(features, context=True)


@pytest.mark.parametrize('context', [0, 1, None, 'False', np.bool_(True)])
def test_basis_arithmetic_mode_must_be_an_explicit_python_boolean(context):
    with pytest.raises(ValueError):
        portable.query_physics_basis(np.zeros((1, 16), dtype='f4'), context=context)


@pytest.mark.parametrize('context', [False, True])
def test_finite_inputs_that_overflow_the_derived_basis_are_rejected(context):
    features = np.zeros((1, 16), dtype='f4')
    features[0, 6] = 200.; features[0, 10] = 1.
    with pytest.raises(ValueError):
        portable.query_physics_basis(features, context=context)


@pytest.mark.parametrize('features', [
    np.zeros(16), np.zeros((2, 15)), np.zeros((2, 17)),
    np.full((2, 16), np.nan), np.full((2, 16), np.inf),
    np.zeros((2, 16), dtype='complex128'),
])
@pytest.mark.parametrize('context', [False, True])
def test_basis_refuses_nonfinite_complex_or_wrong_schema_features(features, context):
    with pytest.raises(ValueError):
        portable.query_physics_basis(features, context=context)


@pytest.mark.parametrize('changes', [
    {'rx_xyz': [[np.nan, 0., 1.], [2., 3., 1.]]},
    {'ap_xyz': [[0., 0., np.inf], [0., 0., 2.]]},
    {'rx_xyz': np.ones((3, 3))}, {'ap_xyz': np.ones(3)},
    {'bbox_dims_m': [20., 0., 4.]}, {'bbox_dims_m': [20., np.inf, 4.]},
    {'bbox_dims_m': [20., 10.]}, {'carrier_hz': [2.4e9, 0.]},
    {'carrier_hz': [-1., 5.2e9]}, {'carrier_hz': [2.4e9, np.nan]},
    {'direction': [0, 2]}, {'direction': [0, .5]},
    {'calibration_pair_count': [-1, 2]}, {'calibration_pair_count': [.5, 2]},
    {'calibration_unknown': [0, 2]}, {'quantization_width_db': [-1., 0.]},
    {'noise_floor_dbm': [np.inf, -100.]}, {'tx_power_dbm': [20., np.nan]},
    {'carrier_hz': np.ones((2, 1))},
    {'cad_boxes_m': [[[6., -1., 0.], [2., 1., 3.]]]},
    {'cad_boxes_m': [[[2., -1., 0.], [6., np.nan, 3.]]]},
])
def test_geometry_refuses_invalid_public_coordinates_and_metadata(changes):
    args = _simple_links(); args.update(changes)
    with pytest.raises(ValueError):
        portable.link_geometry_features(**args)


@pytest.mark.parametrize('name', [
    'rx_xyz', 'ap_xyz', 'bbox_dims_m', 'cad_boxes_m', 'carrier_hz', 'direction',
    'calibration_pair_count', 'calibration_unknown', 'quantization_width_db', 'noise_floor_dbm', 'tx_power_dbm'])
def test_geometry_rejects_complex_inputs_instead_of_discarding_imaginary_parts(name):
    args = _simple_links()
    args[name] = np.asarray(args.get(name, 0), dtype='complex128') + 1j
    with pytest.raises(ValueError):
        portable.link_geometry_features(**args)


def test_link_geometry_refuses_coincident_endpoints():
    args = _simple_links(); args['ap_xyz'] = args['rx_xyz'].copy()
    with pytest.raises(ValueError):
        portable.link_geometry_features(**args)


def test_finite_endpoints_with_overflowing_distance_fail_closed():
    args = _simple_links()
    args['rx_xyz'] = np.full((2, 3), 1e308)
    args['ap_xyz'] = np.full((2, 3), -1e308)
    with pytest.raises(ValueError):
        portable.link_geometry_features(**args)
    with pytest.raises(ValueError):
        portable.segment_box_features(args['rx_xyz'], args['ap_xyz'], [])


@pytest.mark.parametrize('rx,ap,boxes', [
    ([[1., 2.]], [[0., 0., 0.]], []),
    ([[1., 2., 3.]], [[0., 0., 0.], [0., 0., 1.]], []),
    ([[np.nan, 2., 3.]], [[0., 0., 0.]], []),
    ([[1., 2., 3.]], [[0., np.inf, 0.]], []),
    ([[1., 2., 3.]], [[0., 0., 0.]], np.zeros((1, 3, 2))),
    ([[1., 2., 3.]], [[0., 0., 0.]], [[[2., 0., 0.], [1., 1., 1.]]]),
    ([[1. + 1j, 2., 3.]], [[0., 0., 0.]], []),
    ([[1., 2., 3.]], [[0., 0., 0.]], np.zeros((1, 2, 3), dtype='complex128')),
])
def test_segment_operator_fails_closed_for_invalid_public_geometry(rx, ap, boxes):
    with pytest.raises(ValueError):
        portable.segment_box_features(rx, ap, boxes)
