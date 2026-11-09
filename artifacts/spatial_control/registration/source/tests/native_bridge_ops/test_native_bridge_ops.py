"""Analytic/public CAD unit fixtures only; no model, world or packet run."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from wirelessjev.native_bridge import predictor as reference
from wirelessjev.native_bridge_ops import operators as ops
from wirelessjev.twin.warehouse import AP_SITES


def public_geometry():
    return ops.PublicQueryGeometry(np.asarray(AP_SITES), np.array((21., 23., 23.)),
                                   np.array((-94., -88., -85.)))


def public_plan(robots):
    """Explicit software fixture on the published floor, not a simulated trace."""
    robot = np.arange(robots)[:, None]
    horizon = np.arange(9)[None, :]
    return np.stack((1. + (37 * robot + 3 * horizon) % 118,
                     1. + (23 * robot + 5 * horizon) % 78), axis=-1)


def assert_exact(actual, expected):
    assert actual.keys() == expected.keys()
    assert actual["shape"] == expected["shape"]
    for name in ("query", "basis", "fspl", "offset"):
        assert actual[name].dtype == expected[name].dtype == np.float32
        assert actual[name].shape == expected[name].shape
        assert actual[name].tobytes() == expected[name].tobytes(), name


@pytest.mark.parametrize("robots", [0, 1, 7, 100])
def test_fixed_public_warehouse_queries_have_exact_frozen_operand_bytes(robots):
    geometry, plan = public_geometry(), public_plan(robots)
    before = plan.copy()
    assert_exact(ops.query_operands(geometry, plan), reference.query_operands(geometry, plan))
    np.testing.assert_array_equal(plan, before)


def test_bandwise_metadata_and_order_remain_distinct_and_exact():
    geometry, plan = public_geometry(), public_plan(2)
    result = ops.query_operands(geometry, plan)
    features = result["query"].reshape(result["shape"] + (16,))
    np.testing.assert_array_equal(features[0, 0, 0, :, 7],
                                  np.log(np.asarray(reference.NATIVE_CARRIERS_HZ) / 2.4e9).astype("f4"))
    np.testing.assert_array_equal(features[0, 0, 0, :, 14], (geometry.noise / 100).astype("f4"))
    np.testing.assert_array_equal(features[0, 0, 0, :, 15], (geometry.tx / 30).astype("f4"))
    assert len(np.unique(result["fspl"].reshape(result["shape"])[0, 0, 0])) == 3
    assert_exact(result, reference.query_operands(geometry, plan))


def test_geometry_intersection_call_has_one_third_of_full_query_rows(monkeypatch):
    original, calls = ops.vectorized_box_features, []

    def count_rows(rx, ap, boxes):
        calls.append((len(rx), len(ap), boxes.shape))
        return original(rx, ap, boxes)

    # Instrument only the new module's alias, never the frozen native runner.
    monkeypatch.setattr(ops, "vectorized_box_features", count_rows)
    result = ops.query_operands(public_geometry(), public_plan(100))
    assert calls == [(14400, 14400, (33, 2, 3))]
    assert result["query"].shape == (43200, 16)
    assert result["basis"].shape == (43200, 8)


def test_analytic_one_segment_crosses_a_rack_and_nearby_segment_does_not():
    boxes = reference.rack_boxes()
    lo, hi = boxes[0]
    center = (lo + hi) / 2
    rx = np.array(((lo[0] - 1, center[1], center[2]),
                   (lo[0] - 1, lo[1] - 1, center[2])))
    ap = np.array(((hi[0] + 1, center[1], center[2]),
                   (hi[0] + 1, lo[1] - 1, center[2])))
    count, fraction = ops.vectorized_box_features(rx, ap, boxes[:1])
    np.testing.assert_array_equal(count, (1., 0.))
    np.testing.assert_allclose(fraction, ((hi[0] - lo[0]) / (hi[0] - lo[0] + 2), 0.), rtol=0, atol=1e-15)


def test_snapshot_is_immutable_and_copies_original_public_arrays():
    sites = np.asarray(AP_SITES, dtype=float).copy()
    geometry = ops.PublicQueryGeometry(sites, np.array((21., 23., 23.)), np.array((-94., -88., -85.)))
    sites[:] = 0
    assert np.any(geometry.sites)
    with pytest.raises(ValueError):
        geometry.sites[0, 0] = 0
    with pytest.raises(FrozenInstanceError):
        geometry.tx = np.ones(3)


def test_explicit_snapshot_rejects_objects_with_simulator_handles():
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError("No simulator or private attribute may be accessed: " + name)

    with pytest.raises(TypeError, match="PublicQueryGeometry"):
        ops.query_operands(Forbidden(), public_plan(1))


@pytest.mark.parametrize("plan", [np.zeros((3, 8, 2)), np.zeros((3, 9, 3)),
                                  np.full((1, 9, 2), np.nan)])
def test_invalid_public_plans_are_rejected(plan):
    with pytest.raises(ValueError, match="plan"):
        ops.query_operands(public_geometry(), plan)


@pytest.mark.parametrize("sites,tx,noise", [
    (np.empty((0, 2)), (21., 23., 23.), (-94., -88., -85.)),
    (np.zeros((2, 3)), (21., 23., 23.), (-94., -88., -85.)),
    (np.ones((1, 2)), (21., 23.), (-94., -88., -85.)),
    (np.ones((1, 2)), (21., 23., 23.), (-94., np.inf, -85.)),
])
def test_malformed_public_geometry_is_rejected(sites, tx, noise):
    with pytest.raises(ValueError, match="Finite public"):
        ops.PublicQueryGeometry(sites, tx, noise)


def test_changed_public_plan_and_power_are_not_stale_cached_values():
    geometry, plan = public_geometry(), public_plan(3)
    first = ops.query_operands(geometry, plan)
    changed_plan = plan.copy(); changed_plan[0, 0] += (0.125, 0.25)
    second = ops.query_operands(geometry, changed_plan)
    changed_geometry = ops.PublicQueryGeometry(geometry.sites, geometry.tx + (1., 2., 3.), geometry.noise)
    third = ops.query_operands(changed_geometry, plan)
    assert first["query"].tobytes() != second["query"].tobytes()
    assert first["fspl"].tobytes() != third["fspl"].tobytes()
    assert_exact(second, reference.query_operands(geometry, changed_plan))
    assert_exact(third, reference.query_operands(changed_geometry, plan))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fractional_noncontiguous_plan_and_nonstandard_public_site_count(dtype):
    plan = (public_plan(11).astype(dtype) / 1.03125)[:, ::-1, :]
    assert not plan.flags.c_contiguous
    geometry = ops.PublicQueryGeometry(np.asarray(AP_SITES, dtype=dtype)[::3] + (0.03125, 0.0625),
                                      (20.125, 23.25, 24.5), (-94.125, -88.25, -85.5))
    assert_exact(ops.query_operands(geometry, plan), reference.query_operands(geometry, plan))


def test_profile_rejects_operand_dtype_or_bit_changes():
    from wirelessjev.native_bridge_ops.profile import assert_operand_bytes
    value = ops.query_operands(public_geometry(), public_plan(1))
    assert_operand_bytes(value, value)
    changed_dtype = dict(value, fspl=value["fspl"].astype("f8"))
    with pytest.raises(RuntimeError, match="fspl"):
        assert_operand_bytes(value, changed_dtype)
    changed_bits = {**value, "query": value["query"].copy()}
    changed_bits["query"][0, 0] = np.nextafter(changed_bits["query"][0, 0], np.float32(1))
    with pytest.raises(RuntimeError, match="query"):
        assert_operand_bytes(value, changed_bits)


def test_profile_publication_guard_runs_before_array_access(monkeypatch):
    from wirelessjev.native_bridge_ops import profile

    def unpublished():
        raise RuntimeError("Publication missing unit fixture")

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved arrays must not open before source publication")

    monkeypatch.setattr(profile, "check_sources", unpublished)
    monkeypatch.setattr(profile.np, "load", forbidden)
    with pytest.raises(RuntimeError, match="Publication missing"):
        profile.run()
