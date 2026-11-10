"""Actual map/model runtime and queue behavior, outside scientific sites."""
import copy
import numpy as np
import pytest
import torch
from wirelessjev.closed_loop.telemetry import DeliveredScans
from wirelessjev.fleetwm.data import episode
from wirelessjev.frontier.study import mask_scans
from wirelessjev.semantic_jepa.model import SemanticJEPA
from wirelessjev.semantic_jepa.runtime import Predictor, ChargedForecasts
from wirelessjev.semantic_jepa.study import compare_replays
from wirelessjev.semantic_jepa.common import write_once, write_text_once


@pytest.fixture(scope='module')
def public_episode():
    # Known software fixture, not one of the new study's training/test IDs.
    torch.set_num_threads(1)
    full, _, geometry, observed = episode(745, 2, 4)
    return full, geometry, mask_scans(observed, 5)


@pytest.mark.parametrize('family', ['supervised', 'context_jepa', 'semantic_aux', 'semantic_jepa'])
def test_actual_single_member_map_forecast_is_finite_and_time_ordered(public_episode, family):
    full, geometry, observed = public_episode
    torch.manual_seed(4)
    model = SemanticJEPA(family)
    predictor = Predictor(geometry, family, 0, model=model)
    mean, std, kind, seconds = predictor.step(observed.before(60), full.plan[60])
    assert mean.shape == std.shape == (2, 9, 16, 3)
    assert mean.dtype == np.float32
    assert np.isfinite(mean).all() and (std > 0).all() and seconds > 0
    assert kind == 'cold'
    with pytest.raises(ValueError, match='increasing'):
        predictor.step(observed.before(60), full.plan[60])
    assert predictor.step(observed.before(61), full.plan[61])[2] == 'warm'


def test_future_unreceived_scans_cannot_change_actual_public_map(public_episode):
    full, geometry, observed = public_episode
    altered = copy.deepcopy(observed)
    altered.value[60:] = 999
    altered.heard[60:] = True
    altered.scanned[60:] = True
    a = Predictor(geometry, 'radiomap')
    b = Predictor(geometry, 'radiomap')
    ma, sa, _, _ = a.step(observed.before(60), full.plan[60])
    mb, sb, _, _ = b.step(altered.before(60), full.plan[60])
    np.testing.assert_array_equal(ma, mb)
    np.testing.assert_array_equal(sa, sb)


def test_actual_fifo_never_releases_a_late_job_early(public_episode):
    full, geometry, observed = public_episode
    history = DeliveredScans(observed.before(60), 4, 745)
    live = ChargedForecasts(Predictor(geometry, 'radiomap'), history, 4,
                           {'cold': 2., 'warm': 2., 'refit': 2.}, .5)
    live.issue(0, full.plan[60])
    history.pose(1, observed.positions[61])
    live.issue(1, full.plan[61])
    assert live.ready == [250, 500]
    assert live.take(249) is None
    assert live.take(250) == 0 and live.age == 2.5
    assert live.take(499) is None
    assert live.take(500) == 1 and live.age == 4.
    assert all(not row['runtime_overrun'] for row in live.records)


def test_undercharged_actual_map_call_records_overrun(public_episode):
    full, geometry, observed = public_episode
    history = DeliveredScans(observed.before(60), 4, 745)
    live = ChargedForecasts(Predictor(geometry, 'radiomap'), history, 4,
                           {'cold': 0., 'warm': 0., 'refit': 0.}, .5)
    live.issue(0, full.plan[60])
    assert live.records[0]['runtime_overrun'] is True


def test_runtime_rejects_wrong_model_identity_and_bad_public_plan(public_episode):
    full, geometry, observed = public_episode
    with pytest.raises(ValueError, match='mismatch'):
        Predictor(geometry, 'semantic_jepa', 0, model=SemanticJEPA('supervised'))
    with pytest.raises(ValueError):
        Predictor(geometry, 'radiomap', 0)
    predictor = Predictor(geometry, 'radiomap')
    plan = full.plan[60].copy(); plan[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        predictor.step(observed.before(60), plan)
    with pytest.raises(TypeError, match='observable'):
        predictor.step(full, full.plan[60])


def test_exact_replay_membership_and_actual_payload_override_flags():
    key = ('confirmation', 42, .03, 'radiomap', None)
    original = {'case': list(key), 'result': {'misses': 1}, 'forecast_mean_sha256': 'm',
                'forecast_std_sha256': 's', 'telemetry': {'events': []}, 'decision_trace': []}
    replay = copy.deepcopy(original)
    assert compare_replays([original], [replay], [key])[0]['exact'] is True
    replay['result']['misses'] = 2
    replay['declared_exact'] = True
    assert compare_replays([original], [replay], [key])[0]['exact'] is False
    with pytest.raises(ValueError, match='membership'):
        compare_replays([original], [], [key])
    with pytest.raises(ValueError, match='membership'):
        compare_replays([original], [original, original], [key])
    with pytest.raises(ValueError, match='no original'):
        compare_replays([], [original], [key])


def test_atomic_evidence_writers_refuse_changed_evidence(tmp_path):
    path = tmp_path / 'receipt.json'
    write_once(path, {'a': 1})
    write_once(path, {'a': 1})
    with pytest.raises(RuntimeError, match='differs'):
        write_once(path, {'a': 2})
    md = tmp_path / 'report.md'
    write_text_once(md, 'Actual result\n')
    write_text_once(md, 'Actual result\n')
    with pytest.raises(RuntimeError, match='differs'):
        write_text_once(md, 'Different result\n')
    assert not list(tmp_path.glob('.report*'))


def test_unregistered_case_is_rejected_before_world_construction(monkeypatch):
    from wirelessjev.semantic_jepa import study
    def forbidden(*args, **kwargs):
        raise AssertionError('Unregistered world was opened')
    monkeypatch.setattr(study, 'episode', forbidden)
    with pytest.raises(ValueError, match='Unregistered'):
        study.case((('confirmation', 745, .03, 'semantic_jepa', 0), False))
    with pytest.raises(ValueError, match='Unregistered'):
        study.validate_case_key(('development', 2103000, .03, 'semantic_jepa', False))


def test_changed_development_verdict_cannot_open_confirmation(tmp_path, monkeypatch):
    from wirelessjev.semantic_jepa import study, report, common
    import json
    monkeypatch.setattr(common, 'OUT', tmp_path)
    monkeypatch.setattr(common, 'check_sources', lambda: {})
    monkeypatch.setattr(study, 'case_keys', lambda stage: [('fixture',)])
    monkeypatch.setattr(study, 'load_case', lambda key: {'causality_pass': True})
    monkeypatch.setattr(report, 'development_gate', lambda *args, **kwargs: {'passed_costs': []})
    (tmp_path / 'development.json').write_text(json.dumps({'passed_costs': [.03]}))
    def forbidden(*args, **kwargs):
        raise AssertionError('Rejected confirmation data was opened')
    monkeypatch.setattr(study, 'execute', forbidden)
    with pytest.raises(RuntimeError, match='raw archives'):
        study.confirmation()
