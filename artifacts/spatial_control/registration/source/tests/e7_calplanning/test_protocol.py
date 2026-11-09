from pathlib import Path

import numpy as np
import pytest

from wirelessjev.e7_calplanning import common as c,study,verification as v


def test_fixed_scope_budgets_and_no_confirmation_claim():
    p=c.protocol_spec()
    assert p['seeds']==list(range(15,30)) and p['variants']==['corrected_v2','geometry_censored']
    assert p['finite_budget']=={'map_fits':90,'belief_searches':450,'truth_reference_searches':75,'total_searches':525,
        'maximum_search_iterations':21000,'paired_rows':225,'environment_units':15}
    assert 'TAIL' in p['budget_semantics'] and 'NOT chronological' in p['budget_semantics']
    assert p['production_qualified'] is False and p['original_SERVICE_holdouts_permitted'] is False
    assert p['planner']['reference'].endswith('not global optimum')


def test_generation_refuses_unpublished_source_before_any_world(monkeypatch):
    def refuse():raise RuntimeError('unpublished')
    monkeypatch.setattr(c,'check_sources',refuse)
    monkeypatch.setattr(study,'random_env',lambda *_:pytest.fail('must not create environment'))
    with pytest.raises(RuntimeError,match='unpublished'):study.generate_inputs(15)
    with pytest.raises(RuntimeError,match='unpublished'):study.run_environment(15)


def test_durable_exclusive_attempt_rejects_interrupts_and_bad_seeds(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path/'out');monkeypatch.setattr(c,'PROTOCOL',tmp_path/'protocol.json')
    c.write_once(c.PROTOCOL,c.protocol_spec());c.write_once(c.OUT/'registration/source.json',{'fixture':True})
    path=study.begin_attempt(15)
    assert c.read_json(path)['maximum_searches']==35
    with pytest.raises(RuntimeError,match='Partial'):study.begin_attempt(15)
    directory=c.OUT/'cases/16';directory.mkdir();(directory/'partial.npz').write_bytes(b'interrupted')
    with pytest.raises(RuntimeError,match='Partial'):study.begin_attempt(16)
    with pytest.raises(ValueError):study.begin_attempt(True)
    with pytest.raises(ValueError):study.begin_attempt(14)


def test_generation_consumes_canonical_attempt_once_before_any_generator(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path/'out');monkeypatch.setattr(c,'PROTOCOL',tmp_path/'protocol.json')
    monkeypatch.setattr(c,'check_sources',lambda:None)
    c.write_once(c.PROTOCOL,c.protocol_spec());c.write_once(c.OUT/'registration/source.json',{'fixture':True})
    with pytest.raises(RuntimeError,match='unused canonical'):study.generate_inputs(15)
    with pytest.raises(ValueError):study.generate_inputs(0)
    study.begin_attempt(15)
    class StopBeforeWorld(Exception):pass
    def stop(_):raise StopBeforeWorld
    monkeypatch.setattr(study,'random_env',stop)
    with pytest.raises(StopBeforeWorld):study.generate_inputs(15)
    assert (c.OUT/'cases/15/generation.json').exists()
    with pytest.raises(RuntimeError,match='unused canonical'):study.generate_inputs(15)


def test_source_capsule_explicitly_includes_dynamic_calibrator(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'PROTOCOL',tmp_path/'protocol.json')
    monkeypatch.setattr(c,'PREVIOUS',tmp_path/'previous.json');monkeypatch.setattr(c,'CALIBRATOR',tmp_path/'docs/experiments/e7_calibration_study.py')
    monkeypatch.setattr(c,'OLD_REPORTS',());monkeypatch.setattr(c,'preserved',lambda:{'source_sha256':{}})
    # Redirect module source locator to an empty package under the synthetic root.
    module_dir=tmp_path/'src/wirelessjev/e7_calplanning';module_dir.mkdir(parents=True)
    monkeypatch.setattr(c,'__file__',str(module_dir/'common.py'))
    for path in (c.PROTOCOL,c.PREVIOUS,c.CALIBRATOR,module_dir/'common.py'):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixed bytes')
    result=c.sources()
    assert result['docs/experiments/e7_calibration_study.py']==c.digest(c.CALIBRATOR)
    assert 'src/wirelessjev/e7_calplanning/common.py' in result


def test_canonical_reference_rejects_checksum_consistent_wrong_path(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    correct=tmp_path/'cases/15/survey.npz';correct.parent.mkdir(parents=True);correct.write_bytes(b'same')
    wrong=tmp_path/'cases/16/survey.npz';wrong.parent.mkdir();wrong.write_bytes(b'same')
    v.reference({'path':'cases/15/survey.npz','sha256':c.digest(correct)},correct)
    with pytest.raises(RuntimeError,match='canonical'):
        v.reference({'path':'cases/16/survey.npz','sha256':c.digest(wrong)},correct)
    with pytest.raises(RuntimeError,match='changed'):
        v.reference({'path':'cases/15/survey.npz','sha256':'0'*64},correct)


def test_all_harms_signed_deltas_and_environment_units_survive_summary():
    rows=[{'seed':seed,'paired_rows':[{'budget':b,'mission':m,'candidate_minus_v2':(-1. if seed%2 else 2.)}
           for b in c.BUDGETS for m in c.MISSIONS]} for seed in c.SEEDS]
    result=study.summarize(rows)
    for item in result.values():
        assert item['environment_units']==15 and item['mission_pairs']==75
        assert item['environment_harms']==8 and item['mission_pair_harms']==40
        assert item['per_environment_delta'][0]==-1.
        assert np.isclose(item['mean_environment_utility_delta'],6/15)
    c.encoded(result)
    with pytest.raises(ValueError):study.summarize(rows[:-1])


def test_action_guard_preserves_partial_visible_moves_but_rejects_mixed_nonvisible_starts():
    shared={'m':np.asarray(3),'visible':np.zeros((100,3),bool),'starts':np.stack((np.zeros(100,int),np.ones(100,int)))}
    shared['visible'][:,2]=True
    valid=shared['starts'][0].copy();valid[:5]=2
    v.allowed_actions(valid,shared)
    invalid=shared['starts'][0].copy();invalid[:5]=1
    with pytest.raises(RuntimeError,match='outside'):v.allowed_actions(invalid,shared)


def test_passive_closure_refuses_extra_files_or_changed_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out')
    monkeypatch.setattr(c,'check_sources',lambda:None);monkeypatch.setattr(c,'sources',lambda:{'fixed.py':'f'*64})
    monkeypatch.setattr(v,'audit',lambda:{'passes':True})
    fixture=c.OUT/'saved.json';c.write_once(fixture,{'fixture':True})
    v.seal();assert v.verify_closure()['passes']
    extra=c.OUT/'extra.json';extra.write_text('extra')
    with pytest.raises(RuntimeError,match='sealed closure'):v.verify_closure()
    extra.unlink();fixture.write_text('changed')
    with pytest.raises(RuntimeError,match='sealed closure'):v.verify_closure()


def test_first_run_creates_verification_directory_before_passive_audit(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out')
    monkeypatch.setattr(c,'PROTOCOL',c.OUT/'protocol.json');monkeypatch.setattr(c,'check_sources',lambda:None)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(key,'1')
    c.write_once(c.PROTOCOL,c.protocol_spec());c.write_once(c.OUT/'registration/source.json',{'software_fixture':True})
    rows={seed:{'seed':seed,'paired_rows':[{'budget':b,'mission':m,'candidate_minus_v2':0.}
        for b in c.BUDGETS for m in c.MISSIONS]} for seed in c.SEEDS}
    for seed,row in rows.items():c.write_once(c.OUT/'cases'/str(seed)/'receipt.json',row)
    monkeypatch.setattr(study,'run_environment',lambda seed:rows[seed])
    class StopBeforeActualAudit(Exception):pass
    def stop():
        assert (c.OUT/'verification').is_dir()
        assert {p.name for p in (c.OUT/'cases').iterdir()}=={str(seed) for seed in c.SEEDS}
        raise StopBeforeActualAudit
    monkeypatch.setattr(v,'audit',stop)
    with pytest.raises(StopBeforeActualAudit):study.run(workers=1)
