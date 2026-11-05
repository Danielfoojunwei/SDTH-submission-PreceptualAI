"""Operational fixtures only; no research worlds, models or registered edits."""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import types
import pytest
WRAPPER = Path('/workspace/SDTH-submission-PreceptualAI/scripts/run_pilot_fastloop_cached.py')
spec = importlib.util.spec_from_file_location('pilot_cached_wrapper', WRAPPER)
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)

@contextmanager
def real_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)

@pytest.fixture
def fixture(tmp_path):
    root, out = tmp_path, tmp_path / 'artifacts/pilot_fastloop'; out.mkdir(parents=True)
    def write(name, text='fixed'):
        path = out / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text); return path
    for name in w.MANDATORY: write(name)
    write('registration/source/uv.lock', 'locked dependency bytes')
    write('registration/source/src/common.py', 'registered scientific code')
    for folder in w.DEVELOPMENT_FOLDERS:
        write(folder + '/development-unit.data'); write(folder + '/development-unit.lock', 'flock')
    common_file = root / 'src/wirelessjev/pilot_fastloop/common.py'
    common_file.parent.mkdir(parents=True); common_file.write_text('original function source')
    wrapper_file = root / 'scripts/run_pilot_fastloop_cached.py'
    wrapper_file.parent.mkdir(); shutil.copyfile(WRAPPER, wrapper_file)
    state = types.SimpleNamespace(original_calls=mp.Value('i', 0), source_calls=mp.Value('i', 0),
        gate={'passed_costs': [4.], 'nested': {'value': [1, 2]}}, mutate=None, reject_sources=False)
    def sources():
        with state.source_calls.get_lock(): state.source_calls.value += 1
        if state.reject_sources: raise RuntimeError('Original source guard rejected')
        return {'fixed_sources': True}
    def original():
        with state.original_calls.get_lock(): state.original_calls.value += 1
        if state.mutate: state.mutate()
        return state.gate
    def once(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream: json.dump(value, stream, sort_keys=True)
    common = types.SimpleNamespace(ROOT=root, OUT=out, __file__=str(common_file),
        check_sources=sources, verified_development=original, write_once=once, lock=real_lock)
    cache = w.DevelopmentGateCache(common, 'confirmation', wrapper_path=wrapper_file)
    return types.SimpleNamespace(root=root, out=out, write=write, common=common, cache=cache,
                                 state=state, wrapper=wrapper_file)

def test_original_once_unchanged_gate_and_source_guard_every_reuse(fixture):
    f = fixture; original = deepcopy(f.state.gate)
    assert f.cache.bootstrap() == original
    assert f.cache() == f.cache() == original
    assert f.state.original_calls.value == 1 and f.state.source_calls.value == 4
    receipt = json.loads(f.cache.bootstrap_path.read_text())
    assert receipt['before_sha256'] == receipt['after_sha256'] == w.bound_files(f.common)
    assert receipt['gate'] == original and receipt['original_full_verified_development'] is True
    hits = [json.loads(line) for line in f.cache.hits_path.read_text().splitlines()]
    assert len(hits) == 2
    assert all(row['bootstrap_sha256'] == w._sha256(f.cache.bootstrap_path) for row in hits)
    assert all(row['source_guard_checked'] and row['bytes_and_membership_checked'] for row in hits)


def test_deepcopies_detach_original_bootstrap_and_every_return(fixture):
    f = fixture; bootstrap = f.cache.bootstrap()
    bootstrap['nested']['value'].append(3); f.state.gate['nested']['value'].append(4)
    first = f.cache(); first['nested']['value'].append(5)
    assert f.cache()['nested']['value'] == [1, 2]


def test_source_guard_rejection_never_returns_cached_permission(fixture):
    f = fixture; f.cache.bootstrap(); f.state.reject_sources = True
    with pytest.raises(RuntimeError, match='Original source guard'): f.cache()
    assert f.state.source_calls.value == 3 and not f.cache.hits_path.exists()


@pytest.mark.parametrize('name', list(w.MANDATORY) + [
    'registration/source/uv.lock', 'registration/source/src/common.py',
    'cases/development-unit.data', 'matched/cases/development-unit.data',
    'matched/predictions/development-unit.data', 'matched/inputs/development-unit.data'])
@pytest.mark.parametrize('operation', ['modify', 'delete'])
def test_every_bound_dependency_tamper_or_missing_fails(fixture, name, operation):
    f = fixture; f.cache.bootstrap(); path = f.out / name
    if operation == 'modify': path.write_text('changed')
    else: path.unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)): f.cache()
    assert f.state.original_calls.value == 1 and not f.cache.hits_path.exists()


@pytest.mark.parametrize('folder', ['registration/source'] + list(w.DEVELOPMENT_FOLDERS))
def test_bound_membership_addition_fails_without_rebuilding(fixture, folder):
    f = fixture; f.cache.bootstrap(); f.write(folder + '/development-added.data')
    with pytest.raises(RuntimeError, match='membership changed'): f.cache()
    assert f.state.original_calls.value == 1


def test_confirmation_additions_and_namespace_lock_changes_do_not_invalidate(fixture):
    f = fixture; expected = f.cache.bootstrap()
    for folder in w.DEVELOPMENT_FOLDERS:
        f.write(folder + '/confirmation-new.data'); f.write(folder + '/development-unit.lock', 'new flock')
    assert f.cache() == expected


def test_modification_during_original_full_verification_fails_before_install(fixture):
    f = fixture; original = f.common.verified_development
    f.state.mutate = lambda: f.write('matched/inputs/development-unit.data', 'changed during fullverification')
    with pytest.raises(RuntimeError, match='during original verification'): f.cache.bootstrap()
    assert f.common.verified_development is original and not f.cache.bootstrap_path.exists()
    assert not f.cache._ready


@pytest.mark.parametrize('target', ['wrapper', 'common', 'bootstrap'])
def test_operational_or_original_code_and_receipt_tamper_fails(fixture, target):
    f = fixture; f.cache.bootstrap()
    path = f.wrapper if target == 'wrapper' else Path(f.common.__file__) if target == 'common' else f.cache.bootstrap_path
    path.write_text('changed')
    with pytest.raises(RuntimeError, match='membership changed'): f.cache()


def test_bootstrap_once_install_only_after_original_verification(fixture):
    f = fixture
    with pytest.raises(RuntimeError, match='complete first'): f.cache()
    with pytest.raises(RuntimeError, match='Bootstrap the original'): f.cache.install()
    f.cache.bootstrap(); f.cache.install()
    assert f.common.verified_development is f.cache
    with pytest.raises(RuntimeError, match='exactly once'): f.cache.bootstrap()


@pytest.mark.parametrize('phase', ['development', 'all', 'register', 'anything'])
def test_only_confirmation_and_report_accepted(fixture, phase):
    with pytest.raises(ValueError): w.DevelopmentGateCache(fixture.common, phase)


def test_fork_workers_inherit_cache_original_once_atomic_receipts(fixture):
    f = fixture; expected = f.cache.bootstrap(); f.cache.install(); f.cache()
    context = mp.get_context('fork'); queue = context.Queue()
    def child():
        for _ in range(4):
            gate = f.common.verified_development(); assert gate == expected
            gate['nested']['value'].append(os.getpid())
        queue.put((os.getpid(), f.common.verified_development()))
    children = [context.Process(target=child) for _ in range(2)]
    for process in children: process.start()
    results = [queue.get(timeout=10) for _ in children]
    for process in children: process.join(10); assert process.exitcode == 0
    assert all(gate == expected for pid, gate in results)
    assert f.state.original_calls.value == 1 and f.state.source_calls.value == 13
    hits = [json.loads(line) for line in f.cache.hits_path.read_text().splitlines()]
    assert len(hits) == 11 and len({row['pid'] for row in hits}) == 3
    child_hits = [row for row in hits if row['pid'] != os.getpid()]
    assert all(row['cache_parent_pid'] == os.getpid() for row in child_hits)
    assert all(sorted(row['process_hit_index'] for row in child_hits if row['pid'] == pid) == list(range(1, 6))
               for pid, gate in results)
    assert f.cache._gate == expected


def test_bootstrap_membership_addition_rejected(fixture):
    f = fixture; f.state.mutate = lambda: f.write('registration/source/added.py', 'new')
    with pytest.raises(RuntimeError, match='during original verification'): f.cache.bootstrap()
    assert not f.cache._ready


def test_cli_rejects_development_before_import_or_work():
    with pytest.raises(SystemExit): w.main(['development'])
    with pytest.raises(SystemExit): w.main(['confirmation', '--workers', '1'])


@pytest.mark.parametrize('phase', ['confirmation', 'report'])
@pytest.mark.parametrize('failure', [False, True])
def test_main_calls_only_original_phase_then_restores_function_and_argv(fixture, monkeypatch, phase, failure):
    import sys
    f = fixture; original = f.common.verified_development; calls = []
    factory = w.DevelopmentGateCache
    monkeypatch.setattr(w, 'DevelopmentGateCache', lambda common, selected:
        factory(common, selected, wrapper_path=f.wrapper))
    package = types.ModuleType('wirelessjev.pilot_fastloop'); package.common = f.common
    def phase_main():
        calls.append(list(sys.argv))
        assert f.common.verified_development() == f.state.gate
        assert f.common.verified_development() == f.state.gate
        if failure: raise RuntimeError('Original phase failed')
    package.study = types.SimpleNamespace(main=phase_main)
    monkeypatch.setitem(sys.modules, 'wirelessjev.pilot_fastloop', package)
    monkeypatch.setattr(mp, 'get_start_method', lambda **kwargs: 'fork')
    previous_argv = sys.argv
    if failure:
        with pytest.raises(RuntimeError, match='Original phase failed'): w.main([phase])
    else: w.main([phase])
    assert calls == [[str(WRAPPER), phase, '--workers', '2']]
    assert f.common.verified_development is original and sys.argv is previous_argv
    assert f.state.original_calls.value == 1


def test_main_refuses_nonfork_before_package_import(monkeypatch):
    monkeypatch.setattr(mp, 'get_start_method', lambda **kwargs: 'spawn')
    with pytest.raises(RuntimeError, match='fork inheritance'): w.main(['confirmation'])
