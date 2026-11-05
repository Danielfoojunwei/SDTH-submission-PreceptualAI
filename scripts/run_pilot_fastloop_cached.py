#!/usr/bin/env python3
"""Operational byte-checked development gate reuse; science is unchanged."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

MANDATORY = ('registration/source.json', 'registration/git.json', 'protocol.json',
             'execution_order.json', 'development.json', 'qualification.json')
DEVELOPMENT_FOLDERS = ('cases', 'matched/cases', 'matched/predictions', 'matched/inputs')


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def _value_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def bound_files(common):
    """All capsule files (including uv.lock), seals and development-only data."""
    root, out = Path(common.ROOT).resolve(), Path(common.OUT).resolve()
    paths = [out / name for name in MANDATORY]
    capsule = out / 'registration/source'
    if not capsule.is_dir() or capsule.is_symlink():
        raise RuntimeError('Missing or redirected registered scientific capsule')
    capsule_files = [path for path in capsule.rglob('*') if path.is_file()]
    if not capsule_files:
        raise RuntimeError('Empty registered scientific capsule')
    paths.extend(capsule_files)
    for name in DEVELOPMENT_FOLDERS:
        folder = out / name
        if not folder.is_dir() or folder.is_symlink():
            raise RuntimeError('Missing or redirected development namespace: ' + name)
        paths.extend(path for path in folder.glob('development*') if path.suffix != '.lock')
    value = {}
    for path in sorted(set(paths)):
        if (not path.is_file() or path.is_symlink()
                or not path.resolve().is_relative_to(root)):
            raise RuntimeError('Missing or redirected bound development file: ' + str(path))
        value[str(path.relative_to(root))] = _sha256(path)
    return value


class DevelopmentGateCache:
    """Original full verification once; then source/byte-checked deepcopy reuse."""
    def __init__(self, common, phase, *, wrapper_path=None):
        if phase not in ('confirmation', 'report'):
            raise ValueError('Operational cache supports confirmation/report only')
        self.common, self.phase = common, phase
        self.original = common.verified_development
        self.wrapper_path = Path(__file__ if wrapper_path is None else wrapper_path).resolve()
        self.cache_parent_pid = os.getpid()
        self.cache_id = f'{phase}-{time.time_ns()}-{self.cache_parent_pid}'
        prefix = Path(common.OUT) / 'logs' / ('cached-development-' + self.cache_id)
        self.bootstrap_path = Path(str(prefix) + '.bootstrap.json')
        self.hits_path = Path(str(prefix) + '.cache-hits.jsonl')
        self._ready, self._hits, self._hit_pid = False, 0, os.getpid()

    def bootstrap(self):
        if self._ready:
            raise RuntimeError('Bootstrap must run exactly once in the parent')
        if os.getpid() != self.cache_parent_pid:
            raise RuntimeError('Bootstrap must precede worker fork')
        self.common.check_sources()
        wrapper_sha = _sha256(self.wrapper_path)
        original_path = Path(self.common.__file__).resolve()
        original_sha = _sha256(original_path)
        before = bound_files(self.common)
        gate = self.original()  # Unmodified complete raw archive/array/paid/causal verification.
        self.common.check_sources()
        after = bound_files(self.common)
        if (before != after or _sha256(self.wrapper_path) != wrapper_sha
                or _sha256(original_path) != original_sha):
            raise RuntimeError('Development dependencies changed during original verification')
        if not isinstance(gate, dict):
            raise RuntimeError('Original development permission is not a gate dictionary')
        root = Path(self.common.ROOT).resolve()
        receipt = {'schema': 1, 'phase': self.phase, 'cache_id': self.cache_id,
            'recorded_unix': time.time(), 'cache_parent_pid': self.cache_parent_pid,
            'wrapper_path': str(self.wrapper_path.relative_to(root)), 'wrapper_sha256': wrapper_sha,
            'original_common_path': str(original_path.relative_to(root)),
            'original_common_sha256': original_sha,
            'original_full_verified_development': True, 'source_guard_checked': True,
            'before_sha256': before, 'after_sha256': after,
            'file_count': len(before), 'bound_files_sha256': _value_sha256(before),
            'unchanged': True, 'gate': deepcopy(gate), 'gate_sha256': _value_sha256(gate),
            'registered_scientific_source_changed': False,
            'new_worlds_models_or_training_called_by_wrapper': False,
            'cache_scope': 'Only original verified_development result; source, bytes and membership checked every reuse.'}
        self.common.write_once(self.bootstrap_path, receipt)
        self._bound, self._gate = before, deepcopy(gate)
        self._wrapper_sha, self._original_path, self._original_sha = wrapper_sha, original_path, original_sha
        self._bootstrap_sha = _sha256(self.bootstrap_path)
        self._ready = True
        return deepcopy(self._gate)

    def __call__(self):
        if not self._ready:
            raise RuntimeError('Original parent development verification must complete first')
        self.common.check_sources()
        current = bound_files(self.common)
        if (current != self._bound or _sha256(self.wrapper_path) != self._wrapper_sha
                or _sha256(self._original_path) != self._original_sha
                or _sha256(self.bootstrap_path) != self._bootstrap_sha):
            raise RuntimeError('Cached development permission invalid: source, bytes or membership changed')
        if self._hit_pid != os.getpid():
            self._hits, self._hit_pid = 0, os.getpid()
        self._hits += 1
        hit = {'schema': 1, 'phase': self.phase, 'cache_id': self.cache_id,
            'recorded_unix': time.time(), 'pid': os.getpid(), 'parent_pid': os.getppid(),
            'cache_parent_pid': self.cache_parent_pid, 'process_hit_index': self._hits,
            'bootstrap_path': str(self.bootstrap_path.relative_to(self.common.ROOT)),
            'bootstrap_sha256': self._bootstrap_sha, 'wrapper_sha256': self._wrapper_sha,
            'source_guard_checked': True, 'bytes_and_membership_checked': True,
            'bound_files_sha256': _value_sha256(current), 'file_count': len(current),
            'gate_sha256': _value_sha256(self._gate),
            'original_full_verification_reexecuted': False}
        payload = json.dumps(hit, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'
        with self.common.lock(Path(str(self.hits_path) + '.lock')):
            fd = os.open(self.hits_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                if os.write(fd, payload) != len(payload):
                    raise RuntimeError('Partial operational cache-hit receipt append')
                os.fsync(fd)
            finally:
                os.close(fd)
        return deepcopy(self._gate)

    def install(self):
        if not self._ready:
            raise RuntimeError('Bootstrap the original gate before installation')
        if self.common.verified_development is not self.original:
            raise RuntimeError('Original development verification replaced before cache installation')
        self.common.verified_development = self


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('confirmation', 'report'))
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(argv)
    if args.workers != 2:
        parser.error('Use exactly the two registered workers')
    import multiprocessing
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('fork')
    if multiprocessing.get_start_method() != 'fork':
        raise RuntimeError('Operational cache requires parent-verified fork inheritance')
    from wirelessjev.pilot_fastloop import common, study
    cache = DevelopmentGateCache(common, args.phase)
    cache.bootstrap()
    cache.install()
    previous = sys.argv
    sys.argv = [str(Path(__file__)), args.phase, '--workers', str(args.workers)]
    try:
        study.main()  # Original phase, timing and ordering.
    finally:
        sys.argv = previous
        common.verified_development = cache.original


if __name__ == '__main__':
    main()
