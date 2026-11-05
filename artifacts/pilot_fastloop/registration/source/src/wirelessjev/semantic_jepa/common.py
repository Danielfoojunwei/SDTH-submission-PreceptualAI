"""Immutable study identities; previous scientific sources remain untouched."""
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/semantic_jepa'
WORK = ROOT.parent / 'training/semantic-jepa'
PROTOCOL = OUT / 'protocol.json'


def protocol():
    return json.loads(PROTOCOL.read_text())


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def value_digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def write_once(path, value, *, compressed=False):
    data = encoded(value)
    if compressed:
        data = gzip.compress(data, mtime=0)
    _write_bytes_once(path, data)


def write_text_once(path, text):
    _write_bytes_once(path, text.encode())


def _write_bytes_once(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'Immutable evidence differs: {path}')
        return
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise RuntimeError(f'Concurrent evidence differs: {path}')
    finally:
        os.unlink(temporary)


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def source_hashes():
    from ..frontier.study import sources
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((ROOT / 'src/wirelessjev/closed_loop').glob('*.py'))
    paths += list((ROOT / 'tests').glob('test_semantic_jepa*.py'))
    paths += [PROTOCOL, ROOT / 'src/wirelessjev/hardware_budget.py',
              ROOT / 'src/wirelessjev/fleet_control.py',
              ROOT / 'artifacts/fleet_world_model/selection.json']
    paths += [ROOT / f'artifacts/{name}/protocol.json' for name in
              ('fleet_world_model', 'contracted', 'closed_loop', 'hardware_budget')]
    paths += sorted((ROOT / 'artifacts/fleet_world_model/checkpoints').glob('hybrid_jepa-*.pt'))
    return sources() | {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def versions():
    return {'python': platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ('numpy', 'scipy', 'torch')}}


def verify_splits():
    p = protocol()
    names = ('training_sites', 'validation_sites', 'development_sites', 'confirmation_sites')
    sets = [set(p[name]) for name in names] + [{p['calibration_site']}]
    if sum(map(len, sets)) != len(set().union(*sets)):
        raise RuntimeError('Overlapping scientific splits')
    def numbers(value):
        if isinstance(value, dict):
            return set().union(*(numbers(v) for v in value.values())) if value else set()
        if isinstance(value, list):
            return set().union(*(numbers(v) for v in value)) if value else set()
        return {value} if isinstance(value, int) else set()
    current = set().union(*sets)
    for path in (ROOT / 'artifacts').glob('*/protocol.json'):
        if path != PROTOCOL and current & numbers(json.loads(path.read_text())):
            raise RuntimeError(f'New sites overlap prior registration: {path}')


def register():
    verify_splits()
    path = OUT / 'registration/source.json'
    with locked(OUT / 'registration/.lock'):
        if path.exists():
            return check_sources()
        hashes = source_hashes()
        for name, expected in hashes.items():
            target = OUT / 'registration/source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(target) != expected:
                    raise RuntimeError(f'Partial capsule differs: {name}')
            else:
                shutil.copyfile(ROOT / name, target)
        value = {'schema': 1, 'time_unix': time.time(), 'source_sha256': hashes,
                 'software_versions': versions(), 'data_accessed': False,
                 'scientific_source_frozen': True, 'training_fits': 12}
        write_once(path, value)
        return value


def check_sources():
    value = json.loads((OUT / 'registration/source.json').read_text())
    if value['source_sha256'] != source_hashes() or value['software_versions'] != versions():
        raise RuntimeError('Frozen source/protocol/software changed; refusing mixed evidence')
    for name, expected in value['source_sha256'].items():
        if digest(OUT / 'registration/source' / name) != expected:
            raise RuntimeError(f'Source capsule changed: {name}')
    return value
