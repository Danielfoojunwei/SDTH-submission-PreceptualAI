"""Registration and byte guards for the scene-model research track."""
from __future__ import annotations
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/radiojepa'
WORK = ROOT.parent / 'training/radiojepa-scene'
PROTOCOL = OUT / 'protocol.json'
PREVIOUS_COMMIT = 'd533b0add1ecd9046a0e4b42a60cd553c1937c21'
PREVIOUS_MANIFEST = 'artifacts/pilot_fastloop/verification/manifest.json'
PREVIOUS_SHA = '612d2451782675959671157f91e147422632d291b7ff6c4f20743725fa6ec94c'
_PUBLICATION = set()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def value_digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    return write_bytes_once(path, encoded(value))


def write_bytes_once(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise RuntimeError('Refuse to overwrite evidence: ' + str(path))
    return path


def safe_path(name, root=None):
    root = ROOT if root is None else Path(root)
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise RuntimeError('Unsafe evidence path')
    full = root / p
    if not full.resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Evidence escaped its root')
    for parent in (full, *full.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise RuntimeError('Redirected evidence path')
    return full


def protocol():
    return read_json(PROTOCOL)


def data_root(source):
    """Relocate byte-identical downloaded inputs without changing identities."""
    return Path(os.environ.get('RADIOJEPA_DATA_ROOT',source['root']))


def previous_manifest():
    path = safe_path(PREVIOUS_MANIFEST)
    if digest(path) != PREVIOUS_SHA:
        raise RuntimeError('Previous sealed manifest changed')
    return read_json(path)


def verify_previous():
    m = previous_manifest()
    if len(m['sources_sha256']) != 395 or len(m['evidence_sha256']) != 793:
        raise RuntimeError('Previous closure cardinality changed')
    for field in ('sources_sha256', 'evidence_sha256'):
        for name, expected in m[field].items():
            if digest(safe_path(name)) != expected:
                raise RuntimeError('Previous scientific bytes changed: ' + name)
    visible=set()
    for folder in ('artifacts/pilot_fastloop','artifacts/jepa_advantage_diagnosis'):
        base=ROOT/folder
        visible.update(str(p.relative_to(ROOT)) for p in base.rglob('*') if p.is_file()
                       and not any(x.startswith('.') for x in p.relative_to(base).parts)
                       and not p.name.endswith(('.lock','.part')))
    visible.discard(PREVIOUS_MANIFEST)
    if visible!=set(m['evidence_sha256']):
        raise RuntimeError('Previous scientific evidence membership changed')
    for name, expected in m['documents_sha256'].items():
        raw = subprocess.check_output(['git', 'show', PREVIOUS_COMMIT + ':' + name], cwd=ROOT)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError('Historical document changed')
    receipt = read_json(ROOT / 'artifacts/pilot_fastloop/registration/source.json')
    capsule = ROOT / 'artifacts/pilot_fastloop/registration/source'
    mapping = receipt['source_sha256']
    if len(mapping) != 467 or {str(p.relative_to(capsule)) for p in capsule.rglob('*') if p.is_file()} != set(mapping):
        raise RuntimeError('Previous source capsule membership changed')
    for name, expected in mapping.items():
        if digest(safe_path(name, capsule)) != expected:
            raise RuntimeError('Previous source capsule changed')
    return {'commit': PREVIOUS_COMMIT, 'manifest_sha256': PREVIOUS_SHA,
            'sources': 395, 'evidence': 793, 'capsule': 467}


def source_hashes():
    result = dict(previous_manifest()['sources_sha256'])
    files = []
    for directory in ('src/wirelessjev/radiojepa', 'src/wirelessjev/worldmodel', 'src/wirelessjev/jev'):
        files += [p for p in (ROOT / directory).rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.lock')]
    files += list((ROOT / 'tests').glob('test_radiojepa*.py'))
    files += list((ROOT / 'tests').glob('test_scene*.py'))
    files += list((ROOT / 'scripts').glob('*radiojepa*.py'))
    files += [p for p in (ROOT / 'docs').glob('RADIOJEPA_*.md') if p.name != 'RADIOJEPA_RESULTS.md']
    files += [ROOT / 'docs/plans/RADIOJEPA_USER_PLAN.md', PROTOCOL, OUT / 'plan.json',
              OUT / 'datasets.json', OUT / 'rt_registry.json', OUT / '.gitignore',
              ROOT / 'artifacts/warehouse_v2/studies/wifi_link_curves.json']
    for path in sorted(set(files)):
        result[str(path.relative_to(ROOT))] = digest(path)
    return dict(sorted(result.items()))


def register():
    previous = verify_previous()
    for name in ('data','predictions','probes','fits', 'checkpoints', 'results', 'rt_data'):
        if (OUT / name).exists() and any((OUT / name).rglob('*')):
            raise RuntimeError('Source registration must precede experiment outputs')
    if any((OUT/name).exists() for name in ('baselines.json','report.json','training.json')):
        raise RuntimeError('Source registration must precede experiment reports')
    sources = source_hashes()
    preflight_path=OUT/'preflight/tests.json'
    preflight=read_json(preflight_path)
    if (preflight.get('exit_code')!=0 or preflight.get('sources_before_sha256')!=sources
            or preflight.get('sources_after_sha256')!=sources
            or preflight.get('prior_evidence')!=previous):
        raise RuntimeError('Final tests must pass on exactly the source being registered')
    stdout=safe_path(preflight['stdout_path'])
    if digest(stdout)!=preflight['stdout_sha256']:
        raise RuntimeError('Final test log changed')
    capsule = OUT / 'registration/source'
    for name, expected in sources.items():
        target = safe_path(name, capsule)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if digest(target) != expected:
                raise RuntimeError('Registered source capsule differs')
        else:
            shutil.copyfile(safe_path(name), target)
    receipt = {'schema': 1, 'source_sha256': sources, 'previous': previous,
               'preflight_sha256':{str(preflight_path.relative_to(ROOT)):digest(preflight_path),
                                   str(stdout.relative_to(ROOT)):digest(stdout)},
               'scope': 'pretraining and comparative experiments; public acquisition/probes and software fixtures precede registration',
               'radio_map_default_unchanged': True}
    write_once(OUT / 'registration/source.json', receipt)
    return receipt


def check_sources():
    receipt = read_json(OUT / 'registration/source.json')
    if verify_previous()!=receipt['previous']:
        raise RuntimeError('Previous evidence identity changed')
    for name,sha in receipt['preflight_sha256'].items():
        if digest(safe_path(name))!=sha:
            raise RuntimeError('Published preflight proof changed')
    if receipt['source_sha256'] != source_hashes():
        raise RuntimeError('Registered scene-model source changed')
    capsule = OUT / 'registration/source'
    expected = receipt['source_sha256']
    if {str(p.relative_to(capsule)) for p in capsule.rglob('*') if p.is_file()} != set(expected):
        raise RuntimeError('Scene source capsule membership changed')
    for name, sha in expected.items():
        if digest(safe_path(name, capsule)) != sha:
            raise RuntimeError('Scene source capsule bytes changed')
    published = read_json(OUT / 'registration/git.json')
    if published['source_receipt_sha256'] != digest(OUT / 'registration/source.json'):
        raise RuntimeError('Publication receipt changed')
    commit = published['commit']
    if (not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit)
            or not published.get('branch') or not published.get('remote')
            or published.get('remote_observed_commit')!=commit):
        raise RuntimeError('Publication must identify an observed immutable remote commit')
    key = (commit, value_digest(published))
    if key not in _PUBLICATION:
        names = dict(expected)
        names.update(receipt['preflight_sha256'])
        names['artifacts/radiojepa/registration/source.json'] = published['source_receipt_sha256']
        names.update({'artifacts/radiojepa/registration/source/' + n: sha for n, sha in expected.items()})
        proc = subprocess.Popen(['git', 'archive', '--format=tar', commit, *names], cwd=ROOT, stdout=subprocess.PIPE)
        actual = set()
        with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
            for item in archive:
                if item.isdir():
                    continue
                if not item.isfile() or item.name not in names:
                    raise RuntimeError('Unexpected published source member')
                if hashlib.sha256(archive.extractfile(item).read()).hexdigest() != names[item.name]:
                    raise RuntimeError('Published source bytes differ')
                actual.add(item.name)
        if proc.wait() or actual != set(names):
            raise RuntimeError('Published source capsule is incomplete')
        _PUBLICATION.add(key)
    return receipt


def authorize_stage(stage):
    check_sources()
    from .rt_data import fingerprint, geometry_registry
    registry_sha = fingerprint(geometry_registry())
    proof = {'stage': stage, 'authorized': True, 'registry_sha256': registry_sha}
    if stage == 'confirmation':
        path = OUT / 'qualification.json'
        if not path.exists():
            raise RuntimeError('Reserved confirmation geometry remains unopened')
        gate = read_json(path)
        if (gate.get('confirmation_permitted') is not True
                or gate.get('registry_sha256') != registry_sha
                or gate.get('source_receipt_sha256') != digest(OUT / 'registration/source.json')):
            raise RuntimeError('No qualification for reserved confirmation geometry')
        proof['confirmation_permitted'] = True
    elif stage not in ('training', 'validation', 'development'):
        raise ValueError('Unknown stage')
    return proof
