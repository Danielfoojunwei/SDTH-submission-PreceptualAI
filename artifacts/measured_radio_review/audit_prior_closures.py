"""Passive independent byte replay of the ten previously sealed experiments.

Only stdlib file hashing/archive inspection is used. No scientific code,
checkpoint deserializer, model, dataset preparer, simulator or training routine
is imported or called. Existing evidence and deliverables are never written.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REVIEW = Path(__file__).resolve().parent
PREVIOUS_RECEIPT = 'artifacts/smartcar_numpy_review/prior_closure_verification.json'
PREVIOUS_RECEIPT_SHA256 = 'da54d6d7cd6e6240598343f51019921a97a914685c4d76c92e2c448966096bb2'
EXPECTED_MANIFESTS = {
    'artifacts/adaptive_radio/verification/manifest.json': '912f543b363972b56c54c8210cb44ae79a80fa9220d2950cd5f0fc3a89a0fe2c',
    'artifacts/channel_scene/verification/manifest.json': '7e8dfad6e26d0f62630faf6bacf0a30ce108f5363ac420cb8d2edb08ead6270e',
    'artifacts/device_transfer/verification/manifest.json': 'ef30f07741a13512bad757180d3dec868a9b55f55a836799718fac6d3c9ca2d9',
    'artifacts/e7_calplanning/verification/manifest.json': 'b498272bb37b08c9bbe465eb0be49abf52f9923189d15b0e4f85f84d37628077',
    'artifacts/hierarchical_radio/verification/manifest.json': '3785febce39e419a1b36ab8f122f478724edace4afe4b4d41626468ae0d0a287',
    'artifacts/native_bridge/verification/manifest.json': '4d9f95316d9e029365ed64d4b04e43128a49239c4115ebb8a6bb49f11c12aeb9',
    'artifacts/nested_radio/verification/manifest.json': 'f5e7d7c2947119b5c282c626dbce0a4cef387a47a01110dd63415b6b643978cd',
    'artifacts/radiojepa/verification/manifest.json': 'b9114d4cf4beb65ddd4af63c87e7e154edba5c8f1bba4a56117e1e3f369acf31',
    'artifacts/radiojepa_spectral/verification/manifest.json': '53f329a2ce82d6498116fa098b1231ad31c5826be4435b4610c713701bcf598c',
    'artifacts/spatial_control/verification/manifest.json': 'f652af5c0c1671a978ad0b2ebefcf35d7e5a7474dc36b373d63b4f60aa41d28f',
}
EXPECTED_UNIQUE_FILES = 27295
ZIP_PATH = 'artifacts/smartcar_numpy/smartcar-wirelessJEV.zip'
ZIP_SHA256 = '088d148fbae7780327e52e133bbdf8996af89650ec327f34ef1bfe7cf794165f'
EXPECTED_MODELS = {
    'nested_joint-seed0.npz': '76094add20d506f28bbc76ea47aa7736ae6de8363a86ee1d4a19baf99cb5bdf3',
    'nested_joint-seed1.npz': 'a06a042c4fbb9d589db0d3ddd39bd9512c4e1979c4b96c303e4d135272b133eb',
    'nested_joint-seed2.npz': '35331ae0bbee224cb225b34f6946791b91d9cab0c608a28a530249e8256734e0',
}
FIELDS = ('source_sha256', 'evidence_sha256', 'documents_sha256')
SHA256 = re.compile(r'[0-9a-f]{64}\Z')


def stream_sha(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b''):
        digest.update(block)
    return digest.hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return stream_sha(stream)


def safe_path(base, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('Expected a nonempty relative evidence path')
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError('Missing or escaping evidence path: ' + name)
    return path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def verify():
    started = time.perf_counter()
    previous_path = safe_path(ROOT, PREVIOUS_RECEIPT)
    require(file_sha(previous_path) == PREVIOUS_RECEIPT_SHA256, 'Canonical prior receipt bytes changed')
    previous = read(previous_path)
    require(previous['passes'] and previous['scientific_closures'] == 10
            and previous['manifest_sha256'] == EXPECTED_MANIFESTS
            and previous['unique_named_repository_files'] == EXPECTED_UNIQUE_FILES,
            'Canonical prior receipt identities changed')
    union, manifests, schemas = {}, {}, {}
    for name, expected in EXPECTED_MANIFESTS.items():
        path = safe_path(ROOT, name)
        require(file_sha(path) == expected, 'Prior manifest bytes changed: ' + name)
        manifest = read(path)
        manifests[name] = manifest
        schemas[name] = {}
        for field in FIELDS:
            values = manifest.get(field, {})
            require(isinstance(values, dict), 'Unsupported prior filehash schema: ' + name + '/' + field)
            schemas[name][field] = len(values)
            for item, sha in values.items():
                require(isinstance(sha, str) and SHA256.fullmatch(sha), 'Invalid named evidence hash: ' + item)
                require(item not in union or union[item] == sha, 'Prior manifests disagree: ' + item)
                union[item] = sha
    require(len(union) == EXPECTED_UNIQUE_FILES, 'Prior named repository file membership changed')
    total_bytes = 0
    for name, expected in sorted(union.items()):
        path = safe_path(ROOT, name)
        require(file_sha(path) == expected, 'Prior named file bytes changed: ' + name)
        total_bytes += path.stat().st_size

    # The channel-scene closure separately names an external prediction cache.
    # These 57 shards are additional to the 27,295 repository paths above.
    channel = manifests['artifacts/channel_scene/verification/manifest.json']
    external_root = Path(os.environ.get('CHANNEL_SCENE_CACHE_ROOT', str(ROOT.parent / 'training/channel-scene')))
    external = channel['external_prediction_sha256']
    require(len(external) == 57, 'External prediction shard membership changed')
    for name, expected in external.items():
        require(file_sha(safe_path(external_root, name)) == expected, 'External prediction shard changed: ' + name)
    for key in ('cache_manifest', 'prediction_capsule'):
        ref = channel[key]
        require(file_sha(safe_path(external_root, ref['path'])) == ref['sha256'], 'External deliverable changed: ' + key)
    capsule_ref = channel['prediction_capsule']
    capsule_path = safe_path(external_root, capsule_ref['path'])
    require(capsule_path.stat().st_size == capsule_ref['bytes'], 'External capsule length changed')
    expected_members = dict(external)
    expected_members[channel['cache_manifest']['path']] = channel['cache_manifest']['sha256']
    seen = set()
    with tarfile.open(capsule_path, 'r|gz') as capsule:
        for member in capsule:
            prefix = 'channel-scene-predictions/'
            require(member.name.startswith(prefix), 'External capsule prefix changed')
            name = member.name[len(prefix):]
            require(member.isfile() and name in expected_members and name not in seen,
                    'External capsule membership changed: ' + name)
            require(stream_sha(capsule.extractfile(member)) == expected_members[name],
                    'External capsule member changed: ' + name)
            seen.add(name)
    require(seen == set(expected_members), 'External capsule is incomplete')

    archive = safe_path(ROOT, ZIP_PATH)
    require(file_sha(archive) == ZIP_SHA256, 'Published PR29 ZIP changed')
    model_root = ROOT / 'artifacts/smartcar_numpy/models'
    model_manifest = read(model_root / 'manifest.json')
    require({m['path']: m['sha256'] for m in model_manifest['members']} == EXPECTED_MODELS,
            'Released model identities changed')
    with zipfile.ZipFile(archive) as bundle:
        prefix = 'smartcar-wirelessJEV/'
        require(len(bundle.namelist()) == len(set(bundle.namelist())) == 35, 'Published ZIP membership changed')
        require(bundle.read(prefix + 'models/manifest.json') == (model_root / 'manifest.json').read_bytes(),
                'Released model manifest differs from the immutable ZIP')
        bundle_manifest = json.loads(bundle.read(prefix + 'MANIFEST.json'))
        require(set(bundle.namelist()) == {prefix + name for name in bundle_manifest['files_sha256']}
                | {prefix + 'MANIFEST.json'}, 'Published ZIP manifest membership changed')
        for name, expected in bundle_manifest['files_sha256'].items():
            require(hashlib.sha256(bundle.read(prefix + name)).hexdigest() == expected,
                    'Published ZIP internal member changed: ' + name)
        for name, expected in EXPECTED_MODELS.items():
            require(file_sha(safe_path(model_root, name)) == expected, 'Released NPZ bytes changed: ' + name)
            require(hashlib.sha256(bundle.read(prefix + 'models/' + name)).hexdigest() == expected,
                    'Published NPZ bytes changed: ' + name)

    for name in ('artifacts/radiojepa/rt_data/confirmation', 'artifacts/radiojepa/rt_data/confirmation.json'):
        require(not (ROOT / name).exists(), 'Original SERVICE holdout was opened: ' + name)
    return {'schema': 1, 'passes': True, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'verifier_sha256': file_sha(Path(__file__)), 'prior_receipt_sha256': PREVIOUS_RECEIPT_SHA256,
        'manifest_sha256': EXPECTED_MANIFESTS, 'scientific_closures': 10,
        'filehash_schemas_replayed': list(FIELDS), 'schema_entries_by_manifest': schemas,
        'unique_named_repository_files': len(union), 'unique_named_repository_bytes': total_bytes,
        'external_channel_scene_shards_checked': len(external),
        'external_channel_scene_capsule_sha256': capsule_ref['sha256'],
        'external_channel_scene_capsule_members_checked': len(seen),
        'archive_sha256': ZIP_SHA256, 'archive_bytes': archive.stat().st_size,
        'archive_members_checked': len(bundle_manifest['files_sha256']) + 1,
        'released_model_npz_sha256': EXPECTED_MODELS,
        'model_manifest_sha256': file_sha(model_root / 'manifest.json'),
        'original_SERVICE_holdout_unopened': True, 'scientific_calls': 0,
        'model_deserializations': 0, 'hardware_actions': 0, 'SOTA_established': False,
        'elapsed_seconds': time.perf_counter() - started}


if __name__ == '__main__':
    value = verify()
    with (REVIEW / 'prior_closure_verification.json').open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({key: value[key] for key in ('passes', 'scientific_closures',
        'unique_named_repository_files', 'external_channel_scene_shards_checked', 'archive_sha256')}))
