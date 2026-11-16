"""Portable desktop entrypoint with a Torch-free, explicitly shadow-only model."""
import argparse
import json
from pathlib import Path
import sys

from ..smartcar_runtime import workflow as reference
from ..smartcar_runtime.contracts import load_records, site_sha256, validate_query, validate_site


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != 'shadow':
        return reference.main(argv)
    parser = argparse.ArgumentParser(description='Numerical RadioJEPA shadow; no Torch, actuation or calibrated packet risk.')
    parser.add_argument('command', choices=('shadow',))
    parser.add_argument('--site', required=True)
    parser.add_argument('--survey', required=True)
    parser.add_argument('--xyz', nargs=3, type=float, required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    try:
        from .runtime import predict_numpy
        site = validate_site(reference.read_json(args.site))
        records = load_records(args.survey, site)
        point = validate_query(site, args.xyz)
        manifest_path = Path(args.models).resolve()
        manifest = reference.read_json(manifest_path)
        members = manifest.get('members')
        pins = reference.RELEASED_CHECKPOINTS
        if (manifest.get('schema') != 1 or not isinstance(members, list) or len(members) != 3
                or {r.get('name') for r in members if isinstance(r, dict)} != set(pins)
                or any(r.get('source_checkpoint_sha256') != pins.get(r.get('name')) for r in members)):
            raise ValueError('The three released source-checkpoint identities are required')
        result = {'schema': 1, 'site_sha256': site_sha256(site), 'backend': 'numpy_scipy',
                  'experimental_shadow_only': True, 'sent_to_car': False,
                  'hardware_accuracy_measured': False, 'members': {}}
        for member in members:
            path = (manifest_path.parent/member['path']).resolve()
            if not path.is_relative_to(manifest_path.parent):
                raise ValueError('Export path escapes the approved model directory')
            result['members'][member['name']] = predict_numpy(site, records, [point], path,
                member['sha256'], member['source_checkpoint_sha256'])
        reference.write_document(args.out, result)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print('Shadow inference interrupted; no command was sent.', file=sys.stderr)
        return 130
    except Exception:
        print('Numerical shadow failed. Check the admitted survey, site and exact model exports.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
