"""Reproduce training-only frozen-model representation diagnostics."""
import argparse
import json
from pathlib import Path

from wirelessjev.representation_audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.root)
    dest = args.output or args.root/'artifacts/representation_audit/report.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'completed': True, 'packet_models': len(result['packet_models']),
                      'training_examples': result['packet_inputs']['examples'],
                      'native_windows': result['untrained_service_model']['windows'],
                      'report': str(dest)}, sort_keys=True))


if __name__ == '__main__':
    main()
