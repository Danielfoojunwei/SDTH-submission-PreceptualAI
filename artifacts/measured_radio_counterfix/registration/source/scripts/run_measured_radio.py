"""Finite source-published measured RSSI assay; no training or hardware access."""
from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('register', 'publish-proof', 'run', 'verify'))
    parser.add_argument('--source-commit')
    args = parser.parse_args()
    if args.command == 'register':
        from wirelessjev.measured_radio.registration import register
        result = register()
    elif args.command == 'publish-proof':
        from wirelessjev.measured_radio.registration import publication
        if not args.source_commit:
            parser.error('--source-commit is required')
        result = publication(args.source_commit)
    elif args.command == 'run':
        from wirelessjev.measured_radio.study import run
        result = run()
    else:
        from wirelessjev.measured_radio import common as c
        from wirelessjev.measured_radio.study import guard, dataset, validation, select, testing, report, seal
        guard()
        value = dataset()
        if not (c.OUT/'report.json').exists():
            raise RuntimeError('An executed report is required; verify never starts a new scientific arm')
        for partition in ('validation', 'test'):
            if any(not (c.OUT/'predictions'/f'{partition}-locations{b}.json').exists() for b in c.BUDGETS):
                raise RuntimeError('Complete frozen prediction groups are required')
        validation(value)
        selected = select()
        testing(value, selected)
        report(value, selected)
        result = {'passes': True, 'sealed_files': len(seal()['files_sha256']), 'new_scientific_prediction_calls': 0}
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
