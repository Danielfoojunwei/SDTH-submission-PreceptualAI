"""Discover RAN/JEPA literature into a local artifact directory only."""
import argparse
import json
from pathlib import Path

from wirelessjev.research_watch import discover, publish_snapshot, load_snapshot, ResponseArchive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--since', default=None, help='Optional inclusive publication date, YYYY-MM-DD')
    parser.add_argument('--sources', nargs='+', choices=('arxiv', 'openalex'), default=['arxiv', 'openalex'])
    parser.add_argument('--max-results', type=int, default=5, help='Per source/query; hard maximum 50')
    parser.add_argument('--timeout', type=float, default=10., help='Per request; hard maximum 30 seconds')
    parser.add_argument('--max-attempts', type=int, default=2, help='Hard maximum 3 attempts')
    parser.add_argument('--max-run-seconds', type=float, default=180., help='Stop starting requests at this budget; hard maximum 600 seconds')
    parser.add_argument('--allow-incomplete', action='store_true', help='Keep explicit coverage warnings but return zero after publishing a valid partial snapshot')
    args = parser.parse_args(argv)
    latest = args.output_dir/'latest.json'
    previous = load_snapshot(latest) if latest.exists() else None
    snapshot = discover(previous=previous, since=args.since, sources=args.sources,
                        max_results=args.max_results, timeout=args.timeout, max_attempts=args.max_attempts,
                        max_run_seconds=args.max_run_seconds, retain_response=ResponseArchive(args.output_dir))
    paths = publish_snapshot(snapshot, args.output_dir)
    print(json.dumps({'records': len(snapshot['records']), 'changes': snapshot['changes'],
                      'coverage': snapshot['coverage'], **paths}, indent=2))
    return 2 if snapshot['coverage']['incomplete_coverage'] and not args.allow_incomplete else 0


if __name__ == '__main__':
    raise SystemExit(main())
