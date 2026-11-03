"""Execute the committed, fixed-six-fit representation projection ablation."""
import argparse
import json
from pathlib import Path

from wirelessjev.representation_repair import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps({'completed': result['completed'], 'fits': result['fits'],
                      'summary': result['summary']}, indent=2))


if __name__ == '__main__':
    main()
