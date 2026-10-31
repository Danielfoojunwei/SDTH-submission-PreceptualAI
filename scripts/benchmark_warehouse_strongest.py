"""Add the strong RSSI baseline without overwriting prospective evidence."""
from pathlib import Path
import argparse
import torch
from wirelessjev.warehouse_v2.closedloop import rollout
from wirelessjev.warehouse_v2.experiment import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = [rollout(131000+seed, 'strongest_scan') for seed in [0, 1, 2]]
    repo = Path(__file__).resolve().parents[1]
    atomic_json(args.out, {'source': 'closed_loop_warehouse_simulation', 'records': rows,
                          'source_sha256': {str(p.relative_to(repo)): sha256(p) for p in
                                            [Path(__file__), repo/'src/wirelessjev/warehouse_v2/closedloop.py']},
                          'note': 'Additional fixed-policy baseline; existing prospective model reports are preserved'})


if __name__ == '__main__':
    main()
