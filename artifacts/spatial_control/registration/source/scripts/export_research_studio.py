"""Build the separate current-research Studio bundle and standalone local view."""
import argparse
import json

from wirelessjev.research_export.bundle import ROOT,write_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=str,default=str(ROOT))
    parser.add_argument('--cache-root',type=str)
    args=parser.parse_args()
    print(json.dumps(write_bundle(args.root,cache_root=args.cache_root),sort_keys=True))


if __name__=='__main__':main()
