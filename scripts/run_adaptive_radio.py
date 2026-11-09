#!/usr/bin/env python3
"""Run/audit the finite source-published adaptive RF-condition experiment."""
import argparse
import json
from wirelessjev.adaptive_radio import common,study,verification


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('register','run','audit','seal','verify','status'))
    parser.add_argument('--workers',type=int,default=2,choices=(1,2));args=parser.parse_args()
    if args.action=='register':value=common.register()
    elif args.action=='run':value=study.run(args.workers)
    elif args.action=='audit':value=verification.run_audit()
    elif args.action=='seal':value=verification.seal()
    elif args.action=='verify':value=verification.verify_closure()
    else:value={'registered':(common.OUT/'registration/source.json').exists(),
        'published':(common.OUT/'registration/git.json').exists(),
        'completed_fits':len(list((common.OUT/'fits').glob('*.json'))),
        'development':common.read_json(common.OUT/'development.json') if (common.OUT/'development.json').exists() else None,
        'sealed':(common.OUT/'verification/manifest.json').exists(),'original_service_confirmation_opened':False}
    print(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
