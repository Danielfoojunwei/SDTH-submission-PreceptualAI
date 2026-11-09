"""Build the pinned moving Wi-Fi audit and its actual upstream checks."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import hashlib


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/workspace/upstream/warehouse-v2'))
    p.add_argument('--manifest',type=Path,default=Path('/workspace/.cloud-setup/warehouse-v3-upstreams.json'))
    p.add_argument('--build',action='store_true');args=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('pinned_v2_acquisition',repo/'scripts/setup_warehouse_v2.py')
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    manifest=upstream.acquire(args.root);ns3=args.root/'ns3'
    if args.build:
        source=repo/'third_party/ns3/warehouse-contracts.cc'
        shutil.copyfile(source,ns3/'scratch/warehouse-contracts.cc')
        env=dict(os.environ,PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''))
        subprocess.run([sys.executable,'ns3','configure','--enable-modules=wifi,internet,mobility,applications,flow-monitor',
            '--enable-examples','--enable-tests','--build-profile=optimized','--out=build-contracts'],cwd=ns3,env=env,check=True)
        subprocess.run([sys.executable,'ns3','build','warehouse-contracts','test-runner','-j','2'],cwd=ns3,env=env,check=True)
        runner=ns3/'build-contracts/utils/ns3.49-test-runner-optimized'
        for suite in ['waypoint-mobility-model','wifi-mac-queue','wifi-aggregation']:
            subprocess.run([str(runner),f'--suite={suite}'],cwd=ns3,env=env,check=True)
        binary=ns3/'build-contracts/scratch/ns3.49-warehouse-contracts-optimized'
        manifest['moving_packet_audit']={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'binary':str(binary),
            'profile':'NIST, VHT80/HT20, two streams, public transmit powers, fixed primary channels',
            'upstream_tests_passed':['waypoint-mobility-model','wifi-mac-queue','wifi-aggregation']}
    args.manifest.parent.mkdir(parents=True,exist_ok=True);args.manifest.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Pinned warehouse dependencies verified; no training or car action started')


if __name__=='__main__':main()
