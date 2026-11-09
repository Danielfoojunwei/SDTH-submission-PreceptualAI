"""Run a constructed two-client hidden-transmitter sensitivity experiment."""
from pathlib import Path
import argparse
import json
import subprocess
import time
from wirelessjev.warehouse_v2.experiment import atomic_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--binary',type=Path,default=Path('/workspace/upstream/warehouse-v2/ns3/build-contracts/scratch/ns3.49-warehouse-contracts-optimized'))
    args=p.parse_args();root=args.out;root.mkdir(parents=True,exist_ok=True)
    path=root/'topology.txt'
    lines=['WIRELESSJEV_PACKET_V2 2 1 15','60 40 36 1 0.02 18']
    for x in [32,88]:lines.append('4.1 0.1 0 0 '+' '.join([f'{x} 40']*7))
    path.write_text('\n'.join(lines)+'\n');records=[]
    for seed in [170000,170001,170002]:
        for rts in [False,True]:
            out=root/f'{seed}-{rts}.json';start=time.perf_counter()
            subprocess.run([str(args.binary),f'--input={path.resolve()}',f'--output={out.resolve()}',f'--seed={seed}',
                '--duration=8',f'--rts={int(rts)}','--burst=0'],check=True,timeout=180,capture_output=True)
            report=json.loads(out.read_text());rows=report['per_robot']
            records.append({'seed':seed,'rts_cts':rts,'wall_s':time.perf_counter()-start,
                'critical_generated':sum(r['critical_tx'] for r in rows),
                'critical_misses':sum(r['critical_tx']-r['critical_rx']+r['critical_late'] for r in rows),
                'received_bytes':sum(r['delivered_bytes'] for r in rows),'retry_attempts':sum(r['phy_retry_attempts'] for r in rows),
                'report_sha256':sha256(out)})
    atomic_json(root/'hidden_terminal.json',{'source':'actual_ns3_packet_simulation','binary_sha256':sha256(args.binary),
        'input_sha256':sha256(path),'records':records,'robots':2,
        'geometry':'Clients 56m apart, each 28m horizontally from AP; seeks a hidden-transmitter regime under declared log-distance radio',
        'limits':['Constructed mechanism sensitivity, not a 100-robot fleet result',
            'RTS/CTS is a simulator-side capability comparison, not enabled on customer APs',
            'No guarantee RTS improves every topology or traffic regime']})
    atomic_json(root/'status.json',{'state':'completed','report_sha256':sha256(root/'hidden_terminal.json')})
    print('Executed six two-client native sensitivity runs')


if __name__=='__main__':main()
