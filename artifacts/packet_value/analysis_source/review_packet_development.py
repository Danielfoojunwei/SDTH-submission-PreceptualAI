"""Reproduce the rejected development disposition without consuming holdout."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/packet_value'


def review():
    cases=[json.loads(p.read_text()) for p in (OUT/'cases').glob('development-*.json')]
    if len(cases)!=64 or list((OUT/'cases').glob('test-*.json')) or (OUT/'test_source.json').exists():
        raise RuntimeError('Require the complete development cohort and an untouched confirmation cohort')
    losses={}
    for cost in (.03,4.):
        losses[str(cost)]={family:float(np.mean([c['result']['control_miss'] for c in cases
            if c['cost']==cost and c['family']==family])) for family in ('jepa','supervised','radiomap','canonical')}
    if not all(group['jepa']>group['canonical'] for group in losses.values()):
        raise RuntimeError('Changed results require a new acceptance review; no confirmation is automatically invoked')
    path=OUT/'completion.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old.get('status')=='development_rejected':
            print('Verified development rejection; confirmation remains untouched');return
    diagnostic=json.loads((OUT/'diagnostic.json').read_text())
    record={'completed':True,'status':'development_rejected','development_rejected':True,
        'fits':6,'development_runs':64,'test_runs':0,'fresh_sites':0,'label_native_runs':diagnostic['native_runs'],
        'primary_evaluated':False,'reason':'Learned macro-controller suppresses necessary mobility and loses more commands than canonical RadioMap on both development costs; no independent confirmation is warranted.'}
    path.write_text(json.dumps(record,indent=2)+'\n')
    print('Recorded development rejection; confirmation remains untouched')


if __name__=='__main__':review()
