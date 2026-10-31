"""Prospective canonical-clause reader audit after whole-family failure."""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import json
import shutil
import torch
from wirelessjev.policy import WirelessJEV,MODEL_REPO,MODEL_REVISION
from wirelessjev.warehouse_v3.clauses import ClauseReader,ClauseAbstention
from wirelessjev.warehouse_v2.experiment import atomic_json,sha256

TEXTS=[
 'Switch at most 3 robots; require service >= 40% for robots [amr-001, amr-017] at every offset; minimise critical violations; then maximise throughput.',
 'Switch at most 7 robots; require service >= 25% for teleoperated robots at every offset; minimise switches; then maximise throughput.',
 'Switch at most 9 robots; require service >= 65% for video robots at every offset; maximise worst-served 15% service; then minimise switches.',
 'Switch at most 6 robots; if any teleoperated robot has observed service < 35%, prioritise that group; minimise switches; then maximise throughput.',
 'Switch at most 11 robots; if any video robot has observed service < 55%, prioritise that group; minimise critical violations; then maximise throughput.',
 'Switch at most 4 robots; maximise service for robots with x >= 35 m; then minimise switches.',
 'Switch at most 2 robots; maximise service for robots with y < 25 m; then maximise throughput.',
 'Switch at most 0 robots; maximise worst-served 5% service; then minimise critical violations.',
 'Switch at most 10 robots; require service >= 15% for robots with y >= 30 m at every offset; maximise throughput.',
 'Switch at most 1 robots; require service >= 5% for all robots at every offset; minimise switches; then maximise throughput.',
 'Switch at most 13 robots; minimise critical violations; then maximise throughput.',
 'Switch at most 14 robots; maximise service for robots [amr-002, amr-099]; then minimise switches.'
]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    root=args.out;root.mkdir(parents=True,exist_ok=True);repo=Path(__file__).resolve().parents[1]
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    paths=[Path(__file__),repo/'src/wirelessjev/warehouse_v3/clauses.py',repo/'src/wirelessjev/warehouse_v3/missions.py',repo/'src/wirelessjev/policy.py']
    hashes={str(p.relative_to(repo)):sha256(p) for p in paths}
    for p in paths:
        target=root/'source_snapshot'/p.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    atomic_json(root/'provenance.json',{'source_sha256':hashes,'designed_after_whole_family_reader_failure':True,
        'evaluation_texts_sealed_before_execution':TEXTS,'created_at':datetime.now(timezone.utc).isoformat()})
    engine=WirelessJEV(device='cpu');reader=ClauseReader(engine)
    rows=json.loads((root/'progress.json').read_text())['readings'] if (root/'progress.json').exists() else []
    if [r['text'] for r in rows]!=TEXTS[:len(rows)]:raise RuntimeError('Cannot reuse a different clause audit')
    for row in rows:
        for clause in row['receipt']['clauses']:
            if clause['answer_probabilities'] is not None:
                reader.cache[clause['text'].removeprefix('then ')]=(clause['choice'],clause['answer_probabilities'])
    for text in TEXTS[len(rows):]:
        try:contract,receipt=reader.read(text);row={'text':text,'answered':True,'contract':contract.to_dict(),'receipt':receipt}
        except ClauseAbstention as error:row={'text':text,'answered':False,'reason':str(error),'receipt':error.receipt}
        rows.append(row);atomic_json(root/'progress.json',{'readings':rows});print('Fresh typed clause mission',len(rows),row['answered'],flush=True)
    atomic_json(root/'clauses.json',{'source':'actual_pinned_d3_inference','model':{'repo':MODEL_REPO,'revision':MODEL_REVISION},
        'answered':sum(r['answered'] for r in rows),'total':len(rows),'readings':rows,
        'limits':['Designed after inspecting the failed whole-mission family gate; new canonical statements used for this audit',
            'Literal syntax is established by the complete compiler; model checks individual clause meanings',
            'Small author-written grammar audit, not broad language understanding or a physical control result']})
    atomic_json(root/'status.json',{'state':'completed','report_sha256':sha256(root/'clauses.json')})


if __name__=='__main__':main()
