"""Read saved E7 receipts only; never import a model, fitter or planner.

This reviewer checks arithmetic, pairing, signs, identities and preserved raw
reference utilities. It does not independently recompute the physical utility.
Run --self-test for analytic arithmetic checks; --wait waits for all receipts
and report before writing this review's own audit.json outside the sealed run.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import struct
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'artifacts/e7_calplanning'
REVIEW=ROOT/'artifacts/e7_calplanning_review'
SEEDS=tuple(range(15,30))
BUDGETS=(1,5,15)
MISSIONS=('teleop_assist','peak_picking','map_rollout','shift_fairness','scanner_coexistence')
SOURCE_SHA='7559bda6e136e39bb9befb74cc68d1ca9595823f6b7376ec1b843999784d46df'
PROTOCOL_SHA='adf503f90507043759ecf76c62fd9c720bbe9ba87a60c1ba2afbc9e2a12662da'
PUBLICATION_SHA='0cedab85ccaaaf833b079d8a5b81c0a4a21869a5a31b482014815e3f049c68a7'
COMMIT='8549a00e4f7148949109bba63a2a17befb22ec93'
SCOPE='ORACLE-CONDITIONED DESCRIPTIVE; previously opened environment seeds; not deployable qualification'
FLOAT_KEYS=('corrected_v2','geometry_censored','candidate_minus_v2','truth_search_reference',
            'legacy','v2_signed_gap','candidate_signed_gap')


def require(condition,message):
    if not condition:raise RuntimeError(message)


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
    return value.hexdigest()


def read(path):
    def nonfinite(value):raise RuntimeError('Nonfinite JSON value: '+value)
    with Path(path).open() as stream:return json.load(stream,parse_constant=nonfinite)


def number(value):
    require(type(value) in (int,float) and math.isfinite(value),'Finite arithmetic scalar required')
    return float(value)


def close(actual,expected,message):
    require(math.isclose(number(actual),number(expected),rel_tol=1e-11,abs_tol=1e-10),message)


def read_reference_utilities(path):
    """Read only the saved float vector from its standard NumPy byte format."""
    with zipfile.ZipFile(path) as archive:
        with archive.open('utility.npy') as stream:
            require(stream.read(6)==b'\x93NUMPY','Raw reference array magic changed')
            version=tuple(stream.read(2))
            require(version in ((1,0),(2,0)),'Unsupported raw reference format')
            size=2 if version==(1,0) else 4
            length=int.from_bytes(stream.read(size),'little')
            require(length<65536,'Unexpected reference header')
            header=ast.literal_eval(stream.read(length).decode('latin1'))
            require(header['shape']==(5,) and header['fortran_order'] is False,
                    'Five mission reference utilities required')
            kind={'<f8':('d',8),'<f4':('f',4)}.get(header['descr'])
            require(kind is not None,'Reference utility must be a real floating vector')
            code,width=kind;raw=stream.read()
            require(len(raw)==5*width,'Raw reference utility size changed')
            return tuple(number(x) for x in struct.unpack('<'+5*code,raw))


def check_row(row):
    require(set(row)=={'budget','mission',*FLOAT_KEYS},'Unexpected paired-row fields')
    x={key:number(row[key]) for key in FLOAT_KEYS}
    close(x['candidate_minus_v2'],x['geometry_censored']-x['corrected_v2'],
          'Candidate-minus-v2 sign/arithmetic changed')
    close(x['v2_signed_gap'],x['truth_search_reference']-x['corrected_v2'],
          'V2 signed reference gap changed')
    close(x['candidate_signed_gap'],x['truth_search_reference']-x['geometry_censored'],
          'Candidate signed reference gap changed')
    close(x['candidate_minus_v2'],x['v2_signed_gap']-x['candidate_signed_gap'],
          'Paired utility and signed-gap differences disagree')
    return x


def reduce_receipts(receipts):
    require([r['seed'] for r in receipts]==list(SEEDS),'All fifteen environment units required in registered order')
    rows=[]
    order=[(b,m) for b in BUDGETS for m in MISSIONS]
    for receipt in receipts:
        require([(x['budget'],x['mission']) for x in receipt['paired_rows']]==order,
                'Exactly fifteen distinct budget/mission pairs per environment required')
        for row in receipt['paired_rows']:
            check_row(row);rows.append(dict(row,seed=receipt['seed']))
    require(len(rows)==225,'All 225 paired rows required')
    summary={}
    for budget in BUDGETS:
        selected=[r for r in rows if r['budget']==budget]
        per_environment=[statistics.fmean(r['candidate_minus_v2'] for r in selected if r['seed']==seed)
                         for seed in SEEDS]
        summary[str(budget)]={
            'environment_units':15,'mission_pairs':75,
            'mean_environment_utility_delta':statistics.fmean(per_environment),
            'environment_harms':sum(x<0 for x in per_environment),
            'mission_pair_harms':sum(r['candidate_minus_v2']<0 for r in selected),
            'per_environment_delta':per_environment,
        }
    diagnostic={str(budget):{
        'mission_pair_improvements':sum(r['candidate_minus_v2']>0 for r in rows if r['budget']==budget),
        'mission_pair_equal':sum(r['candidate_minus_v2']==0 for r in rows if r['budget']==budget),
        'candidate_above_bounded_truth_reference':sum(r['candidate_signed_gap']<0 for r in rows if r['budget']==budget),
        'v2_above_bounded_truth_reference':sum(r['v2_signed_gap']<0 for r in rows if r['budget']==budget),
        'negative_candidate_truth_utilities':sum(r['geometry_censored']<0 for r in rows if r['budget']==budget),
        'negative_v2_truth_utilities':sum(r['corrected_v2']<0 for r in rows if r['budget']==budget),
        'mean_candidate_truth_utility':statistics.fmean(r['geometry_censored'] for r in rows if r['budget']==budget),
        'mean_v2_truth_utility':statistics.fmean(r['corrected_v2'] for r in rows if r['budget']==budget),
        'mean_candidate_signed_gap':statistics.fmean(r['candidate_signed_gap'] for r in rows if r['budget']==budget),
        'mean_v2_signed_gap':statistics.fmean(r['v2_signed_gap'] for r in rows if r['budget']==budget),
        'minimum_candidate_minus_v2':min(r['candidate_minus_v2'] for r in rows if r['budget']==budget),
        'maximum_candidate_minus_v2':max(r['candidate_minus_v2'] for r in rows if r['budget']==budget),
    } for budget in BUDGETS}
    return summary,diagnostic


def ready():
    missing=[seed for seed in SEEDS if not (RUN/'cases'/str(seed)/'receipt.json').is_file()]
    return {'complete_receipts':15-len(missing),'missing_seeds':missing,
            'aggregate_report_ready':(RUN/'report.json').is_file()}


def audit():
    status=ready();require(not status['missing_seeds'] and status['aggregate_report_ready'],
                          'Wait for all fifteen immutable receipts and aggregate report')
    require(digest(RUN/'registration/source.json')==SOURCE_SHA,'Published source receipt changed')
    require(digest(RUN/'protocol.json')==PROTOCOL_SHA,'Published protocol changed')
    require(digest(RUN/'registration/git.json')==PUBLICATION_SHA,'Publication receipt changed')
    publication=read(RUN/'registration/git.json')
    require(publication['commit']==publication['remote_observed_commit']==COMMIT,'Publication commit mismatch')
    protocol=read(RUN/'protocol.json')
    require(protocol['seeds']==list(SEEDS) and protocol['budgets']==list(BUDGETS)
            and protocol['missions']==list(MISSIONS) and protocol['scope']==SCOPE,'Protocol pairing changed')
    present={x.name for x in (RUN/'cases').iterdir() if x.is_dir()}
    require(present=={str(x) for x in SEEDS},'Unregistered/missing environment directory')
    receipts=[];inputs={};raw_hashes={};iterations=0
    for seed in SEEDS:
        directory=RUN/'cases'/str(seed);path=directory/'receipt.json';receipt=read(path)
        require(type(receipt['seed']) is int and receipt['seed']==seed,'Environment identity mismatch')
        require(receipt['source_receipt_sha256']==SOURCE_SHA and receipt['protocol_sha256']==PROTOCOL_SHA
                and receipt['scope']==SCOPE,'Case source/protocol/scope mismatch')
        require(receipt['fits']==6 and receipt['belief_searches']==30 and receipt['reference_searches']==5,
                'Per-environment fit/search count changed')
        require(receipt['production_qualified'] is False and receipt['default_changed'] is False
                and receipt['original_SERVICE_holdouts_opened'] is False
                and receipt['new_packet_simulations']==0 and receipt['no_fresh_confirmation_claim'] is True,
                'Case scope or qualification changed')
        for reference in receipt['refs'].values():
            require(set(reference)=={'path','sha256'},'Evidence reference schema changed')
            relative=Path(reference['path'])
            require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe raw evidence path')
            evidence=ROOT/relative
            require(evidence.resolve().parent==directory.resolve() and not evidence.is_symlink(),
                    'Raw evidence must remain in its canonical environment')
            require(digest(evidence)==reference['sha256'],'Saved raw evidence bytes changed')
            raw_hashes[reference['path']]=reference['sha256']
        reference=receipt['refs']['truth_references']
        utilities=read_reference_utilities(ROOT/reference['path'])
        for row in receipt['paired_rows']:
            close(row['truth_search_reference'],utilities[MISSIONS.index(row['mission'])],
                  'Paired row differs from saved raw truth-reference utility')
        stats=read(ROOT/receipt['refs']['choice_stats']['path'])+receipt['reference_stats']
        require(len(stats)==35,'Thirty belief and five reference search records required')
        for item in stats:
            require(type(item['iterations']) is int and 0<=item['iterations']<=40,
                    'Search iteration budget changed')
            iterations+=item['iterations']
        require(len(receipt['fit_records'])==6,'Six declared fit records required')
        inputs[str(seed)]=digest(path);receipts.append(receipt)
    require(iterations<=21000,'Global fixed search-iteration budget exceeded')
    summary,diagnostic=reduce_receipts(receipts)
    report=read(RUN/'report.json')
    require(report['source_receipt_sha256']==SOURCE_SHA and report['protocol_sha256']==PROTOCOL_SHA
            and report['scope']==SCOPE and report['seeds']==list(SEEDS),'Aggregate report identity changed')
    require(report['environment_receipts']==inputs,'Aggregate receipt hashes differ')
    require(report['map_fits']==90 and report['belief_searches']==450 and report['reference_searches']==75
            and report['paired_rows']==225 and report['environment_units']==15,'Aggregate budget changed')
    require(report['production_qualified'] is False and report['default_changed'] is False
            and report['original_SERVICE_holdouts_opened'] is False,'Aggregate scope flags changed')
    require(set(report['summary'])==set(summary),'Aggregate budget groups changed')
    for budget,expected in summary.items():
        actual=report['summary'][budget]
        require(set(actual)==set(expected),'Aggregate summary fields changed')
        for key in expected:
            if key=='per_environment_delta':
                require(len(actual[key])==15,'All environment deltas required')
                for a,b in zip(actual[key],expected[key]):close(a,b,'Environment mean differs')
            elif key=='mean_environment_utility_delta':close(actual[key],expected[key],'Aggregate mean differs')
            else:require(actual[key]==expected[key],'Aggregate harm/unit count differs')
    return {'passes':True,'scope':SCOPE,'source_commit':COMMIT,'source_receipt_sha256':SOURCE_SHA,
        'protocol_sha256':PROTOCOL_SHA,'publication_receipt_sha256':PUBLICATION_SHA,
        'reviewer_source_sha256':digest(Path(__file__)),'report_sha256':digest(RUN/'report.json'),
        'environment_receipts':inputs,'raw_evidence_sha256':raw_hashes,'environment_units':15,'paired_rows':225,
        'search_iterations_recorded':iterations,'summary':summary,'sign_diagnostics':diagnostic,
        'method':'stdlib-only independent pairing/arithmetic reducer; saved NPY reference vector read as bytes',
        'recomputed_physical_utility':False,'recomputed_fit_or_search':False,'model_forward_passes':0,
        'signed_gaps_clipped':False,'iid_mission_or_robot_significance_claim':False,
        'production_qualified':False,'default_changed':False,'original_SERVICE_holdouts_opened_by_reviewer':False}


def self_test():
    receipts=[]
    for seed in SEEDS:
        rows=[]
        for budget in BUDGETS:
            for mission in MISSIONS:
                base=-.25;delta=-.1 if seed==15 else .2;candidate=base+delta;reference=-.1
                rows.append({'budget':budget,'mission':mission,'corrected_v2':base,
                    'geometry_censored':candidate,'candidate_minus_v2':delta,'truth_search_reference':reference,
                    'legacy':-.4,'v2_signed_gap':reference-base,'candidate_signed_gap':reference-candidate})
        receipts.append({'seed':seed,'paired_rows':rows})
    summary,diagnostic=reduce_receipts(receipts)
    for budget in BUDGETS:
        require(summary[str(budget)]['environment_harms']==1,'Analytic environment harm check failed')
        require(summary[str(budget)]['mission_pair_harms']==5,'Analytic mission harm check failed')
        close(summary[str(budget)]['mean_environment_utility_delta'],.18,'Analytic reduction check failed')
        require(diagnostic[str(budget)]['candidate_above_bounded_truth_reference']==70,
                'Negative signed gaps must remain visible')
    bad=dict(receipts[0]['paired_rows'][0],candidate_minus_v2=.1)
    try:check_row(bad)
    except RuntimeError:pass
    else:raise RuntimeError('Reversed utility sign was not rejected')
    return {'passes':True,'analytic_fixtures_only':True,'scientific_functions_called':False}


def export_rows(audit_value):
    """Lossless passive convenience copy; the original receipts remain authoritative."""
    stream=io.StringIO(newline='')
    fields=('seed','budget','mission',*FLOAT_KEYS,'receipt_sha256')
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
    count=0
    for seed in SEEDS:
        path=RUN/'cases'/str(seed)/'receipt.json';expected=audit_value['environment_receipts'][str(seed)]
        require(digest(path)==expected,'Authoritative receipt changed before CSV export')
        receipt=read(path)
        for row in receipt['paired_rows']:
            writer.writerow(dict(row,seed=seed,receipt_sha256=expected));count+=1
        require(digest(path)==expected,'Authoritative receipt changed during CSV export')
    require(count==225,'All 225 authoritative paired rows required for CSV export')
    raw=stream.getvalue().encode();destination=REVIEW/'paired_rows.csv'
    try:
        with destination.open('xb') as file:file.write(raw)
    except FileExistsError:require(destination.read_bytes()==raw,'Independent CSV is immutable')
    return {'path':str(destination.relative_to(ROOT)),'sha256':digest(destination),'paired_rows':count,
            'authoritative_sources':'per-environment case receipts, bound by receipt_sha256 column'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true')
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--status',action='store_true')
    args=parser.parse_args()
    if args.self_test:value=self_test()
    elif args.status:value=ready()
    else:
        if args.wait:
            while True:
                state=ready()
                if not state['missing_seeds'] and state['aggregate_report_ready']:break
                print(json.dumps({'waiting':state}),flush=True);time.sleep(15)
        value=audit();value['paired_rows_csv']=export_rows(value);destination=REVIEW/'audit.json'
        raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        try:
            with destination.open('xb') as stream:stream.write(raw)
        except FileExistsError:require(destination.read_bytes()==raw,'Independent review is immutable')
    print(json.dumps(value,sort_keys=True),flush=True)


if __name__=='__main__':main()
