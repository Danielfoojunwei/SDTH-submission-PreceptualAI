"""Apply declared schema defaults to unchanged actual Qwen outputs.

This audits an over-strict validator. It never fills mission-dependent values,
strips formatting, calls another model, or changes the generated text.
"""
from pathlib import Path
import argparse
import json
from wirelessjev.warehouse_v2.experiment import atomic_json,sha256
from wirelessjev.warehouse_v3.missions import Contract,compile_contract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('/workspace/training/warehouse-v3-generative'))
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    original=json.loads((args.run/'generative.json').read_text());rows=[]
    for row in original['readings']:
        result={'text':row['text'],'raw_output_unchanged':row['raw_output'],'original_strict_exact_match':row['exact_contract_match']}
        try:
            contract=Contract.from_dict(json.loads(row['raw_output'].strip()))
            result.update(schema_valid_with_declared_defaults=True,exact_contract_match=contract==compile_contract(row['text']),contract=contract.to_dict())
        except (KeyError,TypeError,ValueError) as error:
            result.update(schema_valid_with_declared_defaults=False,exact_contract_match=False,error=str(error))
        rows.append(result)
    atomic_json(args.out/'schema_audit.json',{'source':'deterministic_schema_audit_of_actual_unchanged_model_outputs',
        'parent_report_sha256':sha256(args.run/'generative.json'),'original_strict_accuracy':original['exact_schema_accuracy'],
        'accuracy_with_declared_defaults':sum(r['exact_contract_match'] for r in rows)/len(rows),'readings':rows,
        'defaults':['Absent nullable group fields use None; absent IDs use []','A missing non-tail fraction is 0.1',
                    'Absent floors mean no floors; absent conditional priority means none','Non-group goals default to all robots'],
        'model_executed_in_this_audit':False,'label_guided_repairs':False,
        'limits':['Designed after finding the over-strict validation issue; original exact-schema score is retained',
            'Unknown or conflicting fields remain errors; mission-specific missing values are never recovered from labels',
            'Malformed JSON, truncation and omitted real constraints still count as failures']})
    atomic_json(args.out/'status.json',{'state':'completed','report_sha256':sha256(args.out/'schema_audit.json')})


if __name__=='__main__':main()
