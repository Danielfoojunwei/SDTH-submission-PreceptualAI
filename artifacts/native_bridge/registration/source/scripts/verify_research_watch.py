"""Read-only validation of bibliographic snapshots and provider response bytes."""
import argparse,hashlib,json
from pathlib import Path
from wirelessjev.research_watch import validate_snapshot,MAX_RESPONSE_BYTES,canonical_json,parse_time


def verify(directory):
    directory=Path(directory);latest=directory/'latest.json'
    if latest.stat().st_size>20*1024*1024:raise ValueError('Snapshot exceeds storage cap')
    snapshot=validate_snapshot(json.loads(latest.read_text()));checked=set();history={}
    for path in sorted((directory/'snapshots').glob('*.json')):
        if path.stat().st_size>20*1024*1024:raise ValueError('Historical snapshot exceeds storage cap')
        item=validate_snapshot(json.loads(path.read_text()))
        history[hashlib.sha256(canonical_json(item)).hexdigest()]=item
    head=hashlib.sha256(canonical_json(snapshot)).hexdigest()
    if head not in history:raise ValueError('Latest snapshot has no immutable historical copy')
    # Check every predecessor, including any valid interrupted-publication
    # branch, rather than validating only today's successful response bodies.
    for start in history:
        seen=set();current=start
        while current is not None:
            if current in seen:raise ValueError('Historical predecessor cycle')
            if current not in history:raise ValueError('Missing historical predecessor snapshot')
            seen.add(current);item=history[current];parent=item['previous_snapshot_sha256']
            if parent in history and parse_time(history[parent]['generated_at'])>parse_time(item['generated_at']):raise ValueError('Historical predecessor follows its child')
            current=parent
    for item in history.values():
        for request in item['request_results']:
            for attempt in request['attempts']:
                if 'response_sha256' not in attempt:continue
                name='responses/'+attempt['response_sha256']+'.bin'
                if attempt.get('response_artifact')!=name:raise ValueError('Provider response must be retained')
                path=directory/name
                if path.stat().st_size!=attempt['response_bytes'] or path.stat().st_size>MAX_RESPONSE_BYTES:raise ValueError('Retained response size differs')
                if hashlib.sha256(path.read_bytes()).hexdigest()!=attempt['response_sha256']:raise ValueError('Retained response checksum differs')
                checked.add(name)
        for record in item['records']:
            for observation in record['observations']:
                expected=observation['source_response_sha256'];name='responses/'+expected+'.bin'
                if name in checked:continue
                path=directory/name
                if path.stat().st_size>MAX_RESPONSE_BYTES or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Historical observation provenance response differs')
                checked.add(name)
    for path in (directory/'responses').glob('*'):
        if not path.is_file() or path.name.startswith('.'):continue
        if path.name!=hashlib.sha256(path.read_bytes()).hexdigest()+'.bin' or path.stat().st_size>MAX_RESPONSE_BYTES:raise ValueError('Historical response filename/checksum differs')
        checked.add('responses/'+path.name)
    return {'valid_snapshot':True,'historical_snapshots_verified':len(history),'retained_provider_responses_verified':len(checked),
        'leads':len(snapshot['records']),'incomplete_coverage':snapshot['coverage']['incomplete_coverage'],
        'automatic_review_or_promotion':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(verify(args.output_dir),indent=2))
