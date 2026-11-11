"""A retained research lead needs its actual untampered provider response."""
import importlib.util
from pathlib import Path
import shutil
import json,hashlib
import pytest
from wirelessjev.research_watch import canonical_json

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('research_watch_archive_check',ROOT/'scripts/verify_research_watch.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


def test_live_snapshot_remains_metadata_only_with_verifiable_response_bytes():
    result=MODULE.verify(ROOT/'artifacts/ran_research/watch')
    assert result['valid_snapshot'] and result['retained_provider_responses_verified']>0
    assert not result['automatic_review_or_promotion']


def test_response_tampering_fails_even_when_the_lead_json_looks_valid(tmp_path):
    target=tmp_path/'watch';shutil.copytree(ROOT/'artifacts/ran_research/watch',target)
    response=next((target/'responses').glob('*.bin'))
    before=response.read_bytes();response.write_bytes(bytes([before[0]^1])+before[1:])
    with pytest.raises(ValueError,match='checksum|provenance'):MODULE.verify(target)


def write_snapshot(target,name,snapshot):
    (target/'snapshots'/name).write_text(json.dumps(snapshot))
    return hashlib.sha256(canonical_json(snapshot)).hexdigest()


def test_latest_cannot_claim_a_missing_predecessor(tmp_path):
    target=tmp_path/'watch';shutil.copytree(ROOT/'artifacts/ran_research/watch',target)
    head=json.loads((target/'latest.json').read_text());head['previous_snapshot_sha256']='a'*64
    write_snapshot(target,'modified-head.json',head);(target/'latest.json').write_text(json.dumps(head))
    with pytest.raises(ValueError,match='predecessor'):MODULE.verify(target)


def test_old_failure_response_is_checked_even_when_not_referenced_by_latest_requests(tmp_path):
    target=tmp_path/'watch';shutil.copytree(ROOT/'artifacts/ran_research/watch',target)
    head=json.loads((target/'latest.json').read_text());old=json.loads(json.dumps(head))
    body=b'explicit historical provider failure fixture';digest=hashlib.sha256(body).hexdigest()
    path=target/'responses'/(digest+'.bin');path.write_bytes(body)
    query=old['request_results'][0];attempt=query['attempts'][0]
    attempt.update(http_status=500,response_sha256=digest,response_bytes=len(body),response_artifact='responses/'+digest+'.bin')
    query.update(retrieval_status='failed',accepted_entries=0,errors=[{'reason':'HTTP 500'}],http_status=500,source_response_sha256=digest)
    old['coverage'].update(successful_requests=9,complete_for_requested_requests=False,incomplete_coverage=True)
    parent=write_snapshot(target,'historical-failure.json',old)
    head['previous_snapshot_sha256']=parent;write_snapshot(target,'new-head.json',head)
    (target/'latest.json').write_text(json.dumps(head));assert MODULE.verify(target)['historical_snapshots_verified']>=3
    path.write_bytes(b'X'+body[1:])
    with pytest.raises(ValueError,match='checksum'):MODULE.verify(target)
