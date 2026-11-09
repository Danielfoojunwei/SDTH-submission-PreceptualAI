"""Network-free contract tests for real provider formats and failure handling."""
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
from pathlib import Path
import pytest
from wirelessjev import research_watch as watch

NOW = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)


class Clock:
    def __init__(self): self.t = 0.; self.waits = []
    def monotonic(self): return self.t
    def sleep(self, seconds): self.waits.append(seconds); self.t += seconds


def atom(identifier='http://arxiv.org/abs/2610.12345v2', title='Wireless JEPA',
         published='2026-10-01T10:00:00Z', updated='2026-10-02T10:00:00Z', doi='10.1234/wireless'):
    return f'<entry><id>{identifier}</id><title>{title}</title><published>{published}</published><updated>{updated}</updated><author><name>A. Researcher</name></author><arxiv:doi>{doi}</arxiv:doi></entry>'


def feed(entries):
    return ('<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom" '
            'xmlns:os="http://a9.com/-/spec/opensearch/1.1/"><os:totalResults>'+str(len(entries))+'</os:totalResults>'+''.join(entries)+'</feed>').encode()


def work(identifier='W123', doi='https://doi.org/10.1234/wireless', title='Wireless JEPA', published='2026-10-01'):
    return {'id':'https://openalex.org/'+identifier,'doi':doi,'title':title,
            'publication_date':published,'updated_date':'2026-10-02T10:00:00',
            'authorships':[{'author':{'display_name':'A. Researcher'}}],
            'primary_location':{'landing_page_url':None}}


def oa(entries): return json.dumps({'meta':{'count':len(entries)},'results':entries}).encode()


def response(source, body, status=200, headers=None):
    return watch.HTTPResponse(watch.ENDPOINTS[source],status,body,headers or {})


def fake_fetch(url, *, source, timeout):
    return response(source,feed([atom()]) if source=='arxiv' else oa([work()]))


def discover(fetch=fake_fetch, **kwargs):
    clock=Clock()
    return watch.discover(now=NOW,fetch=fetch,sleep=clock.sleep,monotonic=clock.monotonic,**kwargs)


def test_official_origins_and_identifier_urls_do_not_follow_other_hosts():
    for source in watch.ENDPOINTS:
        watch.validate_origin(watch.ENDPOINTS[source]+'?search=a',source)
        for url in ['https://elsewhere.example/api/query','http://export.arxiv.org/api/query',
                    'https://api.openalex.org@elsewhere.example/works','https://api.openalex.org:444/works']:
            with pytest.raises(ValueError):watch.validate_origin(url,source)
    redirect=watch.OfficialRedirects('arxiv')
    with pytest.raises(ValueError):redirect.redirect_request(None,None,302,'',{},'https://elsewhere.example/')
    assert watch.normalize_arxiv('https://arxiv.org/abs/2610.12345v7')=='2610.12345'
    assert watch.normalize_doi('https://doi.org/10.1234/ABC')=='10.1234/abc'
    for value in ['https://arxiv.org/abs/2610.12345?token=x','https://arxiv.org:666/abs/2610.12345']:
        with pytest.raises(ValueError):watch.normalize_arxiv(value)


def test_future_and_pre_window_papers_are_excluded_and_malformed_records_are_partial():
    body=feed([atom(),atom(identifier='2610.12346',published='2027-01-01T00:00:00Z'),
               atom(identifier='2610.12347',updated='2027-01-01T00:00:00Z'),
               atom(identifier='2501.12345',published='2025-01-01T00:00:00Z'),atom(identifier='bad-id')])
    rows,failures,excluded=watch.parse_feed('arxiv',response('arxiv',body),'q',NOW,NOW.date().replace(month=1,day=1))
    assert len(rows)==1 and len(failures)==1
    assert sorted(e['reason'] for e in excluded)==['before_since','future_metadata','future_metadata']
    assert rows[0]['identifiers']['arxiv']==['2610.12345']
    with pytest.raises(ValueError):watch.parse_feed('arxiv',response('arxiv',b'<!DOCTYPE feed><feed/>'),'q',NOW)
    with pytest.raises(ValueError):watch.parse_feed('arxiv',response('arxiv',b'<feed xmlns="http://www.w3.org/2005/Atom"/>'),'q',NOW)
    with pytest.raises(ValueError):watch.parse_feed('openalex',response('openalex',b'{"results":[]}'),'q',NOW)


def test_exact_doi_and_arxiv_dedup_is_cross_source_and_not_title_matching():
    snapshot=discover(since='2026-01-01')
    assert len(snapshot['records'])==1
    record=snapshot['records'][0]
    assert record['id']=='arxiv:2610.12345'
    assert len(record['observations'])==10 and record['identifiers']['openalex']==['W123']
    assert record['review']['status']=='lead' and not snapshot['claims']['sota_established']
    r=response('openalex',oa([work(doi='https://doi.org/10.48550/arXiv.2610.12345')]))
    rows,errors,_=watch.parse_feed('openalex',r,'q',NOW)
    assert not errors and rows[0]['identifiers']['arxiv']==['2610.12345']
    other=deepcopy(rows[0]);other['source_id']='W999';other['identifiers']={'arxiv':[],'doi':['10.1234/different'],'openalex':['W999']}
    records,_=watch.merge_records([record['observations'][0],other],None,NOW)
    assert len(records)==2  # identical title alone never merges records


def test_429_retry_after_is_capped_and_actual_failure_bodies_are_retained(tmp_path):
    clock=Clock();calls=[]
    def fetch(url, *, source, timeout):
        calls.append(url)
        return response(source,b'too many requests',429,{'Retry-After':'999999'}) if len(calls)==1 else response(source,oa([]))
    result,attempts,error=watch.fetch_bounded(watch.ENDPOINTS['openalex'],'openalex',now=NOW,timeout=1,max_attempts=2,
        fetch=fetch,sleep=clock.sleep,monotonic=clock.monotonic,last_request={},retain_response=watch.ResponseArchive(tmp_path))
    assert error is None and result.status==200 and len(attempts)==2 and clock.waits==[5.]
    for attempt in attempts:
        content=(tmp_path/attempt['response_artifact']).read_bytes()
        assert watch.sha256(content)==attempt['response_sha256'] and len(content)==attempt['response_bytes']


def test_failed_sources_are_incomplete_and_retain_previous_leads():
    previous=discover()
    def failure(url, *, source, timeout):return response(source,b'access denied',403)
    snapshot=discover(fetch=failure,previous=previous)
    assert snapshot['coverage']['incomplete_coverage'] and snapshot['coverage']['successful_requests']==0
    assert len(snapshot['records'])==1 and len(snapshot['changes']['not_seen_again'])==1
    assert not snapshot['changes']['deleted'] and all(len(r['attempts'])==1 for r in snapshot['request_results'])
    assert all(r['retrieval_status']=='failed' for r in snapshot['request_results'])
    assert snapshot['records'][0]['retrieval_status']=='retained_previous_not_seen'


def test_overlarge_foreign_and_syntactically_bad_responses_are_not_empty_success():
    def large(url, *, source, timeout):return response(source,b'x'*(watch.MAX_RESPONSE_BYTES+1))
    assert discover(fetch=large)['coverage']['successful_requests']==0
    def foreign(url, *, source, timeout):return watch.HTTPResponse('https://elsewhere.example/',200,b'',{})
    assert discover(fetch=foreign)['coverage']['successful_requests']==0
    def malformed(url, *, source, timeout):return response(source,b'<html>unavailable</html>')
    snapshot=discover(fetch=malformed)
    assert snapshot['coverage']['incomplete_coverage'] and all(r['errors'] for r in snapshot['request_results'])


def test_requested_result_cap_is_enforced_even_when_provider_ignores_it():
    def excessive(url, *, source, timeout):
        return response(source,feed([atom(identifier=f'2610.{12345+i}') for i in range(7)]) if source=='arxiv'
                        else oa([work(identifier=f'W{i}',doi=f'10.1234/x{i}') for i in range(7)]))
    snapshot=discover(fetch=excessive,max_results=5)
    assert snapshot['coverage']['incomplete_coverage']
    assert all(r['accepted_entries']==5 and r['retrieval_status']=='partial' for r in snapshot['request_results'])


def test_run_budget_and_invalid_inputs_do_not_start_unbounded_requests():
    clock=Clock();calls=[]
    def slow(url, *, source, timeout):calls.append(url);clock.t+=2;return response(source,feed([]))
    snapshot=watch.discover(now=NOW,fetch=slow,sleep=clock.sleep,monotonic=clock.monotonic,max_run_seconds=1)
    assert len(calls)==1 and snapshot['coverage']['incomplete_coverage']
    assert len(snapshot['request_results'])==10
    assert all(r['retrieval_status']=='failed' for r in snapshot['request_results'][1:])
    for kwargs in [{'since':'2027-01-01'},{'max_results':51},{'timeout':31},{'max_attempts':4},
                   {'max_run_seconds':601},{'sources':['arxiv','arxiv']}]:
        with pytest.raises(ValueError):discover(**kwargs)


def test_atomic_publish_preserves_history_rejects_stale_writers_and_rejects_bad_previous(tmp_path):
    first=discover();paths=watch.publish_snapshot(first,tmp_path,now=NOW)
    initial=(tmp_path/'latest.json').read_bytes()
    assert watch.sha256(initial)==paths['snapshot_sha256']
    second=discover(previous=first)
    watch.publish_snapshot(second,tmp_path,now=NOW)
    assert (tmp_path/'snapshots'/('previous-'+watch.sha256(watch.canonical_json(first))+'.json')).read_bytes()==initial
    latest=(tmp_path/'latest.json').read_bytes()
    with pytest.raises(ValueError,match='changed during discovery'):watch.publish_snapshot(first,tmp_path,now=NOW)
    assert (tmp_path/'latest.json').read_bytes()==latest
    invalid=deepcopy(second);invalid['records'][0]['title']='Fabricated metadata'
    with pytest.raises(ValueError):watch.validate_snapshot(invalid,NOW)
    (tmp_path/'latest.json').write_text('{"schema_version":0}')
    corrupted=(tmp_path/'latest.json').read_bytes()
    with pytest.raises(ValueError):watch.publish_snapshot(first,tmp_path,now=NOW)
    assert (tmp_path/'latest.json').read_bytes()==corrupted


def test_publish_failure_and_concurrent_writer_never_replace_latest(tmp_path,monkeypatch):
    first=discover();watch.publish_snapshot(first,tmp_path,now=NOW)
    old=(tmp_path/'latest.json').read_bytes();second=discover(previous=first)
    with (tmp_path/'.publish.lock').open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='publisher'):watch.publish_snapshot(second,tmp_path,now=NOW)
    replace=watch.os.replace
    def fail_latest(source,destination):
        if Path(destination).name=='latest.json':raise OSError('simulated disk error')
        return replace(source,destination)
    monkeypatch.setattr(watch.os,'replace',fail_latest)
    with pytest.raises(OSError):watch.publish_snapshot(second,tmp_path,now=NOW)
    assert (tmp_path/'latest.json').read_bytes()==old


def test_review_is_never_automatic_and_changed_metadata_requires_explicit_rereview():
    previous=discover();r=previous['records'][0]
    r['review']={'status':'reviewed','reviewer':'Human reviewer','reviewed_at':'2026-10-31T00:00:00Z',
                 'evidence_url':'https://example.org/review/evidence'}
    same=discover(previous=previous)
    assert same['records'][0]['review']['status']=='reviewed'
    def changed(url, *, source, timeout):
        return response(source,feed([atom(title='Corrected title')]) if source=='arxiv' else oa([work()]))
    updated=discover(fetch=changed,previous=previous)
    assert updated['records'][0]['review']['status']=='lead'
    assert updated['records'][0]['previous_reviews'][0]['reviewer']=='Human reviewer'
    assert updated['changes']['updated']==['arxiv:2610.12345']


def test_cli_publishes_partial_snapshot_but_returns_nonzero_by_default(tmp_path,monkeypatch,capsys):
    path=Path(__file__).resolve().parents[1]/'scripts/update_ran_research.py'
    spec=importlib.util.spec_from_file_location('research_watch_cli',path)
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    partial=discover(fetch=lambda url,source,timeout:response(source,b'no access',403))
    monkeypatch.setattr(cli,'discover',lambda **kwargs:partial)
    assert cli.main(['--output-dir',str(tmp_path)])==2
    assert json.loads((tmp_path/'latest.json').read_text())['coverage']['incomplete_coverage']
    # A second publication needs the previous hash. The opt-in does not erase
    # the coverage warning; only its process exit code changes.
    second=discover(fetch=lambda url,source,timeout:response(source,b'no access',403),previous=partial)
    monkeypatch.setattr(cli,'discover',lambda **kwargs:second)
    assert cli.main(['--output-dir',str(tmp_path),'--allow-incomplete'])==0
    assert capsys.readouterr().out.count('"incomplete_coverage": true')==2


def test_tampered_response_and_false_coverage_cannot_be_published(tmp_path):
    clock=Clock()
    first=watch.discover(now=NOW,fetch=fake_fetch,sleep=clock.sleep,monotonic=clock.monotonic,
                         retain_response=watch.ResponseArchive(tmp_path))
    watch.publish_snapshot(first,tmp_path,now=NOW)
    old=(tmp_path/'latest.json').read_bytes()
    clock=Clock()
    second=watch.discover(now=NOW,fetch=fake_fetch,sleep=clock.sleep,monotonic=clock.monotonic,
                          retain_response=watch.ResponseArchive(tmp_path),previous=first)
    attempt=second['request_results'][0]['attempts'][0]
    (tmp_path/attempt['response_artifact']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='receipt'):watch.publish_snapshot(second,tmp_path,now=NOW)
    assert (tmp_path/'latest.json').read_bytes()==old
    bad=deepcopy(first);bad['coverage']['successful_requests']=0
    with pytest.raises(ValueError,match='Coverage'):watch.validate_snapshot(bad,NOW)


def test_trickling_response_has_a_whole_response_elapsed_cap(monkeypatch):
    clock=Clock()
    class Trickle:
        code=200;headers={}
        def __enter__(self):return self
        def __exit__(self,*args):return None
        def geturl(self):return watch.ENDPOINTS['arxiv']
        def read1(self,n):clock.t+=.6;return b'x'
    class Opener:
        def open(self,*args,**kwargs):return Trickle()
    monkeypatch.setattr(watch,'build_opener',lambda *args:Opener())
    monkeypatch.setattr(watch.time,'monotonic',clock.monotonic)
    with pytest.raises(TimeoutError,match='Whole discovery response'):
        watch.urllib_fetch(watch.ENDPOINTS['arxiv'],source='arxiv',timeout=1)
    assert clock.t==1.2
