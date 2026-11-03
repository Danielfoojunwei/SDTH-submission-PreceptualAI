"""Bounded bibliographic discovery; never downloads models or promotes findings.

Provider metadata creates unreviewed leads. Official response origins, exact
identifiers and retained snapshots provide provenance, not scientific validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import xml.etree.ElementTree as ET


QUERIES = (
    ('ran_jepa', 'RAN and wireless JEPA', 'all:"JEPA" AND (all:"wireless" OR all:"radio access")', 'JEPA wireless'),
    ('wireless_world_models', 'Wireless world models', 'all:"world model" AND all:"wireless"', 'wireless world model'),
    ('jepa_control', 'Predictive representations for network control', '(all:"JEPA" OR all:"joint embedding predictive") AND (all:"network" OR all:"control")', 'JEPA network control'),
    ('ran_control', 'Radio access network control', 'all:"radio access network" AND (all:"control" OR all:"resource allocation")', 'radio access network control'),
    ('deadline_uncertainty', 'Wireless deadline risk and uncertainty', '(all:"wireless" OR all:"radio access network") AND (all:"deadline" OR all:"uncertainty")', 'wireless deadline uncertainty'),
)
ENDPOINTS = {'arxiv': 'https://export.arxiv.org/api/query', 'openalex': 'https://api.openalex.org/works'}
MAX_RESPONSE_BYTES = 2_000_000
MAX_SNAPSHOT_BYTES = 20_000_000
ARXIV_ID = re.compile(r'(?:\d{4}\.\d{4,5}|[a-zA-Z][a-zA-Z.\-]+/\d{7})(?:v\d+)?\Z')
DOI = re.compile(r'10\.\d{4,9}/[^\s<>]+\Z', re.I)
HEX = re.compile(r'[0-9a-f]{64}\Z')


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def utc_now():
    return datetime.now(timezone.utc)


def stamp(value):
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def parse_time(value):
    if not isinstance(value, str):
        raise ValueError('Timestamp must be a string')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timestamp must include its timezone')
    return result.astimezone(timezone.utc)


def parse_since(value, now):
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Since must be YYYY-MM-DD')
    result = date.fromisoformat(value)
    if result > now.date():
        raise ValueError('Since cannot be after current UTC date')
    return result


def validate_origin(url, source):
    parsed = urlsplit(url)
    expected = urlsplit(ENDPOINTS[source])
    if (parsed.scheme != 'https' or parsed.hostname != expected.hostname
            or parsed.path != expected.path or parsed.username or parsed.password
            or parsed.port not in (None, 443) or parsed.fragment):
        raise ValueError('Response/redirect origin is not the official discovery endpoint')
    return url


def normalize_arxiv(value):
    if not isinstance(value, str):
        raise ValueError('Invalid arXiv identifier')
    if '://' in value:
        parsed = urlsplit(value)
        if parsed.scheme not in ('http', 'https') or parsed.hostname not in ('arxiv.org', 'www.arxiv.org') or parsed.username or parsed.password:
            raise ValueError('arXiv record identifier has a foreign origin')
        if parsed.query or parsed.fragment or parsed.port not in (None, 80, 443):
            raise ValueError('arXiv identifier URL must not contain extra components')
        if not parsed.path.startswith('/abs/'):
            raise ValueError('arXiv identifier must be an abstract URL')
        value = unquote(parsed.path[5:])
    value = value.removeprefix('arXiv:').strip()
    if not ARXIV_ID.fullmatch(value):
        raise ValueError('Invalid arXiv identifier')
    return re.sub(r'v\d+$', '', value).lower()


def normalize_doi(value):
    if value is None or value == '':
        return None
    if not isinstance(value, str):
        raise ValueError('Invalid DOI')
    value = value.strip()
    if '://' in value:
        parsed = urlsplit(value)
        if parsed.scheme not in ('http', 'https') or parsed.hostname not in ('doi.org', 'dx.doi.org') or parsed.username or parsed.password:
            raise ValueError('DOI URL has a foreign origin')
        if parsed.query or parsed.fragment or parsed.port not in (None, 80, 443):
            raise ValueError('DOI URL must not contain extra components')
        value = unquote(parsed.path.lstrip('/'))
    value = re.sub(r'^doi:\s*', '', value, flags=re.I)
    if len(value) > 512 or not DOI.fullmatch(value):
        raise ValueError('Invalid DOI')
    return value.lower()


class OfficialRedirects(HTTPRedirectHandler):
    max_redirections = 2
    max_repeats = 1
    def __init__(self, source):
        super().__init__()
        self.source = source

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_origin(newurl, self.source)  # reject before following it
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass
class HTTPResponse:
    url: str
    status: int
    body: bytes
    headers: dict


def urllib_fetch(url, *, source, timeout):
    validate_origin(url, source)
    request = Request(url, headers={'User-Agent': 'WirelessJEVResearchWatch/1.0 (bibliographic discovery)',
                                    'Accept-Encoding': 'identity'})
    opener = build_opener(OfficialRedirects(source))
    started = time.monotonic()
    try:
        response = opener.open(request, timeout=timeout)
    except HTTPError as error:
        response = error
    with response:
        chunks, length = [], 0
        # read1 returns after one underlying read instead of waiting to fill a
        # large buffer. The socket timeout bounds a blocking read; the elapsed
        # check also stops a provider that keeps trickling bytes. Redirects are
        # independently capped above. This is not an unbounded full-body read.
        while True:
            if time.monotonic()-started > timeout:
                raise TimeoutError('Whole discovery response exceeded its time budget')
            chunk = response.read1(min(65536, MAX_RESPONSE_BYTES+1-length))
            if not chunk:
                break
            chunks.append(chunk); length += len(chunk)
            if length > MAX_RESPONSE_BYTES:
                raise ValueError('Discovery response exceeds the bounded byte limit')
        body = b''.join(chunks)
        return HTTPResponse(response.geturl(), response.code, body, dict(response.headers))


def request_url(source, query, since, now, max_results):
    if source == 'arxiv':
        lower = since.strftime('%Y%m%d')+'0000' if since else '000001010000'
        search = f'({query[2]}) AND submittedDate:[{lower} TO {now.strftime("%Y%m%d%H%M")}]'
        params = {'search_query': search, 'start': 0, 'max_results': max_results,
                  'sortBy': 'submittedDate', 'sortOrder': 'descending'}
    else:
        filters = ['to_publication_date:'+now.date().isoformat()]
        if since:
            filters.append('from_publication_date:'+since.isoformat())
        params = {'search': query[3], 'filter': ','.join(filters), 'per-page': max_results,
                  'sort': 'publication_date:desc',
                  'select': 'id,doi,title,authorships,publication_date,updated_date,ids,primary_location'}
    return ENDPOINTS[source]+'?'+urlencode(params)


def _retry_wait(headers, now):
    value = next((v for k, v in headers.items() if k.lower() == 'retry-after'), None)
    if value is None:
        return 1.
    try:
        wait = float(value)
    except ValueError:
        try:
            wait = (parsedate_to_datetime(value).astimezone(timezone.utc)-now).total_seconds()
        except (ValueError, TypeError, OverflowError):
            wait = 1.
    return max(0., min(wait, 5.))


def fetch_bounded(url, source, *, now, timeout, max_attempts, fetch, sleep, monotonic, last_request,
                  retain_response=None, deadline=None):
    attempts = []
    response = None
    for attempt in range(max_attempts):
        remaining = float('inf') if deadline is None else deadline-monotonic()
        if remaining <= 0:
            return response, attempts, 'Run time budget exhausted'
        if source == 'arxiv' and source in last_request:
            wait = max(0., 3.-(monotonic()-last_request[source]))
            if wait >= remaining:
                return response, attempts, 'Run time budget exhausted before provider pacing interval'
            if wait:
                sleep(wait)
        last_request[source] = monotonic()
        try:
            remaining = timeout if deadline is None else min(timeout, deadline-monotonic())
            if remaining <= 0:
                return response, attempts, 'Run time budget exhausted'
            candidate = fetch(url, source=source, timeout=remaining)
            validate_origin(candidate.url, source)
            if len(candidate.body) > MAX_RESPONSE_BYTES:
                raise ValueError('Discovery response exceeds byte limit')
            response = candidate
        except (OSError, TimeoutError, ValueError) as error:
            attempts.append({'number': attempt+1, 'error': type(error).__name__+': '+str(error)[:500]})
            if isinstance(error, ValueError):
                return response, attempts, str(error)
            wait = min(2.**attempt, 5.)
        else:
            receipt = {'number': attempt+1, 'http_status': response.status,
                       'response_url': response.url, 'response_sha256': sha256(response.body),
                       'response_bytes': len(response.body)}
            # Storage failures are fatal: never publish a snapshot claiming
            # retained provenance when its actual response bytes were lost.
            if retain_response is not None:
                receipt['response_artifact'] = retain_response(source, response)
            attempts.append(receipt)
            if response.status == 200:
                return response, attempts, None
            retryable = response.status in (408, 429, 500, 502, 503, 504)
            error = f'HTTP {response.status}'
            if not retryable:
                return response, attempts, error
            wait = _retry_wait(response.headers, now)
        if attempt+1 < max_attempts:
            sleep(wait if deadline is None else max(0., min(wait, deadline-monotonic())))
    return response, attempts, 'Retry budget exhausted'


def observation(source, source_id, title, authors, published, updated, doi, arxiv,
                *, query_id, response_url, response_hash, retrieved_at):
    title = ' '.join(title.split()) if isinstance(title, str) else ''
    if not title or not isinstance(authors, list) or any(not isinstance(a, str) or not a.strip() for a in authors):
        raise ValueError('Missing/invalid title or authors')
    published_time = parse_time(published)
    updated_time = parse_time(updated) if updated else None
    now = parse_time(retrieved_at)
    if published_time > now or (updated_time and updated_time > now):
        raise ValueError('future_metadata')
    validate_origin(response_url, source)
    if not HEX.fullmatch(response_hash):
        raise ValueError('Invalid response hash')
    doi = normalize_doi(doi)
    arxiv = normalize_arxiv(arxiv) if arxiv else None
    if doi and doi.startswith('10.48550/arxiv.'):
        doi_arxiv = normalize_arxiv(doi[len('10.48550/arxiv.'):])
        if arxiv and arxiv != doi_arxiv:
            raise ValueError('arXiv DOI and record identifiers disagree')
        arxiv = doi_arxiv
    if source == 'arxiv':
        source_id = normalize_arxiv(source_id)
        if arxiv != source_id:
            raise ValueError('arXiv source and record identifiers disagree')
    elif not isinstance(source_id, str) or not re.fullmatch(r'W\d{1,24}', source_id):
        raise ValueError('Invalid OpenAlex work identifier')
    identifiers = {'arxiv': [arxiv] if arxiv else [], 'doi': [doi] if doi else [],
                   'openalex': [source_id] if source == 'openalex' else []}
    url = ('https://arxiv.org/abs/'+arxiv if arxiv else
           'https://doi.org/'+quote(doi, safe='/():;._-') if doi else 'https://openalex.org/'+source_id)
    return {'source': source, 'source_id': source_id, 'query_id': query_id,
            'title': title, 'authors': [' '.join(a.split()) for a in authors],
            'published_at': stamp(published_time), 'updated_at': stamp(updated_time) if updated_time else None,
            'identifiers': identifiers, 'primary_url': url, 'response_url': response_url,
            'source_response_sha256': response_hash, 'retrieved_at': retrieved_at,
            'retrieval_status': 'retrieved_from_official_discovery_endpoint'}


def parse_feed(source, response, query_id, now, since=None, max_results=50):
    """Parse bounded trusted-origin bytes; exclude future dates explicitly."""
    validate_origin(response.url, source)
    if len(response.body) > MAX_RESPONSE_BYTES:
        raise ValueError('Discovery response exceeds byte limit')
    raw, failures, excluded = [], [], []
    if source == 'arxiv':
        if re.search(br'<!\s*(?:DOCTYPE|ENTITY)\b', response.body, re.I):
            raise ValueError('DTD/entity declarations are forbidden')
        root = ET.fromstring(response.body)
        if root.tag != '{http://www.w3.org/2005/Atom}feed':
            raise ValueError('Expected an arXiv Atom feed')
        total = root.findtext('{http://a9.com/-/spec/opensearch/1.1/}totalResults')
        if total is None or not re.fullmatch(r'\d+', total.strip()):
            raise ValueError('arXiv feed is missing valid result-count metadata')
        atom = '{http://www.w3.org/2005/Atom}'
        entries = root.findall(atom+'entry')
        if len(entries) > max_results:
            failures.append({'reason': 'Provider exceeded requested result cap; remaining entries excluded'})
        for entry in entries[:max_results]:
            text = lambda tag: entry.findtext(tag)
            raw.append({'source_id': text(atom+'id'), 'title': text(atom+'title'),
                'authors': [a.findtext(atom+'name') for a in entry.findall(atom+'author')],
                'published': text(atom+'published'), 'updated': text(atom+'updated'),
                'doi': text('{http://arxiv.org/schemas/atom}doi'), 'arxiv': text(atom+'id')})
    else:
        data = json.loads(response.body)
        if not isinstance(data, dict) or not isinstance(data.get('results'), list):
            raise ValueError('Expected OpenAlex results')
        count = (data.get('meta') or {}).get('count')
        if type(count) is not int or count < 0 or data.get('error'):
            raise ValueError('OpenAlex response is missing valid result-count metadata')
        if len(data['results']) > max_results:
            failures.append({'reason': 'Provider exceeded requested result cap; remaining entries excluded'})
        for work in data['results'][:max_results]:
            try:
                identifier = urlsplit(work['id'])
                if identifier.scheme != 'https' or identifier.hostname != 'openalex.org' or identifier.username or identifier.password:
                    raise ValueError('OpenAlex work has a foreign origin')
                if identifier.query or identifier.fragment or identifier.port not in (None,443):
                    raise ValueError('OpenAlex work identifier contains extra URL components')
                published = work['publication_date']
                if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', published):
                    raise ValueError('Invalid publication date')
                updated = work.get('updated_date')
                if updated and not re.search(r'(Z|[+-]\d{2}:\d{2})$', updated):
                    updated += 'Z'  # OpenAlex timestamps are UTC without a suffix
                landing = (work.get('primary_location') or {}).get('landing_page_url')
                arxiv = None
                if landing and urlsplit(landing).hostname in ('arxiv.org', 'www.arxiv.org'):
                    arxiv = normalize_arxiv(landing)
                raw.append({'source_id': identifier.path.lstrip('/'), 'title': work.get('title'),
                    'authors': [a['author']['display_name'] for a in work.get('authorships', [])],
                    'published': published+'T00:00:00Z', 'updated': updated,
                    'doi': work.get('doi'), 'arxiv': arxiv})
            except (KeyError, TypeError, ValueError) as error:
                failures.append({'reason': str(error)[:300]})
    rows = []
    for item in raw:
        try:
            row = observation(source, **item, query_id=query_id, response_url=response.url,
                              response_hash=sha256(response.body), retrieved_at=stamp(now))
            if since and parse_time(row['published_at']).date() < since:
                excluded.append({'source_id': row['source_id'], 'reason': 'before_since'})
            else:
                rows.append(row)
        except (KeyError, TypeError, ValueError) as error:
            detail = {'source_id': str(item.get('source_id', ''))[:200], 'reason': str(error)[:300]}
            (excluded if str(error) == 'future_metadata' else failures).append(detail)
    return rows, failures, excluded


def _review(value, now):
    if value is not None and not isinstance(value, dict):
        raise ValueError('Invalid review metadata')
    if value is None or value.get('status') == 'lead':
        return {'status': 'lead', 'basis': 'Unreviewed discovery; no SOTA or production qualification'}
    if value.get('status') != 'reviewed' or not value.get('reviewer') or not value.get('evidence_url'):
        raise ValueError('Reviewed status requires an explicit prior reviewer and evidence URL')
    if parse_time(value.get('reviewed_at')) > now:
        raise ValueError('Review time cannot be in the future')
    parsed = urlsplit(value['evidence_url'])
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Review evidence must be an HTTPS URL')
    return dict(value)


def _tokens(observation):
    return {(kind+':'+value) for kind, values in observation['identifiers'].items() for value in values}


def _content(record):
    return {key: record[key] for key in ('title', 'authors', 'published_at', 'updated_at', 'identifiers', 'primary_url')}


def merge_records(observations, previous, now):
    """Exact arXiv/DOI/provider-ID union, never title similarity or fuzzy matching."""
    old = previous['records'] if previous else []
    entries = [(ob, record) for record in old for ob in record['observations']]
    entries += [(ob, None) for ob in observations]
    parent = list(range(len(entries)))
    owners = {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (ob, _) in enumerate(entries):
        for key in _tokens(ob):
            if key in owners:
                parent[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = {}
    for i, entry in enumerate(entries):
        groups.setdefault(find(i), []).append(entry)
    records, added, updated, not_seen, redirects = [], [], [], [], {}
    for group in groups.values():
        old_records = {r['id']: r for _, r in group if r is not None}
        latest = {}
        for ob, _ in group:
            key = (ob['source'], ob['source_id'], ob['query_id'])
            if key not in latest or ob['retrieved_at'] >= latest[key]['retrieved_at']:
                latest[key] = ob
        obs = sorted(latest.values(), key=lambda ob: (ob['source'], ob['source_id'], ob['query_id']))
        identifiers = {kind: sorted({v for ob in obs for v in ob['identifiers'][kind]})
                       for kind in ('arxiv', 'doi', 'openalex')}
        kind = next(k for k in ('arxiv', 'doi', 'openalex') if identifiers[k])
        key = kind+':'+identifiers[kind][0]
        preferred = sorted(obs, key=lambda ob: (ob['source'] == 'arxiv', ob['retrieved_at']), reverse=True)[0]
        seen = any(r is None for _, r in group)
        first = min([r['first_seen_at'] for r in old_records.values()] or [stamp(now)])
        last = stamp(now) if seen else max(r['last_seen_at'] for r in old_records.values())
        record = {'id': key, **{k: preferred[k] for k in ('title', 'authors', 'published_at', 'updated_at', 'primary_url')},
            'identifiers': identifiers, 'observations': obs, 'first_seen_at': first, 'last_seen_at': last,
            'seen_in_current_run': seen, 'retrieval_status': 'retrieved' if seen else 'retained_previous_not_seen',
            'review': {'status': 'lead', 'basis': 'Unreviewed discovery; no SOTA or production qualification'}}
        changed = bool(old_records) and any(_content(r) != _content(record) for r in old_records.values())
        reviews = [_review(r.get('review'), now) for r in old_records.values()]
        reviewed = [r for r in reviews if r['status'] == 'reviewed']
        if reviewed and not changed:
            record['review'] = sorted(reviewed, key=lambda r: r['reviewed_at'], reverse=True)[0]
        elif reviewed:
            record['previous_reviews'] = reviewed
            record['review_update_reason'] = 'Metadata changed; explicit re-review required'
        if not old_records:
            added.append(key)
        elif changed:
            updated.append(key)
        if not seen:
            not_seen.append(key)
        redirects.update({old_key: key for old_key in old_records if old_key != key})
        records.append(record)
    return sorted(records, key=lambda r: r['id']), {'added': sorted(added), 'updated': sorted(updated),
        'not_seen_again': sorted(not_seen), 'identifier_redirects': redirects, 'deleted': []}


def validate_snapshot(snapshot, now=None):
    now = now or utc_now()
    if snapshot.get('schema_version') != 1 or parse_time(snapshot['generated_at']) > now:
        raise ValueError('Invalid/future snapshot header')
    if not isinstance(snapshot.get('records'), list) or not isinstance(snapshot.get('request_results'), list):
        raise ValueError('Invalid snapshot collections')
    if snapshot.get('claims') != {'automatically_reviewed': False, 'production_promoted': False, 'sota_established': False}:
        raise ValueError('Discovery cannot assert review, production or SOTA qualification')
    coverage = snapshot['coverage']
    requested = coverage['requested_sources']
    expected = {(source,query[0]) for source in requested for query in QUERIES}
    actual = [(row['source'],row['query_id']) for row in snapshot['request_results']]
    success = sum(row['retrieval_status']=='success' for row in snapshot['request_results'])
    complete = bool(actual) and success == len(actual)
    if (not requested or len(set(requested)) != len(requested) or not set(requested) <= set(ENDPOINTS)
            or set(actual) != expected or len(actual) != len(expected)
            or coverage['requests'] != len(actual) or coverage['successful_requests'] != success
            or coverage['complete_for_requested_requests'] != complete
            or coverage['incomplete_coverage'] != (not complete or set(requested) != set(ENDPOINTS))):
        raise ValueError('Coverage does not match actual provider/query outcomes')
    for result in snapshot['request_results']:
        source = result['source']
        validate_origin(result['request_url'], source)
        if result['retrieval_status'] not in ('failed', 'partial', 'success'):
            raise ValueError('Invalid query retrieval status')
        if ((result['retrieval_status']=='success') == bool(result['errors'])
                or not 0 <= result['accepted_entries'] <= coverage['max_results_per_query']):
            raise ValueError('Query status does not match its errors/result cap')
        for attempt in result['attempts']:
            if 'response_sha256' in attempt:
                validate_origin(attempt['response_url'], source)
                if not HEX.fullmatch(attempt['response_sha256']):
                    raise ValueError('Invalid retained response hash')
                if not 0 <= attempt['response_bytes'] <= MAX_RESPONSE_BYTES:
                    raise ValueError('Invalid retained response size')
                artifact = attempt.get('response_artifact')
                if artifact and artifact != 'responses/'+attempt['response_sha256']+'.bin':
                    raise ValueError('Invalid response artifact location')
                if coverage['actual_response_retention_enabled'] and not artifact:
                    raise ValueError('Actual response retention is missing a response artifact')
    seen = set()
    for record in snapshot['records']:
        if record['id'] in seen or not record.get('observations'):
            raise ValueError('Duplicate or provenance-free record')
        seen.add(record['id'])
        _review(record.get('review'), now)
        if parse_time(record['first_seen_at']) > now or parse_time(record['last_seen_at']) > now:
            raise ValueError('Future observation history')
        for ob in record['observations']:
            source = ob['source']
            if source not in ENDPOINTS:
                raise ValueError('Unknown provenance source')
            # Revalidate all retained metadata; a malformed previous file must
            # never be overwritten by an apparently successful new snapshot.
            rebuilt = observation(source, ob['source_id'], ob['title'], ob['authors'], ob['published_at'],
                ob['updated_at'], next(iter(ob['identifiers']['doi']), None),
                next(iter(ob['identifiers']['arxiv']), None), query_id=ob['query_id'],
                response_url=ob['response_url'], response_hash=ob['source_response_sha256'],
                retrieved_at=ob['retrieved_at'])
            if rebuilt != ob or parse_time(ob['retrieved_at']) > now:
                raise ValueError('Retained source observation does not validate')
        identifiers = {kind: sorted({v for ob in record['observations'] for v in ob['identifiers'][kind]})
                       for kind in ('arxiv', 'doi', 'openalex')}
        kind = next(k for k in ('arxiv', 'doi', 'openalex') if identifiers[k])
        if record['id'] != kind+':'+identifiers[kind][0] or record['identifiers'] != identifiers:
            raise ValueError('Aggregated record identifiers do not match its provenance')
        preferred = sorted(record['observations'], key=lambda ob: (ob['source']=='arxiv', ob['retrieved_at']), reverse=True)[0]
        if any(record[key] != preferred[key] for key in ('title','authors','published_at','updated_at','primary_url')):
            raise ValueError('Aggregated metadata does not match its source observation')
    return snapshot


def discover(*, since=None, previous=None, sources=('arxiv', 'openalex'), max_results=5,
             timeout=10., max_attempts=2, fetch=urllib_fetch, sleep=time.sleep,
             monotonic=time.monotonic, now=None, retain_response=None, max_run_seconds=180.):
    now = now or utc_now()
    if now.tzinfo is None:
        raise ValueError('Use an explicit UTC-aware clock')
    now = now.astimezone(timezone.utc)
    since_date = parse_since(since, now)
    if not sources or len(set(sources)) != len(sources) or any(s not in ENDPOINTS for s in sources):
        raise ValueError('Choose unique official discovery sources')
    if (not 1 <= max_results <= 50 or not 0 < timeout <= 30 or not 1 <= max_attempts <= 3
            or not 0 < max_run_seconds <= 600):
        raise ValueError('Request bounds exceeded')
    if previous is not None:
        validate_snapshot(previous, now)
    observations, results, last_request = [], [], {}
    deadline = monotonic()+max_run_seconds
    for source in sources:
        for query in QUERIES:
            url = request_url(source, query, since_date, now, max_results)
            response, attempts, error = fetch_bounded(url, source, now=now, timeout=timeout,
                max_attempts=max_attempts, fetch=fetch, sleep=sleep, monotonic=monotonic, last_request=last_request,
                retain_response=retain_response, deadline=deadline)
            row = {'source': source, 'query_id': query[0], 'topic': query[1], 'request_url': url,
                'retrieved_at': stamp(now), 'attempts': attempts, 'retrieval_status': 'failed',
                'response_url': response.url if response else None,
                'source_response_sha256': sha256(response.body) if response else None,
                'http_status': response.status if response else None, 'errors': [], 'excluded': [], 'accepted_entries': 0}
            if error is not None:
                row['errors'].append({'reason': error})
            else:
                try:
                    found, errors, excluded = parse_feed(source, response, query[0], now, since_date, max_results)
                    observations.extend(found)
                    row.update(retrieval_status='partial' if errors else 'success', errors=errors,
                               excluded=excluded, accepted_entries=len(found))
                except (ET.ParseError, json.JSONDecodeError, TypeError, ValueError) as error:
                    row['errors'].append({'reason': type(error).__name__+': '+str(error)[:500]})
            results.append(row)
    records, changes = merge_records(observations, previous, now)
    complete = all(row['retrieval_status'] == 'success' for row in results)
    snapshot = {'schema_version': 1, 'generated_at': stamp(now),
        'previous_snapshot_sha256': sha256(canonical_json(previous)) if previous else None,
        'window': {'since': since, 'until_utc': stamp(now)},
        'coverage': {'complete_for_requested_requests': complete,
            'incomplete_coverage': not complete or set(sources) != set(ENDPOINTS),
            'requested_sources': list(sources), 'unqueried_sources': sorted(set(ENDPOINTS)-set(sources)),
            'bounded_nonexhaustive_search': True, 'requests': len(results),
            'actual_response_retention_enabled': retain_response is not None,
            'successful_requests': sum(r['retrieval_status'] == 'success' for r in results),
            'max_results_per_query': max_results, 'timeout_seconds': timeout, 'max_attempts': max_attempts,
            'run_request_budget_seconds': max_run_seconds},
        'request_results': results, 'records': records, 'changes': changes,
        'claims': {'automatically_reviewed': False, 'production_promoted': False, 'sota_established': False}}
    return validate_snapshot(snapshot, now)


def _atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.'+path.name+'.', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_snapshot(path, *, now=None):
    with Path(path).open('rb') as handle:
        content = handle.read(MAX_SNAPSHOT_BYTES+1)
    if len(content) > MAX_SNAPSHOT_BYTES:
        raise ValueError('Snapshot exceeds the bounded byte limit')
    return validate_snapshot(json.loads(content), now)


class ResponseArchive:
    """Content-addressed actual response bytes, including HTTP failure bodies."""
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def __call__(self, source, response):
        validate_origin(response.url, source)
        if len(response.body) > MAX_RESPONSE_BYTES:
            raise ValueError('Response exceeds archive size cap')
        relative = 'responses/'+sha256(response.body)+'.bin'
        target = self.output_dir/relative
        if target.exists():
            if target.read_bytes() != response.body:
                raise ValueError('Immutable response archive changed')
        else:
            _atomic_bytes(target, response.body)
        return relative


def _publish_locked(snapshot, output_dir, now):
    latest = output_dir/'latest.json'
    previous = load_snapshot(latest, now=now) if latest.exists() else None
    previous_bytes = latest.read_bytes() if previous is not None else None
    identity = sha256(canonical_json(previous)) if previous else None
    if identity != snapshot['previous_snapshot_sha256']:
        raise ValueError('Latest snapshot changed during discovery; preserve it and retry')
    for result in snapshot['request_results']:
        for attempt in result['attempts']:
            if attempt.get('response_artifact'):
                path = output_dir/attempt['response_artifact']
                with path.open('rb') as handle:
                    body = handle.read(MAX_RESPONSE_BYTES+1)
                if len(body) != attempt['response_bytes'] or sha256(body) != attempt['response_sha256']:
                    raise ValueError('Retained source response does not match its receipt')
    if previous_bytes is not None:
        saved = output_dir/'snapshots'/('previous-'+identity+'.json')
        if not saved.exists():
            _atomic_bytes(saved, previous_bytes)
    content = json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False).encode()+b'\n'
    if len(content) > MAX_SNAPSHOT_BYTES:
        raise ValueError('Snapshot exceeds the bounded byte limit; explicit history management required')
    name = snapshot['generated_at'].replace(':', '').replace('-', '')+'-'+sha256(content)[:16]+'.json'
    immutable = output_dir/'snapshots'/name
    if immutable.exists() and immutable.read_bytes() != content:
        raise ValueError('Refusing to replace a different immutable snapshot')
    if not immutable.exists():
        _atomic_bytes(immutable, content)
    _atomic_bytes(latest, content)
    return {'latest': str(latest), 'snapshot': str(immutable), 'snapshot_sha256': sha256(content)}


def publish_snapshot(snapshot, output_dir, *, now=None):
    """Retain snapshots and atomically publish under a nonblocking file lock."""
    now = now or utc_now()
    validate_snapshot(snapshot, now)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir/'.publish.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another snapshot publisher is active; preserve latest and retry') from error
        try:
            return _publish_locked(snapshot, output_dir, now)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
