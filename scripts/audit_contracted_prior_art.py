"""Retrieve and verify supplied literature leads; never invent bibliographic data."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,html,json,re,time
import httpx
import pymupdf

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/contracted/prior_art';ARCHIVE=ROOT.parent/'training/contracted-world-models/literature'
LEADS={'2610.17584':'RAN-JEPA','2604.06882':'Telecom World Models','2607.09798':'JEPA for AI-native 6G','2606.06373':'LatentWave','2608.10222':'JEPA channel model','2610.13315':'JEPA channel model','2610.04345':'System One Models for Wireless Decision-Making',
       '1710.08005':'Smart Predict then Optimize','1911.08265':'MuZero','1705.08500':'Selective Classification for Deep Neural Networks','2101.02703':'Risk-Controlling Prediction Sets','2011.03506':'The Value Equivalence Principle for Model-Based Reinforcement Learning'}
QUERIES=['uncertainty gated selective prediction wireless network control','decision focused learning wireless world model JEPA','value equivalent models decision focused learning','RAN JEPA Telecom World Models']


def get(url):
    r=httpx.get(url,timeout=25,follow_redirects=True,headers={'User-Agent':'WirelessJEVResearchAudit/1.0 (academic source verification)'})
    return r


def lead(item):
    ident,expected=item;row={'id':ident,'supplied_title':expected,'checked_at_unix':time.time(),'full_text_retrieved':False,'attempts':[]}
    for base in ['https://arxiv.org/abs/','https://export.arxiv.org/abs/','https://www.alphaxiv.org/abs/']:
        url=base+ident
        try:
            r=get(url);row['attempts'].append({'url':url,'status':r.status_code})
            if r.status_code!=200:continue
            title=re.search(r'<meta\s+name=["\']citation_title["\']\s+content=["\'](.*?)["\']',r.text,re.S)
            if title:
                row['verified_title']=html.unescape(title.group(1));row['metadata_url']=url
                (ARCHIVE/(ident+'.metadata.html')).write_text(r.text)
                break
        except httpx.HTTPError as e:row['attempts'].append({'url':url,'error':type(e).__name__})
    # Full text only counts after an actual PDF parse; a 200 HTML response is insufficient.
    for base in ['https://arxiv.org/pdf/','https://export.arxiv.org/pdf/']:
        url=base+ident
        try:
            r=get(url);row['attempts'].append({'url':url,'status':r.status_code})
            if r.status_code!=200 or not r.content.startswith(b'%PDF'):continue
            doc=pymupdf.open(stream=r.content,filetype='pdf');text='\n'.join(p.get_text() for p in doc)
            if len(text)<300:continue
            (ARCHIVE/(ident+'.pdf')).write_bytes(r.content);(ARCHIVE/(ident+'.txt')).write_text(text)
            row.update({'full_text_retrieved':True,'pdf_url':url,'pdf_sha256':hashlib.sha256(r.content).hexdigest(),'pages':len(doc),'opening_excerpt':text[:160]})
            terms=['uncertainty','abstain','selective','regret','decision-focused','joint embedding','risk','calibrat']
            row['term_offsets']={term:[m.start() for m in re.finditer(re.escape(term),text,re.I)] for term in terms};doc.close();break
        except (httpx.HTTPError,RuntimeError,ValueError) as e:row['attempts'].append({'url':url,'error':type(e).__name__})
    row['status']='full_text_verified' if row['full_text_retrieved'] else 'metadata_only' if 'verified_title' in row else 'unverified_access'
    if 'verified_title' in row:
        a=set(re.findall('[a-z]+',expected.lower()));b=set(re.findall('[a-z]+',row['verified_title'].lower()));row['title_keyword_overlap']=len(a&b)/max(len(a),1)
        row['supplied_title_match_requires_review']=row['title_keyword_overlap']<.35
    return row


def search(query):
    url='https://api.crossref.org/works'
    try:
        r=httpx.get(url,params={'query':query,'rows':8},timeout=25,follow_redirects=True);row={'query':query,'url':str(r.url),'status':r.status_code}
        if r.status_code==200:row['results']=[{'title':x.get('title',[]),'doi':x.get('DOI'),'url':x.get('URL'),'published':x.get('published',{})} for x in r.json().get('message',{}).get('items',[])]
        return row
    except (httpx.HTTPError,ValueError) as e:return {'query':query,'error':type(e).__name__}


def main():
    OUT.mkdir(parents=True,exist_ok=True);ARCHIVE.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(6) as pool:leads=list(pool.map(lead,LEADS.items()));searches=list(pool.map(search,QUERIES))
    report={'leads':leads,'searches':searches,'exhaustive_novelty_verified':False,'limitations':['IEEE Xplore full-text/index search is not authenticated in this environment','A supplied title is not a verified citation','Identifier/title mismatches are not evidence against a method'],'checked_at_unix':time.time()}
    (OUT/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'leads':[{k:r[k] for k in ['id','status','verified_title'] if k in r} for r in leads],'searches':[{'query':s['query'],'status':s.get('status')} for s in searches]}),flush=True)


if __name__=='__main__':main()
