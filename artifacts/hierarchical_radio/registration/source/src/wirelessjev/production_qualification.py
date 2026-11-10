"""Read-only evidence qualification. This module cannot activate a controller."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from scipy.special import betaincinv

SHA = re.compile(r'[0-9a-f]{64}\Z')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def subject(bundle):
    return identity({k: bundle.get(k) for k in ('candidate_id','mode','hardware_profile','files','protocol','limits')})


@dataclass(frozen=True)
class Limits:
    min_sites: int = 6
    min_training_seeds: int = 3
    miss_margin_ppm: float = 0.
    max_control_miss_ppm: float = 10.
    max_gap_ms: float = 200.
    max_latency_p99_ms: float = 50.
    min_hardware_samples: int = 1000
    max_risk_ece: float = .02
    max_rollback_ms: float = 200.
    min_control_commands: int = 100000
    familywise_alpha: float = .05


def number(value, name, *, count=False):
    if (isinstance(value, bool) or not isinstance(value, (int,float))
            or value < 0 or value > 10**15 or not math.isfinite(value)):
        raise ValueError(name+' must be a finite nonnegative number')
    if count and (type(value) is not int): raise ValueError(name+' must be an integer count')
    return value


def timestamp(value):
    t = datetime.fromisoformat(value.replace('Z','+00:00'))
    if t.tzinfo is None or t > datetime.now(timezone.utc): raise ValueError('Invalid/future evidence time')
    return t


def binomial_upper(failures, episodes, alpha):
    """Exact one-sided bound, conditional on the registered IID episode design."""
    if not 0 < alpha < 1 or not 0 <= failures <= episodes or episodes < 1:
        raise ValueError('Invalid independent-episode counts/confidence level')
    if failures == episodes: return 1.
    if failures == 0: return -math.expm1(math.log(alpha)/episodes)
    return float(betaincinv(failures+1,episodes-failures,1-alpha))


def paired_metrics(rows, sites, seeds, limits):
    """Recompute site-paired intervals; repeated model seeds are not new sites."""
    keyed = {}
    for row in rows:
        key = (row['site_id'], row['training_seed'])
        if key in keyed: raise ValueError('Repeated site/seed pair')
        arms = []
        for family in ('candidate','baseline'):
            a = row[family]; sent = number(a['sent'],'sent',count=True); miss = number(a['misses'],'misses',count=True)
            if not sent or miss > sent: raise ValueError('Invalid miss/sent counts')
            gap = number(a['max_gap_ms'],'max_gap_ms')
            latency = np.asarray([number(v,'latency_ms') for v in a['latency_ms']],dtype=float)
            if len(latency) < 100: raise ValueError('Insufficient raw latency samples')
            arms.append((miss/sent*1e6,gap,float(np.quantile(latency,.99)),int(sent)))
        if arms[0][3] != arms[1][3]: raise ValueError('Paired command exposure differs')
        keyed[key] = arms
    if set(keyed) != {(site,seed) for site in sites for seed in seeds}: raise ValueError('Missing/foreign confirmation site-seed pairs')
    matrix = np.array([[keyed[s,k][0][0]-keyed[s,k][1][0] for k in seeds] for s in sites])
    difference = matrix.mean(1)
    rng = np.random.default_rng(70401)
    draws = rng.integers(len(sites),size=(20000,len(sites)))
    boot = difference[draws].mean(1)
    interval = np.quantile(boot,[.025,.975]).tolist()
    seed_draws = rng.integers(len(seeds),size=(20000,len(seeds)))
    crossed = matrix[draws[:,:,None],seed_draws[:,None,:]].mean((1,2))
    candidate = [a[0] for a in keyed.values()]; baseline = [a[1] for a in keyed.values()]
    result = {'paired_site_difference_ppm':float(difference.mean()),'paired_site_95_ppm':interval,
              'crossed_site_seed_95_ppm':np.quantile(crossed,[.025,.975]).tolist(),
              'sites':len(sites),'training_seeds':len(seeds),'raw_pairs':len(keyed),
              'candidate_ppm':float(np.mean([a[0] for a in candidate])),
              'maximum_gap_ms':max(a[1] for a in candidate),'maximum_p99_latency_ms':max(a[2] for a in candidate)}
    violations = []
    if max(interval[1],result['crossed_site_seed_95_ppm'][1]) > limits.miss_margin_ppm: violations.append('Paired/crossed miss interval exceeds noninferiority margin')
    if any(a[0] > limits.max_control_miss_ppm for a in candidate): violations.append('Absolute command-miss budget exceeded in at least one site/seed')
    if any(a[1] > limits.max_gap_ms or a[1] > b[1] or a[2] > limits.max_latency_p99_ms or a[2] > b[2]
           for a,b in zip(candidate,baseline)): violations.append('Gap/latency cap or paired no-regression requirement failed')
    result['violations'] = violations
    return result


def qualify(bundle, root, *, limits=Limits()):
    """Verify bound bytes, raw measurements and prerequisites; never actuate."""
    root = Path(root).resolve(); blockers = []; verified = {}; metrics = {}; identity_ok = True
    binary_cache = {}; checked_resets = set()
    if not isinstance(bundle,dict): bundle = {}
    def blocked(code, error): blockers.append({'code':code,'detail':str(error)[:1000]})
    try:
        for key,value in asdict(limits).items(): number(value,key,count=key.startswith('min_'))
        if limits.min_sites<2 or limits.min_training_seeds<3 or limits.min_hardware_samples<100:
            raise ValueError('Policy weakens minimum evidence counts')
        if not 0 < limits.familywise_alpha < .5: raise ValueError('Invalid familywise error allocation')
        if bundle.get('limits') != asdict(limits): raise ValueError('Candidate receipts do not bind the operator policy limits')
    except (ValueError,TypeError,AttributeError) as error:
        blocked('limits',error); limits = Limits()
    def reference(ref, *, json_value=False, binary_value=False):
        name, expected = ref['path'], ref['sha256']
        if not isinstance(name,str) or Path(name).is_absolute() or '..' in Path(name).parts or not SHA.fullmatch(expected):
            raise ValueError('Invalid relative evidence reference')
        if binary_value and (name,expected) in binary_cache: return binary_cache[name,expected]
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file(): raise ValueError('Missing/outside evidence: '+name)
        if path.stat().st_size > 1024**3: raise ValueError('Single evidence file exceeds 1 GiB hashing limit')
        content = None
        if json_value or binary_value:
            cap = 32_000_000 if binary_value else 8_000_000
            with path.open('rb') as handle: content = handle.read(cap+1)
            if len(content)>cap: raise ValueError('Receipt/episode ledger exceeds size limit')
        actual = hashlib.sha256(content).hexdigest() if content is not None else digest(path)
        verified[name] = {'expected_sha256':expected,'actual_sha256':actual}
        if actual != expected: raise ValueError('Evidence bytes changed: '+name)
        if binary_value:
            binary_cache[name,expected] = content
            return content
        if not json_value: return expected
        value = json.loads(content)
        if not isinstance(value,dict): raise ValueError('Expected a receipt object')
        return value
    try:
        sid = subject(bundle)
        if (bundle.get('schema_version') != 1 or not isinstance(bundle.get('candidate_id'),str)
                or not bundle['candidate_id'].strip()): raise ValueError('Missing candidate schema/identity')
        if bundle.get('mode') not in ('proposal_only','active_candidate'): raise ValueError('Unknown candidate mode')
    except (ValueError,TypeError,KeyError) as error:
        sid = None; blocked('identity',error); identity_ok = False
    files = bundle.get('files',{})
    if not isinstance(files,dict): files = {}
    for group in ('source','weights','data','licenses'):
        try:
            if not isinstance(files.get(group),dict) or not files[group]: raise ValueError('Missing '+group+' file manifest')
            for path,sha in files[group].items(): reference({'path':path,'sha256':sha})
        except (OSError,ValueError,TypeError,KeyError) as error:
            identity_ok = False; blocked('identity_'+group,error)
    def receipt(role):
        value = reference(bundle[role],json_value=True)
        if value.get('subject_sha256') != sid: raise ValueError(role+' is not bound to this candidate/profile/data')
        return value
    def gate(role, function):
        try: return function()
        except (OSError,ValueError,TypeError,KeyError,IndexError,AttributeError) as error:
            blocked(role,error); return None
    def profile():
        p = bundle['hardware_profile']
        if any(not isinstance(p.get(k),str) or not p[k].strip() for k in ('profile_id','device_model','driver','firmware','protocol','baseline_source_sha256','baseline_endpoint','interface','role')):
            raise ValueError('Incomplete hardware/software profile')
        if number(p['control_period_ms'],'control_period_ms') != 50 or number(p['deadline_ms'],'deadline_ms') != 50:
            raise ValueError('This contract is for the existing 20 Hz / 50 ms warehouse control task')
        if p['interface']!='wlan1' or p['role']!='WIFI': raise ValueError('Only the warehouse wlan1/WIFI role is in scope')
        if p['baseline_source_sha256'] not in files.get('source',{}).values(): raise ValueError('Baseline source is not hash-bound')
        return p
    hw_profile = gate('hardware_profile',profile)
    def rights():
        r = receipt('rights'); grants = r['grants']; needed = set(files['weights'].values())|set(files['data'].values())
        uses = {sha:set() for sha in needed}
        for grant in grants:
            if not grant.get('reviewer') or grant['license_sha256'] not in files['licenses'].values(): raise ValueError('License review lacks bound license/reviewer')
            timestamp(grant['reviewed_at'])
            if grant['artifact_sha256'] in uses: uses[grant['artifact_sha256']].update(grant['allowed_uses'])
        if not uses or any('research' not in u for u in uses.values()): raise ValueError('Research rights not recorded for every weight/data file')
        if any('production_inference' not in u for u in uses.values()): blocked('production_rights','Production inference rights are missing')
        return True
    rights_ok = gate('rights',rights)
    protocol_value = gate('protocol',lambda:reference(bundle['protocol'],json_value=True))
    def independence():
        p = protocol_value['independent_sampling']
        if p.get('unit')!='independently_reset_episode' or p.get('target')!='one_preregistered_uniform_command_per_episode':
            raise ValueError('Packet independence/estimated ESS is not independent reset-episode evidence')
        r = reference(p['design_review'],json_value=True)
        if (r.get('basis')!='independent_randomized_resets' or not isinstance(r.get('reviewer'),str) or not r['reviewer'].strip()
                or r.get('assumption')!='iid_reset_episodes_within_registered_profile'
                or r.get('profile_sha256')!=identity(bundle['hardware_profile'])
                or r.get('shift_status')!='no_detected_shift_in_registered_profile'
                or r.get('on_shift')!='invalidate' or len(r.get('justification',''))<40):
            raise ValueError('Missing/unjustified IID reset design or unsupported/shifted deployment scope')
        if timestamp(r['reviewed_at'])>timestamp(protocol_value['sealed_at']): raise ValueError('Reset design review was not registered before confirmation')
        reference(r['reset_and_sampling_source'])
        if files['source'].get(r['reset_and_sampling_source']['path'])!=r['reset_and_sampling_source']['sha256']:
            raise ValueError('Reset/sampling design is not in the bound source manifest')
        return r
    independence_review = gate('independence',independence)
    def ledger(ref, resets, *, paired):
        values = np.frombuffer(reference(ref,binary_value=True),dtype=np.uint8)
        if not len(values) or np.any(values > (3 if paired else 1)):
            raise ValueError('Invalid raw independent-episode outcome ledger')
        raw_ids = reference(resets,binary_value=True)
        if len(raw_ids)!=8*len(values): raise ValueError('Reset identities do not match observed episode count')
        key = (resets['sha256'],len(values))
        if key not in checked_resets:
            ids = np.frombuffer(raw_ids,dtype='<u8')
            if np.any(ids[1:]<=ids[:-1]): raise ValueError('Repeated/unordered reset identities cannot inflate effective exposure')
            checked_resets.add(key)
        return values
    def independent_control():
        protocol = protocol_value; evaluation = receipt('evaluation')
        train, val, sites = [set(protocol[k]) for k in ('training_sites','validation_sites','confirmation_sites')]
        seeds = protocol['training_seeds']
        if not train or not val or len(sites)<limits.min_sites or train&val or train&sites or val&sites: raise ValueError('Insufficient/disjoint site splits')
        if (len(set(seeds)) != len(seeds) or len(seeds)<max(3,limits.min_training_seeds)
                or len(seeds)>16 or len(sites)>128 or any(type(seed) is not int or seed<0 for seed in seeds)):
            raise ValueError('At least three distinct integer training seeds are required; maximum 16 seeds / 128 sites')
        if timestamp(protocol['sealed_at']) >= timestamp(evaluation['confirmation_opened_at']): raise ValueError('Protocol was not sealed before confirmation')
        fits = [reference(r,json_value=True) for r in bundle['fits']]
        if {f['training_seed'] for f in fits} != set(seeds) or len(fits)!=len(seeds): raise ValueError('Missing fit receipts')
        weights = []
        for fit in fits:
            if (fit.get('subject_sha256')!=sid or fit['source_group_sha256']!=identity(files['source'])
                    or fit['data_group_sha256']!=identity(files['data']) or number(fit['optimizer_steps'],'optimizer_steps',count=True)<1):
                raise ValueError('Fit provenance does not bind actual source/data/optimizer steps')
            if files['weights'].get(fit['weights_path']) != fit['weights_sha256']: raise ValueError('Fit checkpoint does not match candidate weights')
            weights.append(fit['weights_sha256'])
        if len(set(weights))!=len(seeds): raise ValueError('Seed labels reuse the same checkpoint')
        metrics['control'] = paired_metrics(evaluation['rows'],sorted(sites),seeds,limits)
        if independence_review is None: raise ValueError('Absolute risk needs justified independent reset episodes')
        absolute = []
        alpha = limits.familywise_alpha/(2*len(evaluation['rows']))
        for row in evaluation['rows']:
            v = ledger(row['independent_episode_ledger'],row['independent_reset_ids'],paired=True); n = len(v)
            for family,mask in [('candidate',1),('baseline',2)]:
                arm = row[family]
                if arm['sent']!=n or arm['misses']!=int(np.count_nonzero(v & mask)):
                    raise ValueError('Independent episode ledger contradicts raw miss/sent counts')
            upper = binomial_upper(row['candidate']['misses'],n,alpha)*1e6
            absolute.append({'site_id':row['site_id'],'training_seed':row['training_seed'],
                'independent_episodes':n,'candidate_upper_ppm':upper})
        metrics['control']['absolute_risk'] = {'unit':'one_registered_sample_per_independent_reset_episode',
            'conditional_assumption':independence_review['assumption'],'per_cell_alpha':alpha,'cells':absolute}
        if any(c['independent_episodes']<limits.min_control_commands or c['candidate_upper_ppm']>limits.max_control_miss_ppm for c in absolute):
            metrics['control']['violations'].append('Insufficient independent exposure or absolute rare-risk upper bound exceeds budget')
        if metrics['control']['violations']: raise ValueError('; '.join(metrics['control']['violations']))
        return protocol
    protocol = gate('independent_control',independent_control)
    def risk():
        r = receipt('risk')
        if r.get('scope')!='empirical_heldout_calibration' or r.get('target')!='deadline_miss' or r.get('distribution_free_certificate') is not None:
            raise ValueError('Risk must be empirical calibration, not a claimed universal/exchangeability certificate')
        if protocol_value is None or independence_review is None: raise ValueError('Risk calibration needs registered independent episodes and data splits')
        protocol = protocol_value
        calibration = set(r['calibration_sites'])
        if not calibration or calibration & (set(protocol['training_sites'])|set(protocol['confirmation_sites'])): raise ValueError('Risk calibration split is missing or leaked')
        count = misses = error = 0.; bins = set(); seen_bins = set(); bounds = []
        registered = {b['bin_id']:b for b in protocol['risk_bins']}
        if len(registered)!=len(protocol['risk_bins']) or not registered: raise ValueError('Missing/duplicate preregistered risk bins')
        alpha = limits.familywise_alpha/(2*len(registered))
        for row in r['bins']:
            if row['site_id'] not in calibration: raise ValueError('Foreign risk-calibration site')
            n = number(row['trials'],'risk trials',count=True); m = number(row['misses'],'risk misses',count=True); p = number(row['probability'],'risk probability')
            if not n or m>n or p>1: raise ValueError('Invalid risk probability/counts')
            spec = registered[row['bin_id']]
            if row['bin_id'] in seen_bins or type(spec['admitted']) is not bool: raise ValueError('Duplicate/invalid admission bin')
            seen_bins.add(row['bin_id'])
            v = ledger(row['independent_episode_ledger'],row['independent_reset_ids'],paired=False)
            if len(v)!=n or int(v.sum())!=m: raise ValueError('Calibration episode ledger contradicts raw trials/misses')
            upper = binomial_upper(m,n,alpha); ceiling = number(spec['ceiling'],'risk bin ceiling')
            if upper>p or p>ceiling or upper>ceiling or (spec['admitted'] and max(upper,p)>limits.max_control_miss_ppm/1e6):
                raise ValueError('Rare-event underprediction/absolute admission risk bound exceeds registered budget')
            bounds.append({'bin_id':row['bin_id'],'independent_episodes':n,'upper_probability':upper,'admitted':spec['admitted']})
            count += n; misses += m; error += n*abs(p-m/n); bins.add(p)
        if count<1000 or not 0<misses<count or len(bins)<2 or error/count>limits.max_risk_ece: raise ValueError('Insufficient/uncalibrated empirical risk labels')
        if seen_bins!=set(registered) or not any(b['admitted'] for b in bounds): raise ValueError('Missing registered/accepted calibration stratum')
        metrics['risk']={'empirical_ece':error/count,'trials':int(count),'misses':int(misses),
            'episode_bounds':bounds,'per_bin_alpha':alpha,'guarantee':'conditional_on_reviewed_reset_design_not_a_transfer_certificate'}
    gate('risk',risk)
    def hardware():
        r = receipt('hardware')
        if hw_profile is None or r.get('measurement_kind')!='physical_device_run' or not r.get('device_id'): raise ValueError('Actual hardware receipt is missing')
        values = []; seen = set(); last_time = -1
        for sample in r['samples']:
            total = number(sample['end_to_end_ms'],'end_to_end_ms')
            sample_id = sample['sample_id']; started = number(sample['started_at_ns'],'started_at_ns',count=True)
            if not isinstance(sample_id,str) or not sample_id or sample_id in seen or started<=last_time:
                raise ValueError('Repeated or unordered hardware latency samples')
            seen.add(sample_id); last_time = started
            stages = sum(number(sample[k],k) for k in ('observation_ms','inference_ms','transport_ms','actuator_ms'))
            if total<=0 or total<stages or sample.get('timed_out') is not False: raise ValueError('Hardware latency stages/timeouts are invalid')
            values.append(total)
        if len(values)<limits.min_hardware_samples or np.quantile(values,.99)>limits.max_latency_p99_ms or max(values)>hw_profile['deadline_ms']:
            raise ValueError('Physical end-to-end latency count/deadline failed')
        metrics['hardware']={'samples':len(values),'p99_end_to_end_ms':float(np.quantile(values,.99)),'maximum_end_to_end_ms':max(values)}
        return r
    hardware_ok = gate('hardware',hardware)
    def rollback():
        r = receipt('rollback'); events = r['events']
        if (r.get('measurement_kind')!='physical_device_run' or hw_profile is None or hardware_ok is None
                or r.get('device_id')!=hardware_ok['device_id']): raise ValueError('Matching actual hardware rollback evidence is missing')
        if [e['event'] for e in events] != ['outcome_unknown','status_read','baseline_verified']: raise ValueError('Rollback needs observed reconciliation and baseline restoration')
        times = [number(e['at_ms'],'rollback at_ms') for e in events]
        if times!=sorted(times) or times[-1]-times[0]>limits.max_rollback_ms: raise ValueError('Rollback recovery budget exceeded')
        if events[-1].get('controller_sha256')!=hw_profile['baseline_source_sha256'] or events[-1].get('evidence_source')!='hardware_status':
            raise ValueError('Rollback did not verify the bound baseline on hardware')
        if any(events[-1].get('observed_'+key)!=hw_profile[key] for key in ('baseline_endpoint','interface','role')):
            raise ValueError('Hardware status did not verify the baseline link/interface/role')
    gate('rollback',rollback)
    if bundle.get('mode')=='proposal_only': blocked('proposal_only','Proposal/shadow artifacts are never execution eligible')
    sandbox = bool(identity_ok and rights_ok)
    return {'schema_version':1,'candidate_id':bundle.get('candidate_id'),'subject_sha256':sid,
            'sandbox_eligible':sandbox,'latency_shadow_eligible':bool(sandbox and hardware_ok),
            'execution_eligible':not blockers,'blockers':blockers,'verified_files':verified,'metrics':metrics,
            'limits':asdict(limits),'runtime_activated':False,'scientific_or_safety_certificate':False}


def current_repository_status(root):
    """Inventory actual receipts without recasting prototypes as qualified models."""
    root = Path(root); candidates = []
    inventory = [
        ('packet_action_jepa','Development rejected; confirmation sites unused',
         ['artifacts/packet_value/summary.json','artifacts/packet_value/selection.json']),
        ('visual_radio_dinov3_rt_fusion','Pretrained encoder and static RT fixture; fusion untrained',
         ['artifacts/visual_radio/dinov3_access.json','artifacts/visual_radio/rt_probe/receipt.json','artifacts/visual_radio/verification.json']),
        ('continual_response_experiment','Registered experiment; no independently confirmed production model',
         ['artifacts/continual_radio/registration/protocol.json','artifacts/continual_radio/completion.json'])]
    for name,scope,paths in inventory:
        evidence = {p:{'sha256':digest(root/p),'bytes':(root/p).stat().st_size} for p in paths if (root/p).is_file()}
        for p,record in evidence.items():
            if record['bytes'] <= 8_000_000:
                raw = json.loads((root/p).read_text())
                record['observed_fields'] = {k:raw[k] for k in ('status','completed','simulation_only','fresh_sites','test_runs',
                    'primary_evaluated','primary_success','physical_deployment_qualified','geometry_radio_validation_passed') if k in raw}
        missing = [p for p in paths if not (root/p).is_file()]
        bundle = {'schema_version':1,'candidate_id':name,'mode':'proposal_only'}
        candidates.append({'scope':scope,'existing_evidence':evidence,'missing_observed_receipts':missing,'qualification':qualify(bundle,root)})
    return {'schema_version':1,'checker_source_sha256':digest(__file__),
            'read_only_inventory':True,'runtime_activated':False,'candidates':candidates}
