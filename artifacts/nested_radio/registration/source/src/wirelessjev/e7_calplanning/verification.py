"""Passive exact arithmetic replay, with no environment generation, fit or search."""
import numpy as np

from ..missions import WAREHOUSE_MISSIONS
from . import common as c,data


def arrays(path):
    with np.load(path,allow_pickle=False) as saved:return {k:saved[k].copy() for k in saved.files}


def reference(ref,path):
    if set(ref)!=set(('path','sha256')) or ref['path']!=str(path.relative_to(c.ROOT)):
        raise RuntimeError('E7 evidence must use canonical case paths')
    if c.digest(c.safe_path(ref['path'],c.ROOT))!=ref['sha256']:raise RuntimeError('E7 raw evidence changed')


def public_from_arrays(a):
    return data.public_fit_inputs([data.PublicAP(float(x),float(y),str(b),float(tx))
        for (x,y),b,tx in zip(a['ap_xy'],a['ap_band'],a['ap_tx'])],float(a['ap_height']))


def expected_refs(directory):
    result={k:directory/file for k,file in [('attempt','attempt.json'),('generation','generation.json'),('survey','survey.npz'),('public','public.npz'),
        ('shared_ORACLE','shared_ORACLE.npz'),('private_SCORING','private_SCORING.npz'),('choices','choices.npz'),
        ('choice_stats','choice_stats.json'),('truth_references','truth_references.npz')]}
    for b in c.BUDGETS:
        result[f'survey_b{b}']=directory/f'survey_b{b}.npz'
        for kind in c.KINDS:
            for name in ('mean_state','belief'):result[f'{name}_{b}_{kind}']=directory/f'{name}_{b}_{kind}.npz'
    return result


def allowed_actions(actions,shared):
    plans=np.asarray(actions).reshape(-1,100);visible=shared['visible'];starts=shared['starts']
    if (visible.shape!=(100,int(shared['m'])) or starts.shape!=(2,100)
            or starts.dtype.kind not in 'iu' or np.any((starts<0)|(starts>=int(shared['m'])))):
        raise RuntimeError('E7 admitted action masks or starts changed')
    visible_moves=visible[np.arange(100)[None],plans]
    # One search begins from one whole start. Partial visible moves may leave
    # its other coordinates nonvisible; distinct nonvisible starts cannot mix.
    reachable=np.any(np.all(visible_moves[:,None,:] | (plans[:,None,:]==starts[None]),axis=2),axis=1)
    if not reachable.all():raise RuntimeError('E7 action lies outside shared visible mask and admitted starts')


def verify_environment(seed):
    c.check_sources()
    if type(seed)is not int or seed not in c.SEEDS:raise ValueError('Unregistered environment')
    directory=c.OUT/'cases'/str(seed);receipt=c.read_json(directory/'receipt.json')
    if (receipt['seed']!=seed or receipt['source_receipt_sha256']!=c.digest(c.OUT/'registration/source.json')
            or receipt['protocol_sha256']!=c.digest(c.PROTOCOL) or receipt['scope']!=c.protocol_spec()['scope']
            or receipt['fits']!=6 or receipt['belief_searches']!=30 or receipt['reference_searches']!=5
            or receipt['production_qualified'] is not False or receipt['default_changed'] is not False
            or receipt['original_SERVICE_holdouts_opened'] is not False or receipt['new_packet_simulations']!=0
            or receipt['no_fresh_confirmation_claim'] is not True):
        raise RuntimeError('E7 case identity, budget or scope changed')
    expected=expected_refs(directory)
    if set(receipt['refs'])!=set(expected):raise RuntimeError('E7 input/model/score file membership changed')
    if ({p.name for p in directory.iterdir()}!={p.name for p in expected.values()}|{'receipt.json'}
            or any(not p.is_file() or p.is_symlink() for p in directory.iterdir())):
        raise RuntimeError('E7 canonical case membership changed')
    for key,path in expected.items():reference(receipt['refs'][key],path)
    attempt=c.read_json(directory/'attempt.json')
    if (attempt['seed']!=seed or attempt['source_receipt_sha256']!=receipt['source_receipt_sha256']
            or attempt['protocol_sha256']!=receipt['protocol_sha256'] or attempt['maximum_fits']!=6 or attempt['maximum_searches']!=35):
        raise RuntimeError('E7 exclusive attempt changed')
    generation=c.read_json(directory/'generation.json')
    if (generation['seed']!=seed or generation['attempt_sha256']!=c.digest(directory/'attempt.json')
            or generation['source_receipt_sha256']!=receipt['source_receipt_sha256']
            or generation['protocol_sha256']!=receipt['protocol_sha256']):
        raise RuntimeError('E7 single-use generation marker changed')
    public_arrays=arrays(directory/'public.npz');public=public_from_arrays(public_arrays)
    shared=arrays(directory/'shared_ORACLE.npz');private=arrays(directory/'private_SCORING.npz')
    full=data.survey_from_arrays(arrays(directory/'survey.npz'))
    selected=arrays(directory/'choices.npz');choices=selected['actions'];predicted=selected['predicted_utility']
    if choices.shape!=(3,2,5,100) or choices.dtype.kind not in 'iu' or np.any((choices<0)|(choices>=len(public.aps))):
        raise RuntimeError('E7 chosen action axes/range changed')
    if predicted.shape!=(3,2,5) or not np.isfinite(predicted).all():raise RuntimeError('E7 predicted utility axes changed')
    allowed_actions(choices,shared)
    stats=c.read_json(directory/'choice_stats.json')
    expected_order=[(b,k,m) for b in c.BUDGETS for k in c.KINDS for m in c.MISSIONS]
    if ([(s['budget'],s['kind'],s['mission']) for s in stats]!=expected_order
            or [(s['budget'],s['kind']) for s in receipt['fit_records']]!=[(b,k) for b in c.BUDGETS for k in c.KINDS]
            or [s['mission'] for s in receipt['reference_stats']]!=list(c.MISSIONS)
            or any(type(s['iterations'])is not int or not 0<=s['iterations']<=40
                   or type(s['plans_evaluated'])is not int or not 3<=s['plans_evaluated']<=32243
                   or not np.isfinite([s['search_ms'],s['setup_ms']]).all()
                   or min(s['search_ms'],s['setup_ms'])<0 for s in stats+receipt['reference_stats'])
            or any(not np.isfinite(s['elapsed_seconds']) or s['elapsed_seconds']<0 for s in receipt['fit_records'])):
        raise RuntimeError('E7 fit/search order or finite iteration budget changed')
    errors={}
    for bi,b in enumerate(c.BUDGETS):
        expected_survey=data.survey_arrays(data.tail_subset(full,b));actual=arrays(directory/f'survey_b{b}.npz')
        if set(actual)!=set(expected_survey) or any(not np.array_equal(actual[k],v) for k,v in expected_survey.items()):
            raise RuntimeError('E7 nominal tail report/link/busy selection differs')
        sub=data.survey_from_arrays(actual)
        for ki,kind in enumerate(c.KINDS):
            state=arrays(directory/f'mean_state_{b}_{kind}.npz')
            if (str(state['kind'])!=kind or state['Wf'].shape!=(200,2) or state['bf'].shape!=(200,)
                    or state['coefficient'].shape!=(len(public.aps),201) or state['present'].shape!=(len(public.aps),)
                    or state['law'].shape!=(len(data.BANDS),2) or state['rack_scale'].shape!=(len(data.BANDS),)
                    or any(not np.isfinite(state[k]).all() for k in ('Wf','bf','coefficient','law','rack_scale'))):
                raise RuntimeError('E7 fixed variant state changed')
            replay=lambda x,y:data.replay_mean(state,public,x,y)
            belief=data.make_belief(shared,public_arrays['positions'],public,sub,replay)
            saved=arrays(directory/f'belief_{b}_{kind}.npz')
            for name,value in {'ul':np.stack([s.ul for s in belief.snaps]),'dl':np.stack([s.dl for s in belief.snaps]),
                               'external_noise':belief.ext_noise_mw,'availability':belief.avail}.items():
                np.testing.assert_allclose(saved[name],value,rtol=1e-11,atol=1e-10,err_msg='E7 mean-map/planner input replay failed')
            target=private['dl'][0];mean=belief.snaps[0].dl
            errors[f'{b}/{kind}']={key:float(np.sqrt(np.mean((mean[mask]-target[mask])**2))) if mask.any() else None
                for key,mask in [('audible',target>=-90),('inaudible',target< -90),('all',np.ones(target.shape,bool))]}
            for mi,name in enumerate(c.MISSIONS):
                mission=WAREHOUSE_MISSIONS[name];focus=[belief.car_ids.index(x) for x in mission.focus_cars if x in belief.car_ids]
                value=float(belief.utility(choices[bi,ki,mi],mission.weights,focus)[0])
                if not np.isclose(value,predicted[bi,ki,mi],rtol=1e-11,atol=1e-10):raise RuntimeError('Predicted E7 utility replay failed')
    if set(receipt['map_error'])!=set(errors):raise RuntimeError('E7 map-error endpoint membership changed')
    for key,metrics in errors.items():
        if set(receipt['map_error'][key])!=set(metrics):raise RuntimeError('E7 map-error subset changed')
        for name,value in metrics.items():
            actual=receipt['map_error'][key][name]
            if (value is None and actual is not None or value is not None
                    and (actual is None or not np.isclose(actual,value,rtol=1e-11,atol=1e-10))):
                raise RuntimeError('E7 secondary map-error arithmetic changed')
    truth=data.evaluator(shared,list(zip(private['ul'],private['dl'])),external_noise=private['external_noise'],availability=private['availability'])
    references=arrays(directory/'truth_references.npz');rows=[]
    if (references['actions'].shape!=(5,100) or references['actions'].dtype.kind not in 'iu'
            or np.any((references['actions']<0)|(references['actions']>=len(public.aps)))
            or references['utility'].shape!=(5,) or not np.isfinite(references['utility']).all()):
        raise RuntimeError('E7 truth-informed reference axes or range changed')
    allowed_actions(references['actions'],shared)
    for mi,name in enumerate(c.MISSIONS):
        mission=WAREHOUSE_MISSIONS[name];focus=[truth.car_ids.index(x) for x in mission.focus_cars if x in truth.car_ids]
        ref=float(truth.utility(references['actions'][mi],mission.weights,focus)[0])
        if not np.isclose(ref,references['utility'][mi],rtol=1e-11,atol=1e-10):raise RuntimeError('E7 local-search reference score changed')
    for bi,b in enumerate(c.BUDGETS):
        for mi,name in enumerate(c.MISSIONS):
            mission=WAREHOUSE_MISSIONS[name];focus=[truth.car_ids.index(x) for x in mission.focus_cars if x in truth.car_ids]
            values=truth.utility(choices[bi,:,mi],mission.weights,focus);ref=float(references['utility'][mi])
            rows.append({'budget':b,'mission':name,'corrected_v2':float(values[0]),'geometry_censored':float(values[1]),
                'candidate_minus_v2':float(values[1]-values[0]),'truth_search_reference':ref,
                'legacy':float(truth.utility(shared['starts'][1],mission.weights,focus)[0]),
                'v2_signed_gap':float(ref-values[0]),'candidate_signed_gap':float(ref-values[1])})
    if len(receipt['paired_rows'])!=15:raise RuntimeError('All15 mission/budget pairs must be retained')
    for actual,expected in zip(receipt['paired_rows'],rows):
        if actual['budget']!=expected['budget'] or actual['mission']!=expected['mission']:raise RuntimeError('E7 paired row ordering changed')
        for key in set(expected)-{'budget','mission'}:
            if not np.isclose(actual[key],expected[key],rtol=1e-11,atol=1e-10):raise RuntimeError('E7 private score arithmetic changed')
    return {'passes':True,'seed':seed,'paired_rows':15,'new_fits_searches_or_environment_generation':False}


def audit():
    c.check_sources();report=c.read_json(c.OUT/'report.json');rows=[]
    if ({p.name for p in c.OUT.iterdir()}!={'protocol.json','registration','preflight','cases','report.json','verification'}
            or {p.name for p in (c.OUT/'cases').iterdir()}!={str(s) for s in c.SEEDS}
            or any(not p.is_dir() or p.is_symlink() for p in (c.OUT/'cases').iterdir())):
        raise RuntimeError('E7 diagnostic/global case membership changed')
    if (report['source_receipt_sha256']!=c.digest(c.OUT/'registration/source.json')
            or report['protocol_sha256']!=c.digest(c.PROTOCOL) or report['scope']!=c.protocol_spec()['scope']
            or report['seeds']!=list(c.SEEDS) or report['map_fits']!=90 or report['belief_searches']!=450
            or report['reference_searches']!=75 or report['default_changed'] is not False
            or report['original_SERVICE_holdouts_opened'] is not False
            or set(report['environment_receipts'])!={str(s) for s in c.SEEDS}):
        raise RuntimeError('E7 report identity, counts or scope changed')
    for seed in c.SEEDS:
        verify_environment(seed);path=c.OUT/'cases'/str(seed)/'receipt.json'
        if c.digest(path)!=report['environment_receipts'][str(seed)]:raise RuntimeError('E7 case receipt changed')
        rows.append(c.read_json(path))
    from .study import summarize
    if (summarize(rows)!=report['summary'] or report['paired_rows']!=225 or report['environment_units']!=15
            or report['production_qualified'] is not False):raise RuntimeError('E7 environment-clustered summary changed')
    return {'passes':True,'environment_units':15,'paired_rows':225,'map_fits':90,'total_searches':525,
        'no_new_fits_searches_or_generation':True,'scope':c.protocol_spec()['scope'],'production_qualified':False}


def seal():
    c.check_sources();path=c.OUT/'verification/manifest.json'
    evidence={str(p.relative_to(c.ROOT)):c.digest(p) for p in sorted(c.OUT.rglob('*'))
              if p.is_file() and p!=path and not p.is_relative_to(c.OUT/'registration/source')}
    value={'source_sha256':c.sources(),'evidence_sha256':evidence,'scope':c.protocol_spec()['scope'],'production_qualified':False}
    c.write_once(path,value);return value


def verify_closure():
    """Bind the final file set and bytes, then replay only saved arithmetic."""
    c.check_sources();path=c.OUT/'verification/manifest.json';value=c.read_json(path)
    evidence={str(p.relative_to(c.ROOT)):c.digest(p) for p in sorted(c.OUT.rglob('*'))
              if p.is_file() and p!=path and not p.is_relative_to(c.OUT/'registration/source')}
    if (value['source_sha256']!=c.sources() or value['evidence_sha256']!=evidence
            or value['scope']!=c.protocol_spec()['scope'] or value['production_qualified'] is not False):
        raise RuntimeError('E7 sealed closure membership, bytes or scope changed')
    return {**audit(),'sealed_evidence_files':len(evidence),'sealed_source_files':len(value['source_sha256'])}
