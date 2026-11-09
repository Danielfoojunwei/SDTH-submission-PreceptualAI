"""Retrospective service diagnosis; no new worlds, inference or old writes.

Outputs belong only to artifacts/jepa_repair_diagnosis. The completed selected
confirmation is observed data for repair design, never a fresh repair holdout.
"""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/selected_jepa'
OUTPUT = ROOT / 'artifacts/jepa_repair_diagnosis'
BASELINES = ('radiomap', 'frozen_fleet_jepa')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def key(row):
    return row['site'], float(row['handover_seconds']), row['family'], row['training_seed']


def label(row):
    return dict(zip(('site', 'handover_seconds', 'family', 'training_seed'), key(row)))


def merge(intervals, lower=0, upper=6000):
    result = []
    for a, b in sorted((max(lower, int(a)), min(upper, int(b))) for a, b in intervals
                       if min(upper, int(b)) > max(lower, int(a))):
        if result and a <= result[-1][1]:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a, b])
    return result


def miss_floor(intervals):
    # A packet generated at g expires at g+4. A length-L complete outage has
    # L-4 eligible generation slots; at least floor((L-4)/5) for any phase.
    return sum(max(0, (b - a - 4) // 5) for a, b in merge(intervals))


def handovers(row):
    actions = row['result']['packet_detail']['action_trace']
    previous = np.zeros(100, dtype=int)
    prior_ap = np.array(actions[0]['primary_site'], dtype=int)
    duration = round(row['handover_seconds'] * 100)
    events = [[] for _ in range(100)]
    for action in actions:
        until = np.array(action['primary_unavailable_until_slot'], dtype=int)
        aps = np.array(action['primary_site'], dtype=int)
        assert np.all(until >= previous)
        for robot in np.flatnonzero(until > previous):
            events[int(robot)].append({'start_slot': int(until[robot] - duration),
                'end_slot': int(until[robot]), 'observed_by_slot': action['slot_end'],
                'old_ap_at_previous_sample': int(prior_ap[robot]), 'new_ap': int(aps[robot])})
        previous, prior_ap = until, aps
    return events


def longest_interval(row, robot):
    """Infer a >1s maximum from exact current-gap/maximum snapshots.

    For a gap closing between snapshots, its start is visible in the previous
    snapshot and its length in the next maximum. End-of-trial gaps remain
    right censored; their true eventual duration is unknown.
    """
    d = row['result']['packet_detail']; maximum = d['max_gap'][robot]
    if maximum <= 100:
        return None
    trace = d['cumulative']
    hit = next(i for i, e in enumerate(trace) if e['max_gap'][robot] == maximum)
    e = trace[hit]
    if e['current_gap'][robot] == maximum:
        start = e['slot_end'] - maximum
    elif hit and maximum - 100 <= trace[hit-1]['current_gap'][robot] <= maximum:
        start = trace[hit-1]['slot_end'] - trace[hit-1]['current_gap'][robot]
    else:
        return None
    end = start + maximum
    assert 0 <= start < end <= e['slot_end']
    return {'start_slot': start, 'end_slot': end, 'seconds': maximum / 100,
            'right_censored_at_trial_end': end == 6000 and trace[-1]['current_gap'][robot] == maximum}


def plans(row, robot, start=0, end=6000):
    result = []
    for d in row['decision_trace']:
        i = d['decision']; ready = row['jobs'][i]['ready_slot']
        if start <= ready <= end:
            result.append({'slot': ready, 'forecast_decision': i,
                'age_seconds': ready / 100 - i, 'current_ap': d['current'][robot],
                'candidate_ap': d['candidate'][robot], 'gain': d['gain'][robot],
                'eligible': d['eligible'][robot], 'confidence_mask': d['confidence_mask'][robot],
                'after_ap_moves': d['after_ap_moves'][robot]})
    return result


def proxy_count(plan_rows):
    # This is not an admission-denial count: stagger quota and fast pending
    # execution can also leave after_ap_moves unchanged.
    return sum(bool(e['eligible'] and e['confidence_mask'] and
                    e['after_ap_moves'] == e['current_ap']) for e in plan_rows)


def anchor_context(row, robot, anchor, events):
    start, end = anchor['start_slot'], anchor['end_slot']; d = row['result']['packet_detail']
    ps = plans(row, robot, start, end)
    reports = [e for e in row['telemetry']['events'] if e['robot'] == robot]
    before = [e['delivery_slot'] for e in reports if e['delivery_slot'] < start]
    actions = [e for e in d['action_trace'] if start <= e['slot_end'] <= end]
    intervals = merge([(e['start_slot'], e['end_slot']) for e in events[robot]], start, end)
    lo, hi = math.ceil(start / 100), math.floor(end / 100)
    ticks = {e['slot_end']//100: e for e in d['cumulative']}
    count = lambda t, field: 0 if t == 0 else ticks[t][field][robot]
    return {**label(row), 'robot': robot, 'total_misses': d['misses'][robot],
        'max_gap_seconds': d['max_gap'][robot]/100, 'own_longest_gap': longest_interval(row, robot),
        'anchor_window': anchor, 'inside_full_second_window': [lo, hi],
        'misses_inside_full_seconds': count(hi, 'misses') - count(lo, 'misses'),
        'sent_inside_full_seconds': count(hi, 'sent') - count(lo, 'sent'),
        'primary_aps_sampled_in_anchor': sorted({e['primary_site'][robot] for e in actions}),
        'known_handover_outage_overlap_seconds': sum(b-a for a,b in intervals)/100,
        'all_robot_handover_events': events[robot],
        'last_report_before_anchor_seconds': max(before)/100 if before else None,
        'delivered_reports_in_anchor': sum(start <= e['delivery_slot'] <= end for e in reports),
        'planning_opportunities_in_anchor': len(ps),
        'eligible_plans_in_anchor': sum(bool(e['eligible']) for e in ps),
        'eligible_confident_unchanged_ap_proxy': proxy_count(ps),
        'all_plans_select_current_ap': bool(ps and all(e['candidate_ap'] == e['current_ap'] for e in ps)),
        'plans_in_anchor': ps}


def summary(row, events):
    d = row['result']['packet_detail']; intervals = [[(e['start_slot'], e['end_slot']) for e in r] for r in events]
    floors = [miss_floor(r) for r in intervals]; observed = sum(map(len, events))
    coverage = observed == row['result']['roams']
    if row['handover_seconds'] == 4.:
        assert coverage
    assert all(bound <= factual for bound, factual in zip(floors, d['misses']))
    assert sum(floors) <= row['result']['miss_causes']['no_radio_up']
    used = [j for j in row['jobs'] if j['ready_slot'] < 6000]
    all_plans = [e for i in range(100) for e in plans(row, i)]
    return {**label(row), 'control_misses': row['result']['control_misses'],
        'control_loss': row['result']['control_miss'], 'max_gap_seconds': max(d['max_gap'])/100,
        'roams': row['result']['roams'], 'observed_deadline_increases': observed,
        'handover_event_count_matches_roam_counter': coverage,
        'known_handover_robot_seconds': sum(sum(b-a for a,b in merge(r)) for r in intervals)/100,
        'handover_complete_outage_miss_lower_bound': sum(floors),
        'misses_not_accounted_by_this_lower_bound': row['result']['control_misses'] - sum(floors),
        'planned_decision_callbacks': len(row['decision_trace']), 'consumed_forecast_jobs': len(used),
        'max_forecast_ready_age_seconds': max(j['ready_slot']/100-j['decision'] for j in used),
        'mean_forecast_ready_age_seconds': statistics.mean(j['ready_slot']/100-j['decision'] for j in used),
        'eligible_confident_unchanged_ap_proxy': proxy_count(all_plans),
        'mean_field_rmse_db': row['forecast_metrics']['rmse_db'],
        'route_delta_rmse_db': row['forecast_metrics']['route_delta_rmse_db'],
        'ap_choice_proxy_regret_db': row['forecast_metrics']['ap_choice_regret_db'],
        **{'attributed_'+cause: value for cause,value in row['result']['miss_causes'].items()}}


def paired(row, reference, first_events, second_events):
    a,b = row['result']['packet_detail'],reference['result']['packet_detail']
    delta = np.array(a['misses'])-np.array(b['misses']); order=np.argsort(-delta)
    checkpoint=lambda d,t:sum(next(e['misses'] for e in d['cumulative'] if e['slot_end']==t*100))
    result={**label(row),'reference_family':reference['family'],'reference_training_seed':reference['training_seed'],
        'miss_difference':int(delta.sum()),'roam_difference':row['result']['roams']-reference['result']['roams'],
        'rmse_difference_db':row['forecast_metrics']['rmse_db']-reference['forecast_metrics']['rmse_db'],
        'route_delta_rmse_difference_db':row['forecast_metrics']['route_delta_rmse_db']-reference['forecast_metrics']['route_delta_rmse_db'],
        'ap_proxy_regret_difference_db':row['forecast_metrics']['ap_choice_regret_db']-reference['forecast_metrics']['ap_choice_regret_db'],
        'positive_robot_excess_sum':int(np.maximum(delta,0).sum()),
        'negative_robot_excess_sum':int(np.minimum(delta,0).sum()),
        'top_excess_robots':[{'robot':int(i),'miss_difference':int(delta[i]),'candidate_misses':a['misses'][i],
            'reference_misses':b['misses'][i],'candidate_max_gap_seconds':a['max_gap'][i]/100,
            'reference_max_gap_seconds':b['max_gap'][i]/100,'candidate_sampled_handovers':len(first_events[i]),
            'reference_sampled_handovers':len(second_events[i])} for i in order[:10]],
        'cumulative_excess_misses':{str(t):checkpoint(a,t)-checkpoint(b,t) for t in (15,30,45,60)}}
    for field in ('primary_site','control_band','bulk_band','secondary_band'):
        av=np.array([e[field] for e in a['action_trace']]);bv=np.array([e[field] for e in b['action_trace']])
        result['sampled_'+field+'_difference_fraction']=float(np.mean(av!=bv))
    return result


def self_checks():
    assert miss_floor([(0,400)])==79
    assert miss_floor([(0,400),(400,800)])==159
    assert miss_floor([(0,3)])==0
    assert miss_floor([(5997,6400)])==0


def main():
    self_checks()
    p=json.loads((SOURCE/'protocol.json').read_text());candidate=p['candidate_family'];data={};hashes={}
    for path in sorted((SOURCE/'cases').glob('*.json.gz')):
        receipt_path=path.with_name(path.name.removesuffix('.json.gz')+'.receipt.json')
        receipt=json.loads(receipt_path.read_text());assert digest(path)==receipt['archive_sha256']
        row=json.loads(gzip.decompress(path.read_bytes()));assert value_digest(row['result'])==receipt['result_sha256']
        assert row['stage']=='confirmation' and row['replay'] is False and key(row) not in data
        data[key(row)]=row
        hashes[str(path.relative_to(ROOT))]=digest(path);hashes[str(receipt_path.relative_to(ROOT))]=digest(receipt_path)
    assert len(data)==128
    events={identity:handovers(row) for identity,row in data.items()}
    summaries=[summary(row,events[identity]) for identity,row in data.items()]
    groups={}
    for cost in p['handover_seconds']:
        for family in (candidate,'supervised',*BASELINES):
            group=[r for r in summaries if r['handover_seconds']==cost and r['family']==family]
            numeric=[name for name in group[0] if name not in ('site','handover_seconds','family','training_seed','handover_event_count_matches_roam_counter')]
            groups[f'{cost:g}:{family}']={name:float(np.mean([r[name] for r in group])) for name in numeric}
            groups[f'{cost:g}:{family}']['cases']=len(group)
    comparisons=[]
    for identity,row in data.items():
        if row['family'] not in (candidate,'supervised'):
            continue
        for reference in list(BASELINES)+(['supervised'] if row['family']==candidate else []):
            seed=row['training_seed'] if reference=='supervised' else None
            second_key=row['site'],row['handover_seconds'],reference,seed
            comparisons.append(paired(row,data[second_key],events[identity],events[second_key]))
    mismatches={}
    for family in (candidate,'supervised'):
        for reference in (*BASELINES,'supervised'):
            group=[r for r in comparisons if r['family']==family and r['reference_family']==reference]
            if group:
                mismatches[family+'_vs_'+reference]={'paired_cases':len(group),
                    'lower_issued_rmse_but_more_misses':sum(r['rmse_difference_db']<0 and r['miss_difference']>0 for r in group),
                    'lower_AP_proxy_regret_but_more_misses':sum(r['ap_proxy_regret_difference_db']<0 and r['miss_difference']>0 for r in group)}
    # Include every site/robot/cost with a >=10s recorded gap in ANY family.
    focus={}
    for row in data.values():
        for robot,gap in enumerate(row['result']['packet_detail']['max_gap']):
            if gap>=1000:
                group=row['site'],row['handover_seconds'],robot;interval=longest_interval(row,robot)
                if interval is not None and (group not in focus or gap>focus[group]['seconds']*100):
                    focus[group]=interval
    critical=[]
    for (site,cost,robot),anchor in sorted(focus.items()):
        for identity,row in data.items():
            if row['site']==site and row['handover_seconds']==cost:
                critical.append(anchor_context(row,robot,anchor,events[identity]))
    slow=[r for r in summaries if r['handover_seconds']==4.]
    slow_floor=sum(r['handover_complete_outage_miss_lower_bound'] for r in slow)
    slow_misses=sum(r['control_misses'] for r in slow)
    findings={'analysis_scope':'Retrospective observational diagnosis of all 128 completed selected cases. This is observed repair-design data, never a new repair holdout.',
        'source_scientific_commit':'f5c69d55cf110578f7c48de96899a5879bb9f106',
        'source_report_sha256':digest(SOURCE/'report.json'),'analysis_script_sha256':digest(Path(__file__)),
        'main_cases':len(data),'new_worlds_models_inference_or_old_evidence_writes':False,
        'source_case_and_receipt_sha256':hashes,'case_family_means':groups,
        'slow_handover_bound':{'cases':len(slow),'reconstructed_handovers':sum(r['observed_deadline_increases'] for r in slow),
            'matches_all_native_roam_totals':all(r['handover_event_count_matches_roam_counter'] for r in slow),
            'guaranteed_miss_lower_bound':slow_floor,'observed_misses':slow_misses,
            'lower_bound_fraction_of_observed':slow_floor/slow_misses,
            'scope':'Conditional on factual handovers. Not avoidable regret or causal benefit of cancelling moves.'},
        'forecast_proxy_counterexamples':mismatches,'case_metrics':summaries,'paired_comparisons':comparisons,
        'largest_positive_case_excesses':sorted(comparisons,key=lambda r:r['miss_difference'],reverse=True)[:20],
        'critical_gap_site_robot_groups':len(focus),'critical_gap_contexts':critical,
        'interpretation_limits':[
            'Forecast errors score issued frames, not latency-adjusted consumed forecasts; mean RMSE/AP proxy are not native packet-risk metrics.',
            'Decision confidence is not admission. Eligible/confident/unchanged AP may reflect admission, stagger quota or fast cadence-pending execution; no explicit admission mask is archived.',
            'At slow cost utility scores already include outage cost. A reactive-admission veto is a distinct hypothesis from planned-confidence fallback suppression.',
            'Actual pilots, cached neighbor RSSI, per-robot cause flags and all reactive candidates/admission masks are not archived.',
            'Cadence scans start only following control delivery. Inside continuous control silence no fresh scan starts; a prior scan may finish near the beginning.',
            'A different policy succeeds under shared exogenous conditions, but this is not a same-state intervention or proof of exact preventable losses.',
            '100ms sampled actions may miss rapid fast handovers. Slow event counts are checked against exact raw roam totals.',
            'The handover bound uses unknown control phase conservatively. Remaining misses are not proved inevitable or causally attributed.',
            'Rankings and token counts are descriptive, not independent sites or post-selection significance claims.',
            'Recovery, scan scheduling and cadence hypotheses need new matched prospective experiments. No collapse, SOTA, deployment or infinite-stability conclusion follows.']}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'findings.json').write_text(json.dumps(findings,indent=2,allow_nan=False)+'\n')
    with (OUTPUT/'case_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    explanatory = [
        '# Observed JEPA service failures', '',
        'This diagnostic reads all 128 completed selected-study cases and changes no scientific evidence. '
        'The eight warehouse instances are the site units; repeated members are not additional sites. '
        'These outcomes are now observed repair-design data, not fresh holdouts.', '',
        '## The longest tail is sustained association failure, not a long handover', '',
        'At site 2114001, robot 44 has a right-censored 20.73 s control-silence interval from '
        '39.27 s through the trial end at 60 s. CTX all three members, SUP0/1 and RadioMap remain '
        'on AP10 throughout. SUP2 moves AP10→AP2 at 59.85 s, too late to recover. '
        'The old frozen JEPA moves AP10→AP2 at 14.11 s and exits its outage at 18.11 s.', '',
        'In the common full-second window [40,60], all new members and RadioMap lose 400/400 commands '
        'on this robot; the old frozen model loses 4/400. CTX records 21 planning callbacks during '
        'the long tail, all with candidate=current AP10, gain zero and eligibility/confidence false. '
        'Its last delivered RF report was at 34.96 s. There is no planned-eligible/admission-denied '
        'evidence for this particular tail. Reactive candidates and gate return masks were not archived.', '',
        'The source starts cadence scans only after a successful control delivery; continuous silence '
        'therefore prevents fresh scan starts. A scan initiated before silence may finish near its start. '
        'This is a real feedback constraint, but does not prove that allowing a scan or a different roam '
        'would causally save the observed number of losses.', '',
        '## Handover cost and model timing', '',
        f'All {len(slow)} slow cases reconstruct {sum(r["observed_deadline_increases"] for r in slow):,} '
        f'handovers, matching their native roam totals. Complete unavailable intervals imply at least '
        f'{slow_floor:,} of {slow_misses:,} observed misses ({100*slow_floor/slow_misses:.2f}%). '
        'This lower bound conditions on the chosen handovers; it is not avoidable regret.', '',
        'The frozen ensemble consumes 43 forecasts within each 60 s run, with maximum ready age '
        '17.98 s. The new members consume all 60 with maximum age 2.13 s; RadioMap reaches 1.13 s. '
        'These are ages under the sealed FIFO model. Issued-frame forecast errors do not measure '
        'the error of the actually consumed, aged forecast. The old model still roams more often at '
        'fast cost (135.75 versus CTX127.00 mean roams); fewer updates alone is not an explanation.', '',
        '## Proxy objectives do not imply better service', '',
        'Across 48 paired CTX/RadioMap case comparisons, 14 have lower issued-frame RMSE but more '
        'missed commands. Across 48 CTX/old-frozen comparisons, 21 have lower AP-choice proxy regret '
        'but more misses. These are counterexamples to a metric implication, not independent-site '
        'significance or proof of the cause.', '',
        'The largest CTX/old-model case excess is site2114005, member1: +1,002 misses. '
        'Robot31 alone contributes +282, with no CTX handover and a 16.73 s gap; the old model makes '
        'one handover and has a 4 s maximum gap on that robot. Other robots and earlier policy '
        'histories also differ, so this is not a same-state intervention.', '',
        '## Repair hypotheses requiring new matched tests', '',
        '- Put planned admission results in the confidence/fallback path and separately preserve '
        'reactive recovery; log planned eligibility, admission, reactive proposal and execution distinctly.',
        '- Test bounded emergency sensing when delivery-dependent observation collection becomes stale.',
        '- Record actual pilots, cached neighbor RSSI/ages, consumed AP scores and per-action outage '
        'predictions to distinguish wrong beliefs from wrong admission/recovery.',
        '- Compare every controller repair across CTX, supervised, RadioMap and the old model. '
        'A shared controller gain does not establish a JEPA-objective gain.', '',
        'No new world, inference, fitting, calibration or intervention was run. No collapse, SOTA, '
        'production, causal-benefit or indefinite-stability claim follows.', '',
        'Reproduce with `uv run --frozen python scripts/diagnose_jepa_service.py`. '
        '`findings.json` contains every case, paired robot excess, six long-gap groups and timing/proxy limitations.'
    ]
    (OUTPUT/'README.md').write_text('\n'.join(explanatory)+'\n')
    receipt={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'analysis_script_sha256':digest(Path(__file__)),
        'source_report_sha256':digest(SOURCE/'report.json'),'findings_sha256':digest(OUTPUT/'findings.json'),
        'case_metrics_sha256':digest(OUTPUT/'case_metrics.csv'),'readme_sha256':digest(OUTPUT/'README.md'),'scientific_sources_or_prior_outputs_changed':False}
    (OUTPUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'cases':len(data),'critical_gap_groups':len(focus),'slow_handover_bound':findings['slow_handover_bound'],
                      'forecast_proxy_counterexamples':mismatches,'output':str(OUTPUT)},indent=2))


if __name__=='__main__':
    main()
