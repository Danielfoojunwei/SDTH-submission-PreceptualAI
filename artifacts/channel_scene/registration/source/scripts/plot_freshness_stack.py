"""Render completed freshness-stack evidence from passive archives only.

No scientific package, model, simulator, dataset builder or training runner is
imported. Missing/changed evidence fails closed. Figures are derived artifacts.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right
from fractions import Fraction
import gzip
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('radiomap', 'frozen_fleet_jepa')
LABELS = {'radiomap': 'RadioMap', 'frozen_fleet_jepa': 'Frozen JEPA ensemble'}
STAGES = ('fifo', 'latest', 'latest_single', 'latest_single_sense')
FINAL = STAGES[-1]
VARS = {'fifo': 'FIFO', 'latest': 'Latest', 'latest_single': '+ Single', 'latest_single_sense': '+ Sensing'}
COLORS = {'radiomap': '#3278A5', 'frozen_fleet_jepa': '#007E83'}
INK, BLUE, TEAL, ORANGE, RED, PALE = '#203343', '#3278A5', '#007E83', '#BB862F', '#BC4C5E', '#F8FBFD'
REPLAY_FIELDS = ('result', 'forecast_mean_sha256', 'forecast_std_sha256', 'telemetry',
                 'decision_trace', 'admission_trace', 'local_scan_observability', 'action_trace_sha256', 'arrivals', 'queue_mode',
                 'accepted_decisions', 'forecast_metric_decisions')
SCHEDULE_FIELDS = ('decision', 'source_decision', 'job_index', 'mode', 'issue_seconds', 'start_seconds',
    'kind', 'history_cutoff_exclusive', 'ready_slot',
    'actual_prediction_seconds', 'actual_issue_seconds', 'actual_compute_seconds',
    'profile_seconds', 'charged_compute_seconds', 'charged_seconds', 'shared_extra_seconds')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value: raise ValueError(message)


def relative(path):
    try: return str(Path(path).resolve().relative_to(ROOT))
    except ValueError: return str(Path(path).resolve())


def finite(value):
    require(not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value), 'Expected a finite measured number')
    return float(value)


def close(a, b):
    return math.isclose(finite(a), finite(b), rel_tol=1e-11, abs_tol=1e-12)


def variants(cost):
    require(cost == 4., 'Freshness stack is a four-second-only study')
    return STAGES


def keys(p, stage, costs):
    return [(stage, site, float(cost), family, variant) for site in p[stage + '_sites']
            for cost in costs for family in FAMILIES for variant in variants(cost)]


def case_name(key):
    stage, site, cost, family, variant = key
    return f'{stage}-{site}-h{round(cost * 1000):04d}-{family}-{variant}'


def audit_jobs(row, p, profile):
    """Reconstruct sparse arrival admission BEFORE model work and actual charges."""
    jobs, arrivals = row['jobs'], row['arrivals']
    mode = 'fifo' if row['controller_variant'] == 'fifo' else 'latest'
    require(row['queue_mode'] == mode and len(arrivals) == p['duration_seconds'], 'Incomplete arrival ledger')
    previous = 0.; ready = []; sources = []
    for decision, arrival in enumerate(arrivals):
        accepted = mode == 'fifo' or float(decision) >= previous
        index = len(sources)
        require(arrival == {'decision': decision, 'arrival_slot': decision*100, 'accepted': accepted,
            'job_index': index if accepted else None, 'reason': 'accepted' if accepted else 'busy_previous_job',
            'previous_ready_seconds': previous, 'previous_ready_slot': math.ceil(previous*100-1e-10), 'mode': mode},
            'Arrival admission differs from original paid completion')
        if not accepted: continue
        require(index < len(jobs), 'Accepted request lacks a paid job')
        job = jobs[index]
        require(job['decision'] == job['source_decision'] == decision and job['job_index'] == index
                and job['mode'] == mode and job['history_cutoff_exclusive'] == 60 + decision
                and close(job['issue_seconds'],decision) and close(job['start_seconds'],max(float(decision),previous)),
                'Source timestamp, sparse ordinal or causal prefix differs')
        actual = max(finite(job['actual_prediction_seconds']), finite(job['actual_issue_seconds']))
        compute = max(profile[job['kind']], actual)
        require(actual >= 0 and close(job['actual_compute_seconds'], actual)
                and close(job['profile_seconds'],profile[job['kind']])
                and close(job['shared_extra_seconds'],p['extra_compute_seconds'])
                and close(job['charged_compute_seconds'], compute)
                and close(job['charged_seconds'], compute + p['extra_compute_seconds'])
                and job['runtime_overrun'] is (actual > profile[job['kind']]),
                'Actual forecast work was not paid or capacity flag changed')
        previous = max(float(decision), previous) + job['charged_seconds']
        require(job['ready_slot'] == math.ceil(previous * 100 - 1e-10), 'Actual-cost completion differs')
        ready.append(job['ready_slot']); sources.append(decision)
    require(len(sources) == len(jobs) and row['accepted_decisions'] == row['forecast_metric_decisions'] == sources,
        'Computed sparse rows differ from accepted source decisions')
    expected, last = [], -1
    for slot in range(p['duration_seconds'] * 100):
        index = bisect_right(ready, slot) - 1
        if index > last:
            expected.append({'slot': slot, 'forecast_decision': sources[index], 'ready_slot': ready[index],
                             'age_seconds': slot / 100 - sources[index]})
            last = index
    require(row['forecast_consumption'] == expected, 'Consumed forecast precedes completion or is omitted')
    require(row['runtime_overrun'] is any(job['runtime_overrun'] for job in jobs)
            and row['paid_cost_and_consumption_pass'] is True, 'Paid/capacity summary changed')


def timing_metrics(row, p):
    jobs, consumed = row['jobs'], row['forecast_consumption']
    held = [(consumed[i+1]['slot']-1 if i+1<len(consumed) else p['duration_seconds']*100-1)/100-e['forecast_decision']
        for i,e in enumerate(consumed)]
    return {'accepted':len(jobs),'skipped':len(row['arrivals'])-len(jobs),'consumed':len(consumed),
        'unconsumed':len(jobs)-len(consumed),
        'initial_fallback_slots':consumed[0]['slot'] if consumed else p['duration_seconds']*100,
        'consumed_age_max':max((e['age_seconds'] for e in consumed),default=None),
        'held_age_max':max(held,default=None),
        'actual_compute_total':sum(j['actual_compute_seconds'] for j in jobs),
        'charged_service_total':sum(j['charged_seconds'] for j in jobs)}


def scan_metrics(row, p):
    """Audit actual whole scans and positive suppression using archived operands."""
    execution = row['result']['freshness_execution']; final = row['controller_variant'] == FINAL
    require(execution['risk_gated_scans_enabled'] is final, 'Sensing variant changed')
    starts, assessments = execution['scan_starts'], execution['scan_assessments']
    require(all(e['complete_slot']-e['slot']==1+round(p['scan_outage_seconds']*100) for e in starts),
        'Whole-scan completion/outage changed')
    require(final or not assessments, 'Non-sensing stage used suppression')
    jobs={j['decision']:j for j in row['jobs']}; consumed=row['forecast_consumption']
    horizons=(0.,1.,2.,3.,4.,5.,10.,20.,30.)
    def interpolate(t, values):
        i=min(bisect_right(horizons,t)-1,len(horizons)-2)
        return values[i]+(values[i+1]-values[i])*(t-horizons[i])/(horizons[i+1]-horizons[i])
    suppressed=0
    for e in assessments:
        if not e['suppress_scan']:continue
        suppressed+=1;slot=e['slot'];source=e['source_decision'];age=slot/100-source
        current=next((c for c in reversed(consumed) if c['slot']<=slot),None)
        require(current is not None and current['forecast_decision']==source and source in jobs
            and e['forecast_ready_slot']==jobs[source]['ready_slot']<=slot
            and e['forecast_sha256']==jobs[source]['forecast_sha256']
            and close(e['forecast_age_seconds'],age) and 0<=age<=5 and age+5<=30
            and e['primary_available'] is True and e['control_gap_slots']<20
            and e['pilot_site']==e['current_site']
            and 0<=slot-e['pilot_refresh_slot']<10 and 0<=slot-e['pilot_capture_slot']<20,
            'Suppression lacks a fresh consumed forecast/current association pilot')
        rssi=[finite(a)+finite(b)+finite(c)-15. for a,b,c in zip(e['measured_snr_db'],e['noise_dbm'],e['ap_tx_dbm'],strict=True)]
        require(len(rssi)==3 and all(close(a,b) for a,b in zip(rssi,e['current_rssi_dbm'],strict=True))
            and close(max(rssi),e['current_best_rssi_dbm']) and max(rssi)>=-67.,'Suppression pilot arithmetic differs')
        original=e['original_forecast_snr_db']
        require(len(original)==9 and all(len(v)==3 for v in original),'Suppression original forecast axes differ')
        shifted=[[interpolate(age+t,[finite(v[b]) for v in original]) for b in range(3)] for t in range(6)]
        innovation=[[v[b]-shifted[0][b] for b in range(3)] for v in shifted]
        predicted=max(min(rssi[b]+v[b] for v in innovation) for b in range(3))
        require(all(close(a,b) for actual,want in zip(e['forecast_at_age_plus_samples_snr_db'],shifted,strict=True)
                for a,b in zip(actual,want,strict=True))
            and all(close(a,b) for actual,want in zip(e['route_innovation_db'],innovation,strict=True)
                for a,b in zip(actual,want,strict=True))
            and close(predicted,e['predicted_best_worst_rssi_dbm']) and predicted>=-67.,
            'Suppression fixed-band worst-lookahead arithmetic differs')
    duplication=finite(row['result']['dup_fraction'])
    require(row['controller_variant'] not in STAGES[2:] or duplication==0.,'Single stage still uses STR duplication')
    return {'scan_starts':len(starts),'suppressed_assessments':suppressed,'scan_assessments':len(assessments),
        'scheduled_scan_outage_robot_seconds':len(starts)*p['scan_outage_seconds'],
        'dup_fraction':duplication,'airtime_domain_sum':finite(row['result']['airtime_domain_sum'])}


def packet_metrics(row, p):
    detail, flow = row['result']['packet_detail'], row['result']['packet_accounting']
    sent, misses, gap = detail['sent'], detail['misses'], detail['max_gap']
    pending = [int(age >= 0) for age in detail['pending_control_packet_age_slots']]
    require(len(sent) == len(misses) == len(gap) == len(pending) == p['robots']
            and all(type(x) is int and x == 20 * p['duration_seconds'] for x in sent)
            and all(type(x) is int and 0 <= x <= sent[i] for i, x in enumerate(misses)),
            'Native per-robot command counters differ')
    total_sent, total_miss = sum(sent), sum(misses)
    require(row['result']['control_sent'] == total_sent and row['result']['control_misses'] == total_miss,
            'Native global counters differ from robot counters')
    decile = max(1, p['robots'] // 10)
    rates = [m / s for m, s in zip(misses, sent)]
    worst = sum(sorted(rates)[-decile:]) / decile
    pending_worst = sum(sorted((m + q) / s for m, q, s in zip(misses, pending, sent))[-decile:]) / decile
    return {'control_sent': total_sent, 'control_misses': total_miss,
        'command_loss': total_miss / total_sent,
        'command_loss_pending_as_misses': (total_miss + sum(pending)) / total_sent,
        'worst_decile_loss': worst, 'worst_decile_pending_as_misses': pending_worst,
        'max_silence_seconds': max(gap) / 100,
        'video_vs_original': flow['video_delivered_bits'] / max(flow['video_requested_bits'], 1e-9),
        'bulk_mbps': flow['bulk_delivered_bits'] / p['duration_seconds'] / 1e6,
        'roams': row['result']['roams']}


def load_inputs(folder):
    required = ('protocol.json', 'registration/source.json', 'registration/git.json',
                'development.json', 'qualification.json', 'completion.json', 'report.json')
    paths = [folder / name for name in required]
    require(all(path.is_file() for path in paths), 'Finish the registered pipeline before rendering')
    p, reg, meta, dev, qualification, completion, report = [read(path) for path in paths]
    costs = qualification['passed_costs']
    require(tuple(p['families']) == FAMILIES and p['handover_seconds'] == [4.]
            and p['variants_by_cost'] == {'4.0': list(STAGES)}
            and p['robots'] == 100 and p['duration_seconds'] == 60 and p['bootstrap_replicates'] == 20000
            and close(p['extra_compute_seconds'], .5)
            and len(p['confirmation_sites']) == 8 and len(p['development_sites']) == 2
            and len(set(p['confirmation_sites'] + p['development_sites'])) == 10
            and set(costs) <= {4.} and len(costs) == len(set(costs)), 'Protocol/site/cost identity changed')
    require(report['source_sha256'] == completion['source_sha256'] == reg['source_sha256']
            and report['completion_sha256'] == digest(folder / 'completion.json')
            and report['development_sha256'] == qualification['development_sha256'] == digest(folder / 'development.json')
            and completion['qualification_sha256'] == report['qualification_sha256'] == digest(folder / 'qualification.json')
            and report['passed_development_costs'] == completion['passed_costs'] == costs
            and dev['gate']['passed_costs'] == costs and report['development_gate'] == dev['gate']
            and completion['completed'] is True,
            'Final source/permission/report/completion identities differ')
    require(report['primary_multiplicity_budget'] == 3 and len(report['primary']) == 3
            and close(report['primary_upper_quantile'], 1 - .05 / 3)
            and report['production_qualified'] is False and report['hardware_qualified'] is False
            and report['ema_necessity_claimed'] is False and report['activated'] is False,
            'Final claim scope changed')
    execution_order = folder / 'execution_order.json'
    require(execution_order.is_file()
            and completion['execution_order_sha256'] == report['execution_order_sha256']
            == dev['execution_order_sha256'] == digest(execution_order), 'Registered execution order changed')
    paths.append(execution_order)
    for name, expected in reg['source_sha256'].items():
        require(digest(ROOT / name) == expected
                and digest(folder / 'registration/source' / name) == expected, 'Frozen source/capsule/input changed: ' + name)
    latency_path = ROOT / 'artifacts/semantic_jepa/latency.json'; latency = read(latency_path)
    paths.append(latency_path)
    profiles = latency['profiles']
    main_keys = keys(p, 'development', p['handover_seconds']) + keys(p, 'confirmation', costs)
    repeats = [key for key in keys(p, 'confirmation', costs) if key[1] == p['confirmation_sites'][0]]
    row_data, main_fingerprints, offline_checks, fresh, receipt_hashes = {}, {}, [], [], {}
    metrics = {(r['stage'], r['site'], r['handover_seconds'], r['family'], r['controller_variant']): r
               for r in report['every_case']}
    require(set(metrics) == set(keys(p, 'confirmation', costs)), 'Final metric matrix incomplete')
    for subfolder, expected, replay, live in [('cases', main_keys, False, False),
            ('replays', repeats, True, False), ('live_replications', repeats, False, True)]:
        directory = folder / subfolder
        actual = {path.name for path in directory.iterdir() if path.is_file()
                  and not path.name.startswith('.') and not path.name.endswith(('.lock', '.part'))} if directory.exists() else set()
        require(actual == {case_name(k) + suffix for k in expected for suffix in ('.json.gz', '.receipt.json')},
                'Unregistered/incomplete archive matrix: ' + subfolder)
        for key in expected:
            path = directory / (case_name(key) + '.json.gz'); receipt_path = path.with_name(case_name(key) + '.receipt.json')
            receipt = read(receipt_path); row = json.loads(gzip.decompress(path.read_bytes()))
            if key[0]=='confirmation':
                receipt_hashes[subfolder+'/'+receipt_path.name]=digest(receipt_path)
            mode = 'recorded_schedule_offline' if replay else 'actual_cost'
            require(receipt['archive_sha256'] == digest(path) and receipt['result_sha256'] == value_digest(row['result'])
                    and receipt['case'] == row['case'] == list(key) and row['completed'] is True
                    and row['replay'] is replay and row['live_replication'] is live
                    and receipt['replay'] is replay and receipt['live_replication'] is live
                    and row['accounting_mode'] == receipt['accounting_mode'] == mode
                    and row['stage'] == key[0] and row['site'] == key[1] and row['handover_seconds'] == key[2]
                    and row['family'] == key[3] and row['controller_variant'] == key[4]
                    and row['training_seed'] is None and row['simulation_only'] is True
                    and row['production_qualified'] is False and row['hardware_qualified'] is False
                    and row['checkpoint_sha256'] == reg['checkpoint_sha256']
                    and row['source_sha256'] == reg['source_sha256']
                    and row['registration_sha256'] == digest(folder / 'registration/source.json')
                    and row['protocol_sha256'] == digest(folder / 'protocol.json')
                    and row['latency_sha256'] == digest(latency_path), 'Raw archive identity changed: ' + path.name)
            profile = profiles[key[3]]
            if not replay: audit_jobs(row, p, profile)
            require(row['causality_pass'] is True and all(
                e['measurement_slot'] < e['complete_slot'] <= e['delivery_slot']
                and e['first_usable_decision'] == e['delivery_slot'] // 100 + 1 for e in row['telemetry']['events']),
                'Delivered telemetry is noncausal')
            fingerprint = {field: value_digest(row[field]) for field in REPLAY_FIELDS}
            if not replay and not live:
                main_fingerprints[key] = {'fields': fingerprint, 'jobs': row['jobs'],
                    'consumption': row['forecast_consumption'], 'arrivals': row['arrivals'], 'archive_sha': digest(path)}
                values = packet_metrics(row, p)
                if key[0] == 'confirmation':
                    require(all(close(metrics[key][field], value) for field, value in values.items() if field != 'roams'),
                            'Report differs from native counters')
                execution = row['result']['repair_execution']
                require(execution['emergency_recovery_enabled'] is True, 'Recovery flag differs from registered variant')
                for event in execution['emergency_handovers']:
                    require(event['available_silence_slots'] >= event['handover_threshold_slots'] == 420
                        and event['available_silence_slots_after'] == 0
                        and event['silence_slots'] >= 20 and event['current_rssi_dbm'] < -75
                        and event['neighbor_gain_db'] >= 8 and event['neighbor_rssi_dbm'] >= -90
                        and close(event['neighbor_gain_db'], event['neighbor_rssi_dbm'] - event['current_rssi_dbm'])
                        and event['neighbor_heard'] is True
                        and 0 <= event['neighbor_capture_age_slots'] <= event['neighbor_max_age_slots']
                        and event['neighbor_capture_age_slots'] == event['slot'] - event['neighbor_capture_slot']
                        and event['outage_until_slot'] - event['slot'] == round(key[2] * 100)
                        and event['pending_cadence_target_after'] == -1, 'A logged recovery violates its observed trigger or outage')
                require(execution['handover_threshold_slots'] == 420, 'Available-radio threshold differs')
                timing, sensing = timing_metrics(row,p), scan_metrics(row,p)
                if key[0]=='confirmation':
                    reported=metrics[key]['timing']['maximum_held_forecast_age_seconds']
                    require((reported is None and timing['held_age_max'] is None or
                        reported is not None and timing['held_age_max'] is not None and close(reported,timing['held_age_max']))
                        and metrics[key]['timing']['forecast_jobs']==timing['accepted']
                        and metrics[key]['timing']['skipped_arrivals']==timing['skipped'],
                        'Report sparse timing differs from actual interval accounting')
                row_data[key] = {**values, **timing, **sensing, 'jobs': row['jobs'],
                    'startup_seconds': row['predictor_startup_seconds'],
                    'emergency_handovers': len(execution['emergency_handovers']),
                    'emergency_scans': len(execution['emergency_scans']),
                    'watchdog_checks': len(execution['watchdog_checks'])}
            elif replay:
                original = main_fingerprints[key]
                schedule = all(all(a.get(field) == b.get(field) for field in SCHEDULE_FIELDS)
                               for a, b in zip(row['jobs'], original['jobs'], strict=True))
                exact = (fingerprint == original['fields'] and schedule
                    and row['forecast_consumption'] == original['consumption']
                    and row['source_main_archive_sha256'] == original['archive_sha']
                    and row['arrivals']==original['arrivals']
                    and row['recorded_schedule_sha256'] == value_digest({'jobs':original['jobs'],'arrivals':original['arrivals']})
                    and row['recorded_arrivals_sha256'] == value_digest(original['arrivals'])
                    and all(j['matches_recorded_forecast'] is True and j['offline_recorded_schedule'] is True
                            and j['completion_uses_actual_elapsed'] is False for j in row['jobs']))
                offline_checks.append({'case': list(key), 'exact': exact})
            else:
                require(row['source_main_archive_sha256'] == main_fingerprints[key]['archive_sha'],
                        'Fresh replication changed its main-case binding')
                original = row_data[key]; now = packet_metrics(row, p)
                before={j['decision']:j for j in main_fingerprints[key]['jobs']}
                after={j['decision']:j for j in row['jobs']}; shared=sorted(set(before)&set(after))
                fresh.append({'case': list(key), 'miss_difference': now['control_misses'] - original['control_misses'],
                    'changed_ready_jobs':sum(before[k]['ready_slot']!=after[k]['ready_slot'] for k in shared),
                    'added_source_decisions':sorted(set(after)-set(before)),
                    'removed_source_decisions':sorted(set(before)-set(after)),
                    'arrival_admission_exact':row['arrivals']==main_fingerprints[key]['arrivals']})
            paths.extend((path, receipt_path))
    expected_costs = []
    for cost in p['handover_seconds']:
        severe = False
        for family in FAMILIES:
            rates = {}
            for variant in ('fifo', FINAL):
                part = [row_data['development', site, cost, family, variant] for site in p['development_sites']]
                rates[variant] = Fraction(sum(r['control_misses'] for r in part), sum(r['control_sent'] for r in part))
            severe |= rates[FINAL] - rates['fifo'] > Fraction(str(p['development_severe_regression_absolute']))
        require(next(r['severe_regression'] for r in dev['gate']['per_cost'] if r['handover_seconds'] == cost) is bool(severe),
                'Development regression status differs from exact native counters')
        if not severe: expected_costs.append(cost)
    require(costs == expected_costs, 'Confirmation subset differs from mechanically valid development veto')
    require(completion['case_count'] == report['case_count'] == len(keys(p, 'confirmation', costs))
            and completion['archive_receipt_sha256']==receipt_hashes
            and completion['offline_replay_count'] == completion['fresh_live_replication_count'] == len(repeats)
            and completion['all_offline_exact'] is all(r['exact'] for r in offline_checks)
            and {tuple(r['case']): r['exact'] for r in completion['offline_comparisons']} ==
                {tuple(r['case']): r['exact'] for r in offline_checks}, 'Completion/replay claims differ from raw archives')
    require(completion['all_fresh_paid_cost_checks_pass'] is True
            and report['proof']['source_and_archive_pass'] is True
            and report['proof']['mechanical_integrity_pass'] is True
            and report['proof']['paid_cost_and_consumption_pass'] is True
            and report['proof']['causality_pass'] is True
            and report['proof']['exact_offline_replay_pass'] is all(r['exact'] for r in offline_checks)
            and report['proof']['experimental_claim_integrity_pass'] is all(r['exact'] for r in offline_checks),
            'Report proof differs from actual namespaces')
    priors=[]
    previous_hashes={}
    for study in ('jepa_repair','costaware_recovery'):
        parent_folder=ROOT/'artifacts'/study
        parent_paths=[parent_folder/'protocol.json',parent_folder/'report.json',parent_folder/'figures/confirmation.svg',
            parent_folder/'verification/manifest.json']
        require(all(path.is_file() for path in parent_paths),'Retain prior completed study evidence: '+study)
        parent_p,parent_report=read(parent_paths[0]),read(parent_paths[1])
        manifest=read(parent_paths[3])
        for field in ('sources_sha256','evidence_sha256'):
            require(isinstance(manifest[field],dict) and manifest[field],'Missing prior frozen evidence map')
            for name,expected in manifest[field].items():
                path=Path(name)
                require(not path.is_absolute() and '..' not in path.parts
                    and (name not in previous_hashes or previous_hashes[name]==expected)
                    and digest(ROOT/path)==expected,'Previously completed source/evidence changed: '+name)
                previous_hashes[name]=expected;paths.append(ROOT/path)
        require(not set(p['development_sites']+p['confirmation_sites']) &
            set(parent_p['development_sites']+parent_p['confirmation_sites']),'New stack reuses prior sites')
        priors.append({'study':study,'figure':relative(parent_paths[2]),'figure_sha256':digest(parent_paths[2]),
            'report_sha256':digest(parent_paths[1]),'all_primary_experimental_benefits':parent_report['all_three_experimental_benefits']
                if study=='costaware_recovery' else parent_report['all_six_experimental_benefits'],
            'new_sites_disjoint_from_prior':True})
        paths.extend(parent_paths)
    return p,report,profiles,row_data,paths,offline_checks,fresh,priors


def validate_statistics(np, p, report, data):
    """Derive all plotted means/contrasts from native whole-site counters."""
    sites = p['confirmation_sites']; rng = np.random.default_rng(p['bootstrap_seed'])
    indices = rng.integers(0, len(sites), size=(p['bootstrap_replicates'], len(sites)))
    values = lambda cost, arm, field: np.asarray([data['confirmation', site, cost, *arm][field] for site in sites])
    for row in report['means']:
        arm = row['family'], row['controller_variant']; cost = row['handover_seconds']
        require(close(row['command_loss'], values(cost, arm, 'command_loss').mean()), 'Plotted mean differs')
    expected = {(float(cost), label) for cost in p['handover_seconds'] for label in
                ('final_minus_fifo:radiomap', 'final_minus_fifo:frozen_fleet_jepa', 'final_jepa_minus_final_radiomap')}
    require({(r['handover_seconds'], r['contrast_id']) for r in report['primary']} == expected,
            'A registered contrast is missing or substituted')
    require(report['all_three_experimental_benefits'] is all(r['experimental_benefit'] for r in report['primary']),
            'Whole-study three-comparison flag differs')
    for row in report['primary']:
        label = row['contrast_id']
        if label.startswith('final_minus_fifo:'):
            family = label.split(':', 1)[1]
            require((row['candidate_family'], row['candidate_variant'], row['reference_family'], row['reference_variant'])
                    == (family, FINAL, family, 'fifo'), 'Repair contrast changed arms')
        else:
            require((row['candidate_family'], row['candidate_variant'], row['reference_family'], row['reference_variant'])
                    == ('frozen_fleet_jepa', FINAL, 'radiomap', FINAL), 'Model contrast changed arms')
        require((row['status'] == 'tested') is (row['handover_seconds'] in report['passed_development_costs']),
                'Tested versus development-vetoed cost differs')
        if row['status'] != 'tested':
            require(row['mean_difference'] is None and row['experimental_benefit'] is False,
                    'Unrun contrast is represented as an outcome'); continue
        cost = row['handover_seconds']; candidate = row['candidate_family'], row['candidate_variant']
        reference = row['reference_family'], row['reference_variant']
        difference = values(cost, candidate, 'command_loss') - values(cost, reference, 'command_loss')
        endpoint = values(cost, candidate, 'command_loss_pending_as_misses') - values(cost, reference, 'command_loss')
        draws, pending_draws = difference[indices].mean(1), endpoint[indices].mean(1)
        interval = np.quantile(draws, [.025, .975]); upper = np.quantile(draws, p['primary_upper_quantile'])
        endpoint_upper = np.quantile(pending_draws, p['primary_upper_quantile'])
        require(len(row['paired_site_differences']) == len(sites) and len(row['descriptive_95_interval']) == 2,
            'Paired observation/interval axes differ')
        require(close(row['mean_difference'], difference.mean()) and all(close(a, b) for a, b in zip(row['paired_site_differences'], difference))
            and all(close(a, b) for a, b in zip(row['descriptive_95_interval'], interval))
            and close(row['multiplicity_upper'], upper)
            and close(row['endpoint_sensitivity']['multiplicity_upper'], endpoint_upper)
            and row['statistical_pass'] is bool(upper < 0 and endpoint_upper < 0), 'Primary bootstrap differs from paired native counters')
        engineering = next(g for g in report['engineering_gates'] if g['contrast_id'] == row['contrast_id'] and g['handover_seconds'] == cost)
        candidate_worst = values(cost, candidate, 'worst_decile_pending_as_misses').mean()
        reference_worst = values(cost, reference, 'worst_decile_loss').mean()
        checks = {'worst_decile': candidate_worst <= reference_worst + p['worst_decile_margin'],
            'maximum_silence': values(cost, candidate, 'max_silence_seconds').max() <= values(cost, reference, 'max_silence_seconds').max() + p['silence_margin_seconds'],
            'original_video': values(cost, candidate, 'video_vs_original').mean() >= values(cost, reference, 'video_vs_original').mean() - p['video_margin'],
            'bulk_delivery': values(cost, candidate, 'bulk_mbps').mean() >= p['bulk_ratio'] * values(cost, reference, 'bulk_mbps').mean()}
        require(engineering['checks'] == checks and row['engineering_pass'] is bool(all(checks.values()))
            and row['experimental_benefit'] is bool(row['statistical_pass'] and row['engineering_pass']
                  and report['proof']['experimental_claim_integrity_pass']), 'Engineering/benefit flag differs')
    expected_secondary={f'nested:{family}:{after}_minus_{before}':(family,before,after)
        for family in FAMILIES for before,after in zip(STAGES,STAGES[1:])}
    secondary=report['secondary_component_contrasts']
    require(len(secondary)==6 and {r['contrast_id'] for r in secondary}==set(expected_secondary),
        'Nested descriptive contrasts missing/substituted')
    for row in secondary:
        family,before,after=expected_secondary[row['contrast_id']]
        require((row['candidate_family'],row['candidate_variant'],row['reference_family'],row['reference_variant'])
            ==(family,after,family,before) and row['statistical_pass'] is None
            and row['statistical_superiority_claimed'] is False and row['experimental_benefit'] is False,
            'Descriptive nested contrast changed scope or arms')
        if row['status']!='tested':
            require(not report['passed_development_costs'] and row['mean_difference'] is None,'Unrun nested estimate claimed')
            continue
        difference=values(4.,(family,after),'command_loss')-values(4.,(family,before),'command_loss')
        interval=np.quantile(difference[indices].mean(1),[.025,.975])
        require(len(row['paired_site_differences'])==len(sites) and len(row['descriptive_95_interval'])==2
            and close(row['mean_difference'],difference.mean())
            and all(close(a,b) for a,b in zip(row['paired_site_differences'],difference))
            and all(close(a,b) for a,b in zip(row['descriptive_95_interval'],interval)),
            'Nested descriptive interval differs from paired raw counters')


def style(plt):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'text.color': INK,
        'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#C4D2DD',
        'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'axes.facecolor': 'white',
        'figure.facecolor': PALE, 'savefig.facecolor': PALE, 'svg.fonttype': 'none'})


def tidy(ax):
    ax.grid(axis='x', color='#DCE5EB', alpha=.7, linewidth=.7); ax.set_axisbelow(True)
    ax.spines['left'].set_color('#D3DFE7'); ax.tick_params(length=3)


def primary_label(row):
    if row['contrast_id'] == 'final_jepa_minus_final_radiomap': return 'Final JEPA vs final Map'
    return 'Final vs FIFO: ' + ('JEPA' if row['candidate_family'] == 'frozen_fleet_jepa' else 'RadioMap')


def status_panel(ax, records):
    ax.axis('off'); ax.set_xlim(0, 3.3); ax.set_ylim(-.5, 3.2)
    for index, title in enumerate(('Statistics', 'Engineering', 'Benefit')):
        ax.text(index + .42, 2.95, title, ha='center', fontsize=9, fontweight='bold')
    for index, row in enumerate(records):
        for column, field in enumerate(('statistical_pass', 'engineering_pass', 'experimental_benefit')):
            unrun = row['status'] != 'tested'; passed = row[field] is True
            label, color, face = ('UNRUN', '#6F7F8B', '#EDF1F4') if unrun else (
                ('PASS', TEAL, '#E3F2ED') if passed else ('FAIL', RED, '#F9EDEF'))
            ax.text(column + .42, 2.45 - index * .95, label, ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color,
                    bbox={'boxstyle': 'round,pad=.22', 'facecolor': face, 'edgecolor': 'none'})


def prior_references(fig, priors, y=.045):
    fig.text(.045,y,'Earlier failures, nulls and gates remain unchanged. New stack sites are disjoint from both prior studies.',
        fontsize=8.1,color=RED)
    for i,prior in enumerate(priors):
        fig.text(.045+i*.47,y-.022,prior['figure'],fontsize=8.,color=BLUE,
            url='../../'+prior['study']+'/figures/confirmation.svg')


def no_estimate(ax, text='4 s confirmation cases were not run: development veto.'):
    ax.axis('off');ax.text(.5,.5,text,transform=ax.transAxes,ha='center',va='center',
        color='#657684',fontsize=9.5,linespacing=1.8)


def stage_plot(ax,np,p,report,data,field,scale=1.,aggregate='mean',show_sites=True):
    if not report['passed_development_costs']:
        no_estimate(ax);return
    for number,family in enumerate(FAMILIES):
        means=[];offset=(-.055,.055)[number]
        for index,variant in enumerate(STAGES):
            raw=np.asarray([data['confirmation',site,4.,family,variant][field] for site in p['confirmation_sites']])
            values=np.asarray([v for v in raw if v is not None],dtype=float)*scale
            means.append(float(np.mean(values) if aggregate=='mean' else np.max(values)) if len(values) else float('nan'))
            if show_sites and len(values):
                ax.scatter(index+offset+np.linspace(-.022,.022,len(values)),values,s=15,alpha=.30,
                    color=COLORS[family],edgecolors='none')
        ax.plot(np.arange(4)+offset,means,marker='D',markersize=4.5,color=COLORS[family],
            linewidth=1.7,label='Map' if number==0 else 'JEPA ensemble')
    ax.set_xticks(range(4),[VARS[v] for v in STAGES],fontsize=8.5)
    ax.set_xlim(-.32,3.32);ax.grid(axis='y',color='#DCE5EB',alpha=.7,linewidth=.7);ax.set_axisbelow(True)


def evidence_figure(plt,np,p,report,data,priors):
    from matplotlib.lines import Line2D
    fig=plt.figure(figsize=(18.,11.8))
    gs=fig.add_gridspec(2,3,width_ratios=[1.02,1.52,.93],left=.075,right=.98,top=.83,bottom=.255,
        hspace=.65,wspace=.61)
    fig.suptitle('Forecast freshness, radio resources and whole-scan sensing',x=.045,y=.977,
        ha='left',fontsize=19,fontweight='bold')
    fig.text(.045,.935,f"100 simulated robots × 60 s · 4 s handovers · {report['independent_sites']} confirmation sites · "
        f"{report['case_count']} factual runs · original fixed model weights",fontsize=11.3)
    fig.text(.045,.892,'Four fixed nested stages. Final package is the only primary candidate; intermediate stages are descriptive and cannot be selected.',
        fontsize=10.4,color='#516574')
    ax=fig.add_subplot(gs[0,0]);ax.set_title('A  Mean fleet command loss',loc='left',pad=11)
    stage_plot(ax,np,p,report,data,'command_loss',100);ax.set_ylabel('Commands missing deadline (%)')
    if report['passed_development_costs']:ax.legend(frameon=False,fontsize=8.2)
    ax=fig.add_subplot(gs[0,1]);ax.set_title('B  Three registered primary contrasts',loc='left',pad=11)
    tested=bool(report['passed_development_costs'])
    if tested:
        ax.axvline(0,color=INK,linestyle='--',linewidth=1.)
        for index,row in enumerate(report['primary']):
            color=COLORS[row['candidate_family']]
            low,high=(-np.asarray(row['descriptive_95_interval'])*1e5)[::-1]
            ax.scatter(-np.asarray(row['paired_site_differences'])*1e5,index+np.linspace(-.05,.05,8),
                color='#AAB9C2',s=20,alpha=.7)
            ax.plot([low,high],[index,index],color=color,linewidth=3,solid_capstyle='round')
            ax.scatter(-row['mean_difference']*1e5,index,color=INK,marker='D',s=42,zorder=4)
            ax.scatter(-row['multiplicity_upper']*1e5,index-.11,color=ORANGE,marker='^',s=45,zorder=4)
            ax.scatter(-row['endpoint_sensitivity']['multiplicity_upper']*1e5,index+.11,color=RED,marker='v',s=45,zorder=4)
        ax.set_yticks(range(3),[primary_label(r) for r in report['primary']],fontsize=8.5)
        ax.set_ylim(2.6,-.6);tidy(ax)
        ax.set_xlabel('Avoided misses per 100,000 generated commands\n(reference − candidate; positive favors candidate)')
    else:no_estimate(ax)
    ax=fig.add_subplot(gs[0,2]);status_panel(ax,report['primary']);ax.set_title('C  Registered primary gates',loc='left',pad=11)
    ax=fig.add_subplot(gs[1,0]);ax.set_title('D  Worst-decile command loss',loc='left',pad=11)
    stage_plot(ax,np,p,report,data,'worst_decile_pending_as_misses',100)
    ax.set_ylabel('Mean worst 10 robots per site (%)\nPending commands counted as misses')
    ax=fig.add_subplot(gs[1,1]);ax.set_title('E  Nested component contrasts: descriptive only',loc='left',pad=11)
    if tested:
        ax.axvline(0,color=INK,linestyle='--',linewidth=1.)
        labels=[]
        for index,row in enumerate(report['secondary_component_contrasts']):
            color=COLORS[row['candidate_family']]
            low,high=(-np.asarray(row['descriptive_95_interval'])*1e5)[::-1]
            ax.plot([low,high],[index,index],color=color,linewidth=2.5,solid_capstyle='round')
            ax.scatter(-row['mean_difference']*1e5,index,color=color,marker='D',s=32,zorder=4)
            family='Map' if row['candidate_family']=='radiomap' else 'JEPA'
            labels.append(f"{family}: {VARS[row['candidate_variant']]} vs {VARS[row['reference_variant']]}")
        ax.set_yticks(range(6),labels,fontsize=8.);ax.set_ylim(5.6,-.6);tidy(ax)
        ax.set_xlabel('Mean reduction and descriptive 95% interval\nNot factorial; no superiority test or stage selection')
    else:no_estimate(ax)
    ax=fig.add_subplot(gs[1,2]);ax.set_title('F  Longest observed control silence',loc='left',pad=11)
    stage_plot(ax,np,p,report,data,'max_silence_seconds',aggregate='max')
    ax.set_ylabel('Seconds; line = maximum over eight sites')
    handles=[Line2D([],[],color='#AAB9C2',marker='o',linestyle='',label='Eight paired whole-site observations'),
        Line2D([],[],color=INK,marker='D',linestyle='',label='Mean primary reduction'),
        Line2D([],[],color=TEAL,linewidth=3,label='Descriptive 95% site-bootstrap interval'),
        Line2D([],[],color=ORANGE,marker='^',linestyle='',label='Multiplicity-adjusted lower bound'),
        Line2D([],[],color=RED,marker='v',linestyle='',label='Conservative endpoint lower bound')]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.045,.173),frameon=False,ncol=3,fontsize=8.3)
    dev=report['development_gate']['per_cost'][0]
    devtext='; '.join(('Map' if r['family']=='radiomap' else 'JEPA')+f" {r['final_minus_fifo']*100:+.3f} pp" for r in dev['family_checks'])
    fig.text(.045,.145,'Development final − FIFO: '+devtext+f"; status: {dev['status']}. No tuning or intermediate-stage substitution.\n"
        'All three primary tests retain 1 − 0.05/3 bounds from 20,000 whole-site bootstrap draws; observed and conservative pending-command bounds must both favor candidate.\n'
        'Engineering gates retain worst-decile loss, maximum silence, original video and bulk margins. '
        f"Source / causal / paid / exact offline integrity: {report['proof']['experimental_claim_integrity_pass']}.\n"
        'Fixed original JEPA ensemble is one treatment. No new fitting, matched-query RF-accuracy claim, JEPA-objective necessity, hardware qualification or SOTA claim.',
        fontsize=8.1,color='#516574',linespacing=1.65,va='top')
    prior_references(fig,priors)
    return fig


def architecture(ax):
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    nodes=[(.01,.49,.21,.40,'FIFO\nevery request computed\nSTR duplication · full scans','#E9F0F4'),
        (.27,.49,.21,.40,'Latest\nbusy arrivals skipped before work\nno cancellation or overlap','#E3E9F4'),
        (.53,.49,.21,.40,'Latest + single\nSTR simultaneous duplication off\nsequential retries remain','#DDF0EC'),
        (.79,.49,.20,.40,'Final + sensing\nconditional whole 90 ms scans\nfresh pilot + held forecast','#F6EDDA')]
    for x,y,w,h,text,color in nodes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',facecolor=color,edgecolor='#ADC5D3',linewidth=1.))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9.,linespacing=1.55)
    for x in (.23,.49,.75):ax.add_patch(FancyArrowPatch((x,.69),(x+.03,.69),arrowstyle='-|>',mutation_scale=12,color='#567A8A',linewidth=1.5))
    ax.text(.5,.29,'Shared: fixed RadioMap / original JEPA · actual paid FIFO service per accepted job · ordinary outage utility + all-handover admission\n'
        '420 available-radio silent slots for emergency handover; full 4 s paid · current pilots / completed heard neighbors only',
        ha='center',va='center',fontsize=9.1,color=INK,linespacing=1.65)
    ax.text(.5,.06,'Sensing: pilot AP matches, refresh <100 ms; consumed age ≤5 s; current and best-fixed-band worst 0–5 s RSSI ≥−67 dBm.\n'
        'Suppression preserves due=True; reassess after 5 s; 200 ms silence can start the whole due scan immediately. No partial scans or extra observations.',
        ha='center',va='center',fontsize=8.5,color='#516574',linespacing=1.6)


def operations_figure(plt,np,p,report,data,offline,fresh,priors):
    fig=plt.figure(figsize=(18.,13.1))
    gs=fig.add_gridspec(3,3,left=.065,right=.98,top=.86,bottom=.19,height_ratios=[.97,1.,1.],hspace=.68,wspace=.4)
    fig.suptitle('Where the enabling stack spends compute, radio time and information',x=.045,y=.98,
        ha='left',fontsize=18.5,fontweight='bold')
    fig.text(.045,.943,'Actual accepted jobs, source-time age and native whole-scan accounting; all four nested stages are applied to both fixed model families.',fontsize=11)
    ax=fig.add_subplot(gs[0,:]);architecture(ax);ax.set_title('A  Fixed nested intervention stack; no intermediate-arm selection',loc='left',pad=10)
    ax=fig.add_subplot(gs[1,0]);ax.set_title('B  Exact held age and consumption age',loc='left',pad=10)
    stage_plot(ax,np,p,report,data,'held_age_max',aggregate='max',show_sites=False)
    if report['passed_development_costs']:
        for number,family in enumerate(FAMILIES):
            age=[]
            for variant in STAGES:
                measured=[data['confirmation',site,4.,family,variant]['consumed_age_max'] for site in p['confirmation_sites']]
                valid=[value for value in measured if value is not None]
                age.append(max(valid) if valid else float('nan'))
            ax.plot(np.arange(4)+(-.055,.055)[number],age,color=COLORS[family],linestyle='--',linewidth=1.4)
        ax.legend(frameon=False,fontsize=7.9)
    ax.set_ylabel('Maximum seconds over all sites')
    ax.set_xlabel('Solid: held through next consumption −1 slot\nDashed: age when consumed; gaps = no completed forecast',fontsize=8.)
    confirmed={k:r for k,r in data.items() if k[0]=='confirmation'}
    arms=[(family,variant) for family in FAMILIES for variant in STAGES]
    rows=lambda arm:[confirmed['confirmation',site,4.,*arm] for site in p['confirmation_sites']]
    mean=lambda arm,field:float(np.mean([r[field] for r in rows(arm)]))
    ax=fig.add_subplot(gs[1,1]);ax.set_title('C  Sixty arrivals: computed, skipped, consumed',loc='left',pad=10)
    if confirmed:
        accepted=[mean(arm,'accepted') for arm in arms];skipped=[mean(arm,'skipped') for arm in arms]
        ax.bar(range(8),accepted,color=[COLORS[f] for f,v in arms],width=.65,label='Accepted')
        ax.bar(range(8),skipped,bottom=accepted,color='#DDE5EB',width=.65,label='Skipped')
        ax.scatter(range(8),[mean(arm,'consumed') for arm in arms],color=INK,marker='D',s=25,label='Consumed',zorder=4)
        ax.set_xticks(range(8),['M'+str(i+1) for i in range(4)]+['J'+str(i+1) for i in range(4)],fontsize=8.)
        ax.set_ylim(0,74);ax.set_ylabel('Mean requests per 60 s trial')
        ax.legend(frameon=False,fontsize=7.4,loc='upper center',ncol=3,columnspacing=.8,handletextpad=.4)
        ax.set_xlabel('Stages 1–4 in stack order; M = Map, J = JEPA\nAccepted but unconsumed jobs are still paid',fontsize=8.)
    else:no_estimate(ax)
    ax=fig.add_subplot(gs[1,2]);ax.set_title('D  Every accepted job pays its full service',loc='left',pad=10)
    stage_plot(ax,np,p,report,data,'charged_service_total',show_sites=False)
    if confirmed:
        for number,family in enumerate(FAMILIES):
            ax.plot(np.arange(4)+(-.055,.055)[number],[mean((family,v),'actual_compute_total') for v in STAGES],
                color=COLORS[family],linestyle='--',linewidth=1.4)
    ax.set_ylabel('Mean summed job seconds per trial')
    ax.set_xlabel('Solid: max(sealed allowance, actual issue)+0.5 s\nDashed: measured actual issue compute',fontsize=8.)
    ax=fig.add_subplot(gs[2,0]);ax.set_title('E  Actual whole-scan starts and suppression',loc='left',pad=10)
    stage_plot(ax,np,p,report,data,'scan_starts',show_sites=False)
    ax.set_ylabel('Mean scan starts per fleet trial')
    if confirmed:
        ax.set_xlabel('Each actual start schedules a full 90 ms outage\nFinal suppressed assessments: '+
            f"Map {mean(('radiomap',FINAL),'suppressed_assessments'):.1f}; JEPA {mean(('frozen_fleet_jepa',FINAL),'suppressed_assessments'):.1f}\n"
            'Assessments are not unique skipped scans',fontsize=7.9)
    ax=fig.add_subplot(gs[2,1]);ax.axis('off');ax.set_title('F  STR resource and original-traffic tradeoffs',loc='left',pad=10)
    if confirmed:
        table=[]
        for i,arm in enumerate(arms):table.append([('M' if arm[0]=='radiomap' else 'J')+str(i%4+1),
            f"{mean(arm,'dup_fraction')*100:.2f}",f"{mean(arm,'video_vs_original')*100:.2f}",f"{mean(arm,'bulk_mbps'):.2f}",
            f"{mean(arm,'scheduled_scan_outage_robot_seconds'):.2f}"])
        artist=ax.table(cellText=table,colLabels=['Arm','STR slots\n(%)','Video/orig.\n(%)','Bulk\n(Mbps)','Scan charge\n(robot s)'],
            cellLoc='center',colLoc='center',colWidths=[.11,.19,.22,.20,.28],bbox=[0,.12,1,.87])
        artist.auto_set_font_size(False);artist.set_fontsize(7.4)
        for (r,c),cell in artist.get_celld().items():
            cell.set_edgecolor('#D3DFE7');cell.set_facecolor('#E9F0F4' if r==0 else 'white')
        ax.text(0,-.03,'STR = pending-control duplicated robot-slots / all robot-slots.\nScan charge = starts ×0.09 s; not net de-overlapped airtime.',
            transform=ax.transAxes,fontsize=7.6,color='#516574',va='top',linespacing=1.6)
    else:no_estimate(ax)
    ax=fig.add_subplot(gs[2,2]);ax.axis('off');ax.set_title('G  Paid-cost, archive and replication scope',loc='left',pad=10)
    lines=[f"Factual confirmation cases: {len(confirmed)}",f"All source / paid / causal checks: {report['proof']['mechanical_integrity_pass']}",
        f"Actual accepted jobs: {sum(r['accepted'] for r in confirmed.values()):,}",
        f"Skipped before computation: {sum(r['skipped'] for r in confirmed.values()):,}",
        f"Accepted but unconsumed: {sum(r['unconsumed'] for r in confirmed.values()):,}",
        f"Maximum initial reactive fallback: {max(r['initial_fallback_slots'] for r in confirmed.values())/100:.2f} s" if confirmed else 'Initial reactive fallback: unrun',
        f"Profile-exceeded cases: {sum(any(j['runtime_overrun'] for j in r['jobs']) for r in confirmed.values())} (paid diagnostic)",
        f"Exact offline original-schedule replays: {sum(r['exact'] for r in offline)} / {len(offline)}",
        f"Fresh-cost repetitions: {len(fresh)}; extra sites: 0",
        f"Fresh changed accepted source sets: {sum(bool(r['added_source_decisions'] or r['removed_source_decisions']) for r in fresh)}",
        f"Fresh maximum absolute miss change: {max((abs(r['miss_difference']) for r in fresh),default=0)}" if fresh else 'Fresh miss sensitivity: unrun']
    ax.text(0,.96,'\n'.join(lines),transform=ax.transAxes,va='top',fontsize=8.6,color='#516574',linespacing=1.8)
    fig.text(.045,.13,'Held-age extrema use every native slot from consumption through the slot before replacement; initial reactive fallback has no forecast age. '
        'One-second right-boundary snapshots are not substituted.\n'
        'Whole-scan suppression changes paid sensing and delivered information. Latest rejects busy arrivals before work; no overlap or cancellation discount. '
        'Single disables simultaneous STR duplication; sequential retries remain.\n'
        'Actual history-to-forecast issue work is charged; startup is separate. Admission/scoring/scan-gating CPU is not independently scheduled; common +0.5 s remains an assumption.\n'
        'Fixed weights, shared procedural warehouse CAD, assumed routes/pilots, warm histories and finite episodes limit transfer. No model-objective, real-car, full-stack real-time or production claim.',
        fontsize=8.1,color='#516574',linespacing=1.65,va='top')
    prior_references(fig,priors)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=ROOT/'artifacts/freshness_stack')
    parser.add_argument('--output',type=Path,help='Default: FOLDER/figures')
    args=parser.parse_args()
    p,report,profiles,data,inputs,offline,fresh,priors=load_inputs(args.folder)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    style(plt);validate_statistics(np,p,report,data)
    figures=[('confirmation-and-nested-stages',evidence_figure(plt,np,p,report,data,priors)),
        ('freshness-resource-sensing-stack',operations_figure(plt,np,p,report,data,offline,fresh,priors))]
    output=args.output or args.folder/'figures';output.mkdir(parents=True,exist_ok=True);outputs=[]
    for name,fig in figures:
        for extension in ('svg','pdf','png'):
            path=output/f'{name}.{extension}';fig.savefig(path,dpi=190,metadata={'Creator':'plot_freshness_stack.py'});outputs.append(path)
        plt.close(fig)
    receipt={'schema':1,'derived_only':True,'scientific_execution':False,'new_fitting_or_selection':False,
        'script_sha256':digest(Path(__file__)),'report_sha256':digest(args.folder/'report.json'),
        'inputs_sha256':{relative(path):digest(path) for path in sorted(set(inputs))},
        'outputs_sha256':{relative(path):digest(path) for path in outputs},
        'software':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
        'primary_comparisons':report['primary'],'secondary_descriptive_only':report['secondary_component_contrasts'],
        'all_three_experimental_benefits':report['all_three_experimental_benefits'],'passed_development_costs':report['passed_development_costs'],
        'paired_reduction_units':'avoided misses per100000 generated commands; reference minus candidate',
        'uncertainty':'20k whole-site bootstrap; three primary1-.05/3 observed AND conservative endpoint bounds; six nested95% descriptive only',
        'held_age_scope':'Every consumed interval through next consumption slot-1 or final native slot; source decision not job ordinal; no initial fallback age',
        'offline_replay_checks':offline,'fresh_actual_cost_sensitivity':fresh,'prior_studies':priors,
        'per_case_derived_diagnostics':[{k:v for k,v in row.items() if k!='jobs'}|{'case':list(key)} for key,row in data.items()],
        'no_physical_or_production_qualification':True,'limits':report['limits']}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'figures':[relative(path) for path in outputs],'receipt':relative(output/'receipt.json'),'scientific_execution':False}))


if __name__=='__main__':main()

