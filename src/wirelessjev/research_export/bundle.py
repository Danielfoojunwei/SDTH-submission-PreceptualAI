"""Augment the real legacy Studio without editing its frozen generator/assets.

Reads JSON/text and hashes evidence bytes. Never imports a learner, opens NPZ
arrays, evaluates a scientific case, sends credentials or publishes a site.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re


ROOT=Path(__file__).resolve().parents[3]
DEFAULT_CACHE=ROOT.parent/'training/channel-scene'
SEALS={
    'measured_power':'artifacts/radiojepa/verification/manifest.json',
    'spectral_field':'artifacts/radiojepa_spectral/verification/manifest.json',
    'phone_transfer':'artifacts/device_transfer/verification/manifest.json',
    'complex_csi':'artifacts/channel_scene/verification/manifest.json',
    'native_bridge':'artifacts/native_bridge/verification/manifest.json',
}


def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


class Evidence:
    """Read-only evidence boundaries and a per-export verified-byte cache."""
    def __init__(self,root,cache_root=None):
        self.root=Path(root).resolve()
        self.cache_root=Path(cache_root or os.environ.get('CHANNEL_SCENE_CACHE_ROOT',str(DEFAULT_CACHE))).resolve()
        self.sources={};self.verified={};self.sealed={};self.external={};self.file_stats={}

    def path(self,name,*,external=False):
        relative=Path(name);root=self.cache_root if external else self.root
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe evidence path')
        path=root/relative
        if not path.resolve().is_relative_to(root):raise ValueError('Redirected evidence path')
        for parent in (path,*path.parents):
            if parent==root:break
            if parent.is_symlink():raise ValueError('Redirected evidence path')
        return path

    def digest(self,name,*,expected=None,source=False,external=False):
        path=self.path(name,external=external);key=('external:' if external else '')+name
        actual=self.verified.get(key)
        stat=path.stat();signature=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if actual is None or self.file_stats.get(key)!=signature:
            current=file_sha(path)
            if actual is not None and current!=actual:raise ValueError('Evidence bytes changed during export: '+name)
            actual=current;self.verified[key]=actual;self.file_stats[key]=signature
        if expected is not None and actual!=expected:raise ValueError('Evidence bytes changed: '+name)
        if source:self.sources[name]=actual
        if external:self.external[name]=actual
        return actual

    def finish(self):
        """Catch input mutations during export, without rereading stable data arrays."""
        for name,expected in tuple(self.verified.items()):
            external=name.startswith('external:')
            self.digest(name.removeprefix('external:'),expected=expected,external=external)

    def load(self,name,*,expected=None):
        self.digest(name,expected=expected,source=True)
        return json.loads(self.path(name).read_text())

    def text(self,name,*,expected=None):
        self.digest(name,expected=expected,source=True)
        return self.path(name).read_text()

    def seal(self,name):
        value=self.load(name);counts={}
        for key in ('source_sha256','evidence_sha256','documents_sha256'):
            mapping=value.get(key,{})
            if not isinstance(mapping,dict):raise ValueError('Malformed seal: '+name)
            counts[key]=len(mapping)
            for path,expected in mapping.items():self.digest(path,expected=expected)
        if not counts['source_sha256'] or not counts['evidence_sha256']:
            raise ValueError('Source and evidence closure required: '+name)
        result={'path':name,'sha256':self.sources[name],'status':'repository_hashes_verified',
            'source_files':counts['source_sha256'],'evidence_files':counts['evidence_sha256'],
            'document_files':counts['documents_sha256']}
        self.sealed[name]=value
        return value,result

    def report(self,name,seal):
        expected=seal.get('evidence_sha256',{}).get(name) or seal.get('documents_sha256',{}).get(name)
        if expected is None:raise ValueError('Report is not bound by the completed seal: '+name)
        return self.load(name,expected=expected)


def track(title,kind,status,*,scope,units,metrics=None,limits=(),proof=None,**extra):
    value={'title':title,'evidence_kind':kind,'status':status,'scope':scope,'units':units,
        'limits':list(limits),'production_qualified':False,'sota_established':False,**extra}
    if metrics is not None:value['metrics']=metrics
    if proof is not None:value['proof']=proof
    return value


def measured_power(e):
    seal,proof=e.seal(SEALS['measured_power']);r=e.report('artifacts/radiojepa/report.json',seal)
    a=r['aggregate'];rows=[]
    for cell,value in a['cells'].items():
        for recording,domain in value['domains'].items():
            rmse=domain['prediction_metrics']['rmse_db']
            rows.append({'cell':cell,'recording':recording,'mean_rmse_db':rmse['mean'],
                'training_seed_sd_db':rmse['training_seed_sd'],'individual_rmse_db':rmse['individual']})
    factory=[x for x in rows if x['recording']=='dichasus-cf06']
    best=min(factory,key=lambda x:x['mean_rmse_db'])
    baselines={recording:{model:{k:v[k] for k in ('rmse_db','mae_db') if k in v}
        for model,v in models.items()} for recording,models in a['baseline_metrics'].items()}
    return track('Measured array-power prediction','measured_radio','sealed',proof=proof,
        scope=a['scope'],units={'rmse':'dB of referenced channel power; not dBm'},
        metrics={'rows':rows,'completed_fits':a['completed_fits'],'training_seeds':a['training_seeds'],
            'baselines_by_recording':baselines,'factory_comparison':{'recording':'dichasus-cf06',
                'best_reported_neural_cell':best['cell'],'best_reported_neural_rmse_db':best['mean_rmse_db'],
                'constant_rmse_db':baselines['dichasus-cf06']['constant']['rmse_db'],'descriptive_only':True}},
        limits=['Three training seeds are not independent sites.','No isolated consistency-objective advantage or robot-service qualification.'],
        rights={'dataset':'DICHASUS','license':'CC BY 4.0','attribution':'Original release authors; see acquisition audit.',
            'reference':'docs/RADIOJEPA_CSI_DATA_AUDIT.md'})


def spectral(e):
    seal,proof=e.seal(SEALS['spectral_field']);r=e.report('artifacts/radiojepa_spectral/report.json',seal)
    return track('Static spectral RF field','sionna_simulation','sealed',proof=proof,
        scope='24 FIELD layouts after fixed TRAIN/VAL selection; static simulated propagation',
        units={'rmse':'dB','p95_absolute_error':'dB'},
        metrics={'selected_cell':r['selected_cell'],'contrasts':r['contrasts'],
            'matched_alignment_benefit_supported':r['matched_alignment_benefit_supported'],
            'seed_main_rmse_db':r['seed_main_rmse_db'],'coverage':r['coverage']},
        limits=['The selected model improves RF error over the RadioMap operator; matched regression is nearly tied.',
            'This does not establish a JEPA-objective gain, calibrated packet risk or native service reliability.',
            'Layout intervals are conditional on the trained members; all primary query paths were heard.'])


def phones(e):
    seal,proof=e.seal(SEALS['phone_transfer']);r=e.report('artifacts/device_transfer/report.json',seal)
    p=e.load('artifacts/device_transfer/protocol.json')
    return track('Real held-phone localization and calibration','measured_localization','sealed',proof=proof,
        scope='Five held-out phones, one site; official held-out positions; classical references',
        units={'localization_error':'meters'},
        metrics={k:r[k] for k in ('dataset','heldout_phones','physical_sites','official_test_fingerprints',
            'case_count','summaries','support_draws','independent_pretrained_neural_runs')},
        limits=r['limits']+['Support draws are calibration-sampling variation, not independently pretrained neural models.',
            'Some phone-level tails worsen; the best mean is not uniformly best for every device.'],
        rights={'dataset':'TUJI1','license':p['license'],'source':'https://doi.org/10.5281/zenodo.7641701'})


def csi(e):
    seal,proof=e.seal(SEALS['complex_csi']);r=e.report('artifacts/channel_scene/report.json',seal)
    external=seal['external_prediction_sha256']
    for path,expected in external.items():e.digest(path,expected=expected,external=True)
    for key in ('cache_manifest','prediction_capsule'):
        ref=seal[key];e.digest(ref['path'],expected=ref['sha256'],external=True)
    proof.update({'external_prediction_shards_verified':len(external),
        'requires_separate_prediction_capsule':True,'prediction_capsule':seal['prediction_capsule']})
    return track('Measured complex-CSI completion','measured_radio','sealed',proof=proof,
        scope=r['task'],units={'packet_nmse':'dimensionless error energy / target channel energy; lower is better'},
        metrics={'summary':r['summary'],'negative_neural_result':True},limits=r['limitations']+[
            'Neither neural arm beats the strongest simple reference; full-basis Fourier ridge behaves as zero fill on target bins.',
            'The Git PR requires the separately retained prediction capsule for complete passive numerical replay.'],
        rights={'dataset':'DICHASUS','license':'CC BY 4.0','reference':'docs/channel_scene/README.md'})


def native(e):
    seal,proof=e.seal(SEALS['native_bridge']);r=e.report('artifacts/native_bridge/summary.json',seal)
    weights=e.report('artifacts/native_bridge/weights.json',seal);cases=[]
    for case_id,expected in r['cases'].items():
        receipt=e.report(f'artifacts/native_bridge/cases/{case_id}/receipt.json',seal)
        if e.sources[f'artifacts/native_bridge/cases/{case_id}/receipt.json']!=expected:
            raise ValueError('Native summary case identity differs')
        cases.append({'case_id':case_id,'parity':receipt['parity'],'handover_seconds':receipt['case']['handover_seconds'],
            'metrics':receipt['metrics'],
            'paired_delta_candidate_minus_Map':receipt['paired_delta_candidate_minus_Map']})
    means={}
    for cost in r['separate_handover_results']:
        selected=[x for x in cases if not x['parity'] and x['handover_seconds']==float(cost)]
        means[cost]={arm:{'control_miss_fraction':sum(x['metrics'][arm]['control_miss'] for x in selected)/len(selected)}
            for arm in ('radiomap','candidate')}
    return track('100-robot native bridge diagnostic','native_simulation','sealed',proof=proof,
        scope='Four 12-second simulated episodes × two handover costs plus a wiring pair; 18 simulations',
        units={'control_miss':'fraction','gaps':'milliseconds','bulk':'Mbps','handover':'seconds','compute':'seconds'},
        metrics={'selected_cell':weights['selected_cell'],'selected_seed':weights['seed'],
            'separate_handover_results':r['separate_handover_results'],'cases':cases,
            'arm_means_by_handover':means,
            'native_simulations':r['native_simulations'],'wiring_proof':r['wiring_proof']},
        limits=['The candidate is the selected regression seed 0 member, not evidence of a JEPA gain.',
            'All four fast-handover cases worsen control misses; two slow-handover cases are harmed despite the mean gain.',
            'A robot experiences 9.74 seconds maximum silence. Higher bulk throughput does not erase control harms.',
            'RadioMap remains the default; no original SERVICE confirmation, hardware or production qualification.'])


def adaptive(e):
    name='artifacts/adaptive_radio/verification/manifest.json'
    if not e.path(name).exists():
        return track('Adaptive reported-RF assay','sionna_simulation','pending_seal',
            scope='Source-published fixed assay; no final TEST metric is exported until closure exists',
            units={'rf_error':'dB'},limits=['DEV permission and completed fits do not constitute sealed TEST evidence.',
                'Original SERVICE confirmation remains separate.'])
    seal,proof=e.seal(name);r=e.report('artifacts/adaptive_radio/report.json',seal)
    # Preserve the real report schema/negative contrasts; omit repeated hash maps.
    metrics={k:v for k,v in r.items() if k not in ('source_sha256','evidence_sha256')}
    return track('Adaptive reported-RF assay','sionna_simulation','sealed',proof=proof,
        scope='Static condition changes with reported/censored inputs; not learned temporal dynamics',
        units={'rf_error':'dB','condition_delta_error':'dB; distinct from absolute RF RMSE'},metrics=metrics,
        limits=['Preserve VAL-selected regression and every negative matched-objective contrast.',
            'RF error is not a packet/deadline/silence qualification; no default change.'])


def operator_profile(e):
    prefix='artifacts/native_bridge_ops/'
    source=e.load(prefix+'registration/source.json')
    for key in ('source_sha256','preflight_sha256'):
        for name,expected in source[key].items():e.digest(name,expected=expected)
    publication=e.load(prefix+'registration/git.json');r=e.load(prefix+'report.json')
    if (publication['source_receipt_sha256']!=e.sources[prefix+'registration/source.json']
            or publication['commit']!=publication['remote_observed_commit']
            or r['source_receipt_sha256']!=e.sources[prefix+'registration/source.json']
            or r['publication_sha256']!=e.sources[prefix+'registration/git.json']
            or not r['passes'] or not r['exact_operand_byte_parity_every_plan_and_timed_pair']):
        raise ValueError('Operator publication or parity evidence differs')
    review=e.load('artifacts/native_bridge_ops_review/review.json')
    if (not review['passes'] or review['report_sha256_before']!=e.sources[prefix+'report.json']
            or review['report_sha256_after']!=e.sources[prefix+'report.json']):
        raise ValueError('Independent operator review does not bind report bytes')
    return track('Band-independent geometry reuse','software_microbenchmark','verified_profile',
        scope='48 existing public plans; 144 timed pairs, same managed CPU environment',
        units={'call_time':'milliseconds','ratio':'reference median / reuse median'},metrics=r['summary'],
        proof={'report':prefix+'report.json','sha256':e.sources[prefix+'report.json'],
            'source_publication_commit':publication['commit'],'independent_review_passes':True,
            'exact_operand_byte_parity':True},
        limits=r['limitations']+['2.831× is operand-call performance, not model, queue, planner, radio or service speedup.',
            'The optimized operator was not substituted into the frozen native outcomes.'])


def context(e):
    r=e.load('artifacts/native_context_review/diagnosis.json')
    e.digest('artifacts/native_context_review/diagnose_context.py',expected=r['script_sha256'],source=True)
    for name,expected in r['input_sha256'].items():e.digest(name,expected=expected)
    if r['sealed_before']!=r['sealed_after'] or not r['sealed_after']['all_hashes_match']:
        raise ValueError('Passive context review did not preserve the sealed evidence')
    rows=[]
    for snapshot in r['snapshots']:
        target=next(x for x in snapshot['target_ap3'] if x['band_index']==1)
        rows.append({'decision':snapshot['decision'],'own_robot_tokens_all_links':snapshot['own_robot_selected_tokens'],
            'target_selected_observations':target['selected_observations'],
            'spatial_support_counts_m':target['spatial_support_counts_m'],
            'latest_own_target_observation':target['own_robot_observations'][-1],
            'forecast_ready_slot':snapshot['forecast_job_timing']['ready_slot']})
    return track('Robot 7 target-link context diagnosis','passive_completed_case_review','verified_passive',
        scope='Existing native seed 90473001 fast-handover case; two saved context snapshots only',
        units={'distance':'meters','time':'seconds or explicitly labeled 10 ms slots','rssi_or_bound':'dBm'},
        metrics={'snapshots':rows,'model_state_contract':r['model_state_contract']},
        proof={'path':'artifacts/native_context_review/diagnosis.json',
            'sha256':e.sources['artifacts/native_context_review/diagnosis.json'],'sealed_inputs_verified':True},
        limits=['Nearby AP3 / 5.2 GHz bounds were omitted while distant target-link reports were retained.',
            'Robot 7 still supplied other-link tokens; this is target-specific support loss.',
            'No causal context-only intervention, proof of eight-state insufficiency or representation-collapse diagnosis.'])


def suite(e):
    name='artifacts/research_review/whole_suite.json';r=e.load(name)
    log=e.text('artifacts/research_review/whole_suite.log',expected=r['log_sha256'])
    summaries=re.findall(r'(?m)^(\d+) passed, (\d+) warnings? in ([0-9.]+)s',log)
    if (r['exit_code']!=0 or r['unchanged'] is not True or r['source_before_sha256']!=r['source_after_sha256']
            or len(summaries)!=1):raise ValueError('Archived whole-suite proof is incomplete')
    for path,expected in r['source_after_sha256'].items():e.digest(path,expected=expected)
    passed,warnings,seconds=summaries[0]
    return {'status':'archived_source_bound_pass','passed':int(passed),'warnings':int(warnings),
        'pytest_seconds':float(seconds),'whole_process_seconds':r['duration_seconds'],
        'test_modules':len(r['test_files']),'source_files':len(r['source_after_sha256']),
        'scope':r['scope'],'includes_this_new_exporter':False,
        'proof':{'path':name,'sha256':e.sources[name],'log_sha256':r['log_sha256']}}


def original_runs(e):
    name='artifacts/radiojev_cloud/provenance.json';r=e.load(name)
    for filename,expected in r['result_sha256'].items():
        e.digest('artifacts/radiojev_cloud/'+filename,expected=expected,source=True)
    if r['run_complete'] is not True or r['pending_stages']:raise ValueError('Original RadioJEV completion provenance differs')
    return {'status':'completed_archived_chain','stages':list(r['completed_stages']),
        'training_seed':r['seed'],'source':name,'limits':['Original E1–E5 use one training chain; case intervals are not training-seed replication.',
            'E6 is excluded; the base Studio retains all original reports, including negative outcomes.']}


def vision(e):
    name='artifacts/radiojepa_vision/aligned-five/receipt.json';r=e.load(name)
    e.digest('artifacts/radiojepa_vision/aligned-five/features.npz',expected=r['features_sha256'],source=True)
    if r['completed'] is not True or r['actual_frozen_dinov3'] is not True:
        raise ValueError('Actual frozen-feature receipt required')
    return track('DINOv3 visual feature extraction','exploratory_feature_extraction','completed_exploratory_receipt',
        scope='Five selected video frames with publisher-tolerance CSI alignment; no trained RF–vision task',
        units={'alignment_uncertainty':'seconds; see each frame receipt'},
        metrics={'frames':len(r['frames']),'frozen':r['model']['frozen'],
            'parameter_count':r['model']['parameter_count'],
            'RF_accuracy_gain_established':r['RF_accuracy_gain_established'],
            'exact_simultaneity_proven':r['exact_simultaneity_proven'],
            'camera_geometry_calibrated':r['camera_geometry_calibrated'],
            'visual_overlay_label_leakage_audited':r['visual_overlay_label_leakage_audited']},
        proof={'path':name,'sha256':e.sources[name],'features_sha256':r['features_sha256'],
            'status':'feature bytes and receipt verified; not a sealed RF-accuracy experiment'},
        rights={'model':r['model']['repository'],'model_license':r['model']['license'],
            'reference':'docs/DINOV3_ACCESS.md','weights_redistributed_by_this_export':False},
        limits=['No RF gain, camera calibration, exact simultaneity, overlay-leakage clearance or robot-safety result.'])


def software_capabilities(e):
    paths=['src/wirelessjev/sensing_design/__init__.py','src/wirelessjev/sensing_design/operators.py',
        'src/wirelessjev/sensing_design/pilots.py','tests/test_sensing_design_operators.py']
    for path in paths:e.digest(path,source=True)
    return {'sensing_design':{'status':'implemented_software','scope':'Linear target-nullspace certificates and explicit-prior budgeted sensing heuristic',
        'RF_benchmark_or_service_gain_established':False,'source_paths':paths,
        'limits':['Mathematical fixtures are not measured on-air sensing, learned nonlinear identifiability or calibrated service risk.']}}


def exporter_source_hashes(root):
    root=Path(root)
    paths=list((root/'src/wirelessjev/research_export').glob('*.py'))
    paths+=list((root/'tests/research_export').glob('test_*.py'))
    paths+=list((root/'scripts').glob('export_research_studio.py'))
    return {str(path.relative_to(root)):file_sha(path) for path in sorted(paths)}


def exporter_tests(e):
    name='artifacts/research_studio/exporter_tests.json'
    if not e.path(name).exists():return {'status':'not_recorded','included_in_archived_2547':False}
    r=e.load(name);current=exporter_source_hashes(e.root)
    log=e.text('artifacts/research_studio/exporter_tests.log',expected=r['log_sha256'])
    count=re.findall(r'(?m)^(\d+) passed(?:, \d+ warnings?)? in ([0-9.]+)s',log)
    if (r['exit_code']!=0 or r['source_before_sha256']!=current or r['source_after_sha256']!=current or len(count)!=1):
        raise ValueError('Separate exporter fixture proof does not bind current source')
    for path,expected in current.items():e.digest(path,expected=expected,source=True)
    return {'status':'separate_source_bound_pass','passed':int(count[0][0]),'seconds':float(count[0][1]),
        'included_in_archived_2547':False,'proof':{'path':name,'sha256':e.sources[name],'log_sha256':r['log_sha256']}}


def r6_software(e):
    name='artifacts/r6_bridge/software_tests.json'
    if not e.path(name).exists():return None
    r=e.load(name);log=e.text(r['log_path'],expected=r['log_sha256'])
    count=re.findall(r'(?m)^(\d+) passed(?:, \d+ warnings?)? in ([0-9.]+)s',log)
    if (r['exit_code']!=0 or r['source_sha256_before']!=r['source_sha256_after']
            or r['upstream_sha256_before']!=r['upstream_sha256_after']
            or r['frozen_source_maps_before']!=r['frozen_source_maps_after']
            or len(count)!=1 or int(count[0][0])!=r['tests_passed']
            or r['actuation_performed'] or r['model_forwards'] or r['new_RF_cases']
            or r['new_packet_simulations'] or r['hardware_network_calls']):
        raise ValueError('R6 software receipt scope or source proof differs')
    for path,expected in r['source_sha256_after'].items():e.digest(path,expected=expected,source=True)
    return track('Read-only R6 planning and journal bridge','software_integration','verified_software_receipt',
        scope=r['scope'],units={'tests':'software checks; separate scope, do not sum with archived 2547'},
        metrics={'tests_passed':r['tests_passed'],'upstream_commit':r['upstream_commit'],
            'upstream_sha256_recorded':r['upstream_sha256_after'],'actuation_performed':False},
        proof={'path':name,'sha256':e.sources[name],'log_sha256':r['log_sha256'],
            'current_local_source_bytes_verified':True,'upstream_identity_scope':'recorded by original software-test receipt'},
        limits=['Read-only mission preview and journal snapshots do not execute robot commands or repair uncertain outcomes.',
            'RF OutcomeDistribution inputs still DEFER; no qualified service forecast is invented.',
            'These 77 checks have a separate scope; they are not added to the archived 2547 count.'])


def view_fields(key,value):
    """Two concise findings; complete metrics and limits remain in expandable evidence."""
    m=value.get('metrics',{})
    if key=='measured_power':
        c=m['factory_comparison'];headline=f"{c['best_reported_neural_rmse_db']:.3f} vs {c['constant_rmse_db']:.3f} dB · factory"
        findings=['Best reported neural factory mean loses to the constant reference; lower RMSE is better.',
            'One constructed factory split; no aggregate across recordings or isolated JEPA gain.']
    elif key=='spectral_field':
        c=next(x for x in m['contrasts'] if x['regime']=='in_band' and x['metric']=='rmse_db' and x['reference']=='radiomap_operator')
        control=next(x for x in m['contrasts'] if x['regime']=='in_band' and x['reference']=='matched_regression')
        headline=f"{c['candidate_mean']:.3f} vs {c['reference_mean']:.3f} dB · static RF"
        findings=['Selected architecture improves RF RMSE over the RadioMap operator.',
            f"Matched regression is {control['reference_mean']:.3f} dB; no supported consistency-objective gain."]
    elif key=='phone_transfer':
        a=next(x for x in m['summaries'] if x['model']=='source_knn' and x['support_size']==0)
        c=next(x for x in m['summaries'] if x['model']=='ap_shrinkage_knn' and x['support_size']==16)
        headline=f"{a['balanced_phone_mean_2d_error_m']:.3f} → {c['balanced_phone_mean_2d_error_m']:.3f} m · N=16"
        findings=['Classical calibration improves the mean; some phone-level tails worsen.',
            'Five phones at one site, not a neural RadioJEPA device-transfer result.']
    elif key=='complex_csi':
        c=m['summary']['dichasus-dc01'];headline=f"{c['joint']['seed_mean_packet_nmse']:.3f} vs {c['linear']['mean_packet_nmse']:.4f} NMSE · dc01"
        findings=['Joint completion loses decisively to linear interpolation; no useful joint/control gain.',
            'Digitally masked same-packet CSI, not identified physical paths or a world model.']
    elif key=='native_bridge':
        fast=m['arm_means_by_handover']['0.03'];slow=m['arm_means_by_handover']['4.0']
        headline=f"{100*fast['radiomap']['control_miss_fraction']:.3f} → {100*fast['candidate']['control_miss_fraction']:.3f}% misses · fast"
        findings=['Fast handover harms 4/4 cases; the candidate is selected regression, not a JEPA gain.',
            f"Slow mean {100*slow['radiomap']['control_miss_fraction']:.3f} → {100*slow['candidate']['control_miss_fraction']:.3f}%, but 2/4 cases harmed; worst gap 9.74 s."]
    elif key=='geometry_ops':
        headline=f"{m['ratio_of_median_call_times']:.3f}× operand calls"
        findings=[f"{m['reference_median_ms']:.1f} → {m['reuse_median_ms']:.1f} ms with exact operand bytes.",
            'Operator-only CPU profile; no measured full-model or service speedup.']
    elif key=='native_context':
        s=m['snapshots'][0]['spatial_support_counts_m']['5.0'];headline=f"{s['selected']}/{s['available']} nearby target reports retained"
        findings=['Decision 0 retains target reports 54.8/74.3 m away instead of available nearby evidence.',
            'Decision 1 omits its own 0.617 m target bound; no causal proof or collapse diagnosis.']
    elif key=='vision_features':
        headline=f"{m['frames']} frozen visual-feature frames"
        findings=['Actual pretrained DINOv3 extraction completed on selected frames.',
            'No RF gain, calibrated camera geometry or exact simultaneity established.']
    elif key=='r6_software':
        headline=f"{m['tests_passed']} software checks · read-only"
        findings=['Actual pinned mission preview and journal snapshot reader; no actuation.',
            'RF outcomes still DEFER; keep this test scope separate from archived 2547.']
    else:
        headline='Final sealed result pending' if value['status']=='pending_seal' else 'Sealed reported-RF assay'
        findings=value['limits'][:2]
    value['headline']=headline;value['findings']=findings


def build_bundle(root=ROOT,*,cache_root=None,captured_at=None):
    e=Evidence(root,cache_root);base=e.load('studio/data/bundle.json')
    if not isinstance(base,dict) or 'research_evidence' in base:raise ValueError('Use the original legacy Studio bundle')
    # Build every current scientific track only from a verified completed seal.
    tracks={'measured_power':measured_power(e),'spectral_field':spectral(e),
        'phone_transfer':phones(e),'complex_csi':csi(e),'adaptive_rf':adaptive(e),
        'native_bridge':native(e),'geometry_ops':operator_profile(e),'native_context':context(e),'vision_features':vision(e)}
    r6=r6_software(e)
    if r6 is not None:tracks['r6_software']=r6
    for key,value in tracks.items():view_fields(key,value)
    evidence={'schema':1,'captured_at':captured_at or datetime.now(timezone.utc).isoformat(),
        'default_controller':'RadioMap + WirelessJEV','remote_publication_performed':False,
        'remote_publication_note':'Local export only; the external Claude Studio/deck was not republished.',
        'evidence_class':'Mixed measured-data, simulated RF/service, software and passive-review evidence; scopes remain separate.',
        'tracks':tracks,'software':{'whole_suite':suite(e),'capabilities':software_capabilities(e),
            'exporter_tests':exporter_tests(e)},'original_e_runs':original_runs(e),
        'claim_boundaries':['Selected adaptive/native branch is regression; do not label its native outcomes a JEPA gain.',
            'No specialist SOTA, arbitrary-device/world/protocol generalization, 100-real-car benefit or production qualification.',
            'A negative completed result remains completed; pending sealed evidence has no exported TEST metrics.'],
        'legacy_snapshot':{'path':'studio/data/bundle.json','sha256':e.sources['studio/data/bundle.json'],
            'generated_at':base.get('generated_at'),'legacy_values_preserved':True},
        'source_sha256':dict(sorted(e.sources.items()))}
    bundle=deepcopy(base);bundle['research_evidence']=evidence
    e.finish()
    manifest={'schema':1,'input_sha256':dict(sorted(e.sources.items())),
        'verified_repository_files':sum(not k.startswith('external:') for k in e.verified),
        'verified_external_cache_files':len(e.external),'external_sha256':dict(sorted(e.external.items())),
        'scientific_actions':{'fits':0,'model_forwards':0,'simulations':0},'remote_publication_performed':False}
    return bundle,manifest


def review_html(bundle):
    payload=json.dumps(bundle,ensure_ascii=False,allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WirelessJEV · Research evidence</title><style>
:root{color-scheme:dark;font:16px system-ui,sans-serif;background:#081421;color:#dfebf8}body{max-width:1320px;margin:auto;padding:38px}h1{font-size:42px;letter-spacing:-1.5px;margin:8px 0}h2{font-size:23px}p{line-height:1.6;color:#adbed1}header{padding:30px;border-radius:22px;background:linear-gradient(130deg,#153750,#102236)}.tag{color:#7be2d2;text-transform:uppercase;font-size:12px;letter-spacing:2px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:20px;margin:26px 0}.card{border:1px solid #284157;border-radius:18px;background:#102131;padding:22px}.status{display:inline-block;border-radius:20px;background:#1b4050;color:#9ce9da;padding:5px 10px;font-size:12px}.pending{background:#584119;color:#ffe0a0}.metric{font-size:29px;font-weight:700;margin:16px 0;color:#f4f8fd}.limits{font-size:14px;color:#bccbdb;line-height:1.6;padding-left:20px}pre{font:12px/1.6 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;color:#b9cee1;background:#091623;padding:15px;border-radius:10px;max-height:420px;overflow:auto}summary{cursor:pointer;color:#8ccbe9;padding:8px 0}a{color:#82c9ed}footer{padding:25px 0;color:#98acc3}.banner{border-left:4px solid #ecad71;padding:10px 18px;margin:20px 0;background:#33291f}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #304758;padding:8px;text-align:left}th{color:#90b4cc}
</style><header><div class="tag">Preceptual AI · source-bound research</div><h1>What the evidence supports</h1><p id="intro"></p><div class="banner" id="boundary"></div><p id="software"></p></header><main><div id="cards" class="grid"></div><details><summary>Export provenance and original experiment status</summary><pre id="provenance"></pre></details></main><footer>Local review artifact. No external Claude Studio or slide deck was republished. <a href="bundle.json">Download augmented JSON</a></footer><script id="evidence" type="application/json">'''+payload+'''</script><script>
const B=JSON.parse(document.getElementById('evidence').textContent),R=B.research_evidence;
document.getElementById('intro').textContent=R.evidence_class+' Captured '+R.captured_at;
document.getElementById('boundary').textContent=R.default_controller+' remains the default. No new head or fleet controller is production qualified.';
const S=R.software.whole_suite,E=R.software.exporter_tests;document.getElementById('software').textContent=S.passed.toLocaleString()+' archived tests passed across '+S.test_modules+' modules; '+S.warnings+' expected warning. These tests do not establish RF or hardware performance and do not include this new exporter.'+(E?.status==='separate_source_bound_pass'?' Exporter: '+E.passed+' separate source-bound tests passed.':'');
for(const [key,T] of Object.entries(R.tracks)){const card=document.createElement('section');card.className='card';const status=document.createElement('span');status.className='status'+(T.status==='pending_seal'?' pending':'');status.textContent=T.status.replaceAll('_',' ');card.append(status);const title=document.createElement('h2');title.textContent=T.title;card.append(title);const metric=document.createElement('div');metric.className='metric';metric.textContent=T.headline;card.append(metric);const p=document.createElement('p');p.textContent=T.scope;card.append(p);const units=document.createElement('p');units.textContent='Units: '+Object.entries(T.units).map(([k,v])=>k+' — '+v).join('; ');card.append(units);const ul=document.createElement('ul');ul.className='limits';for(const x of T.findings.slice(0,2)){const li=document.createElement('li');li.textContent=x;ul.append(li)}card.append(ul);for(const [label,value] of [['All reported metrics',T.metrics],['Evidence, scope and rights', {scope:T.scope,units:T.units,limits:T.limits,proof:T.proof,rights:T.rights}]]){if(!value)continue;const d=document.createElement('details'),s=document.createElement('summary'),pre=document.createElement('pre');s.textContent=label;pre.textContent=JSON.stringify(value,null,2);d.append(s,pre);card.append(d)}document.getElementById('cards').append(card)}
document.getElementById('provenance').textContent=JSON.stringify({legacy_snapshot:R.legacy_snapshot,original_e_runs:R.original_e_runs,software:R.software,source_sha256:R.source_sha256,claim_boundaries:R.claim_boundaries},null,2);
</script></html>'''


def write_bundle(root=ROOT,*,cache_root=None,captured_at=None):
    bundle,manifest=build_bundle(root,cache_root=cache_root,captured_at=captured_at)
    out=Path(root)/'artifacts/research_studio';out.mkdir(parents=True,exist_ok=True)
    outputs={'bundle.json':encoded(bundle),'index.html':review_html(bundle).encode()}
    for name,payload in outputs.items():
        temporary=out/(name+'.part');temporary.write_bytes(payload);temporary.replace(out/name)
    manifest['outputs_sha256']={name:file_sha(out/name) for name in outputs}
    # Exporter-source identity is separate from prior scientific/test receipts.
    manifest['exporter_source_sha256']=exporter_source_hashes(root)
    (out/'manifest.json').write_bytes(encoded(manifest))
    return {'output':str(out),'tracks':{k:v['status'] for k,v in bundle['research_evidence']['tracks'].items()},
        'outputs_sha256':manifest['outputs_sha256'],'remote_publication_performed':False}
