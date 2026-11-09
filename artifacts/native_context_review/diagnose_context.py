"""One passive pass over already completed native archives; never run a model.

Only stored context, public plans/geometry, delivered history and job metadata
are interpreted. No World, RF simulator, fitting or learned forward is imported.
"""
from pathlib import Path
import gzip
import hashlib
import json
import math

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
FROZEN=ROOT/'artifacts/native_bridge/verification/manifest.json'
FROZEN_SHA='4d9f95316d9e029365ed64d4b04e43128a49239c4115ebb8a6bb49f11c12aeb9'
CASE=ROOT/'artifacts/native_bridge/cases/dev-90473001-ho0.03'
ROBOT=7
CARRIERS=(2.4e9,5.2e9,6.5e9)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def closure():
    if sha(FROZEN)!=FROZEN_SHA:raise RuntimeError('Sealed manifest changed')
    m=read(FROZEN)
    for field,count in [('source_sha256',512),('evidence_sha256',56)]:
        if len(m[field])!=count:raise RuntimeError('Sealed membership changed')
        for path,expected in m[field].items():
            if sha(ROOT/path)!=expected:raise RuntimeError('Sealed bytes changed: '+path)
    return {'manifest_sha256':FROZEN_SHA,'source_files':512,'evidence_files':56,'all_hashes_match':True}


def stats(values):
    x=np.asarray(values,dtype=float)
    return None if not len(x) else {'minimum':float(x.min()),'median':float(np.median(x)),
        'mean':float(x.mean()),'maximum':float(x.max())}


def main():
    if (OUT/'diagnosis.json').exists():raise RuntimeError('Passive diagnosis already exists; do not silently repeat')
    before=closure()
    receipt=read(CASE/'receipt.json')
    raw=json.loads(gzip.decompress((CASE/'native_result.json.gz').read_bytes()))
    arm=raw['arms']['candidate']
    protocol=read(ROOT/'artifacts/native_bridge/protocol.json')
    source_protocol=read(ROOT/'artifacts/adaptive_radio/protocol.json')
    registry=read(ROOT/'artifacts/adaptive_radio/geometry_registry.json')
    with np.load(ROOT/'artifacts/native_bridge/worlds/90473001/world.npz',allow_pickle=False) as z:
        sites=z['sites'].copy();tx=z['tx_dl'].copy();noise=z['noise_db'].copy()
    # Individually requested members only; no hidden world RF arrays are opened.
    with np.load(CASE/'candidate_inputs_forecasts.npz',allow_pickle=False) as z:
        history={k:z['last_history_'+k].copy() for k in ('positions','value','heard','bound','scanned')}
        jobs={i:{k:z[f'job{i}_{k}'].copy() for k in ('context','selected_rows','plan','mean_dbm','std_db',
                'belief_mean','belief_covariance','known','scheduled')} for i in (0,1)}
    events={}
    for event in arm['telemetry']['events']:
        events[(60+event['delivery_slot']//100,event['robot'])]=event
    for (t,r),event in events.items():
        if t>=len(history['value']):continue
        payload=history['value'][t,r].tobytes()+history['heard'][t,r].tobytes()
        if hashlib.sha256(payload).hexdigest()!=event['payload_sha256']:
            raise RuntimeError('Delivered payload does not match retained bin')

    def observation(t,r,s,b,query_xy,decision,selected):
        link=s*3+b;heard=bool(history['heard'][t,r,link]);xy=history['positions'][t,r]
        result={'history_bin':int(t),'robot':int(r),'ap':int(s),'band_index':int(b),
            'carrier_hz':CARRIERS[b],'xy_m':xy.tolist(),'distance_to_query_m':float(np.linalg.norm(xy-query_xy)),
            'heard':heard,'rssi_dbm':float(history['value'][t,r,link]) if heard else None,
            'upper_bound_dbm':None if heard else float(history['bound'][t,r,link]),
            'bin_age_at_issue_seconds':float(decision-(t-60)),
            'retained_as_token':(int(t),int(r),int(s),int(b)) in selected}
        event=events.get((int(t),int(r))) if t>=60 else None
        if event:
            result.update({'measurement_time_s':event['measurement_slot']/100,
                'delivery_time_s':event['delivery_slot']/100,
                'exact_measurement_age_at_issue_s':decision-event['measurement_slot']/100,
                'first_usable_decision':event['first_usable_decision']})
        return result

    snapshots=[]
    for decision,job in jobs.items():
        length=receipt['candidate_jobs'][decision]['history_length']
        if length!=60+decision:raise RuntimeError('Unexpected stored history cutoff')
        selected={tuple(int(v) for v in row) for row in job['selected_rows']}
        query=job['plan'][ROBOT,0]
        rows=[]
        for ordinal,(t,r,s,b) in enumerate(job['selected_rows']):
            if t>=length or not history['scanned'][t,r]:raise RuntimeError('Selected report not available')
            item=observation(t,r,s,b,query,decision,selected)
            expected_heard=item['heard']
            if (bool(job['context'][ordinal,2]>.5)!=expected_heard
                    or bool(job['context'][ordinal,3]>.5)==expected_heard):
                raise RuntimeError('Stored context status changed')
            np.testing.assert_allclose(job['context'][ordinal,4:6]*[120.,80.],item['xy_m'],atol=1.e-5,rtol=0)
            rows.append(item)
        if len(rows)!=64 or not job['known'].all() or not job['scheduled'].all():
            raise RuntimeError('Unexpected candidate context cardinality/status')
        groups=[]
        for s in range(len(sites)):
            for b in (0,1):
                group=[row for row in rows if row['ap']==s and row['band_index']==b]
                groups.append({'ap':s,'band_index':b,'count':len(group),'robots':sorted({x['robot'] for x in group}),
                    'heard':sum(x['heard'] for x in group),'censored':sum(not x['heard'] for x in group),
                    'distance_m':stats([x['distance_to_query_m'] for x in group]),
                    'bin_age_s':stats([x['bin_age_at_issue_seconds'] for x in group])})
        times,robots=np.nonzero(history['scanned'][:length])
        window_start=max(0,length-60)
        map_mask=history['scanned'][window_start:length]
        scan_ordinals=np.cumsum(map_mask,axis=0)
        # This records which existing rows are eligible under the frozen Map
        # fitter's stated censor-thinning rule. It does not fit/replay a Map.
        targets=[]
        for b in (0,1,2):
            available=[observation(t,r,3,b,query,decision,selected) for t,r in zip(times,robots)]
            own=[x for x in available if x['robot']==ROBOT]
            near={str(radius):{'available':sum(x['distance_to_query_m']<=radius for x in available),
                'selected':sum(x['distance_to_query_m']<=radius and x['retained_as_token'] for x in available)}
                for radius in (1.,5.,10.,20.)}
            link=3*3+b
            heard=history['heard'][window_start:length,:,link]
            cens=map_mask&~heard&((scan_ordinals-1)%5==0)
            mt,mr=np.nonzero(heard|cens)
            map_eligible=[observation(t+window_start,r,3,b,query,decision,selected) for t,r in zip(mt,mr)]
            selected_group=[x for x in rows if x['ap']==3 and x['band_index']==b]
            targets.append({'band_index':b,'carrier_hz':CARRIERS[b],
                'selected_count':len(selected_group),'selected_observations':selected_group,
                'available_prefix_count':len(available),'available_heard':sum(x['heard'] for x in available),
                'available_censored':sum(not x['heard'] for x in available),
                'available_nearest_distance_m':min(x['distance_to_query_m'] for x in available),
                'selected_nearest_distance_m':min((x['distance_to_query_m'] for x in selected_group),default=None),
                'spatial_support_counts_m':near,'own_robot_observations':own,
                'nearest_ten_available':sorted(available,key=lambda x:(x['distance_to_query_m'],-x['history_bin'],x['robot']))[:10],
                'map_window_scan_rows':int(map_mask.sum()),'map_target_heard_rows':int(heard.sum()),
                'map_target_censored_rows_after_actual_thinning':int(cens.sum()),
                'map_eligible_nearest_distance_m':min(x['distance_to_query_m'] for x in map_eligible),
                'map_eligible_own_observations':[x for x in map_eligible if x['robot']==ROBOT]})
        # At1.5s the initiating action consumed decision0's existing forecast.
        # This interpolates only the public planned XY, not hidden future RF.
        interpolated=(job['plan'][ROBOT,1]+job['plan'][ROBOT,2])/2 if decision==0 else None
        if interpolated is not None:
            consumption_distances=[float(np.linalg.norm(history['positions'][x['history_bin'],x['robot']]-interpolated))
                for x in rows if x['ap']==3 and x['band_index']==1]
        else:consumption_distances=None
        forecast=[]
        for s in (4,3):
            forecast.append({'ap':s,'rss_5p2_dbm':job['mean_dbm'][ROBOT,:,s,1].tolist(),
                'std_5p2_db':job['std_db'][ROBOT,:,s,1].tolist(),
                'native_uplink_snr_5p2_db':(job['mean_dbm'][ROBOT,:,s,1]-tx[1]+15-noise[1]).tolist()})
        snapshots.append({'decision':decision,'issue_time_s':float(decision),'history_cutoff_exclusive':length,
            'robot7_query_xy_m':query.tolist(),'selected_tokens':len(rows),
            'unique_selected_robots':sorted({x['robot'] for x in rows}),
            'own_robot_selected_tokens':sum(x['robot']==ROBOT for x in rows),
            'all_selected_rows':rows,'groups_ap_source_band':groups,'target_ap3':targets,
            'source_band_counts':{str(b):sum(x['band_index']==b for x in rows) for b in (0,1,2)},
            'forecast_job_timing':arm['forecast_jobs'][decision],
            'forecast_horizons_s':protocol['forecast_horizons_seconds'],'forecasts_robot7':forecast,
            'stored_global_coefficient_mean':job['belief_mean'].tolist(),
            'stored_global_coefficient_covariance_shape':list(job['belief_covariance'].shape),
            'stored_global_coefficient_covariance_diagonal':np.diag(job['belief_covariance']).tolist(),
            'first_consumption_public_plan_xy_at1p5s':None if interpolated is None else interpolated.tolist(),
            'target_selected_distances_to_consumption_public_plan_m':consumption_distances,
            'age_note':'Warm observations retain1s bins, not packet timestamps; postwarm exact ages use matching archived delivered events.'})
    geometries=[x['public_geometry'] for x in registry['stages']['training']]
    source_domain={'training_layouts':len(geometries),
        'width_m_min_max':[min(x['width_m'] for x in geometries),max(x['width_m'] for x in geometries)],
        'depth_m_min_max':[min(x['depth_m'] for x in geometries),max(x['depth_m'] for x in geometries)],
        'AP_counts':sorted({len(x['ap_positions_m']) for x in geometries}),
        'known_tx_dbm_values':sorted({x['known_tx_power_dbm'] for x in registry['stages']['training']}),
        'data_contract':source_protocol['data'],'native_geometry':protocol['geometry'],
        'native_APs':len(sites),'native_tx_dbm':tx.tolist(),'native_noise_dbm':noise.tolist(),
        'native_carriers_hz':list(CARRIERS),'calibration':protocol['calibration'],
        'native6p5_scope':protocol['six_point_five_GHz']}
    inputs=[CASE/'candidate_inputs_forecasts.npz',CASE/'native_result.json.gz',CASE/'receipt.json',
        ROOT/'artifacts/native_bridge/worlds/90473001/world.npz',ROOT/'artifacts/native_bridge/protocol.json',
        ROOT/'artifacts/native_bridge_review/outlier_diagnosis.json',ROOT/'artifacts/adaptive_radio/protocol.json',
        ROOT/'artifacts/adaptive_radio/geometry_registry.json',ROOT/'src/wirelessjev/native_bridge/predictor.py',
        ROOT/'src/wirelessjev/adaptive_radio/model.py',ROOT/'src/wirelessjev/frontier/fast.py',
        ROOT/'src/wirelessjev/closed_loop/telemetry.py',ROOT/'src/wirelessjev/closed_loop/runtime.py']
    after=closure()
    if after!=before:raise RuntimeError('Sealed bytes changed during passive review')
    output={'schema':1,'case_id':receipt['case']['case_id'],'robot_zero_based':ROBOT,
        'scope':'one passive descriptive pass over completed native RAW; no fits, model forwards, context reselection or counterfactual episodes',
        'script_sha256':sha(Path(__file__)),'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
        'sealed_before':before,'sealed_after':after,'snapshots':snapshots,'source_domain':source_domain,
        'model_state_contract':{'stored_mean_dimension':8,'mean_history_path':'All report-specific mean effects go through8globalcoefficients; local_db(query) is a shared query-only fixed learned correction.',
            'covariance_role':'8x8approximateglobalcoefficientcovariance affects forecaststd, not a learned local RF memory.',
            'not_identified':['physicalscatterers','materials','semanticcollapse','causal8dimensionalinsufficiency']},
        'map_contrast':'Cold Map fits per-AP/per-band local random-Fourier features using full60-bin native history; warm calls reuse that fit for20s, so available postwarm rows must not be called newly assimilated Map data.',
        'scientific_actions':{'fits':0,'model_forwards':0,'simulations':0,'counterfactuals':0}}
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'diagnosis.json').open('x') as stream:json.dump(output,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'passes':True,'snapshots':len(snapshots),'sealed':after,'output_sha256':sha(OUT/'diagnosis.json')}))


if __name__=='__main__':main()
