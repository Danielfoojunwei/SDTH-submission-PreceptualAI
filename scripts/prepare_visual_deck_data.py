"""Prepare real evidence and layout data for the cinematic 3D deck."""
from pathlib import Path
import hashlib,json
import numpy as np
from wirelessjev.twin.warehouse import make_warehouse,AP_SITES,RACK_ROWS,RACK_SEGMENTS
from wirelessjev.native.world import build_world
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'docs/slides/concise/assets';out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/'artifacts/native/wifi_vs_5g.json').read_text())['rows']
    values={}
    for condition in ['lenient','main','stress']:
        values[condition]={}
        for arm in ['vertical_client','nr_30ms']:
            rs=[r for r in rows if r['condition']==condition and r['arm']==arm]
            values[condition][arm]={'misses':sum(r['control_misses'] for r in rs),'commands':sum(r['control_sent'] for r in rs),
                'ppm':sum(r['control_misses'] for r in rs)/sum(r['control_sent'] for r in rs)*1e6,
                'availability_1s':float(np.mean([r['availability_1s'] for r in rs]))}
    world=build_world(530,duration_s=5.,n_robots=100,survey_s=60)
    x,y=np.meshgrid(np.linspace(0,120,49),np.linspace(0,80,33))
    mean,std=world.rmap.predict(x,y)
    calibration=json.loads((ROOT/'artifacts/native/learned_campaign.json').read_text())['calibration']
    c=[r for r in calibration if r['condition']=='survey60']
    scene=make_warehouse(530,n_robots=100)
    result={'scene':530,'conceptual_model':True,'field_measurement':False,
        'sites':[list(p) for p in AP_SITES],'rack_rows':list(RACK_ROWS),'rack_segments':[list(p) for p in RACK_SEGMENTS],
        'robots':[[c.x,c.y] for c in scene.cars],'map':{'x':x.tolist(),'y':y.tolist(),'mean_dbm':mean[...,5,1].tolist(),'std_db':std[...,5,1].tolist(),
            'meaning':'Height and colour encode predicted RSSI for site 5, 5 GHz; not a physical RF surface','survey_seconds':60},
        'calibration':{'rmse_db':float(np.mean([r['rmse_db'] for r in c])),'coverage_90':float(np.mean([r['coverage_90'] for r in c]))},
        'results':values,'ratio':values['main']['nr_30ms']['ppm']/values['main']['vertical_client']['ppm'],
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['artifacts/native/wifi_vs_5g.json','artifacts/native/learned_campaign.json','src/wirelessjev/native/world.py','src/wirelessjev/native/radiomap.py']}}
    (out/'scene_data.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Prepared 3D layout and real map grid; main-condition ratio',result['ratio'],flush=True)


if __name__=='__main__':main()
