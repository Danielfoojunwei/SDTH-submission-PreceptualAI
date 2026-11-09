"""Collect registered physical response kernels with real CPU Sionna RT.

Static contexts in ONE warehouse, not physical robot measurements. True channel
couplings are labels, never legitimate unmeasured inference inputs. This stage
does not synthesize a favorable model score or silently replace missing paths.
"""
import argparse,hashlib,importlib.metadata,json,subprocess,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ASSETS=Path('/workspace/upstream/service-world-reference/coupled_raytracer/environments/aws_warehouse/sionna')
TX=np.array([[-1.,-6.,1.8],[1.,-6.,1.8]])
RX=np.array([[1.,-1.,1.],[-1.,-1.,1.]])
POSES=np.array([[x,float(y),1.5] for y in np.linspace(-5.,2.,12) for x in (-.7,.7)])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(root=ROOT):
    root=Path(root);out=root/'artifacts/continual_radio';data=out/'data'
    registration=out/'registration/protocol.json';receipt=out/'data_receipt.json'
    registered=json.loads(registration.read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    source_names=['scripts/collect_continual_radio.py','artifacts/continual_radio/registration/protocol.json']
    source_hash={}
    for name in source_names:
        current=sha(root/name)
        before=hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name],cwd=root)).hexdigest()
        if current!=before:raise RuntimeError('Commit exact collector and registration before collecting: '+name)
        source_hash[name]=current
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        assert previous['completed'] and previous['source_sha256']==source_hash
        assert previous['data_sha256']==sha(data/'kernels.npz')
        for path,expected in previous['path_files_sha256'].items():assert sha(root/path)==expected
        print('Verified completed real RT collection; no new traces',flush=True);return previous
    data.mkdir(parents=True,exist_ok=True)
    import mitsuba as mi
    mi.set_variant('llvm_ad_mono_polarized')
    import drjit as dr
    if hasattr(dr,'set_thread_count'):dr.set_thread_count(1)
    from sionna.rt import load_scene,PlanarArray,Transmitter,Receiver,SceneObject,PathSolver
    started=time.time();scene=load_scene(str(ASSETS/'warehouse.xml'));scene.frequency=5.8e9
    scene.tx_array=PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V')
    scene.rx_array=PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V')
    original_objects=len(scene.objects)
    for i,position in enumerate(TX):scene.add(Transmitter('tx_'+str(i),position=position))
    for i,position in enumerate(RX):scene.add(Receiver('rx_'+str(i),position=position))
    mesh=Path('/workspace/training/visual-radio/rt/explicit_metal_blocker.ply')
    upstream_receipt=json.loads((root/'artifacts/visual_radio/rt_probe/receipt.json').read_text())
    if sha(mesh)!=upstream_receipt['blocker']['mesh_sha256']:raise RuntimeError('Registered original blocker mesh changed')
    blocker=SceneObject(name='continual_metal_blocker',fname=str(mesh),radio_material=scene.radio_materials['itu_metal'],position=POSES[0])
    scene.add(blocker);solver=PathSolver();couplings=[];valid=[];coherent=[];paths_hash={};rows=[]
    for i,position in enumerate(POSES):
        blocker.position=position
        paths=solver(scene,max_depth=2,samples_per_src=2048,max_num_paths_per_src=4096,
            los=True,specular_reflection=True,diffuse_reflection=False,refraction=False,diffraction=False,seed=42)
        a,tau=paths.cir(normalize_delays=False,out_type='numpy');coeff=a[:,0,:,0,:,0]
        power=np.sum(np.abs(coeff)**2,axis=-1);counts=np.asarray(paths.valid).sum(-1).reshape(2,2)
        if not np.isfinite(a).all() or not np.isfinite(power).all():raise FloatingPointError('Actual RT returned nonfinite coefficients')
        path=data/f'context-{i:02d}-cir.npz';np.savez_compressed(path,a=a,tau=tau,path_counts=counts)
        paths_hash[str(path.relative_to(root))]=sha(path)
        couplings.append(power);valid.append((counts>0)&(power>0));coherent.append(np.abs(coeff.sum(-1))**2)
        rows.append({'context_id':i,'pose_m':position.tolist(),'path_counts':counts.tolist(),'power':power.tolist()})
        print(json.dumps({'actual_rt_context':i,'path_counts':counts.tolist()}),flush=True)
    context=np.concatenate([POSES/10,np.tile(TX.flatten()/10,(24,1)),np.tile(RX.flatten()/10,(24,1)),np.full((24,1),5.8/6)],axis=1)
    target=data/'kernels.npz'
    np.savez_compressed(target,coupling=np.asarray(couplings),coupling_valid=np.asarray(valid),
        coherent_coupling=np.asarray(coherent),context=context.astype('f4'),context_id=np.arange(24),pose=POSES,
        tx_positions=TX,rx_positions=RX,frequency_hz=np.array(5.8e9))
    record={'completed':True,'actual_sionna_rt':True,'simulation_only':True,'one_warehouse_only':True,
        'contexts':24,'registration_commit_at_collection':commit,'registration_sha256':sha(registration),
        'source_sha256':source_hash,'data_sha256':sha(target),'path_files_sha256':paths_hash,
        'all_couplings_have_found_paths':bool(np.asarray(valid).all()),'contexts_detail':rows,
        'original_scene_objects':original_objects,'elapsed_seconds':time.time()-started,
        'versions':{n:importlib.metadata.version(n) for n in ('sionna','sionna-rt','mitsuba','drjit')},
        'asset_sha256':{str(p.relative_to(ASSETS)):sha(p) for p in sorted(ASSETS.rglob('*')) if p.is_file()},
        'upstream_fixture_sha256':sha(root/'artifacts/visual_radio/rt_probe/receipt.json'),
        'public_context':'blockerXYZ/10 + configured TX XYZ/10 (6) + RX XYZ/10 (6) + carrierGHz/6; 16 values',
        'physical_quantity':'Incoherent sum of squared complex RT path coefficients, per RX and TX; not measured traffic or packet loss',
        'solver':{'max_depth':2,'samples_per_src':2048,'max_paths_per_src':4096,'seed':42,'los':True,'specular':True,'diffraction':False,'diffuse':False,'refraction':False,'convergence_established':False},
        'limits':['One fixed warehouse and configured material family. Context holdouts are not unseen warehouses.',
            'Static kernel contexts, not a measured robot trajectory. Later activity/device/noise episodes are explicitly constructed simulations.',
            'Unknown channel coupling remains a label/oracle; it cannot be supplied to a learner as a measured inference feature.',
            'No radio command, active hardware probe, learned-policy qualification or production-reliability claim.']}
    receipt.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    if not record['all_couplings_have_found_paths']:raise RuntimeError('Incomplete bounded RT coverage: preserve result and stop before learning')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();result=collect(args.root);print(json.dumps({'completed':result['completed'],'contexts':result['contexts'],'seconds':result['elapsed_seconds']}))
