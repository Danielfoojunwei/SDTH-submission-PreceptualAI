"""Reuse verified observations, and generate disjoint transfer environments."""
from pathlib import Path
import json
import numpy as np
from ..warehouse_horizon.data import CONFIG as OLD_CONFIG,load_examples,save_examples,diverse_candidates
from ..warehouse_horizon.world import World,observe
from ..warehouse_horizon.belief import RadioBelief
from ..warehouse_horizon.planning import Contract,candidate_pool
from ..warehouse_v2.experiment import sha256

DOMAINS={'ordinary':(100,'train',1.),'layout':(100,'shift',1.),
         'fleet200':(200,'shift',1.),'traffic':(100,'train',3.)}


def observation_roles(obs):
    # Explicit synthetic application declarations; never inferred from labels.
    return {'teleoperated':(np.arange(len(obs['current']))%17==0).tolist()}


def ids(obs):
    return [f'amr-{i+1:03d}' for i in range(len(obs['current']))]


def verified_parent(parent):
    parent=Path(parent)
    status=json.loads((parent/'status.json').read_text())
    if status['state']!='completed':raise RuntimeError('Original campaign must be completed')
    for name,digest in status['reports'].items():
        if sha256(parent/name)!=digest:raise RuntimeError('Original evidence changed')
    fingerprint=json.loads((parent/'provenance.json').read_text())['data_fingerprint']
    return {split:[sample for seed in OLD_CONFIG[split]
        for sample in load_examples(parent/'cache'/f'{seed}.npz',fingerprint,seed)] for split in ['train','validation']}


def episode(seed, *, n=100,domain='train',bulk_scale=1.,environment_seed=None,seconds=300):
    world=World(seed,n=n,steps=int(seconds/5)+7,domain=domain,bulk_scale=bulk_scale,environment_seed=environment_seed)
    current=world.initial.copy();history=[];previous=np.zeros(n);since=np.full(n,30.);samples=[]
    rng=np.random.default_rng(seed+7000000)
    for tick in range(int(seconds/5)):
        obs=observe(world.frames[tick],current,tick,previous_service=previous,since_switch=since,history=history)
        action=current.copy()
        if tick>=3 and (tick-3)%6==0:
            belief=RadioBelief(obs);plans,_=candidate_pool(obs,belief,Contract('throughput',12,1.))
            plans=diverse_candidates(plans)
            samples.append({'observation':obs,'plans':plans,'prior':belief.forecast(plans),
                            'target':world.outcomes(tick,plans,current),'scene':seed,'tick':tick})
            action=plans[int(rng.integers(len(plans)))].copy()
        previous,_,_=world.interval(tick,action,current)
        since=np.where(action!=current,0,since)+5.;current=action
    return samples


def held_out(root,fingerprint):
    root=Path(root)
    needed=[root/'cache'/f'{150000+index*1000+offset}.npz' for index in range(len(DOMAINS)) for offset in range(3)]
    if any(not p.exists() for p in needed):
        provenance=json.loads((root/'provenance.json').read_text())
        repo=Path(__file__).resolve().parents[3]
        names=['warehouse_v3/data.py','warehouse_horizon/world.py','warehouse_horizon/belief.py',
               'warehouse_horizon/planning.py','warehouse_v2/data.py','envlearn.py','fastplan.py',
               'twin/warehouse.py','twin/radio.py','hardware.py']
        for name in names:
            relative='src/wirelessjev/'+name
            if sha256(repo/relative)!=provenance['source_sha256'].get(relative):
                raise RuntimeError('Refusing to generate old-fingerprint test data with changed sources; restore the executed capsule or use a new run')
    result={}
    for index,(name,(n,domain,scale)) in enumerate(DOMAINS.items()):
        result[name]=[]
        for offset in range(3):
            seed=150000+index*1000+offset
            path=Path(root)/'cache'/f'{seed}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():samples=load_examples(path,fingerprint,seed)
            else:
                samples=episode(seed,n=n,domain=domain,bulk_scale=scale)
                save_examples(path,samples,fingerprint)
            result[name].extend(samples)
    return result
