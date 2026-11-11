"""An analytic identifiability example, NOT measured warehouse performance."""
from dataclasses import asdict,replace
import hashlib,json
from pathlib import Path
from wirelessjev.visual_radio.physics import Channel,ReceiverState,TransmitterState,free_space_coupling,receiver_link_budget

ROOT=Path(__file__).resolve().parents[1]


def example():
    channel=Channel(5.2e9,20e6)
    receiver=ReceiverState('receiver',(0.,0.,1.),channel,1e-9)
    desired=TransmitterState('desired',(10.,0.,1.),channel,20.,1.)
    near=TransmitterState('near',(1.,0.,1.),channel,20.,1.)
    far=TransmitterState('far',(0.,10.,1.),channel,20.,0.)
    couplings=[free_space_coupling(receiver,t,assume_isotropic_far_field=True) for t in (desired,near,far)]
    a=receiver_link_budget(receiver,desired,[near,far],couplings,inventory_complete=True)
    b=receiver_link_budget(receiver,desired,[replace(near,activity_fraction=0.),replace(far,activity_fraction=1.)],couplings,inventory_complete=True)
    unknown=receiver_link_budget(receiver,desired,[near,far],couplings,inventory_complete=False)
    assert a.sinr_db<b.sinr_db and unknown.sinr_linear is None
    return {'analytic_example_only':True,'measured_data_used':False,'warehouse_or_model_score':False,
        'assumptions':'Declared isotropic far-field free space, flat equal channels, incoherent interference, known binary activity; no scene image or RF measurement is fabricated',
        'same_separately_pooled_factors':{'mean_activity':.5,'mean_interferer_power_dbm':20.,
            'mean_interferer_coupling':sum(c.gain_linear for c in couplings[1:])/2},
        'near_active':asdict(a),'far_active':asdict(b),'sinr_difference_db':b.sinr_db-a.sinr_db,
        'unknown_inventory_sinr':unknown.sinr_linear,
        'conclusion':'Separately pooling activity and coupling loses which transmitter is active. The receiver needs sum_j(P_j * G_jr * activity_j * overlap_jr). Visual appearance alone cannot reveal unseen RF activity.',
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'src/wirelessjev/visual_radio/physics.py',ROOT/'src/wirelessjev/service_world/physics.py']}}


if __name__=='__main__':
    result=example();destination=ROOT/'artifacts/visual_radio/abstraction_checks.json'
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'analytic_only':True,'sinr_difference_db':result['sinr_difference_db'],'unknown_inventory_sinr':None}))
