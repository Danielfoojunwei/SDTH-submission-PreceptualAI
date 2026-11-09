import numpy as np
import torch
from wirelessjev.packet_value.study import make_case,policy_for,verify_seeds
from wirelessjev.packet_value.sim import simulate
from wirelessjev.packet_value.features import action_queries


def test_real_collector_features_and_factual_suffix_use_the_same_physics():
    torch.set_num_threads(1);verify_seeds()
    w,g,controller,kwargs=make_case(299030,6,16,.03);checkpoints=[]
    factual=simulate(w,policy_for(.03),checkpoint_at={500},checkpoint_sink=checkpoints.append,detailed=True,**kwargs)
    cp=checkpoints[0];public=cp.observable();copied=cp.bundle['hooks'][-1]
    assert copied.history.context().shape==(6,6,32)
    assert action_queries(g,public,.03).shape==(6,16,24)
    replay=simulate(w,policy_for(.03),resume=cp,stop_slot=1500,scan_duration_s=.09,detailed=True)
    expected=next(row for row in factual['packet_detail']['cumulative'] if row['slot_end']==1500)
    assert replay['packet_detail']['misses']==expected['misses']
    available=np.flatnonzero((public['outage_remaining']==0)&(public['scan_remaining']==0))
    robot=int(available[0]);current=int(public['current'][robot])
    stay=simulate(w,policy_for(.03),resume=cp,stop_slot=1500,force_action=(robot,current),scan_duration_s=.09,detailed=True)
    frames=stay['packet_detail']['controller_frames']
    assert len(frames)==6 and all(500<=x['slot']<1500 for x in frames)
    assert np.stack([x['frame'][robot] for x in frames]).shape==(6,32)
