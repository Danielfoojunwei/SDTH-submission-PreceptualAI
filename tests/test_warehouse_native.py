"""Protocol boundary checks for packet-supervised decisions."""
from copy import deepcopy
import numpy as np
import pytest
from wirelessjev.warehouse_native.data import (DURATION, BUDGET, MapPrior, candidates,
    downlink, endpoint_target, features, public_observation, topology, utility)


@pytest.fixture(scope='module')
def public():
    return public_observation(190000)


def test_native_prior_uses_declared_map_without_twin_scans(public):
    assert 'history' not in public and 'previous_service' not in public
    prior=MapPrior(public)
    distance=np.sqrt(((public['robot'][0]-public['ap'][0,:2])*[120,80]) @ ((public['robot'][0]-public['ap'][0,:2])*[120,80])+4.7**2)
    loss=(46.7 if public['ap'][0,2] else 40)+33*np.log10(distance)
    assert downlink(public,public['robot'])[0,0]==pytest.approx(public['ap'][0,5]*30-loss)
    assert prior.ev.scene.obstruction is None
    assert not prior.ev.leak.any()
    assert np.allclose(prior.ev.snaps[0].dl,downlink(public,public['route'][0]))
    assert np.allclose(prior.ev.snaps[0].ul,prior.ev.snaps[0].dl-public['ap'][None,:,5]*30+15)


def test_identical_plan_features_and_hard_budget(public):
    prior=MapPrior(public);plans,names,prediction,_=candidates(public,prior)
    assert (plans[0]==public['current']).all()
    assert ((plans!=public['current'][None]).sum(1)<=BUDGET).all()
    x,g=features(public,plans,prediction)
    assert x.shape==(4,100,45) and g.shape==(4,100)
    assert np.isfinite(x).all()
    assert np.all((prediction>=0)&(prediction<=1))
    doubled=np.stack([plans[0],plans[0]])
    xx,gg=features(public,doubled,prior.predict(doubled))
    assert np.array_equal(xx[0],xx[1]) and np.array_equal(gg[0],gg[1])
    lines=topology(public,plans[1]).splitlines()
    assert lines[0]=='WIRELESSJEV_PACKET_V2 100 32 15.0'
    assert len(lines)==133
    assert sum(float(line.split()[3])>0 for line in lines[33:])<=BUDGET


def test_endpoint_demand_includes_muted_and_untransmitted_intervals():
    # One 240-byte on-time critical packet per second: half nominal demand.
    row={'robot':0,'delivered_bytes':240,'critical_rx':1,'critical_late':0}
    report={'duration_s':30,'robots':1,'per_robot':[{'offered_mbps':.00384}],
            'telemetry':[{'second':j,'robots':[deepcopy(row)]} for j in range(1,31)]}
    assert np.allclose(endpoint_target(report,[.00384]),[[.5,.5]])
    for frame in report['telemetry'][:4]:
        frame['robots'][0].update(delivered_bytes=0,critical_rx=0)
    result=endpoint_target(report,[.00384])
    assert np.allclose(result,[[26/60,26/60]])
    assert utility(result)==pytest.approx(26/60)
    report['telemetry'][4]['robots'][0]['critical_late']=1
    assert endpoint_target(report,[.00384])[0,1]==pytest.approx(25/60)
    with pytest.raises(ValueError):endpoint_target(report,[0.])
    report['duration_s']=8
    with pytest.raises(ValueError):endpoint_target(report,[.00384])


def test_native_message_model_equivariant_to_robot_order(public):
    import torch
    from wirelessjev.warehouse_native.model import NativeModel
    prior=MapPrior(public);plans,_,p,_=candidates(public,prior);x,g=features(public,plans,p)
    torch.manual_seed(4);model=NativeModel(graph=True).eval()
    # A nonzero trained-shaped head is needed to test context, not just the prior.
    with torch.no_grad():model.head.weight.normal_(0,.1)
    t=lambda a:torch.as_tensor(a,dtype=torch.float32)
    order=np.random.default_rng(3).permutation(100)
    with torch.no_grad():
        a=model(t(x),torch.tensor(plans),torch.tensor(g),t(p))
        b=model(t(x[:,order]),torch.tensor(plans[:,order]),torch.tensor(g[:,order]),t(p[:,order]))
    assert torch.allclose(a['fractions'][:,order],b['fractions'],atol=2e-6)
    assert torch.allclose(a['failure_logits'][:,order],b['failure_logits'],atol=2e-6)


def test_prediction_requires_no_future_packet_labels(public,tmp_path):
    import torch
    from wirelessjev.warehouse_native.model import NativeModel
    from wirelessjev.warehouse_native.learning import predict
    prior=MapPrior(public);plans,_,p,_=candidates(public,prior);x,g=features(public,plans,p)
    sample={'x':x,'plans':plans,'groups':g,'prior':p,'observation':public}
    model=NativeModel(graph=True).eval()
    saved={'mean':torch.zeros(45),'std':torch.ones(45),'temperature':1.}
    a,risk=predict(tmp_path,'graph-0',sample,loaded=(model,saved))
    assert a.shape==p.shape and risk.shape==p.shape[:2]
    assert np.isfinite(a).all() and np.isfinite(risk).all()
    sample['target']=np.full_like(p,np.nan)
    b,_=predict(tmp_path,'graph-0',sample,loaded=(model,saved))
    assert np.array_equal(a,b)


def test_positive_mean_does_not_override_a_tail_regression():
    from wirelessjev.warehouse_native.learning import gate
    records=[]
    for scene in [1,2,3]:
        records += [{'method':'numerical','scene':scene,'utility':.5,'worst_decile_service':.1,'critical_ontime_demand_fraction':.5},
                    {'method':'graph-0','scene':scene,'utility':.6,'worst_decile_service':.09 if scene==2 else .11,'critical_ontime_demand_fraction':.6}]
    assert not gate(records,'graph-0')['accepted']


def test_archiver_refuses_incomplete_campaign(tmp_path):
    import importlib.util,json
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/archive_warehouse_native.py'
    spec=importlib.util.spec_from_file_location('native_archive',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'status.json').write_text(json.dumps({'state':'running'}))
    with pytest.raises(RuntimeError,match='Only complete'):
        module.archive(tmp_path,tmp_path/'archive')


def test_training_saves_a_finite_reloadable_checkpoint(public,tmp_path):
    import torch
    from wirelessjev.warehouse_native.learning import train,load_model,predict
    prior=MapPrior(public);plans,names,p,_=candidates(public,prior);x,g=features(public,plans,p)
    # Constructed labels verify software checkpointing only; they are never
    # included in the scientific native datasets or results.
    base={'scene':1,'split':'train','unique':np.arange(4),'names':names,'observation':public,
          'x':x,'groups':g,'plans':plans,'prior':p.astype('float32'),'target':np.clip(p*.8,0,1).astype('float32')}
    validation=dict(base,scene=2,split='validation')
    (tmp_path/'protocol.json').write_text('{}\n')
    name=train([base,validation],tmp_path,seed=0,graph=True,epochs=2)
    model,saved=load_model(tmp_path,name)
    assert all(torch.isfinite(t).all() for t in model.parameters())
    a,risk=predict(tmp_path,name,{k:v for k,v in base.items() if k!='target'},loaded=(model,saved))
    assert np.isfinite(a).all() and np.isfinite(risk).all()
    assert (tmp_path/'train-graph-0.json').exists()


def test_resume_reuses_sealed_validation_and_rejects_tampering(tmp_path):
    import importlib.util,json
    from pathlib import Path
    from wirelessjev.warehouse_v2.experiment import sha256
    path=Path(__file__).resolve().parents[1]/'scripts/run_warehouse_native.py'
    spec=importlib.util.spec_from_file_location('native_supervisor',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    selection={'frozen_before_test':True,'test_used_for_model_selection':False,'checkpoints':{}}
    for filename,key in [('protocol.json','protocol_sha256'),('training-protocol.json','training_protocol_sha256'),
                         ('dataset-development.json','development_sha256'),('validation.json','validation_sha256'),
                         ('ridge.json','ridge_sha256')]:
        (tmp_path/filename).write_text(json.dumps({'timing_ms':1.25}))
        selection[key]=sha256(tmp_path/filename)
    (tmp_path/'selection.json').write_text(json.dumps(selection))
    before=(tmp_path/'validation.json').read_bytes()
    assert module.frozen_selection(tmp_path)==selection
    assert module.frozen_selection(tmp_path)==selection
    assert (tmp_path/'validation.json').read_bytes()==before
    (tmp_path/'validation.json').write_text(json.dumps({'timing_ms':1.26}))
    with pytest.raises(RuntimeError,match='validation.json'):
        module.frozen_selection(tmp_path)
