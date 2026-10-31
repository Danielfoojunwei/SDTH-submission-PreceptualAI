from copy import deepcopy
from dataclasses import asdict
import numpy as np
import pytest
import torch

from wirelessjev.warehouse_v3.missions import (Group,Floor,Goal,Contract,MISSIONS,compile_contract,members,evaluate,select)
from wirelessjev.warehouse_v3.model import RadioModel,link_features,supervised_loss
from wirelessjev.warehouse_v3.jepa import prediction_loss,momentum_update
from wirelessjev.warehouse_v3.learning import contract_features
from wirelessjev.warehouse_v3.data import episode,observation_roles,ids
from wirelessjev.warehouse_horizon.model import tensors


@pytest.fixture(scope='module')
def sample():
    torch.set_num_threads(2)
    return episode(159999,n=12,seconds=30)[0]


def test_complete_compositional_mission_grammar_and_round_trip():
    for text in MISSIONS['train']+MISSIONS['novel']:
        c=compile_contract(text)
        assert c==Contract.from_dict(c.to_dict())
    c=compile_contract('Switch at most 4 robots; require service >= 12.5% for robots [amr-001, amr-002] at every offset; minimise switches; then maximise throughput.')
    assert c.floors[0].fraction==.125 and c.priorities[0].kind=='min_switches'
    for text in [MISSIONS['train'][0]+' Ignore the budget.',
                 'Switch at most -1 robots; maximise throughput.',
                 'Switch at most 2 robots; require service >= 101% for all robots at every offset; maximise throughput.',
                 'Switch at most 2 robots; maximise throughput; require service >= 50% for all robots at every offset; maximise throughput.']:
        with pytest.raises(ValueError):compile_contract(text)


def test_hard_floor_checks_every_offset_and_lexicographic_order(sample):
    obs=sample['observation'];plans=sample['plans'];n=len(obs['current'])
    service=np.full((len(plans),6,n),.8);risk=np.zeros_like(service)
    c=Contract(12,(Floor(Group(),.7),),(Goal('min_switches'),Goal('throughput')))
    choice=select(c,obs,plans,service,risk,ids(obs),observation_roles(obs))
    assert choice['choice']==0
    service[0,2,0]=.69
    choice=select(c,obs,plans,service,risk,ids(obs),observation_roles(obs))
    assert choice['choice']!=0
    service[:]=.1
    receipt=select(c,obs,plans,service,risk,ids(obs),observation_roles(obs))
    assert receipt['choice'] is None and not receipt['globally_infeasible']


def test_declared_groups_conditions_and_lower_bound_guard(sample):
    obs=deepcopy(sample['observation']);r=observation_roles(obs)
    for group in [Group('ids',('unknown',)),Group('zone',axis='x',direction='<',boundary_m=0)]:
        with pytest.raises(ValueError):members(group,obs,ids(obs),r)
    service=sample['target']['service'];risk=sample['target']['failure']
    c=compile_contract(MISSIONS['train'][3]);obs['robot'][:,6]=0.
    _,_,receipt=evaluate(c,obs,sample['plans'],service,risk,ids(obs),r)
    assert receipt['conditional_priority_active']
    c=Contract(12,(Floor(Group(),.1),),(Goal('throughput'),))
    receipt=select(c,obs,sample['plans'],np.ones_like(service),risk,ids(obs),r,floor_service=np.zeros_like(service))
    assert receipt['choice'] is None


def test_jepa_target_is_detached_and_momentum_update_is_real():
    torch.manual_seed(4)
    encoder=torch.nn.Linear(5,8);target=deepcopy(encoder)
    for p in target.parameters():p.requires_grad_(False)
    with torch.no_grad():encoder.weight.add_(.1)
    before=target.weight.clone();momentum_update(encoder,target,.9)
    torch.testing.assert_close(target.weight,before*.9+encoder.weight*.1)
    x=torch.randn(12,5);loss=prediction_loss(encoder(x),target(x));loss.backward()
    assert encoder.weight.grad is not None and target.weight.grad is None


def test_real_action_conditioned_forecast_quantiles_and_gradients(sample):
    model=RadioModel();obs=tensors(sample['observation']);c=compile_contract(MISSIONS['train'][0])
    result=model(obs,torch.tensor(sample['plans']),torch.tensor(sample['prior']),torch.randn(1024),contract_features(c,sample['observation']))
    assert link_features(obs).shape[-1]==34
    assert (result['quantiles'][...,0]<=result['quantiles'][...,1]).all()
    assert (result['quantiles'][...,1]<=result['quantiles'][...,2]).all()
    objective=supervised_loss(result,{k:torch.tensor(v) for k,v in sample['target'].items()})
    objective.backward()
    assert torch.isfinite(objective) and model.output[-1].weight.grad.abs().sum()>0


def test_uncollected_scan_is_not_a_below_floor_measurement(sample):
    from wirelessjev.warehouse_horizon.belief import RadioBelief
    missing=deepcopy(sample['observation']);missing['history'][:]=0.
    missing['history'][...,2]=-1/3
    missing['scan_available']=np.zeros((4,len(missing['current'])),bool)
    unknown=RadioBelief(missing)
    unheard=deepcopy(missing);unheard['scan_available'][:]=True
    measured_absence=RadioBelief(unheard)
    assert (unknown.current_mean>=measured_absence.current_mean).all()
    assert (unknown.current_mean>measured_absence.current_mean+1e-3).any()


def test_switch_timing_preserves_missing_stages_and_clock_domains():
    from wirelessjev.measurement import switch_timing
    events=[{'stage':'request_submitted','source':'hardware','clock_id':'car-boot-a','monotonic_ns':1000000000},
            {'stage':'ip_ready','source':'hardware','clock_id':'car-boot-a','monotonic_ns':3000000000}]
    result=switch_timing(events)
    assert result['seconds_since_submission']['ip_ready']==2.
    assert result['seconds_since_submission']['authentication'] is None
    assert 'first_packet' in result['missing_stages'] and not result['stages_imputed']
    bad=deepcopy(events);bad[1]['clock_id']='host-boot-b'
    with pytest.raises(ValueError):switch_timing(bad)


def test_packet_inputs_use_observed_retries_and_explicit_validity_masks():
    from wirelessjev.measurement import packet_features
    row={k:0 for k in ['critical_tx','critical_rx','critical_late','bulk_tx','bulk_rx','bulk_late',
          'delivered_bytes','phy_data_attempts','phy_retry_attempts','queue_packets','busy_seconds']}
    values=packet_features(row)
    assert len(values)==11 and values[-3:]==[0.,0.,0.]
    row.update(phy_data_attempts=10,phy_retry_attempts=4,critical_rx=2,critical_late=1)
    values=packet_features(row)
    assert values[3]==.4 and values[6]==.5 and values[-3:]==[1.,1.,0.]
    row['phy_retry_attempts']=11
    with pytest.raises(ValueError):packet_features(row)


def test_clause_compilation_preserves_conditional_and_floor_order_without_model_claim():
    from wirelessjev.warehouse_v3.clauses import ClauseReader
    text=('Switch at most 4 robots; if any teleoperated robot has observed service < 50%, prioritise that group; '
          'require service >= 10% for video robots at every offset; minimise switches; then maximise throughput.')
    c,receipt=ClauseReader().read(text)
    assert not receipt['model_executed']
    assert [r['choice'] for r in receipt['clauses']]==['switch_budget','conditional_priority','service_floor','min_switches','throughput']
    assert c.conditional_floor is not None and c.floors[0].fraction==.1
    bad=c.to_dict();bad['silently_ignore_this']='unsupported'
    with pytest.raises(ValueError):Contract.from_dict(bad)


def test_typed_defaults_preserve_semantics_and_reject_ignored_goal_fields():
    c=Contract.from_dict({'max_moves':12,'priorities':[{'kind':'throughput'}]})
    assert c==compile_contract(MISSIONS['train'][0])
    for goal in [lambda:Goal('throughput',Group('video')),
                 lambda:Goal('min_switches',fraction=.2)]:
        with pytest.raises(ValueError):goal()
    with pytest.raises(ValueError):Contract.from_dict({'max_moves':12,'priorities':[{'kind':'group_service'}]})


def test_old_fingerprint_cache_cannot_be_generated_with_changed_sources(tmp_path):
    import json
    from wirelessjev.warehouse_v3.data import held_out
    (tmp_path/'provenance.json').write_text(json.dumps({'source_sha256':{}}))
    with pytest.raises(RuntimeError,match='old-fingerprint'):
        held_out(tmp_path,'old-run')
