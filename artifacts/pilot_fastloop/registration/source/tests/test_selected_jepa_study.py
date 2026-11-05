"""Preserve post-development selection and require every real replay field."""
import copy
import json
import pytest
from wirelessjev.selected_jepa import common, study
from wirelessjev.selected_jepa.rollout import REPLAY_FIELDS


def test_selection_records_complete_three_member_profile_and_comparison():
    saved = common.selection()
    assert saved['selected_family'] == common.protocol()['candidate_family'] == 'context_jepa'
    assert saved['source_development_already_observed'] is True
    assert saved['new_confirmation_accessed'] is False
    assert len(saved['candidate_checkpoint_sha256']) == len(saved['supervised_checkpoint_sha256']) == 3
    assert len(saved['reused_timing_profiles']) == 8
    scores = {row['family']: row['mean_loss_ratio_to_radiomap'] for row in saved['candidates']}
    assert scores['context_jepa'] < scores['semantic_jepa']
    assert common.protocol()['primary_upper_quantile'] == 1 - .05 / 6


def test_replay_comparison_checks_every_expected_payload_and_member():
    rows = []
    for key in study.replay_keys():
        row = {'case': list(key)}
        row.update({name: {'fixture': name} for name in REPLAY_FIELDS})
        rows.append(row)
    replays = copy.deepcopy(rows)
    assert len(study.compare_replays(rows, replays)) == 8
    assert all(row['exact'] for row in study.compare_replays(rows, replays))
    for field in REPLAY_FIELDS:
        changed = copy.deepcopy(replays)
        changed[0][field] = 'Different actual payload'
        assert study.compare_replays(rows, changed)[0]['exact'] is False
    with pytest.raises(RuntimeError, match='membership'):
        study.compare_replays(rows, replays[:-1])
    with pytest.raises(RuntimeError, match='membership'):
        study.compare_replays(rows, replays + replays[:1])


def test_fresh_site_guard_rejects_a_previously_registered_site(monkeypatch):
    p = common.protocol()
    p['confirmation_sites'] = [2104000, *p['confirmation_sites'][1:]]
    monkeypatch.setattr(common, 'protocol', lambda: p)
    with pytest.raises(RuntimeError, match='overlaps') as error:
        common.verify_fresh_sites()
    assert 'artifacts/semantic_jepa/protocol.json' in str(error.value)
