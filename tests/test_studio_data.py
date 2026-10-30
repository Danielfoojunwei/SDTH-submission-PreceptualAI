from wirelessjev import studio_data
import json
import pytest


def test_export_prefers_corrected_e7_and_keeps_original_evidence(monkeypatch):
    original = {"summary": {"n_envs": 15}}
    corrected = {"method": "measured-top8-censored-gaussian-v2"}
    lora = {"arm": "radio_lora", "test": {"n": 200}}
    calibration = {"map_only": True, "seeds": list(range(15, 30))}
    reports = {"artifacts/envlearn/report.json": original,
               "artifacts/envlearn_corrected/report.json": corrected,
               "artifacts/radiojev/e1_radio_lora.json": lora,
               "artifacts/envlearn_calibration/report.json": calibration}
    monkeypatch.setattr(studio_data, "_load", reports.get)
    bundle = studio_data.build()
    assert bundle["envlearn"] is corrected
    assert bundle["envlearn_legacy"] is original
    assert bundle["envlearn_calibration"] is calibration
    assert bundle["radiojev"]["e1_radio_lora"] is lora
    assert bundle["radiojev"]["e1_radio_frozen"] is None


def test_export_falls_back_to_historical_e7_without_inventing_results(monkeypatch):
    original = {"summary": {"n_envs": 15}}
    monkeypatch.setattr(studio_data, "_load", {"artifacts/envlearn/report.json": original}.get)
    bundle = studio_data.build()
    assert bundle["envlearn"] is original
    assert all(value is None for value in bundle["radiojev"].values())


def test_archived_cloud_snapshot_survives_export_without_the_training_machine(monkeypatch):
    snapshot = {"status": "completed", "snapshot": True, "stages": {},
                "results": {"e45": {"local": {"mean": 0.2}}}}
    monkeypatch.setattr(studio_data, "_load", {"artifacts/radiojev_cloud/snapshot.json": snapshot}.get)
    bundle = studio_data.build()
    assert bundle["training_run"] is snapshot
    assert bundle["radiojev"]["e45"] is None


def test_training_snapshot_keeps_partial_validation_out_of_test_results(tmp_path, monkeypatch):
    state = {"status": "running", "pid": 10, "smoke": False, "seed": 0, "stages": {
        "e1_no_llm": {"status": "completed"}, "e1_radio_lora": {"status": "running", "pid": 11}}}
    (tmp_path / "status.json").write_text(json.dumps(state))
    result = {"arm": "no_llm", "test": {"n": 200, "regret": 0.001}}
    (tmp_path / "e1_no_llm.json").write_text(json.dumps(result))
    (tmp_path / "e1_radio_lora.json").write_text(json.dumps({"test": {"n": 200, "regret": 999}}))
    (tmp_path / "e1_radio_lora.log").write_text(
        "[01:00:00] radio_lora: epoch 1 train KL 0.4 val regret 0.23\n"
        "[02:00:00] radio_lora: epoch 2 train KL 0.3 val regret 0.13\n")
    monkeypatch.setattr(studio_data, "_training_process_alive", lambda pid, path: True)
    snapshot = studio_data.training_snapshot(tmp_path)
    assert snapshot["status"] == "running"
    assert snapshot["results"] == {"e1_no_llm": result}
    assert snapshot["stages"]["e1_radio_lora"]["history"][-1]["val_regret"] == 0.13
    assert snapshot["stages"]["e3"]["status"] == "pending"


def test_training_snapshot_marks_dead_workers_interrupted_and_ignores_old_attempts(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(json.dumps({"status": "running", "smoke": False,
        "stages": {"e1_radio_lora": {"status": "running", "pid": 12}}}))
    (tmp_path / "e1_radio_lora.log").write_text(
        "epoch 1 train KL 0.2 val regret 0.05\nepoch 2 train KL 0.1 val regret 0.02\n"
        "epoch 1 train KL 0.8 val regret 0.5\n")
    monkeypatch.setattr(studio_data, "_training_process_alive", lambda pid, path: False)
    snapshot = studio_data.training_snapshot(tmp_path)
    assert snapshot["status"] == "interrupted"
    assert snapshot["stages"]["e1_radio_lora"] == {"status": "interrupted", "history": [
        {"epoch": 1, "train_kl": 0.8, "val_regret": 0.5}]}


def test_training_snapshot_rejects_smoke_or_missing_completed_results(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"smoke": True}))
    with pytest.raises(ValueError, match="full training"):
        studio_data.training_snapshot(tmp_path)
    (tmp_path / "status.json").write_text(json.dumps({"status": "completed", "smoke": False,
        "stages": {"e1_no_llm": {"status": "completed"}}}))
    with pytest.raises(FileNotFoundError):
        studio_data.training_snapshot(tmp_path)
