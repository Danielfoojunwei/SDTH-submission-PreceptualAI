"""Controller recovery/failure checks; no model or simulated training is claimed."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def controller(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/train_radiojev.py"
    spec = importlib.util.spec_from_file_location("training_controller_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "prepare", lambda *args: None)
    monkeypatch.setattr(module, "STAGES", [("e1_no_llm", "e1", ["--arm", "no_llm"])])
    monkeypatch.setattr(sys, "argv", [str(path), "--run-dir", str(tmp_path)])
    return module, tmp_path


def test_controller_propagates_failure_and_preserves_existing_evidence(controller, monkeypatch):
    module, run = controller
    evidence = run / "original.json"
    evidence.write_text('{"preserve":true}')
    class FailedChild:
        pid = 100
        def __init__(self, *args, **kwargs): pass
        def wait(self): return 7
    monkeypatch.setattr(module.subprocess, "Popen", FailedChild)
    assert module.main() == 7
    status = json.loads((run / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["stages"]["e1_no_llm"]["returncode"] == 7
    assert evidence.read_text() == '{"preserve":true}'


def test_completed_stage_is_reused_and_old_network_failure_cleared(controller, monkeypatch):
    module, run = controller
    (run / "e1_no_llm.json").write_text('{"test":{"n":200}}')
    (run / "status.json").write_text(json.dumps({"status": "blocked", "reason": "old 403",
        "blocked_stage": "e1_radio_lora", "stages": {"e1_no_llm": {"status": "completed"}}}))
    def unexpected(*args, **kwargs):
        raise AssertionError("A completed stage must not be relaunched")
    monkeypatch.setattr(module.subprocess, "Popen", unexpected)
    assert module.main() == 0
    status = json.loads((run / "status.json").read_text())
    assert status["status"] == "completed"
    assert "reason" not in status and "blocked_stage" not in status
    assert (run / "e1_no_llm.json").read_text() == '{"test":{"n":200}}'


def test_missing_model_access_blocks_before_model_stage(controller, monkeypatch):
    module, run = controller
    monkeypatch.setattr(module, "STAGES", [("e1_radio_lora", "e1", ["--arm", "radio_lora"])])
    def denied(*args, **kwargs):
        raise OSError("Tunnel connection failed: 403 Forbidden")
    monkeypatch.setattr(module.urllib.request, "urlopen", denied)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("Model stage started without access"))
    assert module.main() == 2
    status = json.loads((run / "status.json").read_text())
    assert status["status"] == "blocked" and status["blocked_stage"] == "e1_radio_lora"


def test_missing_completed_output_is_not_treated_as_success(controller):
    module, run = controller
    (run / "status.json").write_text(json.dumps({"status": "running",
        "stages": {"e1_no_llm": {"status": "completed"}}}))
    with pytest.raises(AssertionError):
        module.main()
    assert json.loads((run / "status.json").read_text())["status"] == "failed"
