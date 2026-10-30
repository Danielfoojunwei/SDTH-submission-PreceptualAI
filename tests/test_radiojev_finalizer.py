"""Completion must require full evidence and preserve previously archived work."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("radiojev_finalizer", ROOT / "scripts/finalize_radiojev.py")
finalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalizer)


@pytest.fixture
def complete_run():
    reports = {name: json.loads((ROOT / f"artifacts/radiojev_cloud/{name}.json").read_text())
               for name in finalizer.STAGES if name != "e45"}
    # Synthetic fixture only; never exported as research evidence.
    arm = {"per_site": {site: 0.2 for site in finalizer.SITES},
           "mean": 0.2, "worst": 0.2, "log_verifies": True, "rounds": []}
    reports["e45"] = {"init_per_site": {site: 0.3 for site in finalizer.SITES},
        **{name: copy.deepcopy(arm) for name in ("local", "fedavg_all", "swarm_pers")},
        "e5": {"forgotten_site": "S2_heavy_scanners", "mia_auc_before": 0.6,
               "mia_auc_exact_retrain": 0.5, "mia_auc_gated_distill": 0.52,
               "mia_auc_never_trained": 0.5, "retained_regret_before": 0.2,
               "retained_regret_exact": 0.25, "retained_regret_distill": 0.23,
               "cost_s_exact": 100, "cost_s_distill": 20}}
    state = {"status": "completed", "smoke": False, "seed": 0,
             "stages": {name: {"status": "completed", "returncode": 0} for name in finalizer.STAGES}}
    return state, reports


def test_complete_reports_pass_and_summary_keeps_negative_findings(complete_run):
    state, reports = complete_run
    finalizer.validate_reports(state, reports)
    audit = json.loads((ROOT / "artifacts/radiojev_cloud/e3_label_audit.json").read_text())
    text = finalizer.render_results(reports, None, audit, complete=True)
    assert "All nine" in text and "0.1046" in text
    assert "one-class dataset" in text and "not certified data deletion" in text
    assert "E6 is excluded" in text and "0.231" in text


@pytest.mark.parametrize("fault", ["running", "smoke", "short_test", "nan", "bad_auc", "bad_mean", "bad_signature"])
def test_partial_or_invalid_evidence_is_rejected(complete_run, fault):
    state, reports = complete_run
    if fault == "running": state["status"] = "running"
    if fault == "smoke": state["smoke"] = True
    if fault == "short_test": reports["e2_intent"]["test"]["n"] = 3
    if fault == "nan": reports["e45"]["local"]["mean"] = float("nan")
    if fault == "bad_auc": reports["e45"]["e5"]["mia_auc_before"] = 1.2
    if fault == "bad_mean": reports["e45"]["local"]["mean"] = 0.1
    if fault == "bad_signature": reports["e45"]["swarm_pers"]["log_verifies"] = False
    with pytest.raises(ValueError): finalizer.validate_reports(state, reports)


def test_archive_refuses_to_replace_different_evidence(tmp_path):
    source, destination = tmp_path / "new.json", tmp_path / "archived.json"
    source.write_text('{"regret":0.2}')
    destination.write_text('{"regret":0.1}')
    with pytest.raises(ValueError, match="Refusing to replace"):
        finalizer.preserve_report(source, destination)
    assert destination.read_text() == '{"regret":0.1}'


def test_automatic_publish_refuses_uncommitted_user_changes(tmp_path):
    subprocess.run(["git", "init", "-q", "-b", "review", str(tmp_path)], check=True)
    user_file = tmp_path / "notes.txt"
    user_file.write_text("User work in progress")
    with pytest.raises(ValueError, match="Checkout has changes"):
        finalizer.require_clean_branch(tmp_path, "review")
    assert user_file.read_text() == "User work in progress"


@pytest.mark.parametrize("value,valid", [(0.25, True), (float("inf"), False)])
def test_checkpoint_validation_checks_actual_tensor_values(tmp_path, value, valid):
    import torch
    path = tmp_path / "checkpoint.pt"
    torch.save({"readout": torch.tensor([value])}, path)
    if valid:
        metadata = finalizer.checkpoint_metadata(path)
        assert metadata["finite_tensors"] == 1 and len(metadata["sha256"]) == 64
    else:
        with pytest.raises(ValueError, match="Non-finite checkpoint"):
            finalizer.checkpoint_metadata(path)


def test_complete_pipeline_builds_portable_evidence_only_in_isolated_fixture(tmp_path, monkeypatch, complete_run):
    """Run real exports/builds with synthetic fixture results, never the live archive."""
    import torch
    state, reports = complete_run
    root, run = tmp_path / "project", tmp_path / "run"
    run.mkdir()
    shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "studio", root / "studio", ignore=shutil.ignore_patterns("dist", "data"))
    (root / "docs").mkdir()
    shutil.copyfile(ROOT / "docs/PAPER_MAP.md", root / "docs/PAPER_MAP.md")
    monkeypatch.setenv("PYTHONPATH", str(root / "src"))
    for name, report in reports.items():
        (run / f"{name}.json").write_text(json.dumps(report))
    (run / "status.json").write_text(json.dumps(state))
    sources = {str(p.relative_to(root)): finalizer.sha(p) for p in (root / "src/wirelessjev/radiojev").glob("*.py")}
    (run / "manifest.json").write_text(json.dumps({"source_sha256": sources, "smoke": False, "seed": 0}))
    for name in finalizer.CHECKPOINTS:
        torch.save({"readout": torch.tensor([0.25])}, run / name)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-q", "--allow-empty", "-m", "Fixture"], check=True)
    result = finalizer.finalise(root, run, state)
    assert result["status"] == "completed" and result["published_branch"] is None
    out = root / "artifacts/radiojev_cloud"
    assert json.loads((out / "provenance.json").read_text())["run_complete"] is True
    assert (out / "e45.json").read_bytes() == (run / "e45.json").read_bytes()
    bundle = json.loads((root / "studio/data/bundle.json").read_text())
    assert bundle["training_run"]["status"] == "completed"
    assert len(bundle["training_run"]["results"]) == 9
    assert (root / "studio/dist/wirelessjev-studio.html").exists()
    assert "All nine" in (root / "docs/WAREHOUSE_RESULTS.md").read_text()
    archived = (out / "e45.json").read_bytes()
    torch.save({"readout": torch.tensor([0.5])}, run / finalizer.CHECKPOINTS[0])
    with pytest.raises(ValueError, match="Previously verified checkpoint changed"):
        finalizer.finalise(root, run, state)
    assert (out / "e45.json").read_bytes() == archived


def test_watcher_accepts_worker_exit_when_parent_records_completion(tmp_path, monkeypatch):
    from wirelessjev import studio_data
    state = {"status": "running", "pid": 100,
             "stages": {"e45": {"status": "running", "pid": 101}}}
    path = tmp_path / "status.json"
    path.write_text(json.dumps(state))
    def transition(pid, run):
        path.write_text(json.dumps({**state, "status": "completed"}))
        return False
    monkeypatch.setattr(studio_data, "_training_process_alive", transition)
    monkeypatch.setattr(finalizer, "finalise", lambda *args: {"status": "completed"})
    monkeypatch.setattr(sys, "argv", ["finalize_radiojev.py", "--run-dir", str(tmp_path), "--watch"])
    assert finalizer.main() == 0
    assert json.loads((tmp_path / "finalization_status.json").read_text())["status"] == "completed"


def test_watcher_reports_a_stopped_run_without_publishing(tmp_path, monkeypatch):
    from wirelessjev import studio_data
    (tmp_path / "status.json").write_text(json.dumps({"status": "running", "pid": 100,
        "stages": {"e45": {"status": "running", "pid": 101}}}))
    monkeypatch.setattr(studio_data, "_training_process_alive", lambda *args: False)
    monkeypatch.setattr(finalizer.time, "sleep", lambda *args: None)
    monkeypatch.setattr(finalizer, "finalise", lambda *args: pytest.fail("Stopped run must not publish"))
    monkeypatch.setattr(sys, "argv", ["finalize_radiojev.py", "--run-dir", str(tmp_path), "--watch"])
    with pytest.raises(ValueError, match="Training process stopped"):
        finalizer.main()
    assert json.loads((tmp_path / "finalization_status.json").read_text())["status"] == "failed"
