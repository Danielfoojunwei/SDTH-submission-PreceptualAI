"""Exercise resume/failure behaviour without starting any model runs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def runner(tmp_path):
    artifacts = tmp_path / "artifacts" / "radiojev"
    artifacts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "artifacts/radiojev/run_all.sh"
    script = artifacts / "run_all.sh"
    shutil.copy2(source, script)
    uv = tmp_path / "uv"
    uv.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$CALLS"\nexit "${FAKE_EXIT:-0}"\n')
    uv.chmod(0o755)
    environ = {**os.environ, "WJ_UV_BIN": str(uv), "WJ_PYTHON": sys.executable,
               "CALLS": str(tmp_path / "calls"), "WJ_QUIET_POLLS": "1",
               "WJ_QUIET_INTERVAL": "0"}

    def run(*args, **env):
        return subprocess.run(["bash", str(script), *args, "no-other-radiojev-job-42"],
                              cwd=tmp_path.parent, env={**environ, **env},
                              text=True, capture_output=True, timeout=10)

    return artifacts, tmp_path / "calls", run


def test_dry_run_excludes_completed_arm_and_smoke_without_mutation(runner):
    artifacts, calls, run = runner
    result = run("--dry-run")
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 7
    assert "--arm radio_lora" not in result.stdout and "smoke" not in result.stdout
    assert not calls.exists() and not (artifacts / "run_all.log").exists()


def test_missing_lora_checkpoint_blocks_before_any_run(runner):
    _, calls, run = runner
    result = run()
    assert result.returncode == 2
    assert "Restore" in result.stderr and "ckpt_radio_lora.pt" in result.stderr
    assert not calls.exists()


def test_failure_stops_chain_and_preserves_checkpoints(runner):
    artifacts, calls, run = runner
    checkpoint = artifacts / "ckpt_radio_lora.pt"
    checkpoint.write_bytes(b"preserve-existing-weights")
    result = run(FAKE_EXIT="7")
    assert result.returncode == 7
    assert calls.read_text().splitlines() == [
        "run --frozen python -m wirelessjev.radiojev.e1 --arm radio_frozen"]
    assert checkpoint.read_bytes() == b"preserve-existing-weights"
    assert "ALL DONE" not in (artifacts / "run_all.log").read_text()


@pytest.mark.parametrize("saved_frozen", [True, False])
def test_resume_skips_completed_report_and_never_deletes_its_checkpoint(runner, saved_frozen):
    artifacts, calls, run = runner
    for name in (["ckpt_radio_lora.pt", "ckpt_radio_frozen.pt"] if saved_frozen else ["ckpt_radio_lora.pt"]):
        (artifacts / name).write_bytes(b"weights")
    report = artifacts / "e1_radio_frozen.json"
    report.write_text(json.dumps({"arm": "radio_frozen", "test": {"n": 200}}))
    result = run(FAKE_EXIT="9", WJ_SMOKE="1")
    assert result.returncode == 9
    assert "SKIP completed" in result.stdout
    assert calls.read_text().splitlines() == [
        "run --frozen python -m wirelessjev.radiojev.e1 --arm intent"]
    assert report.exists()
    if saved_frozen:
        assert (artifacts / "ckpt_radio_frozen.pt").read_bytes() == b"weights"
    else:
        assert not (artifacts / "ckpt_radio_frozen.pt").exists()
