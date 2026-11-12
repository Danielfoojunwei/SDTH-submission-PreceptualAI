"""Passive release checks for the final portable software package, not hardware."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REVIEW = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def verify():
    suite = read(REVIEW/"final/software_suite.json")
    assert suite["exit_code"] == 0 and suite["passed"] == 3903 and suite["source_unchanged"]
    assert len(suite["test_modules"]) == 160 and len(suite["source_sha256_after"]) == 591
    assert suite["source_sha256_before"] == suite["source_sha256_after"]
    for name, expected in suite["source_sha256_after"].items():
        assert sha((ROOT/name).read_bytes()) == expected, name
    assert sha((ROOT/suite["log_path"]).read_bytes()) == suite["log_sha256"]
    assert sha((REVIEW/"software_suite_release_runner.py").read_bytes()) == suite["runner_sha256"]
    archive = ROOT/"artifacts/smartcar_deployment/smartcar-wirelessJEV.zip"
    receipt = read(archive.with_suffix(".receipt.json"))
    portable = read(REVIEW/"final/portable_verification.json")
    assert portable["passes"] and portable["hardware_scan_or_actuation"] == portable["SSH_requests"] == 0
    assert portable["real_car_measurements"] == 0 and portable["SOTA_or_hardware_qualification"] is False
    assert portable["archive_sha256"] == receipt["archive_sha256"] == sha(archive.read_bytes())
    assert portable["archive_bytes"] == receipt["archive_bytes"] == archive.stat().st_size
    assert portable["research_training_dependencies_blocked_during_shadow"]
    assert len(portable["released_shadow_members_executed"]) == 3
    assert sha((REVIEW/"verify_portable_release.py").read_bytes()) == portable["verifier_sha256"]
    components = runpy.run_path(str(ROOT/"scripts/build_smartcar_bundle.py"))["COMPONENTS"]
    source_paths = {"src/"+name: "src/"+name for name in components}
    for name in ("README.md", "PI/wj_car_agent.py", "PC/run.py", "verify_bundle.py"):
        source_paths[name] = "deployment/smartcar/"+name
    source_paths["docs/SMARTCAR_DEPLOYMENT.md"] = "docs/SMARTCAR_DEPLOYMENT.md"
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        assert len(names) == len(set(names)) == 28
        prefix = "smartcar-wirelessJEV/"
        manifest = json.loads(zipped.read(prefix+"MANIFEST.json"))
        assert set(names) == {prefix+name for name in manifest["files_sha256"]} | {prefix+"MANIFEST.json"}
        assert manifest["source_commit"] == receipt["source_commit"]
        for name, expected in manifest["files_sha256"].items():
            assert sha(zipped.read(prefix+name)) == expected, name
        for name, path in source_paths.items():
            committed = subprocess.check_output(["git", "show", manifest["source_commit"]+":"+path], cwd=ROOT)
            assert committed == zipped.read(prefix+name) == (ROOT/path).read_bytes(), name
        models = json.loads(zipped.read(prefix+"models/manifest.json"))
        assert models["experimental_shadow_only"] and not models["hardware_transfer_qualified"]
        for member in models["members"]:
            expected = member["sha256"]
            assert sha(zipped.read(prefix+"models/"+member["path"])) == expected
            assert sha((ROOT/"artifacts/nested_radio/checkpoints"/member["path"]).read_bytes()) == expected
    prior = read(REVIEW/"prior_closure_verification.json")
    assert prior["passes"] and prior["scientific_closures"] == 10
    assert prior["unique_named_repository_files"] == 27295 and prior["scientific_calls"] == 0
    for name, expected in prior["manifest_sha256"].items():
        assert sha((ROOT/name).read_bytes()) == expected, name
    docs = ("README.md", "docs/HANDOFF.md", "docs/ONE_CAR_TEST.md", "docs/SMARTCAR_DEPLOYMENT.md",
            "deployment/smartcar/README.md", "artifacts/smartcar_deployment/README.md",
            "artifacts/smartcar_deployment_review/README.md")
    document_hashes, links = {}, 0
    for name in docs:
        path = ROOT/name
        document_hashes[name] = sha(path.read_bytes())
        for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)\n]+)\)", path.read_text()):
            if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target) or target.startswith("#"):
                continue
            resolved = (path.parent/target.split("#", 1)[0]).resolve()
            assert resolved.is_relative_to(ROOT) and resolved.exists(), (name, target)
            links += 1
    return {"schema": 1, "passes": True, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha(Path(__file__).read_bytes()), "archive_sha256": portable["archive_sha256"],
        "archive_bytes": archive.stat().st_size, "bundle_members": 28,
        "bundle_source_commit": manifest["source_commit"], "committed_source_members_checked": len(source_paths),
        "tracked_suite_passed": suite["passed"], "source_files_checked": 591, "test_modules": 160,
        "software_suite_sha256": sha((REVIEW/"final/software_suite.json").read_bytes()),
        "portable_proof_sha256": sha((REVIEW/"final/portable_verification.json").read_bytes()),
        "prior_closures_preserved": 10, "documents_sha256": document_hashes,
        "local_links_checked": links, "real_car_validation": False, "SOTA_established": False,
        "new_registered_scientific_experiments": 0, "hardware_actions": 0,
        "deferred_observation_frontier_excluded": True}


if __name__ == "__main__":
    value = verify()
    with (REVIEW/"final/release_verification.json").open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: value[key] for key in ("passes", "archive_sha256", "tracked_suite_passed", "local_links_checked")}))
