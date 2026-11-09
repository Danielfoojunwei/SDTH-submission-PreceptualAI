"""Temporary-repository source/publication fixtures, not scientific evidence."""
from pathlib import Path
import subprocess

import pytest

from wirelessjev.spatial_control import study


def git(root,*args):
    return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()


@pytest.fixture
def sandbox(tmp_path,monkeypatch):
    root=tmp_path/"work";root.mkdir()
    git(root,"init","-q","-b","fixture-source")
    monkeypatch.setattr(study,"ROOT",root)
    monkeypatch.setattr(study,"OUT",root/"artifacts/spatial_control")
    files={
        "src/visible.py":"# tracked software fixture\n",
        "tests/test_visible.py":"def test_fixture(): assert True\n",
        "src/wirelessjev/spatial_control/new.py":"# untracked new namespace must bind\n",
        "tests/spatial_control/test_new.py":"# new nested test source\n",
        "scripts/run_spatial_control.py":"# fixture entry point\n",
        "docs/spatial_control/PLAN.md":"Software fixture plan\n",
        "pyproject.toml":"[project]\nname='fixture'\nversion='0.0.0'\n",
        "uv.lock":"# fixture lock identity\n",
        "src/wirelessjev/observation_frontier/deferred.py":"# deferred untracked user work\n",
    }
    for name,data in files.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(data)
    git(root,"add","src/visible.py","tests/test_visible.py","pyproject.toml","uv.lock")
    study.write_spec()
    return root


def closure(root,monkeypatch):
    names=study.source_hashes()
    deps={"protocol":study.OUT/"protocol.json",
        "preflight":study.OUT/"preflight/tests.json",
        "preflight_log":study.OUT/"preflight/tests.log",
        "checkpoint":root/"artifacts/fixed-checkpoint.pt",
        "VAL_selection":root/"artifacts/selection.json"}
    for key,path in deps.items():
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((key+" software fixture\n").encode())
    prior={"manifests":{},"named_files":0,"all_named_bytes_checked":False}
    monkeypatch.setattr(study,"prior_identities",lambda **kwargs:prior)
    receipt={"schema":1,"source_sha256":names,"prior":prior,
        **{key:study.reference(path) for key,path in deps.items()}}
    for name in names:
        study.write_bytes(study.OUT/"registration/source"/name,(root/name).read_bytes())
    study.write(study.OUT/"registration/source.json",receipt)
    return receipt


def commit_fixture(root):
    # This repository contains only these software fixtures. No real repository
    # changes, authentication or external network calls occur in this test.
    git(root,"add","src/visible.py","tests/test_visible.py",
        "src/wirelessjev/spatial_control","tests/spatial_control","scripts","docs",
        "pyproject.toml","uv.lock","artifacts")
    git(root,"-c","user.name=Source Fixture","-c","user.email=fixture@example.invalid",
        "commit","-q","-m","Bind temporary source fixture")
    return git(root,"rev-parse","HEAD")


def test_finite_registered_membership_has_48_fresh_and_two_parity_runs():
    protocol=study.spec()
    assert len(protocol["cases"])==8
    assert len({case["id"] for case in protocol["cases"]})==8
    assert {case["seed"] for case in protocol["cases"]}==set(study.SEEDS)
    assert len(study.SEEDS)==4 and len(study.ARMS)==6
    assert sum(len(case["order"]) for case in protocol["cases"])==48
    assert all(set(case["order"])==set(study.ARMS) for case in protocol["cases"])
    assert protocol["maximum_native_simulations"]==50
    assert protocol["disabled_parity_simulations"]==2
    assert protocol["weights"]["new_fits"]==0
    assert protocol["weights"]["selected_cell"]=="regression"
    assert protocol["weights"]["seed"]==0
    assert not protocol["original_SERVICE_opened"] and not protocol["default_changed"]


def test_source_collector_binds_new_namespace_without_scooping_deferred_user_work(sandbox):
    names=study.source_hashes()
    assert "src/visible.py" in names and "tests/test_visible.py" in names
    assert "src/wirelessjev/spatial_control/new.py" in names
    assert "tests/spatial_control/test_new.py" in names
    assert "scripts/run_spatial_control.py" in names
    assert "docs/spatial_control/PLAN.md" in names
    assert "artifacts/spatial_control/protocol.json" in names
    assert "pyproject.toml" in names and "uv.lock" in names
    assert not any("observation_frontier" in name for name in names)


def test_protocol_write_is_idempotent_and_refuses_semantic_change(sandbox):
    before=(study.OUT/"protocol.json").read_bytes()
    study.write_spec()
    assert (study.OUT/"protocol.json").read_bytes()==before
    (study.OUT/"protocol.json").write_text('{}')
    with pytest.raises(RuntimeError,match="Protocol changed"):study.write_spec()


@pytest.mark.parametrize("directory",["worlds","cases"])
def test_registration_refuses_data_or_attempt_membership_before_preflight(sandbox,directory):
    (study.OUT/directory).mkdir()
    with pytest.raises(RuntimeError,match="before fresh data"):study.register()
    assert not (study.OUT/"preflight").exists()


def test_exact_source_closure_checks_bytes_and_membership(sandbox,monkeypatch):
    receipt=closure(sandbox,monkeypatch)
    assert study.check_sources(published=False)==receipt
    (sandbox/"src/visible.py").write_text("# changed\n")
    with pytest.raises(RuntimeError,match="source changed"):study.check_sources(published=False)


def test_new_importable_namespace_file_invalidates_registration(sandbox,monkeypatch):
    closure(sandbox,monkeypatch)
    (sandbox/"src/wirelessjev/spatial_control/extra.py").write_text("# extra new source\n")
    with pytest.raises(RuntimeError,match="source changed"):study.check_sources(published=False)


def test_capsule_byte_changes_are_detected_separately(sandbox,monkeypatch):
    closure(sandbox,monkeypatch)
    (study.OUT/"registration/source/src/visible.py").write_text("# changed snapshot\n")
    with pytest.raises(RuntimeError,match="snapshot changed"):study.check_sources(published=False)


@pytest.mark.parametrize("key",["preflight","preflight_log","checkpoint","VAL_selection"])
def test_bound_dependency_changes_refuse_before_execution(sandbox,monkeypatch,key):
    receipt=closure(sandbox,monkeypatch)
    (sandbox/receipt[key]["path"]).write_text("changed dependency")
    with pytest.raises(RuntimeError,match="Dependency changed"):study.check_sources(published=False)


def test_publication_requires_actual_remote_commit_and_all_committed_bytes(sandbox,monkeypatch,tmp_path):
    closure(sandbox,monkeypatch)
    commit=commit_fixture(sandbox)
    remote=tmp_path/"fixture-remote.git"
    subprocess.run(["git","init","--bare","-q",str(remote)],check=True)
    git(sandbox,"remote","add","origin",str(remote))
    with pytest.raises(RuntimeError,match="Remote did not expose"):study.publish(commit,"fixture-source")
    git(sandbox,"push","-q","origin","fixture-source")
    publication=study.publish(commit,"fixture-source")
    assert publication["commit"]==publication["remote_observed_commit"]==commit
    assert study.check_sources()
    publication["remote_observed_commit"]="0"*40
    (study.OUT/"registration/git.json").write_bytes(study.canonical(publication))
    with pytest.raises(RuntimeError,match="Invalid source publication"):study.check_sources()


def test_remote_ref_alone_cannot_publish_an_uncommitted_source_change(sandbox,monkeypatch,tmp_path):
    # Commit the source before constructing the closure, then bind changed live
    # bytes. The remote ref matches, but the old commit must not pass publication.
    commit=commit_fixture(sandbox)
    (sandbox/"src/visible.py").write_text("# later source identity\n")
    closure(sandbox,monkeypatch)
    remote=tmp_path/"fixture-remote.git"
    subprocess.run(["git","init","--bare","-q",str(remote)],check=True)
    git(sandbox,"remote","add","origin",str(remote));git(sandbox,"push","-q","origin","fixture-source")
    with pytest.raises(RuntimeError,match="Commit source mismatch"):study.publish(commit,"fixture-source")
    assert not (study.OUT/"registration/git.json").exists()


def test_later_unrelated_tracked_modules_do_not_redefine_frozen_dependencies(sandbox,monkeypatch):
    receipt=closure(sandbox,monkeypatch)
    later=sandbox/"src/later_phase/new_module.py"
    later.parent.mkdir(parents=True);later.write_text("# future independent phase\n")
    git(sandbox,"add","src/later_phase/new_module.py")
    assert "src/later_phase/new_module.py" in study.source_hashes()
    assert "src/later_phase/new_module.py" not in receipt["source_sha256"]
    assert study.check_sources(published=False)==receipt


def test_deleted_own_module_fails_exact_membership(sandbox,monkeypatch):
    closure(sandbox,monkeypatch)
    (sandbox/"src/wirelessjev/spatial_control/new.py").unlink()
    with pytest.raises(RuntimeError,match="source changed"):study.check_sources(published=False)


def test_extra_capsule_file_is_not_implicitly_blessed(sandbox,monkeypatch):
    closure(sandbox,monkeypatch)
    (study.OUT/"registration/source/extra.py").write_text("# not registered\n")
    with pytest.raises(RuntimeError,match="snapshot membership"):study.check_sources(published=False)


def test_publication_cache_cannot_rebase_changed_receipt_on_same_commit(sandbox,monkeypatch,tmp_path):
    receipt=closure(sandbox,monkeypatch)
    commit=commit_fixture(sandbox)
    remote=tmp_path/"fixture-remote.git"
    subprocess.run(["git","init","--bare","-q",str(remote)],check=True)
    git(sandbox,"remote","add","origin",str(remote));git(sandbox,"push","-q","origin","fixture-source")
    publication=study.publish(commit,"fixture-source")
    assert study.check_sources()
    path=sandbox/"src/visible.py";path.write_text("# changed after cached publication\n")
    receipt["source_sha256"]["src/visible.py"]=study.digest(path)
    (study.OUT/"registration/source/src/visible.py").write_bytes(path.read_bytes())
    (study.OUT/"registration/source.json").write_bytes(study.canonical(receipt))
    publication["source_sha256"]=study.digest(study.OUT/"registration/source.json")
    (study.OUT/"registration/git.json").write_bytes(study.canonical(publication))
    with pytest.raises(RuntimeError,match="Published source changed"):study.check_sources()
