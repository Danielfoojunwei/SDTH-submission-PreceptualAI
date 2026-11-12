"""Verify the actual ZIP in a clean map environment; fixtures are software only.

This does not fabricate a hardware validation result. No SSH, radio, training
or ray tracer is called. Only the local CLOUD doctor's read-only missing-role
behavior is observed; its result is explicitly not a SmartCar observation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "final"
OUT.mkdir(parents=True, exist_ok=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(map_python):
    archive_path = ROOT / "artifacts/smartcar_deployment/smartcar-wirelessJEV.zip"
    receipt = json.loads(archive_path.with_suffix(".receipt.json").read_text())
    assert digest(archive_path) == receipt["archive_sha256"]
    with tempfile.TemporaryDirectory(prefix="wj-portable-software-") as folder:
        base = Path(folder)
        with zipfile.ZipFile(archive_path) as archive:
            assert len(archive.namelist()) == 28
            assert all(name.startswith("smartcar-wirelessJEV/") and ".." not in Path(name).parts
                       for name in archive.namelist())
            archive.extractall(base)
        bundle = base / "smartcar-wirelessJEV"
        manifest = json.loads((bundle/"MANIFEST.json").read_text())
        env = dict(os.environ, PYTHONPATH="", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        check = subprocess.run([map_python, str(bundle/"verify_bundle.py")], cwd=base,
                               env=env, capture_output=True, text=True, check=True)
        integrity = json.loads(check.stdout)
        assert integrity["passes"] and integrity["files_checked"] == 27
        # These are explicitly synthetic API fixtures, not alleged car captures.
        # The schema's source label is exercised, never presented as evidence.
        setup_code = """
import importlib.util,json,pathlib,sys,time
sys.path.insert(0,sys.argv[1])
from wirelessjev.smartcar_runtime.contracts import site_sha256
from wirelessjev.smartcar_runtime import field
assert pathlib.Path(field.__file__).is_relative_to(pathlib.Path(sys.argv[1]))
assert all(importlib.util.find_spec(name) is None for name in ('torch','sionna','transformers','huggingface_hub'))
site={'schema':1,'site_id':'portable-software-fixture','dimensions_m':[20,10,4],'cad_boxes_m':[],
 'aps':[{'bssid':'02:00:00:00:00:01','ssid':'unit-only','xyz_m':[1,2,3],'frequency_mhz':5180,'tx_power_dbm':20},
        {'bssid':'02:00:00:00:00:02','ssid':'unit-only','xyz_m':[15,2,3],'frequency_mhz':5180,'tx_power_dbm':20}],
 'receiver':{'noise_floor_dbm':-95,'offset_db':0,'quantization_db':0,'reference_gauge_assumed':True}}
now=time.time_ns();rows=[]
for i,x in enumerate((2,3,4)):
 rows.append({'schema':1,'site_id':site['site_id'],'site_sha256':site_sha256(site),'capture_id':'software-only-'+str(i),
  'source':'hardware','interface':'wlan1','fresh_scan':True,'started_unix_ns':100+i*2,'completed_unix_ns':101+i*2,
  'pc_request_started_unix_ns':now-(4-i)*10**9,'pc_received_unix_ns':now-(4-i)*10**9+10**6,
  'pose':{'source':'operator','xyz_m':[x,2,.3]},'current_bssid':site['aps'][0]['bssid'],
  'aps':[{'bssid':a['bssid'],'ssid':a['ssid'],'frequency_mhz':a['frequency_mhz'],'signal_dbm':v}
         for a,v in zip(site['aps'],(-70.,-45.))]})
pathlib.Path('site.json').write_text(json.dumps(site))
pathlib.Path('software-fixture.jsonl').write_text(''.join(json.dumps(r)+'\\n' for r in rows))
import importlib.metadata
print(json.dumps({'numpy':importlib.metadata.version('numpy'),'scipy':importlib.metadata.version('scipy'),
 'torch_present':False,'Sionna_HF_Transformers_present':False,'input_origin':'synthetic_API_fixture_not_hardware_evidence'}))
"""
        setup = subprocess.run([map_python, "-c", setup_code, str(bundle/"src")], cwd=base,
                               env=env, capture_output=True, text=True, check=True)
        minimal = json.loads(setup.stdout)
        entry = str(bundle/"PC/run.py")
        steps = (
            ["validate-site", "--site", "site.json"],
            ["fit-map", "--site", "site.json", "--survey", "software-fixture.jsonl", "--out", "map.json"],
            ["predict", "--site", "site.json", "--map", "map.json", "--xyz", "4", "2", ".3"],
            ["propose", "--site", "site.json", "--survey", "software-fixture.jsonl", "--map", "map.json",
             "--xyz", "4", "2", ".3", "--out", "proposal.json"],
        )
        for args in steps:
            subprocess.run([map_python, entry, *args], cwd=base, env=env,
                           capture_output=True, text=True, check=True)
        proposal = json.loads((base/"proposal.json").read_text())
        assert proposal["decision"] == "PROPOSE_SWITCH" and proposal["sent_to_car"] is False
        assert proposal["probability"] is None
        blocker = """
import importlib.abc,runpy,sys
class BlockResearch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('sionna','mitsuba','drjit','transformers','huggingface_hub','chronos','accelerate'):
   raise RuntimeError('Forbidden research/training dependency')
sys.meta_path.insert(0,BlockResearch())
sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')
"""
        shadow_command = [sys.executable, "-c", blocker, entry, "shadow", "--site", "site.json",
            "--survey", "software-fixture.jsonl", "--xyz", "4", "2", ".3",
            "--models", str(bundle/"models/manifest.json"), "--out", "shadow.json"]
        subprocess.run(shadow_command, cwd=base, env=env, capture_output=True, text=True, check=True)
        shadow = json.loads((base/"shadow.json").read_text())
        assert len(shadow["members"]) == 3 and shadow["sent_to_car"] is False
        for name, rows in shadow["members"].items():
            assert len(rows) == 2 and all(row["status"] == "prediction" for row in rows)
            assert all(row["real_hardware_accuracy_validated"] is False for row in rows)
        doctor = subprocess.run([map_python, str(bundle/"PI/wj_car_agent.py")],
            input='{"action":"doctor"}\n', capture_output=True, text=True, env=env, check=True)
        local = json.loads(doctor.stdout)
        assert local["doctor"]["role_check"]["safe"] is False
        # Read-only software execution may create bytecode only; shipped inputs
        # and all pinned model/source bytes must remain unchanged.
        for name, expected in manifest["files_sha256"].items():
            assert digest(bundle/name) == expected, name
        return {"schema": 1, "passes": True, "verifier_sha256": digest(Path(__file__)),
            "archive_sha256": digest(archive_path), "archive_bytes": archive_path.stat().st_size,
            "bundle_members": 28, "integrity": integrity, "clean_map_environment": minimal,
            "actual_minimal_CLI_commands_passed": [step[0] for step in steps],
            "released_shadow_members_executed": sorted(shadow["members"]),
            "research_training_dependencies_blocked_during_shadow": True,
            "all_shipped_input_hashes_unchanged": True,
            "local_doctor_machine": "cloud_executor_not_SmartCar", "local_missing_radio_guard_passed": True,
            "test_data_origin": "synthetic_API_fixtures_only", "real_car_measurements": 0,
            "SSH_requests": 0, "hardware_scan_or_actuation": 0, "training_or_RT": 0,
            "SOTA_or_hardware_qualification": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-python", required=True)
    value = verify(parser.parse_args().map_python)
    with (OUT/"portable_verification.json").open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: value[key] for key in ("passes", "archive_sha256", "archive_bytes", "released_shadow_members_executed")}))
