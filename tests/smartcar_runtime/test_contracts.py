"""Synthetic contract fixtures test validation, not physical car performance."""
import copy
import time

import pytest

from wirelessjev.smartcar_runtime.contracts import (
    canonical_sha256, load_records, site_sha256, validate_record, validate_site,
)
from wirelessjev.smartcar_runtime.workflow import append_record, ssh_argv


def site():
    return {"schema": 1, "site_id": "software-contract-fixture", "dimensions_m": [20, 10, 4],
            "cad_boxes_m": [], "aps": [{"bssid": "02:00:00:00:00:01", "ssid": "unit-only",
                "xyz_m": [1, 2, 3], "frequency_mhz": 5180, "tx_power_dbm": 20}],
            "receiver": {"noise_floor_dbm": -95, "offset_db": 0, "quantization_db": 0,
                         "reference_gauge_assumed": True}}


def record():
    now = time.time_ns()
    return {"schema": 1, "site_id": site()["site_id"], "site_sha256": site_sha256(site()),
            "capture_id": "fixture-1", "source": "hardware", "interface": "wlan1",
            "fresh_scan": True, "started_unix_ns": 100, "completed_unix_ns": 200,
            "pc_request_started_unix_ns": now-1000, "pc_received_unix_ns": now-100,
            "pose": {"source": "operator", "xyz_m": [2, 2, 0.3]},
            "current_bssid": "02:00:00:00:00:01",
            "aps": [{"bssid": "02:00:00:00:00:01", "ssid": "unit-only",
                     "frequency_mhz": 5180, "signal_dbm": -60}]}


def test_site_normalization_is_hash_stable_and_does_not_mutate_input():
    original = site()
    assert site_sha256(original) == site_sha256(validate_site(original))
    assert original == site()
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


@pytest.mark.parametrize("path,value", [
    (("dimensions_m", 0), float("nan")), (("dimensions_m", 1), True),
    (("aps", 0, "bssid"), "ff:ff:ff:ff:ff:ff"),
    (("aps", 0, "frequency_mhz"), 3000), (("aps", 0, "xyz_m", 0), -1),
    (("receiver", "noise_floor_dbm"), None), (("receiver", "reference_gauge_assumed"), 1),
])
def test_site_rejects_ambiguous_or_nonphysical_input(path, value):
    data = site()
    leaf = data
    for key in path[:-1]:
        leaf = leaf[key]
    leaf[path[-1]] = value
    with pytest.raises(ValueError):
        validate_site(data)


def test_unsynchronized_car_clock_is_not_compared_to_desktop_epoch():
    row = record()
    row["started_unix_ns"] = 10**25
    row["completed_unix_ns"] = 10**25+100
    assert validate_record(site(), row)["completed_unix_ns"] == 10**25+100


@pytest.mark.parametrize("change", [
    {"fresh_scan": False}, {"interface": "wlan0"}, {"source": "simulation"},
    {"site_sha256": "0"*64}, {"pc_received_unix_ns": 10**25},
    {"completed_unix_ns": 1}, {"password": "unit-secret"},
])
def test_unavailable_wrong_site_or_fabricated_boundary_is_rejected(change):
    row = record()
    row.update(change)
    with pytest.raises(ValueError):
        validate_record(site(), row)


def test_missing_unheard_ap_is_not_invented_as_censor_measurement():
    config = site()
    ap = copy.deepcopy(config["aps"][0])
    ap["bssid"] = "02:00:00:00:00:02"
    config["aps"].append(ap)
    row = record()
    row["site_sha256"] = site_sha256(config)
    assert len(validate_record(config, row)["aps"]) == 1


def test_survey_rejects_replayed_capture(tmp_path):
    path = tmp_path / "survey.jsonl"
    row = record()
    append_record(path, row)
    append_record(path, row)
    with pytest.raises(ValueError, match="replayed"):
        load_records(path, site())


def test_journal_rejects_nested_credentials_before_writing(tmp_path):
    path = tmp_path / "journal.jsonl"
    with pytest.raises(ValueError):
        append_record(path, {"nested": [{"password": "unit-secret"}]})
    assert not path.exists()


@pytest.mark.parametrize("host", ["-oProxyCommand=bad", "pi@192.168.1.11", "example.com;id", "bad name"])
def test_ssh_destination_does_not_accept_options_or_shell_syntax(tmp_path, host):
    key = tmp_path / "known_hosts"
    key.write_text("fixture-not-a-trusted-key\n")
    with pytest.raises(ValueError):
        ssh_argv(host, key)


def test_ssh_preserves_strict_pinning_and_nonstandard_port(tmp_path):
    key = tmp_path / "known_hosts"
    key.write_text("fixture-not-a-trusted-key\n")
    argv = ssh_argv("192.168.1.11", key, port=2222)
    assert "StrictHostKeyChecking=yes" in argv
    assert "UserKnownHostsFile="+str(key.resolve()) in argv
    assert argv[-1] == "pi@192.168.1.11" and argv[1:4] == ["-T", "-p", "2222"]
