"""Numerical software fixtures exercise proposal boundaries; no car evidence."""
import copy
import io
import time

from wirelessjev.smartcar_runtime.contracts import site_sha256
from wirelessjev.smartcar_runtime.field import fit_map
from wirelessjev.smartcar_runtime.workflow import main, propose


def fixture():
    site = {"schema": 1, "site_id": "proposal-software-fixture", "dimensions_m": [20, 10, 4],
            "cad_boxes_m": [], "aps": [
                {"bssid": "02:00:00:00:00:01", "ssid": "unit-only", "xyz_m": [1, 2, 3],
                 "frequency_mhz": 5180, "tx_power_dbm": 20},
                {"bssid": "02:00:00:00:00:02", "ssid": "unit-only", "xyz_m": [15, 2, 3],
                 "frequency_mhz": 5180, "tx_power_dbm": 20}],
            "receiver": {"noise_floor_dbm": -95, "offset_db": 0, "quantization_db": 0,
                         "reference_gauge_assumed": True}}
    now = time.time_ns()
    rows = []
    for i, x in enumerate((2, 3, 4)):
        rows.append({"schema": 1, "site_id": site["site_id"], "site_sha256": site_sha256(site),
            "capture_id": "proposal-fixture-"+str(i), "source": "hardware", "interface": "wlan1",
            "fresh_scan": True, "started_unix_ns": 100+i*2, "completed_unix_ns": 101+i*2,
            "pc_request_started_unix_ns": now-(4-i)*1_000_000_000,
            "pc_received_unix_ns": now-(4-i)*1_000_000_000+1_000_000,
            "pose": {"source": "operator", "xyz_m": [x, 2, 0.3]},
            "current_bssid": site["aps"][0]["bssid"],
            "aps": [{"bssid": ap["bssid"], "ssid": ap["ssid"], "frequency_mhz": ap["frequency_mhz"],
                     "signal_dbm": signal} for ap, signal in zip(site["aps"], (-70., -45.))]})
    return site, rows


def test_supported_corroborated_proposal_is_typed_but_never_actuates():
    site, records = fixture()
    result = propose(site, records, fit_map(site, records), [4, 2, 0.3])
    assert result["decision"] == "PROPOSE_SWITCH"
    assert result["target_bssid"] == site["aps"][1]["bssid"]
    assert result["probability"] is None and result["sent_to_car"] is False


def test_exact_survey_refit_is_required_after_another_capture():
    site, records = fixture()
    old = fit_map(site, records[:2])
    result = propose(site, records, old, [4, 2, 0.3])
    assert result["decision"] == "DEFER" and "Refit" in result["reason"]


def test_latest_scan_at_another_position_cannot_authorize_a_proposal():
    site, records = fixture()
    result = propose(site, records, fit_map(site, records), [2, 2, 0.3])
    assert result["decision"] == "DEFER" and "requested position" in result["reason"]


def test_cross_ssid_candidate_is_not_an_eligible_switch():
    site, records = fixture()
    site["aps"][1]["ssid"] = "different-network"
    for row in records:
        row["site_sha256"] = site_sha256(site)
        row["aps"][1]["ssid"] = "different-network"
    result = propose(site, records, fit_map(site, records), [4, 2, 0.3])
    assert result["decision"] == "STAY"


def test_map_cannot_override_contradictory_current_observation():
    site, records = fixture()
    # Many nearby earlier observations can dominate a fit. The independent
    # observed-margin guard must still refuse this freshly contradicted switch.
    for row in records:
        row["aps"][1]["signal_dbm"] = -65.
    result = propose(site, records, fit_map(site, records), [4, 2, 0.3], switch_margin_db=6.)
    assert result["decision"] != "PROPOSE_SWITCH"


def test_switch_cli_rejects_piped_confirmation_before_credentials_or_network(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("02:00:00:00:00:02\n"))
    code = main(["switch", "--host", "not-a-real-host", "--known-hosts", "missing-hosts",
                 "--wifi-json", "missing-private-credentials", "--ssid", "unit-only",
                 "--bssid", "02:00:00:00:00:02", "--journal", "must-not-be-created.jsonl"])
    assert code == 2 and "interactive terminal" in capsys.readouterr().err
