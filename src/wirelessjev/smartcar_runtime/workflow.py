"""Desktop workflow for real, operator-positioned SmartCar radio surveys.

RadioMap is the numerical default. Pinned neural checkpoints run only in shadow.
The only mutation is a separately confirmed existing SmartCarCmd request.
"""
from __future__ import annotations

import argparse
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import time

from .contracts import (bssid, finite, load_records,
                        records_sha256, reject_secrets, site_sha256, validate_query,
                        validate_record, validate_site)

COMMANDS = {
    "agent": 'python3 -u "$HOME/smartcar-wirelessJEV/PI/wj_car_agent.py"',
    "smartcar": ('sudo -n python3 -u "$HOME/SmartCarCmd.py" '
                 '--interface wlan1 --management-interface wlan0'),
}
RELEASED_CHECKPOINTS = {
    "nested_joint-seed0": "57c710af2db4f4bd7abfd031a03a76359c27100568d3ebe277ab3c738a69c3bf",
    "nested_joint-seed1": "2de1c28d4ea41298cac58ca64cabe6d86a8c76688646a3dd4f479ab6686a9b52",
    "nested_joint-seed2": "16571ca0adb781349e1988fae74a648caf06c76c13c1cf5f605b0e5a4fee974c",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_document(path, value):
    reject_secrets(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_record(path, value):
    reject_secrets(value)
    encoded = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        while encoded:
            size = os.write(descriptor, encoded)
            if size <= 0:
                raise OSError("Could not commit journal record")
            encoded = encoded[size:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Journal:
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, event, **fields):
        append_record(self.path, {"schema": 1, "pc_unix_ns": time.time_ns(),
                                  "event": event, **fields})


def ssh_argv(host, known_hosts, user="pi", port=22):
    if not isinstance(host, str) or not host or host.startswith("-"):
        raise ValueError("Use a bare management hostname or IP")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}", host):
            raise ValueError("Use a bare management hostname or IP") from None
    if not isinstance(user, str) or not re.fullmatch(r"[a-z_][a-z0-9_.-]{0,31}", user):
        raise ValueError("Invalid SSH user")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Invalid SSH port")
    known_hosts = Path(known_hosts).expanduser().resolve()
    if not known_hosts.is_file() or known_hosts.stat().st_size == 0:
        raise ValueError("Provide the independently verified, nonempty known_hosts file")
    return ["ssh", "-T", "-p", str(port), "-o", "StrictHostKeyChecking=yes",
            "-o", "UserKnownHostsFile=" + str(known_hosts),
            "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=2", "-o", "NumberOfPasswordPrompts=1",
            user + "@" + host]


class SSHTransport:
    """Finite single-attempt SSH requests; failures are conservatively ambiguous.

Output is bounded while reading, including a separate stderr budget. Remote
stderr is discarded and cannot copy credentials into a receipt. OpenSSH owns
its normal interactive authentication prompt; SSH keys are preferred.
"""
    def __init__(self, host, known_hosts, user="pi", port=22):
        self.argv = ssh_argv(host, known_hosts, user, port)

    def __call__(self, command_key, payload, timeout):
        if command_key not in COMMANDS or not 0 < timeout <= 180:
            raise ValueError("Only fixed, finite SmartCar commands are supported")
        data = (json.dumps(payload, allow_nan=False) + "\n").encode()
        if len(data) > 16384:
            raise ValueError("Request is too large")
        process = subprocess.Popen(self.argv + [COMMANDS[command_key]],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, bufsize=0)
        output, stderr_bytes = bytearray(), 0
        selector = selectors.DefaultSelector()
        deadline = time.monotonic() + timeout
        try:
            for stream in (process.stdin, process.stdout, process.stderr):
                os.set_blocking(stream.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE, "input")
            selector.register(process.stdout, selectors.EVENT_READ, "output")
            selector.register(process.stderr, selectors.EVENT_READ, "error")
            while selector.get_map():
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("SSH request outcome is unknown; do not resend mutations")
                for key, _ in selector.select(min(left, 0.2)):
                    stream, kind = key.fileobj, key.data
                    if kind == "input":
                        written = os.write(stream.fileno(), data)
                        data = data[written:]
                        if not data:
                            selector.unregister(stream)
                            stream.close()
                    else:
                        chunk = os.read(stream.fileno(), 65536)
                        if not chunk:
                            selector.unregister(stream)
                            stream.close()
                        elif kind == "output":
                            output.extend(chunk)
                            if len(output) > 2 * 1024 * 1024:
                                raise ValueError("SSH reply exceeded its output limit")
                        else:
                            stderr_bytes += len(chunk)
                            if stderr_bytes > 65536:
                                raise ValueError("SSH stderr exceeded its output limit")
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
            try:
                value = json.loads(output)
            except (ValueError, UnicodeError):
                raise RuntimeError("SSH reply was not a complete JSON result") from None
            if not isinstance(value, dict):
                raise RuntimeError("SSH reply was not a JSON object")
            if process.returncode and value.get("ok") is not False:
                raise RuntimeError("SSH ended without a trustworthy result")
            return value
        finally:
            selector.close()
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _site(args):
    return validate_site(read_json(args.site))


def _transport(args):
    return SSHTransport(args.host, args.known_hosts, args.user, args.port)


def propose(site, records, map_model, point, *, now_ns=None, maximum_age_s=120.,
            maximum_distance_m=5., switch_margin_db=6., minimum_signal_dbm=-75.):
    """Non-actuating typed RSSI proposal; no invented classifier probability."""
    from .field import predict_map
    now = time.time_ns() if now_ns is None else now_ns
    point = validate_query(site, point)
    values = {"maximum_age_s": maximum_age_s, "maximum_distance_m": maximum_distance_m,
              "switch_margin_db": switch_margin_db, "minimum_signal_dbm": minimum_signal_dbm}
    values = {k: finite(v, k) for k, v in values.items()}
    if min(values["maximum_age_s"], values["maximum_distance_m"]) <= 0 or values["switch_margin_db"] < 0:
        raise ValueError("Positive age/distance limits and a nonnegative margin are required")
    receipt = {"schema": 1, "decision": "DEFER", "target_bssid": None,
               "site_sha256": site_sha256(site), "query_xyz_m": point,
               "sent_to_car": False, "production_qualified": False,
               "model": "radiomap", "probability": None, "limits": values,
               "reason": "No eligible current observation", "assumptions": [
                   "RSSI comparison is not a packet-risk or deadline probability",
                   "No command is authorized or transmitted by this proposal"]}
    admitted = [validate_record(site, r, now_ns=now) for r in records]
    if not admitted:
        return receipt
    if (map_model.get("site_sha256") != site_sha256(site)
            or map_model.get("records_sha256") != records_sha256(admitted)):
        receipt["reason"] = "Refit the numerical map from this exact admitted survey before proposing"
        return receipt
    latest = max(admitted, key=lambda r: r["pc_received_unix_ns"])
    age = (now - latest["pc_request_started_unix_ns"]) / 1e9
    distance = sum((a - b) ** 2 for a, b in zip(point, latest["pose"]["xyz_m"])) ** .5
    receipt.update(source_capture_id=latest["capture_id"], conservative_age_s=age,
                   latest_capture_distance_m=distance, current_bssid=latest.get("current_bssid"))
    if age > values["maximum_age_s"] or distance > 0.5:
        receipt["reason"] = "A recent fresh scan at the requested position is required"
        return receipt
    current = latest.get("current_bssid")
    visible = {r["bssid"]: r for r in latest["aps"]}
    if not current or current not in visible:
        receipt["reason"] = "Current link is not present in the fresh scan"
        return receipt
    configured = {ap["bssid"]: ap for ap in site["aps"]}
    if current not in configured:
        receipt["reason"] = "Current BSSID is not in the declared site"
        return receipt
    predictions = predict_map(map_model, [point])
    candidates = []
    for row in predictions:
        bid = row["bssid"]
        if (row.get("status") != "prediction" or bid not in visible
                or row.get("ssid") != configured[current]["ssid"]):
            continue
        support = row.get("support", {})
        if support.get("nearest_m", float("inf")) > values["maximum_distance_m"]:
            continue
        # Spread is descriptive; this conservative score is not a calibrated quantile.
        score = row["mean_dbm"] - row["std_db"]
        candidates.append({"bssid": bid, "score_dbm": score,
                           "observed_signal_dbm": visible[bid]["signal_dbm"],
                           "mean_dbm": row["mean_dbm"], "std_db": row["std_db"]})
    receipt["candidates"] = candidates
    by_id = {r["bssid"]: r for r in candidates}
    if current not in by_id:
        receipt["reason"] = "Current link lacks numerical map support"
        return receipt
    best = max(candidates, key=lambda r: (r["score_dbm"], r["bssid"]))
    if best["observed_signal_dbm"] < values["minimum_signal_dbm"]:
        receipt["reason"] = "No eligible observed candidate exceeds the RSSI threshold"
        return receipt
    improvement = best["score_dbm"] - by_id[current]["score_dbm"]
    if best["bssid"] != current and improvement >= values["switch_margin_db"]:
        observed_gain = best["observed_signal_dbm"] - by_id[current]["observed_signal_dbm"]
        if observed_gain < values["switch_margin_db"]:
            receipt["reason"] = "Fresh observations do not corroborate the map's switching margin"
        else:
            receipt.update(decision="PROPOSE_SWITCH", target_bssid=best["bssid"],
                           reason="Fresh same-SSID observations and the map both exceed the RSSI margin")
    else:
        receipt.update(decision="STAY", target_bssid=current,
                       reason="Current link wins or the switching margin is not met")
    return receipt


def _profile(path, ssid):
    data = read_json(path)
    if data.get("version") != 1 or not isinstance(data.get("networks"), list):
        raise ValueError("Private WiFi.json must use the existing version-1 network schema")
    candidates = [p for p in data["networks"] if p.get("enabled") is True and p.get("ssid") == ssid]
    if len(candidates) != 1:
        raise ValueError("Exactly one enabled matching Wi-Fi profile is required")
    profile = candidates[0]
    security, password = profile.get("security", "wpa-psk"), profile.get("password", "")
    if security not in ("open", "wpa-psk", "sae") or not isinstance(password, str) or len(password.encode()) > 256:
        raise ValueError("Unsupported private Wi-Fi profile")
    if security != "open" and not password:
        raise ValueError("Encrypted profile requires its private password")
    return {"ssid": ssid, "security": security, "password": password, "enabled": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "status", "capture", "switch"):
        sub = commands.add_parser(name)
        sub.add_argument("--host", required=True)
        sub.add_argument("--known-hosts", required=True)
        sub.add_argument("--user", default="pi")
        sub.add_argument("--port", type=int, default=22)
        if name == "capture":
            sub.add_argument("--site", required=True)
            sub.add_argument("--survey", required=True)
            sub.add_argument("--xyz", nargs=3, type=float, required=True)
            sub.add_argument("--active-scan", action="store_true", required=True)
        if name == "switch":
            sub.add_argument("--wifi-json", required=True)
            sub.add_argument("--ssid", required=True)
            sub.add_argument("--bssid", required=True)
            sub.add_argument("--journal", required=True)
    sub = commands.add_parser("init-site")
    sub.add_argument("--out", required=True)
    sub = commands.add_parser("validate-site")
    sub.add_argument("--site", required=True)
    for name in ("fit-map", "predict", "shadow", "propose"):
        sub = commands.add_parser(name)
        sub.add_argument("--site", required=True)
        if name in ("fit-map", "shadow", "propose"):
            sub.add_argument("--survey", required=True)
        if name in ("predict", "propose"):
            sub.add_argument("--map", required=True)
        if name != "fit-map":
            sub.add_argument("--xyz", nargs=3, type=float, required=True)
        if name in ("fit-map", "shadow", "propose"):
            sub.add_argument("--out", required=True)
        if name == "shadow":
            sub.add_argument("--models", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init-site":
            value = {"schema": 1, "site_id": "replace-with-your-site",
                     "dimensions_m": [None, None, None], "cad_boxes_m": [], "aps": [],
                     "receiver": {"noise_floor_dbm": None, "offset_db": 0.,
                                  "quantization_db": 0., "reference_gauge_assumed": False}}
            write_document(args.out, value)
            result = {"template": str(args.out), "valid_site": False,
                      "next": "Fill measured coordinates and explicitly declared radio assumptions"}
        elif args.command == "validate-site":
            site = _site(args)
            result = {"valid_site": True, "site_sha256": site_sha256(site), "aps": len(site["aps"])}
        elif args.command in ("doctor", "status"):
            key = "agent" if args.command == "doctor" else "smartcar"
            result = _transport(args)(key, {"action": args.command}, 20.)
        elif args.command == "capture":
            site = _site(args)
            point = validate_query(site, args.xyz)
            request = {"action": "capture", "active_scan": True,
                       "site_id": site["site_id"], "site_sha256": site_sha256(site),
                       "pose": {"xyz_m": point, "source": "operator"}}
            started = time.time_ns()
            result = _transport(args)("agent", request, 45.)
            received = time.time_ns()
            if result.get("ok") is not True:
                raise RuntimeError("Collector refused or could not complete a fresh scan; no record admitted")
            row = result["record"]
            if row.get("pose") != request["pose"]:
                raise ValueError("Collector position metadata differs from the explicit survey request")
            row.update(pc_request_started_unix_ns=started, pc_received_unix_ns=received)
            row = validate_record(site, row, now_ns=received)
            append_record(args.survey, row)
            result = {"captured": True, "capture_id": row["capture_id"], "aps": len(row["aps"]),
                      "survey": str(args.survey), "pose": row["pose"],
                      "active_scan_cost_incurred": True, "pc_round_trip_s": (received-started)/1e9}
        elif args.command == "switch":
            from .control import supervised_switch
            profile = _profile(args.wifi_json, args.ssid)
            result = supervised_switch(_transport(args), Journal(args.journal),
                                       profiledict=profile, target_bssid=bssid(args.bssid))
        else:
            from .field import fit_map, predict_map, predict_radiojepa
            site = _site(args)
            if args.command == "fit-map":
                result = fit_map(site, load_records(args.survey, site))
                write_document(args.out, result)
                result = {"map_written": str(args.out), "site_sha256": result["site_sha256"],
                          "records_sha256": result["records_sha256"],
                          "model_sha256": result["model_sha256"], "model": result["model_kind"]}
            elif args.command == "predict":
                model = read_json(args.map)
                if model.get("site_sha256") != site_sha256(site):
                    raise ValueError("Map belongs to another site configuration")
                result = predict_map(model, [validate_query(site, args.xyz)])
            elif args.command == "shadow":
                records = load_records(args.survey, site)
                manifest_path = Path(args.models).resolve()
                manifest = read_json(manifest_path)
                members = manifest.get("members")
                if (manifest.get("schema") != 1 or manifest.get("cell") != "nested_joint"
                        or not isinstance(members, list) or len(members) != 3
                        or {m.get("name") for m in members if isinstance(m, dict)} != set(RELEASED_CHECKPOINTS)
                        or any(m.get("sha256") != RELEASED_CHECKPOINTS.get(m.get("name")) for m in members)):
                    raise ValueError("Shadow requires the three exact released member identities")
                result = {"schema": 1, "site_sha256": site_sha256(site),
                          "experimental_shadow_only": True, "sent_to_car": False,
                          "hardware_accuracy_measured": False, "members": {}}
                for member in members:
                    name, path = member["name"], (manifest_path.parent / member["path"]).resolve()
                    if not path.is_relative_to(manifest_path.parent):
                        raise ValueError("Checkpoint path escapes the bundle model directory")
                    result["members"][name] = predict_radiojepa(site, records,
                        [validate_query(site, args.xyz)], path, member["sha256"])
                write_document(args.out, result)
            else:
                model = read_json(args.map)
                if model.get("site_sha256") != site_sha256(site):
                    raise ValueError("Map belongs to another site configuration")
                result = propose(site, load_records(args.survey, site), model, args.xyz)
                write_document(args.out, result)
        reject_secrets(result)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        if args.command == "doctor":
            doctor = result.get("doctor")
            role = doctor.get("role_check") if isinstance(doctor, dict) else None
            role = role if isinstance(role, dict) else {}
            return 0 if result.get("ok") is True and role.get("safe") is True else 2
        if args.command == "status":
            return 0 if result.get("ok") is True else 2
        if args.command == "switch":
            return 0 if result.get("outcome") in ("APPLIED", "NOT_APPLIED") else 2
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Read status before any further switch; never resend an uncertain mutation.", file=sys.stderr)
        return 130
    except Exception as exc:
        code = "DEPENDENCY_UNAVAILABLE" if isinstance(exc, ImportError) else "WORKFLOW_FAILED"
        if args.command == "validate-site" and isinstance(exc, ValueError):
            print("Invalid site: " + str(exc)[:200], file=sys.stderr)
        print(code + " in " + args.command + ". Check local files/dependencies and read-only doctor/status; no automatic retry was made.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
