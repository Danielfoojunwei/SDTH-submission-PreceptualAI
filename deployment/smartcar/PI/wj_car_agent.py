#!/usr/bin/env python3
"""Read-only Raspberry Pi radio collector; Python 3.9+, standard library only.

One JSON request is read from stdin and one JSON response is written to stdout.
Only doctor and explicitly requested active capture are supported. This program
never changes a connection, interface, address, route, robot or configuration.
"""
import json
import math
import os
from pathlib import Path
import platform
import re
import selectors
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass


INTERFACE = 'wlan1'
MANAGEMENT_INTERFACE = 'wlan0'
MANAGEMENT_IP = '192.168.1.11'
SAFE_PATH = '/usr/sbin:/usr/bin:/sbin:/bin'
MAX_REQUEST_BYTES = 8192
MAX_SCAN_BYTES = 524288
BSSID = re.compile(r'(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}')
SCAN_BSS = re.compile(r'^BSS ((?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2})(?:\([^\r\n]*\))?(?:\s+--\s+[^\r\n]*)?$')


class AgentError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str


def run_command(argv, timeout_s=8.0, max_stdout_bytes=MAX_SCAN_BYTES):
    """Bound wall time and both pipes without shell or unbounded buffering."""
    if (not isinstance(argv, (list, tuple)) or not argv
            or any(not isinstance(arg, str) for arg in argv)):
        raise AgentError('command_invalid', 'An internal command is invalid.')
    try:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
            close_fds=True, env={'PATH': SAFE_PATH, 'LC_ALL': 'C', 'LANG': 'C'})
    except OSError:
        raise AgentError('command_unavailable', 'A required command is unavailable.') from None
    output = bytearray()
    stderr_bytes = 0
    deadline = time.monotonic() + timeout_s
    try:
        with selectors.DefaultSelector() as selected:
            selected.register(process.stdout, selectors.EVENT_READ, 'stdout')
            selected.register(process.stderr, selectors.EVENT_READ, 'stderr')
            while selected.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AgentError('command_timeout', 'A read-only command exceeded its time limit.')
                for key, _ in selected.select(min(remaining, 0.1)):
                    block = os.read(key.fileobj.fileno(), 16384)
                    if not block:
                        selected.unregister(key.fileobj)
                    elif key.data == 'stdout':
                        if len(output) + len(block) > max_stdout_bytes:
                            raise AgentError('command_output_limit', 'A command exceeded its output limit.')
                        output.extend(block)
                    else:
                        stderr_bytes += len(block)
                        if stderr_bytes > 65536:
                            raise AgentError('command_output_limit', 'A command exceeded its output limit.')
            try:
                returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                raise AgentError('command_timeout', 'A read-only command exceeded its time limit.') from None
        try:
            text = output.decode('utf-8', errors='strict')
        except UnicodeDecodeError:
            raise AgentError('command_output_invalid', 'A command returned invalid text.') from None
        return CommandResult(returncode, text)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()


def tool_paths():
    return {name: shutil.which(name, path=SAFE_PATH) for name in ('ip', 'iw', 'sudo')}


def _finite_coordinate(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def validate_request(value):
    if not isinstance(value, dict) or type(value.get('action')) is not str:
        raise AgentError('invalid_request', 'A JSON object with a supported action is required.')
    action = value['action']
    if action == 'doctor':
        if set(value) != {'action'}:
            raise AgentError('invalid_request', 'Doctor accepts only its action field.')
        return value
    if action != 'capture':
        raise AgentError('unsupported_action', 'Only doctor and capture are supported.')
    if set(value) != {'action', 'active_scan', 'pose', 'site_id', 'site_sha256'}:
        raise AgentError('invalid_request', 'Capture requires exactly its documented fields.')
    if value['active_scan'] is not True:
        raise AgentError('active_scan_required', 'Capture requires an explicit active scan request.')
    site_id = value['site_id']
    try:
        site_size = len(site_id.encode('utf-8')) if type(site_id) is str else 0
    except UnicodeEncodeError:
        site_size = 129
    if (type(site_id) is not str or not site_id or site_id != site_id.strip()
            or site_size > 128 or any(ord(c) < 32 or ord(c) == 127 for c in site_id)):
        raise AgentError('invalid_site', 'A bounded nonempty site identifier is required.')
    digest = value['site_sha256']
    if type(digest) is not str or re.fullmatch(r'[0-9a-fA-F]{64}', digest) is None:
        raise AgentError('invalid_site', 'A 64-digit hexadecimal site SHA256 is required.')
    pose = value['pose']
    if (not isinstance(pose, dict) or set(pose) != {'xyz_m', 'source'}
            or pose['source'] not in ('operator', 'external') or type(pose['source']) is not str
            or not isinstance(pose['xyz_m'], list) or len(pose['xyz_m']) != 3):
        raise AgentError('invalid_pose', 'An operator or external XYZ pose in metres is required.')
    coordinates = pose['xyz_m']
    if any(not _finite_coordinate(v) for v in coordinates):
        raise AgentError('invalid_pose', 'Pose coordinates must be three finite non-boolean numbers.')
    return {'action': action, 'active_scan': True, 'site_id': site_id,
        'site_sha256': digest.lower(), 'pose': {'xyz_m': [float(v) for v in coordinates], 'source': pose['source']}}


def _valid_mac(value):
    return (type(value) is str and BSSID.fullmatch(value) is not None
        and value.lower() not in ('00:00:00:00:00:00', 'ff:ff:ff:ff:ff:ff')
        and int(value[:2], 16) % 2 == 0)


def inspect_roles(runner=run_command, sysfs_root=Path('/sys/class/net'), tool_paths=None):
    """Prove the two NIC roles from this kernel, never a factory assumption."""
    paths = tool_paths if tool_paths is not None else globals()['tool_paths']()
    result = {'safe': False, 'interface': INTERFACE, 'role': 'WIFI',
        'management_interface': MANAGEMENT_INTERFACE, 'management_guard': False, 'reasons': []}
    reasons = result['reasons']
    devices = {}
    for name, bus in ((INTERFACE, 'pci'), (MANAGEMENT_INTERFACE, 'sdio')):
        try:
            device = (Path(sysfs_root) / name / 'device').resolve(strict=True)
            subsystem = (device / 'subsystem').resolve(strict=True)
            if not device.is_dir() or not subsystem.is_dir() or subsystem.name != bus:
                reasons.append(name + '_bus_unverified')
            else:
                devices[name] = device
        except (OSError, RuntimeError):
            reasons.append(name + '_device_unavailable')
    if len(devices) == 2 and devices[INTERFACE] == devices[MANAGEMENT_INTERFACE]:
        reasons.append('interfaces_share_device')
    if not paths.get('ip'):
        reasons.append('ip_unavailable')
        return result
    try:
        response = runner([paths['ip'], '-j', 'address', 'show'], timeout_s=5.0, max_stdout_bytes=65536)
        if response.returncode != 0:
            raise ValueError
        addresses = json.loads(response.stdout)
        if not isinstance(addresses, list) or any(not isinstance(row, dict) for row in addresses):
            raise ValueError
        rows = {}
        owners = []
        for row in addresses:
            info = row.get('addr_info')
            if not isinstance(info, list) or any(not isinstance(a, dict) for a in info):
                raise ValueError
            if any(a.get('family') == 'inet' and a.get('local') == MANAGEMENT_IP for a in info):
                owners.append(row.get('ifname'))
            if row.get('ifname') in (INTERFACE, MANAGEMENT_INTERFACE):
                if row['ifname'] in rows:
                    raise ValueError
                rows[row['ifname']] = row
        if set(rows) != {INTERFACE, MANAGEMENT_INTERFACE}:
            raise ValueError
        indices = [rows[n].get('ifindex') for n in (INTERFACE, MANAGEMENT_INTERFACE)]
        macs = [rows[n].get('address') for n in (INTERFACE, MANAGEMENT_INTERFACE)]
        if (any(type(v) is not int or v <= 0 for v in indices) or len(set(indices)) != 2
                or not all(_valid_mac(v) for v in macs) or len({v.lower() for v in macs}) != 2):
            raise ValueError
        result['management_guard'] = owners == [MANAGEMENT_INTERFACE]
        if not result['management_guard']:
            reasons.append('management_address_unverified')
    except (AgentError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        reasons.append('interface_addresses_unverified')
    result['safe'] = not reasons and result['management_guard']
    return result


def _parse_ssid(rendered):
    """Decode iw's byte escapes exactly once, including edge spaces."""
    encoded = bytearray()
    i = 0
    try:
        while i < len(rendered):
            if rendered[i] == '\\':
                escaped = rendered[i:i+4]
                if re.fullmatch(r'\\x[0-9a-fA-F]{2}', escaped) is None:
                    raise ValueError
                encoded.append(int(escaped[2:], 16))
                i += 4
            else:
                encoded.extend(rendered[i].encode('utf-8'))
                i += 1
            if len(encoded) > 32:
                raise ValueError
        value = encoded.decode('utf-8', errors='strict')
        if any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError
        return value
    except (ValueError, UnicodeError):
        raise AgentError('scan_parse_failed', 'The scan SSID is not supported valid UTF-8 text.') from None


def parse_scan(text):
    """Parse genuine iw BSS blocks; preserve reported SSID byte identity.

    iw byte escapes are decoded once, including leading/trailing spaces. SSIDs
    with controls, invalid UTF-8 or more than 32 octets fail explicitly.
    Identical repeated BSS blocks deduplicate; conflicting duplicates fail.
    """
    if type(text) is not str:
        raise AgentError('scan_parse_failed', 'The active scan output is invalid.')
    blocks = []
    current = None
    field_indent = None
    for line in text.splitlines():
        if line.startswith('BSS '):
            match = SCAN_BSS.fullmatch(line)
            if not match or not _valid_mac(match.group(1)):
                raise AgentError('scan_parse_failed', 'The active scan contains an invalid BSSID.')
            current = {'bssid': match.group(1).lower()}
            field_indent = None
            blocks.append(current)
            continue
        if current is None:
            if line.strip():
                raise AgentError('scan_parse_failed', 'The active scan output is not recognized.')
            continue
        field = line.lstrip('\t ')
        indent = line[:len(line) - len(field)]
        for prefix, name in (('freq:', 'frequency_mhz'), ('signal:', 'signal_dbm'), ('SSID:', 'ssid')):
            if not field.startswith(prefix):
                continue
            # Nested information elements (for example nontransmitted MBSSID
            # profiles) may also contain SSID fields. They do not replace the
            # outer BSS report. The first required field fixes its indentation.
            if field_indent is None:
                field_indent = indent
            elif indent != field_indent:
                if indent.startswith(field_indent) and len(indent) > len(field_indent):
                    break
                raise AgentError('scan_parse_failed', 'The active scan field structure is invalid.')
            if name in current:
                raise AgentError('scan_parse_failed', 'The active scan repeats a required field.')
            raw = field[len(prefix):]
            if name == 'ssid':
                # iw prints one formatting space after the colon; every further
                # leading/trailing SSID space belongs to the actual report.
                current[name] = _parse_ssid(raw[1:] if raw.startswith(' ') else raw)
            elif name == 'frequency_mhz':
                if re.fullmatch(r'\s*\d+\s*', raw) is None:
                    raise AgentError('scan_parse_failed', 'The scan frequency is invalid.')
                current[name] = int(raw)
                frequency = current[name]
                if not (2400 <= frequency <= 2500 or 4900 <= frequency <= 5900 or 5925 <= frequency <= 7125):
                    raise AgentError('scan_parse_failed', 'The scan frequency is outside supported Wi-Fi bands.')
            else:
                match = re.fullmatch(r'\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s+dBm\s*', raw)
                if not match:
                    raise AgentError('scan_parse_failed', 'The scan signal is not a dBm report.')
                current[name] = float(match.group(1))
                if not math.isfinite(current[name]) or not -150 <= current[name] <= 0:
                    raise AgentError('scan_parse_failed', 'The scan signal is outside valid dBm bounds.')
            break
    unique = {}
    for block in blocks:
        if set(block) != {'bssid', 'ssid', 'frequency_mhz', 'signal_dbm'}:
            raise AgentError('scan_parse_failed', 'An active-scan BSS lacks a required field.')
        key = block['bssid']
        if key in unique and unique[key] != block:
            raise AgentError('scan_duplicate_conflict', 'The active scan reports conflicting BSSID records.')
        unique[key] = block
    if not unique:
        raise AgentError('scan_empty', 'The active scan returned no usable BSS records.')
    if len(unique) > 512:
        raise AgentError('scan_output_limit', 'The active scan exceeds the supported AP record limit.')
    return [unique[key] for key in sorted(unique)]


def parse_link(text):
    lines = text.splitlines()
    if lines and lines[0].strip() == 'Not connected.':
        return None
    if lines:
        match = re.fullmatch(r'Connected to ((?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}) \(on wlan1\)', lines[0].strip())
        if match and _valid_mac(match.group(1)):
            return match.group(1).lower()
    raise AgentError('link_status_unknown', 'The current client association could not be identified.')


def doctor(runner=run_command, sysfs_root=Path('/sys/class/net'), tool_paths=None):
    paths = tool_paths if tool_paths is not None else globals()['tool_paths']()
    versions = {'python': platform.python_version(), 'kernel': platform.release()}
    for name, flags in (('ip', ['-V']), ('iw', ['--version'])):
        versions[name] = None
        if paths.get(name):
            try:
                response = runner([paths[name]] + flags, timeout_s=3.0, max_stdout_bytes=4096)
                lines = response.stdout.splitlines()
                if response.returncode == 0 and lines:
                    versions[name] = ''.join(c for c in lines[0] if c.isprintable())[:160]
            except AgentError:
                pass
    # Legacy tools live in the login user's home, separately from this agent.
    # Presence is reported without opening or executing either legacy script.
    folder = Path.home()
    effective_uid = os.geteuid()
    return {'ok': True, 'action': 'doctor', 'doctor': {'schema': 1, 'source': 'hardware',
        'read_only': True, 'uid': effective_uid,
        'fresh_scan_requires': 'root or operator-configured noninteractive sudo for exact iw scan argv',
        'privileged_scan_executable': effective_uid == 0 or bool(paths.get('sudo')),
        'scan_privilege_authorized': None,
        'tools': {name: bool(paths.get(name)) for name in ('ip', 'iw', 'sudo')},
        'versions': versions, 'role_check': inspect_roles(runner, sysfs_root, paths),
        'existing_scripts': {name: any((folder / candidate).is_file() for candidate in (name, name + '.py'))
            for name in ('SmartCarCmd', 'SmartCatScan')}}}


def capture(request, runner=run_command, sysfs_root=Path('/sys/class/net'), tool_paths=None):
    request = validate_request(request)
    if request['action'] != 'capture':
        raise AgentError('invalid_request', 'Capture requires the capture action.')
    paths = tool_paths if tool_paths is not None else globals()['tool_paths']()
    if not paths.get('iw'):
        raise AgentError('iw_unavailable', 'The iw tool is required for a fresh active scan.')
    if os.geteuid() == 0:
        scan_command = [paths['iw'], 'dev', INTERFACE, 'scan', 'flush']
    elif paths.get('sudo'):
        scan_command = [paths['sudo'], '-n', paths['iw'], 'dev', INTERFACE, 'scan', 'flush']
    else:
        raise AgentError('scan_privilege_unavailable', 'A fresh scan requires root or configured noninteractive sudo.')
    if not inspect_roles(runner, sysfs_root, paths)['safe']:
        raise AgentError('unsafe_interface_roles', 'Client and management interface roles are not proven safe.')
    started = time.time_ns()
    linked = runner([paths['iw'], 'dev', INTERFACE, 'link'], timeout_s=5.0, max_stdout_bytes=8192)
    current_bssid = None
    if linked.returncode == 0:
        try:
            current_bssid = parse_link(linked.stdout)
        except AgentError:
            pass  # Optional status remains absent; it is never fabricated.
    scan_started = time.time_ns()
    monotonic_started = time.monotonic_ns()
    # Flush the transient BSS cache as part of this explicitly requested scan.
    # A successful ordinary scan dump can otherwise include older observations.
    scanned = runner(scan_command, timeout_s=30.0, max_stdout_bytes=MAX_SCAN_BYTES)
    duration = time.monotonic_ns() - monotonic_started
    scan_completed = time.time_ns()
    if scanned.returncode != 0:
        raise AgentError('active_scan_failed', 'The requested active scan failed; no fresh record was created.')
    aps = parse_scan(scanned.stdout)
    completed = time.time_ns()
    if completed < started or scan_completed < scan_started:
        raise AgentError('clock_changed', 'The wall clock moved backwards during capture.')
    record = {'schema': 1, 'site_sha256': request['site_sha256'], 'site_id': request['site_id'],
        'capture_id': str(uuid.uuid4()), 'pose': request['pose'], 'started_unix_ns': started,
        'completed_unix_ns': completed, 'interface': INTERFACE, 'source': 'hardware',
        'fresh_scan': True, 'scan_cache_flushed': True, 'aps': aps, 'active_scan_cost_incurred': True,
        'scan_started_unix_ns': scan_started, 'scan_completed_unix_ns': scan_completed,
        'scan_duration_ns': duration, 'scan_duration_s': duration / 1e9}
    if current_bssid is not None:
        record['current_bssid'] = current_bssid
    return {'ok': True, 'action': 'capture', 'record': record}


def handle_request(value, runner=run_command, sysfs_root=Path('/sys/class/net'), tool_paths=None):
    request = validate_request(value)
    if request['action'] == 'doctor':
        return doctor(runner, sysfs_root, tool_paths)
    return capture(request, runner, sysfs_root, tool_paths)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AgentError('invalid_json', 'Duplicate JSON object keys are not accepted.')
        result[key] = value
    return result


def main():
    status = 0
    try:
        data = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(data) > MAX_REQUEST_BYTES:
            raise AgentError('request_too_large', 'The JSON request exceeds its size limit.')
        try:
            value = json.loads(data.decode('utf-8'), object_pairs_hook=_unique_object,
                parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()))
        except (UnicodeDecodeError, ValueError, TypeError):
            raise AgentError('invalid_json', 'Exactly one finite JSON request is required.') from None
        response = handle_request(value)
    except AgentError as error:
        status = 1
        response = {'ok': False, 'error': {'code': error.code, 'message': error.message}}
    except Exception:
        status = 1
        response = {'ok': False, 'error': {'code': 'internal_error', 'message': 'The read-only collector could not complete the request.'}}
    sys.stdout.write(json.dumps(response, allow_nan=False, separators=(',', ':')) + '\n')
    return status


if __name__ == '__main__':
    sys.exit(main())
