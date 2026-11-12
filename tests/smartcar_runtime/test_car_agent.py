"""UNIT FIXTURES ONLY: collector contracts, never physical-car validation.

Disposable sysfs trees and an explicit fake ip/iw runner exercise fail-closed
behavior. Successful fixture records stay in memory; these tests do not create
hardware receipts, scan a radio, change a connection, or touch a real car.
The subprocess tests run only harmless local Python programs.
"""
from copy import deepcopy
import ast
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import time
import uuid

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'deployment/smartcar/PI/wj_car_agent.py'
TOOLS = {'ip': '/unit-fixture/ip', 'iw': '/unit-fixture/iw', 'sudo': '/unit-fixture/sudo'}


@pytest.fixture(scope='module')
def agent():
    name = '_wj_readonly_car_agent_unit_fixture'
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    before = sys.modules.get(name)
    sys.modules[name] = module  # dataclasses resolves the defining module.
    spec.loader.exec_module(module)
    yield module
    if before is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = before


@pytest.fixture(autouse=True)
def fixture_root_uid(agent, monkeypatch):
    """Explicit software privilege state; this grants no actual privilege."""
    monkeypatch.setattr(agent.os, 'geteuid', lambda: 0)


@pytest.fixture
def sysfs(tmp_path):
    """An invented kernel-layout fixture, with no hardware assertion."""
    root = tmp_path / 'unit-sys/class/net'
    root.mkdir(parents=True)
    buses = {}
    for bus in ('pci', 'sdio', 'usb'):
        buses[bus] = tmp_path / 'unit-sys/bus' / bus
        buses[bus].mkdir(parents=True)
    for interface, bus in (('wlan0', 'sdio'), ('wlan1', 'pci')):
        device = tmp_path / 'unit-sys/devices' / ('fixture-' + interface)
        device.mkdir(parents=True)
        (device / 'subsystem').symlink_to(buses[bus], target_is_directory=True)
        (root / interface).mkdir()
        (root / interface / 'device').symlink_to(device, target_is_directory=True)
    return root


def addresses():
    return [
        {'ifindex': 2, 'ifname': 'wlan0', 'address': '02:00:00:00:00:10',
         'addr_info': [{'family': 'inet', 'local': '192.168.1.11', 'prefixlen': 24}]},
        {'ifindex': 3, 'ifname': 'wlan1', 'address': '02:00:00:00:00:11',
         'addr_info': [{'family': 'inet', 'local': '10.20.30.40', 'prefixlen': 24}]},
    ]


def request():
    return {'action': 'capture', 'active_scan': True,
        'pose': {'xyz_m': [1, -2.5, 0], 'source': 'operator'},
        'site_id': 'UNIT_FIXTURE_ONLY', 'site_sha256': 'ab' * 32}


def scan_text(*, bssid='AA:BB:CC:DD:EE:10', signal='-52.25', frequency='5180', ssid='Unit fixture'):
    return (f'BSS {bssid}(on wlan1) -- associated\n'
        '\tlast seen: 0 ms ago\n'
        f'\tfreq: {frequency}\n\tcapability: ESS Privacy (0x0011)\n'
        f'\tsignal: {signal} dBm\n\tSSID: {ssid}\n'
        '\tSupported rates: 6.0* 9.0 12.0*\n')


class Runner:
    """Explicit fixture transport: an unexpected command always fails."""
    def __init__(self, module, rows=None, scan=None, link=None):
        self.module = module
        self.rows = addresses() if rows is None else rows
        self.scan = scan_text() if scan is None else scan
        self.link = 'Connected to AA:BB:CC:DD:EE:10 (on wlan1)\n\tsignal: -52 dBm\n' if link is None else link
        self.scan_code = 0
        self.link_code = 0
        self.calls = []

    def __call__(self, argv, **kwargs):
        command = tuple(argv)
        self.calls.append((command, kwargs))
        if command == (TOOLS['ip'], '-j', 'address', 'show'):
            return self.module.CommandResult(0, json.dumps(self.rows))
        if command == (TOOLS['ip'], '-V'):
            return self.module.CommandResult(0, 'ip utility UNIT_FIXTURE\n')
        if command == (TOOLS['iw'], '--version'):
            return self.module.CommandResult(0, 'iw version UNIT_FIXTURE\n')
        if command == (TOOLS['iw'], 'dev', 'wlan1', 'link'):
            return self.module.CommandResult(self.link_code, self.link)
        if command == (TOOLS['iw'], 'dev', 'wlan1', 'scan', 'flush'):
            return self.module.CommandResult(self.scan_code, self.scan)
        if command == (TOOLS['sudo'], '-n', TOOLS['iw'], 'dev', 'wlan1', 'scan', 'flush'):
            return self.module.CommandResult(self.scan_code, self.scan)
        raise AssertionError('Unexpected command in UNIT_FIXTURE: ' + repr(command))


def call_main(module, monkeypatch, payload):
    incoming = io.TextIOWrapper(io.BytesIO(payload), encoding='utf-8')
    outgoing = io.StringIO()
    monkeypatch.setattr(module.sys, 'stdin', incoming)
    monkeypatch.setattr(module.sys, 'stdout', outgoing)
    code = module.main()
    lines = outgoing.getvalue().splitlines()
    assert len(lines) == 1
    return code, json.loads(lines[0]), lines[0]


def test_single_file_uses_standard_library_and_python39_syntax_only():
    text = SOURCE.read_text()
    tree = ast.parse(text, feature_version=(3, 9))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not node.level
            imports.add(node.module.split('.')[0])
    assert imports <= sys.stdlib_module_names
    assert not imports & {'tomllib', 'numpy', 'torch', 'wirelessjev', 'paramiko'}


@pytest.mark.parametrize('source', ['operator', 'external'])
def test_capture_request_normalizes_metadata_without_mutating_input(agent, source):
    value = request(); value['pose']['source'] = source; value['site_sha256'] = 'AB' * 32
    before = deepcopy(value)
    normalized = agent.validate_request(value)
    assert normalized['site_sha256'] == 'ab' * 32
    assert normalized['pose'] == {'xyz_m': [1., -2.5, 0.], 'source': source}
    assert value == before
    assert agent.validate_request({'action': 'doctor'}) == {'action': 'doctor'}


@pytest.mark.parametrize('change', [
    lambda r: r.pop('active_scan'),
    lambda r: r.update(active_scan=False),
    lambda r: r.update(active_scan=1),
    lambda r: r.update(active_scan='true'),
    lambda r: r.update(interface='wlan0'),
    lambda r: r.update(action='connect'),
    lambda r: r.update(action='capture; reboot'),
    lambda r: r.update(site_id=''),
    lambda r: r.update(site_id=' leading'),
    lambda r: r.update(site_id='line\nbreak'),
    lambda r: r.update(site_id='x' * 129),
    lambda r: r.update(site_id='\ud800'),
    lambda r: r.update(site_id=3),
    lambda r: r.update(site_sha256='a' * 63),
    lambda r: r.update(site_sha256='z' * 64),
    lambda r: r['pose'].update(source='inferred'),
    lambda r: r['pose'].update(source=1),
    lambda r: r['pose'].update(extra='unsupported'),
    lambda r: r['pose'].update(xyz_m=[1, 2]),
    lambda r: r['pose'].update(xyz_m=[1, 2, 3, 4]),
    lambda r: r['pose'].update(xyz_m=[True, 2, 3]),
    lambda r: r['pose'].update(xyz_m=[1, float('nan'), 3]),
    lambda r: r['pose'].update(xyz_m=[1, float('inf'), 3]),
    lambda r: r['pose'].update(xyz_m=[1, '2', 3]),
    lambda r: r['pose'].update(xyz_m=[1, None, 3]),
    lambda r: r['pose'].update(xyz_m=[10 ** 1000, 2, 3]),
])
def test_capture_request_refuses_unsupported_or_ambiguous_fields(agent, change):
    value = request(); change(value)
    with pytest.raises(agent.AgentError):
        agent.validate_request(value)


@pytest.mark.parametrize('value', [None, [], 'doctor', {'action': True},
    {'action': 'doctor', 'active_scan': True}, {'action': 'doctor', 'interface': 'wlan0'}])
def test_only_the_two_exact_request_schemas_are_accepted(agent, value):
    with pytest.raises(agent.AgentError):
        agent.validate_request(value)


def test_tools_are_resolved_only_from_the_fixed_system_search_path(agent, monkeypatch):
    calls = []
    def which(name, *, path):
        calls.append((name, path)); return TOOLS[name]
    monkeypatch.setattr(agent.shutil, 'which', which)
    assert agent.tool_paths() == TOOLS
    assert calls == [('ip', agent.SAFE_PATH), ('iw', agent.SAFE_PATH), ('sudo', agent.SAFE_PATH)]


def test_role_guard_proves_pci_sdio_and_unique_management_owner(agent, sysfs):
    runner = Runner(agent)
    result = agent.inspect_roles(runner=runner, sysfs_root=sysfs, tool_paths=TOOLS)
    assert result == {'safe': True, 'interface': 'wlan1', 'role': 'WIFI',
        'management_interface': 'wlan0', 'management_guard': True, 'reasons': []}
    assert [c[0] for c in runner.calls] == [(TOOLS['ip'], '-j', 'address', 'show')]
    assert runner.calls[0][1]['timeout_s'] == 5.
    assert runner.calls[0][1]['max_stdout_bytes'] <= 65536


@pytest.mark.parametrize('interface,bus', [('wlan1', 'usb'), ('wlan1', 'sdio'), ('wlan0', 'pci')])
def test_unproven_physical_radio_bus_is_unsafe(agent, sysfs, interface, bus):
    subsystem = (sysfs / interface / 'device').resolve() / 'subsystem'
    subsystem.unlink()
    subsystem.symlink_to(sysfs.parents[1] / 'bus' / bus, target_is_directory=True)
    result = agent.inspect_roles(Runner(agent), sysfs, TOOLS)
    assert result['safe'] is False and result['reasons']


def test_missing_or_shared_kernel_device_cannot_pass_role_guard(agent, sysfs):
    target = (sysfs / 'wlan0/device').resolve()
    (sysfs / 'wlan1/device').unlink()
    result = agent.inspect_roles(Runner(agent), sysfs, TOOLS)
    assert not result['safe']
    (sysfs / 'wlan1/device').symlink_to(target, target_is_directory=True)
    result = agent.inspect_roles(Runner(agent), sysfs, TOOLS)
    assert not result['safe']


@pytest.mark.parametrize('change', [
    lambda rows: rows[0].update(addr_info=[]),
    lambda rows: rows[1]['addr_info'].append({'family': 'inet', 'local': '192.168.1.11'}),
    lambda rows: rows[0].update(ifindex=True),
    lambda rows: rows[1].update(ifindex=2),
    lambda rows: rows[1].update(address=rows[0]['address']),
    lambda rows: rows[1].update(address='01:00:00:00:00:11'),
    lambda rows: rows[1].update(address='00:00:00:00:00:00'),
    lambda rows: rows[1].update(address='not-a-mac'),
    lambda rows: rows.append(deepcopy(rows[0])),
    lambda rows: rows.append({'ifname': 'eth0', 'addr_info': [{'family': 'inet', 'local': '192.168.1.11'}]}),
    lambda rows: rows[1].update(addr_info=None),
    lambda rows: rows.pop(),
])
def test_ambiguous_addresses_or_interface_identity_fail_closed(agent, sysfs, change):
    rows = addresses(); change(rows)
    result = agent.inspect_roles(Runner(agent, rows=rows), sysfs, TOOLS)
    assert result['safe'] is False and result['reasons']


def test_address_probe_failures_do_not_echo_raw_command_text(agent, sysfs):
    def broken(_argv, **_kwargs):
        return agent.CommandResult(1, 'PRIVATE_FIXTURE_CONTENT')
    result = agent.inspect_roles(broken, sysfs, TOOLS)
    assert not result['safe'] and result['management_guard'] is False
    assert 'PRIVATE_FIXTURE_CONTENT' not in json.dumps(result)
    result = agent.inspect_roles(Runner(agent), sysfs, {'iw': TOOLS['iw'], 'ip': None})
    assert not result['safe'] and 'ip_unavailable' in result['reasons']


def test_doctor_reports_unsafe_roles_without_scanning_or_reading_config(agent, sysfs, tmp_path, monkeypatch):
    (tmp_path / 'SmartCarCmd.py').write_text('PRIVATE_FIXTURE_CONTENT')
    (tmp_path / 'WiFi.json').write_text('PRIVATE_FIXTURE_PASSWORD')
    monkeypatch.setattr(agent.Path, 'home', classmethod(lambda cls: tmp_path))
    runner = Runner(agent, rows=[])
    result = agent.handle_request({'action': 'doctor'}, runner=runner, sysfs_root=sysfs, tool_paths=TOOLS)
    assert result['ok'] is True and result['action'] == 'doctor'
    info = result['doctor']
    assert info['read_only'] is True and info['tools'] == {'ip': True, 'iw': True, 'sudo': True}
    assert info['role_check']['safe'] is False
    assert info['existing_scripts']['SmartCarCmd'] is True
    assert not any('scan' in command for command, _ in runner.calls)
    assert all(command in ((TOOLS['ip'], '-V'), (TOOLS['iw'], '--version'),
        (TOOLS['ip'], '-j', 'address', 'show')) for command, _ in runner.calls)
    assert 'PRIVATE_FIXTURE' not in json.dumps({k:v for k,v in info.items() if k != 'versions'})
    assert 'PASSWORD' not in json.dumps(result)


@pytest.mark.parametrize('uid,sudo,executable', [
    (0, None, True), (1000, TOOLS['sudo'], True), (1000, None, False)])
def test_doctor_reports_scan_privilege_without_attempting_or_claiming_authorization(
        agent, sysfs, monkeypatch, uid, sudo, executable):
    monkeypatch.setattr(agent.os, 'geteuid', lambda: uid)
    runner = Runner(agent)
    paths = dict(TOOLS, sudo=sudo)
    info = agent.doctor(runner=runner, sysfs_root=sysfs, tool_paths=paths)['doctor']
    assert info['uid'] == uid
    assert info['privileged_scan_executable'] is executable
    assert info['scan_privilege_authorized'] is None
    assert info['fresh_scan_requires'] == 'root or operator-configured noninteractive sudo for exact iw scan argv'
    assert info['tools']['sudo'] is bool(sudo)
    assert all(command[0] != TOOLS['sudo'] and 'scan' not in command for command, _ in runner.calls)


def test_scan_keeps_three_wifi_bands_and_exact_ssid_content(agent):
    text = (scan_text(ssid='  Unit fixture  ')
        + scan_text(bssid='AA:BB:CC:DD:EE:12', frequency='2412', signal='-77.5', ssid='')
        + scan_text(bssid='AA:BB:CC:DD:EE:14', frequency='5955', signal='-61', ssid=r'caf\xc3\xa9'))
    records = agent.parse_scan(text)
    assert len(records) == 3
    assert records[0] == {'bssid': 'aa:bb:cc:dd:ee:10', 'ssid': '  Unit fixture  ',
        'frequency_mhz': 5180, 'signal_dbm': -52.25}
    assert records[1]['ssid'] == '' and records[1]['frequency_mhz'] == 2412
    assert records[2]['ssid'] == 'café' and records[2]['frequency_mhz'] == 5955


@pytest.mark.parametrize('rendered,decoded', [
    (r'\x20Warehouse\x20', ' Warehouse '),
    (r'\xc3\xa9tage', 'étage'),
    (r'one\x5cx20pass', r'one\x20pass'),
    (r'literal\x5cname', 'literal\\name'),
    ('x' * 32, 'x' * 32),
    (r'\xc3\xa9' * 16, 'é' * 16),
])
def test_iw_ssid_bytes_decode_once_as_strict_utf8_with_a_32_byte_limit(agent, rendered, decoded):
    assert agent.parse_scan(scan_text(ssid=rendered))[0]['ssid'] == decoded


@pytest.mark.parametrize('rendered', [
    r'bad\x00null', r'bad\x0aline', r'bad\x1bescape', r'bad\x7fdelete',
    r'bad\xffbyte', r'bad\xc0\xafutf8', r'bad\xc3',
    r'bad\x2Ghex', r'bad\qescape', 'bad\\', 'x' * 33, r'\xc3\xa9' * 17,
])
def test_invalid_or_ambiguous_ssid_bytes_fail_without_rewriting_the_network_name(agent, rendered):
    with pytest.raises(agent.AgentError):
        agent.parse_scan(scan_text(ssid=rendered))


@pytest.mark.parametrize('frequency', [2400, 2500, 4900, 5900, 5925, 7125])
def test_scan_accepts_only_documented_wifi_band_endpoints(agent, frequency):
    assert agent.parse_scan(scan_text(frequency=str(frequency)))[0]['frequency_mhz'] == frequency


@pytest.mark.parametrize('frequency', [2399, 2501, 4899, 5901, 5924, 7126])
def test_scan_rejects_frequency_gaps_and_out_of_band_reports(agent, frequency):
    with pytest.raises(agent.AgentError):
        agent.parse_scan(scan_text(frequency=str(frequency)))


@pytest.mark.parametrize('signal', ['-150', '-0', '0'])
def test_scan_accepts_documented_signal_endpoints(agent, signal):
    assert agent.parse_scan(scan_text(signal=signal))[0]['signal_dbm'] == float(signal)


def test_scan_limits_unique_aps_to_the_pc_capture_contract(agent):
    def bss(number):
        return scan_text(bssid=f'02:00:00:00:{number >> 8:02x}:{number & 255:02x}')
    limit = ''.join(bss(i) for i in range(512))
    assert len(agent.parse_scan(limit)) == 512
    with pytest.raises(agent.AgentError):
        agent.parse_scan(limit + bss(512))


def test_identical_scan_duplicates_deduplicate_but_conflicts_fail(agent):
    sample = scan_text()
    assert agent.parse_scan(sample + sample) == agent.parse_scan(sample)
    with pytest.raises(agent.AgentError, match='conflicting'):
        agent.parse_scan(sample + scan_text(signal='-53'))


def test_nested_mbssid_profile_cannot_replace_the_outer_bss_measurement(agent):
    text = scan_text(ssid='Outer fixture') + (
        '\tMultiple BSSID:\n'
        '\t\tNontransmitted BSSID profile:\n'
        '\t\t\tSSID: Nested fixture\n'
        '\t\t\tfreq: 2412\n'
        '\t\t\tsignal: -99 dBm\n')
    assert agent.parse_scan(text) == [{'bssid': 'aa:bb:cc:dd:ee:10',
        'ssid': 'Outer fixture', 'frequency_mhz': 5180, 'signal_dbm': -52.25}]


def test_scan_required_fields_at_inconsistent_indentation_fail_closed(agent):
    text = scan_text().replace('\tsignal: -52.25 dBm', '  signal: -52.25 dBm')
    with pytest.raises(agent.AgentError):
        agent.parse_scan(text)


@pytest.mark.parametrize('text', [
    '', 'arbitrary command warning\n',
    scan_text(bssid='01:BB:CC:DD:EE:10'),
    scan_text(bssid='00:00:00:00:00:00'),
    scan_text().replace('\tfreq: 5180\n', ''),
    scan_text().replace('\tsignal: -52.25 dBm\n', ''),
    scan_text().replace('\tSSID: Unit fixture\n', ''),
    scan_text(frequency='not-a-frequency'),
    scan_text(frequency='5180.0'),
    scan_text(frequency='99999'),
    scan_text(signal='nan'), scan_text(signal='inf'), scan_text(signal='1'), scan_text(signal='-150.1'),
    scan_text().replace(' dBm', ' percent'),
    scan_text(ssid='actual\tcontrol'),
    scan_text() + '\tfreq: 5180\n',
])
def test_malformed_or_unusable_scan_never_becomes_a_fresh_record(agent, text):
    with pytest.raises(agent.AgentError):
        agent.parse_scan(text)


def test_link_status_is_interface_bound_and_unknown_is_not_fabricated(agent):
    assert agent.parse_link('Not connected.\n') is None
    assert agent.parse_link('Connected to AA:BB:CC:DD:EE:10 (on wlan1)\n\tsignal: -50 dBm\n') == 'aa:bb:cc:dd:ee:10'
    for text in ('', 'Connected to AA:BB:CC:DD:EE:10 (on wlan0)', 'unrecognized PRIVATE_FIXTURE'):
        with pytest.raises(agent.AgentError):
            agent.parse_link(text)


def test_capture_executes_only_fixed_readonly_commands_and_explicit_cache_flush(agent, sysfs):
    runner = Runner(agent)
    reply = agent.handle_request(request(), runner=runner, sysfs_root=sysfs, tool_paths=TOOLS)
    assert reply['ok'] is True and reply['action'] == 'capture'
    record = reply['record']
    assert record['schema'] == 1 and record['source'] == 'hardware'
    assert record['interface'] == 'wlan1' and record['fresh_scan'] is True
    assert record['active_scan_cost_incurred'] is True and record['scan_cache_flushed'] is True
    assert record['site_id'] == request()['site_id'] and record['site_sha256'] == 'ab' * 32
    assert record['pose'] == {'xyz_m': [1., -2.5, 0.], 'source': 'operator'}
    assert uuid.UUID(record['capture_id']).version == 4
    assert record['current_bssid'] == 'aa:bb:cc:dd:ee:10'
    assert 0 < record['started_unix_ns'] <= record['scan_started_unix_ns'] <= record['scan_completed_unix_ns'] <= record['completed_unix_ns']
    assert record['scan_duration_ns'] >= 0 and math.isfinite(record['scan_duration_s'])
    assert record['scan_duration_s'] == record['scan_duration_ns'] / 1e9
    assert record['aps'] == agent.parse_scan(scan_text())
    assert [command for command, _ in runner.calls] == [
        (TOOLS['ip'], '-j', 'address', 'show'),
        (TOOLS['iw'], 'dev', 'wlan1', 'link'),
        (TOOLS['iw'], 'dev', 'wlan1', 'scan', 'flush')]
    assert all(0 < params['timeout_s'] <= 30 and 0 < params['max_stdout_bytes'] <= agent.MAX_SCAN_BYTES
               for _command, params in runner.calls)


@pytest.mark.parametrize('uid', [0, 1000])
def test_unsafe_roles_prevent_all_scan_and_link_commands(agent, sysfs, monkeypatch, uid):
    monkeypatch.setattr(agent.os, 'geteuid', lambda: uid)
    runner = Runner(agent, rows=[])
    with pytest.raises(agent.AgentError) as caught:
        agent.capture(request(), runner=runner, sysfs_root=sysfs, tool_paths=TOOLS)
    assert caught.value.code == 'unsafe_interface_roles'
    assert [command for command, _ in runner.calls] == [(TOOLS['ip'], '-j', 'address', 'show')]


def test_nonroot_capture_privileges_only_the_exact_fresh_scan_command(agent, sysfs, monkeypatch):
    monkeypatch.setattr(agent.os, 'geteuid', lambda: 1000)
    runner = Runner(agent)
    result = agent.capture(request(), runner=runner, sysfs_root=sysfs, tool_paths=TOOLS)
    assert result['ok'] is True and result['record']['scan_cache_flushed'] is True
    assert [command for command, _ in runner.calls] == [
        (TOOLS['ip'], '-j', 'address', 'show'),
        (TOOLS['iw'], 'dev', 'wlan1', 'link'),
        (TOOLS['sudo'], '-n', TOOLS['iw'], 'dev', 'wlan1', 'scan', 'flush')]


def test_nonroot_capture_without_sudo_fails_before_any_command(agent, sysfs, monkeypatch):
    monkeypatch.setattr(agent.os, 'geteuid', lambda: 1000)
    runner = Runner(agent)
    with pytest.raises(agent.AgentError) as caught:
        agent.capture(request(), runner=runner, sysfs_root=sysfs, tool_paths=dict(TOOLS, sudo=None))
    assert caught.value.code == 'scan_privilege_unavailable' and runner.calls == []


@pytest.mark.parametrize('uid', [0, 1000])
def test_scan_permission_refusal_has_one_attempt_and_never_retries(agent, sysfs, monkeypatch, uid):
    monkeypatch.setattr(agent.os, 'geteuid', lambda: uid)
    runner = Runner(agent, scan='PRIVATE_FIXTURE_PERMISSION_DENIED')
    runner.scan_code = 1
    with pytest.raises(agent.AgentError) as caught:
        agent.capture(request(), runner=runner, sysfs_root=sysfs, tool_paths=TOOLS)
    assert caught.value.code == 'active_scan_failed'
    assert 'PRIVATE_FIXTURE_PERMISSION_DENIED' not in str(caught.value)
    scans = [command for command, _ in runner.calls if 'scan' in command]
    expected = (TOOLS['iw'], 'dev', 'wlan1', 'scan', 'flush')
    if uid != 0:
        expected = (TOOLS['sudo'], '-n') + expected
    assert scans == [expected] and len(runner.calls) == 3


def test_missing_iw_stops_capture_before_any_probe(agent, sysfs):
    runner = Runner(agent)
    with pytest.raises(agent.AgentError) as caught:
        agent.capture(request(), runner=runner, sysfs_root=sysfs, tool_paths={'ip': TOOLS['ip'], 'iw': None})
    assert caught.value.code == 'iw_unavailable' and runner.calls == []


def test_capture_helper_refuses_doctor_action_before_any_command(agent, sysfs):
    runner = Runner(agent)
    with pytest.raises(agent.AgentError) as caught:
        agent.capture({'action': 'doctor'}, runner=runner, sysfs_root=sysfs, tool_paths=TOOLS)
    assert caught.value.code == 'invalid_request' and runner.calls == []


@pytest.mark.parametrize('status,text', [(1, 'PRIVATE_FIXTURE_ERROR'), (0, '')])
def test_failed_or_empty_active_scan_has_no_partial_success_record(agent, sysfs, status, text):
    runner = Runner(agent, scan=text); runner.scan_code = status
    with pytest.raises(agent.AgentError) as caught:
        agent.capture(request(), runner=runner, sysfs_root=sysfs, tool_paths=TOOLS)
    assert 'PRIVATE_FIXTURE_ERROR' not in str(caught.value)


@pytest.mark.parametrize('link_code,link_text', [(1, 'PRIVATE_LINK_FAILURE'), (0, 'unknown status'), (0, 'Not connected.\n')])
def test_optional_association_status_remains_absent_when_unknown(agent, sysfs, link_code, link_text):
    runner = Runner(agent, link=link_text); runner.link_code = link_code
    record = agent.capture(request(), runner=runner, sysfs_root=sysfs, tool_paths=TOOLS)['record']
    assert 'current_bssid' not in record and record['aps']


def test_backwards_wall_clock_prevents_capture_success(agent, sysfs, monkeypatch):
    clock = iter([100, 110, 109, 120])
    monkeypatch.setattr(agent.time, 'time_ns', lambda: next(clock))
    with pytest.raises(agent.AgentError) as caught:
        agent.capture(request(), runner=Runner(agent), sysfs_root=sysfs, tool_paths=TOOLS)
    assert caught.value.code == 'clock_changed'


@pytest.mark.parametrize('payload', [b'{', b'{} {}', b'{"action":"doctor","action":"doctor"}',
    b'{"action":"doctor","extra":NaN}', b'{"action":"doctor","extra":Infinity}', b'\xff', b' ' * 8193])
def test_main_rejects_bad_json_before_any_handler_or_hardware_probe(agent, monkeypatch, payload):
    called = []
    def forbidden(value):
        called.append(value)
        raise AssertionError('Invalid JSON must not reach a request handler')
    monkeypatch.setattr(agent, 'handle_request', forbidden)
    code, result, raw = call_main(agent, monkeypatch, payload)
    assert code != 0 and result['ok'] is False and called == []
    assert set(result) == {'ok', 'error'} and set(result['error']) == {'code', 'message'}
    assert 'Traceback' not in raw


def test_main_sanitizes_unexpected_internal_exceptions(agent, monkeypatch):
    def broken(_value):
        raise RuntimeError('PRIVATE_FIXTURE_PASSWORD')
    monkeypatch.setattr(agent, 'handle_request', broken)
    code, result, raw = call_main(agent, monkeypatch, b'{"action":"doctor"}')
    assert code != 0 and result == {'ok': False, 'error': {'code': 'internal_error',
        'message': 'The read-only collector could not complete the request.'}}
    assert 'PRIVATE_FIXTURE_PASSWORD' not in raw and 'Traceback' not in raw


def test_run_command_captures_stdout_and_returncode_but_never_stderr(agent):
    program = 'import sys;sys.stdout.write("UNIT_FIXTURE_OK");sys.stderr.write("PRIVATE_FIXTURE_STDERR");sys.exit(7)'
    result = agent.run_command([sys.executable, '-c', program], timeout_s=3., max_stdout_bytes=1024)
    assert result.returncode == 7 and result.stdout == 'UNIT_FIXTURE_OK'
    assert 'PRIVATE_FIXTURE_STDERR' not in repr(result)


@pytest.mark.parametrize('stream', ['stdout', 'stderr'])
def test_run_command_bounds_both_pipes_without_returning_private_output(agent, stream):
    program = f'import sys;sys.{stream}.write("X"*1000000)'
    with pytest.raises(agent.AgentError) as caught:
        agent.run_command([sys.executable, '-c', program], timeout_s=3., max_stdout_bytes=1024)
    assert caught.value.code == 'command_output_limit' and len(str(caught.value)) < 200


def test_run_command_timeout_does_not_wait_for_a_hung_child(agent):
    started = time.monotonic()
    with pytest.raises(agent.AgentError) as caught:
        agent.run_command([sys.executable, '-c', 'import time;time.sleep(10)'], timeout_s=.05, max_stdout_bytes=1024)
    assert caught.value.code == 'command_timeout' and time.monotonic() - started < 3.


def test_run_command_refuses_invalid_utf8_without_echo(agent):
    with pytest.raises(agent.AgentError) as caught:
        agent.run_command([sys.executable, '-c', 'import sys;sys.stdout.buffer.write(bytes([255]))'], timeout_s=3., max_stdout_bytes=1024)
    assert caught.value.code == 'command_output_invalid'
