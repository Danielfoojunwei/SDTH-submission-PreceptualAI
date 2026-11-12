"""Real local subprocess I/O checks; no SSH server or hardware is simulated."""
import sys

import pytest

from wirelessjev.smartcar_runtime.workflow import SSHTransport


def child(code):
    transport = object.__new__(SSHTransport)
    transport.argv = [sys.executable, "-c", code]
    return transport


def test_transport_streams_complete_local_json_reply():
    transport = child("import sys,json; r=json.load(sys.stdin); print(json.dumps({'ok':True,'echo':r}))")
    assert transport("agent", {"action": "software-pipe-test"}, 3)["echo"] == {"action": "software-pipe-test"}


def test_transport_preserves_explicit_refusal_on_nonzero_exit():
    transport = child("import sys;sys.stdin.read();print('{\"ok\":false}');sys.exit(1)")
    assert transport("agent", {}, 3) == {"ok": False}


def test_transport_rejects_truncated_json():
    with pytest.raises(RuntimeError, match="complete JSON"):
        child("import sys;sys.stdin.read();print('{')")("agent", {}, 3)


@pytest.mark.parametrize("channel,limit", [("stdout", 2*1024*1024), ("stderr", 65536)])
def test_transport_caps_streams_while_reading(channel, limit):
    transport = child(f"import sys;sys.stdin.read();sys.{channel}.write('x'*{limit+65536});sys.{channel}.flush()")
    with pytest.raises(ValueError, match="limit"):
        transport("agent", {}, 3)


def test_transport_times_out_without_retry(tmp_path):
    marker = tmp_path / "calls"
    transport = child(f"import pathlib,time;pathlib.Path({str(marker)!r}).write_text('one');time.sleep(20)")
    with pytest.raises(TimeoutError):
        transport("agent", {}, .1)
    assert marker.read_text() == "one"
