"""Ghost robots: real Wi-Fi traffic shaped like a fleet, from one laptop.

With one real SmartCar on site, the other nine robots are emulated as load:
this process sends UDP shaped like each ghost's control, telemetry, video and
map-upload traffic through the same access points, so the real car competes
for real airtime.

    python -m wirelessjev.hil.ghosts send --target 192.168.50.1 --seconds 120
    python -m wirelessjev.hil.ghosts sink --port 47000      # optional receiver

Limits, stated plainly: all ghosts share this laptop's one Wi-Fi station, so
the access point sees one busy client, not nine contending ones; ghosts do
not move. Targets must be private (RFC 1918) or link-local addresses, the
total rate is capped, and every run has a fixed duration.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import ipaddress
import json
import socket
import threading
import time

MAX_TOTAL_MBPS = 80.0
MAX_SECONDS = 900
PAYLOAD = 1200


@dataclass(frozen=True)
class Ghost:
    name: str
    control_hz: float = 20.0       # 600-byte control packets
    telemetry_mbps: float = 0.5
    video_mbps: float = 0.0
    bulk_mbps: float = 0.0

    @property
    def mbps(self) -> float:
        return self.control_hz * 600 * 8 / 1e6 + self.telemetry_mbps + self.video_mbps + self.bulk_mbps


def fleet(n: int = 9, video: int = 1, bulk: int = 4) -> list[Ghost]:
    """Same mix as the warehouse twin: a few video robots, about 45% uploading."""
    out = []
    for i in range(n):
        out.append(Ghost(f"ghost-{i + 1:02d}", video_mbps=5.0 if i < video else 0.0,
                         bulk_mbps=10.0 if video <= i < video + bulk else 0.0))
    return out


def check_target(target: str) -> str:
    from ..carlink import MANAGEMENT_HOST
    ip = ipaddress.ip_address(target)
    if not (ip.is_private or ip.is_link_local) or ip.is_loopback or ip.is_multicast:
        raise ValueError("target must be a private or link-local unicast address on the test network")
    if str(ip) == MANAGEMENT_HOST:
        raise ValueError("target is the car's management address; ghost traffic must never go over the hotspot")
    return str(ip)


def _stream(sock, addr, mbps: float, size: int, stop: threading.Event, counter: list, tag: bytes):
    if mbps <= 0:
        return
    interval = size * 8 / (mbps * 1e6)
    payload = tag + b"\0" * (size - len(tag))
    nxt = time.perf_counter()
    while not stop.is_set():
        now = time.perf_counter()
        if now < nxt:
            time.sleep(min(nxt - now, 0.01))
            continue
        try:
            sock.sendto(payload, addr)
            counter[0] += size
        except OSError:
            counter[1] += 1
        nxt += interval
        if now - nxt > 0.5:            # fell behind: do not burst to catch up
            nxt = now


def send(target: str, port: int, ghosts: list[Ghost], seconds: int, *, sleep=time.sleep) -> dict:
    target = check_target(target)
    if not 1 <= seconds <= MAX_SECONDS:
        raise ValueError("seconds must be 1-900")
    total = sum(g.mbps for g in ghosts)
    if total > MAX_TOTAL_MBPS:
        raise ValueError(f"requested {total:.1f} Mb/s exceeds the {MAX_TOTAL_MBPS:.0f} Mb/s cap")
    stop = threading.Event()
    stats, threads, socks = {}, [], []
    for g in ghosts:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        socks.append(sock)
        stats[g.name] = {"offered_mbps": round(g.mbps, 3), "bytes": [0, 0]}
        tag = g.name.encode()
        for mbps, size in ((g.control_hz * 600 * 8 / 1e6, 600), (g.telemetry_mbps, 200),
                           (g.video_mbps, PAYLOAD), (g.bulk_mbps, PAYLOAD)):
            t = threading.Thread(target=_stream, daemon=True,
                                 args=(sock, (target, port), mbps, size, stop, stats[g.name]["bytes"], tag))
            threads.append(t)
            t.start()
    started = time.time()
    try:
        sleep(seconds)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=1.0)
        for s in socks:
            s.close()
    elapsed = time.time() - started
    return {"target": target, "port": port, "seconds": round(elapsed, 2), "started_unix": started,
            "offered_mbps": round(total, 2),
            "ghosts": {k: {"offered_mbps": v["offered_mbps"], "sent_mbps": round(v["bytes"][0] * 8 / 1e6 / max(elapsed, 1e-9), 3),
                           "send_errors": v["bytes"][1]} for k, v in stats.items()}}


def sink(port: int, seconds: int) -> dict:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", port))
    sock.settimeout(0.5)
    got, end = {}, time.time() + seconds
    while time.time() < end:
        try:
            data, _ = sock.recvfrom(2048)
        except socket.timeout:
            continue
        name = data[:8].decode("ascii", errors="replace")
        got[name] = got.get(name, 0) + len(data)
    sock.close()
    return {"port": port, "seconds": seconds,
            "received_mbps": {k: round(v * 8 / 1e6 / seconds, 3) for k, v in sorted(got.items())}}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("send")
    p.add_argument("--target", required=True, help="private IP on the test network (e.g. the AP or a sink laptop)")
    p.add_argument("--port", type=int, default=47000)
    p.add_argument("--ghosts", type=int, default=9)
    p.add_argument("--video", type=int, default=1)
    p.add_argument("--bulk", type=int, default=4)
    p.add_argument("--seconds", type=int, default=120)
    p = sub.add_parser("sink")
    p.add_argument("--port", type=int, default=47000)
    p.add_argument("--seconds", type=int, default=120)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "send":
        out = send(args.target, args.port, fleet(args.ghosts, args.video, args.bulk), args.seconds)
    else:
        out = sink(args.port, args.seconds)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
