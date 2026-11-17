"""Oversight: one line per action the platform takes or the operator makes, appended to
platform_audit.jsonl in the log folder. Passwords never pass through the platform, and the
log refuses any field that looks like one."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time

_LOCK = threading.Lock()
FORBIDDEN = ("password", "passphrase", "psk", "secret", "token", "api_key")


class Audit:
    def __init__(self, path: Path):
        self.path = path

    def write(self, action: str, **detail) -> dict:
        for k in detail:
            if any(f in k.lower() for f in FORBIDDEN):
                raise ValueError(f"refusing to log a field named {k!r}")
        rec = {"t": round(time.time(), 3), "action": action, **detail}
        with _LOCK, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def tail(self, n: int = 200) -> list[dict]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        except OSError:
            return []
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out[::-1]
