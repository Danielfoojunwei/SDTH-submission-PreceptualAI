"""Markdown tables from artifacts/native/campaign.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .run import OUT

KEYS = [("control_miss", "Control misses", "{:.3%}"),
        ("control_miss_worst_decile", "Worst-decile robot", "{:.3%}"),
        ("control_gap_p99_ms", "p99 longest gap (ms)", "{:.0f}"),
        ("video_on_time", "Video on time", "{:.3f}"),
        ("bulk_mbps", "Bulk (Mb/s)", "{:.0f}"),
        ("airtime_domain_sum", "Airtime", "{:.2f}"),
        ("roams", "Roams", "{:.0f}")]


def table(path: Path = OUT / "campaign.json") -> str:
    rows = json.loads(Path(path).read_text())["rows"]
    arms = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    head = "| Arm | " + " | ".join(k[1] for k in KEYS) + " |"
    lines = [head, "|" + "---|" * (len(KEYS) + 1)]
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        cells = [fmt.format(float(np.mean([r[k] for r in rs]))) for k, _, fmt in KEYS]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
