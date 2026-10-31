"""Analyse provided switch-stage events; missing events remain missing."""
import argparse
import json
from pathlib import Path
from wirelessjev.measurement import switch_timing
from wirelessjev.warehouse_v2.experiment import atomic_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    events=[json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]
    report=switch_timing(events)
    report.update(input_sha256=sha256(args.events),input_origin_authenticated=False,hardware_events_collected_by_this_command=False)
    atomic_json(args.out,report)


if __name__=='__main__':main()
