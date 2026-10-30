"""Reconstruct E3 label balance without loading a model or changing its results.

Run from the repository root with the research environment, for example:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
        docs/experiments/e3_label_audit.py --out /tmp/e3_label_audit.json
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wirelessjev import traffic as T
from wirelessjev.radiojev import e3


def audit():
    samples = [x for x in (e3.admission_sample(60000 + i) for i in range(160))
               if x is not None]
    cut = len(samples) * 2 // 3
    rng = np.random.default_rng(11)
    windows = [T.synth_window(rng) for _ in range(260)]
    traffic_cut = len(windows) * 3 // 4
    result = {"admission": {}, "traffic": {}}
    for split, subset in (("train", samples[:cut]), ("test", samples[cut:])):
        counts = Counter(["lower", "hold", "raise"][x["label"]] for x in subset)
        result["admission"][split] = {
            "n": len(subset), "labels": dict(counts),
            "majority_accuracy": max(counts.values()) / len(subset)}
    for key in T.QUESTIONS:
        result["traffic"][key] = {}
        majority = Counter(getattr(w, key) for w in windows[:traffic_cut]).most_common(1)[0][0]
        for split, subset in (("train", windows[:traffic_cut]), ("test", windows[traffic_cut:])):
            counts = Counter(getattr(w, key) for w in subset)
            result["traffic"][key][split] = {
                "n": len(subset), "labels": dict(counts),
                "training_majority_label": majority,
                "training_majority_accuracy": sum(getattr(w, key) == majority for w in subset) / len(subset)}
    result["method"] = "reconstruct-e3-label-splits-v1"
    result["admission_seed_start"] = 60000
    result["admission_samples"] = 160
    result["traffic_seed"] = 11
    result["traffic_windows"] = 260
    result["e3_source_sha256"] = hashlib.sha256(Path(e3.__file__).read_bytes()).hexdigest()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit(), indent=2) + "\n")
    print(args.out)
