"""100-robot AGV/AMR warehouse benchmark.

    uv run python -m wirelessjev.bench_warehouse --scenes 50 --out artifacts/warehouse

Reports, on held-out warehouse scenes scored with the route-aware oracle:
zero-shot (single choice and option-wise), the 10-car adapter applied
unchanged (transfer), a warehouse-distilled adapter, and one combined adapter
distilled on both sites (also re-checked on the 10-car test split).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .bench import (TAU, ALL_MISSIONS, fleet_examples, fleet_metrics, intent_flip,
                    run_fleet)
from .distill import (dataset_digest, distill_listwise, list_choices, save_adapter)
from .missions import INTENT_FLIP_PAIRS, WAREHOUSE_FLIP_PAIRS, WAREHOUSE_MISSIONS
from .policy import WirelessJEV
from .twin.warehouse import make_warehouse


def _load_readout(path: Path) -> torch.Tensor:
    from safetensors.torch import load_file
    return load_file(str(path / "readout.safetensors"))["readout"].float()


def _apply(cases, readout, true_row, key, temp):
    ex = fleet_examples(cases)
    test = [e for e in ex if e.split == "test"]
    test_cases = [c for c in cases if c["split"] == "test"]
    for c, ch in zip(test_cases, list_choices(readout, true_row, test, temp)):
        c[key] = ch


def _load_ten_car_cases(bench_dir: Path):
    cases_path, hidden_path = bench_dir / "cases.json", bench_dir / "hidden_cache.pt"
    if not (cases_path.exists() and hidden_path.exists()):
        return []
    cases = json.loads(cases_path.read_text())
    hidden = {h["case_id"]: h["hidden"] for h in torch.load(hidden_path)}
    for c in cases:
        c["hidden"] = hidden[c["case_id"]]
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=50)
    ap.add_argument("--out", default="artifacts/warehouse")
    ap.add_argument("--ten-car-bench", default="artifacts/bench")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "progress.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    t0 = time.time()
    engine = WirelessJEV()
    temp = float(engine.runtime.temperature)
    base = engine.base_readout.detach().float().cpu()
    cases = run_fleet(engine, args.scenes, log, scene_fn=make_warehouse,
                      missions=WAREHOUSE_MISSIONS, seed0=20000)["cases"]
    true_row = cases[0]["true_row"]
    pairs = WAREHOUSE_FLIP_PAIRS
    report = {"n_robots": len(cases[0]["scene"].cars), "n_cases": len(cases),
              "parity_err_max": max(c["parity_err"] for c in cases),
              "zero_shot_single_choice": fleet_metrics(cases, "jev_choice"),
              "zero_shot_optionwise": fleet_metrics(cases, "jev"),
              "intent_flip_zero_shot": intent_flip(cases, "jev_choice", pairs)}

    ten_dir = Path(args.ten_car_bench)
    if (ten_dir / "adapters" / "fleet").exists():
        _apply(cases, _load_readout(ten_dir / "adapters" / "fleet"), true_row,
               "jev_transfer", temp)
        report["transfer_10car_adapter"] = fleet_metrics(cases, "jev_transfer")
        report["intent_flip_transfer"] = intent_flip(cases, "jev_transfer", pairs)

    ex = fleet_examples(cases)
    w_readout, info = distill_listwise(base, true_row, ex, temperature=temp)
    save_adapter(out / "adapters" / "warehouse", w_readout, task="warehouse/option_wise",
                 info=info, dataset_digest=dataset_digest(ex))
    _apply(cases, w_readout, true_row, "jev_distilled", temp)
    report["distilled_warehouse"] = fleet_metrics(cases, "jev_distilled")
    report["distilled_val_kl"] = [info["val_kl_before"], info["val_kl_after"]]
    report["intent_flip_distilled"] = intent_flip(cases, "jev_distilled", pairs)
    log("warehouse distillation done")

    ten = _load_ten_car_cases(ten_dir)
    if ten:
        both = ex + fleet_examples(ten)
        c_readout, cinfo = distill_listwise(base, true_row, both, temperature=temp)
        save_adapter(out / "adapters" / "combined", c_readout, task="fleet+warehouse/option_wise",
                     info=cinfo, dataset_digest=dataset_digest(both))
        _apply(cases, c_readout, true_row, "jev_combined", temp)
        _apply(ten, c_readout, true_row, "jev_combined", temp)
        report["combined_on_warehouse"] = fleet_metrics(cases, "jev_combined")
        report["combined_on_10car"] = fleet_metrics(ten, "jev_combined")
        report["intent_flip_combined_warehouse"] = intent_flip(cases, "jev_combined", pairs)
        report["intent_flip_combined_10car"] = intent_flip(ten, "jev_combined", INTENT_FLIP_PAIRS)
        log("combined distillation done")

    lat = [c["ms_optionwise"] for c in cases]
    report["latency_ms"] = {"optionwise_mean": float(np.mean(lat)),
                            "optionwise_p90": float(np.percentile(lat, 90)),
                            "device": engine.device}
    report["wall_clock_s"] = time.time() - t0
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    slim = [{k: v for k, v in c.items() if k not in ("hidden", "scene", "options")}
            for c in cases]
    (out / "cases.json").write_text(json.dumps(slim, default=float))
    torch.save([{"case_id": c["case_id"], "hidden": c["hidden"]} for c in cases],
               out / "hidden_cache.pt")
    log(f"done in {report['wall_clock_s']:.0f} s")


if __name__ == "__main__":
    main()
