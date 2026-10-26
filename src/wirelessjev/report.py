"""Render artifacts/bench/report.json (+ closedloop.json) as one static HTML page.

    uv run python -m wirelessjev.report
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8;--accent:#2f6f5e;--bad:#a8432f;--card:#ffffff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ecebe6;--muted:#a3a29b;--line:#2f2e2a;--accent:#7cc2ab;--bad:#e08b74;--card:#1e1e1c}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ecebe6;--muted:#a3a29b;--line:#2f2e2a;--accent:#7cc2ab;--bad:#e08b74;--card:#1e1e1c}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:960px;margin:0 auto}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
p.lede,.muted{color:var(--muted)}table{border-collapse:collapse;width:100%;margin:8px 0;font-variant-numeric:tabular-nums}
td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}
.wrap{overflow-x:auto}.good{color:var(--accent);font-weight:600}.bad{color:var(--bad);font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:10px 0}
"""


def f(x, d=3):
    if isinstance(x, float):
        return f"{x:.{d}f}"
    return html.escape(str(x))


def table(rows, head):
    out = ["<div class='wrap'><table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    return "".join(out) + "</table></div>"


def render(r: dict, cl: dict | None) -> str:
    z = r["fleet_zero_shot_single_choice"]
    zo = r["fleet_zero_shot_optionwise"]
    d = r["fleet_distilled"]
    dm = d["metrics"]
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>WirelessJEV Milestone 1</title><style>{CSS}</style></head><body><main>",
             "<h1>WirelessJEV — milestone 1</h1>",
             f"<p class='lede'>Simulated 10-car Raspbot fleet (SmartCar.zip hardware: BE200 client radio on wlan1, BSSID selection only). Decisions by frozen <code>{html.escape(r['model']['repo'])}@{r['model']['revision'][:10]}</code> on {html.escape(r['model']['device'])}; every reference number from the exact oracle. {r['n_cases']} fleet decisions; readout parity error ≤ {r['parity_err_max']:.1e}.</p>"]

    parts.append("<h2>Fleet decisions: regret vs. the best option (lower is better)</h2>")
    rows = [["WirelessJEV, distilled (option-wise, 1024-parameter head)", f"<span class='good'>{f(dm['jev_regret'])}</span>", f(dm["jev_top1"]), f(dm["defer_rate"])],
            ["WirelessJEV, zero-shot single choice", f(z["jev_regret"]), f(z["jev_top1"]), f(z["defer_rate"])],
            ["WirelessJEV, zero-shot option-wise", f(zo["jev_regret"]), f(zo["jev_top1"]), f(zo["defer_rate"])]]
    for name in ("milp", "legacy", "local_greedy", "coloring", "hungarian", "random", "strongest_rssi"):
        rows.append([f"Baseline: {name}", f(z[f"baseline_{name}_regret"]), "—", "—"])
    rows += [["Ablation: random option", f(z["ablation_random_option_regret"]), "—", "—"],
             ["Ablation: always DEFER (= keep current)", f(z["ablation_defer_regret"]), "—", "—"]]
    parts.append(table(rows, ["Policy", "Mean regret", "Top-1", "DEFER rate"]))
    parts.append(table([[m, f(z["jev_regret_by_mission"][m]), f(dm["jev_regret_by_mission"][m])]
                        for m in z["jev_regret_by_mission"]],
                       ["Mission", "Zero-shot (single choice)", "Distilled"]))

    fi, fd = r["intent_flip_zero_shot_single_choice"], d["intent_flip"]
    parts.append("<h2>Intent-flip test (same fleet state, different mission text)</h2>")
    parts.append(table([["Zero-shot single choice", fi["oracle_differs"], f(fi["model_flips_when_oracle_differs"]), f(fi["directionally_correct_flips"])],
                        ["Distilled", fd["oracle_differs"], f(fd["model_flips_when_oracle_differs"]), f(fd["directionally_correct_flips"])]],
                       ["Model", "Pairs where oracle choice differs", "Model choice also changes", "Change in the right direction"]))
    parts.append(table([["High contention", f(z["regret_high_contention"]), f(dm["regret_high_contention"])],
                        ["Low contention", f(z["regret_low_contention"]), f(dm["regret_low_contention"])]],
                       ["Contention (by busiest airtime in option set)", "Zero-shot regret", "Distilled regret"]))

    b = r["bias_audit"]
    parts.append("<h2>Bias audit (option-wise mode, zero-shot)</h2>")
    parts.append(table([["Same choice after reversing option order", f(b["order_reversal_consistency"])],
                        ["Padded option captures the choice", f(b["length_padding_capture"])],
                        ["'Recommended by senior engineer' captures the choice", f(b["authority_wording_capture"])],
                        ["Chance capture", f(b["chance_capture"])]], ["Probe", "Rate"]))

    parts.append("<h2>Privacy and unlearning (membership inference AUC; 0.5 = no leakage)</h2>")
    parts.append(table([["Released readout", f(d["mia_auc_base_readout"])], ["Distilled readout", f(d["mia_auc_distilled"])],
                        ["Region R0 before unlearning", f(d["mia_auc_region_before_unlearning"])],
                        ["Region R0 after exact unlearning", f(d["mia_auc_region_after_unlearning"])]], ["Readout", "AUC"]))

    s = r["swarm"]
    parts.append("<h2>Swarm consensus merge</h2>")
    parts.append(table([["Round 1 (9 trusted cars)", s["round1"]["released"], s["round1"]["leader"], ", ".join(s["round1"]["excluded_low_reputation"]), f(s["round1"]["validation"].get("val_regret", float('nan'))), f(s["round1"]["validation"].get("base_val_regret", float('nan')))],
                        ["Round 2 (poisoned updates)", s["round2_poisoned_updates"]["released"], "—", "—", f(s["round2_poisoned_updates"]["validation"].get("val_regret", float('nan'))), f(s["round2_poisoned_updates"]["validation"].get("base_val_regret", float('nan')))]],
                       ["Round", "Released", "Leader", "Excluded", "Val regret", "Base val regret"]))
    parts.append(f"<p class='muted'>Audit log entries: {s['log_entries']}; hash chain and HMAC verify: {s['log_verifies']}.</p>")

    t = r["traffic"]
    parts.append("<h2>Traffic labels (Chen, Lam &amp; Li framework)</h2>")
    la = t["label_accuracy"]
    parts.append(table([[k, f(la["jev_raw"][k]), f(la["jev_digest"][k]), f(t["label_distillation_test_accuracy"][k]["distilled"]), f(la["rule"][k])] for k in la["rule"]],
                       ["Label", "Jev, raw per-step prompt", "Jev, digested stats", "Jev, distilled readout (test)", "Fixed-threshold rules"]))
    parts.append(table([[k, f(v)] for k, v in t["forecast_mae"].items()], ["Forecaster", "MAE (Mbps)"]))
    parts.append(table([[k, f(v)] for k, v in t["residual_ablation_rmse"].items()], ["Residual features (same size)", "RMSE (Mbps)"]))
    parts.append(f"<p class='muted'>{html.escape(t['residual_note'])}.</p>")

    a = r["admission"]
    parts.append("<h2>Admission level (Shen, Lam &amp; Li price-step)</h2>")
    parts.append(table([["Jev", f(a["jev_local_direction_accuracy"])], ["Demand-vs-capacity rule", f(a["demand_rule_accuracy"])], ["Chance", f(a["chance"])]], ["Policy", "Best one-step move accuracy"]))

    lat = r["latency_ms"]
    parts.append("<h2>Latency on this Mac</h2>")
    parts.append(f"<p>Option-wise fleet decision: mean {lat['fleet_optionwise_mean']:.0f} ms, p90 {lat['fleet_optionwise_p90']:.0f} ms on {html.escape(lat['device'])}. That fits the 5 s policy interval and 30 s commitment horizon. It does not fit anything sub-10 ms; that loop stays conventional.</p>")

    if cl:
        parts.append("<h2>Closed loop: 300 s, 10 cars, mission switch, misbehaving car, coordinator loss</h2>")
        rows = [[p, f(v["mean_utility"]), v["total_handovers"], v["critical_violation_steps"], v["connects_beyond_one_per_txn"]] for p, v in cl["summary"].items()]
        parts.append(table(rows, ["Policy", "Mean realised mission utility", "Handovers", "Steps with a critical violation", "Connect requests resent"]))
    parts.append("<p class='muted'>Digital-twin results only. No car, radio or network was touched.</p></main></body></html>")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="artifacts/bench/report.json")
    ap.add_argument("--closedloop", default="artifacts/closedloop.json")
    ap.add_argument("--out", default="artifacts/report.html")
    args = ap.parse_args()
    r = json.loads(Path(args.bench).read_text())
    cl_path = Path(args.closedloop)
    cl = json.loads(cl_path.read_text()) if cl_path.exists() else None
    Path(args.out).write_text(render(r, cl))
    print(args.out)


if __name__ == "__main__":
    main()
