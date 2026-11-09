"""Read-only independent audit of the already sealed measured-CSI study.

No model imports, forwards, fitting, hyperparameter search, or new scientific
cases. Reconstructs the registered scores from saved predictions and original
measured I/Q bytes. Writes only its sibling review.json after every check.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/channel_scene"
WORK = Path(os.environ.get("CHANNEL_SCENE_CACHE_ROOT", str(ROOT.parent / "training/channel-scene")))
DATA = Path(os.environ.get("CHANNEL_SCENE_REVIEW_DATA_ROOT", "/workspace/upstream/scene-radio/csi/dichasus"))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def path_below(root, name):
    result = root / name
    assert result.resolve().is_relative_to(root.resolve()) and not result.is_symlink(), name
    return result


def check_files(root, entries):
    for name, expected in entries.items():
        assert sha(path_below(root, name)) == expected, name


def load_member(reference):
    assert reference["storage"] in {"repository", "external_workspace"}
    root = ROOT if reference["storage"] == "repository" else WORK
    path = path_below(root, reference["path"])
    assert sha(path) == reference["sha256"]
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def equal_metrics(actual, expected):
    assert set(actual) == set(expected)
    for key, value in actual.items():
        if isinstance(value, (bool, int)):
            assert expected[key] == value, key
        else:
            assert np.isclose(value, expected[key], rtol=1e-12, atol=1e-12), (key, value, expected[key])


def packet_scores(prediction, target):
    assert prediction.shape == target.shape and prediction.ndim == 3
    assert np.isfinite(prediction).all() and np.isfinite(target).all()
    error = np.square(np.abs(prediction.astype(np.complex128) - target))
    energy = np.square(np.abs(target.astype(np.complex128)))
    denominator = energy.sum(axis=(1, 2))
    keep = denominator > 0
    nmse = error[keep].sum(axis=(1, 2)) / denominator[keep]
    return {
        "mean_packet_nmse": float(nmse.mean()),
        "median_packet_nmse": float(np.median(nmse)),
        "p90_packet_nmse": float(np.quantile(nmse, 0.9)),
        "global_nmse": float(error[keep].sum() / energy[keep].sum()),
        "packets": int(keep.sum()),
        "zero_target_packets_excluded": int((~keep).sum()),
        "case_quantiles_are_not_iid_confidence_intervals": True,
    }


def spatial_partition(positions, guard):
    # Independent reconstruction of the frozen public pose-only rule.
    xy = np.asarray(positions, dtype=np.float64)[:, :2]
    low, high = np.quantile(xy[:, 1], [0.6, 0.8])
    train = np.flatnonzero(xy[:, 1] < low - guard / 2)
    val = np.flatnonzero((xy[:, 1] > low + guard / 2) & (xy[:, 1] < high - guard / 2))
    test = np.flatnonzero(xy[:, 1] > high + guard / 2)

    def near(query, reference):
        return (np.square(xy[query, None] - xy[None, reference]).sum(-1) < guard**2).any(-1)

    train = train[~near(train, test)]
    val = val[~near(val, test)]
    train = train[~near(train, val)]
    return {"training": train, "validation": val, "cf06_test": test,
            "excluded": np.setdiff1d(np.arange(len(xy)), np.concatenate([train, val, test]))}


def measured_values(name, protocol):
    family = {"dichasus-cf06": "cf0x", "dichasus-ad00": "adxx", "dichasus-dc01": "dcxx"}[name]
    files = {"npz": DATA / (name + "-first1024.npz"),
             "receipt": DATA / (name + "-first1024.json"),
             "binding": DATA / (name + "-binding.json"),
             "calibration": DATA.parent / "research/dichasus" /
             ("content__datasets__data__dichasus-" + family + "__reftx-offsets-" + name + ".json")}
    identities = {"measured:" + name + ":" + key: sha(path) for key, path in files.items()}
    assert identities == protocol["observation_sha256"][name]
    receipt, binding, offsets = (read(files[key]) for key in ("receipt", "binding", "calibration"))
    assert receipt["all_length_and_payload_crc32c_pass"] is True
    assert receipt["parsed_npz_sha256"] == identities["measured:" + name + ":npz"]
    bins = np.asarray(protocol["raw_bin_indices"], dtype=int)
    with np.load(files["npz"], allow_pickle=False) as saved:
        raw = saved["csi"]
        assert raw.dtype == np.float32 and raw.shape[2:] == (1024, 2)
        assert np.isfinite(raw).all()
        # FFT-shift then subsample, without constructing any model input or output.
        raw_indices = (bins + 512) % 1024
        value = raw[:, :, raw_indices, 0] + 1j * raw[:, :, raw_indices, 1]
        times = np.asarray(saved["time"], dtype=np.float64).reshape(-1)
        positions = saved["pos-tachy"].copy() if name == "dichasus-cf06" else None
    sto, cpo = np.asarray(offsets["sto"]), np.asarray(offsets["cpo"])
    phase = sto[:, None] * (2 * np.pi * bins / 1024) + cpo[:, None]
    value = (value * np.exp(1j * phase)).astype(np.complex64)
    frequency = (bins - 512) * binding["bandwidth_hz"] / 1024
    assert len(value) == receipt["acquired_records"] == 1024
    assert len(times) == len(value) and np.isfinite(times).all()
    assert binding["frequency_hz"] == receipt["frequency_hz"]
    assert binding["bandwidth_hz"] == receipt["bandwidth_hz"]
    return value, frequency, positions, {
        "source_sha256": identities, "packets_acquired": len(value), "antennas": value.shape[1],
        "carrier_hz": binding["frequency_hz"], "bandwidth_hz": binding["bandwidth_hz"],
        "capture_elapsed_s": float(times[-1] - times[0]),
        "timestamps_nondecreasing": bool((np.diff(times) >= 0).all()),
        "calibration": "publisher STO/CPO on shifted original bins; no target-derived gain/phase fit",
        "power": binding["power_units"], "clock": binding["clock"],
    }


def byte_closure(manifest):
    check_files(ROOT, manifest["source_sha256"])
    check_files(ROOT, manifest["evidence_sha256"])
    check_files(WORK, manifest["external_prediction_sha256"])
    for field in ("cache_manifest", "prediction_capsule"):
        entry = manifest[field]
        assert sha(path_below(WORK, entry["path"])) == entry["sha256"]
    old_path = ROOT / "artifacts/radiojepa_spectral/verification/manifest.json"
    assert sha(old_path) == manifest["previous"]["manifest_sha256"]
    old = read(old_path)
    check_files(ROOT, old["source_sha256"])
    check_files(ROOT, old["evidence_sha256"])
    return {"source_files": len(manifest["source_sha256"]),
            "evidence_files": len(manifest["evidence_sha256"]),
            "external_prediction_shards": len(manifest["external_prediction_sha256"]),
            "preserved_old_sources": len(old["source_sha256"]),
            "preserved_old_evidence": len(old["evidence_sha256"])}


def main():
    manifest_path = OUT / "verification/manifest.json"
    manifest_before = sha(manifest_path)
    manifest = read(manifest_path)
    before = byte_closure(manifest)
    assert before == {"source_files": 491, "evidence_files": 24, "external_prediction_shards": 57,
                      "preserved_old_sources": 475, "preserved_old_evidence": 5161}
    protocol, inventory, fits, predictions, report = (read(OUT / name) for name in
        ("protocol.json", "inventory.json", "fits/completed.json", "predictions/completed.json", "report.json"))
    source_receipt, publication = (read(OUT / ("registration/" + name)) for name in ("source.json", "git.json"))
    assert source_receipt["source_sha256"] == manifest["source_sha256"]
    assert publication["source_receipt_sha256"] == sha(OUT / "registration/source.json")
    assert publication["remote_observed_commit"] == publication["commit"]
    capsule_sources = OUT / "registration/source"
    check_files(capsule_sources, source_receipt["source_sha256"])
    assert {str(p.relative_to(capsule_sources)) for p in capsule_sources.rglob("*") if p.is_file()} == set(source_receipt["source_sha256"])
    # Verify the published frozen source, rather than trusting a receipt alone.
    registered = source_receipt["source_sha256"] | source_receipt["preflight_sha256"]
    registered["artifacts/channel_scene/registration/source.json"] = sha(OUT / "registration/source.json")
    registered.update({"artifacts/channel_scene/registration/source/" + k: v for k, v in source_receipt["source_sha256"].items()})
    proc = subprocess.Popen(["git", "archive", "--format=tar", publication["commit"], *registered], cwd=ROOT, stdout=subprocess.PIPE)
    seen = set()
    with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
        for member in archive:
            if member.isdir():
                continue
            assert member.isfile() and member.name in registered
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == registered[member.name]
            seen.add(member.name)
    assert proc.wait() == 0 and seen == set(registered)
    assert fits["source_sha256"] == predictions["source_sha256"] == report["source_sha256"] == manifest["source_sha256"]
    assert predictions["fit_receipt_sha256"] == sha(OUT / "fits/completed.json")
    assert report["predictions_receipt_sha256"] == sha(OUT / "predictions/completed.json")
    assert protocol["seeds"] == [17, 43, 89] and protocol["epochs"] == 40
    assert fits["held_recordings_opened_before_all_fits"] is False
    runs = []
    for row in fits["fits"]:
        history = row["history"]
        assert [r["epoch"] for r in history] == list(range(1, 41))
        assert np.isfinite([[r["loss"], r["validation_packet_nmse"]] for r in history]).all()
        selected = min(history, key=lambda r: r["validation_packet_nmse"])
        assert selected["epoch"] == row["selected_epoch"] and row["held_recordings_opened"] is False
        assert row == read(OUT / f"fits/{row['arm']}-{row['seed']}.json")
        weights = load_member(row["weights"])
        assert weights["encoder.0.weight"].shape == (64, 195)
        assert weights["encoder.2.weight"].shape == (24, 64)
        assert weights["decoder.weight"].shape == (128, 24)
        runs.append({"arm": row["arm"], "seed": row["seed"], "epochs": 40,
                     "selected_epoch": row["selected_epoch"], "selected_validation_packet_nmse": selected["validation_packet_nmse"],
                     "validation_latent": row["validation_latent"]})
    assert {(r["arm"], r["seed"]) for r in runs} == {(a, s) for a in ("joint", "regression") for s in [17, 43, 89]}
    for kind, fields in (("fourier", ("n_taps", "ridge")), ("lmmse", ("shrinkage", "noise_fraction"))):
        candidates = fits[kind + "_candidates"]
        grid = ([protocol["fourier_taps"], protocol["fourier_ridge"]] if kind == "fourier"
                else [protocol["lmmse_shrinkage"], protocol["lmmse_noise_fraction"]])
        assert [(r[fields[0]], r[fields[1]]) for r in candidates] == [(a, b) for a in grid[0] for b in grid[1]]
        assert fits[kind] == min(candidates, key=lambda r: r["validation_packet_nmse"])
    baselines = {f"{b}_{v}" for b in ("linear", "fourier", "lmmse") for v in ("a", "b")}
    members = baselines | {f"{a}_{s}_{v}" for a in ("joint", "regression") for s in [17, 43, 89] for v in ("a", "b")}
    expected_domains = {"dichasus-cf06", "dichasus-ad00", "dichasus-dc01"}
    assert set(predictions["predictions"]) == set(report["summary"]) == set(report["cases"]) == expected_domains
    pilot_union = np.r_[protocol["pilot_a"], protocol["pilot_b"]]
    assert len(set(pilot_union)) == 32 and not set(pilot_union) & set(protocol["target"])
    all_cases, domains, summary, zero_fill_ratios = {}, {}, {}, {}
    for name in sorted(expected_domains):
        values, frequency, positions, metadata = measured_values(name, protocol)
        if name == "dichasus-cf06":
            partition = spatial_partition(positions, protocol["guard_m"])
            for key, rows in partition.items():
                assert np.array_equal(rows, inventory["rows"][key])
            train = values[partition["training"]]
            pilot_scale = float(np.sqrt(np.mean(np.square(np.abs(train[:, :, pilot_union].astype(np.complex128))))))
            assert np.isclose(pilot_scale, inventory["training_pilot_scale"], rtol=1e-12, atol=0)
            assert np.array_equal(frequency, inventory["actual_sampled_offsets_hz"])
            flat = train.reshape(-1, 64).astype(np.complex128)
            covariance = flat.T @ flat.conj() / len(flat)
            covariance = (covariance + covariance.conj().T) / 2
            shrink = fits["lmmse"]["shrinkage"]
            covariance = (1 - shrink) * covariance + shrink * np.diag(covariance.diagonal().real)
            saved_state = load_member(fits["baseline_state"])
            assert np.allclose(covariance, saved_state["covariance"], rtol=1e-12, atol=1e-12)
            assert int(saved_state["training_count"]) == len(flat) == 17952
            rows = partition["cf06_test"]
            metadata["partition_rows"] = {k: len(v) for k, v in partition.items()}
            metadata["train_pilot_rms_recomputed"] = pilot_scale
        else:
            rows = np.arange(1024)
        reference = predictions["predictions"][name]
        assert reference["source_sha256"] == metadata["source_sha256"]
        assert set(reference["members"]) == members | {"frames"}
        assert np.array_equal(load_member(reference["members"]["frames"])["value"], rows)
        truth = values[rows][:, :, protocol["target"]]
        assert list(truth.shape) == reference["target_shape"]
        cases, ratios = {}, {}
        for key in sorted(members):
            ref = reference["members"][key]
            assert ref["path"] == f"predictions/{name}/{key}.npz"
            prediction = load_member(ref)["value"]
            result = packet_scores(prediction, truth)
            equal_metrics(result, report["cases"][name][key])
            cases[key] = result
            if key.startswith("fourier_"):
                ratios[key] = float(np.square(np.abs(prediction.astype(np.complex128))).sum() /
                                    np.square(np.abs(truth.astype(np.complex128))).sum())
        domain_summary = {}
        for arm in ("joint", "regression"):
            seeds = [float(np.mean([cases[f"{arm}_{seed}_{view}"]["mean_packet_nmse"] for view in ("a", "b")])) for seed in [17, 43, 89]]
            domain_summary[arm] = {"seed_mean_packet_nmse": float(np.mean(seeds)),
                                   "seed_std_packet_nmse": float(np.std(seeds, ddof=1)), "individual_seed_nmse": seeds}
            for field in ("seed_mean_packet_nmse", "seed_std_packet_nmse", "individual_seed_nmse"):
                assert np.allclose(domain_summary[arm][field], report["summary"][name][arm][field], rtol=1e-12, atol=1e-12)
        for baseline in ("linear", "fourier", "lmmse"):
            value = float(np.mean([cases[baseline + "_" + view]["mean_packet_nmse"] for view in ("a", "b")]))
            assert np.isclose(value, report["summary"][name][baseline]["mean_packet_nmse"], rtol=1e-12, atol=1e-12)
            domain_summary[baseline] = value
        all_cases[name], domains[name], summary[name], zero_fill_ratios[name] = cases, metadata, domain_summary, ratios
    # Exact source-geometry identity, not a newly scored dataset case.
    k, m = np.arange(64)[:, None], np.arange(64)[None, :]
    basis = np.exp(-2j * np.pi * k * m / 64)
    a, b, target = (np.asarray(protocol[x]) for x in ("pilot_a", "pilot_b", "target"))
    assert np.array_equal(a, np.arange(0, 64, 4)) and np.array_equal(b, np.arange(2, 64, 4))
    assert np.array_equal(target, np.arange(1, 64, 2))
    cross = basis[target] @ basis[pilot_union].conj().T
    ambiguity = {"single_view_rank": int(np.linalg.matrix_rank(basis[a])),
                 "union_view_rank": int(np.linalg.matrix_rank(basis[pilot_union])),
                 "coefficient_count": 64,
                 "max_unobserved_observed_gram_abs": float(abs(cross).max()),
                 "a_taps_0_16_difference_max": float(abs(basis[a, 0] - basis[a, 16]).max()),
                 "union_taps_0_32_difference_max": float(abs(basis[pilot_union, 0] - basis[pilot_union, 32]).max()),
                 "target_taps_0_32_difference_abs_min": float(abs(basis[target, 0] - basis[target, 32]).min()),
                 "fourier_prediction_to_truth_energy": zero_fill_ratios}
    assert ambiguity["single_view_rank"] == 16 and ambiguity["union_view_rank"] == 32
    # Verify the portable artifact contains exactly all prediction shards.
    expected_tar = manifest["external_prediction_sha256"] | {"deliverables/cache_manifest.json": manifest["cache_manifest"]["sha256"]}
    seen = set()
    with tarfile.open(WORK / manifest["prediction_capsule"]["path"], "r:gz") as archive:
        for member in archive:
            name = member.name.removeprefix("channel-scene-predictions/")
            assert member.isfile() and name in expected_tar and name not in seen
            digest = hashlib.sha256()
            stream = archive.extractfile(member)
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
            assert digest.hexdigest() == expected_tar[name]
            seen.add(name)
    assert seen == set(expected_tar)
    assert manifest_before == sha(manifest_path) and byte_closure(manifest) == before
    review = {"schema": 1, "passes": True, "review_script_sha256": sha(__file__),
              "sealed_manifest_sha256_before": manifest_before, "sealed_manifest_sha256_after": sha(manifest_path),
              "closure": before, "published_source_commit": publication["commit"],
              "report_sha256": sha(OUT / "report.json"), "fit_receipt_sha256": sha(OUT / "fits/completed.json"),
              "prediction_receipt_sha256": sha(OUT / "predictions/completed.json"),
              "prediction_capsule": manifest["prediction_capsule"],
              "independent_scored_domain_model_views": 54, "independent_summary_rows": 15,
              "training_covariance_reconstructed_without_refitting": True,
              "pose_partition_and_train_pilot_scale_reconstructed": True,
              "registered_finite_validation_candidate_membership_and_first_minimum_checked": True,
              "selected_checkpoint_metadata_first_minimum_checked": True,
              "neural_validation_scores_independently_replayed": False,
              "validation_candidate_predictions_independently_replayed": False,
              "model_forward_calls": 0, "fits": 0, "new_scientific_cases": 0,
              "scope": "Independent raw-I/Q target scoring and receipt/budget/selection metadata audit; owner raw.json separately records weight/pilot replay.",
              "domains": domains, "runs": runs, "cases": all_cases, "summary": summary,
              "aliasing_geometry": ambiguity, "production_qualified": False,
              "limitations": ["Previously inspected, temporally correlated first1024 prefixes; one corpus and no global unseen-data assertion.",
                              "Packets, antennas and subcarriers are correlated; no iid antenna confidence intervals.",
                              "Three neural training seeds quantify initialization/order variability only.",
                              "Domain/band/environment/receiver geometry shifts are confounded.",
                              "No warehouse action, protocol transfer, throughput, BLER or robot service outcome is measured."]}
    destination = Path(__file__).with_name("review.json")
    payload = json.dumps(review, indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination.write_text(payload)
    print(json.dumps({"passes": True, "path": str(destination.relative_to(ROOT)), "sha256": sha(destination),
                      "case_scores": 54, "summary_rows": 15, "closure": before}))


if __name__ == "__main__":
    main()
