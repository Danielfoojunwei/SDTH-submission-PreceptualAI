"""Finite, resumable scene-level training with equal information and budgets."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import io
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from . import common as c, data
from .model import SpectralScene, balanced_objective

INITIAL = ("joint_none", "joint_sigreg", "joint_vicreg")
CONTROLS = ("matched_regression", "geometry_only")
TARGET_NAMES = ("center_dbm", "lower_dbm", "upper_dbm", "heard", "censored", "censor_upper_dbm")


def tensor(value):
    array = np.array(value, copy=True)
    return torch.from_numpy(array if array.dtype == bool else array.astype("f4"))


def batch(examples):
    names = ("context_a", "context_b", "mask_a", "mask_b", "query", "query_basis",
             "query_fspl_dbm", "query_offset_db")
    result = {name: tensor(np.stack([getattr(e, name) for e in examples])) for name in names}
    result["targets"] = {name: tensor(np.stack([getattr(e.targets, name) for e in examples]))
                         for name in TARGET_NAMES}
    return result


def settings(cell, selected_prior=None):
    if cell in INITIAL:
        prior = cell.removeprefix("joint_")
    elif cell in CONTROLS and selected_prior in ("none", "sigreg", "vicreg"):
        prior = selected_prior
    else:
        raise ValueError("Unregistered cell/prior")
    return {"regularizer": prior, "geometry_only": cell == "geometry_only"}


def selected_prior():
    selection = c.read_json(c.OUT / "selection.json")
    scores = {}
    for cell in INITIAL:
        receipts = [c.read_json(c.OUT / "fits" / f"{cell}-seed{seed}.json") for seed in (0, 1, 2)]
        values = [r["validation_mean_layout_rmse_db"] for r in receipts]
        candidate = selection["candidates"][cell]
        hashes = [c.digest(c.OUT / "fits" / f"{cell}-seed{seed}.json") for seed in (0, 1, 2)]
        if (candidate["seed_rmse_db"] != values or candidate["fit_receipts_sha256"] != hashes
                or candidate["mean_rmse_db"] != float(np.mean(values))):
            raise RuntimeError("VAL selection inputs changed")
        scores[cell] = candidate["mean_rmse_db"]
    winner = min(INITIAL, key=scores.get)
    prior = winner.removeprefix("joint_")
    if (selection["selected_cell"] != winner or selection["selected_prior"] != prior
            or selection["source_receipt_sha256"] != c.digest(c.OUT / "registration/source.json")
            or selection["protocol_sha256"] != c.digest(c.PROTOCOL)):
        raise RuntimeError("Controls must use the fixed three-seed VAL winner")
    return prior


def identity():
    c.check_sources()
    return {"source_receipt_sha256": c.digest(c.OUT / "registration/source.json"),
            "protocol_sha256": c.digest(c.OUT / "protocol.json"),
            "data_receipt_sha256": c.digest(c.OUT / "data/prepared-training.json"),
            "training_rt_receipt_sha256": c.digest(c.OUT / "rt_data/training.json"),
            "validation_rt_receipt_sha256": c.digest(c.OUT / "rt_data/validation.json")}


def validate_history(history, *, complete=False):
    if not isinstance(history, list) or len(history) > 60 or (complete and len(history) != 60):
        raise RuntimeError("Unexpected training history length")
    for epoch, entry in enumerate(history):
        if (entry["epoch"] != epoch or len(entry["updates"]) != 8
                or not np.isfinite(entry["validation_mean_layout_rmse_db"])
                or any(not isinstance(update, dict) or "loss" not in update
                       or not np.isfinite(list(update.values())).all() for update in entry["updates"])):
            raise RuntimeError("Training history/budget changed")


def validate_completed(saved, ident, cell, seed, option, checkpoint_path):
    if (saved["identity"] != ident or saved["settings"] != option
            or saved["checkpoint_sha256"] != c.digest(checkpoint_path)
            or saved["completed"] is not True or saved["cell"] != cell or saved["seed"] != seed
            or saved["run_id"] != f"{cell}-seed{seed}" or saved["epochs"] != 60 or saved["updates"] != 480):
        raise RuntimeError("Completed fit identity changed")
    validate_history(saved["history"], complete=True)
    scores = [h["validation_mean_layout_rmse_db"] for h in saved["history"]]
    if (saved["selected_epoch"] != int(np.argmin(scores))
            or saved["validation_mean_layout_rmse_db"] != min(scores)):
        raise RuntimeError("Checkpoint selection differs from the fixed validation rule")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint["identity"] != ident or checkpoint["cell"] != cell or checkpoint["seed"] != seed
            or checkpoint["settings"] != option or checkpoint["selected_epoch"] != saved["selected_epoch"]):
        raise RuntimeError("Checkpoint metadata changed")
    return checkpoint


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".resume-", suffix=".pt", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_checkpoint(path, value):
    stream = io.BytesIO()
    torch.save(value, stream)
    c.write_bytes_once(path, stream.getvalue())


def dropout_context(context, mask, rng, probability=.15):
    """Remove complete observations, keeping at least one source per band."""
    retained = mask.clone()
    drop = tensor(rng.random(tuple(mask.shape)) >= probability)
    retained &= drop
    band = context[..., 11]  # context4 + geometry7 = log(f/2.4GHz)
    for row in range(len(mask)):
        for value in torch.unique(band[row][mask[row]]):
            eligible = torch.where(mask[row] & (band[row] == value))[0]
            if len(eligible) and not retained[row, eligible].any():
                retained[row, eligible[0]] = True
    return retained


@torch.no_grad()
def _validation(model, values):
    predicted, _ = model(values["context_a"], values["mask_a"], values["query"],
                         values["query_basis"], values["query_fspl_dbm"], values["query_offset_db"])
    scores = []
    for row in range(len(predicted)):
        known = values["targets"]["heard"][row]
        if not known.any():
            raise RuntimeError("Validation scene has no heard query observations")
        residual = predicted[row][known] - values["targets"]["center_dbm"][row][known]
        scores.append(float(residual.double().square().mean().sqrt()))
    return float(np.mean(scores))


def fit(task):
    cell, seed, chosen_prior = task
    ident = identity()
    if seed not in (0, 1, 2):
        raise ValueError("Unregistered independent training seed")
    run = f"{cell}-seed{seed}"
    receipt_path = c.OUT / "fits" / f"{run}.json"
    checkpoint_path = c.OUT / "checkpoints" / f"{run}.pt"
    if cell in CONTROLS and chosen_prior != selected_prior():
        raise RuntimeError("Control prior differs from the published validation selection")
    option = settings(cell, chosen_prior)
    if receipt_path.exists():
        saved = c.read_json(receipt_path)
        validate_completed(saved, ident, cell, seed, option, checkpoint_path)
        return saved
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpectralScene(**option)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1.e-4)
    training = batch(data.examples("training"))
    validation = batch(data.examples("validation"))
    if training["context_a"].shape[0] != 48:
        raise RuntimeError("The scene regularizer requires 48 distinct registered TRAIN layouts")
    resume = c.WORK / f"{run}.last.pt"
    history, best, best_epoch, best_state, elapsed = [], float("inf"), None, None, 0.
    if resume.exists():
        prior = torch.load(resume, map_location="cpu", weights_only=True)
        if (prior["identity"] != ident or prior["settings"] != option or prior["cell"] != cell
                or prior["seed"] != seed or prior["run_id"] != run):
            raise RuntimeError("Resume source/data identity changed")
        validate_history(prior["history"])
        if prior["history"]:
            scores = [h["validation_mean_layout_rmse_db"] for h in prior["history"]]
            if prior["best_epoch"] != int(np.argmin(scores)) or prior["best"] != min(scores):
                raise RuntimeError("Resume selection changed")
        model.load_state_dict(prior["state"])
        optimizer.load_state_dict(prior["optimizer"])
        torch.set_rng_state(prior["torch_rng"])
        rng.bit_generator.state = prior["numpy_rng"]
        history, best, best_epoch = prior["history"], prior["best"], prior["best_epoch"]
        best_state, elapsed = prior["best_state"], prior["elapsed_seconds"]
    began = time.perf_counter()
    for epoch in range(len(history), 60):
        model.train()
        logs = []
        for update in range(8):
            mask_a = dropout_context(training["context_a"], training["mask_a"], rng)
            mask_b = dropout_context(training["context_b"], training["mask_b"], rng)
            optimizer.zero_grad(set_to_none=True)
            loss, details = balanced_objective(
                model, (training["context_a"], mask_a), (training["context_b"], mask_b),
                training["query"], training["query_basis"], training["query_fspl_dbm"],
                training["query_offset_db"], training["targets"], alignment=cell != "matched_regression",
                diagnose_backbone=update == 0)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            logs.append({"loss": float(loss.detach()), "total_gradient_before_clip": float(norm),
                         **{key: float(value) for key, value in details.items()}})
        model.eval()
        score = _validation(model, validation)
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite validation score")
        if score < best:
            best, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append({"epoch": epoch, "validation_mean_layout_rmse_db": score,
                        "updates": logs})
        atomic_save(resume, {"identity": ident, "cell": cell, "seed": seed, "run_id": run,
                    "settings": option, "state": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                    "numpy_rng": rng.bit_generator.state, "history": history, "best": best,
                    "best_epoch": best_epoch, "best_state": best_state,
                    "elapsed_seconds": elapsed + time.perf_counter() - began})
    model.load_state_dict(best_state)
    elapsed += time.perf_counter() - began
    save_checkpoint(checkpoint_path, {"identity": ident, "cell": cell, "seed": seed,
                    "settings": option, "state": best_state, "selected_epoch": best_epoch})
    result = {"schema": 1, "run_id": run, "cell": cell, "seed": seed, "settings": option,
              "identity": ident, "completed": True, "epochs": 60, "updates": 480,
              "selected_epoch": best_epoch, "validation_mean_layout_rmse_db": best,
              "checkpoint_sha256": c.digest(checkpoint_path), "history": history,
              "elapsed_seconds": elapsed, "distinct_training_layouts": 48,
              "learned_parameters": sum(p.numel() for p in model.parameters()),
              "uncertainty_calibrated": False}
    c.write_once(receipt_path, result)
    return result


def _run(tasks, workers):
    if workers not in (1, 2):
        raise ValueError("CPU protocol admits one or two fit workers")
    if workers == 1:
        return [fit(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fit, tasks))


def train_initial(workers=2):
    # Write the immutable VAL data receipts once before concurrent workers read.
    data.examples("validation")
    return _run([(cell, seed, None) for cell in INITIAL for seed in (0, 1, 2)], workers)


def train_controls(chosen_prior, workers=2):
    if chosen_prior != selected_prior():
        raise RuntimeError("Control prior differs from the fixed validation selection")
    settings("matched_regression", chosen_prior)
    return _run([(cell, seed, chosen_prior) for cell in CONTROLS for seed in (0, 1, 2)], workers)


def load(cell, seed):
    c.check_sources()
    if cell not in (*INITIAL, *CONTROLS) or seed not in (0, 1, 2):
        raise ValueError("Unregistered checkpoint")
    receipt = c.read_json(c.OUT / "fits" / f"{cell}-seed{seed}.json")
    path = c.OUT / "checkpoints" / f"{cell}-seed{seed}.pt"
    expected_option = settings(cell, selected_prior() if cell in CONTROLS else None)
    checkpoint = validate_completed(receipt, identity(), cell, seed, expected_option, path)
    model = SpectralScene(**checkpoint["settings"])
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model


@torch.no_grad()
def encode(model, example, view="a"):
    if view not in ("a", "b"):
        raise ValueError("Unknown public scene view")
    return model.encode(tensor(getattr(example, "context_" + view))[None],
                        tensor(getattr(example, "mask_" + view))[None])


@torch.no_grad()
def predict(model, example, view="a", *, intervention=None, z_override=None):
    z = encode(model, example, view)
    if intervention == "zero":
        z = torch.zeros_like(z)
    elif intervention is not None:
        raise ValueError("Use explicit z_override for across-layout shuffle")
    if z_override is not None:
        z = tensor(z_override).reshape_as(z)
    mean = model.render(z, tensor(example.query)[None], tensor(example.query_basis)[None],
                        tensor(example.query_fspl_dbm)[None], tensor(example.query_offset_db)[None])
    result = mean[0].numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite scene prediction")
    return result, np.full_like(result, 4.)
