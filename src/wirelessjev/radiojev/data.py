"""Scene datasets for RadioJEV: candidate plans, radio graphs and exact
utilities of every plan under every mission, cached to disk."""
from __future__ import annotations

from pathlib import Path
import pickle
import time

import numpy as np

from ..mission_texts import NOVEL, PARAPHRASES, TRAIN_PARAPHRASES, all_known
from ..missions import WAREHOUSE_MISSIONS
from ..twin.warehouse import make_warehouse
from .candidates import candidate_plans
from .features import plan_graph

ALL_MISSIONS = {**all_known(), **{k: v[0] for k, v in NOVEL.items()}}
MISSION_IDS = sorted(ALL_MISSIONS)


def mission_texts(split: str) -> dict[str, list[str]]:
    """train: original + first TRAIN_PARAPHRASES paraphrases of known missions;
    validation: next paraphrase; test: last paraphrase; novel: never-seen."""
    out = {}
    if split == "novel":
        return {k: list(v[1]) for k, v in NOVEL.items()}
    for mid, m in all_known().items():
        p = PARAPHRASES[mid]
        if split == "train":
            out[mid] = [m.intent] + p[:TRAIN_PARAPHRASES]
        elif split == "validation":
            out[mid] = [p[TRAIN_PARAPHRASES]]
        else:
            out[mid] = [p[-1]]
    return out


def build_scene(seed: int, scene_fn=make_warehouse, k: int = 8) -> dict:
    scene = scene_fn(seed)
    fp, plans = candidate_plans(scene, WAREHOUSE_MISSIONS, k=k, seed=seed)
    ev = fp.ev
    zeros = np.zeros(ev.n)
    graphs = [plan_graph(ev, a, zeros) for a in plans]
    A = np.stack(plans)
    util = {}
    for mid, m in ALL_MISSIONS.items():
        focus = [ev.car_ids.index(c) for c in m.focus_cars if c in ev.car_ids]
        util[mid] = ev.utility(A, m.weights, focus).astype(np.float32)
    return {"seed": seed, "plans": A, "graphs": graphs, "util": util,
            "car_ids": ev.car_ids, "region": scene.region}


def build(path: Path, seeds: dict[str, list[int]], scene_fn=make_warehouse, log=print) -> dict:
    if path.exists():
        return pickle.loads(path.read_bytes())
    data, t0 = {}, time.time()
    for split, ss in seeds.items():
        data[split] = []
        for i, s in enumerate(ss):
            data[split].append(build_scene(s, scene_fn))
            if (i + 1) % 20 == 0:
                log(f"{split} scenes {i + 1}/{len(ss)} ({time.time() - t0:.0f} s)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(data))
    return data
