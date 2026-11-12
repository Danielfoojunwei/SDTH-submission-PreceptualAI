"""Missions that a weight vector cannot express, with exact oracles.

Each mission's utility is a non-linear function of per-robot outcomes over the
planned-route horizon: thresholds, budgets, spatial zones and worst cases.
Train missions have paraphrases (two for training, one held out); NOVEL_NL
missions change the numbers or the zone and never appear in training, so a
model must read them from the text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Outcomes:
    """Per-plan outcomes, arrays over K plans (and n robots)."""
    sat: np.ndarray          # (K, S, n) service per snapshot
    U: np.ndarray            # (K,) mean over snapshots
    jain: np.ndarray         # (K,)
    viol: np.ndarray         # (K,) mean count over snapshots
    moves: np.ndarray        # (K,) reconnections now
    x: np.ndarray            # (n,)
    y: np.ndarray            # (n,)
    video: np.ndarray        # (n,) bool


def _mean_sat(o: Outcomes, mask) -> np.ndarray:
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.ones(len(o.U))
    return o.sat[:, :, mask].mean((1, 2))


def video_floor_then_fair(floor: float):
    def f(o: Outcomes):
        if not o.video.any():
            return o.jain
        worst = o.sat[:, :, o.video].min(2).mean(1)
        return np.where(worst >= floor, o.jain, worst - 1.0)
    return f


def move_budget(limit: int):
    return lambda o: np.where(o.moves <= limit, o.U, o.U - 0.2 * (o.moves - limit))


def zone(pred: Callable[[np.ndarray, np.ndarray], np.ndarray]):
    return lambda o: _mean_sat(o, pred(o.x, o.y)) + 0.1 * o.U


def worst_case(o: Outcomes):
    return o.sat.min(2).mean(1)


def safety_then_data(o: Outcomes):
    return o.U - 10.0 * o.viol / o.sat.shape[2]


NL_MISSIONS = {
    "video_floor_then_fair": (video_floor_then_fair(0.9), [
        "Every robot streaming video must keep at least 90% of its rate; subject to that, share capacity as evenly as possible.",
        "Video robots need 90% of their requested rate or more. Once that holds, spread capacity fairly.",
        "First guarantee each video stream 90% of its rate, then make the allocation as fair as possible."]),
    "move_budget_5": (move_budget(5), [
        "Reconnect at most 5 robots in this plan; within that limit, maximise the total data delivered.",
        "No more than five BSSID switches this round. Inside that budget, push the most data.",
        "You may move up to 5 robots to new access points; use them to maximise fleet throughput."]),
    "zone_front": (zone(lambda x, y: y < 40), [
        "Robots working in the front half of the hall, near shipping (y below 40 m), matter most; maximise their service.",
        "Prioritise the robots in the shipping half of the warehouse (the front 40 metres).",
        "The front half of the hall is where the work is today: serve those robots best."]),
    "zone_back": (zone(lambda x, y: y >= 40), [
        "Robots in the back half of the hall, near receiving (y of 40 m or more), matter most; maximise their service.",
        "Prioritise the robots in the receiving half of the warehouse (the back 40 metres).",
        "Today the work is in the back half of the hall: serve those robots best."]),
    "worst_case": (worst_case, [
        "Maximise the service of the single worst-served robot.",
        "Make the robot with the poorest link as well off as possible.",
        "Focus entirely on the least-served robot in the fleet."]),
    "safety_then_data": (safety_then_data, [
        "First, no safety link may fall below its floor; only then maximise total data.",
        "Safety heartbeats come first. Once every one is above its minimum, maximise throughput.",
        "Keep all safety links above their floors before anything else, then deliver as much data as possible."]),
}
TRAIN_NL = 2   # paraphrases 0,1 train; 2 held out

NOVEL_NL = {
    "video_floor_80": (video_floor_then_fair(0.8), [
        "Each video robot must keep at least 80% of its rate; after that, be as fair as possible.",
        "Guarantee video streams 80% of their rate, then share the rest evenly."]),
    "move_budget_2": (move_budget(2), [
        "Reconnect at most 2 robots this round; within that, maximise total data.",
        "Only two BSSID switches are allowed. Get the most data inside that budget."]),
    "zone_left": (zone(lambda x, y: x < 60), [
        "Robots in the left half of the hall (x below 60 m) matter most; maximise their service.",
        "Prioritise robots working on the left side of the warehouse, the first 60 metres."]),
}


def nl_texts(split: str) -> dict[str, list[str]]:
    if split == "novel":
        return {k: v[1] for k, v in NOVEL_NL.items()}
    if split == "train":
        return {k: v[1][:TRAIN_NL] for k, v in NL_MISSIONS.items()}
    return {k: [v[1][TRAIN_NL]] for k, v in NL_MISSIONS.items()}


def nl_utility(mid: str, o: Outcomes) -> np.ndarray:
    table = {**NL_MISSIONS, **NOVEL_NL}
    return table[mid][0](o).astype(np.float32)
