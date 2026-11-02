"""Mission text -> cross-layer policy for the native stack.

WirelessJEV's intent reader (frozen d3-lite, `wirelessjev.intent`) turns the
operator's words into nine objective weights and the robots it names. This
module compiles those into concrete knobs at every layer the stack controls:

| Weight | Knob it sets |
|---|---|
| ho (avoid reconnections) | planned-roam gain threshold |
| fair (no robot starved) | per-AP load penalty in roam planning |
| focus (named robot) | second radio and full video share reserved for it; never power-reduced |
| viol (safety floors) | trust-gate z (more caution before acting on the map) |
| prot (protected equipment) | penalty on links whose channel carries scanner traffic |
| W (interference low) | transmit-power politeness on/off |
| V (each robot's rate) | adaptive-video share of predicted goodput |
"""
from __future__ import annotations

from dataclasses import replace
import re

from .sim import Policy, policies


def robot_indices(names) -> tuple[int, ...]:
    out = []
    for name in names:
        m = re.fullmatch(r"amr-(\d{3})", name)
        if m:
            out.append(int(m.group(1)) - 1)
    return tuple(out)


def intent_policy(weights: dict, focus_names=(), *, dual_hardware: str = "focus") -> Policy:
    """dual_hardware: "all" (every robot has a second radio), "focus" (only
    robots the mission names get one, as when adapters are scarce), "none"."""
    base = {p.name: p for p in policies()}["native"]
    focus = robot_indices(focus_names) if weights.get("focus", 0) >= 2.0 else ()
    return replace(
        base, name="native_intent", gate=True, online_map=True, cadence_roam=True,
        adaptive_video=True,
        roam_gain_db=5.0 + 1.5 * weights["ho"],
        load_penalty=0.15 + 0.2 * weights["fair"],
        gate_z=1.5 + 0.25 * weights["viol"],
        avoid_occupied=min(1.0, weights["prot"] / 5.0),
        power_control=weights["W"] >= 0.3,
        video_share=0.4 + 0.15 * weights["V"],
        focus_robots=focus,
        dual_radio=dual_hardware == "all" or (dual_hardware == "focus" and bool(focus)),
        dual_focus_only=dual_hardware == "focus")
