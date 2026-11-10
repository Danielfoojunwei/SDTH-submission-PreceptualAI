"""Numerical teachers and baselines. None of them learn; all are exact or
deterministic heuristics, so they can label decisions and serve as
comparators for WirelessJEV."""
from .assign import (milp_assignment, hungarian_access, coloring_assignment,
                     local_search, legacy_policy, strongest_rssi,
                     random_assignment, local_greedy, BASELINES)
from .admission import admission_oracle

__all__ = [
    "milp_assignment", "hungarian_access", "coloring_assignment",
    "local_search", "legacy_policy", "strongest_rssi", "random_assignment",
    "local_greedy", "BASELINES", "admission_oracle",
]
