"""Observable-only inputs to an optional roaming gate and its diagnostic trace."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RoamDecision:
    decision: int
    current: np.ndarray
    candidate: np.ndarray
    gain: np.ndarray
    eligible: np.ndarray
    scores: np.ndarray

    @classmethod
    def capture(cls,decision,current,candidate,gain,eligible,scores):
        arrays=[np.array(x,copy=True) for x in [current,candidate,gain,eligible,scores]]
        for a in arrays:a.flags.writeable=False
        return cls(decision,*arrays)


def checked_mask(result,n):
    a=np.asarray(result)
    if a.dtype!=np.bool_ or a.shape!=(n,):raise ValueError('Roam gate must return one boolean per robot')
    return a
