"""Explicit prediction boundary: no future observations or truth-label fields."""
from dataclasses import dataclass
import numpy as np
from .train import predict


@dataclass(frozen=True)
class DeviceInputs:
    context: np.ndarray
    planned_route: np.ndarray
    geometric_prior: np.ndarray


def forecast(models,inputs: DeviceInputs,intent_weights):
    if not isinstance(inputs,DeviceInputs):raise TypeError('Use DeviceInputs; labels are not prediction inputs')
    return predict(models,{'ctx':inputs.context,'route':inputs.planned_route,'prior':inputs.geometric_prior},intent_weights)
