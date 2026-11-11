"""Mask declared simultaneous scene views, with no same-feature copying."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SimultaneousViews:
    scene_id: str
    input_capture_s: float
    target_capture_s: float
    input_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    capture_tolerance_s: float = 0.
    provenance: str = 'same_packet'


def validate_disjoint_views(views: SimultaneousViews, feature_count: int):
    """Check supplied capture metadata, not its independent authenticity.

    Indices refer to the same recorded RF feature space. Different devices or
    modalities need an explicit common scene/capture alignment upstream.
    """
    if (not isinstance(views, SimultaneousViews) or not isinstance(views.scene_id, str)
            or not views.scene_id or type(feature_count) is not int or feature_count < 2):
        raise ValueError('Use identified scenes and an explicit RF feature grid')
    times = (views.input_capture_s, views.target_capture_s, views.capture_tolerance_s)
    if not all(math.isfinite(t) for t in times) or views.capture_tolerance_s < 0:
        raise ValueError('Capture times and tolerance must be finite')
    if views.provenance not in ('same_packet', 'time_aligned_capture'):
        raise ValueError('Temporal next-step pairs are not simultaneous scene views')
    if (views.provenance == 'same_packet' and views.input_capture_s != views.target_capture_s
            or abs(views.input_capture_s - views.target_capture_s) > views.capture_tolerance_s):
        raise ValueError('Views do not share the declared capture interval')
    left, right = views.input_indices, views.target_indices
    for index in (left, right):
        if (not isinstance(index, tuple) or not index or len(set(index)) != len(index)
                or any(type(k) is not int or not 0 <= k < feature_count for k in index)):
            raise ValueError('Feature indices must be distinct and in range')
    if set(left) & set(right):
        raise ValueError('A recorded feature cannot be both input and prediction target')
    masks = []
    for index in (left, right):
        mask = np.zeros(feature_count, bool); mask[list(index)] = True
        mask.setflags(write=False); masks.append(mask)
    return tuple(masks)


def coherence_mask(group_ids, input_groups, target_groups, *, coordinates=None, minimum_gap=0.):
    """Mask whole AP/band/antenna groups with an explicitly supplied gap.

    A gap is in the coordinate's physical units (Hz, metres or seconds). Its
    relation to coherence is estimated from training data/metadata, not from
    the hidden target values. lambda/2 is not assumed a universal threshold.
    """
    groups = np.asarray(group_ids)
    if groups.ndim != 1 or not len(groups):
        raise ValueError('Use one group identity per recorded RF feature')
    left, right = set(input_groups), set(target_groups)
    if not left or not right or left & right or not (left | right) <= set(groups.tolist()):
        raise ValueError('Whole input and target groups must exist and be disjoint')
    if not math.isfinite(minimum_gap) or minimum_gap < 0:
        raise ValueError('A coherence gap must be finite and nonnegative')
    a, b = np.isin(groups, list(left)), np.isin(groups, list(right))
    if minimum_gap:
        x = np.asarray(coordinates, dtype=float)
        if x.shape not in ((len(groups),), (len(groups), 2), (len(groups), 3)) or not np.isfinite(x).all():
            raise ValueError('A physical gap needs finite per-feature coordinates')
        if x.ndim == 1: x = x[:, None]
        gap = np.linalg.norm(x[a, None] - x[None, b], axis=-1).min()
        if gap < minimum_gap:
            raise ValueError('Mask groups violate the declared physical coherence gap')
    for value in (a, b): value.setflags(write=False)
    return a, b
