"""Optical/RF time alignment without treating unreceived frames as observed."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VisualEvent:
    frame_id:str
    captured_seconds:float
    available_seconds:float
    source_sha256:str
    calibration_id:str|None=None

    def __post_init__(self):
        if not self.frame_id or len(self.source_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.source_sha256):raise ValueError('Frame identity and content SHA-256 required')
        if not all(math.isfinite(t) and t>=0 for t in (self.captured_seconds,self.available_seconds)) or self.available_seconds<self.captured_seconds:raise ValueError('Invalid capture/delivery clock')


def latest_visual_event(events,decision_seconds,*,max_age_seconds,calibration_id=None):
    """Newest captured frame actually available now on the SAME calibrated clock.

    Clock synchronization/offsets must be established outside this function.
    No match returns None, never a duplicated or future observation. Capture
    age and availability are distinct; out-of-order delivery remains causal.
    """
    if not math.isfinite(decision_seconds) or decision_seconds<0 or not math.isfinite(max_age_seconds) or max_age_seconds<0:raise ValueError('Invalid event-selection clock')
    usable=[e for e in events if e.available_seconds<=decision_seconds
            and 0<=decision_seconds-e.captured_seconds<=max_age_seconds
            and (calibration_id is None or e.calibration_id==calibration_id)]
    return max(usable,key=lambda e:(e.captured_seconds,e.available_seconds,e.frame_id)) if usable else None


@dataclass(frozen=True)
class CausalPatchFrame:
    event:VisualEvent
    patches:object
    observed:object
    patch_xy:object
    age_seconds:object


def select_dino_observation(batch,decision_seconds,*,feature_available_seconds,max_age_seconds,calibration_id=None):
    """Bridge actual DinoPatchBatch output into a causal RF-time observation.

    Unknown capture times (including the downloaded static photo probes) are
    ineligible for temporal RF fusion. Delivery time is not capture time.
    Caller must supply actual encoding-completion times on the SAME clock,
    separately from sensor delivery. Never present features before computation
    finishes. Returns original-image patch centers and explicit capture ages.
    """
    import torch
    if len(feature_available_seconds)!=len(batch.provenance):raise ValueError('Feature completion timestamps must match delivered frames')
    events=[];indices={}
    for index,provenance in enumerate(batch.provenance):
        ready=float(feature_available_seconds[index]);delivered=float(batch.observation_time_seconds[index])
        if not math.isfinite(ready) or not math.isfinite(delivered) or delivered<0 or ready<delivered:raise ValueError('Encoding cannot finish before valid sensor delivery')
        if not bool(batch.capture_time_known[index]):continue
        event=VisualEvent(provenance['frame_id'],float(batch.capture_time_seconds[index]),
            ready,provenance['canonical_rgb_sha256'],
            provenance.get('calibration_id'))
        if event.frame_id in indices:raise ValueError('Duplicate delivered frame identity')
        events.append(event);indices[event.frame_id]=index
    selected=latest_visual_event(events,decision_seconds,max_age_seconds=max_age_seconds,calibration_id=calibration_id)
    if selected is None:return None
    index=indices[selected.frame_id];tokens=batch.patch_tokens[index]
    boxes=batch.patch_boxes_original_pixels[index].to(dtype=tokens.dtype,device=tokens.device)
    width,height=batch.image_geometry[index]['original_size_wh']
    if width<=0 or height<=0:raise ValueError('Invalid original image geometry')
    xy=(boxes[:,:2]+boxes[:,2:])*.5/boxes.new_tensor([width,height])
    age=torch.full(tokens.shape[:1],decision_seconds-selected.captured_seconds,dtype=tokens.dtype,device=tokens.device)
    return CausalPatchFrame(selected,tokens,batch.patch_valid[index],xy,age)
