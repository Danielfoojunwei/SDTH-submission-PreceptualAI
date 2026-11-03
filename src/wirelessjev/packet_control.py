"""Keep unqualified packet critics outside the baseline's actuation path."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .native.gate import checked_mask

ROOT=Path(__file__).resolve().parents[2]


def require_confirmed_packet_profile():
    path=ROOT/'artifacts/packet_value/summary.json'
    if not path.exists():
        raise RuntimeError('Packet JEPA has no completed independent qualification')
    evidence=json.loads(path.read_text())
    if (evidence.get('status')!='evaluated' or not evidence.get('research_promotion_criterion_passed')
            or not evidence.get('primary_success') or not evidence.get('additional_promotion_success')):
        raise RuntimeError('Packet JEPA is unqualified; retain canonical RadioMap control')
    return evidence


class ShadowPacketAdmission:
    """Observe critic advice while executing the baseline's admission exactly.

    A wide/absent estimate, rejection or critic exception never means "stay".
    There is deliberately no active-override mode: the rejected ten-second
    labels do not validate baseline-continuation overrides. Such a mode needs
    its own branch contract, development evaluation and independent evidence.
    Shadow work uses one bounded background worker. CPU contention still needs
    hardware qualification; admission never waits for that worker.
    """
    def __init__(self,baseline,critic):
        self.baseline,self.critic=baseline,critic
        self.observations=[]
        self.worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix='packet-shadow')
        self.pending=None

    def _evaluate(self,context):
        advice=self.critic(context)
        if advice is None:return {'critic_status':'abstain'}
        mask=checked_mask(advice,len(context.current))
        return {'critic_status':'shadow_only','critic_approvals':int(mask.sum())}

    def _collect(self):
        if self.pending is None or not self.pending[1].done():return
        record,future=self.pending
        try:record.update(future.result())
        except Exception as error:record.update(critic_status='error_fallback',error_type=type(error).__name__)
        self.pending=None

    def __call__(self,context):
        baseline=np.array(checked_mask(self.baseline(context),len(context.current)),copy=True)
        record={'decision':int(context.decision),'baseline_approvals':int(baseline.sum()),'active_override':False}
        self._collect()
        if self.pending is None:
            record['critic_status']='queued';self.pending=(record,self.worker.submit(self._evaluate,context))
        else:record['critic_status']='busy_skipped'
        self.observations.append(record)
        return baseline

    def close(self):
        """Join only outside actuation, during explicit shutdown."""
        self.worker.shutdown(wait=True);self._collect()
