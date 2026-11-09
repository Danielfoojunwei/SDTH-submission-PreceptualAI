"""Keep unqualified packet critics outside the baseline's actuation path."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from threading import RLock,current_thread
import numpy as np
from .native.gate import checked_mask,RoamDecision

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
    def __init__(self,baseline,critic,*,max_observations=128,max_context_bytes=1024*1024):
        for name,value in [('max_observations',max_observations),('max_context_bytes',max_context_bytes)]:
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<1:
                raise ValueError(f'{name} must be a positive integer')
        self.baseline,self.critic=baseline,critic
        self.max_context_bytes=int(max_context_bytes)
        self.observations=deque(maxlen=int(max_observations))
        self.worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix='packet-shadow')
        self.pending=None;self.closed=False;self._lock=RLock();self._worker_thread=None

    def _snapshot(self,context):
        if not isinstance(context,RoamDecision):raise ValueError('Use an explicit roaming decision')
        if (isinstance(context.decision,(bool,np.bool_))
                or not isinstance(context.decision,(int,np.integer))):
            raise ValueError('Decision index must be an integer')
        arrays=(context.current,context.candidate,context.gain,context.eligible,context.scores)
        if any(type(a) is not np.ndarray for a in arrays):raise ValueError('Expected decision arrays')
        n=len(context.current)
        if (sum(a.nbytes for a in arrays)>self.max_context_bytes
                or any(a.shape!=(n,) for a in arrays[:4]) or context.scores.ndim!=2
                or context.scores.shape[0]!=n or context.eligible.dtype!=np.bool_):
            raise ValueError('Decision snapshot exceeds its limit or has invalid dimensions')
        for a in arrays:
            if ((not np.issubdtype(a.dtype,np.number) and a.dtype!=np.bool_)
                    or np.issubdtype(a.dtype,np.complexfloating)):
                raise ValueError('Decision fields must be numeric')
            if not np.isfinite(a).all():raise ValueError('Nonfinite decision field')
        return RoamDecision.capture(context.decision,*arrays)

    def _evaluate(self,context):
        with self._lock:
            self._worker_thread=current_thread()
            if self.closed:return {'critic_status':'closed_skipped'}
        try:
            advice=self.critic(context)
            if advice is None:return {'critic_status':'abstain'}
            mask=checked_mask(advice,len(context.current))
            return {'critic_status':'shadow_only','critic_approvals':int(mask.sum())}
        except BaseException as error:
            return {'critic_status':'error_fallback','error_type':type(error).__name__[:80]}

    def _collect(self):
        if self.pending is None or not self.pending[1].done():return
        record,future=self.pending
        try:record.update(future.result())
        except Exception as error:record.update(critic_status='error_fallback',error_type=type(error).__name__)
        if self.closed:record['completed_after_close']=True
        self.pending=None

    def __call__(self,context):
        # The authoritative baseline is evaluated and copied before any shadow
        # work. Critic state, closure and executor failures cannot replace it.
        baseline=np.array(checked_mask(self.baseline(context),len(context.current)),copy=True)
        try:decision=context.decision
        except Exception:decision=None
        decision=(int(decision) if isinstance(decision,(int,np.integer))
                  and not isinstance(decision,(bool,np.bool_)) and abs(int(decision))<=2**63-1 else None)
        record={'decision':decision,'baseline_approvals':int(baseline.sum()),'active_override':False}
        with self._lock:
            self._collect()
            if self.closed:record['critic_status']='closed_baseline'
            elif self.pending is not None:record['critic_status']='busy_skipped'
            else:
                try:
                    snapshot=self._snapshot(context)
                    future=self.worker.submit(self._evaluate,snapshot)
                except Exception as error:
                    record.update(critic_status='submission_error_fallback',error_type=type(error).__name__[:80])
                else:
                    record['critic_status']='queued';self.pending=(record,future)
            self.observations.append(record)
        return baseline

    def close(self):
        """Never join a hung critic; False reports a still-running thread.

        The thread cannot be killed here, and Python may wait for it during
        interpreter shutdown. Use process isolation for enforceable termination.
        """
        with self._lock:
            first=not self.closed;self.closed=True
        if first:self.worker.shutdown(wait=False,cancel_futures=False)
        with self._lock:
            self._collect()
            alive=self._worker_thread is not None and self._worker_thread.is_alive()
            return self.pending is None and not alive
