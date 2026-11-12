"""Shared observable packet-value controller with explicit ten-second holds.

The ledger records nominal interruption exposure and a model's action-value
reservation once per executed switch. Neither is a bound on factual cumulative
misses or on the unavailable RadioMap counterfactual after actions diverge.
"""
from collections import deque
import copy
import time

import numpy as np
import torch

from .features import PublicHistory, action_queries, NumericPacketValue, immediate_mask


class PacketController:
    def __init__(self, g, cost, model=None, correction=0., head_seconds=.01):
        if not np.isfinite(cost) or not 0 <= cost <= 10:
            raise ValueError('A switch cost must fit the ten-second action horizon')
        if not np.isfinite(correction) or correction < 0:
            raise ValueError('Use a finite nonnegative admission correction')
        if not np.isfinite(head_seconds) or head_seconds < 0:
            raise ValueError('Use a finite nonnegative charged head duration')
        self.geometry, self.cost = g, float(cost)
        self.model, self.correction = model, float(correction)
        self.head_seconds = float(head_seconds)
        self.history = PublicHistory(g, cost)
        self.numeric = NumericPacketValue(cost) if model is None else None
        if model is not None:
            model.eval()
        self.observations = []
        self.decisions = []
        self.executions = []
        self.head_runtimes = []
        self.next_decision = None
        self.pending = deque()
        self.previous_ready_seconds = 0.
        self.published = 0
        self._available_reservations = {}
        self._booked_executions = set()
        self.known_outage_seconds = 0.
        self.booked_predicted_delta_misses = 0.

    @torch.no_grad()
    def observe(self, public):
        """Assimilate one public second; no head job without a ready forecast."""
        slot = int(public['slot'])
        if slot % 100:
            raise ValueError('Controller observations must occur at whole seconds')
        frame = self.history.update(public)
        self.observations.append({'slot': slot, 'frame': np.array(frame, copy=True)})
        current = np.asarray(public['current'], dtype=int)
        n = len(current)
        if self.next_decision is None:
            self.next_decision = np.zeros(n, dtype=int)
        elif self.next_decision.shape != (n,):
            raise ValueError('Fleet membership changed; reset the controller')
        if public['forecast'] is None:
            return
        available = ((np.asarray(public['outage_remaining']) <= 0)
                     & (np.asarray(public['scan_remaining']) <= 0))
        due = (slot >= self.next_decision) & available
        if not due.any():
            return
        started = time.perf_counter()
        if self.model is None:
            mean = self.numeric(public)
        else:
            context = torch.as_tensor(self.history.context(), dtype=torch.float32)
            query = torch.as_tensor(action_queries(self.geometry, public, self.cost), dtype=torch.float32)
            mean = self.model.forward_actions(context, query)['delta_misses'].detach().cpu().numpy()
        mean = np.asarray(mean, dtype=float)
        expected_shape = (n, len(self.geometry.sites))
        if mean.shape != expected_shape or not np.isfinite(mean).all():
            raise ValueError('Packet values must be finite N x AP action differences')
        score = mean+self.correction
        score = np.where(immediate_mask(self.geometry, public), score, np.inf)
        ar = np.arange(n)
        score[ar, current] = 0.
        target = score.argmin(1).astype(int)
        delta = score[ar, target]
        eligible = due & (target != current) & (delta < -1.)
        # A rejected/non-due action is an explicit stay, not an untracked hint.
        target = np.where(eligible, target, current)
        delta = np.where(eligible, delta, 0.)
        elapsed = time.perf_counter()-started
        self.head_runtimes.append(elapsed)
        issue_seconds = slot/100
        self.previous_ready_seconds = max(issue_seconds, self.previous_ready_seconds)+self.head_seconds
        ready = int(np.ceil(self.previous_ready_seconds*100-1e-10))
        self.next_decision[due] = ready+1000
        for key in [key for key in self._available_reservations if due[key[0]]]:
            del self._available_reservations[key]
        job_id = len(self.decisions)
        self.pending.append({'id': job_id, 'ready_slot': ready, 'issued_slot': slot,
                             'current': current.copy(), 'target': target.copy(),
                             'eligible': eligible.copy(), 'delta': delta.copy()})
        self.decisions.append({'id': job_id, 'issued_slot': slot, 'ready_slot': ready,
                               'committed_robots': np.flatnonzero(due).tolist(),
                               'current': current.tolist(), 'target': target.tolist(),
                               'eligible': eligible.tolist(), 'corrected_delta_misses': delta.tolist(),
                               'actual_head_seconds': elapsed, 'charged_head_seconds': self.head_seconds})

    def take(self, slot):
        """Return the oldest completed FIFO proposal, never a future result."""
        slot = int(slot)
        if not self.pending or self.pending[0]['ready_slot'] > slot:
            return None
        job = self.pending.popleft()
        valid_until = job['issued_slot']+100
        eligible = job['eligible'].copy()
        if slot >= valid_until:
            eligible[:] = False
        self.decisions[job['id']]['published_slot'] = slot
        self.decisions[job['id']]['valid_until'] = valid_until
        self.decisions[job['id']]['expired_at_publication'] = slot >= valid_until
        for i in np.flatnonzero(eligible):
            key = (int(i), int(job['target'][i]))
            self._available_reservations[key] = {'decision_id': job['id'],
                                               'issued_slot': job['issued_slot'],
                                               'ready_slot': job['ready_slot'],
                                               'valid_until': valid_until,
                                               'delta': float(job['delta'][i])}
        self.published += 1
        return {'current': job['current'].copy(), 'target': job['target'].copy(),
                'eligible': eligible, 'valid_until': valid_until}

    def executed(self, slot, ids, targets):
        """Book actual switches once, including cadence-delayed execution."""
        slot = int(slot)
        ids, targets = np.asarray(ids, dtype=int), np.asarray(targets, dtype=int)
        if (self.next_decision is None or ids.ndim != 1 or targets.shape != ids.shape
                or (ids < 0).any() or (ids >= len(self.next_decision)).any()
                or (targets < 0).any() or (targets >= len(self.geometry.sites)).any()):
            raise ValueError('Invalid executed primary-radio actions')
        for robot, target in zip(ids, targets, strict=True):
            robot, target = int(robot), int(target)
            event_key = (slot, robot, target)
            if event_key in self._booked_executions:
                continue
            reservation = self._available_reservations.pop((robot, target), None)
            if reservation is not None and slot >= reservation['valid_until']:
                reservation = None  # A stale forecast cannot earn a fresh credit.
            delta = None if reservation is None else reservation['delta']
            if delta is not None and delta >= -1.:
                raise RuntimeError('Only admitted negative action values may be reserved')
            self._booked_executions.add(event_key)
            self.next_decision[robot] = slot+1000
            self.history.executed(slot, np.array([robot]))
            self.known_outage_seconds += self.cost
            if delta is not None:
                self.booked_predicted_delta_misses += delta
            self.executions.append({'slot': slot, 'robot': robot, 'target': target,
                                    'decision_id': None if reservation is None else reservation['decision_id'],
                                    'nominal_outage_seconds': self.cost,
                                    'nominal_command_equivalent_exposure': self.cost*20,
                                    'booked_predicted_delta_misses': delta})
            # A later actual switch cannot spend an older unexecuted reservation.
            for key in [key for key in self._available_reservations if key[0] == robot]:
                del self._available_reservations[key]

    def summary(self):
        return {'family': 'numeric' if self.model is None else self.model.config.get('family', 'learned'),
                'correction_commands': self.correction, 'charged_head_seconds': self.head_seconds,
                'observations': len(self.observations), 'jobs_issued': len(self.decisions),
                'jobs_published': self.published, 'jobs_pending': len(self.pending),
                'decisions': copy.deepcopy(self.decisions), 'executions': copy.deepcopy(self.executions),
                'head_prediction_seconds': list(self.head_runtimes),
                'known_outage_seconds': self.known_outage_seconds,
                'nominal_command_equivalent_exposure': self.known_outage_seconds*20,
                'booked_predicted_delta_misses': self.booked_predicted_delta_misses,
                'not_factual_guarantee': True,
                'ledger_scope': 'Nominal handover exposure and once-only predicted reservations; no bound on realized excess misses or recovery of past losses'}
