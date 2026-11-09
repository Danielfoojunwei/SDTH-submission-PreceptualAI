"""Real executor fault/recovery contracts; no radio or device actions."""
from concurrent.futures import wait,ThreadPoolExecutor
from threading import Event,Thread
import time
import numpy as np
import pytest

from wirelessjev.fleetwm.data import Observations
from wirelessjev.frontier_control import AsyncForecastService
from wirelessjev.packet_control import ShadowPacketAdmission
from wirelessjev.native.gate import RoamDecision


def observations(t=60,n=2,sites=2):
    links=sites*3
    return Observations(np.zeros((t,n,2),dtype='f4'),np.ones((t,n,links),dtype='f4'),
                        np.ones((t,n,links),dtype=bool),np.full((t,n,links),-90.,dtype='f4'),
                        np.ones((t,n),dtype=bool))


def route(n=2):
    return np.full((n,9,2),2.,dtype='f4')


class Forecaster:
    def __init__(self,mode='valid',release=None,callback=None):
        self.mode=mode;self.release=release;self.callback=callback;self.entered=Event()
        self.input_values=None;self.output=None

    def forecast(self,history,plan):
        self.entered.set()
        if self.release is not None:self.release.wait()
        if self.callback is not None:self.callback()
        if self.mode=='error':raise ValueError('worker unavailable')
        if self.mode=='base_error':raise KeyboardInterrupt('worker failure')
        self.input_values=(float(history.value[0,0,0]),float(plan[0,0,0]))
        shape=(len(plan),9,history.value.shape[2]//3,3)
        mean=np.full(shape,sum(self.input_values),dtype='f4');std=np.ones(shape,dtype='f4')
        metadata={'family':'test_worker','history_cutoff_exclusive':len(history.value)}
        if self.mode=='nan_mean':mean.flat[0]=np.nan
        if self.mode=='inf_std':std.flat[0]=np.inf
        if self.mode=='negative_std':std.flat[0]=-1
        if self.mode=='shape':mean=mean[:,:1]
        if self.mode=='metadata':metadata['extra']=np.ones(100)
        if self.mode=='cutoff':metadata['history_cutoff_exclusive']+=1
        if self.mode=='nonfinite_metadata':metadata['duration']=np.nan
        self.output=(mean,std)
        return mean,std,metadata


def finished(future):
    done,_=wait([future],timeout=3)
    assert future in done,'real worker did not complete'


def eventually_closed(service):
    deadline=time.monotonic()+2
    while not service.close():
        assert time.monotonic()<deadline,'released worker did not exit'
        time.sleep(.001)


@pytest.mark.parametrize('mode',['error','base_error','nan_mean','inf_std','negative_std','shape','metadata','cutoff','nonfinite_metadata'])
def test_forecast_faults_disable_the_worker_and_return_baseline_fallback(mode):
    service=AsyncForecastService(Forecaster(mode))
    try:
        assert service.submit(observations(),route(),60.)
        finished(service.pending[0][1])
        assert service.poll(60.) is None and service.failed
        assert not service.submit(observations(61),route(),61.)
        assert service.events[-1]['status']=='worker_failure_fallback'
    finally:service.close()


def test_forecast_clock_freshness_and_snapshot_isolation():
    release=Event();model=Forecaster(release=release)
    service=AsyncForecastService(model,max_forecast_age_s=1.)
    history=observations();plan=route()
    try:
        assert service.submit(history,plan,60.) and model.entered.wait(1)
        future=service.pending[0][1]
        history.value[:]=100.;plan[:]=200.
        release.set();finished(future)
        assert model.input_values==(1.,2.)
        # Returned source buffers are also isolated from the worker's arrays.
        model.output[0][:]=999.
        assert service.poll(59.9) is None
        mean,_,metadata=service.poll(60.)
        np.testing.assert_array_equal(mean,np.full(mean.shape,3.,dtype='f4'))
        assert metadata['history_cutoff_exclusive']==60
        assert service.poll(61.01) is None
        assert service.events[-1]['status']=='expired_forecast_fallback'
        assert service.poll(60.5) is None
        assert service.events[-1]['status']=='backward_clock_fallback'
        for bad in (np.nan,np.inf,True,'62'):
            assert service.poll(bad) is None
    finally:release.set();service.close()


@pytest.mark.parametrize('bad',[0,-1,1.5,True,np.nan])
def test_forecast_pending_bound_is_a_positive_integer(bad):
    with pytest.raises(ValueError):AsyncForecastService(Forecaster(),max_pending=bad)


def test_invalid_history_limits_and_backward_submission_do_not_reach_the_model():
    model=Forecaster();service=AsyncForecastService(model,max_history_steps=61,max_events=3)
    try:
        assert not service.submit(observations(62),route(),62.)
        history=observations();history.positions.flat[0]=np.nan
        assert not service.submit(history,route(),60.)
        history=observations()
        history=Observations(history.positions,history.value,np.zeros((60,2,5),dtype=bool),history.bound,history.scanned)
        assert not service.submit(history,route(),60.)
    finally:service.close()
    tiny=AsyncForecastService(model,max_snapshot_bytes=1)
    try:assert not tiny.submit(observations(),route(),60.)
    finally:tiny.close()
    service=AsyncForecastService(Forecaster(),max_events=3)
    try:
        assert service.submit(observations(),route(),60.)
        finished(service.pending[0][1]);assert service.poll(60.) is not None
        assert not service.submit(observations(),route(),59.)
        for _ in range(10):assert not service.submit(observations(),route(),np.nan)
        assert len(service.events)<=3
    finally:service.close()


def test_forecast_close_is_nonblocking_idempotent_and_cannot_kill_a_thread():
    release=Event();model=Forecaster(release=release);service=AsyncForecastService(model,max_pending=1)
    try:
        assert service.submit(observations(),route(),60.) and model.entered.wait(1)
        future=service.pending[0][1]
        assert not service.submit(observations(61),route(),61.)
        start=time.monotonic();assert service.close() is False
        assert time.monotonic()-start<.2
        assert service.close() is False and service.poll(60.) is None
        assert not service.submit(observations(),route(),60.)
        release.set();finished(future);eventually_closed(service)
    finally:release.set();service.close()


def test_forecast_worker_can_request_close_without_joining_itself():
    model=Forecaster();service=AsyncForecastService(model);model.callback=service.close
    try:
        assert service.submit(observations(),route(),60.)
        with service._lock:
            future=(service.pending[0][1] if service.pending else service._closing_futures[0])
        finished(future)
        assert service.closed and service.poll(60.) is None
    finally:service.close()


def test_forecast_executor_submission_failure_is_a_fallback():
    service=AsyncForecastService(Forecaster())
    service.pool.shutdown(wait=False,cancel_futures=True)
    try:
        assert not service.submit(observations(),route(),60.)
        assert service.poll(60.) is None and service.failed
        assert service.events[-1]['status']=='submission_failure_fallback'
    finally:service.close()


def test_concurrent_forecast_submissions_share_one_bounded_capacity():
    release=Event();model=Forecaster(release=release);service=AsyncForecastService(model,max_pending=2)
    try:
        assert service.submit(observations(),route(),60.) and model.entered.wait(1)
        running=service.pending[0][1]
        with ThreadPoolExecutor(max_workers=8) as callers:
            accepted=list(callers.map(lambda _:service.submit(observations(61),route(),61.),range(40)))
        assert sum(accepted)==1 and len(service.pending)==2 and service._reservations==0
        queued=service.pending[1][1]
        assert service.close() is False
        release.set();finished(running);finished(queued);eventually_closed(service)
    finally:release.set();service.close()


def test_forecast_close_never_runs_blocking_future_callbacks_on_its_caller():
    release=Event();callback_release=Event();callback_started=Event();close_done=Event()
    service=AsyncForecastService(Forecaster(release=release),max_pending=2)
    closer=None
    try:
        assert service.submit(observations(),route(),60.) and service.model.entered.wait(1)
        assert service.submit(observations(61),route(),61.)
        queued=service.pending[1][1]
        def callback(future):callback_started.set();callback_release.wait()
        queued.add_done_callback(callback)
        def shutdown():service.close();close_done.set()
        closer=Thread(target=shutdown);closer.start()
        assert close_done.wait(.2),'close ran a queued cancellation callback synchronously'
        release.set();assert callback_started.wait(2)
        assert service.close() is False  # Future.done alone does not prove its thread exited.
        callback_release.set();finished(queued);eventually_closed(service)
    finally:
        release.set();callback_release.set()
        if closer is not None:closer.join(2)
        service.close()


def context():
    return RoamDecision.capture(1,np.array([0,1]),np.array([1,0]),np.array([6.,7.]),
                                np.array([True,True]),np.zeros((2,2)))


def test_shadow_inputs_are_isolated_and_closed_calls_keep_the_baseline():
    original=context();baseline=np.array([True,False])
    def mutate(copy):
        copy.current.flags.writeable=True;copy.current[:]=99
        copy.eligible.flags.writeable=True;copy.eligible[:]=False
        return np.zeros(2,dtype=bool)
    shadow=ShadowPacketAdmission(lambda c:baseline,mutate)
    try:
        returned=shadow(original);future=shadow.pending[1];finished(future)
        np.testing.assert_array_equal(returned,baseline)
        np.testing.assert_array_equal(original.current,[0,1])
        np.testing.assert_array_equal(original.eligible,[True,True])
        assert returned is not baseline
        eventually_closed(shadow)
        np.testing.assert_array_equal(shadow(original),baseline)
        assert shadow.observations[-1]['critic_status']=='closed_baseline'
    finally:shadow.close()


def test_shadow_executor_failure_and_snapshot_limit_preserve_baseline():
    mask=np.array([False,True]);c=context()
    shadow=ShadowPacketAdmission(lambda c:mask,lambda c:np.ones(2,dtype=bool))
    shadow.worker.shutdown(wait=False,cancel_futures=True)
    try:
        np.testing.assert_array_equal(shadow(c),mask)
        assert shadow.observations[-1]['critic_status']=='submission_error_fallback'
    finally:shadow.close()

    limited=ShadowPacketAdmission(lambda c:mask,lambda c:None,max_context_bytes=1)
    try:
        np.testing.assert_array_equal(limited(c),mask)
        assert limited.pending is None
    finally:limited.close()
    malformed=RoamDecision.capture('invalid',c.current,c.candidate,c.gain,c.eligible,c.scores)
    shadow=ShadowPacketAdmission(lambda c:mask,lambda c:None)
    try:
        np.testing.assert_array_equal(shadow(malformed),mask)
        assert shadow.pending is None
    finally:shadow.close()


def test_missing_shadow_metadata_does_not_interrupt_a_valid_baseline():
    class ContextWithoutDecision:
        current=np.array([0,1])
    mask=np.array([True,False])
    shadow=ShadowPacketAdmission(lambda c:mask,lambda c:None)
    try:
        np.testing.assert_array_equal(shadow(ContextWithoutDecision()),mask)
        assert shadow.pending is None
        eventually_closed(shadow)
        np.testing.assert_array_equal(shadow(ContextWithoutDecision()),mask)
    finally:shadow.close()


def test_shadow_hung_worker_has_bounded_telemetry_and_nonblocking_close():
    entered=Event();release=Event();mask=np.array([True,False])
    def hung(c):entered.set();release.wait();return np.zeros(2,dtype=bool)
    shadow=ShadowPacketAdmission(lambda c:mask,hung,max_observations=4)
    try:
        np.testing.assert_array_equal(shadow(context()),mask);assert entered.wait(1)
        future=shadow.pending[1]
        start=time.monotonic()
        for _ in range(30):np.testing.assert_array_equal(shadow(context()),mask)
        assert time.monotonic()-start<.2 and len(shadow.observations)==4
        assert shadow.pending[1] is future
        start=time.monotonic();assert shadow.close() is False
        assert time.monotonic()-start<.2
        np.testing.assert_array_equal(shadow(context()),mask)
        assert shadow.close() is False
        release.set();finished(future);eventually_closed(shadow)
    finally:release.set();shadow.close()


def test_shadow_callback_and_base_exception_cannot_change_actuation():
    mask=np.array([True,False]);holder={}
    def callback(c):holder['shadow'].close();raise SystemExit('critic stopped')
    shadow=ShadowPacketAdmission(lambda c:mask,callback);holder['shadow']=shadow
    try:
        np.testing.assert_array_equal(shadow(context()),mask)
        future=shadow.pending[1];finished(future)
        np.testing.assert_array_equal(shadow(context()),mask)
        assert shadow.closed
        assert shadow.observations[-2]['active_override'] is False
    finally:shadow.close()


def test_shadow_close_reports_a_worker_stuck_in_a_future_callback():
    release=Event();entered=Event();callback_release=Event();callback_started=Event()
    mask=np.array([True,False])
    def critic(c):entered.set();release.wait();return np.zeros(2,dtype=bool)
    shadow=ShadowPacketAdmission(lambda c:mask,critic)
    try:
        np.testing.assert_array_equal(shadow(context()),mask);assert entered.wait(1)
        future=shadow.pending[1]
        def callback(f):callback_started.set();callback_release.wait()
        future.add_done_callback(callback)
        release.set();assert callback_started.wait(2)
        start=time.monotonic();assert shadow.close() is False
        assert time.monotonic()-start<.2
        np.testing.assert_array_equal(shadow(context()),mask)
        callback_release.set();eventually_closed(shadow)
    finally:release.set();callback_release.set();shadow.close()
