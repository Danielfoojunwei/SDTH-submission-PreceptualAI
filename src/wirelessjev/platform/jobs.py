"""Run one command at a time as a subprocess and keep its output; feed waypoint marks to laps."""
from __future__ import annotations

from collections import deque
import itertools
import os
import subprocess
import threading
import time


class Job:
    def __init__(self, job_id: int, step_id: str, argv: list[str], shown: str):
        self.id, self.step_id, self.argv, self.shown = job_id, step_id, argv, shown
        self.lines: deque[str] = deque(maxlen=4000)
        self.started, self.ended, self.returncode = time.time(), None, None
        self.proc: subprocess.Popen | None = None
        self.marks: list[dict] = []
        self.lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.ended is None

    def summary(self, tail: int = 200) -> dict:
        with self.lock:
            lines = list(self.lines)[-tail:]
        return {"id": self.id, "step": self.step_id, "command": self.shown, "running": self.running,
                "returncode": self.returncode, "started": self.started, "ended": self.ended,
                "seconds": round((self.ended or time.time()) - self.started, 1), "marks": self.marks, "output": lines}


class Runner:
    """At most one job at a time: the car takes one request at a time."""

    def __init__(self, cwd, env_extra: dict | None = None, on_finish=None):
        self.cwd, self.env_extra, self.on_finish = cwd, env_extra or {}, on_finish
        self.jobs: list[Job] = []
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def current(self) -> Job | None:
        return next((j for j in reversed(self.jobs) if j.running), None)

    def start(self, step_id: str, argv: list[str], shown: str) -> Job:
        with self._lock:
            if self.current() is not None:
                raise RuntimeError("another command is still running; wait for it or stop it first")
            job = Job(next(self._ids), step_id, argv, shown)
            env = {**os.environ, **self.env_extra, "PYTHONUNBUFFERED": "1"}
            job.proc = subprocess.Popen(argv, cwd=self.cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            self.jobs.append(job)
        threading.Thread(target=self._pump, args=(job,), daemon=True).start()
        return job

    def _pump(self, job: Job) -> None:
        for line in job.proc.stdout:
            with job.lock:
                job.lines.append(line.rstrip("\n"))
        job.returncode = job.proc.wait()
        job.ended = time.time()
        try:
            job.proc.stdin.close()
        except OSError:
            pass
        if self.on_finish:
            self.on_finish(job)

    def mark(self, label: str) -> dict:
        job = self.current()
        if job is None or "lap" not in job.argv:
            raise RuntimeError("marks only go to a lap that is recording")
        job.proc.stdin.write(label + "\n")
        job.proc.stdin.flush()
        rec = {"label": label, "t": time.time()}
        job.marks.append(rec)
        return rec

    def stop(self) -> bool:
        job = self.current()
        if job is None:
            return False
        job.proc.terminate()
        return True

    def get(self, job_id: int) -> Job | None:
        return next((j for j in self.jobs if j.id == job_id), None)
