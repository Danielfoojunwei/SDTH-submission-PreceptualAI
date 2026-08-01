from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Sequence


class CommandError(RuntimeError):
    """Raised when a required operating-system command fails."""


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int


def run_command(argv: Sequence[str], *, timeout_seconds: float = 10.0) -> CommandResult:
    if not argv:
        raise ValueError("argv cannot be empty")
    try:
        environment = os.environ.copy()
        environment["LC_ALL"] = "C"
        completed = subprocess.run(
            [str(value) for value in argv],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=environment,
        )
    except FileNotFoundError as exc:
        raise CommandError(f"required executable is missing: {argv[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CommandError(f"command timed out after {timeout_seconds}s: {argv[0]}") from exc
    return CommandResult(
        argv=tuple(str(value) for value in argv),
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def run_checked(argv: Sequence[str], *, timeout_seconds: float = 10.0) -> str:
    result = run_command(argv, timeout_seconds=timeout_seconds)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise CommandError(f"command failed ({result.returncode}): {argv[0]}: {detail}")
    return result.stdout


def run_json(argv: Sequence[str], *, timeout_seconds: float = 10.0) -> object:
    output = run_checked(argv, timeout_seconds=timeout_seconds)
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise CommandError(f"command did not return valid JSON: {argv[0]}") from exc
