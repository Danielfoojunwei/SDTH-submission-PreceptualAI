from __future__ import annotations

import math
import re
from typing import Any, Iterable

_PROVENANCE_HASH = re.compile(r"^[0-9a-fA-F]{64}$")


def bounded_number(value: Any, name: str, minimum: float, maximum: float) -> float:
    """Return a finite float within [minimum, maximum] or raise ValueError."""
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be a finite number between {minimum} and {maximum}")
    return number


def positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return number


def require_finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def non_empty_str(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text or "REPLACE_" in text:
        raise ValueError(f"{name} must be provisioned with a real value")
    return text


def sha256_hex(value: Any, name: str) -> str:
    text = str(value)
    if not _PROVENANCE_HASH.fullmatch(text):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return text.lower()


def frozen_str_set(values: Iterable[Any], name: str) -> frozenset[str]:
    result = frozenset(str(value).strip() for value in values)
    if any(not value for value in result):
        raise ValueError(f"{name} cannot contain empty values")
    return result
