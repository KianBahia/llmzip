"""Cumulative frequency tables for the arithmetic coder.

A table of n symbol frequencies becomes `cum` of length n + 1 with cum[0] = 0 and
cum[n] = total; symbol s owns [cum[s], cum[s+1]). Everything returned to callers
is a Python int, ready to pass to the coder.
"""

import numpy as np

from llmzip.coder import MAX_TOTAL


def validate(freqs, total: int) -> None:
    """Raise ValueError unless every freq >= 1, sum == total and total fits the coder."""
    arr = np.asarray(freqs, dtype=np.int64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("freqs must be a non-empty 1-D sequence")
    if (arr < 1).any():
        raise ValueError(f"every frequency must be >= 1 (min is {int(arr.min())})")
    s = int(arr.sum())
    if s != total:
        raise ValueError(f"frequencies sum to {s}, expected {total}")
    if total > MAX_TOTAL:
        raise ValueError(f"total {total} exceeds coder maximum {MAX_TOTAL}")


def cumulative(freqs, total: int | None = None) -> np.ndarray:
    """int64 array of length n + 1. Validates against `total` if given (else its own sum)."""
    arr = np.asarray(freqs, dtype=np.int64)
    validate(arr, int(arr.sum()) if total is None else total)
    cum = np.zeros(arr.size + 1, dtype=np.int64)
    np.cumsum(arr, out=cum[1:])
    return cum


def symbol_interval(cum, sym: int) -> tuple[int, int]:
    return int(cum[sym]), int(cum[sym + 1])


def find_symbol(cum, target: int) -> int:
    """The symbol s with cum[s] <= target < cum[s+1] (binary search)."""
    return int(np.searchsorted(cum, target, side="right")) - 1
