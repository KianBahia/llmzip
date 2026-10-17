"""Logits -> integer frequency table. The ONLY such path; encoder and decoder share it.

Scheme (deterministic given identical logits):
  p      = softmax(float64(logits)), computed stably (subtract max, exp, divide by sum)
  budget = TOTAL - V                  (reserves frequency 1 for every token)
  freqs  = floor(p * budget) + 1
  freqs[argmax(p)] += TOTAL - sum(freqs)   (np.argmax picks the first index on ties)
Since sum(floor(p * budget)) lies in (budget - V, budget], the remainder is in [0, V].
"""

import numpy as np
import torch

from llmzip.config import FREQ_TOTAL_BITS

TOTAL = 1 << FREQ_TOTAL_BITS

# Part of the model fingerprint. Bump the version whenever this module's output
# can change, so files made with the old scheme are rejected instead of misdecoded.
PROBS_SCHEME = "floor+1, remainder-to-argmax, v1"


def softmax64(logits: torch.Tensor) -> np.ndarray:
    x = logits.detach().numpy().astype(np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def logits_to_freqs(logits: torch.Tensor) -> np.ndarray:
    """int64 frequencies, each >= 1, summing to exactly 2**FREQ_TOTAL_BITS."""
    if logits.ndim != 1:
        raise ValueError(f"expected 1-D logits, got shape {tuple(logits.shape)}")
    p = softmax64(logits)
    v = p.size
    budget = TOTAL - v
    freqs = np.floor(p * budget).astype(np.int64) + 1
    remainder = TOTAL - int(freqs.sum())
    assert 0 <= remainder <= v, f"remainder {remainder} out of range"
    freqs[int(np.argmax(p))] += remainder
    assert int(freqs.min()) >= 1 and int(freqs.sum()) == TOTAL
    return freqs
