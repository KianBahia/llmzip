import subprocess
import sys

import pytest
import torch

from llmzip import config
from llmzip.model import NextTokenModel, logits_digest

TEXT = "The quick brown fox jumps over the lazy dog. It was a sunny day in the park."


@pytest.fixture(scope="module")
def model():
    return NextTokenModel()


def test_model_loads(model):
    assert model.vocab_size == 50257
    assert torch.get_num_threads() == config.NUM_THREADS
    assert not model.model.training


def test_next_logits_shape(model):
    first = model.start()
    nxt = model.next_logits(model.encode("Hello")[0])
    for t in (first, nxt):
        assert t.shape == (model.vocab_size,)
        assert t.dtype == torch.float32


def test_two_runs_bit_identical(model):
    ids = model.encode(TEXT)
    a = [t.numpy().tobytes() for t in model.logits_sequence(ids)]
    b = [t.numpy().tobytes() for t in model.logits_sequence(ids)]
    assert len(a) == len(ids)
    assert a == b


def test_fingerprint_stable(model):
    fp = model.model_fingerprint()
    assert len(fp) == 16
    assert fp == model.model_fingerprint()


def test_context_window_guard(model):
    model.start()
    model._pos = model.max_positions
    with pytest.raises(ValueError):
        model.next_logits(0)
    model.reset()


@pytest.mark.slow
def test_cross_process_bit_identical(model):
    parent = logits_digest(model, TEXT)
    for _ in range(2):
        out = subprocess.run(
            [sys.executable, "-m", "llmzip.model", str(config.NUM_THREADS)],
            input=TEXT, capture_output=True, text=True, encoding="utf-8", check=True,
        )
        assert out.stdout.strip().splitlines()[-1] == parent


NO_NETWORK_SCRIPT = """
import socket, sys
def _blocked(*a, **k):
    print("NETWORK ATTEMPT", a[:2], file=sys.stderr)
    raise OSError("network disabled in test")
socket.socket.connect = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
from llmzip.model import NextTokenModel
print(NextTokenModel().model_fingerprint())
"""

# Fingerprint at NUM_THREADS=1 (phase 4 inputs: + numpy, coder/freq bits, tokenizer
# flags, PROBS_SCHEME). Changing it makes existing .lz files undecodable: update deliberately.
EXPECTED_FINGERPRINT = "2bb9a6826f48e26a"


@pytest.mark.slow
def test_loads_without_network():
    out = subprocess.run(
        [sys.executable, "-c", NO_NETWORK_SCRIPT], capture_output=True, text=True, check=True
    )
    assert "NETWORK ATTEMPT" not in out.stderr
    assert out.stdout.strip().splitlines()[-1] == EXPECTED_FINGERPRINT


def test_batched_vs_incremental_info(model):
    """Non-gating: documents why HARD RULE 3 forbids batched probabilities."""
    ids = model.encode(TEXT)
    incremental = torch.stack(model.logits_sequence(ids))
    with torch.no_grad():
        batched = model.model(input_ids=torch.tensor([[config.BOS_TOKEN_ID] + ids[:-1]])).logits[0]
    diff = (batched - incremental).abs().max().item()
    print(f"\nbatched vs incremental: bit-identical={torch.equal(batched, incremental)} max|diff|={diff:.3e}")
