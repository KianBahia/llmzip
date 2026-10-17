import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from llmzip.llmcodec import (
    NotRoundTrippable,
    TextTooLong,
    compress_text,
    decompress_text,
    max_tokens,
)
from llmzip.model import NextTokenModel
from llmzip.probs import TOTAL, logits_to_freqs, softmax64

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from smoke_model import PARAGRAPH  # noqa: E402

PYTHON_SNIPPET = '''def fib(n: int) -> int:
    """Return the n-th Fibonacci number."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


if __name__ == "__main__":
    print([fib(i) for i in range(10)])
'''

JSON_SNIPPET = '{\n  "name": "llmzip",\n  "version": "0.1.0",\n  "tags": ["compression", "gpt2"],\n  "stable": false\n}\n'

MARKDOWN_SNIPPET = "# Notes\n\n- **Bold** item\n- `code` item\n\n> A quote.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"

EDGE_CASES = {
    "crlf": "line one\r\nline two\r\n",
    "tabs": "\tindented\twith\ttabs",
    "trailing_ws": "trailing spaces   \n\n",
    "blank_lines": "a\n\n\n\n\nb",
    "emoji": "I love pizza 🍕🔥 and 👩‍👩‍👧",
    "cjk": "日本語のテキストと中文文本",
    "combining": "café vs café",
    "nul": "before\x00after",
    "leading_spaces": "    leading spaces",
    "endoftext": "end of doc<|endoftext|>start of next",
    "punctuation": "Wait , what . ! ? ... !!",
}

SHORT_TEXT = "The quick brown fox jumps over the lazy dog."


@pytest.fixture(scope="module")
def model():
    return NextTokenModel()


def roundtrip(text, model):
    payload, n = compress_text(text, model)
    assert decompress_text(payload, n, model) == text
    return payload, n


# --- probs ----------------------------------------------------------------------


def test_freqs_invariants(model):
    logits = model.start()
    freqs = logits_to_freqs(logits)
    assert freqs.shape == (model.vocab_size,)
    assert freqs.dtype == np.int64
    assert int(freqs.sum()) == TOTAL
    assert int(freqs.min()) >= 1
    p = softmax64(logits)
    base = np.floor(p * (TOTAL - p.size)).astype(np.int64) + 1
    top = int(np.argmax(p))
    remainder = TOTAL - int(base.sum())
    assert freqs[top] == base[top] + remainder
    assert np.array_equal(np.delete(freqs, top), np.delete(base, top))
    assert np.array_equal(freqs, logits_to_freqs(logits))


def test_freqs_rejects_bad_shape(model):
    with pytest.raises(ValueError):
        logits_to_freqs(model.start().unsqueeze(0))


FREQS_DIGEST_SCRIPT = """
import hashlib, sys
from llmzip.model import NextTokenModel
from llmzip.probs import logits_to_freqs
m = NextTokenModel()
h = hashlib.sha256()
for logits in m.logits_sequence(m.encode(sys.argv[1])):
    h.update(logits_to_freqs(logits).tobytes())
print(h.hexdigest())
"""


@pytest.mark.slow
def test_freqs_identical_in_fresh_process(model):
    h = hashlib.sha256()
    for logits in model.logits_sequence(model.encode(SHORT_TEXT)):
        h.update(logits_to_freqs(logits).tobytes())
    out = subprocess.run(
        [sys.executable, "-c", FREQS_DIGEST_SCRIPT, SHORT_TEXT],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == h.hexdigest()


# --- round trips ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [PARAGRAPH, PYTHON_SNIPPET, JSON_SNIPPET, MARKDOWN_SNIPPET, "x", " "],
    ids=["paragraph", "python", "json", "markdown", "single_char", "single_space"],
)
def test_roundtrip(text, model):
    roundtrip(text, model)


@pytest.mark.parametrize("name", list(EDGE_CASES))
def test_roundtrip_tokenizer_edge_cases(name, model):
    text = EDGE_CASES[name]
    _, n = roundtrip(text, model)
    assert n == len(model.encode(text))
    if name == "endoftext":
        assert 50256 not in model.encode(text)


def test_empty(model):
    assert compress_text("", model) == (b"", 0)
    assert decompress_text(b"", 0, model) == ""


# --- limits and errors ----------------------------------------------------------


def text_with_exactly(n_tokens: int, model) -> str:
    """Decode a prefix of a long text's ids so it re-tokenizes to exactly n_tokens."""
    ids = model.encode((PARAGRAPH + "\n\n") * 8)
    for k in range(n_tokens, n_tokens - 20, -1):
        for extra in range(0, 20):
            text = model.decode(ids[: k + extra])
            if len(model.encode(text)) == n_tokens:
                return text
    raise AssertionError(f"could not build a {n_tokens}-token text")


def test_exactly_max_tokens_roundtrips(model):
    text = text_with_exactly(max_tokens(model), model)
    assert max_tokens(model) == 1023
    _, n = roundtrip(text, model)
    assert n == 1023


def test_one_over_max_raises(model):
    text = text_with_exactly(max_tokens(model) + 1, model)
    with pytest.raises(TextTooLong):
        compress_text(text, model)


def test_not_round_trippable(model):
    with pytest.raises(NotRoundTrippable):
        compress_text("lone surrogate \ud800 here", model)


# --- efficiency -----------------------------------------------------------------


def test_efficiency_vs_ideal(model):
    ids = model.encode(PARAGRAPH)
    ideal_bits = quantized_bits = 0.0
    for token, logits in zip(ids, model.logits_sequence(ids)):
        ideal_bits -= math.log2(softmax64(logits)[token])
        quantized_bits -= math.log2(int(logits_to_freqs(logits)[token]) / TOTAL)
    payload, _ = roundtrip(PARAGRAPH, model)
    ideal, quantized = ideal_bits / 8, quantized_bits / 8
    print(f"\nideal {ideal:.2f} B, quantized {quantized:.2f} B, actual {len(payload)} B")
    assert len(payload) <= ideal * 1.01 + 8


# --- cross-process --------------------------------------------------------------


def run_codec(*args) -> dict:
    out = subprocess.run(
        [sys.executable, "-m", "llmzip.llmcodec", *map(str, args)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.slow
@pytest.mark.parametrize(
    "text",
    [PARAGRAPH, "a\r\nb\x00c 🍕 <|endoftext|> café\r\n"],
    ids=["paragraph", "edge_mix"],
)
def test_cross_process_roundtrip(text, tmp_path):
    src, packed, restored = tmp_path / "in.txt", tmp_path / "packed.bin", tmp_path / "out.txt"
    src.write_bytes(text.encode("utf-8"))
    run_codec("compress", src, packed)
    run_codec("decompress", packed, restored)
    assert restored.read_bytes() == src.read_bytes()
