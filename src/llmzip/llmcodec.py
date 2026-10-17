"""In-memory text compression: GPT-2 probabilities + the integer arithmetic coder.

Encoder and decoder both pull their frequency tables from `_cum_tables`, which feeds
the model one token at a time (HARD RULE 3) and converts logits with
`probs.logits_to_freqs` (the only logits -> frequencies path).
"""

import json
import sys
import time
from pathlib import Path

from llmzip.coder import Decoder, Encoder
from llmzip.freqtable import cumulative, find_symbol, symbol_interval
from llmzip.model import NextTokenModel
from llmzip.probs import TOTAL, logits_to_freqs


class NotRoundTrippable(ValueError):
    """decode(encode(text)) != text, so the tokenizer path would not be lossless."""


class TextTooLong(ValueError):
    """More tokens than fit in one GPT-2 context (chunking comes in phase 4)."""


_model: NextTokenModel | None = None


def get_model() -> NextTokenModel:
    global _model
    if _model is None:
        _model = NextTokenModel()
    return _model


def max_tokens(model: NextTokenModel) -> int:
    return model.max_positions - 1  # BOS takes one position


def tokenize_checked(text: str, model: NextTokenModel) -> list[int]:
    try:
        ids = model.encode(text)
        decoded = model.decode(ids)
    except Exception as e:  # e.g. lone surrogates make the tokenizer raise TypeError
        raise NotRoundTrippable(f"tokenizer failed: {e}") from e
    if decoded != text:
        raise NotRoundTrippable("decode(encode(text)) differs from the input")
    if len(ids) > max_tokens(model):
        raise TextTooLong(f"{len(ids)} tokens exceeds the limit of {max_tokens(model)}")
    return ids


def _cum_tables(model: NextTokenModel, n: int):
    """Yield the cumulative table for step i; receive (send) token i before step i+1."""
    logits = model.start()
    for i in range(n):
        token = yield cumulative(logits_to_freqs(logits), TOTAL)
        if i < n - 1:
            logits = model.next_logits(token)


def compress_text(text: str, model: NextTokenModel | None = None) -> tuple[bytes, int]:
    if text == "":
        return b"", 0
    model = model or get_model()
    ids = tokenize_checked(text, model)
    enc = Encoder()
    tables = _cum_tables(model, len(ids))
    cum = next(tables)
    for i, token in enumerate(ids):
        lo, hi = symbol_interval(cum, token)
        enc.encode(lo, hi, TOTAL)
        if i < len(ids) - 1:
            cum = tables.send(token)
    return enc.finish(), len(ids)


def decompress_text(payload: bytes, n_tokens: int, model: NextTokenModel | None = None) -> str:
    if n_tokens == 0:
        return ""
    model = model or get_model()
    dec = Decoder(payload)
    tables = _cum_tables(model, n_tokens)
    cum = next(tables)
    ids = []
    for i in range(n_tokens):
        token = find_symbol(cum, dec.get_target(TOTAL))
        lo, hi = symbol_interval(cum, token)
        dec.consume(lo, hi, TOTAL)
        ids.append(token)
        if i < n_tokens - 1:
            cum = tables.send(token)
    return model.decode(ids)


if __name__ == "__main__":
    # Fresh-process round-trip helper (test/demo framing, NOT the phase 4 format):
    #   python -m llmzip.llmcodec compress   TEXT_IN  PAYLOAD_OUT
    #   python -m llmzip.llmcodec decompress PAYLOAD_IN TEXT_OUT
    # Payload file = 4-byte big-endian n_tokens + payload. Prints {"seconds", "tokens"}.
    mode, src, dst = sys.argv[1:4]
    model = get_model()
    if mode == "compress":
        text = Path(src).read_bytes().decode("utf-8")  # bytes, so CRLF is untouched
        t0 = time.perf_counter()
        payload, n = compress_text(text, model)
        elapsed = time.perf_counter() - t0
        Path(dst).write_bytes(n.to_bytes(4, "big") + payload)
    elif mode == "decompress":
        blob = Path(src).read_bytes()
        n = int.from_bytes(blob[:4], "big")
        t0 = time.perf_counter()
        text = decompress_text(blob[4:], n, model)
        elapsed = time.perf_counter() - t0
        Path(dst).write_bytes(text.encode("utf-8"))
    else:
        sys.exit(f"unknown mode {mode!r}")
    print(json.dumps({"seconds": elapsed, "tokens": n}))
