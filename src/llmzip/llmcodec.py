"""Text compression: GPT-2 probabilities + the integer arithmetic coder, in blocks.

The whole text is tokenized (and round-trip checked) once; the ids are split into
blocks of at most `block_size` tokens, and each block is coded independently,
starting from a fresh BOS with an empty KV cache. Decompression decodes all blocks
into one id list and detokenizes ONCE, so multi-byte characters split across a
block edge are reassembled correctly.

Encoder and decoder both pull their frequency tables from `_cum_tables`, which feeds
the model one token at a time (HARD RULE 3) and converts logits with
`probs.logits_to_freqs` (the only logits -> frequencies path).
"""

from collections.abc import Callable

from llmzip.coder import Decoder, Encoder
from llmzip.config import MAX_BLOCK_TOKENS
from llmzip.errors import LlmzipError
from llmzip.freqtable import cumulative, find_symbol, symbol_interval
from llmzip.model import NextTokenModel
from llmzip.probs import TOTAL, logits_to_freqs

Progress = Callable[[int], None] | None  # called with the number of tokens just coded


class NotRoundTrippable(LlmzipError, ValueError):
    """decode(encode(text)) != text, so the tokenizer path would not be lossless."""


_model: NextTokenModel | None = None


def get_model() -> NextTokenModel:
    global _model
    if _model is None:
        _model = NextTokenModel()
    return _model


def _check_block_size(block_size: int) -> None:
    if not 1 <= block_size <= MAX_BLOCK_TOKENS:
        raise ValueError(f"block_size must be in 1..{MAX_BLOCK_TOKENS}, got {block_size}")


def tokenize_checked(text: str, model: NextTokenModel) -> list[int]:
    try:
        ids = model.encode(text)
        decoded = model.decode(ids)
    except Exception as e:  # e.g. lone surrogates make the tokenizer raise TypeError
        raise NotRoundTrippable(f"tokenizer failed: {e}") from e
    if decoded != text:
        raise NotRoundTrippable("decode(encode(text)) differs from the input")
    return ids


def _cum_tables(model: NextTokenModel, n: int):
    """Yield the cumulative table for step i; receive (send) token i before step i+1."""
    logits = model.start()
    for i in range(n):
        token = yield cumulative(logits_to_freqs(logits), TOTAL)
        if i < n - 1:
            logits = model.next_logits(token)


def encode_block(ids: list[int], model: NextTokenModel, progress: Progress = None) -> bytes:
    """Code one block (<= MAX_BLOCK_TOKENS ids) from a fresh BOS. Same bits as inside compress_text."""
    enc = Encoder()
    tables = _cum_tables(model, len(ids))
    cum = next(tables)
    for i, token in enumerate(ids):
        lo, hi = symbol_interval(cum, token)
        enc.encode(lo, hi, TOTAL)
        if progress:
            progress(1)
        if i < len(ids) - 1:
            cum = tables.send(token)
    return enc.finish()


def _decode_block(payload: bytes, n: int, model: NextTokenModel, progress: Progress) -> list[int]:
    dec = Decoder(payload)
    tables = _cum_tables(model, n)
    cum = next(tables)
    ids = []
    for i in range(n):
        token = find_symbol(cum, dec.get_target(TOTAL))
        lo, hi = symbol_interval(cum, token)
        dec.consume(lo, hi, TOTAL)
        ids.append(token)
        if progress:
            progress(1)
        if i < n - 1:
            cum = tables.send(token)
    return ids


def compress_text(
    text: str,
    model: NextTokenModel | None = None,
    block_size: int = MAX_BLOCK_TOKENS,
    progress: Progress = None,
    ids: list[int] | None = None,
    done_blocks: list[bytes] | None = None,
) -> tuple[list[bytes], int]:
    """Return (block payloads, total tokens). Pass `ids` if already tokenize_checked.

    `done_blocks` are leading blocks already produced by `encode_block` for these ids
    and this block_size (e.g. by the auto-mode size estimate); they are not re-encoded.
    """
    _check_block_size(block_size)
    if text == "":
        return [], 0
    model = model or get_model()
    if ids is None:
        ids = tokenize_checked(text, model)
    blocks = list(done_blocks or [])
    for i in range(len(blocks) * block_size, len(ids), block_size):
        blocks.append(encode_block(ids[i : i + block_size], model, progress))
    return blocks, len(ids)


def decompress_text(
    blocks: list[bytes],
    n_tokens: int,
    model: NextTokenModel | None = None,
    block_size: int = MAX_BLOCK_TOKENS,
    progress: Progress = None,
) -> str:
    _check_block_size(block_size)
    if n_tokens == 0:
        return ""
    if len(blocks) != -(-n_tokens // block_size):
        raise ValueError(f"{len(blocks)} blocks inconsistent with {n_tokens} tokens")
    model = model or get_model()
    ids: list[int] = []
    for b, payload in enumerate(blocks):
        n = min(block_size, n_tokens - b * block_size)
        ids.extend(_decode_block(payload, n, model, progress))
    return model.decode(ids)
