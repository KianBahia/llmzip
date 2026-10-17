"""Bytes in, .lz bytes out (and back): mode routing + container + verification.

Routing for mode="auto" ("never worse"):
  empty input              -> STORED
  invalid UTF-8            -> candidates: STORED, ZSTD
  not round-trippable      -> candidates: STORED, ZSTD
  otherwise                -> candidates: STORED, ZSTD, LLM
The smallest packed container wins; ties prefer STORED, then ZSTD, then LLM
(cheapest to decompress). All modes share the same header, so this equals
"smaller of zstd and LLM, unless not smaller than the raw bytes -> STORED".

Skip heuristic (auto mode, text longer than 2 blocks, unless skip=False): code only
the first block, estimate the full LLM payload as its bits/byte x total bytes, and
drop the LLM candidate if that estimate exceeds the zstd frame. Otherwise the first
block is reused, so it is never coded twice and the output is identical to a
no-skip run.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from llmzip import container as ct
from llmzip.config import MAX_BLOCK_TOKENS
from llmzip.errors import LlmzipError
from llmzip.fallback import zstd_compress, zstd_decompress
from llmzip.llmcodec import (
    NotRoundTrippable,
    compress_text,
    decompress_text,
    encode_block,
    get_model,
    tokenize_checked,
)
from llmzip.model import NextTokenModel

MODES = ("auto", "llm", "zstd", "store")

# Called once when LLM coding starts with the total token count; returns a per-token callback.
ProgressFactory = Callable[[int], Callable[[int], None]] | None


@dataclass
class CompressInfo:
    mode: int
    original_size: int
    compressed_size: int
    chars: int | None = None
    n_tokens: int | None = None
    candidates: dict[str, int] = field(default_factory=dict)  # mode name -> container size
    llm_skipped: tuple[int, int] | None = None  # (estimated LLM payload B, zstd frame B)


def _text_or_none(data: bytes) -> str | None:
    try:
        return data.decode("utf-8")  # strict; never utf-8-sig, so a BOM is kept as U+FEFF
    except UnicodeDecodeError:
        return None


def _stored(data: bytes, crc: int) -> bytes:
    return ct.pack(ct.Container(ct.STORED, len(data), crc, data))


def _zstd(data: bytes, crc: int) -> bytes:
    return ct.pack(ct.Container(ct.ZSTD, len(data), crc, zstd_compress(data)))


def _llm(data: bytes, crc: int, text: str, ids: list[int], model: NextTokenModel,
         block_size: int, cb, done_blocks: list[bytes] | None = None) -> bytes:
    blocks, n_tokens = compress_text(text, model, block_size, cb, ids=ids, done_blocks=done_blocks)
    body = ct.LLMBody(model.fingerprint_bytes(), block_size, n_tokens, blocks)
    return ct.pack(ct.Container(ct.LLM, len(data), crc, body))


def skip_estimate(first: bytes, first_ids: list[int], model: NextTokenModel,
                  total_bytes: int, zstd_frame: int) -> tuple[int, bool]:
    """(estimated full LLM payload in bytes, whether to skip the LLM) from the first block."""
    prefix_bytes = len(model.decode(first_ids).encode("utf-8"))
    estimate = len(first) * total_bytes // max(prefix_bytes, 1)
    return estimate, estimate > zstd_frame


def compress_bytes(
    data: bytes,
    mode: str = "auto",
    block_size: int = MAX_BLOCK_TOKENS,
    progress: ProgressFactory = None,
    model: NextTokenModel | None = None,
    skip: bool = True,
) -> tuple[bytes, CompressInfo]:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    crc = ct.crc32(data)
    text = _text_or_none(data)
    candidates: dict[int, bytes] = {}
    ids = None
    skipped = None

    if mode in ("auto", "store") or not data:
        candidates[ct.STORED] = _stored(data, crc)
    if data and mode in ("auto", "zstd"):
        candidates[ct.ZSTD] = _zstd(data, crc)
    if data and mode in ("auto", "llm") and text is not None:
        model = model or get_model()
        try:
            ids = tokenize_checked(text, model)
        except NotRoundTrippable:
            if mode == "llm":
                raise
        if ids is not None:
            cb = progress(len(ids)) if progress else None
            done = None
            if mode == "auto" and skip and len(ids) > 2 * block_size:
                first = encode_block(ids[:block_size], model, cb)
                zstd_frame = len(candidates[ct.ZSTD]) - ct.HEADER_SIZE
                estimate, skip_llm = skip_estimate(first, ids[:block_size], model, len(data), zstd_frame)
                if skip_llm:
                    skipped = (estimate, zstd_frame)
                done = [first]
            if skipped is None:
                candidates[ct.LLM] = _llm(data, crc, text, ids, model, block_size, cb, done)
    if data and mode == "llm" and text is None:
        raise LlmzipError("--mode llm needs valid UTF-8 text; this input is not")

    # Smallest wins; dict order STORED, ZSTD, LLM breaks ties toward cheaper decoding.
    chosen = min(candidates, key=lambda m: (len(candidates[m]), m))
    blob = candidates[chosen]
    info = CompressInfo(
        mode=chosen,
        original_size=len(data),
        compressed_size=len(blob),
        chars=len(text) if text is not None else None,
        n_tokens=len(ids) if ids is not None else None,
        candidates={ct.MODE_NAMES[m]: len(b) for m, b in candidates.items()},
        llm_skipped=skipped,
    )
    return blob, info


def decompress_bytes(
    blob: bytes,
    progress: ProgressFactory = None,
    model: NextTokenModel | None = None,
) -> tuple[bytes, ct.Container]:
    c = ct.unpack(blob)
    if c.mode == ct.STORED:
        data = c.body
    elif c.mode == ct.ZSTD:
        data = zstd_decompress(c.body, c.original_size)
    else:
        body = c.body
        model = model or get_model()
        ct.check_fingerprint(body, model.fingerprint_bytes())  # before any decoding
        cb = progress(body.n_tokens) if progress else None
        text = decompress_text(body.blocks, body.n_tokens, model, body.block_size, cb)
        data = text.encode("utf-8")
    ct.verify(c, data)
    return data, c
