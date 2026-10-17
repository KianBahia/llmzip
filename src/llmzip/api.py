"""Bytes in, .lz bytes out (and back): mode routing + container + verification.

Routing for mode="auto" ("never worse"):
  empty input              -> STORED
  invalid UTF-8            -> candidates: STORED, ZSTD
  not round-trippable      -> candidates: STORED, ZSTD
  otherwise                -> candidates: STORED, ZSTD, LLM
The smallest packed container wins; ties prefer STORED, then ZSTD, then LLM
(cheapest to decompress). All modes share the same header, so this equals
"smaller of zstd and LLM, unless not smaller than the raw bytes -> STORED".
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from llmzip import container as ct
from llmzip.config import MAX_BLOCK_TOKENS
from llmzip.errors import LlmzipError
from llmzip.fallback import zstd_compress, zstd_decompress
from llmzip.llmcodec import NotRoundTrippable, compress_text, decompress_text, get_model, tokenize_checked
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
         block_size: int, progress: ProgressFactory) -> bytes:
    cb = progress(len(ids)) if progress else None
    blocks, n_tokens = compress_text(text, model, block_size, cb, ids=ids)
    body = ct.LLMBody(model.fingerprint_bytes(), block_size, n_tokens, blocks)
    return ct.pack(ct.Container(ct.LLM, len(data), crc, body))


def compress_bytes(
    data: bytes,
    mode: str = "auto",
    block_size: int = MAX_BLOCK_TOKENS,
    progress: ProgressFactory = None,
    model: NextTokenModel | None = None,
) -> tuple[bytes, CompressInfo]:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    crc = ct.crc32(data)
    text = _text_or_none(data)
    candidates: dict[int, bytes] = {}
    ids = None

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
            candidates[ct.LLM] = _llm(data, crc, text, ids, model, block_size, progress)
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
