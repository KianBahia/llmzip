"""The .lz container format, version 1. All integers little-endian.

Header (18 bytes):
    magic          4s   b"LLMZ"
    version        u8   1
    mode           u8   0 = STORED, 1 = ZSTD, 2 = LLM
    original_size  u64  bytes
    crc32          u32  zlib.crc32 of the ORIGINAL bytes
Body:
    STORED  the raw bytes (exactly original_size)
    ZSTD    one zstd frame
    LLM     fingerprint 8s | block_size u16 | n_tokens u64 | n_blocks u32
            | n_blocks x uvarint (LEB128) block payload lengths
            | block payloads concatenated

`unpack` validates structure only; `verify` checks size + CRC after decoding.
"""

import struct
import zlib
from dataclasses import dataclass

from llmzip.config import MAX_BLOCK_TOKENS
from llmzip.errors import (
    BadMagic,
    ChecksumMismatch,
    CorruptContainer,
    FingerprintMismatch,
    Truncated,
    UnsupportedVersion,
)

MAGIC = b"LLMZ"
VERSION = 1
STORED, ZSTD, LLM = 0, 1, 2
MODE_NAMES = {STORED: "STORED", ZSTD: "ZSTD", LLM: "LLM"}

_HEADER = struct.Struct("<4sBBQI")
_LLM_HEADER = struct.Struct("<8sHQI")
HEADER_SIZE = _HEADER.size
_U64_MAX = (1 << 64) - 1


@dataclass
class LLMBody:
    fingerprint: bytes  # 8 raw bytes
    block_size: int
    n_tokens: int
    blocks: list[bytes]


@dataclass
class Container:
    mode: int
    original_size: int
    crc32: int
    body: bytes | LLMBody


def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


# --- uvarint (unsigned LEB128) ----------------------------------------------------


def uvarint_encode(n: int) -> bytes:
    if not 0 <= n <= _U64_MAX:
        raise ValueError(f"uvarint out of range: {n}")
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def uvarint_decode(buf: bytes, pos: int) -> tuple[int, int]:
    """Return (value, new_pos). At most 10 bytes; the value must fit in u64."""
    value = shift = 0
    for i in range(10):
        if pos + i >= len(buf):
            raise Truncated("file ends inside a block length")
        byte = buf[pos + i]
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if value > _U64_MAX:
                raise CorruptContainer("block length overflows u64")
            return value, pos + i + 1
        shift += 7
    raise CorruptContainer("block length uvarint longer than 10 bytes")


# --- pack / unpack ----------------------------------------------------------------


def pack(c: Container) -> bytes:
    header = _HEADER.pack(MAGIC, VERSION, c.mode, c.original_size, c.crc32)
    if c.mode in (STORED, ZSTD):
        return header + c.body
    if c.mode == LLM:
        b = c.body
        lengths = b"".join(uvarint_encode(len(p)) for p in b.blocks)
        return (
            header
            + _LLM_HEADER.pack(b.fingerprint, b.block_size, b.n_tokens, len(b.blocks))
            + lengths
            + b"".join(b.blocks)
        )
    raise ValueError(f"unknown mode {c.mode}")


def unpack(blob: bytes) -> Container:
    if len(blob) < len(MAGIC) or blob[: len(MAGIC)] != MAGIC:
        raise BadMagic("not an llmzip file (bad magic)")
    if len(blob) < len(MAGIC) + 1:
        raise Truncated("file ends inside the header")
    if blob[len(MAGIC)] != VERSION:
        raise UnsupportedVersion(f"unsupported format version {blob[len(MAGIC)]} (this build reads {VERSION})")
    if len(blob) < HEADER_SIZE:
        raise Truncated("file ends inside the header")
    _, _, mode, size, crc = _HEADER.unpack_from(blob)
    body = blob[HEADER_SIZE:]

    if mode == STORED:
        if len(body) < size:
            raise Truncated(f"stored body has {len(body)} of {size} bytes")
        if len(body) > size:
            raise CorruptContainer("trailing data after stored body")
        return Container(mode, size, crc, body)
    if mode == ZSTD:
        if not body:
            raise Truncated("missing zstd frame")
        return Container(mode, size, crc, body)
    if mode != LLM:
        raise CorruptContainer(f"unknown mode {mode}")

    if len(body) < _LLM_HEADER.size:
        raise Truncated("file ends inside the LLM header")
    fp, block_size, n_tokens, n_blocks = _LLM_HEADER.unpack_from(body)
    # Sanity checks before any decoding, so corrupt counts can't trigger huge work.
    if not 1 <= block_size <= MAX_BLOCK_TOKENS:
        raise CorruptContainer(f"invalid block size {block_size}")
    if n_tokens > size:  # every GPT-2 token decodes to at least one byte
        raise CorruptContainer(f"{n_tokens} tokens cannot fit in {size} bytes")
    if n_blocks != -(-n_tokens // block_size):
        raise CorruptContainer(f"{n_blocks} blocks inconsistent with {n_tokens} tokens")
    if (n_tokens == 0) != (size == 0):
        raise CorruptContainer("token count inconsistent with original size")
    pos = _LLM_HEADER.size
    lengths = []
    for _ in range(n_blocks):
        length, pos = uvarint_decode(body, pos)
        lengths.append(length)
    if len(body) - pos < sum(lengths):
        raise Truncated(f"LLM payload has {len(body) - pos} of {sum(lengths)} bytes")
    if len(body) - pos > sum(lengths):
        raise CorruptContainer("trailing data after LLM payload")
    blocks = []
    for length in lengths:
        blocks.append(body[pos : pos + length])
        pos += length
    return Container(mode, size, crc, LLMBody(fp, block_size, n_tokens, blocks))


def check_fingerprint(body: LLMBody, local: bytes) -> None:
    if body.fingerprint != local:
        raise FingerprintMismatch(
            f"model fingerprint mismatch: file {body.fingerprint.hex()}, local {local.hex()} "
            "(different model, library versions or settings)"
        )


def verify(c: Container, data: bytes) -> None:
    if len(data) != c.original_size:
        raise ChecksumMismatch(f"size mismatch: got {len(data)} bytes, expected {c.original_size}")
    if crc32(data) != c.crc32:
        raise ChecksumMismatch(f"CRC32 mismatch: got {crc32(data):08x}, expected {c.crc32:08x}")
