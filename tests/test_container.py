import struct

import pytest

from llmzip import container as ct
from llmzip.errors import (
    BadMagic,
    ChecksumMismatch,
    ContainerError,
    CorruptContainer,
    FingerprintMismatch,
    Truncated,
    UnsupportedVersion,
)
from llmzip.fallback import zstd_compress, zstd_decompress

DATA = b"hello hello hello hello world\n" * 20
FP = bytes.fromhex("0123456789abcdef")


def stored():
    return ct.Container(ct.STORED, len(DATA), ct.crc32(DATA), DATA)


def zstd():
    return ct.Container(ct.ZSTD, len(DATA), ct.crc32(DATA), zstd_compress(DATA))


def llm(blocks=(b"\x01\x02", b"", b"\xff" * 300), block_size=50, n_tokens=120):
    body = ct.LLMBody(FP, block_size, n_tokens, list(blocks))
    return ct.Container(ct.LLM, len(DATA), ct.crc32(DATA), body)


# --- round trips ------------------------------------------------------------------


@pytest.mark.parametrize("make", [stored, zstd, llm])
def test_pack_unpack_roundtrip(make):
    c = make()
    blob = ct.pack(c)
    assert blob[:4] == b"LLMZ" and blob[4] == 1 and blob[5] == c.mode
    assert ct.unpack(blob) == c


def test_header_layout_little_endian():
    blob = ct.pack(stored())
    magic, version, mode, size, crc = struct.unpack_from("<4sBBQI", blob)
    assert (magic, version, mode, size, crc) == (b"LLMZ", 1, ct.STORED, len(DATA), ct.crc32(DATA))
    assert ct.HEADER_SIZE == 18


def test_stored_and_zstd_payloads_verify():
    for make in (stored, zstd):
        c = ct.unpack(ct.pack(make()))
        data = c.body if c.mode == ct.STORED else zstd_decompress(c.body, c.original_size)
        ct.verify(c, data)


# --- uvarint ------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 127, 128, 255, 16383, 16384, 2**32, 2**63, 2**64 - 1])
def test_uvarint_roundtrip(n):
    enc = ct.uvarint_encode(n)
    assert ct.uvarint_decode(enc + b"tail", 0) == (n, len(enc))


def test_uvarint_known_encodings():
    assert ct.uvarint_encode(0) == b"\x00"
    assert ct.uvarint_encode(127) == b"\x7f"
    assert ct.uvarint_encode(128) == b"\x80\x01"
    assert len(ct.uvarint_encode(2**64 - 1)) == 10


def test_uvarint_errors():
    with pytest.raises(Truncated):
        ct.uvarint_decode(b"\x80\x80", 0)
    with pytest.raises(CorruptContainer):
        ct.uvarint_decode(b"\x80" * 11, 0)
    with pytest.raises(CorruptContainer):
        ct.uvarint_decode(b"\xff" * 9 + b"\x7f", 0)  # > u64
    with pytest.raises(ValueError):
        ct.uvarint_encode(-1)
    with pytest.raises(ValueError):
        ct.uvarint_encode(2**64)


# --- corruption ---------------------------------------------------------------------


def test_bad_magic():
    with pytest.raises(BadMagic):
        ct.unpack(b"PK\x03\x04" + ct.pack(stored())[4:])
    with pytest.raises(BadMagic):
        ct.unpack(b"")


def test_unsupported_version():
    blob = bytearray(ct.pack(stored()))
    blob[4] = 2
    with pytest.raises(UnsupportedVersion):
        ct.unpack(bytes(blob))


@pytest.mark.parametrize("make", [stored, zstd, llm])
def test_truncated_header(make):
    blob = ct.pack(make())
    for cut in (5, 10, 17):
        with pytest.raises(Truncated):
            ct.unpack(blob[:cut])


@pytest.mark.parametrize("make", [stored, llm])
def test_truncated_body(make):
    blob = ct.pack(make())
    with pytest.raises(Truncated):
        ct.unpack(blob[:-1])


def test_truncated_llm_header():
    blob = ct.pack(llm())
    with pytest.raises(Truncated):
        ct.unpack(blob[: ct.HEADER_SIZE + 10])


@pytest.mark.parametrize("make", [stored, llm])
def test_trailing_data(make):
    with pytest.raises(CorruptContainer):
        ct.unpack(ct.pack(make()) + b"\x00")


def test_unknown_mode():
    blob = bytearray(ct.pack(stored()))
    blob[5] = 7
    with pytest.raises(CorruptContainer):
        ct.unpack(bytes(blob))


def test_inconsistent_block_count():
    with pytest.raises(CorruptContainer):
        ct.unpack(ct.pack(llm(n_tokens=200)))  # 200/50 = 4 blocks, but 3 stored


@pytest.mark.parametrize("block_size", [0, 1024])
def test_invalid_block_size(block_size):
    with pytest.raises(CorruptContainer):
        ct.unpack(ct.pack(llm(block_size=block_size, blocks=[b"x"], n_tokens=1)))


def test_huge_token_count_rejected_before_decoding():
    # A corrupted n_tokens must not lead to a 2**56-token decode.
    blob = bytearray(ct.pack(llm()))
    off = ct.HEADER_SIZE + 8 + 2
    blob[off : off + 8] = struct.pack("<Q", 2**56)
    with pytest.raises(CorruptContainer):
        ct.unpack(bytes(blob))


@pytest.mark.parametrize("make", [stored, zstd])
def test_flipped_byte_detected(make):
    blob = bytearray(ct.pack(make()))
    blob[ct.HEADER_SIZE + 5] ^= 0x40
    with pytest.raises(ContainerError):
        c = ct.unpack(bytes(blob))
        data = c.body if c.mode == ct.STORED else zstd_decompress(c.body, c.original_size)
        ct.verify(c, data)


def test_flipped_crc_detected():
    blob = bytearray(ct.pack(stored()))
    blob[14] ^= 0x01  # inside the crc32 field
    c = ct.unpack(bytes(blob))
    with pytest.raises(ChecksumMismatch):
        ct.verify(c, c.body)


def test_fingerprint_check():
    c = llm()
    ct.check_fingerprint(c.body, FP)
    with pytest.raises(FingerprintMismatch, match="0123456789abcdef"):
        ct.check_fingerprint(c.body, bytes(8))
