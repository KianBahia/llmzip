"""zstd fallback for input the LLM path can't (or shouldn't) handle."""

import zstandard

from llmzip.errors import CorruptContainer

ZSTD_LEVEL = 19


def zstd_compress(data: bytes) -> bytes:
    return zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(data)


def zstd_decompress(frame: bytes, original_size: int) -> bytes:
    try:
        return zstandard.ZstdDecompressor().decompress(frame, max_output_size=original_size)
    except zstandard.ZstdError as e:
        raise CorruptContainer(f"corrupt zstd data: {e}") from e
