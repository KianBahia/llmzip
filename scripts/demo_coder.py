"""Compress files' raw bytes with an adaptive order-0 byte model + the arithmetic coder.

Run: uv run python scripts/demo_coder.py FILE [FILE ...]
"""

import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from llmzip.coder import Decoder, Encoder
from llmzip.freqtable import cumulative, find_symbol, symbol_interval

INCREMENT = 32
RESCALE_AT = 1 << 16


class Order0Model:
    """Adaptive byte counts: start at 1, +INCREMENT per byte, halve (keeping >= 1) past RESCALE_AT."""

    def __init__(self) -> None:
        self.counts = [1] * 256
        self.total = 256

    def cum(self):
        return cumulative(self.counts, self.total)

    def update(self, sym: int) -> None:
        self.counts[sym] += INCREMENT
        self.total += INCREMENT
        if self.total > RESCALE_AT:
            self.counts = [max(1, c >> 1) for c in self.counts]
            self.total = sum(self.counts)


def compress(data: bytes) -> tuple[bytes, list[tuple[int, int, int]]]:
    model, enc, intervals = Order0Model(), Encoder(), []
    for b in data:
        cum = model.cum()
        lo, hi = symbol_interval(cum, b)
        enc.encode(lo, hi, model.total)
        intervals.append((lo, hi, model.total))
        model.update(b)
    return enc.finish(), intervals


def decompress(blob: bytes, n: int) -> bytes:
    model, dec, out = Order0Model(), Decoder(blob), bytearray()
    for _ in range(n):
        cum = model.cum()
        s = find_symbol(cum, dec.get_target(model.total))
        lo, hi = symbol_interval(cum, s)
        dec.consume(lo, hi, model.total)
        out.append(s)
        model.update(s)
    return bytes(out)


def coder_only_rates(blob: bytes, intervals) -> tuple[float, float]:
    """Symbols/sec for the bare coder, replaying precomputed intervals (no model, no search)."""
    t0 = time.perf_counter()
    enc = Encoder()
    for lo, hi, total in intervals:
        enc.encode(lo, hi, total)
    assert enc.finish() == blob
    t1 = time.perf_counter()
    dec = Decoder(blob)
    for lo, hi, total in intervals:
        dec.get_target(total)
        dec.consume(lo, hi, total)
    t2 = time.perf_counter()
    n = len(intervals)
    return n / (t1 - t0), n / (t2 - t1)


def main(paths: list[str]) -> int:
    ok = True
    for path in paths:
        data = Path(path).read_bytes()
        n = len(data)
        t0 = time.perf_counter()
        blob, intervals = compress(data)
        t1 = time.perf_counter()
        restored = decompress(blob, n)
        t2 = time.perf_counter()
        match = restored == data
        ok &= match
        gz = len(subprocess.run(["gzip", "-9", "-c"], input=data, capture_output=True, check=True).stdout)
        entropy = sum(-c * math.log2(c / n) for c in Counter(data).values()) / 8 if n else 0.0
        enc_only, dec_only = coder_only_rates(blob, intervals) if n else (0.0, 0.0)
        print(f"{path}")
        print(f"  round trip:        {'OK' if match else 'MISMATCH'}")
        print(f"  original:          {n} bytes")
        print(f"  llmzip order-0:    {len(blob)} bytes ({len(blob) * 8 / max(n, 1):.3f} bits/byte)")
        print(f"  static order-0 H:  {entropy:.1f} bytes (reference)")
        print(f"  gzip -9:           {gz} bytes")
        print(f"  with model:        encode {n / (t1 - t0):,.0f} sym/s, decode {n / (t2 - t1):,.0f} sym/s")
        print(f"  coder only:        encode {enc_only:,.0f} sym/s, decode {dec_only:,.0f} sym/s")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
