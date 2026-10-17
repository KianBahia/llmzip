"""Integer-only binary arithmetic coder (Witten, Neal & Cleary, 1987).

The coder never sees probabilities, only integer intervals: each symbol is given
as [cum_low, cum_high) out of `total`, and the table may change at every step.
No floats are used anywhere in this module (HARD RULE 4).

Registers are P = CODER_PRECISION_BITS wide. Invariants:
  * 0 <= low <= high <= FULL; the current interval is [low, high] inclusive.
  * After renormalization, low < HALF <= high, and NOT (QUARTER <= low and
    high < 3*QUARTER). Together these imply range = high - low + 1 > QUARTER.
  * total <= QUARTER (asserted in config), so range >= total and every symbol
    with frequency >= 1 narrows to a non-empty interval.
Renormalization:
  * E1: high < HALF   -> interval in lower half: emit 0, shift.
  * E2: low >= HALF   -> interval in upper half: emit 1, shift.
  * E3: QUARTER <= low and high < 3*QUARTER -> interval straddles the middle:
        remember a "pending" bit (emitted, inverted, after the next E1/E2), shift.

Bit order: bits are packed MSB-first within each byte; bytes in order. The
decoder treats every bit past the end of the data as 0, so the encoder strips
trailing zero bytes. The number of symbols is not stored; the caller knows it.
"""

from llmzip.config import CODER_PRECISION_BITS

P = CODER_PRECISION_BITS
FULL = (1 << P) - 1
HALF = 1 << (P - 1)
QUARTER = 1 << (P - 2)
THREE_QUARTERS = 3 * QUARTER
MAX_TOTAL = QUARTER


def _check_interval(cum_low: int, cum_high: int, total: int) -> None:
    # numpy integers are not `int` subclasses; rejecting them enforces conversion.
    if not (type(cum_low) is int and type(cum_high) is int and type(total) is int):
        raise TypeError("coder arguments must be Python ints")
    if not 0 <= cum_low < cum_high <= total <= MAX_TOTAL:
        raise ValueError(
            f"invalid interval [{cum_low}, {cum_high}) of total {total} (max total {MAX_TOTAL})"
        )


class BitWriter:
    """Accumulates bits MSB-first into bytes."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._acc = 0
        self._nbits = 0

    def write_bit(self, bit: int) -> None:
        self._acc = (self._acc << 1) | bit
        self._nbits += 1
        if self._nbits == 8:
            self._buf.append(self._acc)
            self._acc = 0
            self._nbits = 0

    def write_bits(self, bit: int, count: int) -> None:
        for _ in range(count):
            self.write_bit(bit)

    def getvalue(self) -> bytes:
        """All bits written so far; the final partial byte is zero-padded."""
        out = bytes(self._buf)
        if self._nbits:
            out += bytes([self._acc << (8 - self._nbits)])
        return out


class BitReader:
    """Reads bits MSB-first; returns 0 for every bit past the end of the data."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read_bit(self) -> int:
        i = self._pos >> 3
        self._pos += 1
        if i >= len(self._data):
            return 0
        return (self._data[i] >> (7 - ((self._pos - 1) & 7))) & 1


class Encoder:
    def __init__(self) -> None:
        self._low = 0
        self._high = FULL
        self._pending = 0
        self._count = 0
        self._out = BitWriter()

    def encode(self, cum_low: int, cum_high: int, total: int) -> None:
        _check_interval(cum_low, cum_high, total)
        low, high, pending, out = self._low, self._high, self._pending, self._out
        r = high - low + 1
        high = low + r * cum_high // total - 1
        low = low + r * cum_low // total
        while True:
            if high < HALF:  # E1
                out.write_bit(0)
                out.write_bits(1, pending)
                pending = 0
            elif low >= HALF:  # E2
                out.write_bit(1)
                out.write_bits(0, pending)
                pending = 0
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < THREE_QUARTERS:  # E3
                pending += 1
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
        self._low, self._high, self._pending = low, high, pending
        self._count += 1

    def finish(self) -> bytes:
        # Zero symbols: return b"" as a special case. Normal termination always
        # emits bits, but with nothing encoded there is nothing to disambiguate,
        # and a Decoder given b"" never reads when asked for 0 symbols.
        if self._count == 0:
            return b""
        # Emit enough bits to pick a value inside [low, high] when followed by
        # zeros: low < HALF <= high, so if low < QUARTER the value QUARTER
        # ("01" then zeros) is inside, otherwise low >= QUARTER implies
        # high >= 3*QUARTER and HALF ("10" then zeros) is inside.
        out = self._out
        self._pending += 1
        if self._low < QUARTER:
            out.write_bit(0)
            out.write_bits(1, self._pending)
        else:
            out.write_bit(1)
            out.write_bits(0, self._pending)
        # Trailing zero bytes are implied by the decoder's zero-fill.
        return out.getvalue().rstrip(b"\x00")


class Decoder:
    def __init__(self, data: bytes) -> None:
        self._in = BitReader(data)
        self._low = 0
        self._high = FULL
        self._code: int | None = None  # filled lazily so b"" is never read for 0 symbols

    def _ensure_code(self) -> None:
        if self._code is None:
            code = 0
            for _ in range(P):
                code = (code << 1) | self._in.read_bit()
            self._code = code

    def get_target(self, total: int) -> int:
        """Return a value in [0, total) that lies in the next symbol's interval."""
        if not (type(total) is int and 0 < total <= MAX_TOTAL):
            raise ValueError(f"invalid total {total}")
        self._ensure_code()
        r = self._high - self._low + 1
        return ((self._code - self._low + 1) * total - 1) // r

    def consume(self, cum_low: int, cum_high: int, total: int) -> None:
        """Remove the decoded symbol's interval; must mirror Encoder.encode exactly."""
        _check_interval(cum_low, cum_high, total)
        self._ensure_code()
        low, high, code, read_bit = self._low, self._high, self._code, self._in.read_bit
        r = high - low + 1
        high = low + r * cum_high // total - 1
        low = low + r * cum_low // total
        while True:
            if high < HALF:  # E1
                pass
            elif low >= HALF:  # E2
                low -= HALF
                high -= HALF
                code -= HALF
            elif low >= QUARTER and high < THREE_QUARTERS:  # E3
                low -= QUARTER
                high -= QUARTER
                code -= QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            code = (code << 1) | read_bit()
        self._low, self._high, self._code = low, high, code
