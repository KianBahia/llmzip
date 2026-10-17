import math

import numpy as np
import pytest

from llmzip.coder import MAX_TOTAL, BitReader, BitWriter, Decoder, Encoder
from llmzip.freqtable import cumulative, find_symbol, symbol_interval, validate

TOTAL = 1 << 24
VOCAB = 50257


def roundtrip(seq, table_fn) -> bytes:
    """Encode then decode `seq`. table_fn(step, history) -> cum, called identically on both sides."""
    enc = Encoder()
    history: list[int] = []
    for i, s in enumerate(seq):
        cum = table_fn(i, history)
        lo, hi = symbol_interval(cum, s)
        enc.encode(lo, hi, int(cum[-1]))
        history.append(s)
    data = enc.finish()

    dec = Decoder(data)
    out: list[int] = []
    for i in range(len(seq)):
        cum = table_fn(i, out)
        total = int(cum[-1])
        s = find_symbol(cum, dec.get_target(total))
        lo, hi = symbol_interval(cum, s)
        dec.consume(lo, hi, total)
        out.append(s)
    assert out == list(seq)
    return data


def fixed(cum):
    return lambda i, h: cum


def skewed_freqs(n=256, top=0.999, total=TOTAL):
    """Symbol 0 gets ~`top` of the mass; the rest share the remainder (each >= 1)."""
    rest = n - 1
    f0 = int(total * top)
    other = total - f0
    freqs = [other // rest] * rest
    freqs[0] += other - sum(freqs)
    return [f0] + freqs


def sample(freqs, n, seed):
    p = np.asarray(freqs, dtype=np.float64)
    return [int(x) for x in np.random.default_rng(seed).choice(len(freqs), size=n, p=p / p.sum())]


def entropy_bits(freqs, seq):
    total = sum(freqs)
    return sum(-math.log2(freqs[s] / total) for s in seq)


# --- fixed tables ---------------------------------------------------------------


def test_uniform_256():
    seq = [int(x) for x in np.random.default_rng(0).integers(0, 256, 5000)]
    data = roundtrip(seq, fixed(cumulative([1] * 256)))
    assert abs(len(data) - 5000) <= 4  # 8 bits/symbol


def test_skewed_999():
    freqs = skewed_freqs()
    seq = sample(freqs, 20000, seed=1)
    assert 0 < sum(1 for s in seq if s != 0) < 100
    roundtrip(seq, fixed(cumulative(freqs, TOTAL)))


# --- changing tables ------------------------------------------------------------


def test_adaptive_order0():
    def table(i, history):  # counts derived purely from history, +1 per symbol
        counts = [1] * 256
        for s in history:
            counts[s] += 1
        return cumulative(counts)

    rng = np.random.default_rng(2)
    seq = [int(x) for x in rng.choice(256, size=1500, p=np.r_[[0.5], np.full(255, 0.5 / 255)])]
    roundtrip(seq, table)


def llm_like_freqs(step: int) -> tuple[np.ndarray, np.ndarray]:
    """Vocab 50257, total 2**24: freq 1 everywhere, a few heavy symbols redrawn per step."""
    rng = np.random.default_rng([12345, step])
    freqs = np.ones(VOCAB, dtype=np.int64)
    heavy = rng.choice(VOCAB, size=int(rng.integers(1, 8)), replace=False)
    weights = rng.dirichlet(np.ones(heavy.size))
    extra = rng.multinomial(TOTAL - VOCAB, weights)
    freqs[heavy] += extra
    return freqs, heavy


def test_llm_like_stress():
    n_steps = 1500
    rng = np.random.default_rng(99)
    seq = []
    rare_run = 0
    for step in range(n_steps):
        freqs, heavy = llm_like_freqs(step)
        if step % 300 < 30:  # runs of 30 consecutive rarest symbols
            mode = "rare"
        elif step in (500, 501):
            mode = "edges"
        else:
            mode = "rare" if rng.random() < 0.3 else "heavy"
        if mode == "heavy":
            s = int(heavy[np.argmax(freqs[heavy])])
        elif mode == "edges":
            s = 0 if step == 500 else VOCAB - 1
        else:
            while True:
                s = int(rng.integers(0, VOCAB))
                if freqs[s] == 1:
                    break
        seq.append(s)
        if freqs[s] == 1:
            rare_run += 1
    assert rare_run >= 300

    def table(i, h):
        freqs, _ = llm_like_freqs(i)
        return cumulative(freqs, TOTAL)

    roundtrip(seq, table)


# --- edge cases -----------------------------------------------------------------


def test_empty_sequence():
    assert Encoder().finish() == b""
    roundtrip([], fixed(cumulative([1] * 10)))


def test_single_symbol():
    roundtrip([3], fixed(cumulative([1] * 10)))


def test_single_symbol_alphabet():
    roundtrip([0] * 50, fixed(cumulative([1])))


def test_first_and_last_index():
    cum = cumulative(skewed_freqs(VOCAB, top=0.5))
    roundtrip([0, VOCAB - 1, 0, 0, VOCAB - 1, VOCAB - 1], fixed(cum))


def test_long_run_same_symbol():
    freqs = skewed_freqs()
    roundtrip([0] * 10000, fixed(cumulative(freqs)))
    roundtrip([200] * 10000, fixed(cumulative(freqs)))  # a rare symbol, repeated


def test_long_input_100k():
    freqs = skewed_freqs(256, top=0.6)
    seq = sample(freqs, 100_000, seed=3)
    roundtrip(seq, fixed(cumulative(freqs)))


# --- efficiency -----------------------------------------------------------------


@pytest.mark.parametrize("top", [0.999, 0.9, 0.5])
def test_efficiency_vs_entropy(top):
    freqs = skewed_freqs(256, top=top)
    seq = sample(freqs, 100_000, seed=4)
    data = roundtrip(seq, fixed(cumulative(freqs)))
    ideal = entropy_bits(freqs, seq) / 8
    print(f"\ntop={top}: {len(data)} bytes vs entropy {ideal:.1f} bytes ({len(data) / ideal - 1:+.3%})")
    assert len(data) <= ideal * 1.01 + 8


# --- property test --------------------------------------------------------------


@pytest.mark.parametrize("seed", range(200))
def test_random_property(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 5001))
    length = int(rng.integers(0, 301))
    changing = seed % 2 == 1

    def make_freqs(r):
        total = int(r.integers(n, TOTAL + 1))
        return 1 + r.multinomial(total - n, r.dirichlet(np.full(n, 0.3)))

    base = cumulative(make_freqs(rng))
    if changing:
        def table(i, h):
            return cumulative(make_freqs(np.random.default_rng([seed, i])))
    else:
        table = fixed(base)

    seq = []
    for i in range(length):
        cum = table(i, seq)
        # Mix uniform-over-symbols (rare-heavy) and probability-weighted picks.
        if rng.random() < 0.5:
            seq.append(int(rng.integers(0, n)))
        else:
            seq.append(find_symbol(cum, int(rng.integers(0, int(cum[-1])))))
    roundtrip(seq, table)


# --- determinism and validation -------------------------------------------------


def test_deterministic_bytes():
    freqs = skewed_freqs(256, top=0.7)
    seq = sample(freqs, 5000, seed=5)
    assert roundtrip(seq, fixed(cumulative(freqs))) == roundtrip(seq, fixed(cumulative(freqs)))


def test_encoder_rejects_bad_intervals():
    enc = Encoder()
    with pytest.raises(ValueError):
        enc.encode(5, 5, 10)  # zero width (freq 0)
    with pytest.raises(ValueError):
        enc.encode(0, 11, 10)  # cum_high > total
    with pytest.raises(ValueError):
        enc.encode(0, 1, MAX_TOTAL + 1)
    with pytest.raises(TypeError):
        enc.encode(np.int64(0), 1, 10)


def test_freqtable_validation_and_search():
    with pytest.raises(ValueError):
        validate([1, 0, 3], 4)
    with pytest.raises(ValueError):
        validate([1, 2, 3], 7)
    with pytest.raises(ValueError):
        cumulative([1, 2, 3], total=7)
    cum = cumulative([2, 1, 3])
    assert list(cum) == [0, 2, 3, 6]
    assert symbol_interval(cum, 1) == (2, 3)
    assert all(type(x) is int for x in symbol_interval(cum, 1))
    assert [find_symbol(cum, t) for t in range(6)] == [0, 0, 1, 2, 2, 2]
    assert type(find_symbol(cum, 0)) is int


def test_bit_io_msb_first_and_zero_fill():
    w = BitWriter()
    for b in [1, 0, 1, 1, 0, 0, 0, 0, 1]:
        w.write_bit(b)
    assert w.getvalue() == bytes([0b10110000, 0b10000000])
    r = BitReader(b"\x80")
    assert [r.read_bit() for _ in range(10)] == [1] + [0] * 9
