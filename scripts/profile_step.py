"""Per-token time profile of the LLM coding loop (measurement only; changes nothing).

Run: uv run python scripts/profile_step.py [FILE]   (default: bench/samples/prose.txt, first ~300 tokens)

Mirrors one encode + one decode pass of llmcodec, timing each stage separately.
"""

import sys
import time
from pathlib import Path

from llmzip.coder import Decoder, Encoder
from llmzip.freqtable import cumulative, find_symbol, symbol_interval
from llmzip.model import NextTokenModel
from llmzip.probs import TOTAL, logits_to_freqs

ROOT = Path(__file__).resolve().parents[1]
N_TOKENS = 300


def main(path: Path) -> int:
    model = NextTokenModel()
    ids = model.encode(path.read_bytes().decode("utf-8"))[:N_TOKENS]
    n = len(ids)
    t = dict.fromkeys(["forward", "logits_to_freqs", "cumulative", "interval/search", "coder"], 0.0)
    clock = time.perf_counter

    model.logits_sequence(ids[:20])  # warm-up

    # Encode pass.
    enc = Encoder()
    t0 = clock()
    logits = model.start()
    t["forward"] += clock() - t0
    cums = []
    for i, token in enumerate(ids):
        t0 = clock(); freqs = logits_to_freqs(logits); t1 = clock()
        cum = cumulative(freqs, TOTAL); t2 = clock()
        lo, hi = symbol_interval(cum, token); t3 = clock()
        enc.encode(lo, hi, TOTAL); t4 = clock()
        t["logits_to_freqs"] += t1 - t0
        t["cumulative"] += t2 - t1
        t["interval/search"] += t3 - t2
        t["coder"] += t4 - t3
        cums.append(cum)
        if i < n - 1:
            t0 = clock(); logits = model.next_logits(token); t["forward"] += clock() - t0
    payload = enc.finish()

    # Decode pass, coder + search only (tables reused from above: identical by construction).
    dec = Decoder(payload)
    out = []
    for cum in cums:
        t0 = clock(); target = dec.get_target(TOTAL); t1 = clock()
        s = find_symbol(cum, target); lo, hi = symbol_interval(cum, s); t2 = clock()
        dec.consume(lo, hi, TOTAL); t3 = clock()
        t["coder"] += (t1 - t0) + (t3 - t2)
        t["interval/search"] += t2 - t1
        out.append(s)
    assert out == ids

    # Per token: forward/probs/cumulative happen once per token per direction; the decode
    # side is estimated as equal to encode. Coder/search are measured for both directions.
    per_dir = {k: (v if k in ("coder", "interval/search") else 2 * v) / (2 * n) for k, v in t.items()}
    total = sum(per_dir.values())
    print(f"{path.name}: {n} tokens; average per token per direction (encode or decode)\n")
    print(f"{'stage':<18} {'µs/token':>10} {'share':>7}")
    for k, v in per_dir.items():
        print(f"{k:<18} {v * 1e6:>10.1f} {v / total:>7.1%}")
    print(f"{'total':<18} {total * 1e6:>10.1f} {1.0:>7.1%}   (~{1 / total:.0f} tok/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "bench" / "samples" / "prose.txt"))
