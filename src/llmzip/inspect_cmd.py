"""`llmzip inspect`: per-token surprise, exactly as compression sees it.

Read-only with respect to compression. Tokens come from `tokenize_checked`, blocks
restart from `model.start()` like `llmcodec`, and every step uses `probs.softmax64`
(ideal bits) and `probs.logits_to_freqs` (quantized bits, the ones actually paid).
Each block is also arithmetic-coded in the same pass with the real coder and the
same tables; tests assert those bytes equal `llmcodec.compress_text`'s output.
"""

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from llmzip.coder import Encoder
from llmzip.config import MAX_BLOCK_TOKENS
from llmzip.errors import LlmzipError
from llmzip.freqtable import cumulative, symbol_interval
from llmzip.llmcodec import tokenize_checked
from llmzip.model import NextTokenModel
from llmzip.probs import TOTAL, logits_to_freqs, softmax64

# Quantized-bits buckets for the heatmap: (upper bound, background, foreground, label).
BUCKETS = [
    (0.5, "#2c7bb6", "white", "<0.5"),
    (1.5, "#abd9e9", "black", "0.5-1.5"),
    (3.0, "#ffffbf", "black", "1.5-3"),
    (6.0, "#fdae61", "black", "3-6"),
    (12.0, "#f46d43", "black", "6-12"),
    (math.inf, "#a50026", "white", ">=12"),
]
AUTOPILOT_BITS = 0.5


@dataclass
class TokenInfo:
    id: int
    text: str  # this token's share of the original text ("" if its bytes end mid-character)
    ideal_bits: float
    quant_bits: float
    prob: float
    block: int
    top3: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class Analysis:
    tokens: list[TokenInfo]
    total_tokens: int  # before --max-tokens truncation
    truncated: bool
    block_size: int
    chars: int
    predicted_payload: int  # sum over blocks of ceil(quantized bits / 8)
    actual_payload: int  # bytes from arithmetic-coding the same blocks
    block_payloads: list[bytes]

    @property
    def ideal_bits(self) -> float:
        return sum(t.ideal_bits for t in self.tokens)

    @property
    def quant_bits(self) -> float:
        return sum(t.quant_bits for t in self.tokens)

    @property
    def predictability(self) -> float:
        return sum(t.quant_bits < 1 for t in self.tokens) / max(len(self.tokens), 1)


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte <-> printable-unicode table (from the original encoder.py)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_BYTE_DECODER = {c: b for b, c in _bytes_to_unicode().items()}


def token_segments(model: NextTokenModel, ids: list[int]) -> list[str]:
    """Split the decoded text into one segment per token (concatenation == decoded text).

    A character whose UTF-8 bytes span several tokens is attributed to the token that
    completes it; the earlier tokens get "".
    """
    pieces = model.tokenizer.convert_ids_to_tokens(ids)
    out, pending = [], b""
    for piece in pieces:
        pending += bytes(_BYTE_DECODER[c] for c in piece)
        try:
            out.append(pending.decode("utf-8"))
            pending = b""
        except UnicodeDecodeError:
            out.append("")
    if pending:
        raise LlmzipError("token bytes do not form valid UTF-8")
    return out


def analyze(text: str, model: NextTokenModel, block_size: int = MAX_BLOCK_TOKENS,
            max_tokens: int | None = 5000, progress=None) -> Analysis:
    """progress: optional factory(total_tokens) -> advance(n) callback, as in api."""
    if not 1 <= block_size <= MAX_BLOCK_TOKENS:
        raise ValueError(f"block_size must be in 1..{MAX_BLOCK_TOKENS}")
    all_ids = tokenize_checked(text, model)
    ids = all_ids[:max_tokens] if max_tokens else all_ids
    segments = token_segments(model, ids)
    if max_tokens is None or len(ids) == len(all_ids):
        if "".join(segments) != text:
            raise LlmzipError("internal error: token segments do not reproduce the text")

    advance = progress(len(ids)) if progress else None
    tokens: list[TokenInfo] = []
    payloads: list[bytes] = []
    predicted = 0
    for b, start in enumerate(range(0, len(ids), block_size)):
        block = ids[start : start + block_size]
        enc = Encoder()
        block_bits = 0.0
        logits = model.start()
        for i, tok in enumerate(block):
            p = softmax64(logits)
            freqs = logits_to_freqs(logits)
            lo, hi = symbol_interval(cumulative(freqs, TOTAL), tok)
            enc.encode(lo, hi, TOTAL)
            quant = -math.log2(int(freqs[tok]) / TOTAL)
            top = np.argpartition(p, -3)[-3:]
            top = top[np.argsort(-p[top], kind="stable")]
            tokens.append(TokenInfo(
                id=tok, text=segments[start + i], ideal_bits=-math.log2(float(p[tok])),
                quant_bits=quant, prob=float(p[tok]), block=b,
                top3=[(int(t), float(p[t])) for t in top],
            ))
            block_bits += quant
            if advance:
                advance(1)
            if i < len(block) - 1:
                logits = model.next_logits(tok)
        payloads.append(enc.finish())
        predicted += math.ceil(block_bits / 8)

    return Analysis(
        tokens=tokens, total_tokens=len(all_ids), truncated=len(ids) < len(all_ids),
        block_size=block_size, chars=sum(len(t.text) for t in tokens),
        predicted_payload=predicted, actual_payload=sum(map(len, payloads)), block_payloads=payloads,
    )


def top_surprising(a: Analysis, n: int) -> list[int]:
    """Indices of the n tokens with the most quantized bits, descending (stable on ties)."""
    return sorted(range(len(a.tokens)), key=lambda i: -a.tokens[i].quant_bits)[:n]


def autopilot_runs(a: Analysis, n: int = 5) -> list[tuple[int, int]]:
    """(start, length) of the n longest runs of consecutive tokens each < AUTOPILOT_BITS."""
    runs, start = [], None
    for i, t in enumerate(a.tokens + [None]):
        cheap = t is not None and t.quant_bits < AUTOPILOT_BITS
        if cheap and start is None:
            start = i
        elif not cheap and start is not None:
            runs.append((start, i - start))
            start = None
    return sorted(runs, key=lambda r: (-r[1], r[0]))[:n]


def context(a: Analysis, i: int, before: int = 25, after: int = 15) -> str:
    left = "".join(t.text for t in a.tokens[max(0, i - 12) : i])[-before:]
    right = "".join(t.text for t in a.tokens[i + 1 : i + 8])[:after]
    flat = lambda s: s.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")  # noqa: E731
    return f"{flat(left)}[{flat(a.tokens[i].text)}]{flat(right)}"


def summary(a: Analysis) -> dict:
    return {
        "tokens": len(a.tokens),
        "total_tokens": a.total_tokens,
        "truncated": a.truncated,
        "characters": a.chars,
        "blocks": len(a.block_payloads),
        "block_size": a.block_size,
        "ideal_bits": a.ideal_bits,
        "quantized_bits": a.quant_bits,
        "bits_per_char": a.quant_bits / max(a.chars, 1),
        "predicted_payload_bytes": a.predicted_payload,
        "actual_payload_bytes": a.actual_payload,
        "predictability_pct": 100 * a.predictability,
    }


def to_json(a: Analysis, model: NextTokenModel, top_n: int) -> dict:
    return {
        "summary": summary(a),
        "tokens": [
            {"id": t.id, "text": t.text, "ideal_bits": t.ideal_bits, "quantized_bits": t.quant_bits,
             "block": t.block}
            for t in a.tokens
        ],
        "top_surprising": [
            {"index": i, **{k: v for k, v in asdict(a.tokens[i]).items() if k != "top3"},
             "top3": [{"token": model.decode([t]), "prob": p} for t, p in a.tokens[i].top3]}
            for i in top_surprising(a, top_n)
        ],
        "autopilot_runs": [{"start": s, "length": n} for s, n in autopilot_runs(a)],
    }


# --- rendering ----------------------------------------------------------------------


def bucket(bits: float) -> tuple[str, str, str]:
    for limit, bg, fg, label in BUCKETS:
        if bits < limit:
            return bg, fg, label
    return BUCKETS[-1][1:]


def render(a: Analysis, model: NextTokenModel, console, top_n: int, heatmap: bool) -> None:
    from rich.table import Table
    from rich.text import Text

    if heatmap:
        text = Text()
        for t in a.tokens:
            bg, fg, _ = bucket(t.quant_bits)
            text.append(t.text, style=f"{fg} on {bg}")
        console.print(text, soft_wrap=True)  # keep the file's own line breaks; let the terminal wrap
        legend = Text("quantized bits/token: ")
        for _, bg, fg, label in BUCKETS:
            legend.append(f" {label} ", style=f"{fg} on {bg}")
            legend.append(" ")
        console.print(legend)
    else:
        listing = " ".join(f"{t.text!r}[{t.quant_bits:.1f}]" for t in a.tokens[:200])
        console.print("tokens with quantized bits (first 200):", soft_wrap=False)
        console.print(listing, markup=False, highlight=False)
    if a.truncated:
        console.print(f"NOTE: output truncated to the first {len(a.tokens)} of {a.total_tokens} tokens "
                      "(--max-tokens).", markup=False)

    s = summary(a)
    console.print()
    console.print(
        f"tokens {s['tokens']}  characters {s['characters']}  blocks {s['blocks']}  "
        f"bits {s['quantized_bits']:.1f} (ideal {s['ideal_bits']:.1f})  bits/char {s['bits_per_char']:.3f}\n"
        f"predicted LLM payload {s['predicted_payload_bytes']} B  actual payload {s['actual_payload_bytes']} B  "
        f"predictability {s['predictability_pct']:.1f}% of tokens < 1 bit",
        markup=False, highlight=False,
    )

    table = Table(title=f"Top {top_n} most surprising tokens")
    for col in ["#", "token", "bits", "prob", "context", "model's top 3"]:
        table.add_column(col, justify="right" if col in ("#", "bits", "prob") else "left")
    for i in top_surprising(a, top_n):
        t = a.tokens[i]
        top3 = ", ".join(f"{model.decode([tid])!r} {p:.2f}" for tid, p in t.top3)
        # Text(), not str: rich would parse "[import]" in the context as markup and drop it.
        table.add_row(str(i), Text(repr(t.text)), f"{t.quant_bits:.1f}", f"{t.prob:.2g}",
                      Text(context(a, i)), Text(top3))
    console.print(table)

    runs = Table(title=f"Autopilot stretches (consecutive tokens each < {AUTOPILOT_BITS} bits)")
    for col in ["start", "tokens", "bits", "text"]:
        runs.add_column(col, justify="left" if col == "text" else "right")
    for start, n in autopilot_runs(a):
        seg = a.tokens[start : start + n]
        snippet = "".join(t.text for t in seg).replace("\n", "\\n")
        runs.add_row(str(start), str(n), f"{sum(t.quant_bits for t in seg):.1f}",
                     Text(snippet if len(snippet) <= 70 else snippet[:67] + "..."))
    console.print(runs)
