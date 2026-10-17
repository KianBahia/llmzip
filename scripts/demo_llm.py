"""Compress files with GPT-2 (forced LLM mode); decompress in a FRESH process via the CLI.

Run: uv run python scripts/demo_llm.py FILE [FILE ...]
Reports ideal / quantized / actual payload sizes and in-process encode/decode tok/s.
"""

import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from llmzip import container as ct
from llmzip.config import MAX_BLOCK_TOKENS
from llmzip.llmcodec import NotRoundTrippable, compress_text, decompress_text, get_model, tokenize_checked
from llmzip.probs import TOTAL, logits_to_freqs, softmax64


def external_size(cmd: list[str], data: bytes) -> int:
    return len(subprocess.run(cmd, input=data, capture_output=True, check=True).stdout)


def ideal_and_quantized_bytes(model, ids: list[int]) -> tuple[float, float]:
    """Per-block, matching how the codec restarts the context every MAX_BLOCK_TOKENS."""
    ideal = quantized = 0.0
    for b in range(0, len(ids), MAX_BLOCK_TOKENS):
        block = ids[b : b + MAX_BLOCK_TOKENS]
        for token, logits in zip(block, model.logits_sequence(block)):
            ideal -= math.log2(softmax64(logits)[token])
            quantized -= math.log2(int(logits_to_freqs(logits)[token]) / TOTAL)
    return ideal / 8, quantized / 8


def decompress_in_fresh_process(blob: bytes, tmp: Path) -> bytes:
    packed, restored = tmp / "packed.lz", tmp / "restored.txt"
    packed.write_bytes(blob)
    subprocess.run(
        [sys.executable, "-m", "llmzip.cli", "decompress", str(packed), "-o", str(restored), "--quiet"],
        capture_output=True, text=True, check=True,
    )
    return restored.read_bytes()


def main(paths: list[str]) -> int:
    model = get_model()
    rows, ok = [], True
    for path in paths:
        raw = Path(path).read_bytes()
        text = raw.decode("utf-8")
        try:
            ids = tokenize_checked(text, model)
        except NotRoundTrippable as e:
            print(f"skip {path}: {e}")
            continue
        t0 = time.perf_counter()
        blocks, n = compress_text(text, model, ids=ids)
        enc_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        assert decompress_text(blocks, n, model) == text
        dec_s = time.perf_counter() - t0
        payload = b"".join(blocks)
        body = ct.LLMBody(model.fingerprint_bytes(), MAX_BLOCK_TOKENS, n, blocks)
        blob = ct.pack(ct.Container(ct.LLM, len(raw), ct.crc32(raw), body))
        with tempfile.TemporaryDirectory() as tmp:
            restored = decompress_in_fresh_process(blob, Path(tmp))
        match = restored == raw
        ok &= match
        ideal, quantized = ideal_and_quantized_bytes(model, ids)
        rows.append({
            "file": Path(path).name,
            "match": match,
            "chars": len(text),
            "bytes": len(raw),
            "tokens": n,
            "payload": len(payload),
            "container": len(blob),
            "ideal": ideal,
            "quantized": quantized,
            "bpc": len(payload) * 8 / len(text),
            "gzip": external_size(["gzip", "-9", "-c"], raw),
            "xz": external_size(["xz", "-9", "-c"], raw),
            "zstd": external_size(["zstd", "-19", "-c", "-q"], raw),
            "enc_tok_s": n / enc_s,
            "dec_tok_s": n / dec_s,
        })

    print(
        f"{'file':<16} {'ok':<3} {'bytes':>6} {'tokens':>6} {'payload':>7} {'.lz':>6} {'ideal':>7} {'quant':>7} "
        f"{'bits/ch':>7} {'gzip-9':>7} {'xz-9':>6} {'zstd-19':>7} {'enc t/s':>8} {'dec t/s':>8}"
    )
    for r in rows:
        print(
            f"{r['file']:<16} {'OK' if r['match'] else 'BAD':<3} {r['bytes']:>6} {r['tokens']:>6} "
            f"{r['payload']:>7} {r['container']:>6} {r['ideal']:>7.1f} {r['quantized']:>7.1f} {r['bpc']:>7.3f} "
            f"{r['gzip']:>7} {r['xz']:>6} {r['zstd']:>7} {r['enc_tok_s']:>8.1f} {r['dec_tok_s']:>8.1f}"
        )
    print("\npayload = coded blocks only; .lz = full container. ok = byte-identical after")
    print("decompressing the .lz in a fresh process with the CLI.")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
