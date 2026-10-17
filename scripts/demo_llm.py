"""Compress files with GPT-2 + the arithmetic coder; decompress in a FRESH process.

Run: uv run python scripts/demo_llm.py FILE [FILE ...]   (each file <= 1023 tokens)
"""

import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from llmzip.llmcodec import NotRoundTrippable, TextTooLong, compress_text, get_model, tokenize_checked
from llmzip.probs import TOTAL, logits_to_freqs, softmax64


def external_size(cmd: list[str], data: bytes) -> int:
    return len(subprocess.run(cmd, input=data, capture_output=True, check=True).stdout)


def ideal_and_quantized_bytes(model, ids: list[int]) -> tuple[float, float]:
    ideal = quantized = 0.0
    for token, logits in zip(ids, model.logits_sequence(ids)):
        ideal -= math.log2(softmax64(logits)[token])
        quantized -= math.log2(int(logits_to_freqs(logits)[token]) / TOTAL)
    return ideal / 8, quantized / 8


def decompress_in_fresh_process(payload: bytes, n: int, tmp: Path) -> tuple[bytes, float]:
    packed, restored = tmp / "packed.bin", tmp / "restored.txt"
    packed.write_bytes(n.to_bytes(4, "big") + payload)
    out = subprocess.run(
        [sys.executable, "-m", "llmzip.llmcodec", "decompress", str(packed), str(restored)],
        capture_output=True, text=True, check=True,
    )
    seconds = json.loads(out.stdout.strip().splitlines()[-1])["seconds"]
    return restored.read_bytes(), seconds


def main(paths: list[str]) -> int:
    model = get_model()
    rows, ok = [], True
    for path in paths:
        raw = Path(path).read_bytes()
        text = raw.decode("utf-8")
        try:
            ids = tokenize_checked(text, model)
        except (TextTooLong, NotRoundTrippable) as e:
            print(f"skip {path}: {e}")
            continue
        t0 = time.perf_counter()
        payload, n = compress_text(text, model)
        enc_s = time.perf_counter() - t0
        with tempfile.TemporaryDirectory() as tmp:
            restored, dec_s = decompress_in_fresh_process(payload, n, Path(tmp))
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
        f"{'file':<16} {'ok':<3} {'bytes':>6} {'tokens':>6} {'llmzip':>7} {'ideal':>7} {'quant':>7} "
        f"{'bits/ch':>7} {'gzip-9':>7} {'xz-9':>6} {'zstd-19':>7} {'enc t/s':>8} {'dec t/s':>8}"
    )
    for r in rows:
        print(
            f"{r['file']:<16} {'OK' if r['match'] else 'BAD':<3} {r['bytes']:>6} {r['tokens']:>6} "
            f"{r['payload']:>7} {r['ideal']:>7.1f} {r['quantized']:>7.1f} {r['bpc']:>7.3f} "
            f"{r['gzip']:>7} {r['xz']:>6} {r['zstd']:>7} {r['enc_tok_s']:>8.1f} {r['dec_tok_s']:>8.1f}"
        )
    print("\nDecompression ran in a fresh process for every file; ok = byte-identical.")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
